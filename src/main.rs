use std::cmp::PartialOrd;
use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::num::NonZeroU32;
use std::ops::ControlFlow;
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;
use std::process::ExitCode;

use crate::cli::{Command, Config, ANSII_CLEAR, ANSII_COLOR_RED, ANSII_COLOR_YELLOW};

pub mod cli;
pub mod x86;

const NUM_REGISTERS: usize = 1 << 15;

macro_rules! warn {
    ($pat:expr) => {{
        eprint!("{ANSII_COLOR_YELLOW}warning{ANSII_CLEAR}: ");
        eprintln!($pat);
    }};
}

macro_rules! error {
    ($pat:expr) => {{
        eprint!("{ANSII_COLOR_RED}error{ANSII_CLEAR}: ");
        eprintln!($pat);
    }};
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Token {
    Shl,
    Shr,
    Inc,
    Dec,
    Output,
    Input,
    LSquare,
    RSquare,
}

impl std::fmt::Display for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Token::Shl => write!(f, "<"),
            Token::Shr => write!(f, ">"),
            Token::Inc => write!(f, "+"),
            Token::Dec => write!(f, "-"),
            Token::Output => write!(f, "."),
            Token::Input => write!(f, ","),
            Token::LSquare => write!(f, "["),
            Token::RSquare => write!(f, "]"),
        }
    }
}

impl Token {
    pub fn is_combinable(self) -> bool {
        match self {
            Token::Shl | Token::Shr | Token::Inc | Token::Dec => true,
            Token::Output | Token::Input | Token::LSquare | Token::RSquare => false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Instruction {
    Shl(u16),
    Shr(u16),
    Inc(i16, u8),
    Dec(i16, u8),
    Output,
    Input,
    /// Jump to the position if the current register value is zero.
    JumpZ(Jump),
    /// Jump to the position if the current register value is not zero.
    JumpNz(Jump),

    /// Clear the current register:
    /// ```bf
    /// [
    ///     -
    /// ]
    /// ```
    Zero(i16),
    Set(i16, u8),
    /// Add current register value to register at offset.
    Add(i16),
    /// Subtract current register value from register at offset.
    Sub(i16),
    /// Multiply current register value and add to register at offset.
    AddMul(i16, u8),
    /// Multiply current register value and subtraction from register at offset.
    SubMul(i16, u8),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Jump {
    Location(NonZeroU32),
    Redundant,
}

impl Jump {
    pub fn is_redundant(&self) -> bool {
        matches!(self, Self::Redundant)
    }
}

impl std::fmt::Display for Instruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Instruction::Shl(n) => write!(f, "< ({n})"),
            Instruction::Shr(n) => write!(f, "> ({n})"),
            Instruction::Inc(0, n) => write!(f, "+ ({n})"),
            Instruction::Inc(o, n) => write!(f, "<{o}> + ({n})"),
            Instruction::Dec(0, n) => write!(f, "- ({n})"),
            Instruction::Dec(o, n) => write!(f, "<{o}> - ({n})"),
            Instruction::Output => write!(f, "out"),
            Instruction::Input => write!(f, "in"),
            Instruction::JumpZ(Jump::Redundant) => write!(f, "[ !"),
            Instruction::JumpZ(Jump::Location(_)) => write!(f, "["),
            Instruction::JumpNz(Jump::Redundant) => write!(f, "] !"),
            Instruction::JumpNz(Jump::Location(_)) => write!(f, "]"),

            Instruction::Zero(0) => write!(f, "zero"),
            Instruction::Zero(o) => write!(f, "<{o}> zero"),
            Instruction::Set(0, n) => write!(f, "set {n}"),
            Instruction::Set(o, n) => write!(f, "<{o}> set {n}"),
            Instruction::Add(o) => write!(f, "<{o}> add"),
            Instruction::Sub(o) => write!(f, "<{o}> sub"),
            Instruction::AddMul(o, n) => write!(f, "<{o}> addmul({n})"),
            Instruction::SubMul(o, n) => write!(f, "<{o}> submul({n})"),
        }
    }
}

fn main() -> ExitCode {
    let (config, command, path) = match cli::parse_args() {
        ControlFlow::Continue(c) => c,
        ControlFlow::Break(e) => return e,
    };

    let input = std::fs::read_to_string(&path).unwrap();
    let bytes = input.as_bytes();

    let mut line = 1;
    let mut line_start = 0;
    let mut par_stack = Vec::new();
    let mut tokens = Vec::new();
    let mut mismatched = false;
    for (i, b) in bytes.iter().enumerate() {
        let t = match *b {
            b'<' => Token::Shl,
            b'>' => Token::Shr,
            b'+' => Token::Inc,
            b'-' => Token::Dec,
            b'.' => Token::Output,
            b',' => Token::Input,
            b'[' => {
                let col = input[line_start..i].chars().count();
                par_stack.push((line, col));
                Token::LSquare
            }
            b']' => {
                if par_stack.pop().is_none() {
                    let col = input[line_start..i].chars().count();
                    error!("missing opening bracket for [{line}:{col}]");
                    mismatched = true;
                }
                Token::RSquare
            }
            b'\n' => {
                line += 1;
                line_start = i + 1;
                continue;
            }
            _ => continue,
        };
        tokens.push(t);
    }
    for &(line, col) in par_stack.iter() {
        error!("missing closing bracket for [{line}:{col}]");
        mismatched = true;
    }
    if mismatched {
        return ExitCode::FAILURE;
    }

    // combine instructions
    let mut instructions = tokens
        .chunk_by(|a, b| a.is_combinable() && a == b)
        .inspect(|c| {
            if config.verbose >= 2 && c.len() > 1 {
                println!("combine {}", c.len());
            }
        })
        .map(|chunk| match chunk[0] {
            Token::Shl => Instruction::Shl(chunk.len() as u16),
            Token::Shr => Instruction::Shr(chunk.len() as u16),
            Token::Inc => Instruction::Inc(0, chunk.len() as u8),
            Token::Dec => Instruction::Dec(0, chunk.len() as u8),
            Token::Output => Instruction::Output,
            Token::Input => Instruction::Input,
            Token::LSquare => Instruction::JumpZ(Jump::Location(NonZeroU32::MAX)),
            Token::RSquare => Instruction::JumpNz(Jump::Location(NonZeroU32::MAX)),
        })
        .collect::<Vec<_>>();
    if config.verbose >= 1 {
        println!("============================================================");
        println!(
            "tokens before {} after: {} ({:.3}%)",
            tokens.len(),
            instructions.len(),
            100.0 * instructions.len() as f32 / tokens.len() as f32,
        );
        println!("============================================================");
    }
    if config.verbose >= 3 || command == Command::Format {
        cli::print_brainfuck_code(&instructions);
        if command == Command::Format {
            return ExitCode::SUCCESS;
        }
        if config.verbose >= 1 {
            println!("============================================================");
        }
    }

    if config.optimize {
        if config.print_unoptimized_ir {
            cli::print_instructions(&instructions);
            println!("============================================================");
        }

        let prev_len = instructions.len();
        // zero register
        if config.o_zeros {
            use Instruction::*;

            let mut i = 0;
            while i + 3 < instructions.len() {
                let [a, b, c] = &instructions[i..i + 3] else {
                    unreachable!()
                };
                if let (JumpZ(_), Dec(0, 1), JumpNz(_)) = (a, b, c) {
                    let range = i..i + 3;
                    if config.verbose >= 2 {
                        println!("replaced {range:?} with zero");
                    }
                    instructions.drain(range);
                    instructions.insert(i, Zero(0));
                }

                i += 1;
            }
        }

        if config.o_dead_code || config.o_init || config.o_jumps {
            optimize_static_code(&config, &mut instructions);
        }

        if config.o_arithmetic || config.o_jumps {
            let mut i = 0;
            while i < instructions.len() {
                arithmetic_loop_pass(&config, &mut instructions, i);
                i += 1;
            }
        }

        if config.o_dead_code || config.o_init || config.o_jumps {
            optimize_static_code(&config, &mut instructions);
        }

        // remove optimized jumps
        if config.o_jumps {
            let mut par_stack = Vec::new();
            let mut i = 0;
            while i < instructions.len() {
                let inst = instructions[i];
                match inst {
                    Instruction::JumpZ(jump) => par_stack.push((i, jump.is_redundant())),
                    Instruction::JumpNz(end_jump) => {
                        let Some((start_idx, start_redundant)) = par_stack.pop() else {
                            unreachable!("mismatched brackets")
                        };

                        if start_redundant && end_jump.is_redundant() {
                            if config.verbose >= 2 {
                                println!("remove redundant jump pair at {start_idx} and {i}");
                            }
                            instructions.remove(i);
                            instructions.remove(start_idx);

                            i -= 2;
                        }
                    }
                    _ => (),
                }
                i += 1;
            }
            if !par_stack.is_empty() {
                unreachable!("mismatched brackets")
            }
        }

        if config.verbose >= 1 {
            if config.verbose >= 2 {
                println!("============================================================");
            }
            println!(
                "instructions before {} after: {} ({:.3}%)",
                prev_len,
                instructions.len(),
                100.0 * instructions.len() as f32 / prev_len as f32,
            );
            println!("============================================================");
        }
    }

    // update jump indices
    let mut par_stack = Vec::new();
    for (i, instruction) in instructions.iter_mut().enumerate() {
        match instruction {
            Instruction::JumpZ(closing_idx_ref) => par_stack.push((i, closing_idx_ref)),
            Instruction::JumpNz(opening_idx_ref) => {
                let Some((opening_idx, closing_idx_ref)) = par_stack.pop() else {
                    unreachable!("mismatched brackets")
                };

                if let Jump::Location(loc) = opening_idx_ref {
                    *loc = unsafe { NonZeroU32::new_unchecked(opening_idx as u32 + 1) };
                }
                if let Jump::Location(loc) = closing_idx_ref {
                    *loc = unsafe { NonZeroU32::new_unchecked(i as u32 + 1) };
                }
            }
            _ => (),
        }
    }
    if !par_stack.is_empty() {
        unreachable!("mismatched brackets")
    }

    if config.verbose >= 3 || command == Command::Ir {
        cli::print_instructions(&instructions);
        if command == Command::Ir {
            return ExitCode::SUCCESS;
        } else {
            println!("============================================================");
        }
    }

    match command {
        Command::Format => unreachable!(),
        Command::Ir => unreachable!(),
        Command::Run => run(&instructions),
        Command::Compile => {
            let code = x86::compile(&instructions);
            let path: &Path = path.as_ref();
            let bin_path = path.with_extension("elf");
            let mut file = OpenOptions::new()
                .write(true)
                .truncate(true)
                .create(true)
                .mode(0o755)
                .open(bin_path)
                .unwrap();
            file.write_all(&code).unwrap();
        }
    }

    ExitCode::SUCCESS
}

fn run(instructions: &[Instruction]) {
    let mut ip = 0;
    let mut rp: i16 = 0;
    let mut registers = [0u8; NUM_REGISTERS];
    loop {
        let Some(inst) = instructions.get(ip) else {
            break;
        };

        match *inst {
            Instruction::Shl(n) => rp -= n as i16,
            Instruction::Shr(n) => rp += n as i16,
            Instruction::Inc(o, n) => {
                let r = &mut registers[(rp + o) as usize];
                *r = r.wrapping_add(n);
            }
            Instruction::Dec(o, n) => {
                let r = &mut registers[(rp + o) as usize];
                *r = r.wrapping_sub(n);
            }
            Instruction::Output => {
                _ = std::io::stdout().write(&registers[rp as usize..][..1]);
            }
            Instruction::Input => {
                _ = std::io::stdin().read(&mut registers[rp as usize..][..1]);
            }
            Instruction::JumpZ(Jump::Location(idx)) => {
                if registers[rp as usize] == 0 {
                    ip = idx.get() as usize;
                    continue;
                }
            }
            Instruction::JumpZ(Jump::Redundant) => (),
            Instruction::JumpNz(Jump::Location(idx)) => {
                if registers[rp as usize] > 0 {
                    ip = idx.get() as usize;
                    continue;
                }
            }
            Instruction::JumpNz(Jump::Redundant) => (),

            Instruction::Zero(o) => registers[(rp + o) as usize] = 0,
            Instruction::Set(o, n) => registers[(rp + o) as usize] = n,
            Instruction::Add(o) => {
                let val = registers[rp as usize];
                let r = &mut registers[(rp + o) as usize];
                *r = r.wrapping_add(val);
            }
            Instruction::Sub(o) => {
                let val = registers[rp as usize];
                let r = &mut registers[(rp + o) as usize];
                *r = r.wrapping_sub(val);
            }
            Instruction::AddMul(o, n) => {
                let val = registers[rp as usize];
                let r = &mut registers[(rp + o) as usize];
                *r = r.wrapping_add(n.wrapping_mul(val));
            }
            Instruction::SubMul(o, n) => {
                let val = registers[rp as usize];
                let r = &mut registers[(rp + o) as usize];
                *r = r.wrapping_sub(n.wrapping_mul(val));
            }
        }

        ip += 1;
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum IterationDiff {
    /// The change each loop iteration will have on the iteration register.
    Diff(i16),
    /// The loop always zeros the iteration register, this is equivalent to an if statement.
    Zeroed,
    /// The loop always zeros the iteration register, and then performs other operations on the
    /// iteration register. If the zeroed diff results in 0, this is also equivalent to an if
    /// statement, and the register is just used as temporary storage. Otherwise this is an
    /// infinite loop.
    ZeroedDiff(i16),
}

impl IterationDiff {
    fn inc(&mut self, inc: u8) {
        use IterationDiff::*;
        let inc = inc as i16;
        match self {
            Diff(d) | ZeroedDiff(d) => *d += inc,
            Zeroed => *self = ZeroedDiff(inc),
        }
    }

    fn dec(&mut self, dec: u8) {
        use IterationDiff::*;
        let dec = dec as i16;
        match self {
            Diff(d) | ZeroedDiff(d) => *d -= dec,
            Zeroed => *self = ZeroedDiff(-dec),
        }
    }

    fn zero(&mut self) {
        use IterationDiff::*;
        *self = Zeroed;
    }

    fn set(&mut self, n: u8) {
        use IterationDiff::*;
        *self = ZeroedDiff(n as i16);
    }
}

fn arithmetic_loop_pass(config: &Config, instructions: &mut Vec<Instruction>, i: usize) {
    use Instruction::*;

    let JumpZ(_) = instructions[i] else { return };

    let start = i + 1;
    let mut end = None;
    for (j, inst) in instructions[start..].iter().enumerate() {
        match inst {
            JumpZ(_) => break,
            JumpNz(jump) => {
                end = Some((jump, start + j));
                break;
            }
            _ => (),
        }
    }
    let Some((end_jump, end)) = end else { return };
    let inner = &instructions[start..end];
    let mut offset = 0;
    let mut num_arith = 0;
    let mut iteration_diff = IterationDiff::Diff(0);
    for inst in inner {
        match *inst {
            Shl(n) => offset -= n as i16,
            Shr(n) => offset += n as i16,
            Inc(o, n) => {
                if offset + o == 0 {
                    iteration_diff.inc(n);
                } else {
                    num_arith += 1;
                }
            }
            Dec(o, n) => {
                if offset + o == 0 {
                    iteration_diff.dec(n);
                } else {
                    num_arith += 1;
                }
            }
            Zero(o) => {
                if offset + o == 0 {
                    iteration_diff.zero();
                } else {
                    num_arith += 1;
                }
            }
            Set(o, n) => {
                if offset + o == 0 {
                    iteration_diff.set(n);
                } else {
                    num_arith += 1;
                }
            }
            Output | Input | JumpZ(_) | JumpNz(_) | Add(_) | Sub(_) | AddMul(..) | SubMul(..) => {
                return
            }
        }
    }

    if offset != 0 {
        return;
    }

    match iteration_diff {
        IterationDiff::Diff(-1) => (),
        IterationDiff::Zeroed | IterationDiff::ZeroedDiff(0) => {
            if config.o_jumps {
                let JumpNz(jump) = &mut instructions[end] else {
                    unreachable!();
                };
                *jump = Jump::Redundant;
                if config.verbose >= 2 {
                    println!("redundant jump if not zero at {}", end);
                }
            }
            return;
        }
        IterationDiff::Diff(0) | IterationDiff::ZeroedDiff(_) => {
            if !end_jump.is_redundant() {
                let range = start - 1..end + 1;
                let l = &instructions[range.clone()];
                warn!("infinite loop detected at {range:?}:\n{l:?}");
            }
            return;
        }
        IterationDiff::Diff(_) => return,
    }

    if !config.o_arithmetic {
        return;
    }

    let mut offset = 0;
    let mut replacements = Vec::with_capacity(num_arith + 1);
    for inst in inner.iter() {
        match *inst {
            Shl(n) => offset -= n as i16,
            Shr(n) => offset += n as i16,
            Inc(o, n) => {
                let offset = offset + o;
                if offset != 0 {
                    if let Some(set) = find_set_instruction_at_offset(&mut replacements, offset) {
                        *set.inst = Set(offset, set.prev_val.wrapping_add(n));
                    } else {
                        let replacement = match n {
                            1 => Add(offset),
                            _ => AddMul(offset, n),
                        };
                        replacements.push(replacement);
                    }
                }
            }
            Dec(o, n) => {
                let offset = offset + o;
                if offset != 0 {
                    if let Some(set) = find_set_instruction_at_offset(&mut replacements, offset) {
                        *set.inst = Set(offset, set.prev_val.wrapping_add(n));
                    } else {
                        let replacement = match n {
                            1 => Sub(offset),
                            _ => SubMul(offset, n),
                        };
                        replacements.push(replacement);
                    }
                }
            }
            Zero(o) => {
                let offset = offset + o;
                if offset != 0 {
                    if let Some(set) = find_set_instruction_at_offset(&mut replacements, offset) {
                        *set.inst = Zero(offset)
                    } else {
                        replacements.extend([
                            JumpZ(Jump::Location(NonZeroU32::MAX)),
                            Zero(offset),
                            JumpNz(Jump::Redundant),
                        ]);
                    }
                }
            }
            Set(o, n) => {
                let offset = offset + o;
                if offset != 0 {
                    if let Some(set) = find_set_instruction_at_offset(&mut replacements, offset) {
                        *set.inst = Set(offset, n);
                    } else {
                        replacements.extend([
                            JumpZ(Jump::Location(NonZeroU32::MAX)),
                            Zero(offset),
                            JumpNz(Jump::Redundant),
                        ]);
                    }
                }
            }
            Output | Input | JumpZ(_) | JumpNz(_) | Add(_) | Sub(_) | AddMul(..) | SubMul(..) => {
                unreachable!()
            }
        }
    }
    replacements.push(Zero(0));

    let range = start - 1..end + 1;
    if config.verbose >= 2 {
        println!("replaced {range:?} with {replacements:?}");
    }
    _ = instructions.splice(range, replacements);
}

struct SetInstruction<'a> {
    inst: &'a mut Instruction,
    prev_val: u8,
}

fn find_set_instruction_at_offset(
    instructions: &mut [Instruction],
    offset: i16,
) -> Option<SetInstruction<'_>> {
    for inst in instructions.iter_mut().rev() {
        match *inst {
            Instruction::Zero(o) if offset == o => {
                return Some(SetInstruction { inst, prev_val: 0 })
            }
            Instruction::Set(o, n) if offset == o => {
                return Some(SetInstruction { inst, prev_val: n })
            }
            _ => (),
        }
    }

    None
}

fn optimize_static_code(config: &Config, instructions: &mut Vec<Instruction>) {
    let mut registers = [0u8; NUM_REGISTERS];
    let mut rp: i16 = 0;
    let mut i = 0;
    while i < instructions.len() {
        match static_code_execution_pass(config, instructions, i, &mut registers, &mut rp) {
            ControlFlow::Continue(true) => i += 1,
            ControlFlow::Continue(false) => (),
            ControlFlow::Break(()) => {
                if i > 0 && config.o_init {
                    let all_set = instructions[0..i - 1]
                        .iter()
                        .all(|inst| matches!(inst, Instruction::Set(..)));
                    let last_shr = matches!(instructions[i - 1], Instruction::Shr(_));
                    if all_set && last_shr {
                        return;
                    }

                    let replacements = registers
                        .iter()
                        .enumerate()
                        .filter_map(|(i, n)| {
                            if *n == 0 {
                                return None;
                            }
                            Some(Instruction::Set(i as i16, *n))
                        })
                        .chain(Some(Instruction::Shr(rp as u16)))
                        .collect::<Vec<_>>();
                    if config.verbose >= 2 {
                        let range = 0..i;
                        println!("replaced {range:?} with static initialization {replacements:?}");
                    }
                    instructions.splice(0..i, replacements);
                }
                return;
            }
        }
    }

    // TODO: if all instructions could be executed at compile time the program doesn't produce any
    // output and could thus be empty
}

fn static_code_execution_pass(
    config: &Config,
    instructions: &mut Vec<Instruction>,
    i: usize,
    registers: &mut [u8; NUM_REGISTERS],
    rp: &mut i16,
) -> ControlFlow<(), bool> {
    let Some(inst) = instructions.get_mut(i) else {
        unreachable!()
    };

    match inst {
        Instruction::Shl(n) => *rp -= *n as i16,
        Instruction::Shr(n) => *rp += *n as i16,
        Instruction::Inc(o, n) => {
            let r = &mut registers[(*rp + *o) as usize];
            *r = r.wrapping_add(*n);
        }
        Instruction::Dec(o, n) => {
            let r = &mut registers[(*rp + *o) as usize];
            *r = r.wrapping_sub(*n);
        }
        Instruction::Output => return ControlFlow::Break(()),
        Instruction::Input => return ControlFlow::Break(()),
        Instruction::JumpZ(jump) => {
            let val = registers[*rp as usize];
            if val != 0 {
                if config.o_jumps {
                    if config.verbose >= 2 {
                        println!("redundant jump if zero at {}", i);
                    }
                    *jump = Jump::Redundant;
                }
                return ControlFlow::Break(());
            }
            if !config.o_dead_code {
                return ControlFlow::Break(());
            }

            remove_dead_code(config, instructions, i);
            return ControlFlow::Continue(false);
        }
        Instruction::JumpNz(_) => return ControlFlow::Break(()),

        Instruction::Zero(o) => registers[(*rp + *o) as usize] = 0,
        Instruction::Set(o, n) => registers[(*rp + *o) as usize] = *n,
        Instruction::Add(o) => {
            let val = registers[*rp as usize];
            let r = &mut registers[(*rp + *o) as usize];
            *r = r.wrapping_add(val);
        }
        Instruction::Sub(o) => {
            let val = registers[*rp as usize];
            let r = &mut registers[(*rp + *o) as usize];
            *r = r.wrapping_sub(val);
        }
        Instruction::AddMul(o, n) => {
            let val = registers[*rp as usize];
            let r = &mut registers[(*rp + *o) as usize];
            *r = r.wrapping_add(n.wrapping_mul(val));
        }
        Instruction::SubMul(o, n) => {
            let val = registers[*rp as usize];
            let r = &mut registers[(*rp + *o) as usize];
            *r = r.wrapping_sub(n.wrapping_mul(val));
        }
    }

    ControlFlow::Continue(true)
}

fn remove_dead_code(config: &Config, instructions: &mut Vec<Instruction>, start: usize) {
    let mut jump_stack = 0;

    for (i, inst) in instructions[start..].iter().enumerate() {
        match inst {
            Instruction::JumpZ(_) => jump_stack += 1,
            Instruction::JumpNz(_) => {
                jump_stack -= 1;
                if jump_stack == 0 {
                    let range = start..start + i + 1;
                    if config.verbose >= 2 {
                        println!("removed dead code at {range:?}");
                    }
                    instructions.drain(range);
                    return;
                }
            }
            _ => (),
        }
    }

    unreachable!()
}
