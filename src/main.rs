use std::cmp::PartialOrd;
use std::io::{Read, Write};
use std::path::Path;
use std::process::ExitCode;

pub mod x86;

const NUM_REGISTERS: usize = 30000;

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
    Inc(u8),
    Dec(u8),
    Output,
    Input,
    /// Jump to the position if the current register value is zero.
    JumpZ(u32),
    /// Jump to the position if the current register value is not zero.
    JumpNz(u32),

    /// Clear the current register:
    /// ```bf
    /// [
    ///     -
    /// ]
    /// ```
    Zero(i16),
    /// Add current register value to register at offset.
    Add(i16),
    /// Subtract current register value from register at offset.
    Sub(i16),
    /// Multiply current register value and add to register at offset.
    AddMul(i16, u8),
    /// Multiply current register value and subtraction from register at offset.
    SubMul(i16, u8),
}

impl std::fmt::Display for Instruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Instruction::Shl(n) => write!(f, "< ({n})"),
            Instruction::Shr(n) => write!(f, "> ({n})"),
            Instruction::Inc(n) => write!(f, "+ ({n})"),
            Instruction::Dec(n) => write!(f, "- ({n})"),
            Instruction::Output => write!(f, "out"),
            Instruction::Input => write!(f, "in"),
            Instruction::JumpZ(_) => write!(f, "["),
            Instruction::JumpNz(_) => write!(f, "]"),

            Instruction::Zero(o) => write!(f, "<{o}> zero"),
            Instruction::Add(o) => write!(f, "<{o}> add"),
            Instruction::Sub(o) => write!(f, "<{o}> sub"),
            Instruction::AddMul(o, n) => write!(f, "<{o}> addmul({n})"),
            Instruction::SubMul(o, n) => write!(f, "<{o}> submul({n})"),
        }
    }
}

struct Config {
    verbose: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Command {
    Format,
    Ir,
    Run,
    Compile,
}

fn main() -> ExitCode {
    let mut args = std::env::args();
    _ = args.next();
    let command = match args.next().as_deref() {
        Some("format") => Command::Format,
        Some("ir") => Command::Ir,
        Some("run") => Command::Run,
        Some("compile") => Command::Compile,
        Some(a) => {
            eprintln!("invalid command: `{a}`");
            return ExitCode::FAILURE;
        }
        None => {
            eprintln!("missing first positional argument <command>");
            return ExitCode::FAILURE;
        }
    };

    let mut path = None;
    let mut config = Config { verbose: 0 };
    for a in args {
        if let Some(n) = a.strip_prefix("--") {
            match n {
                "verbose" => config.verbose += 1,
                _ => {
                    eprintln!("unexpected argument `{a}`");
                    return ExitCode::FAILURE;
                }
            }
        } else if let Some(n) = a.strip_prefix('-') {
            for c in n.chars() {
                match c {
                    'v' => config.verbose += 1,
                    _ => {
                        eprintln!("unexpected flag `{c}`");
                        return ExitCode::FAILURE;
                    }
                }
            }
        } else {
            if path.is_some() {
                eprintln!("unexpected positional argument `{a}`");
                return ExitCode::FAILURE;
            }
            path = Some(a);
        }
    }
    let Some(path) = path else {
        eprintln!("missing second positional argument <path>");
        return ExitCode::FAILURE;
    };
    let input = std::fs::read_to_string(&path).unwrap();
    let bytes = input.as_bytes();

    let tokens = bytes
        .iter()
        .filter_map(|b| {
            let t = match *b {
                b'<' => Token::Shl,
                b'>' => Token::Shr,
                b'+' => Token::Inc,
                b'-' => Token::Dec,
                b'.' => Token::Output,
                b',' => Token::Input,
                b'[' => Token::LSquare,
                b']' => Token::RSquare,
                _ => return None,
            };
            Some(t)
        })
        .collect::<Vec<_>>();

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
            Token::Inc => Instruction::Inc(chunk.len() as u8),
            Token::Dec => Instruction::Dec(chunk.len() as u8),
            Token::Output => Instruction::Output,
            Token::Input => Instruction::Input,
            Token::LSquare => Instruction::JumpZ(0),
            Token::RSquare => Instruction::JumpNz(0),
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
        print_brainfuck_code(&instructions);
        if command == Command::Format {
            return ExitCode::SUCCESS;
        }
    }

    let prev_len = instructions.len();
    // zero register
    {
        use Instruction::*;

        let mut i = 0;
        while i + 3 < instructions.len() {
            let [a, b, c] = &instructions[i..i + 3] else {
                unreachable!()
            };
            if let (JumpZ(_), Dec(1), JumpNz(_)) = (a, b, c) {
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
    // arithmetic instructions
    {
        let mut i = 0;
        while i < instructions.len() {
            arithmetic_loop_pass(&config, &mut instructions, i);
            i += 1;
        }
    }
    if config.verbose >= 1 {
        println!("============================================================");
        println!(
            "instructions before {} after: {} ({:.3}%)",
            prev_len,
            instructions.len(),
            100.0 * instructions.len() as f32 / prev_len as f32,
        );
        println!("============================================================");
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

                *opening_idx_ref = opening_idx as u32 + 1;
                *closing_idx_ref = i as u32 + 1;
            }
            _ => (),
        }
    }
    if !par_stack.is_empty() {
        unreachable!("mismatched brackets")
    }

    if config.verbose >= 3 || command == Command::Ir {
        print_instructions(&instructions);
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
            std::fs::write(bin_path, &code).unwrap();
        }
    }

    ExitCode::SUCCESS
}

fn run(instructions: &[Instruction]) {
    let mut ip = 0;
    let mut rp: usize = 0;
    let mut registers = [0u8; NUM_REGISTERS];
    loop {
        let Some(b) = instructions.get(ip) else {
            break;
        };

        match *b {
            Instruction::Shl(n) => rp -= n as usize,
            Instruction::Shr(n) => rp += n as usize,
            Instruction::Inc(n) => registers[rp] = registers[rp].wrapping_add(n),
            Instruction::Dec(n) => registers[rp] = registers[rp].wrapping_sub(n),
            Instruction::Output => {
                _ = std::io::stdout().write(&registers[rp..rp + 1]);
            }
            Instruction::Input => {
                _ = std::io::stdin().read(&mut registers[rp..rp + 1]);
            }
            Instruction::JumpZ(idx) => {
                if registers[rp] == 0 {
                    ip = idx as usize;
                    continue;
                }
            }
            Instruction::JumpNz(idx) => {
                if registers[rp] > 0 {
                    ip = idx as usize;
                    continue;
                }
            }

            Instruction::Zero(o) => registers[(rp as isize + o as isize) as usize] = 0,
            Instruction::Add(o) => {
                let val = registers[rp];
                let r = &mut registers[(rp as isize + o as isize) as usize];
                *r = r.wrapping_add(val);
            }
            Instruction::Sub(o) => {
                let val = registers[rp];
                let r = &mut registers[(rp as isize + o as isize) as usize];
                *r = r.wrapping_sub(val);
            }
            Instruction::AddMul(o, n) => {
                let val = n.wrapping_mul(registers[rp]);
                let r = &mut registers[(rp as isize + o as isize) as usize];
                *r = r.wrapping_add(val);
            }
            Instruction::SubMul(o, n) => {
                let val = n.wrapping_mul(registers[rp]);
                let r = &mut registers[(rp as isize + o as isize) as usize];
                *r = r.wrapping_sub(val);
            }
        }

        ip += 1;
    }
}

fn print_brainfuck_code(instructions: &[Instruction]) {
    let mut indent = 0;
    for i in instructions.iter() {
        if let Instruction::JumpNz(_) = i {
            indent -= 1
        }
        for _ in 0..indent {
            print!("    ");
        }
        match i {
            Instruction::Shl(n) => println!("{:<<width$}", "", width = *n as usize),
            Instruction::Shr(n) => println!("{:><width$}", "", width = *n as usize),
            Instruction::Inc(n) => println!("{:+<width$}", "", width = *n as usize),
            Instruction::Dec(n) => println!("{:-<width$}", "", width = *n as usize),
            Instruction::Output => println!("."),
            Instruction::Input => println!(","),
            Instruction::JumpZ(_) => println!("["),
            Instruction::JumpNz(_) => println!("]"),

            Instruction::Zero(_) => unreachable!(),
            Instruction::Add(_) => unreachable!(),
            Instruction::Sub(_) => unreachable!(),
            Instruction::AddMul(_, _) => unreachable!(),
            Instruction::SubMul(_, _) => unreachable!(),
        }
        if let Instruction::JumpZ(_) = i {
            indent += 1
        }
    }
}

fn print_instructions(instructions: &[Instruction]) {
    let mut indent = 0;
    for i in instructions.iter() {
        if let Instruction::JumpNz(_) = i {
            indent -= 1
        }
        for _ in 0..indent {
            print!("    ");
        }
        println!("{i}");
        if let Instruction::JumpZ(_) = i {
            indent += 1
        }
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
}

fn arithmetic_loop_pass(config: &Config, instructions: &mut Vec<Instruction>, i: usize) {
    use Instruction::*;

    let JumpZ(_) = instructions[i] else { return };

    let start = i + 1;
    let mut end = None;
    for (j, inst) in instructions[start..].iter().enumerate() {
        match inst {
            JumpZ(_) => break,
            JumpNz(_) => {
                end = Some(start + j);
                break;
            }
            _ => (),
        }
    }
    let Some(end) = end else { return };
    let inner = &instructions[start..end];
    let mut offset = 0;
    let mut num_arith = 0;
    let mut iteration_diff = IterationDiff::Diff(0);
    for inst in inner {
        match inst {
            Shl(n) => offset -= *n as i16,
            Shr(n) => offset += *n as i16,
            Inc(n) => {
                if offset == 0 {
                    iteration_diff.inc(*n);
                } else {
                    num_arith += 1;
                }
            }
            Dec(n) => {
                if offset == 0 {
                    iteration_diff.dec(*n);
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
        IterationDiff::Diff(_) => return,
        // TODO: consider removing trailing check
        IterationDiff::Zeroed => return,
        // TODO: consider removing trailing check
        IterationDiff::ZeroedDiff(0) => return,
        IterationDiff::ZeroedDiff(_) => {
            let range = start - 1..end + 1;
            let l = &instructions[range.clone()];
            println!("\x1b[93mwarning\x1b[0m: infinite loop detected at {range:?}:\n{l:?}");
            return;
        }
    }

    let mut offset = 0;
    let mut replacements = Vec::with_capacity(num_arith + 1);
    for inst in inner.iter() {
        match inst {
            Shl(n) => offset -= *n as i16,
            Shr(n) => offset += *n as i16,
            Inc(n) => {
                if offset != 0 {
                    let replacement = match n {
                        1 => Add(offset),
                        _ => AddMul(offset, *n),
                    };
                    replacements.push(replacement);
                }
            }
            Dec(n) => {
                if offset != 0 {
                    let replacement = match n {
                        1 => Sub(offset),
                        _ => SubMul(offset, *n),
                    };
                    replacements.push(replacement);
                }
            }
            Zero(o) => {
                if offset + o != 0 {
                    replacements.push(Zero(offset + o));
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
