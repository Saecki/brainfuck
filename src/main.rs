use std::cmp::PartialOrd;
use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::num::NonZeroU32;
use std::ops::ControlFlow;
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;
use std::process::ExitCode;

use crate::cli::Command;

pub mod cli;
pub mod optim;
pub mod x86;

const NUM_REGISTERS: usize = 1 << 15;

#[macro_export]
macro_rules! warn {
    ($pat:expr) => {{
        use crate::cli::{ANSII_CLEAR, ANSII_COLOR_YELLOW};
        eprint!("{ANSII_COLOR_YELLOW}warning{ANSII_CLEAR}: ");
        eprintln!($pat);
    }};
}

#[macro_export]
macro_rules! error {
    ($pat:expr) => {{
        use crate::cli::{ANSII_CLEAR, ANSII_COLOR_RED};
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
            if config.verbose >= 3 && c.len() > 1 {
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
            optim::replace_zeros(&config, &mut instructions);
        }

        if config.o_dead_code || config.o_init || config.o_jumps {
            optim::optimize_static_code(&config, &mut instructions);
        }

        if config.o_jumps {
            optim::remove_redundant_jump_pairs(&config, &mut instructions);
        }

        if config.o_arithmetic || config.o_jumps {
            let mut i = 0;
            while i < instructions.len() {
                optim::arithmetic_loop_pass(&config, &mut instructions, i);
                i += 1;
            }
        }

        if config.o_simplify {
            optim::simplify_code(&config, &mut instructions);
        }

        if config.o_dead_code || config.o_init || config.o_jumps {
            optim::optimize_static_code(&config, &mut instructions);
        }

        if config.o_jumps {
            optim::remove_redundant_jump_pairs(&config, &mut instructions);
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
    let mut jump_stack = Vec::new();
    for (i, instruction) in instructions.iter_mut().enumerate() {
        match instruction {
            Instruction::JumpZ(closing_idx_ref) => jump_stack.push((i, closing_idx_ref)),
            Instruction::JumpNz(opening_idx_ref) => {
                let Some((opening_idx, closing_idx_ref)) = jump_stack.pop() else {
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
    if !jump_stack.is_empty() {
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
            let code = x86::compile(&config, &instructions);
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
