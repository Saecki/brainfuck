use std::ops::ControlFlow;
use std::path::PathBuf;
use std::process::ExitCode;

use crate::Instruction;

pub const ANSII_CLEAR: &str = "\x1b[0m";
pub const ANSII_UNDERLINED: &str = "\x1b[4m";
pub const ANSII_COLOR_RED: &str = "\x1b[91m";
pub const ANSII_COLOR_YELLOW: &str = "\x1b[93m";

pub struct Config {
    pub verbose: u8,
    pub print_unoptimized_ir: bool,
    pub optimize: bool,
    pub o_zeros: bool,
    pub o_arithmetic: bool,
    pub o_jumps: bool,
    pub o_dead_code: bool,
    pub o_init: bool,
    pub o_simplify: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Command {
    Format,
    Ir,
    Run,
    Compile,
}

macro_rules! input_error {
    ($pat:expr) => {{
        eprint!("{ANSII_COLOR_RED}argument error: ");
        eprint!($pat);
        eprintln!("{ANSII_CLEAR}");
        eprintln!();
        print_help();
        return ControlFlow::Break(ExitCode::FAILURE);
    }};
}

pub fn parse_args() -> ControlFlow<ExitCode, (Config, Command, PathBuf)> {
    let mut args = std::env::args();
    _ = args.next();
    let command = match args.next().as_deref() {
        Some("format") => Command::Format,
        Some("ir") => Command::Ir,
        Some("run") => Command::Run,
        Some("compile") => Command::Compile,
        Some("help") => {
            print_help();
            return ControlFlow::Break(ExitCode::SUCCESS);
        }
        Some(a) => {
            input_error!("invalid command: `{a}`");
        }
        None => {
            input_error!("missing first positional argument <command>");
        }
    };

    let mut path = None;
    let mut config = Config {
        verbose: 0,
        print_unoptimized_ir: false,
        optimize: true,
        o_zeros: true,
        o_arithmetic: true,
        o_jumps: true,
        o_dead_code: true,
        o_init: true,
        o_simplify: true,
    };
    for a in args {
        if let Some(n) = a.strip_prefix("--") {
            match n {
                "verbose" => config.verbose += 1,
                "print-unoptimized-ir" => config.print_unoptimized_ir = true,
                "debug" => config.optimize = false,
                "no-optimize-zeroes" => config.o_zeros = false,
                "no-optimize-arithmetic" => config.o_arithmetic = false,
                "no-optimize-jumps" => config.o_jumps = false,
                "no-optimize-dead-code" => config.o_dead_code = false,
                "no-optimize-init" => config.o_init = false,
                "no-optimize-simplify" => config.o_simplify = false,
                _ => input_error!("unexpected argument `{a}`"),
            }
        } else if let Some(n) = a.strip_prefix('-') {
            for c in n.chars() {
                match c {
                    'v' => config.verbose += 1,
                    'u' => config.print_unoptimized_ir = true,
                    'd' => config.optimize = false,
                    _ => input_error!("unexpected flag `{c}`"),
                }
            }
        } else {
            if path.is_some() {
                input_error!("unexpected positional argument `{a}`");
            }
            path = Some(a);
        }
    }
    let Some(path) = path else {
        input_error!("missing second positional argument <path>");
    };

    ControlFlow::Continue((config, command, path.into()))
}

fn print_help() {
    eprintln!(
        "\
brainfuck <mode> [<option>] <path>

{ANSII_UNDERLINED}modes{ANSII_CLEAR}
    format          pretty print brainfuck code
    ir              print the intermediate representation
    run             interpret the ir
    compile         generate an ELF64 x86-64 system-v executable
    help            print this help message

{ANSII_UNDERLINED}options{ANSII_CLEAR}
    -v,--verbose                change verbosity level via number of occurences [0..=3]
    -u,--print-unoptimized-ir   print the ir before optimizations are applied
    -d,--debug                  disable all optimizations
       --no-optimize-zeros      disable zeroing optimization
       --no-optimize-arithmetic disable arithmetic optimizations
       --no-optimize-jumps      disable redundant jump elmination
       --no-optimize-dead-code  disable dead code elmination
       --no-optimize-init       disable state initialization optimization
       --no-optimize-simplify   disable code simplification
    "
    );
}

pub fn print_brainfuck_code(instructions: &[Instruction]) {
    let mut indent = 0;
    for inst in instructions.iter() {
        if let Instruction::JumpNz(_) = inst {
            indent -= 1
        }
        for _ in 0..indent {
            print!("    ");
        }
        match *inst {
            Instruction::Shl(n) => println!("{:<<width$}", "", width = n as usize),
            Instruction::Shr(n) => println!("{:><width$}", "", width = n as usize),
            Instruction::Inc(0, n) => println!("{:+<width$}", "", width = n as usize),
            Instruction::Inc(_, _) => unreachable!(),
            Instruction::Dec(0, n) => println!("{:-<width$}", "", width = n as usize),
            Instruction::Dec(_, _) => unreachable!(),
            Instruction::Output => println!("."),
            Instruction::Input => println!(","),
            Instruction::JumpZ(_) => println!("["),
            Instruction::JumpNz(_) => println!("]"),

            Instruction::Zero(_) => unreachable!(),
            Instruction::Set(_, _) => unreachable!(),
            Instruction::Add(_) => unreachable!(),
            Instruction::Sub(_) => unreachable!(),
            Instruction::AddMul(_, _) => unreachable!(),
            Instruction::SubMul(_, _) => unreachable!(),
        }
        if let Instruction::JumpZ(_) = inst {
            indent += 1
        }
    }
}

pub fn print_instructions(instructions: &[Instruction]) {
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
