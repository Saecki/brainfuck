use std::cmp::PartialOrd;
use std::io::{Read, Write};
use std::process::ExitCode;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Token {
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
    fn is_combinable(self) -> bool {
        match self {
            Token::Shl | Token::Shr | Token::Inc | Token::Dec => true,
            Token::Output | Token::Input | Token::LSquare | Token::RSquare => false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Instruction {
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
    Zero,
    // additional instructions
    /// Add current register value to register at offset, clears the current register.
    /// Equivalent to the following code for `Add(1)`:
    /// ```bf
    /// [
    ///     >
    ///     +
    ///     <
    ///     -
    /// ]
    /// ```
    Add(i16),
    /// Subtract current register value from register at offset, clears the current register.
    /// Equivalent to the following code for `Sub(1)`:
    /// ```bf
    /// [
    ///     >
    ///     -
    ///     <
    ///     -
    /// ]
    /// ```
    Sub(i16),
    /// Multiply current register value and add to register at offset, clears the current register.
    /// Equivalent to the following code for `AddMul(1, -5)`:
    /// ```bf
    /// [
    ///     >
    ///     -----
    ///     <
    ///     -
    /// ]
    /// ```
    AddMul(i16, i16),
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

            Instruction::Zero => write!(f, "zero"),
            Instruction::Add(i) => write!(f, "<{i}> +="),
            Instruction::Sub(i) => write!(f, "<{i}> -="),
            Instruction::AddMul(i, n) => write!(f, "<{i}> +=*({n})"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Command {
    Format,
    Ir,
    Run,
}

fn main() -> ExitCode {
    let mut args = std::env::args();
    _ = args.next();
    let command = match args.next().as_deref() {
        Some("format") => Command::Format,
        Some("ir") => Command::Ir,
        Some("run") => Command::Run,
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
    let mut verbose = false;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--verbose" => verbose = true,
            _ if a.starts_with("-") => {
                eprintln!("unexpected argument `{a}`");
                return ExitCode::FAILURE;
            }
            _ => {
                if path.is_some() {
                    eprintln!("unexpected positional argument `{a}`");
                    return ExitCode::FAILURE;
                }
                path = Some(a);
            }
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
            if verbose && c.len() > 1 {
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
    if verbose {
        println!("============================================================");
        println!(
            "tokens before {} after: {} ({:.3}%)",
            tokens.len(),
            instructions.len(),
            100.0 * instructions.len() as f32 / tokens.len() as f32,
        );
        println!("============================================================");
    }
    if verbose || command == Command::Format {
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
        while i + 3 < instructions.len() - 3 {
            let [a, b, c] = &instructions[i..i + 3] else {
                unreachable!()
            };
            match (a, b, c) {
                (JumpZ(_), Dec(1), JumpNz(_)) => {
                    let range = i..i + 3;
                    if verbose {
                        println!("replaced {range:?} with zero");
                    }
                    instructions.drain(range);
                    instructions.insert(i, Zero);
                }
                _ => (),
            }

            i += 1;
        }
    }
    // arithmetic instructions
    {
        use Instruction::*;

        let mut i = 0;
        while i + 6 < instructions.len() - 6 {
            let [a, b, c, d, e, f] = &instructions[i..i + 6] else {
                unreachable!()
            };
            let (offset, inst) = match (a, b, c, d, e, f) {
                (JumpZ(_), Shr(r), inst, Shl(l), Dec(1), JumpNz(_))
                | (JumpZ(_), Dec(1), Shr(r), inst, Shl(l), JumpNz(_))
                    if r == l =>
                {
                    (*r as i16, inst)
                }

                (JumpZ(_), Shl(l), inst, Shr(r), Dec(1), JumpNz(_))
                | (JumpZ(_), Dec(1), Shl(l), inst, Shr(r), JumpNz(_))
                    if l == r =>
                {
                    (-(*l as i16), inst)
                }

                _ => {
                    i += 1;
                    continue;
                }
            };

            let replacement = match inst {
                Inc(1) => Add(offset),
                Dec(1) => Sub(offset),
                Inc(n) => AddMul(offset, *n as i16),
                Dec(n) => AddMul(offset, -(*n as i16)),
                _ => {
                    i += 1;
                    continue;
                }
            };

            let range = i..i + 6;
            if verbose {
                println!("replaced {range:?} with {replacement:?}");
            }
            instructions.drain(range);
            instructions.insert(i, replacement);

            i += 1;
        }
    }
    if verbose {
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

    if verbose || command == Command::Ir {
        print_instructions(&instructions);
        if command == Command::Ir {
            return ExitCode::SUCCESS;
        } else {
            println!("============================================================");
        }
    }

    const LEN: usize = 30000;
    let mut ip = 0;
    let mut pointer: usize = 0;
    let mut registers: [u8; LEN] = [0; LEN];
    loop {
        let Some(b) = instructions.get(ip) else {
            break;
        };

        match *b {
            Instruction::Shl(n) => pointer -= n as usize,
            Instruction::Shr(n) => pointer += n as usize,
            Instruction::Inc(n) => registers[pointer] = registers[pointer].wrapping_add(n),
            Instruction::Dec(n) => registers[pointer] = registers[pointer].wrapping_sub(n),
            Instruction::Output => {
                _ = std::io::stdout().write(&registers[pointer..pointer + 1]);
            }
            Instruction::Input => {
                _ = std::io::stdin().read(&mut registers[pointer..pointer + 1]);
            }
            Instruction::JumpZ(idx) => {
                if registers[pointer] == 0 {
                    ip = idx as usize;
                    continue;
                }
            }
            Instruction::JumpNz(idx) => {
                if registers[pointer] > 0 {
                    ip = idx as usize;
                    continue;
                }
            }

            Instruction::Zero => registers[pointer] = 0,
            Instruction::Add(i) => {
                let val = registers[pointer];
                let r = &mut registers[(pointer as isize + i as isize) as usize];
                *r = r.wrapping_add(val);
                registers[pointer] = 0;
            }
            Instruction::Sub(i) => {
                let val = registers[pointer];
                let r = &mut registers[(pointer as isize + i as isize) as usize];
                *r = r.wrapping_sub(val);
                registers[pointer] = 0;
            }
            Instruction::AddMul(i, n) => {
                let val = n.wrapping_mul(registers[pointer] as i16);
                let r = &mut registers[(pointer as isize + i as isize) as usize];
                *r = r.wrapping_add(val as u8);
                registers[pointer] = 0;
            }
        }

        ip += 1;
    }

    ExitCode::SUCCESS
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

            Instruction::Zero => unreachable!(),
            Instruction::Add(_) => unreachable!(),
            Instruction::Sub(_) => unreachable!(),
            Instruction::AddMul(_, _) => unreachable!(),
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
