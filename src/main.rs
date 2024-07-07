use std::cmp::PartialOrd;
use std::env::args;
use std::io::{Read, Write};

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
    OffsetInc(i16, u8),
    OffsetDec(i16, u8),
    Output,
    Input,
    LSquare(u32),
    RSquare(u32),
}

fn main() {
    let path = args().skip(1).next().unwrap();
    let input = std::fs::read_to_string(&path).unwrap();
    let bytes = input.as_bytes();

    let mut ip = 0;

    const LEN: usize = 30000;
    let mut pointer: usize = 0;
    let mut registers: [u8; LEN] = [0; LEN];

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
        .map(|chunk| match chunk[0] {
            Token::Shl => Instruction::Shl(chunk.len() as u16),
            Token::Shr => Instruction::Shr(chunk.len() as u16),
            Token::Inc => Instruction::Inc(chunk.len() as u8),
            Token::Dec => Instruction::Dec(chunk.len() as u8),
            Token::Output => Instruction::Output,
            Token::Input => Instruction::Input,
            Token::LSquare => Instruction::LSquare(0),
            Token::RSquare => Instruction::RSquare(0),
        })
        .collect::<Vec<_>>();

    // offset instructions
    {
        let mut i = 0;
        while i + 3 < instructions.len() - 3 {
            let [a, b, c] = &instructions[i..i + 3] else {
                unreachable!()
            };
            match (a, b, c) {
                (Instruction::Shl(l), inst, Instruction::Shr(r)) if l == r => {
                    let offset_instruction = match inst {
                        Instruction::Inc(n) => Instruction::OffsetInc(-(*l as i16), *n),
                        Instruction::Dec(n) => Instruction::OffsetDec(-(*l as i16), *n),
                        _ => {
                            i += 1;
                            continue;
                        }
                    };
                    instructions.drain(i..i + 3);
                    instructions.insert(i, offset_instruction);
                    println!("replaced {i}..{}", i + 3);
                }
                (Instruction::Shr(r), inst, Instruction::Shl(l)) if r == l => {
                    let offset_instruction = match inst {
                        Instruction::Inc(n) => Instruction::OffsetInc(*l as i16, *n),
                        Instruction::Dec(n) => Instruction::OffsetDec(*l as i16, *n),
                        _ => {
                            i += 1;
                            continue;
                        }
                    };
                    instructions.drain(i..i + 3);
                    instructions.insert(i, offset_instruction);
                    println!("replaced {i}..{}", i + 3);
                }
                _ => (),
            }

            i += 1;
        }
    }

    // update jump indices
    let mut par_stack = Vec::new();
    for (i, instruction) in instructions.iter_mut().enumerate() {
        match instruction {
            Instruction::LSquare(closing_idx_ref) => par_stack.push((i, closing_idx_ref)),
            Instruction::RSquare(opening_idx_ref) => {
                let Some((opening_idx, closing_idx_ref)) = par_stack.pop() else {
                    unreachable!("mismatched parentheses")
                };

                *opening_idx_ref = opening_idx as u32 + 1;
                *closing_idx_ref = i as u32 + 1;
            }
            _ => (),
        }
    }

    loop {
        let Some(b) = instructions.get(ip) else {
            break;
        };

        match *b {
            Instruction::Shl(n) => pointer -= n as usize,
            Instruction::Shr(n) => pointer += n as usize,
            Instruction::Inc(n) => registers[pointer] += n,
            Instruction::Dec(n) => registers[pointer] -= n,
            Instruction::OffsetInc(i, n) => {
                registers[(pointer as isize + i as isize) as usize] += n
            }
            Instruction::OffsetDec(i, n) => {
                registers[(pointer as isize + i as isize) as usize] -= n
            }
            Instruction::Output => {
                _ = std::io::stdout().write(&registers[pointer..pointer + 1]);
            }
            Instruction::Input => {
                _ = std::io::stdin().read(&mut registers[pointer..pointer + 1]);
            }
            Instruction::LSquare(idx) => {
                if registers[pointer] == 0 {
                    ip = idx as usize;
                    continue;
                }
            }
            Instruction::RSquare(idx) => {
                if registers[pointer] > 0 {
                    ip = idx as usize;
                    continue;
                }
            }
        }

        ip += 1;
    }
}
