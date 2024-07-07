use std::env::args;
use std::io::{Read, Write};
use std::cmp::PartialOrd;

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

fn main() {
    let path = args().skip(1).next().unwrap();
    let input = std::fs::read_to_string(&path).unwrap();
    let bytes = input.as_bytes();

    let mut token_cursor = 0;

    const LEN: usize = 30000;
    let mut pointer: usize = 0;
    let mut registers: [u8; LEN] = [0; LEN];

    let tokens = bytes.iter().filter_map(|b| {
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
    }).collect::<Vec<_>>();

    loop {
        let Some(b) = tokens.get(token_cursor) else {
            break;
        };

        match *b {
            Token::Shl => {
                pointer -= 1;
            }
            Token::Shr => {
                pointer += 1;
            }
            Token::Inc => {
                registers[pointer] += 1;
            }
            Token::Dec => {
                registers[pointer] -= 1;
            }
            Token::Output => {
                _ = std::io::stdout().write(&registers[pointer..pointer + 1]);
            }
            Token::Input => {
                _ = std::io::stdin().read(&mut registers[pointer..pointer + 1]);
            }
            Token::LSquare => {
                if registers[pointer] == 0 {
                    let mut par_stack = 1;
                    token_cursor += 1;
                    for (i, b) in tokens[token_cursor..].iter().enumerate() {
                        match b {
                            Token::LSquare => {
                                par_stack += 1;
                            }
                            Token::RSquare => {
                                par_stack -= 1;
                                if par_stack == 0 {
                                    token_cursor += i;
                                    break;
                                }
                            }
                            _ => (),
                        }
                    }
                    continue;
                }
            }
            Token::RSquare => {
                if registers[pointer] > 0 {
                    let mut par_stack = 1;
                    token_cursor -= 1;
                    for (i, b) in tokens[..=token_cursor].iter().rev().enumerate() {
                        match b {
                            Token::RSquare => {
                                par_stack += 1;
                            }
                            Token::LSquare => {
                                par_stack -= 1;
                                if par_stack == 0 {
                                    token_cursor -= i;
                                    break;
                                }
                            }
                            _ => (),
                        }
                    }
                    continue;
                }
            }
        }

        token_cursor += 1;
    }
}
