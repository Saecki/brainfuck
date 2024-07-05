use std::env::args;
use std::io::{Read, Write};

fn main() {
    let path = args().skip(1).next().unwrap();
    let input = std::fs::read_to_string(&path).unwrap();
    let bytes = input.as_bytes();

    let mut byte_cursor = 0;

    const LEN: usize = 30000;
    let mut pointer: usize = 0;
    let mut registers: [u8; LEN] = [0; LEN];

    loop {
        let Some(b) = bytes.get(byte_cursor) else {
            break;
        };

        match *b {
            b'<' => {
                pointer -= 1;
            }
            b'>' => {
                pointer += 1;
            }
            b'+' => {
                registers[pointer] += 1;
            }
            b'-' => {
                registers[pointer] -= 1;
            }
            b'.' => {
                _ = std::io::stdout().write(&registers[pointer..pointer + 1]);
            }
            b',' => {
                _ = std::io::stdin().read(&mut registers[pointer..pointer + 1]);
            }
            b'[' => {
                if registers[pointer] == 0 {
                    let mut par_stack = 1;
                    byte_cursor += 1;
                    for (i, b) in bytes[byte_cursor..].iter().enumerate() {
                        match b {
                            b'[' => {
                                par_stack += 1;
                            }
                            b']' => {
                                par_stack -= 1;
                                if par_stack == 0 {
                                    byte_cursor += i;
                                    break;
                                }
                            }
                            _ => (),
                        }
                    }
                    continue;
                }
            }
            b']' => {
                if registers[pointer] > 0 {
                    let mut par_stack = 1;
                    byte_cursor -= 1;
                    for (i, b) in bytes[..=byte_cursor].iter().rev().enumerate() {
                        match b {
                            b']' => {
                                par_stack += 1;
                            }
                            b'[' => {
                                par_stack -= 1;
                                if par_stack == 0 {
                                    byte_cursor -= i;
                                    break;
                                }
                            }
                            _ => (),
                        }
                    }
                    continue;
                }
            }
            _ => (),
        }

        byte_cursor += 1;
    }
}
