use std::num::NonZeroU32;
use std::ops::ControlFlow;

use crate::cli::Config;
use crate::{warn, Instruction, Jump, NUM_REGISTERS};

enum IndexInc {
    Zero = 0,
    One = 1,
}

pub fn replace_zeros(config: &Config, instructions: &mut Vec<Instruction>) {
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Update {
    /// A relative change to a register value.
    /// Just [`Instruction::Inc`] and [`Instruction::Dec`].
    Relative(i8),
    /// An absolute change to a register value.
    /// A [`Instruction::Zero`] or [`Instruction::Set`] and then optional
    /// [`Instruction::Inc`] or [`Instruction::Dec`]
    Absolute(i8),
}

impl Update {
    fn inc(&mut self, inc: u8) {
        use Update::*;
        match self {
            Relative(d) => *d = d.wrapping_add(inc as i8),
            Absolute(d) => *d = d.wrapping_add(inc as i8),
        }
    }

    fn dec(&mut self, dec: u8) {
        use Update::*;
        match self {
            Relative(d) => *d = d.wrapping_sub(dec as i8),
            Absolute(d) => *d = d.wrapping_sub(dec as i8),
        }
    }

    fn set(&mut self, n: u8) {
        use Update::*;
        *self = Absolute(n as i8);
    }
}

pub fn arithmetic_loop_pass(config: &Config, instructions: &mut Vec<Instruction>, i: usize) {
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
    let mut iteration_diff = Update::Relative(0);
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
                    iteration_diff.set(0);
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
        // The loop is decremented by 1 in each iteration
        Update::Relative(-1) => (),
        // The loop always zeros the iteration register
        Update::Absolute(0) => {
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
        // The loop doesn't alter the value inside the iteration register, or sets it to a constant
        // non zero value
        Update::Relative(0) | Update::Absolute(_) => {
            if !end_jump.is_redundant() {
                let range = start - 1..end + 1;
                let l = &instructions[range.clone()];
                warn!("infinite loop detected at {range:?}:\n{l:?}");
            }
            return;
        }
        // An relative increment other than -1, the loop might never complete depending on how the
        // iteration register was initialized
        // TODO: handle +1 case
        Update::Relative(_) => return,
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

pub fn optimize_static_code(config: &Config, instructions: &mut Vec<Instruction>) {
    let mut registers = [0u8; NUM_REGISTERS];
    let mut rp: i16 = 0;
    let mut i = 0;
    while i < instructions.len() {
        match static_code_execution_pass(config, instructions, i, &mut registers, &mut rp) {
            ControlFlow::Continue(index_inc) => i += index_inc as usize,
            ControlFlow::Break(()) => {
                if i > 0 && config.o_init {
                    let all_set = instructions[0..i - 1]
                        .iter()
                        .all(|inst| matches!(inst, Instruction::Set(..)));
                    let last_set_or_shr = matches!(
                        instructions[i - 1],
                        Instruction::Set(..) | Instruction::Shr(_)
                    );
                    if all_set && last_set_or_shr {
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
                        .chain((rp != 0).then_some(Instruction::Shr(rp as u16)))
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
) -> ControlFlow<(), IndexInc> {
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
            return ControlFlow::Continue(IndexInc::Zero);
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

    ControlFlow::Continue(IndexInc::One)
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

pub fn remove_redundant_jump_pairs(config: &Config, instructions: &mut Vec<Instruction>) {
    let mut jump_stack = Vec::new();
    let mut i = 0;
    while i < instructions.len() {
        let inst = instructions[i];
        match inst {
            Instruction::JumpZ(jump) => jump_stack.push((i, jump.is_redundant())),
            Instruction::JumpNz(end_jump) => {
                let Some((start_idx, start_redundant)) = jump_stack.pop() else {
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
    if !jump_stack.is_empty() {
        unreachable!("mismatched brackets")
    }
}

pub fn simplify_code(config: &Config, instructions: &mut Vec<Instruction>) {
    use Instruction::*;

    let mut i = 0;
    while i < instructions.len() {
        let inst = instructions[i];
        let index_inc = match inst {
            Shl(n) => combine_shifts(config, instructions, i, -(n as i16)),
            Shr(n) => combine_shifts(config, instructions, i, n as i16),

            Inc(o, n) => combine_sets(config, instructions, i, o, Update::Relative(n as i8)),
            Dec(o, n) => combine_sets(
                config,
                instructions,
                i,
                o,
                Update::Relative((n as i8).wrapping_neg()),
            ),
            Zero(o) => combine_sets(config, instructions, i, o, Update::Absolute(0)),
            Set(o, n) => combine_sets(config, instructions, i, o, Update::Absolute(n as i8)),

            Add(o) => combine_add_sub(config, instructions, i, o, 1),
            Sub(o) => combine_add_sub(config, instructions, i, o, -1),
            AddMul(o, n) => combine_add_sub(config, instructions, i, o, n as i8),
            SubMul(o, n) => combine_add_sub(config, instructions, i, o, (n as i8).wrapping_neg()),

            Output => IndexInc::One,
            Input => IndexInc::One,
            JumpZ(_) => IndexInc::One,
            JumpNz(_) => IndexInc::One,
        };

        i += index_inc as usize;
    }
}

fn combine_shifts(
    config: &Config,
    instructions: &mut Vec<Instruction>,
    start: usize,
    mut shift: i16,
) -> IndexInc {
    use Instruction::*;

    let mut i = start + 1;
    while i < instructions.len() {
        match instructions[i] {
            Shl(n) => shift -= n as i16,
            Shr(n) => shift += n as i16,
            _ => break,
        }

        i += 1;
    }

    if start + 1 == i {
        return IndexInc::One;
    };

    let range = start..i;
    let replacement = match shift {
        ..=-1 => Shl((-shift) as u16),
        0 => {
            if config.verbose >= 2 {
                let removed = &instructions[range.clone()];
                println!("remove redundant shifts {range:?} {removed:?}");
            }
            instructions.drain(range);
            return IndexInc::Zero;
        }
        1.. => Shr(shift as u16),
    };
    if config.verbose >= 2 {
        let removed = &instructions[range.clone()];
        println!("simplify shifts {range:?} {removed:?} with {replacement:?}");
    }
    instructions.splice(range, Some(replacement));

    IndexInc::One
}

fn combine_sets(
    config: &Config,
    instructions: &mut Vec<Instruction>,
    start: usize,
    offset: i16,
    mut acc: Update,
) -> IndexInc {
    use Instruction::*;

    let mut i = start + 1;
    while i < instructions.len() {
        match instructions[i] {
            Inc(o, n) if offset == o => acc.inc(n),
            Dec(o, n) if offset == o => acc.dec(n),
            Zero(o) if offset == o => acc.set(0),
            Set(o, n) if offset == o => acc.set(n),
            _ => break,
        }

        i += 1;
    }

    if start + 1 == i {
        return IndexInc::One;
    }

    let range = start..i;
    let replacement = match acc {
        Update::Relative(n) => match n {
            ..=-1 => Dec(offset, n.wrapping_neg() as u8),
            0 => {
                if config.verbose >= 2 {
                    let removed = &instructions[range.clone()];
                    println!("removed redundant {range:?} {removed:?}");
                }
                instructions.drain(range);
                return IndexInc::Zero;
            }
            1.. => Inc(offset, n as u8),
        },
        Update::Absolute(0) => Zero(offset),
        Update::Absolute(n) => Set(offset, n as u8),
    };
    if config.verbose >= 2 {
        let removed = &instructions[range.clone()];
        println!("simplify {range:?} {removed:?} with {replacement:?}");
    }
    instructions.splice(range, Some(replacement));

    IndexInc::One
}

fn combine_add_sub(
    config: &Config,
    instructions: &mut Vec<Instruction>,
    start: usize,
    offset: i16,
    mut factor: i8,
) -> IndexInc {
    use Instruction::*;

    let mut i = start + 1;
    while i < instructions.len() {
        match instructions[i] {
            Add(o) if offset == o => factor = factor.wrapping_add(1),
            Sub(o) if offset == o => factor = factor.wrapping_sub(1),
            AddMul(o, n) if offset == o => factor = factor.wrapping_add(n as i8),
            SubMul(o, n) if offset == o => factor = factor.wrapping_sub(n as i8),

            // if field is zeroed all instruction combined until here are redundant
            Zero(o) | Set(o, _) if o == offset => {
                let range = start..i + 1;
                if config.verbose >= 2 {
                    let removed = &instructions[range.clone()];
                    println!("removed redundant {range:?} {removed:?}");
                }
                instructions.drain(range);
                return IndexInc::Zero;
            }
            _ => break,
        }

        i += 1;
    }
    if start + 1 == i {
        return IndexInc::One;
    }

    let range = start..i;
    let replacement = match factor {
        ..=-2 => SubMul(offset, -factor as u8),
        ..=-1 => Sub(offset),
        0 => {
            if config.verbose >= 2 {
                let removed = &instructions[range.clone()];
                println!("remove redundant {range:?} {removed:?}");
            }
            instructions.drain(range);
            return IndexInc::Zero;
        }
        1 => Add(offset),
        2.. => AddMul(offset, factor as u8),
    };
    if config.verbose >= 2 {
        let removed = &instructions[range.clone()];
        println!("simplify {range:?} {removed:?} with {replacement:?}");
    }
    instructions.splice(range, Some(replacement));

    IndexInc::One
}
