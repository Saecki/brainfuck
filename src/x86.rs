use crate::cli::Config;
use crate::{Instruction, NUM_REGISTERS};

#[repr(C)]
pub struct ElfFileHeader {
    pub ei_magic: [u8; 4],
    pub ei_class: u8,
    pub ei_data: u8,
    pub ei_version: u8,
    pub ei_osabi: u8,
    pub ei_abiversion: u8,
    pub ei_pad: [u8; 7],

    pub e_type: u16,
    pub e_machine: u16,
    pub e_version: u32,
    pub e_entry: u64,
    /// program header offset
    pub e_phoff: u64,
    /// section header offset
    pub e_shoff: u64,
    pub e_flags: u32,
    /// size of this header
    pub e_ehsize: u16,
    /// program header entry size
    pub e_phentsize: u16,
    /// number of program header table entries
    pub e_phnum: u16,
    /// section header entry size
    pub e_shentsize: u16,
    /// number of section header table entries
    pub e_shnum: u16,
    /// index of section header table entry that contains section names
    pub e_shstrndx: u16,
}

#[repr(C)]
pub struct ElfProgramHeader {
    pub p_type: u32,
    pub p_flags: u32,
    pub p_offset: u64,
    pub p_vaddr: u64,
    pub p_paddr: u64,
    pub p_filesz: u64,
    pub p_memsz: u64,
    pub p_align: u64,
}

/// Address mode
#[derive(Clone, Copy)]
pub enum ModRm {
    /// Use address stored in the register or the [`Sib`] to read value from memory
    Indirect(RmI),
    /// Use address stored in the register or the [`Sib`] + a i8 displacement to read value from memory
    IndirectDisp8(RmID),
    /// Use address stored in the register or the [`Sib`] + a i32 displacement to read value from memory
    IndirectDisp32(RmID),
    /// Use value stored in the register
    Register(Reg),
}

impl ModRm {
    pub const fn mode(&self) -> u8 {
        match self {
            ModRm::Indirect(..) => 0b00,
            ModRm::IndirectDisp8(..) => 0b01,
            ModRm::IndirectDisp32(..) => 0b10,
            ModRm::Register(..) => 0b11,
        }
    }

    pub const fn rm(&self) -> u8 {
        match *self {
            ModRm::Indirect(rm) => rm as u8,
            ModRm::IndirectDisp8(rm) => rm as u8,
            ModRm::IndirectDisp32(rm) => rm as u8,
            ModRm::Register(rm) => rm as u8,
        }
    }
}

/// R/M value for indirect addressing mode
#[derive(Clone, Copy)]
pub enum RmI {
    RegRax = 0b000,
    RegRcx = 0b001,
    RegRdx = 0b010,
    RegRbx = 0b011,
    /// Use a scaled index byte [`Sib`] following the opcode
    Sib = 0b100,
    /// Use *only* a constant 32-bit displacement
    Disp32 = 0b101,
    RegRsi = 0b110,
    RegRdi = 0b111,
}

/// R/M value for indirect addressing mode with displacement
#[derive(Clone, Copy)]
pub enum RmID {
    RegRax = 0b000,
    RegRcx = 0b001,
    RegRdx = 0b010,
    RegRbx = 0b011,
    Sib = 0b100,
    RegRbp = 0b101,
    RegRsi = 0b110,
    RegRdi = 0b111,
}

#[derive(Clone, Copy)]
pub enum Reg {
    Rax = 0x0,
    Rcx = 0x1,
    Rdx = 0x2,
    Rbx = 0x3,
    Rsp = 0x4,
    Rbp = 0x5,
    Rsi = 0x6,
    Rdi = 0x7,
}

/// Generate a `MOD-REG_R/M` byte with a `reg` field
pub const fn modrm_reg(modrm: ModRm, reg: Reg) -> u8 {
    (modrm.mode() << 6) | ((reg as u8) << 3) | modrm.rm()
}

/// Generate a `MOD-REG_R/M` byte with an op-code extension
pub const fn modrm_ext(modrm: ModRm, ext: u8) -> u8 {
    (modrm.mode() << 6) | (ext << 3) | modrm.rm()
}

/// Calculate a memory address using the values stored inside the [`Sib::base`] register plus the
/// [`Sib::idx`] register multiplied by a [`Scale`] factor:
///
/// `base + scale * idx`
///
/// or
///
/// `base[scale * idx]`
#[derive(Clone, Copy)]
pub struct Sib {
    pub scale: Scale,
    pub idx: Reg,
    pub base: Reg,
}

impl Sib {
    pub const fn new(scale: Scale, idx: Reg, base: Reg) -> Self {
        Self { scale, idx, base }
    }

    pub const fn sib(&self) -> u8 {
        ((self.scale as u8) << 6) | ((self.idx as u8) << 3) | (self.base as u8)
    }
}

/// [`Sib::scale`]
#[derive(Clone, Copy)]
pub enum Scale {
    /// 1 byte scale
    B1 = 0b00,
    /// 2 byte scale
    B2 = 0b01,
    /// 4 byte scale
    B4 = 0b10,
    /// 8 byte scale
    B8 = 0b11,
}

#[macro_export]
macro_rules! const_assert {
    ($x:expr $(,)?) => {
        #[allow(unknown_lints, clippy::eq_op)]
        const _: [(); 0 - !{
            const ASSERT: bool = $x;
            ASSERT
        } as usize] = [];
    };
}

/// Generate a 64-bit x86 linux ELF binary
pub fn compile(config: &Config, instructions: &[Instruction]) -> Vec<u8> {
    const B64_ELF_HEADER_LEN: usize = 0x40;
    const B64_PROGRAM_HEADER_LEN: usize = 0x38;
    const PROGRAM_OFFSET: usize = B64_ELF_HEADER_LEN + B64_PROGRAM_HEADER_LEN;
    const VADDR: usize = 0x40_0000;

    const_assert!(B64_ELF_HEADER_LEN == std::mem::size_of::<ElfFileHeader>());
    const_assert!(B64_PROGRAM_HEADER_LEN == std::mem::size_of::<ElfProgramHeader>());

    let elf_header: [u8; B64_ELF_HEADER_LEN] = {
        let header = ElfFileHeader {
            // e_ident
            ei_magic: *b"\x7fELF",
            ei_class: 0x02, // 64-bit
            ei_data: 0x01,  // little-endian
            ei_version: 0x01,
            ei_osabi: 0x00, // system-v
            ei_abiversion: 0x00,
            ei_pad: [0x00; 7], // reserved

            e_type: 0x0002,    // executable
            e_machine: 0x003E, // AMD x86-64
            e_version: 1,
            e_entry: (VADDR + PROGRAM_OFFSET) as u64, // entry point offset
            e_phoff: B64_ELF_HEADER_LEN as u64, // program header immediately follows the ELF header
            e_shoff: 0,                         // no table thus 0 offset
            e_flags: 0x0000_0000,               // no flags
            e_ehsize: B64_ELF_HEADER_LEN as u16,
            e_phentsize: B64_PROGRAM_HEADER_LEN as u16,
            e_phnum: 1,
            e_shentsize: 0, // no table thus irrelevant
            e_shnum: 0,     // no table thus 0 entries
            e_shstrndx: 0,
        };

        unsafe { std::mem::transmute(header) }
    };

    let program_header: [u8; B64_PROGRAM_HEADER_LEN] = {
        let header = ElfProgramHeader {
            p_type: 0x0000_0001,                      // loadable segment
            p_flags: 0x0000_0007,                     // read write execute
            p_offset: PROGRAM_OFFSET as u64,          // loadable segment starts after this header
            p_vaddr: (VADDR + PROGRAM_OFFSET) as u64, // virtual address to place the segment at
            p_paddr: 0,                               // physical address is not used
            p_filesz: 0,     // size of the whole file (write after code generation)
            p_memsz: 0,      // size of the whole file (write after code generation)
            p_align: 0x1000, // no alignment
        };

        unsafe { std::mem::transmute(header) }
    };

    let mut code = (elf_header.iter().copied())
        .chain(program_header.iter().copied())
        .collect();

    write_instructions(config, &mut code, instructions);

    // update loadable segment size
    {
        let size = code.len() as u64 - PROGRAM_OFFSET as u64;
        let program_header = &mut code[B64_ELF_HEADER_LEN..][..B64_PROGRAM_HEADER_LEN];
        const P_FILESZ: usize = std::mem::offset_of!(ElfProgramHeader, p_filesz);
        program_header[P_FILESZ..P_FILESZ + 8].copy_from_slice(&u64::to_le_bytes(size));
        const P_MEMSZ: usize = std::mem::offset_of!(ElfProgramHeader, p_memsz);
        program_header[P_MEMSZ..P_MEMSZ + 8].copy_from_slice(&u64::to_le_bytes(size));
    }

    if config.verbose >= 1 {
        const K: usize = 1024;
        const M: usize = K * K;
        const G: usize = K * M;
        print!("generated code size: ");
        let size = code.len();
        match size {
            _ if size < K => println!("{}b", size),
            _ if size < M => println!("{:.2}kb", size as f32 / K as f32),
            _ if size < G => println!("{:.2}Mb", size as f32 / M as f32),
            _ => println!("{size}Gb"),
        }
        println!("============================================================");
    }

    code
}

fn write_instructions(config: &Config, code: &mut Vec<u8>, instructions: &[Instruction]) {
    // prepare brainfuck registers array
    {
        // allocate stack space for brainfuck registers array
        write(code, sub_imm32_from_r64(Reg::Rsp, NUM_REGISTERS as i32));

        const NUM_ITERATIONS: i32 = NUM_REGISTERS as i32 / 8;
        write(code, mov_imm32_to_r64(Reg::Rcx, NUM_ITERATIONS));

        let loop_start = code.len();
        write(code, sub_imm8_from_r32(Reg::Rcx, 0x01));

        // write 0_i64 to stack at `rsp + 8 * rcx` using a scaled index byte (SIB)
        const SIB: Sib = Sib::new(Scale::B8, Reg::Rcx, Reg::Rsp);
        write(code, mov_imm32_to_sib64(SIB, 0));

        write(code, cmp_r32_with_imm8(Reg::Rcx, 0x00));

        // the instruction pointer will have already moved to the next instruction, so it will be
        // after the jump instruction
        let loop_end = code.len() + const { jnz_rel8(0).len() };
        let rel_jump = (loop_start as isize - loop_end as isize) as i8;
        write(code, jnz_rel8(rel_jump))
    }

    // scaled index byte used to index into the brainfuck register array
    const SIB: Sib = Sib::new(Scale::B1, Reg::Rcx, Reg::Rsp);

    // stores if the jump is redundant, and the location after the opening jump (`[`), the jump
    // offset is stored inside the 4 bytes before that
    let mut jump_stack = Vec::new();
    for inst in instructions.iter() {
        match *inst {
            #[rustfmt::skip]
            Instruction::Shl(n) => match n {
                0..=127 => write(code, sub_imm8_from_r32(Reg::Rcx, n as i8)),
                _ =>       write(code, sub_imm32_from_r32(Reg::Rcx, n as i32)),
            },
            #[rustfmt::skip]
            Instruction::Shr(n) => match n {
                0..=127 => write(code, add_imm8_to_r32(Reg::Rcx, n as i8)),
                _ =>       write(code, add_imm32_to_r32(Reg::Rcx, n as i32)),
            },
            #[rustfmt::skip]
            Instruction::Inc(disp, n) => match disp {
                0 =>          write(code, add_imm8_to_sib8(SIB, n)),
                -128..=127 => write(code, add_imm8_to_sib8_disp8(SIB, disp as i8, n)),
                _ =>          write(code, add_imm8_to_sib8_disp32(SIB, disp as i32, n)),
            },
            #[rustfmt::skip]
            Instruction::Dec(disp, n) => match disp {
                0 =>          write(code, sub_imm8_from_sib8(SIB, n)),
                -128..=127 => write(code, sub_imm8_from_sib8_disp8(SIB, disp as i8, n)),
                _ =>          write(code, sub_imm8_from_sib8_disp32(SIB, disp as i32, n)),
            },
            Instruction::Output => {
                const SYSCALL_WRITE: i32 = 1;
                write(code, mov_imm32_to_r64(Reg::Rax, SYSCALL_WRITE));

                const STDOUT_FD: i32 = 1;
                write(code, mov_imm32_to_r64(Reg::Rdi, STDOUT_FD));

                // write address of string to `rsi`
                write(code, mov_r64_to_r64(Reg::Rsp, Reg::Rsi));
                write(code, add_r64_to_r64(Reg::Rcx, Reg::Rsi));

                const STRING_LEN: i32 = 1;
                write(code, mov_imm32_to_r64(Reg::Rdx, STRING_LEN));

                write(code, push_r64(Reg::Rcx));
                write(code, SYSCALL);
                write(code, pop_r64(Reg::Rcx));
            }
            Instruction::Input => {
                const _SYSCALL_READ: i32 = 0;
                write(code, xor_r64_r64(Reg::Rax, Reg::Rax));

                const _STDIN_FD: i32 = 0;
                write(code, xor_r64_r64(Reg::Rdi, Reg::Rdi));

                // write address of string to `rsi`
                write(code, mov_r64_to_r64(Reg::Rsp, Reg::Rsi));
                write(code, add_r64_to_r64(Reg::Rcx, Reg::Rsi));

                const STRING_LEN: i32 = 1;
                write(code, mov_imm32_to_r64(Reg::Rdx, STRING_LEN));

                write(code, push_r64(Reg::Rcx));
                write(code, SYSCALL);
                write(code, pop_r64(Reg::Rcx));
            }
            Instruction::JumpZ(jump) => {
                let redundant = jump.is_redundant();
                if !redundant {
                    write(code, cmp_sib8_with_imm8(SIB, 0));
                    // actual jump offset is updated when writing the matching JumpNz (`]`) instruction
                    write(code, jz_rel32(0));
                }

                let pos = code.len();
                jump_stack.push((redundant, pos));
            }
            Instruction::JumpNz(jump) => {
                let Some((start_redundant, start_pos)) = jump_stack.pop() else {
                    unreachable!()
                };

                const CMP_INST_LEN: usize = cmp_sib8_with_imm8(SIB, 0).len();
                const REL8_INST_LEN: usize = CMP_INST_LEN + jnz_rel8(0).len();
                const REL32_JUMP_INST_LEN: usize = jnz_rel32(0).len();
                const REL32_INST_LEN: usize = CMP_INST_LEN + REL32_JUMP_INST_LEN;
                let pos = code.len();
                let offset_without_inst = pos - start_pos;
                let redundant = jump.is_redundant();

                let (rel8, offset) = if redundant {
                    (offset_without_inst < 128, offset_without_inst)
                } else if offset_without_inst + REL8_INST_LEN < 128 {
                    (true, offset_without_inst + REL8_INST_LEN)
                } else {
                    (false, offset_without_inst + REL32_INST_LEN)
                };

                if config.verbose >= 3 {
                    if rel8 {
                        println!("using rel8 jump for offset: {offset}");
                    } else {
                        println!("using rel32 jump for offset: {offset}");
                    }
                }

                if !start_redundant {
                    if rel8 {
                        let jump_inst = jz_rel8(offset as i8);
                        code.splice(start_pos - REL32_JUMP_INST_LEN..start_pos, jump_inst);
                    } else {
                        let offset = i32::to_le_bytes(offset as i32);
                        code[start_pos - 4..start_pos].copy_from_slice(&offset);
                    }
                }

                if !redundant {
                    write(code, cmp_sib8_with_imm8(SIB, 0));
                    if rel8 {
                        write(code, jnz_rel8(-(offset as i8)));
                    } else {
                        write(code, jnz_rel32(-(offset as i32)));
                    }
                }
            }

            #[rustfmt::skip]
            Instruction::Zero(disp) => match disp {
                0 =>          write(code, mov_imm8_to_sib8(SIB, 0x00)),
                -128..=127 => write(code, mov_imm8_to_sib8_disp8(SIB, disp as i8, 0x00)),
                _ =>          write(code, mov_imm8_to_sib8_disp32(SIB, disp as i32, 0x00)),
            },
            #[rustfmt::skip]
            Instruction::Set(disp, n) => match disp {
                0 =>          write(code, mov_imm8_to_sib8(SIB, n)),
                -128..=127 => write(code, mov_imm8_to_sib8_disp8(SIB, disp as i8, n)),
                _ =>          write(code, mov_imm8_to_sib8_disp32(SIB, disp as i32, n)),
            },
            Instruction::Add(disp) => {
                write(code, mov_sib8_to_r8(SIB, Reg::Rax));
                #[rustfmt::skip]
                match disp {
                    0 =>          write(code, add_r8_to_sib8(Reg::Rax, SIB)),
                    -128..=127 => write(code, add_r8_to_sib8_disp8(Reg::Rax, SIB, disp as i8)),
                    _ =>          write(code, add_r8_to_sib8_disp32(Reg::Rax, SIB, disp as i32)),
                };
            }
            Instruction::Sub(disp) => {
                write(code, mov_sib8_to_r8(SIB, Reg::Rax));
                #[rustfmt::skip]
                match disp {
                    0 =>          write(code, sub_r8_from_sib8(Reg::Rax, SIB)),
                    -128..=127 => write(code, sub_r8_from_sib8_disp8(Reg::Rax, SIB, disp as i8)),
                    _ =>          write(code, sub_r8_from_sib8_disp32(Reg::Rax, SIB, disp as i32)),
                };
            }
            Instruction::AddMul(disp, n) => {
                write(code, mov_imm8_to_r8(Reg::Rax, n));
                write(code, mul_al_with_sib8(SIB));
                #[rustfmt::skip]
                match disp {
                    0 =>          write(code, add_r8_to_sib8(Reg::Rax, SIB)),
                    -128..=127 => write(code, add_r8_to_sib8_disp8(Reg::Rax, SIB, disp as i8)),
                    _ =>          write(code, add_r8_to_sib8_disp32(Reg::Rax, SIB, disp as i32)),
                };
            }
            Instruction::SubMul(disp, n) => {
                write(code, mov_imm8_to_r8(Reg::Rax, n));
                write(code, mul_al_with_sib8(SIB));
                #[rustfmt::skip]
                match disp {
                    0 =>          write(code, sub_r8_from_sib8(Reg::Rax, SIB)),
                    -128..=127 => write(code, sub_r8_from_sib8_disp8(Reg::Rax, SIB, disp as i8)),
                    _ =>          write(code, sub_r8_from_sib8_disp32(Reg::Rax, SIB, disp as i32)),
                };
            }
        }
    }

    // pop brainfuck registers array off the stack
    write(code, add_imm32_to_r64(Reg::Rsp, NUM_REGISTERS as i32));

    const SYSCALL_EXIT: i32 = 60;
    write(code, mov_imm32_to_r64(Reg::Rax, SYSCALL_EXIT));

    // clear the edi register
    write(code, xor_r64_r64(Reg::Rdi, Reg::Rdi));

    write(code, SYSCALL);
}

fn write<const SIZE: usize>(code: &mut Vec<u8>, instruction: [u8; SIZE]) {
    code.extend_from_slice(&instruction);
}

/// prefix for some 64-bit instructions
const REXW: u8 = 0x48;

// ========================================
//                   ADD
// ========================================

// 8-bit

/// `00 /r` : `ADD r/m8 r8` : add r8 to r/m8
pub const fn add_r8_to_r8(src: Reg, dest: Reg) -> [u8; 2] {
    let modrm = modrm_reg(ModRm::Register(dest), src);
    [0x00, modrm]
}
/// `00 /r` : `ADD r/m8 r8` : add r8 to r/m8
pub const fn add_r8_to_sib8(src: Reg, dest: Sib) -> [u8; 3] {
    let modrm = modrm_reg(ModRm::Indirect(RmI::Sib), src);
    [0x00, modrm, dest.sib()]
}
/// `00 /r` : `ADD r/m8 r8` : add r8 to r/m8
pub const fn add_r8_to_sib8_disp8(src: Reg, dest: Sib, disp: i8) -> [u8; 4] {
    let modrm = modrm_reg(ModRm::IndirectDisp8(RmID::Sib), src);
    let [disp] = i8::to_le_bytes(disp);
    [0x00, modrm, dest.sib(), disp]
}
/// `00 /r` : `ADD r/m8 r8` : add r8 to r/m8
pub const fn add_r8_to_sib8_disp32(src: Reg, dest: Sib, disp: i32) -> [u8; 7] {
    let modrm = modrm_reg(ModRm::IndirectDisp32(RmID::Sib), src);
    let [b0, b1, b2, b3] = i32::to_le_bytes(disp);
    [0x00, modrm, dest.sib(), b0, b1, b2, b3]
}

/// `80 /0 ib`: `ADD r/m8 imm8` : add imm8 to r/m8
pub const fn add_imm8_to_r8(dest: Reg, ib: u8) -> [u8; 3] {
    let modrm = modrm_ext(ModRm::Register(dest), 0);
    [0x80, modrm, ib]
}
/// `80 /0 ib`: `ADD r/m8 imm8` : add imm8 to r/m8
pub const fn add_imm8_to_sib8(dest: Sib, ib: u8) -> [u8; 4] {
    const MODRM: u8 = modrm_ext(ModRm::Indirect(RmI::Sib), 0);
    [0x80, MODRM, dest.sib(), ib]
}
/// `80 /0 ib`: `ADD r/m8 imm8` : add imm8 to r/m8
pub const fn add_imm8_to_sib8_disp8(dest: Sib, disp: i8, ib: u8) -> [u8; 5] {
    const MODRM: u8 = modrm_ext(ModRm::IndirectDisp8(RmID::Sib), 0);
    let [disp] = i8::to_le_bytes(disp);
    [0x80, MODRM, dest.sib(), disp, ib]
}
/// `80 /0 ib`: `ADD r/m8 imm8` : add imm8 to r/m8
pub const fn add_imm8_to_sib8_disp32(dest: Sib, disp: i32, ib: u8) -> [u8; 8] {
    const MODRM: u8 = modrm_ext(ModRm::IndirectDisp32(RmID::Sib), 0);
    let [b0, b1, b2, b3] = i32::to_le_bytes(disp);
    [0x80, MODRM, dest.sib(), b0, b1, b2, b3, ib]
}

// 32-bit

/// `83 /0 ib` : `ADD r/m32 imm8` : add imm8 sign extended to 32-bits to r/m32
pub const fn add_imm8_to_r32(dest: Reg, ib: i8) -> [u8; 3] {
    let modrm = modrm_ext(ModRm::Register(dest), 0);
    let [ib] = i8::to_le_bytes(ib);
    [0x83, modrm, ib]
}

/// `81 /0 id` : `ADD r/m32 imm32` : add imm32 to r/m32
pub const fn add_imm32_to_r32(dest: Reg, id: i32) -> [u8; 6] {
    let modrm = modrm_ext(ModRm::Register(dest), 0);
    let [b0, b1, b2, b3] = i32::to_le_bytes(id);
    [0x81, modrm, b0, b1, b2, b3]
}

// 64-bit

/// `REX.W 01 /r` : `ADD r/m64 r64` : add r64 to r/m64
pub const fn add_r64_to_r64(src: Reg, dest: Reg) -> [u8; 3] {
    let modrm = modrm_reg(ModRm::Register(dest), src);
    [REXW, 0x01, modrm]
}

/// `REX.W 81 /0 id` : `ADD r/m64 imm32` : add imm32 sign extended to 64-bits to r/m64
pub const fn add_imm32_to_r64(dest: Reg, id: i32) -> [u8; 7] {
    let modrm = modrm_ext(ModRm::Register(dest), 0);
    let [b0, b1, b2, b3] = i32::to_le_bytes(id);
    [REXW, 0x81, modrm, b0, b1, b2, b3]
}

// ========================================
//                   SUB
// ========================================

// 8-bit

/// `28 /r` : `SUB r/m8 r8` : subtract r8 from r/m8
pub const fn sub_r8_from_r8(src: Reg, sib: Sib) -> [u8; 3] {
    let modrm = modrm_reg(ModRm::Indirect(RmI::Sib), src);
    [0x28, modrm, sib.sib()]
}
/// `28 /r` : `SUB r/m8 r8` : subtract r8 from r/m8
pub const fn sub_r8_from_sib8(src: Reg, dest: Sib) -> [u8; 3] {
    let modrm = modrm_reg(ModRm::Indirect(RmI::Sib), src);
    [0x28, modrm, dest.sib()]
}
/// `28 /r` : `SUB r/m8 r8` : subtract r8 from r/m8
pub const fn sub_r8_from_sib8_disp8(src: Reg, dest: Sib, disp: i8) -> [u8; 4] {
    let modrm = modrm_reg(ModRm::IndirectDisp8(RmID::Sib), src);
    let [disp] = i8::to_le_bytes(disp);
    [0x28, modrm, dest.sib(), disp]
}
/// `28 /r` : `SUB r/m8 r8` : subtract r8 from r/m8
pub const fn sub_r8_from_sib8_disp32(src: Reg, dest: Sib, disp: i32) -> [u8; 7] {
    let modrm = modrm_reg(ModRm::IndirectDisp32(RmID::Sib), src);
    let [b0, b1, b2, b3] = i32::to_le_bytes(disp);
    [0x28, modrm, dest.sib(), b0, b1, b2, b3]
}

/// `80 /5 ib`: `SUB r/m8 imm8` : subtract imm8 from r/m8
pub const fn sub_imm8_from_r8(dest: Reg, ib: u8) -> [u8; 3] {
    let modrm = modrm_ext(ModRm::Register(dest), 5);
    [0x80, modrm, ib]
}
/// `80 /5 ib`: `SUB r/m8 imm8` : subtract imm8 from r/m8
pub const fn sub_imm8_from_sib8(dest: Sib, ib: u8) -> [u8; 4] {
    const MODRM: u8 = modrm_ext(ModRm::Indirect(RmI::Sib), 5);
    [0x80, MODRM, dest.sib(), ib]
}
/// `80 /5 ib`: `SUB r/m8 imm8` : subtract imm8 from r/m8
pub const fn sub_imm8_from_sib8_disp8(dest: Sib, disp: i8, ib: u8) -> [u8; 5] {
    const MODRM: u8 = modrm_ext(ModRm::IndirectDisp8(RmID::Sib), 5);
    let [disp] = i8::to_le_bytes(disp);
    [0x80, MODRM, dest.sib(), disp, ib]
}
/// `80 /5 ib`: `SUB r/m8 imm8` : subtract imm8 from r/m8
pub const fn sub_imm8_from_sib8_disp32(dest: Sib, disp: i32, ib: u8) -> [u8; 8] {
    const MODRM: u8 = modrm_ext(ModRm::IndirectDisp32(RmID::Sib), 5);
    let [b0, b1, b2, b3] = i32::to_le_bytes(disp);
    [0x80, MODRM, dest.sib(), b0, b1, b2, b3, ib]
}

// 32-bit

/// `83 /5 ib` : `SUB r/m32 imm8` : subtract imm8 sign extended to 32-bit from r/m32
pub const fn sub_imm8_from_r32(dest: Reg, ib: i8) -> [u8; 3] {
    let modrm = modrm_ext(ModRm::Register(dest), 5);
    let [ib] = i8::to_le_bytes(ib);
    [0x83, modrm, ib]
}

/// `81 /5 id` : `SUB r/m32 imm32` : subtract imm32 from r/m32
pub const fn sub_imm32_from_r32(dest: Reg, id: i32) -> [u8; 6] {
    let modrm = modrm_ext(ModRm::Register(dest), 5);
    let [b0, b1, b2, b3] = i32::to_le_bytes(id);
    [0x81, modrm, b0, b1, b2, b3]
}

// 64-bit

/// `REX.W 81 /5 id` : `SUB r/m64 imm32` : subtract imm32 sign extended to 64-bits from r/m64
pub const fn sub_imm32_from_r64(dest: Reg, id: i32) -> [u8; 7] {
    let modrm = modrm_ext(ModRm::Register(dest), 5);
    let [b0, b1, b2, b3] = i32::to_le_bytes(id);
    [REXW, 0x81, modrm, b0, b1, b2, b3]
}

// ========================================
//                   MOV
// ========================================

// 8-bit

/// `8A /r`: `MOV r8 r/m8` : move r/m8 to r8
pub const fn mov_sib8_to_r8(src: Sib, dest: Reg) -> [u8; 3] {
    let modrm = modrm_reg(ModRm::Indirect(RmI::Sib), dest);
    [0x8A, modrm, src.sib()]
}

/// `C6 /0 ib` : `MOV r/m8 imm8` : move imm8 to r/m8
pub const fn mov_imm8_to_r8(dest: Reg, ib: u8) -> [u8; 3] {
    let modrm = modrm_ext(ModRm::Register(dest), 0);
    [0xC6, modrm, ib]
}
/// `C6 /0 ib` : `MOV r/m8 imm8` : move imm8 to r/m8
pub const fn mov_imm8_to_sib8(dest: Sib, ib: u8) -> [u8; 4] {
    let modrm = modrm_ext(ModRm::Indirect(RmI::Sib), 0);
    [0xC6, modrm, dest.sib(), ib]
}
/// `C6 /0 ib` : `MOV r/m8 imm8` : move imm8 to r/m8
pub const fn mov_imm8_to_sib8_disp8(dest: Sib, disp: i8, ib: u8) -> [u8; 5] {
    let modrm = modrm_ext(ModRm::IndirectDisp8(RmID::Sib), 0);
    let [disp] = i8::to_le_bytes(disp);
    [0xC6, modrm, dest.sib(), disp, ib]
}
/// `C6 /0 ib` : `MOV r/m8 imm8` : move imm8 to r/m8
pub const fn mov_imm8_to_sib8_disp32(sib: Sib, disp: i32, ib: u8) -> [u8; 8] {
    let modrm = modrm_ext(ModRm::IndirectDisp32(RmID::Sib), 0);
    let [b0, b1, b2, b3] = i32::to_le_bytes(disp);
    [0xC6, modrm, sib.sib(), b0, b1, b2, b3, ib]
}

// 64-bit

/// `REX.W 89 /r` : `MOV r/m64 r64` : move r64 to r/m64
pub const fn mov_r64_to_r64(src: Reg, dest: Reg) -> [u8; 3] {
    let modrm = modrm_reg(ModRm::Register(dest), src);
    [REXW, 0x89, modrm]
}

/// `REX.W C7 /0 id` : `MOV r/m64 imm32` : move imm32 sign extended to 64-bits to r/m64
pub const fn mov_imm32_to_r64(dest: Reg, id: i32) -> [u8; 7] {
    let modrm = modrm_ext(ModRm::Register(dest), 0);
    let [b0, b1, b2, b3] = i32::to_le_bytes(id);
    [REXW, 0xC7, modrm, b0, b1, b2, b3]
}
/// `REX.W C7 /0 id` : `MOV r/m64 imm32` : move imm32 sign extended to 64-bits to r/m64
pub const fn mov_imm32_to_sib64(dest: Sib, id: i32) -> [u8; 8] {
    let modrm = modrm_ext(ModRm::Indirect(RmI::Sib), 0);
    let [b0, b1, b2, b3] = i32::to_le_bytes(id);
    [REXW, 0xC7, modrm, dest.sib(), b0, b1, b2, b3]
}

// ========================================
//                   MISC
// ========================================

/// `F6 /4`: `MUL r/m8` : multiply al with r/m8 into ax
pub const fn mul_al_with_sib8(src: Sib) -> [u8; 3] {
    let modrm = modrm_ext(ModRm::Indirect(RmI::Sib), 4);
    [0xF6, modrm, src.sib()]
}

/// `31 /r`: `XOR r/m64 r64` : performs r/m64 xor r64 into r/m64
pub const fn xor_r64_r64(src: Reg, dest: Reg) -> [u8; 3] {
    let modrm = modrm_reg(ModRm::Register(dest), src);
    [REXW, 0x31, modrm]
}

/// `80 /7 ib` : `CMP r/m8 imm8` : compare r/m8 with imm8
pub const fn cmp_sib8_with_imm8(src: Sib, ib: u8) -> [u8; 4] {
    let modrm = const { modrm_ext(ModRm::Indirect(RmI::Sib), 7) };
    [0x80, modrm, src.sib(), ib]
}

/// `83 /7 ib` : `CMP r/m32 imm8` : compare r/m32 with imm8
pub const fn cmp_r32_with_imm8(src: Reg, ib: i8) -> [u8; 3] {
    let modrm = modrm_ext(ModRm::Register(src), 7);
    let [ib] = i8::to_le_bytes(ib);
    [0x83, modrm, ib]
}

/// `74 cb` : `JZ rel8` : jump rel8 if zero
pub const fn jz_rel8(cb: i8) -> [u8; 2] {
    let [cb] = i8::to_le_bytes(cb);
    [0x74, cb]
}

/// `0F 84 cd` : `JZ rel32` : jump rel32 if zero
pub const fn jz_rel32(cd: i32) -> [u8; 6] {
    let [b0, b1, b2, b3] = i32::to_le_bytes(cd);
    [0x0F, 0x84, b0, b1, b2, b3]
}

/// `75 cd` : `JNZ rel8` : jump rel8 if not zero
pub const fn jnz_rel8(cd: i8) -> [u8; 2] {
    let [cd] = i8::to_le_bytes(cd);
    [0x75, cd]
}

/// `0F 85 cd` : `JNZ rel32` : jump rel32 if not zero
pub const fn jnz_rel32(cd: i32) -> [u8; 6] {
    let [b0, b1, b2, b3] = i32::to_le_bytes(cd);
    [0x0F, 0x85, b0, b1, b2, b3]
}

/// `FF /6`: `PUSH r64` : push r64 onto the stack
pub const fn push_r64(src: Reg) -> [u8; 2] {
    let modrm = modrm_ext(ModRm::Register(src), 6);
    [0xFF, modrm]
}

/// `8F /0`: `POP r64` : pop r64 off the stack
pub const fn pop_r64(dest: Reg) -> [u8; 2] {
    let modrm = modrm_ext(ModRm::Register(dest), 0);
    [0x8F, modrm]
}

/// `0F 05`: `SYSCALL` : fast system call
pub const SYSCALL: [u8; 2] = [0x0F, 0x05];
