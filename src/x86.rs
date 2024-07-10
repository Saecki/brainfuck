use std::io::Write;

use crate::{Instruction, NUM_REGISTERS};

#[derive(Clone, Copy)]
pub enum ModRm {
    Indirect(Indirect),
    Indirect1ByteDisp(IndirectDisp),
    Indirect4ByteDisp(IndirectDisp),
    Register(Reg),
}

#[derive(Clone, Copy)]
pub enum Indirect {
    RegEax = 0b000,
    RegEcx = 0b001,
    RegEdx = 0b010,
    RegEbx = 0b011,
    Sib = 0b100,
    Disp4byte = 0b101,
    RegEsi = 0b110,
    RegEdi = 0b111,
}

#[derive(Clone, Copy)]
pub enum IndirectDisp {
    RegEax = 0b000,
    RegEcx = 0b001,
    RegEdx = 0b010,
    RegEbx = 0b011,
    Sib = 0b100,
    RegEbp = 0b101,
    RegEsi = 0b110,
    RegEdi = 0b111,
}

impl ModRm {
    pub const fn mode(&self) -> u8 {
        match self {
            ModRm::Indirect(..) => 0b00,
            ModRm::Indirect1ByteDisp(..) => 0b01,
            ModRm::Indirect4ByteDisp(..) => 0b10,
            ModRm::Register(..) => 0b11,
        }
    }

    pub const fn rm(&self) -> u8 {
        match *self {
            ModRm::Indirect(rm) => rm as u8,
            ModRm::Indirect1ByteDisp(rm) => rm as u8,
            ModRm::Indirect4ByteDisp(rm) => rm as u8,
            ModRm::Register(rm) => rm as u8,
        }
    }
}

#[derive(Clone, Copy)]
pub enum Reg {
    Eax = 0x0,
    Ecx = 0x1,
    Edx = 0x2,
    Ebx = 0x3,
    Esp = 0x4,
    Ebp = 0x5,
    Esi = 0x6,
    Edi = 0x7,
}

pub const fn normal_mod_reg_rm(modrm: ModRm, reg: Reg) -> u8 {
    ((modrm.mode() as u8) << 6) | ((reg as u8) << 3) | (modrm.rm() as u8)
}

pub const fn extension_mod_reg_rm(modrm: ModRm, ext: u8) -> u8 {
    ((modrm.mode() as u8) << 6) | (ext << 3) | (modrm.rm() as u8)
}

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

pub const fn sib(scale: Scale, index: Reg, base: Reg) -> u8 {
    ((scale as u8) << 6) | ((index as u8) << 3) | (base as u8)
}

/// Generate a 32-bit x86 linux ELF binary
pub fn compile(instructions: &[Instruction]) -> Vec<u8> {
    const B32BIT_ELF_HEADER_LEN: usize = 0x34;
    const B32BIT_PROGRAM_HEADER_LEN: usize = 0x20;

    let mut elf_header = [0u8; B32BIT_ELF_HEADER_LEN];
    {
        // e_ident
        elf_header[0x00..0x04].copy_from_slice(b"\x7fELF"); // EI_MAG
        elf_header[0x04] = 0x1; // EI_CLASS      : 32-bit
        elf_header[0x05] = 0x1; // EI_DATA       : little-endian
        elf_header[0x06] = 0x1; // EI_VERSION    : 1
        elf_header[0x07] = 0x3; // EI_OSABI      : linux
        elf_header[0x08] = 0x3; // EI_ABIVERSION : 0
                                // EI_PAD        : reserved

        // e_type: executable
        elf_header[0x10..0x12].copy_from_slice(&u16::to_le_bytes(0x0002));

        // e_machine: AMD x86-64
        elf_header[0x12..0x14].copy_from_slice(&u16::to_le_bytes(0x003E));

        // e_version: 1
        elf_header[0x14..0x18].copy_from_slice(&u32::to_le_bytes(0x00000001));

        // e_entry: entry point offset
        // TODO: write

        // e_phoff: program header table offset immediately follows the ELF header
        elf_header[0x1C..0x20].copy_from_slice(&u32::to_le_bytes(B32BIT_ELF_HEADER_LEN as u32));

        // e_shoff: section header table offset
        // TODO: write section header table offset

        // e_flags: no flags

        // e_ehsize: ELF header size is 52 for 32-bit binaries
        elf_header[0x28..0x2a].copy_from_slice(&u16::to_le_bytes(B32BIT_ELF_HEADER_LEN as u16));

        // e_phentsize: program header table size
        // TODO: write

        // e_phnum: program header table entry count
        // TODO: write

        // e_shentsize: section header table size
        // TODO: write

        // e_shnum:  section header table entry count
        // TODO: write

        // e_shstrndx: section header table entry index that contains the section names
        // TODO: write
    }

    let mut program_header = [0u8; B32BIT_PROGRAM_HEADER_LEN];
    {
        // p_type:
        // TODO
        program_header[0x00..0x04].copy_from_slice(&u32::to_le_bytes(0x00000000));

        // p_offset:
        // TODO
        program_header[0x04..0x08].copy_from_slice(&u32::to_le_bytes(0x00000000));

        // p_vaddr:
        // TODO
        program_header[0x08..0x0C].copy_from_slice(&u32::to_le_bytes(0x00000000));

        // p_paddr:
        // TODO
        program_header[0x0C..0x10].copy_from_slice(&u32::to_le_bytes(0x00000000));

        // p_filesz:
        // TODO
        program_header[0x10..0x14].copy_from_slice(&u32::to_le_bytes(0x00000000));

        // p_memsz:
        // TODO
        program_header[0x14..0x18].copy_from_slice(&u32::to_le_bytes(0x00000000));

        // p_flags:
        // TODO
        program_header[0x18..0x1C].copy_from_slice(&u32::to_le_bytes(0x00000000));

        // p_align:
        // TODO
        program_header[0x1C..0x20].copy_from_slice(&u32::to_le_bytes(0x00000000));
    }

    let mut code = (elf_header.iter().copied())
        .chain(program_header.iter().copied())
        .collect();

    // prepare brainfuck registers
    {
        // `81 /5 id`: allocate stack space for 30000 elements
        let modrm = const { extension_mod_reg_rm(ModRm::Register(Reg::Esp), 5) };
        let [b0, b1, b2, b3] = u32::to_le_bytes(NUM_REGISTERS as u32);
        write_x86_instruction(&mut code, [0x81, modrm, b0, b1, b2, b3]);

        // `C7 /0 id`: write ITERATIONS to `eax`
        const NUM_ITERATIONS: u32 = NUM_REGISTERS as u32 / 4;
        let modrm = const { extension_mod_reg_rm(ModRm::Register(Reg::Eax), 0) };
        let [b0, b1, b2, b3] = u32::to_le_bytes(NUM_ITERATIONS as u32);
        write_x86_instruction(&mut code, [0xC7, modrm, b0, b1, b2, b3]);

        let loop_start = code.len();
        // `2C id`: subtract 1 from `eax`
        let [b0, b1, b2, b3] = u32::to_le_bytes(0x01);
        write_x86_instruction(&mut code, [0x2D, b0, b1, b2, b3]);

        // `C7 /0 id`: write 0u32 to stack at `esp + 4 * eax` using a scaled index byte (SIB)
        let modrm = const { extension_mod_reg_rm(ModRm::Indirect(Indirect::Sib), 0) };
        let sib = const { sib(Scale::B4, Reg::Eax, Reg::Esp) };
        let [b0, b1, b2, b3] = u32::to_le_bytes(0);
        write_x86_instruction(&mut code, [0xC7, modrm, sib, b0, b1, b2, b3]);

        // `3D id`: cmp `eax` to `0`
        let [b0, b1, b2, b3] = u32::to_le_bytes(0);
        write_x86_instruction(&mut code, [0x3D, b0, b1, b2, b3]);

        let loop_end = code.len();
        // `75 cb`: jump if not 0
        let [rel_jump] = i8::to_le_bytes((loop_start as isize - loop_end as isize) as i8);
        write_x86_instruction(&mut code, [0x75, rel_jump]);
    }

    // `31 /r` zero our register pointer in the `eax` register
    let modrm = const { normal_mod_reg_rm(ModRm::Register(Reg::Eax), Reg::Eax) };
    write_x86_instruction(&mut code, [0x31, modrm]);

    // generate code
    // stores location after opening jump (`[`), the jump offset is stored inside the 4-bytes
    // before that.
    let mut jump_stack = Vec::new();
    for inst in instructions.iter() {
        match inst {
            Instruction::Shl(n) => {
                // `2D id`: sub immediate value from `eax`
                let [b0, b1] = u16::to_le_bytes(*n);
                write_x86_instruction(&mut code, [0x2D, b0, b1, 0, 0]);
            }
            Instruction::Shr(n) => {
                // `05 id`: add immediate value to `eax`
                let [b0, b1] = u16::to_le_bytes(*n);
                write_x86_instruction(&mut code, [0x05, b0, b1, 0, 0]);
            }
            Instruction::Inc(n) => {
                // `80 /0 id`: add immediate value to `esp + 1 * eax`
                let modrm = const { extension_mod_reg_rm(ModRm::Indirect(Indirect::Sib), 0) };
                let sib = const { sib(Scale::B1, Reg::Eax, Reg::Esp) };
                write_x86_instruction(&mut code, [0x80, modrm, sib, *n]);
            }
            Instruction::Dec(n) => {
                // `80 /5 id`: add immediate value to `esp + 1 * eax`
                let modrm = const { extension_mod_reg_rm(ModRm::Indirect(Indirect::Sib), 5) };
                let sib = const { sib(Scale::B1, Reg::Eax, Reg::Esp) };
                write_x86_instruction(&mut code, [0x80, modrm, sib, *n]);
            }
            Instruction::Output => todo!(),
            Instruction::Input => todo!(),
            Instruction::JumpZ(_) => {
                // `80 /7 ib`: cmp with `0`
                let modrm = const { extension_mod_reg_rm(ModRm::Indirect(Indirect::Sib), 7) };
                let sib = const { sib(Scale::B1, Reg::Eax, Reg::Esp) };
                write_x86_instruction(&mut code, [0x80, modrm, sib, 0x00]);

                // `0F 84 cd`: jump if zero
                let offset = u32::to_le_bytes(0);
                write_x86_instruction(&mut code, [0x0F, 0x84]);

                jump_stack.push(code.len());
            }
            Instruction::JumpNz(_) => todo!(),
            Instruction::Zero(_) => todo!(),
            Instruction::Add(_) => todo!(),
            Instruction::Sub(_) => todo!(),
            Instruction::AddMul(_, _) => todo!(),
            Instruction::SubMul(_, _) => todo!(),
        }
    }

    code
}

fn write_x86_instruction<const SIZE: usize>(code: &mut Vec<u8>, instruction: [u8; SIZE]) {
    _ = code.write_all(&instruction);
}
