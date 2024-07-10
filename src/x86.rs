use crate::{Instruction, NUM_REGISTERS};

#[derive(Clone, Copy)]
pub enum ModRm {
    Indirect(RmI),
    Indirect1ByteDisp(RmID),
    Indirect4ByteDisp(RmID),
    Register(Reg),
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

/// R/M value for indirect addressing mode
#[derive(Clone, Copy)]
pub enum RmI {
    RegEax = 0b000,
    RegEcx = 0b001,
    RegEdx = 0b010,
    RegEbx = 0b011,
    Sib = 0b100,
    Disp4byte = 0b101,
    RegEsi = 0b110,
    RegEdi = 0b111,
}

/// R/M value for indirect addressing mode with displacement
#[derive(Clone, Copy)]
pub enum RmID {
    RegEax = 0b000,
    RegEcx = 0b001,
    RegEdx = 0b010,
    RegEbx = 0b011,
    Sib = 0b100,
    RegEbp = 0b101,
    RegEsi = 0b110,
    RegEdi = 0b111,
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

/// Generate a `MOD-REG_R/M` byte with a `reg` field
pub const fn normal_modrm(modrm: ModRm, reg: Reg) -> u8 {
    ((modrm.mode() as u8) << 6) | ((reg as u8) << 3) | (modrm.rm() as u8)
}

/// Generate a `MOD-REG_R/M` byte with an op-code extension
pub const fn ext_modrm(modrm: ModRm, ext: u8) -> u8 {
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

pub const fn gen_sib(scale: Scale, index: Reg, base: Reg) -> u8 {
    ((scale as u8) << 6) | ((index as u8) << 3) | (base as u8)
}

/// Generate a 32-bit x86 linux ELF binary
pub fn compile(instructions: &[Instruction]) -> Vec<u8> {
    const B32BIT_ELF_HEADER_LEN: usize = 0x34;
    const B32BIT_PROGRAM_HEADER_LEN: usize = 0x20;
    const ENTRY_POINT: usize = B32BIT_ELF_HEADER_LEN + B32BIT_PROGRAM_HEADER_LEN;

    let mut elf_header = [0u8; B32BIT_ELF_HEADER_LEN];
    {
        // e_ident
        elf_header[0x00..0x04].copy_from_slice(b"\x7fELF"); // EI_MAG
        elf_header[0x04] = 0x1; // EI_CLASS      : 32-bit
        elf_header[0x05] = 0x1; // EI_DATA       : little-endian
        elf_header[0x06] = 0x1; // EI_VERSION    : 1
        elf_header[0x07] = 0x3; // EI_OSABI      : linux
        elf_header[0x08] = 0x0; // EI_ABIVERSION : 0
                                // EI_PAD        : reserved

        // e_type: executable
        elf_header[0x10..0x12].copy_from_slice(&u16::to_le_bytes(0x0002));

        // e_machine: AMD x86-64
        elf_header[0x12..0x14].copy_from_slice(&u16::to_le_bytes(0x003E));

        // e_version: 1
        elf_header[0x14..0x18].copy_from_slice(&u32::to_le_bytes(0x00000001));

        // e_entry: entry point offset
        elf_header[0x18..0x1C].copy_from_slice(&u32::to_le_bytes(ENTRY_POINT as u32));

        // e_phoff: program header table offset immediately follows the ELF header
        elf_header[0x1C..0x20].copy_from_slice(&u32::to_le_bytes(B32BIT_ELF_HEADER_LEN as u32));

        // e_shoff: section header table offset
        // no table thus 0 offset

        // e_flags: no flags

        // e_ehsize: ELF header size is 52 (0x34) for 32-bit binaries
        elf_header[0x28..0x2A].copy_from_slice(&u16::to_le_bytes(B32BIT_ELF_HEADER_LEN as u16));

        // e_phentsize: program header table size
        elf_header[0x2A..0x2C].copy_from_slice(&u16::to_le_bytes(B32BIT_PROGRAM_HEADER_LEN as u16));

        // e_phnum: program header table entry count
        elf_header[0x2C..0x2E].copy_from_slice(&u16::to_le_bytes(1));

        // e_shentsize: section header table size
        // no table thus irrelevant

        // e_shnum: section header table entry count
        // no table thus 0 entries

        // e_shstrndx: section header table entry index that contains the section names
        // no table thus irrelevant
    }

    let mut program_header = [0u8; B32BIT_PROGRAM_HEADER_LEN];
    {
        // p_type: loadable segment
        program_header[0x00..0x04].copy_from_slice(&u32::to_le_bytes(0x00000001));

        // p_offset: 0 (includes the whole file)

        // p_vaddr: virtual address to place the segment at
        program_header[0x08..0x0C].copy_from_slice(&u32::to_le_bytes(0x00001000));

        // p_paddr: physical address is not used (just set it to the same as the virtual address)
        program_header[0x0C..0x10].copy_from_slice(&u32::to_le_bytes(0x00001000));

        // p_filesz: size of the whole file
        // write after code generation

        // p_memsz: size of the whole file
        // write after code generation

        // p_flags: executable
        program_header[0x18..0x1C].copy_from_slice(&u32::to_le_bytes(0x00000001));

        // p_align: no alignment
        program_header[0x1C..0x20].copy_from_slice(&u32::to_le_bytes(0x00000001));
    }

    let mut code = (elf_header.iter().copied())
        .chain(program_header.iter().copied())
        .collect();

    // prepare brainfuck registers
    {
        // `81 /5 id`: allocate stack space for 30000 elements
        let modrm = const { ext_modrm(ModRm::Register(Reg::Esp), 5) };
        let [b0, b1, b2, b3] = u32::to_le_bytes(NUM_REGISTERS as u32);
        write_instruction(&mut code, [0x81, modrm, b0, b1, b2, b3]);

        // `C7 /0 id`: write ITERATIONS to `ecx`
        const NUM_ITERATIONS: u32 = NUM_REGISTERS as u32 / 4;
        let modrm = const { ext_modrm(ModRm::Register(Reg::Ecx), 0) };
        let [b0, b1, b2, b3] = u32::to_le_bytes(NUM_ITERATIONS as u32);
        write_instruction(&mut code, [0xC7, modrm, b0, b1, b2, b3]);

        let loop_start = code.len();
        // `83 /5 ib`: subtract 1 from `ecx`
        let modrm = const { ext_modrm(ModRm::Register(Reg::Ecx), 5) };
        write_instruction(&mut code, [0x83, modrm, 0x1]);

        // `C7 /0 id`: write 0u32 to stack at `esp + 4 * ecx` using a scaled index byte (SIB)
        let modrm = const { ext_modrm(ModRm::Indirect(RmI::Sib), 0) };
        let sib = const { gen_sib(Scale::B4, Reg::Ecx, Reg::Esp) };
        let [b0, b1, b2, b3] = u32::to_le_bytes(0);
        write_instruction(&mut code, [0xC7, modrm, sib, b0, b1, b2, b3]);

        // `83 /7 ib`: cmp `ecx` to `0`
        let modrm = const { ext_modrm(ModRm::Register(Reg::Ecx), 7) };
        write_instruction(&mut code, [0x83, modrm, 0x00]);

        let loop_end = code.len();
        // `75 cb`: jump if not 0
        let [rel_jump] = i8::to_le_bytes((loop_start as isize - loop_end as isize) as i8);
        write_instruction(&mut code, [0x75, rel_jump]);
    }

    // generate code

    // stores if the jump is redundant, and the location before opening jump (`[`), the jump offset
    // is stored inside the [6..10] bytes after that
    let mut jump_stack = Vec::new();
    for inst in instructions.iter() {
        match inst {
            Instruction::Shl(n) => {
                // `81 /5 id`: sub immediate value from `ecx`
                let [b0, b1] = u16::to_le_bytes(*n);
                let modrm = const { ext_modrm(ModRm::Register(Reg::Ecx), 5) };
                write_instruction(&mut code, [0x81, modrm, b0, b1, 0, 0]);
            }
            Instruction::Shr(n) => {
                // `81 /0 id`: add immediate value to `ecx`
                let [b0, b1] = u16::to_le_bytes(*n);
                let modrm = const { ext_modrm(ModRm::Register(Reg::Ecx), 0) };
                write_instruction(&mut code, [0x81, modrm, b0, b1, 0, 0]);
            }
            Instruction::Inc(n) => {
                // `80 /0 ib`: add immediate value to `esp + 1 * ecx`
                let modrm = const { ext_modrm(ModRm::Indirect(RmI::Sib), 0) };
                let sib = const { gen_sib(Scale::B1, Reg::Ecx, Reg::Esp) };
                write_instruction(&mut code, [0x80, modrm, sib, *n]);
            }
            Instruction::Dec(n) => {
                // `80 /5 ib`: add immediate value to `esp + 1 * ecx`
                let modrm = const { ext_modrm(ModRm::Indirect(RmI::Sib), 5) };
                let sib = const { gen_sib(Scale::B1, Reg::Ecx, Reg::Esp) };
                write_instruction(&mut code, [0x80, modrm, sib, *n]);
            }
            Instruction::Output => todo!(),
            Instruction::Input => todo!(),
            Instruction::JumpZ(jump) => {
                let pos = code.len();
                let redundant = jump.is_redundant();
                if !redundant {
                    // `80 /7 ib`: cmp with `0`
                    let modrm = const { ext_modrm(ModRm::Indirect(RmI::Sib), 7) };
                    let sib = const { gen_sib(Scale::B1, Reg::Ecx, Reg::Esp) };
                    write_instruction(&mut code, [0x80, modrm, sib, 0x00]);

                    // `0F 84 cd`: jump if zero
                    let [b0, b1, b2, b3] = u32::to_le_bytes(0);
                    write_instruction(&mut code, [0x0F, 0x84, b0, b1, b2, b3]);
                }

                jump_stack.push((redundant, pos));
            }
            Instruction::JumpNz(jump) => {
                let Some((start_redundant, start_pos)) = jump_stack.pop() else {
                    unreachable!()
                };

                let pos = code.len();
                let offset = pos as i32 - start_pos as i32;
                let redundant = jump.is_redundant();
                if !redundant {
                    let offset = offset - (10 * start_redundant as i32);

                    // `80 /7 ib`: cmp with `0`
                    let modrm = const { ext_modrm(ModRm::Indirect(RmI::Sib), 7) };
                    let sib = const { gen_sib(Scale::B1, Reg::Ecx, Reg::Esp) };
                    write_instruction(&mut code, [0x80, modrm, sib, 0x00]);

                    // `0F 85 cd`: jump if not zero
                    let [b0, b1, b2, b3] = i32::to_le_bytes(offset);
                    write_instruction(&mut code, [0x0F, 0x85, b0, b1, b2, b3]);
                }

                if !start_redundant {
                    let offset = offset + (10 * (!redundant) as i32);
                    code[start_pos + 6..start_pos + 10].copy_from_slice(&i32::to_le_bytes(offset));
                }
            }

            // TODO: use Indirect or Indirect1ByteDisp addressing modes if possible to save space
            Instruction::Zero(disp) => {
                // `C6 /0 ib`: move immediate value to `esp + 1 * ecx + disp` using sib and disp
                let modrm = const { ext_modrm(ModRm::Indirect4ByteDisp(RmID::Sib), 0) };
                let sib = const { gen_sib(Scale::B1, Reg::Ecx, Reg::Esp) };
                let [b0, b1, b2, b3] = i32::to_le_bytes(*disp as i32);
                write_instruction(&mut code, [0xC6, modrm, sib, b0, b1, b2, b3, 0x00]);
            }
            Instruction::Add(disp) => {
                // `8A /r`: move from `esp + 1 * ecx` to `al` using sib
                let modrm = const { normal_modrm(ModRm::Indirect(RmI::Sib), Reg::Eax) };
                let sib = const { gen_sib(Scale::B1, Reg::Ecx, Reg::Esp) };
                write_instruction(&mut code, [0x8A, modrm, sib]);

                // `00 /r`: add `al` to `esp + 1 * ecx + disp` using sib and disp
                let modrm = const { normal_modrm(ModRm::Indirect4ByteDisp(RmID::Sib), Reg::Eax) };
                let sib = const { gen_sib(Scale::B1, Reg::Ecx, Reg::Esp) };
                let [b0, b1, b2, b3] = i32::to_le_bytes(*disp as i32);
                write_instruction(&mut code, [0x00, modrm, sib, b0, b1, b2, b3]);
            }
            Instruction::Sub(disp) => {
                // `8A /r`: move from `esp + 1 * ecx` to `al` using sib
                let modrm = const { normal_modrm(ModRm::Indirect(RmI::Sib), Reg::Eax) };
                let sib = const { gen_sib(Scale::B1, Reg::Ecx, Reg::Esp) };
                write_instruction(&mut code, [0x8A, modrm, sib]);

                // `28 /r`: subtract `al` from `esp + 1 * ecx + disp` using sib and disp
                let modrm = const { normal_modrm(ModRm::Indirect4ByteDisp(RmID::Sib), Reg::Eax) };
                let sib = const { gen_sib(Scale::B1, Reg::Ecx, Reg::Esp) };
                let [b0, b1, b2, b3] = i32::to_le_bytes(*disp as i32);
                write_instruction(&mut code, [0x28, modrm, sib, b0, b1, b2, b3]);
            }
            Instruction::AddMul(disp, n) => {
                // `8A /r`: move from `esp + 1 * ecx` to `al` using sib
                let modrm = const { normal_modrm(ModRm::Indirect(RmI::Sib), Reg::Eax) };
                let sib = const { gen_sib(Scale::B1, Reg::Ecx, Reg::Esp) };
                write_instruction(&mut code, [0x8A, modrm, sib]);

                // `C6 /0 ib`: move immediate value `dl`
                let modrm = const { ext_modrm(ModRm::Register(Reg::Edx), 0) };
                write_instruction(&mut code, [0xC6, modrm, *n]);

                // `F6 /4`: multiply `al` with `dl` into `eax`
                let modrm = const { ext_modrm(ModRm::Register(Reg::Edx), 4) };
                write_instruction(&mut code, [0xF6, modrm]);

                // `00 /r`: add `al` to `esp + 1 * ecx + disp` using sib and disp
                let modrm = const { normal_modrm(ModRm::Indirect4ByteDisp(RmID::Sib), Reg::Eax) };
                let sib = const { gen_sib(Scale::B1, Reg::Ecx, Reg::Esp) };
                let [b0, b1, b2, b3] = i32::to_le_bytes(*disp as i32);
                write_instruction(&mut code, [0x00, modrm, sib, b0, b1, b2, b3]);
            }
            Instruction::SubMul(disp, n) => {
                // `8A /r`: move from `esp + 1 * ecx` to `al` using sib
                let modrm = const { normal_modrm(ModRm::Indirect(RmI::Sib), Reg::Eax) };
                let sib = const { gen_sib(Scale::B1, Reg::Ecx, Reg::Esp) };
                write_instruction(&mut code, [0x8A, modrm, sib]);

                // `C6 /0 ib`: move immediate value `dl`
                let modrm = const { ext_modrm(ModRm::Register(Reg::Edx), 0) };
                write_instruction(&mut code, [0xC6, modrm, *n]);

                // `F6 /4`: multiply `al` with `dl` into `eax`
                let modrm = const { ext_modrm(ModRm::Register(Reg::Edx), 4) };
                write_instruction(&mut code, [0xF6, modrm]);

                // `28 /r`: subtract `al` from `esp + 1 * ecx + disp` using sib and disp
                let modrm = const { normal_modrm(ModRm::Indirect4ByteDisp(RmID::Sib), Reg::Eax) };
                let sib = const { gen_sib(Scale::B1, Reg::Ecx, Reg::Esp) };
                let [b0, b1, b2, b3] = i32::to_le_bytes(*disp as i32);
                write_instruction(&mut code, [0x28, modrm, sib, b0, b1, b2, b3]);
            }
        }
    }

    // update loadable segment size
    {
        let size = code.len() as u32;
        let program_header = &mut code[B32BIT_ELF_HEADER_LEN..][..B32BIT_PROGRAM_HEADER_LEN];
        // p_filesz: size of the whole file
        program_header[0x10..0x14].copy_from_slice(&u32::to_le_bytes(size));
        // p_memsz: size of the whole file
        program_header[0x14..0x18].copy_from_slice(&u32::to_le_bytes(size));
    }

    code
}

fn write_instruction<const SIZE: usize>(code: &mut Vec<u8>, instruction: [u8; SIZE]) {
    code.extend_from_slice(&instruction);
}
