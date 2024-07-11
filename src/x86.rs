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
    (modrm.mode() << 6) | ((reg as u8) << 3) | modrm.rm()
}

/// Generate a `MOD-REG_R/M` byte with an op-code extension
pub const fn ext_modrm(modrm: ModRm, ext: u8) -> u8 {
    (modrm.mode() << 6) | (ext << 3) | modrm.rm()
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

#[repr(C)]
struct ElfFileHeader {
    ei_magic: [u8; 4],
    ei_class: u8,
    ei_data: u8,
    ei_version: u8,
    ei_osabi: u8,
    ei_abiversion: u8,
    ei_pad: [u8; 7],

    e_type: u16,
    e_machine: u16,
    e_version: u32,
    e_entry: u64,
    /// program header offset
    e_phoff: u64,
    /// section header offset
    e_shoff: u64,
    e_flags: u32,
    /// size of this header
    e_ehsize: u16,
    /// program header entry size
    e_phentsize: u16,
    /// number of program header table entries
    e_phnum: u16,
    /// section header entry size
    e_shentsize: u16,
    /// number of section header table entries
    e_shnum: u16,
    /// index of section header table entry that contains section names
    e_shstrndx: u16,
}

#[repr(C)]
struct ElfProgramHeader {
    p_type: u32,
    p_flags: u32,
    p_offset: u64,
    p_vaddr: u64,
    p_paddr: u64,
    p_filesz: u64,
    p_memsz: u64,
    p_align: u64,
}

#[macro_export]
macro_rules! const_assert {
    ($x:expr $(,)?) => {
        #[allow(unknown_lints, eq_op)]
        const _: [(); 0 - !{
            const ASSERT: bool = $x;
            ASSERT
        } as usize] = [];
    };
}

/// Generate a 64-bit x86 linux ELF binary
pub fn compile(instructions: &[Instruction]) -> Vec<u8> {
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
            p_offset: PROGRAM_OFFSET as u64,          // (includes the whole file)
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

    // prepare brainfuck registers
    {
        // `81 /5 id`: allocate stack space for 30000 elements
        let modrm = const { ext_modrm(ModRm::Register(Reg::Esp), 5) };
        let [b0, b1, b2, b3] = u32::to_le_bytes(NUM_REGISTERS as u32);
        write_instruction(&mut code, [0x81, modrm, b0, b1, b2, b3]);

        // `C7 /0 id`: write ITERATIONS to `ecx`
        const NUM_ITERATIONS: u32 = NUM_REGISTERS as u32 / 4;
        let modrm = const { ext_modrm(ModRm::Register(Reg::Ecx), 0) };
        let [b0, b1, b2, b3] = u32::to_le_bytes(NUM_ITERATIONS);
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
            Instruction::Output => {
                // `C7 /0 id`: move immediate value to `eax`
                let modrm = const { ext_modrm(ModRm::Register(Reg::Eax), 0) };
                const WRITE_SYSCALL: u32 = 1;
                let [b0, b1, b2, b3] = u32::to_le_bytes(WRITE_SYSCALL);
                write_instruction(&mut code, [0xC6, modrm, b0, b1, b2, b3]);

                // `C7 /0 id`: move immediate value to `edi`
                let modrm = const { ext_modrm(ModRm::Register(Reg::Edi), 0) };
                const STDOUT_FD: u32 = 1;
                let [b0, b1, b2, b3] = u32::to_le_bytes(STDOUT_FD);
                write_instruction(&mut code, [0xC6, modrm, b0, b1, b2, b3]);

                // write address of string to `esi`
                // `89 /r`: move from `esp` to `esi`
                let modrm = const { normal_modrm(ModRm::Register(Reg::Esi), Reg::Esp) };
                write_instruction(&mut code, [0x89, modrm]);
                // `01 /r`: add `ecx` to `esi`
                let modrm = const { normal_modrm(ModRm::Register(Reg::Esi), Reg::Ecx) };
                write_instruction(&mut code, [0x01, modrm]);

                // `C7 /0 id`: move immediate value to `edx`
                let modrm = const { ext_modrm(ModRm::Register(Reg::Edx), 0) };
                const STRING_LEN: u32 = 1;
                let [b0, b1, b2, b3] = u32::to_le_bytes(STRING_LEN);
                write_instruction(&mut code, [0xC6, modrm, b0, b1, b2, b3]);

                // `0F 05`: SYSCALL - fast system call
                write_instruction(&mut code, [0x0F, 0x05]);
            }
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
        let size = code.len() as u64 - PROGRAM_OFFSET as u64;
        let program_header = &mut code[B64_ELF_HEADER_LEN..][..B64_PROGRAM_HEADER_LEN];
        const P_FILESZ: usize = std::mem::offset_of!(ElfProgramHeader, p_filesz);
        program_header[P_FILESZ..P_FILESZ + 8].copy_from_slice(&u64::to_le_bytes(size));
        const P_MEMSZ: usize = std::mem::offset_of!(ElfProgramHeader, p_memsz);
        program_header[P_MEMSZ..P_MEMSZ + 8].copy_from_slice(&u64::to_le_bytes(size));
    }

    code
}

fn write_instruction<const SIZE: usize>(code: &mut Vec<u8>, instruction: [u8; SIZE]) {
    code.extend_from_slice(&instruction);
}
