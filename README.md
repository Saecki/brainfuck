# brainfuck
A somewhat optimizing brainfuck interpreter and compiler

## Usage

```
brainfuck <mode> [<option>] <path>

modes
    format          pretty print brainfuck code
    ir              print the intermediate representation
    run             interpret the ir
    compile         generate an ELF64 x86-64 system-v executable
    help            print this help message

options
    -v,--verbose                change verbosity level via number of occurences [0..=3]
    -d,--debug                  disable all optimizations
       --no-optimize-zeros      disable zeroing optimization
       --no-optimize-arithmetic disable arithmetic optimizations
       --no-optimize-jumps      disable redundant jump elmination
       --no-optimize-dead-code  disable dead code elmination
```
