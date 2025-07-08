# C Libraries

This directory contains C libraries used by PECOS. Each subdirectory contains a separate C library.

## Libraries

### rng_pcg/
PCG random number generator implementation.
- `rng_pcg.h` - Header file with function declarations
- `rng_pcg.c` - Implementation

## Adding New Libraries

To add a new C library:
1. Create a new subdirectory with the library name
2. Place the C source and header files in that directory
3. Update `crates/pecos-clibs/build.rs` to compile the new library
4. Add Rust FFI bindings in `crates/pecos-clibs/src/lib.rs`
5. If needed for Python, add PyO3 wrappers in `python/pecos-rslib/`
