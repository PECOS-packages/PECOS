# pecos-qis

QIS (Quantum Instruction Set) infrastructure for PECOS.

## Purpose

Provides the complete QIS execution pipeline: compiling QIS programs (LLVM IR) and executing them via Selene's quantum simulator.

## Architecture

```
QisEngine
├── QisInterface (compiles programs, collects operations)
│   └── QisHeliosInterface (Selene Helios compiler)
└── QisRuntime (executes quantum operations)
    └── SeleneRuntime (Selene simulator)
```

## Key Types

- `QisEngine` - Classical control engine for QIS programs
- `QisInterface` trait - Program compilation interface
- `QisRuntime` trait - Quantum operation execution
- `QisHeliosInterface` - Selene Helios-based interface (feature: `selene`)
- `SeleneRuntime` - Selene simulator wrapper (feature: `selene`)

## Features

- `selene` (default): Selene-based implementation
- `llvm`: LLVM IR program support

## Usage

```rust
use pecos_qis::{qis_engine, helios_interface_builder, selene_simple_runtime};

let engine = qis_engine()
    .runtime(selene_simple_runtime()?)
    .interface(helios_interface_builder())
    .program(qis_program)
    .build()?;
```

## Runtime libraries in development and tests

Cargo's build-directory layout is internal. PECOS obtains runtime library paths
from Cargo's `compiler-artifact` JSON messages and stages them in the flat
`<cargo target_directory>/pecos-qis-runtime/<debug|release|native>/runtime/` directory.
With `--tests`, the separate `.../<profile>/tests/` directory includes the test fixture.
On Unix, each staged file is replaced atomically using a temporary file and rename.
On Windows, a loaded library cannot be replaced until the process using it exits;
close that process (for example a Python session using `pecos_rslib`) and retry.
Replacement is per file, not per set: a reader racing a restage can see a mix of
old and new libraries. Obsolete files are removed only after all replacements succeed.
`just clean` removes the entire staged directory.
The library set is declared in `pecos-qis`'s `package.metadata.pecos.runtime-libraries`.
On Windows the FFI DLL's Cargo-reported `.dll.lib` is included too.

For Python, run `just build-debug`; both that command and editable/wheel builds
bundle the libraries in `pecos_rslib/_native`. For Rust tests, use
`just rstest debug` or `pecos rust test --profile debug`. For direct Cargo tests:

```sh
runtime_dir=$(cargo run --locked -p pecos-cli -- qis-runtime --profile debug --tests)
export PECOS_QIS_FFI_PATH="$runtime_dir" PECOS_SELENE_DIR="$runtime_dir"
cargo test --locked -p pecos-qis --lib
```

Use the same profile and build environment for preparation and execution.
`qis-runtime --target-dir <path>` and `CARGO_TARGET_DIR` select Cargo's target
directory. Omit `--tests` when the descriptor test fixture is not needed.

`pecos run` and `pecos compile` do not build runtime libraries implicitly.
The FFI lookup order is `PECOS_QIS_FFI_PATH`, the registered bundle, then a library
directly beside the executable. `PECOS_QIS_FFI_PATH` accepts a file or directory;
an invalid value fails immediately. Selene runtime constructors use
`PECOS_SELENE_DIR`, then the bundle when the variable is unset. An invalid
`PECOS_SELENE_DIR` fails immediately. The separate `find_selene_runtime` utility
also searches `/usr/local/lib`, `/usr/lib`, and `/opt/pecos/lib`.
Every directory is searched only for the exact platform filename directly inside it;
`deps/` subdirectories and hash-suffixed filenames are never searched.
No runtime lookup walks Cargo target directories, workspace parents, or the cwd.

## Acknowledgements

This crate integrates with [Selene](https://github.com/Quantinuum/selene), a quantum computer emulation platform developed by Quantinuum.
