# QIR Runtime Library

The QIR (Quantum Intermediate Representation) compiler in PECOS uses a Rust runtime library to implement quantum operations. This library is automatically built by the `build.rs` script in the `pecos-engines` crate.

## Requirements

To use QIR functionality, you need:

- **LLVM version 14 specifically**:
  - On Linux: Install using your package manager (e.g., `sudo apt install llvm-14`)
  - On macOS: Install using Homebrew (`brew install llvm@14`)
  - On Windows: Download and install LLVM 14.x from the [LLVM website](https://releases.llvm.org/download.html#14.0.0)

- **Required tools**:
  - Linux/macOS: The `llc` compiler tool must be in your PATH
  - Windows: The `clang` compiler must be in your PATH

**Note**: PECOS requires LLVM version 14.x specifically, not newer versions. LLVM 15 or later versions are not compatible with PECOS's QIR implementation.

If LLVM 14 is not installed or the required tools aren't found, QIR functionality will be disabled but the rest of PECOS will continue to work normally.

## How It Works

The `build.rs` script:

1. Runs automatically when building the `pecos-engines` crate
2. Checks for LLVM 14+ dependencies
3. Checks if the QIR runtime library needs to be rebuilt
4. Builds the library only if necessary (if source files have changed)
5. Places the built library in both `target/debug` and `target/release` directories

When the QIR compiler runs, it looks for the pre-built library in these locations. If the library is not found, the compiler will attempt to build it by running `cargo build -p pecos-engines` before raising an error.

## Benefits

- **Faster compilation**: Tests and examples that use QIR run much faster
- **Reduced resource usage**: The QIR runtime is only built when necessary
- **Consistent behavior**: The same runtime library is used for all compilations
- **Automatic**: No manual steps required - everything happens during normal build process
- **Resilient**: Automatically attempts to build the library if it's missing

## Technical Details

The QIR runtime library includes:
- `src/engines/qir/runtime.rs` - The main QIR runtime implementation
- `src/engines/qir/common.rs` - Common utilities for the QIR runtime
- `src/engines/qir/state.rs` - State management for QIR execution
- `src/result_id.rs` - Result ID type definitions
- `src/byte_message/quantum_cmd.rs` - Quantum command definitions

## Troubleshooting

If you encounter issues with the QIR runtime library, you can:

1. Delete the pre-built library to force a rebuild:
   ```bash
   rm -f target/debug/libqir_runtime.a target/release/libqir_runtime.a  # Linux/macOS
   ```
   ```powershell
   Remove-Item -Force target\debug\qir_runtime.lib, target\release\qir_runtime.lib  # Windows
   ```

2. Rebuild the library explicitly:
   ```bash
   cargo clean -p pecos-engines
   cargo build -p pecos-engines
   ```

3. Check the build output for errors by running with verbose output:
   ```bash
   CARGO_LOG=debug cargo build -p pecos-engines
   ```

If the QIR runtime library cannot be found or built, you'll see an error message indicating the issue. This typically means that the `build.rs` script failed to build the library. Check the build output for errors and ensure that all dependencies are installed correctly, including LLVM and Clang which are required for building the QIR runtime.
