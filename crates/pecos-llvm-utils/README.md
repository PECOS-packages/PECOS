# pecos-llvm-utils

LLVM detection, installation, and management for PECOS.

This crate provides functionality to locate and install LLVM 14 across different platforms (macOS, Linux, Windows).

## Features

- **Automatic LLVM Detection**: Finds LLVM 14 in common system locations
- **LLVM Installation**: Downloads and installs LLVM 14.0.6 to user data directory
- **Cross-platform**: Works on macOS, Linux, and Windows
- **Pure Rust**: No external dependencies (tar, 7zip, etc.) required for installation
- **Build Script Integration**: Can be used in `build.rs` files
- **Command-line Tool**: `pecos-llvm` binary for all LLVM operations

### Installation Details

The installer uses pure Rust dependencies for archive extraction:
- **Unix systems (macOS/Linux)**: Uses `xz2` and `tar` crates for .tar.xz extraction
- **Windows**: Uses `sevenz-rust` crate for .7z extraction

No external tools (tar, 7zip) are required - everything is handled through Rust libraries.

## Command-line Tool: `pecos-llvm`

The `pecos-llvm` binary provides several subcommands:

### Find LLVM

```bash
# Find and print LLVM path
pecos-llvm find

# Print export command for shell evaluation
pecos-llvm find --export
```

### Check LLVM Availability

```bash
# Check if LLVM is available (exit code 0 if found, 1 if not)
pecos-llvm check

# Quiet mode (no output)
pecos-llvm check --quiet
```

### Install LLVM

```bash
# Install LLVM 14.0.6 to ~/.pecos/llvm
pecos-llvm install

# Force reinstall
pecos-llvm install --force
```

### Show Version

```bash
# Show LLVM version information
pecos-llvm version
```

## Usage in build.rs

```rust
use pecos_llvm_utils::{find_llvm_14, get_repo_root_from_manifest, print_llvm_not_found_error};

fn main() {
    let repo_root = get_repo_root_from_manifest();
    match find_llvm_14(repo_root) {
        Some(path) => {
            println!("cargo:warning=Found LLVM 14 at: {}", path.display());
        }
        None => {
            print_llvm_not_found_error();
            panic!("LLVM 14 required but not found");
        }
    }
}
```

## Shell Scripts

The `pecos-llvm` tool can be wrapped in shell scripts:

### Bash/Zsh
```bash
#!/bin/bash
# Install LLVM
cargo run --release -p pecos-llvm-utils --bin pecos-llvm -- install

# Set environment variable
export LLVM_SYS_140_PREFIX=$(cargo run --release -p pecos-llvm-utils --bin pecos-llvm -- find 2>/dev/null)
```

### PowerShell
```powershell
# Install LLVM
cargo run --release -p pecos-llvm-utils --bin pecos-llvm -- install

# Set environment variable
$env:LLVM_SYS_140_PREFIX = (cargo run --release -p pecos-llvm-utils --bin pecos-llvm -- find 2>$null)
```

## Detection Priority

The crate searches for LLVM 14 in the following order:

1. **Home directory**: `~/.pecos/llvm` (where `pecos-llvm install` puts it)
2. **Project-local**: `llvm/` directory (relative to repository root, for backward compatibility)
3. **System installations**:
   - **macOS**: Homebrew installations (`/opt/homebrew/opt/llvm@14`, `/usr/local/opt/llvm@14`)
   - **Linux**: Package manager installations (`/usr/lib/llvm-14`, `/usr/local/llvm-14`)

## License

Apache-2.0
