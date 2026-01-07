fn main() {
    // Always validate LLVM since this crate requires LLVM
    validate_llvm();
}

fn validate_llvm() {
    use pecos_build::llvm::is_valid_llvm_14;
    use std::env;
    use std::path::PathBuf;

    // Check if LLVM_SYS_140_PREFIX is already set and valid
    if let Ok(sys_prefix) = env::var("LLVM_SYS_140_PREFIX") {
        let path = PathBuf::from(&sys_prefix);
        if is_valid_llvm_14(&path) {
            // LLVM is configured and valid, we're good!
            return;
        }
        eprintln!("\n═══════════════════════════════════════════════════════════════");
        eprintln!("ERROR: Invalid LLVM_SYS_140_PREFIX");
        eprintln!("═══════════════════════════════════════════════════════════════");
        eprintln!();
        eprintln!("LLVM_SYS_140_PREFIX is set to: {sys_prefix}");
        eprintln!("But this is not a valid LLVM 14 installation.");
        eprintln!();
        eprintln!("Please either:");
        eprintln!("  1. Fix the path to point to a valid LLVM 14 installation");
        eprintln!("  2. Unset it and configure LLVM:");
        eprintln!("     unset LLVM_SYS_140_PREFIX");
        eprintln!("     cargo run -p pecos -- llvm configure");
        eprintln!("═══════════════════════════════════════════════════════════════\n");
        panic!("Invalid LLVM_SYS_140_PREFIX. See error message above.");
    }

    // LLVM_SYS_140_PREFIX not set - print setup instructions
    print_llvm_not_found_error_extended();
    panic!("LLVM 14 not configured. See error message above for setup instructions.");
}

fn print_llvm_not_found_error_extended() {
    eprintln!("\n═══════════════════════════════════════════════════════════════");
    eprintln!("LLVM 14 Setup Required for pecos-qir");
    eprintln!("═══════════════════════════════════════════════════════════════");
    eprintln!();
    eprintln!("The pecos-qir crate requires LLVM 14 for QIR generation.");
    eprintln!("Choose one of these installation methods:");
    eprintln!();
    eprintln!("Option 1: Use pecos-llvm installer (recommended)");
    eprintln!("  cargo run -p pecos -- llvm install");
    eprintln!("  cargo build");
    eprintln!();
    eprintln!("  The installer automatically configures PECOS.");
    eprintln!("  (Downloads LLVM 14.0.6 to ~/.pecos/llvm/ - ~400MB, ~5 minutes)");
    eprintln!();

    #[cfg(target_os = "macos")]
    {
        eprintln!("Option 2: Install via Homebrew");
        eprintln!("  # Install LLVM 14");
        eprintln!("  brew install llvm@14");
        eprintln!();
        eprintln!("  # Configure PECOS to use it");
        eprintln!("  cargo run -p pecos -- llvm configure");
        eprintln!();
        eprintln!("  # Build PECOS");
        eprintln!("  cargo build");
        eprintln!();
        eprintln!("  Note: Works on both Intel and Apple Silicon Macs");
        eprintln!();
    }

    #[cfg(target_os = "linux")]
    {
        eprintln!("Option 2: Install via system package manager");
        eprintln!();
        eprintln!("  Debian/Ubuntu:");
        eprintln!("    sudo apt update");
        eprintln!("    sudo apt install llvm-14 llvm-14-dev");
        eprintln!();
        eprintln!("  Fedora/RHEL:");
        eprintln!("    sudo dnf install llvm14 llvm14-devel");
        eprintln!();
        eprintln!("  Arch Linux:");
        eprintln!("    # LLVM 14 may need to be built from AUR");
        eprintln!("    yay -S llvm14");
        eprintln!();
        eprintln!("  Then configure and build:");
        eprintln!("    cargo run -p pecos -- llvm configure");
        eprintln!("    cargo build");
        eprintln!();
    }

    #[cfg(target_os = "windows")]
    {
        eprintln!("Option 2: Manual installation (advanced)");
        eprintln!();
        eprintln!("  WARNING: The official LLVM installer lacks development files.");
        eprintln!("  You need a FULL development package from community sources:");
        eprintln!();
        eprintln!("  Recommended sources:");
        eprintln!("    https://github.com/bitgate/llvm-windows-full-builds");
        eprintln!("    https://github.com/vovkos/llvm-package-windows");
        eprintln!();
        eprintln!("  After extracting to C:\\LLVM (or similar):");
        eprintln!("    set LLVM_SYS_140_PREFIX=C:\\LLVM");
        eprintln!("    cargo run -p pecos -- llvm configure");
        eprintln!("    cargo build");
        eprintln!();
    }

    eprintln!("Alternative: Set LLVM path manually");
    eprintln!("  Instead of 'configure', you can set environment variables:");
    eprintln!();
    #[cfg(target_os = "windows")]
    eprintln!("    set LLVM_SYS_140_PREFIX=C:\\path\\to\\llvm");
    #[cfg(not(target_os = "windows"))]
    eprintln!("    export LLVM_SYS_140_PREFIX=/path/to/llvm");
    #[cfg(not(target_os = "windows"))]
    eprintln!("  Or add llvm-config to PATH:");
    #[cfg(not(target_os = "windows"))]
    eprintln!("    export PATH=\"/path/to/llvm/bin:$PATH\"");
    eprintln!();
    eprintln!("For detailed instructions, see:");
    eprintln!(
        "  https://github.com/PECOS-packages/PECOS/blob/master/docs/user-guide/getting-started.md"
    );
    eprintln!();
    eprintln!("═══════════════════════════════════════════════════════════════\n");
}
