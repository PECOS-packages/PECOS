fn main() {
    env_logger::init();
    // Always validate LLVM since this crate requires LLVM
    validate_llvm();
}

fn validate_llvm() {
    use pecos_build::llvm::{LLVM_SYS_PREFIX_ENV, REQUIRED_VERSION, is_valid_llvm};
    use std::env;
    use std::path::PathBuf;

    // Check if LLVM_SYS_PREFIX_ENV is already set and valid
    if let Ok(sys_prefix) = env::var(LLVM_SYS_PREFIX_ENV) {
        let path = PathBuf::from(&sys_prefix);
        if is_valid_llvm(&path) {
            // LLVM is configured and valid, we're good!
            return;
        }
        eprintln!("\n═══════════════════════════════════════════════════════════════");
        eprintln!("ERROR: Invalid {LLVM_SYS_PREFIX_ENV}");
        eprintln!("═══════════════════════════════════════════════════════════════");
        eprintln!();
        eprintln!("{LLVM_SYS_PREFIX_ENV} is set to: {sys_prefix}");
        eprintln!("But this is not a valid LLVM {REQUIRED_VERSION} installation.");
        eprintln!();
        eprintln!("Please either:");
        eprintln!("  1. Fix the path to point to a valid LLVM {REQUIRED_VERSION} installation");
        eprintln!("  2. Unset it and configure LLVM:");
        eprintln!("     unset {LLVM_SYS_PREFIX_ENV}");
        eprintln!("     pecos llvm configure");
        eprintln!("═══════════════════════════════════════════════════════════════\n");
        panic!("Invalid {LLVM_SYS_PREFIX_ENV}. See error message above.");
    }

    // LLVM_SYS_PREFIX_ENV not set - print setup instructions
    print_llvm_not_found_error_extended();
    panic!(
        "LLVM {REQUIRED_VERSION} not configured. See error message above for setup instructions."
    );
}

fn print_llvm_not_found_error_extended() {
    use pecos_build::llvm::{LLVM_SYS_PREFIX_ENV, REQUIRED_VERSION};

    eprintln!("\n═══════════════════════════════════════════════════════════════");
    eprintln!("LLVM {REQUIRED_VERSION} Setup Required for pecos-qir");
    eprintln!("═══════════════════════════════════════════════════════════════");
    eprintln!();
    eprintln!("The pecos-qir crate requires LLVM {REQUIRED_VERSION} for QIR generation.");
    eprintln!("Choose one of these installation methods:");
    eprintln!();
    eprintln!("Option 1: Use pecos setup (recommended)");
    eprintln!("  pecos setup");
    eprintln!("  cargo build");
    eprintln!();
    eprintln!("  This detects and installs all missing dependencies.");
    eprintln!(
        "  (LLVM {REQUIRED_VERSION}: several hundred MB download, installs to ~/.pecos/deps/llvm-{REQUIRED_VERSION}/)"
    );
    eprintln!();

    #[cfg(target_os = "macos")]
    {
        eprintln!("Option 2: Install via Homebrew");
        eprintln!("  # Install LLVM 21");
        eprintln!("  brew install llvm@21");
        eprintln!();
        eprintln!("  # Configure PECOS to use it");
        eprintln!("  pecos llvm configure");
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
        eprintln!("    sudo apt install llvm-21 llvm-21-dev");
        eprintln!();
        eprintln!("  Fedora/RHEL:");
        eprintln!("    sudo dnf install llvm21 llvm21-devel");
        eprintln!();
        eprintln!("  Arch Linux:");
        eprintln!("    # LLVM 21 may need to come from an alternate repository");
        eprintln!("    yay -S llvm21");
        eprintln!();
        eprintln!("  Then configure and build:");
        eprintln!("    pecos llvm configure");
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
        eprintln!("    set {LLVM_SYS_PREFIX_ENV}=C:\\LLVM");
        eprintln!("    pecos llvm configure");
        eprintln!("    cargo build");
        eprintln!();
    }

    eprintln!("Alternative: Set LLVM path manually");
    eprintln!("  Instead of 'configure', you can set environment variables:");
    eprintln!();
    #[cfg(target_os = "windows")]
    eprintln!("    set {LLVM_SYS_PREFIX_ENV}=C:\\path\\to\\llvm");
    #[cfg(not(target_os = "windows"))]
    eprintln!("    export {LLVM_SYS_PREFIX_ENV}=/path/to/llvm");
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
