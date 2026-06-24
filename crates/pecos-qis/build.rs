//! Build script for pecos-qis
//!
//! Handles:
//! - LLVM validation (when `llvm` feature is enabled)
//! - Selene shim and Helios interface building (when `selene` feature is enabled)

use std::env;
use std::path::PathBuf;

#[cfg(feature = "selene")]
#[path = "build_selene.rs"]
mod build_selene;

fn main() {
    // Initialize logger for build script
    env_logger::init();

    // Only run LLVM validation if the llvm feature is enabled
    #[cfg(feature = "llvm")]
    validate_llvm();

    // Embed LLVM bin path at compile time for runtime use
    if let Ok(llvm_prefix) = env::var(pecos_build::llvm::LLVM_SYS_PREFIX_ENV) {
        let llvm_bin = PathBuf::from(&llvm_prefix).join("bin");
        println!("cargo:rustc-env=PECOS_LLVM_BIN_PATH={}", llvm_bin.display());
    }

    // Build Selene-specific components only when the selene feature is enabled
    #[cfg(feature = "selene")]
    build_selene::build_selene_components();
}

#[cfg(feature = "llvm")]
fn validate_llvm() {
    use pecos_build::llvm::{LLVM_SYS_PREFIX_ENV, REQUIRED_VERSION, is_valid_llvm};

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

#[cfg(feature = "llvm")]
fn print_llvm_not_found_error_extended() {
    use pecos_build::llvm::{LLVM_SYS_PREFIX_ENV, REQUIRED_VERSION};

    eprintln!("\n═══════════════════════════════════════════════════════════════");
    eprintln!("LLVM {REQUIRED_VERSION} Setup Required");
    eprintln!("═══════════════════════════════════════════════════════════════");
    eprintln!();
    eprintln!("PECOS needs LLVM {REQUIRED_VERSION}. Choose one of these installation methods:");
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
    eprintln!("Don't need LLVM IR support? Build without it:");
    eprintln!("  cargo build --no-default-features");
    eprintln!("═══════════════════════════════════════════════════════════════\n");
}
