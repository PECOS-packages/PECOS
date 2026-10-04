//! Selene-specific build logic for pecos-qis
//!
//! This module handles building the Helios interface.
//! It is only compiled when the `selene` feature is enabled.

use log::info;
use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Build the Selene Helios interface
pub fn build_selene_components() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Find or build libhelios_selene_interface.a
    find_or_build_helios_lib(&out_dir);

    // Note: We don't export Selene runtime paths as environment variables here because
    // the Selene runtimes are dependencies that may not be built yet when this build
    // script runs. Runtime detection is done at runtime instead (see selene_runtimes.rs).
}

fn find_or_build_helios_lib(out_dir: &Path) {
    let helios_lib = out_dir.join("libhelios_selene_interface.a");

    // Check if already exists in our output directory
    if helios_lib.exists() {
        println!("cargo:rustc-env=HELIOS_LIB_PATH={}", helios_lib.display());
        return;
    }

    // Build from Cargo-downloaded Selene dependency
    #[cfg(feature = "selene-runtimes")]
    match build_helios_from_cargo_dependency(out_dir) {
        Ok(()) => {
            println!("cargo:rustc-env=HELIOS_LIB_PATH={}", helios_lib.display());
        }
        Err(e) => {
            panic!("Failed to build Helios interface from Selene dependency: {e}");
        }
    }

    #[cfg(not(feature = "selene-runtimes"))]
    panic!(
        "Failed to build Helios interface library. The selene-runtimes feature must be enabled."
    );
}

/// Build Helios interface library from Cargo-downloaded Selene dependency
#[cfg(feature = "selene-runtimes")]
fn build_helios_from_cargo_dependency(out_dir: &Path) -> Result<(), String> {
    use cargo_metadata::MetadataCommand;

    info!("Building Helios interface from Selene dependency");

    // Get cargo metadata to find Selene source
    let metadata = MetadataCommand::new()
        .exec()
        .map_err(|e| format!("Failed to get cargo metadata: {e}"))?;

    // Find the selene-simple-runtime package (which depends on selene-core)
    let selene_pkg = metadata
        .packages
        .iter()
        .find(|p| p.name == "selene-simple-runtime")
        .ok_or_else(|| "Could not find selene-simple-runtime in cargo metadata".to_string())?;

    // Get the path to the Selene repository root
    // The manifest path is something like .../selene-ext/runtimes/simple/Cargo.toml
    // We need to go up three levels to get to the Selene root
    let manifest_dir = selene_pkg
        .manifest_path
        .parent()
        .and_then(|p| p.parent())
        .and_then(|p| p.parent())
        .and_then(|p| p.parent())
        .ok_or_else(|| "Could not determine Selene root from manifest path".to_string())?;

    let selene_root = manifest_dir.as_std_path();

    // Build Helios interface from Selene source
    let helios_path = selene_root.join("selene-ext/interfaces/helios_qis");
    let interface_c = helios_path.join("c/src/interface.c");
    let helios_include_dir = helios_path.join("c/include");
    let base_qis_include_dir = selene_root.join("selene-ext/interfaces/base_qis/c/include");
    let selene_include_dir = selene_root.join("selene-sim/c/include");

    if !interface_c.exists() {
        return Err(format!(
            "Helios interface.c not found at: {}",
            interface_c.display()
        ));
    }

    let interface_o = out_dir.join("interface.o");
    let helios_lib = out_dir.join("libhelios_selene_interface.a");

    // Compile interface.c to object file
    // Try to find an available C compiler (clang or gcc)
    let compiler = find_c_compiler();
    let mut compile_cmd = Command::new(&compiler);
    compile_cmd.arg("-c");

    // -fPIC is not supported (and not needed) on Windows MSVC
    #[cfg(not(target_os = "windows"))]
    compile_cmd.arg("-fPIC");

    compile_cmd
        .arg("-O2")
        .arg("-std=c11")
        .arg("-D_USE_MATH_DEFINES")
        .arg("-DM_PI=3.14159265358979323846") // Define M_PI directly
        .arg("-DSELENE_LOG_LEVEL=0")
        .arg("-Wno-macro-redefined") // Suppress the redefinition warning
        .arg("-I")
        .arg(&helios_include_dir)
        .arg("-I")
        .arg(&base_qis_include_dir)
        .arg("-I")
        .arg(&selene_include_dir)
        .arg("-o")
        .arg(&interface_o)
        .arg(&interface_c);

    // On macOS, add SDK flags to ensure compiler can find system headers/libraries
    #[cfg(target_os = "macos")]
    add_macos_sdk_flags(&mut compile_cmd);

    let output = compile_cmd
        .output()
        .map_err(|e| format!("Failed to execute clang: {e}"))?;

    if !output.status.success() {
        return Err(format!(
            "Failed to compile interface.c:\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    // Create static library from object file
    let mut ar_cmd = Command::new("ar");
    ar_cmd.arg("rcs").arg(&helios_lib).arg(&interface_o);

    let output = ar_cmd
        .output()
        .map_err(|e| format!("Failed to execute ar: {e}"))?;

    if !output.status.success() {
        return Err(format!(
            "Failed to create libhelios_selene_interface.a:\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    info!("Successfully built Helios interface from Selene dependency");

    // Tell cargo to recompile if Selene files change
    println!("cargo:rerun-if-changed={}", interface_c.display());

    Ok(())
}

/// Find an available C compiler on the system
///
/// Tries to find clang or gcc, in that order of preference.
/// On Windows, just tries "clang" which will be found in PATH if available.
fn find_c_compiler() -> String {
    if cfg!(target_os = "windows") {
        // On Windows, try clang from PATH
        if Command::new("clang").arg("--version").output().is_ok() {
            return "clang".to_string();
        }
        // Fall back to cc which might be MSVC cl.exe
        return "cc".to_string();
    }

    // On Unix-like systems, try various compilers in order
    let compilers = vec![
        "/usr/bin/clang",
        "clang",
        "/usr/bin/gcc",
        "gcc",
        "/usr/bin/cc",
        "cc",
    ];

    for compiler in &compilers {
        if Command::new(compiler).arg("--version").output().is_ok() {
            return (*compiler).to_string();
        }
    }

    // If nothing works, return "cc" and let it fail with a better error
    "cc".to_string()
}

/// Get the macOS SDK path using xcrun
///
/// Returns None if not on macOS or if xcrun fails
#[cfg(target_os = "macos")]
fn get_macos_sdk_path() -> Option<String> {
    let output = Command::new("xcrun")
        .args(["--show-sdk-path"])
        .output()
        .ok()?;

    if output.status.success() {
        Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
    } else {
        None
    }
}

/// Add macOS SDK flags to a compiler command
///
/// This ensures that the compiler can find system headers and libraries
/// when using pre-built LLVM that doesn't have SDK paths configured.
#[cfg(target_os = "macos")]
fn add_macos_sdk_flags(cmd: &mut Command) {
    if let Some(sdk_path) = get_macos_sdk_path() {
        cmd.arg(format!("-isysroot{sdk_path}"));
        cmd.arg(format!("-L{sdk_path}/usr/lib"));
    }
}
