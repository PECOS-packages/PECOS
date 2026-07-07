//! Selene-specific build logic for pecos-qis
//!
//! This module handles building the Selene shim library and Helios interface.
//! It is only compiled when the `selene` feature is enabled.

use log::info;
#[cfg(target_os = "windows")]
use pecos_build::llvm::LLVM_SYS_PREFIX_ENV;
use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Build all Selene-specific components (shim library, Helios interface)
pub fn build_selene_components() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Find or build libhelios_selene_interface.a
    find_or_build_helios_lib(&out_dir);

    // Note: We don't export Selene runtime paths as environment variables here because
    // the Selene runtimes are dependencies that may not be built yet when this build
    // script runs. Runtime detection is done at runtime instead (see selene_runtimes.rs).

    // Tell cargo to rerun this build script if pecos-qis-ffi changes
    println!("cargo:rerun-if-changed=../pecos-qis-ffi/src");

    // Build the PECOS Selene shim library
    let output_file = build_shim_library(&out_dir);

    // Create Windows import library if needed
    create_windows_import_library(&out_dir);

    // Set environment variable so Rust code can find the shim
    println!(
        "cargo:rustc-env=PECOS_SELENE_SHIM_PATH={}",
        output_file.display()
    );

    // Tell cargo to recompile if the C source changes
    println!("cargo:rerun-if-changed=src/c/selene_shim.c");
}

/// Build the PECOS Selene shim library as a shared library
fn build_shim_library(out_dir: &Path) -> PathBuf {
    // Build our PECOS shim with undefined __quantum__* symbols
    // These will be resolved at runtime from libpecos_qis_ffi.so/.dylib/.dll
    let source_file = PathBuf::from("src/c/selene_shim.c");
    let output_file = if cfg!(target_os = "macos") {
        out_dir.join("libpecos_selene.dylib")
    } else if cfg!(target_os = "windows") {
        out_dir.join("pecos_selene.dll")
    } else {
        out_dir.join("libpecos_selene.so")
    };

    // Build the C shim as a shared library with undefined __quantum__* symbols
    // These symbols will be resolved from libpecos_qis_ffi.so at runtime
    // Try to find an available C compiler (clang or gcc)
    let compiler = find_c_compiler();
    let mut cmd = Command::new(&compiler);
    cmd.arg("-shared");

    // -fPIC is not supported (and not needed) on Windows MSVC
    #[cfg(not(target_os = "windows"))]
    cmd.arg("-fPIC");

    cmd.arg("-O2").arg("-o").arg(&output_file).arg(&source_file);

    // -lm is not needed on Windows
    #[cfg(not(target_os = "windows"))]
    cmd.arg("-lm");

    // On macOS, we need to allow undefined symbols
    if cfg!(target_os = "macos") {
        cmd.arg("-undefined");
        cmd.arg("dynamic_lookup");
    }

    // On Windows, link against pecos_qis_ffi import library instead of allowing undefined symbols
    if cfg!(target_os = "windows") {
        // Find the pecos-qis-ffi import library in the target directory
        let target_dir = env::var("OUT_DIR")
            .map(|d| {
                PathBuf::from(d)
                    .parent()
                    .unwrap()
                    .parent()
                    .unwrap()
                    .parent()
                    .unwrap()
                    .to_path_buf()
            })
            .expect("OUT_DIR not set");

        let qis_ffi_lib = target_dir.join("deps").join("pecos_qis_ffi.dll.lib");

        if qis_ffi_lib.exists() {
            println!(
                "cargo:warning=Linking shim against QIS FFI import library: {}",
                qis_ffi_lib.display()
            );
            cmd.arg(qis_ffi_lib.to_str().unwrap());
        } else {
            println!(
                "cargo:warning=QIS FFI import library not found at {}, symbols may not resolve",
                qis_ffi_lib.display()
            );
            // Fall back to allowing unresolved symbols
            cmd.arg("-Wl,/FORCE:UNRESOLVED");
        }
    }

    // Add include paths if needed
    if let Ok(selene_include) = env::var("SELENE_INCLUDE_PATH") {
        cmd.arg(format!("-I{selene_include}"));
    }

    // On macOS, add SDK flags to ensure compiler can find system headers/libraries
    #[cfg(target_os = "macos")]
    add_macos_sdk_flags(&mut cmd);

    let output = cmd.output().expect("Failed to execute clang");

    assert!(
        output.status.success(),
        "Failed to compile selene shim:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    output_file
}

/// Create a Windows import library for the shim DLL
#[cfg(target_os = "windows")]
fn create_windows_import_library(out_dir: &Path) {
    let import_lib = out_dir.join("pecos_selene.lib");
    let def_file = out_dir.join("pecos_selene.def");

    // Create a .def file listing all exported selene_* functions
    let def_content = r"LIBRARY pecos_selene.dll
EXPORTS
    selene_qalloc
    selene_qfree
    selene_rxy
    selene_rz
    selene_rzz
    selene_qubit_reset
    selene_qubit_measure
    selene_qubit_lazy_measure
    selene_qubit_lazy_measure_leaked
    selene_future_read_bool
    selene_future_read_u64
    selene_refcount_increment
    selene_refcount_decrement
    selene_print_bool
    selene_print_i64
    selene_print_u64
    selene_print_f64
    selene_print_bool_array
    selene_print_i64_array
    selene_print_u64_array
    selene_print_f64_array
    selene_print_panic
    selene_dump_state
    selene_set_tc
    selene_get_tc
    selene_get_current_shot
    selene_local_barrier
    selene_global_barrier
    selene_shot_count
    selene_on_shot_start
    selene_on_shot_end
    selene_load_config
    selene_exit
    selene_print_exit
    selene_random_seed
    selene_random_advance
    selene_random_u32
    selene_random_u32_bounded
    selene_random_f64
    selene_custom_runtime_call
    pecos_call_qmain_with_setjmp
    pecos_call_void_main_with_setjmp
";

    std::fs::write(&def_file, def_content).expect("Failed to write .def file");

    // Try to use llvm-dlltool (from LLVM) or dlltool (from MinGW) to generate import library
    // First try llvm-dlltool which should be available with our LLVM installation
    let dlltool_result = if let Ok(llvm_prefix) = env::var(LLVM_SYS_PREFIX_ENV) {
        let llvm_dlltool = PathBuf::from(llvm_prefix)
            .join("bin")
            .join("llvm-dlltool.exe");
        if llvm_dlltool.exists() {
            Command::new(&llvm_dlltool)
                .arg("-m")
                .arg("i386:x86-64")
                .arg("-d") // Use -d for .def file input
                .arg(&def_file)
                .arg("-l")
                .arg(&import_lib)
                .output()
        } else {
            Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "llvm-dlltool not found",
            ))
        }
    } else {
        Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("{LLVM_SYS_PREFIX_ENV} not set"),
        ))
    };

    // If llvm-dlltool failed, try regular dlltool (from MinGW/MSYS2)
    let dlltool_result = dlltool_result.or_else(|_| {
        Command::new("dlltool")
            .arg("-m")
            .arg("i386:x86-64")
            .arg("-d") // Use -d for .def file input
            .arg(&def_file)
            .arg("-l")
            .arg(&import_lib)
            .output()
    });

    if let Ok(output) = dlltool_result {
        if output.status.success() {
            info!("Generated import library: {}", import_lib.display());
            // Set environment variable for the import library path
            println!(
                "cargo:rustc-env=PECOS_SELENE_SHIM_LIB={}",
                import_lib.display()
            );
        } else {
            println!("cargo:warning=Failed to generate import library with dlltool");
            println!(
                "cargo:warning=stderr: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
    } else {
        println!("cargo:warning=Could not find llvm-dlltool or dlltool to generate import library");
        println!("cargo:warning=Linking against the shim may fail on Windows");
    }
}

/// No-op on non-Windows platforms
#[cfg(not(target_os = "windows"))]
fn create_windows_import_library(_out_dir: &Path) {}

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
