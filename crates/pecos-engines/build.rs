use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Build script for the pecos-engines crate
///
/// This script automatically builds the QIR runtime library that is used by the QIR compiler.
/// The library is built only when necessary (when source files have changed).
fn main() {
    // Use a more surgical approach to rebuild triggers
    // Only track the specific files and environment variables we care about

    // Only track build.rs itself - this is the most critical
    println!("cargo:rerun-if-changed=build.rs");

    // Track QIR source files
    for file in QIR_SOURCE_FILES {
        println!("cargo:rerun-if-changed={file}");
    }

    // Track only pecos-core/Cargo.toml for major version changes
    println!("cargo:rerun-if-changed=../pecos-core/Cargo.toml");

    // Only track environment variables specifically for LLVM paths
    // Intentionally NOT tracking PATH as it changes too often
    println!("cargo:rerun-if-env-changed=PECOS_LLVM_PATH");
    println!("cargo:rerun-if-env-changed=LLVM_HOME");

    // Check for LLVM dependencies first
    match check_llvm_dependencies() {
        Ok(version) => {
            println!("Found LLVM version {version}");
            // Build the QIR runtime library
            if let Err(e) = build_qir_runtime() {
                eprintln!("Warning: Failed to build QIR runtime library: {e}");
                eprintln!("QIR compilation will be slower as it will build the runtime on-demand.");
            }
        }
        Err(e) => {
            println!("cargo:warning=LLVM dependency check failed: {e}");
            eprintln!("Warning: {e}");
            eprintln!(
                "QIR functionality will be unavailable. Install LLVM version 14 (specifically 'llc' tool) to enable QIR support."
            );
            eprintln!("QIR tests will be skipped, but other tests will continue to run.");
        }
    }
}

/// Check for required LLVM dependencies
/// Returns the LLVM version if found and meets requirements
fn check_llvm_dependencies() -> Result<String, String> {
    // Use a simple caching mechanism to avoid checking repeatedly
    const CACHE_FILE: &str = "target/qir_runtime_build/llvm_version_cache.txt";

    // First, try to read from the cache
    if let Ok(cached_version) = fs::read_to_string(CACHE_FILE) {
        let cached_version = cached_version.trim();

        // Only return the cached version if it's valid (version 14.x)
        if cached_version.starts_with("14.") || cached_version == "14" {
            println!("Using cached LLVM version: {cached_version}");
            return Ok(cached_version.to_string());
        }
    }

    // If no cache or invalid version, check normally
    let tool_path = find_tool_in_path()?;
    let version = check_llvm_version(&tool_path)?;

    // Cache the result for next time
    if let Some(parent) = std::path::Path::new(CACHE_FILE).parent() {
        let _ = fs::create_dir_all(parent);
    }
    let _ = fs::write(CACHE_FILE, &version);

    Ok(version)
}

/// Find LLVM tool in the system path
fn find_tool_in_path() -> Result<PathBuf, String> {
    // Set the tool name based on platform
    #[cfg(not(target_os = "windows"))]
    let tool_name = "llc";
    #[cfg(target_os = "windows")]
    let tool_name = "clang";

    // Create executable name with extension if needed
    let executable_name = if cfg!(windows) {
        format!("{tool_name}.exe")
    } else {
        tool_name.to_string()
    };

    // Define standard search locations
    let env_vars = ["PECOS_LLVM_PATH", "LLVM_HOME"];

    // Try environment variables first
    for env_var in &env_vars {
        if let Ok(llvm_path) = env::var(env_var) {
            let tool_path = PathBuf::from(llvm_path).join("bin").join(&executable_name);
            if tool_path.exists() {
                return Ok(tool_path);
            }
        }
    }

    // Try to find in PATH directly
    if let Ok(path_var) = env::var("PATH") {
        let separator = if cfg!(windows) { ';' } else { ':' };
        for path_entry in path_var.split(separator) {
            let full_path = Path::new(path_entry).join(&executable_name);
            if full_path.exists() {
                return Ok(full_path);
            }
        }
    }

    // If we get here, the tool wasn't found
    Err(format!(
        "Required LLVM tool '{tool_name}' not found. Please install LLVM version 14 to enable QIR functionality."
    ))
}

/// Check LLVM version and verify it meets specific version requirements (LLVM 14.x only)
fn check_llvm_version(tool_path: &Path) -> Result<String, String> {
    // Get the version output
    let output = Command::new(tool_path)
        .arg("--version")
        .output()
        .map_err(|e| format!("Failed to check LLVM version: {e}"))?;

    if !output.status.success() {
        return Err("Failed to get LLVM version. Tool returned non-zero status.".to_string());
    }

    let version_output = String::from_utf8_lossy(&output.stdout);
    let first_line = version_output
        .lines()
        .next()
        .ok_or_else(|| "Empty LLVM version output".to_string())?;

    // Extract version number - first look for X.Y.Z format
    let version = first_line
        .split_whitespace()
        .find(|&part| part.contains('.') && part.chars().any(|c| c.is_ascii_digit()))
        // If no X.Y.Z format found, look for just numbers
        .or_else(|| {
            first_line
                .split_whitespace()
                .find(|&part| part.chars().all(|c| c.is_ascii_digit()))
        })
        .ok_or_else(|| format!("Could not parse version from: {first_line}"))?;

    // Extract major version and check requirements
    let major_version = version
        .split('.')
        .next()
        .ok_or_else(|| format!("Malformed LLVM version: {version}"))?;

    let major = major_version
        .parse::<u32>()
        .map_err(|_| format!("Failed to parse LLVM major version: {major_version}"))?;

    if major != 14 {
        return Err(format!(
            "LLVM version {version} is not compatible. PECOS requires LLVM version 14.x specifically for QIR functionality."
        ));
    }

    Ok(version.to_string())
}

// Source files that trigger rebuilds when changed
const QIR_SOURCE_FILES: [&str; 5] = [
    "src/engines/qir/runtime.rs",
    "src/engines/qir/common.rs",
    "src/engines/qir/state.rs",
    "src/core/result_id.rs",
    "src/byte_message/quantum_cmd.rs",
];

// File paths to copy or modify
struct FilePaths {
    common: (PathBuf, PathBuf),
    state: (PathBuf, PathBuf),
    result_id: (PathBuf, PathBuf),
    quantum_cmd: (PathBuf, PathBuf),
    runtime: (PathBuf, PathBuf),
    byte_message: PathBuf,
    cargo_toml: PathBuf,
    lib_rs: PathBuf,
}

fn build_qir_runtime() -> Result<(), String> {
    println!("Building QIR runtime library...");

    // Get the workspace root directory
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let workspace_dir = manifest_dir.parent().unwrap().parent().unwrap();

    // Determine library name based on platform
    let lib_filename = if cfg!(windows) {
        "qir_runtime.lib"
    } else {
        "libqir_runtime.a"
    };

    // Check if the library already exists and is up-to-date
    let debug_lib_path = workspace_dir.join(format!("target/debug/{lib_filename}"));
    let release_lib_path = workspace_dir.join(format!("target/release/{lib_filename}"));

    // Check if we need to rebuild
    if !needs_rebuild(&manifest_dir, &debug_lib_path)
        && !needs_rebuild(&manifest_dir, &release_lib_path)
    {
        println!("QIR runtime library is up-to-date, skipping build.");
        return Ok(());
    }

    // Create a temporary directory for building
    let build_dir = workspace_dir.join("target/qir_runtime_build");
    fs::create_dir_all(&build_dir).map_err(|e| format!("Failed to create build directory: {e}"))?;
    fs::create_dir_all(build_dir.join("src/byte_message"))
        .map_err(|e| format!("Failed to create source directories: {e}"))?;

    // Set up file paths
    let paths = setup_file_paths(&manifest_dir, &build_dir);

    // Setup temporary project
    setup_temp_project(workspace_dir, &paths)?;

    // Build the library
    println!("Running cargo build in {}...", build_dir.display());
    if !run_cargo_build(&build_dir)? {
        return Err("Cargo build failed".to_string());
    }

    // Check if library was built
    let built_lib_path = build_dir.join(format!("target/release/{lib_filename}"));
    if !built_lib_path.exists() {
        return Err(format!(
            "Library not found at: {}",
            built_lib_path.display()
        ));
    }

    // Copy the built library to the target directories
    for target_dir in ["debug", "release"] {
        let target_path = workspace_dir.join(format!("target/{target_dir}/{lib_filename}"));
        fs::create_dir_all(target_path.parent().unwrap())
            .map_err(|e| format!("Failed to create target directory: {e}"))?;
        fs::copy(&built_lib_path, &target_path)
            .map_err(|e| format!("Failed to copy library to {}: {e}", target_path.display()))?;
    }

    println!("QIR runtime library built successfully!");
    Ok(())
}

fn setup_file_paths(manifest_dir: &Path, build_dir: &Path) -> FilePaths {
    FilePaths {
        common: (
            manifest_dir.join("src/engines/qir/common.rs"),
            build_dir.join("src/common.rs"),
        ),
        state: (
            manifest_dir.join("src/engines/qir/state.rs"),
            build_dir.join("src/state.rs"),
        ),
        result_id: (
            manifest_dir.join("src/core/result_id.rs"),
            build_dir.join("src/result_id.rs"),
        ),
        quantum_cmd: (
            manifest_dir.join("src/byte_message/quantum_cmd.rs"),
            build_dir.join("src/byte_message/quantum_cmd.rs"),
        ),
        runtime: (
            manifest_dir.join("src/engines/qir/runtime.rs"),
            build_dir.join("src/lib.rs"),
        ),
        byte_message: build_dir.join("src/byte_message.rs"),
        cargo_toml: build_dir.join("Cargo.toml"),
        lib_rs: build_dir.join("src/lib.rs"),
    }
}

fn setup_temp_project(workspace_dir: &Path, paths: &FilePaths) -> Result<(), String> {
    // Create Cargo.toml
    let cargo_toml_content = format!(
        r#"[package]
name = "qir_runtime"
version = "0.1.0"
edition = "2021"

[lib]
name = "qir_runtime"
crate-type = ["staticlib"]

[dependencies]
once_cell = "1.8.0"
pecos-core = {{ version = "=0.1.1", path = "{}" }}

[workspace]
resolver = "2"
members = ["."]
"#,
        workspace_dir.join("crates/pecos-core").display()
    );
    fs::write(&paths.cargo_toml, cargo_toml_content)
        .map_err(|e| format!("Failed to write Cargo.toml: {e}"))?;

    // Copy common.rs
    fs::copy(&paths.common.0, &paths.common.1)
        .map_err(|e| format!("Failed to copy common.rs: {e}"))?;

    // Copy and modify state.rs
    let state_content =
        fs::read_to_string(&paths.state.0).map_err(|e| format!("Failed to read state.rs: {e}"))?;
    let modified_state =
        state_content.replace("use crate::engines::qir::common::", "use crate::common::");
    fs::write(&paths.state.1, modified_state)
        .map_err(|e| format!("Failed to write state.rs: {e}"))?;

    // Copy result_id.rs
    fs::copy(&paths.result_id.0, &paths.result_id.1)
        .map_err(|e| format!("Failed to copy result_id.rs: {e}"))?;

    // Copy and modify quantum_cmd.rs
    let quantum_cmd_content = fs::read_to_string(&paths.quantum_cmd.0)
        .map_err(|e| format!("Failed to read quantum_cmd.rs: {e}"))?;
    let modified_quantum_cmd = quantum_cmd_content.replace(
        "use crate::core::result_id::ResultId;",
        "use crate::result_id::ResultId;",
    );
    fs::write(&paths.quantum_cmd.1, modified_quantum_cmd)
        .map_err(|e| format!("Failed to write quantum_cmd.rs: {e}"))?;

    // Create byte_message.rs
    fs::write(
        &paths.byte_message,
        "pub mod quantum_cmd;\npub use quantum_cmd::QuantumCmd;\n",
    )
    .map_err(|e| format!("Failed to write byte_message.rs: {e}"))?;

    // Read and modify runtime.rs
    let runtime_content = fs::read_to_string(&paths.runtime.0)
        .map_err(|e| format!("Failed to read runtime.rs: {e}"))?;

    // More careful replacements to ensure imports are correct
    let modified_runtime = runtime_content
        .replace("use crate::engines::qir::common::", "use crate::common::")
        .replace("use crate::engines::qir::state::", "use crate::state::")
        .replace(
            "use crate::byte_message::quantum_cmd::",
            "use crate::byte_message::",
        )
        .replace("use crate::core::result_id::", "use crate::result_id::");

    // Add module declarations and write lib.rs
    let module_declarations =
        "pub mod byte_message;\npub mod result_id;\npub mod common;\npub mod state;\n\n";

    // Ensure MEASUREMENT_RESULTS is property initialized and used
    let fixed_runtime = format!("{module_declarations}{modified_runtime}");
    fs::write(&paths.lib_rs, fixed_runtime).map_err(|e| format!("Failed to write lib.rs: {e}"))?;

    // On Windows, create a DEF file for exports
    if cfg!(windows) {
        let def_file_path = paths.cargo_toml.with_file_name("qir_runtime.def");
        let def_file_content = r"EXPORTS
    qir_runtime_reset
    qir_runtime_get_binary_commands
    qir_runtime_free_binary_commands
    __quantum__qis__rz__body
    __quantum__qis__r1xy__body
    __quantum__qis__h__body
    __quantum__qis__x__body
    __quantum__qis__y__body
    __quantum__qis__z__body
    __quantum__qis__cx__body
    __quantum__qis__cz__body
    __quantum__qis__szz__body
    __quantum__qis__rzz__body
    __quantum__qis__m__body
    __quantum__qis__reset__body
    __quantum__rt__qubit_allocate
    __quantum__rt__result_allocate
    __quantum__rt__qubit_release
    __quantum__rt__result_release
    __quantum__rt__message
    __quantum__rt__record
    __quantum__rt__result_record_output
";
        fs::write(&def_file_path, def_file_content)
            .map_err(|e| format!("Failed to write DEF file: {e}"))?;
    }

    Ok(())
}

fn run_cargo_build(build_dir: &Path) -> Result<bool, String> {
    let output = Command::new("cargo")
        .arg("build")
        .arg("--release")
        .current_dir(build_dir)
        .output()
        .map_err(|e| format!("Failed to execute cargo: {e}"))?;

    if !output.status.success() {
        if cfg!(windows) {
            // Only show detailed output on Windows where CI issues are more common
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            println!("Cargo build failed: {}", output.status);
            println!("Stdout:\n{stdout}");
            println!("Stderr:\n{stderr}");
        }
        return Ok(false);
    }

    Ok(true)
}

fn needs_rebuild(manifest_dir: &Path, lib_path: &Path) -> bool {
    // If the library doesn't exist, we need to build it
    if !lib_path.exists() {
        println!(
            "QIR runtime library not found at {}, rebuilding",
            lib_path.display()
        );
        return true;
    }

    // Check library size - if it's suspiciously small, rebuild
    if let Ok(metadata) = fs::metadata(lib_path) {
        if metadata.len() < 1000 {
            // Arbitrary small size check
            println!(
                "QIR runtime library at {} appears to be too small ({}b), rebuilding",
                lib_path.display(),
                metadata.len()
            );
            return true;
        }
    } else {
        println!("Could not read metadata for QIR runtime library, rebuilding");
        return true;
    }

    // Get the modification time of the library
    let Ok(lib_modified) = fs::metadata(lib_path).and_then(|m| m.modified()) else {
        println!("Could not determine modification time of QIR runtime library, rebuilding");
        return true;
    };

    // Only check if build.rs has changed - the most critical file
    if let Ok(metadata) = fs::metadata(manifest_dir.join("build.rs")) {
        if let Ok(modified) = metadata.modified() {
            if modified > lib_modified {
                println!("build.rs is newer than library, rebuilding");
                return true;
            }
        }
    }

    // Check pecos-core version but only Cargo.toml
    let core_cargo_path = manifest_dir.parent().unwrap().join("pecos-core/Cargo.toml");
    if let Ok(metadata) = fs::metadata(&core_cargo_path) {
        if let Ok(modified) = metadata.modified() {
            if modified > lib_modified {
                println!("pecos-core Cargo.toml is newer than library, rebuilding");
                return true;
            }
        }
    }

    // Check if any source files are newer than the library
    for file in QIR_SOURCE_FILES {
        let file_path = manifest_dir.join(file);
        if let Ok(metadata) = fs::metadata(&file_path) {
            if let Ok(modified) = metadata.modified() {
                if modified > lib_modified {
                    println!("Source file {file_path:?} is newer than library, rebuilding");
                    return true;
                }
            }
        } else {
            // If a source file is missing, that's a problem and we should rebuild
            println!("Source file {file_path:?} not found, rebuilding");
            return true;
        }
    }

    false
}
