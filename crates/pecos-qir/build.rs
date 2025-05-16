use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

//------------------------------------------------------------------------------
// Configuration Constants
//------------------------------------------------------------------------------

// Source files that trigger rebuilds when changed
const QIR_SOURCE_FILES: [&str; 5] = [
    "src/runtime.rs",
    "src/common.rs",
    "src/state.rs",
    "../pecos-engines/src/core/result_id.rs",
    "../pecos-engines/src/byte_message/quantum_cmd.rs",
];

// LLVM version required by PECOS
const REQUIRED_LLVM_VERSION: u32 = 14;

// LLVM version cache location
const LLVM_CACHE_FILE: &str = "target/qir_runtime_build/llvm_version_cache.txt";

// Environment variables to check for LLVM path
const LLVM_ENV_VARS: [&str; 2] = ["PECOS_LLVM_PATH", "LLVM_HOME"];

/// Build script for the pecos-qir crate
///
/// This script automatically builds the QIR runtime library that is used by the QIR compiler.
/// The library is built only when necessary (when source files have changed or the build
/// environment has been modified).
///
/// # Key behaviors:
/// - Builds the QIR runtime library as a static library (.a or .lib)
/// - Checks for LLVM dependencies (specifically version 14)
/// - Optimizes build performance by selectively tracking files that trigger rebuilds
/// - Provides clear error messages when dependencies are missing
fn main() {
    // Configure rebuild triggers - only track specific files and environment variables
    configure_rebuild_triggers();

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
                "QIR functionality will be unavailable. Install LLVM version {REQUIRED_LLVM_VERSION} (specifically 'llc' tool) to enable QIR support."
            );
            eprintln!("QIR tests will be skipped, but other tests will continue to run.");
        }
    }
}

/// Configure which files and environment variables should trigger rebuilds
fn configure_rebuild_triggers() {
    // Track build.rs itself - this is the most critical
    println!("cargo:rerun-if-changed=build.rs");

    // Track QIR source files
    for file in QIR_SOURCE_FILES {
        println!("cargo:rerun-if-changed={file}");
    }

    // Track only pecos-core/Cargo.toml for major version changes
    println!("cargo:rerun-if-changed=../pecos-core/Cargo.toml");

    // Track environment variables specifically for LLVM paths
    // Intentionally NOT tracking PATH as it changes too often
    for env_var in LLVM_ENV_VARS {
        println!("cargo:rerun-if-env-changed={env_var}");
    }
}

/// Check for required LLVM dependencies (must be version 14.x)
///
/// Tries to use a cached version first, then searches for the tool and verifies its version.
///
/// # Returns
/// - `Ok(String)` - The LLVM version string if found and compatible
/// - `Err(String)` - A descriptive error message if the dependency check fails
fn check_llvm_dependencies() -> Result<String, String> {
    // Try to get cached version first
    if let Ok(cached_version) = fs::read_to_string(LLVM_CACHE_FILE) {
        let cached_version = cached_version.trim();
        if cached_version.starts_with(&format!("{REQUIRED_LLVM_VERSION}."))
            || cached_version == REQUIRED_LLVM_VERSION.to_string()
        {
            println!("Using cached LLVM version: {cached_version}");
            return Ok(cached_version.to_string());
        }
    }

    // Find the tool and check its version
    let tool_path = find_tool_in_path()?;
    let version = check_llvm_version(&tool_path)?;

    // Cache the result for next time
    if let Some(parent) = Path::new(LLVM_CACHE_FILE).parent() {
        let _ = fs::create_dir_all(parent);
    }
    let _ = fs::write(LLVM_CACHE_FILE, &version);

    Ok(version)
}

/// Find LLVM tool (llc on Unix, clang on Windows) in the system
///
/// Searches in environment variables and system PATH
///
/// # Returns
/// - `Ok(PathBuf)` - Path to the found tool
/// - `Err(String)` - Error message if tool not found
fn find_tool_in_path() -> Result<PathBuf, String> {
    // Determine the tool name based on platform
    #[cfg(not(target_os = "windows"))]
    let tool_name = "llc";
    #[cfg(target_os = "windows")]
    let tool_name = "clang";

    // Add .exe extension on Windows
    let executable_name = if cfg!(windows) {
        format!("{tool_name}.exe")
    } else {
        tool_name.to_string()
    };

    // Try environment variables first
    for env_var in LLVM_ENV_VARS {
        if let Ok(llvm_path) = env::var(env_var) {
            let tool_path = PathBuf::from(llvm_path).join("bin").join(&executable_name);
            if tool_path.exists() {
                return Ok(tool_path);
            }
        }
    }

    // Try system PATH
    if let Ok(path_var) = env::var("PATH") {
        let separator = if cfg!(windows) { ';' } else { ':' };
        for path_entry in path_var.split(separator) {
            let full_path = Path::new(path_entry).join(&executable_name);
            if full_path.exists() {
                return Ok(full_path);
            }
        }
    }

    Err(format!(
        "Required LLVM tool '{tool_name}' not found. Please install LLVM version {REQUIRED_LLVM_VERSION}."
    ))
}

/// Check LLVM version and verify it's compatible with PECOS requirements
///
/// # Arguments
/// * `tool_path` - Path to the LLVM tool executable
///
/// # Returns
/// - `Ok(String)` - The version string if compatible
/// - `Err(String)` - Error message if version check fails or incompatible
fn check_llvm_version(tool_path: &Path) -> Result<String, String> {
    // Run the version command
    let output = Command::new(tool_path)
        .arg("--version")
        .output()
        .map_err(|e| format!("Failed to check LLVM version: {e}"))?;

    if !output.status.success() {
        return Err("Failed to get LLVM version. Tool returned non-zero status.".to_string());
    }

    // Parse the output to find version number
    let version_text = String::from_utf8_lossy(&output.stdout);
    let first_line = version_text
        .lines()
        .next()
        .ok_or_else(|| "Empty LLVM version output".to_string())?;

    // Extract version string using two different patterns
    let version = first_line
        .split_whitespace()
        // Look for X.Y.Z format with digits
        .find(|&part| part.contains('.') && part.chars().any(|c| c.is_ascii_digit()))
        // Or just a plain number
        .or_else(|| {
            first_line
                .split_whitespace()
                .find(|&part| part.chars().all(|c| c.is_ascii_digit()))
        })
        .ok_or_else(|| format!("Could not parse version from: {first_line}"))?;

    // Extract major version and verify compatibility
    let major = version
        .split('.')
        .next()
        .and_then(|v| v.parse::<u32>().ok())
        .ok_or_else(|| format!("Malformed LLVM version: {version}"))?;

    if major != REQUIRED_LLVM_VERSION {
        return Err(format!(
            "LLVM version {version} not compatible. PECOS requires version {REQUIRED_LLVM_VERSION}.x."
        ));
    }

    Ok(version.to_string())
}

/// File paths used during the QIR runtime build process
///
/// Contains source and destination paths for all files that need to be
/// copied or modified during the QIR runtime library build process
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

/// Build the QIR runtime library
///
/// This function:
/// 1. Creates a temporary build directory
/// 2. Copies and modifies necessary source files
/// 3. Sets up a minimal Cargo project
/// 4. Builds the static library
/// 5. Copies the result to the target directories
///
/// The build is skipped if the library already exists and is up-to-date.
///
/// # Returns
/// - `Ok(())` - Build successful or skipped (up-to-date)
/// - `Err(String)` - Error message if build fails
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

    // Check for potentially corrupted libraries
    let debug_corrupted = debug_lib_path.exists()
        && fs::metadata(&debug_lib_path).map(|m| m.len()).unwrap_or(0) < 1000;
    let release_corrupted = release_lib_path.exists()
        && fs::metadata(&release_lib_path)
            .map(|m| m.len())
            .unwrap_or(0)
            < 1000;

    if debug_corrupted || release_corrupted {
        println!("Detected potentially corrupted QIR runtime library, forcing rebuild");
    }
    // Skip build if libraries exist and are up-to-date
    else if !needs_rebuild(&manifest_dir, &debug_lib_path)
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

    // Set up file paths and create temporary project
    let paths = setup_file_paths(&manifest_dir, &build_dir);
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

        // Verify that the library was copied correctly
        if !target_path.exists() || fs::metadata(&target_path).map(|m| m.len()).unwrap_or(0) < 1000
        {
            return Err(format!(
                "Library copy verification failed at {}",
                target_path.display()
            ));
        }
    }

    println!("QIR runtime library built successfully!");
    Ok(())
}

/// Set up file paths for the QIR runtime build
///
/// Creates a `FilePaths` struct with source and destination paths for all files
/// that need to be copied or modified during the build process.
///
/// # Arguments
/// * `manifest_dir` - Path to the crate's manifest directory
/// * `build_dir` - Path to the temporary build directory
///
/// # Returns
/// A `FilePaths` struct with all required paths
fn setup_file_paths(manifest_dir: &Path, build_dir: &Path) -> FilePaths {
    // Define paths for pecos-engines source files
    let pecos_engines_dir = manifest_dir.parent().unwrap().join("pecos-engines");

    FilePaths {
        common: (
            manifest_dir.join("src/common.rs"),
            build_dir.join("src/common.rs"),
        ),
        state: (
            manifest_dir.join("src/state.rs"),
            build_dir.join("src/state.rs"),
        ),
        result_id: (
            pecos_engines_dir.join("src/core/result_id.rs"),
            build_dir.join("src/result_id.rs"),
        ),
        quantum_cmd: (
            pecos_engines_dir.join("src/byte_message/quantum_cmd.rs"),
            build_dir.join("src/byte_message/quantum_cmd.rs"),
        ),
        runtime: (
            manifest_dir.join("src/runtime.rs"),
            build_dir.join("src/lib.rs"),
        ),
        byte_message: build_dir.join("src/byte_message.rs"),
        cargo_toml: build_dir.join("Cargo.toml"),
        lib_rs: build_dir.join("src/lib.rs"),
    }
}

/// Set up a temporary Cargo project for building the QIR runtime
///
/// Creates a standalone Cargo project with all necessary source files.
///
/// # Arguments
/// * `workspace_dir` - Path to the workspace root directory
/// * `paths` - `FilePaths` struct with all source and destination paths
///
/// # Returns
/// - `Ok(())` - Setup successful
/// - `Err(String)` - Error message if setup fails
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
pecos-core = {{ path = "{}" }}

[workspace]
resolver = "2"
members = ["."]
"#,
        workspace_dir.join("crates/pecos-core").display()
    );
    fs::write(&paths.cargo_toml, cargo_toml_content)
        .map_err(|e| format!("Failed to write Cargo.toml: {e}"))?;

    // Perform file operations one by one

    // 1. Copy common.rs
    fs::copy(&paths.common.0, &paths.common.1)
        .map_err(|e| format!("Failed to copy common.rs: {e}"))?;

    // 2. Copy result_id.rs
    fs::copy(&paths.result_id.0, &paths.result_id.1)
        .map_err(|e| format!("Failed to copy result_id.rs: {e}"))?;

    // 3. Copy state.rs (no need to modify imports)
    let state_content =
        fs::read_to_string(&paths.state.0).map_err(|e| format!("Failed to read state.rs: {e}"))?;
    fs::write(&paths.state.1, state_content)
        .map_err(|e| format!("Failed to write state.rs: {e}"))?;

    // 4. Modify quantum_cmd.rs: update imports
    let quantum_cmd_content = fs::read_to_string(&paths.quantum_cmd.0)
        .map_err(|e| format!("Failed to read quantum_cmd.rs: {e}"))?;
    let modified_quantum_cmd = quantum_cmd_content.replace(
        "use crate::core::result_id::ResultId;",
        "use crate::result_id::ResultId;",
    );
    fs::write(&paths.quantum_cmd.1, modified_quantum_cmd)
        .map_err(|e| format!("Failed to write quantum_cmd.rs: {e}"))?;

    // 5. Create byte_message.rs module file
    fs::write(
        &paths.byte_message,
        "pub mod quantum_cmd;\npub use quantum_cmd::QuantumCmd;\n",
    )
    .map_err(|e| format!("Failed to write byte_message.rs: {e}"))?;

    // 6. Create lib.rs with modified runtime content
    let runtime_content = fs::read_to_string(&paths.runtime.0)
        .map_err(|e| format!("Failed to read runtime.rs: {e}"))?;

    // Update imports
    let modified_runtime = runtime_content
        .replace(
            "use pecos_engines::byte_message::",
            "use crate::byte_message::",
        )
        .replace(
            "use pecos_engines::core::result_id::",
            "use crate::result_id::",
        );

    // Add module declarations
    let module_declarations =
        "pub mod byte_message;\npub mod result_id;\npub mod common;\npub mod state;\n\n";

    fs::write(
        &paths.lib_rs,
        format!("{module_declarations}{modified_runtime}"),
    )
    .map_err(|e| format!("Failed to write lib.rs: {e}"))?;

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

/// Run 'cargo build --release' in the temporary project directory
///
/// # Arguments
/// * `build_dir` - Path to the temporary build directory
///
/// # Returns
/// - `Ok(true)` - Build successful
/// - `Ok(false)` - Build failed but not due to a system error
/// - `Err(String)` - Error message if command execution fails
fn run_cargo_build(build_dir: &Path) -> Result<bool, String> {
    let output = Command::new("cargo")
        .arg("build")
        .arg("--release")
        .current_dir(build_dir)
        .output()
        .map_err(|e| format!("Failed to execute cargo: {e}"))?;

    if !output.status.success() {
        // On Windows, show detailed output where CI issues are more common
        if cfg!(windows) {
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

/// Check if the QIR runtime library needs to be rebuilt
///
/// # Returns
/// * `true` if any of these conditions are met:
///   - Library doesn't exist or is too small
///   - build.rs is newer than the library
///   - pecos-core/Cargo.toml is newer than the library
///   - Any source file is newer than the library
/// * `false` if library is up-to-date
fn needs_rebuild(manifest_dir: &Path, lib_path: &Path) -> bool {
    // Check if library exists and has reasonable size
    if !lib_path.exists() {
        println!(
            "QIR runtime library not found at {}, rebuilding",
            lib_path.display()
        );
        return true;
    }

    // Get library metadata
    let Ok(lib_metadata) = fs::metadata(lib_path) else {
        println!("Could not read metadata for QIR runtime library, rebuilding");
        return true;
    };

    // Check if library is suspiciously small
    if lib_metadata.len() < 1000 {
        println!(
            "QIR runtime library too small ({}b), rebuilding",
            lib_metadata.len()
        );
        return true;
    }

    // Get library modification time
    let Ok(lib_modified) = lib_metadata.modified() else {
        println!("Could not determine library modification time, rebuilding");
        return true;
    };

    // Check if any critical file is newer than the library
    let check_file = |path: &Path, desc: &str| -> bool {
        if !path.exists() {
            println!("{desc} not found, rebuilding");
            return true;
        }

        match fs::metadata(path).and_then(|meta| meta.modified()) {
            Ok(time) if time > lib_modified => {
                println!("{desc} is newer than library, rebuilding");
                true
            }
            Err(_) => {
                println!("Cannot check time of {desc}, rebuilding");
                true
            }
            _ => false,
        }
    };

    // Check build script and core dependency
    if check_file(&manifest_dir.join("build.rs"), "build.rs")
        || check_file(
            &manifest_dir.parent().unwrap().join("pecos-core/Cargo.toml"),
            "pecos-core Cargo.toml",
        )
    {
        return true;
    }

    // Check source files
    for file in QIR_SOURCE_FILES {
        if check_file(&manifest_dir.join(file), &format!("Source file {file}")) {
            return true;
        }
    }

    false // Library is up-to-date
}
