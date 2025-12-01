//! Helios interface executor
//!
//! This module implements the `QisInterface` trait for Selene's Helios compiler.

use libloading::{Library, Symbol};
use log::{debug, error, info, warn};
use pecos_qis_core::qis_interface::{InterfaceError, ProgramFormat, QisInterface};
use pecos_qis_ffi_types::OperationCollector;
use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use tempfile::NamedTempFile;

// FFI function type aliases for dlopen symbol lookup
type ResetInterfaceFn = unsafe extern "C" fn();
type GetOperationsFn = unsafe extern "C" fn() -> *mut OperationCollector;
type CallQmainFn = unsafe extern "C" fn(extern "C" fn(u64) -> u64) -> u64;

/// Derive the project target directory from the compile-time embedded Helios path.
///
/// The compile-time path looks like:
/// `/path/to/project/target/release/build/pecos-qis-selene-HASH/out/libhelios_selene_interface.a`
///
/// We want to extract `/path/to/project/target` so we can search for other build hashes.
fn get_helios_target_dir() -> Option<PathBuf> {
    let compile_time_path = PathBuf::from(env!("HELIOS_LIB_PATH"));
    // Go up from: lib -> out -> pecos-qis-selene-HASH -> build -> release/debug -> target
    compile_time_path
        .parent() // out/
        .and_then(|p| p.parent()) // pecos-qis-selene-HASH/
        .and_then(|p| p.parent()) // build/
        .and_then(|p| p.parent()) // release or debug
        .and_then(|p| p.parent()) // target/
        .map(std::path::Path::to_path_buf)
}

/// Search for the Helios library in a target directory
fn search_helios_in_target(target_dir: &Path, lib_name: &str) -> Option<PathBuf> {
    for profile in ["release", "debug"] {
        let build_dir = target_dir.join(profile).join("build");
        if build_dir.exists()
            && let Ok(entries) = std::fs::read_dir(&build_dir)
        {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                if name_str.starts_with("pecos-qis-selene-") {
                    let lib_path = entry.path().join("out").join(lib_name);
                    if lib_path.exists() {
                        debug!("Found Helios library at: {}", lib_path.display());
                        return Some(lib_path);
                    }
                }
            }
        }
    }
    None
}

/// Find the Helios interface library with the following priority:
/// 1. Runtime `HELIOS_LIB_PATH` environment variable (explicit override)
/// 2. Embedded path from build time (compile-time `HELIOS_LIB_PATH`)
/// 3. Search target directory derived from compile-time path (handles hash changes)
/// 4. Search target directory relative to current working directory
/// 5. Search relative to the executable
///
/// Returns the path to `libhelios_selene_interface.a` or an error with helpful suggestions.
fn find_helios_lib() -> Result<PathBuf, InterfaceError> {
    const LIB_NAME: &str = "libhelios_selene_interface.a";

    // 1. Check runtime environment variable (explicit override)
    if let Ok(path_str) = std::env::var("HELIOS_LIB_PATH") {
        let path = PathBuf::from(&path_str);
        if path.exists() {
            debug!(
                "Using Helios library from HELIOS_LIB_PATH env var: {}",
                path.display()
            );
            return Ok(path);
        }
        warn!(
            "HELIOS_LIB_PATH is set to '{path_str}' but file does not exist, searching other locations..."
        );
    }

    // 2. Check compile-time embedded path
    let compile_time_path = PathBuf::from(env!("HELIOS_LIB_PATH"));
    if compile_time_path.exists() {
        debug!(
            "Using Helios library from compile-time path: {}",
            compile_time_path.display()
        );
        return Ok(compile_time_path);
    }

    // 3. Search target directory derived from compile-time path
    // This handles cases where the build hash changed but the target dir is the same
    if let Some(target_dir) = get_helios_target_dir()
        && let Some(path) = search_helios_in_target(&target_dir, LIB_NAME)
    {
        return Ok(path);
    }

    // 4. Search target directory relative to current working directory
    let mut candidate_paths = Vec::new();
    if let Ok(cwd) = std::env::current_dir() {
        let target_dir = cwd.join("target");
        if let Some(path) = search_helios_in_target(&target_dir, LIB_NAME) {
            return Ok(path);
        }
    }

    // 5. Search relative to executable
    if let Ok(exe_path) = std::env::current_exe()
        && let Some(exe_dir) = exe_path.parent()
    {
        // Check same directory as executable
        candidate_paths.push(exe_dir.join(LIB_NAME));
        // Check lib subdirectory
        candidate_paths.push(exe_dir.join("lib").join(LIB_NAME));
        // Check parent directory (for bundled installations)
        if let Some(parent) = exe_dir.parent() {
            candidate_paths.push(parent.join("lib").join(LIB_NAME));
        }
    }

    // Try each candidate
    for path in &candidate_paths {
        if path.exists() {
            debug!("Found Helios library at: {}", path.display());
            return Ok(path.clone());
        }
    }

    // Nothing found - provide helpful error message
    let searched_locations = candidate_paths
        .iter()
        .map(|p| format!("  - {}", p.display()))
        .collect::<Vec<_>>()
        .join("\n");

    Err(InterfaceError::LoadError(format!(
        "Could not find Helios interface library ({LIB_NAME}).\n\n\
        The compile-time path no longer exists:\n  {}\n\n\
        This usually happens after a partial rebuild. To fix this:\n\
        1. Run: cargo clean -p pecos-qis-selene\n\
        2. Rebuild: cargo build --release\n\n\
        Or set HELIOS_LIB_PATH environment variable to the library location.\n\n\
        Searched locations:\n{searched_locations}",
        compile_time_path.display()
    )))
}

/// Find an LLVM tool with the following priority:
/// 1. Embedded path from build time (`PECOS_LLVM_BIN_PATH`)
/// 2. Runtime `LLVM_SYS_140_PREFIX` environment variable
/// 3. Fall back to PATH
fn find_llvm_tool(tool_name: &str) -> PathBuf {
    let tool_exe = if cfg!(windows) {
        format!("{tool_name}.exe")
    } else {
        tool_name.to_string()
    };

    option_env!("PECOS_LLVM_BIN_PATH")
        .and_then(|bin_path| {
            let path = PathBuf::from(bin_path).join(&tool_exe);
            if path.exists() {
                debug!(
                    "Using {} from embedded PECOS_LLVM_BIN_PATH: {}",
                    tool_name,
                    path.display()
                );
                Some(path)
            } else {
                None
            }
        })
        .or_else(|| {
            std::env::var("LLVM_SYS_140_PREFIX")
                .ok()
                .and_then(|prefix| {
                    let path = PathBuf::from(prefix).join("bin").join(&tool_exe);
                    if path.exists() {
                        debug!(
                            "Using {} from LLVM_SYS_140_PREFIX: {}",
                            tool_name,
                            path.display()
                        );
                        Some(path)
                    } else {
                        None
                    }
                })
        })
        .unwrap_or_else(|| {
            debug!("Using {tool_name} from PATH");
            PathBuf::from(tool_name)
        })
}

/// Helios interface implementation
///
/// This interface:
/// 1. Links program bitcode with libhelios.a to create an executable
/// 2. Loads the executable in-process using dlopen (libloading)
/// 3. Calls `qmain()` to execute the program
/// 4. Collects operations via thread-local storage in the PECOS shim
pub struct QisHeliosInterface {
    /// Path to the linked executable (if created)
    executable_path: Option<PathBuf>,

    /// The program bytes
    program: Vec<u8>,

    /// The program format
    format: ProgramFormat,

    /// Metadata about the interface
    metadata: BTreeMap<String, String>,

    /// Keep temporary files alive (`TempPath` auto-deletes when dropped)
    temp_files: Vec<tempfile::TempPath>,
}

impl QisHeliosInterface {
    /// Create a new Helios interface
    #[must_use]
    pub fn new() -> Self {
        Self {
            executable_path: None,
            program: Vec::new(),
            format: ProgramFormat::QisBitcode,
            metadata: BTreeMap::new(),
            temp_files: Vec::new(),
        }
    }

    /// Find the `libpecos_qis_ffi` library by searching common locations
    fn find_pecos_qis_lib() -> Result<PathBuf, InterfaceError> {
        // On Windows, Rust cdylibs don't use the "lib" prefix
        // On Unix (Linux/macOS), they do use the "lib" prefix
        let (lib_prefix, lib_ext) = if cfg!(target_os = "windows") {
            ("", "dll")
        } else if cfg!(target_os = "macos") {
            ("lib", "dylib")
        } else {
            ("lib", "so")
        };

        let lib_name = format!("{lib_prefix}pecos_qis_ffi.{lib_ext}");

        debug!(
            "Looking for QIS FFI library: {lib_name} on {}",
            std::env::consts::OS
        );

        let exe_dir = std::env::current_exe()
            .ok()
            .and_then(|exe| exe.parent().map(std::path::Path::to_path_buf))
            .ok_or_else(|| {
                InterfaceError::ExecutionError(
                    "Failed to determine executable directory".to_string(),
                )
            })?;

        debug!("Executable directory: {}", exe_dir.display());

        let mut candidate_paths = vec![
            exe_dir.join(&lib_name),
            exe_dir.join(format!("deps/{lib_name}")),
        ];

        if let Some(parent) = exe_dir.parent() {
            candidate_paths.push(parent.join(&lib_name));
            candidate_paths.push(parent.join(format!("deps/{lib_name}")));
        }

        if let Ok(current_dir) = std::env::current_dir() {
            debug!("Current directory: {}", current_dir.display());
            candidate_paths.push(current_dir.join(format!("target/debug/{lib_name}")));
            candidate_paths.push(current_dir.join(format!("target/debug/deps/{lib_name}")));
            candidate_paths.push(current_dir.join(format!("target/release/{lib_name}")));
            candidate_paths.push(current_dir.join(format!("target/release/deps/{lib_name}")));

            // Search up the directory tree for workspace root (when running from Python)
            let mut search_dir = current_dir.as_path();
            for _ in 0..5 {
                // Search up to 5 levels
                if let Some(parent) = search_dir.parent() {
                    candidate_paths.push(parent.join(format!("target/debug/{lib_name}")));
                    candidate_paths.push(parent.join(format!("target/debug/deps/{lib_name}")));
                    candidate_paths.push(parent.join(format!("target/release/{lib_name}")));
                    candidate_paths.push(parent.join(format!("target/release/deps/{lib_name}")));
                    search_dir = parent;
                } else {
                    break;
                }
            }
        }

        debug!("Searching {} candidate paths...", candidate_paths.len());

        // Check each path and report which ones exist
        let mut found_files = Vec::new();
        for path in &candidate_paths {
            if path.exists() {
                debug!("Found library: {}", path.display());
                found_files.push(path.clone());
            }
        }

        if found_files.is_empty() {
            warn!("No matching files found!");
            warn!("Searched paths:");
            for (i, path) in candidate_paths.iter().enumerate() {
                warn!("  {}: {}", i + 1, path.display());
            }
        }

        candidate_paths
            .iter()
            .find(|p| p.exists())
            .ok_or_else(|| {
                InterfaceError::ExecutionError(format!(
                    "Failed to find {lib_name}. Searched in: {candidate_paths:?}"
                ))
            })
            .cloned()
    }

    /// Collect operations from thread-local storage via the QIS cdylib
    fn collect_operations_from_lib(
        pecos_qis_lib: &Library,
    ) -> Result<OperationCollector, InterfaceError> {
        let get_ops_fn: Symbol<GetOperationsFn> = unsafe {
            pecos_qis_lib
                .get(b"pecos_qis_get_operations\0")
                .map_err(|e| {
                    InterfaceError::ExecutionError(format!(
                        "Failed to find get_operations function: {e}"
                    ))
                })?
        };
        let operations_ptr = unsafe { get_ops_fn() };
        let operations = unsafe { Box::from_raw(operations_ptr) };
        Ok(*operations)
    }

    /// Load a library with `RTLD_GLOBAL` and return both the global and lookup handles
    #[cfg(unix)]
    fn load_library_with_rtld_global(
        path: &std::path::Path,
        error_msg: &str,
    ) -> Result<(libloading::os::unix::Library, Library), InterfaceError> {
        let lib_global = unsafe {
            libloading::os::unix::Library::open(
                Some(path),
                libloading::os::unix::RTLD_LAZY | libloading::os::unix::RTLD_GLOBAL,
            )
            .map_err(|e| InterfaceError::ExecutionError(format!("{error_msg}: {e}")))?
        };

        let lib = unsafe {
            Library::new(path)
                .map_err(|e| InterfaceError::ExecutionError(format!("{error_msg} (lookup): {e}")))?
        };

        Ok((lib_global, lib))
    }

    /// Load a library on Windows (no `RTLD_GLOBAL` equivalent - symbols are searched in load order)
    #[cfg(windows)]
    fn load_library_with_rtld_global(
        path: &std::path::Path,
        error_msg: &str,
    ) -> Result<(Library, Library), InterfaceError> {
        // On Windows, there's no RTLD_GLOBAL flag. Symbols are automatically visible
        // to subsequently loaded libraries through the normal DLL search mechanism.
        // We load the library twice to maintain the same API as Unix.
        let lib_global = unsafe {
            Library::new(path)
                .map_err(|e| InterfaceError::ExecutionError(format!("{error_msg}: {e}")))?
        };

        let lib = unsafe {
            Library::new(path)
                .map_err(|e| InterfaceError::ExecutionError(format!("{error_msg} (lookup): {e}")))?
        };

        Ok((lib_global, lib))
    }

    /// Get the qmain and setjmp wrapper function symbols from the libraries
    fn get_execution_symbols<'a>(
        program_lib: &'a Library,
        shim_lib: &'a Library,
    ) -> Result<
        (
            Symbol<'a, extern "C" fn(u64) -> u64>,
            Symbol<'a, CallQmainFn>,
        ),
        InterfaceError,
    > {
        // Get the qmain or main function symbol
        let qmain_fn: Symbol<extern "C" fn(u64) -> u64> = unsafe {
            program_lib
                .get(b"qmain\0")
                .or_else(|_| program_lib.get(b"main\0"))
                .map_err(|e| {
                    InterfaceError::ExecutionError(format!(
                        "Failed to find qmain or main entry point: {e}"
                    ))
                })?
        };

        // Get the setjmp wrapper function
        let call_with_setjmp: Symbol<CallQmainFn> = unsafe {
            shim_lib
                .get(b"pecos_call_qmain_with_setjmp\0")
                .map_err(|e| {
                    InterfaceError::ExecutionError(format!("Failed to find setjmp wrapper: {e}"))
                })?
        };

        Ok((qmain_fn, call_with_setjmp))
    }

    /// Add platform-specific linker flags to the clang command
    fn add_platform_linker_flags(clang_cmd: &mut Command) {
        if cfg!(target_os = "windows") {
            // Windows-specific flags
            debug!("Adding Windows-specific linker flags...");
            // On Windows, clang uses MSVC's linker (link.exe) or lld-link
            // The -shared flag is enough for basic DLL creation
            // Undefined symbols are allowed by default on Windows - they'll be resolved at load time
        } else {
            // Unix-like platforms (Linux, macOS)
            // -fPIC is not supported on Windows MSVC (and not needed for DLLs)
            clang_cmd.arg("-fPIC");

            // Export dynamic flag differs by platform
            if cfg!(target_os = "macos") {
                // macOS ld flags:
                // - export_dynamic: Make all symbols visible for dlopen
                // - undefined dynamic_lookup: Allow undefined symbols (resolved at runtime via RTLD_GLOBAL)
                debug!("Adding macOS-specific linker flags...");
                clang_cmd.arg("-Wl,-export_dynamic");
                clang_cmd.arg("-Wl,-undefined,dynamic_lookup");

                // On macOS, we need to specify the SDK path for LLVM clang to find system libraries
                // This is required because LLVM's clang (unlike Apple's clang) doesn't automatically
                // know where to find macOS system libraries in the dyld cache
                // Use xcrun to get the SDK path
                debug!("Running xcrun --show-sdk-path...");
                match Command::new("xcrun").args(["--show-sdk-path"]).output() {
                    Ok(output) => {
                        if output.status.success() {
                            if let Ok(sdk_path) = String::from_utf8(output.stdout) {
                                let sdk_path = sdk_path.trim();
                                debug!("SDK path: {sdk_path}");
                                clang_cmd.arg("-isysroot");
                                clang_cmd.arg(sdk_path);
                                // Add library search path so linker can find pthread, etc.
                                clang_cmd.arg(format!("-L{sdk_path}/usr/lib"));
                            } else {
                                warn!("xcrun output was not valid UTF-8");
                            }
                        } else {
                            warn!("xcrun failed with status: {}", output.status);
                            warn!("xcrun stderr: {}", String::from_utf8_lossy(&output.stderr));
                        }
                    }
                    Err(e) => {
                        warn!("Failed to run xcrun: {e}");
                    }
                }

                // macOS provides math functions through libSystem - don't link -lm separately
                // On macOS Big Sur+, libm.dylib doesn't exist as a separate file - it's in the dyld cache
                clang_cmd.arg("-lpthread").arg("-ldl");
            } else {
                // Linux
                clang_cmd.arg("-Wl,--export-dynamic"); // GNU ld flag (double dash)
                // Unix-specific libraries (Linux needs -lm explicitly)
                clang_cmd.arg("-lm").arg("-lpthread").arg("-ldl");
            }
        }
    }

    /// Link the program with Helios interface to create a shared library
    #[allow(clippy::too_many_lines)]
    fn create_shared_library(&mut self) -> Result<PathBuf, InterfaceError> {
        // Find the Helios library using robust search
        let helios_lib_path = find_helios_lib()?;
        let helios_lib_path = helios_lib_path.to_string_lossy().to_string();

        // Create temporary files for the program
        let mut program_file = NamedTempFile::new()
            .map_err(|e| InterfaceError::LoadError(format!("Failed to create temp file: {e}")))?;

        // Get the program file path that we'll pass to clang
        // We need to keep the TempPath alive until after clang finishes
        let program_temp_path = match self.format {
            ProgramFormat::QisBitcode | ProgramFormat::LlvmBitcode => {
                // Write bitcode directly
                program_file.write_all(&self.program).map_err(|e| {
                    InterfaceError::LoadError(format!("Failed to write bitcode: {e}"))
                })?;
                program_file.into_temp_path()
            }
            ProgramFormat::LlvmIrText => {
                debug!("Converting LLVM IR text to bitcode using llvm-as...");
                // Convert text to bitcode using llvm-as
                program_file.write_all(&self.program).map_err(|e| {
                    InterfaceError::LoadError(format!("Failed to write LLVM IR: {e}"))
                })?;
                program_file.flush().map_err(|e| {
                    InterfaceError::LoadError(format!("Failed to flush LLVM IR: {e}"))
                })?;

                let ir_path = program_file.into_temp_path();

                let bitcode_file = NamedTempFile::with_suffix(".bc").map_err(|e| {
                    InterfaceError::LoadError(format!("Failed to create bitcode file: {e}"))
                })?;

                let llvm_as_cmd = find_llvm_tool("llvm-as");

                let output = Command::new(&llvm_as_cmd)
                    .arg("-o")
                    .arg(bitcode_file.path())
                    .arg(&ir_path)
                    .output()
                    .map_err(|e| {
                        InterfaceError::LoadError(format!("Failed to run llvm-as: {e}"))
                    })?;

                if !output.status.success() {
                    return Err(InterfaceError::LoadError(format!(
                        "llvm-as failed: {}",
                        String::from_utf8_lossy(&output.stderr)
                    )));
                }

                // Convert bitcode file to persistent path and keep it alive
                bitcode_file.into_temp_path()
            }
            ProgramFormat::HugrBytes => {
                return Err(InterfaceError::InvalidFormat(
                    "HUGR bytes should be compiled to LLVM first".to_string(),
                ));
            }
        };

        // On Windows, check if we need to add a qmain wrapper for programs that only have main
        #[cfg(target_os = "windows")]
        let program_temp_path = {
            // Use llvm-nm to check which symbols exist in the bitcode
            let llvm_nm_cmd = find_llvm_tool("llvm-nm");

            let nm_output = Command::new(&llvm_nm_cmd)
                .arg(&program_temp_path)
                .output()
                .map_err(|e| InterfaceError::LoadError(format!("Failed to run llvm-nm: {e}")))?;

            if !nm_output.status.success() {
                return Err(InterfaceError::LoadError(format!(
                    "llvm-nm failed: {}",
                    String::from_utf8_lossy(&nm_output.stderr)
                )));
            }

            let nm_output_str = String::from_utf8_lossy(&nm_output.stdout);
            let qmain_found = nm_output_str
                .lines()
                .any(|line| line.contains(" T ") && line.contains("qmain"));
            let main_found = nm_output_str.lines().any(|line| {
                line.contains(" T ") && (line.contains(" main") || line.ends_with(" main"))
            });

            debug!("Symbol check: qmain_found={qmain_found}, main_found={main_found}");

            // If we have qmain or neither, use the original bitcode
            if qmain_found || !main_found {
                program_temp_path
            } else {
                // We have main but not qmain - create a wrapper
                debug!("Creating qmain wrapper for program with only @main");

                // Create wrapper LLVM IR that calls main
                let wrapper_ir = r"
; Wrapper to provide qmain entry point for programs with only @main
declare void @main()

define i64 @qmain(i64 %arg) {
entry:
  call void @main()
  ret i64 0
}
";

                // Write wrapper IR to temp file
                let wrapper_ir_file = NamedTempFile::with_suffix(".ll").map_err(|e| {
                    InterfaceError::LoadError(format!("Failed to create wrapper IR file: {e}"))
                })?;
                std::fs::write(wrapper_ir_file.path(), wrapper_ir).map_err(|e| {
                    InterfaceError::LoadError(format!("Failed to write wrapper IR: {e}"))
                })?;

                // Compile wrapper IR to bitcode
                let wrapper_bc_file = NamedTempFile::with_suffix(".bc").map_err(|e| {
                    InterfaceError::LoadError(format!("Failed to create wrapper BC file: {e}"))
                })?;

                let llvm_as_cmd = find_llvm_tool("llvm-as");

                let as_output = Command::new(&llvm_as_cmd)
                    .arg("-o")
                    .arg(wrapper_bc_file.path())
                    .arg(wrapper_ir_file.path())
                    .output()
                    .map_err(|e| {
                        InterfaceError::LoadError(format!("Failed to run llvm-as on wrapper: {e}"))
                    })?;

                if !as_output.status.success() {
                    return Err(InterfaceError::LoadError(format!(
                        "llvm-as on wrapper failed: {}",
                        String::from_utf8_lossy(&as_output.stderr)
                    )));
                }

                // Link original bitcode with wrapper using llvm-link
                let linked_bc_file = NamedTempFile::with_suffix(".bc").map_err(|e| {
                    InterfaceError::LoadError(format!("Failed to create linked BC file: {e}"))
                })?;

                let llvm_link_cmd = find_llvm_tool("llvm-link");

                let link_output = Command::new(&llvm_link_cmd)
                    .arg("-o")
                    .arg(linked_bc_file.path())
                    .arg(&program_temp_path)
                    .arg(wrapper_bc_file.path())
                    .output()
                    .map_err(|e| {
                        InterfaceError::LoadError(format!("Failed to run llvm-link: {e}"))
                    })?;

                if !link_output.status.success() {
                    return Err(InterfaceError::LoadError(format!(
                        "llvm-link failed: {}",
                        String::from_utf8_lossy(&link_output.stderr)
                    )));
                }

                debug!("Successfully created qmain wrapper");
                linked_bc_file.into_temp_path()
            }
        };

        #[cfg(not(target_os = "windows"))]
        let program_temp_path = program_temp_path;

        // Create shared library path with platform-appropriate extension
        let lib_suffix = if cfg!(target_os = "windows") {
            ".dll"
        } else {
            ".so"
        };
        debug!("Creating shared library temp file with suffix {lib_suffix}...");

        // IMPORTANT: On Windows, we need to get a temp path but NOT create the file yet
        // because MSVC's link.exe wants to create the DLL file itself
        #[cfg(target_os = "windows")]
        let (so_file, so_path_for_clang) = {
            use tempfile::Builder;
            // Create a temp file to reserve the name, then immediately close and delete it
            let temp = Builder::new().suffix(lib_suffix).tempfile().map_err(|e| {
                InterfaceError::LoadError(format!("Failed to create temp file: {e}"))
            })?;

            // Get the path before the file is deleted
            let path = temp.path().to_path_buf();
            debug!(
                "Windows: Reserved temp path (will be deleted): {}",
                path.display()
            );
            debug!("Windows: File exists before drop: {}", path.exists());

            // Drop temp explicitly to delete the file
            drop(temp);

            debug!("Windows: File exists after drop: {}", path.exists());

            // We keep the path but the file is deleted - link.exe will create it
            ((), path)
        };

        #[cfg(not(target_os = "windows"))]
        let (so_file, so_path_for_clang) = {
            let temp = NamedTempFile::with_suffix(lib_suffix).map_err(|e| {
                InterfaceError::LoadError(format!("Failed to create library file: {e}"))
            })?;
            let path = temp.path().to_path_buf();
            (temp, path)
        };

        debug!("Temp library path: {}", so_path_for_clang.display());

        // Link using clang to create a shared library:
        // program.bc + libhelios.a → program.so/.dll
        // The resulting shared library will:
        // - Export qmain symbol
        // - Have undefined selene_* symbols (to be resolved by our shim at runtime)
        debug!(
            "Linking: {} + {} -> {}",
            program_temp_path.display(),
            helios_lib_path,
            so_path_for_clang.display()
        );

        // Build clang command with platform-specific flags
        // Try to find clang: first check LLVM_SYS_140_PREFIX, then fall back to PATH
        let clang_cmd_path = std::env::var("LLVM_SYS_140_PREFIX")
            .ok()
            .and_then(|prefix| {
                let mut path = PathBuf::from(prefix);
                path.push("bin");
                path.push(if cfg!(windows) { "clang.exe" } else { "clang" });
                if path.exists() {
                    debug!("Using clang from LLVM_SYS_140_PREFIX: {}", path.display());
                    Some(path)
                } else {
                    None
                }
            })
            .unwrap_or_else(|| {
                debug!("Using clang from PATH");
                PathBuf::from("clang")
            });

        let mut clang_cmd = Command::new(&clang_cmd_path);

        // On Windows, we need to be more careful with paths and flags
        #[cfg(target_os = "windows")]
        {
            debug!("Windows: Using DLL path: {}", so_path_for_clang.display());

            // On Windows, we need to link against both import libraries (.lib files)
            // to populate the import table for selene_* and __quantum__* symbols

            // Get the selene shim import library path (set by build.rs)
            let shim_lib_path = std::env::var("PECOS_SELENE_SHIM_LIB")
                .ok()
                .or_else(|| option_env!("PECOS_SELENE_SHIM_LIB").map(String::from))
                .ok_or_else(|| {
                    InterfaceError::LoadError(
                        "PECOS selene shim import library not found - build script may have failed to generate it".to_string(),
                    )
                })?;

            // Find the pecos_qis_ffi.dll.lib import library
            let pecos_qis_lib_path = Self::find_pecos_qis_lib()?;
            let qis_ffi_import_lib = pecos_qis_lib_path.with_extension("dll.lib");

            if !qis_ffi_import_lib.exists() {
                return Err(InterfaceError::LoadError(format!(
                    "PECOS QIS FFI import library not found at: {} - Rust should have created this",
                    qis_ffi_import_lib.display()
                )));
            }

            debug!("Windows: Linking against selene shim import library: {shim_lib_path}");
            debug!(
                "Windows: Linking against QIS FFI import library: {}",
                qis_ffi_import_lib.display()
            );

            clang_cmd
                .arg("-shared") // Create shared library instead of executable
                .arg("-o")
                .arg(&so_path_for_clang)
                .arg(&program_temp_path)
                .arg(&qis_ffi_import_lib) // Link QIS FFI import library for setup/teardown/___* symbols
                .arg(&shim_lib_path) // Link against selene shim import library to resolve selene_* symbols
                // NOTE: On Windows, DO NOT link helios_lib_path - it conflicts with DLL symbols
                // The static library contains stub implementations that we replace with DLL versions
                .arg("-Wl,/EXPORT:qmain"); // Export qmain symbol for GetProcAddress
            debug!(
                "Windows: Linking against selene shim import library to resolve selene_* symbols"
            );
            debug!("Windows: Exporting qmain entry point (auto-wrapped from main if needed)");
        }

        #[cfg(not(target_os = "windows"))]
        {
            clang_cmd
                .arg("-shared") // Create shared library instead of executable
                .arg("-o")
                .arg(&so_path_for_clang)
                .arg(&program_temp_path)
                .arg(&helios_lib_path);
        }

        // Add platform-specific linker flags
        Self::add_platform_linker_flags(&mut clang_cmd);

        // Debug: Print the full clang command
        debug!("Full clang command: {clang_cmd:?}");

        let output = clang_cmd
            .output()
            .map_err(|e| InterfaceError::LoadError(format!("Failed to run clang: {e}")))?;

        if !output.status.success() {
            error!("Linking FAILED!");
            debug!("stderr: {}", String::from_utf8_lossy(&output.stderr));
            debug!("stdout: {}", String::from_utf8_lossy(&output.stdout));

            // On Windows, check if we're still getting LNK2019 errors for selene_* symbols
            #[cfg(target_os = "windows")]
            {
                let stderr_str = String::from_utf8_lossy(&output.stderr);
                if stderr_str.contains("LNK2019") {
                    error!("LNK2019 UNRESOLVED SYMBOL ERRORS DETECTED");
                    for line in stderr_str.lines() {
                        if line.contains("LNK2019") || line.contains("unresolved external symbol") {
                            error!("  {line}");
                        }
                    }
                }
            }

            return Err(InterfaceError::LoadError(format!(
                "Linking failed: {}",
                String::from_utf8_lossy(&output.stderr)
            )));
        }

        // Verify the DLL/SO file was created
        info!("Linking succeeded!");
        debug!(
            "Checking if output file exists: {}",
            so_path_for_clang.display()
        );
        if so_path_for_clang.exists() {
            if let Ok(metadata) = std::fs::metadata(&so_path_for_clang) {
                debug!("Output file size: {} bytes", metadata.len());
            }
        } else {
            warn!("Output file does not exist after successful link!");
        }

        // Keep the temporary files alive by storing the TempPaths
        #[cfg(target_os = "windows")]
        {
            // On Windows, link.exe created the DLL file, so we just use the path we reserved
            // We need to manually track this file for cleanup
            // Note: so_file is () on Windows (since we deleted the temp file before linking)
            // so there's nothing to drop
            let () = so_file; // Silence unused variable warning

            debug!(
                "Windows: DLL created by link.exe at: {}",
                so_path_for_clang.display()
            );

            // Store the program bitcode temp path
            self.temp_files.push(program_temp_path);

            // We'll store the DLL path directly since it was created by link.exe
            // Note: This file won't be auto-deleted, but that's okay for temp testing
            // In production, we'd want to use a proper temp file wrapper
        }

        #[cfg(not(target_os = "windows"))]
        {
            let so_temp_path = so_file.into_temp_path();

            // Store both the program bitcode and the .so file to keep them alive
            self.temp_files.push(program_temp_path);
            self.temp_files.push(so_temp_path);
        }

        let so_path = so_path_for_clang.clone();

        self.executable_path = Some(so_path.clone());

        self.metadata
            .insert("library_path".to_string(), so_path.display().to_string());
        self.metadata
            .insert("helios_lib".to_string(), helios_lib_path);

        Ok(so_path)
    }

    /// Execute the program by loading it in-process and calling `qmain()`
    fn execute_program(&mut self) -> Result<OperationCollector, InterfaceError> {
        let so_path = self.executable_path.as_ref().ok_or_else(|| {
            InterfaceError::ExecutionError("No shared library created".to_string())
        })?;

        // Get the path to our PECOS selene shim library
        let shim_path = crate::shim::get_shim_library_path().ok_or_else(|| {
            InterfaceError::ExecutionError(
                "PECOS selene shim library not found - build script may have failed".to_string(),
            )
        })?;

        // Architecture note:
        // The __quantum__* FFI symbols are in libpecos_qis_ffi.so (Rust cdylib from pecos-qis-ffi).
        // The selene_* symbols are in libpecos_selene.so (C shim).
        //
        // Symbol resolution chain:
        //   qmain() → ___qalloc() → selene_qalloc() → __quantum__rt__qubit_allocate()
        //
        // We need to load libs in order with RTLD_GLOBAL so symbols are visible:
        //   1. libpecos_qis_ffi.so (provides __quantum__*)
        //   2. libpecos_selene.so (provides selene_*, calls __quantum__*)
        //   3. program.so (provides qmain, calls selene_*)

        // Step 1: Find and load libpecos_qis_ffi.so with RTLD_GLOBAL
        // This provides the __quantum__* symbols for the shim to resolve
        debug!("Finding PECOS QIS FFI library");
        let pecos_qis_lib_path = Self::find_pecos_qis_lib()?;
        debug!(
            "Successfully found QIS FFI library at: {}",
            pecos_qis_lib_path.display()
        );

        debug!("Loading QIS FFI library with RTLD_GLOBAL...");
        let (pecos_qis_lib_global, pecos_qis_lib) = Self::load_library_with_rtld_global(
            &pecos_qis_lib_path,
            "Failed to load PECOS QIS cdylib",
        )?;
        debug!("QIS FFI library loaded successfully!");

        // Step 2: Reset the QIS interface via the cdylib
        // IMPORTANT: We call the cdylib's version to ensure we're using the same thread-local
        // storage instance that the shim will use
        let reset_fn: Symbol<ResetInterfaceFn> = unsafe {
            pecos_qis_lib
                .get(b"pecos_qis_reset_interface\0")
                .map_err(|e| {
                    InterfaceError::ExecutionError(format!("Failed to find reset function: {e}"))
                })?
        };
        unsafe { reset_fn() };

        // Step 3: Load our PECOS C shim with RTLD_GLOBAL
        // The shim has undefined __quantum__* symbols that will resolve to the cdylib
        let (shim_lib_global, shim_lib) =
            Self::load_library_with_rtld_global(&shim_path, "Failed to load PECOS C shim library")?;

        // Step 4: Load the program.so with RTLD_GLOBAL so it can resolve selene_* symbols
        // It will find selene_* symbols from our shim (loaded with RTLD_GLOBAL above)
        debug!("Loading program.so with RTLD_GLOBAL...");
        let (program_lib_global, program_lib) =
            Self::load_library_with_rtld_global(so_path, "Failed to load program library")?;

        // Step 5: Get the execution symbols (qmain and setjmp wrapper)
        let (qmain_fn, call_with_setjmp) = Self::get_execution_symbols(&program_lib, &shim_lib)?;

        // Step 6: Call qmain via our setjmp wrapper
        // The call chain will be:
        //   pecos_call_qmain_with_setjmp(qmain) [from our shim]
        //   → setjmp(user_program_jmpbuf) [saves stack state for longjmp]
        //   → qmain(0) [user code in program.so]
        //   → ___qalloc() [from libhelios.a linked into program.so]
        //   → selene_qalloc() [from libpecos_selene.so C shim]
        //   → __quantum__rt__qubit_allocate() [from libpecos_qis_ffi.so]
        //   → pecos_qis_ffi::with_interface() [thread-local in current process]
        // If an error occurs:
        //   → longjmp(user_program_jmpbuf, error_code) [jumps back to setjmp]
        //   → wrapper catches error and returns error code
        let result = unsafe { call_with_setjmp(*qmain_fn) };
        if result != 0 {
            return Err(InterfaceError::ExecutionError(format!(
                "qmain returned error code: {result}"
            )));
        }
        info!("qmain executed successfully!");

        // Step 7: Collect the operations from thread-local storage via the cdylib
        // IMPORTANT: We call the cdylib's version to get the operations from the same
        // thread-local storage instance that the shim used
        let operations = Self::collect_operations_from_lib(&pecos_qis_lib)?;

        // Keep libraries loaded until we're done
        drop(program_lib);
        drop(program_lib_global);
        drop(shim_lib);
        drop(shim_lib_global);
        drop(pecos_qis_lib);
        drop(pecos_qis_lib_global);

        Ok(operations)
    }
}

impl Default for QisHeliosInterface {
    fn default() -> Self {
        Self::new()
    }
}

impl QisInterface for QisHeliosInterface {
    fn load_program(
        &mut self,
        program_bytes: &[u8],
        format: ProgramFormat,
    ) -> Result<(), InterfaceError> {
        debug!("load_program() called");
        debug!("Program bytes length: {}", program_bytes.len());
        debug!("Program format: {format:?}");

        // Check if Helios can handle this format
        match format {
            ProgramFormat::QisBitcode | ProgramFormat::LlvmBitcode | ProgramFormat::LlvmIrText => {
                debug!("Format is compatible, storing program...");
                self.program = program_bytes.to_vec();
                self.format = format;

                // Create the shared library by linking
                self.create_shared_library()?;

                Ok(())
            }
            ProgramFormat::HugrBytes => {
                error!("HUGR bytes format not supported");
                Err(InterfaceError::InvalidFormat(
                    "Helios interface requires HUGR to be compiled to LLVM first".to_string(),
                ))
            }
        }
    }

    fn collect_operations(&mut self) -> Result<OperationCollector, InterfaceError> {
        // Execute the program and collect operations
        self.execute_program()
    }

    fn execute_with_measurements(
        &mut self,
        _measurements: BTreeMap<usize, bool>,
    ) -> Result<OperationCollector, InterfaceError> {
        // TODO: Implement measurement support by pre-populating results via cdylib
        // For now, just execute the program normally
        self.execute_program()
    }

    fn metadata(&self) -> BTreeMap<String, String> {
        self.metadata.clone()
    }

    fn name(&self) -> &'static str {
        "Helios (dlopen)"
    }

    fn reset(&mut self) -> Result<(), InterfaceError> {
        // Reset is not needed for this interface - it happens at the start of execute_program
        Ok(())
    }
}
