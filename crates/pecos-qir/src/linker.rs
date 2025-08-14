//! QIR Linker Module
//!
//! This module is responsible for compiling QIR programs (.ll files) and linking them
//! with the pre-built runtime library to create dynamically loadable libraries.
//!
//! # Overview
//!
//! The QIR compilation process involves:
//! 1. Compiling the QIR file to an object file using LLVM tools
//! 2. Getting the pre-built runtime library from `RuntimeBuilder`
//! 3. Linking them together to create a shared library (.so/.dll/.dylib)
//!
//! # Rebuild Strategy
//!
//! The system manages two types of artifacts that may need rebuilding:
//!
//! ## 1. Static Runtime Library (`~/.cargo/pecos-qir/libpecos_qir.a`)
//!
//! The runtime library rebuild is triggered by:
//! - **Missing library**: If the library doesn't exist at all
//! - **Source changes**: When pecos-qir source files are newer than the library
//! - **Dependency changes**: When Cargo.lock indicates dependency updates
//!
//! The detection happens in two phases:
//! - **Detection phase** (build.rs during `cargo build/test/check`):
//!   - Checks if runtime library exists and is up-to-date
//!   - Creates marker file (`~/.cargo/pecos-qir/.needs_rebuild`) if rebuild needed
//!   - Removes marker if everything is current
//! - **Build phase** (`RuntimeBuilder` when compiling QIR):
//!   - Checks for missing library OR marker file existence
//!   - Builds the static library if needed
//!   - Removes marker file after successful build
//!
//! ## 2. QIR Executables (Compiled QIR linked with runtime)
//!
//! QIR executable rebuild is triggered by:
//! - **Missing executable**: If the compiled library doesn't exist
//! - **QIR source changes**: When the .ll file is newer than the executable
//! - **Runtime library changes**: When the runtime library is newer than the executable
//!
//! The `QirLinker::compile` method handles this by:
//! 1. Checking for cached QIR executable
//! 2. Ensuring runtime library is built/current (via `RuntimeBuilder`)
//! 3. Comparing timestamps: executable vs QIR source and runtime library
//! 4. Rebuilding if any dependency is newer
//!
//! This design ensures seamless operation where rebuilds happen automatically
//! when needed, while avoiding unnecessary recompilation through smart caching.

#[cfg(target_os = "macos")]
use crate::platform::macos::MacOSCompiler;
#[cfg(target_os = "windows")]
use crate::platform::windows::WindowsCompiler;
use crate::platform::{executable_name, standard_llvm_paths};
use crate::runtime_builder::RuntimeBuilder;
use log::{debug, info, warn};
use pecos_core::errors::PecosError;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Links QIR programs with the runtime library to create dynamically loadable libraries
pub struct QirLinker;

impl QirLinker {
    /// Compile and link a QIR program with the runtime to create a dynamically loadable library
    ///
    /// This method orchestrates the complete QIR compilation process with intelligent
    /// caching and rebuild detection.
    ///
    /// # Process Overview
    ///
    /// 1. **Validate** the QIR file exists and is not empty
    /// 2. **Check cache** for existing compiled library
    /// 3. **Ensure runtime** library is built and up-to-date
    /// 4. **Validate cache** by comparing timestamps
    /// 5. **Rebuild if needed** when:
    ///    - No cached library exists
    ///    - QIR source is newer than cached library
    ///    - Runtime library is newer than cached library
    /// 6. **Return** path to the compiled library
    ///
    /// # Arguments
    ///
    /// * `qir_file` - Path to the QIR (.ll) file to compile
    /// * `output_dir` - Optional output directory (defaults to `<qir_dir>/build/`)
    ///
    /// # Returns
    ///
    /// Path to the compiled shared library (.so/.dll/.dylib)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The QIR file does not exist or is empty
    /// - LLVM tools are not installed or are the wrong version
    /// - Compilation of the QIR file fails
    /// - Linking the object file with the runtime library fails
    /// - File system operations fail
    pub fn compile<P: AsRef<Path>>(
        qir_file: P,
        output_dir: Option<P>,
    ) -> Result<PathBuf, PecosError> {
        let qir_file = qir_file.as_ref();
        // Validate the QIR file
        Self::validate_qir_file(qir_file)?;

        // Determine output directory
        let output_dir = Self::prepare_output_directory(qir_file, output_dir)?;

        // Step 1: Check for cached QIR executable
        // We check cache first to avoid updating runtime timestamp unnecessarily
        if let Some(cached_lib) = Self::find_cached_library(qir_file, &output_dir)? {
            // Step 2: Ensure runtime library is built/current
            // RuntimeBuilder checks for missing library OR marker file
            let rust_runtime_lib = RuntimeBuilder::build_runtime()?;

            // Step 3: Validate cached executable is still valid
            // Compare modification times: cached library vs runtime library
            let cached_metadata = fs::metadata(&cached_lib)?;
            let cached_mtime = cached_metadata.modified().map_err(PecosError::IO)?;

            let runtime_metadata = fs::metadata(&rust_runtime_lib)?;
            let runtime_mtime = runtime_metadata.modified().map_err(PecosError::IO)?;

            // If cached library is newer than (or same age as) runtime, use it
            if cached_mtime >= runtime_mtime {
                debug!("Using cached library: {}", cached_lib.display());
                return Ok(cached_lib);
            }

            // Runtime was updated, need to relink
            info!("Cached library is older than runtime library, rebuilding...");
            // Fall through to rebuild
        } else {
            // No cached library exists, ensure runtime is built before we compile
            RuntimeBuilder::build_runtime()?;
        }

        info!("Starting compilation: {}", qir_file.display());

        // Step 4: Build QIR executable
        // Get the runtime library path (already built in steps above)
        let rust_runtime_lib = RuntimeBuilder::build_runtime()?;

        // Generate consistent file paths for caching
        let (object_file, library_file) = Self::generate_file_paths(qir_file, &output_dir);

        // Compile QIR to object file using LLVM
        Self::compile_to_object_file(qir_file, &object_file)?;

        // Link object file with runtime library to create final executable
        Self::link_shared_library(&object_file, &rust_runtime_lib, &library_file)?;

        info!("Compilation successful: {}", library_file.display());

        Ok(library_file)
    }

    /// Find a cached library if it exists and is up-to-date
    fn find_cached_library(
        qir_file: &Path,
        output_dir: &Path,
    ) -> Result<Option<PathBuf>, PecosError> {
        let qir_metadata = fs::metadata(qir_file)?;
        let qir_modified = qir_metadata.modified().map_err(PecosError::IO)?;

        let file_stem = qir_file
            .file_stem()
            .unwrap_or_else(|| "qir_program".as_ref())
            .to_string_lossy();

        let lib_extension = Self::get_library_extension();
        let library_file = output_dir.join(format!("lib{file_stem}.{lib_extension}"));

        // Check if the library file exists
        if library_file.exists() {
            // Check if library is newer than QIR file
            if let Ok(lib_metadata) = fs::metadata(&library_file)
                && let Ok(lib_modified) = lib_metadata.modified()
                && lib_modified >= qir_modified
            {
                return Ok(Some(library_file));
            }
        }

        Ok(None)
    }

    /// Validate that the QIR file exists and is not empty
    fn validate_qir_file(qir_file: &Path) -> Result<(), PecosError> {
        let metadata = fs::metadata(qir_file).map_err(|_| {
            PecosError::Resource(format!("QIR file not found: {}", qir_file.display()))
        })?;

        if metadata.len() == 0 {
            return Err(PecosError::Resource(format!(
                "QIR file is empty: {}",
                qir_file.display()
            )));
        }

        Ok(())
    }

    /// Prepare the output directory
    fn prepare_output_directory<P: AsRef<Path>>(
        qir_file: &Path,
        output_dir: Option<P>,
    ) -> Result<PathBuf, PecosError> {
        let output_dir = output_dir.map_or_else(
            || {
                qir_file
                    .parent()
                    .unwrap_or_else(|| Path::new("."))
                    .join("build")
            },
            |d| d.as_ref().to_path_buf(),
        );

        Self::ensure_dir(&output_dir)?;
        Ok(output_dir)
    }

    /// Generate file paths for object file and library file
    fn generate_file_paths(qir_file: &Path, output_dir: &Path) -> (PathBuf, PathBuf) {
        let file_stem = qir_file
            .file_stem()
            .unwrap_or_else(|| "qir_program".as_ref())
            .to_string_lossy();

        // Use consistent filenames for proper caching
        let object_file = output_dir.join(format!("{file_stem}.o"));
        let library_file =
            output_dir.join(format!("lib{file_stem}.{}", Self::get_library_extension()));

        (object_file, library_file)
    }

    /// Get the platform-specific library extension
    fn get_library_extension() -> &'static str {
        if cfg!(target_os = "linux") {
            "so"
        } else if cfg!(target_os = "macos") {
            "dylib"
        } else {
            "dll"
        }
    }

    /// Find an LLVM tool in the system
    pub(crate) fn find_llvm_tool(tool_name: &str) -> Option<PathBuf> {
        let exec_name = executable_name(tool_name);

        // Check environment variables first
        for env_var in ["PECOS_LLVM_PATH", "LLVM_HOME"] {
            if let Ok(path) = std::env::var(env_var) {
                let tool_path = PathBuf::from(path).join("bin").join(&exec_name);
                if tool_path.exists() {
                    debug!("Found {tool_name} from {env_var}: {}", tool_path.display());
                    return Some(tool_path);
                }
            }
        }

        // Check PATH using which/where
        let command = if cfg!(target_os = "windows") {
            "where"
        } else {
            "which"
        };

        if let Ok(output) = Command::new(command).arg(tool_name).output()
            && output.status.success()
            && let Ok(path_str) = String::from_utf8(output.stdout)
            && let Some(first_line) = path_str.lines().next()
        {
            let path = PathBuf::from(first_line.trim());
            if path.exists() {
                debug!("Found {tool_name} from PATH: {}", path.display());
                return Some(path);
            }
        }

        // Check standard locations
        for base_path in standard_llvm_paths() {
            let tool_path = base_path.join(&exec_name);
            if tool_path.exists() {
                debug!("Found {tool_name} at: {}", tool_path.display());
                return Some(tool_path);
            }
        }

        None
    }

    /// Check LLVM version (requires LLVM 14.x)
    pub(crate) fn check_llvm_version(tool_path: &Path) -> Result<String, String> {
        let output = Command::new(tool_path)
            .arg("--version")
            .output()
            .map_err(|e| format!("Version check failed: {e}"))?;

        if !output.status.success() {
            return Err("Version check failed".to_string());
        }

        let version_output = String::from_utf8_lossy(&output.stdout);
        let version = Self::extract_version(&version_output)?;

        // Check major version
        let major = version
            .split('.')
            .next()
            .and_then(|v| v.parse::<u32>().ok())
            .ok_or("Invalid version format")?;

        if major != 14 {
            return Err(format!("LLVM {version} not supported. Requires LLVM 14.x"));
        }

        Ok(version.to_string())
    }

    /// Extract version number from version output
    fn extract_version(output: &str) -> Result<&str, &'static str> {
        output
            .lines()
            .next()
            .ok_or("Empty version output")?
            .split_whitespace()
            .find(|s| {
                s.chars().any(|c| c.is_ascii_digit())
                    && (s.contains('.') || s.parse::<u32>().is_ok())
            })
            .ok_or("No version found")
    }

    /// Compile QIR file to object file using LLVM tools
    fn compile_to_object_file(qir_file: &Path, object_file: &Path) -> Result<(), PecosError> {
        debug!(
            "Compiling: {} -> {}",
            qir_file.display(),
            object_file.display()
        );

        // Ensure the output directory exists
        if let Some(parent) = object_file.parent() {
            Self::ensure_dir(parent)?;
        }

        #[cfg(target_os = "windows")]
        {
            let clang = Self::find_llvm_tool("clang").ok_or_else(|| {
                PecosError::Processing(
                    "clang not found. Install LLVM 14 and add to PATH.".to_string(),
                )
            })?;

            // Verify LLVM version
            Self::check_llvm_version(&clang).map_err(PecosError::Processing)?;

            debug!("Using clang: {}", clang.display());

            WindowsCompiler::compile_to_object_file(
                qir_file,
                object_file,
                &clang,
                Self::handle_command_error,
                Self::handle_command_status,
            )
        }
        #[cfg(not(target_os = "windows"))]
        {
            let llc_path = Self::find_llvm_tool("llc").ok_or_else(|| {
                PecosError::Processing("llc not found. Install LLVM 14 (e.g., 'apt install llvm-14' or 'brew install llvm@14').".to_string())
            })?;

            // Verify LLVM version
            Self::check_llvm_version(&llc_path).map_err(PecosError::Processing)?;

            let result = Command::new(llc_path)
                .args(["-filetype=obj", "-o"])
                .arg(object_file)
                .arg(qir_file)
                .output();

            let output = Self::handle_command_error(result, "Failed to run llc")?;
            Self::handle_command_status(&output, "llc")?;

            debug!("Successfully compiled QIR to object file");
            Ok(())
        }
    }

    /// Link object file and runtime library into a shared library
    fn link_shared_library(
        object_file: &Path,
        rust_runtime_lib: &Path,
        library_file: &Path,
    ) -> Result<(), PecosError> {
        debug!("Linking object file and runtime library...");

        // Ensure the output directory exists
        if let Some(parent) = library_file.parent() {
            Self::ensure_dir(parent)?;
        }

        // Verify input files exist
        for (file, desc) in [
            (object_file, "Object file"),
            (rust_runtime_lib, "Runtime library"),
        ] {
            if !file.exists() {
                return Err(PecosError::Processing(format!(
                    "{desc} not found: {}",
                    file.display()
                )));
            }
        }

        #[cfg(target_os = "windows")]
        {
            let clang = Self::find_llvm_tool("clang").ok_or_else(|| {
                PecosError::Processing(
                    "clang not found in system. Please install LLVM tools.".to_string(),
                )
            })?;

            WindowsCompiler::link_shared_library(
                object_file,
                rust_runtime_lib,
                library_file,
                &clang,
                Self::handle_command_error,
                Self::handle_command_status,
            )
        }
        #[cfg(target_os = "macos")]
        {
            MacOSCompiler::link_shared_library(
                object_file,
                rust_runtime_lib,
                library_file,
                Self::handle_command_error,
                Self::handle_command_status,
            )
        }
        #[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
        {
            let result = Command::new("gcc")
                .args(["-shared", "-o"])
                .arg(library_file)
                .arg(object_file)
                .arg(rust_runtime_lib)
                .output();

            let output = Self::handle_command_error(result, "Failed to execute gcc")?;
            Self::handle_command_status(&output, "gcc")?;

            debug!("Linked: {}", library_file.display());
            Ok(())
        }
    }

    /// Helper function to handle command execution errors
    fn handle_command_error<T>(
        result: std::io::Result<T>,
        error_msg: &str,
    ) -> Result<T, PecosError> {
        result.map_err(|e| {
            warn!("{error_msg}: {e}");
            PecosError::Processing(format!("QIR compilation failed: {error_msg}: {e}"))
        })
    }

    /// Helper function to handle command execution status
    fn handle_command_status(
        output: &std::process::Output,
        command_name: &str,
    ) -> Result<(), PecosError> {
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let error = PecosError::Processing(format!(
                "QIR compilation failed: {command_name} failed with status: {} and error: {stderr}",
                output.status
            ));
            warn!("{error}");
            return Err(error);
        }
        Ok(())
    }

    /// Ensure a directory exists, creating it if necessary
    fn ensure_dir(path: &Path) -> Result<(), PecosError> {
        if !path.exists() {
            fs::create_dir_all(path)
                .map_err(|e| PecosError::Processing(format!("Failed to create directory: {e}")))?;
        }
        Ok(())
    }
}
