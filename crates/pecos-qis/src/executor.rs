//! Helios interface executor
//!
//! This module implements the `QisInterface` trait for Selene's Helios compiler.

use crate::qis_interface::{DynamicSyncHandle, InterfaceError, ProgramFormat, QisInterface};
use libloading::{Library, Symbol};
use log::{debug, error, info, warn};
use pecos_qis_ffi_types::OperationCollector;
use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;
use tempfile::NamedTempFile;

/// Process-wide singleton for the QIS FFI library.
///
/// On macOS, loading the same dynamic library multiple times creates separate
/// thread-local storage (TLS) instances for each load. This causes crashes when
/// code in one library instance tries to access TLS data from another instance.
///
/// By making this a singleton, all code in the process shares the same library
/// instance and the same TLS, avoiding the macOS TLS isolation issue.
///
/// We store the initialization result (Ok or Err) and both handles:
/// - The `RTLD_GLOBAL` handle to keep symbols visible to other libraries
/// - The regular Library handle for symbol lookup
///
/// Note: We use `SharedLibrary` wrapper to make the library handle `Sync`.
/// This is safe because:
/// 1. The library is loaded once and never dropped (lives for process lifetime)
/// 2. On Unix, `dlsym()` is thread-safe once the library is loaded
/// 3. We only read from the library (no mutation after initialization)
static QIS_FFI_LIB_SINGLETON: OnceLock<Result<SharedLibrary, String>> = OnceLock::new();

/// Process-wide singleton for the shim library.
///
/// The PECOS C shim library (`libpecos_selene.so/dylib`) provides the selene_*
/// functions that bridge to __quantum__* FFI functions. On macOS, loading and
/// unloading this library repeatedly (once per shot in dynamic execution mode)
/// can cause issues with the dynamic linker.
///
/// By making it a singleton, we load it once and keep it for the process lifetime.
static SHIM_LIB_SINGLETON: OnceLock<Result<SharedLibrary, String>> = OnceLock::new();

/// Process-wide cache for program libraries (keyed by file path).
///
/// When engines are cloned for parallel shot execution, each clone creates its own
/// interface, which would normally load its own program library. On macOS, this
/// repeated loading causes dynamic linker issues.
///
/// By caching program libraries by their file path, clones that use the same
/// compiled program share the same loaded library instance.
///
/// We use `Box<SharedLibrary>` so that when the `BTreeMap` grows/reallocates, the
/// actual library data stays in place on the heap (only the Box pointer moves).
static PROGRAM_LIB_CACHE: OnceLock<
    std::sync::Mutex<std::collections::BTreeMap<PathBuf, Box<SharedLibrary>>>,
> = OnceLock::new();

/// Cache mapping program content hash to compiled shared library path.
///
/// When multiple interfaces are created with the same program content (e.g., when
/// engines are cloned for parallel execution), this cache ensures they all use
/// the same compiled shared library file.
static COMPILED_PROGRAM_CACHE: OnceLock<
    std::sync::Mutex<std::collections::BTreeMap<u64, PathBuf>>,
> = OnceLock::new();

/// Tracks whether cache cleanup has been performed (once per process).
static CACHE_CLEANUP_DONE: OnceLock<()> = OnceLock::new();

/// Directory for persistent compiled program cache.
///
/// Unlike temp files that are deleted when the process exits, files in this directory
/// persist across process invocations. This enables:
/// - Tests running in parallel (each subprocess can reuse previously compiled programs)
/// - Repeated `pecos run` commands to reuse cached compilation
///
/// The cache is cleaned up periodically (files older than 24 hours are removed on startup).
fn get_persistent_cache_dir() -> Result<PathBuf, InterfaceError> {
    // Use PECOS_CACHE_DIR if set, otherwise use a subdirectory of the system temp dir
    let cache_dir = std::env::var("PECOS_CACHE_DIR").map_or_else(
        |_| std::env::temp_dir().join("pecos_compiled_cache"),
        PathBuf::from,
    );

    // Ensure the directory exists
    std::fs::create_dir_all(&cache_dir)
        .map_err(|e| InterfaceError::LoadError(format!("Failed to create cache directory: {e}")))?;

    // Cleanup old files (older than 24 hours) - do this once per process
    CACHE_CLEANUP_DONE.get_or_init(|| {
        cleanup_old_cache_files(&cache_dir, 24 * 60 * 60); // 24 hours
    });

    Ok(cache_dir)
}

/// Remove cache files older than the specified age in seconds
fn cleanup_old_cache_files(cache_dir: &Path, max_age_secs: u64) {
    let now = std::time::SystemTime::now();
    let Ok(entries) = std::fs::read_dir(cache_dir) else {
        return;
    };
    for entry in entries.flatten() {
        let dominated_by_age = entry
            .metadata()
            .ok()
            .and_then(|m| m.modified().ok())
            .and_then(|modified| now.duration_since(modified).ok())
            .is_some_and(|age| age.as_secs() > max_age_secs);

        if dominated_by_age {
            debug!("Removing old cache file: {}", entry.path().display());
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

/// File-based lock for cross-process synchronization during compilation.
///
/// This prevents multiple processes from compiling the same program simultaneously,
/// which would waste resources and potentially cause race conditions.
///
/// The lock is acquired by creating a `.lock` file with `O_CREAT | O_EXCL` semantics.
/// If the lock file already exists, the process waits and retries.
struct CompilationLock {
    lock_path: PathBuf,
}

impl CompilationLock {
    /// Maximum time to wait for the lock (in seconds)
    const MAX_WAIT_SECS: u64 = 120;
    /// Time between retry attempts (in milliseconds)
    const RETRY_DELAY_MS: u64 = 100;
    /// Maximum age of a lock file before considering it stale (in seconds)
    const STALE_LOCK_SECS: u64 = 300;

    /// Try to acquire a compilation lock for the given cache path.
    ///
    /// Returns `Some(lock)` if acquired, `None` if the compiled file appeared while waiting.
    fn acquire(cache_path: &Path) -> Result<Option<Self>, InterfaceError> {
        let lock_path = cache_path.with_extension("lock");
        let start = std::time::Instant::now();

        loop {
            // Try to create the lock file exclusively
            match std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&lock_path)
            {
                Ok(file) => {
                    // Write our PID to help debug stale locks
                    use std::io::Write;
                    let mut file = file;
                    let _ = writeln!(file, "{}", std::process::id());
                    debug!("Acquired compilation lock: {}", lock_path.display());
                    return Ok(Some(Self { lock_path }));
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    // Lock exists - another process is compiling

                    // Check if the compiled file appeared (other process finished)
                    if cache_path.exists() {
                        debug!(
                            "Compiled file appeared while waiting for lock: {}",
                            cache_path.display()
                        );
                        // Try to clean up stale lock if we can
                        let _ = std::fs::remove_file(&lock_path);
                        return Ok(None);
                    }

                    // Check if lock is stale (process crashed)
                    let lock_age = std::fs::metadata(&lock_path)
                        .ok()
                        .and_then(|m| m.modified().ok())
                        .and_then(|modified| {
                            std::time::SystemTime::now().duration_since(modified).ok()
                        });

                    if let Some(age) = lock_age.filter(|a| a.as_secs() > Self::STALE_LOCK_SECS) {
                        warn!(
                            "Removing stale compilation lock ({}s old): {}",
                            age.as_secs(),
                            lock_path.display()
                        );
                        let _ = std::fs::remove_file(&lock_path);
                        continue; // Try again immediately
                    }

                    // Check timeout
                    if start.elapsed().as_secs() > Self::MAX_WAIT_SECS {
                        return Err(InterfaceError::LoadError(format!(
                            "Timeout waiting for compilation lock: {}",
                            lock_path.display()
                        )));
                    }

                    // Wait and retry
                    debug!(
                        "Waiting for compilation lock: {} (elapsed: {:?})",
                        lock_path.display(),
                        start.elapsed()
                    );
                    std::thread::sleep(std::time::Duration::from_millis(Self::RETRY_DELAY_MS));
                }
                Err(e) => {
                    return Err(InterfaceError::LoadError(format!(
                        "Failed to create compilation lock: {e}"
                    )));
                }
            }
        }
    }
}

impl Drop for CompilationLock {
    fn drop(&mut self) {
        debug!("Releasing compilation lock: {}", self.lock_path.display());
        let _ = std::fs::remove_file(&self.lock_path);
    }
}

/// Thread-safe wrapper for a loaded dynamic library.
///
/// This wrapper exists because `libloading::Library` is `!Sync` by default
/// (for safety on some platforms). However, for our use case:
/// - The library is loaded once at startup and lives for the process lifetime
/// - We only use it for symbol lookups (dlsym), which are thread-safe on Unix
/// - We never drop the library (it's in a static singleton)
///
/// SAFETY: This is only safe because we never drop the library and only use
/// thread-safe operations (dlsym for symbol lookup).
///
/// IMPORTANT: The library handles are wrapped in `ManuallyDrop` to prevent
/// calling `dlclose()` during process exit. Calling `dlclose()` during shutdown
/// can cause hangs because:
/// 1. Thread-local storage may already be partially torn down
/// 2. Other static destructors may be running concurrently
/// 3. The LLVM JIT runtime may be in an inconsistent state
///
/// Since these libraries live for the process lifetime, it's safe (and necessary)
/// to let the OS clean them up during process termination instead of explicitly
/// calling `dlclose()`.
struct SharedLibrary {
    /// The `RTLD_GLOBAL` handle - keeps symbols visible to other libraries
    /// Wrapped in `ManuallyDrop` to prevent `dlclose()` during process exit
    #[cfg(unix)]
    _global_handle: std::mem::ManuallyDrop<libloading::os::unix::Library>,
    #[cfg(windows)]
    _global_handle: std::mem::ManuallyDrop<Library>,
    /// The regular handle for symbol lookups
    /// Wrapped in `ManuallyDrop` to prevent `dlclose()` during process exit
    lib: std::mem::ManuallyDrop<Library>,
}

// SAFETY: See struct documentation above. The library handle is only used for
// thread-safe dlsym operations and is never dropped.
unsafe impl Sync for SharedLibrary {}
unsafe impl Send for SharedLibrary {}

impl SharedLibrary {
    /// Get a symbol from the library
    ///
    /// # Safety
    /// Same safety requirements as `Library::get`
    unsafe fn get<T>(&self, symbol: &[u8]) -> Result<Symbol<'_, T>, libloading::Error> {
        // SAFETY: We're inside an unsafe fn, caller is responsible for safety
        unsafe { self.lib.get(symbol) }
    }

    /// Get a reference to the inner Library for legacy code
    fn inner(&self) -> &Library {
        &self.lib
    }
}

// FFI function type aliases for dlopen symbol lookup
type ResetInterfaceFn = unsafe extern "C" fn();
type GetOperationsFn = unsafe extern "C" fn() -> *mut OperationCollector;
type CallQmainFn = unsafe extern "C" fn(extern "C" fn(u64) -> u64) -> u64;
type CallVoidMainFn = unsafe extern "C" fn(extern "C" fn()) -> u64;

/// The entry-point shape found in a compiled QIR program, bundled with the
/// matching setjmp wrapper from the C shim. Each variant pairs the function
/// pointer ABI with the shim that calls it -- mixing them (e.g. calling a
/// `void main()` through the qmain wrapper) is undefined behaviour.
enum ExecutionEntryPoint<'a> {
    Qmain {
        func: Symbol<'a, extern "C" fn(u64) -> u64>,
        call: Symbol<'a, CallQmainFn>,
    },
    VoidMain {
        func: Symbol<'a, extern "C" fn()>,
        call: Symbol<'a, CallVoidMainFn>,
    },
}
type WaitForNeedResultFn = unsafe extern "C" fn(u64) -> u64;
type SetMeasurementResultFn = unsafe extern "C" fn(u64, bool);
type SignalResultReadyFn = unsafe extern "C" fn();
type AbortExecutionFn = unsafe extern "C" fn();
type GetNamedResultsJsonFn = unsafe extern "C" fn() -> *mut std::ffi::c_char;
type FreeNamedResultsJsonFn = unsafe extern "C" fn(*mut std::ffi::c_char);

/// Synchronization handle for main thread communication with worker thread
///
/// This handle allows the main thread to call FFI functions for synchronization
/// while the interface is running on a worker thread. It uses the same singleton
/// library instance as the worker thread, ensuring TLS consistency on macOS.
pub struct HeliosSyncHandle;

impl HeliosSyncHandle {
    /// Create a new sync handle
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Get the singleton library for FFI calls
    fn get_lib() -> Result<&'static SharedLibrary, InterfaceError> {
        QisHeliosInterface::get_qis_ffi_lib_singleton()
    }
}

impl Default for HeliosSyncHandle {
    fn default() -> Self {
        Self::new()
    }
}

impl DynamicSyncHandle for HeliosSyncHandle {
    fn wait_for_need_result(&self, timeout_ms: u64) -> Option<u64> {
        let lib = Self::get_lib().ok()?;
        let wait_fn: Symbol<WaitForNeedResultFn> =
            unsafe { lib.get(b"pecos_wait_for_need_result\0").ok()? };
        let result_id = unsafe { wait_fn(timeout_ms) };
        if result_id == u64::MAX {
            None
        } else {
            Some(result_id)
        }
    }

    fn set_measurement_result(&self, result_id: u64, value: bool) -> Result<(), InterfaceError> {
        let lib = Self::get_lib()?;
        let set_fn: Symbol<SetMeasurementResultFn> = unsafe {
            lib.get(b"pecos_set_measurement_result\0").map_err(|e| {
                InterfaceError::ExecutionError(format!(
                    "Failed to find pecos_set_measurement_result: {e}"
                ))
            })?
        };
        unsafe { set_fn(result_id, value) };
        debug!("HeliosSyncHandle: Set measurement result {result_id} = {value}");
        Ok(())
    }

    fn signal_result_ready(&self) -> Result<(), InterfaceError> {
        let lib = Self::get_lib()?;
        let signal_fn: Symbol<SignalResultReadyFn> = unsafe {
            lib.get(b"pecos_signal_result_ready\0").map_err(|e| {
                InterfaceError::ExecutionError(format!(
                    "Failed to find pecos_signal_result_ready: {e}"
                ))
            })?
        };
        unsafe { signal_fn() };
        debug!("HeliosSyncHandle: Signaled result ready");
        Ok(())
    }

    fn get_pending_operations(
        &self,
    ) -> Result<Vec<pecos_qis_ffi_types::Operation>, InterfaceError> {
        let lib = Self::get_lib()?;
        // Use pecos_get_pending_operations which reads from the execution context
        // (not pecos_qis_get_operations which reads thread-local storage)
        let get_ops_fn: Symbol<GetOperationsFn> = unsafe {
            lib.get(b"pecos_get_pending_operations\0").map_err(|e| {
                InterfaceError::ExecutionError(format!(
                    "Failed to find pecos_get_pending_operations: {e}"
                ))
            })?
        };
        let collector = unsafe {
            let ptr = get_ops_fn();
            if ptr.is_null() {
                return Ok(Vec::new());
            }
            Box::from_raw(ptr)
        };
        Ok(collector.operations)
    }

    fn abort_execution(&self) -> Result<(), InterfaceError> {
        let lib = Self::get_lib()?;
        let abort_fn: Symbol<AbortExecutionFn> = unsafe {
            lib.get(b"pecos_abort_dynamic_execution\0").map_err(|e| {
                InterfaceError::ExecutionError(format!(
                    "Failed to find pecos_abort_dynamic_execution: {e}"
                ))
            })?
        };
        unsafe { abort_fn() };
        debug!("HeliosSyncHandle: Aborted execution");
        Ok(())
    }

    fn get_named_results(
        &self,
    ) -> Result<std::collections::BTreeMap<String, Vec<bool>>, InterfaceError> {
        let lib = Self::get_lib()?;

        // Get the JSON string
        let get_fn: Symbol<GetNamedResultsJsonFn> = unsafe {
            lib.get(b"pecos_get_named_results_json\0").map_err(|e| {
                InterfaceError::ExecutionError(format!(
                    "Failed to find pecos_get_named_results_json: {e}"
                ))
            })?
        };

        let ptr = unsafe { get_fn() };
        if ptr.is_null() {
            // No named results - return empty map
            return Ok(std::collections::BTreeMap::new());
        }

        // Convert to Rust string
        let c_str = unsafe { std::ffi::CStr::from_ptr(ptr) };
        let json_str = c_str.to_str().map_err(|e| {
            InterfaceError::ExecutionError(format!("Invalid UTF-8 in named results JSON: {e}"))
        })?;

        // Parse JSON
        let result: std::collections::BTreeMap<String, Vec<bool>> = serde_json::from_str(json_str)
            .map_err(|e| {
                InterfaceError::ExecutionError(format!("Failed to parse named results JSON: {e}"))
            })?;

        // Free the JSON string
        let free_fn: Symbol<FreeNamedResultsJsonFn> = unsafe {
            lib.get(b"pecos_free_named_results_json\0").map_err(|e| {
                InterfaceError::ExecutionError(format!(
                    "Failed to find pecos_free_named_results_json: {e}"
                ))
            })?
        };
        unsafe { free_fn(ptr) };

        debug!("HeliosSyncHandle: Got {} named results", result.len());
        Ok(result)
    }
}

/// Derive the project target directory from the compile-time embedded Helios path.
///
/// The compile-time path looks like:
/// `/path/to/project/target/release/build/pecos-qis-HASH/out/libhelios_selene_interface.a`
///
/// We want to extract `/path/to/project/target` so we can search for other build hashes.
fn get_helios_target_dir() -> Option<PathBuf> {
    let compile_time_path = PathBuf::from(env!("HELIOS_LIB_PATH"));
    // Go up from: lib -> out -> pecos-qis-HASH -> build -> release/debug -> target
    compile_time_path
        .parent() // out/
        .and_then(|p| p.parent()) // pecos-qis-HASH/
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
                if name_str.starts_with("pecos-qis-") {
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
        1. Run: cargo clean -p pecos-qis\n\
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

// FFI function types for dynamic circuit coordination
// These must be called via the dynamically loaded library to use the same statics

/// Opaque type representing an execution context
#[repr(C)]
pub struct ExecutionContext {
    _private: [u8; 0],
}

/// Wrapper for `ExecutionContext` pointer that is Send + Sync
///
/// This is safe because:
/// 1. The `ExecutionContext` is internally thread-safe (uses atomic operations and mutexes)
/// 2. Each execution context is designed to be shared between a worker thread and main thread
/// 3. The pointer is only used to call FFI functions that handle their own synchronization
struct ExecutionContextPtr(*mut ExecutionContext);

// SAFETY: ExecutionContext is internally thread-safe and designed for cross-thread sharing
unsafe impl Send for ExecutionContextPtr {}
unsafe impl Sync for ExecutionContextPtr {}

type CreateExecutionContextFn = unsafe extern "C" fn() -> *mut ExecutionContext;
type DestroyExecutionContextFn = unsafe extern "C" fn(ctx: *mut ExecutionContext);
type RegisterExecutionContextFn = unsafe extern "C" fn(ctx: *mut ExecutionContext);
type EnableDynamicModeFn = unsafe extern "C" fn();
type DisableDynamicModeFn = unsafe extern "C" fn();

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

    // Note: The QIS FFI library, shim library, and program libraries are stored in
    // process-wide caches/singletons to avoid macOS TLS/dynamic linker issues.
    // Program libraries are cached by path in PROGRAM_LIB_CACHE.
    /// Execution context for dynamic circuit coordination
    /// Created when dynamic mode is enabled, destroyed when disabled
    execution_context: Option<ExecutionContextPtr>,
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
            execution_context: None,
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

        let profile_order = if cfg!(debug_assertions) {
            ["debug", "release"]
        } else {
            ["release", "debug"]
        };

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
            for profile in &profile_order {
                candidate_paths.push(current_dir.join(format!("target/{profile}/{lib_name}")));
                candidate_paths.push(current_dir.join(format!("target/{profile}/deps/{lib_name}")));
            }

            // Search up the directory tree for workspace root (when running from Python)
            let mut search_dir = current_dir.as_path();
            for _ in 0..5 {
                // Search up to 5 levels
                if let Some(parent) = search_dir.parent() {
                    for profile in &profile_order {
                        candidate_paths.push(parent.join(format!("target/{profile}/{lib_name}")));
                        candidate_paths
                            .push(parent.join(format!("target/{profile}/deps/{lib_name}")));
                    }
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

    /// Get or initialize the process-wide QIS FFI library singleton.
    ///
    /// This ensures that all code in the process uses the same library instance,
    /// which is critical on macOS where multiple library loads create separate TLS instances.
    ///
    /// Returns a reference to the `SharedLibrary` wrapper for symbol lookups.
    fn get_qis_ffi_lib_singleton() -> Result<&'static SharedLibrary, InterfaceError> {
        let result = QIS_FFI_LIB_SINGLETON.get_or_init(|| match Self::find_pecos_qis_lib() {
            Ok(lib_path) => {
                debug!(
                    "Initializing QIS FFI library singleton from: {}",
                    lib_path.display()
                );

                match Self::load_library_with_rtld_global(
                    &lib_path,
                    "Failed to load QIS FFI library singleton",
                ) {
                    Ok((lib_global, lib)) => {
                        debug!("QIS FFI library singleton initialized successfully");
                        Ok(SharedLibrary {
                            _global_handle: std::mem::ManuallyDrop::new(lib_global),
                            lib: std::mem::ManuallyDrop::new(lib),
                        })
                    }
                    Err(e) => Err(e.to_string()),
                }
            }
            Err(e) => Err(e.to_string()),
        });

        result
            .as_ref()
            .map_err(|e| InterfaceError::ExecutionError(e.clone()))
    }

    /// Get or cache a program library by path.
    ///
    /// When engines are cloned for parallel shot execution, each clone creates a new
    /// interface and would normally load its own program library. On macOS, this
    /// repeated loading causes dynamic linker issues.
    ///
    /// By caching program libraries by path, all clones that compile to the same
    /// shared library path share the same loaded library instance.
    fn get_or_cache_program_lib(path: &Path) -> Result<&'static SharedLibrary, InterfaceError> {
        let cache = PROGRAM_LIB_CACHE
            .get_or_init(|| std::sync::Mutex::new(std::collections::BTreeMap::new()));

        // First check: quick lookup with lock held briefly
        {
            let cache_guard = cache.lock().map_err(|e| {
                InterfaceError::ExecutionError(format!("Failed to lock program cache: {e}"))
            })?;

            if let Some(boxed_lib) = cache_guard.get(path) {
                debug!("Using cached program library for: {}", path.display());
                // SAFETY: Box ensures stable heap address, BTreeMap never removes, OnceLock ensures lifetime
                let ptr: *const SharedLibrary = std::ptr::from_ref::<SharedLibrary>(boxed_lib);
                return Ok(unsafe { &*ptr });
            }
        } // Lock released here before slow library loading

        // Load library WITHOUT holding the lock - this is the slow part
        debug!("Loading program library (outside lock): {}", path.display());
        let (lib_global, lib) =
            Self::load_library_with_rtld_global(path, "Failed to load program library")?;
        let shared_lib = SharedLibrary {
            _global_handle: std::mem::ManuallyDrop::new(lib_global),
            lib: std::mem::ManuallyDrop::new(lib),
        };

        // Second check: re-acquire lock and check if another thread already inserted
        let mut cache_guard = cache.lock().map_err(|e| {
            InterfaceError::ExecutionError(format!("Failed to lock program cache: {e}"))
        })?;

        // Double-check: another thread may have inserted while we were loading
        let ptr: *const SharedLibrary = if let Some(boxed_lib) = cache_guard.get(path) {
            debug!(
                "Another thread already cached library for: {}",
                path.display()
            );
            // Use the existing one (drop our loaded library)
            std::ptr::from_ref::<SharedLibrary>(boxed_lib)
        } else {
            // We're first, insert ours
            debug!("Caching program library: {}", path.display());
            cache_guard.insert(path.to_path_buf(), Box::new(shared_lib));
            std::ptr::from_ref::<SharedLibrary>(cache_guard.get(path).expect("just inserted"))
        };

        // SAFETY: Box ensures stable heap address, BTreeMap never removes, OnceLock ensures lifetime
        Ok(unsafe { &*ptr })
    }

    /// Get or initialize the process-wide shim library singleton.
    ///
    /// The PECOS C shim library provides the selene_* functions. On macOS,
    /// loading and unloading it repeatedly can cause dynamic linker issues.
    /// By making it a singleton, we load once and keep it for the process lifetime.
    fn get_shim_lib_singleton() -> Result<&'static SharedLibrary, InterfaceError> {
        let result = SHIM_LIB_SINGLETON.get_or_init(|| {
            let shim_path = crate::shim::get_shim_library_path().ok_or_else(|| {
                "PECOS selene shim library not found - build script may have failed".to_string()
            })?;

            debug!(
                "Initializing shim library singleton from: {}",
                shim_path.display()
            );

            match Self::load_library_with_rtld_global(
                &shim_path,
                "Failed to load PECOS C shim library singleton",
            ) {
                Ok((lib_global, lib)) => {
                    debug!("Shim library singleton initialized successfully");
                    Ok(SharedLibrary {
                        _global_handle: std::mem::ManuallyDrop::new(lib_global),
                        lib: std::mem::ManuallyDrop::new(lib),
                    })
                }
                Err(e) => Err(e.to_string()),
            }
        });

        result
            .as_ref()
            .map_err(|e| InterfaceError::ExecutionError(e.clone()))
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
        use std::os::windows::ffi::OsStrExt;

        // AddDllDirectory FFI - adds directories to the DLL search path
        // This is better than SetDllDirectoryW because it allows multiple directories
        type DllDirectoryCookie = *mut std::ffi::c_void;

        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn AddDllDirectory(path: *const u16) -> DllDirectoryCookie;
            fn RemoveDllDirectory(cookie: DllDirectoryCookie) -> i32;
            fn SetDefaultDllDirectories(flags: u32) -> i32;
        }

        // LOAD_LIBRARY_SEARCH flags
        const LOAD_LIBRARY_SEARCH_DEFAULT_DIRS: u32 = 0x0000_1000;
        const LOAD_LIBRARY_SEARCH_USER_DIRS: u32 = 0x0000_0400;

        // On Windows, there's no RTLD_GLOBAL flag. DLL dependencies need to be
        // findable via the standard DLL search order. For program.dll, we need
        // pecos_qis_ffi.dll and the shim DLL to be findable.
        //
        // We use AddDllDirectory to temporarily add the directories containing
        // our FFI DLLs to the search path.

        // Find the directories containing our dependency DLLs
        let qis_ffi_path = Self::find_pecos_qis_lib().ok();
        let qis_ffi_dir = qis_ffi_path.as_ref().and_then(|p| p.parent());

        let shim_path = crate::shim::get_shim_library_path();
        let shim_dir = shim_path.as_ref().and_then(|p| p.parent());

        // Combine both directories (they may be different)
        let dll_dirs: Vec<&std::path::Path> =
            [qis_ffi_dir, shim_dir].into_iter().flatten().collect();

        // Set the default search order to include user-added directories
        unsafe {
            SetDefaultDllDirectories(
                LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_USER_DIRS,
            );
        }

        // Add each DLL directory to the search path
        let mut cookies: Vec<DllDirectoryCookie> = Vec::new();
        for dir in &dll_dirs {
            let dir_wide: Vec<u16> = dir.as_os_str().encode_wide().chain(Some(0)).collect();
            let cookie = unsafe { AddDllDirectory(dir_wide.as_ptr()) };
            if cookie.is_null() {
                warn!("Windows: Failed to add DLL directory: {}", dir.display());
            } else {
                debug!("Windows: Added DLL search directory: {}", dir.display());
                cookies.push(cookie);
            }
        }

        // Load the library
        let load_result = (|| {
            let lib_global = unsafe {
                Library::new(path)
                    .map_err(|e| InterfaceError::ExecutionError(format!("{error_msg}: {e}")))?
            };

            let lib = unsafe {
                Library::new(path).map_err(|e| {
                    InterfaceError::ExecutionError(format!("{error_msg} (lookup): {e}"))
                })?
            };

            Ok((lib_global, lib))
        })();

        // Remove the added DLL directories
        for cookie in cookies {
            unsafe { RemoveDllDirectory(cookie) };
        }

        load_result
    }

    /// Get the entry point and matching setjmp wrapper from the libraries.
    ///
    /// QIR programs can use one of two entry-point signatures:
    /// - `i64 @qmain(i64)` -- the Helios / adaptive profile. pecos-hugr-qis
    ///   emits this (its `LLVM_MAIN` constant in compiler.rs is `"qmain"`),
    ///   and the pecos-phir RON pipeline fixtures (`ron_support.rs`,
    ///   `qis_pipeline_tests`) use it. The return value is an error code.
    /// - `void @main()` -- the "base profile" form. pecos-phir's MLIR/QIR text
    ///   path matches `@main` directly (see `mlir_toolchain.rs`), and PECOS's QIR
    ///   text tests use this. It's also the most common form for
    ///   externally-authored programs.
    ///
    /// Calling a `void @main()` function through the qmain ABI (`u64 fn(u64)`)
    /// is undefined behaviour: the return register is never set, so what looks
    /// like a "random error code" is actually whatever was in the register on
    /// return. We dispatch on the symbol that's present so each kind is called
    /// with the correct ABI.
    ///
    /// **Known limitation:** dispatch is name-only. A program with the
    /// off-spec signature `void @qmain()` or `i64 @main(i64)` would be
    /// misclassified. The robust fix would be to inspect the LLVM module's
    /// function type before linking and reject (or dispatch on) any signature
    /// other than the two canonical shapes; that requires plumbing the IR
    /// through to this lookup, so it's deferred until we encounter such a
    /// program in practice. Until then, callers should stick to the two
    /// canonical signatures above.
    fn get_execution_symbols<'a>(
        program_lib: &'a Library,
        shim_lib: &'a Library,
    ) -> Result<ExecutionEntryPoint<'a>, InterfaceError> {
        // Prefer `qmain` (Helios profile); fall back to `main` (void-return form).
        // We look up qmain first because it's the only one we want to call
        // through the i64-returning wrapper.
        let qmain_fn: Result<Symbol<'a, extern "C" fn(u64) -> u64>, _> =
            unsafe { program_lib.get(b"qmain\0") };

        if let Ok(func) = qmain_fn {
            let call: Symbol<'a, CallQmainFn> = unsafe {
                shim_lib
                    .get(b"pecos_call_qmain_with_setjmp\0")
                    .map_err(|e| {
                        InterfaceError::ExecutionError(format!(
                            "Failed to find pecos_call_qmain_with_setjmp wrapper: {e}"
                        ))
                    })?
            };
            return Ok(ExecutionEntryPoint::Qmain { func, call });
        }

        // No qmain -- try `main` and dispatch through the void-main wrapper so
        // we don't read a garbage value out of the return register.
        let main_fn: Symbol<'a, extern "C" fn()> = unsafe {
            program_lib.get(b"main\0").map_err(|e| {
                InterfaceError::ExecutionError(format!(
                    "Failed to find qmain or main entry point: {e}"
                ))
            })?
        };
        let call: Symbol<'a, CallVoidMainFn> = unsafe {
            shim_lib
                .get(b"pecos_call_void_main_with_setjmp\0")
                .map_err(|e| {
                    InterfaceError::ExecutionError(format!(
                        "Failed to find pecos_call_void_main_with_setjmp wrapper: {e}"
                    ))
                })?
        };
        Ok(ExecutionEntryPoint::VoidMain {
            func: main_fn,
            call,
        })
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
        use std::hash::{Hash, Hasher};

        // Compute content hash for caching
        // We include the format as a discriminator in case the same bytes could be
        // interpreted differently (e.g., bitcode vs text IR)
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.program.hash(&mut hasher);
        std::mem::discriminant(&self.format).hash(&mut hasher);
        let content_hash = hasher.finish();

        // Check if we already have a compiled library for this content
        let compiled_cache = COMPILED_PROGRAM_CACHE
            .get_or_init(|| std::sync::Mutex::new(std::collections::BTreeMap::new()));

        // Check cache with lock held briefly, then release before loading
        let cached_path_opt: Option<PathBuf> = {
            let cache_guard = compiled_cache.lock().map_err(|e| {
                InterfaceError::LoadError(format!("Failed to lock compiled cache: {e}"))
            })?;

            if let Some(cached_path) = cache_guard.get(&content_hash) {
                debug!(
                    "Using cached compiled library for content hash {content_hash:016x}: {}",
                    cached_path.display()
                );
                // Verify the file still exists (might have been cleaned up)
                if cached_path.exists() {
                    Some(cached_path.clone())
                } else {
                    debug!("Cached path no longer exists, will recompile");
                    None
                }
            } else {
                None
            }
        }; // Lock released here before potentially slow operations

        // If we found a cached path in the in-process cache, load it
        if let Some(cached_path) = cached_path_opt {
            self.executable_path = Some(cached_path.clone());
            let _lib = Self::get_or_cache_program_lib(&cached_path)?;
            debug!("Successfully loaded cached program library from in-process cache");
            return Ok(cached_path);
        }

        // Check for a persistent cache file (survives process restarts)
        let cache_dir = get_persistent_cache_dir()?;
        let lib_suffix = if cfg!(target_os = "windows") {
            ".dll"
        } else {
            ".so"
        };
        let persistent_cache_path =
            cache_dir.join(format!("program_{content_hash:016x}{lib_suffix}"));

        if persistent_cache_path.exists() {
            debug!(
                "Found persistent cache file: {}",
                persistent_cache_path.display()
            );
            // Load the cached library
            match Self::get_or_cache_program_lib(&persistent_cache_path) {
                Ok(_lib) => {
                    // Update in-process cache
                    {
                        let compiled_cache = COMPILED_PROGRAM_CACHE.get_or_init(|| {
                            std::sync::Mutex::new(std::collections::BTreeMap::new())
                        });
                        if let Ok(mut cache_guard) = compiled_cache.lock() {
                            cache_guard.insert(content_hash, persistent_cache_path.clone());
                        }
                    }
                    self.executable_path = Some(persistent_cache_path.clone());
                    info!(
                        "Loaded program from persistent cache: {}",
                        persistent_cache_path.display()
                    );
                    return Ok(persistent_cache_path);
                }
                Err(e) => {
                    // Cache file is invalid, remove it and recompile
                    warn!(
                        "Persistent cache file invalid ({}), will recompile: {}",
                        e,
                        persistent_cache_path.display()
                    );
                    let _ = std::fs::remove_file(&persistent_cache_path);
                }
            }
        }

        // Acquire compilation lock to prevent multiple processes from compiling simultaneously
        // This is a cross-process lock using file system primitives
        let Some(_compilation_lock) = CompilationLock::acquire(&persistent_cache_path)? else {
            // Another process compiled it while we waited - load the cached version
            debug!("Another process compiled the program, loading from cache");
            match Self::get_or_cache_program_lib(&persistent_cache_path) {
                Ok(_lib) => {
                    let compiled_cache = COMPILED_PROGRAM_CACHE
                        .get_or_init(|| std::sync::Mutex::new(std::collections::BTreeMap::new()));
                    if let Ok(mut cache_guard) = compiled_cache.lock() {
                        cache_guard.insert(content_hash, persistent_cache_path.clone());
                    }
                    self.executable_path = Some(persistent_cache_path.clone());
                    info!(
                        "Loaded program compiled by another process: {}",
                        persistent_cache_path.display()
                    );
                    return Ok(persistent_cache_path);
                }
                Err(e) => {
                    // The file that appeared is invalid - we need to recompile
                    // But we don't have the lock, so we need to acquire it
                    warn!("Cached file from other process is invalid: {e}");
                    let _ = std::fs::remove_file(&persistent_cache_path);
                    // Retry by recursively calling ourselves (will try to get lock again)
                    return self.create_shared_library();
                }
            }
        };

        // Double-check the file doesn't exist after acquiring the lock
        // (another process may have created it between our check and lock acquisition)
        if persistent_cache_path.exists() {
            match Self::get_or_cache_program_lib(&persistent_cache_path) {
                Ok(_lib) => {
                    let compiled_cache = COMPILED_PROGRAM_CACHE
                        .get_or_init(|| std::sync::Mutex::new(std::collections::BTreeMap::new()));
                    if let Ok(mut cache_guard) = compiled_cache.lock() {
                        cache_guard.insert(content_hash, persistent_cache_path.clone());
                    }
                    self.executable_path = Some(persistent_cache_path.clone());
                    info!(
                        "Loaded program from cache (appeared after lock): {}",
                        persistent_cache_path.display()
                    );
                    return Ok(persistent_cache_path);
                }
                Err(e) => {
                    warn!("Cache file invalid after acquiring lock: {e}");
                    let _ = std::fs::remove_file(&persistent_cache_path);
                }
            }
        }

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

        // We already determined lib_suffix above for persistent cache
        // Create shared library path - use persistent cache path
        debug!(
            "Will compile to persistent cache: {}",
            persistent_cache_path.display()
        );

        // Use persistent cache path directly
        // We compile to a temp file first, then rename to avoid partial/corrupted cache files
        let so_path_for_clang = {
            // Use a temp file with a unique suffix to avoid conflicts during compilation
            let temp_path = persistent_cache_path.with_extension(format!(
                "{}.compiling.{}",
                lib_suffix.trim_start_matches('.'),
                std::process::id()
            ));
            debug!("Compiling to temp path first: {}", temp_path.display());
            temp_path
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
                .arg(&program_temp_path);
            // NOTE: We intentionally do NOT link helios_lib_path here.
            // The helios library statically defines ___read_future_bool which would
            // shadow our dynamic version from libpecos_qis_ffi.so.
            // Instead, we let all ___* symbols resolve at runtime from libpecos_qis_ffi.so
            // which is loaded with RTLD_GLOBAL before program.so.
            // This enables dynamic circuits because our ___read_future_bool has the
            // callback mechanism to pause and get measurement results from the simulator.
            debug!(
                "Not linking helios library - ___* symbols will resolve from libpecos_qis_ffi.so at runtime"
            );
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

        // Rename temp file to persistent cache path
        // Use rename for atomicity on Unix (instant, no partial writes)
        // On some filesystems, rename across mount points may fail, so we fall back to copy+delete
        let final_path = if std::fs::rename(&so_path_for_clang, &persistent_cache_path).is_ok() {
            debug!(
                "Renamed compiled library to persistent cache: {}",
                persistent_cache_path.display()
            );
            persistent_cache_path.clone()
        } else {
            // Rename failed (possibly cross-filesystem), try copy
            debug!("Rename failed, trying copy for persistent cache...");
            if std::fs::copy(&so_path_for_clang, &persistent_cache_path).is_ok() {
                let _ = std::fs::remove_file(&so_path_for_clang);
                debug!(
                    "Copied compiled library to persistent cache: {}",
                    persistent_cache_path.display()
                );
                persistent_cache_path.clone()
            } else {
                // Copy also failed, use the temp path (it will work, just won't persist)
                warn!(
                    "Failed to create persistent cache, using temp path: {}",
                    so_path_for_clang.display()
                );
                so_path_for_clang.clone()
            }
        };

        // Keep the program bitcode temp path alive (not the .so since it's now in cache)
        self.temp_files.push(program_temp_path);

        let so_path = final_path;

        self.executable_path = Some(so_path.clone());

        self.metadata
            .insert("library_path".to_string(), so_path.display().to_string());
        self.metadata
            .insert("helios_lib".to_string(), helios_lib_path);

        // Cache the compiled path for content-based lookup
        {
            let compiled_cache = COMPILED_PROGRAM_CACHE
                .get_or_init(|| std::sync::Mutex::new(std::collections::BTreeMap::new()));
            if let Ok(mut cache_guard) = compiled_cache.lock() {
                cache_guard.insert(content_hash, so_path.clone());
                debug!(
                    "Cached compiled library for content hash {content_hash:016x}: {}",
                    so_path.display()
                );
            }
        }

        // Load the program library into the global cache.
        // This avoids repeated library load/unload cycles which cause instability on macOS.
        debug!("Loading program library into global cache...");
        let _lib = Self::get_or_cache_program_lib(&so_path)?;
        debug!("Program library loaded into cache successfully");

        Ok(so_path)
    }

    /// Execute the program by loading it in-process and calling `qmain()`
    ///
    /// If `measurements` is Some, pre-populate the measurement results via the cdylib
    /// before executing. This enables dynamic circuits where conditionals depend on
    /// measurement results from previous simulation passes.
    fn execute_program(
        &mut self,
        measurements: Option<&BTreeMap<usize, bool>>,
    ) -> Result<OperationCollector, InterfaceError> {
        // Verify the executable path is set
        let so_path = self.executable_path.as_ref().ok_or_else(|| {
            InterfaceError::ExecutionError(
                "No program library path. Call load_program() first.".to_string(),
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

        // Step 1: Get the process-wide QIS FFI library singleton
        // This provides the __quantum__* symbols for the shim to resolve.
        //
        // IMPORTANT: We use a process-wide singleton to ensure all code uses the same
        // library instance. On macOS, loading the same library multiple times creates
        // separate thread-local storage (TLS) instances, which causes crashes when
        // the execution context is accessed from a different library instance.
        // The singleton ensures all clones of QisEngine share the same TLS.
        let pecos_qis_lib = Self::get_qis_ffi_lib_singleton()?;
        debug!("Using QIS FFI library from process-wide singleton");

        // Always register the execution context on the current thread
        // This is necessary because TLS registration is per-thread, so the worker thread
        // needs to register the same context that was created on the main thread
        let current_thread_id = std::thread::current().id();
        if let Some(ExecutionContextPtr(ctx)) = self.execution_context {
            let register_fn: Symbol<RegisterExecutionContextFn> = unsafe {
                pecos_qis_lib
                    .get(b"pecos_register_execution_context\0")
                    .map_err(|e| {
                        InterfaceError::ExecutionError(format!(
                            "Failed to find pecos_register_execution_context: {e}"
                        ))
                    })?
            };
            debug!("execute_program: registering context {ctx:?} on thread {current_thread_id:?}");
            unsafe { register_fn(ctx) };
            debug!("execute_program: context {ctx:?} registered on thread {current_thread_id:?}");
        } else {
            debug!(
                "execute_program: NO execution context to register on thread {current_thread_id:?}"
            );
        }

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

        // Step 2b: Pre-populate measurement results if provided
        // This enables dynamic circuits - the results are stored in the cdylib's thread-local
        // storage where ___read_future_bool will find them
        if let Some(measurements) = measurements {
            type SetMeasurementResultFn = unsafe extern "C" fn(result_id: u64, value: bool);
            let set_result_fn: Symbol<SetMeasurementResultFn> = unsafe {
                pecos_qis_lib
                    .get(b"pecos_set_measurement_result\0")
                    .map_err(|e| {
                        InterfaceError::ExecutionError(format!(
                            "Failed to find pecos_set_measurement_result: {e}"
                        ))
                    })?
            };

            for (&result_id, &value) in measurements {
                debug!("Pre-populating measurement result via cdylib: {result_id} = {value}");
                unsafe { set_result_fn(result_id as u64, value) };
            }
        }

        // Step 3: Get the PECOS C shim library from the singleton
        // The shim has undefined __quantum__* symbols that will resolve to the cdylib
        // We use a singleton to avoid repeated library load/unload cycles on macOS
        let shim_lib = Self::get_shim_lib_singleton()?;
        debug!("Using shim library from process-wide singleton");

        // Step 4: Get the program library from the global cache
        // The program library is cached to avoid repeated load/unload cycles on macOS.
        let program_lib = Self::get_or_cache_program_lib(so_path)?;
        debug!("Using cached program library");

        // Step 5: Get the execution entry point (qmain or main) and matching
        // setjmp wrapper from the shim.
        let entry_point = Self::get_execution_symbols(program_lib.inner(), shim_lib.inner())?;

        // Step 6: Call the entry point via the matching setjmp wrapper.
        // The call chain will be:
        //   pecos_call_qmain_with_setjmp(qmain) [from our shim]
        //   → setjmp(user_program_jmpbuf) [saves stack state for longjmp]
        //   → qmain(0)  -or-  main()  [user code in program.so]
        //   → ___qalloc() [from libhelios.a linked into program.so]
        //   → selene_qalloc() [from libpecos_selene.so C shim]
        //   → __quantum__rt__qubit_allocate() [from libpecos_qis_ffi.so]
        //   → pecos_qis_ffi::with_interface() [thread-local in current process]
        // If an error occurs:
        //   → longjmp(user_program_jmpbuf, error_code) [jumps back to setjmp]
        //   → wrapper catches error and returns error code
        let (entry_label, result) = match &entry_point {
            ExecutionEntryPoint::Qmain { func, call } => ("qmain", unsafe { call(**func) }),
            ExecutionEntryPoint::VoidMain { func, call } => ("main", unsafe { call(**func) }),
        };
        if result != 0 {
            return Err(InterfaceError::ExecutionError(format!(
                "{entry_label} returned error code: {result}"
            )));
        }
        info!("{entry_label} executed successfully!");

        // Step 7: Collect the operations from thread-local storage via the cdylib
        // IMPORTANT: We call the cdylib's version to get the operations from the same
        // thread-local storage instance that the shim used
        let operations = Self::collect_operations_from_lib(pecos_qis_lib.inner())?;

        // Note: All libraries (QIS FFI, shim, and program) are in process-wide caches.
        // They remain loaded for the process lifetime to avoid macOS dynamic linker issues.

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
        // Execute the program and collect operations (no pre-populated measurements)
        self.execute_program(None)
    }

    fn execute_with_measurements(
        &mut self,
        measurements: BTreeMap<usize, bool>,
    ) -> Result<OperationCollector, InterfaceError> {
        // Execute with pre-populated measurements via the cdylib
        // This enables dynamic circuits where conditionals depend on measurement results
        self.execute_program(Some(&measurements))
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

    // ========================================================================
    // Dynamic execution methods
    // ========================================================================

    fn supports_dynamic(&self) -> bool {
        true
    }

    fn enable_dynamic_mode(&mut self) -> Result<(), InterfaceError> {
        let main_thread_id = std::thread::current().id();
        debug!("Enabling dynamic execution mode on main thread {main_thread_id:?}");

        // Get the process-wide QIS FFI library singleton
        // IMPORTANT: We use a process-wide singleton to ensure all code uses the same
        // library instance. On macOS, loading the library twice creates separate TLS
        // instances, causing crashes when the execution context is accessed from a
        // different library instance.
        let lib = Self::get_qis_ffi_lib_singleton()?;
        debug!("Using QIS FFI library from process-wide singleton for dynamic mode");

        // Destroy any previous execution context from a previous shot.
        // This is safe because we're at the start of a new shot, so the main thread
        // is no longer using the old context (it was kept alive during disable_dynamic_mode
        // to avoid a use-after-free race condition).
        if let Some(ExecutionContextPtr(old_ctx)) = self.execution_context.take() {
            debug!(
                "enable_dynamic_mode: destroying previous execution context {old_ctx:?} on thread {main_thread_id:?}"
            );
            let destroy_fn: Symbol<DestroyExecutionContextFn> = unsafe {
                lib.get(b"pecos_destroy_execution_context\0").map_err(|e| {
                    InterfaceError::ExecutionError(format!(
                        "Failed to find pecos_destroy_execution_context: {e}"
                    ))
                })?
            };
            unsafe { destroy_fn(old_ctx) };
        }

        // Create a new execution context for this shot
        let create_fn: Symbol<CreateExecutionContextFn> = unsafe {
            lib.get(b"pecos_create_execution_context\0").map_err(|e| {
                InterfaceError::ExecutionError(format!(
                    "Failed to find pecos_create_execution_context: {e}"
                ))
            })?
        };
        let ctx = unsafe { create_fn() };
        debug!(
            "enable_dynamic_mode: created execution context {ctx:?} on main thread {main_thread_id:?}"
        );
        self.execution_context = Some(ExecutionContextPtr(ctx));

        // Register the execution context on this (main) thread
        let register_fn: Symbol<RegisterExecutionContextFn> = unsafe {
            lib.get(b"pecos_register_execution_context\0")
                .map_err(|e| {
                    InterfaceError::ExecutionError(format!(
                        "Failed to find pecos_register_execution_context: {e}"
                    ))
                })?
        };
        debug!(
            "enable_dynamic_mode: registering context {ctx:?} on main thread {main_thread_id:?}"
        );
        unsafe { register_fn(ctx) };
        debug!("enable_dynamic_mode: context {ctx:?} registered on main thread {main_thread_id:?}");

        // Now enable dynamic mode
        let enable_fn: Symbol<EnableDynamicModeFn> = unsafe {
            lib.get(b"pecos_enable_dynamic_mode\0").map_err(|e| {
                InterfaceError::ExecutionError(format!(
                    "Failed to find pecos_enable_dynamic_mode: {e}"
                ))
            })?
        };
        unsafe { enable_fn() };
        debug!(
            "enable_dynamic_mode: dynamic mode enabled via FFI on main thread {main_thread_id:?}"
        );

        Ok(())
    }

    fn disable_dynamic_mode(&mut self) -> Result<(), InterfaceError> {
        let worker_thread_id = std::thread::current().id();
        debug!("Disabling dynamic execution mode on worker thread {worker_thread_id:?}");

        // Get the process-wide QIS FFI library singleton
        let lib = Self::get_qis_ffi_lib_singleton()?;

        // Disable dynamic mode first (signals worker_complete and notifies waiters)
        let disable_fn: Symbol<DisableDynamicModeFn> = unsafe {
            lib.get(b"pecos_disable_dynamic_mode\0").map_err(|e| {
                InterfaceError::ExecutionError(format!(
                    "Failed to find pecos_disable_dynamic_mode: {e}"
                ))
            })?
        };
        unsafe { disable_fn() };
        debug!(
            "disable_dynamic_mode: dynamic mode disabled via FFI on worker thread {worker_thread_id:?}"
        );

        // Unregister the execution context from this (worker) thread's TLS
        let register_fn: Symbol<RegisterExecutionContextFn> = unsafe {
            lib.get(b"pecos_register_execution_context\0")
                .map_err(|e| {
                    InterfaceError::ExecutionError(format!(
                        "Failed to find pecos_register_execution_context: {e}"
                    ))
                })?
        };
        debug!(
            "disable_dynamic_mode: unregistering context from worker thread {worker_thread_id:?}"
        );
        unsafe { register_fn(std::ptr::null_mut()) };
        debug!(
            "disable_dynamic_mode: context unregistered from worker thread {worker_thread_id:?}"
        );

        // IMPORTANT: Do NOT destroy the execution context here!
        // The main thread may still be inside pecos_wait_for_need_result using the context.
        // The context will be destroyed in enable_dynamic_mode() before the next shot starts,
        // at which point the main thread is guaranteed to not be using the old context.
        // This prevents a use-after-free race condition.

        Ok(())
    }

    fn wait_for_result_needed(&self, timeout_ms: u64) -> Option<u64> {
        // Get the process-wide QIS FFI library singleton
        let lib = Self::get_qis_ffi_lib_singleton().ok()?;

        let wait_fn: Symbol<WaitForNeedResultFn> =
            unsafe { lib.get(b"pecos_wait_for_need_result\0").ok()? };
        let result_id = unsafe { wait_fn(timeout_ms) };
        if result_id == u64::MAX {
            None
        } else {
            Some(result_id)
        }
    }

    fn set_measurement_result(
        &mut self,
        result_id: u64,
        value: bool,
    ) -> Result<(), InterfaceError> {
        // Get the process-wide QIS FFI library singleton
        let lib = Self::get_qis_ffi_lib_singleton()?;

        let set_fn: Symbol<SetMeasurementResultFn> = unsafe {
            lib.get(b"pecos_set_measurement_result\0").map_err(|e| {
                InterfaceError::ExecutionError(format!(
                    "Failed to find pecos_set_measurement_result: {e}"
                ))
            })?
        };
        unsafe { set_fn(result_id, value) };
        debug!("Set measurement result via FFI: {result_id} = {value}");
        Ok(())
    }

    fn signal_result_ready(&mut self) -> Result<(), InterfaceError> {
        // Get the process-wide QIS FFI library singleton
        let lib = Self::get_qis_ffi_lib_singleton()?;

        let signal_fn: Symbol<SignalResultReadyFn> = unsafe {
            lib.get(b"pecos_signal_result_ready\0").map_err(|e| {
                InterfaceError::ExecutionError(format!(
                    "Failed to find pecos_signal_result_ready: {e}"
                ))
            })?
        };
        unsafe { signal_fn() };
        debug!("Signaled result ready via FFI");
        Ok(())
    }

    fn get_pending_operations(
        &self,
    ) -> Result<Vec<pecos_qis_ffi_types::Operation>, InterfaceError> {
        // Get the process-wide QIS FFI library singleton
        let lib = Self::get_qis_ffi_lib_singleton()?;

        // Get operations from the library's thread-local storage
        let get_ops_fn: Symbol<GetOperationsFn> = unsafe {
            lib.get(b"pecos_qis_get_operations\0").map_err(|e| {
                InterfaceError::ExecutionError(format!(
                    "Failed to find pecos_qis_get_operations: {e}"
                ))
            })?
        };
        let collector = unsafe {
            let ptr = get_ops_fn();
            if ptr.is_null() {
                return Ok(Vec::new());
            }
            Box::from_raw(ptr)
        };
        Ok(collector.operations)
    }

    fn get_qis_ffi_lib_path(&self) -> Option<std::path::PathBuf> {
        Self::find_pecos_qis_lib().ok()
    }

    fn get_execution_context_ptr(&self) -> Option<*mut std::ffi::c_void> {
        self.execution_context
            .as_ref()
            .map(|ExecutionContextPtr(ptr)| (*ptr).cast::<std::ffi::c_void>())
    }

    fn get_sync_handle(&self) -> Option<Box<dyn DynamicSyncHandle>> {
        // Return a handle that uses the singleton library for FFI calls
        // This ensures TLS consistency between main thread and worker thread on macOS
        Some(Box::new(HeliosSyncHandle::new()))
    }
}

impl Drop for QisHeliosInterface {
    fn drop(&mut self) {
        // Intentionally skip cleanup of execution context during drop.
        //
        // IMPORTANT: The FFI calls to unregister and destroy the execution context
        // (pecos_register_execution_context and pecos_destroy_execution_context) access
        // thread-local storage (TLS). During process shutdown, TLS may already be partially
        // torn down, which can cause the FFI calls to hang indefinitely. This was the
        // root cause of intermittent test hangs (occurring ~15-20% of the time).
        //
        // Since drop() is typically called during process exit (when the program is terminating),
        // it's safe to skip the cleanup:
        // - The memory will be reclaimed by the OS when the process exits
        // - The TLS entry will be cleaned up by the OS
        //
        // Note: During normal operation (multi-shot execution), the context is cleaned up
        // in enable_dynamic_mode() at the start of each new shot, before the previous
        // context is needed. The context is NOT cleaned up in disable_dynamic_mode() to
        // avoid a use-after-free race condition where the main thread might still be
        // accessing the context when the worker thread tries to destroy it.
        //
        // The drop() path is only reached for the LAST shot's context when:
        // 1. The program is exiting after all shots complete
        // 2. There was a panic or early return
        //
        // In both cases, leaking the context is acceptable and avoids the TLS hang.
        let _ = self.execution_context.take();
    }
}
