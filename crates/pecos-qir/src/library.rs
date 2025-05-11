use crate::common::get_thread_id;
use libloading::{Library, Symbol};
use log::{debug, trace, warn};
use pecos_core::errors::PecosError;
use pecos_engines::byte_message::QuantumCmd;
use std::collections::HashMap;
use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::thread;
use std::time::Duration;

/// QIR Library for executing quantum programs
///
/// This struct represents a loaded QIR library that can be used to execute
/// quantum programs. It provides methods for calling functions in the library
/// and retrieving the generated quantum commands.
///
/// # Thread Safety
///
/// The QIR Library is designed to be thread-safe and can be used from multiple
/// threads. Each thread gets its own copy of the library to avoid conflicts.
///
/// # Error Handling
///
/// Errors are propagated through the Result type and include context about
/// the operation that failed.
///
/// # Examples
///
/// ```no_run
/// use pecos_qir::library::QirLibrary;
/// use std::path::Path;
///
/// // Load a QIR library from a file
/// let library = QirLibrary::load(Path::new("path/to/library.so")).unwrap();
///
/// // Call the main function in the library
/// library.call_function(b"main").unwrap();
///
/// // Get the generated quantum commands
/// let commands = library.get_binary_commands().unwrap();
///
/// // Reset the library state
/// library.reset().unwrap();
/// ```
pub struct QirLibrary {
    /// The loaded dynamic library
    library: Mutex<Library>,

    /// Path to the library file
    path: PathBuf,

    /// Map of measurement results
    measurement_results: HashMap<String, u32>,
}

impl Clone for QirLibrary {
    fn clone(&self) -> Self {
        let thread_id = get_thread_id();
        debug!(
            "QIR Library: [Thread {}] Cloning library from {:?}",
            thread_id, self.path
        );

        // Load the library again from the same path with retries
        match Self::load_library_with_retries(&self.path, 3, &thread_id) {
            Ok(mut library) => {
                // Copy the measurement results using clone_from for efficiency
                library
                    .measurement_results
                    .clone_from(&self.measurement_results);
                library
            }
            Err(e) => {
                // If we can't load the library, panic with a clear error message
                panic!("Failed to clone QIR library: {e}");
            }
        }
    }
}

impl QirLibrary {
    /// Load a QIR library from the given path
    ///
    /// This method loads a compiled QIR library from the specified path and
    /// initializes the `QirLibrary` struct for interacting with it.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the compiled QIR library
    ///
    /// # Returns
    ///
    /// * `Result<Self, PecosError>` - The loaded library if successful
    ///
    /// # Errors
    ///
    /// This method can return the following errors:
    /// * `PecosError::ResourceError` - If the library file does not exist or cannot be loaded
    ///
    /// # Thread Safety
    ///
    /// This method implements retry logic for handling "Text file busy" errors
    /// that can occur when multiple threads try to load the same library file
    /// simultaneously.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, PecosError> {
        let path = path.as_ref();
        let thread_id = get_thread_id();

        debug!(
            "QIR: [Thread {}] Loading library from {:?}",
            thread_id, path
        );

        // Check if the file exists
        if !path.exists() {
            return Err(Self::log_error(
                "File not found",
                format!("Path: {}", path.display()),
                &thread_id,
            ));
        }

        // Try to load the library with retries
        let max_retries = 3;
        Self::load_library_with_retries(path, max_retries, &thread_id)
    }

    /// Helper function to implement exponential backoff
    fn sleep_with_backoff(retry_count: usize, thread_id: &str) {
        let sleep_duration =
            Duration::from_millis(100 * 2u64.pow(u32::try_from(retry_count).unwrap_or(0)));
        debug!(
            "QIR: [Thread {}] Sleeping for {:?} before retry",
            thread_id, sleep_duration
        );
        thread::sleep(sleep_duration);
    }

    /// Helper function to load a library with retries
    ///
    /// This function attempts to load a library from the given path, with retries
    /// if the initial attempt fails due to "Text file busy" errors.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the library file
    /// * `max_retries` - Maximum number of retry attempts
    /// * `thread_id` - Thread ID for logging
    ///
    /// # Returns
    ///
    /// * `Result<Self, PecosError>` - The loaded library if successful
    fn load_library_with_retries(
        path: &Path,
        max_retries: usize,
        thread_id: &str,
    ) -> Result<Self, PecosError> {
        let mut retry_count = 0;

        while retry_count < max_retries {
            debug!(
                "QIR: [Thread {}] Loading library attempt {}/{}",
                thread_id,
                retry_count + 1,
                max_retries
            );

            // Try to load the library using the path directly
            match unsafe { Library::new(path) } {
                Ok(library) => {
                    debug!(
                        "QIR: [Thread {}] Successfully loaded library from {:?}",
                        thread_id, path
                    );
                    return Ok(Self {
                        library: Mutex::new(library),
                        path: path.to_path_buf(),
                        measurement_results: HashMap::new(),
                    });
                }
                Err(e) => {
                    Self::log_error(
                        "Failed to load library",
                        format!("Attempt {}/{}: {}", retry_count + 1, max_retries, e),
                        thread_id,
                    );

                    // Sleep before retrying, with exponential backoff
                    Self::sleep_with_backoff(retry_count, thread_id);
                    retry_count += 1;
                }
            }
        }

        // If we get here, all attempts failed
        Err(Self::log_error(
            "Failed to load library after multiple attempts",
            format!("Max retries ({max_retries}) exceeded"),
            thread_id,
        ))
    }

    /// Calls a function in the loaded library
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the function to call
    ///
    /// # Returns
    ///
    /// * `Result<i32, PecosError>` - The return value of the function if successful
    ///
    /// # Errors
    ///
    /// This method can return the following errors:
    /// * `PecosError::Resource` - If the function is not found in the library or the call fails
    ///
    /// # Panics
    ///
    /// This function will panic if the internal mutex is poisoned.
    pub fn call_function(&self, name: &[u8]) -> Result<i32, PecosError> {
        let thread_id = get_thread_id();
        debug!(
            "QIR Library: [Thread {}] Calling function {:?}",
            thread_id, name
        );

        unsafe {
            // Get the function pointer
            let library_guard = self.library.lock().unwrap();
            let func: Symbol<unsafe extern "C" fn() -> i32> = library_guard
                .get(name)
                .map_err(|e| Self::log_error("Failed to get function", e, &thread_id))?;

            // Call the function
            let result = func();
            debug!(
                "QIR Library: [Thread {}] Function call returned {}",
                thread_id, result
            );
            Ok(result)
        }
    }

    /// Resets the QIR runtime
    ///
    /// This method calls the `qir_runtime_reset` function in the loaded library
    /// to reset the QIR runtime state.
    ///
    /// # Returns
    ///
    /// * `Result<(), PecosError>` - Success or error
    ///
    /// # Errors
    ///
    /// This method can return the following errors:
    /// * `PecosError::Resource` - If the reset function is not found in the library or the call fails
    ///
    /// # Panics
    ///
    /// This function will panic if the internal mutex is poisoned.
    pub fn reset(&self) -> Result<(), PecosError> {
        let thread_id = get_thread_id();
        debug!("QIR Library: [Thread {}] Resetting QIR runtime", thread_id);

        unsafe {
            // Get the function pointer
            let library_guard = self.library.lock().unwrap();
            let reset: Symbol<unsafe extern "C" fn()> = library_guard
                .get(b"qir_runtime_reset")
                .map_err(|e| Self::log_error("Failed to get reset function", e, &thread_id))?;

            // Call the function
            reset();
            debug!(
                "QIR Library: [Thread {}] Successfully reset QIR runtime",
                thread_id
            );
        }

        Ok(())
    }

    /// Gets the binary commands generated by the QIR runtime
    ///
    /// This method calls the `qir_runtime_get_binary_commands` function in the loaded library
    /// to get the binary commands generated by the QIR runtime.
    ///
    /// # Returns
    ///
    /// * `Result<Vec<QuantumCmd>, PecosError>` - The binary commands if successful
    ///
    /// # Errors
    ///
    /// This method can return the following errors:
    /// * `PecosError::LibraryError` - If the function is not found in the library or the call fails
    ///
    /// # Panics
    ///
    /// This function will panic if the internal mutex is poisoned.
    pub fn get_binary_commands(&self) -> Result<Vec<QuantumCmd>, PecosError> {
        let thread_id = get_thread_id();

        debug!(
            "QIR Library: [Thread {}] Getting binary commands",
            thread_id
        );

        // Get the get_binary_commands function
        let library_guard = self.library.lock().unwrap();
        let get_binary_commands: Symbol<unsafe extern "C" fn() -> *mut c_void> = unsafe {
            library_guard
                .get(b"qir_runtime_get_binary_commands")
                .map_err(|e| {
                    Self::log_error(
                        "Failed to get qir_runtime_get_binary_commands symbol",
                        e,
                        &thread_id,
                    )
                })?
        };

        // Get the free_binary_commands function
        let free_binary_commands: Symbol<unsafe extern "C" fn(*mut c_void)> = unsafe {
            library_guard
                .get(b"qir_runtime_free_binary_commands")
                .map_err(|e| {
                    Self::log_error(
                        "Failed to get qir_runtime_free_binary_commands symbol",
                        e,
                        &thread_id,
                    )
                })?
        };

        // Call the get_binary_commands function
        let commands_ptr = unsafe { get_binary_commands() };
        if commands_ptr.is_null() {
            return Err(Self::log_error(
                "Got null pointer from qir_runtime_get_binary_commands",
                "Cannot retrieve commands",
                &thread_id,
            ));
        }

        // Parse the binary commands
        let commands = unsafe { parse_binary_commands(commands_ptr) };

        // Free the commands
        unsafe { free_binary_commands(commands_ptr) };

        Ok(commands)
    }

    /// Helper function to log errors with thread ID context
    fn log_error<E: std::fmt::Display>(context: &str, error: E, thread_id: &str) -> PecosError {
        let error_msg = format!("{context}: {error}");
        warn!("QIR Library: [Thread {}] {}", thread_id, error_msg);
        PecosError::Resource(error_msg.to_string())
    }
}

impl Drop for QirLibrary {
    fn drop(&mut self) {
        let thread_id = get_thread_id();
        debug!("QIR Library: [Thread {}] Dropping library", thread_id);
    }
}

/// Parse binary commands from a raw pointer
///
/// # Safety
///
/// This function is unsafe because it dereferences a raw pointer.
/// The pointer must be valid and point to a valid binary command structure.
unsafe fn parse_binary_commands(commands_ptr: *mut c_void) -> Vec<QuantumCmd> {
    let thread_id = get_thread_id();

    // This implementation depends on the actual structure of the data
    // For now, we'll assume it's a Vec<QuantumCmd> that we can clone
    let commands = if let Ok(cmds) = std::panic::catch_unwind(|| {
        let cmd_vec_ptr = commands_ptr.cast::<Vec<QuantumCmd>>();
        unsafe { (*cmd_vec_ptr).clone() }
    }) {
        cmds
    } else {
        warn!(
            "QIR Library: [Thread {}] Panic while parsing commands from runtime",
            thread_id
        );
        Vec::new()
    };

    debug!(
        "QIR Library: [Thread {}] Got {} binary commands",
        thread_id,
        commands.len()
    );

    // Log a sample of commands for debugging
    if !commands.is_empty() && log::log_enabled!(log::Level::Trace) {
        let sample_size = std::cmp::min(5, commands.len());
        trace!(
            "QIR Library: [Thread {}] First {} commands:",
            thread_id, sample_size
        );
        for (i, cmd) in commands.iter().take(sample_size).enumerate() {
            trace!("QIR Library: [Thread {}]   {}: {:?}", thread_id, i, cmd);
        }

        if commands.len() > sample_size {
            trace!(
                "QIR Library: [Thread {}]   ... and {} more",
                thread_id,
                commands.len() - sample_size
            );
        }
    }

    commands
}
