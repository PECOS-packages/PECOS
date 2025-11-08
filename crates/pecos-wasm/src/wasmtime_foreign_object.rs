// Copyright 2025 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
// the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

//! Wasmtime-based WebAssembly foreign object implementation
//!
//! This module provides a thread-safe, production-ready WebAssembly foreign object
//! implementation using the Wasmtime runtime. It supports timeout handling, proper
//! resource cleanup, and type conversions.

use crate::foreign_object::ForeignObject;
use log::{debug, warn};
use parking_lot::{Mutex, RwLock};
use pecos_core::errors::PecosError;
use std::any::Any;
use std::path::Path;
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use wasmtime::{
    Config, Engine, Func, Instance, Module, Store, StoreLimits, StoreLimitsBuilder, Trap, Val,
};

/// Length of each tick in milliseconds (10ms per tick)
const WASM_EXECUTION_TICK_LENGTH_MS: u64 = 10;
/// Default timeout in seconds (1 second to match Python implementation)
const DEFAULT_TIMEOUT_SECONDS: f64 = 1.0;

/// Store context holding resource limits
#[derive(Debug)]
struct StoreContext {
    limits: StoreLimits,
}

impl StoreContext {
    fn new(memory_size: Option<usize>) -> Self {
        let mut builder = StoreLimitsBuilder::new();
        if let Some(size) = memory_size {
            builder = builder.memory_size(size);
        }
        Self {
            limits: builder.build(),
        }
    }
}

/// WebAssembly foreign object implementation using Wasmtime
///
/// This implementation provides:
/// - Thread-safe execution with RwLock/Mutex synchronization
/// - Configurable timeout via epoch interruption (default: 1 second)
/// - Configurable memory limits (default: unlimited)
/// - Type conversion between i32/i64 with bounds checking and warnings
/// - Function discovery and caching
/// - Proper resource cleanup via Drop trait
///
/// # Example
///
/// ```no_run
/// # use pecos_wasm::{WasmForeignObject, ForeignObject};
/// // Create with defaults (1-second timeout, unlimited memory)
/// let mut wasm = WasmForeignObject::new("math.wasm").unwrap();
/// wasm.init().unwrap();
///
/// // Or create with custom timeout (5 seconds)
/// let mut wasm = WasmForeignObject::with_timeout("math.wasm", 5.0).unwrap();
///
/// // Or create with custom memory limit (10 MB)
/// let mut wasm = WasmForeignObject::with_limits("math.wasm", 5.0, Some(10 * 1024 * 1024)).unwrap();
///
/// // Execute a function
/// let result = wasm.exec("add", &[5, 3]).unwrap();
/// assert_eq!(result, vec![8]);
/// ```
#[derive(Debug)]
pub struct WasmForeignObject {
    /// WebAssembly binary
    #[allow(dead_code)]
    wasm_bytes: Vec<u8>,
    /// Wasmtime engine
    #[allow(dead_code)]
    engine: Engine,
    /// Wasmtime module
    module: Module,
    /// Wasmtime store (thread-safe)
    store: RwLock<Store<StoreContext>>,
    /// Wasmtime instance (thread-safe)
    instance: RwLock<Option<Instance>>,
    /// Cached function names (thread-safe)
    func_names: Mutex<Option<Vec<String>>>,
    /// Stop flag for epoch increment thread
    stop_flag: Arc<RwLock<bool>>,
    /// Last function call results
    last_results: Vec<Val>,
    /// Timeout in seconds for WASM execution (default: 1.0 second)
    timeout_seconds: f64,
    /// Maximum memory size in bytes per linear memory (default: None = unlimited)
    memory_size: Option<usize>,
}

impl WasmForeignObject {
    /// Calculate maximum ticks from timeout in seconds
    ///
    /// # Parameters
    ///
    /// * `timeout_seconds` - Timeout in seconds
    ///
    /// # Returns
    ///
    /// Number of ticks before timeout
    fn calculate_max_ticks(timeout_seconds: f64) -> u64 {
        // Ensure non-negative timeout (clamp to 0 for negative values)
        // Casting is safe for reasonable timeout values (< 18 quadrillion seconds)
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let timeout_ms = (timeout_seconds.max(0.0) * 1000.0).round() as u64;
        timeout_ms / WASM_EXECUTION_TICK_LENGTH_MS
    }

    /// Create a new WebAssembly foreign object from a file with default timeout
    ///
    /// # Parameters
    ///
    /// * `path` - Path to the WebAssembly file (.wasm or .wat)
    ///
    /// # Returns
    ///
    /// A new WebAssembly foreign object with 1-second timeout
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or if WebAssembly compilation fails
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self, PecosError> {
        Self::with_timeout(path, DEFAULT_TIMEOUT_SECONDS)
    }

    /// Create a new WebAssembly foreign object from a file with custom timeout
    ///
    /// # Parameters
    ///
    /// * `path` - Path to the WebAssembly file (.wasm or .wat)
    /// * `timeout_seconds` - Timeout in seconds for WASM execution
    ///
    /// # Returns
    ///
    /// A new WebAssembly foreign object
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or if WebAssembly compilation fails
    pub fn with_timeout<P: AsRef<Path>>(path: P, timeout_seconds: f64) -> Result<Self, PecosError> {
        Self::with_limits(path, timeout_seconds, None)
    }

    /// Create a new WebAssembly foreign object from a file with custom limits
    ///
    /// # Parameters
    ///
    /// * `path` - Path to the WebAssembly file (.wasm or .wat)
    /// * `timeout_seconds` - Timeout in seconds for WASM execution
    /// * `memory_size` - Optional maximum memory size in bytes per linear memory (None = unlimited)
    ///
    /// # Returns
    ///
    /// A new WebAssembly foreign object
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or if WebAssembly compilation fails
    pub fn with_limits<P: AsRef<Path>>(
        path: P,
        timeout_seconds: f64,
        memory_size: Option<usize>,
    ) -> Result<Self, PecosError> {
        // Read the WebAssembly file
        let wasm_bytes = std::fs::read(path)
            .map_err(|e| PecosError::Input(format!("Failed to read WebAssembly file: {e}")))?;

        Self::from_bytes_with_limits(&wasm_bytes, timeout_seconds, memory_size)
    }

    /// Create a new WebAssembly foreign object from bytes with default timeout
    ///
    /// # Parameters
    ///
    /// * `wasm_bytes` - WebAssembly binary
    ///
    /// # Returns
    ///
    /// A new WebAssembly foreign object with 1-second timeout
    ///
    /// # Errors
    ///
    /// Returns an error if WebAssembly compilation fails
    pub fn from_bytes(wasm_bytes: &[u8]) -> Result<Self, PecosError> {
        Self::from_bytes_with_limits(wasm_bytes, DEFAULT_TIMEOUT_SECONDS, None)
    }

    /// Create a new WebAssembly foreign object from bytes with custom timeout
    ///
    /// # Parameters
    ///
    /// * `wasm_bytes` - WebAssembly binary
    /// * `timeout_seconds` - Timeout in seconds for WASM execution
    ///
    /// # Returns
    ///
    /// A new WebAssembly foreign object
    ///
    /// # Errors
    ///
    /// Returns an error if WebAssembly compilation fails
    pub fn from_bytes_with_timeout(
        wasm_bytes: &[u8],
        timeout_seconds: f64,
    ) -> Result<Self, PecosError> {
        Self::from_bytes_with_limits(wasm_bytes, timeout_seconds, None)
    }

    /// Create a new WebAssembly foreign object from bytes with custom limits
    ///
    /// # Parameters
    ///
    /// * `wasm_bytes` - WebAssembly binary
    /// * `timeout_seconds` - Timeout in seconds for WASM execution
    /// * `memory_size` - Optional maximum memory size in bytes per linear memory (None = unlimited)
    ///
    /// # Returns
    ///
    /// A new WebAssembly foreign object
    ///
    /// # Errors
    ///
    /// Returns an error if WebAssembly compilation fails
    pub fn from_bytes_with_limits(
        wasm_bytes: &[u8],
        timeout_seconds: f64,
        memory_size: Option<usize>,
    ) -> Result<Self, PecosError> {
        // Create a new WebAssembly engine with epoch interruption enabled
        let mut config = Config::new();
        config.epoch_interruption(true);
        let engine = Engine::new(&config).map_err(|e| {
            PecosError::Processing(format!("Failed to create WebAssembly engine: {e}"))
        })?;

        // Create a new store with resource limits
        let store_context = StoreContext::new(memory_size);
        let mut store = Store::new(&engine, store_context);

        // Set the resource limiter
        store.limiter(|ctx| &mut ctx.limits);

        // Compile the WebAssembly module
        let module = Module::new(&engine, wasm_bytes).map_err(|e| {
            PecosError::Processing(format!("Failed to compile WebAssembly module: {e}"))
        })?;

        let stop_flag = Arc::new(RwLock::new(false));
        let engine_clone = engine.clone();
        let stop_flag_clone = stop_flag.clone();

        // Start the epoch increment thread for timeout handling
        thread::spawn(move || {
            while !*stop_flag_clone.read() {
                // Increment the epoch every tick length
                engine_clone.increment_epoch();
                thread::sleep(Duration::from_millis(WASM_EXECUTION_TICK_LENGTH_MS));
            }
        });

        let mut foreign_object = Self {
            wasm_bytes: wasm_bytes.to_vec(),
            engine,
            module,
            store: RwLock::new(store),
            instance: RwLock::new(None),
            func_names: Mutex::new(None),
            stop_flag,
            last_results: Vec::new(),
            timeout_seconds,
            memory_size,
        };

        // Create the instance
        foreign_object.new_instance()?;

        Ok(foreign_object)
    }

    /// Get a function from the WebAssembly instance
    ///
    /// # Parameters
    ///
    /// * `func_name` - Name of the function to get
    ///
    /// # Returns
    ///
    /// The WebAssembly function
    ///
    /// # Errors
    ///
    /// Returns an error if the function is not found
    fn get_function(&self, func_name: &str) -> Result<Func, PecosError> {
        // Get the instance
        let instance = self.instance.read();
        let instance = instance
            .as_ref()
            .ok_or_else(|| PecosError::Resource("WebAssembly instance not created".to_string()))?;

        // Get the function
        let mut store = self.store.write();
        let func = instance.get_func(&mut *store, func_name).ok_or_else(|| {
            PecosError::Resource(format!("WebAssembly function '{func_name}' not found"))
        })?;

        Ok(func)
    }

    /// Call before each shot to reset variables (if function exists)
    ///
    /// This is a convenience method that calls the `shot_reinit` function in the
    /// WebAssembly module if it exists. If the function doesn't exist, this is a no-op.
    ///
    /// # Errors
    ///
    /// Returns an error if the `shot_reinit` function exists but execution fails
    pub fn shot_reinit(&mut self) -> Result<(), PecosError> {
        let funcs = self.get_funcs();
        if funcs.contains(&"shot_reinit".to_string()) {
            self.exec("shot_reinit", &[])?;
        }
        Ok(())
    }

    /// Get the WebAssembly binary bytes
    ///
    /// This is useful for serialization and cloning.
    ///
    /// # Returns
    ///
    /// A reference to the WebAssembly binary bytes
    #[must_use]
    pub fn wasm_bytes(&self) -> &[u8] {
        &self.wasm_bytes
    }

    /// Get the configured timeout in seconds
    ///
    /// # Returns
    ///
    /// The timeout in seconds for WASM execution
    #[must_use]
    pub fn timeout_seconds(&self) -> f64 {
        self.timeout_seconds
    }

    /// Get the configured memory size limit
    ///
    /// # Returns
    ///
    /// The memory size limit in bytes per linear memory (None = unlimited)
    #[must_use]
    pub fn memory_size(&self) -> Option<usize> {
        self.memory_size
    }
}

impl ForeignObject for WasmForeignObject {
    fn clone_box(&self) -> Box<dyn ForeignObject> {
        // Create a new instance from the same bytes with the same timeout and memory limit
        let mut result =
            Self::from_bytes_with_limits(&self.wasm_bytes, self.timeout_seconds, self.memory_size)
                .expect("Failed to clone WasmForeignObject");

        // Initialize it the same way
        if self.instance.read().is_some() {
            let _ = result.new_instance();
        }

        Box::new(result)
    }

    fn init(&mut self) -> Result<(), PecosError> {
        // Create a new instance
        self.new_instance()?;

        // Check if the init function exists
        let funcs = self.get_funcs();
        if !funcs.contains(&"init".to_string()) {
            return Err(PecosError::Input(
                "WebAssembly module must contain an 'init' function".to_string(),
            ));
        }

        // Call the init function
        self.exec("init", &[])?;

        Ok(())
    }

    fn new_instance(&mut self) -> Result<(), PecosError> {
        let mut store = self.store.write();

        // Create a new instance
        let instance = Instance::new(&mut *store, &self.module, &[]).map_err(|e| {
            PecosError::Processing(format!("Failed to create WebAssembly instance: {e}"))
        })?;

        // Store the instance
        *self.instance.write() = Some(instance);

        Ok(())
    }

    fn get_funcs(&self) -> Vec<String> {
        // Check if we've already cached the function names
        if let Some(ref funcs) = *self.func_names.lock() {
            return funcs.clone();
        }

        // Get the function names
        let mut funcs = Vec::new();
        for export in self.module.exports() {
            if export.ty().func().is_some() {
                funcs.push(export.name().to_string());
            }
        }

        // Cache the function names
        *self.func_names.lock() = Some(funcs.clone());

        funcs
    }

    fn exec(&mut self, func_name: &str, args: &[i64]) -> Result<Vec<i64>, PecosError> {
        debug!("Executing WebAssembly function '{func_name}' with args {args:?}");

        // Get the function
        let func = self.get_function(func_name)?;

        // Get store early to check function signature
        let mut store = self.store.write();
        let func_type = func.ty(&*store);
        let param_types: Vec<_> = func_type.params().collect();

        // Convert the arguments based on function signature
        let wasm_args: Vec<_> = args
            .iter()
            .enumerate()
            .map(|(i, a)| {
                // Get the expected parameter type (or default to i32)
                let param_type = param_types.get(i);

                match param_type {
                    Some(wasmtime::ValType::I64) => {
                        // Function expects i64, pass directly
                        wasmtime::Val::I64(*a)
                    }
                    Some(wasmtime::ValType::I32) | None => {
                        // Function expects i32 or unknown, convert with bounds checking
                        let value = if *a > i64::from(i32::MAX) {
                            warn!("Argument value {a} exceeds i32::MAX, clamping to i32::MAX");
                            i32::MAX
                        } else if *a < i64::from(i32::MIN) {
                            warn!("Argument value {a} is less than i32::MIN, clamping to i32::MIN");
                            i32::MIN
                        } else {
                            // Safe: we've verified the value is in range
                            i32::try_from(*a).expect("Value should be in range after bounds check")
                        };
                        wasmtime::Val::I32(value)
                    }
                    _ => {
                        // Unsupported parameter type, default to i32
                        warn!("Unexpected parameter type for argument {i}, defaulting to i32");
                        let value = i32::try_from(*a).unwrap_or(i32::MAX);
                        wasmtime::Val::I32(value)
                    }
                }
            })
            .collect();

        // Set execution deadline based on configured timeout
        let max_ticks = Self::calculate_max_ticks(self.timeout_seconds);
        store.set_epoch_deadline(max_ticks);

        // Get the number of results
        let results_len = func_type.results().len();

        // Handle functions based on their return type
        let result = if results_len == 0 {
            // Function returns nothing (like init)
            func.call(&mut *store, &wasm_args, &mut [])
        } else {
            // Function returns something, create an appropriate buffer
            let mut results_buffer = vec![Val::I32(0); results_len];
            debug!(
                "Calling WebAssembly function '{func_name}' with args {wasm_args:?}, expecting {results_len} results"
            );
            let res = func.call(&mut *store, &wasm_args, &mut results_buffer);

            // Store the results if successful
            if res.is_ok() {
                debug!("WebAssembly function returned {results_buffer:?}");
                self.last_results = results_buffer;
            }

            res
        };

        // Handle the result
        match result {
            Ok(()) => {
                if results_len == 0 {
                    // Functions with no return value
                    Ok(vec![0])
                } else {
                    // Convert the results back to i64
                    let results: Vec<i64> = self
                        .last_results
                        .iter()
                        .map(|r| match r {
                            Val::I32(val) => i64::from(*val),
                            Val::I64(val) => *val,
                            _ => {
                                warn!("Unexpected result type from WebAssembly function");
                                0
                            }
                        })
                        .collect();

                    if results.is_empty() {
                        // If there are no results, return a zero
                        Ok(vec![0])
                    } else {
                        Ok(results)
                    }
                }
            }
            Err(e) => {
                // Check if the error is a timeout
                if let Some(trap) = e.downcast_ref::<Trap>()
                    && trap.to_string().contains("interrupt")
                {
                    return Err(PecosError::Processing(format!(
                        "WebAssembly function '{func_name}' timed out after {}s",
                        self.timeout_seconds
                    )));
                }

                Err(PecosError::Processing(format!(
                    "WebAssembly function '{func_name}' failed with error: {e}"
                )))
            }
        }
    }

    fn teardown(&mut self) {
        // Set the stop flag to stop the epoch increment thread
        *self.stop_flag.write() = true;
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl Drop for WasmForeignObject {
    fn drop(&mut self) {
        // Set the stop flag to stop the epoch increment thread
        *self.stop_flag.write() = true;
    }
}
