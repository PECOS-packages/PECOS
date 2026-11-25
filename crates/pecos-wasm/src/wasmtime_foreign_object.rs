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

//! Wasmtime-based WebAssembly foreign object implementation.
//!
//! Uses the Wasmtime runtime with timeout handling and type conversions.

use crate::foreign_object::ForeignObject;
use log::{debug, warn};
use parking_lot::RwLock;
use pecos_core::errors::PecosError;
use std::any::Any;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread;
use std::time::Duration;
use wasmtime::{
    Config, Engine, Func, Instance, Module, Store, StoreLimits, StoreLimitsBuilder, Trap, Val,
};

/// Scalar types supported by the strict Guppy WASM interface.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WasmScalarType {
    /// Signed 64-bit integer.
    I64,
    /// IEEE 754 double precision, transported without changing its bits.
    F64,
    /// Signed 32-bit integer in a Guppy 64-bit transport slot. Arguments wrap to their
    /// low 32 bits and results are sign-extended into the 64-bit transport slot.
    I32,
    /// Select i32 or i64 from the actual export signature for each integer lane.
    IntegerAuto,
}

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

/// Immutable compiled WebAssembly code that can create independent instances.
///
/// Clones share compiled code and the epoch clock, never stores or mutable WASM
/// state. Retaining this value avoids recompilation between shots.
#[derive(Clone, Debug)]
pub struct CompiledWasmModule {
    bytes: Arc<[u8]>,
    engine: Engine,
    module: Module,
    clock: Arc<EpochClock>,
}

#[derive(Debug)]
struct EpochClock {
    stop: Arc<AtomicBool>,
}

impl Drop for EpochClock {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

impl CompiledWasmModule {
    /// Compile bytes without executing the module's start function or init export.
    ///
    /// # Errors
    /// Returns an error if engine creation or compilation fails.
    pub fn new(bytes: &[u8]) -> Result<Self, PecosError> {
        let mut config = Config::new();
        config.epoch_interruption(true);
        let engine = Engine::new(&config).map_err(|e| {
            PecosError::Processing(format!("Failed to create WebAssembly engine: {e}"))
        })?;
        let module = Module::new(&engine, bytes).map_err(|e| {
            PecosError::Processing(format!("Failed to compile WebAssembly module: {e}"))
        })?;
        let stop = Arc::new(AtomicBool::new(false));
        let thread_stop = stop.clone();
        let thread_engine = engine.clone();
        // Capture neither the artifact nor its clock: either would form a cycle.
        thread::spawn(move || {
            while !thread_stop.load(Ordering::Relaxed) {
                thread_engine.increment_epoch();
                thread::sleep(Duration::from_millis(WASM_EXECUTION_TICK_LENGTH_MS));
            }
        });
        Ok(Self {
            bytes: Arc::from(bytes),
            engine,
            module,
            clock: Arc::new(EpochClock { stop }),
        })
    }

    /// Original input bytes, used for exact cache identity.
    #[must_use]
    pub fn wasm_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Create fresh mutable state with the default execution limits.
    ///
    /// # Errors
    /// Returns an error if instantiation or the start function fails.
    pub fn instantiate(&self) -> Result<WasmForeignObject, PecosError> {
        self.instantiate_with_limits(DEFAULT_TIMEOUT_SECONDS, None)
    }

    /// Create a fresh store and instance with independent resource limits.
    ///
    /// # Errors
    /// Returns an error if instantiation or the start function fails.
    pub fn instantiate_with_limits(
        &self,
        timeout_seconds: f64,
        memory_size: Option<usize>,
    ) -> Result<WasmForeignObject, PecosError> {
        let mut store = Store::new(&self.engine, StoreContext::new(memory_size));
        store.limiter(|ctx| &mut ctx.limits);
        let mut object = WasmForeignObject {
            wasm_bytes: self.bytes.clone(),
            module: self.module.clone(),
            store: RwLock::new(store),
            instance: RwLock::new(None),
            func_names: OnceLock::new(),
            clock: Some(self.clock.clone()),
            last_results: Vec::new(),
            timeout_seconds,
            memory_size,
        };
        object.new_instance()?;
        Ok(object)
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
    wasm_bytes: Arc<[u8]>,
    /// Wasmtime module
    module: Module,
    /// Wasmtime store (thread-safe)
    store: RwLock<Store<StoreContext>>,
    /// Wasmtime instance (thread-safe)
    instance: RwLock<Option<Instance>>,
    /// Cached function names (initialized once, lock-free reads)
    func_names: OnceLock<Arc<[String]>>,
    /// A live instance retains its shared epoch clock until teardown or drop.
    clock: Option<Arc<EpochClock>>,
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
        // Round positive limits up so a sub-tick timeout never becomes an
        // already-expired deadline. Leave headroom for Wasmtime's epoch + delta.
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss
        )]
        let ticks = (timeout_seconds.max(0.0) * 1000.0 / WASM_EXECUTION_TICK_LENGTH_MS as f64)
            .ceil()
            .min((u64::MAX / 10) as f64) as u64;
        ticks
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
        CompiledWasmModule::new(wasm_bytes)?.instantiate_with_limits(timeout_seconds, memory_size)
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

    /// Validate the scalar i64 ABI used by public Guppy WASM calls.
    ///
    /// # Errors
    /// Returns an error for a missing export, differing arity, or non-i64 types.
    pub fn validate_i64_signature(
        &self,
        name: &str,
        inputs: usize,
        outputs: usize,
    ) -> Result<(), PecosError> {
        let Some(wasmtime::ExternType::Func(signature)) = self.module.get_export(name) else {
            return Err(PecosError::Input(format!(
                "Missing WASM function export: {name}"
            )));
        };
        if signature.params().len() != inputs
            || signature.results().len() != outputs
            || !signature
                .params()
                .chain(signature.results())
                .all(|ty| matches!(ty, wasmtime::ValType::I64))
        {
            return Err(PecosError::Input(format!(
                "WASM i64 signature mismatch for {name}"
            )));
        }
        Ok(())
    }

    /// Execute a strictly typed scalar call using one raw 64-bit slot per value.
    ///
    /// Float slots contain IEEE 754 bits, not integer conversions. This keeps
    /// the Guppy interface separate from the legacy integer/clamping API.
    ///
    /// # Errors
    /// Rejects missing exports, argument/signature mismatches, traps and timeouts.
    pub fn exec_scalar_bits(
        &mut self,
        name: &str,
        inputs: &[WasmScalarType],
        outputs: &[WasmScalarType],
        arguments: &[i64],
    ) -> Result<Vec<i64>, PecosError> {
        let func = self.get_function(name)?;
        let mut store = self.store.write();
        let signature = func.ty(&*store);
        let matches = |expected: &WasmScalarType, actual: wasmtime::ValType| {
            matches!(
                (expected, actual),
                (WasmScalarType::I64, wasmtime::ValType::I64)
                    | (WasmScalarType::I32, wasmtime::ValType::I32)
                    | (WasmScalarType::F64, wasmtime::ValType::F64)
                    | (
                        WasmScalarType::IntegerAuto,
                        wasmtime::ValType::I32 | wasmtime::ValType::I64
                    )
            )
        };
        if arguments.len() != inputs.len()
            || signature.params().len() != inputs.len()
            || signature.results().len() != outputs.len()
            || !inputs
                .iter()
                .zip(signature.params())
                .all(|(a, b)| matches(a, b))
            || !outputs
                .iter()
                .zip(signature.results())
                .all(|(a, b)| matches(a, b))
        {
            return Err(PecosError::Input(format!(
                "WASM scalar signature mismatch for {name}"
            )));
        }
        // The complete signature is validated before any execution. IntegerAuto
        // resolves each lane independently; float lanes remain exact.
        let values: Vec<_> = signature
            .params()
            .zip(arguments)
            .map(|(ty, bits)| match ty {
                wasmtime::ValType::I64 => Ok(Val::I64(*bits)),
                wasmtime::ValType::I32 => {
                    let [a, b, c, d, ..] = bits.to_le_bytes();
                    Ok(Val::I32(i32::from_le_bytes([a, b, c, d])))
                }
                wasmtime::ValType::F64 => Ok(Val::F64(u64::from_ne_bytes(bits.to_ne_bytes()))),
                _ => Err(PecosError::Input(
                    "Unsupported WASM scalar argument type".into(),
                )),
            })
            .collect::<Result<_, _>>()?;
        let mut results = vec![Val::I64(0); outputs.len()];
        store.set_epoch_deadline(Self::calculate_max_ticks(self.timeout_seconds));
        func.call(&mut *store, &values, &mut results)
            .map_err(|error| self.execution_error(name, &error))?;
        results
            .into_iter()
            .map(|value| match value {
                Val::I32(bits) => Ok(i64::from(bits)),
                Val::I64(bits) => Ok(bits),
                Val::F64(bits) => Ok(i64::from_ne_bytes(bits.to_ne_bytes())),
                _ => Err(PecosError::Input(
                    "Unexpected WASM scalar result type".into(),
                )),
            })
            .collect()
    }

    fn execution_error(&self, name: &str, error: &wasmtime::Error) -> PecosError {
        if let Some(trap) = error.downcast_ref::<Trap>() {
            let trap_str = trap.to_string();
            if trap_str.contains("interrupt") {
                return PecosError::Processing(format!(
                    "WebAssembly execution timed out after {}s (possible infinite loop in '{name}')",
                    self.timeout_seconds
                ));
            }
            if trap_str.contains("integer divide by zero") {
                return PecosError::RuntimeDivisionByZero;
            }
            return PecosError::Generic(format!("WebAssembly function '{name}' trapped: {trap}"));
        }
        PecosError::Generic(format!("WebAssembly function '{name}' failed: {error}"))
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
        if self.has_func("shot_reinit") {
            self.exec("shot_reinit", &[])?;
        }
        Ok(())
    }

    /// Check if a function exists in the WebAssembly module (zero-allocation)
    ///
    /// This is more efficient than `get_funcs().contains()` for existence checks
    /// as it doesn't allocate or clone the function list.
    ///
    /// # Parameters
    ///
    /// * `func_name` - Name of the function to check
    ///
    /// # Returns
    ///
    /// `true` if the function exists, `false` otherwise
    pub fn has_func(&self, func_name: &str) -> bool {
        self.module
            .exports()
            .any(|e| e.name() == func_name && e.ty().func().is_some())
    }

    /// Resolve a zero-based export-list position to a function name.
    ///
    /// All exports occupy positions, including memories, tables and globals.
    /// This is the order used by Guppy's `@wasm(index)`, not the WebAssembly
    /// internal function-index space.
    ///
    /// # Errors
    /// Rejects out-of-range positions and exports that are not functions.
    pub fn exported_function_name(&self, index: u64) -> Result<String, PecosError> {
        let export = usize::try_from(index)
            .ok()
            .and_then(|index| self.module.exports().nth(index))
            .ok_or_else(|| {
                PecosError::Input(format!("WASM export index {index} is out of range"))
            })?;
        if export.ty().func().is_none() {
            return Err(PecosError::Input(format!(
                "WASM export index {index} is not a function ({})",
                export.name()
            )));
        }
        Ok(export.name().to_owned())
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
        if !self.has_func("init") {
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

        // Instantiation can execute enough Wasm work to observe an epoch
        // interrupt. Arm the deadline before Instance::new, just as exec does
        // before calling an exported function.
        let max_ticks = Self::calculate_max_ticks(self.timeout_seconds);
        store.set_epoch_deadline(max_ticks);

        // Create a new instance
        let instance = Instance::new(&mut *store, &self.module, &[]).map_err(|e| {
            PecosError::Processing(format!("Failed to create WebAssembly instance: {e}"))
        })?;

        // Store the instance
        *self.instance.write() = Some(instance);

        Ok(())
    }

    fn get_funcs(&self) -> Vec<String> {
        // Use OnceLock::get_or_init for lock-free cached access
        // The Arc is cloned (cheap), then converted to Vec for trait compatibility
        let arc_funcs = self.func_names.get_or_init(|| {
            // Collect function names and convert to Arc<[String]>
            let funcs: Vec<String> = self
                .module
                .exports()
                .filter_map(|export| {
                    if export.ty().func().is_some() {
                        Some(export.name().to_string())
                    } else {
                        None
                    }
                })
                .collect();
            Arc::from(funcs)
        });

        // Convert Arc<[String]> to Vec<String> for trait compatibility
        // This clones the strings but Arc::clone is cheap
        arc_funcs.to_vec()
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
            Err(error) => Err(self.execution_error(func_name, &error)),
        }
    }

    fn teardown(&mut self) {
        // Other objects and cached compiled code may still need this clock.
        self.clock.take();
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::WasmForeignObject;

    #[test]
    fn managed_roots_survive_collection_and_stores_are_independent() {
        use crate::ForeignObject;
        let module = super::CompiledWasmModule::new(
            br#"(module
            (type $node (struct (field (mut i64)) (field (mut (ref null $node)))))
            (global $live (mut (ref null $node)) (ref.null $node))
            (func (export "init")
                i64.const 42 ref.null $node struct.new $node global.set $live)
            (func (export "cycle") (local $node (ref null $node))
                i64.const 9 ref.null $node struct.new $node local.tee $node
                local.get $node struct.set $node 1)
            (func (export "get") (result i64) global.get $live struct.get $node 0)
            (func (export "increment")
                global.get $live global.get $live struct.get $node 0
                i64.const 1 i64.add struct.set $node 0)
            (func (export "reference") (result (ref null $node)) global.get $live)
            (func (export "consume") (param (ref null $node))))"#,
        )
        .unwrap();
        let mut first = module.instantiate().unwrap();
        let mut second = module.instantiate().unwrap();
        first.exec("init", &[]).unwrap();
        second.exec("init", &[]).unwrap();
        for expected in 43..51 {
            first.exec("increment", &[]).unwrap();
            for _ in 0..1000 {
                first.exec("cycle", &[]).unwrap();
            }
            // Explicitly invoke collection: allocation alone does not prove
            // that a moving collector preserves live module roots.
            first.store.write().gc(None).unwrap();
            assert_eq!(first.exec("get", &[]).unwrap(), [expected]);
            assert_eq!(second.exec("get", &[]).unwrap(), [42]);
        }
        assert!(first.validate_i64_signature("reference", 0, 1).is_err());
        assert!(first.validate_i64_signature("consume", 1, 0).is_err());
        first.teardown();
        second.store.write().gc(None).unwrap();
        assert_eq!(second.exec("get", &[]).unwrap(), [42]);
    }

    #[test]
    fn positive_deadlines_round_up_and_leave_epoch_headroom() {
        for (seconds, expected) in [
            (f64::from_bits(1), 1),
            (0.001, 1),
            (0.01, 1),
            (0.011, 2),
            (1.0, 100),
        ] {
            assert_eq!(WasmForeignObject::calculate_max_ticks(seconds), expected);
        }
        let largest = WasmForeignObject::calculate_max_ticks(f64::MAX);
        assert!(largest > 0 && largest < u64::MAX / 2);
        assert_eq!(WasmForeignObject::calculate_max_ticks(0.0), 0);
    }
}
