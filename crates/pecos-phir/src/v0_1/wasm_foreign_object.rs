#[cfg(feature = "wasm")]
use crate::v0_1::foreign_objects::ForeignObject;
#[cfg(feature = "wasm")]
use log::{debug, warn};
#[cfg(feature = "wasm")]
use parking_lot::{Mutex, RwLock};
#[cfg(feature = "wasm")]
use pecos_core::errors::PecosError;
#[cfg(feature = "wasm")]
use std::any::Any;
#[cfg(feature = "wasm")]
use std::path::Path;
#[cfg(feature = "wasm")]
use std::sync::Arc;
#[cfg(feature = "wasm")]
use std::thread;
#[cfg(feature = "wasm")]
use std::time::Duration;
#[cfg(feature = "wasm")]
use wasmtime::{Config, Engine, Func, Instance, Module, Store, Trap, Val};

#[cfg(feature = "wasm")]
const WASM_EXECUTION_MAX_TICKS: u64 = 10_000;
#[cfg(feature = "wasm")]
const WASM_EXECUTION_TICK_LENGTH_MS: u64 = 10;

/// WebAssembly foreign object implementation for executing WebAssembly functions
#[cfg(feature = "wasm")]
#[derive(Debug)]
pub struct WasmtimeForeignObject {
    /// WebAssembly binary
    #[allow(dead_code)]
    wasm_bytes: Vec<u8>,
    /// Wasmtime engine
    #[allow(dead_code)]
    engine: Engine,
    /// Wasmtime module
    module: Module,
    /// Wasmtime store
    store: RwLock<Store<()>>,
    /// Wasmtime instance
    instance: RwLock<Option<Instance>>,
    /// Available functions
    func_names: Mutex<Option<Vec<String>>>,
    /// Timeout flag for long-running operations
    stop_flag: Arc<RwLock<bool>>,
    /// Last function call results
    last_results: Vec<Val>,
}

#[cfg(feature = "wasm")]
impl WasmtimeForeignObject {
    /// Create a new WebAssembly foreign object from a file
    ///
    /// # Parameters
    ///
    /// * `path` - Path to the WebAssembly file (.wasm or .wat)
    ///
    /// # Returns
    ///
    /// A new WebAssembly foreign object
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or if WebAssembly compilation fails
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self, PecosError> {
        // Read the WebAssembly file
        let wasm_bytes = std::fs::read(path)
            .map_err(|e| PecosError::Input(format!("Failed to read WebAssembly file: {e}")))?;

        Self::from_bytes(&wasm_bytes)
    }

    /// Create a new WebAssembly foreign object from bytes
    ///
    /// # Parameters
    ///
    /// * `wasm_bytes` - WebAssembly binary
    ///
    /// # Returns
    ///
    /// A new WebAssembly foreign object
    ///
    /// # Errors
    ///
    /// Returns an error if WebAssembly compilation fails
    pub fn from_bytes(wasm_bytes: &[u8]) -> Result<Self, PecosError> {
        // Create a new WebAssembly engine
        let mut config = Config::new();
        config.epoch_interruption(true);
        let engine = Engine::new(&config).map_err(|e| {
            PecosError::Processing(format!("Failed to create WebAssembly engine: {e}"))
        })?;

        // Create a new store
        let store = Store::new(&engine, ());

        // Compile the WebAssembly module
        let module = Module::new(&engine, wasm_bytes).map_err(|e| {
            PecosError::Processing(format!("Failed to compile WebAssembly module: {e}"))
        })?;

        let stop_flag = Arc::new(RwLock::new(false));
        let engine_clone = engine.clone();
        let stop_flag_clone = stop_flag.clone();

        // Start the epoch increment thread
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
}

#[cfg(feature = "wasm")]
impl ForeignObject for WasmtimeForeignObject {
    fn clone_box(&self) -> Box<dyn ForeignObject> {
        // Create a new instance from the same bytes
        let mut result =
            Self::from_bytes(&self.wasm_bytes).expect("Failed to clone WasmtimeForeignObject");

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

        // Convert the arguments
        let wasm_args: Vec<_> = args
            .iter()
            .map(|a| {
                // Try to convert i64 to i32 with proper bounds checking
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
            })
            .collect();

        // Execute the function
        let mut store = self.store.write();
        store.set_epoch_deadline(WASM_EXECUTION_MAX_TICKS);

        // Get the function type to determine the number of results
        let func_type = func.ty(&*store);
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
                if let Some(trap) = e.downcast_ref::<Trap>() {
                    if trap.to_string().contains("interrupt") {
                        let timeout_ms = WASM_EXECUTION_MAX_TICKS * WASM_EXECUTION_TICK_LENGTH_MS;
                        return Err(PecosError::Processing(format!(
                            "WebAssembly function '{func_name}' timed out after {timeout_ms}ms"
                        )));
                    }
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

#[cfg(feature = "wasm")]
impl Drop for WasmtimeForeignObject {
    fn drop(&mut self) {
        // Set the stop flag to stop the epoch increment thread
        *self.stop_flag.write() = true;
    }
}
