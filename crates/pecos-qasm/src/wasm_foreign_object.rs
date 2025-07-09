//! WebAssembly Foreign Object Implementation
//!
//! This module provides WebAssembly support for QASM simulations, allowing you to call
//! WASM functions from within QASM programs.
//!
//! # Example
//!
//! ## QASM Usage
//!
//! ```text
//! OPENQASM 2.0;
//! creg a[10];
//! creg b[10];
//! creg result[10];
//!
//! a = 5;
//! b = 3;
//! result = add(a, b);      // Call WASM function
//! void_func(a, b);         // Call void WASM function
//! a = get_value();         // Call WASM function with no args
//! ```
//!
//! ## Rust Usage
//!
//! ```no_run
//! # #[cfg(feature = "wasm")] {
//! use pecos_qasm::simulation::qasm_sim;
//!
//! let qasm = r#"
//!     OPENQASM 2.0;
//!     creg a[10];
//!     creg b[10];
//!     creg result[10];
//!
//!     a = 5;
//!     b = 3;
//!     result = add(a, b);
//! "#;
//!
//! // Run simulation with WASM module
//! let results = qasm_sim(qasm)
//!     .wasm("math.wasm")
//!     .run(100)
//!     .expect("Failed to run simulation");
//!
//! // Process results
//! for shot in &results.shots {
//!     let result_value = shot.data.get("result").unwrap();
//!     println!("Result: {:?}", result_value);
//! }
//! # }
//! ```
//!
//! # Requirements
//!
//! - WASM modules must export an `init()` function that is called at the start of each shot
//! - Functions can accept i32/i64 parameters and return i32/i64 values
//! - Built-in functions (sin, cos, tan, exp, ln, sqrt) cannot be overridden
//!
//! # Build-time Validation
//!
//! All function calls are validated at build time to ensure they exist in the WASM module.
//! This eliminates runtime errors for missing functions.

#[cfg(feature = "wasm")]
use crate::foreign_objects::ForeignObject;
#[cfg(feature = "wasm")]
use log::debug;
#[cfg(feature = "wasm")]
use pecos_core::errors::PecosError;
#[cfg(feature = "wasm")]
use std::collections::BTreeMap;
#[cfg(feature = "wasm")]
use std::path::Path;
#[cfg(feature = "wasm")]
use wasmtime::{Engine, Func, Instance, Module, Store, Val};

/// WebAssembly foreign object implementation for executing WebAssembly functions
///
/// Note: This implementation assumes that all function validation has been done
/// at build time. Function lookups use `expect()` instead of error handling
/// because we've already validated that:
/// 1. The WASM module exports an 'init' function
/// 2. All functions called from QASM exist in the WASM module
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
    store: Store<()>,
    /// Wasmtime instance
    instance: Option<Instance>,
    /// Cached function references
    function_cache: BTreeMap<String, Func>,
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
        let engine = Engine::default();

        // Create a new store
        let store = Store::new(&engine, ());

        // Compile the WebAssembly module
        let module = Module::new(&engine, wasm_bytes).map_err(|e| {
            PecosError::Processing(format!("Failed to compile WebAssembly module: {e}"))
        })?;

        Ok(Self {
            wasm_bytes: wasm_bytes.to_vec(),
            engine,
            module,
            store,
            instance: None,
            function_cache: BTreeMap::new(),
        })
    }

    /// Get the list of exported function names from the module
    #[must_use]
    pub fn get_exported_functions(&self) -> Vec<String> {
        let mut functions = Vec::new();
        for export in self.module.exports() {
            if matches!(export.ty(), wasmtime::ExternType::Func(_)) {
                functions.push(export.name().to_string());
            }
        }
        functions
    }

    /// Convert i64 to i32 with bounds checking
    fn i64_to_i32(value: i64) -> Result<i32, PecosError> {
        if value > i64::from(i32::MAX) || value < i64::from(i32::MIN) {
            Err(PecosError::Input(format!(
                "Value {value} is out of range for i32"
            )))
        } else {
            #[allow(clippy::cast_possible_truncation)]
            Ok(value as i32)
        }
    }
}

#[cfg(feature = "wasm")]
impl ForeignObject for WasmtimeForeignObject {
    fn clone_box(&self) -> Box<dyn ForeignObject> {
        Box::new(Self {
            wasm_bytes: self.wasm_bytes.clone(),
            engine: self.engine.clone(),
            module: self.module.clone(),
            store: Store::new(&self.engine, ()),
            instance: None,
            function_cache: BTreeMap::new(),
        })
    }

    fn init(&mut self) -> Result<(), PecosError> {
        // Create a new instance
        let instance = Instance::new(&mut self.store, &self.module, &[])
            .map_err(|e| PecosError::Processing(format!("WASM instantiation failed: {e}")))?;

        // Get the init function (we already validated it exists at build time)
        let init_func = instance
            .get_func(&mut self.store, "init")
            .expect("init function should exist (validated at build time)");

        self.instance = Some(instance);

        // Clear the function cache (will be populated on first use)
        self.function_cache.clear();

        // Call init
        match init_func.call(&mut self.store, &[], &mut []) {
            Ok(()) => {
                debug!("WebAssembly init function called successfully");
                Ok(())
            }
            Err(e) => Err(PecosError::Processing(format!(
                "WebAssembly function 'init' failed: {e}"
            ))),
        }
    }

    fn new_instance(&mut self) -> Result<(), PecosError> {
        // For QASM, we'll call init() at the start of each shot
        // If no instance exists yet, do full initialization
        if self.instance.is_none() {
            return self.init();
        }

        // Otherwise just call the init function to reset state
        let instance = self.instance.as_ref().expect("instance should exist");
        let init_func = instance
            .get_func(&mut self.store, "init")
            .expect("init function should exist (validated at build time)");

        init_func
            .call(&mut self.store, &[], &mut [])
            .map_err(|e| PecosError::Processing(format!("WebAssembly function 'init' failed: {e}")))
    }

    fn exec(&mut self, func_name: &str, args: &[i64]) -> Result<Vec<i64>, PecosError> {
        let instance = self.instance.as_ref().ok_or_else(|| {
            PecosError::Processing("WebAssembly instance not initialized".to_string())
        })?;

        // Get the function from cache or fetch and cache it
        let func = if let Some(cached_func) = self.function_cache.get(func_name) {
            *cached_func
        } else {
            // Get the function (we already validated it exists at build time)
            let func = instance
                .get_func(&mut self.store, func_name)
                .unwrap_or_else(|| {
                    panic!("Function '{func_name}' should exist (validated at build time)")
                });
            self.function_cache.insert(func_name.to_string(), func);
            func
        };

        // Get function type
        let func_ty = func.ty(&self.store);
        let params = func_ty.params();
        let results = func_ty.results();

        // Check parameter count
        if params.len() != args.len() {
            return Err(PecosError::Processing(format!(
                "Function '{func_name}' expects {} arguments, got {}",
                params.len(),
                args.len()
            )));
        }

        // Convert arguments
        let mut wasm_args = Vec::new();
        for (i, (param_ty, &arg)) in params.zip(args.iter()).enumerate() {
            match param_ty {
                wasmtime::ValType::I32 => {
                    let val = Self::i64_to_i32(arg)?;
                    wasm_args.push(Val::I32(val));
                }
                wasmtime::ValType::I64 => {
                    wasm_args.push(Val::I64(arg));
                }
                _ => {
                    return Err(PecosError::Processing(format!(
                        "Unsupported parameter type for argument {i} of function '{func_name}'"
                    )));
                }
            }
        }

        // Prepare result buffer
        let mut wasm_results = vec![Val::I32(0); results.len()];

        // Call the function
        match func.call(&mut self.store, &wasm_args, &mut wasm_results) {
            Ok(()) => {
                // Convert results to i64
                let mut results_i64 = Vec::new();
                for (i, val) in wasm_results.iter().enumerate() {
                    match val {
                        Val::I32(v) => results_i64.push(i64::from(*v)),
                        Val::I64(v) => results_i64.push(*v),
                        _ => {
                            return Err(PecosError::Processing(format!(
                                "Unsupported return type for result {i} of function '{func_name}'"
                            )));
                        }
                    }
                }
                Ok(results_i64)
            }
            Err(e) => Err(PecosError::Processing(format!(
                "WebAssembly function '{func_name}' failed: {e}"
            ))),
        }
    }
}
