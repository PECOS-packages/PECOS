// Copyright 2025 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License.You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! PyO3 bindings for WebAssembly foreign object
//!
//! This module provides Python bindings for the Rust WasmForeignObject implementation,
//! allowing Python code to use the Rust Wasmtime runtime instead of the Python wasmtime package.

use pecos_wasm::{ForeignObject, WasmForeignObject};
use pyo3::exceptions::{PyException, PyFileNotFoundError, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::path::Path;

/// Python wrapper for WasmForeignObject
///
/// This class provides the same interface as the Python WasmtimeObj class,
/// but uses the Rust implementation under the hood for better performance
/// and thread safety.
#[pyclass(name = "RsWasmForeignObject")]
pub struct PyWasmForeignObject {
    inner: WasmForeignObject,
}

#[pymethods]
impl PyWasmForeignObject {
    /// Create a new WebAssembly foreign object
    ///
    /// Args:
    ///     file: Path to WASM file (str) or WASM bytes (bytes)
    ///
    /// Returns:
    ///     New WebAssembly foreign object instance
    ///
    /// Raises:
    ///     FileNotFoundError: If file path doesn't exist
    ///     RuntimeError: If WASM compilation fails
    #[new]
    fn new(_py: Python<'_>, file: &Bound<'_, PyAny>) -> PyResult<Self> {
        // Try to extract as bytes first
        if let Ok(bytes) = file.downcast::<PyBytes>() {
            let wasm_bytes = bytes.as_bytes();
            let inner = WasmForeignObject::from_bytes(wasm_bytes).map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to load WASM from bytes: {e}"))
            })?;
            return Ok(Self { inner });
        }

        // Try to extract as string path
        if let Ok(path_str) = file.extract::<String>() {
            let path = Path::new(&path_str);
            if !path.exists() {
                return Err(PyFileNotFoundError::new_err(format!(
                    "WASM file not found: {path_str}"
                )));
            }

            let inner = WasmForeignObject::new(path).map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to load WASM from file: {e}"))
            })?;
            return Ok(Self { inner });
        }

        // If neither worked, return error
        Err(PyException::new_err(
            "Expected str (file path) or bytes (WASM binary)",
        ))
    }

    /// Initialize the WASM module
    ///
    /// This must be called before using the object. It creates a new instance
    /// and calls the 'init' function in the WASM module.
    ///
    /// Raises:
    ///     RuntimeError: If init function is missing or execution fails
    fn init(&mut self) -> PyResult<()> {
        self.inner
            .init()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to initialize WASM: {e}")))
    }

    /// Reset variables before each shot
    ///
    /// Calls the 'shot_reinit' function in the WASM module if it exists.
    /// This is a no-op if the function doesn't exist.
    ///
    /// Raises:
    ///     RuntimeError: If shot_reinit function exists but execution fails
    fn shot_reinit(&mut self) -> PyResult<()> {
        self.inner.shot_reinit().map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to call shot_reinit: {e}"))
        })
    }

    /// Create a new WASM instance
    ///
    /// Resets the object's internal state by creating a fresh instance.
    ///
    /// Raises:
    ///     RuntimeError: If instance creation fails
    fn new_instance(&mut self) -> PyResult<()> {
        self.inner.new_instance().map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to create new instance: {e}"))
        })
    }

    /// Get list of exported function names
    ///
    /// Returns:
    ///     List of function names exported by the WASM module
    fn get_funcs(&self) -> Vec<String> {
        self.inner.get_funcs()
    }

    /// Execute a WASM function
    ///
    /// Args:
    ///     func_name: Name of the function to execute
    ///     args: List of integer arguments (i64)
    ///
    /// Returns:
    ///     Tuple containing the function results (or single 0 for void functions)
    ///
    /// Raises:
    ///     RuntimeError: If function not found or execution fails
    fn exec(&mut self, py: Python<'_>, func_name: &str, args: Vec<i64>) -> PyResult<PyObject> {
        let results = self.inner.exec(func_name, &args).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to execute '{func_name}': {e}"))
        })?;

        // Convert Vec<i64> to Python - single value as int, multiple as tuple
        if results.len() == 1 {
            // Return single value directly (matching Python behavior)
            Ok(results[0].into_pyobject(py)?.into_any().unbind())
        } else {
            // Return tuple for multiple values
            let tuple = pyo3::types::PyTuple::new(py, results.iter())?;
            Ok(tuple.into_any().unbind())
        }
    }

    /// Cleanup resources
    ///
    /// Stops the epoch increment thread. This is called automatically
    /// when the object is dropped, but can be called explicitly.
    fn teardown(&mut self) {
        self.inner.teardown();
    }

    /// Serialize to dictionary for pickling
    ///
    /// Returns:
    ///     Dictionary containing 'fobj_class' and 'wasm_bytes'
    fn to_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = pyo3::types::PyDict::new(py);

        // Get the Python class for fobj_class
        let module = py.import("pecos_rslib")?;
        let cls = module.getattr("RsWasmForeignObject")?;
        dict.set_item("fobj_class", cls)?;

        // Get WASM bytes
        let wasm_bytes = PyBytes::new(py, self.inner.wasm_bytes());
        dict.set_item("wasm_bytes", wasm_bytes)?;

        Ok(dict.into())
    }

    /// Deserialize from dictionary (for pickling)
    ///
    /// Args:
    ///     wasmtime_dict: Dictionary containing 'fobj_class' and 'wasm_bytes'
    ///
    /// Returns:
    ///     New instance created from the dictionary
    #[staticmethod]
    fn from_dict(py: Python<'_>, wasmtime_dict: &Bound<'_, PyAny>) -> PyResult<Self> {
        use pyo3::types::PyDictMethods;
        let dict = wasmtime_dict.downcast::<pyo3::types::PyDict>()?;
        let wasm_bytes = dict
            .get_item("wasm_bytes")?
            .ok_or_else(|| PyException::new_err("Missing 'wasm_bytes' in dictionary"))?;

        Self::new(py, &wasm_bytes)
    }

    /// Support for pickle (Python serialization)
    fn __getstate__(&self, py: Python<'_>) -> PyResult<PyObject> {
        self.to_dict(py)
    }

    /// Support for pickle (Python deserialization)
    fn __setstate__(&mut self, py: Python<'_>, state: &Bound<'_, PyAny>) -> PyResult<()> {
        // Create new object and swap the inner value
        let new_obj = Self::from_dict(py, state)?;
        // Replace inner by creating a new instance from the same bytes
        let wasm_bytes = new_obj.inner.wasm_bytes();
        self.inner = WasmForeignObject::from_bytes(wasm_bytes).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to deserialize WASM object: {e}"))
        })?;
        Ok(())
    }
}

impl Drop for PyWasmForeignObject {
    fn drop(&mut self) {
        // Ensure teardown is called when the object is dropped
        self.inner.teardown();
    }
}
