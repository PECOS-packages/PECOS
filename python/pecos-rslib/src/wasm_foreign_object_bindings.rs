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

//! `PyO3` bindings for WebAssembly foreign object
//!
//! This module provides Python bindings for the Rust `WasmForeignObject` implementation,
//! allowing Python code to use the Rust Wasmtime runtime instead of the Python wasmtime package.

use pecos::wasm::{ForeignObject, WasmForeignObject};
use pyo3::exceptions::{PyException, PyFileNotFoundError, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::path::Path;

pyo3::create_exception!(pecos_rslib, WasmError, pyo3::exceptions::PyException);

/// Python wrapper for `WasmForeignObject`
///
/// This class provides WebAssembly execution capabilities using the Rust
/// Wasmtime runtime for better performance and thread safety.
///
/// WebAssembly modules can be loaded from files (.wasm or .wat) or directly
/// from binary bytes in memory. Use the explicit classmethods for clarity:
///
/// - `WasmForeignObject.from_file()` - Load from a file path
/// - `WasmForeignObject.from_bytes()` - Load from binary bytes in memory
///
/// Example:
///
/// ```ignore
/// # Load from file
/// wasm = WasmForeignObject.from_file("math.wasm")
/// wasm.init()
/// result = wasm.exec("add", [5, 3])
///
/// # Load from bytes (e.g., downloaded or embedded)
/// with open("math.wasm", "rb") as f:
///     wasm_bytes = f.read()
/// wasm = WasmForeignObject.from_bytes(wasm_bytes)
/// wasm.init()
/// ```
#[pyclass(name = "WasmForeignObject", module = "pecos_rslib")]
pub struct PyWasmForeignObject {
    pub(crate) inner: WasmForeignObject,
}

impl PyWasmForeignObject {
    /// Get a cloned boxed `ForeignObject` for use with `HugrEngine`
    pub fn clone_boxed(&self) -> Box<dyn ForeignObject> {
        self.inner.clone_box()
    }
}

#[pymethods]
impl PyWasmForeignObject {
    /// Create a new WebAssembly foreign object
    ///
    /// This constructor accepts either a file path or raw WASM bytes.
    /// For clearer code, prefer using the explicit classmethods:
    ///
    /// - `WasmForeignObject.from_file(path)` - Load from a file
    /// - `WasmForeignObject.from_bytes(data)` - Load from bytes
    ///
    /// Args:
    ///     file: Path to WASM file (str or pathlib.Path) or WASM bytes (bytes)
    ///     timeout: Optional timeout in seconds (default: 1.0 second)
    ///     `memory_size`: Optional maximum memory size in bytes per linear memory
    ///                  (default: None = unlimited)
    ///
    /// Returns:
    ///     New WebAssembly foreign object instance
    ///
    /// Raises:
    ///     `FileNotFoundError`: If file path doesn't exist
    ///     `RuntimeError`: If WASM compilation fails
    ///
    /// Example:
    ///
    /// ```ignore
    /// # From file path
    /// wasm = WasmForeignObject("math.wasm")
    ///
    /// # From bytes
    /// wasm = WasmForeignObject(wasm_bytes)
    ///
    /// # With custom timeout (5 seconds)
    /// wasm = WasmForeignObject("math.wasm", timeout=5.0)
    /// ```
    #[new]
    #[pyo3(signature = (file, timeout=None, memory_size=None))]
    fn new(
        _py: Python<'_>,
        file: &Bound<'_, PyAny>,
        timeout: Option<f64>,
        memory_size: Option<usize>,
    ) -> PyResult<Self> {
        let timeout_seconds = timeout.unwrap_or(1.0);

        // Try to extract as bytes first
        if let Ok(bytes) = file.cast::<PyBytes>() {
            let wasm_bytes = bytes.as_bytes();
            let inner =
                WasmForeignObject::from_bytes_with_limits(wasm_bytes, timeout_seconds, memory_size)
                    .map_err(|e| {
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

            let inner = WasmForeignObject::with_limits(path, timeout_seconds, memory_size)
                .map_err(|e| {
                    PyRuntimeError::new_err(format!("Failed to load WASM from file: {e}"))
                })?;
            return Ok(Self { inner });
        }

        // Try to handle pathlib.Path objects via __fspath__ protocol
        if file.hasattr("__fspath__")? {
            let path_str = file.call_method0("__fspath__")?.extract::<String>()?;
            let path = Path::new(&path_str);
            if !path.exists() {
                return Err(PyFileNotFoundError::new_err(format!(
                    "WASM file not found: {path_str}"
                )));
            }

            let inner = WasmForeignObject::with_limits(path, timeout_seconds, memory_size)
                .map_err(|e| {
                    PyRuntimeError::new_err(format!("Failed to load WASM from file: {e}"))
                })?;
            return Ok(Self { inner });
        }

        // If none of the above worked, return error
        Err(PyException::new_err(
            "Expected str (file path), pathlib.Path, or bytes (WASM binary)",
        ))
    }

    /// Create a WebAssembly foreign object from a file
    ///
    /// Loads a WebAssembly module from a .wasm (binary) or .wat (text) file.
    ///
    /// Args:
    ///     path: Path to the WASM file (str or pathlib.Path)
    ///     timeout: Optional timeout in seconds for function execution (default: 1.0)
    ///     `memory_size`: Optional maximum memory size in bytes per linear memory
    ///                  (default: None = unlimited)
    ///
    /// Returns:
    ///     New WebAssembly foreign object instance
    ///
    /// Raises:
    ///     `FileNotFoundError`: If the file doesn't exist
    ///     `RuntimeError`: If WASM compilation fails
    ///
    /// Example:
    ///
    /// ```ignore
    /// # Basic usage
    /// wasm = WasmForeignObject.from_file("math.wasm")
    /// wasm.init()
    /// result = wasm.exec("add", [5, 3])
    ///
    /// # With custom timeout (5 seconds) and memory limit (10 MB)
    /// wasm = WasmForeignObject.from_file(
    ///     "compute.wasm",
    ///     timeout=5.0,
    ///     memory_size=10 * 1024 * 1024
    /// )
    /// ```
    #[staticmethod]
    #[pyo3(signature = (path, timeout=None, memory_size=None))]
    fn from_file(
        path: &Bound<'_, PyAny>,
        timeout: Option<f64>,
        memory_size: Option<usize>,
    ) -> PyResult<Self> {
        let timeout_seconds = timeout.unwrap_or(1.0);

        // Extract path string (handle both str and pathlib.Path)
        let path_str = if let Ok(s) = path.extract::<String>() {
            s
        } else if path.hasattr("__fspath__")? {
            path.call_method0("__fspath__")?.extract::<String>()?
        } else {
            return Err(PyException::new_err(
                "Expected str or pathlib.Path for 'path' argument",
            ));
        };

        let path = Path::new(&path_str);
        if !path.exists() {
            return Err(PyFileNotFoundError::new_err(format!(
                "WASM file not found: {path_str}"
            )));
        }

        let inner = WasmForeignObject::with_limits(path, timeout_seconds, memory_size)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to load WASM from file: {e}")))?;

        Ok(Self { inner })
    }

    /// Create a WebAssembly foreign object from bytes
    ///
    /// Loads a WebAssembly module directly from binary bytes in memory.
    /// This is useful when the WASM binary is downloaded, embedded, or
    /// generated programmatically.
    ///
    /// Args:
    ///     data: WASM binary as bytes
    ///     timeout: Optional timeout in seconds for function execution (default: 1.0)
    ///     `memory_size`: Optional maximum memory size in bytes per linear memory
    ///                  (default: None = unlimited)
    ///
    /// Returns:
    ///     New WebAssembly foreign object instance
    ///
    /// Raises:
    ///     `RuntimeError`: If WASM compilation fails
    ///
    /// Example:
    ///
    /// ```ignore
    /// # Load from file into bytes
    /// with open("math.wasm", "rb") as f:
    ///     wasm_bytes = f.read()
    /// wasm = WasmForeignObject.from_bytes(wasm_bytes)
    /// wasm.init()
    ///
    /// # From downloaded content
    /// import requests
    /// response = requests.get("https://example.com/module.wasm")
    /// wasm = WasmForeignObject.from_bytes(response.content)
    ///
    /// # With custom timeout
    /// wasm = WasmForeignObject.from_bytes(wasm_bytes, timeout=5.0)
    /// ```
    #[staticmethod]
    #[pyo3(signature = (data, timeout=None, memory_size=None))]
    fn from_bytes(
        data: &Bound<'_, PyBytes>,
        timeout: Option<f64>,
        memory_size: Option<usize>,
    ) -> PyResult<Self> {
        let timeout_seconds = timeout.unwrap_or(1.0);
        let wasm_bytes = data.as_bytes();

        let inner =
            WasmForeignObject::from_bytes_with_limits(wasm_bytes, timeout_seconds, memory_size)
                .map_err(|e| {
                    PyRuntimeError::new_err(format!("Failed to load WASM from bytes: {e}"))
                })?;

        Ok(Self { inner })
    }

    /// Initialize the WASM module
    ///
    /// This must be called before using the object. It creates a new instance
    /// and calls the 'init' function in the WASM module.
    ///
    /// Raises:
    ///     `RuntimeError`: If init function is missing or execution fails
    fn init(&mut self) -> PyResult<()> {
        self.inner
            .init()
            .map_err(|e| PyErr::new::<WasmError, _>(format!("Failed to initialize WASM: {e}")))
    }

    /// Reset variables before each shot
    ///
    /// Calls the '`shot_reinit`' function in the WASM module if it exists.
    /// This is a no-op if the function doesn't exist.
    ///
    /// Raises:
    ///     `RuntimeError`: If `shot_reinit` function exists but execution fails
    fn shot_reinit(&mut self) -> PyResult<()> {
        self.inner
            .shot_reinit()
            .map_err(|e| PyErr::new::<WasmError, _>(format!("Failed to call shot_reinit: {e}")))
    }

    /// Create a new WASM instance
    ///
    /// Resets the object's internal state by creating a fresh instance.
    ///
    /// Raises:
    ///     `RuntimeError`: If instance creation fails
    fn new_instance(&mut self) -> PyResult<()> {
        self.inner
            .new_instance()
            .map_err(|e| PyErr::new::<WasmError, _>(format!("Failed to create new instance: {e}")))
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
    ///     `func_name`: Name of the function to execute
    ///     args: List of integer arguments (i64)
    ///
    /// Returns:
    ///     Tuple containing the function results (or single 0 for void functions)
    ///
    /// Raises:
    ///     `RuntimeError`: If function not found or execution fails
    #[allow(clippy::needless_pass_by_value)] // PyO3 extracts Python sequences as Vec
    fn exec(&mut self, py: Python<'_>, func_name: &str, args: Vec<i64>) -> PyResult<Py<PyAny>> {
        let results = self
            .inner
            .exec(func_name, &args)
            .map_err(|e| PyErr::new::<WasmError, _>(e.to_string()))?;

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

    /// Get the WebAssembly binary bytes
    ///
    /// Returns:
    ///     The WASM binary as bytes
    #[getter]
    fn wasm_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, self.inner.wasm_bytes())
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
    ///     Dictionary containing '`fobj_class`', '`wasm_bytes`', 'timeout', and '`memory_size`'
    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dict = pyo3::types::PyDict::new(py);

        // Get the Python class for fobj_class
        let module = py.import("pecos_rslib")?;
        let cls = module.getattr("WasmForeignObject")?;
        dict.set_item("fobj_class", cls)?;

        // Get WASM bytes
        let wasm_bytes = PyBytes::new(py, self.inner.wasm_bytes());
        dict.set_item("wasm_bytes", wasm_bytes)?;

        // Get timeout
        dict.set_item("timeout", self.inner.timeout_seconds())?;

        // Get memory_size (None or usize)
        if let Some(size) = self.inner.memory_size() {
            dict.set_item("memory_size", size)?;
        } else {
            dict.set_item("memory_size", py.None())?;
        }

        Ok(dict.into())
    }

    /// Deserialize from dictionary (for pickling)
    ///
    /// Args:
    ///     `wasmtime_dict`: Dictionary containing '`fobj_class`', '`wasm_bytes`', and optionally 'timeout' and '`memory_size`'
    ///
    /// Returns:
    ///     New instance created from the dictionary
    #[staticmethod]
    fn from_dict(py: Python<'_>, wasmtime_dict: &Bound<'_, PyAny>) -> PyResult<Self> {
        use pyo3::types::PyDictMethods;
        let dict = wasmtime_dict.cast::<pyo3::types::PyDict>()?;
        let wasm_bytes = dict
            .get_item("wasm_bytes")?
            .ok_or_else(|| PyException::new_err("Missing 'wasm_bytes' in dictionary"))?;

        // Get timeout if present (default to 1.0 for backward compatibility)
        let timeout = dict
            .get_item("timeout")?
            .and_then(|t| t.extract::<f64>().ok());

        // Get memory_size if present (default to None for backward compatibility)
        let memory_size = dict
            .get_item("memory_size")?
            .and_then(|m| m.extract::<usize>().ok());

        Self::new(py, &wasm_bytes, timeout, memory_size)
    }

    /// Support for pickle (Python serialization)
    ///
    /// Returns a tuple (callable, args) that can be used to reconstruct the object.
    /// We use `from_dict` as the callable and the serialized dict as the argument.
    fn __reduce__(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, (Py<PyAny>,))> {
        // Get the from_dict classmethod
        let module = py.import("pecos_rslib")?;
        let cls = module.getattr("WasmForeignObject")?;
        let from_dict = cls.getattr("from_dict")?;

        // Get the state dict
        let state = self.to_dict(py)?;

        Ok((from_dict.into(), (state,)))
    }
}

pub fn register_wasm_types(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent_module.py();
    parent_module.add("WasmError", py.get_type::<WasmError>())?;
    Ok(())
}

impl Drop for PyWasmForeignObject {
    fn drop(&mut self) {
        // Ensure teardown is called when the object is dropped
        self.inner.teardown();
    }
}
