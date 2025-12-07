// Copyright 2025 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! Python bindings for WebAssembly programs.
//!
//! This module provides `PyO3` bindings for WASM and WAT program types, enabling Python code
//! to work with WebAssembly programs for quantum simulation.

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyType};

/// A WebAssembly (WASM) program wrapper.
///
/// This class holds compiled WebAssembly bytecode that can be used for
/// quantum circuit execution in WASM-based runtimes.
#[pyclass(name = "Wasm")]
pub struct PyWasm {
    wasm_bytes: Vec<u8>,
}

#[pymethods]
impl PyWasm {
    /// Create a new WASM program from bytes.
    ///
    /// Args:
    ///     `wasm_bytes`: The compiled WASM bytecode
    #[new]
    fn new(wasm_bytes: Vec<u8>) -> Self {
        PyWasm { wasm_bytes }
    }

    /// Create a WASM program from bytes (class method).
    ///
    /// Args:
    ///     `wasm_bytes`: The compiled WASM bytecode
    ///
    /// Returns:
    ///     `Wasm`: A new WASM program instance
    #[classmethod]
    fn from_bytes(_cls: &Bound<'_, PyType>, wasm_bytes: Vec<u8>) -> Self {
        PyWasm { wasm_bytes }
    }

    /// Get the WASM bytecode.
    ///
    /// Returns:
    ///     bytes: The WASM bytecode
    fn bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.wasm_bytes)
    }

    fn __repr__(&self) -> String {
        format!("Wasm({} bytes)", self.wasm_bytes.len())
    }
}

/// A WebAssembly Text (WAT) program wrapper.
///
/// This class holds WAT source code (the textual representation of WASM)
/// that can be compiled to WASM for execution.
#[pyclass(name = "Wat")]
pub struct PyWat {
    source: String,
}

#[pymethods]
impl PyWat {
    /// Create a new WAT program from source code.
    ///
    /// Args:
    ///     source: The WAT source code
    #[new]
    fn new(source: String) -> Self {
        PyWat { source }
    }

    /// Create a WAT program from a string (class method).
    ///
    /// Args:
    ///     source: The WAT source code
    ///
    /// Returns:
    ///     `Wat`: A new WAT program instance
    #[classmethod]
    fn from_string(_cls: &Bound<'_, PyType>, source: String) -> Self {
        PyWat { source }
    }

    fn __str__(&self) -> &str {
        &self.source
    }

    fn __repr__(&self) -> String {
        let preview = if self.source.len() > 50 {
            format!("{}...", &self.source[..50])
        } else {
            self.source.clone()
        };
        format!("Wat('{preview}')")
    }
}

/// Register the WASM program types with the Python module.
pub fn register_wasm_programs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyWasm>()?;
    m.add_class::<PyWat>()?;
    Ok(())
}
