// Copyright 2024 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Python bindings for QIR generation using the Rust pecos-llvm crate

use pecos::prelude::QirBuilder;
use pyo3::prelude::*;

/// Python wrapper for QirBuilder
///
/// This class provides QIR (Quantum Intermediate Representation) generation
/// functionality from Python, replacing the llvmlite dependency.
///
/// # Example (from Python):
/// ```python
/// from pecos_rslib import QirBuilder
///
/// builder = QirBuilder("0.1.1")
/// builder.create_qreg("q", 2)
/// builder.create_creg("c", 2, True)
/// builder.apply_gate("h", [0], [])
/// builder.apply_gate("cx", [0, 1], [])
/// builder.measure_to_bit(0, "c", 0)
/// builder.measure_to_bit(1, "c", 1)
/// ir = builder.get_output()
/// print(ir)
/// ```
#[pyclass(name = "QirBuilder")]
pub struct PyQirBuilder {
    // We use Box::leak to create a 'static reference to the context
    // This is a controlled memory leak that's acceptable for these short-lived builders
    builder: QirBuilder<'static>,
}

// SAFETY: PyQirBuilder is safe to Send/Sync because:
// 1. Python's GIL ensures single-threaded access to Python objects
// 2. The LLVM context is only accessed from the Python thread that created it
// 3. We never share the builder across threads - it's owned by a Python object
unsafe impl Send for PyQirBuilder {}
unsafe impl Sync for PyQirBuilder {}

#[pymethods]
impl PyQirBuilder {
    /// Create a new QIR builder
    ///
    /// Args:
    ///     pecos_version: Version string to embed in generated IR
    ///
    /// Returns:
    ///     A new QirBuilder instance
    #[new]
    fn new(pecos_version: &str) -> PyResult<Self> {
        let builder = QirBuilder::new_with_leaked_context(pecos_version)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(Self { builder })
    }

    /// Create a quantum register
    ///
    /// Args:
    ///     name: Name of the quantum register
    ///     size: Number of qubits in the register
    fn create_qreg(&mut self, name: &str, size: usize) -> PyResult<()> {
        self.builder
            .create_qreg(name, size)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Create a classical register
    ///
    /// Args:
    ///     name: Name of the classical register
    ///     size: Number of bits in the register
    ///     is_result: Whether this register contains measurement results
    fn create_creg(&mut self, name: &str, size: usize, is_result: bool) -> PyResult<()> {
        self.builder
            .create_creg(name, size, is_result)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Apply a quantum gate to qubits
    ///
    /// Args:
    ///     gate_name: Name of the gate (e.g., "h", "cx", "rz")
    ///     qubits: List of qubit indices to apply the gate to
    ///     params: List of gate parameters (e.g., rotation angles)
    ///
    /// # Example:
    /// ```python
    /// builder.apply_gate("h", [0], [])           # Hadamard on qubit 0
    /// builder.apply_gate("cx", [0, 1], [])       # CNOT from qubit 0 to 1
    /// builder.apply_gate("rz", [0], [1.57])      # RZ(π/2) on qubit 0
    /// ```
    fn apply_gate(&mut self, gate_name: &str, qubits: Vec<usize>, params: Vec<f64>) -> PyResult<()> {
        self.builder
            .apply_gate(gate_name, &qubits, &params)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Measure a qubit to a classical bit
    ///
    /// Args:
    ///     qubit_idx: Index of the qubit to measure
    ///     creg_name: Name of the classical register to store the result
    ///     bit_idx: Index of the bit in the classical register
    fn measure_to_bit(&mut self, qubit_idx: usize, creg_name: &str, bit_idx: usize) -> PyResult<()> {
        self.builder
            .measure_to_bit(qubit_idx, creg_name, bit_idx)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Get the generated LLVM IR as a string
    ///
    /// Returns:
    ///     The complete QIR LLVM IR as a string
    fn get_output(&self) -> String {
        self.builder.get_output()
    }

    /// Get the generated LLVM bitcode as bytes
    ///
    /// Returns:
    ///     The LLVM bitcode as bytes
    fn get_bitcode(&self) -> Vec<u8> {
        self.builder.get_bitcode()
    }

    /// Get the number of qubits allocated
    #[getter]
    fn qubit_count(&self) -> usize {
        self.builder.qubit_count()
    }

    /// Get the number of measurements performed
    #[getter]
    fn measure_count(&self) -> usize {
        self.builder.measure_count()
    }

    fn __repr__(&self) -> String {
        "QirBuilder()".to_string()
    }
}

/// Register QIR functions and classes with the Python module
pub fn register_qir_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyQirBuilder>()?;
    Ok(())
}
