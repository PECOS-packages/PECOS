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

use pecos::prelude::{ByteMessage, ByteMessageBuilder, dump_batch};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyType};

/// Python wrapper for Rust ByteMessageBuilder
#[pyclass(name = "ByteMessageBuilder")]
pub struct PyByteMessageBuilder {
    inner: ByteMessageBuilder,
}

#[pymethods]
impl PyByteMessageBuilder {
    /// Create a new ByteMessageBuilder
    #[new]
    fn new() -> Self {
        Self {
            inner: ByteMessageBuilder::new(),
        }
    }

    /// Configure the builder for quantum operations
    #[pyo3(text_signature = "($self)")]
    fn for_quantum_operations(&mut self) {
        let _ = self.inner.for_quantum_operations();
    }

    /// Configure the builder for measurement results
    #[pyo3(text_signature = "($self)")]
    fn for_measurement_results(&mut self) {
        let _ = self.inner.for_measurement_results();
    }

    /// Add an X gate to the message
    #[pyo3(text_signature = "($self, qubit)")]
    fn add_x(&mut self, qubit: usize) {
        self.inner.add_x(&[qubit]);
    }

    /// Add a Y gate to the message
    #[pyo3(text_signature = "($self, qubit)")]
    fn add_y(&mut self, qubit: usize) {
        self.inner.add_y(&[qubit]);
    }

    /// Add a Z gate to the message
    #[pyo3(text_signature = "($self, qubit)")]
    fn add_z(&mut self, qubit: usize) {
        self.inner.add_z(&[qubit]);
    }

    /// Add an H gate to the message
    #[pyo3(text_signature = "($self, qubit)")]
    fn add_h(&mut self, qubit: usize) {
        self.inner.add_h(&[qubit]);
    }

    /// Add a CX (CNOT) gate to the message
    #[pyo3(text_signature = "($self, control, target)")]
    fn add_cx(&mut self, control: usize, target: usize) {
        self.inner.add_cx(&[control], &[target]);
    }

    /// Add an RZ gate to the message
    #[pyo3(text_signature = "($self, theta, qubit)")]
    fn add_rz(&mut self, theta: f64, qubit: usize) {
        self.inner.add_rz(theta, &[qubit]);
    }

    /// Add an RZZ gate to the message
    #[pyo3(text_signature = "($self, theta, qubit1, qubit2)")]
    fn add_rzz(&mut self, theta: f64, qubit1: usize, qubit2: usize) {
        self.inner.add_rzz(theta, &[qubit1], &[qubit2]);
    }

    /// Add an SZZ gate to the message
    #[pyo3(text_signature = "($self, qubit1, qubit2)")]
    fn add_szz(&mut self, qubit1: usize, qubit2: usize) {
        self.inner.add_szz(&[qubit1], &[qubit2]);
    }

    /// Add an R1XY gate to the message
    #[pyo3(text_signature = "($self, theta, phi, qubit)")]
    fn add_r1xy(&mut self, theta: f64, phi: f64, qubit: usize) {
        self.inner.add_r1xy(theta, phi, &[qubit]);
    }

    /// Add a measurement gate to the message
    #[pyo3(text_signature = "($self, qubit, result_id)")]
    fn add_measurement(&mut self, qubit: usize, result_id: usize) {
        self.inner.add_measurements(&[qubit], &[result_id]);
    }

    /// Add a qubit preparation gate to the message
    #[pyo3(text_signature = "($self, qubit)")]
    fn add_prep(&mut self, qubit: usize) {
        self.inner.add_prep(&[qubit]);
    }

    /// Add a flush command to the message
    #[pyo3(text_signature = "($self, is_last=False)")]
    fn add_flush(&mut self, is_last: Option<bool>) {
        self.inner.add_flush(is_last.unwrap_or(false));
    }

    /// Build the message and return a PyByteMessage
    #[pyo3(text_signature = "($self)")]
    fn build(&mut self) -> PyByteMessage {
        PyByteMessage {
            inner: self.inner.build(),
        }
    }

    /// Clear the builder and reset to initial state
    #[pyo3(text_signature = "($self)")]
    fn clear(&mut self) {
        self.inner.clear();
    }

    /// Reset the builder to initial state while preserving capacity
    #[pyo3(text_signature = "($self)")]
    fn reset(&mut self) {
        self.inner.reset();
    }
}

/// Python wrapper for Rust ByteMessage
#[pyclass(name = "ByteMessage")]
pub struct PyByteMessage {
    inner: ByteMessage,
}

#[pymethods]
impl PyByteMessage {
    /// Create a new empty ByteMessage
    #[classmethod]
    fn create_empty(_cls: &Bound<PyType>) -> Self {
        Self {
            inner: ByteMessage::create_empty(),
        }
    }

    /// Create a new ByteMessageBuilder
    #[classmethod]
    fn builder(_cls: &Bound<PyType>) -> PyByteMessageBuilder {
        PyByteMessageBuilder::new()
    }

    /// Create a ByteMessageBuilder configured for quantum operations
    #[classmethod]
    fn quantum_operations_builder(_cls: &Bound<PyType>) -> PyByteMessageBuilder {
        let mut builder = PyByteMessageBuilder::new();
        builder.for_quantum_operations();
        builder
    }

    /// Create a ByteMessageBuilder configured for measurement results
    #[classmethod]
    fn measurement_results_builder(_cls: &Bound<PyType>) -> PyByteMessageBuilder {
        let mut builder = PyByteMessageBuilder::new();
        builder.for_measurement_results();
        builder
    }

    /// Create a flush message
    #[classmethod]
    fn create_flush(_cls: &Bound<PyType>) -> Self {
        Self {
            inner: ByteMessage::create_flush(),
        }
    }

    /// Get the ByteMessage as bytes
    #[pyo3(text_signature = "($self)")]
    fn as_bytes(&self, py: Python<'_>) -> PyObject {
        PyBytes::new(py, self.inner.as_bytes()).into()
    }

    /// Check if the message is empty
    #[pyo3(text_signature = "($self)")]
    fn is_empty(&self) -> bool {
        self.inner.is_empty().unwrap_or(true)
    }

    /// Parse quantum operations from the message
    #[pyo3(text_signature = "($self)")]
    fn parse_quantum_operations(&self, py: Python<'_>) -> PyResult<Vec<PyObject>> {
        let mut results = Vec::new();

        for op in self
            .inner
            .parse_quantum_operations()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
        {
            let dict = PyDict::new(py);

            // Convert gate_type to a string
            dict.set_item("gate_type", op.gate_type.to_string())?;
            dict.set_item("qubits", op.qubits.clone())?;

            // Handle params vector
            if !op.params.is_empty() {
                dict.set_item("params", op.params.clone())?;
            }

            if let Some(result_id) = op.result_id {
                dict.set_item("result_id", result_id)?;
            }

            results.push(dict.into());
        }

        Ok(results)
    }

    /// Dump the batch contents for debugging
    #[pyo3(text_signature = "($self)")]
    fn dump_batch(&self) -> String {
        dump_batch(self.inner.as_bytes())
    }

    /// Get measurement results as a list of (result_id, outcome) tuples
    #[pyo3(text_signature = "($self)")]
    pub fn measurement_results(&self, py: Python<'_>) -> PyResult<PyObject> {
        let results = self
            .inner
            .measurement_results_as_vec()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        // Create a list of lists, where each inner list has two elements
        let result_list = PyList::empty(py);
        for (result_id, outcome) in results {
            // For each measurement, create a small list with [result_id, outcome]
            let inner_list = PyList::empty(py);
            inner_list.append(result_id)?;
            inner_list.append(outcome as usize)?;

            // Add the inner list to the result list
            result_list.append(inner_list)?;
        }

        Ok(result_list.into())
    }
}

// Add these methods outside of #[pymethods] since they're for internal Rust use only
impl PyByteMessage {
    /// Clone the inner `ByteMessage` of this `PyByteMessage`
    pub fn clone_inner(&self) -> ByteMessage {
        self.inner.clone()
    }

    /// Create a new `PyByteMessage` from a `ByteMessage`
    pub fn from_byte_message(message: ByteMessage) -> Self {
        Self { inner: message }
    }
}
