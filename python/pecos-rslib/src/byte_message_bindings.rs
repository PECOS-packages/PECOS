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

use crate::dtypes::AngleParam;
use crate::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyType};

/// Python wrapper for Rust `ByteMessageBuilder`
#[pyclass(name = "ByteMessageBuilder", module = "pecos_rslib")]
pub struct PyByteMessageBuilder {
    inner: ByteMessageBuilder,
}

#[pymethods]
impl PyByteMessageBuilder {
    /// Create a new `ByteMessageBuilder`
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

    /// Configure the builder for measurement outcomes
    #[pyo3(text_signature = "($self)")]
    fn for_outcomes(&mut self) {
        let _ = self.inner.for_outcomes();
    }

    /// Add X gate(s): `x([0, 1, 2])`
    fn x(&mut self, qubits: Vec<usize>) {
        self.inner.x(&qubits);
    }

    /// Add Y gate(s): `y([0, 1, 2])`
    fn y(&mut self, qubits: Vec<usize>) {
        self.inner.y(&qubits);
    }

    /// Add Z gate(s): `z([0, 1, 2])`
    fn z(&mut self, qubits: Vec<usize>) {
        self.inner.z(&qubits);
    }

    /// Add H gate(s): `h([0, 1, 2])`
    fn h(&mut self, qubits: Vec<usize>) {
        self.inner.h(&qubits);
    }

    /// Add CX (CNOT) gate(s): `cx([(c0, t0), (c1, t1)])`
    fn cx(&mut self, pairs: Vec<(usize, usize)>) {
        self.inner.cx(&pairs);
    }

    /// Add RZ gate(s): `rz(theta, [q0, q1])`
    fn rz(&mut self, theta: AngleParam, qubits: Vec<usize>) {
        self.inner.rz(theta.0, &qubits);
    }

    /// Add RZZ gate(s): `rzz(theta, [(q0, q1), (q2, q3)])`
    fn rzz(&mut self, theta: AngleParam, pairs: Vec<(usize, usize)>) {
        self.inner.rzz(theta.0, &pairs);
    }

    /// Add CY gate(s): `cy([(c0, t0), (c1, t1)])`
    fn cy(&mut self, pairs: Vec<(usize, usize)>) {
        self.inner.cy(&pairs);
    }

    /// Add CZ gate(s): `cz([(c0, t0), (c1, t1)])`
    fn cz(&mut self, pairs: Vec<(usize, usize)>) {
        self.inner.cz(&pairs);
    }

    /// Add SZZ gate(s): `szz([(q0, q1), (q2, q3)])`
    fn szz(&mut self, pairs: Vec<(usize, usize)>) {
        self.inner.szz(&pairs);
    }

    /// Add `SZZdg` gate(s): `szzdg([(q0, q1), (q2, q3)])`
    fn szzdg(&mut self, pairs: Vec<(usize, usize)>) {
        self.inner.szzdg(&pairs);
    }

    /// Add SZ gate(s): `sz([q0, q1])`
    fn sz(&mut self, qubits: Vec<usize>) {
        self.inner.sz(&qubits);
    }

    /// Add `SZdg` gate(s): `szdg([q0, q1])`
    fn szdg(&mut self, qubits: Vec<usize>) {
        self.inner.szdg(&qubits);
    }

    /// Add T gate(s): `t([q0, q1])`
    fn t(&mut self, qubits: Vec<usize>) {
        self.inner.t(&qubits);
    }

    /// Add Tdg gate(s): `tdg([q0, q1])`
    fn tdg(&mut self, qubits: Vec<usize>) {
        self.inner.tdg(&qubits);
    }

    /// Add RX gate(s): `rx(theta, [q0, q1])`
    fn rx(&mut self, theta: AngleParam, qubits: Vec<usize>) {
        self.inner.rx(theta.0, &qubits);
    }

    /// Add RY gate(s): `ry(theta, [q0, q1])`
    fn ry(&mut self, theta: AngleParam, qubits: Vec<usize>) {
        self.inner.ry(theta.0, &qubits);
    }

    /// Add R1XY gate(s): `r1xy(theta, phi, [q0, q1])`
    fn r1xy(&mut self, theta: AngleParam, phi: AngleParam, qubits: Vec<usize>) {
        self.inner.r1xy(theta.0, phi.0, &qubits);
    }

    /// Add U gate(s): `u(theta, phi, lambda_, [q0, q1])`
    fn u(&mut self, theta: AngleParam, phi: AngleParam, lambda_: AngleParam, qubits: Vec<usize>) {
        self.inner.u(theta.0, phi.0, lambda_.0, &qubits);
    }

    /// SX (sqrt-X) gate(s): `sx([q0, q1])`
    fn sx(&mut self, qubits: Vec<usize>) {
        self.inner.sx(&qubits);
    }

    /// `SXdg` (sqrt-X dagger) gate(s): `sxdg([q0, q1])`
    fn sxdg(&mut self, qubits: Vec<usize>) {
        self.inner.sxdg(&qubits);
    }

    /// SY (sqrt-Y) gate(s): `sy([q0, q1])`
    fn sy(&mut self, qubits: Vec<usize>) {
        self.inner.sy(&qubits);
    }

    /// `SYdg` (sqrt-Y dagger) gate(s): `sydg([q0, q1])`
    fn sydg(&mut self, qubits: Vec<usize>) {
        self.inner.sydg(&qubits);
    }

    /// SWAP gate(s): `swap([(q0, q1), (q2, q3)])`
    fn swap(&mut self, pairs: Vec<(usize, usize)>) {
        self.inner.swap(&pairs);
    }

    /// SXX (sqrt-XX) gate(s): `sxx([(q0, q1)])`
    fn sxx(&mut self, pairs: Vec<(usize, usize)>) {
        self.inner.sxx(&pairs);
    }

    /// `SXXdg` (sqrt-XX dagger) gate(s): `sxxdg([(q0, q1)])`
    fn sxxdg(&mut self, pairs: Vec<(usize, usize)>) {
        self.inner.sxxdg(&pairs);
    }

    /// SYY (sqrt-YY) gate(s): `syy([(q0, q1)])`
    fn syy(&mut self, pairs: Vec<(usize, usize)>) {
        self.inner.syy(&pairs);
    }

    /// `SYYdg` (sqrt-YY dagger) gate(s): `syydg([(q0, q1)])`
    fn syydg(&mut self, pairs: Vec<(usize, usize)>) {
        self.inner.syydg(&pairs);
    }

    /// RXX gate(s): `rxx(theta, [(q0, q1)])`
    fn rxx(&mut self, theta: AngleParam, pairs: Vec<(usize, usize)>) {
        self.inner.rxx(theta.0, &pairs);
    }

    /// RYY gate(s): `ryy(theta, [(q0, q1)])`
    fn ryy(&mut self, theta: AngleParam, pairs: Vec<(usize, usize)>) {
        self.inner.ryy(theta.0, &pairs);
    }

    /// Z-basis measurement(s): `mz([0, 1, 2])`
    fn mz(&mut self, qubits: Vec<usize>) {
        self.inner.mz(&qubits);
    }

    /// PZ (preparation/reset) gate(s): `pz([0, 1, 2])`
    fn pz(&mut self, qubits: Vec<usize>) {
        self.inner.pz(&qubits);
    }

    /// Build the message and return a `PyByteMessage`
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

    #[allow(clippy::unused_self)]
    fn __repr__(&self) -> String {
        "ByteMessageBuilder()".to_string()
    }
}

/// Python wrapper for Rust `ByteMessage`
#[pyclass(name = "ByteMessage", module = "pecos_rslib")]
pub struct PyByteMessage {
    inner: ByteMessage,
}

#[pymethods]
impl PyByteMessage {
    /// Create a new `ByteMessageBuilder`
    #[classmethod]
    fn builder(_cls: &Bound<PyType>) -> PyByteMessageBuilder {
        PyByteMessageBuilder::new()
    }

    /// Create a `ByteMessageBuilder` configured for quantum operations
    #[classmethod]
    fn quantum_operations_builder(_cls: &Bound<PyType>) -> PyByteMessageBuilder {
        let mut builder = PyByteMessageBuilder::new();
        builder.for_quantum_operations();
        builder
    }

    /// Create a `ByteMessageBuilder` configured for measurement outcomes
    #[classmethod]
    fn outcomes_builder(_cls: &Bound<PyType>) -> PyByteMessageBuilder {
        let mut builder = PyByteMessageBuilder::new();
        builder.for_outcomes();
        builder
    }

    /// Create an empty message
    #[classmethod]
    fn create_empty(_cls: &Bound<PyType>) -> Self {
        Self {
            inner: ByteMessage::create_empty(),
        }
    }

    /// Get the `ByteMessage` as bytes
    #[pyo3(text_signature = "($self)")]
    fn as_bytes(&self, py: Python<'_>) -> Py<PyAny> {
        PyBytes::new(py, self.inner.as_bytes()).into()
    }

    /// Check if the message is empty
    #[pyo3(text_signature = "($self)")]
    fn is_empty(&self) -> bool {
        self.inner.is_empty().unwrap_or(true)
    }

    /// Parse quantum operations from the message
    #[pyo3(text_signature = "($self)")]
    fn parse_quantum_operations(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        let mut results = Vec::new();

        for op in self.inner.quantum_ops().map_err(|e| {
            PyRuntimeError::new_err(format!(
                "Failed to parse quantum operations in Python bindings: {e}"
            ))
        })? {
            let dict = PyDict::new(py);

            // Convert gate_type to a string
            dict.set_item("gate_type", op.gate_type.to_string())?;
            let qubits_as_usize: Vec<usize> = op.qubits.iter().map(|q| **q).collect();
            dict.set_item("qubits", qubits_as_usize)?;

            // Handle angles vector (rotation angles stored as Angle64, convert to radians for Python)
            if !op.angles.is_empty() {
                let angles_radians: Vec<f64> = op
                    .angles
                    .iter()
                    .map(pecos_core::Angle::to_radians)
                    .collect();
                dict.set_item("angles", angles_radians)?;
            }

            // Handle params vector (other non-angle parameters)
            if !op.params.is_empty() {
                dict.set_item("params", op.params.to_vec())?;
            }

            // result_id no longer exists on GateCommand

            results.push(dict.into());
        }

        Ok(results)
    }

    /// Dump the batch contents for debugging
    #[pyo3(text_signature = "($self)")]
    fn dump_batch(&self) -> String {
        dump_batch(self.inner.as_bytes())
    }

    /// Get measurement results as a list of (`result_id`, outcome) tuples
    #[pyo3(text_signature = "($self)")]
    pub fn measurement_results(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        // Get raw outcomes
        let outcomes = self.inner.outcomes().map_err(|e| {
            PyRuntimeError::new_err(format!(
                "Failed to extract measurement results in Python bindings: {e}"
            ))
        })?;

        // Create a list of lists, where each inner list has two elements
        let result_list = PyList::empty(py);
        for (result_id, outcome) in outcomes.into_iter().enumerate() {
            // For each measurement, create a small list with [result_id, outcome]
            let inner_list = PyList::empty(py);
            inner_list.append(result_id)?;
            inner_list.append(outcome as usize)?;

            // Add the inner list to the result list
            result_list.append(inner_list)?;
        }

        Ok(result_list.into())
    }

    fn __repr__(&self) -> String {
        let bytes_len = self.inner.as_bytes().len();
        format!("ByteMessage(size={bytes_len} bytes)")
    }

    /// Get the size of the message in bytes
    #[getter]
    fn size(&self) -> usize {
        self.inner.as_bytes().len()
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
