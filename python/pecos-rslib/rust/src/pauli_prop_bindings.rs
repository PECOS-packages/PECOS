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

use pecos_core::{Set, VecSet};
use pecos_qsim::{CliffordGateable, QuantumSimulator, StdPauliProp};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySet};
use std::collections::BTreeMap;

/// Python wrapper for the Rust `PauliProp` simulator
///
/// This simulator tracks how Pauli operators propagate through Clifford circuits.
/// It's particularly useful for fault propagation and stabilizer simulations.
#[pyclass(name = "PauliProp")]
pub struct PyPauliProp {
    inner: StdPauliProp,
}

#[pymethods]
impl PyPauliProp {
    /// Create a new `PauliProp` simulator
    ///
    /// Args:
    ///     `num_qubits`: Optional number of qubits (for string representation)
    ///     `track_sign`: Whether to track sign and phase
    #[new]
    #[pyo3(signature = (num_qubits=None, track_sign=false))]
    pub fn new(num_qubits: Option<usize>, track_sign: bool) -> Self {
        let inner = if track_sign {
            if let Some(n) = num_qubits {
                StdPauliProp::with_sign_tracking(n)
            } else {
                // Default to tracking with 0 qubits if not specified
                StdPauliProp::with_sign_tracking(0)
            }
        } else {
            StdPauliProp::new()
        };

        PyPauliProp { inner }
    }

    /// Reset the simulator state
    pub fn reset(&mut self) {
        self.inner.reset();
    }

    /// Check if a qubit has an X operator
    pub fn contains_x(&self, qubit: usize) -> bool {
        self.inner.contains_x(qubit)
    }

    /// Check if a qubit has a Z operator
    pub fn contains_z(&self, qubit: usize) -> bool {
        self.inner.contains_z(qubit)
    }

    /// Check if a qubit has a Y operator
    pub fn contains_y(&self, qubit: usize) -> bool {
        self.inner.contains_y(qubit)
    }

    /// Add an X operator to a qubit
    pub fn add_x(&mut self, qubit: usize) {
        self.inner.add_x(qubit);
    }

    /// Add a Z operator to a qubit
    pub fn add_z(&mut self, qubit: usize) {
        self.inner.add_z(qubit);
    }

    /// Add a Y operator to a qubit
    pub fn add_y(&mut self, qubit: usize) {
        self.inner.add_y(qubit);
    }

    /// Flip the sign of the Pauli string
    pub fn flip_sign(&mut self) {
        self.inner.flip_sign();
    }

    /// Add imaginary factors
    pub fn flip_img(&mut self, num_is: usize) {
        self.inner.flip_img(num_is);
    }

    /// Add Pauli operators from a dictionary
    ///
    /// Args:
    ///     paulis: Dictionary with keys "X", "Y", "Z" mapping to sets of qubit indices
    pub fn add_paulis(&mut self, paulis: &Bound<'_, PyDict>) -> PyResult<()> {
        let mut btree_map = BTreeMap::new();

        // Convert Python dict to BTreeMap<String, VecSet<usize>>
        for (key, value) in paulis.iter() {
            let key_str: String = key.extract()?;

            if let Ok(py_set) = value.downcast::<PySet>() {
                let mut vec_set = VecSet::new();
                for item in py_set.iter() {
                    let qubit: usize = item.extract()?;
                    vec_set.insert(qubit);
                }
                btree_map.insert(key_str, vec_set);
            } else {
                // Try to handle it as a Python set-like object
                let iter = value.call_method0("__iter__")?;
                let mut vec_set = VecSet::new();
                while let Ok(item) = iter.call_method0("__next__") {
                    let qubit: usize = item.extract()?;
                    vec_set.insert(qubit);
                }
                btree_map.insert(key_str, vec_set);
            }
        }

        self.inner.add_paulis(&btree_map);
        Ok(())
    }

    /// Get the weight of the Pauli string (number of non-identity operators)
    pub fn weight(&self) -> usize {
        self.inner.weight()
    }

    /// Get the sign string representation
    pub fn sign_string(&self) -> String {
        self.inner.sign_string()
    }

    /// Get the sparse string representation
    pub fn sparse_string(&self) -> String {
        self.inner.sparse_string()
    }

    /// Get the dense string representation (for `StdPauliProp`)
    pub fn dense_string(&self) -> String {
        self.inner.dense_string()
    }

    /// Get the full Pauli string with sign
    pub fn to_pauli_string(&self) -> String {
        self.inner.to_pauli_string()
    }

    /// Get the full dense Pauli string with sign
    pub fn to_dense_string(&self) -> String {
        self.inner.to_dense_string()
    }

    // Clifford gates

    /// Apply Hadamard gate
    pub fn h(&mut self, qubit: usize) {
        self.inner.h(qubit);
    }

    /// Apply S gate (sqrt(Z))
    pub fn sz(&mut self, qubit: usize) {
        self.inner.sz(qubit);
    }

    /// Apply sqrt(X) gate
    pub fn sx(&mut self, qubit: usize) {
        self.inner.sx(qubit);
    }

    /// Apply sqrt(Y) gate
    pub fn sy(&mut self, qubit: usize) {
        self.inner.sy(qubit);
    }

    /// Apply CNOT/CX gate
    pub fn cx(&mut self, control: usize, target: usize) {
        self.inner.cx(control, target);
    }

    /// Apply CY gate
    pub fn cy(&mut self, control: usize, target: usize) {
        self.inner.cy(control, target);
    }

    /// Apply CZ gate
    pub fn cz(&mut self, control: usize, target: usize) {
        self.inner.cz(control, target);
    }

    /// Apply SWAP gate
    pub fn swap(&mut self, q1: usize, q2: usize) {
        self.inner.swap(q1, q2);
    }

    /// Measure in Z basis
    pub fn mz(&mut self, qubit: usize) -> bool {
        self.inner.mz(qubit).outcome
    }

    /// Check if this is the identity operator
    pub fn is_identity(&self) -> bool {
        self.inner.is_identity()
    }

    /// Get the sign as a boolean (false for +, true for -)
    pub fn get_sign(&self) -> bool {
        self.inner.get_sign()
    }

    /// Get the imaginary component (0 for real, 1 for imaginary)
    pub fn get_img(&self) -> u8 {
        self.inner.get_img()
    }

    /// Get all faults as a dictionary (compatible with Python `PauliFaultProp`)
    pub fn get_faults(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new(py);

        // Get X-only qubits
        let x_set = PySet::empty(py)?;
        for qubit in self.inner.get_x_only_qubits() {
            x_set.add(qubit)?;
        }
        dict.set_item("X", x_set)?;

        // Get Y qubits
        let y_set = PySet::empty(py)?;
        for qubit in self.inner.get_y_qubits() {
            y_set.add(qubit)?;
        }
        dict.set_item("Y", y_set)?;

        // Get Z-only qubits
        let z_set = PySet::empty(py)?;
        for qubit in self.inner.get_z_only_qubits() {
            z_set.add(qubit)?;
        }
        dict.set_item("Z", z_set)?;

        Ok(dict.into())
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!("PauliProp({})", self.inner.to_pauli_string())
    }

    /// String representation
    fn __str__(&self) -> String {
        self.inner.to_pauli_string()
    }
}
