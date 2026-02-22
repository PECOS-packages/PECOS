// Copyright 2025 The PECOS Developers
use pecos::prelude::*;
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

use pyo3::prelude::*;
use pyo3::types::{PyDict, PySet};
use std::collections::BTreeMap;

/// Python wrapper for the Rust `PauliProp` simulator
///
/// This simulator tracks how Pauli operators propagate through Clifford circuits.
/// It's particularly useful for fault propagation and stabilizer simulations.
#[pyclass(name = "PauliProp", module = "pecos_rslib")]
pub struct PyPauliProp {
    inner: PauliProp,
    num_qubits: Option<usize>,
    track_sign: bool,
}

impl PyPauliProp {
    /// Helper method to build faults dictionary
    fn build_faults_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
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
}

#[pymethods]
impl PyPauliProp {
    /// Create a new `PauliProp` simulator
    ///
    /// Args:
    ///     `num_qubits`: Optional number of qubits (for string representation)
    ///     `track_sign`: Whether to track sign and phase
    #[new]
    #[pyo3(signature = (num_qubits=None, *, track_sign=false))]
    pub fn new(num_qubits: Option<usize>, track_sign: bool) -> Self {
        let inner = if track_sign {
            if let Some(n) = num_qubits {
                PauliProp::with_sign_tracking(n)
            } else {
                // Default to tracking with 0 qubits if not specified
                PauliProp::with_sign_tracking(0)
            }
        } else {
            PauliProp::new()
        };

        PyPauliProp {
            inner,
            num_qubits,
            track_sign,
        }
    }

    /// Get `num_qubits` (for backwards compatibility)
    #[getter]
    pub fn num_qubits(&self) -> Option<usize> {
        self.num_qubits
    }

    /// Get `track_sign` setting (for backwards compatibility)
    #[getter]
    pub fn track_sign(&self) -> bool {
        self.track_sign
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

            if let Ok(py_set) = value.cast::<PySet>() {
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

    /// Get the dense string representation (for `PauliProp`)
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
        self.inner.h(&[QubitId(qubit)]);
    }

    /// Apply S gate (sqrt(Z))
    pub fn sz(&mut self, qubit: usize) {
        self.inner.sz(&[QubitId(qubit)]);
    }

    /// Apply sqrt(X) gate
    pub fn sx(&mut self, qubit: usize) {
        self.inner.sx(&[QubitId(qubit)]);
    }

    /// Apply sqrt(Y) gate
    pub fn sy(&mut self, qubit: usize) {
        self.inner.sy(&[QubitId(qubit)]);
    }

    /// Apply CNOT/CX gate
    pub fn cx(&mut self, control: usize, target: usize) {
        self.inner.cx(&[QubitId(control), QubitId(target)]);
    }

    /// Apply CY gate
    pub fn cy(&mut self, control: usize, target: usize) {
        self.inner.cy(&[QubitId(control), QubitId(target)]);
    }

    /// Apply CZ gate
    pub fn cz(&mut self, control: usize, target: usize) {
        self.inner.cz(&[QubitId(control), QubitId(target)]);
    }

    /// Apply SWAP gate
    pub fn swap(&mut self, q1: usize, q2: usize) {
        self.inner.swap(&[QubitId(q1), QubitId(q2)]);
    }

    /// Measure in Z basis
    pub fn mz(&mut self, qubit: usize) -> bool {
        self.inner.mz(&[QubitId(qubit)])[0].outcome
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
    /// Also accessible as a property via the `faults` getter
    pub fn get_faults(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.build_faults_dict(py)
    }

    /// Property getter for faults (backwards compatibility with `PauliPropRs` wrapper)
    #[getter(faults)]
    pub fn get_faults_property(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.build_faults_dict(py)
    }

    /// Set faults by clearing and adding new ones
    pub fn set_faults(&mut self, paulis: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
        self.reset();
        if let Some(p) = paulis {
            self.add_paulis(p)?;
        }
        Ok(())
    }

    /// Alias for `get_sign` (backwards compatibility)
    pub fn get_sign_bool(&self) -> bool {
        self.inner.get_sign()
    }

    /// Alias for `get_img` (backwards compatibility)
    pub fn get_img_value(&self) -> u8 {
        self.inner.get_img()
    }

    /// Alias for `to_pauli_string` (backwards compatibility with `PauliFaultProp`)
    pub fn fault_string(&self) -> String {
        self.inner.to_pauli_string()
    }

    /// Alias for weight (backwards compatibility with `PauliFaultProp`)
    pub fn fault_wt(&self) -> usize {
        self.inner.weight()
    }

    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyTuple>> {
        let faults = self.build_faults_dict(py)?;
        let sign = self.inner.get_sign();
        let img = self.inner.get_img();

        let cls = py.get_type::<PyPauliProp>();
        let from_pickle = cls.getattr("_from_pickle")?;
        pyo3::types::PyTuple::new(
            py,
            &[
                from_pickle.into_any(),
                pyo3::types::PyTuple::new(
                    py,
                    &[
                        self.num_qubits.into_pyobject(py)?.into_any(),
                        self.track_sign.into_pyobject(py)?.to_owned().into_any(),
                        faults.into_bound(py).into_any(),
                        sign.into_pyobject(py)?.to_owned().into_any(),
                        img.into_pyobject(py)?.into_any(),
                    ],
                )?
                .into_any(),
            ],
        )
    }

    #[staticmethod]
    fn _from_pickle(
        num_qubits: Option<usize>,
        track_sign: bool,
        faults: &Bound<'_, PyDict>,
        sign: bool,
        img: u8,
    ) -> PyResult<Self> {
        let mut obj = PyPauliProp::new(num_qubits, track_sign);
        obj.add_paulis(faults)?;
        // Restore sign: if stored sign is negative, flip it (default is false/positive)
        if sign {
            obj.inner.flip_sign();
        }
        // Restore img: add the stored imaginary count
        if img > 0 {
            obj.inner.flip_img(img as usize);
        }
        Ok(obj)
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
