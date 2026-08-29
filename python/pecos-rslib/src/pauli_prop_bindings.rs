// Copyright 2026 The PECOS Developers
use crate::prelude::*;
use crate::simulator_utils::{extract_angle, extract_angles};
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

use pecos_simulators::clifford_rotation::CliffordRotation;
use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PySet, PyTuple};
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

    #[expect(clippy::too_many_lines)]
    #[pyo3(signature = (symbol, location, params=None))]
    fn run_1q_gate(
        &mut self,
        symbol: &str,
        location: usize,
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Option<u8>> {
        let q = &[QubitId(location)];
        match symbol {
            "I" => Ok(None),
            "X" => {
                self.inner.x(q);
                Ok(None)
            }
            "Y" => {
                self.inner.y(q);
                Ok(None)
            }
            "Z" => {
                self.inner.z(q);
                Ok(None)
            }
            "H" | "H1" | "H+z+x" => {
                self.inner.h(q);
                Ok(None)
            }
            "H2" | "H-z-x" => {
                self.inner.h2(q);
                Ok(None)
            }
            "H3" | "H+y-z" => {
                self.inner.h3(q);
                Ok(None)
            }
            "H4" | "H-y-z" => {
                self.inner.h4(q);
                Ok(None)
            }
            "H5" | "H-x+y" => {
                self.inner.h5(q);
                Ok(None)
            }
            "H6" | "H-x-y" => {
                self.inner.h6(q);
                Ok(None)
            }
            "F" | "F1" => {
                self.inner.f(q);
                Ok(None)
            }
            "Fdg" | "F1d" | "F1dg" => {
                self.inner.fdg(q);
                Ok(None)
            }
            "F2" => {
                self.inner.f2(q);
                Ok(None)
            }
            "F2dg" | "F2d" => {
                self.inner.f2dg(q);
                Ok(None)
            }
            "F3" => {
                self.inner.f3(q);
                Ok(None)
            }
            "F3dg" | "F3d" => {
                self.inner.f3dg(q);
                Ok(None)
            }
            "F4" => {
                self.inner.f4(q);
                Ok(None)
            }
            "F4dg" | "F4d" => {
                self.inner.f4dg(q);
                Ok(None)
            }
            "Q" | "SX" | "SqrtX" => {
                self.inner.sx(q);
                Ok(None)
            }
            "Qd" | "SXdg" | "SqrtXd" | "SqrtXdg" => {
                self.inner.sxdg(q);
                Ok(None)
            }
            "R" | "SY" | "SqrtY" => {
                self.inner.sy(q);
                Ok(None)
            }
            "Rd" | "SYdg" | "SqrtYd" | "SqrtYdg" => {
                self.inner.sydg(q);
                Ok(None)
            }
            "S" | "SZ" | "SqrtZ" => {
                self.inner.sz(q);
                Ok(None)
            }
            "Sd" | "SZdg" | "SqrtZd" | "SqrtZdg" => {
                self.inner.szdg(q);
                Ok(None)
            }
            "RX" => {
                let angle = extract_angle(params, "RX")?;
                self.inner
                    .try_rx(angle, q)
                    .map_err(pyo3::exceptions::PyValueError::new_err)?;
                Ok(None)
            }
            "RY" => {
                let angle = extract_angle(params, "RY")?;
                self.inner
                    .try_ry(angle, q)
                    .map_err(pyo3::exceptions::PyValueError::new_err)?;
                Ok(None)
            }
            "RZ" => {
                let angle = extract_angle(params, "RZ")?;
                self.inner
                    .try_rz(angle, q)
                    .map_err(pyo3::exceptions::PyValueError::new_err)?;
                Ok(None)
            }
            "RXY1Q" | "R1XY" => {
                let angles = extract_angles(params, "RXY1Q", 2)?;
                self.inner
                    .try_rxy1q(angles[0], angles[1], q)
                    .map_err(pyo3::exceptions::PyValueError::new_err)?;
                Ok(None)
            }
            "U" => {
                let angles = extract_angles(params, "U", 3)?;
                self.inner
                    .try_u(angles[0], angles[1], angles[2], q)
                    .map_err(pyo3::exceptions::PyValueError::new_err)?;
                Ok(None)
            }
            "PZ" | "PZForced" | "Init" | "Init +Z" | "init |0>" | "leak" | "leak |0>"
            | "unleak |0>" => {
                self.inner.pz(q);
                Ok(None)
            }
            "PNZ" | "Init -Z" | "init |1>" | "leak |1>" | "unleak |1>" => {
                self.inner.pnz(q);
                Ok(None)
            }
            "PX" | "Init +X" | "init |+>" => {
                self.inner.px(q);
                Ok(None)
            }
            "PNX" | "Init -X" | "init |->" => {
                self.inner.pnx(q);
                Ok(None)
            }
            "PY" | "Init +Y" | "init |+i>" => {
                self.inner.py(q);
                Ok(None)
            }
            "PNY" | "Init -Y" | "init |-i>" => {
                self.inner.pny(q);
                Ok(None)
            }
            "MZ" | "MZForced" | "Measure" | "measure Z" | "Measure +Z" => {
                Ok(Some(u8::from(self.inner.mz(q)[0].outcome)))
            }
            "MX" | "Measure +X" | "measure X" => Ok(Some(u8::from(self.inner.mx(q)[0].outcome))),
            "MY" | "Measure +Y" | "measure Y" => Ok(Some(u8::from(self.inner.my(q)[0].outcome))),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Unsupported single-qubit gate",
            )),
        }
    }

    #[pyo3(signature = (symbol, location, params=None))]
    fn run_2q_gate(
        &mut self,
        symbol: &str,
        location: &Bound<'_, PyTuple>,
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Option<u8>> {
        if location.len() != 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Two-qubit gate requires exactly 2 qubit locations",
            ));
        }

        let q1: usize = location.get_item(0)?.extract()?;
        let q2: usize = location.get_item(1)?.extract()?;
        let pair = &[(QubitId(q1), QubitId(q2))];

        match symbol {
            "CX" | "CNOT" => {
                self.inner.cx(pair);
                Ok(None)
            }
            "CY" => {
                self.inner.cy(pair);
                Ok(None)
            }
            "CZ" => {
                self.inner.cz(pair);
                Ok(None)
            }
            "SXX" | "SqrtXX" | "MS" | "MSXX" => {
                self.inner.sxx(pair);
                Ok(None)
            }
            "SXXdg" | "SqrtXXd" | "SqrtXXdg" => {
                self.inner.sxxdg(pair);
                Ok(None)
            }
            "SYY" | "SqrtYY" => {
                self.inner.syy(pair);
                Ok(None)
            }
            "SYYdg" | "SqrtYYd" | "SqrtYYdg" => {
                self.inner.syydg(pair);
                Ok(None)
            }
            "SZZ" | "SqrtZZ" => {
                self.inner.szz(pair);
                Ok(None)
            }
            "SZZdg" | "SqrtZZd" | "SqrtZZdg" => {
                self.inner.szzdg(pair);
                Ok(None)
            }
            "SWAP" => {
                self.inner.swap(pair);
                Ok(None)
            }
            "G" | "G2" => {
                self.inner.g(pair);
                Ok(None)
            }
            "Gdg" => {
                self.inner.gdg(pair);
                Ok(None)
            }
            "ISWAP" => {
                self.inner.iswap(pair);
                Ok(None)
            }
            "ISWAPdg" => {
                self.inner.iswapdg(pair);
                Ok(None)
            }
            "RXX" => {
                let angle = extract_angle(params, "RXX")?;
                self.inner
                    .try_rxx(angle, pair)
                    .map_err(pyo3::exceptions::PyValueError::new_err)?;
                Ok(None)
            }
            "RYY" => {
                let angle = extract_angle(params, "RYY")?;
                self.inner
                    .try_ryy(angle, pair)
                    .map_err(pyo3::exceptions::PyValueError::new_err)?;
                Ok(None)
            }
            "RZZ" => {
                let angle = extract_angle(params, "RZZ")?;
                self.inner
                    .try_rzz(angle, pair)
                    .map_err(pyo3::exceptions::PyValueError::new_err)?;
                Ok(None)
            }
            "RXXRYYRZZ" | "RZZRYYRXX" | "R2XXYYZZ" | "RXXYYZZ" => {
                let angles = extract_angles(params, "RXXRYYRZZ", 3)?;
                self.inner
                    .try_rxxryyrzz(angles[0], angles[1], angles[2], pair)
                    .map_err(pyo3::exceptions::PyValueError::new_err)?;
                Ok(None)
            }
            "II" => Ok(None),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Unsupported two-qubit gate",
            )),
        }
    }

    #[pyo3(signature = (symbol, location, params=None))]
    fn run_gate_internal(
        &mut self,
        symbol: &str,
        location: &Bound<'_, PyTuple>,
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Option<u8>> {
        match location.len() {
            1 => self.run_1q_gate(symbol, location.get_item(0)?.extract()?, params),
            2 => self.run_2q_gate(symbol, location, params),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Gate location must be specified for either 1 or 2 qubits",
            )),
        }
    }

    #[pyo3(signature = (symbol, locations, **params))]
    fn run_gate(
        &mut self,
        symbol: &str,
        locations: &Bound<'_, PyAny>,
        params: Option<&Bound<'_, PyDict>>,
        py: Python<'_>,
    ) -> PyResult<Py<PyDict>> {
        self.run_gate_highlevel(symbol, locations, params, py)
    }

    #[pyo3(signature = (symbol, locations, **params))]
    fn run_gate_highlevel(
        &mut self,
        symbol: &str,
        locations: &Bound<'_, PyAny>,
        params: Option<&Bound<'_, PyDict>>,
        py: Python<'_>,
    ) -> PyResult<Py<PyDict>> {
        let output = PyDict::new(py);

        if matches!(symbol, "force output" | "check" | "measure") {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Unsupported gate",
            ));
        }

        if let Some(p) = params
            && let Ok(Some(simulate_gate)) = p.get_item("simulate_gate")
            && let Ok(false) = simulate_gate.extract::<bool>()
        {
            return Ok(output.into());
        }

        let locations_set: Bound<PySet> = locations.clone().cast_into()?;
        if locations_set.is_empty() {
            return Ok(output.into());
        }

        let has_special_params = params.is_some_and(|p| !p.is_empty());
        if !has_special_params
            && let Some(result) = crate::simulator_utils::try_clifford_batch_dispatch(
                &mut self.inner,
                symbol,
                &locations_set,
                py,
            )?
        {
            return Ok(result);
        }

        for location in locations_set.iter() {
            let loc_tuple: Bound<'_, PyTuple> = if location.is_instance_of::<PyTuple>() {
                location.clone().cast_into()?
            } else {
                PyTuple::new(py, std::slice::from_ref(&location))?
            };
            if let Some(value) = self.run_gate_internal(symbol, &loc_tuple, params)? {
                output.set_item(location, value)?;
            }
        }

        Ok(output.into())
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

    /// Track X Pauli(s) on qubit(s)
    pub fn track_x(&mut self, qubits: Vec<usize>) {
        self.inner.track_x(&qubits);
    }

    /// Track Z Pauli(s) on qubit(s)
    pub fn track_z(&mut self, qubits: Vec<usize>) {
        self.inner.track_z(&qubits);
    }

    /// Track Y Pauli(s) on qubit(s)
    pub fn track_y(&mut self, qubits: Vec<usize>) {
        self.inner.track_y(&qubits);
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

    /// Apply Hadamard gate(s)
    pub fn h(&mut self, qubits: Vec<usize>) {
        let qs: Vec<QubitId> = qubits.into_iter().map(QubitId).collect();
        self.inner.h(&qs);
    }

    /// Apply S gate(s) (sqrt(Z))
    pub fn sz(&mut self, qubits: Vec<usize>) {
        let qs: Vec<QubitId> = qubits.into_iter().map(QubitId).collect();
        self.inner.sz(&qs);
    }

    /// Apply sqrt(X) gate(s)
    pub fn sx(&mut self, qubits: Vec<usize>) {
        let qs: Vec<QubitId> = qubits.into_iter().map(QubitId).collect();
        self.inner.sx(&qs);
    }

    /// Apply sqrt(Y) gate(s)
    pub fn sy(&mut self, qubits: Vec<usize>) {
        let qs: Vec<QubitId> = qubits.into_iter().map(QubitId).collect();
        self.inner.sy(&qs);
    }

    /// Apply CNOT/CX gate(s)
    pub fn cx(&mut self, pairs: Vec<(usize, usize)>) {
        let ps: Vec<(QubitId, QubitId)> = pairs
            .into_iter()
            .map(|(c, t)| (QubitId(c), QubitId(t)))
            .collect();
        self.inner.cx(&ps);
    }

    /// Apply CY gate(s)
    pub fn cy(&mut self, pairs: Vec<(usize, usize)>) {
        let ps: Vec<(QubitId, QubitId)> = pairs
            .into_iter()
            .map(|(c, t)| (QubitId(c), QubitId(t)))
            .collect();
        self.inner.cy(&ps);
    }

    /// Apply CZ gate(s)
    pub fn cz(&mut self, pairs: Vec<(usize, usize)>) {
        let ps: Vec<(QubitId, QubitId)> = pairs
            .into_iter()
            .map(|(c, t)| (QubitId(c), QubitId(t)))
            .collect();
        self.inner.cz(&ps);
    }

    /// Apply SWAP gate(s)
    pub fn swap(&mut self, pairs: Vec<(usize, usize)>) {
        let ps: Vec<(QubitId, QubitId)> = pairs
            .into_iter()
            .map(|(a, b)| (QubitId(a), QubitId(b)))
            .collect();
        self.inner.swap(&ps);
    }

    /// Measure in Z basis
    pub fn mz(&mut self, qubits: Vec<usize>) -> Vec<bool> {
        let qs: Vec<QubitId> = qubits.into_iter().map(QubitId).collect();
        self.inner.mz(&qs).into_iter().map(|r| r.outcome).collect()
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

    #[getter]
    fn bindings(slf: PyRef<'_, Self>) -> PyResult<crate::simulator_utils::GateBindingsDict> {
        let py = slf.py();
        let sim_obj: Py<PyAny> = slf.into_bound_py_any(py)?.unbind();
        Ok(crate::simulator_utils::GateBindingsDict::new(sim_obj))
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
