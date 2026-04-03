// Copyright 2024 The PECOS Developers
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
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

// Import the Rust types with renamed aliases to distinguish from Python wrapper types
// These are re-exported by pecos::prelude when the quest feature is enabled
use crate::prelude::{
    QuestDensityMatrix as RustQuestDensityMatrix, QuestStateVec as RustQuestStateVec,
};

/// The struct represents the `QuEST` state-vector simulator exposed to Python
#[pyclass]
pub struct QuestStateVec {
    inner: RustQuestStateVec,
}

#[pymethods]
impl QuestStateVec {
    /// Creates a new `QuEST` state-vector simulator with the specified number of qubits
    ///
    /// # Arguments
    /// * `num_qubits` - Number of qubits in the system
    /// * `seed` - Optional seed for the random number generator
    #[new]
    #[pyo3(signature = (num_qubits, seed=None))]
    pub fn new(num_qubits: usize, seed: Option<u64>) -> Self {
        QuestStateVec {
            inner: match seed {
                Some(s) => RustQuestStateVec::with_seed(num_qubits, s),
                None => RustQuestStateVec::new(num_qubits),
            },
        }
    }

    /// Returns the number of qubits in the simulator
    fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    /// Resets the quantum state to the all-zero state
    fn reset(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.inner.reset();
        slf
    }

    /// Prepares a computational basis state
    fn prepare_computational_basis(&mut self, index: usize) {
        self.inner.prepare_computational_basis(index);
    }

    /// Gets the probability of a computational basis state
    fn probability(&self, index: usize) -> f64 {
        self.inner.probability(index)
    }

    /// Gets the amplitude of a computational basis state as a complex number
    fn get_amplitude(&self, index: usize) -> (f64, f64) {
        let amp = self.inner.get_amplitude(index);
        (amp.re, amp.im)
    }

    /// Executes a single-qubit gate based on the provided symbol and location
    ///
    /// `symbol`: The gate symbol (e.g., "X", "H", "Z", "RX", "RY", "RZ")
    /// `location`: The qubit index to apply the gate to
    /// `params`: Optional parameters for parameterized gates
    ///
    /// Returns an optional result, usually `None` unless a measurement is performed
    #[allow(clippy::too_many_lines)]
    #[pyo3(signature = (symbol, location, params=None))]
    fn run_1q_gate(
        &mut self,
        symbol: &str,
        location: usize,
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Option<u8>> {
        match symbol {
            "X" => {
                self.inner.x(&[QubitId(location)]);
                Ok(None)
            }
            "Y" => {
                self.inner.y(&[QubitId(location)]);
                Ok(None)
            }
            "Z" => {
                self.inner.z(&[QubitId(location)]);
                Ok(None)
            }
            "H" => {
                self.inner.h(&[QubitId(location)]);
                Ok(None)
            }
            // Note: S and S† gates are not implemented in QuEST wrapper yet
            "RX" => {
                if let Some(params) = params {
                    match params.get_item("angle") {
                        Ok(Some(py_any)) => {
                            if let Ok(angle) = py_any.extract::<AngleParam>() {
                                self.inner.rx(angle.0, &[QubitId(location)]);
                            } else {
                                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                    "Expected a valid angle parameter for RX gate",
                                ));
                            }
                        }
                        Ok(None) => {
                            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                "Angle parameter missing for RX gate",
                            ));
                        }
                        Err(err) => {
                            return Err(err);
                        }
                    }
                }
                Ok(None)
            }
            "RY" => {
                if let Some(params) = params {
                    match params.get_item("angle") {
                        Ok(Some(py_any)) => {
                            if let Ok(angle) = py_any.extract::<AngleParam>() {
                                self.inner.ry(angle.0, &[QubitId(location)]);
                            } else {
                                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                    "Expected a valid angle parameter for RY gate",
                                ));
                            }
                        }
                        Ok(None) => {
                            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                "Angle parameter missing for RY gate",
                            ));
                        }
                        Err(err) => {
                            return Err(err);
                        }
                    }
                }
                Ok(None)
            }
            "RZ" => {
                if let Some(params) = params {
                    match params.get_item("angle") {
                        Ok(Some(py_any)) => {
                            if let Ok(angle) = py_any.extract::<AngleParam>() {
                                self.inner.rz(angle.0, &[QubitId(location)]);
                            } else {
                                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                    "Expected a valid angle parameter for RZ gate",
                                ));
                            }
                        }
                        Ok(None) => {
                            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                "Angle parameter missing for RZ gate",
                            ));
                        }
                        Err(err) => {
                            return Err(err);
                        }
                    }
                }
                Ok(None)
            }
            "MZ" => {
                let results = self.inner.mz(&[QubitId(location)]);
                Ok(Some(u8::from(results[0].outcome)))
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Unknown single-qubit gate: {symbol}"
            ))),
        }
    }

    /// Executes a two-qubit gate based on the provided symbol and locations
    ///
    /// `symbol`: The gate symbol (e.g., "CX", "CY", "CZ", "RXX", "RYY", "RZZ")
    /// `locations`: Tuple of (control, target) qubit indices
    /// `params`: Optional parameters for parameterized gates
    #[pyo3(signature = (symbol, locations, params=None))]
    fn run_2q_gate(
        &mut self,
        symbol: &str,
        locations: &Bound<'_, PyTuple>,
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        if locations.len() != 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Two-qubit gate requires exactly 2 qubit indices",
            ));
        }

        let control = locations.get_item(0)?.extract::<usize>()?;
        let target = locations.get_item(1)?.extract::<usize>()?;

        match symbol {
            "CX" | "CNOT" => {
                self.inner.cx(&[(QubitId(control), QubitId(target))]);
                Ok(())
            }
            "CY" => {
                self.inner.cy(&[(QubitId(control), QubitId(target))]);
                Ok(())
            }
            "CZ" => {
                self.inner.cz(&[(QubitId(control), QubitId(target))]);
                Ok(())
            }
            "RXX" => {
                if let Some(params) = params {
                    match params.get_item("angle") {
                        Ok(Some(py_any)) => {
                            if let Ok(angle) = py_any.extract::<AngleParam>() {
                                self.inner
                                    .rxx(angle.0, &[(QubitId(control), QubitId(target))]);
                                Ok(())
                            } else {
                                Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                    "Expected a valid angle parameter for RXX gate",
                                ))
                            }
                        }
                        Ok(None) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            "Angle parameter missing for RXX gate",
                        )),
                        Err(err) => Err(err),
                    }
                } else {
                    Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "RXX gate requires angle parameter",
                    ))
                }
            }
            "RYY" => {
                if let Some(params) = params {
                    match params.get_item("angle") {
                        Ok(Some(py_any)) => {
                            if let Ok(angle) = py_any.extract::<AngleParam>() {
                                self.inner
                                    .ryy(angle.0, &[(QubitId(control), QubitId(target))]);
                                Ok(())
                            } else {
                                Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                    "Expected a valid angle parameter for RYY gate",
                                ))
                            }
                        }
                        Ok(None) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            "Angle parameter missing for RYY gate",
                        )),
                        Err(err) => Err(err),
                    }
                } else {
                    Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "RYY gate requires angle parameter",
                    ))
                }
            }
            "RZZ" => {
                if let Some(params) = params {
                    match params.get_item("angle") {
                        Ok(Some(py_any)) => {
                            if let Ok(angle) = py_any.extract::<AngleParam>() {
                                self.inner
                                    .rzz(angle.0, &[(QubitId(control), QubitId(target))]);
                                Ok(())
                            } else {
                                Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                    "Expected a valid angle parameter for RZZ gate",
                                ))
                            }
                        }
                        Ok(None) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            "Angle parameter missing for RZZ gate",
                        )),
                        Err(err) => Err(err),
                    }
                } else {
                    Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "RZZ gate requires angle parameter",
                    ))
                }
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Unknown two-qubit gate: {symbol}"
            ))),
        }
    }

    /// Applies a T gate to the specified qubit
    fn t_gate(&mut self, location: usize) {
        self.inner.t(&[QubitId(location)]);
    }

    /// Applies a T-dagger gate to the specified qubit
    fn tdg_gate(&mut self, location: usize) {
        self.inner.tdg(&[QubitId(location)]);
    }

    /// Applies a square root of XX gate to two qubits
    fn sxx_gate(&mut self, control: usize, target: usize) {
        self.inner.sxx(&[(QubitId(control), QubitId(target))]);
    }

    /// Applies a square root of YY gate to two qubits
    fn syy_gate(&mut self, control: usize, target: usize) {
        self.inner.syy(&[(QubitId(control), QubitId(target))]);
    }

    /// Applies a square root of ZZ gate to two qubits
    fn szz_gate(&mut self, control: usize, target: usize) {
        self.inner.szz(&[(QubitId(control), QubitId(target))]);
    }
    /// Applies an R1XY gate to the specified qubit
    fn r1xy_gate(&mut self, theta: AngleParam, phi: AngleParam, location: usize) {
        self.inner.r1xy(theta.0, phi.0, &[QubitId(location)]);
    }

    /// Applies RXXRYYRZZ gate (combination of RXX, RYY, RZZ) to two qubits
    /// NOTE: This uses the trait implementation which may differ from `StateVec`'s decomposition
    /// For consistency with `StateVec` tests, the Python bindings use manual decompositions
    fn rxxryyrzz_gate(
        &mut self,
        theta: AngleParam,
        phi: AngleParam,
        lambda: AngleParam,
        q1: usize,
        q2: usize,
    ) {
        // Use the trait implementation directly
        // Note: The trait's rxxryyrzz has a different decomposition than StateVec's
        // which is why Python bindings use manual decompositions for RXX, RYY, RZZ
        self.inner
            .rxxryyrzz(theta.0, phi.0, lambda.0, &[(QubitId(q1), QubitId(q2))]);
    }

    /// Applies a SWAP gate to two qubits
    fn swap_gate(&mut self, control: usize, target: usize) {
        self.inner.swap(&[(QubitId(control), QubitId(target))]);
    }

    /// Applies H2 gate variant
    fn h2_gate(&mut self, location: usize) {
        self.inner.h2(&[QubitId(location)]);
    }

    /// Applies H3 gate variant
    fn h3_gate(&mut self, location: usize) {
        self.inner.h3(&[QubitId(location)]);
    }

    /// Applies H4 gate variant
    fn h4_gate(&mut self, location: usize) {
        self.inner.h4(&[QubitId(location)]);
    }

    /// Applies H5 gate variant
    fn h5_gate(&mut self, location: usize) {
        self.inner.h5(&[QubitId(location)]);
    }

    /// Applies H6 gate variant
    fn h6_gate(&mut self, location: usize) {
        self.inner.h6(&[QubitId(location)]);
    }

    /// Measures in the X basis
    fn mx_gate(&mut self, location: usize) -> u8 {
        let results = self.inner.mx(&[QubitId(location)]);
        u8::from(results[0].outcome)
    }

    /// Measures in the Y basis
    fn my_gate(&mut self, location: usize) -> u8 {
        let results = self.inner.my(&[QubitId(location)]);
        u8::from(results[0].outcome)
    }

    /// Applies a square root of X gate to the specified qubit
    fn sx_gate(&mut self, location: usize) {
        self.inner.sx(&[QubitId(location)]);
    }

    /// Applies a square root of X-dagger gate to the specified qubit
    fn sxdg_gate(&mut self, location: usize) {
        self.inner.sxdg(&[QubitId(location)]);
    }

    /// Applies a square root of Y gate to the specified qubit
    fn sy_gate(&mut self, location: usize) {
        self.inner.sy(&[QubitId(location)]);
    }

    /// Applies a square root of Y-dagger gate to the specified qubit
    fn sydg_gate(&mut self, location: usize) {
        self.inner.sydg(&[QubitId(location)]);
    }

    /// Applies a square root of Z gate to the specified qubit
    fn sz_gate(&mut self, location: usize) {
        self.inner.sz(&[QubitId(location)]);
    }

    /// Applies a square root of Z-dagger gate to the specified qubit
    fn szdg_gate(&mut self, location: usize) {
        self.inner.szdg(&[QubitId(location)]);
    }

    /// String representation of the simulator
    fn __repr__(&self) -> String {
        format!("QuestStateVec(num_qubits={})", self.inner.num_qubits())
    }
}

/// The struct represents the `QuEST` density matrix simulator exposed to Python
#[pyclass]
pub struct QuestDensityMatrix {
    inner: RustQuestDensityMatrix,
}

#[pymethods]
impl QuestDensityMatrix {
    /// Creates a new `QuEST` density matrix simulator with the specified number of qubits
    ///
    /// # Arguments
    /// * `num_qubits` - Number of qubits in the system
    /// * `seed` - Optional seed for the random number generator
    #[new]
    #[pyo3(signature = (num_qubits, seed=None))]
    pub fn new(num_qubits: usize, seed: Option<u64>) -> Self {
        QuestDensityMatrix {
            inner: match seed {
                Some(s) => RustQuestDensityMatrix::with_seed(num_qubits, s),
                None => RustQuestDensityMatrix::new(num_qubits),
            },
        }
    }

    /// Returns the number of qubits in the simulator
    fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    /// Resets the quantum state to the all-zero state
    fn reset(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.inner.reset();
        slf
    }

    /// Prepares a computational basis state
    fn prepare_computational_basis(&mut self, index: usize) {
        self.inner.prepare_computational_basis(index);
    }

    /// Gets the probability of a computational basis state
    fn probability(&self, index: usize) -> f64 {
        self.inner.probability(index)
    }

    // Note: calculate_purity is not exposed in QuEST wrapper yet

    /// Executes a single-qubit gate based on the provided symbol and location
    ///
    /// `symbol`: The gate symbol (e.g., "X", "H", "Z", "RX", "RY", "RZ")
    /// `location`: The qubit index to apply the gate to
    /// `params`: Optional parameters for parameterized gates
    ///
    /// Returns an optional result, usually `None` unless a measurement is performed
    #[allow(clippy::too_many_lines)]
    #[pyo3(signature = (symbol, location, params=None))]
    fn run_1q_gate(
        &mut self,
        symbol: &str,
        location: usize,
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Option<u8>> {
        match symbol {
            "X" => {
                self.inner.x(&[QubitId(location)]);
                Ok(None)
            }
            "Y" => {
                self.inner.y(&[QubitId(location)]);
                Ok(None)
            }
            "Z" => {
                self.inner.z(&[QubitId(location)]);
                Ok(None)
            }
            "H" => {
                self.inner.h(&[QubitId(location)]);
                Ok(None)
            }
            // Note: S and S† gates are not implemented in QuEST wrapper yet
            "RX" => {
                if let Some(params) = params {
                    match params.get_item("angle") {
                        Ok(Some(py_any)) => {
                            if let Ok(angle) = py_any.extract::<AngleParam>() {
                                self.inner.rx(angle.0, &[QubitId(location)]);
                            } else {
                                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                    "Expected a valid angle parameter for RX gate",
                                ));
                            }
                        }
                        Ok(None) => {
                            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                "Angle parameter missing for RX gate",
                            ));
                        }
                        Err(err) => {
                            return Err(err);
                        }
                    }
                }
                Ok(None)
            }
            "RY" => {
                if let Some(params) = params {
                    match params.get_item("angle") {
                        Ok(Some(py_any)) => {
                            if let Ok(angle) = py_any.extract::<AngleParam>() {
                                self.inner.ry(angle.0, &[QubitId(location)]);
                            } else {
                                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                    "Expected a valid angle parameter for RY gate",
                                ));
                            }
                        }
                        Ok(None) => {
                            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                "Angle parameter missing for RY gate",
                            ));
                        }
                        Err(err) => {
                            return Err(err);
                        }
                    }
                }
                Ok(None)
            }
            "RZ" => {
                if let Some(params) = params {
                    match params.get_item("angle") {
                        Ok(Some(py_any)) => {
                            if let Ok(angle) = py_any.extract::<AngleParam>() {
                                self.inner.rz(angle.0, &[QubitId(location)]);
                            } else {
                                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                    "Expected a valid angle parameter for RZ gate",
                                ));
                            }
                        }
                        Ok(None) => {
                            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                "Angle parameter missing for RZ gate",
                            ));
                        }
                        Err(err) => {
                            return Err(err);
                        }
                    }
                }
                Ok(None)
            }
            "MZ" => {
                let results = self.inner.mz(&[QubitId(location)]);
                Ok(Some(u8::from(results[0].outcome)))
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Unknown single-qubit gate: {symbol}"
            ))),
        }
    }

    /// Executes a two-qubit gate based on the provided symbol and locations
    ///
    /// `symbol`: The gate symbol (e.g., "CX", "CY", "CZ", "RXX", "RYY", "RZZ")
    /// `locations`: Tuple of (control, target) qubit indices
    /// `params`: Optional parameters for parameterized gates
    #[pyo3(signature = (symbol, locations, params=None))]
    fn run_2q_gate(
        &mut self,
        symbol: &str,
        locations: &Bound<'_, PyTuple>,
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        if locations.len() != 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Two-qubit gate requires exactly 2 qubit indices",
            ));
        }

        let control = locations.get_item(0)?.extract::<usize>()?;
        let target = locations.get_item(1)?.extract::<usize>()?;

        match symbol {
            "CX" | "CNOT" => {
                self.inner.cx(&[(QubitId(control), QubitId(target))]);
                Ok(())
            }
            "CY" => {
                self.inner.cy(&[(QubitId(control), QubitId(target))]);
                Ok(())
            }
            "CZ" => {
                self.inner.cz(&[(QubitId(control), QubitId(target))]);
                Ok(())
            }
            "RXX" => {
                if let Some(params) = params {
                    match params.get_item("angle") {
                        Ok(Some(py_any)) => {
                            if let Ok(angle) = py_any.extract::<AngleParam>() {
                                self.inner
                                    .rxx(angle.0, &[(QubitId(control), QubitId(target))]);
                                Ok(())
                            } else {
                                Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                    "Expected a valid angle parameter for RXX gate",
                                ))
                            }
                        }
                        Ok(None) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            "Angle parameter missing for RXX gate",
                        )),
                        Err(err) => Err(err),
                    }
                } else {
                    Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "RXX gate requires angle parameter",
                    ))
                }
            }
            "RYY" => {
                if let Some(params) = params {
                    match params.get_item("angle") {
                        Ok(Some(py_any)) => {
                            if let Ok(angle) = py_any.extract::<AngleParam>() {
                                self.inner
                                    .ryy(angle.0, &[(QubitId(control), QubitId(target))]);
                                Ok(())
                            } else {
                                Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                    "Expected a valid angle parameter for RYY gate",
                                ))
                            }
                        }
                        Ok(None) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            "Angle parameter missing for RYY gate",
                        )),
                        Err(err) => Err(err),
                    }
                } else {
                    Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "RYY gate requires angle parameter",
                    ))
                }
            }
            "RZZ" => {
                if let Some(params) = params {
                    match params.get_item("angle") {
                        Ok(Some(py_any)) => {
                            if let Ok(angle) = py_any.extract::<AngleParam>() {
                                self.inner
                                    .rzz(angle.0, &[(QubitId(control), QubitId(target))]);
                                Ok(())
                            } else {
                                Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                    "Expected a valid angle parameter for RZZ gate",
                                ))
                            }
                        }
                        Ok(None) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            "Angle parameter missing for RZZ gate",
                        )),
                        Err(err) => Err(err),
                    }
                } else {
                    Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "RZZ gate requires angle parameter",
                    ))
                }
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Unknown two-qubit gate: {symbol}"
            ))),
        }
    }

    /// Applies a T gate to the specified qubit
    fn t_gate(&mut self, location: usize) {
        self.inner.t(&[QubitId(location)]);
    }

    /// Applies a T-dagger gate to the specified qubit
    fn tdg_gate(&mut self, location: usize) {
        self.inner.tdg(&[QubitId(location)]);
    }

    /// Applies a square root of XX gate to two qubits
    fn sxx_gate(&mut self, control: usize, target: usize) {
        self.inner.sxx(&[(QubitId(control), QubitId(target))]);
    }

    /// Applies a square root of YY gate to two qubits
    fn syy_gate(&mut self, control: usize, target: usize) {
        self.inner.syy(&[(QubitId(control), QubitId(target))]);
    }

    /// Applies a square root of ZZ gate to two qubits
    fn szz_gate(&mut self, control: usize, target: usize) {
        self.inner.szz(&[(QubitId(control), QubitId(target))]);
    }
    /// Applies an R1XY gate to the specified qubit
    fn r1xy_gate(&mut self, theta: AngleParam, phi: AngleParam, location: usize) {
        self.inner.r1xy(theta.0, phi.0, &[QubitId(location)]);
    }

    /// Applies RXXRYYRZZ gate (combination of RXX, RYY, RZZ) to two qubits
    /// NOTE: This uses the trait implementation which may differ from `StateVec`'s decomposition
    /// For consistency with `StateVec` tests, the Python bindings use manual decompositions
    fn rxxryyrzz_gate(
        &mut self,
        theta: AngleParam,
        phi: AngleParam,
        lambda: AngleParam,
        q1: usize,
        q2: usize,
    ) {
        // Use the trait implementation directly
        // Note: The trait's rxxryyrzz has a different decomposition than StateVec's
        // which is why Python bindings use manual decompositions for RXX, RYY, RZZ
        self.inner
            .rxxryyrzz(theta.0, phi.0, lambda.0, &[(QubitId(q1), QubitId(q2))]);
    }

    /// Applies a SWAP gate to two qubits
    fn swap_gate(&mut self, control: usize, target: usize) {
        self.inner.swap(&[(QubitId(control), QubitId(target))]);
    }

    /// Applies H2 gate variant
    fn h2_gate(&mut self, location: usize) {
        self.inner.h2(&[QubitId(location)]);
    }

    /// Applies H3 gate variant
    fn h3_gate(&mut self, location: usize) {
        self.inner.h3(&[QubitId(location)]);
    }

    /// Applies H4 gate variant
    fn h4_gate(&mut self, location: usize) {
        self.inner.h4(&[QubitId(location)]);
    }

    /// Applies H5 gate variant
    fn h5_gate(&mut self, location: usize) {
        self.inner.h5(&[QubitId(location)]);
    }

    /// Applies H6 gate variant
    fn h6_gate(&mut self, location: usize) {
        self.inner.h6(&[QubitId(location)]);
    }

    /// Measures in the X basis
    fn mx_gate(&mut self, location: usize) -> u8 {
        let results = self.inner.mx(&[QubitId(location)]);
        u8::from(results[0].outcome)
    }

    /// Measures in the Y basis
    fn my_gate(&mut self, location: usize) -> u8 {
        let results = self.inner.my(&[QubitId(location)]);
        u8::from(results[0].outcome)
    }

    /// Applies a square root of X gate to the specified qubit
    fn sx_gate(&mut self, location: usize) {
        self.inner.sx(&[QubitId(location)]);
    }

    /// Applies a square root of X-dagger gate to the specified qubit
    fn sxdg_gate(&mut self, location: usize) {
        self.inner.sxdg(&[QubitId(location)]);
    }

    /// Applies a square root of Y gate to the specified qubit
    fn sy_gate(&mut self, location: usize) {
        self.inner.sy(&[QubitId(location)]);
    }

    /// Applies a square root of Y-dagger gate to the specified qubit
    fn sydg_gate(&mut self, location: usize) {
        self.inner.sydg(&[QubitId(location)]);
    }

    /// Applies a square root of Z gate to the specified qubit
    fn sz_gate(&mut self, location: usize) {
        self.inner.sz(&[QubitId(location)]);
    }

    /// Applies a square root of Z-dagger gate to the specified qubit
    fn szdg_gate(&mut self, location: usize) {
        self.inner.szdg(&[QubitId(location)]);
    }

    /// String representation of the simulator
    fn __repr__(&self) -> String {
        format!("QuestDensityMatrix(num_qubits={})", self.inner.num_qubits())
    }
}
