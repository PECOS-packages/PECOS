// Copyright 2025 The PECOS Developers
use crate::dtypes::AngleParam;
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
use pyo3::types::{PyDict, PyTuple};

/// The struct represents the Qulacs state-vector simulator exposed to Python
#[pyclass(name = "Qulacs")]
pub struct PyQulacs {
    inner: QulacsStateVec,
}

impl PyQulacs {
    /// Handle simple two-qubit gates that don't require parameters
    fn handle_simple_2q_gate(
        &mut self,
        symbol: &str,
        q1: usize,
        q2: usize,
    ) -> PyResult<Option<u8>> {
        let pair = &[(QubitId(q1), QubitId(q2))];
        match symbol {
            "CX" => {
                self.inner.cx(pair);
            }
            "CY" => {
                self.inner.cy(pair);
            }
            "CZ" => {
                self.inner.cz(pair);
            }
            "SWAP" => {
                self.inner.swap(pair);
            }
            "G" | "G2" => {
                self.inner.g(pair);
            }
            "SXX" => {
                self.inner.rxx(Angle64::QUARTER_TURN, pair);
            }
            "SXXdg" => {
                self.inner.rxx(-Angle64::QUARTER_TURN, pair);
            }
            "SYY" => {
                self.inner.ryy(Angle64::QUARTER_TURN, pair);
            }
            "SYYdg" => {
                self.inner.ryy(-Angle64::QUARTER_TURN, pair);
            }
            "SZZ" | "SqrtZZ" => {
                self.inner.rzz(Angle64::QUARTER_TURN, pair);
            }
            "SZZdg" => {
                self.inner.rzz(-Angle64::QUARTER_TURN, pair);
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Unknown simple two-qubit gate",
                ));
            }
        }
        Ok(None)
    }

    /// Helper method to extract angle parameter from dict
    fn extract_angle_param(params: &Bound<'_, PyDict>, gate_name: &str) -> PyResult<Angle64> {
        match params.get_item("angle") {
            Ok(Some(py_any)) => py_any.extract::<AngleParam>().map(|a| a.0).map_err(|_| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Expected a valid angle parameter for {gate_name} gate"
                ))
            }),
            Ok(None) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Angle parameter missing for {gate_name} gate"
            ))),
            Err(err) => Err(err),
        }
    }

    /// Helper method to extract angles parameter from dict
    fn extract_angles_param(
        params: &Bound<'_, PyDict>,
        gate_name: &str,
        expected_count: usize,
    ) -> PyResult<Vec<Angle64>> {
        match params.get_item("angles") {
            Ok(Some(py_any)) => {
                let angles = py_any.extract::<Vec<AngleParam>>().map_err(|_| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Expected valid angles parameter for {gate_name} gate"
                    ))
                })?;
                if angles.len() == expected_count {
                    Ok(angles.into_iter().map(|a| a.0).collect())
                } else {
                    Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "{gate_name} requires exactly {expected_count} angles"
                    )))
                }
            }
            Ok(None) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Angles parameter missing for {gate_name} gate"
            ))),
            Err(err) => Err(err),
        }
    }
}

#[pymethods]
impl PyQulacs {
    /// Creates a new Qulacs state-vector simulator with the specified number of qubits
    ///
    /// # Arguments
    /// * `num_qubits` - Number of qubits in the system
    /// * `seed` - Optional seed for the random number generator
    #[new]
    #[pyo3(signature = (num_qubits, seed=None))]
    pub fn new(num_qubits: usize, seed: Option<u64>) -> Self {
        PyQulacs {
            inner: match seed {
                Some(s) => QulacsStateVec::with_seed(num_qubits, s),
                None => QulacsStateVec::new(num_qubits),
            },
        }
    }

    /// Resets the quantum state to the all-zero state
    fn reset(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.inner.reset();
        slf
    }

    /// Executes a single-qubit gate based on the provided symbol and location
    ///
    /// `symbol`: The gate symbol (e.g., "X", "H", "Z")
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
        // Check bounds
        if location >= self.inner.num_qubits() {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "Qubit index {} out of range for {} qubits",
                location,
                self.inner.num_qubits()
            )));
        }

        let q = &[QubitId(location)];
        match symbol {
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
            "H" => {
                self.inner.h(q);
                Ok(None)
            }
            "SX" => {
                self.inner.sx(q);
                Ok(None)
            }
            "SXdg" => {
                self.inner.sxdg(q);
                Ok(None)
            }
            "SY" => {
                self.inner.sy(q);
                Ok(None)
            }
            "SYdg" => {
                self.inner.sydg(q);
                Ok(None)
            }
            "SZ" => {
                self.inner.sz(q);
                Ok(None)
            }
            "SZdg" => {
                self.inner.szdg(q);
                Ok(None)
            }
            "F" | "F1" => {
                // F gate is implemented via CliffordGateable trait
                self.inner.f(q);
                Ok(None)
            }
            "Fdg" | "F1dg" => {
                // F dagger is implemented via CliffordGateable trait
                self.inner.fdg(q);
                Ok(None)
            }
            "T" => {
                self.inner.t(q);
                Ok(None)
            }
            "Tdg" => {
                self.inner.tdg(q);
                Ok(None)
            }
            "RX" => {
                if let Some(params) = params {
                    match params.get_item("angle") {
                        Ok(Some(py_any)) => {
                            if let Ok(angle) = py_any.extract::<AngleParam>() {
                                self.inner.rx(angle.0, q);
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
                } else {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Angle parameter required for RX gate",
                    ));
                }
                Ok(None)
            }
            "RY" => {
                if let Some(params) = params {
                    match params.get_item("angle") {
                        Ok(Some(py_any)) => {
                            if let Ok(angle) = py_any.extract::<AngleParam>() {
                                self.inner.ry(angle.0, q);
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
                } else {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Angle parameter required for RY gate",
                    ));
                }
                Ok(None)
            }
            "RZ" => {
                if let Some(params) = params {
                    match params.get_item("angle") {
                        Ok(Some(py_any)) => {
                            if let Ok(angle) = py_any.extract::<AngleParam>() {
                                self.inner.rz(angle.0, q);
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
                } else {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Angle parameter required for RZ gate",
                    ));
                }
                Ok(None)
            }
            "R1XY" => {
                if let Some(params) = params {
                    match params.get_item("angles") {
                        Ok(Some(py_any)) => {
                            if let Ok(angles) = py_any.extract::<Vec<AngleParam>>() {
                                if angles.len() >= 2 {
                                    // R1XY = RZ(phi-pi/2) * RY(theta) * RZ(-phi+pi/2)
                                    // where theta = angles[0], phi = angles[1]
                                    let theta = angles[0].0;
                                    let phi = angles[1].0;
                                    let pi_half = Angle64::QUARTER_TURN;

                                    self.inner.rz(-phi + pi_half, q);
                                    self.inner.ry(theta, q);
                                    self.inner.rz(phi - pi_half, q);
                                } else {
                                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                        "R1XY requires at least 2 angles",
                                    ));
                                }
                            } else {
                                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                    "Expected a list of angles for R1XY gate",
                                ));
                            }
                        }
                        Ok(None) => {
                            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                "Angles parameter missing for R1XY gate",
                            ));
                        }
                        Err(err) => {
                            return Err(err);
                        }
                    }
                } else {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Angles parameter required for R1XY gate",
                    ));
                }
                Ok(None)
            }
            "H2" => {
                // H2 is implemented via CliffordGateable trait
                self.inner.h2(q);
                Ok(None)
            }
            "H3" => {
                // H3 is implemented via CliffordGateable trait
                self.inner.h3(q);
                Ok(None)
            }
            "H4" => {
                // H4 is implemented via CliffordGateable trait
                self.inner.h4(q);
                Ok(None)
            }
            "H5" => {
                // H5 is implemented via CliffordGateable trait
                self.inner.h5(q);
                Ok(None)
            }
            "H6" => {
                // H6 is implemented via CliffordGateable trait
                self.inner.h6(q);
                Ok(None)
            }
            "F2" => {
                // F2 is implemented via CliffordGateable trait
                self.inner.f2(q);
                Ok(None)
            }
            "F2dg" | "F2d" => {
                // F2dg is implemented via CliffordGateable trait
                self.inner.f2dg(q);
                Ok(None)
            }
            "F3" => {
                // F3 is implemented via CliffordGateable trait
                self.inner.f3(q);
                Ok(None)
            }
            "F3dg" | "F3d" => {
                // F3dg is implemented via CliffordGateable trait
                self.inner.f3dg(q);
                Ok(None)
            }
            "F4" => {
                // F4 is implemented via CliffordGateable trait
                self.inner.f4(q);
                Ok(None)
            }
            "F4dg" | "F4d" => {
                // F4dg is implemented via CliffordGateable trait
                self.inner.f4dg(q);
                Ok(None)
            }
            "MZ" => {
                let results = self.inner.mz(q);
                Ok(Some(u8::from(results[0].outcome)))
            }
            "MX" => {
                let results = self.inner.mx(q);
                Ok(Some(u8::from(results[0].outcome)))
            }
            "MY" => {
                let results = self.inner.my(q);
                Ok(Some(u8::from(results[0].outcome)))
            }
            "PZ" => {
                // Project to |0⟩ state using CliffordGateable trait
                self.inner.pz(q);
                Ok(None)
            }
            "PnZ" => {
                // Project to |1⟩ state using CliffordGateable trait
                self.inner.pnz(q);
                Ok(None)
            }
            "PX" => {
                // Project to |+⟩ state
                self.inner.prepare_computational_basis(0);
                self.inner.h(q);
                Ok(None)
            }
            "PnX" => {
                // Project to |-⟩ state
                self.inner.prepare_computational_basis(1 << location);
                self.inner.h(q);
                Ok(None)
            }
            "PY" => {
                // Project to |+i⟩ state
                self.inner.prepare_computational_basis(0);
                self.inner.h(q);
                self.inner.sz(q);
                Ok(None)
            }
            "PnY" => {
                // Project to |-i⟩ state
                self.inner.prepare_computational_basis(0);
                self.inner.h(q);
                self.inner.szdg(q);
                Ok(None)
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Unsupported single-qubit gate",
            )),
        }
    }

    /// Executes a two-qubit gate based on the provided symbol and locations
    ///
    /// `symbol`: The gate symbol (e.g., "CX", "CZ")
    /// `location`: A tuple specifying the two qubits to apply the gate to
    /// `params`: Optional parameters for parameterized gates
    ///
    /// Returns an optional result, usually `None` unless a measurement is performed
    #[pyo3(signature = (symbol, location, params))]
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

        // Check bounds
        let num_qubits = self.inner.num_qubits();
        if q1 >= num_qubits || q2 >= num_qubits {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "Qubit indices ({q1}, {q2}) out of range for {num_qubits} qubits"
            )));
        }

        let pair = &[(QubitId(q1), QubitId(q2))];
        match symbol {
            "CX" | "CY" | "CZ" | "SWAP" | "G" | "SXX" | "SXXdg" | "SYY" | "SYYdg" | "SZZ"
            | "SqrtZZ" | "SZZdg" | "G2" => self.handle_simple_2q_gate(symbol, q1, q2),
            "RZZ" => {
                let params = params.ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Angle parameter required for RZZ gate",
                    )
                })?;
                let angle = Self::extract_angle_param(params, "RZZ")?;
                self.inner.rzz(angle, pair);
                Ok(None)
            }
            "RXX" => {
                let params = params.ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Angle parameter required for RXX gate",
                    )
                })?;
                let angle = Self::extract_angle_param(params, "RXX")?;
                self.inner.rxx(angle, pair);
                Ok(None)
            }
            "RYY" => {
                let params = params.ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Angle parameter required for RYY gate",
                    )
                })?;
                let angle = Self::extract_angle_param(params, "RYY")?;
                self.inner.ryy(angle, pair);
                Ok(None)
            }
            "RXXRYYRZZ" | "RZZRYYRXX" => {
                let params = params.ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Angles parameter required for RXXRYYRZZ gate",
                    )
                })?;
                let angles = Self::extract_angles_param(params, "RXXRYYRZZ", 3)?;
                // Use the rxxryyrzz method from ArbitraryRotationGateable trait
                // angles[0] = theta (XX), angles[1] = phi (YY), angles[2] = lambda (ZZ)
                self.inner.rxxryyrzz(angles[0], angles[1], angles[2], pair);
                Ok(None)
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Unsupported two-qubit gate",
            )),
        }
    }

    /// Dispatches a gate to the appropriate handler based on the number of qubits specified
    ///
    /// `symbol`: The gate symbol
    /// `location`: A tuple specifying the qubits to apply the gate to
    /// `params`: Optional parameters for parameterized gates
    #[pyo3(signature = (symbol, location, params=None))]
    fn run_gate(
        &mut self,
        symbol: &str,
        location: &Bound<'_, PyTuple>,
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Option<u8>> {
        match location.len() {
            1 => {
                let qubit: usize = location.get_item(0)?.extract()?;
                self.run_1q_gate(symbol, qubit, params)
            }
            2 => self.run_2q_gate(symbol, location, params),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Gate location must be specified for either 1 or 2 qubits",
            )),
        }
    }

    /// Provides direct access to the current state vector as a Python property
    #[getter]
    fn vector(&self) -> Vec<(f64, f64)> {
        self.inner
            .state()
            .iter()
            .map(|complex| (complex.re, complex.im))
            .collect()
    }

    /// Get the number of qubits in the system
    #[getter]
    fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    /// Returns the probability of each computational basis state as a real-valued array.
    ///
    /// Each entry is |amplitude|^2 for the corresponding basis state.
    #[getter]
    fn probabilities(&self) -> Vec<f64> {
        self.inner
            .state()
            .iter()
            .map(num_complex::Complex::norm_sqr)
            .collect()
    }

    /// Get the probability of measuring a specific basis state
    fn probability(&self, basis_state: usize) -> f64 {
        self.inner.probability(basis_state)
    }

    /// Prepare the state as a specific computational basis state
    fn prepare_computational_basis(&mut self, basis_state: usize) {
        self.inner.prepare_computational_basis(basis_state);
    }

    /// Prepare all qubits in the |+⟩ state
    fn prepare_plus_state(&mut self) {
        self.inner.prepare_plus_state();
    }
}
