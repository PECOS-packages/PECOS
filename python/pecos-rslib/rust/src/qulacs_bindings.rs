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

use pecos_qsim::{ArbitraryRotationGateable, CliffordGateable, QuantumSimulator};
use pecos_qulacs::QulacsStateVec;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

/// The struct represents the Qulacs state-vector simulator exposed to Python
#[pyclass]
pub struct RsQulacs {
    inner: QulacsStateVec,
}

#[pymethods]
impl RsQulacs {
    /// Creates a new Qulacs state-vector simulator with the specified number of qubits
    ///
    /// # Arguments
    /// * `num_qubits` - Number of qubits in the system
    /// * `seed` - Optional seed for the random number generator
    #[new]
    #[pyo3(signature = (num_qubits, seed=None))]
    pub fn new(num_qubits: usize, seed: Option<u64>) -> Self {
        RsQulacs {
            inner: match seed {
                Some(s) => QulacsStateVec::with_seed(num_qubits, s),
                None => QulacsStateVec::new(num_qubits),
            },
        }
    }

    /// Resets the quantum state to the all-zero state
    fn reset(&mut self) {
        self.inner.reset();
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
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                format!("Qubit index {} out of range for {} qubits", location, self.inner.num_qubits())
            ));
        }
        
        match symbol {
            "X" => {
                self.inner.x(location);
                Ok(None)
            }
            "Y" => {
                self.inner.y(location);
                Ok(None)
            }
            "Z" => {
                self.inner.z(location);
                Ok(None)
            }
            "H" => {
                self.inner.h(location);
                Ok(None)
            }
            "SX" => {
                self.inner.sx(location);
                Ok(None)
            }
            "SXdg" => {
                self.inner.sxdg(location);
                Ok(None)
            }
            "SY" => {
                self.inner.sy(location);
                Ok(None)
            }
            "SYdg" => {
                self.inner.sydg(location);
                Ok(None)
            }
            "SZ" => {
                self.inner.sz(location);
                Ok(None)
            }
            "SZdg" => {
                self.inner.szdg(location);
                Ok(None)
            }
            "F" | "F1" => {
                // F gate is implemented via CliffordGateable trait
                self.inner.f(location);
                Ok(None)
            }
            "Fdg" | "F1dg" => {
                // F dagger is implemented via CliffordGateable trait
                self.inner.fdg(location);
                Ok(None)
            }
            "T" => {
                self.inner.t(location);
                Ok(None)
            }
            "Tdg" => {
                self.inner.tdg(location);
                Ok(None)
            }
            "RX" => {
                if let Some(params) = params {
                    match params.get_item("angle") {
                        Ok(Some(py_any)) => {
                            if let Ok(angle) = py_any.extract::<f64>() {
                                self.inner.rx(angle, location);
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
                            if let Ok(angle) = py_any.extract::<f64>() {
                                self.inner.ry(angle, location);
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
                            if let Ok(angle) = py_any.extract::<f64>() {
                                self.inner.rz(angle, location);
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
                            if let Ok(angles) = py_any.extract::<Vec<f64>>() {
                                if angles.len() >= 2 {
                                    // R1XY = RZ(phi-pi/2) * RY(theta) * RZ(-phi+pi/2)
                                    // where theta = angles[0], phi = angles[1]
                                    let theta = angles[0];
                                    let phi = angles[1];
                                    let pi_half = std::f64::consts::PI / 2.0;
                                    
                                    self.inner.rz(-phi + pi_half, location);
                                    self.inner.ry(theta, location);
                                    self.inner.rz(phi - pi_half, location);
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
                self.inner.h2(location);
                Ok(None)
            }
            "H3" => {
                // H3 is implemented via CliffordGateable trait
                self.inner.h3(location);
                Ok(None)
            }
            "H4" => {
                // H4 is implemented via CliffordGateable trait
                self.inner.h4(location);
                Ok(None)
            }
            "H5" => {
                // H5 is implemented via CliffordGateable trait
                self.inner.h5(location);
                Ok(None)
            }
            "H6" => {
                // H6 is implemented via CliffordGateable trait
                self.inner.h6(location);
                Ok(None)
            }
            "F2" => {
                // F2 is implemented via CliffordGateable trait
                self.inner.f2(location);
                Ok(None)
            }
            "F2dg" | "F2d" => {
                // F2dg is implemented via CliffordGateable trait
                self.inner.f2dg(location);
                Ok(None)
            }
            "F3" => {
                // F3 is implemented via CliffordGateable trait
                self.inner.f3(location);
                Ok(None)
            }
            "F3dg" | "F3d" => {
                // F3dg is implemented via CliffordGateable trait
                self.inner.f3dg(location);
                Ok(None)
            }
            "F4" => {
                // F4 is implemented via CliffordGateable trait
                self.inner.f4(location);
                Ok(None)
            }
            "F4dg" | "F4d" => {
                // F4dg is implemented via CliffordGateable trait
                self.inner.f4dg(location);
                Ok(None)
            }
            "MZ" => {
                let result = self.inner.mz(location);
                Ok(Some(u8::from(result.outcome)))
            }
            "MX" => {
                let result = self.inner.mx(location);
                Ok(Some(u8::from(result.outcome)))
            }
            "MY" => {
                let result = self.inner.my(location);
                Ok(Some(u8::from(result.outcome)))
            }
            "PZ" => {
                // Project to |0⟩ state using CliffordGateable trait
                self.inner.pz(location);
                Ok(None)
            }
            "PnZ" => {
                // Project to |1⟩ state using CliffordGateable trait
                self.inner.pnz(location);
                Ok(None)
            }
            "PX" => {
                // Project to |+⟩ state
                self.inner.prepare_computational_basis(0);
                self.inner.h(location);
                Ok(None)
            }
            "PnX" => {
                // Project to |-⟩ state
                self.inner.prepare_computational_basis(1 << location);
                self.inner.h(location);
                Ok(None)
            }
            "PY" => {
                // Project to |+i⟩ state
                self.inner.prepare_computational_basis(0);
                self.inner.h(location);
                self.inner.sz(location);
                Ok(None)
            }
            "PnY" => {
                // Project to |-i⟩ state
                self.inner.prepare_computational_basis(0);
                self.inner.h(location);
                self.inner.szdg(location);
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
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                format!("Qubit indices ({}, {}) out of range for {} qubits", q1, q2, num_qubits)
            ));
        }

        match symbol {
            "CX" => {
                self.inner.cx(q1, q2);
                Ok(None)
            }
            "CY" => {
                self.inner.cy(q1, q2);
                Ok(None)
            }
            "CZ" => {
                self.inner.cz(q1, q2);
                Ok(None)
            }
            "SWAP" => {
                self.inner.swap(q1, q2);
                Ok(None)
            }
            "G" => {
                // G gate is implemented via CliffordGateable trait
                self.inner.g(q1, q2);
                Ok(None)
            }
            "RZZ" => {
                if let Some(params) = params {
                    match params.get_item("angle") {
                        Ok(Some(py_any)) => {
                            if let Ok(angle) = py_any.extract::<f64>() {
                                self.inner.rzz(angle, q1, q2);
                            } else {
                                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                    "Expected a valid angle parameter for RZZ gate",
                                ));
                            }
                        }
                        Ok(None) => {
                            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                "Angle parameter missing for RZZ gate",
                            ));
                        }
                        Err(err) => {
                            return Err(err);
                        }
                    }
                } else {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Angle parameter required for RZZ gate",
                    ));
                }
                Ok(None)
            }
            "RXX" => {
                if let Some(params) = params {
                    match params.get_item("angle") {
                        Ok(Some(py_any)) => {
                            if let Ok(angle) = py_any.extract::<f64>() {
                                self.inner.rxx(angle, q1, q2);
                            } else {
                                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                    "Expected a valid angle parameter for RXX gate",
                                ));
                            }
                        }
                        Ok(None) => {
                            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                "Angle parameter missing for RXX gate",
                            ));
                        }
                        Err(err) => {
                            return Err(err);
                        }
                    }
                } else {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Angle parameter required for RXX gate",
                    ));
                }
                Ok(None)
            }
            "RYY" => {
                if let Some(params) = params {
                    match params.get_item("angle") {
                        Ok(Some(py_any)) => {
                            if let Ok(angle) = py_any.extract::<f64>() {
                                self.inner.ryy(angle, q1, q2);
                            } else {
                                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                    "Expected a valid angle parameter for RYY gate",
                                ));
                            }
                        }
                        Ok(None) => {
                            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                "Angle parameter missing for RYY gate",
                            ));
                        }
                        Err(err) => {
                            return Err(err);
                        }
                    }
                } else {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Angle parameter required for RYY gate",
                    ));
                }
                Ok(None)
            }
            "SXX" => {
                // Use the CliffordGateable trait method if available, else use RXX
                // Since we have RXX which is more efficient, use it directly
                self.inner.rxx(std::f64::consts::FRAC_PI_2, q1, q2);
                Ok(None)
            }
            "SXXdg" => {
                // SXXdg = RXX(-π/2)
                self.inner.rxx(-std::f64::consts::FRAC_PI_2, q1, q2);
                Ok(None)
            }
            "SYY" => {
                // SYY = RYY(π/2)
                self.inner.ryy(std::f64::consts::FRAC_PI_2, q1, q2);
                Ok(None)
            }
            "SYYdg" => {
                // SYYdg = RYY(-π/2)
                self.inner.ryy(-std::f64::consts::FRAC_PI_2, q1, q2);
                Ok(None)
            }
            "SZZ" | "SqrtZZ" => {
                // SZZ = RZZ(π/2)
                self.inner.rzz(std::f64::consts::FRAC_PI_2, q1, q2);
                Ok(None)
            }
            "SZZdg" => {
                // SZZdg = RZZ(-π/2)
                self.inner.rzz(-std::f64::consts::FRAC_PI_2, q1, q2);
                Ok(None)
            }
            "RZZRYYRXX" => {
                if let Some(params) = params {
                    match params.get_item("angles") {
                        Ok(Some(py_any)) => {
                            if let Ok(angles) = py_any.extract::<Vec<f64>>() {
                                if angles.len() == 3 {
                                    // R2XXYYZZ expects angles for XX, YY, ZZ in that order
                                    // But we apply gates as RZZ @ RYY @ RXX (reverse order)
                                    // So: RZZ(angles[2]), RYY(angles[1]), RXX(angles[0])
                                    self.inner.rzz(angles[2], q1, q2);
                                    self.inner.ryy(angles[1], q1, q2);
                                    self.inner.rxx(angles[0], q1, q2);
                                } else {
                                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                        "RZZRYYRXX requires exactly 3 angles",
                                    ));
                                }
                            } else {
                                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                    "Expected a list of angles for RZZRYYRXX gate",
                                ));
                            }
                        }
                        Ok(None) => {
                            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                "Angles parameter missing for RZZRYYRXX gate",
                            ));
                        }
                        Err(err) => {
                            return Err(err);
                        }
                    }
                } else {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Angles parameter required for RZZRYYRXX gate",
                    ));
                }
                Ok(None)
            }
            "G2" => {
                // G2 gate implementation (using decomposition)
                // G2 = H(q2) CX(q1,q2) H(q1) H(q2) CX(q1,q2) H(q2)
                self.inner.h(q2);
                self.inner.cx(q1, q2);
                self.inner.h(q1);
                self.inner.h(q2);
                self.inner.cx(q1, q2);
                self.inner.h(q2);
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