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

use pecos::prelude::*;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

/// The struct represents the state-vector simulator exposed to Python
#[pyclass]
pub struct RsStateVec {
    inner: StateVec,
}

#[pymethods]
impl RsStateVec {
    /// Creates a new state-vector simulator with the specified number of qubits
    ///
    /// # Arguments
    /// * `num_qubits` - Number of qubits in the system
    /// * `seed` - Optional seed for the random number generator
    #[new]
    #[pyo3(signature = (num_qubits, seed=None))]
    pub fn new(num_qubits: usize, seed: Option<u64>) -> Self {
        RsStateVec {
            inner: match seed {
                Some(s) => StateVec::with_seed(num_qubits, s),
                None => StateVec::new(num_qubits),
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
    /// `params`: Optional parameters for parameterized gates (currently unused here)
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
                }
                Ok(None)
            }
            "R1XY" => {
                if let Some(params) = params {
                    match params.get_item("angles") {
                        Ok(Some(py_any)) => {
                            // Extract as a sequence of f64 values
                            if let Ok(angles) = py_any.extract::<Vec<f64>>() {
                                if angles.len() >= 2 {
                                    self.inner.r1xy(angles[0], angles[1], location);
                                } else {
                                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                        "R1XY gate requires two angle parameters",
                                    ));
                                }
                            } else {
                                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                    "Expected valid angle parameters for R1XY gate",
                                ));
                            }
                        }
                        Ok(None) => {
                            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                "Angle parameters missing for R1XY gate",
                            ));
                        }
                        Err(err) => {
                            return Err(err);
                        }
                    }
                }
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

            "H" => {
                self.inner.h(location);
                Ok(None)
            }
            "H2" => {
                self.inner.h2(location);
                Ok(None)
            }
            "H3" => {
                self.inner.h3(location);
                Ok(None)
            }
            "H4" => {
                self.inner.h4(location);
                Ok(None)
            }
            "H5" => {
                self.inner.h5(location);
                Ok(None)
            }
            "H6" => {
                self.inner.h6(location);
                Ok(None)
            }
            "F" => {
                self.inner.f(location);
                Ok(None)
            }
            "Fdg" => {
                self.inner.fdg(location);
                Ok(None)
            }
            "F2" => {
                self.inner.f2(location);
                Ok(None)
            }
            "F2dg" => {
                self.inner.f2dg(location);
                Ok(None)
            }
            "F3" => {
                self.inner.f3(location);
                Ok(None)
            }
            "F3dg" => {
                self.inner.f3dg(location);
                Ok(None)
            }
            "F4" => {
                self.inner.f4(location);
                Ok(None)
            }
            "F4dg" => {
                self.inner.f4dg(location);
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
            "PZ" => {
                self.inner.pz(location);
                Ok(None)
            }
            "PX" => {
                self.inner.px(location);
                Ok(None)
            }
            "PY" => {
                self.inner.py(location);
                Ok(None)
            }
            "PnZ" => {
                self.inner.pnz(location);
                Ok(None)
            }
            "PnX" => {
                self.inner.pnx(location);
                Ok(None)
            }
            "PnY" => {
                self.inner.pny(location);
                Ok(None)
            }
            "MZ" | "MX" | "MY" => {
                let result = match symbol {
                    "MZ" => self.inner.mz(location),
                    "MX" => self.inner.mx(location),
                    "MY" => self.inner.my(location),
                    _ => unreachable!(),
                };
                Ok(Some(u8::from(result.outcome)))
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
    /// `params`: Optional parameters for parameterized gates (currently unused here)
    ///
    /// Returns an optional result, usually `None` unless a measurement is performed
    #[allow(clippy::too_many_lines)]
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
            "SXX" => {
                self.inner.sxx(q1, q2);
                Ok(None)
            }
            "SXXdg" => {
                self.inner.sxxdg(q1, q2);
                Ok(None)
            }
            "SYY" => {
                self.inner.syy(q1, q2);
                Ok(None)
            }
            "SYYdg" => {
                self.inner.syydg(q1, q2);
                Ok(None)
            }
            "SZZ" => {
                self.inner.szz(q1, q2);
                Ok(None)
            }
            "SZZdg" => {
                self.inner.szzdg(q1, q2);
                Ok(None)
            }
            "SWAP" => {
                self.inner.swap(q1, q2);
                Ok(None)
            }
            "G2" => {
                self.inner.g(q1, q2);
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
                }
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
                }
                Ok(None)
            }

            "RZZRYYRXX" => {
                if let Some(params) = params {
                    match params.get_item("angles") {
                        Ok(Some(py_any)) => {
                            if let Ok(angles) = py_any.extract::<Vec<f64>>() {
                                if angles.len() >= 3 {
                                    self.inner
                                        .rzzryyrxx(angles[0], angles[1], angles[2], q1, q2);
                                } else {
                                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                        "RZZRYYRXX gate requires three angle parameters",
                                    ));
                                }
                            } else {
                                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                    "Expected valid angle parameters for RZZRYYRXX gate",
                                ));
                            }
                        }
                        Ok(None) => {
                            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                "Angle parameters missing for RZZRYYRXX gate",
                            ));
                        }
                        Err(err) => {
                            return Err(err);
                        }
                    }
                }
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
}
