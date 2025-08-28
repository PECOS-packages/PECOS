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

use pecos_cppsparsesim::CppSparseStab;
use pecos_qsim::{CliffordGateable, QuantumSimulator};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

// Monte Carlo engines create independent simulator copies for each thread.
// CppSparseStab implements Send, so each thread gets exclusive access to its own instance.
#[pyclass(name = "CppSparseSim")]
pub struct CppSparseSim {
    inner: CppSparseStab,
}

#[pymethods]
impl CppSparseSim {
    #[new]
    #[pyo3(signature = (num_qubits, seed=None))]
    fn new(num_qubits: usize, seed: Option<u64>) -> Self {
        let inner = match seed {
            Some(s) => CppSparseStab::with_seed(num_qubits, s),
            None => CppSparseStab::new(num_qubits),
        };
        CppSparseSim { inner }
    }

    fn set_seed(&mut self, seed: u64) {
        self.inner.set_seed(seed);
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn __repr__(&self) -> String {
        format!("CppSparseSim(num_qubits={})", self.inner.num_qubits())
    }

    #[getter]
    fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

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
            "MZForced" => {
                if let Some(params) = params {
                    // Extract forced_outcome as integer first, then convert to bool
                    let forced_int = params
                        .get_item("forced_outcome")?
                        .ok_or_else(|| {
                            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                "MZForced requires a 'forced_outcome' parameter",
                            )
                        })?
                        .extract::<i32>()?;
                    let forced_value = forced_int != 0;
                    let result = self.inner.force_measure(location, forced_value);
                    Ok(Some(u8::from(result.outcome)))
                } else {
                    Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "MZForced requires a 'forced_outcome' parameter",
                    ))
                }
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Unsupported single-qubit gate: {symbol}"
            ))),
        }
    }

    fn run_2q_gate(
        &mut self,
        symbol: &str,
        location: &Bound<'_, PyTuple>,
        _params: Option<&Bound<'_, PyDict>>,
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
            "SWAP" => {
                self.inner.swap(q1, q2);
                Ok(None)
            }
            "G2" => {
                self.inner.g2(q1, q2);
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
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Unsupported two-qubit gate: {symbol}"
            ))),
        }
    }

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
                "Gates must have either 1 or 2 qubit locations",
            )),
        }
    }

    // Additional methods that mirror SparseSim's API
    fn h(&mut self, qubit: usize) {
        self.inner.h(qubit);
    }

    fn x(&mut self, qubit: usize) {
        self.inner.x(qubit);
    }

    fn y(&mut self, qubit: usize) {
        self.inner.y(qubit);
    }

    fn z(&mut self, qubit: usize) {
        self.inner.z(qubit);
    }

    fn cx(&mut self, control: usize, target: usize) {
        self.inner.cx(control, target);
    }

    fn mz(&mut self, qubit: usize) -> bool {
        self.inner.mz(qubit).outcome
    }

    fn mx(&mut self, qubit: usize) -> bool {
        self.inner.mx(qubit).outcome
    }

    fn my(&mut self, qubit: usize) -> bool {
        self.inner.my(qubit).outcome
    }

    fn stab_tableau(&mut self) -> String {
        self.inner.stab_tableau()
    }

    fn destab_tableau(&mut self) -> String {
        self.inner.destab_tableau()
    }
}
