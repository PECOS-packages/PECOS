// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
// except in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the
// License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
// express or implied. See the License for the specific language governing permissions and
// limitations under the License.

use pecos_core::{Angle64, QubitId};
use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable, QuantumSimulator};
use pecos_stab_tn::stab_mps::mast::Mast;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySet, PyTuple};

#[pyclass(name = "Mast", module = "pecos_rslib_exp")]
pub struct PyMast {
    inner: Mast,
}

impl PyMast {
    fn check_qubit(&self, q: usize, method: &str) -> PyResult<()> {
        if q >= self.inner.num_qubits() {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "{method}: qubit {q} out of bounds (num_qubits={})",
                self.inner.num_qubits()
            )));
        }
        Ok(())
    }
}

#[pymethods]
impl PyMast {
    #[new]
    #[pyo3(signature = (num_qubits, max_non_clifford, seed=None, lazy_measure=false, merge_rz=false))]
    fn new(
        num_qubits: usize,
        max_non_clifford: usize,
        seed: Option<u64>,
        lazy_measure: bool,
        merge_rz: bool,
    ) -> Self {
        let mut mast = if let Some(s) = seed {
            Mast::with_seed(num_qubits, max_non_clifford, s)
        } else {
            Mast::new(num_qubits, max_non_clifford)
        };
        if lazy_measure {
            mast = mast.with_lazy_measure(true);
        }
        if merge_rz {
            mast = mast.with_merge_rz(true);
        }
        PyMast { inner: mast }
    }

    fn reset(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.inner.reset();
        slf
    }

    #[getter]
    fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    #[getter]
    fn num_data_qubits(&self) -> usize {
        self.inner.num_data_qubits()
    }

    #[getter]
    fn num_ancillas_used(&self) -> usize {
        self.inner.num_ancillas_used()
    }

    #[getter]
    fn max_bond_dim(&self) -> usize {
        self.inner.max_bond_dim()
    }

    fn flush(&mut self) {
        self.inner.flush();
    }

    fn project_all(&mut self) {
        self.inner.project_all();
    }

    // ---- Gate dispatch ----

    #[pyo3(signature = (symbol, location, params=None))]
    fn run_1q_gate(
        &mut self,
        symbol: &str,
        location: usize,
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Option<u8>> {
        self.check_qubit(location, symbol)?;
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
            "F" | "F1" => {
                self.inner.f(q);
                Ok(None)
            }
            "Fdg" | "F1d" | "F1dg" => {
                self.inner.fdg(q);
                Ok(None)
            }
            "SX" | "SqrtX" | "Q" => {
                self.inner.sx(q);
                Ok(None)
            }
            "SXdg" | "SqrtXdg" | "SqrtXd" | "Qd" => {
                self.inner.sxdg(q);
                Ok(None)
            }
            "SY" | "SqrtY" | "R" => {
                self.inner.sy(q);
                Ok(None)
            }
            "SYdg" | "SqrtYdg" | "SqrtYd" | "Rd" => {
                self.inner.sydg(q);
                Ok(None)
            }
            "S" | "SZ" | "SqrtZ" => {
                self.inner.sz(q);
                Ok(None)
            }
            "Sd" | "SZdg" | "SqrtZdg" | "SqrtZd" => {
                self.inner.szdg(q);
                Ok(None)
            }
            "RX" => {
                let angle = crate::extract_angle(params, "RX")?;
                self.inner.rx(angle, q);
                Ok(None)
            }
            "RY" => {
                let angle = crate::extract_angle(params, "RY")?;
                self.inner.ry(angle, q);
                Ok(None)
            }
            "RZ" => {
                let angle = crate::extract_angle(params, "RZ")?;
                self.inner.rz(angle, q);
                Ok(None)
            }
            "T" => {
                self.inner.rz(Angle64::QUARTER_TURN / 2u64, q);
                Ok(None)
            }
            "Tdg" => {
                self.inner.rz(-(Angle64::QUARTER_TURN / 2u64), q);
                Ok(None)
            }
            "PZ" | "Init" | "init |0>" => {
                let results = self.inner.mz(q);
                if results[0].outcome {
                    self.inner.x(q);
                }
                Ok(None)
            }
            "PX" | "Init +X" | "init |+>" => {
                let results = self.inner.mz(q);
                if results[0].outcome {
                    self.inner.x(q);
                }
                self.inner.h(q);
                Ok(None)
            }
            "MZ" | "Measure" | "measure Z" => {
                let result = self
                    .inner
                    .mz(q)
                    .into_iter()
                    .next()
                    .expect("measurement returned no results");
                Ok(Some(u8::from(result.outcome)))
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Unsupported single-qubit gate: {symbol}"
            ))),
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
        self.check_qubit(q1, symbol)?;
        self.check_qubit(q2, symbol)?;
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
            "SXX" => {
                self.inner.sxx(pair);
                Ok(None)
            }
            "SXXdg" => {
                self.inner.sxxdg(pair);
                Ok(None)
            }
            "SYY" => {
                self.inner.syy(pair);
                Ok(None)
            }
            "SYYdg" => {
                self.inner.syydg(pair);
                Ok(None)
            }
            "SZZ" => {
                self.inner.szz(pair);
                Ok(None)
            }
            "SZZdg" => {
                self.inner.szzdg(pair);
                Ok(None)
            }
            "SWAP" => {
                self.inner.swap(pair);
                Ok(None)
            }
            "RZZ" => {
                let angle = crate::extract_angle(params, "RZZ")?;
                self.inner.rzz(angle, pair);
                Ok(None)
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Unsupported two-qubit gate: {symbol}"
            ))),
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
        let output = PyDict::new(py);
        let locations_set: Bound<PySet> = locations.clone().cast_into()?;
        for location in locations_set.iter() {
            let loc_tuple: Bound<'_, PyTuple> = if location.is_instance_of::<PyTuple>() {
                location.clone().cast_into()?
            } else {
                PyTuple::new(py, std::slice::from_ref(&location))?
            };
            let result = match loc_tuple.len() {
                1 => {
                    let qubit: usize = loc_tuple.get_item(0)?.extract()?;
                    self.run_1q_gate(symbol, qubit, params)?
                }
                2 => self.run_2q_gate(symbol, &loc_tuple, params)?,
                _ => {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Gate location must be 1 or 2 qubits",
                    ));
                }
            };
            if let Some(value) = result {
                output.set_item(location, value)?;
            }
        }
        Ok(output.into())
    }
}
