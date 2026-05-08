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

#![allow(clippy::needless_pass_by_value)] // PyO3 requires passing extracted types by value

use pecos_core::{Angle64, QubitId};
use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable, QuantumSimulator};
use pecos_stab_tn::stab_mps::{PauliKind, StabMps};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySet, PyTuple};

#[pyclass(name = "StabMps", module = "pecos_rslib_exp")]
pub struct PyStabMps {
    inner: StabMps,
}

impl PyStabMps {
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
impl PyStabMps {
    #[new]
    #[pyo3(signature = (
        num_qubits,
        seed=None,
        max_bond_dim=None,
        merge_rz=None,
        pauli_frame_tracking=None,
        lazy_measure=None,
        for_qec=None,
        auto_grow_bond_dim=None,
        auto_grow_max_bond_dim=None,
        max_truncation_error=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        num_qubits: usize,
        seed: Option<u64>,
        max_bond_dim: Option<usize>,
        merge_rz: Option<bool>,
        pauli_frame_tracking: Option<bool>,
        lazy_measure: Option<bool>,
        for_qec: Option<bool>,
        auto_grow_bond_dim: Option<f64>,
        auto_grow_max_bond_dim: Option<usize>,
        max_truncation_error: Option<f64>,
    ) -> Self {
        let mut b = StabMps::builder(num_qubits);
        if let Some(s) = seed {
            b = b.seed(s);
        }
        if for_qec == Some(true) {
            b = b.for_qec();
        }
        if let Some(bd) = max_bond_dim {
            b = b.max_bond_dim(bd);
        }
        if merge_rz == Some(true) {
            b = b.merge_rz(true);
        }
        if pauli_frame_tracking == Some(true) {
            b = b.pauli_frame_tracking(true);
        }
        if lazy_measure == Some(true) {
            b = b.lazy_measure(true);
        }
        if let Some(t) = auto_grow_bond_dim {
            b = b.auto_grow_bond_dim(t);
        }
        if let Some(c) = auto_grow_max_bond_dim {
            b = b.auto_grow_max_bond_dim(c);
        }
        if let Some(e) = max_truncation_error {
            b = b.max_truncation_error(e);
        }
        PyStabMps { inner: b.build() }
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
    fn max_bond_dim(&self) -> usize {
        self.inner.max_bond_dim()
    }

    #[getter]
    fn truncation_error(&self) -> f64 {
        self.inner.truncation_error()
    }

    #[getter]
    fn pragmatic_drift_count(&self) -> u64 {
        self.inner.pragmatic_drift_count()
    }

    fn is_state_exact(&self) -> bool {
        self.inner.is_state_exact()
    }

    fn flush(&mut self) {
        self.inner.flush();
    }

    fn flush_pauli_frame_to_state(&mut self) {
        self.inner.flush_pauli_frame_to_state();
    }

    fn state_vector(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let sv = self.inner.state_vector();
        let list: Vec<(f64, f64)> = sv.iter().map(|c| (c.re, c.im)).collect();
        Ok(PyList::new(py, &list)?.unbind())
    }

    fn prob_bitstring(&self, bitstring: Vec<bool>) -> f64 {
        self.inner.prob_bitstring(&bitstring)
    }

    // ---- QEC helpers ----

    fn reset_qubit(&mut self, q: usize) -> PyResult<bool> {
        self.check_qubit(q, "reset_qubit")?;
        Ok(self.inner.reset_qubit(QubitId(q)))
    }

    fn pz(&mut self, q: usize) -> PyResult<()> {
        self.check_qubit(q, "pz")?;
        self.inner.pz(QubitId(q));
        Ok(())
    }

    fn px(&mut self, q: usize) -> PyResult<()> {
        self.check_qubit(q, "px")?;
        self.inner.px(QubitId(q));
        Ok(())
    }

    fn inject_x_in_frame(&mut self, q: usize) -> PyResult<()> {
        self.check_qubit(q, "inject_x_in_frame")?;
        self.inner.inject_x_in_frame(QubitId(q));
        Ok(())
    }

    fn inject_y_in_frame(&mut self, q: usize) -> PyResult<()> {
        self.check_qubit(q, "inject_y_in_frame")?;
        self.inner.inject_y_in_frame(QubitId(q));
        Ok(())
    }

    fn inject_z_in_frame(&mut self, q: usize) -> PyResult<()> {
        self.check_qubit(q, "inject_z_in_frame")?;
        self.inner.inject_z_in_frame(QubitId(q));
        Ok(())
    }

    fn inject_paulis_in_frame(&mut self, paulis: Vec<(usize, String)>) -> PyResult<()> {
        let converted: Vec<(QubitId, PauliKind)> = paulis
            .into_iter()
            .map(|(q, s)| {
                let kind = match s.as_str() {
                    "X" => PauliKind::X,
                    "Y" => PauliKind::Y,
                    "Z" => PauliKind::Z,
                    _ => {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                            "Unknown Pauli kind: {s}. Use 'X', 'Y', or 'Z'."
                        )));
                    }
                };
                Ok((QubitId(q), kind))
            })
            .collect::<PyResult<Vec<_>>>()?;
        self.inner.inject_paulis_in_frame(&converted);
        Ok(())
    }

    fn frame_x_bit(&self, q: usize) -> bool {
        self.inner.frame_x_bit(QubitId(q))
    }

    fn frame_z_bit(&self, q: usize) -> bool {
        self.inner.frame_z_bit(QubitId(q))
    }

    fn apply_depolarizing(&mut self, q: usize, p: f64) -> Option<String> {
        self.inner
            .apply_depolarizing(QubitId(q), p)
            .map(|k| format!("{k:?}"))
    }

    fn apply_depolarizing_all(&mut self, qubits: Vec<usize>, p: f64) {
        let qs: Vec<QubitId> = qubits.into_iter().map(QubitId).collect();
        self.inner.apply_depolarizing_all(&qs, p);
    }

    fn extract_syndromes(
        &mut self,
        generators: Vec<Vec<(usize, String)>>,
        ancilla_qubits: Vec<usize>,
    ) -> PyResult<Vec<bool>> {
        let gens: Vec<Vec<(usize, PauliKind)>> = generators
            .into_iter()
            .map(|g| {
                g.into_iter()
                    .map(|(q, s)| {
                        let kind = match s.as_str() {
                            "X" => PauliKind::X,
                            "Y" => PauliKind::Y,
                            "Z" => PauliKind::Z,
                            _ => {
                                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                    format!("Unknown Pauli: {s}"),
                                ));
                            }
                        };
                        Ok((q, kind))
                    })
                    .collect::<PyResult<Vec<_>>>()
            })
            .collect::<PyResult<Vec<_>>>()?;
        let ancs: Vec<QubitId> = ancilla_qubits.into_iter().map(QubitId).collect();
        Ok(self.inner.extract_syndromes(&gens, &ancs))
    }

    fn pauli_expectation(&self, pauli_string: Vec<(usize, String)>) -> PyResult<f64> {
        let ps: Vec<(usize, PauliKind)> = pauli_string
            .into_iter()
            .map(|(q, s)| {
                let kind = match s.as_str() {
                    "X" => PauliKind::X,
                    "Y" => PauliKind::Y,
                    "Z" => PauliKind::Z,
                    _ => {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                            "Unknown Pauli: {s}"
                        )));
                    }
                };
                Ok((q, kind))
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(self.inner.pauli_expectation(&ps))
    }

    fn code_state_fidelity(&self, stabilizers: Vec<Vec<(usize, String)>>) -> PyResult<f64> {
        let stabs: Vec<Vec<(usize, PauliKind)>> = stabilizers
            .into_iter()
            .map(|g| {
                g.into_iter()
                    .map(|(q, s)| {
                        let kind = match s.as_str() {
                            "X" => PauliKind::X,
                            "Y" => PauliKind::Y,
                            "Z" => PauliKind::Z,
                            _ => {
                                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                    format!("Unknown Pauli: {s}"),
                                ));
                            }
                        };
                        Ok((q, kind))
                    })
                    .collect::<PyResult<Vec<_>>>()
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(self.inner.code_state_fidelity(&stabs))
    }

    fn sample_bitstring(&mut self, num_shots: usize) -> Vec<Vec<bool>> {
        self.inner.sample_bitstring(num_shots)
    }

    // ---- Gate dispatch (matches pecos-rslib pattern) ----

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
                self.inner.pz(QubitId(location));
                Ok(None)
            }
            "PX" | "Init +X" | "init |+>" => {
                self.inner.px(QubitId(location));
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
