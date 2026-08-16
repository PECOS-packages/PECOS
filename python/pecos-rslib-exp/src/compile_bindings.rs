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
use pecos_stab_tn::stab_mps::compile::{InjectionMode, SimulatorKind, StabMpsCompile};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySet, PyTuple};

/// Compile-only STN tractability analyzer.
///
/// Replay the same gate stream that will be executed, then call `recommend()`
/// or `advise()`. The analyzer updates a stabilizer tableau and GF(2) flip
/// matrix without allocating an MPS. Recommendations are heuristic. Qubits are
/// zero-based; any bit rows elsewhere in the STN API use `bits[q] == qubit q`.
///
/// # Gate symbols
///
/// The three STN classes accept the same dispatch symbols:
///
/// | Arity | Accepted symbols | Parameters |
/// | --- | --- | --- |
/// | 1 | `I`; `X`; `Y`; `Z` | none |
/// | 1 | `H`, `H1`, `H+z+x`; `F`, `F1`; `Fdg`, `F1d`, `F1dg` | none |
/// | 1 | `SX`, `SqrtX`, `Q`; `SXdg`, `SqrtXdg`, `SqrtXd`, `Qd` | none |
/// | 1 | `SY`, `SqrtY`, `R`; `SYdg`, `SqrtYdg`, `SqrtYd`, `Rd` | none |
/// | 1 | `S`, `SZ`, `SqrtZ`; `Sd`, `SZdg`, `SqrtZdg`, `SqrtZd` | none |
/// | 1 | `RX`; `RY`; `RZ` | `angle` in radians |
/// | 1 | `T`; `Tdg` | none |
/// | 1 | `PZ`, `Init`, `init \|0>`; `PX`, `Init +X`, `init \|+>` | none |
/// | 1 | `MZ`, `Measure`, `measure Z` | none; returns 0 or 1 |
/// | 2 | `CX`, `CNOT`; `CY`; `CZ`; `SXX`; `SXXdg`; `SYY`; `SYYdg`; `SZZ`; `SZZdg`; `SWAP` | none |
/// | 2 | `RZZ` | `angle` in radians |
#[pyclass(name = "StabMpsCompile", module = "pecos_rslib_exp")]
pub struct PyStabMpsCompile {
    inner: StabMpsCompile,
}

impl PyStabMpsCompile {
    fn check_qubit(&self, q: isize, method: &str) -> PyResult<usize> {
        let Ok(q) = usize::try_from(q) else {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "{method}: qubit {q} out of bounds (num_qubits={})",
                self.inner.num_qubits()
            )));
        };
        if q >= self.inner.num_qubits() {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "{method}: qubit {q} out of bounds (num_qubits={})",
                self.inner.num_qubits()
            )));
        }
        Ok(q)
    }
}

fn simulator_name(kind: SimulatorKind) -> &'static str {
    match kind {
        SimulatorKind::StateVector => "state_vector",
        SimulatorKind::CHForm => "ch_form",
        SimulatorKind::StabVec => "stab_vec",
        SimulatorKind::StabMps => "stab_mps",
        SimulatorKind::Mast => "mast",
    }
}

fn injection_name(mode: InjectionMode) -> &'static str {
    match mode {
        InjectionMode::Direct => "direct",
        InjectionMode::Immediate => "immediate",
        InjectionMode::Deferred => "deferred",
    }
}

#[pymethods]
impl PyStabMpsCompile {
    /// Create an empty compile-only analyzer for `num_qubits` qubits.
    ///
    /// Gate replay costs approximately `O(t*n**2)` for `t` non-Clifford gates
    /// and `n` qubits, without coefficient-MPS allocation.
    #[new]
    fn new(num_qubits: usize) -> Self {
        Self {
            inner: StabMpsCompile::new(num_qubits),
        }
    }

    /// Clear the replayed circuit and all counters, returning `self`.
    fn reset(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.inner.reset();
        slf
    }

    #[getter]
    /// Number of qubits being analyzed.
    fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    #[getter]
    /// Non-Clifford gates that OFD would absorb by consuming a free qubit.
    fn absorbed(&self) -> u64 {
        self.inner.absorbed()
    }

    #[getter]
    /// Non-Clifford gates predicted to grow the coefficient-MPS bond dimension.
    fn grown(&self) -> u64 {
        self.inner.grown()
    }

    #[getter]
    /// Non-Clifford RZ gates reduced to the stabilizer-only branch.
    fn stabilizer(&self) -> u64 {
        self.inner.stabilizer()
    }

    #[getter]
    /// Total analyzed non-Clifford gates: absorbed plus grown plus stabilizer.
    fn total_nonclifford(&self) -> u64 {
        self.inner.total_nonclifford()
    }

    #[getter]
    /// Total analyzed non-Clifford RZ gates.
    fn nonclifford_rz_total(&self) -> u64 {
        self.inner.nonclifford_rz_total()
    }

    #[getter]
    /// RZ gates whose deferred-injection `RZ(2*angle)` correction is Clifford.
    fn injectable_clifford_correction(&self) -> u64 {
        self.inner.injectable_clifford_correction()
    }

    #[getter]
    /// OFD nullity: tracked flip-pattern count minus GF(2) rank.
    fn nullity(&self) -> usize {
        self.inner.nullity()
    }

    #[getter]
    /// GF(2) rank of accumulated flip patterns.
    fn rank(&self) -> usize {
        self.inner.rank()
    }

    #[getter]
    /// Theoretical STN bond upper bound, `2**nullity`.
    ///
    /// Saturates at the platform's maximum unsigned integer on overflow.
    fn bond_dim_bound(&self) -> usize {
        self.inner.bond_dim_bound()
    }

    /// Return a heuristic simulator recommendation dictionary.
    ///
    /// The `simulator` and `reason` fields use these ordered thresholds: pure
    /// Clifford -> `ch_form`; otherwise `n <= 14` -> `state_vector`; otherwise
    /// nullity `<= 6` -> `stab_mps`; otherwise non-Clifford count `<= 40` ->
    /// `stab_vec`; otherwise -> `stab_mps` with adaptive growth suggested.
    fn recommend(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let recommendation = self.inner.recommend();
        let result = PyDict::new(py);
        result.set_item("simulator", simulator_name(recommendation.kind))?;
        result.set_item("reason", recommendation.reason)?;
        Ok(result.unbind())
    }

    /// Return simulator, injection, capacity, warning, and reason fields.
    ///
    /// `ancilla_budget` counts fresh ancillas available for deferral. `None`
    /// yields `deferred_feasible=None` and a warning; a sufficient budget yields
    /// deferred injection; an insufficient nonzero budget yields immediate
    /// injection; zero yields direct application and the same insufficient-
    /// budget warning. Injection is always direct when the selected simulator
    /// is not `stab_mps` or `mast`. `deferred_ancillas_required` counts every
    /// non-Clifford RZ, while `injectable_count` counts only gates with Clifford
    /// corrections. Recommendations are heuristic.
    #[pyo3(signature = (ancilla_budget=None))]
    fn advise(&self, py: Python<'_>, ancilla_budget: Option<usize>) -> PyResult<Py<PyDict>> {
        let advice = self.inner.advise(ancilla_budget);
        let result = PyDict::new(py);
        result.set_item("simulator", simulator_name(advice.simulator))?;
        result.set_item("injection", injection_name(advice.injection))?;
        result.set_item("injectable_count", advice.injectable_count)?;
        result.set_item(
            "deferred_ancillas_required",
            advice.deferred_ancillas_required,
        )?;
        result.set_item("deferred_feasible", advice.deferred_feasible)?;
        result.set_item("warnings", advice.warnings)?;
        result.set_item("reason", advice.reason)?;
        Ok(result.unbind())
    }

    // ---- Gate dispatch (matches StabMps and Mast) ----

    /// Replay one accepted gate symbol on one qubit.
    ///
    /// `location` is zero-based. RX, RY, and RZ require
    /// `params={"angle": radians}`. Measurement symbols return `0` or `1`;
    /// other gates return `None`. Raises `IndexError` for an invalid qubit and
    /// `ValueError` for an unknown symbol or invalid angle. See `run_gate` for
    /// the complete symbol table.
    #[pyo3(signature = (symbol, location, params=None))]
    fn run_1q_gate(
        &mut self,
        symbol: &str,
        location: isize,
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Option<u8>> {
        let location = self.check_qubit(location, symbol)?;
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

    /// Replay one accepted two-qubit gate on an ordered qubit pair.
    ///
    /// `location` must be a two-element tuple of distinct zero-based indices;
    /// the first is the control for controlled gates. RZZ requires
    /// `params={"angle": radians}`. Raises `IndexError` or `ValueError` for
    /// invalid arguments. See `run_gate` for the complete symbol table.
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
        let q1 = self.check_qubit(location.get_item(0)?.extract::<isize>()?, symbol)?;
        let q2 = self.check_qubit(location.get_item(1)?.extract::<isize>()?, symbol)?;
        if q1 == q2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Two-qubit gate requires distinct qubit locations",
            ));
        }
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

    /// Replay a gate on a set of one- or two-qubit locations.
    ///
    /// `locations` must be a `set`; each item is either a zero-based qubit
    /// integer or an ordered two-integer tuple. Rotation keyword `angle` is in
    /// radians. The returned dictionary contains entries only for measurement
    /// locations, mapped to integer outcomes. Bit-bearing results use the
    /// shared convention `bits[q] == qubit q` wherever represented as rows.
    ///
    /// # Gate symbols
    ///
    /// | Arity | Accepted symbols | Parameters |
    /// | --- | --- | --- |
    /// | 1 | `I`; `X`; `Y`; `Z` | none |
    /// | 1 | `H`, `H1`, `H+z+x`; `F`, `F1`; `Fdg`, `F1d`, `F1dg` | none |
    /// | 1 | `SX`, `SqrtX`, `Q`; `SXdg`, `SqrtXdg`, `SqrtXd`, `Qd` | none |
    /// | 1 | `SY`, `SqrtY`, `R`; `SYdg`, `SqrtYdg`, `SqrtYd`, `Rd` | none |
    /// | 1 | `S`, `SZ`, `SqrtZ`; `Sd`, `SZdg`, `SqrtZdg`, `SqrtZd` | none |
    /// | 1 | `RX`; `RY`; `RZ` | `angle` in radians |
    /// | 1 | `T`; `Tdg` | none |
    /// | 1 | `PZ`, `Init`, `init \|0>`; `PX`, `Init +X`, `init \|+>` | none |
    /// | 1 | `MZ`, `Measure`, `measure Z` | none; returns 0 or 1 |
    /// | 2 | `CX`, `CNOT`; `CY`; `CZ`; `SXX`; `SXXdg`; `SYY`; `SYYdg`; `SZZ`; `SZZdg`; `SWAP` | none |
    /// | 2 | `RZZ` | `angle` in radians |
    ///
    /// Raises `TypeError` when `locations` or its items have the wrong Python
    /// shape, `IndexError` for an out-of-range qubit, and `ValueError` for bad
    /// arity, repeated pair members, an unsupported symbol, or an invalid angle.
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
                    let qubit: isize = loc_tuple.get_item(0)?.extract()?;
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
        Ok(output.unbind())
    }
}
