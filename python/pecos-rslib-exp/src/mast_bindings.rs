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
use pecos_stab_tn::stab_mps::mast::{Mast, ProjectionOrder};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySet, PyTuple};

/// Python MAST simulator.
///
/// Telemetry and capacity reads report stored state without materializing
/// pending lazy-measurement operations or merged RZ rotations. Exceeding the
/// constructor's `max_non_clifford` capacity raises a `PanicException`;
/// `remaining_injections` exposes available capacity, and
/// `StabMpsCompile.advise()` reports the required capacity for an analyzed
/// circuit. `project_all()` completes deferred injections explicitly; any MZ
/// gate does so automatically before measuring data. Qubit indices are
/// zero-based, and any bit rows elsewhere in the STN API use `bits[q] == qubit q`.
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
#[pyclass(name = "Mast", module = "pecos_rslib_exp")]
pub struct PyMast {
    inner: Mast,
}

impl PyMast {
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

#[pymethods]
impl PyMast {
    #[new]
    /// Create a MAST simulator.
    ///
    /// `projection_order` accepts `"min_span"` or `"input"`; `None` uses
    /// the Rust default (`"min_span"`). Boolean options are tri-state:
    /// `None` preserves the Rust default, while an explicit bool calls the
    /// corresponding Rust setter. Exceeding `max_non_clifford` raises a
    /// `PanicException`; inspect `remaining_injections` before adding work.
    /// `StabMpsCompile.advise()` reports the required deferred capacity for an
    /// analyzed circuit.
    ///
    /// `seed` initializes PECOS's buffered RapidHash RNG and the tableau. Fresh
    /// instances with the same configuration and call sequence reproduce
    /// stochastic results; `reset()` rewinds both streams to the seed.
    #[pyo3(signature = (
        num_qubits,
        max_non_clifford,
        seed=None,
        lazy_measure=None,
        merge_rz=None,
        projection_order=None,
        numerical_flag_redetection=None,
    ))]
    fn new(
        num_qubits: usize,
        max_non_clifford: usize,
        seed: Option<u64>,
        lazy_measure: Option<bool>,
        merge_rz: Option<bool>,
        projection_order: Option<&str>,
        numerical_flag_redetection: Option<bool>,
    ) -> PyResult<Self> {
        if num_qubits.checked_add(max_non_clifford).is_none() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "num_qubits + max_non_clifford exceeds platform capacity",
            ));
        }
        let mut mast = if let Some(s) = seed {
            Mast::with_seed(num_qubits, max_non_clifford, s)
        } else {
            Mast::new(num_qubits, max_non_clifford)
        };
        if let Some(value) = lazy_measure {
            mast = mast.with_lazy_measure(value);
        }
        if let Some(value) = merge_rz {
            mast = mast.with_merge_rz(value);
        }
        if let Some(value) = numerical_flag_redetection {
            mast = mast.with_numerical_flag_redetection(value);
        }
        if let Some(order) = projection_order {
            let order = match order {
                "min_span" => ProjectionOrder::MinSpan,
                "input" => ProjectionOrder::Input,
                _ => {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "projection_order must be 'min_span' or 'input'",
                    ));
                }
            };
            mast = mast.projection_order(order);
        }
        Ok(PyMast { inner: mast })
    }

    /// Reset data, ancillas, capacity use, and diagnostics, returning `self`.
    ///
    /// Configuration is retained. A seeded simulator rewinds both RNG streams
    /// to the construction seed; an unseeded simulator obtains fresh entropy.
    fn reset(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.inner.reset();
        slf
    }

    #[getter]
    /// Number of addressable data qubits.
    ///
    /// Preallocated injection ancillas are internal and excluded.
    fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    #[getter]
    /// Number of data qubits, equal to `num_qubits` for this binding.
    fn num_data_qubits(&self) -> usize {
        self.inner.num_data_qubits()
    }

    #[getter]
    /// Number of fresh ancilla slots consumed by injections.
    ///
    /// Pending unflushed rotations are not reflected and can consume capacity
    /// when later materialized.
    fn num_ancillas_used(&self) -> usize {
        self.inner.num_ancillas_used()
    }

    #[getter]
    /// Number of additional injections available before capacity is exhausted.
    ///
    /// Pending unflushed rotations are not reflected and can consume capacity
    /// when later materialized.
    /// Exceeding capacity through a later gate raises `PanicException`.
    fn remaining_injections(&self) -> usize {
        self.inner.remaining_injections()
    }

    #[getter]
    /// Largest bond dimension currently present in the coefficient MPS.
    ///
    /// Pending unflushed operations are not reflected and can later consume
    /// injection capacity. Deferred injections remain unprojected until
    /// `project_all()` or an MZ gate.
    fn max_bond_dim(&self) -> usize {
        self.inner.max_bond_dim()
    }

    /// Return diagnostics for deferred magic-state projections since reset.
    ///
    /// Pending unflushed operations are not reflected and can later consume
    /// injection capacity. Each dictionary reports ancilla, support size, MPS
    /// span, and bond dimensions before and after projection.
    fn projection_records(&self, py: Python<'_>) -> PyResult<Vec<Py<PyDict>>> {
        self.inner
            .projection_records()
            .iter()
            .map(|record| {
                let result = PyDict::new(py);
                result.set_item("ancilla", record.ancilla)?;
                result.set_item("support_size", record.support_size)?;
                result.set_item("mps_span", record.mps_span)?;
                result.set_item("bond_before", record.bond_before)?;
                result.set_item("bond_after", record.bond_after)?;
                Ok(result.unbind())
            })
            .collect()
    }

    #[getter]
    /// Peak MPS bond dimension observed during deferred projections.
    ///
    /// Returns zero until `project_all()` or an MZ gate projects an injection.
    /// Pending unflushed operations are not reflected and can later consume
    /// injection capacity.
    fn projection_peak_bond(&self) -> usize {
        self.inner.projection_peak_bond()
    }

    /// Return runtime non-Clifford path counters as a dictionary.
    ///
    /// Pending unflushed operations are not reflected and can later consume
    /// injection capacity. Deferred injections remain unprojected.
    fn stats(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        crate::stab_mps_stats_to_dict(py, &self.inner.stats)
    }

    /// Materialize lazy-measurement operations and pending merged RZ rotations.
    ///
    /// Flushing a non-Clifford merged rotation can consume injection capacity.
    /// This does not project already deferred injections.
    fn flush(&mut self) {
        self.inner.flush();
    }

    /// Project all deferred magic-state ancillas and apply their corrections.
    ///
    /// This is the explicit MAST completion step. An MZ gate calls it
    /// automatically before measuring the requested data qubit.
    fn project_all(&mut self) {
        self.inner.project_all();
    }

    // ---- Gate dispatch ----

    /// Apply one accepted gate symbol to one data qubit.
    ///
    /// `location` is zero-based. RX, RY, and RZ require
    /// `params={"angle": radians}`. MZ returns `0` or `1` and first completes
    /// every deferred injection; other gates return `None`. Raises `IndexError`
    /// for an invalid qubit, `ValueError` for an unknown symbol or invalid angle,
    /// and `PanicException` if injection capacity is exceeded. See `run_gate`
    /// for the complete symbol table.
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

    /// Apply one accepted two-qubit gate to an ordered data-qubit pair.
    ///
    /// `location` must be a two-element tuple of distinct zero-based indices;
    /// the first is the control for controlled gates. RZZ requires
    /// `params={"angle": radians}`. Raises `IndexError`, `ValueError`, or
    /// `PanicException` for invalid indices/arguments or exhausted injection
    /// capacity. See `run_gate` for the complete symbol table.
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

    /// Apply a gate to a set of one- or two-qubit locations.
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
    /// shape, `IndexError` for an out-of-range qubit, `ValueError` for bad arity,
    /// repeated pair members, an unsupported symbol, or an invalid angle, and
    /// `PanicException` if injection capacity is exceeded.
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
        Ok(output.into())
    }
}
