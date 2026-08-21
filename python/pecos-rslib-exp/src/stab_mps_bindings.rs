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

use pecos_core::clifford_rep::CliffordRep;
use pecos_core::{Angle64, Pauli, PauliOperator, PauliString, QuarterPhase, QubitId};
use pecos_simulators::{ArbitraryRotationGateable, CHForm, CliffordGateable, QuantumSimulator};
use pecos_stab_tn::stab_mps::{PauliKind, StabMps};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyList, PySet, PyTuple};

/// Python stabilizer-MPS simulator.
///
/// Read methods materialize pending lazy-measurement operations and merged RZ
/// rotations before returning, except the pure diagnostics `is_state_exact()`
/// and `uncompensated_pre_reduction_count`. Bitstrings use qubit-index order: `bits[q]` is
/// the bit for qubit `q`, and input items must be actual Python `bool` values.
/// A tracked Pauli frame remains separate until
/// `flush_pauli_frame_to_state()` is called. The `for_qec` constructor keyword is an enable-only
/// preset switch: `True` applies it, while `False` and `None` are identical
/// no-ops.
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
#[pyclass(name = "StabMps", module = "pecos_rslib_exp")]
pub struct PyStabMps {
    inner: StabMps,
}

#[derive(Clone, Copy)]
enum StabilizerPrepGate {
    H(usize),
    Sdg(usize),
    X(usize),
    Cx(usize, usize),
    Cz(usize, usize),
    Swap(usize, usize),
}

impl StabilizerPrepGate {
    fn clifford(self, num_qubits: usize) -> CliffordRep {
        let gate = match self {
            Self::H(q) => CliffordRep::h(q),
            Self::Sdg(q) => CliffordRep::szdg(q),
            Self::X(q) => CliffordRep::x(q),
            Self::Cx(control, target) => CliffordRep::cx(control, target),
            Self::Cz(q0, q1) => CliffordRep::cz(q0, q1),
            Self::Swap(q0, q1) => CliffordRep::swap(q0, q1),
        };
        gate.extended_to(num_qubits)
    }

    fn apply_inverse(self, simulator: &mut CHForm) {
        match self {
            Self::H(q) => {
                simulator.h(&[QubitId(q)]);
            }
            Self::Sdg(q) => {
                simulator.sz(&[QubitId(q)]);
            }
            Self::X(q) => {
                simulator.x(&[QubitId(q)]);
            }
            Self::Cx(control, target) => {
                simulator.cx(&[(QubitId(control), QubitId(target))]);
            }
            Self::Cz(q0, q1) => {
                simulator.cz(&[(QubitId(q0), QubitId(q1))]);
            }
            Self::Swap(q0, q1) => {
                simulator.swap(&[(QubitId(q0), QubitId(q1))]);
            }
        }
    }
}

fn apply_stabilizer_prep_gate(
    rows: &mut [PauliString],
    gates: &mut Vec<StabilizerPrepGate>,
    gate: StabilizerPrepGate,
    num_qubits: usize,
) {
    let clifford = gate.clifford(num_qubits);
    for row in rows {
        *row = clifford.apply(row);
    }
    gates.push(gate);
}

fn stabilizer_state_from_generators(
    num_qubits: usize,
    generators: Vec<Vec<(isize, String)>>,
    seed: u64,
) -> PyResult<CHForm> {
    if generators.len() != num_qubits {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "expected {num_qubits} independent stabilizer generators, got {}",
            generators.len()
        )));
    }

    let mut rows = Vec::with_capacity(num_qubits);
    for (generator_index, generator) in generators.into_iter().enumerate() {
        let mut seen = vec![false; num_qubits];
        let mut paulis = Vec::with_capacity(generator.len());
        for (q, value) in generator {
            let Ok(q) = usize::try_from(q) else {
                return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                    "stabilizer generator {generator_index}: qubit {q} out of bounds (num_qubits={num_qubits})"
                )));
            };
            if q >= num_qubits {
                return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                    "stabilizer generator {generator_index}: qubit {q} out of bounds (num_qubits={num_qubits})"
                )));
            }
            if std::mem::replace(&mut seen[q], true) {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "stabilizer generator {generator_index}: duplicate qubit {q}"
                )));
            }
            let pauli = match value.as_str() {
                "X" => Pauli::X,
                "Y" => Pauli::Y,
                "Z" => Pauli::Z,
                _ => {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Unknown Pauli: {value}"
                    )));
                }
            };
            paulis.push((pauli, QubitId(q)));
        }
        let row = PauliString::with_phase_and_paulis(QuarterPhase::PlusOne, paulis);
        if row.is_identity() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "stabilizer generator {generator_index} must not be identity"
            )));
        }
        rows.push(row);
    }

    for left in 0..num_qubits {
        for right in left + 1..num_qubits {
            if !rows[left].commutes_with(&rows[right]) {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "stabilizer generators {left} and {right} do not commute"
                )));
            }
        }
    }

    // Symplectic Gaussian elimination maps the supplied generators to +Z_i.
    // Reversing the recorded Clifford gates then prepares their unique common
    // +1 eigenstate in CH form.
    let mut gates = Vec::new();
    for pivot in 0..num_qubits {
        let mut candidate = (pivot..num_qubits).find_map(|row| {
            (pivot..num_qubits)
                .find(|&q| matches!(rows[row].get(q), Pauli::X | Pauli::Y))
                .map(|q| (row, q))
        });
        if candidate.is_none() {
            candidate = (pivot..num_qubits).find_map(|row| {
                (pivot..num_qubits)
                    .find(|&q| rows[row].get(q) == Pauli::Z)
                    .map(|q| (row, q))
            });
            if let Some((_, q)) = candidate {
                apply_stabilizer_prep_gate(
                    &mut rows,
                    &mut gates,
                    StabilizerPrepGate::H(q),
                    num_qubits,
                );
            }
        }
        let Some((candidate_row, candidate_qubit)) = candidate else {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "stabilizer generators are not independent",
            ));
        };

        rows.swap(pivot, candidate_row);
        if candidate_qubit != pivot {
            apply_stabilizer_prep_gate(
                &mut rows,
                &mut gates,
                StabilizerPrepGate::Swap(pivot, candidate_qubit),
                num_qubits,
            );
        }
        if rows[pivot].get(pivot) == Pauli::Y {
            apply_stabilizer_prep_gate(
                &mut rows,
                &mut gates,
                StabilizerPrepGate::Sdg(pivot),
                num_qubits,
            );
        }

        for q in pivot + 1..num_qubits {
            match rows[pivot].get(q) {
                Pauli::I => {}
                Pauli::X => apply_stabilizer_prep_gate(
                    &mut rows,
                    &mut gates,
                    StabilizerPrepGate::Cx(pivot, q),
                    num_qubits,
                ),
                Pauli::Y => {
                    apply_stabilizer_prep_gate(
                        &mut rows,
                        &mut gates,
                        StabilizerPrepGate::Sdg(q),
                        num_qubits,
                    );
                    apply_stabilizer_prep_gate(
                        &mut rows,
                        &mut gates,
                        StabilizerPrepGate::Cx(pivot, q),
                        num_qubits,
                    );
                }
                Pauli::Z => apply_stabilizer_prep_gate(
                    &mut rows,
                    &mut gates,
                    StabilizerPrepGate::Cz(pivot, q),
                    num_qubits,
                ),
            }
        }

        for row in pivot + 1..num_qubits {
            if rows[row].get(pivot) == Pauli::X {
                rows[row] = rows[pivot].multiply(&rows[row]);
            }
        }
        apply_stabilizer_prep_gate(
            &mut rows,
            &mut gates,
            StabilizerPrepGate::H(pivot),
            num_qubits,
        );
        if rows[pivot].phase() == QuarterPhase::MinusOne {
            apply_stabilizer_prep_gate(
                &mut rows,
                &mut gates,
                StabilizerPrepGate::X(pivot),
                num_qubits,
            );
        }
        if rows[pivot].phase() != QuarterPhase::PlusOne
            || rows[pivot].get(pivot) != Pauli::Z
            || rows[pivot].paulis().len() != 1
        {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "stabilizer generators do not define a valid stabilizer state",
            ));
        }
    }

    let mut state = CHForm::new_with_seed(num_qubits, seed);
    for gate in gates.into_iter().rev() {
        gate.apply_inverse(&mut state);
    }
    Ok(state)
}

impl PyStabMps {
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

    fn check_probability(p: f64, method: &str) -> PyResult<()> {
        if !(0.0..=1.0).contains(&p) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "{method}: probability must be finite and in [0, 1]"
            )));
        }
        Ok(())
    }

    fn bitstring(&self, value: &Bound<'_, PyAny>, method: &str) -> PyResult<Vec<bool>> {
        let iterator = value.try_iter().map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "{method}: bitstring must be an iterable of bool values"
            ))
        })?;
        let mut bits = Vec::new();
        for (index, item) in iterator.enumerate() {
            let item = item?;
            if !item.is_instance_of::<PyBool>() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "{method}: bitstring item {index} must be bool"
                )));
            }
            bits.push(item.extract::<bool>()?);
        }
        if bits.len() != self.inner.num_qubits() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "{method}: bitstring length {}, expected {}",
                bits.len(),
                self.inner.num_qubits()
            )));
        }
        Ok(bits)
    }

    fn pauli_kind(value: &str) -> PyResult<PauliKind> {
        match value {
            "X" => Ok(PauliKind::X),
            "Y" => Ok(PauliKind::Y),
            "Z" => Ok(PauliKind::Z),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Unknown Pauli: {value}. Use 'X', 'Y', or 'Z'."
            ))),
        }
    }

    fn pauli_string(
        &self,
        values: Vec<(isize, String)>,
        method: &str,
    ) -> PyResult<Vec<(usize, PauliKind)>> {
        values
            .into_iter()
            .map(|(q, value)| {
                let q = self.check_qubit(q, method)?;
                Ok((q, Self::pauli_kind(&value)?))
            })
            .collect()
    }
}

#[pymethods]
impl PyStabMps {
    /// Create a stabilizer-MPS simulator.
    ///
    /// Boolean options other than `for_qec` are tri-state: `None` preserves
    /// the Rust builder default, while `True` or `False` explicitly enables or
    /// disables the option. `for_qec` is enable-only: `True` applies the
    /// preset, while `False` and `None` are identical no-ops.
    /// `measurement` accepts `"exact"`, `"pragmatic"`, or `"lazy"`; when
    /// omitted, the normal default is exact. An explicit value is applied
    /// after `for_qec` and therefore overrides that preset's exact policy.
    /// `max_truncation_error=None` preserves the builder default of
    /// `1e-8`; a float overrides it, and `0.0` disables adaptive truncation
    /// while retaining the SVD cutoff and bond cap. Negative and non-finite
    /// values raise `ValueError`.
    /// `merge_rz` defaults to true for throughput; MAST defaults it to false so
    /// every call immediately exposes its injection-capacity cost. Numerical
    /// flag redetection self-disables while lazy deferred operations are pending.
    ///
    /// `seed` seeds PECOS's buffered RapidHash RNG and the stabilizer tableau.
    /// Fresh instances with the same configuration and call sequence reproduce
    /// stochastic results. For seeded simulators, `reset()` draws the rebuilt
    /// tableau and continuing simulator-RNG seeds from the current simulator
    /// stream rather than replaying the construction stream.
    #[new]
    #[pyo3(signature = (
        num_qubits,
        seed=None,
        max_bond_dim=None,
        merge_rz=None,
        pauli_frame_tracking=None,
        measurement=None,
        for_qec=None,
        auto_grow_bond_dim=None,
        auto_grow_max_bond_dim=None,
        max_truncation_error=None,
        svd_cutoff=None,
        numerical_flag_redetection=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        num_qubits: usize,
        seed: Option<u64>,
        max_bond_dim: Option<usize>,
        merge_rz: Option<bool>,
        pauli_frame_tracking: Option<bool>,
        measurement: Option<&str>,
        for_qec: Option<bool>,
        auto_grow_bond_dim: Option<f64>,
        auto_grow_max_bond_dim: Option<usize>,
        max_truncation_error: Option<f64>,
        svd_cutoff: Option<f64>,
        numerical_flag_redetection: Option<bool>,
    ) -> PyResult<Self> {
        if max_truncation_error.is_some_and(|error| !error.is_finite() || error < 0.0) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "max_truncation_error must be finite and non-negative",
            ));
        }
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
        if let Some(v) = merge_rz {
            b = b.merge_rz(v);
        }
        if let Some(v) = pauli_frame_tracking {
            b = b.pauli_frame_tracking(v);
        }
        if let Some(value) = measurement {
            b = b.measurement(crate::parse_measurement_mode(value)?);
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
        if let Some(c) = svd_cutoff {
            b = b.svd_cutoff(c);
        }
        if let Some(v) = numerical_flag_redetection {
            b = b.numerical_flag_redetection(v);
        }
        Ok(PyStabMps { inner: b.build() })
    }

    /// Reset the quantum state and diagnostics to `|0...0>` and return `self`.
    ///
    /// Configuration is retained. A seeded simulator draws the rebuilt tableau
    /// and continuing simulator-RNG seeds from its current simulator stream,
    /// giving deterministic continuation. An unseeded simulator obtains fresh
    /// entropy.
    fn reset(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.inner.reset();
        slf
    }

    #[getter]
    /// Number of simulated qubits.
    fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    #[getter]
    /// Largest bond dimension currently present.
    ///
    /// Pending lazy operations and merged rotations are flushed first. This
    /// may perform MPS work and mutate diagnostics.
    fn max_bond_dim(&mut self) -> usize {
        self.inner.flush();
        self.inner.max_bond_dim()
    }

    #[getter]
    /// Accumulated approximate infidelity from SVD truncation.
    ///
    /// Pending work is flushed first. Zero means no tracked singular-value
    /// weight above the configured cutoff has been discarded.
    fn truncation_error(&mut self) -> f64 {
        self.inner.flush();
        self.inner.truncation_error()
    }

    #[getter]
    /// Number of pragmatic measurements with uncompensated pre-reduction.
    fn uncompensated_pre_reduction_count(&self) -> u64 {
        self.inner.uncompensated_pre_reduction_count()
    }

    #[getter]
    fn summed_discarded_weight(&self) -> f64 {
        self.inner.summed_discarded_weight()
    }

    #[getter]
    fn lifetime_peak_bond(&self) -> usize {
        self.inner.lifetime_peak_bond()
    }

    #[getter]
    fn branch_vanish_retry_count(&self) -> u64 {
        self.inner.branch_vanish_retry_count()
    }

    #[getter]
    fn deferred_branch_lost_count(&self) -> u64 {
        self.inner.deferred_branch_lost_count()
    }

    #[getter]
    fn measurement(&self) -> &'static str {
        match self.inner.measurement_mode() {
            pecos_stab_tn::stab_mps::MeasurementMode::Exact => "exact",
            pecos_stab_tn::stab_mps::MeasurementMode::Pragmatic => "pragmatic",
            pecos_stab_tn::stab_mps::MeasurementMode::Lazy => "lazy",
        }
    }

    /// Whether the stored tableau/MPS exactly represents the physical state.
    ///
    /// This conservative sufficient predicate covers all pending-state,
    /// policy, uncompensated-reduction, truncation-weight, and deferred-loss
    /// guards.
    fn is_state_exact(&self) -> bool {
        self.inner.is_state_exact()
    }

    /// Materialize pending lazy-measurement operations and merged RZ rotations.
    ///
    /// Read methods call this automatically. It does not materialize a tracked
    /// Pauli frame; use `flush_pauli_frame_to_state()` for that.
    fn flush(&mut self) {
        self.inner.flush();
    }

    /// Materialize tracked Pauli-frame bits into the quantum state.
    ///
    /// This also flushes pending lazy operations and merged rotations. Call it
    /// before physical-state reads when `pauli_frame_tracking=True`.
    fn flush_pauli_frame_to_state(&mut self) {
        self.inner.flush_pauli_frame_to_state();
    }

    /// Return the dense state vector as `(real, imag)` pairs.
    ///
    /// Indexing is little-endian: the entry for `bits` is at
    /// `sum(int(bits[q]) << q)`. This allocates `2**num_qubits` amplitudes and
    /// constructs dense operators, so it is restricted to `num_qubits <= 14`.
    /// Prefer `amplitude_iterative`, `prob_bitstring`, `pauli_expectation`, or
    /// `sample_bitstrings` for scalable reads. Pending work is auto-flushed;
    /// a tracked Pauli frame must be materialized explicitly.
    ///
    /// Raises `ValueError` when more than 14 qubits are present.
    fn state_vector(&mut self, py: Python<'_>) -> PyResult<Py<PyList>> {
        if self.inner.num_qubits() > 14 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "state_vector requires n <= 14",
            ));
        }
        self.inner.flush();
        let sv = self.inner.state_vector();
        let list: Vec<(f64, f64)> = sv.iter().map(|c| (c.re, c.im)).collect();
        Ok(PyList::new(py, &list)?.unbind())
    }

    /// Return a dense-state wavefunction amplitude as `(real, imag)`.
    ///
    /// `bitstring` must contain exactly `num_qubits` Python `bool` values and
    /// `bitstring[q]` specifies qubit `q`. This materializes the full `2**n`
    /// state and is restricted to `n <= 14`; prefer `amplitude_iterative` for
    /// larger systems. Pending work is auto-flushed; materialize a tracked
    /// Pauli frame explicitly.
    ///
    /// Raises `ValueError` for a malformed bitstring or `n > 14`.
    fn amplitude(&mut self, bitstring: &Bound<'_, PyAny>) -> PyResult<(f64, f64)> {
        let bitstring = self.bitstring(bitstring, "amplitude")?;
        if self.inner.num_qubits() > 14 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "amplitude requires n <= 14",
            ));
        }
        self.inner.flush();
        let amplitude = self.inner.amplitude(&bitstring);
        Ok((amplitude.re, amplitude.im))
    }

    /// Return a CAMPS-native iterative amplitude as `(real, imag)`.
    ///
    /// `bitstring` must contain exactly `num_qubits` Python `bool` values and
    /// `bitstring[q]` specifies qubit `q`. Forced projections avoid dense-state
    /// materialization and scale with MPS contractions. Pending work is
    /// auto-flushed; materialize a tracked Pauli frame explicitly.
    ///
    /// Raises `ValueError` for a malformed bitstring.
    fn amplitude_iterative(&mut self, bitstring: &Bound<'_, PyAny>) -> PyResult<(f64, f64)> {
        let bitstring = self.bitstring(bitstring, "amplitude_iterative")?;
        self.inner.flush();
        let amplitude = self.inner.amplitude_iterative(&bitstring);
        Ok((amplitude.re, amplitude.im))
    }

    /// Estimate overlap with a stabilizer state by Monte Carlo sampling.
    ///
    /// `stabilizers` is a complete list of `num_qubits` independent, commuting
    /// +1 generators; each generator is a list of `(qubit, "X"|"Y"|"Z")`
    /// factors. `num_samples` controls statistical error, and `rng_seed`
    /// controls the estimator stream (default 42). Returns `(real, imag)`.
    /// Cost is linear in `num_samples` times sequential stabilizer sampling and
    /// iterative-amplitude work. Pending simulator work is auto-flushed.
    ///
    /// Raises `IndexError` for an out-of-range qubit and `ValueError` for zero
    /// samples, invalid/incomplete/noncommuting generators, or more than 64 qubits.
    #[pyo3(signature = (stabilizers, *, num_samples, rng_seed=None))]
    fn overlap_with_stabilizer(
        &mut self,
        stabilizers: Vec<Vec<(isize, String)>>,
        num_samples: usize,
        rng_seed: Option<u64>,
    ) -> PyResult<(f64, f64)> {
        if self.inner.num_qubits() > 64 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "overlap_with_stabilizer requires n <= 64",
            ));
        }
        if num_samples == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "num_samples must be greater than zero",
            ));
        }
        let state = stabilizer_state_from_generators(
            self.inner.num_qubits(),
            stabilizers,
            rng_seed.unwrap_or(42),
        )?;
        self.inner.flush();
        let overlap = self
            .inner
            .overlap_with_stabilizer(&state, num_samples, rng_seed);
        Ok((overlap.re, overlap.im))
    }

    /// Return the computational-basis probability of `bitstring`.
    ///
    /// The iterable must contain exactly `num_qubits` Python `bool` values;
    /// `bitstring[q]` specifies qubit `q`. The method uses iterative forced
    /// MPS projections rather than a dense state. Pending work is auto-flushed;
    /// materialize a tracked Pauli frame explicitly.
    ///
    /// Raises `ValueError` for a malformed bitstring.
    fn prob_bitstring(&mut self, bitstring: &Bound<'_, PyAny>) -> PyResult<f64> {
        let bitstring = self.bitstring(bitstring, "prob_bitstring")?;
        self.inner.flush();
        Ok(self.inner.prob_bitstring(&bitstring))
    }

    /// Return second Renyi entropy across the cut after qubits `[0, cut)`.
    ///
    /// Entropy uses the natural logarithm. This method materializes the full
    /// state and is limited to `num_qubits <= 14`; pending work is auto-flushed.
    /// Raises `ValueError` unless `0 < cut < num_qubits`, or when `n > 14`.
    fn renyi_s2(&mut self, cut: usize) -> PyResult<f64> {
        let num_qubits = self.inner.num_qubits();
        if cut == 0 || cut >= num_qubits {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "cut must be in (0, num_qubits)",
            ));
        }
        if num_qubits > 14 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "renyi_s2 requires n <= 14 (uses full state vector)",
            ));
        }
        self.inner.flush();
        Ok(self.inner.renyi_s2(cut))
    }

    /// Return second Renyi entropy via Pauli-coefficient enumeration.
    ///
    /// `cut` divides qubits `[0, cut)` from `[cut, num_qubits)`. Cost is
    /// exponential in the nonzero local Bloch components and is capped by the
    /// implementation. Pending work is auto-flushed. Raises `ValueError` for an
    /// invalid cut or an enumeration above the safety limit.
    fn s2_pce(&mut self, cut: usize) -> PyResult<f64> {
        self.inner.flush();
        self.inner
            .s2_pce(cut)
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
    }

    /// Return second Renyi entropy via the PCMPS fallback hierarchy.
    ///
    /// `cut` divides qubits `[0, cut)` from `[cut, num_qubits)`. This tries the
    /// GF(2) null-space methods before Pauli enumeration. Pending work is
    /// auto-flushed. Raises `ValueError` for an invalid cut or excessive
    /// enumeration size.
    fn s2_pcmps(&mut self, cut: usize) -> PyResult<f64> {
        self.inner.flush();
        self.inner
            .s2_pcmps(cut)
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
    }

    /// Run up to `max_sweeps` full-chain Clifford disentangling sweeps.
    ///
    /// Each bond tries 20 two-qubit Clifford candidates using SVD-based entropy
    /// estimates, so this is substantially costlier than a gate update. Use it
    /// after batches that grew bonds. Returns the number of accepted exact
    /// Clifford transformations; ordinary configured SVD truncation still applies.
    fn disentangle(&mut self, max_sweeps: usize) -> usize {
        self.inner.disentangle(max_sweeps)
    }

    #[getter]
    /// Number of SVDs for which the configured maximum bond dimension was binding.
    ///
    /// Pending work is auto-flushed. A nonzero value is a warning to inspect
    /// `truncation_error` and consider a larger cap.
    fn bond_cap_hits(&mut self) -> u64 {
        self.inner.flush();
        self.inner.bond_cap_hits()
    }

    /// Return OFD nullity, the number of tracked flip patterns beyond GF(2) rank.
    ///
    /// Pending work is auto-flushed. The associated ideal bond bound is
    /// `2**ofd_nullity()`.
    fn ofd_nullity(&mut self) -> usize {
        self.inner.flush();
        self.inner.ofd_nullity()
    }

    /// Return the OFD theoretical bond dimension, `2**nullity`.
    ///
    /// Pending work is auto-flushed. The Rust calculation saturates at the
    /// platform's maximum unsigned integer if the power overflows.
    fn theoretical_min_bond_dim(&mut self) -> usize {
        self.inner.flush();
        self.inner.theoretical_min_bond_dim()
    }

    /// Return the GF(2) rank of non-Clifford flip patterns absorbed by OFD.
    ///
    /// Pending work is auto-flushed.
    fn ofd_disentangled_count(&mut self) -> usize {
        self.inner.flush();
        self.inner.ofd_disentangled_count()
    }

    /// Return the total non-Clifford flip patterns recorded in the OFD basis.
    ///
    /// Pending work is auto-flushed.
    fn ofd_total_absorbed(&mut self) -> u64 {
        self.inner.flush();
        u64::try_from(self.inner.ofd_total_absorbed())
            .expect("usize fits in u64 on supported Python targets")
    }

    /// Return runtime non-Clifford path counters as a dictionary.
    ///
    /// Pending work is auto-flushed. Values count dispatch paths since
    /// construction or the last reset.
    fn stats(&mut self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        self.inner.flush();
        crate::stab_mps_stats_to_dict(py, &self.inner.stats)
    }

    // ---- QEC helpers ----

    /// Measure qubit `q`, reset it to `|0>`, and return the observed bit.
    ///
    /// Raises `IndexError` when `q` is outside `[0, num_qubits)`.
    fn reset_qubit(&mut self, q: isize) -> PyResult<bool> {
        let q = self.check_qubit(q, "reset_qubit")?;
        Ok(self.inner.reset_qubit(QubitId(q)))
    }

    /// Prepare qubit `q` in `|0>` by measurement and conditional X.
    ///
    /// The discarded measurement makes this a state-preparation operation.
    /// Raises `IndexError` for an out-of-range qubit.
    fn pz(&mut self, q: isize) -> PyResult<()> {
        let q = self.check_qubit(q, "pz")?;
        self.inner.pz(QubitId(q));
        Ok(())
    }

    /// Prepare qubit `q` in `|+>` by Z reset followed by H.
    ///
    /// Raises `IndexError` for an out-of-range qubit.
    fn px(&mut self, q: isize) -> PyResult<()> {
        let q = self.check_qubit(q, "px")?;
        self.inner.px(QubitId(q));
        Ok(())
    }

    /// Toggle a Pauli X error in the classical frame at qubit `q`.
    ///
    /// Intended when `pauli_frame_tracking=True`. Raises `IndexError` for an
    /// out-of-range qubit. This is O(1) and does not update the MPS immediately.
    fn inject_x_in_frame(&mut self, q: isize) -> PyResult<()> {
        let q = self.check_qubit(q, "inject_x_in_frame")?;
        self.inner.inject_x_in_frame(QubitId(q));
        Ok(())
    }

    /// Toggle a Pauli Y error in the classical frame at qubit `q`.
    ///
    /// Intended when `pauli_frame_tracking=True`. Raises `IndexError` for an
    /// out-of-range qubit. This is O(1) and does not update the MPS immediately.
    fn inject_y_in_frame(&mut self, q: isize) -> PyResult<()> {
        let q = self.check_qubit(q, "inject_y_in_frame")?;
        self.inner.inject_y_in_frame(QubitId(q));
        Ok(())
    }

    /// Toggle a Pauli Z error in the classical frame at qubit `q`.
    ///
    /// Intended when `pauli_frame_tracking=True`. Raises `IndexError` for an
    /// out-of-range qubit. This is O(1) and does not update the MPS immediately.
    fn inject_z_in_frame(&mut self, q: isize) -> PyResult<()> {
        let q = self.check_qubit(q, "inject_z_in_frame")?;
        self.inner.inject_z_in_frame(QubitId(q));
        Ok(())
    }

    /// Toggle several classical-frame Pauli factors.
    ///
    /// `paulis` contains `(qubit, "X"|"Y"|"Z")` pairs. Raises `IndexError`
    /// for an out-of-range qubit and `ValueError` for any other Pauli name.
    /// Cost is linear in the number of factors and does not update the MPS.
    fn inject_paulis_in_frame(&mut self, paulis: Vec<(isize, String)>) -> PyResult<()> {
        let converted: Vec<(QubitId, PauliKind)> = self
            .pauli_string(paulis, "inject_paulis_in_frame")?
            .into_iter()
            .map(|(q, kind)| (QubitId(q), kind))
            .collect();
        self.inner.inject_paulis_in_frame(&converted);
        Ok(())
    }

    /// Return the X component of the tracked Pauli frame at qubit `q`.
    ///
    /// Raises `IndexError` for an out-of-range qubit.
    fn frame_x_bit(&self, q: isize) -> PyResult<bool> {
        let q = self.check_qubit(q, "frame_x_bit")?;
        Ok(self.inner.frame_x_bit(QubitId(q)))
    }

    /// Return the Z component of the tracked Pauli frame at qubit `q`.
    ///
    /// Raises `IndexError` for an out-of-range qubit.
    fn frame_z_bit(&self, q: isize) -> PyResult<bool> {
        let q = self.check_qubit(q, "frame_z_bit")?;
        Ok(self.inner.frame_z_bit(QubitId(q)))
    }

    /// Apply single-qubit depolarizing noise with total probability `p`.
    ///
    /// Returns `None` or the sampled Pauli name `"X"`, `"Y"`, or `"Z"`.
    /// `p` is dimensionless and must be finite in `[0, 1]`. Raises `IndexError`
    /// for an invalid qubit and `ValueError` for an invalid probability.
    fn apply_depolarizing(&mut self, q: isize, p: f64) -> PyResult<Option<String>> {
        let q = self.check_qubit(q, "apply_depolarizing")?;
        Self::check_probability(p, "apply_depolarizing")?;
        Ok(self
            .inner
            .apply_depolarizing(QubitId(q), p)
            .map(|k| format!("{k:?}")))
    }

    /// Apply X with probability `p` and return whether it was applied.
    ///
    /// `p` is dimensionless and must be finite in `[0, 1]`. Raises `IndexError`
    /// for an invalid qubit and `ValueError` for an invalid probability.
    fn apply_bit_flip(&mut self, q: isize, p: f64) -> PyResult<bool> {
        let q = self.check_qubit(q, "apply_bit_flip")?;
        Self::check_probability(p, "apply_bit_flip")?;
        Ok(self.inner.apply_bit_flip(QubitId(q), p))
    }

    /// Apply Z with probability `p` and return whether it was applied.
    ///
    /// `p` is dimensionless and must be finite in `[0, 1]`. Raises `IndexError`
    /// for an invalid qubit and `ValueError` for an invalid probability.
    fn apply_phase_flip(&mut self, q: isize, p: f64) -> PyResult<bool> {
        let q = self.check_qubit(q, "apply_phase_flip")?;
        Self::check_probability(p, "apply_phase_flip")?;
        Ok(self.inner.apply_phase_flip(QubitId(q), p))
    }

    /// Apply independent single-qubit depolarizing noise to `qubits`.
    ///
    /// Each qubit receives a non-identity Pauli with total dimensionless
    /// probability `p`. Raises `IndexError` for any invalid qubit and
    /// `ValueError` unless `p` is finite and in `[0, 1]`.
    fn apply_depolarizing_all(&mut self, qubits: Vec<isize>, p: f64) -> PyResult<()> {
        let qubits = qubits
            .into_iter()
            .map(|q| self.check_qubit(q, "apply_depolarizing_all"))
            .collect::<PyResult<Vec<_>>>()?;
        Self::check_probability(p, "apply_depolarizing_all")?;
        let qs: Vec<QubitId> = qubits.into_iter().map(QubitId).collect();
        self.inner.apply_depolarizing_all(&qs, p);
        Ok(())
    }

    /// Extract one syndrome bit per Pauli generator using supplied ancillas.
    ///
    /// Each generator is a list of `(data_qubit, "X"|"Y"|"Z")` factors;
    /// `ancilla_qubits` must contain one distinct non-overlapping ancilla per
    /// generator. Returns bits in generator order. Raises `IndexError` for an
    /// invalid qubit and `ValueError` for invalid Paulis, length mismatch, or
    /// an ancilla that overlaps its generator.
    fn extract_syndromes(
        &mut self,
        generators: Vec<Vec<(isize, String)>>,
        ancilla_qubits: Vec<isize>,
    ) -> PyResult<Vec<bool>> {
        if generators.len() != ancilla_qubits.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "extract_syndromes: one ancilla per generator required",
            ));
        }
        let ancilla_qubits = ancilla_qubits
            .into_iter()
            .map(|q| self.check_qubit(q, "extract_syndromes"))
            .collect::<PyResult<Vec<_>>>()?;
        let gens: Vec<Vec<(usize, PauliKind)>> = generators
            .into_iter()
            .map(|generator| self.pauli_string(generator, "extract_syndromes"))
            .collect::<PyResult<Vec<_>>>()?;
        for (generator, &ancilla) in gens.iter().zip(&ancilla_qubits) {
            if generator.iter().any(|&(q, _)| q == ancilla) {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "extract_syndromes: ancilla {ancilla} overlaps with generator data qubit"
                )));
            }
        }
        let ancs: Vec<QubitId> = ancilla_qubits.into_iter().map(QubitId).collect();
        Ok(self.inner.extract_syndromes(&gens, &ancs))
    }

    /// Return the expectation of a Hermitian Pauli string.
    ///
    /// `pauli_string` lists non-identity `(qubit, "X"|"Y"|"Z")` factors.
    /// Pending work is auto-flushed; materialize a tracked Pauli frame
    /// explicitly. Raises `IndexError` for an invalid qubit and `ValueError`
    /// for an invalid Pauli name.
    fn pauli_expectation(&mut self, pauli_string: Vec<(isize, String)>) -> PyResult<f64> {
        let ps = self.pauli_string(pauli_string, "pauli_expectation")?;
        self.inner.flush();
        Ok(self.inner.pauli_expectation(&ps))
    }

    /// Return fidelity with the subspace stabilized by `stabilizers`.
    ///
    /// Each generator lists `(qubit, "X"|"Y"|"Z")` factors. Cost is
    /// exponential in generator count (`2**k` expectations), so `k <= 30` is
    /// enforced. Pending work is auto-flushed. Raises `IndexError` for an
    /// invalid qubit and `ValueError` for an invalid Pauli or too many generators.
    fn code_state_fidelity(&mut self, stabilizers: Vec<Vec<(isize, String)>>) -> PyResult<f64> {
        if stabilizers.len() > 30 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "code_state_fidelity supports at most 30 stabilizer generators",
            ));
        }
        let stabs: Vec<Vec<(usize, PauliKind)>> = stabilizers
            .into_iter()
            .map(|generator| self.pauli_string(generator, "code_state_fidelity"))
            .collect::<PyResult<Vec<_>>>()?;
        self.inner.flush();
        Ok(self.inner.code_state_fidelity(&stabs))
    }

    /// Sample `num_shots` computational-basis rows by cloning once per shot.
    ///
    /// Every returned row uses `row[q] == qubit q`; the original state is
    /// preserved and its RNG advances. Prefer `sample_bitstrings`: this method
    /// pays for a full clone and collapse per shot, while prefix sharing has
    /// measured tens-to-hundreds-fold speedups on the repository's 1,000-shot
    /// example workloads. The two sampler methods do not share an RNG stream,
    /// so their seeded results are not shot-for-shot comparable. A negative or
    /// oversized count raises `OverflowError`.
    fn sample_bitstring(&mut self, num_shots: usize) -> Vec<Vec<bool>> {
        self.inner.sample_bitstring(num_shots)
    }

    /// Sample `num_shots` computational-basis rows with shared prefixes.
    ///
    /// Every returned row uses `row[q] == qubit q`. The original state is
    /// preserved and its RNG advances. Distinct measurement-prefix projections
    /// are shared across all shots taking that branch, avoiding the per-shot
    /// cloning cost of `sample_bitstring`; the repository's 1,000-shot example
    /// measures hardware-dependent tens-to-hundreds-fold speedups. Output is in
    /// lexicographic tree order, not input shot order. Pending merged rotations
    /// and lazy operations are handled internally. The two sampler methods do
    /// not share an RNG stream, so their seeded results are not shot-for-shot
    /// comparable. A negative or oversized count raises `OverflowError`.
    fn sample_bitstrings(&mut self, num_shots: usize) -> Vec<Vec<bool>> {
        self.inner.sample_bitstrings(num_shots)
    }

    // ---- Gate dispatch (matches pecos-rslib pattern) ----

    /// Apply one accepted gate symbol to one qubit.
    ///
    /// `location` is a zero-based qubit index. RX, RY, and RZ require
    /// `params={"angle": radians}`; angles are floating-point radians.
    /// Measurement symbols return `0` or `1`; other gates return `None`.
    /// Raises `IndexError` for an invalid qubit and `ValueError` for an unknown
    /// symbol or missing/non-numeric angle. See `run_gate` for the symbol table.
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

    /// Apply one accepted two-qubit gate symbol to an ordered qubit pair.
    ///
    /// `location` must be a two-element tuple of distinct zero-based indices;
    /// for controlled gates the first qubit is the control. RZZ requires
    /// `params={"angle": radians}`. Raises `IndexError` for an out-of-range
    /// index and `ValueError` for bad arity, repeated qubits, an unknown symbol,
    /// or a missing/non-numeric angle. See `run_gate` for the symbol table.
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
        Ok(output.into())
    }
}
