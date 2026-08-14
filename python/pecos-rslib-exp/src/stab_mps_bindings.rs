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
use pyo3::types::{PyDict, PyList, PySet, PyTuple};

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
    generators: Vec<Vec<(usize, String)>>,
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
            if q >= num_qubits {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
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
    /// Create a stabilizer-MPS simulator.
    ///
    /// Boolean options are tri-state: `None` preserves the Rust builder
    /// default, while `True` or `False` explicitly enables or disables the
    /// option. `max_truncation_error=None` preserves the builder default of
    /// `1e-8`; a float overrides it, and `0.0` disables adaptive truncation
    /// while retaining the SVD cutoff and bond cap.
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
        lazy_measure: Option<bool>,
        for_qec: Option<bool>,
        auto_grow_bond_dim: Option<f64>,
        auto_grow_max_bond_dim: Option<usize>,
        max_truncation_error: Option<f64>,
        svd_cutoff: Option<f64>,
        numerical_flag_redetection: Option<bool>,
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
        if let Some(v) = merge_rz {
            b = b.merge_rz(v);
        }
        if let Some(v) = pauli_frame_tracking {
            b = b.pauli_frame_tracking(v);
        }
        if let Some(v) = lazy_measure {
            b = b.lazy_measure(v);
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

    /// Wavefunction amplitude for a computational-basis bitstring.
    fn amplitude(&self, bitstring: Vec<bool>) -> PyResult<(f64, f64)> {
        if bitstring.len() != self.inner.num_qubits() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "bitstring length mismatch",
            ));
        }
        if self.inner.num_qubits() > 14 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "amplitude requires n <= 14",
            ));
        }
        let amplitude = self.inner.amplitude(&bitstring);
        Ok((amplitude.re, amplitude.im))
    }

    /// CAMPS-native iterative wavefunction amplitude.
    fn amplitude_iterative(&self, bitstring: Vec<bool>) -> PyResult<(f64, f64)> {
        if bitstring.len() != self.inner.num_qubits() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "bitstring length mismatch",
            ));
        }
        let amplitude = self.inner.amplitude_iterative(&bitstring);
        Ok((amplitude.re, amplitude.im))
    }

    /// Monte Carlo estimate of the overlap with a stabilizer state specified
    /// by a complete set of independent +1 Pauli generators.
    #[pyo3(signature = (stabilizers, *, num_samples, rng_seed=None))]
    fn overlap_with_stabilizer(
        &self,
        stabilizers: Vec<Vec<(usize, String)>>,
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
        let overlap = self
            .inner
            .overlap_with_stabilizer(&state, num_samples, rng_seed);
        Ok((overlap.re, overlap.im))
    }

    fn prob_bitstring(&self, bitstring: Vec<bool>) -> f64 {
        self.inner.prob_bitstring(&bitstring)
    }

    /// Second Renyi entropy from the full state vector.
    fn renyi_s2(&self, cut: usize) -> PyResult<f64> {
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
        Ok(self.inner.renyi_s2(cut))
    }

    /// Second Renyi entropy via Pauli coefficient enumeration.
    fn s2_pce(&self, cut: usize) -> PyResult<f64> {
        self.inner
            .s2_pce(cut)
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
    }

    /// Second Renyi entropy via the PCMPS hierarchy.
    fn s2_pcmps(&self, cut: usize) -> PyResult<f64> {
        self.inner
            .s2_pcmps(cut)
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
    }

    /// Run Clifford disentangling sweeps and return the gates applied.
    fn disentangle(&mut self, max_sweeps: usize) -> usize {
        self.inner.disentangle(max_sweeps)
    }

    #[getter]
    fn bond_cap_hits(&self) -> u64 {
        self.inner.bond_cap_hits()
    }

    fn ofd_nullity(&self) -> usize {
        self.inner.ofd_nullity()
    }

    fn theoretical_min_bond_dim(&self) -> usize {
        self.inner.theoretical_min_bond_dim()
    }

    fn ofd_disentangled_count(&self) -> usize {
        self.inner.ofd_disentangled_count()
    }

    fn ofd_total_absorbed(&self) -> u64 {
        u64::try_from(self.inner.ofd_total_absorbed())
            .expect("usize fits in u64 on supported Python targets")
    }

    /// Runtime non-Clifford path counters.
    fn stats(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        crate::stab_mps_stats_to_dict(py, &self.inner.stats)
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

    fn apply_bit_flip(&mut self, q: usize, p: f64) -> PyResult<bool> {
        self.check_qubit(q, "apply_bit_flip")?;
        Ok(self.inner.apply_bit_flip(QubitId(q), p))
    }

    fn apply_phase_flip(&mut self, q: usize, p: f64) -> PyResult<bool> {
        self.check_qubit(q, "apply_phase_flip")?;
        Ok(self.inner.apply_phase_flip(QubitId(q), p))
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

    /// Prefix-sharing perfect sampling: shares each distinct measurement-prefix
    /// projection across all shots taking that branch. Output is in
    /// lexicographic tree order, not per-shot order.
    fn sample_bitstrings(&mut self, num_shots: usize) -> Vec<Vec<bool>> {
        self.inner.sample_bitstrings(num_shots)
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
