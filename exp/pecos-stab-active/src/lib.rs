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

//! Exact per-shot simulation in active stabilizer coordinates.
//!
//! Inspired by Clifft (arXiv:2604.27058) and `SymFT` (arXiv:2607.28600).
//! With signed Hermitian-Y generators and ordered indices `active`, the state is
//! `sum_x a[x] D_active[0]^x_0 ... D_active[k-1]^x_{k-1} |phi>`.
//! Dormant stabilizers fix every represented state. Only the amplitude vector has
//! exponential size, `2^k`; tableau storage and Clifford updates are polynomial.
//!
//! The contract preserves relative phases up to one overall global phase and
//! floating-point roundoff. Consequently `StateVectorSimulator` is intentionally
//! not implemented. Clifford angles use exact `Angle64` comparisons.
//!
//! # Panics
//! Rotation methods (including trait defaults built on them) panic when promotion
//! would exceed the configured active width, with `active width W exceeds limit L`.
//! Gate and measurement indices must be valid and batches must use distinct qubits.

use num_complex::Complex64;
use pecos_core::{Angle64, QubitId};
use pecos_random::{PecosRng, RngManageable};
use pecos_simulators::{
    ArbitraryRotationGateable, CliffordGateable, ForcedMeasurement, MeasurementResult,
    QuantumSimulator, SparseStabY,
};
use pecos_stab_tn::stab_mps::coordinate_tableau::{
    self, CoordinateDecomposition, CoordinateGate, MeasurementCase,
};
use pecos_stab_tn::stab_mps::measure::EXPECTATION_ENDPOINT_TOLERANCE;
pub use pecos_stab_tn::stab_mps::pauli_decomp::PauliKindForDecomp;

pub mod heisenberg;
pub use heisenberg::{
    AffineSign, CompileError, HeisenbergOp, HeisenbergProgram, ShotResult, VirtualPauli,
};

/// A normalized dense vector on an ordered subset of stabilizer coordinates.
///
/// The default maximum active width is 26. Physical Cliffords update only the
/// signed tableau. Global phase is unspecified; relative phases are preserved.
#[derive(Clone, Debug)]
pub struct StabActive {
    tableau: SparseStabY,
    active: Vec<usize>,
    amplitudes: Vec<Complex64>,
    rng: PecosRng,
    peak_width: usize,
    max_width: usize,
}

impl StabActive {
    /// Construct `|0^n>` with a fresh random seed and no active coordinates.
    #[must_use]
    pub fn new(num_qubits: usize) -> Self {
        Self::with_seed(num_qubits, pecos_random::rng_manageable::time_seed())
    }

    /// Construct `|0^n>` with a reproducible measurement RNG.
    #[must_use]
    pub fn with_seed(num_qubits: usize, seed: u64) -> Self {
        Self {
            tableau: SparseStabY::with_seed(num_qubits, seed).with_destab_sign_tracking(),
            active: Vec::new(),
            amplitudes: vec![Complex64::new(1.0, 0.0)],
            rng: PecosRng::seed_from_u64(seed),
            peak_width: 0,
            max_width: 26,
        }
    }

    /// Set the maximum permitted active width, retaining the represented state.
    ///
    /// # Panics
    /// Panics if the limit is below the current width or cannot index a complex
    /// amplitude vector on this platform.
    #[must_use]
    pub fn with_max_active_width(mut self, limit: usize) -> Self {
        assert!(
            limit <= (isize::MAX.unsigned_abs() / size_of::<Complex64>()).ilog2() as usize,
            "active width limit cannot index a complex vector"
        );
        assert!(
            self.active.len() <= limit,
            "active width {} exceeds limit {limit}",
            self.active.len()
        );
        self.max_width = limit;
        self
    }

    /// Number of bits indexing the current amplitude vector.
    #[must_use]
    pub fn active_width(&self) -> usize {
        self.active.len()
    }

    /// Largest active width since construction or the most recent reset.
    #[must_use]
    pub fn peak_active_width(&self) -> usize {
        self.peak_width
    }

    fn parts(&self, pauli: &[(usize, PauliKindForDecomp)]) -> CoordinateDecomposition {
        coordinate_tableau::decompose(&self.tableau, &self.active, pauli, false)
    }

    fn normalize(&mut self) {
        let norm = self
            .amplitudes
            .iter()
            .map(Complex64::norm_sqr)
            .sum::<f64>()
            .sqrt();
        assert!(norm > 0.0, "cannot normalize a zero state");
        for amplitude in &mut self.amplitudes {
            *amplitude /= norm;
        }
    }

    fn pauli_rotation(&mut self, theta: Angle64, pauli: &[(usize, PauliKindForDecomp)]) {
        self.rotate_pauli(theta, pauli, false);
    }

    /// Apply `exp(-i (-1)^negative theta P / 2)` for a Hermitian Pauli tensor P.
    /// Empty support denotes identity; its physically irrelevant global phase is omitted.
    /// Exact quarter-turn angles update only the signed Clifford tableau.
    ///
    /// # Panics
    /// Panics for repeated or invalid qubits, or if promotion exceeds the width limit.
    pub fn rotate_pauli(
        &mut self,
        theta: Angle64,
        pauli: &[(usize, PauliKindForDecomp)],
        negative: bool,
    ) {
        for (i, &(q, _)) in pauli.iter().enumerate() {
            assert!(
                q < self.num_qubits() && !pauli[..i].iter().any(|&(r, _)| r == q),
                "Pauli factors must name distinct valid qubits"
            );
        }
        if pauli.is_empty() {
            return;
        }
        let theta = if negative { -theta } else { theta };
        if let Some(turns) = clifford_turns(theta) {
            rotate_tableau(&mut self.tableau, pauli, turns);
            return;
        }
        let mut parts = self.parts(pauli);
        if !parts.dormant_flips.is_empty() {
            let width = self.active.len() + 1;
            assert!(
                width <= self.max_width,
                "active width {width} exceeds limit {}",
                self.max_width
            );
            coordinate_tableau::promote(&mut self.tableau, &mut self.active, pauli, false);
            self.amplitudes
                .resize(self.amplitudes.len() * 2, Complex64::new(0.0, 0.0));
            self.peak_width = self.peak_width.max(width);
            parts = self.parts(pauli);
        }
        let flip = mask(&parts.active_flips);
        let sign = mask(&parts.active_signs);
        // Convert the fixed-point magnitude before restoring the sign: subtracting
        // two floating-point turns would erase tiny negative rotations near zero.
        let radians = if theta > Angle64::HALF_TURN {
            -(-theta).to_radians()
        } else {
            theta.to_radians()
        };
        let (sine, cosine) = (radians / 2.0).sin_cos();
        let coefficient = Complex64::new(0.0, -sine) * parts.phase;
        if flip == 0 {
            for (index, amplitude) in self.amplitudes.iter_mut().enumerate() {
                *amplitude *= cosine + coefficient * parity(index & sign);
            }
        } else {
            for index in 0..self.amplitudes.len() {
                let partner = index ^ flip;
                if index < partner {
                    let a = self.amplitudes[index];
                    let b = self.amplitudes[partner];
                    self.amplitudes[index] = cosine * a + coefficient * parity(partner & sign) * b;
                    self.amplitudes[partner] = cosine * b + coefficient * parity(index & sign) * a;
                }
            }
        }
        self.normalize();
    }

    fn active_expectation(&self, parts: &CoordinateDecomposition) -> Complex64 {
        let flip = mask(&parts.active_flips);
        let sign = mask(&parts.active_signs);
        self.amplitudes
            .iter()
            .enumerate()
            .map(|(index, a)| {
                self.amplitudes[index ^ flip].conj() * parts.phase * parity(index & sign) * a
            })
            .sum()
    }

    /// Return the outcome-one probability used by measurement and the dense oracle.
    ///
    /// Active expectations within [`EXPECTATION_ENDPOINT_TOLERANCE`] of either
    /// endpoint are treated as that exact eigenvalue before computing probability
    /// or drawing from the RNG. This matches the shared measurement policy and
    /// prevents a forced projector from amplifying a cancellation residue.
    fn probability_one(&self, parts: &CoordinateDecomposition) -> f64 {
        match parts.measurement_case() {
            MeasurementCase::Random => 0.5,
            MeasurementCase::Deterministic => f64::from(parts.phase.re < 0.0),
            MeasurementCase::Active => {
                let expectation = self.active_expectation(parts).re;
                let expectation = if 1.0 - expectation.abs() <= EXPECTATION_ENDPOINT_TOLERANCE {
                    expectation.signum()
                } else {
                    expectation
                };
                ((1.0 - expectation) / 2.0).clamp(0.0, 1.0)
            }
        }
    }

    fn coordinate_gate(&mut self, gate: CoordinateGate) {
        match gate {
            CoordinateGate::Sdg(bit) => {
                for (index, amplitude) in self.amplitudes.iter_mut().enumerate() {
                    if index & (1 << bit) != 0 {
                        *amplitude *= Complex64::new(0.0, -1.0);
                    }
                }
            }
            CoordinateGate::H(bit) => {
                for index in 0..self.amplitudes.len() {
                    if index & (1 << bit) == 0 {
                        let partner = index ^ (1 << bit);
                        let a = self.amplitudes[index];
                        let b = self.amplitudes[partner];
                        self.amplitudes[index] = (a + b) * std::f64::consts::FRAC_1_SQRT_2;
                        self.amplitudes[partner] = (a - b) * std::f64::consts::FRAC_1_SQRT_2;
                    }
                }
            }
            CoordinateGate::Cx(control, target) => {
                for index in 0..self.amplitudes.len() {
                    if index & (1 << control) != 0 && index & (1 << target) == 0 {
                        self.amplitudes.swap(index, index ^ (1 << target));
                    }
                }
            }
        }
    }

    fn measure(&mut self, qubit: usize, forced: Option<bool>) -> MeasurementResult {
        self.measure_pauli_inner(&[(qubit, PauliKindForDecomp::Z)], false, forced)
    }

    /// Measure `(-1)^negative P`, returning its raw signed outcome (one means -1).
    /// Empty support denotes the signed identity and has a deterministic outcome.
    ///
    /// # Panics
    /// Panics for repeated or invalid qubits.
    pub fn measure_pauli(
        &mut self,
        pauli: &[(usize, PauliKindForDecomp)],
        negative: bool,
    ) -> MeasurementResult {
        self.measure_pauli_inner(pauli, negative, None)
    }

    fn measure_pauli_inner(
        &mut self,
        pauli: &[(usize, PauliKindForDecomp)],
        negative: bool,
        forced: Option<bool>,
    ) -> MeasurementResult {
        let parts = coordinate_tableau::decompose(&self.tableau, &self.active, pauli, negative);
        let probability = self.probability_one(&parts);
        let is_deterministic = probability <= 0.0 || probability >= 1.0;
        let outcome = if is_deterministic {
            probability >= 1.0
        } else {
            forced.unwrap_or_else(|| self.rng.random_bool(probability))
        };
        match parts.measurement_case() {
            MeasurementCase::Random => {
                coordinate_tableau::measure_random(
                    &mut self.tableau,
                    &self.active,
                    pauli,
                    negative,
                    outcome,
                );
            }
            MeasurementCase::Deterministic => {}
            MeasurementCase::Active => {
                let basis = coordinate_tableau::measurement_basis(
                    &mut self.tableau,
                    &self.active,
                    pauli,
                    negative,
                );
                for gate in basis.gates {
                    self.coordinate_gate(gate);
                }
                let value = outcome ^ basis.negative;
                let bit_mask = 1 << basis.pivot_bit;
                self.amplitudes = self
                    .amplitudes
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| (index & bit_mask != 0) == value)
                    .map(|(_, &a)| a)
                    .collect();
                coordinate_tableau::demote(
                    &mut self.tableau,
                    &mut self.active,
                    basis.pivot_bit,
                    value,
                );
                self.normalize();
            }
        }
        MeasurementResult {
            outcome,
            is_deterministic,
        }
    }
}

fn clifford_turns(theta: Angle64) -> Option<usize> {
    [
        Angle64::ZERO,
        Angle64::QUARTER_TURN,
        Angle64::HALF_TURN,
        Angle64::THREE_QUARTERS_TURN,
    ]
    .iter()
    .position(|&angle| theta == angle)
}

fn rotate_tableau(tableau: &mut SparseStabY, pauli: &[(usize, PauliKindForDecomp)], turns: usize) {
    let target = QubitId(pauli[pauli.len() - 1].0);
    for &(q, kind) in pauli {
        if kind == PauliKindForDecomp::Y {
            tableau.szdg(&[QubitId(q)]);
        }
        if kind != PauliKindForDecomp::Z {
            tableau.h(&[QubitId(q)]);
        }
    }
    for &(q, _) in &pauli[..pauli.len() - 1] {
        tableau.cx(&[(QubitId(q), target)]);
    }
    for _ in 0..turns {
        tableau.sz(&[target]);
    }
    for &(q, _) in pauli[..pauli.len() - 1].iter().rev() {
        tableau.cx(&[(QubitId(q), target)]);
    }
    for &(q, kind) in pauli.iter().rev() {
        if kind != PauliKindForDecomp::Z {
            tableau.h(&[QubitId(q)]);
        }
        if kind == PauliKindForDecomp::Y {
            tableau.sz(&[QubitId(q)]);
        }
    }
}

fn mask(bits: &[usize]) -> usize {
    bits.iter().fold(0, |value, &bit| value | (1 << bit))
}
fn parity(value: usize) -> f64 {
    if value.count_ones().is_multiple_of(2) {
        1.0
    } else {
        -1.0
    }
}

impl QuantumSimulator for StabActive {
    fn reset(&mut self) -> &mut Self {
        self.tableau.reset();
        self.active.clear();
        self.amplitudes.clear();
        self.amplitudes.push(Complex64::new(1.0, 0.0));
        self.peak_width = 0;
        self
    }
    fn num_qubits(&self) -> usize {
        self.tableau.num_qubits()
    }
}

impl CliffordGateable for StabActive {
    fn sz(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.tableau.sz(qubits);
        self
    }
    fn h(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.tableau.h(qubits);
        self
    }
    fn cx(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        self.tableau.cx(pairs);
        self
    }
    fn mz(&mut self, qubits: &[QubitId]) -> Vec<MeasurementResult> {
        qubits
            .iter()
            .map(|q| self.measure(q.index(), None))
            .collect()
    }
}

impl ArbitraryRotationGateable for StabActive {
    /// Apply `exp(-i theta X/2)` to each qubit.
    ///
    /// # Panics
    /// Panics on invalid qubits or when `active width W exceeds limit L`.
    fn rx(&mut self, theta: Angle64, qubits: &[QubitId]) -> &mut Self {
        for q in qubits {
            self.pauli_rotation(theta, &[(q.index(), PauliKindForDecomp::X)]);
        }
        self
    }
    /// Apply `exp(-i theta Z/2)` to each qubit.
    ///
    /// # Panics
    /// Panics on invalid qubits or when `active width W exceeds limit L`.
    fn rz(&mut self, theta: Angle64, qubits: &[QubitId]) -> &mut Self {
        for q in qubits {
            self.pauli_rotation(theta, &[(q.index(), PauliKindForDecomp::Z)]);
        }
        self
    }
    /// Apply `exp(-i theta Z_a Z_b/2)` to each pair.
    ///
    /// # Panics
    /// Panics on invalid qubits or when `active width W exceeds limit L`.
    fn rzz(&mut self, theta: Angle64, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        for (a, b) in pairs {
            self.pauli_rotation(
                theta,
                &[
                    (a.index(), PauliKindForDecomp::Z),
                    (b.index(), PauliKindForDecomp::Z),
                ],
            );
        }
        self
    }
}

impl ForcedMeasurement for StabActive {
    fn mz_forced(&mut self, qubit: usize, forced_outcome: bool) -> MeasurementResult {
        self.measure(qubit, Some(forced_outcome))
    }
}

impl RngManageable for StabActive {
    type Rng = PecosRng;
    fn set_rng(&mut self, rng: PecosRng) {
        self.rng = rng;
    }
    fn rng(&self) -> &PecosRng {
        &self.rng
    }
    fn rng_mut(&mut self) -> &mut PecosRng {
        &mut self.rng
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod compatibility_tests;
