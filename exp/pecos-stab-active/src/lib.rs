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
    QuantumSimulator,
};
use pecos_stab_tn::stab_mps::coordinate_tableau::{
    CoordinateDecomposition, CoordinateGate, MeasurementCase,
};
use pecos_stab_tn::stab_mps::measure::EXPECTATION_ENDPOINT_TOLERANCE;
pub use pecos_stab_tn::stab_mps::pauli_decomp::PauliKindForDecomp;

mod structure;
use structure::{ActiveStructure, MeasurementData};

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
    structure: ActiveStructure,
    amplitudes: Vec<Complex64>,
    rng: PecosRng,
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
            structure: ActiveStructure::with_seed(num_qubits, seed),
            amplitudes: vec![Complex64::new(1.0, 0.0)],
            rng: PecosRng::seed_from_u64(seed),
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
            self.structure.width() <= limit,
            "active width {} exceeds limit {limit}",
            self.structure.width()
        );
        self.max_width = limit;
        self
    }

    /// Number of bits indexing the current amplitude vector.
    #[must_use]
    pub fn active_width(&self) -> usize {
        self.structure.width()
    }

    /// Largest active width since construction or the most recent reset.
    #[must_use]
    pub fn peak_active_width(&self) -> usize {
        self.structure.peak_width()
    }

    #[cfg(test)]
    fn parts(&self, pauli: &[(usize, PauliKindForDecomp)]) -> CoordinateDecomposition {
        self.structure.measurement(pauli, false).into_parts()
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
        let rotation = self.structure.rotation(theta, pauli, negative);
        if rotation.needs_promotion() {
            let width = self.structure.width() + 1;
            assert!(
                width <= self.max_width,
                "active width {width} exceeds limit {}",
                self.max_width
            );
        }
        let Some(work) = self.structure.apply_rotation(rotation) else {
            return;
        };
        if work.double {
            self.amplitudes
                .resize(self.amplitudes.len() * 2, Complex64::new(0.0, 0.0));
        }
        let theta = work.theta;
        let parts = work.parts;
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
    fn measurement_probability(&self, measurement: &MeasurementData) -> f64 {
        let parts = measurement.parts();
        match measurement.case() {
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

    #[cfg(test)]
    fn probability_one(&self, parts: &CoordinateDecomposition) -> f64 {
        self.measurement_probability(&MeasurementData::new(parts.clone()))
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
        let measurement = self.structure.measurement(pauli, negative);
        let probability = self.measurement_probability(measurement.data());
        let is_deterministic = probability <= 0.0 || probability >= 1.0;
        let outcome = if is_deterministic {
            probability >= 1.0
        } else {
            forced.unwrap_or_else(|| self.rng.random_bool(probability))
        };
        if let Some(active) = self.structure.apply_measurement(measurement, outcome) {
            for &gate in &active.gates {
                self.coordinate_gate(gate);
            }
            let projection = active.projection;
            let value = projection.value;
            let bit_mask = 1 << projection.pivot_bit;
            self.amplitudes = self
                .amplitudes
                .iter()
                .enumerate()
                .filter(|(index, _)| (index & bit_mask != 0) == value)
                .map(|(_, &a)| a)
                .collect();
            self.structure.demote(projection);
            self.normalize();
        }
        MeasurementResult {
            outcome,
            is_deterministic,
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
        self.structure.reset();
        self.amplitudes.clear();
        self.amplitudes.push(Complex64::new(1.0, 0.0));
        self
    }
    fn num_qubits(&self) -> usize {
        self.structure.num_qubits()
    }
}

impl CliffordGateable for StabActive {
    fn sz(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.structure.sz(qubits);
        self
    }
    fn h(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.structure.h(qubits);
        self
    }
    fn cx(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        self.structure.cx(pairs);
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
