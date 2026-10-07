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

//! Exact simulation of third-order phase-polynomial (3PP) states.
//!
//! The state is `2^(-r/2) sum_y w8^p(y) |x0 xor B y>`, with full-column-rank
//! binary `B` and a sparse, ordered parity sum `p` modulo eight. The constant
//! phase is discarded; amplitudes and vectors use this stored convention.
//!
//! Supports X, Y, Z, S, T and their inverses, CX, CZ, controlled S and its
//! inverse, CCZ, Z/X preparations, all Z strings, and exactly the compatible
//! X strings of arXiv:2610.06811, Section 3, Theorem 3.2. Other X measurements
//! return [`IncompatibleMeasurement`] without modifying state or randomness.
//! There is no Hadamard or general Clifford interface.
//!
//! For `K` stored parities, storage is `O(n^2 + Kn)` bits, affine restrictions
//! cost `O((n+K)n + Kn log(K+1))` bit operations, and derivative construction
//! and measurement/expectation queries cost `O(Kn^2 + n^3)` arithmetic operations.
//! Each operation adds only a bounded number of parities, so `K = O(m)` after
//! `m` gates/measurements and total simulation cost is polynomial in circuit size.
//! Exact quadratic sums use variable elimination; only [`PhasePoly::state_vector`]
//! enumerates exponentially many basis states. Floating point is used only at
//! query boundaries and for case-2c sampling via a uniform `f64` comparison.
//! Probability-1/2 outcomes use a fair random bit.
//!
//! # Panics
//! Gate and measurement indices must be in range. Every batch/string must
//! contain distinct qubits, including across pairs/triples in a batch.
//!
//! ```
//! use pecos_phase_poly::{PhasePoly, XMeasurementCase};
//! use pecos_core::QubitId;
//! let mut state = PhasePoly::with_seed(1, 17);
//! state.px(&[QubitId(0)]).t(&[QubitId(0)]);
//! let (probabilities, case) = state.x_probabilities(&[QubitId(0)]).unwrap();
//! assert_eq!(case, XMeasurementCase::Case2c);
//! assert!(probabilities[0] > 0.85);
//! ```

use binary::{Mask, solve};
use num_complex::Complex64;
use pecos_core::{Pauli, PauliString, QuarterPhase, QubitId};
use pecos_random::PecosRng;
use pecos_simulators::{ForcedMeasurement, MeasurementResult, QuantumSimulator};
use quadratic::{Clifford, GaussSum, root};
use std::collections::BTreeMap;

mod binary;
pub mod distillation;
mod measurement;
mod quadratic;
pub mod runner;
pub use measurement::{IncompatibleMeasurement, XMeasurementCase};

#[cfg(test)]
mod tests;

/// A fixed-qubit 3PP state with an independent, reproducible measurement RNG.
///
/// Global phase is omitted. Masks always have `ceil(n/64)` words, even when
/// coordinates are appended or deleted, so equal parities are equal map keys.
#[derive(Clone, Debug)]
pub struct PhasePoly {
    x0: Vec<bool>,
    rows: Vec<Mask>,
    r: usize,
    phase: BTreeMap<Mask, u8>,
    rng: PecosRng,
}

impl PhasePoly {
    /// Construct `|0^n>` with a fresh time-based random seed.
    #[must_use]
    pub fn new(num_qubits: usize) -> Self {
        Self::with_seed(num_qubits, pecos_random::rng_manageable::time_seed())
    }

    /// Construct `|0^n>` with a reproducible measurement RNG.
    #[must_use]
    pub fn with_seed(num_qubits: usize, seed: u64) -> Self {
        Self {
            x0: vec![false; num_qubits],
            rows: vec![Mask::zero(num_qubits); num_qubits],
            r: 0,
            phase: BTreeMap::new(),
            rng: PecosRng::seed_from_u64(seed),
        }
    }

    /// Dimension of the current affine support (between zero and `num_qubits`).
    #[must_use]
    pub fn support_dimension(&self) -> usize {
        self.r
    }

    /// Number of nonzero parity terms in the stored phase polynomial.
    #[must_use]
    pub fn num_phase_terms(&self) -> usize {
        self.phase.len()
    }

    fn validate(&self, qubits: impl IntoIterator<Item = QubitId>) {
        let mut seen = Mask::zero(self.x0.len());
        for q in qubits {
            assert!(q.0 < self.x0.len(), "qubit index out of range");
            assert!(!seen.get(q.0), "qubits must be distinct");
            seen.toggle(q.0);
        }
    }

    fn add_term(&mut self, mask: Mask, coefficient: u8) {
        if mask.is_zero() {
            return;
        }
        let coefficient = (self.phase.get(&mask).copied().unwrap_or(0) + coefficient) % 8;
        if coefficient == 0 {
            self.phase.remove(&mask);
        } else {
            self.phase.insert(mask, coefficient);
        }
    }

    fn add_affine(&mut self, mask: Mask, constant: bool, coefficient: u8) {
        self.add_term(
            mask,
            if constant {
                (8 - coefficient) % 8
            } else {
                coefficient
            },
        );
    }

    /// Boolean inclusion-exclusion for products of at most three affine forms.
    /// arXiv:2610.06811, phase-polynomials appendix, equivalent-forms theorem.
    fn add_product(&mut self, forms: &[(Mask, bool)], coefficient: u8) {
        let unit = coefficient / (1 << (forms.len() - 1));
        for subset in 1_usize..1 << forms.len() {
            let mut mask = Mask::zero(self.x0.len());
            let mut constant = false;
            for (i, (form, bit)) in forms.iter().enumerate() {
                if subset & (1 << i) != 0 {
                    mask.xor(form);
                    constant ^= bit;
                }
            }
            let coefficient = if subset.count_ones() % 2 == 1 {
                unit
            } else {
                (8 - unit) % 8
            };
            self.add_affine(mask, constant, coefficient);
        }
    }

    fn diagonal(&mut self, qubits: &[QubitId], coefficient: u8) -> &mut Self {
        self.validate(qubits.iter().copied());
        for q in qubits {
            self.add_affine(self.rows[q.0].clone(), self.x0[q.0], coefficient);
        }
        self
    }

    /// Apply X to each qubit, changing only the affine offset (paper Section 3).
    pub fn x(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.validate(qubits.iter().copied());
        for q in qubits {
            self.x0[q.0] ^= true;
        }
        self
    }

    /// Apply Y as Z followed by X; the overall factor of i is discarded.
    pub fn y(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.z(qubits).x(qubits)
    }

    /// Apply Z, adding four times each physical bit to the phase.
    pub fn z(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.diagonal(qubits, 4)
    }

    /// Apply S (`diag(1,i)`), named `sz` in the PECOS gate interface.
    pub fn sz(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.diagonal(qubits, 2)
    }

    /// Apply S dagger (`diag(1,-i)`).
    pub fn szdg(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.diagonal(qubits, 6)
    }

    /// Apply T (`diag(1,exp(i*pi/4))`); paper Section 3, phase updates.
    pub fn t(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.diagonal(qubits, 1)
    }

    /// Apply T dagger.
    pub fn tdg(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.diagonal(qubits, 7)
    }

    fn cx_one(&mut self, control: usize, target: usize) {
        self.x0[target] ^= self.x0[control];
        let row = self.rows[control].clone();
        self.rows[target].xor(&row);
    }

    /// Apply controlled X for `(control, target)` pairs; paper Section 3.
    pub fn cx(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        self.validate(pairs.iter().flat_map(|&(a, b)| [a, b]));
        for &(a, b) in pairs {
            self.cx_one(a.0, b.0);
        }
        self
    }

    fn controlled_phase(&mut self, pairs: &[(QubitId, QubitId)], coefficient: u8) -> &mut Self {
        self.validate(pairs.iter().flat_map(|&(a, b)| [a, b]));
        for &(a, b) in pairs {
            self.add_product(
                &[
                    (self.rows[a.0].clone(), self.x0[a.0]),
                    (self.rows[b.0].clone(), self.x0[b.0]),
                ],
                coefficient,
            );
        }
        self
    }

    /// Apply CZ to each pair, adding `4ab` to the phase.
    pub fn cz(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        self.controlled_phase(pairs, 4)
    }

    /// Apply controlled S to each pair, adding `2ab` to the phase.
    pub fn cs(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        self.controlled_phase(pairs, 2)
    }

    /// Apply controlled S dagger to each pair, adding `-2ab` to the phase.
    pub fn csdg(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        self.controlled_phase(pairs, 6)
    }

    /// Apply CCZ on triples `(a,b,c)`. All three arguments are symmetric.
    /// Every qubit across the whole batch must be distinct.
    pub fn ccz(&mut self, triples: &[(QubitId, QubitId, QubitId)]) -> &mut Self {
        self.validate(triples.iter().flat_map(|&(a, b, c)| [a, b, c]));
        for &(a, b, c) in triples {
            let forms = [a, b, c].map(|q| (self.rows[q.0].clone(), self.x0[q.0]));
            self.add_product(&forms, 4);
        }
        self
    }

    /// Reset each qubit to zero by a sampled Z measurement and conditional X.
    pub fn pz(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.validate(qubits.iter().copied());
        for &q in qubits {
            if self.measure_z(&[q], None).outcome {
                self.x0[q.0] ^= true;
            }
        }
        self
    }

    /// Reset each qubit to plus: reset to zero, then append a free coordinate.
    pub fn px(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.pz(qubits);
        for q in qubits {
            self.rows[q.0].toggle(self.r);
            self.r += 1;
        }
        self
    }

    /// Affine pullback and support restriction. See arXiv:2610.06811,
    /// Section 3 (Z measurements) and the affine-pullback corollary.
    fn restrict(&mut self, mask: &Mask, constant: bool) {
        let pivot = mask.ones().next().expect("nonzero constraint");
        for (row, offset) in self.rows.iter_mut().zip(&mut self.x0) {
            if row.get(pivot) {
                row.xor(mask);
                *offset ^= constant;
            }
            row.remove(pivot, self.r);
        }
        for (mut term, mut coefficient) in std::mem::take(&mut self.phase) {
            if term.get(pivot) {
                term.xor(mask);
                if constant {
                    coefficient = (8 - coefficient) % 8;
                }
            }
            term.remove(pivot, self.r);
            self.add_term(term, coefficient);
        }
        self.r -= 1;
    }

    /// Weighted discrete derivative, from the derivative theorem in
    /// arXiv:2610.06811, phase-polynomials appendix.
    fn derivative(&self, translation: &Mask) -> Clifford {
        let mut derivative = Clifford::zero(self.r, self.x0.len());
        for (mask, &coefficient) in &self.phase {
            if mask.dot(translation) {
                derivative.k = (derivative.k + coefficient) % 8;
                derivative.add_affine((8 - 2 * coefficient % 8) % 8, false, mask);
            }
        }
        derivative
    }

    /// Query a basis amplitude using bits ordered by physical qubit number.
    /// Returns zero outside the affine support. The phase constant is zero.
    /// Implements arXiv:2610.06811, efficient-simulation appendix, amplitudes.
    ///
    /// # Panics
    /// Panics unless `bits.len() == num_qubits()`.
    #[must_use]
    pub fn amplitude(&self, bits: &[bool]) -> Complex64 {
        assert_eq!(
            bits.len(),
            self.x0.len(),
            "basis bit count must equal qubit count"
        );
        let rhs: Vec<_> = bits.iter().zip(&self.x0).map(|(a, b)| a ^ b).collect();
        let Some(y) = solve(&self.rows, &rhs, self.r, self.x0.len()) else {
            return Complex64::new(0.0, 0.0);
        };
        let phase = self
            .phase
            .iter()
            .filter(|(mask, _)| mask.dot(&y))
            .fold(0, |p, (_, a)| (p + a) % 8);
        GaussSum { phase, power: 0 }
            .value(i64::try_from(self.r).expect("allocated support dimension fits i64"))
    }

    /// Materialize the exponentially sized vector, for testing small systems.
    /// As in [`pecos_simulators::StateVec`], qubit q is bit q of the basis index
    /// (qubit zero is least significant). Uses the same phase as [`Self::amplitude`].
    ///
    /// # Panics
    /// Panics if the vector cannot be indexed/allocated on this platform.
    #[must_use]
    pub fn state_vector(&self) -> Vec<Complex64> {
        let n = self.x0.len();
        assert!(
            n < usize::BITS as usize,
            "state vector index overflows usize"
        );
        let mut state = vec![Complex64::new(0.0, 0.0); 1_usize << n];
        let magnitude = GaussSum { phase: 0, power: 0 }
            .value(i64::try_from(self.r).expect("support fits i64"))
            .re;
        for assignment in 0_usize..1 << self.r {
            let mut y = Mask::zero(n);
            for bit in 0..self.r {
                if assignment & (1 << bit) != 0 {
                    y.toggle(bit);
                }
            }
            let index = self
                .rows
                .iter()
                .zip(&self.x0)
                .enumerate()
                .fold(0, |x, (q, (row, offset))| {
                    x | (usize::from(row.dot(&y) ^ offset) << q)
                });
            let phase = self
                .phase
                .iter()
                .filter(|(mask, _)| mask.dot(&y))
                .fold(0, |p, (_, a)| (p + a) % 8);
            state[index] = root(phase) * magnitude;
        }
        state
    }

    /// Exact signed Hermitian Pauli expectation, converted to `f64` at the end.
    /// Uses [`PauliString`] with real overall sign and `Y = i X Z` on each site.
    /// Implements arXiv:2610.06811, efficient-simulation appendix, equation
    /// `pauli-expectation-gauss-sum`; no support enumeration is performed.
    ///
    /// # Panics
    /// Panics for an imaginary overall sign or repeated/out-of-range qubits.
    #[must_use]
    pub fn expectation(&self, pauli: &PauliString) -> f64 {
        let mut phase = match pauli.phase() {
            QuarterPhase::PlusOne => 0,
            QuarterPhase::MinusOne => 4,
            _ => panic!("Pauli expectation requires a Hermitian (real signed) string"),
        };
        self.validate(pauli.iter_pairs().map(|(_, q)| q));
        let mut translation = vec![false; self.x0.len()];
        let mut sign_mask = Mask::zero(self.x0.len());
        let mut sign_constant = false;
        for (p, q) in pauli.iter_pairs() {
            if matches!(p, Pauli::X | Pauli::Y) {
                translation[q.0] = true;
            }
            if matches!(p, Pauli::Z | Pauli::Y) {
                sign_mask.xor(&self.rows[q.0]);
                sign_constant ^= self.x0[q.0];
            }
            if p == Pauli::Y {
                phase = (phase + 2) % 8;
            }
        }
        let Some(t) = solve(&self.rows, &translation, self.r, self.x0.len()) else {
            return 0.0;
        };
        let mut quadratic = self.derivative(&t);
        quadratic.negate();
        quadratic.k = (quadratic.k + phase) % 8;
        quadratic.add_affine(4, sign_constant, &sign_mask);
        quadratic.gauss_sum().map_or(0.0, |sum| {
            // Hermiticity forces a real root before any floating-point conversion.
            assert!(
                sum.phase == 0 || sum.phase == 4,
                "Hermitian expectation must be real"
            );
            sum.value(2 * i64::try_from(self.r).expect("allocated support fits i64"))
                .re
        })
    }
}

impl QuantumSimulator for PhasePoly {
    /// Restore `|0^n>` while preserving the RNG stream and fixed qubit count.
    fn reset(&mut self) -> &mut Self {
        self.x0.fill(false);
        self.rows.fill(Mask::zero(self.x0.len()));
        self.r = 0;
        self.phase.clear();
        self
    }

    fn num_qubits(&self) -> usize {
        self.x0.len()
    }
}

impl ForcedMeasurement for PhasePoly {
    /// Force only nondeterministic Z outcomes; deterministic outcomes prevail.
    /// Panics if `qubit` is out of range.
    fn mz_forced(&mut self, qubit: usize, forced_outcome: bool) -> MeasurementResult {
        self.mz_string_forced(&[QubitId(qubit)], forced_outcome)
    }
}
