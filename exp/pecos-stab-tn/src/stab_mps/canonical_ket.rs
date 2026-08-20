// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
// except in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the
// License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
// either express or implied. See the License for the specific language governing permissions and
// limitations under the License.

//! Polynomial-time relative amplitudes of the canonical ket represented by a stabilizer tableau.
//!
//! `StabMps::state_vector` fixes a tableau's otherwise arbitrary ket phase by
//! normalizing the first nonzero column of `prod_k (I + S_k) / 2`. Equivalently,
//! the numerically first computational-basis word in the support has a positive
//! real amplitude. This module implements that same convention without building
//! a dense projector. The resulting amplitudes are exact relative to that
//! convention; the represented physical state, like every ket, still has one
//! arbitrary global phase.

use num_complex::Complex64;
use pecos_simulators::{Gens, SparseStabY};

use super::tableau_compose::multiply_row_within;

/// An exact fourth root of unity, represented by its exponent in `i^exponent`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct QuarterPhase(u8);

impl QuarterPhase {
    const ONE: Self = Self(0);

    fn times_i_pow(&mut self, exponent: usize) {
        self.0 = (self.0 + (exponent & 3) as u8) & 3;
    }

    fn to_complex(self) -> Complex64 {
        match self.0 {
            0 => Complex64::new(1.0, 0.0),
            1 => Complex64::new(0.0, 1.0),
            2 => Complex64::new(-1.0, 0.0),
            3 => Complex64::new(0.0, -1.0),
            _ => unreachable!("quarter-phase exponent is reduced modulo four"),
        }
    }
}

/// Row-reduced description of the canonical representative selected by a tableau.
#[derive(Clone)]
pub(super) struct CanonicalKet {
    num_qubits: usize,
    reduced_stabs: Gens,
    /// `(pivot_qubit, generator_row)` in increasing qubit order.
    x_pivots: Vec<(usize, usize)>,
    first_support: Vec<bool>,
    support_amplitude_magnitude: f64,
}

impl CanonicalKet {
    /// Build the canonical ket in O(n^3) bit operations.
    pub(super) fn new(tableau: &SparseStabY) -> Self {
        let num_qubits = tableau.num_qubits();
        let mut reduced_stabs = tableau.stabs().clone();
        let mut pivot_rows = vec![false; num_qubits];
        let mut x_pivots = Vec::new();

        // Reduced row echelon form of the stabilizer X block. Stabilizer rows
        // commute, so signed generator multiplication is a valid GF(2) row op.
        for qubit in 0..num_qubits {
            let Some(pivot_row) = (0..num_qubits)
                .find(|&row| !pivot_rows[row] && reduced_stabs.row_x[row].contains(qubit))
            else {
                continue;
            };
            for row in 0..num_qubits {
                if row != pivot_row && reduced_stabs.row_x[row].contains(qubit) {
                    multiply_row_within(&mut reduced_stabs, row, pivot_row, num_qubits);
                }
            }
            pivot_rows[pivot_row] = true;
            x_pivots.push((qubit, pivot_row));
        }

        // Rows outside the X-pivot basis are diagonal stabilizers. Their +1
        // eigenvalue equations determine an affine computational-basis coset.
        let mut equations = Vec::with_capacity(num_qubits - x_pivots.len());
        for (row, &is_pivot) in pivot_rows.iter().enumerate() {
            if is_pivot {
                continue;
            }
            assert!(
                reduced_stabs.row_x[row].is_empty(),
                "X reduction left a non-pivot stabilizer row with X support"
            );
            let sign_exponent = usize::from(reduced_stabs.signs_i.contains(row))
                + 2 * usize::from(reduced_stabs.signs_minus.contains(row));
            assert_eq!(
                sign_exponent & 1,
                0,
                "a diagonal stabilizer generator cannot have an imaginary eigenvalue"
            );
            let mut equation = vec![false; num_qubits + 1];
            for qubit in &reduced_stabs.row_z[row] {
                equation[qubit] = true;
            }
            equation[num_qubits] = sign_exponent == 2;
            equations.push(equation);
        }

        let mut first_support = solve_affine_system(equations, num_qubits)
            .expect("a valid stabilizer tableau must have nonempty computational support");

        // The solution set differs by the row space of the X block. Clearing
        // its pivots from q0 upward gives the smallest MSB-first projector
        // column, exactly matching state_vector's scan order.
        for &(pivot_qubit, row) in &x_pivots {
            if first_support[pivot_qubit] {
                for qubit in &reduced_stabs.row_x[row] {
                    first_support[qubit] ^= true;
                }
            }
        }

        let support_amplitude_magnitude = 2.0_f64.powf(-(x_pivots.len() as f64) / 2.0);
        Self {
            num_qubits,
            reduced_stabs,
            x_pivots,
            first_support,
            support_amplitude_magnitude,
        }
    }

    fn first_support(&self) -> &[bool] {
        &self.first_support
    }

    /// Return one amplitude in `state_vector`'s canonical representative.
    ///
    /// Relative phases within the ket are exact; the physical state remains
    /// defined only up to a common global phase.
    pub(super) fn amplitude(&self, bitstring: &[bool]) -> Complex64 {
        assert_eq!(
            bitstring.len(),
            self.num_qubits,
            "bitstring length mismatch"
        );
        let mut remaining = bitstring
            .iter()
            .zip(&self.first_support)
            .map(|(&target, &first)| target ^ first)
            .collect::<Vec<_>>();
        let mut current = self.first_support.clone();
        let mut phase = QuarterPhase::ONE;

        for &(pivot_qubit, row) in &self.x_pivots {
            if !remaining[pivot_qubit] {
                continue;
            }
            phase.times_i_pow(pauli_row_action_exponent(
                &self.reduced_stabs,
                row,
                &current,
            ));
            for qubit in &self.reduced_stabs.row_x[row] {
                current[qubit] ^= true;
                remaining[qubit] ^= true;
            }
        }

        if remaining.into_iter().any(|bit| bit) {
            return Complex64::new(0.0, 0.0);
        }
        debug_assert_eq!(current, bitstring);
        phase.to_complex() * self.support_amplitude_magnitude
    }
}

/// Return `<target|P|ket>` for one signed Y-convention Pauli generator row.
fn pauli_applied_amplitude_at(
    gens: &Gens,
    row: usize,
    ket: &CanonicalKet,
    target: &[bool],
) -> Complex64 {
    assert_eq!(target.len(), ket.num_qubits, "bitstring length mismatch");
    let mut source = target.to_vec();
    for qubit in &gens.row_x[row] {
        source[qubit] ^= true;
    }
    let source_amplitude = ket.amplitude(&source);
    if source_amplitude == Complex64::new(0.0, 0.0) {
        return source_amplitude;
    }
    let phase = QuarterPhase((pauli_row_action_exponent(gens, row, &source) & 3) as u8);
    phase.to_complex() * source_amplitude
}

/// Return `<target|D_0^x0 ... D_{n-1}^xn|phi>` in the same effective
/// multiplication order used by `StabMps::state_vector`.
#[cfg(test)]
fn destabilizer_basis_amplitude(
    tableau: &SparseStabY,
    coefficient_bits: &[u8],
    target: &[bool],
) -> Complex64 {
    let ket = CanonicalKet::new(tableau);
    destabilizer_basis_amplitude_with_ket(tableau, coefficient_bits, target, &ket)
}

fn destabilizer_basis_amplitude_with_ket(
    tableau: &SparseStabY,
    coefficient_bits: &[u8],
    target: &[bool],
    ket: &CanonicalKet,
) -> Complex64 {
    let num_qubits = tableau.num_qubits();
    assert_eq!(
        coefficient_bits.len(),
        num_qubits,
        "coefficient bitstring length mismatch"
    );
    assert_eq!(target.len(), num_qubits, "target bitstring length mismatch");

    // Determine the source basis word which the selected destabilizer product
    // maps to target. X support composes by XOR independently of row order.
    let mut source = target.to_vec();
    for (row, &selected) in coefficient_bits.iter().enumerate() {
        assert!(selected <= 1, "coefficient bit must be zero or one");
        if selected == 1 {
            for qubit in &tableau.destabs().row_x[row] {
                source[qubit] ^= true;
            }
        }
    }

    let source_amplitude = ket.amplitude(&source);
    if source_amplitude == Complex64::new(0.0, 0.0) {
        return source_amplitude;
    }

    // state_vector applies row 0 first, then row 1, etc. Track the phase of
    // that same left-multiplication sequence on the source computational ket.
    let mut current = source;
    let mut phase = QuarterPhase::ONE;
    for (row, &selected) in coefficient_bits.iter().enumerate() {
        if selected == 0 {
            continue;
        }
        phase.times_i_pow(pauli_row_action_exponent(tableau.destabs(), row, &current));
        for qubit in &tableau.destabs().row_x[row] {
            current[qubit] ^= true;
        }
    }
    debug_assert_eq!(current, target);
    phase.to_complex() * source_amplitude
}

fn normalized_terminal_phase(
    tableau: &SparseStabY,
    coefficient_bits: &[u8],
    target: &[bool],
    ket: &CanonicalKet,
) -> Complex64 {
    let amplitude = destabilizer_basis_amplitude_with_ket(tableau, coefficient_bits, target, ket);
    let magnitude = amplitude.norm();
    assert!(
        magnitude.is_finite() && magnitude > 0.0,
        "projected tableau basis vector must have nonzero target amplitude"
    );
    amplitude / magnitude
}

fn right_compose_h_scalar(
    before: &SparseStabY,
    before_ket: &CanonicalKet,
    after_ket: &CanonicalKet,
    q: usize,
) -> Complex64 {
    let target = after_ket.first_support();
    let denominator = after_ket.amplitude(target);
    let numerator = (before_ket.amplitude(target)
        + pauli_applied_amplitude_at(before.destabs(), q, before_ket, target))
        * std::f64::consts::FRAC_1_SQRT_2;
    normalized_scalar(
        numerator,
        denominator,
        "right-composed H canonical-ket scalar",
    )
}

fn right_compose_x_scalar(
    before: &SparseStabY,
    before_ket: &CanonicalKet,
    after_ket: &CanonicalKet,
    q: usize,
) -> Complex64 {
    let target = after_ket.first_support();
    normalized_scalar(
        pauli_applied_amplitude_at(before.destabs(), q, before_ket, target),
        after_ket.amplitude(target),
        "right-composed X canonical-ket scalar",
    )
}

fn forced_measurement_scalar(
    before_ket: &CanonicalKet,
    after_ket: &CanonicalKet,
    q: usize,
    outcome: bool,
    probability: f64,
) -> Complex64 {
    assert!(
        probability > 0.0,
        "cannot phase-track a zero-probability branch"
    );
    let target = after_ket.first_support();
    assert_eq!(
        target[q], outcome,
        "projected canonical ket must have the forced measurement outcome"
    );
    normalized_scalar(
        before_ket.amplitude(target) / probability.sqrt(),
        after_ket.amplitude(target),
        "forced stabilizer measurement canonical-ket scalar",
    )
}

/// Scalar accumulator for the phase-sensitive forced-projection walk.
///
/// The cached ket is valid across generator rebasing and right-composed
/// CX/Z/CZ/SZ/SZ-dagger operations: those operations preserve the stabilizer
/// group because their virtual gates fix `|0...0>`. Every operation that does
/// change that group (the tracked X/H rotations and `mz_forced`) refreshes the
/// cache before it can be reused. This avoids rebuilding the incoming
/// O(n^3) canonical ket at each consecutive scalar site.
#[derive(Clone)]
pub(super) struct CanonicalPhaseTracker {
    scalar: Complex64,
    ket: Option<CanonicalKet>,
}

impl CanonicalPhaseTracker {
    pub(super) fn new() -> Self {
        Self {
            scalar: Complex64::new(1.0, 0.0),
            ket: None,
        }
    }

    pub(super) fn scalar(&self) -> Complex64 {
        self.scalar
    }

    fn take_before_ket(&mut self, before: &SparseStabY) -> CanonicalKet {
        let ket = self.ket.take().unwrap_or_else(|| CanonicalKet::new(before));
        // Self-enforce the reuse invariant: the cache is only valid across
        // operations that preserve the stabilizer group, so the cached ket's
        // support must match one rebuilt from the current tableau. A mismatch
        // means a group-changing operation was inserted between scalar sites
        // without refreshing the cache -- the silent-phase failure class of
        // issue #562.
        debug_assert_eq!(
            ket.first_support(),
            CanonicalKet::new(before).first_support(),
            "cached canonical ket is stale: a group-changing tableau operation \
             ran without refreshing the phase tracker"
        );
        ket
    }

    /// Track the unit-modulus scalar introduced by a right-composed H.
    pub(super) fn right_compose_h(&mut self, before: &SparseStabY, after: &SparseStabY, q: usize) {
        let before_ket = self.take_before_ket(before);
        let after_ket = CanonicalKet::new(after);
        self.scalar *= right_compose_h_scalar(before, &before_ket, &after_ket, q);
        self.ket = Some(after_ket);
    }

    /// Track the unit-modulus scalar introduced by a right-composed X.
    pub(super) fn right_compose_x(&mut self, before: &SparseStabY, after: &SparseStabY, q: usize) {
        let before_ket = self.take_before_ket(before);
        let after_ket = CanonicalKet::new(after);
        self.scalar *= right_compose_x_scalar(before, &before_ket, &after_ket, q);
        self.ket = Some(after_ket);
    }

    /// Track the unit-modulus scalar introduced by pure-stabilizer projection.
    pub(super) fn forced_measurement(
        &mut self,
        before: &SparseStabY,
        after: &SparseStabY,
        q: usize,
        outcome: bool,
        probability: f64,
    ) {
        let before_ket = self.take_before_ket(before);
        let after_ket = CanonicalKet::new(after);
        self.scalar *= forced_measurement_scalar(&before_ket, &after_ket, q, outcome, probability);
        self.ket = Some(after_ket);
    }

    /// Unit phase of the selected terminal tableau basis vector, relative to
    /// `state_vector`'s canonical representative.
    pub(super) fn terminal_tableau_basis_phase(
        &self,
        tableau: &SparseStabY,
        coefficient_bits: &[u8],
        target: &[bool],
    ) -> Complex64 {
        if let Some(ket) = &self.ket {
            normalized_terminal_phase(tableau, coefficient_bits, target, ket)
        } else {
            let ket = CanonicalKet::new(tableau);
            normalized_terminal_phase(tableau, coefficient_bits, target, &ket)
        }
    }
}

fn normalized_scalar(numerator: Complex64, denominator: Complex64, context: &str) -> Complex64 {
    assert!(
        denominator.norm_sqr() > 0.0 && numerator.norm_sqr() > 0.0,
        "{context} must compare nonzero amplitudes"
    );
    let scalar = numerator / denominator;
    let magnitude = scalar.norm();
    assert!(
        magnitude.is_finite() && (magnitude - 1.0).abs() <= 1e-10,
        "{context} must be unit magnitude, got {scalar:?}"
    );
    scalar / magnitude
}

/// Exponent `e` such that a signed row maps `|bits>` to `i^e |bits xor x>`.
fn pauli_row_action_exponent(gens: &Gens, row: usize, bits: &[bool]) -> usize {
    let mut exponent =
        usize::from(gens.signs_i.contains(row)) + 2 * usize::from(gens.signs_minus.contains(row));
    for (qubit, &bit) in bits.iter().enumerate() {
        let x = gens.row_x[row].contains(qubit);
        let z = gens.row_z[row].contains(qubit);
        exponent += usize::from(x && z); // Y = iXZ.
        exponent += 2 * usize::from(z && bit);
    }
    exponent & 3
}

/// Solve `A x = b` over GF(2), choosing zero for every free variable.
fn solve_affine_system(mut equations: Vec<Vec<bool>>, num_variables: usize) -> Option<Vec<bool>> {
    let mut pivot_row = 0usize;
    let mut pivots = Vec::new();
    for column in 0..num_variables {
        let Some(found) = (pivot_row..equations.len()).find(|&row| equations[row][column]) else {
            continue;
        };
        equations.swap(pivot_row, found);
        let pivot = equations[pivot_row].clone();
        for (row, equation) in equations.iter_mut().enumerate() {
            if row == pivot_row || !equation[column] {
                continue;
            }
            for (entry, &pivot_entry) in equation[column..].iter_mut().zip(&pivot[column..]) {
                *entry ^= pivot_entry;
            }
        }
        pivots.push((column, pivot_row));
        pivot_row += 1;
    }

    if equations.iter().any(|equation| {
        !equation[..num_variables].iter().any(|&value| value) && equation[num_variables]
    }) {
        return None;
    }

    let mut solution = vec![false; num_variables];
    for (column, row) in pivots {
        solution[column] = equations[row][num_variables];
    }
    Some(solution)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stab_mps::StabMps;
    use nalgebra::DMatrix;
    use pecos_core::QubitId;
    use pecos_simulators::CliffordGateable;

    #[test]
    fn known_y_state_uses_positive_first_amplitude() {
        let mut tableau = SparseStabY::new(1).with_destab_sign_tracking();
        tableau.h(&[QubitId(0)]).sz(&[QubitId(0)]);
        let ket = CanonicalKet::new(&tableau);
        assert_eq!(ket.first_support(), &[false]);
        assert_eq!(
            ket.amplitude(&[false]),
            Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0)
        );
        assert_eq!(
            ket.amplitude(&[true]),
            Complex64::new(0.0, std::f64::consts::FRAC_1_SQRT_2)
        );
    }

    #[test]
    fn pauli_action_includes_both_stored_sign_bits() {
        for (minus, i_bit, expected) in [
            (false, false, Complex64::new(1.0, 0.0)),
            (true, false, Complex64::new(-1.0, 0.0)),
            (false, true, Complex64::new(0.0, 1.0)),
            (true, true, Complex64::new(0.0, -1.0)),
        ] {
            let mut stn = StabMps::new(1);
            if minus {
                stn.tableau.destabs_mut().signs_minus.insert(0);
            }
            if i_bit {
                stn.tableau.destabs_mut().signs_i.insert(0);
            }
            let x = DMatrix::from_row_slice(
                2,
                2,
                &[
                    Complex64::new(0.0, 0.0),
                    Complex64::new(1.0, 0.0),
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                ],
            );
            stn.mps.apply_one_site_gate(0, &x).unwrap();
            let dense = stn.state_vector();
            assert_eq!(
                destabilizer_basis_amplitude(&stn.tableau, &[1], &[true]),
                expected
            );
            assert_eq!(dense[1], expected);
        }
    }

    #[test]
    fn destabilizer_product_uses_state_vector_row_order() {
        let mut tableau = SparseStabY::new(2).with_destab_sign_tracking();
        tableau.h(&[QubitId(0)]);
        tableau.sz(&[QubitId(0)]);
        tableau.cx(&[(QubitId(0), QubitId(1))]);
        let bits = [1, 1];
        let mut stn = StabMps::new(2);
        stn.tableau = tableau;
        let x = DMatrix::from_row_slice(
            2,
            2,
            &[
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        );
        stn.mps.apply_one_site_gate(0, &x).unwrap();
        stn.mps.apply_one_site_gate(1, &x).unwrap();
        let dense = stn.state_vector();
        for (index, expected) in dense.into_iter().enumerate() {
            let target = [index & 1 != 0, index & 2 != 0];
            assert_eq!(
                destabilizer_basis_amplitude(&stn.tableau, &bits, &target),
                expected
            );
        }
    }
}
