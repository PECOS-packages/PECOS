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

//! Phase-complete changes of stabilizer coordinates, without amplitudes.
//!
//! Active bit `b` names generator pair `active[b]`. All other coordinates are
//! fixed to the +1 eigenspace of their stabilizers. Products use Hermitian
//! `Y = iXZ` and the order `P = phase * product(D) * product(S)`.

use num_complex::Complex64;
use pecos_core::IndexSet;
use pecos_simulators::{GensGeneric, SparseStabY};

use super::pauli_decomp::{PauliKindForDecomp, decompose_pauli_string};
use super::tableau_compose::{
    multiply_row, multiply_row_within, right_compose_cx, right_compose_h, right_compose_sz,
    right_compose_x,
};

/// A Pauli expressed in ordered active bits and dormant generator indices.
#[derive(Clone, Debug)]
pub struct CoordinateDecomposition {
    /// Active bit positions flipped by the Pauli.
    pub active_flips: Vec<usize>,
    /// Active bit positions contributing input parity.
    pub active_signs: Vec<usize>,
    /// Dormant generator indices flipped by the Pauli.
    pub dormant_flips: Vec<usize>,
    /// Dormant stabilizer factors, retained for basis changes.
    pub dormant_signs: Vec<usize>,
    /// Scalar in `P = phase * product(D) * product(S)`.
    pub phase: Complex64,
}

/// Structural measurement case; an active case can still have certain outcome.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MeasurementCase {
    /// A dormant flip gives two equiprobable outcomes.
    Random,
    /// Only dormant stabilizers remain, so the phase fixes the outcome.
    Deterministic,
    /// The amplitudes determine the outcome; one active bit can be removed.
    Active,
}

impl CoordinateDecomposition {
    /// Classify by support, independently of amplitudes.
    #[must_use]
    pub fn measurement_case(&self) -> MeasurementCase {
        if !self.dormant_flips.is_empty() {
            MeasurementCase::Random
        } else if self.active_flips.is_empty() && self.active_signs.is_empty() {
            MeasurementCase::Deterministic
        } else {
            MeasurementCase::Active
        }
    }
}

fn validate(tableau: &SparseStabY, active: &[usize], pauli: &[(usize, PauliKindForDecomp)]) {
    assert!(
        tableau.tracks_destab_signs(),
        "destabilizer sign tracking is required"
    );
    for (bit, &index) in active.iter().enumerate() {
        assert!(
            index < tableau.num_qubits() && !active[..bit].contains(&index),
            "invalid active indices"
        );
    }
    for (i, &(qubit, _)) in pauli.iter().enumerate() {
        assert!(
            qubit < tableau.num_qubits() && !pauli[..i].iter().any(|&(q, _)| q == qubit),
            "Pauli factors must name distinct valid qubits"
        );
    }
}

/// Decompose a signed Hermitian physical Pauli into active and dormant parts.
///
/// `negative` specifies an overall minus sign. Active entries are bit positions,
/// not generator indices. Dormant sign factors act as +1 on the active subspace.
///
/// # Panics
/// Panics for repeated/out-of-range active indices or physical qubits, or when
/// destabilizer sign tracking is disabled.
#[must_use]
pub fn decompose(
    tableau: &SparseStabY,
    active: &[usize],
    pauli: &[(usize, PauliKindForDecomp)],
    negative: bool,
) -> CoordinateDecomposition {
    validate(tableau, active, pauli);
    let (flips, signs, phase) = decompose_pauli_string(tableau.stabs(), tableau.destabs(), pauli);
    let split = |indices: Vec<usize>| {
        let mut bits = Vec::new();
        let mut dormant = Vec::new();
        for index in indices {
            if let Some(bit) = active.iter().position(|&entry| entry == index) {
                bits.push(bit);
            } else {
                dormant.push(index);
            }
        }
        (bits, dormant)
    };
    let (active_flips, dormant_flips) = split(flips);
    let (active_signs, dormant_signs) = split(signs);
    CoordinateDecomposition {
        active_flips,
        active_signs,
        dormant_flips,
        dormant_signs,
        phase: if negative { -phase } else { phase },
    }
}

fn replace_row<S: IndexSet>(
    gens: &mut GensGeneric<S>,
    row: usize,
    pauli: &[(usize, PauliKindForDecomp)],
    negative: bool,
) {
    for q in gens.row_x[row].iter() {
        gens.col_x[q].remove(row);
    }
    for q in gens.row_z[row].iter() {
        gens.col_z[q].remove(row);
    }
    gens.row_x[row].clear();
    gens.row_z[row].clear();
    gens.signs_minus.remove(row);
    gens.signs_i.remove(row);
    if negative {
        gens.signs_minus.insert(row);
    }
    for &(q, kind) in pauli {
        if kind != PauliKindForDecomp::Z {
            gens.row_x[row].insert(q);
            gens.col_x[q].insert(row);
        }
        if kind != PauliKindForDecomp::X {
            gens.row_z[row].insert(q);
            gens.col_z[q].insert(row);
        }
    }
}

// The dormant S_h fixes every old active basis vector. Multiplying any
// non-pivot generator by S_h therefore preserves those basis vectors, with
// the exact row-product phase, while removing its anticommutation with P.
fn localize_dormant(
    tableau: &mut SparseStabY,
    active: &[usize],
    pauli: &[(usize, PauliKindForDecomp)],
    negative: bool,
) -> usize {
    let parts = decompose(tableau, active, pauli, negative);
    let pivot = *parts
        .dormant_flips
        .first()
        .expect("Pauli must have a dormant flip");
    let n = tableau.num_qubits();
    for index in parts
        .active_flips
        .iter()
        .map(|&b| active[b])
        .chain(parts.dormant_flips)
    {
        if index != pivot {
            multiply_row_within(tableau.stabs_mut(), index, pivot, n);
        }
    }
    let (stabs, destabs) = tableau.stabs_and_destabs_mut();
    for index in parts
        .active_signs
        .iter()
        .map(|&b| active[b])
        .chain(parts.dormant_signs)
    {
        if index != pivot {
            multiply_row(destabs, index, stabs, pivot, n);
        }
    }
    replace_row(destabs, pivot, pauli, negative);
    pivot
}

/// Make the entire signed Pauli the destabilizer of a dormant flip and append
/// that coordinate to `active`. Existing basis states have the new bit zero.
///
/// # Panics
/// Panics on invalid inputs to [`decompose`] or if there is no dormant flip.
pub fn promote(
    tableau: &mut SparseStabY,
    active: &mut Vec<usize>,
    pauli: &[(usize, PauliKindForDecomp)],
    negative: bool,
) -> usize {
    let pivot = localize_dormant(tableau, active, pauli, negative);
    active.push(pivot);
    pivot
}

/// Install `(-1)^outcome P` as a dormant stabilizer, preserving all active
/// coefficients of the normalized projected state. The outcome is a fair coin.
///
/// # Panics
/// Panics on invalid inputs to [`decompose`] or if there is no dormant flip.
pub fn measure_random(
    tableau: &mut SparseStabY,
    active: &[usize],
    pauli: &[(usize, PauliKindForDecomp)],
    negative: bool,
    outcome: bool,
) {
    let pivot = localize_dormant(tableau, active, pauli, negative);
    right_compose_h(tableau, pivot);
    if outcome {
        right_compose_x(tableau, pivot);
    }
}

/// An inverse coordinate operation to apply to amplitudes after right composition.
/// Arguments are active bit positions, never physical qubit indices.
#[derive(Clone, Copy, Debug)]
pub enum CoordinateGate {
    /// Apply H to this active bit.
    H(usize),
    /// Apply S dagger to this active bit (the tableau was right-composed by S).
    Sdg(usize),
    /// Apply CX from the first active bit to the second.
    Cx(usize, usize),
}

/// A coordinate change reducing a measured Pauli to `(-1)^negative S_pivot`.
#[derive(Debug)]
pub struct MeasurementBasis {
    /// Inverse changes to apply to amplitudes in the returned order.
    pub gates: Vec<CoordinateGate>,
    /// Active bit to project and subsequently remove.
    pub pivot_bit: usize,
    /// Sign of the measured Pauli relative to the new pivot stabilizer.
    pub negative: bool,
}

/// Right-compose coordinate Cliffords to reduce an active Pauli to one
/// stabilizer, including its dormant sign factors. Apply the returned inverse
/// gates to amplitudes before projecting the pivot bit.
///
/// # Panics
/// Panics on invalid inputs to [`decompose`] or unless the case is active.
pub fn measurement_basis(
    tableau: &mut SparseStabY,
    active: &[usize],
    pauli: &[(usize, PauliKindForDecomp)],
    negative: bool,
) -> MeasurementBasis {
    let parts = decompose(tableau, active, pauli, negative);
    assert_eq!(parts.measurement_case(), MeasurementCase::Active);
    let mut gates = Vec::new();
    for &bit in &parts.active_flips {
        let index = active[bit];
        if parts.active_signs.contains(&bit) {
            right_compose_sz(tableau, index);
            gates.push(CoordinateGate::Sdg(bit));
        }
        right_compose_h(tableau, index);
        gates.push(CoordinateGate::H(bit));
    }
    let diagonal = decompose(tableau, active, pauli, negative);
    let pivot_bit = diagonal.active_signs[0];
    let pivot = active[pivot_bit];
    for &bit in &diagonal.active_signs[1..] {
        right_compose_cx(tableau, active[bit], pivot);
        gates.push(CoordinateGate::Cx(bit, pivot_bit));
    }
    for index in diagonal.dormant_signs {
        // A dormant control is zero; its inverse CX leaves amplitudes unchanged.
        right_compose_cx(tableau, index, pivot);
    }
    let final_parts = decompose(tableau, active, pauli, negative);
    MeasurementBasis {
        gates,
        pivot_bit,
        negative: final_parts.phase.re < 0.0,
    }
}

/// Make the selected coordinate value its new zero state, then remove that bit
/// from `active`. The caller must project and compact its amplitudes accordingly.
///
/// # Panics
/// Panics if `pivot_bit` is not active or the tableau/active indices are invalid.
pub fn demote(tableau: &mut SparseStabY, active: &mut Vec<usize>, pivot_bit: usize, value: bool) {
    validate(tableau, active, &[]);
    let pivot = active[pivot_bit];
    if value {
        right_compose_x(tableau, pivot);
    }
    active.remove(pivot_bit);
}

#[cfg(test)]
mod tests;
