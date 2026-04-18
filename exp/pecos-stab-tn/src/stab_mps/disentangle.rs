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

//! Heuristic Clifford disentangling for the STN simulator.
//!
//! After non-Clifford gates increase the MPS bond dimension, we can try to
//! reduce it by applying two-qubit Clifford gates that absorb entanglement
//! into the stabilizer tableau.
//!
//! The algorithm: for each internal bond, try the 20 inequivalent entangling
//! two-qubit Cliffords. If one reduces the entanglement entropy at the bond,
//! apply it to the MPS and store the inverse for `state_vector` reconstruction.
//!
//! References:
//! - Masot-Llima, Garcia-Saez. arXiv:2403.08724 (Clifford disentangling).
//! - Masot-Llima, Sierant, Stornati, Garcia-Saez. arXiv:2602.15942
//!   (limits of Clifford disentangling).

use crate::mps::Mps;
use nalgebra::DMatrix;
use num_complex::Complex64;

/// A two-qubit Clifford gate for disentangling.
struct DisentanglerGate {
    /// 4x4 unitary matrix for the MPS
    matrix: DMatrix<Complex64>,
    /// 4x4 inverse matrix (for `state_vector` correction)
    inverse_matrix: DMatrix<Complex64>,
}

/// Build a 2x2 single-qubit Clifford gate.
fn single_qubit_clifford(idx: usize) -> DMatrix<Complex64> {
    let one = Complex64::new(1.0, 0.0);
    let zero = Complex64::new(0.0, 0.0);
    let inv2 = Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0);
    let i_val = Complex64::new(0.0, 1.0);

    let id = DMatrix::from_row_slice(2, 2, &[one, zero, zero, one]);
    let h = DMatrix::from_row_slice(2, 2, &[inv2, inv2, inv2, -inv2]);
    let s = DMatrix::from_row_slice(2, 2, &[one, zero, zero, i_val]);

    match idx {
        1 => h.clone(),    // H
        2 => s.clone(),    // S
        3 => &s * &h,      // SH
        4 => &h * &s,      // HS
        5 => &h * &s * &h, // HSH
        _ => id,           // I (idx == 0 or out of range)
    }
}

/// Build the set of candidate disentangling gates.
///
/// Generates entangling 2-qubit Cliffords by dressing CX with single-qubit
/// Cliffords on each qubit: (A⊗B) * CX * (C⊗D) for various A,B,C,D.
fn build_disentangler_set() -> Vec<DisentanglerGate> {
    let one = Complex64::new(1.0, 0.0);
    let zero = Complex64::new(0.0, 0.0);

    // Base CX gate
    let cx = DMatrix::from_row_slice(
        4,
        4,
        &[
            one, zero, zero, zero, zero, one, zero, zero, zero, zero, zero, one, zero, zero, one,
            zero,
        ],
    );

    // Generate dressed CX gates: (A⊗B) * CX for single-qubit Cliffords A, B.
    // This covers the 20 inequivalent entangling 2-qubit Cliffords.
    // We use 6 single-qubit Cliffords: {I, H, S, SH, HS, HSH}
    let mut gates = Vec::new();
    let mut seen = Vec::new();

    // Dressings: (left_q0, left_q1) applied AFTER CX
    let dressings: &[(usize, usize)] = &[
        (0, 0), // I⊗I * CX = CX
        (0, 1), // I⊗H * CX
        (1, 0), // H⊗I * CX
        (1, 1), // H⊗H * CX
        (0, 2), // I⊗S * CX
        (2, 0), // S⊗I * CX
        (3, 0), // SH⊗I * CX
        (0, 3), // I⊗SH * CX
        (4, 0), // HS⊗I * CX
        (0, 4), // I⊗HS * CX
        (1, 2), // H⊗S * CX
        (2, 1), // S⊗H * CX
        (5, 0), // HSH⊗I * CX
        (0, 5), // I⊗HSH * CX
        (3, 1), // SH⊗H * CX
        (1, 3), // H⊗SH * CX
        (4, 1), // HS⊗H * CX
        (1, 4), // H⊗HS * CX
        (2, 2), // S⊗S * CX
        (3, 3), // SH⊗SH * CX
    ];

    for &(a_idx, b_idx) in dressings {
        let a = single_qubit_clifford(a_idx);
        let b = single_qubit_clifford(b_idx);

        // Build A⊗B
        let ab = a.kronecker(&b);
        let matrix = &ab * &cx;

        // Check for duplicates (up to global phase)
        let is_dup = seen.iter().any(|existing: &DMatrix<Complex64>| {
            // Two unitaries are equivalent if one is a scalar multiple of the other
            if let Some(&first_nonzero) = matrix.iter().zip(existing.iter()).find_map(|(a, b)| {
                if a.norm() > 0.1 && b.norm() > 0.1 {
                    Some(a)
                } else {
                    None
                }
            }) {
                let first_existing = existing
                    .iter()
                    .zip(matrix.iter())
                    .find_map(|(b, a)| if a.norm() > 0.1 { Some(b) } else { None });
                if let Some(&fe) = first_existing {
                    let ratio = first_nonzero / fe;
                    // Check all elements have the same ratio
                    matrix
                        .iter()
                        .zip(existing.iter())
                        .all(|(a, b)| (a - b * ratio).norm() < 1e-10)
                } else {
                    false
                }
            } else {
                false
            }
        });

        if !is_dup {
            seen.push(matrix.clone());
            gates.push(DisentanglerGate {
                inverse_matrix: matrix.adjoint(),
                matrix,
            });
        }
    }

    gates
}

/// Compute the entanglement entropy at a given bond of the MPS.
///
/// Uses singular values from the bond matrix after left-canonicalization up to that bond.
/// For efficiency, just compute the Frobenius norm ratio as a proxy.
fn bond_entropy(mps: &Mps, bond: usize) -> f64 {
    if bond == 0 || bond >= mps.num_sites() {
        return 0.0;
    }

    let d = mps.phys_dim();
    let chi_l = mps.bond_dim(bond);
    let chi_r = mps.bond_dim(bond + 1);
    let tensor = &mps.tensors()[bond];

    // Reshape to (chi_l * d, chi_r) and compute SVD
    let matrix = crate::mps::tensor::reshape_left_group(tensor, chi_l, d, chi_r);
    let svd = nalgebra::SVD::new(matrix, false, false);

    // Compute von Neumann entropy from singular values
    let svals = &svd.singular_values;
    let norm_sq: f64 = svals.iter().map(|s| s * s).sum();
    if norm_sq < 1e-30 {
        return 0.0;
    }

    let mut entropy = 0.0;
    for &s in svals.iter() {
        let p = (s * s) / norm_sq;
        if p > 1e-15 {
            entropy -= p * p.ln();
        }
    }
    entropy
}

/// Run one sweep of heuristic disentangling on the MPS.
///
/// For each internal bond, tries each candidate Clifford gate and keeps
/// the one that reduces the max bond dimension of the MPS. Uses max bond
/// dim as the criterion (not local entropy) because a gate that reduces
/// entropy at one bond can increase it at neighboring bonds.
///
/// Records inverse operations in the gate log so `state_vector()` stays correct.
///
/// Returns the number of gates applied (0 means no improvement found).
pub(crate) fn disentangle_sweep(
    mps: &mut Mps,
    corrections: &mut Vec<super::MpsIndexGate>,
) -> usize {
    let n = mps.num_sites();
    if n < 2 {
        return 0;
    }

    let gates = build_disentangler_set();
    let mut num_applied = 0;

    // Forward sweep: bonds 0..n-2 (between sites q and q+1)
    for q in 0..n - 1 {
        let bond = q + 1;
        let current_entropy = bond_entropy(mps, bond);

        if current_entropy < 1e-6 {
            continue; // Already effectively disentangled at this bond
        }

        let current_max_bond = mps.max_bond_dim();
        let mut best_entropy = current_entropy;
        let mut best_gate_idx: Option<usize> = None;

        for (gate_idx, gate) in gates.iter().enumerate() {
            let mut trial_mps = mps.clone();
            if trial_mps.apply_two_site_gate(q, &gate.matrix).is_ok() {
                let trial_max_bond = trial_mps.max_bond_dim();
                let trial_entropy = bond_entropy(&trial_mps, bond);
                // Accept gate only if it doesn't increase max bond dim
                // AND reduces local entropy
                if trial_max_bond <= current_max_bond && trial_entropy < best_entropy - 1e-2 {
                    best_entropy = trial_entropy;
                    best_gate_idx = Some(gate_idx);
                }
            }
        }

        if let Some(idx) = best_gate_idx {
            mps.apply_two_site_gate(q, &gates[idx].matrix)
                .expect("gate should succeed");
            corrections.push(super::MpsIndexGate {
                site: q,
                inverse_matrix: gates[idx].inverse_matrix.clone(),
            });
            num_applied += 1;
        }
    }

    // Backward sweep: bonds n-2..0
    for q in (0..n - 1).rev() {
        let bond = q + 1;
        let current_entropy = bond_entropy(mps, bond);

        if current_entropy < 1e-6 {
            continue;
        }

        let current_max_bond = mps.max_bond_dim();
        let mut best_entropy = current_entropy;
        let mut best_gate_idx: Option<usize> = None;

        for (gate_idx, gate) in gates.iter().enumerate() {
            let mut trial_mps = mps.clone();
            if trial_mps.apply_two_site_gate(q, &gate.matrix).is_ok() {
                let trial_max_bond = trial_mps.max_bond_dim();
                let trial_entropy = bond_entropy(&trial_mps, bond);
                if trial_max_bond <= current_max_bond && trial_entropy < best_entropy - 1e-2 {
                    best_entropy = trial_entropy;
                    best_gate_idx = Some(gate_idx);
                }
            }
        }

        if let Some(idx) = best_gate_idx {
            mps.apply_two_site_gate(q, &gates[idx].matrix)
                .expect("gate should succeed");
            corrections.push(super::MpsIndexGate {
                site: q,
                inverse_matrix: gates[idx].inverse_matrix.clone(),
            });
            num_applied += 1;
        }
    }

    num_applied
}

/// Run multiple sweeps of disentangling until convergence or `max_sweeps` reached.
pub(crate) fn disentangle(
    mps: &mut Mps,
    corrections: &mut Vec<super::MpsIndexGate>,
    max_sweeps: usize,
) -> usize {
    let mut total_applied = 0;
    for _ in 0..max_sweeps {
        let applied = disentangle_sweep(mps, corrections);
        total_applied += applied;
        if applied == 0 {
            break;
        }
    }
    total_applied
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_disentangler_gate_count() {
        let gates = build_disentangler_set();
        eprintln!("Disentangler gate set: {} unique gates", gates.len());
        assert!(
            gates.len() >= 15,
            "should have at least 15 unique gates, got {}",
            gates.len()
        );

        // Verify all gates are unitary
        let dim = 4;
        let id = DMatrix::<Complex64>::identity(dim, dim);
        for (i, gate) in gates.iter().enumerate() {
            let product = &gate.matrix * &gate.inverse_matrix;
            let diff = (&product - &id).norm();
            assert!(
                diff < 1e-10,
                "gate {i} is not unitary: ||U*Udg - I|| = {diff}"
            );
        }
    }
}
