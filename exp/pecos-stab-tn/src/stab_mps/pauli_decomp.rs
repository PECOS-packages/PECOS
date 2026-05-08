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

//! Decompose Pauli operators in the stabilizer/destabilizer basis.
//!
//! The stabilizer tableau defines a basis for the Pauli group. Given stabilizer
//! generators {`S_0`, ..., S_{N-1}} and destabilizer generators {`D_0`, ..., D_{N-1}}
//! where `D_i` anticommutes with `S_i` and commutes with all other `S_j`, any Pauli
//! operator P can be written as:
//!
//! ```text
//! P = phase * prod_i S_i^{s_i} * prod_j D_j^{d_j}
//! ```
//!
//! For the STN simulator, we need to decompose `Z_q` (Z on qubit q) in this basis.
//! The decomposition determines how RZ(theta) acts on the MPS.
//!
//! Uses the Y-convention phase table (matching the stabilizer-TN reference and
//! PECOS `SparseStabY`), where (x=1, z=1) represents Y (Hermitian, Y²=I).
//!
//! Reference: stabilizer-TN `gate_decomposition` function.

use num_complex::Complex64;
use pecos_core::IndexSet;
use pecos_simulators::GensGeneric;

/// Single-qubit Pauli kind for `decompose_pauli_string`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PauliKindForDecomp {
    X,
    Y,
    Z,
}

/// Decompose an arbitrary multi-qubit physical Pauli string into MPS-frame
/// `(flip_sites, sign_sites, phase)` such that
///   `C† · P · C = phase · X_{flip_sites} · Z_{sign_sites}`
/// where `C` is the Clifford encoded by the tableau. Used for
/// expectation-value computation `⟨Ψ|P|Ψ⟩ = ⟨MPS|C†·P·C|MPS⟩` at any `n`.
///
/// Algorithm: each single-qubit factor's anticommutation set with the
/// stabilizer generators (`stabs`/`destabs`) determines its contribution
/// to `flip_sites`/`sign_sites`. Multiple factors XOR the contributions.
/// `Y = i·X·Z` adds combined contributions plus an `i` phase factor per
/// `Y` factor (`i^{Y-count}`).
///
/// Anticommutation lookups:
/// - `X_q` anticom with `S_j` ⇔ `S_j` has `Z` or `Y` at q ⇔ `j ∈ stabs.col_z[q]`.
/// - `Z_q` anticom with `S_j` ⇔ `S_j` has `X` or `Y` at q ⇔ `j ∈ stabs.col_x[q]`.
/// - `Y_q` anticom with `S_j` ⇔ `S_j` has `X`-only or `Z`-only at q ⇔
///   `j ∈ (stabs.col_x[q] ⊕ stabs.col_z[q])`.
/// # Panics
///
/// Panics if any qubit index in `pauli` is >= the number of qubits in the
/// tableau.
pub fn decompose_pauli_string<S: IndexSet>(
    stabs: &GensGeneric<S>,
    destabs: &GensGeneric<S>,
    pauli: &[(usize, PauliKindForDecomp)],
) -> (Vec<usize>, Vec<usize>, Complex64) {
    let n = stabs.get_num_qubits();
    // Y-convention single-qubit Pauli multiplication phase table.
    // Indexing: Pauli codes I=0, Z=1, X=2, Y=3 (= 2*x_bit + z_bit).
    // Entry y_table[a][b] = phase of (Pauli a) · (Pauli b) under the
    // convention that Y = iXZ = (1,1) bit pattern is "pure Y" with no
    // implicit phase.
    let y_table: [[Complex64; 4]; 4] = {
        let one = Complex64::new(1.0, 0.0);
        let pi = Complex64::new(0.0, 1.0);
        let mi = Complex64::new(0.0, -1.0);
        [
            [one, one, one, one], // I · {I, Z, X, Y}
            [one, one, pi, mi],   // Z · {I, Z, X, Y}
            [one, mi, one, pi],   // X · {I, Z, X, Y}
            [one, pi, mi, one],   // Y · {I, Z, X, Y}
        ]
    };
    let pauli_code = |k: PauliKindForDecomp| -> u8 {
        match k {
            PauliKindForDecomp::Z => 1,
            PauliKindForDecomp::X => 2,
            PauliKindForDecomp::Y => 3,
        }
    };

    // Aggregate per-qubit Pauli factors with Pauli multiplication phase.
    // per_q[q] = (current Pauli bits at qubit q, accumulated phase).
    let mut per_q: Vec<(u8, Complex64)> = vec![(0, Complex64::new(1.0, 0.0)); n];
    for &(q, kind) in pauli {
        assert!(q < n, "decompose_pauli_string: qubit {q} >= num_qubits {n}");
        let new_code = pauli_code(kind);
        let cur = per_q[q];
        let phase_factor = y_table[cur.0 as usize][new_code as usize];
        per_q[q] = ((cur.0 ^ new_code), cur.1 * phase_factor);
    }

    // Aggregate flip/sign from per-qubit Pauli bits + total user-supplied
    // Pauli phase. Each X-bit at q contributes the X anticommutation set;
    // each Z-bit at q contributes the Z anticommutation set. A qubit with
    // Y bits (1, 1) contributes BOTH (XOR of X- and Z-anticom sets); the
    // associated `i` factor of Y = iXZ is captured naturally by the
    // y_table in `compute_decomposition_phase`.
    let mut flip = S::new();
    let mut sign = S::new();
    let mut total_phase = Complex64::new(1.0, 0.0);
    for (q, &(bits, p)) in per_q.iter().enumerate() {
        total_phase *= p;
        let x_bit = (bits >> 1) & 1;
        let z_bit = bits & 1;
        if x_bit == 1 {
            flip.xor_assign(&stabs.col_z[q]);
            sign.xor_assign(&destabs.col_z[q]);
        }
        if z_bit == 1 {
            flip.xor_assign(&stabs.col_x[q]);
            sign.xor_assign(&destabs.col_x[q]);
        }
    }

    let flip_vec: Vec<usize> = flip.iter().collect();
    let sign_vec: Vec<usize> = sign.iter().collect();

    let phase_from_compute = compute_decomposition_phase(stabs, destabs, &flip_vec, &sign_vec);
    let final_phase = phase_from_compute * total_phase;
    (flip_vec, sign_vec, final_phase)
}

/// Result of decomposing `Z_q` in the stabilizer/destabilizer basis.
#[derive(Debug)]
pub enum ZDecomposition {
    /// `Z_q` is in the stabilizer group (no destabilizer component).
    ///
    /// `Z_q` = phase * prod_{j in `sign_sites`} `S_j`
    ///
    /// On the MPS, each `S_j` contributes (-1) when the j-th destabilizer
    /// index is active. When `sign_sites` is empty, this is a global scalar.
    Stabilizer {
        /// Overall phase from the decomposition.
        phase: Complex64,
        /// Stabilizer indices whose product appears in the decomposition.
        /// The MPS picks up (-1) for each site j where the destabilizer is active.
        sign_sites: Vec<usize>,
    },

    /// `Z_q` has a destabilizer component.
    ///
    /// `Z_q` = phase * (prod_{j in `flip_sites`} `D_j`) * (prod_{k in `sign_sites`} `S_k`)
    ///
    /// Acting on the MPS:
    /// - Each `D_j` flips (X gate) the physical index at MPS site j
    /// - Each `S_k` contributes (-1) (Z gate) when the k-th destabilizer is active
    /// - The overall complex phase is included
    DestabilizerFlip {
        /// Destabilizer indices that get flipped (X gates on MPS).
        flip_sites: Vec<usize>,
        /// Overall complex phase from the decomposition.
        phase: Complex64,
        /// Stabilizer indices whose product appears in the decomposition.
        /// The MPS picks up (-1) for each site k where the destabilizer is active.
        sign_sites: Vec<usize>,
    },
}

/// Brute-force verify a decomposition by constructing 2^n x 2^n matrices.
/// Only usable for small n (say n<=6). Returns true if correct.
///
/// # Panics
///
/// Panics if the number of qubits exceeds what can be represented as a
/// matrix dimension (realistically n > 20).
pub fn verify_decomposition_brute_force<S: IndexSet>(
    stabs: &GensGeneric<S>,
    destabs: &GensGeneric<S>,
    q: usize,
    decomp: &ZDecomposition,
) -> bool {
    use nalgebra::DMatrix;

    let n = stabs.get_num_qubits();
    let dim = 1usize << n;
    let i_mat = DMatrix::<Complex64>::identity(2, 2);
    let x_mat = DMatrix::from_row_slice(
        2,
        2,
        &[
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
    );
    let z_mat_1q = DMatrix::from_row_slice(
        2,
        2,
        &[
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(-1.0, 0.0),
        ],
    );
    let y_mat = DMatrix::from_row_slice(
        2,
        2,
        &[
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, -1.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(0.0, 0.0),
        ],
    );

    let gen_matrix = |gens: &GensGeneric<S>, row: usize| -> DMatrix<Complex64> {
        let mut result = DMatrix::from_element(1, 1, Complex64::new(1.0, 0.0));
        for qq in 0..n {
            let has_x = gens.row_x[row].contains(qq);
            let has_z = gens.row_z[row].contains(qq);
            let pauli = match (has_x, has_z) {
                (false, false) => &i_mat,
                (true, false) => &x_mat,
                (false, true) => &z_mat_1q,
                (true, true) => &y_mat,
            };
            result = result.kronecker(pauli);
        }
        let mut phase = Complex64::new(1.0, 0.0);
        if gens.signs_minus.contains(row) {
            phase *= Complex64::new(-1.0, 0.0);
        }
        if gens.signs_i.contains(row) {
            phase *= Complex64::new(0.0, 1.0);
        }
        result * phase
    };

    // Build Z_q matrix
    let mut z_mat = DMatrix::from_element(1, 1, Complex64::new(1.0, 0.0));
    for qq in 0..n {
        let p = if qq == q { &z_mat_1q } else { &i_mat };
        z_mat = z_mat.kronecker(p);
    }

    match decomp {
        ZDecomposition::Stabilizer { phase, sign_sites } => {
            let mut product = DMatrix::<Complex64>::identity(dim, dim);
            for &k in sign_sites {
                product = gen_matrix(stabs, k) * product;
            }
            product *= *phase;
            (&z_mat - &product).norm() < 1e-10
        }
        ZDecomposition::DestabilizerFlip {
            flip_sites,
            phase,
            sign_sites,
        } => {
            let mut product = DMatrix::<Complex64>::identity(dim, dim);
            for &k in sign_sites {
                product = gen_matrix(stabs, k) * product;
            }
            for &j in flip_sites {
                product = gen_matrix(destabs, j) * product;
            }
            product *= *phase;
            let diff = (&z_mat - &product).norm();
            if diff > 1e-10 {
                // Find the correct phase by dividing z_mat by the unsigned product
                let mut unsigned_product = DMatrix::<Complex64>::identity(dim, dim);
                for &k in sign_sites {
                    unsigned_product = gen_matrix(stabs, k) * unsigned_product;
                }
                for &j in flip_sites {
                    unsigned_product = gen_matrix(destabs, j) * unsigned_product;
                }
                // Z = correct_phase * unsigned_product, so correct_phase = (Z * unsigned_product†)[0,0]
                // Since both are unitary Pauli products, the correct phase is Z[0,0] / product[0,0]
                // Or more robustly: trace(Z * product†) / dim
                let adj = unsigned_product.adjoint();
                let correct = (&z_mat * &adj).trace()
                    / Complex64::new(f64::from(u32::try_from(dim).unwrap()), 0.0);
                eprintln!("  PHASE MISMATCH: diff={diff:.4e}");
                eprintln!("    computed phase={phase}");
                eprintln!("    correct  phase={correct:.4}");
                eprintln!("    flip={flip_sites:?}, sign={sign_sites:?}");
            }
            diff < 1e-10
        }
    }
}

/// Decompose `Z_q` in the stabilizer/destabilizer basis.
///
/// This mirrors the measurement logic in `SparseStabY`: `Z_q` is deterministic
/// (in the stabilizer group) when `stabs.col_x[q]` is empty, and requires
/// destabilizer decomposition otherwise.
pub fn decompose_z<S: IndexSet>(
    stabs: &GensGeneric<S>,
    destabs: &GensGeneric<S>,
    q: usize,
) -> ZDecomposition {
    if stabs.col_x[q].is_empty() {
        // Z_q commutes with all stabilizers -> it's in the stabilizer group.
        // Z_q = phase * prod_{j in sign_sites} S_j
        // where sign_sites = destabs.col_x[q] (destabilizers that anticommute with Z_q)
        let sign = compute_stabilizer_sign(stabs, destabs, q);
        let sign_sites: Vec<usize> = destabs.col_x[q].iter().collect();
        let phase = if sign < 0.0 {
            Complex64::new(-1.0, 0.0)
        } else {
            Complex64::new(1.0, 0.0)
        };
        ZDecomposition::Stabilizer { phase, sign_sites }
    } else {
        // Z_q anticommutes with at least one stabilizer.
        // Find the destabilizer row that anticommutes with Z_q.
        decompose_z_nondeterministic(stabs, destabs, q)
    }
}

/// Compute the sign of `Z_q` when it is in the stabilizer group.
///
/// `Z_q` = (+/-1) * product of stabilizers. The sign is computed by tracking
/// how the destabilizer generators that have X on qubit q combine.
///
/// This follows the same logic as `SparseStabY::deterministic_meas`.
fn compute_stabilizer_sign<S: IndexSet>(
    stabs: &GensGeneric<S>,
    destabs: &GensGeneric<S>,
    q: usize,
) -> f64 {
    // Count minus signs from the destabilizer generators that have X on qubit q.
    // These destabs "activate" certain stabilizers to reconstruct Z_q.
    let mut num_minuses = destabs.col_x[q].intersection_count(&stabs.signs_minus);
    let mut num_is = destabs.col_x[q].intersection_count(&stabs.signs_i);

    // Y-convention correction: add n_Y per participating stab
    for row in destabs.col_x[q].iter() {
        num_is += stabs.row_x[row].intersection_count(&stabs.row_z[row]);
    }

    // W-convention commutation phase accumulation
    let mut cumulative_x = S::new();
    for row in destabs.col_x[q].iter() {
        num_minuses += stabs.row_z[row].intersection_count(&cumulative_x);
        cumulative_x.xor_assign(&stabs.row_x[row]);
    }

    // Convert i-count to sign. For the Stabilizer branch the total
    // phase must be real (±1), so i-count must be even.
    debug_assert!(
        num_is.is_multiple_of(2),
        "stabilizer sign: i-count {num_is} must be even for real phase"
    );
    if num_is % 4 == 2 {
        num_minuses += 1;
    }

    if num_minuses & 1 != 0 { -1.0 } else { 1.0 }
}

/// Decompose `Z_q` when it anticommutes with at least one stabilizer.
///
/// `Z_q` = phase * (product of some stabilizers) * `D_k`
///
/// We need to find:
/// - k: which destabilizer to flip
/// - phase: the overall complex phase
/// - `sign_sites`: which other stabilizer indices contribute signs
fn decompose_z_nondeterministic<S: IndexSet>(
    stabs: &GensGeneric<S>,
    destabs: &GensGeneric<S>,
    q: usize,
) -> ZDecomposition {
    // The destabilizer col_x[q] tells us which destabilizer generators have X on qubit q.
    // These are the ones that anticommute with Z_q.
    //
    // For STN, we need to pick one destabilizer D_k such that Z_q can be written as
    // a product of stabilizers times D_k. The standard choice: pick the first
    // destabilizer that anticommutes with Z_q (analogous to how measurement picks
    // the first anticommuting stabilizer, but here we look at destabilizers).
    //
    // Actually, the key insight: in the stabilizer formalism, Z_q anticommutes with
    // stabilizer generators indexed by stabs.col_x[q]. The destabilizer D_k that
    // "pairs" with one of these stabilizers is the one we use.
    //
    // For the simplest decomposition: take the first anticommuting stabilizer index k.
    // Then Z_q = (phase) * (product of stabilizers that anticommute with D_k's effect) * D_k.
    //
    // The key relationship: destabilizer D_k anticommutes with S_k and commutes with
    // all other stabilizers. When we write Z_q in terms of destabilizers and stabilizers,
    // the destabilizer component is determined by which stabilizers Z_q anticommutes with.

    // Pick the first stabilizer that anticommutes with Z_q.
    // The paired destabilizer at that index is our flip site.
    // Now we need to figure out what stabilizer product accompanies the destabilizers.
    // Z_q * D_k should commute with all stabilizers (since Z_q anticommutes with S_k
    // and D_k anticommutes with S_k, their product commutes with S_k).
    // But Z_q might also anticommute with other stabilizers S_j (j != k).
    // For those, we need additional destabilizer flips... or stabilizer factors.
    //
    // Actually, in the STN framework the decomposition is simpler. We express Z_q as:
    //   Z_q = phase * (prod of some S_j's) * (prod of some D_j's)
    //
    // The destabilizer part: check destabs.col_x[q] to find all destabilizers that
    // have X or Y on qubit q. Wait -- that's the wrong direction.
    //
    // Let me reconsider. The correct approach follows from the symplectic structure:
    //
    // The stabilizer/destabilizer tableau T = [D_0, ..., D_{n-1}, S_0, ..., S_{n-1}]
    // forms a symplectic basis. Any Pauli P can be uniquely decomposed as:
    //   P = phase * prod_i D_i^{d_i} * prod_j S_j^{s_j}
    //
    // where d_i = 1 iff P anticommutes with S_i,
    // and   s_j = 1 iff P anticommutes with D_j.
    //
    // For P = Z_q:
    // - d_i = 1 iff Z_q anticommutes with S_i, i.e., S_i has X or Y on qubit q
    //   -> d_i = 1 for i in stabs.col_x[q]
    // - s_j = 1 iff Z_q anticommutes with D_j, i.e., D_j has X or Y on qubit q
    //   -> s_j = 1 for j in destabs.col_x[q]
    //
    // The phase comes from the ordering and signs of the generators.

    // Collect the destabilizer flip sites (d_i = 1)
    let flip_sites: Vec<usize> = stabs.col_x[q].iter().collect();

    // Collect the stabilizer sign sites (s_j = 1)
    let sign_sites: Vec<usize> = destabs.col_x[q].iter().collect();

    let phase = compute_decomposition_phase(stabs, destabs, &flip_sites, &sign_sites);

    ZDecomposition::DestabilizerFlip {
        flip_sites,
        phase,
        sign_sites,
    }
}

/// Compute the complex phase of the decomposition `Z_q` = phase * prod(D) * prod(S).
///
/// Uses the Y-convention per-qubit phase table (matching the stabilizer-TN reference).
/// In Y convention: (x=1, z=1) = Y = iXZ, and Y^2 = I.
fn compute_decomposition_phase<S: IndexSet>(
    stabs: &GensGeneric<S>,
    destabs: &GensGeneric<S>,
    flip_sites: &[usize],
    sign_sites: &[usize],
) -> Complex64 {
    // Cumulative Pauli string (x, z) parts and complex phase.
    let mut cum_x = S::new();
    let mut cum_z = S::new();
    let mut phase = Complex64::new(1.0, 0.0);

    // Helper: multiply cumulative Pauli by a generator, accumulating phase.
    let multiply_generator = |cum_x: &mut S,
                              cum_z: &mut S,
                              phase: &mut Complex64,
                              gen_x: &S,
                              gen_z: &S,
                              gen_is_minus: bool,
                              gen_is_i: bool| {
        // Generator's own sign
        if gen_is_minus {
            *phase *= Complex64::new(-1.0, 0.0);
        }
        if gen_is_i {
            *phase *= Complex64::new(0.0, 1.0);
        }

        // Per-qubit phase using Y-convention table (matches reference).
        // In Y convention: I=0, Z=1, X=2, Y=3 where (1,1) = Y = iXZ.
        // Reference: phase_mat = [[1,1,1,1],[1,1,1j,-1j],[1,-1j,1,1j],[1,1j,-1j,1]]
        let y_table: [[Complex64; 4]; 4] = {
            let one = Complex64::new(1.0, 0.0);
            let pi = Complex64::new(0.0, 1.0);
            let mi = Complex64::new(0.0, -1.0);
            [
                [one, one, one, one], // I * {I,Z,X,Y}
                [one, one, pi, mi],   // Z * {I,Z,X,Y}
                [one, mi, one, pi],   // X * {I,Z,X,Y}
                [one, pi, mi, one],   // Y * {I,Z,X,Y}
            ]
        };

        for q in gen_x.iter() {
            let p1 = 2 * usize::from(cum_x.contains(q)) + usize::from(cum_z.contains(q));
            let p2 = 2 + usize::from(gen_z.contains(q));
            *phase *= y_table[p1][p2];
        }
        for q in gen_z.iter() {
            if gen_x.contains(q) {
                continue;
            }
            let p1 = 2 * usize::from(cum_x.contains(q)) + usize::from(cum_z.contains(q));
            *phase *= y_table[p1][1];
        }

        // Update cumulative Pauli
        cum_x.xor_assign(gen_x);
        cum_z.xor_assign(gen_z);
    };

    // Multiply destabilizers (D_j for j in flip_sites)
    for &j in flip_sites {
        let is_minus = destabs.signs_minus.contains(j);
        let is_i = destabs.signs_i.contains(j);
        multiply_generator(
            &mut cum_x,
            &mut cum_z,
            &mut phase,
            &destabs.row_x[j],
            &destabs.row_z[j],
            is_minus,
            is_i,
        );
    }

    // Multiply stabilizers (S_k for k in sign_sites)
    for &k in sign_sites {
        let is_minus = stabs.signs_minus.contains(k);
        let is_i = stabs.signs_i.contains(k);
        multiply_generator(
            &mut cum_x,
            &mut cum_z,
            &mut phase,
            &stabs.row_x[k],
            &stabs.row_z[k],
            is_minus,
            is_i,
        );
    }

    // (Z_q-specific sanity check removed: this routine is now also used by
    // `decompose_pauli_string` for arbitrary Pauli decompositions where the
    // cumulative X part can be non-empty.)

    // phase = product_phase (the phase of prod(D)*prod(S) as a Pauli string).
    // We need decomp_phase such that Z_q = decomp_phase * prod.
    // So decomp_phase = 1 / product_phase.

    if phase.norm_sqr() > 1e-20 {
        Complex64::new(1.0, 0.0) / phase
    } else {
        Complex64::new(1.0, 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;
    use pecos_core::QubitId;
    use pecos_simulators::{CliffordGateable, SparseStabY};

    /// Build the 2^n x 2^n Pauli matrix for generator `row` of `gens`.
    /// In Y-convention: (x=1,z=1) = Y (Hermitian).
    fn generator_matrix<S: IndexSet>(
        gens: &GensGeneric<S>,
        row: usize,
        n: usize,
    ) -> DMatrix<Complex64> {
        let i_mat = DMatrix::<Complex64>::identity(2, 2);
        let x_mat = DMatrix::from_row_slice(
            2,
            2,
            &[
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        );
        let z_mat = DMatrix::from_row_slice(
            2,
            2,
            &[
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(-1.0, 0.0),
            ],
        );
        let y_mat = DMatrix::from_row_slice(
            2,
            2,
            &[
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, -1.0),
                Complex64::new(0.0, 1.0),
                Complex64::new(0.0, 0.0),
            ],
        );

        // Build tensor product of per-qubit Paulis
        let mut result = DMatrix::from_element(1, 1, Complex64::new(1.0, 0.0));
        for q in 0..n {
            let has_x = gens.row_x[row].contains(q);
            let has_z = gens.row_z[row].contains(q);
            let pauli = match (has_x, has_z) {
                (false, false) => &i_mat,
                (true, false) => &x_mat,
                (false, true) => &z_mat,
                (true, true) => &y_mat, // Y convention
            };
            result = result.kronecker(pauli);
        }

        // Apply phase: (-1)^minus * i^i_bit
        let mut phase = Complex64::new(1.0, 0.0);
        if gens.signs_minus.contains(row) {
            phase *= Complex64::new(-1.0, 0.0);
        }
        if gens.signs_i.contains(row) {
            phase *= Complex64::new(0.0, 1.0);
        }

        result * phase
    }

    /// Build `Z_q` as a 2^n x 2^n matrix.
    fn z_matrix(q: usize, n: usize) -> DMatrix<Complex64> {
        let i_mat = DMatrix::<Complex64>::identity(2, 2);
        let z_mat = DMatrix::from_row_slice(
            2,
            2,
            &[
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(-1.0, 0.0),
            ],
        );
        let mut result = DMatrix::from_element(1, 1, Complex64::new(1.0, 0.0));
        for qq in 0..n {
            let pauli = if qq == q { &z_mat } else { &i_mat };
            result = result.kronecker(pauli);
        }
        result
    }

    /// Brute-force verify decomposition phase by matrix multiplication.
    fn verify_decomposition_phase(sim: &SparseStabY, q: usize) {
        let n = sim.stabs().get_num_qubits();
        let decomp = decompose_z(sim.stabs(), sim.destabs(), q);

        let z_mat = z_matrix(q, n);

        match &decomp {
            ZDecomposition::Stabilizer { phase, sign_sites } => {
                // Z_q = phase * prod(S_k for k in sign_sites)
                let dim = 1usize << n;
                let mut product = DMatrix::<Complex64>::identity(dim, dim);
                for &k in sign_sites {
                    product = generator_matrix(sim.stabs(), k, n) * product;
                }
                product *= *phase;
                // Check Z_q == product
                let diff = (&z_mat - &product).norm();
                assert!(
                    diff < 1e-10,
                    "Stabilizer decomp for Z_{q}: ||Z - phase*prod(S)|| = {diff:.4e}, phase={phase}"
                );
            }
            ZDecomposition::DestabilizerFlip {
                flip_sites,
                phase,
                sign_sites,
            } => {
                // Z_q = phase * prod(D_j) * prod(S_k)
                let dim = 1usize << n;
                let mut product = DMatrix::<Complex64>::identity(dim, dim);
                // Multiply stabilizers first (rightmost)
                for &k in sign_sites {
                    product = generator_matrix(sim.stabs(), k, n) * product;
                }
                // Then destabilizers
                for &j in flip_sites {
                    product = generator_matrix(sim.destabs(), j, n) * product;
                }
                product *= *phase;
                let diff = (&z_mat - &product).norm();
                assert!(
                    diff < 1e-10,
                    "DestabFlip decomp for Z_{q}: ||Z - phase*prod(D)*prod(S)|| = {diff:.4e}\n  \
                     phase={phase}, flip={flip_sites:?}, sign={sign_sites:?}"
                );
            }
        }
    }

    // Phase verification is done via the verification test (uses StabMps which has destab sign fixups).

    #[test]
    fn test_destab_sign_tracking_z() {
        // Initial state: D_0 = X_0. Z(0) conjugates X_0 → -X_0.
        // With destab sign tracking, the minus should appear in signs_minus.
        let mut sim = SparseStabY::new(2).with_destab_sign_tracking();

        // Initial: D_0 = X_0 (no minus)
        assert!(sim.destabs().row_x[0].contains(0));
        assert!(!sim.destabs().signs_minus.contains(0));

        // Z(0) should conjugate: Z*X*Z = -X
        sim.z(&[QubitId(0)]);

        assert!(
            sim.destabs().row_x[0].contains(0),
            "D[0] should still have X on q0"
        );
        assert!(
            sim.destabs().signs_minus.contains(0),
            "D[0] should have minus=true after Z (Z*X*Z = -X)"
        );
    }

    #[test]
    fn test_decomposition_phase_brute_force_seed102_circuit() {
        // Reproduce the seed 102 Clifford prefix before the failing T gate.
        // Gate sequence from fuzz(2,10,102):
        //   rz(q1), sz(q1), sz(q0), h(q0), sz(q0), x(q1), cx(1,0), x(q1), t(q0), x(q0)
        // The first rz is on initial state (scalar), so the first DestabilizerFlip
        // is at t(q0) = step 9.
        let q0 = QubitId(0);
        let q1 = QubitId(1);
        let mut sim = SparseStabY::new(2);

        // Step 1: rz(q1) — on initial state, this is a scalar. Skip for tableau.
        // Step 2: sz(q1)
        sim.sz(&[q1]);
        // Step 3: sz(q0)
        sim.sz(&[q0]);
        // Step 4: h(q0)
        sim.h(&[q0]);
        // Step 5: sz(q0)
        sim.sz(&[q0]);
        // Step 6: x(q1)
        sim.x(&[q1]);
        // Step 7: cx(1, 0)
        sim.cx(&[(q1, q0)]);
        // Step 8: x(q1)
        sim.x(&[q1]);

        // Now verify the decomposition of Z_0 (the T gate target)
        eprintln!("=== Seed 102 state before T(q0) ===");
        let n = 2;
        for i in 0..n {
            let stab_x: Vec<usize> = sim.stabs().row_x[i].iter().collect();
            let stab_z: Vec<usize> = sim.stabs().row_z[i].iter().collect();
            let destab_x: Vec<usize> = sim.destabs().row_x[i].iter().collect();
            let destab_z: Vec<usize> = sim.destabs().row_z[i].iter().collect();
            let s_minus = sim.stabs().signs_minus.contains(i);
            let s_i = sim.stabs().signs_i.contains(i);
            let d_minus = sim.destabs().signs_minus.contains(i);
            let d_i = sim.destabs().signs_i.contains(i);
            eprintln!("  stab[{i}]: x={stab_x:?} z={stab_z:?} minus={s_minus} i={s_i}");
            eprintln!("  destab[{i}]: x={destab_x:?} z={destab_z:?} minus={d_minus} i={d_i}");
        }

        let decomp = decompose_z(sim.stabs(), sim.destabs(), 0);
        eprintln!("  decomposition: {decomp:?}");

        verify_decomposition_phase(&sim, 0);
        verify_decomposition_phase(&sim, 1);
    }

    #[test]
    fn test_z_on_initial_state_is_stabilizer() {
        // Initial state |00>: stabilizers are Z_0, Z_1, destabilizers X_0, X_1
        // Z_0 is in the stabilizer group with phase +1
        // sign_sites = destabs.col_x[0] = {0} (X_0 has X on q0)
        let sim = SparseStabY::new(2);
        let decomp = decompose_z(sim.stabs(), sim.destabs(), 0);
        match decomp {
            ZDecomposition::Stabilizer { phase, .. } => {
                assert!(
                    (phase.re - 1.0).abs() < f64::EPSILON,
                    "Z_0 should have phase +1 on |0>"
                );
            }
            ZDecomposition::DestabilizerFlip { .. } => panic!("Z_0 should be a stabilizer on |00>"),
        }
    }

    #[test]
    fn test_z_on_x_state_is_stabilizer_minus() {
        // State |10>: X on qubit 0. Z_0 eigenvalue is -1.
        let mut sim = SparseStabY::new(2);
        sim.x(&[QubitId(0)]);
        let decomp = decompose_z(sim.stabs(), sim.destabs(), 0);
        match decomp {
            ZDecomposition::Stabilizer { phase, .. } => {
                assert!(
                    (phase.re + 1.0).abs() < f64::EPSILON,
                    "Z_0 should have phase -1 on |1>"
                );
            }
            ZDecomposition::DestabilizerFlip { .. } => panic!("Z_0 should be a stabilizer on |10>"),
        }
    }

    #[test]
    fn test_z_after_hadamard_is_destabilizer() {
        // State |+> = H|0>: stabilizer is X, destabilizer is Z
        let mut sim = SparseStabY::new(1);
        sim.h(&[QubitId(0)]);
        let decomp = decompose_z(sim.stabs(), sim.destabs(), 0);
        match decomp {
            ZDecomposition::DestabilizerFlip {
                flip_sites,
                sign_sites,
                ..
            } => {
                assert_eq!(flip_sites, vec![0], "should flip destabilizer 0");
                assert!(sign_sites.is_empty(), "no sign sites for simple case");
            }
            ZDecomposition::Stabilizer { .. } => panic!("Z should be a destabilizer flip after H"),
        }
    }

    /// Diagnostic: print (phase, `flip_sites`, `sign_sites`, Ys) for a variety
    /// of states. Used to understand what cases `decompose_z` actually produces.
    #[test]
    fn test_decomposition_cases_survey() {
        // Helper: apply gates, decompose Z_q, report (phase, Ys, flip, sign)
        let survey_q = |label: &str, q: usize, gates: fn(&mut SparseStabY)| {
            let mut sim = SparseStabY::new(3);
            gates(&mut sim);
            let decomp = decompose_z(sim.stabs(), sim.destabs(), q);
            match decomp {
                ZDecomposition::Stabilizer {
                    phase,
                    ref sign_sites,
                } => {
                    eprintln!(
                        "{label} Z_{q}: Stabilizer phase={phase:.3} sign_sites={sign_sites:?}"
                    );
                }
                ZDecomposition::DestabilizerFlip {
                    ref flip_sites,
                    phase,
                    ref sign_sites,
                } => {
                    let ys = flip_sites.iter().filter(|f| sign_sites.contains(f)).count();
                    eprintln!(
                        "{label} Z_{q}: DestabFlip phase={phase:.3} flip={flip_sites:?} sign={sign_sites:?} Ys={ys}"
                    );
                }
            }
        };
        let survey = |label: &str, gates: fn(&mut SparseStabY)| {
            let mut sim = SparseStabY::new(3);
            gates(&mut sim);
            let decomp = decompose_z(sim.stabs(), sim.destabs(), 0);
            match decomp {
                ZDecomposition::Stabilizer {
                    phase,
                    ref sign_sites,
                } => {
                    eprintln!("{label}: Stabilizer phase={phase:.3} sign_sites={sign_sites:?}");
                }
                ZDecomposition::DestabilizerFlip {
                    ref flip_sites,
                    phase,
                    ref sign_sites,
                } => {
                    let ys = flip_sites.iter().filter(|f| sign_sites.contains(f)).count();
                    eprintln!(
                        "{label}: DestabFlip phase={phase:.3} flip={flip_sites:?} sign={sign_sites:?} Ys={ys}"
                    );
                }
            }
        };

        survey("|0⟩", |s| {
            let _ = s;
        });
        survey("H|0⟩", |s| {
            s.h(&[QubitId(0)]);
        });
        survey("X|0⟩=|1⟩", |s| {
            s.x(&[QubitId(0)]);
        });
        survey("SH|0⟩", |s| {
            s.h(&[QubitId(0)]);
            s.sz(&[QubitId(0)]);
        });
        // Multi-site cases via various tableau setups
        // CX then decompose Z_0 or Z_1 -- decomp depends on which qubit
        survey_q("H,CX(0,1)", 0, |s| {
            s.h(&[QubitId(0)]);
            s.cx(&[(QubitId(0), QubitId(1))]);
        });
        survey_q("H,CX(0,1)", 1, |s| {
            s.h(&[QubitId(0)]);
            s.cx(&[(QubitId(0), QubitId(1))]);
        });
        survey_q("H,H,CX(0,1)", 0, |s| {
            s.h(&[QubitId(0)]);
            s.h(&[QubitId(1)]);
            s.cx(&[(QubitId(0), QubitId(1))]);
        });
        survey_q("H,H,CX(0,1)", 1, |s| {
            s.h(&[QubitId(0)]);
            s.h(&[QubitId(1)]);
            s.cx(&[(QubitId(0), QubitId(1))]);
        });
        // Looking for multi-site: stab has X on multiple qubits after some sequence
        survey_q("H(0),CX(0,1),H(0)", 0, |s| {
            s.h(&[QubitId(0)]);
            s.cx(&[(QubitId(0), QubitId(1))]);
            s.h(&[QubitId(0)]);
        });
        survey_q("H(0),CX(0,1),H(0),CX(0,1)", 0, |s| {
            s.h(&[QubitId(0)]);
            s.cx(&[(QubitId(0), QubitId(1))]);
            s.h(&[QubitId(0)]);
            s.cx(&[(QubitId(0), QubitId(1))]);
        });
        // After H,CX,H on q0: X_0 X_1 stab, should give decompose with multiple flips
        survey_q("H(0),CX(0,1),H(1)", 0, |s| {
            s.h(&[QubitId(0)]);
            s.cx(&[(QubitId(0), QubitId(1))]);
            s.h(&[QubitId(1)]);
        });
        // GHZ-like state
        survey_q("H,CX(0,1),CX(1,2)", 0, |s| {
            s.h(&[QubitId(0)]);
            s.cx(&[(QubitId(0), QubitId(1))]);
            s.cx(&[(QubitId(1), QubitId(2))]);
        });
        survey_q("H,CX(0,1),CX(1,2)", 2, |s| {
            s.h(&[QubitId(0)]);
            s.cx(&[(QubitId(0), QubitId(1))]);
            s.cx(&[(QubitId(1), QubitId(2))]);
        });
        // Setup with Y-type stabilizers: SH gives Y-basis
        survey_q("SH,CX(0,1)", 0, |s| {
            s.h(&[QubitId(0)]);
            s.sz(&[QubitId(0)]);
            s.cx(&[(QubitId(0), QubitId(1))]);
        });
        survey_q("SH,CX(0,1)", 1, |s| {
            s.h(&[QubitId(0)]);
            s.sz(&[QubitId(0)]);
            s.cx(&[(QubitId(0), QubitId(1))]);
        });
        // Try to get Ys > 0: need same generator index in both flip_sites and sign_sites
        // Requires stab_k and destab_k both have X on same qubit q
        survey_q("S(0),H(0)", 0, |s| {
            s.sz(&[QubitId(0)]);
            s.h(&[QubitId(0)]);
        });
        // H,S(0) gives stab=Y_0, destab=X_0. col_x[0] for both = {0}. Ys=1!
        // For Z_0 decomp, flip = stabs.col_x[0] = {0}, sign = destabs.col_x[0] = {0}, Ys=1
        survey_q("H(0),S(0)", 0, |s| {
            s.h(&[QubitId(0)]);
            s.sz(&[QubitId(0)]);
        });
        // More Y cases
        survey_q("H(0),S(0),CX(0,1)", 0, |s| {
            s.h(&[QubitId(0)]);
            s.sz(&[QubitId(0)]);
            s.cx(&[(QubitId(0), QubitId(1))]);
        });
        survey_q("H(0),S(0),CX(0,1)", 1, |s| {
            s.h(&[QubitId(0)]);
            s.sz(&[QubitId(0)]);
            s.cx(&[(QubitId(0), QubitId(1))]);
        });
        survey_q("H(1),S(1),CX(0,1)", 0, |s| {
            s.h(&[QubitId(1)]);
            s.sz(&[QubitId(1)]);
            s.cx(&[(QubitId(0), QubitId(1))]);
        });
        survey_q("H,H,S(0),S(1),CX(0,1)", 0, |s| {
            s.h(&[QubitId(0)]);
            s.h(&[QubitId(1)]);
            s.sz(&[QubitId(0)]);
            s.sz(&[QubitId(1)]);
            s.cx(&[(QubitId(0), QubitId(1))]);
        });
        survey_q("H,H,S(0),S(1),CX(0,1)", 1, |s| {
            s.h(&[QubitId(0)]);
            s.h(&[QubitId(1)]);
            s.sz(&[QubitId(0)]);
            s.sz(&[QubitId(1)]);
            s.cx(&[(QubitId(0), QubitId(1))]);
        });
    }

    #[test]
    fn test_z_after_bell_state() {
        // Bell state: H on q0, then CX(q0, q1)
        // Stabilizers: X_0 X_1, Z_0 Z_1
        // Z_0 anticommutes with X_0 X_1 (has X on q0)
        let mut sim = SparseStabY::new(2);
        sim.h(&[QubitId(0)]);
        sim.cx(&[(QubitId(0), QubitId(1))]);
        let decomp = decompose_z(sim.stabs(), sim.destabs(), 0);
        match decomp {
            ZDecomposition::DestabilizerFlip { flip_sites, .. } => {
                // Should have exactly one flip site
                assert_eq!(flip_sites.len(), 1, "should have one flip site");
            }
            ZDecomposition::Stabilizer { .. } => {
                panic!("Z_0 should be a destabilizer flip in Bell state")
            }
        }
    }
}
