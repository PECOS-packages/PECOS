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

//! Shared measurement logic for STN and MAST simulators.
//!
//! Measures a qubit in the Z basis using the stabilizer tableau for structure
//! and the MPS for probability computation and projection.
//!
//! The measurement protocol decomposes `Z_q` in the stabilizer basis, computes
//! the expectation value from the MPS, samples an outcome, and projects the
//! MPS using the (I + sign * `Z_q)/2` projector. After projection, the measured
//! site collapses to sigma=0 (the stabilizer eigenstate).
//!
//! Reference: Masot-Llima, Garcia-Saez. arXiv:2403.08724, Section III.

use super::pauli_decomp::{ZDecomposition, decompose_z};
use crate::mps::Mps;
use nalgebra::DMatrix;
use num_complex::Complex64;
use pecos_core::BitSet;
use pecos_random::PecosRng;
use pecos_simulators::{CliffordGateable, MeasurementResult, SparseStabY};

/// Virtual coefficient-MPS sites whose disentangling proofs must be rechecked
/// after a physical measurement projection.
#[derive(Debug, Default)]
pub(super) struct ProjectionUpdate {
    /// Site intended to be placed in `|0>` by the measurement basis rotation.
    pub(super) collapsed_site: Option<usize>,
    /// Other sites touched by projection or compensating virtual gates.
    pub(super) modified_sites: Vec<usize>,
}

/// Probability and virtual-site metadata returned by a forced projection.
pub(super) struct ForcedProjectionResult {
    pub(super) probability: f64,
    pub(super) update: ProjectionUpdate,
}

/// Sampled measurement and virtual-site metadata for a live simulator update.
pub(super) struct LiveMeasurementResult {
    pub(super) measurement: MeasurementResult,
    pub(super) update: ProjectionUpdate,
}

/// Check if the MPS is trivial (all sites in a computational basis state).
fn is_mps_trivial(mps: &Mps) -> bool {
    mps.max_bond_dim() == 1
        && mps.tensors().iter().all(|t| {
            let chi_r = t.ncols() / 2;
            let b0_norm: f64 = (0..t.nrows())
                .flat_map(|i| (0..chi_r).map(move |j| t[(i, j)].norm_sqr()))
                .sum();
            let b1_norm: f64 = (0..t.nrows())
                .flat_map(|i| (0..chi_r).map(move |j| t[(i, chi_r + j)].norm_sqr()))
                .sum();
            b0_norm < 1e-12 || b1_norm < 1e-12
        })
}

/// Put a computational-basis product coefficient MPS into `|0...0>` while
/// preserving the represented physical state by absorbing its basis Xs into
/// the Clifford tableau.
///
/// The tableau-only measurement fast path is valid for `C|0...0>`, not for a
/// general `C|b>`. Non-Clifford evolution and earlier exact projections can
/// produce the latter even though every MPS bond is one.
fn canonicalize_trivial_mps_basis(tableau: &mut SparseStabY, mps: &mut Mps) -> Vec<usize> {
    let norm_squared = mps.norm_squared();
    debug_assert!(
        (norm_squared - 1.0).abs() < 1e-8,
        "trivial-basis canonicalization requires a normalized MPS, got norm²={norm_squared}"
    );
    let x_gate = DMatrix::from_row_slice(
        2,
        2,
        &[
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
    );
    let mut modified_sites = Vec::new();
    for site in 0..mps.num_sites() {
        let chi_r = mps.bond_dim(site + 1);
        let block_0 = crate::mps::tensor::phys_block(&mps.tensors()[site], 0, chi_r);
        let block_0_norm: f64 = block_0.iter().map(num_complex::Complex::norm_sqr).sum();
        // Every project_forced_z entry path normalizes before this helper can
        // be reached again, so the block weight has unit-state scale and this
        // absolute zero threshold is intentional.
        if block_0_norm < 1e-12 {
            // C|...1...> = (C X_site)(X_site|...1...>).
            crate::stab_mps::tableau_compose::right_compose_x(tableau, site);
            mps.apply_one_site_gate(site, &x_gate)
                .expect("MPS op on valid site");
            modified_sites.push(site);
        }
    }
    modified_sites
}

/// Compute `<mps| phase · X_flip · Z_sign |mps>` via clone + inner product.
///
/// Returns the expectation value of the Pauli string. Z applied first, then
/// X (matches the measurement projection convention in this module).
///
/// # Panics
///
/// Panics if any MPS gate application fails on a valid site (should not happen
/// for in-range sites).
#[must_use]
pub fn pauli_expectation(
    mps: &Mps,
    flip_sites: &[usize],
    sign_sites: &[usize],
    phase: Complex64,
) -> Complex64 {
    if flip_sites.is_empty() && sign_sites.is_empty() {
        return phase;
    }
    let x_gate = DMatrix::from_row_slice(
        2,
        2,
        &[
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
    );
    let z_diag = [Complex64::new(1.0, 0.0), Complex64::new(-1.0, 0.0)];
    let mut mps_op = mps.clone();
    for &k in sign_sites {
        mps_op
            .apply_diagonal_one_site(k, &z_diag)
            .expect("MPS op on valid site");
    }
    for &j in flip_sites {
        mps_op
            .apply_one_site_gate(j, &x_gate)
            .expect("MPS op on valid site");
    }
    let raw = mps_inner_product(mps, &mps_op);
    phase * raw
}

/// Compute `<mps|Z_q|mps>` by applying the decomposition to a clone and taking the inner product.
///
/// Returns the raw expectation value (before multiplying by the decomposition phase).
/// The full expectation is: `phase * apply_z_to_clone_and_overlap(...)`.
#[must_use]
pub fn z_expectation_value(tableau: &SparseStabY, mps: &Mps, q: usize) -> Complex64 {
    let decomp = decompose_z(tableau.stabs(), tableau.destabs(), q);
    match decomp {
        ZDecomposition::Stabilizer { phase, sign_sites } => {
            pauli_expectation(mps, &[], &sign_sites, phase)
        }
        ZDecomposition::DestabilizerFlip {
            flip_sites,
            phase,
            sign_sites,
        } => pauli_expectation(mps, &flip_sites, &sign_sites, phase),
    }
}

fn forced_outcome_probability(expectation: f64, outcome: bool) -> f64 {
    let probability_zero = f64::midpoint(1.0, expectation).clamp(0.0, 1.0);
    if outcome {
        1.0 - probability_zero
    } else {
        probability_zero
    }
}

/// Compute the inner product <`mps_a|mps_b`> by contracting from left to right.
fn mps_inner_product(mps_a: &Mps, mps_b: &Mps) -> Complex64 {
    let d = mps_a.phys_dim();
    let mut transfer = DMatrix::from_element(1, 1, Complex64::new(1.0, 0.0));

    for q in 0..mps_a.num_sites() {
        let chi_r_a = mps_a.bond_dim(q + 1);
        let chi_r_b = mps_b.bond_dim(q + 1);
        let t_a = &mps_a.tensors()[q];
        let t_b = &mps_b.tensors()[q];

        let mut new_transfer = DMatrix::zeros(chi_r_a, chi_r_b);
        for sigma in 0..d {
            let block_a = crate::mps::tensor::phys_block(t_a, sigma, chi_r_a);
            let block_b = crate::mps::tensor::phys_block(t_b, sigma, chi_r_b);
            let conj_a_t = block_a.conjugate().transpose();
            let tmp = &conj_a_t * &transfer * &block_b;
            new_transfer += tmp;
        }
        transfer = new_transfer;
    }

    transfer[(0, 0)]
}

/// Find the stabilizer index that `mz_forced` will select for replacement.
///
/// This is the minimum-weight stabilizer that anticommutes with `Z_q`,
/// matching the logic in `SparseStabY::nondeterministic_meas`.
fn find_replaced_stabilizer(tableau: &SparseStabY, q_idx: usize) -> usize {
    let stabs = tableau.stabs();
    let col_x = &stabs.col_x[q_idx];

    let mut best_id = None;
    let mut best_weight = usize::MAX;
    for stab_id in col_x {
        let weight = stabs.row_x[stab_id].len() + stabs.row_z[stab_id].len();
        if weight < best_weight {
            best_weight = weight;
            best_id = Some(stab_id);
            if weight == 1 {
                break;
            }
        }
    }
    best_id.expect("col_x should be non-empty for DestabilizerFlip case")
}

/// Test hook for `pre_reduce_for_measurement`.
pub fn pre_reduce_for_measurement_pub(tableau: &mut SparseStabY, mps: &mut Mps, q_idx: usize) {
    let _ = pre_reduce_for_measurement(tableau, mps, q_idx, true);
}

/// Pre-reduce the stabilizer tableau so that `Z_q` anticommutes with at most
/// one stabilizer. For each other anti-commuting stab:
///   - Tableau: `S[other] *= S[replaced]`, `D[replaced] *= D[other]` (via
///     full Y-convention `multiply_row`, including sign/phase tracking).
///   - MPS (when `apply_mps_compensation=true`): apply virtual-frame
///     `CNOT(c=replaced, t=other)` for CAMPS state preservation. The tableau
///     change transforms the Clifford as `C → C · CNOT` — applying the
///     same CNOT to the MPS (self-inverse) compensates so
///     `C'·MPS_new = C·MPS_old`. Non-adjacent CNOTs use
///     `apply_long_range_two_site_gate`.
///
/// `apply_mps_compensation` is `true` for exact-state callers
/// (`project_forced_z`, `project_forced_z_unnormalized`) used by
/// `prob_bitstring` / `amplitude_iterative`. It is `false` for random
/// measurement (`measure_qubit_stab_mps`): the state representation becomes
/// inconsistent with the tableau after row ops, but measurement
/// statistics stay correct and subsequent measurements remain
/// self-consistent. Skipping compensation avoids SWAP-chain bond growth
/// during measurement-heavy circuits (MAST magic-state injection).
///
/// Proper long-term fix: lazy virtual-frame tracking — accumulate a
/// deferred Clifford V such that effective MPS = V·stored MPS, conjugate
/// Pauli strings by V before applying to stored MPS, flush only when MPS
/// must be read directly.
fn pre_reduce_for_measurement(
    tableau: &mut SparseStabY,
    mps: &mut Mps,
    q_idx: usize,
    apply_mps_compensation: bool,
) -> Vec<usize> {
    let col_x = &tableau.stabs().col_x[q_idx];
    if col_x.len() <= 1 {
        return Vec::new();
    }

    let replaced_idx = find_replaced_stabilizer(tableau, q_idx);
    let anticom: Vec<usize> = tableau.stabs().col_x[q_idx]
        .iter()
        .filter(|&id| id != replaced_idx)
        .collect();

    if apply_mps_compensation {
        // Exact callers need the phase-complete Clifford composition and the
        // matching inverse action on the coefficient MPS.
        let mut modified_sites = Vec::with_capacity(1 + anticom.len());
        modified_sites.push(replaced_idx);
        for other_id in anticom {
            crate::stab_mps::tableau_compose::right_compose_cx(tableau, replaced_idx, other_id);
            apply_cnot_to_mps(mps, replaced_idx, other_id);
            modified_sites.push(other_id);
        }
        modified_sites.sort_unstable();
        modified_sites.dedup();
        modified_sites
    } else {
        // Preserve the established pragmatic measurement path byte-for-byte:
        // it intentionally updates only the structural row frame and does not
        // compensate the MPS. Its tracked drift is outside the exact route.
        let n = tableau.num_qubits();
        let stabs_snapshot = tableau.stabs().clone();
        let destabs_snapshot = tableau.destabs().clone();
        for other_id in anticom {
            crate::stab_mps::tableau_compose::multiply_row(
                tableau.stabs_mut(),
                other_id,
                &stabs_snapshot,
                replaced_idx,
                n,
            );
            crate::stab_mps::tableau_compose::multiply_row(
                tableau.destabs_mut(),
                replaced_idx,
                &destabs_snapshot,
                other_id,
                n,
            );
        }
        Vec::new()
    }
}

fn apply_cnot_to_mps(mps: &mut Mps, control: usize, target: usize) {
    // Optimization: if the control site has no |1⟩_virt amplitude, CNOT is
    // identity on this MPS — skip to avoid bond-dim blowup from SWAP chains.
    // Mirror: if control has no |0⟩_virt amp, CNOT reduces to X on target.
    if mps_site_block_is_zero(mps, control, 1) {
        return;
    }
    if mps_site_block_is_zero(mps, control, 0) {
        // Control is |1⟩ → CNOT unconditionally flips target = X on target.
        let x_gate = DMatrix::from_row_slice(
            2,
            2,
            &[
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        );
        mps.apply_one_site_gate(target, &x_gate)
            .expect("MPS op on valid site");
        return;
    }

    // General case: apply full CNOT.
    let o = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let cnot_c_lo = DMatrix::from_row_slice(
        4,
        4,
        &[one, o, o, o, o, one, o, o, o, o, o, one, o, o, one, o],
    );
    let cnot_c_hi = DMatrix::from_row_slice(
        4,
        4,
        &[one, o, o, o, o, o, o, one, o, o, one, o, o, one, o, o],
    );
    let (q0, q1, gate) = if control < target {
        (control, target, cnot_c_lo)
    } else {
        (target, control, cnot_c_hi)
    };
    if q1 == q0 + 1 {
        mps.apply_two_site_gate(q0, &gate)
            .expect("MPS op on valid site");
    } else {
        mps.apply_long_range_two_site_gate(q0, q1, &gate)
            .expect("MPS op on valid site");
    }
}

/// A deferred Clifford primitive in the virtual-frame queue.
///
/// The queue represents a Clifford `V = ops[last] · ... · ops[0]` where
/// index 0 is the first pushed (earliest applied if flushed). Each primitive
/// has a cheap Heisenberg conjugation rule (bit XOR on flip/sign sets) and
/// a cheap MPS application (single-site for H, diagonal for CZ, SWAP-chain
/// for CNOT).
#[derive(Clone, Copy, Debug)]
pub enum DeferredOp {
    /// CNOT(control, target).
    Cnot(usize, usize),
    /// Hadamard on qubit.
    H(usize),
    /// CZ(a, b) — symmetric.
    Cz(usize, usize),
    /// Pauli Z on qubit. Used for outcome-dependent W basis rotation:
    /// for outcome=1, W includes a `Z_id` factor to flip `Z_id` → -`X_id`.
    Z(usize),
    /// Phase gate adjoint (SZ†) — needed for outcome-dependent W when
    /// flip and sign overlap at id (Y-like Pauli).
    SZdg(usize),
    /// Phase gate SZ — needed for outcome-dependent W when the
    /// decomposition phase `sp` is purely imaginary. Conjugation rule:
    /// SZ†·P·SZ — if X at q, toggle Z at q and multiply phase by -i.
    SZ(usize),
}

fn toggle(v: &mut Vec<usize>, x: usize) {
    if let Some(pos) = v.iter().position(|&y| y == x) {
        v.swap_remove(pos);
    } else {
        v.push(x);
    }
}

/// Conjugate a Pauli `P = X_flip · Z_sign` by `V†` where
/// `V = ops[last] · ops[last-1] · ... · ops[0]`. Updates `flip_sites` and
/// `sign_sites` in place to represent `V† · P · V`. The scalar phase is
/// unchanged (CNOT/H/CZ conjugation preserves phase of the product).
///
/// Heisenberg rules:
/// - CNOT(c, t): `X_c -> X_c · X_t`; `Z_t -> Z_c · Z_t`.
/// - H(q): swap `X_q` and `Z_q` (swap q between flip and sign).
/// - CZ(a, b): `X_a -> X_a · Z_b`; `X_b -> X_b · Z_a`.
///
/// Order: `V† P V = op_0·...·op_last·P·op_last·...·op_0`, so iterate `ops`
/// in REVERSE (innermost conjugation by `op_last` first).
pub fn conjugate_pauli_by_deferred_ops(
    flip_sites: &mut Vec<usize>,
    sign_sites: &mut Vec<usize>,
    phase: &mut Complex64,
    ops: &[DeferredOp],
) {
    for op in ops.iter().rev() {
        match *op {
            DeferredOp::Cnot(c, t) => {
                let has_x_c = flip_sites.contains(&c);
                let has_z_t = sign_sites.contains(&t);
                if has_x_c {
                    toggle(flip_sites, t);
                }
                if has_z_t {
                    toggle(sign_sites, c);
                }
            }
            DeferredOp::H(q) => {
                let has_x = flip_sites.contains(&q);
                let has_z = sign_sites.contains(&q);
                // Swap membership of q between flip and sign.
                if has_x != has_z {
                    if has_x {
                        toggle(flip_sites, q);
                        toggle(sign_sites, q);
                    } else {
                        toggle(sign_sites, q);
                        toggle(flip_sites, q);
                    }
                }
                // If both: Y → -Y (H·Y·H = -Y). Membership stays. Phase flips.
                if has_x && has_z {
                    *phase = -*phase;
                }
            }
            DeferredOp::Cz(a, b) => {
                let has_x_a = flip_sites.contains(&a);
                let has_x_b = flip_sites.contains(&b);
                if has_x_a {
                    toggle(sign_sites, b);
                }
                if has_x_b {
                    toggle(sign_sites, a);
                }
            }
            DeferredOp::Z(q) => {
                // Z·X_q·Z = -X_q. If X present at q (and Z not at q), phase flips.
                // Z·Y_q·Z = -Y_q (Y has X factor). So if X present regardless of Z, phase flips.
                // Z·Z_q·Z = Z_q. No flip if only Z at q.
                if flip_sites.contains(&q) {
                    *phase = -*phase;
                }
            }
            DeferredOp::SZdg(q) => {
                // SZdg conjugation: SZdg†·P·SZdg = SZ·P·SZdg.
                // SZ·X·SZdg = Y = iXZ; SZ·Z·SZdg = Z.
                // If X at q and Z not at q: add q to sign, phase *= i.
                // If X at q and Z at q: SZ·Y·SZdg = i·(SZ·X·SZdg)·(SZ·Z·SZdg) = i·Y·Z = i·(iXZ)·Z = -X.
                //   So XZ → X only (toggle z), phase *= i (aggregate: p · iXZ · Z = ip·X).
                // Matrix sanity-check: SZ = [[1,0],[0,i]], SZdg = [[1,0],[0,-i]],
                //   Y = [[0,-i],[i,0]].
                //   SZ·Y·SZdg = [[1,0],[0,i]]·[[0,-i],[i,0]]·[[1,0],[0,-i]]
                //             = [[0,-i],[-1,0]]·[[1,0],[0,-i]]
                //             = [[0, -1],[-1, 0]] = -X. ✓
                let has_x = flip_sites.contains(&q);
                let has_z = sign_sites.contains(&q);
                if has_x && !has_z {
                    // X only → XZ (add Z), phase *= i.
                    toggle(sign_sites, q);
                    *phase *= Complex64::new(0.0, 1.0);
                } else if has_x && has_z {
                    // XZ → X only (remove Z), phase *= i.
                    toggle(sign_sites, q);
                    *phase *= Complex64::new(0.0, 1.0);
                }
                // Z only or none: unchanged.
            }
            DeferredOp::SZ(q) => {
                // SZ conjugation: SZdg·P·SZ.
                // SZdg·X·SZ = -Y = -i·X·Z; SZdg·Z·SZ = Z; SZdg·Y·SZ = X.
                // X only → X·Z, phase *= -i.
                // X·Z → X only, phase *= -i.
                // Z only or none: unchanged.
                let has_x = flip_sites.contains(&q);
                if has_x {
                    toggle(sign_sites, q);
                    *phase *= Complex64::new(0.0, -1.0);
                }
            }
        }
    }
}

/// Return the MPS-site support of the current conjugated observable `C† Z_q C`.
///
/// The support is derived from the same `decompose_z` result used by the
/// measurement protocol. When lazy measurement has accumulated a virtual
/// Clifford frame, the decomposition is conjugated through that frame before
/// its X and Z sites are combined.
#[must_use]
pub(crate) fn conjugated_z_support(
    tableau: &SparseStabY,
    q_idx: usize,
    deferred: &[DeferredOp],
) -> Vec<usize> {
    let (mut flip_sites, mut sign_sites, mut phase) =
        match decompose_z(tableau.stabs(), tableau.destabs(), q_idx) {
            ZDecomposition::Stabilizer { phase, sign_sites } => (Vec::new(), sign_sites, phase),
            ZDecomposition::DestabilizerFlip {
                flip_sites,
                phase,
                sign_sites,
            } => (flip_sites, sign_sites, phase),
        };
    conjugate_pauli_by_deferred_ops(&mut flip_sites, &mut sign_sites, &mut phase, deferred);
    flip_sites.extend(sign_sites);
    flip_sites.sort_unstable();
    flip_sites.dedup();
    flip_sites
}

/// Backwards-compatible CNOT-only conjugation wrapper. CNOT conjugation
/// doesn't touch phase, so this discards the phase output.
pub fn conjugate_pauli_by_deferred(
    flip_sites: &mut Vec<usize>,
    sign_sites: &mut Vec<usize>,
    cnots: &[(usize, usize)],
) {
    let ops: Vec<DeferredOp> = cnots.iter().map(|&(c, t)| DeferredOp::Cnot(c, t)).collect();
    let mut phase = Complex64::new(1.0, 0.0);
    conjugate_pauli_by_deferred_ops(flip_sites, sign_sites, &mut phase, &ops);
}

/// Apply the deferred op queue `V = ops[last]·...·ops[0]` to `mps` and clear.
///
/// # Panics
///
/// Panics if any MPS gate application fails on a valid site.
pub fn flush_deferred_ops(mps: &mut Mps, ops: &mut Vec<DeferredOp>) {
    let h_gate = DMatrix::from_row_slice(
        2,
        2,
        &[
            Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0),
            Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0),
            Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0),
            Complex64::new(-std::f64::consts::FRAC_1_SQRT_2, 0.0),
        ],
    );
    let cz_diag = [
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(-1.0, 0.0),
    ];
    for op in ops.iter() {
        match *op {
            DeferredOp::Cnot(c, t) => apply_cnot_to_mps(mps, c, t),
            DeferredOp::H(q) => {
                mps.apply_one_site_gate(q, &h_gate)
                    .expect("MPS op on valid site");
            }
            DeferredOp::Cz(a, b) => {
                // CZ is diagonal; use apply_two_site_gate (adjacent) or
                // long-range two-site (non-adjacent). Either preserves bond
                // dim since it's diagonal in the product basis.
                let (q0, q1) = if a < b { (a, b) } else { (b, a) };
                let o = Complex64::new(0.0, 0.0);
                let cz = DMatrix::from_row_slice(
                    4,
                    4,
                    &[
                        cz_diag[0], o, o, o, o, cz_diag[1], o, o, o, o, cz_diag[2], o, o, o, o,
                        cz_diag[3],
                    ],
                );
                if q1 == q0 + 1 {
                    mps.apply_two_site_gate(q0, &cz)
                        .expect("MPS op on valid site");
                } else {
                    mps.apply_long_range_two_site_gate(q0, q1, &cz)
                        .expect("MPS op on valid site");
                }
            }
            DeferredOp::Z(q) => {
                let z_diag = [Complex64::new(1.0, 0.0), Complex64::new(-1.0, 0.0)];
                mps.apply_diagonal_one_site(q, &z_diag)
                    .expect("MPS op on valid site");
            }
            DeferredOp::SZdg(q) => {
                let sdg_diag = [Complex64::new(1.0, 0.0), Complex64::new(0.0, -1.0)];
                mps.apply_diagonal_one_site(q, &sdg_diag)
                    .expect("MPS op on valid site");
            }
            DeferredOp::SZ(q) => {
                let s_diag = [Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)];
                mps.apply_diagonal_one_site(q, &s_diag)
                    .expect("MPS op on valid site");
            }
        }
    }
    ops.clear();
}

/// Backwards-compatible CNOT-only flush wrapper.
pub fn flush_deferred(mps: &mut Mps, cnots: &mut Vec<(usize, usize)>) {
    let mut ops: Vec<DeferredOp> = cnots.iter().map(|&(c, t)| DeferredOp::Cnot(c, t)).collect();
    flush_deferred_ops(mps, &mut ops);
    cnots.clear();
}

/// Returns true if `mps` tensor at `site` has the σ=`block`'s elements all
/// below tolerance (i.e., site has no amplitude at that physical dim value).
fn mps_site_block_is_zero(mps: &Mps, site: usize, block: usize) -> bool {
    let chi_r = mps.bond_dim(site + 1);
    let t = &mps.tensors()[site];
    let start_col = block * chi_r;
    for i in 0..t.nrows() {
        for j in 0..chi_r {
            if t[(i, start_col + j)].norm_sqr() > 1e-20 {
                return false;
            }
        }
    }
    true
}

/// Project qubit `q_idx` onto `outcome` without renormalizing. Returns
/// `false` if the projection is to a zero-probability outcome.
///
/// Unlike `project_forced_z`, the MPS is left UNNORMALIZED: its norm drops
/// by `sqrt(conditional_prob)` after this call. This is what lets the caller
/// recover the complex amplitude from the selected virtual-basis coefficient.
///
/// Used by `StabMps::amplitude_iterative` (Liu-Clark VI.B).
///
/// # Panics
///
/// Panics if any MPS gate application fails on a valid site.
pub fn project_forced_z_unnormalized(
    tableau: &mut SparseStabY,
    mps: &mut Mps,
    q_idx: usize,
    outcome: bool,
) -> bool {
    let carried_norm = mps.norm_squared().sqrt();
    if carried_norm < 1e-30 {
        return false;
    }
    mps.scale(Complex64::new(carried_norm.recip(), 0.0));
    let probability = project_forced_z(tableau, mps, q_idx, outcome);
    if probability < 1e-20 {
        return false;
    }
    mps.scale(Complex64::new(carried_norm * probability.sqrt(), 0.0));
    true
}

/// Recover the unique MPS-frame computational-basis index selected after all
/// physical Z outcomes have been projected.
///
/// A forced projection through the stabilizer branch can leave the selected
/// coefficient at a nonzero virtual index. The final tableau supplies the
/// invertible GF(2) system relating physical Z eigenvalues to virtual
/// stabilizer-sign bits.
///
/// # Panics
///
/// Panics if the outcome length differs from the tableau size or the fully
/// projected tableau does not define an invertible real-valued Z constraint
/// system. Those cases indicate an internal projection inconsistency.
#[must_use]
pub(super) fn projected_mps_basis_index(tableau: &SparseStabY, outcomes: &[bool]) -> Vec<u8> {
    let n = tableau.num_qubits();
    assert_eq!(outcomes.len(), n, "projected outcome length mismatch");

    let mut matrix = vec![vec![false; n + 1]; n];
    for (q, &outcome) in outcomes.iter().enumerate() {
        let ZDecomposition::Stabilizer { phase, sign_sites } =
            decompose_z(tableau.stabs(), tableau.destabs(), q)
        else {
            panic!("projected tableau still has a destabilizer component at qubit {q}");
        };
        assert!(
            phase.im.abs() < 1e-10,
            "projected Z decomposition has non-real phase at qubit {q}"
        );
        for site in sign_sites {
            matrix[q][site] = true;
        }
        matrix[q][n] = outcome ^ (phase.re < 0.0);
    }

    let mut pivot_row = 0;
    let mut pivot_columns = vec![usize::MAX; n];
    for col in 0..n {
        let found = matrix[pivot_row..]
            .iter()
            .position(|row| row[col])
            .map(|offset| pivot_row + offset);
        if let Some(found_row) = found {
            matrix.swap(pivot_row, found_row);
            let pivot = matrix[pivot_row].clone();
            for (row_index, row) in matrix.iter_mut().enumerate() {
                if row_index != pivot_row && row[col] {
                    for (cell, &pivot_cell) in row.iter_mut().zip(pivot.iter()) {
                        *cell ^= pivot_cell;
                    }
                }
            }
            pivot_columns[pivot_row] = col;
            pivot_row += 1;
        }
    }
    assert_eq!(
        pivot_row, n,
        "projected Z constraints are not an invertible MPS-frame basis"
    );

    let mut index = vec![0; n];
    for row in 0..n {
        index[pivot_columns[row]] = u8::from(matrix[row][n]);
    }
    index
}

/// Right-compose the phase-complete measurement basis rotation `W`.
///
/// Keeping an independently predicted tableau lets the exact projection path
/// recover the Pauli gauge omitted by `SparseStabY::nondeterministic_meas`
/// when it structurally XORs destabilizer rows without propagating their
/// phases.
fn right_compose_measurement_basis_rotation(
    tableau: &mut SparseStabY,
    id: usize,
    phase: Complex64,
    sign_sites: &[usize],
    outcome: bool,
) {
    let signed_phase = Complex64::new(if outcome { -1.0 } else { 1.0 }, 0.0) * phase;
    let id_in_sign = sign_sites.contains(&id);

    if signed_phase.im.abs() < 1e-9 {
        assert!(
            !id_in_sign,
            "real measurement phase {signed_phase:?} must not overlap the flip site"
        );
        if signed_phase.re < 0.0 {
            crate::stab_mps::tableau_compose::right_compose_z(tableau, id);
        }
        for &site in sign_sites {
            if site != id {
                crate::stab_mps::tableau_compose::right_compose_cz(tableau, id, site);
            }
        }
    } else {
        assert!(
            id_in_sign,
            "imaginary measurement phase {signed_phase:?} must overlap the flip site"
        );
        assert!(
            signed_phase.re.abs() < 1e-9,
            "measurement phase must be a fourth root of unity: {signed_phase:?}"
        );
        for &site in sign_sites {
            if site != id {
                crate::stab_mps::tableau_compose::right_compose_cz(tableau, id, site);
            }
        }
        if signed_phase.im > 0.0 {
            crate::stab_mps::tableau_compose::right_compose_sz(tableau, id);
        } else {
            crate::stab_mps::tableau_compose::right_compose_szdg(tableau, id);
        }
    }
    crate::stab_mps::tableau_compose::right_compose_h(tableau, id);
}

/// Apply the inverse of the virtual Pauli gauge between two structurally
/// identical Clifford tableaux.
///
/// The forced measurement outcome fixes the stabilizer-row signs, leaving only
/// destabilizer-row sign differences. Each such difference is a right-composed
/// Z on that virtual site, so applying Z to the MPS realizes the inverse gauge.
fn compensate_measurement_pauli_gauge(
    mps: &mut Mps,
    predicted: &SparseStabY,
    measured: &SparseStabY,
) -> Vec<usize> {
    assert_eq!(predicted.num_qubits(), measured.num_qubits());
    let rows_match = |predicted_rows: &[BitSet], measured_rows: &[BitSet]| {
        predicted_rows
            .iter()
            .zip(measured_rows)
            .all(|(predicted_row, measured_row)| predicted_row.iter().eq(measured_row.iter()))
    };
    // Compare semantic set contents rather than BitSet's backing allocation:
    // operations above bit 63 can leave different numbers of trailing zero
    // words in otherwise identical sparse rows.
    assert!(rows_match(
        &predicted.stabs().row_x,
        &measured.stabs().row_x
    ));
    assert!(rows_match(
        &predicted.stabs().row_z,
        &measured.stabs().row_z
    ));
    assert!(rows_match(
        &predicted.destabs().row_x,
        &measured.destabs().row_x
    ));
    assert!(rows_match(
        &predicted.destabs().row_z,
        &measured.destabs().row_z
    ));
    assert!(
        predicted
            .stabs()
            .signs_i
            .iter()
            .eq(measured.stabs().signs_i.iter())
    );
    assert!(
        predicted
            .destabs()
            .signs_i
            .iter()
            .eq(measured.destabs().signs_i.iter())
    );

    let z_diag = [Complex64::new(1.0, 0.0), Complex64::new(-1.0, 0.0)];

    // `SparseStabY::apply_outcome`, called by the nondeterministic
    // `mz_forced` path, forces the replacement stabilizer row's sign to the
    // requested outcome. The predicted basis rotation encodes that same sign,
    // so an X-gauge difference is structurally unreachable here.
    debug_assert!(
        predicted
            .stabs()
            .signs_minus
            .iter()
            .eq(measured.stabs().signs_minus.iter()),
        "apply_outcome must leave no stabilizer-sign (X-gauge) difference"
    );
    let mut modified_sites = Vec::new();
    for site in 0..predicted.num_qubits() {
        let differs = predicted.destabs().signs_minus.contains(site)
            != measured.destabs().signs_minus.contains(site);
        if differs {
            mps.apply_diagonal_one_site(site, &z_diag)
                .expect("MPS op on valid site");
            modified_sites.push(site);
        }
    }
    modified_sites
}

/// Convert a normalized single-flip Pauli eigenstate into the coefficient
/// basis selected by the corresponding tableau measurement.
///
/// For every real or imaginary signed phase handled by `W`, the projected
/// eigenvalue equation relates the flip site's two blocks so that
/// `W^-1 |psi_projected> = sqrt(2) P_0 |psi_projected>`. Selecting the zero
/// block is both exact and local; it avoids materializing W's potentially
/// long-range CZ chain on an MPS that is already known to lie in this
/// eigenspace.
fn collapse_projected_flip_site(mps: &mut Mps, id: usize) {
    let chi_r = mps.bond_dim(id + 1);
    let block_0 = crate::mps::tensor::phys_block(&mps.tensors()[id], 0, chi_r)
        * Complex64::new(std::f64::consts::SQRT_2, 0.0);
    let zero = DMatrix::zeros(mps.tensors()[id].nrows(), chi_r);
    crate::mps::tensor::set_phys_block(&mut mps.tensors_mut()[id], 0, chi_r, &block_0);
    crate::mps::tensor::set_phys_block(&mut mps.tensors_mut()[id], 1, chi_r, &zero);
}

/// Project a real single-site X branch and absorb its measurement-basis H
/// without constructing a rank-doubled MPS sum.
fn project_single_flip_without_sign(
    mps: &mut Mps,
    id: usize,
    signed_phase: Complex64,
    probability: f64,
) {
    assert!(
        signed_phase.im.abs() < 1e-9,
        "single-flip branch without Z support must have real phase"
    );
    let chi_r = mps.bond_dim(id + 1);
    let block_0 = crate::mps::tensor::phys_block(&mps.tensors()[id], 0, chi_r);
    let block_1 = crate::mps::tensor::phys_block(&mps.tensors()[id], 1, chi_r);
    let denominator = Complex64::new((2.0 * probability).max(1e-20).sqrt(), 0.0);
    let projected = (block_0 + block_1 * signed_phase) / denominator;
    let zero = DMatrix::zeros(mps.tensors()[id].nrows(), chi_r);
    crate::mps::tensor::set_phys_block(&mut mps.tensors_mut()[id], 0, chi_r, &projected);
    crate::mps::tensor::set_phys_block(&mut mps.tensors_mut()[id], 1, chi_r, &zero);
}

/// Remove the direct-sum projector's structurally oversized virtual bonds
/// without discarding any component of the state.
///
/// A right-canonical QR sweep is an exact factorization and bounds every bond
/// by the Hilbert-space dimension to its right. This avoids applying the
/// cutoff-based SVD compressor to the cancellation-heavy projected sum. Only
/// a bond that still exceeds the caller's explicit approximation cap is sent
/// through the configured compressor.
fn reduce_exact_projection_bonds(mps: &mut Mps) {
    mps.right_canonicalize();
    if mps.max_bond_dim() > mps.config().max_bond_dim {
        mps.compress();
    }
}

/// Project qubit `q_idx` onto a forced Z-basis outcome and return its
/// probability together with the affected virtual coefficient-MPS sites.
///
/// Mirrors `measure_qubit_stab_mps` but deterministic: no RNG, the outcome is
/// supplied by the caller. Useful for bitstring-probability computation
/// (Liu-Clark 2412.17209 Algorithm 3 / VI.A).
///
/// # Panics
///
/// Panics if any MPS gate application fails on a valid site.
pub(super) fn project_forced_z_with_update(
    tableau: &mut SparseStabY,
    mps: &mut Mps,
    q_idx: usize,
    outcome: bool,
) -> ForcedProjectionResult {
    if is_mps_trivial(mps) {
        // A trivial coefficient MPS represents a pure stabilizer state. First
        // absorb a possible nonzero virtual basis word into the tableau; only
        // then can its forced update supply the exact probability and state.
        let modified_sites = canonicalize_trivial_mps_basis(tableau, mps);
        let decomp = decompose_z(tableau.stabs(), tableau.destabs(), q_idx);
        let probability = match decomp {
            ZDecomposition::Stabilizer { phase, .. } => {
                if (phase.re < 0.0) == outcome {
                    1.0
                } else {
                    0.0
                }
            }
            ZDecomposition::DestabilizerFlip { .. } => 0.5,
        };
        if probability > 0.0 {
            tableau.mz_forced(q_idx, outcome);
        }
        return ForcedProjectionResult {
            probability,
            update: ProjectionUpdate {
                collapsed_site: None,
                modified_sites,
            },
        };
    }

    // Reduce to one virtual X site while compensating every generator-basis
    // CNOT on the MPS. This preserves C·MPS exactly.
    let mut modified_sites = pre_reduce_for_measurement(tableau, mps, q_idx, true);
    let norm_squared = mps.norm_squared();
    assert!(norm_squared > 1e-20, "cannot project a zero-norm MPS");
    let expectation = (z_expectation_value(tableau, mps, q_idx).re / norm_squared).clamp(-1.0, 1.0);
    let decomposition = decompose_z(tableau.stabs(), tableau.destabs(), q_idx);

    match decomposition {
        ZDecomposition::Stabilizer { phase, sign_sites } => {
            let probability = forced_outcome_probability(expectation, outcome);
            if probability < 1e-20 {
                return ForcedProjectionResult {
                    probability: 0.0,
                    update: ProjectionUpdate {
                        collapsed_site: None,
                        modified_sites,
                    },
                };
            }
            apply_pauli_projection(
                mps,
                &[],
                &sign_sites,
                phase,
                if outcome { -1.0 } else { 1.0 },
                probability,
            );
            // The physical observable was already in the stabilizer span, so
            // mz_forced performs no Clifford-basis change. The projected
            // stabilizer-sign superposition remains encoded in the MPS.
            if !sign_sites.is_empty() {
                reduce_exact_projection_bonds(mps);
            }
            mps.normalize();
            modified_sites.extend(sign_sites);
            modified_sites.sort_unstable();
            modified_sites.dedup();
            ForcedProjectionResult {
                probability,
                update: ProjectionUpdate {
                    collapsed_site: None,
                    modified_sites,
                },
            }
        }
        ZDecomposition::DestabilizerFlip {
            flip_sites,
            phase,
            sign_sites,
        } => {
            debug_assert_eq!(
                flip_sites.len(),
                1,
                "forced projection must have one flip after pre-reduction"
            );
            let probability = forced_outcome_probability(expectation, outcome);
            if probability < 1e-20 {
                return ForcedProjectionResult {
                    probability: 0.0,
                    update: ProjectionUpdate {
                        collapsed_site: None,
                        modified_sites,
                    },
                };
            }
            let sign_f = if outcome { -1.0 } else { 1.0 };
            let is_local_projection = sign_sites.is_empty();
            if is_local_projection {
                project_single_flip_without_sign(
                    mps,
                    flip_sites[0],
                    Complex64::new(sign_f, 0.0) * phase,
                    probability,
                );
            } else {
                apply_pauli_projection(mps, &flip_sites, &sign_sites, phase, sign_f, probability);
                // On the projected Pauli eigenspace, W^-1 is exactly sqrt(2)
                // times selection of the flip site's zero block.
                collapse_projected_flip_site(mps, flip_sites[0]);
            }

            assert!(
                tableau.tracks_destab_signs(),
                "exact forced projection requires destabilizer-sign tracking"
            );
            let mut predicted_tableau = tableau.clone();
            right_compose_measurement_basis_rotation(
                &mut predicted_tableau,
                flip_sites[0],
                phase,
                &sign_sites,
                outcome,
            );

            let result = tableau.mz_forced(q_idx, outcome);
            debug_assert_eq!(result.outcome, outcome);
            let gauge_sites = compensate_measurement_pauli_gauge(mps, &predicted_tableau, tableau);
            if !is_local_projection {
                reduce_exact_projection_bonds(mps);
            }
            mps.normalize();
            modified_sites.extend(flip_sites.iter().copied());
            modified_sites.extend(sign_sites);
            modified_sites.extend(gauge_sites);
            modified_sites.sort_unstable();
            modified_sites.dedup();
            ForcedProjectionResult {
                probability,
                update: ProjectionUpdate {
                    collapsed_site: Some(flip_sites[0]),
                    modified_sites,
                },
            }
        }
    }
}

/// Project qubit `q_idx` onto a forced Z-basis outcome and return the
/// probability of that outcome given the current state.
///
/// # Panics
///
/// Panics if an MPS operation fails on a valid site.
pub fn project_forced_z(
    tableau: &mut SparseStabY,
    mps: &mut Mps,
    q_idx: usize,
    outcome: bool,
) -> f64 {
    project_forced_z_with_update(tableau, mps, q_idx, outcome).probability
}

/// Measure qubit `q_idx` in the Z basis using the STN protocol.
///
/// Uses the tableau for structure (stabilizer/destabilizer decomposition)
/// and the MPS for probability computation and projection.
/// Lazy-compensation measurement (V2): accumulates `pre_reduce` CNOTs AND
/// the post-projection `W⁻¹` (single-qubit H + diagonal CZs) into a
/// `DeferredOp` queue. Uses `V†`-conjugated Pauli for projection. State
/// invariant: `effective = C_tableau · V_deferred · stored_mps`.
///
/// Derivation:
/// - After `pre_reduce` row ops, tableau's C -> C*A (A = product of CNOTs).
///   Push each CNOT to V: `V_new = A * V_old` (left-multiply).
/// - After projection `(I + sp*P)/2` in effective frame, stored MPS is
///   projected via conjugated `Q = V^dag * P * V`: `stored' = (I+sp*Q)/2*stored`.
/// - `mz_forced` updates tableau: C*A -> C*A*W where `W*Z_id*W^dag = P`.
///   To preserve `effective = C_tableau * V * stored`, absorb `W^-1` into
///   V: `V_new = W^-1 * V` (append `W^-1`'s primitives at end of queue).
/// - For single-flip `P = X_id * Z_{sign}`, `W = CZ(id, s_1)*...*CZ(id, s_k)*H_id`
///   and `W^-1 = H_id * CZ(id, s_1)*...*CZ(id, s_k)`. All cheap primitives
///   (single-site H, diagonal CZ).
///
/// # Panics
///
/// Panics if the tableau measurement iterator is empty (should not happen).
pub(super) fn measure_qubit_stab_mps_lazy_with_update(
    tableau: &mut SparseStabY,
    mps: &mut Mps,
    rng: &mut PecosRng,
    q_idx: usize,
    deferred: &mut Vec<DeferredOp>,
) -> LiveMeasurementResult {
    if is_mps_trivial(mps) {
        let measurement = tableau
            .mz(&[pecos_core::QubitId(q_idx)])
            .into_iter()
            .next()
            .expect("MPS op on valid site");
        return LiveMeasurementResult {
            measurement,
            update: ProjectionUpdate::default(),
        };
    }

    // Push pre_reduce CNOTs to deferred instead of applying eagerly.
    let mut modified_sites = Vec::new();
    {
        let col_x = &tableau.stabs().col_x[q_idx];
        if col_x.len() > 1 {
            let replaced_idx = find_replaced_stabilizer(tableau, q_idx);
            let n = tableau.num_qubits();
            let anticom: Vec<usize> = tableau.stabs().col_x[q_idx]
                .iter()
                .filter(|&id| id != replaced_idx)
                .collect();
            let stabs_snapshot = tableau.stabs().clone();
            let destabs_snapshot = tableau.destabs().clone();
            modified_sites.push(replaced_idx);
            for other_id in anticom {
                crate::stab_mps::tableau_compose::multiply_row(
                    tableau.stabs_mut(),
                    other_id,
                    &stabs_snapshot,
                    replaced_idx,
                    n,
                );
                crate::stab_mps::tableau_compose::multiply_row(
                    tableau.destabs_mut(),
                    replaced_idx,
                    &destabs_snapshot,
                    other_id,
                    n,
                );
                deferred.push(DeferredOp::Cnot(replaced_idx, other_id));
                modified_sites.push(other_id);
            }
        }
    }
    modified_sites.sort_unstable();
    modified_sites.dedup();

    let decomp = decompose_z(tableau.stabs(), tableau.destabs(), q_idx);
    match decomp {
        ZDecomposition::Stabilizer { phase, sign_sites } => {
            let mut flip_conj: Vec<usize> = Vec::new();
            let mut sign_conj: Vec<usize> = sign_sites;
            let mut phase_conj = phase;
            conjugate_pauli_by_deferred_ops(
                &mut flip_conj,
                &mut sign_conj,
                &mut phase_conj,
                deferred,
            );

            let ev = pauli_expectation(mps, &flip_conj, &sign_conj, phase_conj).re;

            if sign_conj.is_empty() && flip_conj.is_empty() {
                let outcome = phase_conj.re < 0.0;
                tableau.mz_forced(q_idx, outcome);
                return LiveMeasurementResult {
                    measurement: MeasurementResult {
                        outcome,
                        is_deterministic: true,
                    },
                    update: ProjectionUpdate {
                        collapsed_site: None,
                        modified_sites,
                    },
                };
            }
            let prob_plus = f64::midpoint(1.0, ev).clamp(0.0, 1.0);
            let is_determ = (ev.abs() - 1.0).abs() < 1e-6;
            let outcome = if is_determ {
                ev < 0.0
            } else {
                rng.random_bool(1.0 - prob_plus)
            };
            let sign_f = if outcome { -1.0 } else { 1.0 };
            let prob = if outcome { 1.0 - prob_plus } else { prob_plus };
            apply_pauli_projection(mps, &flip_conj, &sign_conj, phase_conj, sign_f, prob);
            mps.compress();
            modified_sites.extend(flip_conj);
            modified_sites.extend(sign_conj);
            modified_sites.sort_unstable();
            modified_sites.dedup();
            LiveMeasurementResult {
                measurement: MeasurementResult {
                    outcome,
                    is_deterministic: is_determ,
                },
                update: ProjectionUpdate {
                    collapsed_site: None,
                    modified_sites,
                },
            }
        }
        ZDecomposition::DestabilizerFlip {
            flip_sites,
            phase,
            sign_sites,
        } => {
            // Pre_reduce ensures flip_sites.len() == 1. Let id = flip_sites[0].
            // Mz_forced will transform tableau as C → C·W where
            //   W · Z_id · W† = X_id · Z_{sign_sites}
            //   (the decomposition's Pauli content, phase absorbed in sp).
            // Valid W: CZ(id, s_1)·...·CZ(id, s_k) · H_id.
            // To preserve invariant, V_new = W⁻¹ · V_old. W⁻¹ = H_id · CZ_chain
            // (reversed product with self-adjoint primitives).
            let id = if flip_sites.len() == 1 {
                flip_sites[0]
            } else {
                // Shouldn't happen after pre_reduce; use first as fallback.
                debug_assert!(
                    !flip_sites.is_empty(),
                    "lazy measure: flip_sites empty in DestabilizerFlip"
                );
                flip_sites[0]
            };

            // Conjugate the PRE-basis-rotation Pauli by existing V†.
            let mut flip_conj: Vec<usize> = flip_sites.clone();
            let mut sign_conj: Vec<usize> = sign_sites.clone();
            let mut phase_conj = phase;
            conjugate_pauli_by_deferred_ops(
                &mut flip_conj,
                &mut sign_conj,
                &mut phase_conj,
                deferred,
            );

            let ev = pauli_expectation(mps, &flip_conj, &sign_conj, phase_conj).re;
            let prob_plus = f64::midpoint(1.0, ev).clamp(0.0, 1.0);
            let outcome = rng.random_bool(1.0 - prob_plus);
            let sign_f = if outcome { -1.0 } else { 1.0 };
            let prob = if outcome { 1.0 - prob_plus } else { prob_plus };

            // Project stored MPS via conjugated Pauli.
            apply_pauli_projection(mps, &flip_conj, &sign_conj, phase_conj, sign_f, prob);
            mps.compress();
            // Absorb W⁻¹ into V. W satisfies:
            //   W · Z_id · W† = sp · X_flip · Z_sign  (MPS-frame post-measurement Pauli)
            // where `sp = sign_f · phase_conj` (sign_f = -1 if outcome else +1).
            // sp is one of {+1, -1, +i, -i}. Hermiticity of Z_id forces a
            // dichotomy on `X_flip · Z_sign` (single flip = {id}):
            //   - id ∉ sign: X_id · Z_sign is Hermitian, sp must be real.
            //   - id ∈ sign: X_id · Z_id · Z_rest = -i·Y_id·Z_rest is
            //     anti-Hermitian, sp must be imaginary.
            //
            // Basis-rotation constructions (each giving W·Z_id·W† = target):
            //   Real sp, id ∉ sign:
            //     sp = +1: W = [CZ(id, s) for s∈sign] · H_id
            //     sp = -1: W = Z_id · [CZ(id, s) for s∈sign] · H_id
            //   Imaginary sp, id ∈ sign:
            //     sp = +i: W = [CZ(id, s) for s∈sign\id] · SZ_id · H_id
            //     sp = -i: W = [CZ(id, s) for s∈sign\id] · SZdg_id · H_id
            //
            // W⁻¹ reverses the product and adjoints each primitive. Deferred
            // queue push order is application order (first-pushed applied
            // first), which corresponds to rightmost-in-product. So push
            // W⁻¹'s primitives right-to-left:
            //
            // W is determined by mz_forced's action on the CURRENT tableau
            // (post-pre_reduce). Use the original decomposition `phase`, not
            // the V-conjugated `phase_conj` — V-conjugation is for MPS
            // operations only; the tableau sees the original decomposition.
            let sp = Complex64::new(sign_f, 0.0) * phase;
            let id_in_sign = sign_sites.contains(&id);
            if sp.im.abs() < 1e-9 {
                // Real sp branch. id must not be in sign.
                debug_assert!(
                    !id_in_sign,
                    "lazy measure: real sp={sp:?} but id in sign (expected imaginary)"
                );
                if sp.re < 0.0 {
                    deferred.push(DeferredOp::Z(id));
                }
                for &s in &sign_sites {
                    if s != id {
                        deferred.push(DeferredOp::Cz(id, s));
                    }
                }
            } else {
                // Imaginary sp branch. id must be in sign.
                debug_assert!(
                    id_in_sign,
                    "lazy measure: imaginary sp={sp:?} but id not in sign (expected real)"
                );
                debug_assert!(
                    sp.re.abs() < 1e-9,
                    "lazy measure: sp={sp:?} not pure imaginary"
                );
                for &s in &sign_sites {
                    if s != id {
                        deferred.push(DeferredOp::Cz(id, s));
                    }
                }
                // W inner rotation: SZ for sp=+i, SZdg for sp=-i.
                // W⁻¹'s corresponding primitive: SZdg for sp=+i, SZ for sp=-i.
                if sp.im > 0.0 {
                    deferred.push(DeferredOp::SZdg(id));
                } else {
                    deferred.push(DeferredOp::SZ(id));
                }
            }
            deferred.push(DeferredOp::H(id));

            tableau.mz_forced(q_idx, outcome);
            modified_sites.extend(flip_conj);
            modified_sites.extend(sign_conj);
            modified_sites.push(id);
            modified_sites.sort_unstable();
            modified_sites.dedup();
            LiveMeasurementResult {
                measurement: MeasurementResult {
                    outcome,
                    is_deterministic: false,
                },
                update: ProjectionUpdate {
                    // The queued W⁻¹ makes `id` equal |0> only in the
                    // effective `V * stored_mps` frame. Disentangling flags
                    // describe the stored tensors, where the conjugated
                    // projection modified `id`, so no stored-frame collapse
                    // proof may be installed here.
                    collapsed_site: None,
                    modified_sites,
                },
            }
        }
    }
}

/// Measure `q_idx` while accumulating virtual Clifford compensation in
/// `deferred`.
///
/// # Panics
///
/// Panics if an MPS operation fails on a valid site.
pub fn measure_qubit_stab_mps_lazy(
    tableau: &mut SparseStabY,
    mps: &mut Mps,
    rng: &mut PecosRng,
    q_idx: usize,
    deferred: &mut Vec<DeferredOp>,
) -> MeasurementResult {
    measure_qubit_stab_mps_lazy_with_update(tableau, mps, rng, q_idx, deferred).measurement
}

/// Apply projection `(I + sign_f · phase · X_flip · Z_sign) / 2` to `mps`,
/// normalized by `1/√prob`. Uses MPS addition; no site-collapse step
/// (caller is responsible for collapse if exact state needed).
fn apply_pauli_projection(
    mps: &mut Mps,
    flip_sites: &[usize],
    sign_sites: &[usize],
    phase: Complex64,
    sign_f: f64,
    prob: f64,
) {
    let x_gate = DMatrix::from_row_slice(
        2,
        2,
        &[
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
    );
    let z_diag = [Complex64::new(1.0, 0.0), Complex64::new(-1.0, 0.0)];
    let denom = Complex64::new(2.0 * prob.max(1e-20).sqrt(), 0.0);
    if flip_sites.is_empty() && sign_sites.is_empty() {
        mps.scale(Complex64::new(1.0, 0.0) + Complex64::new(sign_f, 0.0) * phase);
        mps.scale(Complex64::new(1.0, 0.0) / denom);
        return;
    }
    let mut mps_z = mps.clone();
    for &k in sign_sites {
        mps_z
            .apply_diagonal_one_site(k, &z_diag)
            .expect("MPS op on valid site");
    }
    for &j in flip_sites {
        mps_z
            .apply_one_site_gate(j, &x_gate)
            .expect("MPS op on valid site");
    }
    mps_z.scale(Complex64::new(sign_f, 0.0) * phase / denom);
    mps.scale(Complex64::new(1.0, 0.0) / denom);
    *mps = mps.add(&mps_z);
}

/// Measure qubit `q_idx` in the Z basis using the STN protocol.
///
/// # Panics
///
/// Panics if the tableau measurement iterator is empty (should not happen).
pub(super) fn measure_qubit_stab_mps_with_update(
    tableau: &mut SparseStabY,
    mps: &mut Mps,
    rng: &mut PecosRng,
    q_idx: usize,
) -> LiveMeasurementResult {
    // Trivial MPS: delegate to tableau
    if is_mps_trivial(mps) {
        let measurement = tableau
            .mz(&[pecos_core::QubitId(q_idx)])
            .into_iter()
            .next()
            .expect("MPS op on valid site");
        return LiveMeasurementResult {
            measurement,
            update: ProjectionUpdate::default(),
        };
    }

    // Pre-reduce the tableau so that Z_q has at most one anticommuting stabilizer.
    // This avoids the problematic multi-flip projection path.
    //
    // MPS compensation is intentionally SKIPPED here (`false`). Random
    // measurement doesn't require exact (tableau, mps) consistency — the
    // sampled outcome statistics and subsequent measurement stats remain
    // self-consistent (same row ops happen in both forward and reverse
    // comparisons). Compensation would trigger O(N) long-range CNOTs per
    // measurement (SWAP chain -> exponential bond growth in MAST's
    // measurement-heavy workload). Exact-state paths
    // (`project_forced_z`, `project_forced_z_unnormalized`) pass `true`.
    let _ = pre_reduce_for_measurement(tableau, mps, q_idx, false);

    // Compute the expectation value <Z_q>
    let ev = z_expectation_value(tableau, mps, q_idx).re;

    let decomp = decompose_z(tableau.stabs(), tableau.destabs(), q_idx);

    match decomp {
        ZDecomposition::Stabilizer { phase, sign_sites } => {
            // Z_q is in the stabilizer group: measurement is deterministic.
            if sign_sites.is_empty() {
                let outcome = phase.re < 0.0;
                tableau.mz_forced(q_idx, outcome);
                return LiveMeasurementResult {
                    measurement: MeasurementResult {
                        outcome,
                        is_deterministic: true,
                    },
                    update: ProjectionUpdate::default(),
                };
            }
            let prob_plus = f64::midpoint(1.0, ev).clamp(0.0, 1.0);

            // Check if measurement is deterministic (ev ≈ ±1)
            let is_determ = (ev.abs() - 1.0).abs() < 1e-6;
            let outcome = if is_determ {
                ev < 0.0
            } else {
                rng.random_bool(1.0 - prob_plus)
            };

            let sign_f = if outcome { -1.0 } else { 1.0 };
            let prob = if outcome { 1.0 - prob_plus } else { prob_plus };

            let z_diag = [Complex64::new(1.0, 0.0), Complex64::new(-1.0, 0.0)];
            let mut mps_z = mps.clone();
            for &k in &sign_sites {
                mps_z
                    .apply_diagonal_one_site(k, &z_diag)
                    .expect("MPS op on valid site");
            }
            mps_z.scale(
                Complex64::new(sign_f, 0.0) * phase
                    / Complex64::new(2.0 * prob.max(1e-20).sqrt(), 0.0),
            );
            mps.scale(Complex64::new(1.0 / (2.0 * prob.max(1e-20).sqrt()), 0.0));
            *mps = mps.add(&mps_z);
            mps.compress();

            tableau.mz_forced(q_idx, outcome);
            let mut modified_sites = sign_sites;
            modified_sites.sort_unstable();
            modified_sites.dedup();
            LiveMeasurementResult {
                measurement: MeasurementResult {
                    outcome,
                    is_deterministic: is_determ,
                },
                update: ProjectionUpdate {
                    collapsed_site: None,
                    modified_sites,
                },
            }
        }

        ZDecomposition::DestabilizerFlip {
            flip_sites,
            phase,
            sign_sites,
        } => {
            let prob_plus = f64::midpoint(1.0, ev).clamp(0.0, 1.0);
            let outcome = rng.random_bool(1.0 - prob_plus);
            let prob = if outcome { 1.0 - prob_plus } else { prob_plus };

            if flip_sites.len() == 1 && sign_sites.is_empty() {
                // Single flip at site k. Project to eigenstate of phase*X_k.
                // After mz_forced: the projected state always goes to σ=0,
                // because mz_forced encodes the outcome in the stabilizer sign.
                let k = flip_sites[0];
                let chi_r = mps.bond_dim(k + 1);
                let sign_f = if outcome { -1.0 } else { 1.0 };
                let sp = Complex64::new(sign_f, 0.0) * phase;

                let block_0 = crate::mps::tensor::phys_block(&mps.tensors()[k], 0, chi_r);
                let block_1 = crate::mps::tensor::phys_block(&mps.tensors()[k], 1, chi_r);
                let projected = (&block_0 + &block_1 * sp)
                    / Complex64::new((2.0 * prob).max(1e-20).sqrt(), 0.0);

                let zero = DMatrix::zeros(mps.tensors()[k].nrows(), chi_r);
                crate::mps::tensor::set_phys_block(&mut mps.tensors_mut()[k], 0, chi_r, &projected);
                crate::mps::tensor::set_phys_block(&mut mps.tensors_mut()[k], 1, chi_r, &zero);
                mps.normalize();
            } else {
                // Multi-site case with sign_sites: use MPS addition then collapse flip site.
                let sign_f = if outcome { -1.0 } else { 1.0 };
                let x_gate = DMatrix::from_row_slice(
                    2,
                    2,
                    &[
                        Complex64::new(0.0, 0.0),
                        Complex64::new(1.0, 0.0),
                        Complex64::new(1.0, 0.0),
                        Complex64::new(0.0, 0.0),
                    ],
                );
                let z_diag = [Complex64::new(1.0, 0.0), Complex64::new(-1.0, 0.0)];

                let mut mps_z = mps.clone();
                // Apply Z first, then X (order must match z_expectation_value).
                for &k in &sign_sites {
                    mps_z
                        .apply_diagonal_one_site(k, &z_diag)
                        .expect("MPS op on valid site");
                }
                for &j in &flip_sites {
                    mps_z
                        .apply_one_site_gate(j, &x_gate)
                        .expect("MPS op on valid site");
                }
                mps_z.scale(
                    Complex64::new(sign_f, 0.0) * phase
                        / Complex64::new(2.0 * prob.max(1e-20).sqrt(), 0.0),
                );
                mps.scale(Complex64::new(1.0 / (2.0 * prob.max(1e-20).sqrt()), 0.0));
                *mps = mps.add(&mps_z);
                mps.compress();

                // Collapse the flip site to σ=0. After the MPS addition projector,
                // block_1 = sp * block_0 (eigenstate condition). After mz_forced,
                // σ=0 is the stabilizer eigenstate. Just zero out σ=1 and renormalize.
                if flip_sites.len() == 1 {
                    let k = flip_sites[0];
                    let chi_r = mps.bond_dim(k + 1);
                    let zero = DMatrix::zeros(mps.tensors()[k].nrows(), chi_r);
                    crate::mps::tensor::set_phys_block(&mut mps.tensors_mut()[k], 1, chi_r, &zero);
                }

                mps.normalize();
            }

            tableau.mz_forced(q_idx, outcome);
            let collapsed_site = (flip_sites.len() == 1).then_some(flip_sites[0]);
            let mut modified_sites = flip_sites;
            modified_sites.extend(sign_sites);
            modified_sites.sort_unstable();
            modified_sites.dedup();
            LiveMeasurementResult {
                measurement: MeasurementResult {
                    outcome,
                    is_deterministic: false,
                },
                update: ProjectionUpdate {
                    collapsed_site,
                    modified_sites,
                },
            }
        }
    }
}

/// Measure qubit `q_idx` in the Z basis using the eager STN protocol.
///
/// # Panics
///
/// Panics if an MPS operation fails on a valid site.
pub fn measure_qubit_stab_mps(
    tableau: &mut SparseStabY,
    mps: &mut Mps,
    rng: &mut PecosRng,
    q_idx: usize,
) -> MeasurementResult {
    measure_qubit_stab_mps_with_update(tableau, mps, rng, q_idx).measurement
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mps::MpsConfig;

    fn sort_dedup(v: &mut Vec<usize>) {
        v.sort_unstable();
        v.dedup();
    }

    #[test]
    fn conjugate_single_cnot_x_on_control() {
        // V = CNOT(0,1). V†·X_0·V = X_0·X_1.
        let mut flip = vec![0];
        let mut sign: Vec<usize> = vec![];
        conjugate_pauli_by_deferred(&mut flip, &mut sign, &[(0, 1)]);
        sort_dedup(&mut flip);
        assert_eq!(flip, vec![0, 1]);
        assert!(sign.is_empty());
    }

    #[test]
    fn conjugate_single_cnot_z_on_target() {
        // V = CNOT(0,1). V†·Z_1·V = Z_0·Z_1.
        let mut flip: Vec<usize> = vec![];
        let mut sign = vec![1];
        conjugate_pauli_by_deferred(&mut flip, &mut sign, &[(0, 1)]);
        sort_dedup(&mut sign);
        assert!(flip.is_empty());
        assert_eq!(sign, vec![0, 1]);
    }

    #[test]
    fn conjugate_cnot_x_on_target_unchanged() {
        // V = CNOT(0,1). V†·X_1·V = X_1 (target X unchanged).
        let mut flip = vec![1];
        let mut sign: Vec<usize> = vec![];
        conjugate_pauli_by_deferred(&mut flip, &mut sign, &[(0, 1)]);
        assert_eq!(flip, vec![1]);
        assert!(sign.is_empty());
    }

    #[test]
    fn conjugate_cnot_z_on_control_unchanged() {
        // V = CNOT(0,1). V†·Z_0·V = Z_0 (control Z unchanged).
        let mut flip: Vec<usize> = vec![];
        let mut sign = vec![0];
        conjugate_pauli_by_deferred(&mut flip, &mut sign, &[(0, 1)]);
        assert!(flip.is_empty());
        assert_eq!(sign, vec![0]);
    }

    #[test]
    fn conjugate_two_cnots_cancels() {
        // V = CNOT(0,1)·CNOT(0,1) = I. V†·X_0·V = X_0.
        let mut flip = vec![0];
        let mut sign: Vec<usize> = vec![];
        conjugate_pauli_by_deferred(&mut flip, &mut sign, &[(0, 1), (0, 1)]);
        sort_dedup(&mut flip);
        assert_eq!(flip, vec![0]);
    }

    #[test]
    fn conjugate_cnot_chain_fanout() {
        // V = CNOT(0,3)·CNOT(0,2)·CNOT(0,1) — fan-out from qubit 0.
        // V†·X_0·V = ? Chain conjugation: innermost first.
        // Step 1 (CNOT(0,3)): X_0 -> X_0·X_3. flip={0,3}.
        // Step 2 (CNOT(0,2)): X_0 -> X_0·X_2. flip={0,2,3}.
        // Step 3 (CNOT(0,1)): X_0 -> X_0·X_1. flip={0,1,2,3}.
        let mut flip = vec![0];
        let mut sign: Vec<usize> = vec![];
        // Pushed in chronological order: first pushed = CNOT(0,1).
        // V = last·...·first = CNOT(0,3)·CNOT(0,2)·CNOT(0,1).
        conjugate_pauli_by_deferred(&mut flip, &mut sign, &[(0, 1), (0, 2), (0, 3)]);
        sort_dedup(&mut flip);
        assert_eq!(flip, vec![0, 1, 2, 3]);
        assert!(sign.is_empty());
    }

    #[test]
    fn flush_deferred_matches_eager() {
        // Two MPS: one where we apply CNOTs eagerly, one where we flush
        // the queue at the end. Final states should agree.
        let config = MpsConfig::default();
        let num_qubits = 4;

        let mut mps_eager = Mps::new(num_qubits, config.clone());
        // Put into a non-trivial state first: apply H on site 0 via
        // single-site gate (to avoid bond-dim 1 trivial case).
        let h = DMatrix::from_row_slice(
            2,
            2,
            &[
                Complex64::new(0.5_f64.sqrt(), 0.0),
                Complex64::new(0.5_f64.sqrt(), 0.0),
                Complex64::new(0.5_f64.sqrt(), 0.0),
                Complex64::new(-0.5_f64.sqrt(), 0.0),
            ],
        );
        mps_eager
            .apply_one_site_gate(0, &h)
            .expect("MPS op on valid site");
        let mut mps_lazy = mps_eager.clone();

        // Apply CNOT(0,1), CNOT(0,2), CNOT(1,3) eagerly.
        let cnots = vec![(0usize, 1usize), (0, 2), (1, 3)];
        for &(c, t) in &cnots {
            apply_cnot_to_mps(&mut mps_eager, c, t);
        }

        // Flush the same CNOTs.
        let mut queue = cnots;
        flush_deferred(&mut mps_lazy, &mut queue);
        assert!(queue.is_empty());

        // Compare state vectors.
        let sv_e = mps_eager.state_vector();
        let sv_l = mps_lazy.state_vector();
        assert_eq!(sv_e.len(), sv_l.len());
        for (a, b) in sv_e.iter().zip(sv_l.iter()) {
            assert!((a - b).norm() < 1e-10, "eager vs lazy differ: {a} vs {b}");
        }
    }
}
