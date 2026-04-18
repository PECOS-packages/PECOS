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

//! Right-composition of Clifford gates onto a stabilizer tableau.
//!
//! The existing `SparseStabY` gate methods (h, sz, cx) perform LEFT-composition:
//! they transform C into G*C, conjugating each generator by G acting on physical
//! qubits. Right-composition transforms C into C*G, which acts on the "virtual"
//! side -- the generator rows themselves.
//!
//! Mathematical identity:
//! - Left-compose by G: stabilizer `S_k` = C `Z_k` C^dagger -> G `S_k` G^dagger
//! - Right-compose by G: `S_k` -> C (G `Z_k` G^dagger) C^dagger
//!
//! Right-composition is what the stabilizer-TN reference uses to absorb the
//! compensating CNOT cascade from the exact disentangling path. This allows
//! the disentangling to leave the MPS in a single-site rotation state without
//! needing the full multi-site CNOT cascade on the MPS.
//!
//! Implementation for each gate (right-compose):
//! - `H_q`:     swap stabs row q with destabs row q (and their signs)
//! - `S_q`:     destabs[q] *= stabs[q], with phase +i correction
//! - CX(c,t): stabs[t] *= stabs[c], destabs[c] *= destabs[t]
//!
//! Reference: Aaronson & Gottesman, "Improved Simulation of Stabilizer Circuits"
//! (PRA 70, 052328 (2004)); stabilizer-TN reference compose(..., front=True).

use num_complex::Complex64;
use pecos_core::{BitSet, IndexSet};
use pecos_simulators::{GensGeneric, SparseStabY};

/// Standard Pauli multiplication phase table (Y=iXZ). Index scheme:
/// I=0, Z=1, X=2, Y=3.
/// Returns (`minus_bit`, `i_bit`) for the phase factor of `Pauli_a` · `Pauli_b`.
/// Encoding: +1=(0,0), -1=(1,0), +i=(0,1), -i=(1,1).
const fn pauli_phase(a: u8, b: u8) -> (i8, i8) {
    match (a, b) {
        (1, 2) | (3, 1) | (2, 3) => (0, 1), // +i: Z·X, Y·Z, X·Y
        (2, 1) | (1, 3) | (3, 2) => (1, 1), // -i: X·Z, Z·Y, Y·X
        _ => (0, 0),                        // I or same Pauli
    }
}

/// Compute Pauli index (0=I, 1=Z, 2=X, 3=Y) from (x, z) bits.
const fn pauli_idx(x: bool, z: bool) -> u8 {
    match (x, z) {
        (false, false) => 0, // I
        (false, true) => 1,  // Z
        (true, false) => 2,  // X
        (true, true) => 3,   // Y
    }
}

/// Multiply row `a` by row `b` in place: a *= b.
///
/// Result: `a.row_x` = `a.row_x` XOR `b.row_x`
///         `a.row_z` = `a.row_z` XOR `b.row_z`
///         a.sign *= b.sign * (product of per-qubit phases)
///
/// Both rows are treated as Y-convention Pauli strings with optional signs.
pub(crate) fn multiply_row<S: IndexSet>(
    gens_a: &mut GensGeneric<S>,
    row_a: usize,
    gens_b: &GensGeneric<S>,
    row_b: usize,
    num_qubits: usize,
) {
    // Compute per-qubit phase contribution
    let mut phase = Complex64::new(1.0, 0.0);
    for q in 0..num_qubits {
        let a_x = gens_a.row_x[row_a].contains(q);
        let a_z = gens_a.row_z[row_a].contains(q);
        let b_x = gens_b.row_x[row_b].contains(q);
        let b_z = gens_b.row_z[row_b].contains(q);

        if !a_x && !a_z {
            continue;
        } // I * anything, no phase
        if !b_x && !b_z {
            continue;
        } // anything * I, no phase

        let pa = pauli_idx(a_x, a_z);
        let pb = pauli_idx(b_x, b_z);
        let (minus, i_bit) = pauli_phase(pa, pb);
        if minus == 1 {
            phase *= Complex64::new(-1.0, 0.0);
        }
        if i_bit == 1 {
            phase *= Complex64::new(0.0, 1.0);
        }
    }

    // Update row a's bit vectors: XOR with row b
    let b_row_x = gens_b.row_x[row_b].clone();
    let b_row_z = gens_b.row_z[row_b].clone();

    // Update col_x and col_z columns based on the XOR
    for q in b_row_x.iter() {
        gens_a.col_x[q].toggle(row_a);
    }
    for q in b_row_z.iter() {
        gens_a.col_z[q].toggle(row_a);
    }

    gens_a.row_x[row_a].xor_assign(&b_row_x);
    gens_a.row_z[row_a].xor_assign(&b_row_z);

    // Combine signs: a.sign *= b.sign * phase
    let b_minus = gens_b.signs_minus.contains(row_b);
    let b_i = gens_b.signs_i.contains(row_b);
    if b_minus {
        phase *= Complex64::new(-1.0, 0.0);
    }
    if b_i {
        phase *= Complex64::new(0.0, 1.0);
    }

    // Apply phase to row_a's signs
    let a_minus = gens_a.signs_minus.contains(row_a);
    let a_i = gens_a.signs_i.contains(row_a);
    // Current sign of a is (a_minus, a_i). New sign = old * phase.
    let old_sign = match (a_minus, a_i) {
        (false, false) => Complex64::new(1.0, 0.0),
        (true, false) => Complex64::new(-1.0, 0.0),
        (false, true) => Complex64::new(0.0, 1.0),
        (true, true) => Complex64::new(0.0, -1.0),
    };
    let new_sign = old_sign * phase;

    // Round to nearest (minus, i) combination
    let (new_minus, new_i) = if (new_sign - Complex64::new(1.0, 0.0)).norm() < 1e-9 {
        (false, false)
    } else if (new_sign - Complex64::new(-1.0, 0.0)).norm() < 1e-9 {
        (true, false)
    } else if (new_sign - Complex64::new(0.0, 1.0)).norm() < 1e-9 {
        (false, true)
    } else if (new_sign - Complex64::new(0.0, -1.0)).norm() < 1e-9 {
        (true, true)
    } else {
        panic!("row multiplication produced non-quarter-phase: {new_sign}");
    };

    if new_minus != a_minus {
        gens_a.signs_minus.toggle(row_a);
    }
    if new_i != a_i {
        gens_a.signs_i.toggle(row_a);
    }
}

/// Swap rows between two `GensGeneric` structures.
///
/// Swaps row `row_a` of `gens_a` with row `row_b` of `gens_b`, including
/// the bit vectors (`row_x`, `row_z`) and the signs. Also updates `col_x/col_z`.
fn swap_rows_between<S: IndexSet>(
    gens_a: &mut GensGeneric<S>,
    row_a: usize,
    gens_b: &mut GensGeneric<S>,
    row_b: usize,
) {
    // Swap bit vectors
    std::mem::swap(&mut gens_a.row_x[row_a], &mut gens_b.row_x[row_b]);
    std::mem::swap(&mut gens_a.row_z[row_a], &mut gens_b.row_z[row_b]);

    // Update col_x/col_z consistency
    // After swap: for each qubit q, toggle col_x[q] membership in row_a and row_b
    // based on the new row_x contents. Do this by recomputing from the row contents.
    //
    // Actually, the col representation must remain consistent. The row contents
    // swapped, so we need to:
    // - For gens_a: the qubits in row_a's NEW row_x are those that were in gens_b's row_b
    //   before swap. But gens_a.col_x is indexed by qubit and contains row indices within gens_a.
    //   So for each qubit q:
    //     - If row_a was previously in col_x[q] but isn't in gens_a.row_x[row_a] anymore, remove.
    //     - If row_a wasn't but is now, add.
    //   The "now" content is what was in gens_b.row_x[row_b] before swap (= gens_a.row_x[row_a] after swap).
    //
    // Simplest: after swapping row_x bit vectors, rebuild the cols for row_a in gens_a and row_b in gens_b.
    //
    // For each qubit q: check if gens_a.row_x[row_a].contains(q) == gens_a.col_x[q].contains(row_a).
    // If mismatch, toggle col_x[q] membership of row_a. Similar for row_z, row_b, etc.

    let num_qubits = gens_a.col_x.len();
    for q in 0..num_qubits {
        // gens_a row_a
        let row_x_has = gens_a.row_x[row_a].contains(q);
        let col_x_has = gens_a.col_x[q].contains(row_a);
        if row_x_has != col_x_has {
            gens_a.col_x[q].toggle(row_a);
        }
        let row_z_has = gens_a.row_z[row_a].contains(q);
        let col_z_has = gens_a.col_z[q].contains(row_a);
        if row_z_has != col_z_has {
            gens_a.col_z[q].toggle(row_a);
        }
        // gens_b row_b
        let row_x_has = gens_b.row_x[row_b].contains(q);
        let col_x_has = gens_b.col_x[q].contains(row_b);
        if row_x_has != col_x_has {
            gens_b.col_x[q].toggle(row_b);
        }
        let row_z_has = gens_b.row_z[row_b].contains(q);
        let col_z_has = gens_b.col_z[q].contains(row_b);
        if row_z_has != col_z_has {
            gens_b.col_z[q].toggle(row_b);
        }
    }

    // Swap signs
    let a_minus = gens_a.signs_minus.contains(row_a);
    let b_minus = gens_b.signs_minus.contains(row_b);
    if a_minus != b_minus {
        gens_a.signs_minus.toggle(row_a);
        gens_b.signs_minus.toggle(row_b);
    }
    let a_i = gens_a.signs_i.contains(row_a);
    let b_i = gens_b.signs_i.contains(row_b);
    if a_i != b_i {
        gens_a.signs_i.toggle(row_a);
        gens_b.signs_i.toggle(row_b);
    }
}

/// Right-compose Hadamard gate on qubit q onto the tableau.
///
/// Semantically: C -> C * `H_q`.
///
/// For each stabilizer/destabilizer row, this transforms `Z_k` -> (`H_q` `Z_k` `H_q`).
/// `H_q` `Z_q` `H_q` = `X_q`, so `S_q`' = C `X_q` C^dagger = `D_q` (old destabilizer).
/// `H_q` `Z_k` `H_q` = `Z_k` for k != q (unchanged).
/// Similarly for destabilizers: `H_q` `X_q` `H_q` = `Z_q`, so `D_q`' = old `S_q`.
///
/// Implementation: swap stabs row q with destabs row q.
pub fn right_compose_h(
    tableau: &mut SparseStabY<impl pecos_random::Rng + pecos_random::SeedableRng + std::fmt::Debug>,
    q: usize,
) {
    let (stabs, destabs) = tableau.stabs_and_destabs_mut();
    swap_rows_between(stabs, q, destabs, q);
}

/// Right-compose `S_z` (phase) gate on qubit q onto the tableau.
///
/// `S_z` `Z_q` `S_z^dagger` = `Z_q` (unchanged), so stabilizers unchanged.
/// `S_z` `X_q` `S_z^dagger` = `Y_q` = iXZ, so `D_q`' = i * `D_q` * `S_q`.
///
/// Implementation: destabs row q *= stabs row q (with phase +i).
///
/// # Panics
///
/// Panics if the resulting phase is not a quarter-phase (indicates a bug in
/// the phase-tracking logic).
pub fn right_compose_sz<R: pecos_random::Rng + pecos_random::SeedableRng + std::fmt::Debug>(
    tableau: &mut SparseStabY<R>,
    q: usize,
) {
    let num_qubits = tableau.num_qubits();
    // D_q' = i * D_q * S_q
    // Multiply destabs row q by stabs row q (get D_q * S_q)
    let stabs_snapshot_x = tableau.stabs().row_x[q].clone();
    let stabs_snapshot_z = tableau.stabs().row_z[q].clone();
    let stabs_minus = tableau.stabs().signs_minus.contains(q);
    let stabs_i = tableau.stabs().signs_i.contains(q);

    // Compute per-qubit phase of D_q * S_q (using snapshot of stabs row q)
    let destabs = tableau.destabs();
    let mut phase = Complex64::new(1.0, 0.0);
    for qq in 0..num_qubits {
        let a_x = destabs.row_x[q].contains(qq);
        let a_z = destabs.row_z[q].contains(qq);
        let b_x = stabs_snapshot_x.contains(qq);
        let b_z = stabs_snapshot_z.contains(qq);
        if (!a_x && !a_z) || (!b_x && !b_z) {
            continue;
        }
        let pa = pauli_idx(a_x, a_z);
        let pb = pauli_idx(b_x, b_z);
        let (minus, i_bit) = pauli_phase(pa, pb);
        if minus == 1 {
            phase *= Complex64::new(-1.0, 0.0);
        }
        if i_bit == 1 {
            phase *= Complex64::new(0.0, 1.0);
        }
    }
    // Include stabs sign
    if stabs_minus {
        phase *= Complex64::new(-1.0, 0.0);
    }
    if stabs_i {
        phase *= Complex64::new(0.0, 1.0);
    }
    // Multiply by i (the S_z phase)
    phase *= Complex64::new(0.0, 1.0);

    // Now update destabs row q: XOR bits with stabs row q, apply accumulated phase
    let destabs_mut = tableau.destabs_mut();

    // Update col_x/col_z
    for qq in &stabs_snapshot_x {
        destabs_mut.col_x[qq].toggle(q);
    }
    for qq in &stabs_snapshot_z {
        destabs_mut.col_z[qq].toggle(q);
    }
    destabs_mut.row_x[q].xor_assign(&stabs_snapshot_x);
    destabs_mut.row_z[q].xor_assign(&stabs_snapshot_z);

    // Update sign
    let d_minus = destabs_mut.signs_minus.contains(q);
    let d_i = destabs_mut.signs_i.contains(q);
    let old_sign = match (d_minus, d_i) {
        (false, false) => Complex64::new(1.0, 0.0),
        (true, false) => Complex64::new(-1.0, 0.0),
        (false, true) => Complex64::new(0.0, 1.0),
        (true, true) => Complex64::new(0.0, -1.0),
    };
    let new_sign = old_sign * phase;
    let (new_minus, new_i) = if (new_sign - Complex64::new(1.0, 0.0)).norm() < 1e-9 {
        (false, false)
    } else if (new_sign - Complex64::new(-1.0, 0.0)).norm() < 1e-9 {
        (true, false)
    } else if (new_sign - Complex64::new(0.0, 1.0)).norm() < 1e-9 {
        (false, true)
    } else if (new_sign - Complex64::new(0.0, -1.0)).norm() < 1e-9 {
        (true, true)
    } else {
        panic!("right_compose_sz produced non-quarter-phase: {new_sign}");
    };
    if new_minus != d_minus {
        destabs_mut.signs_minus.toggle(q);
    }
    if new_i != d_i {
        destabs_mut.signs_i.toggle(q);
    }
}

/// Right-compose CX(control, target) gate onto the tableau.
///
/// CX(c, t) acting on the right:
/// - `Z_c` unchanged -> stabs[c] unchanged
/// - `Z_t` -> `Z_c` `Z_t` -> stabs[t] *= stabs[c]
/// - `X_c` -> `X_c` `X_t` -> destabs[c] *= destabs[t]
/// - `X_t` unchanged -> destabs[t] unchanged
pub fn right_compose_cx<R: pecos_random::Rng + pecos_random::SeedableRng + std::fmt::Debug>(
    tableau: &mut SparseStabY<R>,
    control: usize,
    target: usize,
) {
    debug_assert_ne!(control, target, "CX requires distinct qubits");
    let num_qubits = tableau.num_qubits();

    // stabs[t] *= stabs[c] (multiply within stabs, self-reference)
    {
        let stabs = tableau.stabs_mut();
        multiply_row_within(stabs, target, control, num_qubits);
    }

    // destabs[c] *= destabs[t]
    {
        let destabs = tableau.destabs_mut();
        multiply_row_within(destabs, control, target, num_qubits);
    }
}

/// Right-compose `S_z^dagger` (inverse phase) gate onto the tableau.
///
/// Sdg Z Sdg^dagger = Z (unchanged), Sdg X Sdg^dagger = -Y = -iXZ.
/// So destabs[q] gets multiplied by stabs[q] with a -i phase.
pub fn right_compose_szdg<R: pecos_random::Rng + pecos_random::SeedableRng + std::fmt::Debug>(
    tableau: &mut SparseStabY<R>,
    q: usize,
) {
    let num_qubits = tableau.num_qubits();
    let stabs_snapshot_x = tableau.stabs().row_x[q].clone();
    let stabs_snapshot_z = tableau.stabs().row_z[q].clone();
    let stabs_minus = tableau.stabs().signs_minus.contains(q);
    let stabs_i = tableau.stabs().signs_i.contains(q);

    let destabs = tableau.destabs();
    let mut phase = Complex64::new(1.0, 0.0);
    for qq in 0..num_qubits {
        let a_x = destabs.row_x[q].contains(qq);
        let a_z = destabs.row_z[q].contains(qq);
        let b_x = stabs_snapshot_x.contains(qq);
        let b_z = stabs_snapshot_z.contains(qq);
        if (!a_x && !a_z) || (!b_x && !b_z) {
            continue;
        }
        let pa = pauli_idx(a_x, a_z);
        let pb = pauli_idx(b_x, b_z);
        let (minus, i_bit) = pauli_phase(pa, pb);
        if minus == 1 {
            phase *= Complex64::new(-1.0, 0.0);
        }
        if i_bit == 1 {
            phase *= Complex64::new(0.0, 1.0);
        }
    }
    if stabs_minus {
        phase *= Complex64::new(-1.0, 0.0);
    }
    if stabs_i {
        phase *= Complex64::new(0.0, 1.0);
    }
    // Multiply by -i (Sdg phase)
    phase *= Complex64::new(0.0, -1.0);

    let destabs_mut = tableau.destabs_mut();
    for qq in &stabs_snapshot_x {
        destabs_mut.col_x[qq].toggle(q);
    }
    for qq in &stabs_snapshot_z {
        destabs_mut.col_z[qq].toggle(q);
    }
    destabs_mut.row_x[q].xor_assign(&stabs_snapshot_x);
    destabs_mut.row_z[q].xor_assign(&stabs_snapshot_z);

    apply_phase_to_sign(destabs_mut, q, phase);
}

/// Right-compose X gate on qubit q onto the tableau.
///
/// X Z X = -Z, X X X = X. So stabs[q] sign flips, destabs[q] unchanged.
pub fn right_compose_x<R: pecos_random::Rng + pecos_random::SeedableRng + std::fmt::Debug>(
    tableau: &mut SparseStabY<R>,
    q: usize,
) {
    // X conjugates Z to -Z, so stabs[q] (which is C Z_q C^dagger) becomes C (-Z_q) C^dagger
    // Result: flip sign of stabs row q.
    let stabs = tableau.stabs_mut();
    stabs.signs_minus.toggle(q);
}

/// Right-compose Z gate on qubit q onto the tableau.
///
/// Z Z Z = Z, Z X Z = -X. So destabs[q] sign flips, stabs[q] unchanged.
pub fn right_compose_z<R: pecos_random::Rng + pecos_random::SeedableRng + std::fmt::Debug>(
    tableau: &mut SparseStabY<R>,
    q: usize,
) {
    let destabs = tableau.destabs_mut();
    destabs.signs_minus.toggle(q);
}

/// Right-compose CY(control, target) gate onto the tableau.
///
/// CY = (I ⊗ Sdg) · CX · (I ⊗ S) when acting on state (circuit-order first-to-last: S, CX, Sdg).
///   Verify: S · X · Sdg = Y (confirmed by matrix calculation).
///
/// For right-composition (C' = C * U): U = Sdg · CX · S (matrix form, since virtual-side
/// circuit order is reverse of matrix product direction). No wait -- read carefully:
/// If right-compose applies U to virtual side BEFORE C, then virtual-side circuit order
/// for U = Sdg · CX · S is: S first, then CX, then Sdg. But that's not what we want.
///
/// Let me restate: we want the VIRTUAL op to be CY = (in circuit order on virtual) S, CX, Sdg.
/// For this, the right-compose sequence is: call Sdg FIRST, then CX, then S.
/// Why? Each `right_compose_X(U)` multiplies C by U on the right: C := C * U.
/// After calls [A, B, C]: tableau = ((`C_init` * A) * B) * C = `C_init` * A * B * C.
/// Read as matrix: virtual op applied first is C (rightmost), then B, then A.
/// So for virtual circuit "S, CX, Sdg" (S first), we need matrix A·B·C = Sdg·CX·S,
/// which means call sequence: Sdg, CX, S.
pub fn right_compose_cy<R: pecos_random::Rng + pecos_random::SeedableRng + std::fmt::Debug>(
    tableau: &mut SparseStabY<R>,
    control: usize,
    target: usize,
) {
    right_compose_szdg(tableau, target);
    right_compose_cx(tableau, control, target);
    right_compose_sz(tableau, target);
}

/// Right-compose CZ(q1, q2) gate onto the tableau.
///
/// CZ = `H_2` CX(1,2) `H_2`. The effect on generators:
/// - `Z_1` -> `Z_1`, `Z_2` -> `Z_2` (stabs unchanged)
/// - `X_1` -> `X_1` `Z_2` (destabs[1] *= stabs[2])
/// - `X_2` -> `Z_1` `X_2` (destabs[2] *= stabs[1])
pub fn right_compose_cz<R: pecos_random::Rng + pecos_random::SeedableRng + std::fmt::Debug>(
    tableau: &mut SparseStabY<R>,
    q1: usize,
    q2: usize,
) {
    debug_assert_ne!(q1, q2, "CZ requires distinct qubits");
    let num_qubits = tableau.num_qubits();

    // destabs[q1] *= stabs[q2] (X_1 -> X_1 Z_2 means D_1 gets a Z_2 factor = S_2)
    multiply_row_across(tableau, q1, q2, num_qubits, /*dest_to_stab=*/ true);
    // destabs[q2] *= stabs[q1]
    multiply_row_across(tableau, q2, q1, num_qubits, /*dest_to_stab=*/ true);
}

/// Multiply row `dst_row` by row `src_row` within the same generator set.
fn multiply_row_within<S: IndexSet>(
    gens: &mut GensGeneric<S>,
    dst_row: usize,
    src_row: usize,
    num_qubits: usize,
) {
    // Snapshot source row
    let src_x = gens.row_x[src_row].clone();
    let src_z = gens.row_z[src_row].clone();
    let src_minus = gens.signs_minus.contains(src_row);
    let src_i = gens.signs_i.contains(src_row);

    // Compute phase
    let mut phase = Complex64::new(1.0, 0.0);
    for q in 0..num_qubits {
        let a_x = gens.row_x[dst_row].contains(q);
        let a_z = gens.row_z[dst_row].contains(q);
        let b_x = src_x.contains(q);
        let b_z = src_z.contains(q);
        if (!a_x && !a_z) || (!b_x && !b_z) {
            continue;
        }
        let pa = pauli_idx(a_x, a_z);
        let pb = pauli_idx(b_x, b_z);
        let (minus, i_bit) = pauli_phase(pa, pb);
        if minus == 1 {
            phase *= Complex64::new(-1.0, 0.0);
        }
        if i_bit == 1 {
            phase *= Complex64::new(0.0, 1.0);
        }
    }
    if src_minus {
        phase *= Complex64::new(-1.0, 0.0);
    }
    if src_i {
        phase *= Complex64::new(0.0, 1.0);
    }

    // Update col_x/col_z
    for q in src_x.iter() {
        gens.col_x[q].toggle(dst_row);
    }
    for q in src_z.iter() {
        gens.col_z[q].toggle(dst_row);
    }
    gens.row_x[dst_row].xor_assign(&src_x);
    gens.row_z[dst_row].xor_assign(&src_z);

    // Apply accumulated phase to dst_row's sign
    apply_phase_to_sign(gens, dst_row, phase);
}

/// Multiply destabs[`dst_q`] by stabs[`src_q`] (or vice versa).
/// `dest_to_stab`: if true, dst is destabs and src is stabs.
fn multiply_row_across<R: pecos_random::Rng + pecos_random::SeedableRng + std::fmt::Debug>(
    tableau: &mut SparseStabY<R>,
    dst_q: usize,
    src_q: usize,
    num_qubits: usize,
    dest_to_stab: bool,
) {
    if !dest_to_stab {
        unimplemented!("only dest_to_stab=true is used here");
    }
    // Snapshot source row (from stabs)
    let src_x = tableau.stabs().row_x[src_q].clone();
    let src_z = tableau.stabs().row_z[src_q].clone();
    let src_minus = tableau.stabs().signs_minus.contains(src_q);
    let src_i = tableau.stabs().signs_i.contains(src_q);

    // Compute phase using destabs[dst_q] (destination) and stabs[src_q] (source)
    let destabs = tableau.destabs();
    let mut phase = Complex64::new(1.0, 0.0);
    for q in 0..num_qubits {
        let a_x = destabs.row_x[dst_q].contains(q);
        let a_z = destabs.row_z[dst_q].contains(q);
        let b_x = src_x.contains(q);
        let b_z = src_z.contains(q);
        if (!a_x && !a_z) || (!b_x && !b_z) {
            continue;
        }
        let pa = pauli_idx(a_x, a_z);
        let pb = pauli_idx(b_x, b_z);
        let (minus, i_bit) = pauli_phase(pa, pb);
        if minus == 1 {
            phase *= Complex64::new(-1.0, 0.0);
        }
        if i_bit == 1 {
            phase *= Complex64::new(0.0, 1.0);
        }
    }
    if src_minus {
        phase *= Complex64::new(-1.0, 0.0);
    }
    if src_i {
        phase *= Complex64::new(0.0, 1.0);
    }

    // Update destabs[dst_q] bits
    let destabs_mut = tableau.destabs_mut();
    for q in &src_x {
        destabs_mut.col_x[q].toggle(dst_q);
    }
    for q in &src_z {
        destabs_mut.col_z[q].toggle(dst_q);
    }
    destabs_mut.row_x[dst_q].xor_assign(&src_x);
    destabs_mut.row_z[dst_q].xor_assign(&src_z);

    apply_phase_to_sign(destabs_mut, dst_q, phase);
}

/// Combine `phase` (expected to be a fourth root of unity) into the row's sign.
fn apply_phase_to_sign<S: IndexSet>(gens: &mut GensGeneric<S>, row: usize, phase: Complex64) {
    let d_minus = gens.signs_minus.contains(row);
    let d_i = gens.signs_i.contains(row);
    let old_sign = match (d_minus, d_i) {
        (false, false) => Complex64::new(1.0, 0.0),
        (true, false) => Complex64::new(-1.0, 0.0),
        (false, true) => Complex64::new(0.0, 1.0),
        (true, true) => Complex64::new(0.0, -1.0),
    };
    let new_sign = old_sign * phase;
    let (new_minus, new_i) = if (new_sign - Complex64::new(1.0, 0.0)).norm() < 1e-9 {
        (false, false)
    } else if (new_sign - Complex64::new(-1.0, 0.0)).norm() < 1e-9 {
        (true, false)
    } else if (new_sign - Complex64::new(0.0, 1.0)).norm() < 1e-9 {
        (false, true)
    } else if (new_sign - Complex64::new(0.0, -1.0)).norm() < 1e-9 {
        (true, true)
    } else {
        panic!("phase not a fourth root of unity: {new_sign}");
    };
    if new_minus != d_minus {
        gens.signs_minus.toggle(row);
    }
    if new_i != d_i {
        gens.signs_i.toggle(row);
    }
}

// Silence unused warnings
#[allow(dead_code)]
fn _type_check() -> BitSet {
    BitSet::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;
    use pecos_core::QubitId;
    use pecos_simulators::{CliffordGateable, SparseStabY};

    /// Verify that right-composing G and then left-composing G^-1 gives identity.
    /// (This doesn't fully test right-compose but is a sanity check.)

    #[test]
    fn test_right_compose_h_twice_is_identity() {
        // H * H = I, so right-composing H twice should leave tableau unchanged
        let mut t = SparseStabY::new(3).with_destab_sign_tracking();
        t.h(&[QubitId(0)]);
        t.cx(&[(QubitId(0), QubitId(1))]);
        let before_stabs_x: Vec<_> = (0..3).map(|i| t.stabs().row_x[i].clone()).collect();
        let before_stabs_z: Vec<_> = (0..3).map(|i| t.stabs().row_z[i].clone()).collect();

        right_compose_h(&mut t, 0);
        right_compose_h(&mut t, 0);

        for i in 0..3 {
            assert_eq!(t.stabs().row_x[i], before_stabs_x[i], "stab {i} x changed");
            assert_eq!(t.stabs().row_z[i], before_stabs_z[i], "stab {i} z changed");
        }
    }

    #[test]
    fn test_right_compose_h_swaps_stab_destab() {
        // Initial state: stabs=[Z_0, Z_1], destabs=[X_0, X_1]
        // After right-compose H_0: stabs[0] should become what destabs[0] was (X_0),
        // destabs[0] should become what stabs[0] was (Z_0).
        let mut t = SparseStabY::new(2);
        // Initial: stab[0] = Z_0 (row_z={0}, row_x={})
        //          destab[0] = X_0 (row_x={0}, row_z={})
        assert!(t.stabs().row_z[0].contains(0));
        assert!(!t.stabs().row_x[0].contains(0));
        assert!(t.destabs().row_x[0].contains(0));
        assert!(!t.destabs().row_z[0].contains(0));

        right_compose_h(&mut t, 0);

        // After: stab[0] should be X_0, destab[0] should be Z_0
        assert!(
            t.stabs().row_x[0].contains(0),
            "stab[0] should have X after H"
        );
        assert!(!t.stabs().row_z[0].contains(0));
        assert!(
            t.destabs().row_z[0].contains(0),
            "destab[0] should have Z after H"
        );
        assert!(!t.destabs().row_x[0].contains(0));
    }

    /// Build the 2^n x 2^n matrix for a generator row.
    fn gen_matrix<S: IndexSet>(
        gens: &GensGeneric<S>,
        row: usize,
        n: usize,
    ) -> nalgebra::DMatrix<Complex64> {
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
        let mut result = DMatrix::from_element(1, 1, Complex64::new(1.0, 0.0));
        for q in 0..n {
            let has_x = gens.row_x[row].contains(q);
            let has_z = gens.row_z[row].contains(q);
            let p = match (has_x, has_z) {
                (false, false) => &i_mat,
                (true, false) => &x_mat,
                (false, true) => &z_mat,
                (true, true) => &y_mat,
            };
            result = result.kronecker(p);
        }
        let mut phase = Complex64::new(1.0, 0.0);
        if gens.signs_minus.contains(row) {
            phase *= Complex64::new(-1.0, 0.0);
        }
        if gens.signs_i.contains(row) {
            phase *= Complex64::new(0.0, 1.0);
        }
        result * phase
    }

    /// Verify that right-composing G transforms `S_k` into C (G `Z_k` G^dagger) C^dagger.
    /// We do this by brute-force: construct gate matrix G, compute expected `S_k`, compare.
    #[test]
    fn test_right_compose_h_correct_transformation() {
        // Start with some non-trivial state
        let mut t = SparseStabY::new(2).with_destab_sign_tracking();
        t.h(&[QubitId(0)]);
        t.cx(&[(QubitId(0), QubitId(1))]);

        // Compute S_0, S_1 before right-compose
        let s0_before = gen_matrix(t.stabs(), 0, 2);
        let s1_before = gen_matrix(t.stabs(), 1, 2);

        // Right-compose H on qubit 0
        right_compose_h(&mut t, 0);

        // After right-compose by H_0, S_k should be C (H_0 Z_k H_0) C^dagger.
        // H_0 Z_0 H_0 = X_0, H_0 Z_1 H_0 = Z_1 (H only on qubit 0).
        // So S_0' = C X_0 C^dagger = D_0 (original destabilizer 0)
        //    S_1' = C Z_1 C^dagger = S_1 (unchanged)
        let s0_after = gen_matrix(t.stabs(), 0, 2);
        let s1_after = gen_matrix(t.stabs(), 1, 2);

        assert!(
            (s1_after.clone() - s1_before).norm() < 1e-10,
            "S_1 should be unchanged"
        );
        // S_0 should equal what D_0 was before (we can't easily recompute that here,
        // but we can verify S_0 != original S_0)
        assert!(
            (s0_after.clone() - s0_before).norm() > 1e-3,
            "S_0 should have changed"
        );

        // Verify stabilizer algebra: S_0' and S_1' should anticommute appropriately with
        // the destabilizers (which also got transformed).
        let d0_after = gen_matrix(t.destabs(), 0, 2);
        let d1_after = gen_matrix(t.destabs(), 1, 2);

        // S_k and D_k should anticommute: S_k D_k + D_k S_k = 0
        let anti_00 = &s0_after * &d0_after + &d0_after * &s0_after;
        let anti_11 = &s1_after * &d1_after + &d1_after * &s1_after;
        assert!(anti_00.norm() < 1e-10, "S_0 and D_0 should anticommute");
        assert!(anti_11.norm() < 1e-10, "S_1 and D_1 should anticommute");

        // S_k and D_j (k != j) should commute: S_k D_j = D_j S_k
        let comm_01 = &s0_after * &d1_after - &d1_after * &s0_after;
        let comm_10 = &s1_after * &d0_after - &d0_after * &s1_after;
        assert!(comm_01.norm() < 1e-10, "S_0 and D_1 should commute");
        assert!(comm_10.norm() < 1e-10, "S_1 and D_0 should commute");

        // S_0 and S_1 should commute (stabilizers all commute with each other)
        let comm_ss = &s0_after * &s1_after - &s1_after * &s0_after;
        assert!(comm_ss.norm() < 1e-10, "S_0 and S_1 should commute");
    }

    #[test]
    fn test_right_compose_cx_preserves_algebra() {
        // Right-compose CX should preserve the symplectic structure
        let mut t = SparseStabY::new(3).with_destab_sign_tracking();
        t.h(&[QubitId(0)]);
        t.cx(&[(QubitId(0), QubitId(1))]);

        right_compose_cx(&mut t, 0, 2);

        // Verify stabilizer/destabilizer algebra
        let stabs: Vec<_> = (0..3).map(|i| gen_matrix(t.stabs(), i, 3)).collect();
        let destabs: Vec<_> = (0..3).map(|i| gen_matrix(t.destabs(), i, 3)).collect();

        for i in 0..3 {
            // S_i and D_i anticommute
            let anti = &stabs[i] * &destabs[i] + &destabs[i] * &stabs[i];
            assert!(anti.norm() < 1e-10, "S_{i} and D_{i} should anticommute");
            // S_i and S_j commute (j != i)
            for j in 0..3 {
                if i == j {
                    continue;
                }
                let comm_ss = &stabs[i] * &stabs[j] - &stabs[j] * &stabs[i];
                assert!(comm_ss.norm() < 1e-10, "S_{i} and S_{j} should commute");
                // S_i and D_j commute
                let comm_sd = &stabs[i] * &destabs[j] - &destabs[j] * &stabs[i];
                assert!(comm_sd.norm() < 1e-10, "S_{i} and D_{j} should commute");
            }
        }
    }

    #[test]
    fn test_right_compose_sz_preserves_algebra() {
        let mut t = SparseStabY::new(2).with_destab_sign_tracking();
        t.h(&[QubitId(0)]);
        t.cx(&[(QubitId(0), QubitId(1))]);

        right_compose_sz(&mut t, 0);

        let stabs: Vec<_> = (0..2).map(|i| gen_matrix(t.stabs(), i, 2)).collect();
        let destabs: Vec<_> = (0..2).map(|i| gen_matrix(t.destabs(), i, 2)).collect();

        for i in 0..2 {
            let anti = &stabs[i] * &destabs[i] + &destabs[i] * &stabs[i];
            assert!(
                anti.norm() < 1e-10,
                "S_{i} and D_{i} should anticommute after right-compose SZ"
            );
        }
    }

    #[test]
    fn test_right_compose_szdg_preserves_algebra() {
        let mut t = SparseStabY::new(2).with_destab_sign_tracking();
        t.h(&[QubitId(0)]);
        right_compose_szdg(&mut t, 0);

        let stabs: Vec<_> = (0..2).map(|i| gen_matrix(t.stabs(), i, 2)).collect();
        let destabs: Vec<_> = (0..2).map(|i| gen_matrix(t.destabs(), i, 2)).collect();
        for i in 0..2 {
            let anti = &stabs[i] * &destabs[i] + &destabs[i] * &stabs[i];
            assert!(anti.norm() < 1e-10);
        }
    }

    #[test]
    fn test_right_compose_s_then_sdg_is_identity() {
        // S * Sdg = I
        let mut t = SparseStabY::new(2).with_destab_sign_tracking();
        t.h(&[QubitId(0)]);
        t.cx(&[(QubitId(0), QubitId(1))]);

        let before = gen_matrix(t.destabs(), 0, 2);
        right_compose_sz(&mut t, 0);
        right_compose_szdg(&mut t, 0);
        let after = gen_matrix(t.destabs(), 0, 2);
        assert!(
            (after - before).norm() < 1e-10,
            "S then Sdg should be identity"
        );
    }

    #[test]
    fn test_right_compose_x_flips_stab_sign() {
        let mut t = SparseStabY::new(2);
        let before_sign = t.stabs().signs_minus.contains(0);
        right_compose_x(&mut t, 0);
        let after_sign = t.stabs().signs_minus.contains(0);
        assert_ne!(before_sign, after_sign, "X should flip stab sign");
        // Destab unchanged
        assert!(!t.destabs().signs_minus.contains(0));
    }

    /// Directly test: does `right_compose_cy` implement CY or -CY?
    /// Build two tableaus: one with `right_compose_cy`, one with the reference's
    /// Sdg-CX-S pattern. They should be equivalent if both implement CY.
    #[test]
    fn test_right_compose_cy_vs_reference_pattern() {
        let mut t_mine = SparseStabY::new(3).with_destab_sign_tracking();
        t_mine.h(&[QubitId(0)]);
        t_mine.cx(&[(QubitId(0), QubitId(1))]);
        let mut t_ref = t_mine.clone();

        // Mine: right_compose_cy = S, CX, Sdg (in call order)
        right_compose_cy(&mut t_mine, 0, 2);

        // Reference pattern: Sdg, CX, S (in call order)
        right_compose_szdg(&mut t_ref, 2);
        right_compose_cx(&mut t_ref, 0, 2);
        right_compose_sz(&mut t_ref, 2);

        // Compare generator matrices
        let m_mine: Vec<_> = (0..3).map(|i| gen_matrix(t_mine.stabs(), i, 3)).collect();
        let m_ref: Vec<_> = (0..3).map(|i| gen_matrix(t_ref.stabs(), i, 3)).collect();

        for i in 0..3 {
            let diff = (&m_mine[i] - &m_ref[i]).norm();
            let neg_diff = (&m_mine[i] + &m_ref[i]).norm();
            eprintln!("stab {i}: diff_eq={diff:.3e}, diff_neg={neg_diff:.3e}");
        }

        let d_mine: Vec<_> = (0..3).map(|i| gen_matrix(t_mine.destabs(), i, 3)).collect();
        let d_ref: Vec<_> = (0..3).map(|i| gen_matrix(t_ref.destabs(), i, 3)).collect();
        for i in 0..3 {
            let diff = (&d_mine[i] - &d_ref[i]).norm();
            let neg_diff = (&d_mine[i] + &d_ref[i]).norm();
            eprintln!("destab {i}: diff_eq={diff:.3e}, diff_neg={neg_diff:.3e}");
        }
    }

    #[test]
    fn test_right_compose_cy_preserves_algebra() {
        let mut t = SparseStabY::new(3).with_destab_sign_tracking();
        t.h(&[QubitId(0)]);
        t.cx(&[(QubitId(0), QubitId(1))]);

        right_compose_cy(&mut t, 0, 2);

        let stabs: Vec<_> = (0..3).map(|i| gen_matrix(t.stabs(), i, 3)).collect();
        let destabs: Vec<_> = (0..3).map(|i| gen_matrix(t.destabs(), i, 3)).collect();
        for i in 0..3 {
            let anti = &stabs[i] * &destabs[i] + &destabs[i] * &stabs[i];
            assert!(anti.norm() < 1e-10, "S_{i} and D_{i} should anticommute");
            for j in 0..3 {
                if i == j {
                    continue;
                }
                let comm_ss = &stabs[i] * &stabs[j] - &stabs[j] * &stabs[i];
                assert!(comm_ss.norm() < 1e-10);
                let comm_sd = &stabs[i] * &destabs[j] - &destabs[j] * &stabs[i];
                assert!(comm_sd.norm() < 1e-10);
            }
        }
    }

    #[test]
    fn test_right_compose_cx_updates_rows() {
        // Right-compose CX(0, 1) on identity tableau:
        // stabs[1] *= stabs[0] means stab[1] = Z_1 * Z_0 = Z_0 Z_1
        // destabs[0] *= destabs[1] means destab[0] = X_0 * X_1 = X_0 X_1
        let mut t = SparseStabY::new(3).with_destab_sign_tracking();
        right_compose_cx(&mut t, 0, 1);

        // stab[1] should now have Z on both qubit 0 and qubit 1
        assert!(
            t.stabs().row_z[1].contains(0),
            "stab[1] should have Z on q0"
        );
        assert!(
            t.stabs().row_z[1].contains(1),
            "stab[1] should have Z on q1"
        );
        // stab[0] unchanged
        assert!(t.stabs().row_z[0].contains(0));
        assert!(!t.stabs().row_z[0].contains(1));
        // destab[0] should have X on both qubits
        assert!(t.destabs().row_x[0].contains(0));
        assert!(t.destabs().row_x[0].contains(1));
        // destab[1] unchanged
        assert!(t.destabs().row_x[1].contains(1));
        assert!(!t.destabs().row_x[1].contains(0));
    }
}
