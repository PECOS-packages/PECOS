// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License.You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! Single-qubit Clifford frame tracking.
//!
//! Represents one of the 24 single-qubit Clifford gates (modulo global phase)
//! as a compact `u8` index with lookup tables for O(1) composition, inverse,
//! and Pauli image queries.
//!
//! # Design
//!
//! Each element is identified by its Heisenberg-picture action on Pauli operators:
//! for gate C, the images C†XC and C†ZC uniquely determine the element (the Y
//! image is derived from Y = iXZ). All lookup tables are computed at compile time
//! from these Heisenberg actions.
//!
//! Indices 0–3 are the four Paulis (I, X, Y, Z), enabling fast `is_pauli()` checks.
//!
//! The CZ lookup table is derived from the `GraphSim` reference implementation of
//! Anders & Briegel, "Fast simulation of stabilizer circuits using a graph-state
//! representation", [arXiv:quant-ph/0504117](https://arxiv.org/abs/quant-ph/0504117).

/// Which Pauli axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum PauliAxis {
    X = 0,
    Y = 1,
    Z = 2,
}

/// A signed Pauli axis: ±X, ±Y, or ±Z.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SignedPauli {
    pub axis: PauliAxis,
    pub positive: bool,
}

/// A single-qubit Clifford gate modulo global phase.
///
/// The 24 elements are stored as a `u8` index into compile-time lookup tables.
/// Indices 0–3 are Paulis (I, X, Y, Z).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CliffordFrame(u8);

// ============================================================================
// Heisenberg action data
// ============================================================================
//
// Each element is defined by (x_axis, x_neg, z_axis, z_neg) where:
//   axis ∈ {0=X, 1=Y, 2=Z}, neg = true means the image is negated.
//
// The Y image is derived: Y = iXZ → C†YC = i·(C†XC)·(C†ZC).
//
// Convention: compose(frame, gate) gives the new frame after applying `gate`
// to a qubit whose accumulated frame is `frame`. This corresponds to the
// matrix product gate · frame.
//
// Heisenberg of compose(a, b): apply b's action first, then a's action.
// Result: x_image = a.apply(b.x_image), z_image = a.apply(b.z_image).

/// Heisenberg actions for all 24 elements: (`x_axis`, `x_neg`, `z_axis`, `z_neg`).
const HEIS: [(u8, bool, u8, bool); 24] = [
    // Paulis (indices 0-3)
    (0, false, 2, false), //  0: I       X→+X  Z→+Z
    (0, false, 2, true),  //  1: X       X→+X  Z→-Z
    (0, true, 2, true),   //  2: Y       X→-X  Z→-Z
    (0, true, 2, false),  //  3: Z       X→-X  Z→+Z
    // S-class
    (1, true, 2, false),  //  4: S       X→-Y  Z→+Z
    (1, false, 2, false), //  5: Sdg     X→+Y  Z→+Z
    // H-class
    (2, false, 0, false), //  6: H       X→+Z  Z→+X
    (1, false, 0, false), //  7: SH      X→+Y  Z→+X
    (2, false, 1, true),  //  8: HS      X→+Z  Z→-Y
    (2, true, 0, false),  //  9: ZH      X→-Z  Z→+X   (= SYdg)
    (2, false, 0, true),  // 10: HZ      X→+Z  Z→-X   (= SY)
    (1, true, 0, false),  // 11: SdgH    X→-Y  Z→+X
    // SHS-class
    (0, false, 1, true),  // 12: SHS     X→+X  Z→-Y   (= SXdg)
    (0, false, 1, false), // 13: HSH     X→+X  Z→+Y   (= SX)
    (2, false, 1, false), // 14: SHSH    X→+Z  Z→+Y
    (2, true, 1, true),   // 15: S²HS    X→-Z  Z→-Y
    (1, true, 0, true),   // 16: SHS²    X→-Y  Z→-X
    (0, true, 1, true),   // 17: S³HS    X→-X  Z→-Y
    (2, true, 0, true),   // 18: S²HS²   X→-Z  Z→-X
    (0, true, 1, false),  // 19: S²HSH   X→-X  Z→+Y
    (1, true, 2, true),   // 20: HS²HS   X→-Y  Z→-Z
    (1, false, 0, true),  // 21: S³HS²   X→+Y  Z→-X
    (2, true, 1, false),  // 22: S³HSH   X→-Z  Z→+Y
    (1, false, 2, true),  // 23: HS²HS³  X→+Y  Z→-Z
];

/// Generator sequences for each element (0=H, 1=S), applied left-to-right
/// as matrix products: seq [a,b,c] means the gate C = a·b·c (a applied last to state).
/// Used for flushing the Clifford frame into H+S gate sequences.
pub const GENERATORS: [[u8; 7]; 24] = {
    // We store fixed-size arrays with 0xFF as padding (unused entries).
    const P: u8 = 0xFF; // padding
    const H: u8 = 0;
    const S: u8 = 1;
    [
        [P, P, P, P, P, P, P], //  0: I
        [H, S, S, H, P, P, P], //  1: X = H·S²·H
        [S, S, H, S, S, H, P], //  2: Y = S²·H·S²·H
        [S, S, P, P, P, P, P], //  3: Z = S²
        [S, P, P, P, P, P, P], //  4: S
        [S, S, S, P, P, P, P], //  5: Sdg = S³
        [H, P, P, P, P, P, P], //  6: H
        [S, H, P, P, P, P, P], //  7: SH
        [H, S, P, P, P, P, P], //  8: HS
        [S, S, H, P, P, P, P], //  9: S²H
        [H, S, S, P, P, P, P], // 10: HS²
        [S, S, S, H, P, P, P], // 11: S³H
        [S, H, S, P, P, P, P], // 12: SHS
        [H, S, H, P, P, P, P], // 13: HSH
        [S, H, S, H, P, P, P], // 14: SHSH
        [S, S, H, S, P, P, P], // 15: S²HS
        [S, H, S, S, P, P, P], // 16: SHS²
        [S, S, S, H, S, P, P], // 17: S³HS
        [S, S, H, S, S, P, P], // 18: S²HS²
        [S, S, H, S, H, P, P], // 19: S²HSH
        [H, S, S, H, S, P, P], // 20: HS²HS
        [S, S, S, H, S, S, P], // 21: S³HS²
        [S, S, S, H, S, H, P], // 22: S³HSH
        [H, S, S, H, S, S, S], // 23: HS²HS³
    ]
};

/// Length of each generator sequence (number of non-padding entries).
pub const GEN_LENS: [u8; 24] = [
    0, 4, 6, 2, 1, 3, 1, 2, 2, 3, 3, 4, 3, 3, 4, 4, 4, 5, 5, 5, 5, 6, 6, 7,
];

// ============================================================================
// Const-fn helpers for table computation
// ============================================================================

/// Compute Y-axis image from X and Z images.
/// Y = iXZ, so C†YC = i·(C†XC)·(C†ZC).
const fn y_image(x_axis: u8, x_neg: bool, z_axis: u8, z_neg: bool) -> (u8, bool) {
    let y_axis = 3 - x_axis - z_axis;
    // Positive cyclic order: (0→1→2→0). eps=+1 if (x_axis+1)%3 == z_axis.
    let eps_positive = (x_axis + 1) % 3 == z_axis;
    let xor = x_neg != z_neg;
    // Sign of Y image: -eps * sign_x * sign_z
    let y_neg = if eps_positive { !xor } else { xor };
    (y_axis, y_neg)
}

/// Get all three Heisenberg images (X, Y, Z) for element i.
const fn all_images(i: usize) -> [(u8, bool); 3] {
    let (xa, xn, za, zn) = HEIS[i];
    let (ya, yn) = y_image(xa, xn, za, zn);
    [(xa, xn), (ya, yn), (za, zn)]
}

/// Apply a Clifford's Heisenberg action to a signed Pauli.
const fn apply_action(imgs: [(u8, bool); 3], p_axis: u8, p_neg: bool) -> (u8, bool) {
    let (img_axis, img_neg) = imgs[p_axis as usize];
    (img_axis, p_neg != img_neg)
}

/// Find which element has the given (`x_image`, `z_image`).
#[allow(clippy::cast_possible_truncation)] // loop bound 24 fits in u8
const fn find_element(x_axis: u8, x_neg: bool, z_axis: u8, z_neg: bool) -> u8 {
    let mut k = 0;
    while k < 24 {
        let (ka, kn, kza, kzn) = HEIS[k];
        if ka == x_axis && kn == x_neg && kza == z_axis && kzn == z_neg {
            return k as u8;
        }
        k += 1;
    }
    255 // unreachable for valid inputs
}

// ============================================================================
// Compile-time table computation
// ============================================================================

/// Compose(a, b) = element of matrix b·a.
/// Heisenberg: apply b first, then a.
/// `result_x` = a.apply(b.x), `result_z` = a.apply(b.z).
const fn compute_compose() -> [[u8; 24]; 24] {
    let mut table = [[0u8; 24]; 24];
    let mut i = 0;
    while i < 24 {
        let i_imgs = all_images(i);
        let mut j = 0;
        while j < 24 {
            let (jx, jxn, jz, jzn) = HEIS[j];
            let rx = apply_action(i_imgs, jx, jxn);
            let rz = apply_action(i_imgs, jz, jzn);
            table[i][j] = find_element(rx.0, rx.1, rz.0, rz.1);
            j += 1;
        }
        i += 1;
    }
    table
}

#[allow(clippy::cast_possible_truncation)] // loop bound 24 fits in u8
const fn compute_inverse() -> [u8; 24] {
    let compose = compute_compose();
    let mut inv = [255u8; 24];
    let mut i = 0;
    while i < 24 {
        let mut j = 0;
        while j < 24 {
            if compose[i][j] == 0 {
                inv[i] = j as u8;
                break;
            }
            j += 1;
        }
        i += 1;
    }
    inv
}

/// Determine the coset representative index from unsigned axis pair.
/// The 6 coset reps of the Pauli subgroup are:
///   I(0), S(4), H(6), SH(7), HS(8), SHS(12).
const fn coset_rep_for_axes(x_axis: u8, z_axis: u8) -> u8 {
    match (x_axis, z_axis) {
        (0, 2) => 0,  // identity perm
        (1, 2) => 4,  // S: (XY) swap
        (2, 0) => 6,  // H: (XZ) swap
        (1, 0) => 7,  // SH: (XYZ) cycle
        (2, 1) => 8,  // HS: (XZY) cycle
        (0, 1) => 12, // SHS: (YZ) swap
        _ => 255,     // unreachable
    }
}

/// Decompose each element as Pauli · Coset: `C_matrix` = P · V.
/// compose(V, P) == C (since compose(a,b) = element of b·a).
/// To flush C: apply V physically first, then P.
const fn compute_decompose() -> [(u8, u8); 24] {
    let compose = compute_compose();
    let inv = compute_inverse();
    let mut table = [(0u8, 0u8); 24];
    let mut i = 0;
    while i < 24 {
        let (xa, _, za, _) = HEIS[i];
        let coset = coset_rep_for_axes(xa, za);
        // C = P · V → P = C · V^{-1} as matrix product.
        // compose(a, b) = b·a, so compose(V^{-1}, C) = C·V^{-1} = P.
        // But actually: P_matrix = C_matrix · V_inverse_matrix
        // compose(a, b) = b·a → need compose(v_inv, i) to get i · v_inv
        // Wait: compose(a, b) = b·a. I want C·V^{-1} = P.
        // So I need compose(V^{-1}, C_idx)? No: compose(a, b) = b·a.
        // compose(v_inv, c) = c·v_inv. But I want c·v_inv = P. Hmm.
        // Actually: P = C · V^{-1}. And compose(a, b) gives element of b·a.
        // So I need b·a = C·V^{-1}, meaning b=C, a=V^{-1}.
        // compose(V^{-1}, C) = C · V^{-1} = P. Yes!
        let v_inv = inv[coset as usize];
        let pauli = compose[v_inv as usize][i];
        table[i] = (pauli, coset);
        i += 1;
    }
    table
}

// ============================================================================
// Static tables (computed at compile time)
// ============================================================================

const COMPOSE: [[u8; 24]; 24] = compute_compose();
const INVERSE: [u8; 24] = compute_inverse();
const DECOMPOSE: [(u8, u8); 24] = compute_decompose();

// ============================================================================
// VOP removal decomposition table (for graph state simulator)
// ============================================================================

/// Maximum length of a VOP removal sequence.
const VOP_DECOMP_MAX_LEN: usize = 5;

/// Decomposition of each Clifford into a sequence of LC generators.
///
/// For the graph state simulator's `remove_vop`, each of the 24 Cliffords
/// can be decomposed as a product of two generators:
///   U = local complement on vertex v (right-multiplies v's VOP by SXDG, index 12)
///   V = local complement on neighbor vb (right-multiplies v's VOP by SZ, index 4)
///
/// The sequence is stored as (length, [steps]), where each step is 0=U or 1=V.
/// Steps are applied in reverse order (last step first).
#[allow(clippy::cast_possible_truncation)] // all indices bounded by 24
const fn compute_vop_decomp() -> [(u8, [u8; VOP_DECOMP_MAX_LEN]); 24] {
    // U = SXDG (index 12), V = SZ (index 4)
    // Right-multiplying element e by SXDG: compose[12][e] = e * SXDG as element
    // Right-multiplying element e by SZ: compose[4][e] = e * SZ as element

    // BFS from identity (0) through right-multiplication by SXDG^{-1} and SZ^{-1}
    // (equivalently, searching backward: which elements can reach 0?)
    // Actually: we do forward BFS from 0, applying right-mult by SXDG and SZ.
    // If we reach element C via path g1, g2, ..., gn, it means
    //   0 * g1 * g2 * ... * gn = C, i.e. I * g1 * ... * gn = C
    //   So C = g1 * g2 * ... * gn.
    //   To go from C back to I: C * gn^{-1} * ... * g1^{-1} = I.
    //   For the remove_vop algorithm, we need to apply LCs that right-multiply by
    //   the generators (not their inverses). So we need a different approach.

    // Better: BFS from each element toward identity.
    // From element e, applying generator U (right-mult by SXDG): next = compose[12][e]
    // From element e, applying generator V (right-mult by SZ): next = compose[4][e]
    // We want the shortest path from e to 0.

    // Reverse BFS from 0: predecessors of element `next` under U are elements e
    // such that compose[12][e] = next (e * SXDG = next, so e = next * SXDG^{-1}).
    // Similarly for V. Since SXDG^4 = I (order 4), SXDG^{-1} = SXDG^3.
    // SZ^4 = I, SZ^{-1} = SZ^3 = SZDG.

    // Simpler: compute inverse of generators
    let inv = compute_inverse();
    let sxdg_inv = inv[12]; // SXDG^{-1}
    let sz_inv = inv[4]; // SZ^{-1}

    // For element e: predecessor via U is compose[sxdg_inv][e]? No.
    // If applying U to element p gives e (i.e., compose[12][p] = e, meaning p * SXDG = e),
    // then p = e * SXDG^{-1} = compose[sxdg_inv][e]... wait:
    // compose[a][b] = COMPOSE[a][b] = element of b * a.
    // Hmm no: compose(self=a, gate=b) = COMPOSE[a][b] = element of (b * a).
    // We want p * SXDG = e, so p = e * SXDG^{-1}.
    // e * SXDG^{-1}: this is right-mult of e by SXDG^{-1} = COMPOSE[sxdg_inv][e].
    // Wait: COMPOSE[self][gate] = gate * self. So COMPOSE[sxdg_inv][e] = e * sxdg_inv.
    // Yes!

    // BFS from 0 (identity), expanding via inverse generators.
    // visited[e] = true if we've found the path to e.
    // parent_gen[e] = which generator (0=U, 1=V) was applied to reach e from its parent.
    // parent[e] = the parent element.

    let mut result = [(0u8, [0u8; VOP_DECOMP_MAX_LEN]); 24];
    let mut visited = [false; 24];
    let mut parent = [255u8; 24]; // parent element
    let mut parent_gen = [255u8; 24]; // 0=U, 1=V
    let mut queue = [0u8; 24];
    let mut q_head = 0usize;
    let mut q_tail = 0usize;

    // Start BFS from identity
    visited[0] = true;
    queue[q_tail] = 0;
    q_tail += 1;

    while q_head < q_tail {
        let current = queue[q_head] as usize;
        q_head += 1;

        // Try expanding via U (predecessor under U is: compose[sxdg_inv][current])
        // This represents: neighbor = current * SXDG^{-1}
        // If we go from neighbor by applying U, we get neighbor * SXDG = current
        let u_nbr = COMPOSE[sxdg_inv as usize][current] as usize;
        if !visited[u_nbr] {
            visited[u_nbr] = true;
            parent[u_nbr] = current as u8;
            parent_gen[u_nbr] = 0; // U
            queue[q_tail] = u_nbr as u8;
            q_tail += 1;
        }

        // Try expanding via V
        let v_nbr = COMPOSE[sz_inv as usize][current] as usize;
        if !visited[v_nbr] {
            visited[v_nbr] = true;
            parent[v_nbr] = current as u8;
            parent_gen[v_nbr] = 1; // V
            queue[q_tail] = v_nbr as u8;
            q_tail += 1;
        }
    }

    // Reconstruct paths. For element e, trace back to 0 to get the sequence.
    // The sequence represents: applying gen at e brings us closer to I.
    // parent_gen[e] = the generator that was applied to reach parent[e] from e (so to say).
    // Wait: actually parent[e] is closer to I, and parent_gen[e] is the generator
    // that when applied to e gives parent[e].
    let mut e = 0;
    while e < 24 {
        if e == 0 {
            result[0] = (0, [0; VOP_DECOMP_MAX_LEN]);
        } else {
            let mut path = [0u8; VOP_DECOMP_MAX_LEN];
            let mut len = 0usize;
            let mut cur = e;
            while cur != 0 {
                path[len] = parent_gen[cur];
                len += 1;
                cur = parent[cur] as usize;
            }
            // path[0..len] is the sequence from e toward I (forward order).
            // The remove_vop algorithm should apply these in order:
            // first path[0], then path[1], etc.
            result[e] = (len as u8, path);
        }
        e += 1;
    }

    result
}

/// VOP removal decomposition table.
///
/// `VOP_DECOMP[i]` = `(len, steps)` where `steps[0..len]` are the generators
/// (0=U on vertex, 1=V on neighbor) to apply in order to reduce element `i` to identity.
pub const VOP_DECOMP: [(u8, [u8; VOP_DECOMP_MAX_LEN]); 24] = compute_vop_decomp();

// ============================================================================
// CZ (cphase) lookup table
// ============================================================================

/// Mapping from reference (`GraphSim`) Clifford indices to our `CliffordFrame` indices.
/// Derived by generating all 24 elements from H and S in both systems.
const REF_TO_OURS: [u8; 24] = [
    0, 1, 2, 3, 20, 5, 4, 23, 18, 10, 6, 9, 17, 19, 12, 13, 14, 15, 22, 8, 7, 11, 21, 16,
];

/// Mapping from our `CliffordFrame` indices to reference (`GraphSim`) indices.
const OURS_TO_REF: [u8; 24] = [
    0, 1, 2, 3, 6, 5, 10, 20, 19, 11, 9, 21, 14, 15, 16, 17, 23, 12, 8, 13, 4, 22, 18, 7,
];

/// Reference CZ table from `GraphSim` (Anders & Briegel), indexed by reference indices.
/// Layout: `REF_CPHASE[was_edge][v1_ref][v2_ref]` = `[new_edge, new_v1_ref, new_v2_ref]`.
/// This is the verified table from `cphase.tbl` in the `GraphSim` reference implementation.
#[rustfmt::skip]
const REF_CPHASE: [[[[u8; 3]; 24]; 24]; 2] = [
    // was_edge = 0
    [
        [[1,0,0],[1,0,0],[1,0,3],[1,0,3],[1,0,5],[1,0,5],[1,0,6],[1,0,6],[0,3,8],[0,3,8],[0,0,10],[0,0,10],[1,0,3],[1,0,3],[1,0,0],[1,0,0],[1,0,6],[1,0,6],[1,0,5],[1,0,5],[0,0,10],[0,0,10],[0,3,8],[0,3,8]],
        [[1,0,0],[1,0,0],[1,0,3],[1,0,3],[1,0,5],[1,0,5],[1,0,6],[1,0,6],[0,2,8],[0,2,8],[0,0,10],[0,0,10],[1,0,3],[1,0,3],[1,0,0],[1,0,0],[1,0,6],[1,0,6],[1,0,5],[1,0,5],[0,0,10],[0,0,10],[0,2,8],[0,2,8]],
        [[1,2,3],[1,0,1],[1,0,2],[1,2,0],[1,0,4],[1,2,6],[1,2,5],[1,0,7],[0,0,8],[0,0,8],[0,2,10],[0,2,10],[1,0,2],[1,0,2],[1,0,1],[1,0,1],[1,0,7],[1,0,7],[1,0,4],[1,0,4],[0,2,10],[0,2,10],[0,0,8],[0,0,8]],
        [[1,3,0],[1,0,1],[1,0,2],[1,3,3],[1,0,4],[1,3,5],[1,3,6],[1,0,7],[0,0,8],[0,0,8],[0,3,10],[0,3,10],[1,0,2],[1,0,2],[1,0,1],[1,0,1],[1,0,7],[1,0,7],[1,0,4],[1,0,4],[0,3,10],[0,3,10],[0,0,8],[0,0,8]],
        [[1,4,3],[1,4,3],[1,4,0],[1,4,0],[1,4,6],[1,4,6],[1,4,5],[1,4,5],[0,6,8],[0,6,8],[0,4,10],[0,4,10],[1,4,0],[1,4,0],[1,4,3],[1,4,3],[1,4,5],[1,4,5],[1,4,6],[1,4,6],[0,4,10],[0,4,10],[0,6,8],[0,6,8]],
        [[1,5,0],[1,5,0],[1,5,3],[1,5,3],[1,5,5],[1,5,5],[1,5,6],[1,5,6],[0,6,8],[0,6,8],[0,5,10],[0,5,10],[1,5,3],[1,5,3],[1,5,0],[1,5,0],[1,5,6],[1,5,6],[1,5,5],[1,5,5],[0,5,10],[0,5,10],[0,6,8],[0,6,8]],
        [[1,6,0],[1,5,1],[1,5,2],[1,6,3],[1,5,4],[1,6,5],[1,6,6],[1,5,7],[0,5,8],[0,5,8],[0,6,10],[0,6,10],[1,5,2],[1,5,2],[1,5,1],[1,5,1],[1,5,7],[1,5,7],[1,5,4],[1,5,4],[0,6,10],[0,6,10],[0,5,8],[0,5,8]],
        [[1,6,0],[1,4,2],[1,4,1],[1,6,3],[1,4,7],[1,6,5],[1,6,6],[1,4,4],[0,4,8],[0,4,8],[0,6,10],[0,6,10],[1,4,1],[1,4,1],[1,4,2],[1,4,2],[1,4,4],[1,4,4],[1,4,7],[1,4,7],[0,6,10],[0,6,10],[0,4,8],[0,4,8]],
        [[0,8,3],[0,8,2],[0,8,0],[0,8,0],[0,8,6],[0,8,6],[0,8,5],[0,8,4],[0,8,8],[0,8,8],[0,8,10],[0,8,10],[0,8,0],[0,8,0],[0,8,2],[0,8,2],[0,8,4],[0,8,4],[0,8,6],[0,8,6],[0,8,10],[0,8,10],[0,8,8],[0,8,8]],
        [[0,8,3],[0,8,2],[0,8,0],[0,8,0],[0,8,6],[0,8,6],[0,8,5],[0,8,4],[0,8,8],[0,8,8],[0,8,10],[0,8,10],[0,8,0],[0,8,0],[0,8,2],[0,8,2],[0,8,4],[0,8,4],[0,8,6],[0,8,6],[0,8,10],[0,8,10],[0,8,8],[0,8,8]],
        [[0,10,0],[0,10,0],[0,10,2],[0,10,3],[0,10,4],[0,10,5],[0,10,6],[0,10,6],[0,10,8],[0,10,8],[0,10,10],[0,10,10],[0,10,2],[0,10,2],[0,10,0],[0,10,0],[0,10,6],[0,10,6],[0,10,4],[0,10,4],[0,10,10],[0,10,10],[0,10,8],[0,10,8]],
        [[0,10,0],[0,10,0],[0,10,2],[0,10,3],[0,10,4],[0,10,5],[0,10,6],[0,10,6],[0,10,8],[0,10,8],[0,10,10],[0,10,10],[0,10,2],[0,10,2],[0,10,0],[0,10,0],[0,10,6],[0,10,6],[0,10,4],[0,10,4],[0,10,10],[0,10,10],[0,10,8],[0,10,8]],
        [[1,2,3],[1,0,1],[1,0,2],[1,2,0],[1,0,4],[1,2,6],[1,2,5],[1,0,7],[0,0,8],[0,0,8],[0,2,10],[0,2,10],[1,0,2],[1,0,2],[1,0,1],[1,0,1],[1,0,7],[1,0,7],[1,0,4],[1,0,4],[0,2,10],[0,2,10],[0,0,8],[0,0,8]],
        [[1,2,3],[1,0,1],[1,0,2],[1,2,0],[1,0,4],[1,2,6],[1,2,5],[1,0,7],[0,0,8],[0,0,8],[0,2,10],[0,2,10],[1,0,2],[1,0,2],[1,0,1],[1,0,1],[1,0,7],[1,0,7],[1,0,4],[1,0,4],[0,2,10],[0,2,10],[0,0,8],[0,0,8]],
        [[1,0,0],[1,0,0],[1,0,3],[1,0,3],[1,0,5],[1,0,5],[1,0,6],[1,0,6],[0,2,8],[0,2,8],[0,0,10],[0,0,10],[1,0,3],[1,0,3],[1,0,0],[1,0,0],[1,0,6],[1,0,6],[1,0,5],[1,0,5],[0,0,10],[0,0,10],[0,2,8],[0,2,8]],
        [[1,0,0],[1,0,0],[1,0,3],[1,0,3],[1,0,5],[1,0,5],[1,0,6],[1,0,6],[0,2,8],[0,2,8],[0,0,10],[0,0,10],[1,0,3],[1,0,3],[1,0,0],[1,0,0],[1,0,6],[1,0,6],[1,0,5],[1,0,5],[0,0,10],[0,0,10],[0,2,8],[0,2,8]],
        [[1,6,0],[1,4,2],[1,4,1],[1,6,3],[1,4,7],[1,6,5],[1,6,6],[1,4,4],[0,4,8],[0,4,8],[0,6,10],[0,6,10],[1,4,1],[1,4,1],[1,4,2],[1,4,2],[1,4,4],[1,4,4],[1,4,7],[1,4,7],[0,6,10],[0,6,10],[0,4,8],[0,4,8]],
        [[1,6,0],[1,4,2],[1,4,1],[1,6,3],[1,4,7],[1,6,5],[1,6,6],[1,4,4],[0,4,8],[0,4,8],[0,6,10],[0,6,10],[1,4,1],[1,4,1],[1,4,2],[1,4,2],[1,4,4],[1,4,4],[1,4,7],[1,4,7],[0,6,10],[0,6,10],[0,4,8],[0,4,8]],
        [[1,4,3],[1,4,3],[1,4,0],[1,4,0],[1,4,6],[1,4,6],[1,4,5],[1,4,5],[0,6,8],[0,6,8],[0,4,10],[0,4,10],[1,4,0],[1,4,0],[1,4,3],[1,4,3],[1,4,5],[1,4,5],[1,4,6],[1,4,6],[0,4,10],[0,4,10],[0,6,8],[0,6,8]],
        [[1,4,3],[1,4,3],[1,4,0],[1,4,0],[1,4,6],[1,4,6],[1,4,5],[1,4,5],[0,6,8],[0,6,8],[0,4,10],[0,4,10],[1,4,0],[1,4,0],[1,4,3],[1,4,3],[1,4,5],[1,4,5],[1,4,6],[1,4,6],[0,4,10],[0,4,10],[0,6,8],[0,6,8]],
        [[0,10,0],[0,10,0],[0,10,2],[0,10,3],[0,10,4],[0,10,5],[0,10,6],[0,10,6],[0,10,8],[0,10,8],[0,10,10],[0,10,10],[0,10,2],[0,10,2],[0,10,0],[0,10,0],[0,10,6],[0,10,6],[0,10,4],[0,10,4],[0,10,10],[0,10,10],[0,10,8],[0,10,8]],
        [[0,10,0],[0,10,0],[0,10,2],[0,10,3],[0,10,4],[0,10,5],[0,10,6],[0,10,6],[0,10,8],[0,10,8],[0,10,10],[0,10,10],[0,10,2],[0,10,2],[0,10,0],[0,10,0],[0,10,6],[0,10,6],[0,10,4],[0,10,4],[0,10,10],[0,10,10],[0,10,8],[0,10,8]],
        [[0,8,3],[0,8,2],[0,8,0],[0,8,0],[0,8,6],[0,8,6],[0,8,5],[0,8,4],[0,8,8],[0,8,8],[0,8,10],[0,8,10],[0,8,0],[0,8,0],[0,8,2],[0,8,2],[0,8,4],[0,8,4],[0,8,6],[0,8,6],[0,8,10],[0,8,10],[0,8,8],[0,8,8]],
        [[0,8,3],[0,8,2],[0,8,0],[0,8,0],[0,8,6],[0,8,6],[0,8,5],[0,8,4],[0,8,8],[0,8,8],[0,8,10],[0,8,10],[0,8,0],[0,8,0],[0,8,2],[0,8,2],[0,8,4],[0,8,4],[0,8,6],[0,8,6],[0,8,10],[0,8,10],[0,8,8],[0,8,8]],
    ],
    // was_edge = 1
    [
        [[0,0,0],[0,3,0],[0,3,2],[0,0,3],[0,3,4],[0,0,5],[0,0,6],[0,3,6],[1,5,23],[1,5,22],[1,5,21],[1,5,20],[0,5,2],[0,6,2],[0,5,0],[0,6,0],[0,6,6],[0,5,6],[0,6,4],[0,5,4],[1,5,10],[1,5,11],[1,5,8],[1,5,9]],
        [[0,0,3],[0,2,2],[0,2,0],[0,0,0],[0,2,6],[0,0,6],[0,0,5],[0,2,4],[1,4,23],[1,4,22],[1,4,21],[1,4,20],[0,6,0],[0,4,0],[0,6,2],[0,4,2],[0,4,4],[0,6,4],[0,4,6],[0,6,6],[1,4,10],[1,4,11],[1,4,8],[1,4,9]],
        [[0,2,3],[0,0,2],[0,0,0],[0,2,0],[0,0,6],[0,2,6],[0,2,5],[0,0,4],[1,4,22],[1,4,23],[1,4,20],[1,4,21],[0,4,0],[0,6,0],[0,4,2],[0,6,2],[0,6,4],[0,4,4],[0,6,6],[0,4,6],[1,4,11],[1,4,10],[1,4,9],[1,4,8]],
        [[0,3,0],[0,0,0],[0,0,2],[0,3,3],[0,0,4],[0,3,5],[0,3,6],[0,0,6],[1,5,22],[1,5,23],[1,5,20],[1,5,21],[0,6,2],[0,5,2],[0,6,0],[0,5,0],[0,5,6],[0,6,6],[0,5,4],[0,6,4],[1,5,11],[1,5,10],[1,5,9],[1,5,8]],
        [[0,4,3],[0,6,2],[0,6,0],[0,4,0],[0,6,6],[0,4,6],[0,4,5],[0,6,4],[1,0,21],[1,0,20],[1,0,23],[1,0,22],[0,0,0],[0,2,0],[0,0,2],[0,2,2],[0,2,4],[0,0,4],[0,2,6],[0,0,6],[1,0,8],[1,0,9],[1,0,10],[1,0,11]],
        [[0,5,0],[0,6,0],[0,6,2],[0,5,3],[0,6,4],[0,5,5],[0,5,6],[0,6,6],[1,0,22],[1,0,23],[1,0,20],[1,0,21],[0,3,2],[0,0,2],[0,3,0],[0,0,0],[0,0,6],[0,3,6],[0,0,4],[0,3,4],[1,0,11],[1,0,10],[1,0,9],[1,0,8]],
        [[0,6,0],[0,5,0],[0,5,2],[0,6,3],[0,5,4],[0,6,5],[0,6,6],[0,5,6],[1,0,23],[1,0,22],[1,0,21],[1,0,20],[0,0,2],[0,3,2],[0,0,0],[0,3,0],[0,3,6],[0,0,6],[0,3,4],[0,0,4],[1,0,10],[1,0,11],[1,0,8],[1,0,9]],
        [[0,6,3],[0,4,2],[0,4,0],[0,6,0],[0,4,6],[0,6,6],[0,6,5],[0,4,4],[1,0,20],[1,0,21],[1,0,22],[1,0,23],[0,2,0],[0,0,0],[0,2,2],[0,0,2],[0,0,4],[0,2,4],[0,0,6],[0,2,6],[1,0,9],[1,0,8],[1,0,11],[1,0,10]],
        [[1,22,6],[1,20,5],[1,20,6],[1,22,5],[1,20,3],[1,22,0],[1,22,3],[1,20,0],[0,0,0],[0,0,2],[0,2,2],[0,2,0],[0,6,6],[0,4,4],[0,6,4],[0,4,6],[0,4,2],[0,6,0],[0,4,0],[0,6,2],[0,2,4],[0,2,6],[0,0,6],[0,0,4]],
        [[1,22,5],[1,20,6],[1,20,5],[1,22,6],[1,20,0],[1,22,3],[1,22,0],[1,20,3],[0,2,0],[0,2,2],[0,0,2],[0,0,0],[0,4,6],[0,6,4],[0,4,4],[0,6,6],[0,6,2],[0,4,0],[0,6,0],[0,4,2],[0,0,4],[0,0,6],[0,2,6],[0,2,4]],
        [[1,20,6],[1,20,7],[1,20,4],[1,20,5],[1,20,1],[1,20,0],[1,20,3],[1,20,2],[0,2,2],[0,2,0],[0,0,0],[0,0,2],[0,6,4],[0,4,6],[0,6,6],[0,4,4],[0,4,0],[0,6,2],[0,4,2],[0,6,0],[0,0,6],[0,0,4],[0,2,4],[0,2,6]],
        [[1,20,5],[1,20,4],[1,20,7],[1,20,6],[1,20,2],[1,20,3],[1,20,0],[1,20,1],[0,0,2],[0,0,0],[0,2,0],[0,2,2],[0,4,4],[0,6,6],[0,4,6],[0,6,4],[0,6,0],[0,4,2],[0,6,2],[0,4,0],[0,2,6],[0,2,4],[0,0,4],[0,0,6]],
        [[0,2,5],[0,0,6],[0,0,4],[0,2,6],[0,0,0],[0,2,3],[0,2,0],[0,0,2],[0,6,6],[0,6,4],[0,4,6],[0,4,4],[1,16,18],[1,16,19],[1,16,16],[1,16,17],[1,16,12],[1,16,13],[1,16,14],[1,16,15],[0,4,2],[0,4,0],[0,6,2],[0,6,0]],
        [[0,2,6],[0,0,4],[0,0,6],[0,2,5],[0,0,2],[0,2,0],[0,2,3],[0,0,0],[0,4,4],[0,4,6],[0,6,4],[0,6,6],[1,16,17],[1,16,16],[1,16,19],[1,16,18],[1,16,15],[1,16,14],[1,16,13],[1,16,12],[0,6,0],[0,6,2],[0,4,0],[0,4,2]],
        [[0,0,5],[0,2,6],[0,2,4],[0,0,6],[0,2,0],[0,0,3],[0,0,0],[0,2,2],[0,4,6],[0,4,4],[0,6,6],[0,6,4],[1,16,16],[1,16,17],[1,16,18],[1,16,19],[1,16,14],[1,16,15],[1,16,12],[1,16,13],[0,6,2],[0,6,0],[0,4,2],[0,4,0]],
        [[0,0,6],[0,2,4],[0,2,6],[0,0,5],[0,2,2],[0,0,0],[0,0,3],[0,2,0],[0,6,4],[0,6,6],[0,4,4],[0,4,6],[1,16,19],[1,16,18],[1,16,17],[1,16,16],[1,16,13],[1,16,12],[1,16,15],[1,16,14],[0,4,0],[0,4,2],[0,6,0],[0,6,2]],
        [[0,6,6],[0,4,4],[0,4,6],[0,6,5],[0,4,2],[0,6,0],[0,6,3],[0,4,0],[0,2,4],[0,2,6],[0,0,4],[0,0,6],[1,12,16],[1,12,17],[1,12,18],[1,12,19],[1,12,14],[1,12,15],[1,12,12],[1,12,13],[0,0,0],[0,0,2],[0,2,0],[0,2,2]],
        [[0,6,5],[0,4,6],[0,4,4],[0,6,6],[0,4,0],[0,6,3],[0,6,0],[0,4,2],[0,0,6],[0,0,4],[0,2,6],[0,2,4],[1,12,19],[1,12,18],[1,12,17],[1,12,16],[1,12,13],[1,12,12],[1,12,15],[1,12,14],[0,2,2],[0,2,0],[0,0,2],[0,0,0]],
        [[0,4,6],[0,6,4],[0,6,6],[0,4,5],[0,6,2],[0,4,0],[0,4,3],[0,6,0],[0,0,4],[0,0,6],[0,2,4],[0,2,6],[1,12,18],[1,12,19],[1,12,16],[1,12,17],[1,12,12],[1,12,13],[1,12,14],[1,12,15],[0,2,0],[0,2,2],[0,0,0],[0,0,2]],
        [[0,4,5],[0,6,6],[0,6,4],[0,4,6],[0,6,0],[0,4,3],[0,4,0],[0,6,2],[0,2,6],[0,2,4],[0,0,6],[0,0,4],[1,12,17],[1,12,16],[1,12,19],[1,12,18],[1,12,15],[1,12,14],[1,12,13],[1,12,12],[0,0,2],[0,0,0],[0,2,2],[0,2,0]],
        [[1,10,5],[1,8,6],[1,8,5],[1,10,6],[1,8,0],[1,10,3],[1,10,0],[1,8,3],[0,4,2],[0,4,0],[0,6,0],[0,6,2],[0,2,4],[0,0,6],[0,2,6],[0,0,4],[0,0,0],[0,2,2],[0,0,2],[0,2,0],[0,6,6],[0,6,4],[0,4,4],[0,4,6]],
        [[1,10,6],[1,8,5],[1,8,6],[1,10,5],[1,8,3],[1,10,0],[1,10,3],[1,8,0],[0,6,2],[0,6,0],[0,4,0],[0,4,2],[0,0,4],[0,2,6],[0,0,6],[0,2,4],[0,2,0],[0,0,2],[0,2,2],[0,0,0],[0,4,6],[0,4,4],[0,6,4],[0,6,6]],
        [[1,8,5],[1,8,4],[1,8,7],[1,8,6],[1,8,2],[1,8,3],[1,8,0],[1,8,1],[0,6,0],[0,6,2],[0,4,2],[0,4,0],[0,2,6],[0,0,4],[0,2,4],[0,0,6],[0,0,2],[0,2,0],[0,0,0],[0,2,2],[0,4,4],[0,4,6],[0,6,6],[0,6,4]],
        [[1,8,6],[1,8,7],[1,8,4],[1,8,5],[1,8,1],[1,8,0],[1,8,3],[1,8,2],[0,4,0],[0,4,2],[0,6,2],[0,6,0],[0,0,6],[0,2,4],[0,0,4],[0,2,6],[0,2,2],[0,0,0],[0,2,0],[0,0,2],[0,6,4],[0,6,6],[0,4,6],[0,4,4]],
    ],
];

/// Compute the CZ lookup table by remapping the reference `GraphSim` table
/// to our `CliffordFrame` index system.
///
/// For each `(was_edge, v1, v2)`, finds `(new_edge, v1', v2')` such that
/// after applying CZ to state `(V1 x V2) |G_{was_edge}>`,
/// the result is `(V1' x V2') |G_{new_edge}>`.
///
/// Layout: `[was_edge * 24 + v1][v2]` = `[new_edge, v1', v2']`.
const fn compute_cphase_table() -> [[[u8; 3]; 24]; 48] {
    let mut table = [[[0u8; 3]; 24]; 48];

    let mut we: usize = 0;
    while we < 2 {
        let mut v1: usize = 0;
        while v1 < 24 {
            let mut v2: usize = 0;
            while v2 < 24 {
                // Map our indices to reference indices
                let rv1 = OURS_TO_REF[v1] as usize;
                let rv2 = OURS_TO_REF[v2] as usize;

                // Look up in reference table
                let [ne, rnv1, rnv2] = REF_CPHASE[we][rv1][rv2];

                // Map reference result indices back to our indices
                let onv1 = REF_TO_OURS[rnv1 as usize];
                let onv2 = REF_TO_OURS[rnv2 as usize];

                table[we * 24 + v1][v2] = [ne, onv1, onv2];
                v2 += 1;
            }
            v1 += 1;
        }
        we += 1;
    }

    table
}

/// CZ lookup table, computed at compile time.
///
/// `CPHASE_TBL[was_edge * 24 + vop1][vop2]` = `[new_edge, new_vop1, new_vop2]`
pub const CPHASE_TBL: [[[u8; 3]; 24]; 48] = compute_cphase_table();

// ============================================================================
// Phase cocycle tables for exact phase tracking
// ============================================================================

/// The 8th roots of unity: e^{ikπ/4} for k = 0..7, stored as [re, im].
pub const PHASE_ROOTS: [[f64; 2]; 8] = {
    const R: f64 = std::f64::consts::FRAC_1_SQRT_2;
    [
        [1.0, 0.0],  // k=0: 1
        [R, R],      // k=1: e^{iπ/4}
        [0.0, 1.0],  // k=2: i
        [-R, R],     // k=3: e^{i3π/4}
        [-1.0, 0.0], // k=4: -1
        [-R, -R],    // k=5: e^{i5π/4}
        [0.0, -1.0], // k=6: -i
        [R, -R],     // k=7: e^{i7π/4}
    ]
};

/// Representative 2x2 unitary matrix for each of the 24 elements.
/// Layout: [`a_re`, `a_im`, `b_re`, `b_im`, `c_re`, `c_im`, `d_re`, `d_im`] for [[a,b],[c,d]].
/// Computed from generator sequences (H, S).
pub const ELEMENT_MATRIX: [[f64; 8]; 24] = {
    const R: f64 = std::f64::consts::FRAC_1_SQRT_2;
    const H: f64 = 0.5;
    [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  //  0: I
        [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],  //  1: X
        [0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0], //  2: Y (= iXZ, differs from std Y by phase -i)
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0], //  3: Z
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  //  4: S
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], //  5: Sdg
        [R, 0.0, R, 0.0, R, 0.0, -R, 0.0],         //  6: H
        [R, 0.0, R, 0.0, 0.0, R, 0.0, -R],         //  7: SH
        [R, 0.0, 0.0, R, R, 0.0, 0.0, -R],         //  8: HS
        [R, 0.0, R, 0.0, -R, 0.0, R, 0.0],         //  9: ZH (= SYdg rep)
        [R, 0.0, -R, 0.0, R, 0.0, R, 0.0],         // 10: HZ (= SY rep)
        [R, 0.0, R, 0.0, 0.0, -R, 0.0, R],         // 11: SdgH
        [R, 0.0, 0.0, R, 0.0, R, R, 0.0],          // 12: SHS (= SXdg rep)
        [H, H, H, -H, H, -H, H, H],                // 13: HSH (= SX)
        [H, H, H, -H, H, H, -H, H],                // 14: SHSH
        [R, 0.0, 0.0, R, -R, 0.0, 0.0, R],         // 15: S^2HS
        [R, 0.0, -R, 0.0, 0.0, R, 0.0, R],         // 16: SHS^2
        [R, 0.0, 0.0, R, 0.0, -R, -R, 0.0],        // 17: S^3HS
        [R, 0.0, -R, 0.0, -R, 0.0, -R, 0.0],       // 18: S^2HS^2
        [H, H, H, -H, -H, H, -H, -H],              // 19: S^2HSH
        [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],  // 20: HS^2HS
        [R, 0.0, -R, 0.0, 0.0, -R, 0.0, -R],       // 21: S^3HS^2
        [H, H, H, -H, -H, -H, H, -H],              // 22: S^3HSH
        [0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0], // 23: HS^2HS^3
    ]
};

/// Phase cocycle table: `PHASE_COCYCLE`[frame][gate] gives the phase correction
/// (as an 8th-root index 0-7) when composing `gate` onto `frame`.
///
/// For element matrices `M_i` and `M_j`:
///   `M_j` · `M_i` = `e^{i·PHASE_COCYCLE`[i][j]·π/4} · M_{COMPOSE[i][j]}
pub const PHASE_COCYCLE: [[u8; 24]; 24] = [
    [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ],
    [
        0, 0, 0, 0, 2, 6, 0, 0, 1, 0, 4, 0, 1, 7, 7, 1, 4, 1, 4, 7, 2, 4, 7, 6,
    ],
    [
        0, 4, 4, 0, 6, 2, 4, 4, 6, 4, 0, 4, 6, 2, 2, 6, 0, 6, 0, 2, 6, 0, 2, 2,
    ],
    [
        0, 4, 4, 0, 0, 0, 0, 0, 7, 0, 0, 0, 7, 1, 1, 7, 0, 7, 0, 1, 0, 0, 1, 0,
    ],
    [
        0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 1, 1, 0, 7, 0, 7, 1, 4, 7, 1, 0,
    ],
    [
        0, 0, 4, 0, 0, 0, 7, 7, 0, 7, 0, 7, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 4,
    ],
    [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 6, 0, 6,
    ],
    [
        0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 7, 0, 2, 1, 1, 0, 7, 6, 7, 1, 4, 7, 1, 0,
    ],
    [
        0, 7, 7, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 2, 1, 4, 0, 1, 2, 0, 5,
    ],
    [
        0, 4, 4, 0, 0, 0, 0, 2, 7, 0, 0, 6, 7, 1, 1, 7, 0, 7, 0, 1, 6, 0, 1, 2,
    ],
    [
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 4, 0, 1, 7, 7, 1, 2, 1, 4, 7, 2, 6, 7, 6,
    ],
    [
        0, 6, 6, 0, 0, 0, 7, 7, 0, 7, 0, 7, 0, 7, 1, 0, 0, 0, 0, 3, 0, 0, 1, 4,
    ],
    [
        0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 7, 1, 2, 1, 1, 4, 7, 2, 7, 1, 3, 7, 1, 7,
    ],
    [
        0, 7, 7, 0, 0, 0, 0, 0, 1, 0, 2, 0, 1, 0, 2, 1, 2, 1, 2, 0, 1, 2, 6, 5,
    ],
    [
        0, 1, 1, 0, 0, 0, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 7, 2, 5, 1, 3, 7, 1, 7,
    ],
    [
        0, 3, 3, 0, 0, 0, 0, 2, 7, 4, 0, 2, 7, 2, 2, 7, 0, 7, 0, 2, 5, 0, 2, 1,
    ],
    [
        0, 2, 2, 0, 0, 0, 1, 1, 4, 1, 6, 1, 2, 1, 1, 4, 6, 6, 6, 1, 4, 6, 1, 0,
    ],
    [
        0, 5, 5, 0, 0, 0, 7, 7, 0, 7, 1, 7, 0, 3, 1, 0, 1, 0, 1, 3, 7, 1, 5, 3,
    ],
    [
        0, 4, 4, 0, 0, 0, 4, 2, 6, 4, 0, 6, 6, 2, 2, 6, 0, 6, 0, 2, 6, 0, 2, 2,
    ],
    [
        0, 3, 3, 0, 0, 0, 2, 2, 1, 2, 0, 2, 7, 2, 2, 5, 0, 7, 0, 2, 5, 0, 2, 1,
    ],
    [
        0, 0, 0, 4, 2, 2, 0, 0, 2, 0, 3, 0, 2, 7, 7, 2, 3, 2, 3, 7, 2, 3, 7, 6,
    ],
    [
        0, 6, 6, 0, 0, 0, 6, 6, 0, 6, 1, 6, 0, 7, 5, 0, 1, 0, 1, 3, 0, 1, 5, 4,
    ],
    [
        0, 5, 5, 0, 0, 0, 1, 7, 0, 5, 1, 7, 0, 3, 3, 0, 1, 0, 1, 3, 7, 1, 3, 3,
    ],
    [
        0, 0, 0, 4, 6, 6, 7, 7, 2, 7, 4, 7, 2, 7, 7, 2, 4, 2, 4, 7, 2, 4, 7, 6,
    ],
];

/// Phase correction from standard gate matrix to generator-based element matrix.
/// `standard_gate_matrix` = `e^{i·GATE_PHASE_DELTA`[idx]·π/4} · `ELEMENT_MATRIX`[idx]
///
/// Only entries for indices used as gates matter:
///   0(I)=0, 1(X)=0, 2(Y)=6, 3(Z)=0, 4(S)=0, 5(Sdg)=0, 6(H)=0,
///   9(SYdg)=7, 10(SY)=1, 12(SXdg)=7, 13(SX)=0
pub const GATE_PHASE_DELTA: [u8; 24] = [
    0, 0, 6, 0, 0, 0, 0, 0, 0, 7, 1, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];

// ============================================================================
// CliffordFrame implementation
// ============================================================================

impl CliffordFrame {
    // Named constants for all single-qubit Cliffords used as gates
    pub const IDENTITY: Self = Self(0);
    pub const X: Self = Self(1);
    pub const Y: Self = Self(2);
    pub const Z: Self = Self(3);
    pub const SZ: Self = Self(4);
    pub const SZDG: Self = Self(5);
    pub const H: Self = Self(6);
    pub const SX: Self = Self(13); // HSH
    pub const SXDG: Self = Self(12); // SHS
    pub const SY: Self = Self(10); // HZ = HS²
    pub const SYDG: Self = Self(9); // ZH = S²H

    /// Compose: `new_frame` = frame.compose(gate).
    ///
    /// Returns the frame after applying `gate` to a qubit with accumulated
    /// frame `self`. Corresponds to the matrix product gate · self.
    #[inline]
    #[must_use]
    pub fn compose(self, gate: Self) -> Self {
        Self(COMPOSE[self.0 as usize][gate.0 as usize])
    }

    /// Inverse of this Clifford.
    #[inline]
    #[must_use]
    pub fn inverse(self) -> Self {
        Self(INVERSE[self.0 as usize])
    }

    /// Where this Clifford maps the Z axis (Heisenberg picture: C†ZC).
    #[inline]
    #[must_use]
    pub fn z_image(self) -> SignedPauli {
        let (_, _, za, zn) = HEIS[self.0 as usize];
        SignedPauli {
            axis: axis_from_u8(za),
            positive: !zn,
        }
    }

    /// Where this Clifford maps the X axis (Heisenberg picture: C†XC).
    #[inline]
    #[must_use]
    pub fn x_image(self) -> SignedPauli {
        let (xa, xn, _, _) = HEIS[self.0 as usize];
        SignedPauli {
            axis: axis_from_u8(xa),
            positive: !xn,
        }
    }

    /// Where this Clifford maps the Y axis (derived from X and Z images).
    #[inline]
    #[must_use]
    pub fn y_image(self) -> SignedPauli {
        let (xa, xn, za, zn) = HEIS[self.0 as usize];
        let (ya, yn) = y_image(xa, xn, za, zn);
        SignedPauli {
            axis: axis_from_u8(ya),
            positive: !yn,
        }
    }

    /// Whether this is a Pauli gate (I, X, Y, or Z).
    #[inline]
    #[must_use]
    pub fn is_pauli(self) -> bool {
        self.0 < 4
    }

    /// Whether this is the identity.
    #[inline]
    #[must_use]
    pub fn is_identity(self) -> bool {
        self.0 == 0
    }

    /// Whether this Clifford is diagonal in the computational basis.
    ///
    /// A Clifford is diagonal iff its Z-image axis is Z (maps Z to +/-Z).
    /// The four diagonal Cliffords are: I, Z, S, Sdg.
    #[inline]
    #[must_use]
    pub fn is_diagonal(self) -> bool {
        HEIS[self.0 as usize].2 == 2 // z_axis == Z
    }

    /// Decompose into Pauli × Coset: `self_matrix` = pauli · coset.
    ///
    /// The coset representative is one of {I, S, H, SH, HS, SHS}.
    /// To physically apply this Clifford: first apply coset, then pauli.
    #[inline]
    #[must_use]
    pub fn decompose_pauli_coset(self) -> (Self, Self) {
        let (p, c) = DECOMPOSE[self.0 as usize];
        (Self(p), Self(c))
    }

    /// Get the raw index (for debugging/testing).
    #[inline]
    #[must_use]
    pub fn index(self) -> u8 {
        self.0
    }

    /// Construct from a raw index. Only valid for indices 0..24.
    #[inline]
    #[must_use]
    pub fn from_index(idx: u8) -> Self {
        debug_assert!(idx < 24, "CliffordFrame index out of range: {idx}");
        Self(idx)
    }

    /// Pauli symplectic representation: (`x_bit`, `z_bit`).
    /// Only valid for Pauli elements (index 0-3).
    /// I=(false,false), X=(true,false), Y=(true,true), Z=(false,true).
    #[inline]
    #[must_use]
    pub fn pauli_xz_bits(self) -> (bool, bool) {
        const PAULI_XZ: [(bool, bool); 4] =
            [(false, false), (true, false), (true, true), (false, true)];
        debug_assert!(self.is_pauli());
        PAULI_XZ[self.0 as usize]
    }

    /// Construct a Pauli from symplectic (`x_bit`, `z_bit`) representation.
    #[inline]
    #[must_use]
    pub fn pauli_from_xz(x: bool, z: bool) -> Self {
        const XZ_TO_IDX: [[u8; 2]; 2] = [[0, 3], [1, 2]]; // [x][z]
        Self(XZ_TO_IDX[usize::from(x)][usize::from(z)])
    }

    /// Push this Pauli frame through a CX gate (Heisenberg picture).
    ///
    /// Given Pauli frames (ctrl, targ) before CX, returns (ctrl', targ') after.
    /// Only valid when both self and `targ_frame` are Paulis.
    ///
    /// CX propagation rules (symplectic):
    ///   xc' = xc,  zc' = zc ⊕ zt,  xt' = xc ⊕ xt,  zt' = zt
    #[inline]
    #[must_use]
    pub fn push_through_cx(ctrl_pauli: Self, targ_pauli: Self) -> (Self, Self, u8) {
        // Phase: CX†(P1⊗P2)CX = phase * P1'⊗P2'. Nonzero only for XZ→YY and YY→XZ.
        // Lookup: index = ctrl_pauli * 4 + targ_pauli (both 0-3).
        const CX_PHASE: [u8; 16] = [0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 4, 0, 0, 0, 0, 0];
        let (xc, zc) = ctrl_pauli.pauli_xz_bits();
        let (xt, zt) = targ_pauli.pauli_xz_bits();
        let new_ctrl = Self::pauli_from_xz(xc, zc ^ zt);
        let new_targ = Self::pauli_from_xz(xc ^ xt, zt);
        let phase = CX_PHASE[ctrl_pauli.0 as usize * 4 + targ_pauli.0 as usize];
        (new_ctrl, new_targ, phase)
    }

    /// Push Pauli frames through a CZ gate.
    ///
    /// CZ propagation (symplectic):
    ///   xc' = xc,  zc' = zc ⊕ xt,  xt' = xt,  zt' = zt ⊕ xc
    #[inline]
    #[must_use]
    pub fn push_through_cz(ctrl_pauli: Self, targ_pauli: Self) -> (Self, Self, u8) {
        // Phase: nonzero only for XY→YX and YX→XY.
        const CZ_PHASE: [u8; 16] = [0, 0, 0, 0, 0, 0, 4, 0, 0, 4, 0, 0, 0, 0, 0, 0];
        let (xc, zc) = ctrl_pauli.pauli_xz_bits();
        let (xt, zt) = targ_pauli.pauli_xz_bits();
        let new_ctrl = Self::pauli_from_xz(xc, zc ^ xt);
        let new_targ = Self::pauli_from_xz(xt, zt ^ xc);
        let phase = CZ_PHASE[ctrl_pauli.0 as usize * 4 + targ_pauli.0 as usize];
        (new_ctrl, new_targ, phase)
    }

    /// Push Pauli frames through a SWAP gate.
    ///
    /// SWAP simply exchanges the two frames.
    #[inline]
    #[must_use]
    pub fn push_through_swap(ctrl_pauli: Self, targ_pauli: Self) -> (Self, Self) {
        (targ_pauli, ctrl_pauli)
    }

    /// Push Pauli frames through an SZZ gate. Returns (`new_p1`, `new_p2`, phase).
    ///
    /// SZZ = exp(-iπ/4 ZZ). Symplectic:
    ///   x1' = x1,  z1' = z1 ⊕ x1 ⊕ x2,  x2' = x2,  z2' = z2 ⊕ x1 ⊕ x2
    /// Element-convention phase: i when exactly one x-bit is set (index 2),
    /// identity otherwise.
    #[inline]
    #[must_use]
    pub fn push_through_szz(p1: Self, p2: Self) -> (Self, Self, u8) {
        let (x1, z1) = p1.pauli_xz_bits();
        let (x2, z2) = p2.pauli_xz_bits();
        let x_xor = x1 ^ x2;
        let new_p1 = Self::pauli_from_xz(x1, z1 ^ x_xor);
        let new_p2 = Self::pauli_from_xz(x2, z2 ^ x_xor);
        let phase = if x_xor { 2 } else { 0 };
        (new_p1, new_p2, phase)
    }

    /// Push Pauli frames through an iSWAP gate. Returns (`new_p1`, `new_p2`, phase).
    ///
    /// iSWAP = SWAP · diag(1,i,i,1)-like. Combines SWAP with SZZ-like z-update.
    /// Symplectic:
    ///   x1' = x2,  z1' = z2 ⊕ x1 ⊕ x2
    ///   x2' = x1,  z2' = z1 ⊕ x1 ⊕ x2
    /// Element-convention phase: 2*(x1 XOR x2).
    #[inline]
    #[must_use]
    pub fn push_through_iswap(p1: Self, p2: Self) -> (Self, Self, u8) {
        let (x1, z1) = p1.pauli_xz_bits();
        let (x2, z2) = p2.pauli_xz_bits();
        let x_xor = x1 ^ x2;
        let new_p1 = Self::pauli_from_xz(x2, z2 ^ x_xor);
        let new_p2 = Self::pauli_from_xz(x1, z1 ^ x_xor);
        let phase = if x_xor { 2 } else { 0 };
        (new_p1, new_p2, phase)
    }

    /// Push Pauli frames through an SXX gate. Returns (`new_p1`, `new_p2`, phase).
    ///
    /// SXX = exp(-iπ/4 XX) = (H⊗H)·SZZ·(H⊗H). Symplectic:
    ///   x1' = x1 ⊕ z1 ⊕ z2,  z1' = z1
    ///   x2' = x2 ⊕ z1 ⊕ z2,  z2' = z2
    /// Element-convention phase: 6*(z1 XOR z2) mod 8 (= -i when one z-bit set).
    #[inline]
    #[must_use]
    pub fn push_through_sxx(p1: Self, p2: Self) -> (Self, Self, u8) {
        let (x1, z1) = p1.pauli_xz_bits();
        let (x2, z2) = p2.pauli_xz_bits();
        let z_xor = z1 ^ z2;
        let new_p1 = Self::pauli_from_xz(x1 ^ z_xor, z1);
        let new_p2 = Self::pauli_from_xz(x2 ^ z_xor, z2);
        let phase = if z_xor { 6 } else { 0 };
        (new_p1, new_p2, phase)
    }

    /// Push Pauli frames through an SYY gate. Returns (`new_p1`, `new_p2`, phase).
    ///
    /// SYY = (Sdg⊗Sdg)·SXX·(S⊗S). Derived by composing S/Sdg/SXX push-throughs.
    /// Symplectic:
    ///   x1' = z1 ⊕ z2 ⊕ x2,  z1' = x1 ⊕ x2 ⊕ z2
    ///   x2' = z1 ⊕ z2 ⊕ x1,  z2' = x1 ⊕ x2 ⊕ z1
    /// Phase: complex formula derived from S/SXX/Sdg composition.
    #[inline]
    #[must_use]
    pub fn push_through_syy(p1: Self, p2: Self) -> (Self, Self, u8) {
        let (x1, z1) = p1.pauli_xz_bits();
        let (x2, z2) = p2.pauli_xz_bits();

        // Step 1: S-conjugation on both qubits: (x,z) → (x, z⊕x), phase = 2*x per qubit
        let sz1 = z1 ^ x1;
        let sz2 = z2 ^ x2;
        let s_phase = (if x1 { 2u8 } else { 0 } + if x2 { 2u8 } else { 0 }) % 8;

        // Step 2: SXX push-through on (x1, sz1, x2, sz2)
        let szz_xor = sz1 ^ sz2;
        let sxx_x1 = x1 ^ szz_xor;
        let sxx_x2 = x2 ^ szz_xor;
        let sxx_phase = if szz_xor { 6u8 } else { 0 };

        // Step 3: Sdg-conjugation on both qubits: (x,z) → (x, z⊕x), phase = 6*x per qubit
        let final_z1 = sz1 ^ sxx_x1;
        let final_z2 = sz2 ^ sxx_x2;
        let sdg_phase = (if sxx_x1 { 6u8 } else { 0 } + if sxx_x2 { 6u8 } else { 0 }) % 8;

        let new_p1 = Self::pauli_from_xz(sxx_x1, final_z1);
        let new_p2 = Self::pauli_from_xz(sxx_x2, final_z2);
        let phase = (s_phase + sxx_phase + sdg_phase) % 8;
        (new_p1, new_p2, phase)
    }

    /// Push Pauli frames through a G gate. Returns (`new_p1`, `new_p2`, phase).
    ///
    /// G = CZ · (H⊗H) · CZ. Derived by composing CZ and H push-throughs.
    #[inline]
    #[must_use]
    pub fn push_through_g(p1: Self, p2: Self) -> (Self, Self, u8) {
        let (x1, z1) = p1.pauli_xz_bits();
        let (x2, z2) = p2.pauli_xz_bits();

        // Step 1: CZ push-through: z1' = z1⊕x2, z2' = z2⊕x1, phase = 4*(x1&x2)
        let cz1_z1 = z1 ^ x2;
        let cz1_z2 = z2 ^ x1;
        let cz1_phase = if x1 && x2 { 4u8 } else { 0 };

        // Step 2: H-conjugation on both: (x,z) → (z,x), phase = 4*(x&z) per qubit
        let h_x1 = cz1_z1;
        let h_z1 = x1;
        let h_x2 = cz1_z2;
        let h_z2 = x2;
        let h_phase = (if x1 && cz1_z1 { 4u8 } else { 0 } + if x2 && cz1_z2 { 4u8 } else { 0 }) % 8;

        // Step 3: CZ push-through: phase = 4*(h_x1 & h_x2)
        let final_z1 = h_z1 ^ h_x2;
        let final_z2 = h_z2 ^ h_x1;
        let cz2_phase = if h_x1 && h_x2 { 4u8 } else { 0 };

        let new_p1 = Self::pauli_from_xz(h_x1, final_z1);
        let new_p2 = Self::pauli_from_xz(h_x2, final_z2);
        let phase = (cz1_phase + h_phase + cz2_phase) % 8;
        (new_p1, new_p2, phase)
    }
}

#[inline]
const fn axis_from_u8(a: u8) -> PauliAxis {
    match a {
        0 => PauliAxis::X,
        1 => PauliAxis::Y,
        _ => PauliAxis::Z,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    const SQRT2_INV: f64 = std::f64::consts::FRAC_1_SQRT_2;
    const ZERO: Complex64 = Complex64::new(0.0, 0.0);
    const ONE: Complex64 = Complex64::new(1.0, 0.0);
    const NEG1: Complex64 = Complex64::new(-1.0, 0.0);
    const IONE: Complex64 = Complex64::new(0.0, 1.0);

    type Mat2 = [[Complex64; 2]; 2];

    fn mat_i() -> Mat2 {
        [[ONE, ZERO], [ZERO, ONE]]
    }
    fn mat_h() -> Mat2 {
        let v = Complex64::new(SQRT2_INV, 0.0);
        let nv = Complex64::new(-SQRT2_INV, 0.0);
        [[v, v], [v, nv]]
    }
    fn mat_s() -> Mat2 {
        [[ONE, ZERO], [ZERO, IONE]]
    }
    fn mat_x() -> Mat2 {
        [[ZERO, ONE], [ONE, ZERO]]
    }
    fn mat_y() -> Mat2 {
        [[ZERO, Complex64::new(0.0, -1.0)], [IONE, ZERO]]
    }
    fn mat_z() -> Mat2 {
        [[ONE, ZERO], [ZERO, NEG1]]
    }

    fn mat_mul(a: &Mat2, b: &Mat2) -> Mat2 {
        let mut r = [[ZERO; 2]; 2];
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    r[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        r
    }

    fn mat_dag(a: &Mat2) -> Mat2 {
        [
            [a[0][0].conj(), a[1][0].conj()],
            [a[0][1].conj(), a[1][1].conj()],
        ]
    }

    /// Check if two matrices are equal up to global phase.
    fn eq_mod_phase(a: &Mat2, b: &Mat2) -> bool {
        let mut ratio: Option<Complex64> = None;
        for i in 0..2 {
            for j in 0..2 {
                let an = a[i][j].norm();
                let bn = b[i][j].norm();
                if an > 1e-10 && bn > 1e-10 {
                    let r = a[i][j] / b[i][j];
                    if let Some(prev) = ratio {
                        if (r - prev).norm() > 1e-6 {
                            return false;
                        }
                    } else {
                        ratio = Some(r);
                    }
                } else if an > 1e-10 || bn > 1e-10 {
                    return false;
                }
            }
        }
        true
    }

    /// Compute the matrix for element `idx` from its generator sequence.
    fn element_matrix(idx: usize) -> Mat2 {
        let gens = [mat_h(), mat_s()];
        let len = GEN_LENS[idx] as usize;
        let mut result = mat_i();
        for &generator in &GENERATORS[idx][..len] {
            let g = generator as usize;
            result = mat_mul(&result, &gens[g]);
        }
        result
    }

    /// Extract the Heisenberg action of a matrix on a Pauli.
    fn heisenberg_image(u: &Mat2, pauli: &Mat2) -> Mat2 {
        mat_mul(&mat_mul(&mat_dag(u), pauli), u)
    }

    /// Identify which signed Pauli a 2x2 matrix is (must be ±X, ±Y, or ±Z).
    fn identify_signed_pauli(m: &Mat2) -> (u8, bool) {
        let paulis = [mat_x(), mat_y(), mat_z()];
        for (axis, p) in paulis.iter().enumerate() {
            if eq_mod_phase(m, p) {
                // Check if positive or negative by comparing entries
                // Find first nonzero entry
                for i in 0..2 {
                    for j in 0..2 {
                        if p[i][j].norm() > 1e-10 {
                            let ratio = m[i][j] / p[i][j];
                            let is_neg = ratio.re < 0.0;
                            #[allow(clippy::cast_possible_truncation)] // axis index 0..2
                            return (axis as u8, is_neg);
                        }
                    }
                }
            }
        }
        panic!("Matrix is not a signed Pauli");
    }

    // ---- Tests ----

    #[test]
    fn test_all_24_distinct() {
        // Verify all 24 elements have distinct Heisenberg actions
        for i in 0..24 {
            for j in (i + 1)..24 {
                assert_ne!(
                    HEIS[i], HEIS[j],
                    "Elements {i} and {j} have identical Heisenberg actions"
                );
            }
        }
    }

    #[test]
    fn test_all_24_matrices_distinct() {
        let matrices: Vec<Mat2> = (0..24).map(element_matrix).collect();
        for i in 0..24 {
            for j in (i + 1)..24 {
                assert!(
                    !eq_mod_phase(&matrices[i], &matrices[j]),
                    "Elements {i} and {j} have identical matrices (mod phase)"
                );
            }
        }
    }

    #[test]
    fn test_heisenberg_actions_match_matrices() {
        let paulis = [mat_x(), mat_y(), mat_z()];
        for idx in 0..24 {
            let u = element_matrix(idx);
            // Check X image
            let x_img = heisenberg_image(&u, &paulis[0]);
            let (xa, xn) = identify_signed_pauli(&x_img);
            let (exp_xa, exp_xn, _, _) = HEIS[idx];
            assert_eq!(
                (xa, xn),
                (exp_xa, exp_xn),
                "Element {idx}: X image mismatch. Got ({xa},{xn}), expected ({exp_xa},{exp_xn})"
            );

            // Check Z image
            let z_img = heisenberg_image(&u, &paulis[2]);
            let (za, zn) = identify_signed_pauli(&z_img);
            let (_, _, exp_za, exp_zn) = HEIS[idx];
            assert_eq!(
                (za, zn),
                (exp_za, exp_zn),
                "Element {idx}: Z image mismatch. Got ({za},{zn}), expected ({exp_za},{exp_zn})"
            );
        }
    }

    #[test]
    fn test_compose_matches_matrix_multiplication() {
        let matrices: Vec<Mat2> = (0..24).map(element_matrix).collect();
        for i in 0..24 {
            for j in 0..24 {
                // compose(i, j) should be the element of matrix j·i
                let product = mat_mul(&matrices[j], &matrices[i]);
                let expected_idx = COMPOSE[i][j] as usize;
                assert!(
                    eq_mod_phase(&product, &matrices[expected_idx]),
                    "compose({i}, {j}) = {expected_idx}, but matrix product doesn't match"
                );
            }
        }
    }

    #[test]
    fn test_inverse() {
        for i in 0..24 {
            let inv = INVERSE[i];
            assert_eq!(
                COMPOSE[i][inv as usize], 0,
                "Element {i}: compose(i, inv) should be identity, got {}",
                COMPOSE[i][inv as usize]
            );
            assert_eq!(
                COMPOSE[inv as usize][i], 0,
                "Element {i}: compose(inv, i) should be identity, got {}",
                COMPOSE[inv as usize][i]
            );
        }
    }

    #[test]
    fn test_inverse_matches_matrix_adjoint() {
        let matrices: Vec<Mat2> = (0..24).map(element_matrix).collect();
        for i in 0..24 {
            let inv_idx = INVERSE[i] as usize;
            let inv_mat = mat_dag(&matrices[i]);
            assert!(
                eq_mod_phase(&inv_mat, &matrices[inv_idx]),
                "Element {i}: matrix adjoint doesn't match inverse element {inv_idx}"
            );
        }
    }

    #[test]
    fn test_named_constants() {
        // Verify named constants match expected matrices
        let cases: Vec<(CliffordFrame, Mat2)> = vec![
            (CliffordFrame::IDENTITY, mat_i()),
            (CliffordFrame::X, mat_x()),
            (CliffordFrame::Y, mat_y()),
            (CliffordFrame::Z, mat_z()),
            (CliffordFrame::H, mat_h()),
            (CliffordFrame::SZ, mat_s()),
        ];
        for (frame, expected) in &cases {
            let actual = element_matrix(frame.index() as usize);
            assert!(
                eq_mod_phase(&actual, expected),
                "Named constant {:?} (idx={}) doesn't match expected matrix",
                frame,
                frame.index()
            );
        }
    }

    #[test]
    fn test_sx_is_hsh() {
        let sx_mat = element_matrix(CliffordFrame::SX.index() as usize);
        let hsh = mat_mul(&mat_h(), &mat_mul(&mat_s(), &mat_h()));
        assert!(eq_mod_phase(&sx_mat, &hsh), "SX should equal HSH");
    }

    #[test]
    fn test_compose_semantics() {
        // frame=I, apply H -> frame=H
        let f = CliffordFrame::IDENTITY.compose(CliffordFrame::H);
        assert_eq!(f, CliffordFrame::H);

        // frame=H, apply S -> frame should correspond to matrix S·H
        let f2 = f.compose(CliffordFrame::SZ);
        let sh_mat = mat_mul(&mat_s(), &mat_h());
        let f2_mat = element_matrix(f2.index() as usize);
        assert!(
            eq_mod_phase(&f2_mat, &sh_mat),
            "H.compose(S) should give element for S·H"
        );
    }

    #[test]
    #[allow(clippy::cast_possible_truncation)] // loop index bounded by 24
    fn test_decompose() {
        for i in 0..24 {
            let (pauli, coset) = DECOMPOSE[i];
            // Verify pauli is actually a Pauli
            assert!(
                pauli < 4,
                "Element {i}: decompose pauli={pauli} is not a Pauli"
            );
            // Verify coset is one of the 6 reps
            assert!(
                [0, 4, 6, 7, 8, 12].contains(&coset),
                "Element {i}: decompose coset={coset} is not a valid coset rep"
            );
            // Verify composition: compose(coset, pauli) == i
            // compose(a, b) = element of b·a, so compose(coset, pauli) = pauli·coset = C
            let composed = COMPOSE[coset as usize][pauli as usize];
            assert_eq!(
                composed, i as u8,
                "Element {i}: compose(coset={coset}, pauli={pauli}) = {composed}, expected {i}"
            );
        }
    }

    #[test]
    fn test_is_pauli() {
        assert!(CliffordFrame::IDENTITY.is_pauli());
        assert!(CliffordFrame::X.is_pauli());
        assert!(CliffordFrame::Y.is_pauli());
        assert!(CliffordFrame::Z.is_pauli());
        assert!(!CliffordFrame::H.is_pauli());
        assert!(!CliffordFrame::SZ.is_pauli());
    }

    #[test]
    fn test_z_image() {
        // I maps Z to +Z
        let img = CliffordFrame::IDENTITY.z_image();
        assert_eq!(img.axis, PauliAxis::Z);
        assert!(img.positive);

        // H maps Z to +X
        let img = CliffordFrame::H.z_image();
        assert_eq!(img.axis, PauliAxis::X);
        assert!(img.positive);

        // X maps Z to -Z
        let img = CliffordFrame::X.z_image();
        assert_eq!(img.axis, PauliAxis::Z);
        assert!(!img.positive);

        // S maps Z to +Z
        let img = CliffordFrame::SZ.z_image();
        assert_eq!(img.axis, PauliAxis::Z);
        assert!(img.positive);
    }

    #[test]
    fn test_pauli_xz_roundtrip() {
        for idx in 0..4u8 {
            let frame = CliffordFrame(idx);
            let (x, z) = frame.pauli_xz_bits();
            let reconstructed = CliffordFrame::pauli_from_xz(x, z);
            assert_eq!(frame, reconstructed, "Pauli roundtrip failed for idx={idx}");
        }
    }

    #[test]
    fn test_push_through_cx() {
        // X_ctrl through CX -> X_ctrl X_targ
        let (c, t, ph) = CliffordFrame::push_through_cx(CliffordFrame::X, CliffordFrame::IDENTITY);
        assert_eq!(c, CliffordFrame::X);
        assert_eq!(t, CliffordFrame::X);
        assert_eq!(ph, 0);

        // I_ctrl, Z_targ through CX -> Z_ctrl Z_targ
        let (c, t, ph) = CliffordFrame::push_through_cx(CliffordFrame::IDENTITY, CliffordFrame::Z);
        assert_eq!(c, CliffordFrame::Z);
        assert_eq!(t, CliffordFrame::Z);
        assert_eq!(ph, 0);

        // Z_ctrl through CX -> Z_ctrl
        let (c, t, ph) = CliffordFrame::push_through_cx(CliffordFrame::Z, CliffordFrame::IDENTITY);
        assert_eq!(c, CliffordFrame::Z);
        assert_eq!(t, CliffordFrame::IDENTITY);
        assert_eq!(ph, 0);

        // I_ctrl, X_targ through CX -> X_targ unchanged
        let (c, t, ph) = CliffordFrame::push_through_cx(CliffordFrame::IDENTITY, CliffordFrame::X);
        assert_eq!(c, CliffordFrame::IDENTITY);
        assert_eq!(t, CliffordFrame::X);
        assert_eq!(ph, 0);
    }

    #[test]
    fn test_push_through_cz() {
        // X_ctrl through CZ -> X_ctrl Z_targ
        let (c, t, ph) = CliffordFrame::push_through_cz(CliffordFrame::X, CliffordFrame::IDENTITY);
        assert_eq!(c, CliffordFrame::X);
        assert_eq!(t, CliffordFrame::Z);
        assert_eq!(ph, 0);

        // I_ctrl, X_targ through CZ -> Z_ctrl X_targ
        let (c, t, ph) = CliffordFrame::push_through_cz(CliffordFrame::IDENTITY, CliffordFrame::X);
        assert_eq!(c, CliffordFrame::Z);
        assert_eq!(t, CliffordFrame::X);
        assert_eq!(ph, 0);

        // Z_ctrl, Z_targ through CZ -> unchanged
        let (c, t, ph) = CliffordFrame::push_through_cz(CliffordFrame::Z, CliffordFrame::Z);
        assert_eq!(c, CliffordFrame::Z);
        assert_eq!(t, CliffordFrame::Z);
        assert_eq!(ph, 0);
    }

    #[test]
    fn test_group_closure() {
        // Verify COMPOSE table has no 255 entries (all compositions found)
        for i in 0..24 {
            for j in 0..24 {
                assert_ne!(
                    COMPOSE[i][j], 255,
                    "compose({i}, {j}) not found in element table"
                );
            }
        }
    }

    #[test]
    #[allow(clippy::cast_possible_truncation)] // loop index bounded by 24
    fn test_identity_is_neutral() {
        for i in 0..24 {
            assert_eq!(COMPOSE[0][i], i as u8, "I·{i} should be {i}");
            assert_eq!(COMPOSE[i][0], i as u8, "{i}·I should be {i}");
        }
    }

    #[test]
    fn test_associativity() {
        // (a·b)·c = a·(b·c) for all triples
        for a in 0..24u8 {
            for b in 0..24u8 {
                let ab = COMPOSE[a as usize][b as usize];
                for c in 0..24u8 {
                    let ab_c = COMPOSE[ab as usize][c as usize];
                    let bc = COMPOSE[b as usize][c as usize];
                    let a_bc = COMPOSE[a as usize][bc as usize];
                    assert_eq!(
                        ab_c, a_bc,
                        "Associativity failed: ({a}·{b})·{c} = {ab_c} != {a}·({b}·{c}) = {a_bc}"
                    );
                }
            }
        }
    }

    /// Extract the global phase ratio as an 8th-root-of-unity index.
    /// ratio = actual / representative. Should be e^{i*k*pi/4} for k in 0..7.
    #[allow(clippy::cast_possible_truncation)] // k bounded by 8 phase roots
    fn phase_index(ratio: Complex64) -> u8 {
        let phases = PHASE_ROOTS;
        for (k, &[re, im]) in phases.iter().enumerate() {
            if (ratio - Complex64::new(re, im)).norm() < 1e-6 {
                return k as u8;
            }
        }
        panic!("Phase ratio {ratio:?} is not an 8th root of unity");
    }

    /// Compute the global phase ratio between matrix product and representative.
    fn compute_cocycle_entry(matrices: &[Mat2], i: usize, j: usize) -> u8 {
        let product = mat_mul(&matrices[j], &matrices[i]);
        let k = COMPOSE[i][j] as usize;
        let rep = &matrices[k];
        let mut ratio = Complex64::new(0.0, 0.0);
        for r in 0..2 {
            for c in 0..2 {
                if rep[r][c].norm() > 1e-10 {
                    ratio = product[r][c] / rep[r][c];
                    break;
                }
            }
            if ratio.norm() > 1e-10 {
                break;
            }
        }
        phase_index(ratio)
    }

    #[test]
    fn test_verify_element_matrix() {
        let matrices: Vec<Mat2> = (0..24).map(element_matrix).collect();
        for idx in 0..24 {
            let m = &matrices[idx];
            let em = ELEMENT_MATRIX[idx];
            for (r, c) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                let expected = Complex64::new(em[r * 4 + c * 2], em[r * 4 + c * 2 + 1]);
                let actual = m[r][c];
                assert!(
                    (actual - expected).norm() < 1e-10,
                    "ELEMENT_MATRIX[{idx}][{r}][{c}]: expected {actual:?}, got {expected:?}"
                );
            }
        }
    }

    #[test]
    fn test_verify_phase_cocycle() {
        let matrices: Vec<Mat2> = (0..24).map(element_matrix).collect();
        for i in 0..24 {
            for j in 0..24 {
                let expected = compute_cocycle_entry(&matrices, i, j);
                assert_eq!(
                    PHASE_COCYCLE[i][j], expected,
                    "PHASE_COCYCLE[{i}][{j}]: expected {expected}, got {}",
                    PHASE_COCYCLE[i][j]
                );
            }
        }
    }

    #[test]
    fn test_cocycle_satisfies_associativity() {
        // For the cocycle to be consistent: composing three elements
        // must give the same total phase regardless of grouping.
        // (a * b) * c vs a * (b * c):
        // phase(a,b) + phase(compose(a,b), c) == phase(b,c) + phase(a, compose(b,c))
        for a in 0..24usize {
            for b in 0..24usize {
                let ab = COMPOSE[a][b] as usize;
                for c in 0..24usize {
                    let bc = COMPOSE[b][c] as usize;
                    let lhs =
                        (u16::from(PHASE_COCYCLE[a][b]) + u16::from(PHASE_COCYCLE[ab][c])) % 8;
                    let rhs =
                        (u16::from(PHASE_COCYCLE[b][c]) + u16::from(PHASE_COCYCLE[a][bc])) % 8;
                    assert_eq!(lhs, rhs, "Cocycle associativity failed for ({a},{b},{c})");
                }
            }
        }
    }

    #[test]
    fn test_verify_gate_phase_deltas() {
        // Verify that GATE_PHASE_DELTA correctly maps standard gate matrices
        // to element matrices.
        let std_gates: [(usize, Mat2); 11] = [
            (0, mat_i()),
            (1, mat_x()),
            (2, mat_y()),
            (3, mat_z()),
            (4, mat_s()),
            (5, {
                let mi = Complex64::new(0.0, -1.0);
                [[ONE, ZERO], [ZERO, mi]]
            }),
            (6, mat_h()),
            (13, {
                // SX
                let a = Complex64::new(0.5, 0.5);
                let b = Complex64::new(0.5, -0.5);
                [[a, b], [b, a]]
            }),
            (12, {
                // SXdg
                let a = Complex64::new(0.5, -0.5);
                let b = Complex64::new(0.5, 0.5);
                [[a, b], [b, a]]
            }),
            (10, {
                // SY
                let a = Complex64::new(0.5, 0.5);
                let b = Complex64::new(-0.5, -0.5);
                [[a, b], [Complex64::new(0.5, 0.5), a]]
            }),
            (9, {
                // SYdg
                let a = Complex64::new(0.5, -0.5);
                let b = Complex64::new(0.5, -0.5);
                let c = Complex64::new(-0.5, 0.5);
                [[a, b], [c, a]]
            }),
        ];

        for &(idx, ref std_mat) in &std_gates {
            let elem_mat = element_matrix(idx);
            // Find ratio = std / elem
            let mut ratio = Complex64::new(0.0, 0.0);
            for r in 0..2 {
                for c in 0..2 {
                    if elem_mat[r][c].norm() > 1e-10 {
                        ratio = std_mat[r][c] / elem_mat[r][c];
                        break;
                    }
                }
                if ratio.norm() > 1e-10 {
                    break;
                }
            }
            let delta = phase_index(ratio);
            assert_eq!(
                GATE_PHASE_DELTA[idx], delta,
                "GATE_PHASE_DELTA[{idx}]: expected {delta}, got {}",
                GATE_PHASE_DELTA[idx]
            );
        }
    }

    #[test]
    fn test_h_generates_order_2() {
        let h = CliffordFrame::H;
        assert_eq!(h.compose(h), CliffordFrame::IDENTITY, "H² should be I");
    }

    #[test]
    fn test_s_generates_order_4() {
        let s = CliffordFrame::SZ;
        let s2 = s.compose(s);
        let s3 = s2.compose(s);
        let s4 = s3.compose(s);
        assert_eq!(s2, CliffordFrame::Z, "S² should be Z");
        assert_eq!(s3, CliffordFrame::SZDG, "S³ should be Sdg");
        assert_eq!(s4, CliffordFrame::IDENTITY, "S⁴ should be I");
    }

    #[test]
    fn test_cx_cz_pauli_passthrough() {
        // Verify CX and CZ Pauli pass-through in the ELEMENT_MATRIX convention.
        //
        // CX is phase-free: conjugating any Pauli tensor product by CX yields
        // exactly the expected Pauli tensor product with no extra phase.
        //
        // CZ picks up a sign of (-1)^{xc * xt} in the element convention,
        // where xc, xt are the X-bits of the input Paulis. This comes from
        // two sources: (1) the Pauli anticommutation sign when CZ introduces
        // Z factors that must be moved past X factors on the same qubit, and
        // (2) the element convention phase for Y (Y_elem = i * Y_std). These
        // combine to give phase = -1 exactly when both inputs have X-bit set
        // (i.e., both are X or Y).

        const ELEM: &[[f64; 8]; 24] = &ELEMENT_MATRIX;

        // --- 4x4 complex matrix helpers using [re, im] pairs ---
        // A 4x4 complex matrix is [[f64; 2]; 16], stored row-major:
        //   entry (r, c) at index r*4 + c.
        type Mat4 = [[f64; 2]; 16];

        const CZERO: [f64; 2] = [0.0, 0.0];

        fn cmul(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
            [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]]
        }

        fn cadd(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
            [a[0] + b[0], a[1] + b[1]]
        }

        fn mat4_zero() -> Mat4 {
            [CZERO; 16]
        }

        fn mat4_mul(a: &Mat4, b: &Mat4) -> Mat4 {
            let mut r = mat4_zero();
            for i in 0..4 {
                for j in 0..4 {
                    let mut s = CZERO;
                    for k in 0..4 {
                        s = cadd(s, cmul(a[i * 4 + k], b[k * 4 + j]));
                    }
                    r[i * 4 + j] = s;
                }
            }
            r
        }

        fn mat4_dag(a: &Mat4) -> Mat4 {
            let mut r = mat4_zero();
            for i in 0..4 {
                for j in 0..4 {
                    let v = a[j * 4 + i];
                    r[i * 4 + j] = [v[0], -v[1]]; // conjugate transpose
                }
            }
            r
        }

        /// Build a 4x4 matrix from the tensor product of two `ELEMENT_MATRIX` entries.
        fn tensor(a_idx: usize, b_idx: usize) -> Mat4 {
            let a = &ELEM[a_idx];
            let b = &ELEM[b_idx];
            // a is [[a00, a01], [a10, a11]] with a_rc = (a[r*4+c*2], a[r*4+c*2+1])
            // tensor product: (A tensor B)_{(ia,ib),(ja,jb)} = A_{ia,ja} * B_{ib,jb}
            let mut r = mat4_zero();
            for ia in 0..2 {
                for ja in 0..2 {
                    let a_val = [a[ia * 4 + ja * 2], a[ia * 4 + ja * 2 + 1]];
                    for ib in 0..2 {
                        for jb in 0..2 {
                            let b_val = [b[ib * 4 + jb * 2], b[ib * 4 + jb * 2 + 1]];
                            let row = ia * 2 + ib;
                            let col = ja * 2 + jb;
                            r[row * 4 + col] = cmul(a_val, b_val);
                        }
                    }
                }
            }
            r
        }

        /// CX matrix in computational basis (control qubit 0, target qubit 1):
        /// |00> -> |00>, |01> -> |01>, |10> -> |11>, |11> -> |10>
        fn mat_cx() -> Mat4 {
            let mut m = mat4_zero();
            let one = [1.0, 0.0];
            m[0] = one; // |00> -> |00>
            m[4 + 1] = one; // |01> -> |01>
            m[2 * 4 + 3] = one; // |10> -> |11>
            m[3 * 4 + 2] = one; // |11> -> |10>
            m
        }

        /// CZ matrix: diag(1, 1, 1, -1)
        fn mat_cz() -> Mat4 {
            let mut m = mat4_zero();
            let one = [1.0, 0.0];
            m[0] = one;
            m[4 + 1] = one;
            m[2 * 4 + 2] = one;
            m[3 * 4 + 3] = [-1.0, 0.0];
            m
        }

        fn mat4_eq(a: &Mat4, b: &Mat4, tol: f64) -> bool {
            for i in 0..16 {
                let dr = a[i][0] - b[i][0];
                let di = a[i][1] - b[i][1];
                if (dr * dr + di * di).sqrt() > tol {
                    return false;
                }
            }
            true
        }

        /// Extract (`x_bit`, `z_bit`) from Pauli index 0..3.
        fn pauli_xz(idx: usize) -> (bool, bool) {
            match idx {
                0 => (false, false), // I
                1 => (true, false),  // X
                2 => (true, true),   // Y
                3 => (false, true),  // Z
                _ => unreachable!(),
            }
        }

        /// Construct Pauli index from (`x_bit`, `z_bit`).
        fn pauli_from_xz(x: bool, z: bool) -> usize {
            match (x, z) {
                (false, false) => 0,
                (true, false) => 1,
                (true, true) => 2,
                (false, true) => 3,
            }
        }

        // --- Test CX ---
        let cx = mat_cx();
        let cx_dag = mat4_dag(&cx); // CX is self-adjoint, but compute anyway

        for pc in 0..4 {
            for pt in 0..4 {
                let input = tensor(pc, pt);
                // CX * input * CX^dag
                let conjugated = mat4_mul(&cx, &mat4_mul(&input, &cx_dag));

                // Compute expected output Paulis via symplectic rules
                let (xc, zc) = pauli_xz(pc);
                let (xt, zt) = pauli_xz(pt);
                let pc_out = pauli_from_xz(xc, zc ^ zt);
                let pt_out = pauli_from_xz(xc ^ xt, zt);

                let expected = tensor(pc_out, pt_out);

                assert!(
                    mat4_eq(&conjugated, &expected, 1e-10),
                    "CX phase-free check failed for pc={pc}, pt={pt}: \
                     expected Pauli ({pc_out}, {pt_out})"
                );
            }
        }

        // --- Test CZ ---
        // CZ picks up (-1)^{xc*xt} in the element convention.
        let cz = mat_cz();
        let cz_dag = mat4_dag(&cz); // CZ is self-adjoint

        for pc in 0..4 {
            for pt in 0..4 {
                let input = tensor(pc, pt);
                // CZ * input * CZ^dag
                let conjugated = mat4_mul(&cz, &mat4_mul(&input, &cz_dag));

                // Compute expected output Paulis via symplectic rules
                let (xc, zc) = pauli_xz(pc);
                let (xt, zt) = pauli_xz(pt);
                let pc_out = pauli_from_xz(xc, zc ^ xt);
                let pt_out = pauli_from_xz(xt, zt ^ xc);

                // Element-convention phase: (-1) when both X-bits are set
                let phase: f64 = if xc && xt { -1.0 } else { 1.0 };

                let raw_expected = tensor(pc_out, pt_out);
                let mut expected = mat4_zero();
                for i in 0..16 {
                    expected[i] = [raw_expected[i][0] * phase, raw_expected[i][1] * phase];
                }

                assert!(
                    mat4_eq(&conjugated, &expected, 1e-10),
                    "CZ check failed for pc={pc}, pt={pt}: \
                     expected Pauli ({pc_out}, {pt_out}) with phase={phase}"
                );
            }
        }
    }

    /// Verify a two-qubit `push_through` against 4x4 matrix conjugation.
    ///
    /// For each of the 16 Pauli⊗Pauli inputs, computes G†·(P1⊗P2)·G via
    /// matrix multiplication and checks that it matches the `push_through` output.
    #[allow(clippy::cast_possible_truncation)] // Pauli indices bounded by 4
    fn verify_push_through_all_16(
        gate_matrix: &[[f64; 2]; 16],
        push_through: fn(CliffordFrame, CliffordFrame) -> (CliffordFrame, CliffordFrame, u8),
        gate_name: &str,
    ) {
        const ELEM: &[[f64; 8]; 24] = &ELEMENT_MATRIX;
        const CZERO: [f64; 2] = [0.0, 0.0];

        fn cmul(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
            [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]]
        }

        fn cadd(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
            [a[0] + b[0], a[1] + b[1]]
        }

        type Mat4 = [[f64; 2]; 16];

        fn mat4_zero() -> Mat4 {
            [CZERO; 16]
        }

        fn mat4_mul(a: &Mat4, b: &Mat4) -> Mat4 {
            let mut r = mat4_zero();
            for i in 0..4 {
                for j in 0..4 {
                    let mut s = CZERO;
                    for k in 0..4 {
                        s = cadd(s, cmul(a[i * 4 + k], b[k * 4 + j]));
                    }
                    r[i * 4 + j] = s;
                }
            }
            r
        }

        fn mat4_dag(a: &Mat4) -> Mat4 {
            let mut r = mat4_zero();
            for i in 0..4 {
                for j in 0..4 {
                    let v = a[j * 4 + i];
                    r[i * 4 + j] = [v[0], -v[1]];
                }
            }
            r
        }

        fn tensor(a_idx: usize, b_idx: usize) -> Mat4 {
            let a = &ELEM[a_idx];
            let b = &ELEM[b_idx];
            let mut r = mat4_zero();
            for ia in 0..2 {
                for ja in 0..2 {
                    let a_val = [a[ia * 4 + ja * 2], a[ia * 4 + ja * 2 + 1]];
                    for ib in 0..2 {
                        for jb in 0..2 {
                            let b_val = [b[ib * 4 + jb * 2], b[ib * 4 + jb * 2 + 1]];
                            r[(ia * 2 + ib) * 4 + (ja * 2 + jb)] = cmul(a_val, b_val);
                        }
                    }
                }
            }
            r
        }

        fn mat4_eq(a: &Mat4, b: &Mat4, tol: f64) -> bool {
            for i in 0..16 {
                let dr = a[i][0] - b[i][0];
                let di = a[i][1] - b[i][1];
                if (dr * dr + di * di).sqrt() > tol {
                    return false;
                }
            }
            true
        }

        let g = *gate_matrix;
        let g_dag = mat4_dag(&g);

        for pc in 0..4usize {
            for pt in 0..4usize {
                let input = tensor(pc, pt);
                let conjugated = mat4_mul(&g_dag, &mat4_mul(&input, &g));

                let p1 = CliffordFrame(pc as u8);
                let p2 = CliffordFrame(pt as u8);
                let (new_p1, new_p2, phase_idx) = push_through(p1, p2);

                let raw_expected = tensor(new_p1.index() as usize, new_p2.index() as usize);
                let [pr, pi] = PHASE_ROOTS[phase_idx as usize];
                let mut expected = mat4_zero();
                for i in 0..16 {
                    expected[i] = [
                        raw_expected[i][0] * pr - raw_expected[i][1] * pi,
                        raw_expected[i][0] * pi + raw_expected[i][1] * pr,
                    ];
                }

                assert!(
                    mat4_eq(&conjugated, &expected, 1e-10),
                    "{gate_name} push-through failed for P1={pc}, P2={pt}: \
                     got Pauli ({},{}) phase={phase_idx}, but matrix doesn't match",
                    new_p1.index(),
                    new_p2.index()
                );
            }
        }
    }

    #[test]
    fn test_push_through_szz() {
        // SZZ = exp(-iπ/4 ZZ) = diag(e^{-iπ/4}, e^{iπ/4}, e^{iπ/4}, e^{-iπ/4})
        let w = std::f64::consts::FRAC_1_SQRT_2;
        let mut szz = [[0.0, 0.0]; 16];
        szz[0] = [w, -w]; // e^{-iπ/4}
        szz[4 + 1] = [w, w]; // e^{iπ/4}
        szz[2 * 4 + 2] = [w, w]; // e^{iπ/4}
        szz[3 * 4 + 3] = [w, -w]; // e^{-iπ/4}
        verify_push_through_all_16(&szz, CliffordFrame::push_through_szz, "SZZ");
    }

    #[test]
    fn test_push_through_iswap() {
        // iSWAP: |00⟩→|00⟩, |01⟩→i|10⟩, |10⟩→i|01⟩, |11⟩→|11⟩
        let mut iswap = [[0.0, 0.0]; 16];
        iswap[0] = [1.0, 0.0];
        iswap[4 + 2] = [0.0, 1.0]; // i
        iswap[2 * 4 + 1] = [0.0, 1.0]; // i
        iswap[3 * 4 + 3] = [1.0, 0.0];
        verify_push_through_all_16(&iswap, CliffordFrame::push_through_iswap, "iSWAP");
    }

    #[test]
    fn test_push_through_sxx() {
        // SXX = exp(-iπ/4 XX) = (I - iXX)/√2
        let w = std::f64::consts::FRAC_1_SQRT_2;
        let mut sxx = [[0.0, 0.0]; 16];
        sxx[0] = [w, 0.0];
        sxx[3] = [0.0, -w];
        sxx[4 + 1] = [w, 0.0];
        sxx[4 + 2] = [0.0, -w];
        sxx[2 * 4 + 1] = [0.0, -w];
        sxx[2 * 4 + 2] = [w, 0.0];
        sxx[3 * 4] = [0.0, -w];
        sxx[3 * 4 + 3] = [w, 0.0];
        verify_push_through_all_16(&sxx, CliffordFrame::push_through_sxx, "SXX");
    }

    #[test]
    fn test_push_through_syy() {
        // SYY = (Sdg⊗Sdg)·SXX·(S⊗S)
        // Build from matrix multiplication
        type Mat4 = [[f64; 2]; 16];
        const CZERO: [f64; 2] = [0.0, 0.0];
        fn mat4_zero() -> Mat4 {
            [CZERO; 16]
        }
        fn cmul(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
            [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]]
        }
        fn cadd(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
            [a[0] + b[0], a[1] + b[1]]
        }
        fn mat4_mul(a: &Mat4, b: &Mat4) -> Mat4 {
            let mut r = mat4_zero();
            for i in 0..4 {
                for j in 0..4 {
                    let mut s = CZERO;
                    for k in 0..4 {
                        s = cadd(s, cmul(a[i * 4 + k], b[k * 4 + j]));
                    }
                    r[i * 4 + j] = s;
                }
            }
            r
        }
        let w = std::f64::consts::FRAC_1_SQRT_2;
        // S⊗S tensor product: diag(1, i, i, -1)
        let mut ss = mat4_zero();
        ss[0] = [1.0, 0.0];
        ss[4 + 1] = [0.0, 1.0]; // i
        ss[2 * 4 + 2] = [0.0, 1.0]; // i
        ss[3 * 4 + 3] = [-1.0, 0.0]; // i*i = -1

        // Sdg⊗Sdg: diag(1, -i, -i, -1)
        let mut sdsd = mat4_zero();
        sdsd[0] = [1.0, 0.0];
        sdsd[4 + 1] = [0.0, -1.0]; // -i
        sdsd[2 * 4 + 2] = [0.0, -1.0]; // -i
        sdsd[3 * 4 + 3] = [-1.0, 0.0]; // -1

        // SXX
        let mut sxx = mat4_zero();
        sxx[0] = [w, 0.0];
        sxx[3] = [0.0, -w];
        sxx[4 + 1] = [w, 0.0];
        sxx[4 + 2] = [0.0, -w];
        sxx[2 * 4 + 1] = [0.0, -w];
        sxx[2 * 4 + 2] = [w, 0.0];
        sxx[3 * 4] = [0.0, -w];
        sxx[3 * 4 + 3] = [w, 0.0];

        let syy = mat4_mul(&sdsd, &mat4_mul(&sxx, &ss));
        verify_push_through_all_16(&syy, CliffordFrame::push_through_syy, "SYY");
    }

    #[test]
    fn test_push_through_g() {
        // G = CZ · (H⊗H) · CZ
        type Mat4 = [[f64; 2]; 16];
        const CZERO: [f64; 2] = [0.0, 0.0];
        fn mat4_zero() -> Mat4 {
            [CZERO; 16]
        }
        fn cmul(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
            [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]]
        }
        fn cadd(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
            [a[0] + b[0], a[1] + b[1]]
        }
        fn mat4_mul(a: &Mat4, b: &Mat4) -> Mat4 {
            let mut r = mat4_zero();
            for i in 0..4 {
                for j in 0..4 {
                    let mut s = CZERO;
                    for k in 0..4 {
                        s = cadd(s, cmul(a[i * 4 + k], b[k * 4 + j]));
                    }
                    r[i * 4 + j] = s;
                }
            }
            r
        }
        // CZ = diag(1, 1, 1, -1)
        let mut cz = mat4_zero();
        cz[0] = [1.0, 0.0];
        cz[4 + 1] = [1.0, 0.0];
        cz[2 * 4 + 2] = [1.0, 0.0];
        cz[3 * 4 + 3] = [-1.0, 0.0];

        // H⊗H: tensor product of H with itself
        let mut hh = mat4_zero();
        // H = [[r,r],[r,-r]]
        // H⊗H = [[r*r, r*r, r*r, r*r],
        //         [r*r, -r*r, r*r, -r*r],
        //         [r*r, r*r, -r*r, -r*r],
        //         [r*r, -r*r, -r*r, r*r]]
        let h2 = 0.5; // r*r
        hh[0] = [h2, 0.0];
        hh[1] = [h2, 0.0];
        hh[2] = [h2, 0.0];
        hh[3] = [h2, 0.0];
        hh[4] = [h2, 0.0];
        hh[4 + 1] = [-h2, 0.0];
        hh[4 + 2] = [h2, 0.0];
        hh[4 + 3] = [-h2, 0.0];
        hh[2 * 4] = [h2, 0.0];
        hh[2 * 4 + 1] = [h2, 0.0];
        hh[2 * 4 + 2] = [-h2, 0.0];
        hh[2 * 4 + 3] = [-h2, 0.0];
        hh[3 * 4] = [h2, 0.0];
        hh[3 * 4 + 1] = [-h2, 0.0];
        hh[3 * 4 + 2] = [-h2, 0.0];
        hh[3 * 4 + 3] = [h2, 0.0];

        let g = mat4_mul(&cz, &mat4_mul(&hh, &cz));
        verify_push_through_all_16(&g, CliffordFrame::push_through_g, "G");
    }
}
