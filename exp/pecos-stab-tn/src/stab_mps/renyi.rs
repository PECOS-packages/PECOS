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

//! CAMPS-native second Rényi entropy `S_2`.
//!
//! Implements Pauli Coefficient Enumeration (PCE) from Liu-Clark 2412.17209
//! Section VI.C. For a CAMPS state ρ = C |φ⟩⟨φ| C† where |φ⟩ is a product MPS:
//!
//! 1. For each MPS site j, find non-vanishing single-qubit Paulis (up to 3 per site)
//!    and their coefficients (Bloch-vector components).
//! 2. Map each `G̃_k` to `G_k` = C · `G̃_k` · C† (a Pauli string on all N qubits).
//! 3. Gaussian-eliminate the region-B part of `G_k` to find generators {`Q_j`} supported
//!    only on region A.
//! 4. Enumerate all 2^M combinations of `Q_j`'s, compute coefficients, and sum
//!    squared coefficients to get `Tr(ρ_A²)`.
//!
//! PCE complexity: O(M · 2^M) enumeration. For magic-doped Clifford circuits with
//! few T gates, M is small relative to N, making PCE practical for many qubits.
//!
//! A PCMPS variant (next step) represents the coefficient state as an MPS to
//! reach 100+ qubits; deferred.

use crate::mps::Mps;
use pecos_simulators::SparseStabY;

/// A Pauli string on N qubits: (x, z) bitvectors and a real coefficient.
///
/// The coefficient is real because every Pauli string produced here is
/// Hermitian (r·σ with real r, conjugated by a Clifford stays Hermitian).
#[derive(Clone, Debug)]
pub(crate) struct PauliString {
    pub x: Vec<bool>,
    pub z: Vec<bool>,
    pub coef: f64,
}

impl PauliString {
    pub fn identity(n: usize) -> Self {
        Self {
            x: vec![false; n],
            z: vec![false; n],
            coef: 1.0,
        }
    }

    /// Returns true if this Pauli is supported only on qubits in `region_a`.
    pub fn supported_on(&self, region_a: &[bool]) -> bool {
        for (q, in_a) in region_a.iter().enumerate() {
            if !in_a && (self.x[q] || self.z[q]) {
                return false;
            }
        }
        true
    }
}

/// Compute Bloch vector (`r_x`, `r_y`, `r_z`) for each MPS site, assuming bond dim 1.
///
/// `r_x` = Tr(X `ρ_j`) = 2 `Re(ψ_0`* `ψ_1`) (for normalized state |`ψ_j`⟩ = `ψ_0|0`⟩ + `ψ_1|1`⟩)
/// `r_y` = Tr(Y `ρ_j`) = -2 `Im(ψ_0`* `ψ_1`)
/// `r_z` = Tr(Z `ρ_j`) = |`ψ_0|²` - |`ψ_1|²`
pub(crate) fn bloch_vectors(mps: &Mps) -> Result<Vec<(f64, f64, f64)>, String> {
    if mps.max_bond_dim() != 1 {
        return Err(format!(
            "bloch_vectors requires bond dim 1, got {}",
            mps.max_bond_dim()
        ));
    }
    let n = mps.num_sites();
    let mut out = Vec::with_capacity(n);
    for j in 0..n {
        let t = &mps.tensors()[j];
        // Tensor shape: (chi_l=1, 2*chi_r=2). Extract psi_0 = t[0, 0], psi_1 = t[0, 1].
        let psi0 = t[(0, 0)];
        let psi1 = t[(0, 1)];
        let rx = 2.0 * (psi0.conj() * psi1).re;
        let ry = -2.0 * (psi0.conj() * psi1).im;
        let rz = psi0.norm_sqr() - psi1.norm_sqr();
        out.push((rx, ry, rz));
    }
    Ok(out)
}

/// Map a single-qubit Pauli at site j through the tableau's Clifford C.
/// Returns G = C · `P_j` · C† as a `PauliString` on N qubits.
///
/// Uses: `destab_j` = C `X_j` C†, `stab_j` = C `Z_j` C†. Then C `Y_j` C† = i · `destab_j` · `stab_j`
/// = (`destab_j` · `stab_j` with sign/phase adjustments).
pub(crate) fn map_pauli_through_tableau(
    tableau: &SparseStabY,
    site: usize,
    pauli: char,
    coef: f64,
) -> PauliString {
    let n = tableau.num_qubits();
    match pauli {
        'X' => {
            let mut x = vec![false; n];
            let mut z = vec![false; n];
            for q in &tableau.destabs().row_x[site] {
                x[q] = true;
            }
            for q in &tableau.destabs().row_z[site] {
                z[q] = true;
            }
            let mut c = coef;
            if tableau.destabs().signs_minus.contains(site) {
                c = -c;
            }
            // signs_i contributes i^count. For Hermitian generators the
            // total phase (including Y-count) must be real. Count Y-bits
            // (x AND z set) to get the Y-count, combine with signs_i.
            if tableau.destabs().signs_i.contains(site) {
                let y_count: usize = x
                    .iter()
                    .zip(z.iter())
                    .filter(|(xi, zi)| **xi && **zi)
                    .count();
                // Total i-count: 1 (from signs_i) + y_count. Must be even
                // for a real coefficient.
                let total_i = 1 + y_count;
                debug_assert!(
                    total_i.is_multiple_of(2),
                    "destab signs_i + Y-count must be even for real coefficient, got {total_i}"
                );
                if total_i % 4 == 2 {
                    c = -c;
                }
            }
            PauliString { x, z, coef: c }
        }
        'Z' => {
            let mut x = vec![false; n];
            let mut z = vec![false; n];
            for q in &tableau.stabs().row_x[site] {
                x[q] = true;
            }
            for q in &tableau.stabs().row_z[site] {
                z[q] = true;
            }
            let mut c = coef;
            if tableau.stabs().signs_minus.contains(site) {
                c = -c;
            }
            if tableau.stabs().signs_i.contains(site) {
                let y_count: usize = x
                    .iter()
                    .zip(z.iter())
                    .filter(|(xi, zi)| **xi && **zi)
                    .count();
                let total_i = 1 + y_count;
                debug_assert!(
                    total_i.is_multiple_of(2),
                    "stab signs_i + Y-count must be even for real coefficient, got {total_i}"
                );
                if total_i % 4 == 2 {
                    c = -c;
                }
            }
            PauliString { x, z, coef: c }
        }
        'Y' => {
            // Y_j = i · X_j · Z_j  ⇒  C Y_j C† = i · D · S
            //   where D = C X_j C†  and  S = C Z_j C†.
            // D and S anticommute (X_j, Z_j anticommute; conjugation preserves it),
            // so the accumulated per-qubit i-count is odd, making i · D·S a real
            // Hermitian Pauli with ±1 sign.
            let dx = map_pauli_through_tableau(tableau, site, 'X', 1.0);
            let dz = map_pauli_through_tableau(tableau, site, 'Z', 1.0);
            let mut x = vec![false; n];
            let mut z = vec![false; n];
            let mut minus_count: u32 = 0;
            let mut i_count: u32 = 0;
            for q in 0..n {
                x[q] = dx.x[q] ^ dz.x[q];
                z[q] = dx.z[q] ^ dz.z[q];
                let pa = pauli_idx(dx.x[q], dx.z[q]);
                let pb = pauli_idx(dz.x[q], dz.z[q]);
                let (m, i_b) = pauli_phase(pa, pb);
                minus_count += u32::from(m);
                i_count += u32::from(i_b);
            }
            // Total phase of i · D·S = i^{1 + i_count} · (-1)^{minus_count}.
            // Must be real (i_count odd). Result sign:
            //   i^{1+k} for k odd, k=2j+1 → i^{2j+2} = (-1)^{j+1}
            debug_assert_eq!(i_count % 2, 1, "C Y C† must be Hermitian");
            let j = (i_count - 1) / 2;
            let mut c = coef * dx.coef * dz.coef;
            if (j + 1) % 2 == 1 {
                c = -c;
            }
            if minus_count % 2 == 1 {
                c = -c;
            }
            PauliString { x, z, coef: c }
        }
        'I' => PauliString::identity(n),
        _ => panic!("invalid pauli: {pauli}"),
    }
}

/// Y-convention Pauli index (I=0, Z=1, X=2, Y=3) from (x, z) bits.
fn pauli_idx(x: bool, z: bool) -> u8 {
    match (x, z) {
        (false, false) => 0,
        (false, true) => 1,
        (true, false) => 2,
        (true, true) => 3,
    }
}

/// Phase (`minus_bit`, `i_bit`) of `Pauli_a` · `Pauli_b` with I=0,Z=1,X=2,Y=3.
/// Encoding: +1=(0,0), -1=(1,0), +i=(0,1), -i=(1,1).
fn pauli_phase(a: u8, b: u8) -> (u8, u8) {
    match (a, b) {
        (1, 2) | (3, 1) | (2, 3) => (0, 1), // +i: Z·X, Y·Z, X·Y
        (2, 1) | (1, 3) | (3, 2) => (1, 1), // -i: X·Z, Z·Y, Y·X
        _ => (0, 0),                        // I or same Pauli
    }
}

/// (Deprecated / unused) Gauss-eliminate Pauli strings to find subset
/// supported only on region A. Kept for reference; PCE now enumerates
/// per-site choices directly since generators from same site anti-commute.
#[allow(dead_code)]
pub(crate) fn restrict_to_region_a(
    generators: Vec<PauliString>,
    region_a_mask: &[bool],
    num_qubits: usize,
) -> Vec<PauliString> {
    let region_b_mask: Vec<bool> = region_a_mask.iter().map(|b| !b).collect();
    // Build matrix: each row is 2N-bit (x then z), coefficient tracked separately.
    // Eliminate rows by pivoting on region_b bits first.
    let mut rows: Vec<PauliString> = generators;
    // Bits of region B: for each qubit q in B, the x bit (position q) and z bit
    // (position num_qubits + q). We want to zero these out.
    let mut b_bit_positions: Vec<(usize, bool)> = Vec::new(); // (qubit, is_x)
    for (q, &is_b) in region_b_mask.iter().enumerate() {
        if is_b {
            b_bit_positions.push((q, true)); // x-bit
            b_bit_positions.push((q, false)); // z-bit
        }
    }

    let row_bit = |row: &PauliString, q: usize, is_x: bool| -> bool {
        if is_x { row.x[q] } else { row.z[q] }
    };

    let mut current = 0;
    for &(q, is_x) in &b_bit_positions {
        if current >= rows.len() {
            break;
        }
        // Find row with 1 in this bit.
        let found = rows[current..]
            .iter()
            .position(|row| row_bit(row, q, is_x))
            .map(|offset| current + offset);
        if let Some(piv) = found {
            rows.swap(current, piv);
            for r in 0..rows.len() {
                if r != current && row_bit(&rows[r], q, is_x) {
                    // Combine rows as Pauli product: P_r · P_current.
                    // Bits XOR; coefficient picks up per-qubit phase.
                    // For commuting-generator case, total phase is real (±1);
                    // i-count must be even.
                    let mut minus_count: u32 = 0;
                    let mut i_count: u32 = 0;
                    for qq in 0..num_qubits {
                        let pa = pauli_idx(rows[r].x[qq], rows[r].z[qq]);
                        let pb = pauli_idx(rows[current].x[qq], rows[current].z[qq]);
                        let (m, i_b) = pauli_phase(pa, pb);
                        minus_count += u32::from(m);
                        i_count += u32::from(i_b);
                        rows[r].x[qq] ^= rows[current].x[qq];
                        rows[r].z[qq] ^= rows[current].z[qq];
                    }
                    // PCE assumes generators mutually commute → i_count even.
                    // Non-commuting generators (e.g. same-site X and Y) violate
                    // this; we approximate with |phase|.
                    let j = i_count / 2;
                    let mut combined = rows[r].coef * rows[current].coef;
                    if j % 2 == 1 {
                        combined = -combined;
                    }
                    if minus_count % 2 == 1 {
                        combined = -combined;
                    }
                    rows[r].coef = combined;
                }
            }
            current += 1;
        }
    }

    // Return rows with zero support on B.
    rows.into_iter()
        .filter(|r| r.supported_on(region_a_mask))
        .collect()
}

/// Compute `S_2` entropy via Pauli Coefficient Enumeration (PCE).
///
/// Formula: `Tr(ρ_A²)` = (`1/2^{N_A`}) · Σ_{P̃ : supp(C P̃ C†) ⊆ A} ∏_j `c_j(P̃_j)²`
/// where `c_j(P̃_j)` = `Tr(ρ_j` · `P̃_j`) ∈ {1, `r_x`, `r_y`, `r_z`}.
///
/// We enumerate site-independent choices (I or one of the non-zero Paulis at
/// each site), map the product through the tableau, and keep terms supported
/// on A. Combinations count is ∏_j (1 + `count_j`). For Clifford+T circuits,
/// most sites have `count_j` ∈ {0,1}; full-magic sites have `count_j` = 3 giving
/// up to 4^N. Errors out above 2^22 combos.
///
/// # Errors
///
/// Returns an error string if the mask length doesn't match the MPS, or if
/// the number of Pauli combinations exceeds the safety limit.
///
/// # Panics
///
/// Panics if the region-A size exceeds u16 range (would require > 65535 qubits).
pub fn compute_s2_pce(
    mps: &Mps,
    tableau: &SparseStabY,
    region_a_mask: &[bool],
) -> Result<f64, String> {
    let n = mps.num_sites();
    if region_a_mask.len() != n {
        return Err("region_a_mask length mismatch".into());
    }

    let bvs = bloch_vectors(mps)?;
    let tol = 1e-12;

    // For each site, list the available (coef, mapped_PauliString) choices.
    // The identity choice has coef=1 and a zero PauliString on all qubits.
    // Non-identity choices include non-zero X/Y/Z Bloch components.
    let mut site_choices: Vec<Vec<(f64, PauliString)>> = Vec::with_capacity(n);
    let mut total_combos: u128 = 1;
    for (j, &(rx, ry, rz)) in bvs.iter().enumerate() {
        let mut opts: Vec<(f64, PauliString)> = Vec::with_capacity(4);
        opts.push((1.0, PauliString::identity(n)));
        if rx.abs() > tol {
            opts.push((rx, map_pauli_through_tableau(tableau, j, 'X', 1.0)));
        }
        if ry.abs() > tol {
            opts.push((ry, map_pauli_through_tableau(tableau, j, 'Y', 1.0)));
        }
        if rz.abs() > tol {
            opts.push((rz, map_pauli_through_tableau(tableau, j, 'Z', 1.0)));
        }
        total_combos = total_combos.saturating_mul(opts.len() as u128);
        site_choices.push(opts);
    }
    if total_combos > (1u128 << 22) {
        return Err(format!("PCE would enumerate {total_combos} > limit 2^22"));
    }

    // Enumerate all combinations as mixed-radix index across sites.
    let n_a: usize = region_a_mask.iter().filter(|&&b| b).count();
    let mut tr_sq: f64 = 0.0;
    let mut idx = vec![0usize; n];
    loop {
        // Combine: product of per-site Bloch coefficients × XOR of mapped Paulis
        // (cross-site Paulis commute so total sign is product of each mapped coef's sign).
        let mut combined_x = vec![false; n];
        let mut combined_z = vec![false; n];
        let mut coef = 1.0;
        for (j, opts) in site_choices.iter().enumerate() {
            let (bloch, ps) = &opts[idx[j]];
            coef *= bloch;
            // Accumulate Pauli product (cross-site, so just XOR; per-qubit phase
            // is trivial because different-site Paulis share no qubit support in
            // P̃, but the *mapped* ps spans all qubits, so we still track phase).
            let mut minus_count: u32 = 0;
            let mut i_count: u32 = 0;
            for q in 0..n {
                let pa = pauli_idx(combined_x[q], combined_z[q]);
                let pb = pauli_idx(ps.x[q], ps.z[q]);
                let (m, i_b) = pauli_phase(pa, pb);
                minus_count += u32::from(m);
                i_count += u32::from(i_b);
                combined_x[q] ^= ps.x[q];
                combined_z[q] ^= ps.z[q];
            }
            // All mapped P_j from different sites commute → i_count even.
            debug_assert_eq!(i_count % 2, 0, "cross-site mapped Paulis must commute");
            let j_half = i_count / 2;
            coef *= ps.coef;
            if (j_half + minus_count) % 2 == 1 {
                coef = -coef;
            }
        }

        // Check support on region A.
        let mut on_a = true;
        for q in 0..n {
            if !region_a_mask[q] && (combined_x[q] || combined_z[q]) {
                on_a = false;
                break;
            }
        }
        if on_a {
            tr_sq += coef * coef;
        }

        // Advance mixed-radix counter.
        let mut carry = true;
        for j in 0..n {
            if !carry {
                break;
            }
            idx[j] += 1;
            if idx[j] >= site_choices[j].len() {
                idx[j] = 0;
            } else {
                carry = false;
            }
        }
        if carry {
            break;
        }
    }
    // n_a is bounded by the number of qubits (enforced by early checks) so n_a <= 22.
    tr_sq *= 0.5_f64.powi(i32::from(u16::try_from(n_a).expect("n_a fits in u16")));

    if tr_sq < 1e-30 {
        Ok(f64::INFINITY)
    } else {
        Ok(-tr_sq.ln())
    }
}

/// `S_2` via full `F_2` enumeration of the 2N-bit Pauli null-space with
/// site-separable squared-weights. Handles arbitrary (multi-axis) Bloch.
///
/// Formula: `Tr(ρ_A²)` = (`1/2^{N_A`}) · Σ_{P̃ : supp(C P̃ C†) ⊆ A} ∏_j `w_j(P̃_j)`
/// where weights are squared Bloch components in Y-convention:
///   `w_j(x̃=0,z̃=0)` = 1                 (I)
///   `w_j(x̃=1,z̃=0)` = `r_x²`               (X)
///   `w_j(x̃=0,z̃=1)` = `r_z²`               (Z)
///   `w_j(x̃=1,z̃=1)` = `r_y²`               (XZ ≈ -iY in standard Pauli)
///
/// Constraints: 2(N-N_A) linear parity checks over `F_2^{2N`} encoding that
/// (C P̃ C†) has no support on region B. Gauss-eliminate to find null
/// space of dim `d`; enumerate 2^d terms. Errors out above 2^22.
///
/// Covers the single-axis case as a special case (inactive axes add
/// unit-weight constraints that reduce effective null dim).
///
/// # Errors
///
/// Returns an error string if the mask length doesn't match the MPS, or if
/// the null-space dimension exceeds the enumeration limit.
///
/// # Panics
///
/// Panics if the region-A size exceeds u16 range (would require > 65535 qubits).
pub fn compute_s2_pcmps_tn(
    mps: &Mps,
    tableau: &SparseStabY,
    region_a_mask: &[bool],
) -> Result<f64, String> {
    let n = mps.num_sites();
    if region_a_mask.len() != n {
        return Err("region_a_mask length mismatch".into());
    }
    let bvs = bloch_vectors(mps)?;
    let tol = 1e-12;

    // Variables: for site j, x̃_j = bit 2j, z̃_j = bit 2j+1. Total 2N bits.
    let n_bits = 2 * n;

    // Per-site weights indexed by (x̃, z̃) ∈ {0,1}² ordered 00, 10, 01, 11.
    // Also record which patterns are forced zero (constraints to add).
    let mut weights: Vec<[f64; 4]> = Vec::with_capacity(n);
    let mut zero_bit_constraints: Vec<Vec<bool>> = Vec::new();
    for (j, &(rx, ry, rz)) in bvs.iter().enumerate() {
        let rx2 = rx * rx;
        let ry2 = ry * ry;
        let rz2 = rz * rz;
        weights.push([1.0, rx2, rz2, ry2]);
        // Zero-weight patterns ⇒ force bit combinations to 0.
        // If rx = 0 AND ry = 0: force x̃_j = 0 (eliminates X and Y options).
        if rx2 < tol && ry2 < tol {
            let mut row = vec![false; n_bits];
            row[2 * j] = true;
            zero_bit_constraints.push(row);
        }
        // If rz = 0 AND ry = 0: force z̃_j = 0.
        if rz2 < tol && ry2 < tol {
            let mut row = vec![false; n_bits];
            row[2 * j + 1] = true;
            zero_bit_constraints.push(row);
        }
        // If rx = 0 AND rz = 0 (only Y): force x̃_j = z̃_j.
        if rx2 < tol && rz2 < tol && ry2 > tol {
            let mut row = vec![false; n_bits];
            row[2 * j] = true;
            row[2 * j + 1] = true;
            zero_bit_constraints.push(row);
        }
    }

    // Support-on-A constraints: for each B-site q, 2 linear constraints (x and z bits of CP̃C†).
    // (CP̃C†)_q x-bit = ⊕_j (x̃_j · destab[j].row_x[q] ⊕ z̃_j · stab[j].row_x[q])
    // (CP̃C†)_q z-bit = ⊕_j (x̃_j · destab[j].row_z[q] ⊕ z̃_j · stab[j].row_z[q])
    let destabs = tableau.destabs();
    let stabs = tableau.stabs();
    let mut support_constraints: Vec<Vec<bool>> = Vec::new();
    for (q, &is_a) in region_a_mask.iter().enumerate() {
        if is_a {
            continue;
        }
        let mut row_x = vec![false; n_bits];
        let mut row_z = vec![false; n_bits];
        for j in 0..n {
            if destabs.row_x[j].contains(q) {
                row_x[2 * j] ^= true;
            }
            if stabs.row_x[j].contains(q) {
                row_x[2 * j + 1] ^= true;
            }
            if destabs.row_z[j].contains(q) {
                row_z[2 * j] ^= true;
            }
            if stabs.row_z[j].contains(q) {
                row_z[2 * j + 1] ^= true;
            }
        }
        if row_x.iter().any(|&b| b) {
            support_constraints.push(row_x);
        }
        if row_z.iter().any(|&b| b) {
            support_constraints.push(row_z);
        }
    }

    // Combine all constraints.
    let mut a_rows: Vec<Vec<bool>> = Vec::new();
    a_rows.extend(zero_bit_constraints);
    a_rows.extend(support_constraints);
    let n_rows = a_rows.len();

    // RREF over F_2.
    let mut pivot_col_of_row: Vec<Option<usize>> = vec![None; n_rows];
    let mut col_is_pivot: Vec<bool> = vec![false; n_bits];
    let mut r = 0;
    for c in 0..n_bits {
        if r >= n_rows {
            break;
        }
        let found = a_rows[r..]
            .iter()
            .position(|row| row[c])
            .map(|offset| r + offset);
        if let Some(rr) = found {
            a_rows.swap(r, rr);
            let pivot_row = a_rows[r].clone();
            for (rr, row) in a_rows.iter_mut().enumerate() {
                if rr != r && row[c] {
                    for (cell, &piv) in row.iter_mut().zip(pivot_row.iter()) {
                        *cell ^= piv;
                    }
                }
            }
            pivot_col_of_row[r] = Some(c);
            col_is_pivot[c] = true;
            r += 1;
        }
    }
    let rank = r;
    let free_cols: Vec<usize> = (0..n_bits).filter(|&c| !col_is_pivot[c]).collect();
    let null_dim = free_cols.len();

    let n_a: usize = region_a_mask.iter().filter(|&&b| b).count();

    if null_dim > 30 {
        return Err(format!("PCMPS-TN null-space dim {null_dim} > 30"));
    }

    // Build null-space basis vectors. Pack as bitmasks:
    //   - For n ≤ 32 (n_bits ≤ 64): single u128 per basis vector.
    //   - Otherwise: Vec<u64>.
    // Per-site weight lookup then becomes a single shift+mask.
    let basis_u128: Option<Vec<u128>> = if n_bits <= 128 {
        Some(
            free_cols
                .iter()
                .map(|&f| {
                    let mut bits: u128 = 1u128 << f;
                    for rr in 0..rank {
                        if let Some(p) = pivot_col_of_row[rr]
                            && a_rows[rr][f]
                        {
                            bits ^= 1u128 << p;
                        }
                    }
                    bits
                })
                .collect(),
        )
    } else {
        None
    };

    let total_combos = 1usize << null_dim;

    let accumulate_combo_u128 = |basis_masks: &[u128], combo: usize| -> f64 {
        let mut bits: u128 = 0;
        for (k, &mask) in basis_masks.iter().enumerate() {
            if (combo >> k) & 1 == 1 {
                bits ^= mask;
            }
        }
        let mut w: f64 = 1.0;
        for (j, wj) in weights.iter().enumerate() {
            let idx = ((bits >> (2 * j)) & 0b11) as usize;
            w *= wj[idx];
            if w == 0.0 {
                return 0.0;
            }
        }
        w
    };

    let tr_sq: f64 = if let Some(basis_masks) = basis_u128.as_ref() {
        use rayon::prelude::*;
        if total_combos >= (1 << 14) {
            (0..total_combos)
                .into_par_iter()
                .map(|combo| accumulate_combo_u128(basis_masks, combo))
                .sum()
        } else {
            (0..total_combos)
                .map(|combo| accumulate_combo_u128(basis_masks, combo))
                .sum()
        }
    } else {
        // Fall back: boolean-vector enumeration (slow but correct for n > 64).
        let mut basis: Vec<Vec<bool>> = Vec::with_capacity(null_dim);
        for &f in &free_cols {
            let mut v = vec![false; n_bits];
            v[f] = true;
            for rr in 0..rank {
                if let Some(p) = pivot_col_of_row[rr]
                    && a_rows[rr][f]
                {
                    v[p] = true;
                }
            }
            basis.push(v);
        }
        let mut sum: f64 = 0.0;
        for combo in 0..total_combos {
            let mut bits = vec![false; n_bits];
            for (k, bk) in basis.iter().enumerate() {
                if (combo >> k) & 1 == 1 {
                    for (bi, &bki) in bits.iter_mut().zip(bk.iter()) {
                        *bi ^= bki;
                    }
                }
            }
            let mut w: f64 = 1.0;
            for j in 0..n {
                let x = bits[2 * j];
                let z = bits[2 * j + 1];
                let idx = usize::from(x) | (usize::from(z) << 1);
                w *= weights[j][idx];
                if w == 0.0 {
                    break;
                }
            }
            sum += w;
        }
        sum
    };
    let tr_sq = tr_sq / f64::from(1u32 << u32::try_from(n_a).unwrap());

    if tr_sq < 1e-30 {
        Ok(f64::INFINITY)
    } else {
        Ok(-tr_sq.ln())
    }
}

/// Faster `S_2` via GF(2) null-space enumeration (PCMPS-style).
///
/// Applicable when every MPS site has exactly ONE non-zero Bloch component
/// (i.e. lies on a Pauli axis). Most STN Clifford+T states satisfy this
/// because T gates absorb into the tableau, leaving MPS sites as |0⟩ (Z-axis).
///
/// Algorithm:
/// 1. Each site j contributes one binary variable `v_j` ∈ {0,1} (I vs `P_j`).
/// 2. Support-on-A constraint on (C·P·C†) is a system of linear equations
///    in `v_j` over GF(2).
/// 3. Enumerate only the null space (dim d), not 2^N combos — 2^d evaluations.
///
/// For single-axis states this is typically d ≈ `N_A`, i.e. `2^{N_A`} not 2^N.
/// Returns error if any site has multi-axis Bloch (caller can fall back to PCE).
///
/// # Errors
///
/// Returns an error string if the mask length doesn't match the MPS, if any
/// site has multi-axis Bloch vectors, or if the null-space exceeds the
/// enumeration limit.
///
/// # Panics
///
/// Panics if the region-A or null-space size exceeds i16 range.
pub fn compute_s2_pcmps(
    mps: &Mps,
    tableau: &SparseStabY,
    region_a_mask: &[bool],
) -> Result<f64, String> {
    let n = mps.num_sites();
    if region_a_mask.len() != n {
        return Err("region_a_mask length mismatch".into());
    }
    let bvs = bloch_vectors(mps)?;
    let tol = 1e-12;

    // Per-site single-axis variable: (bloch_coef, mapped_pauli_string).
    // Fail fast if any site has ≥ 2 non-zero Bloch components.
    let mut vars: Vec<(f64, PauliString)> = Vec::with_capacity(n);
    for (j, &(rx, ry, rz)) in bvs.iter().enumerate() {
        let cands = [(rx, 'X'), (ry, 'Y'), (rz, 'Z')];
        let nonzero: Vec<&(f64, char)> = cands.iter().filter(|(r, _)| r.abs() > tol).collect();
        if nonzero.len() != 1 {
            return Err(format!(
                "PCMPS needs 1 non-zero Bloch axis per site; site {j} has {} (rx={rx}, ry={ry}, rz={rz})",
                nonzero.len()
            ));
        }
        let (r, p) = *nonzero[0];
        vars.push((r, map_pauli_through_tableau(tableau, j, p, 1.0)));
    }

    // Build GF(2) constraint matrix A (rows = B-site bit constraints, cols = vars).
    // For each B site q, two rows (x-bit, z-bit) constraining the combined Pauli
    // to have 0 at that bit position.
    let n_vars = vars.len();
    let mut a_rows: Vec<Vec<bool>> = Vec::new();
    for (q, &is_a) in region_a_mask.iter().enumerate() {
        if is_a {
            continue;
        }
        let row_x: Vec<bool> = vars.iter().map(|v| v.1.x[q]).collect();
        let row_z: Vec<bool> = vars.iter().map(|v| v.1.z[q]).collect();
        if row_x.iter().any(|&b| b) {
            a_rows.push(row_x);
        }
        if row_z.iter().any(|&b| b) {
            a_rows.push(row_z);
        }
    }

    // Gauss-eliminate to RREF; record pivot cols and free cols.
    let n_rows = a_rows.len();
    let mut pivot_col_of_row: Vec<Option<usize>> = vec![None; n_rows];
    let mut col_is_pivot: Vec<bool> = vec![false; n_vars];
    let mut r = 0;
    for c in 0..n_vars {
        if r >= n_rows {
            break;
        }
        let found = a_rows[r..]
            .iter()
            .position(|row| row[c])
            .map(|offset| r + offset);
        if let Some(rr) = found {
            a_rows.swap(r, rr);
            let pivot_row = a_rows[r].clone();
            for (rr, row) in a_rows.iter_mut().enumerate() {
                if rr != r && row[c] {
                    for (cell, &piv) in row.iter_mut().zip(pivot_row.iter()) {
                        *cell ^= piv;
                    }
                }
            }
            pivot_col_of_row[r] = Some(c);
            col_is_pivot[c] = true;
            r += 1;
        }
    }
    let rank = r;
    let free_cols: Vec<usize> = (0..n_vars).filter(|&c| !col_is_pivot[c]).collect();
    let null_dim = free_cols.len();
    debug_assert_eq!(rank + null_dim, n_vars);

    let n_a: usize = region_a_mask.iter().filter(|&&b| b).count();

    // Short-circuit: all-Clifford states have |var_coef · ps.coef| = 1 at every
    // variable. Then every null-space combination contributes coef² = 1,
    // so tr_sq = 2^null_dim / 2^N_A.
    let all_clifford = vars
        .iter()
        .all(|(r, ps)| (r.abs() * ps.coef.abs() - 1.0).abs() < 1e-9);
    if all_clifford {
        let diff = i16::try_from(null_dim).expect("null_dim fits in i16")
            - i16::try_from(n_a).expect("n_a fits in i16");
        let s2 = -f64::from(diff) * (2.0f64).ln();
        return Ok(s2);
    }

    if null_dim > 22 {
        return Err(format!(
            "PCMPS null-space dim {null_dim} > 22 (non-Clifford)"
        ));
    }

    // Null-space basis: for each free col f, basis vector e_f has v_f = 1 and
    // v_p = A[r_p][f] for each pivot row r_p (pivot col p).
    let mut basis: Vec<Vec<bool>> = Vec::with_capacity(null_dim);
    for &f in &free_cols {
        let mut v = vec![false; n_vars];
        v[f] = true;
        for rr in 0..rank {
            if let Some(p) = pivot_col_of_row[rr]
                && a_rows[rr][f]
            {
                v[p] = true;
            }
        }
        basis.push(v);
    }

    // Enumerate 2^null_dim combinations of basis vectors; for each, compute
    // product coefficient and accumulate coef².
    let total_combos = 1usize << null_dim;
    let mut tr_sq: f64 = 0.0;
    for combo in 0..total_combos {
        // XOR combination of basis vectors selected by bits of combo.
        let mut selection = vec![false; n_vars];
        for (k, bk) in basis.iter().enumerate() {
            if (combo >> k) & 1 == 1 {
                for (sel, &bki) in selection.iter_mut().zip(bk.iter()) {
                    *sel ^= bki;
                }
            }
        }
        // Compute combined Pauli string and coefficient.
        let mut cx = vec![false; n];
        let mut cz = vec![false; n];
        let mut coef: f64 = 1.0;
        let mut mc: u32 = 0;
        let mut ic: u32 = 0;
        for (i, sel) in selection.iter().enumerate() {
            if !sel {
                continue;
            }
            let (bloch, ps) = &vars[i];
            coef *= bloch * ps.coef;
            for q in 0..n {
                let pa = pauli_idx(cx[q], cz[q]);
                let pb = pauli_idx(ps.x[q], ps.z[q]);
                let (m, i_b) = pauli_phase(pa, pb);
                mc += u32::from(m);
                ic += u32::from(i_b);
                cx[q] ^= ps.x[q];
                cz[q] ^= ps.z[q];
            }
        }
        debug_assert_eq!(ic % 2, 0, "null-space combos must give even i-count");
        if (ic / 2 + mc) % 2 == 1 {
            coef = -coef;
        }
        debug_assert!(
            cx.iter()
                .zip(&cz)
                .enumerate()
                .all(|(q, (x, z))| region_a_mask[q] || (!x && !z)),
            "null-space vector violated support constraint (GF(2) bug)"
        );
        tr_sq += coef * coef;
    }
    // n_a is bounded by the number of qubits (enforced by early checks) so n_a <= 22.
    tr_sq *= 0.5_f64.powi(i32::from(u16::try_from(n_a).expect("n_a fits in u16")));

    if tr_sq < 1e-30 {
        Ok(f64::INFINITY)
    } else {
        Ok(-tr_sq.ln())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stab_mps::StabMps;
    use pecos_core::QubitId;
    use pecos_simulators::CliffordGateable;

    #[test]
    fn test_bloch_vector_zero_state() {
        let stn = StabMps::new(3);
        let bvs = bloch_vectors(stn.mps()).unwrap();
        // |0⟩ state: (0, 0, +1).
        for (rx, ry, rz) in bvs {
            assert!((rx - 0.0).abs() < 1e-9);
            assert!((ry - 0.0).abs() < 1e-9);
            assert!((rz - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn test_pauli_string_supported_on() {
        let mut p = PauliString::identity(4);
        p.x[0] = true;
        p.z[1] = true;
        let region_a = vec![true, true, false, false];
        assert!(p.supported_on(&region_a));
        p.x[2] = true;
        assert!(!p.supported_on(&region_a));
    }

    #[test]
    fn test_map_pauli_identity_tableau() {
        // Trivial tableau (identity Clifford): C P C† = P.
        // X at site 0 maps to X_0 on all qubits.
        let stn = StabMps::new(3);
        let g = map_pauli_through_tableau(stn.tableau(), 0, 'X', 1.0);
        assert!(g.x[0] && !g.z[0]);
        assert!(!g.x[1] && !g.z[1]);
        assert!(!g.x[2] && !g.z[2]);
        assert!((g.coef - 1.0).abs() < 1e-9);

        // Z at site 1.
        let g = map_pauli_through_tableau(stn.tableau(), 1, 'Z', 1.0);
        assert!(!g.x[1] && g.z[1]);
        assert!(!g.x[0] && !g.z[0]);
    }

    #[test]
    fn test_pce_s2_zero_state() {
        // |0⟩^N: all stabilized by Z_j. S_2 = 0 for any bipartition.
        let stn = StabMps::new(4);
        let mask = vec![true, true, false, false];
        let s2 = compute_s2_pce(stn.mps(), stn.tableau(), &mask).unwrap();
        eprintln!("zero state S_2 = {s2}");
        assert!(s2.abs() < 1e-9, "zero state should have S_2=0, got {s2}");
    }

    #[test]
    fn test_pce_s2_bell_state() {
        // Bell state: S_2 = ln(2).
        let mut stn = StabMps::new(2);
        stn.h(&[QubitId(0)]);
        stn.cx(&[(QubitId(0), QubitId(1))]);
        let mask = vec![true, false]; // q0 in A, q1 in B
        let s2 = compute_s2_pce(stn.mps(), stn.tableau(), &mask).unwrap();
        eprintln!("Bell S_2 (PCE) = {s2}, expected ln(2) = {}", (2.0f64).ln());
        assert!((s2 - (2.0f64).ln()).abs() < 1e-9);
    }

    #[test]
    fn test_pce_matches_sv_for_clifford_plus_t() {
        use pecos_core::Angle64;
        use pecos_simulators::ArbitraryRotationGateable;
        // H on all, CX, T on q0 (creates entangled magic state),
        // CX between regions. Real Clifford+T circuit.
        let mut stn = StabMps::new(4);
        stn.h(&[QubitId(0), QubitId(1), QubitId(2), QubitId(3)]);
        stn.cx(&[(QubitId(0), QubitId(2))]); // entangle A-B boundary
        stn.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(0)]); // T on q0
        stn.cx(&[(QubitId(1), QubitId(3))]); // entangle more
        let mask = vec![true, true, false, false]; // A = {0,1}, B = {2,3}

        let s2_pce = compute_s2_pce(stn.mps(), stn.tableau(), &mask).unwrap();
        let s2_sv = stn.renyi_s2(2);
        eprintln!("PCE: {s2_pce:.6}, SV: {s2_sv:.6}");
        assert!(
            (s2_pce - s2_sv).abs() < 1e-6,
            "PCE should match SV now that Y-sign is tracked: PCE={s2_pce} SV={s2_sv}"
        );
    }

    #[test]
    fn test_pce_entangled_clifford_plus_t() {
        // Genuinely entangled Clifford+T across A-B boundary.
        use pecos_core::Angle64;
        use pecos_simulators::ArbitraryRotationGateable;
        let mut stn = StabMps::new(4);
        stn.h(&[QubitId(0)]);
        stn.cx(&[(QubitId(0), QubitId(2))]); // Bell across A-B
        stn.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(2)]); // T on B-side
        stn.h(&[QubitId(1)]);
        stn.cx(&[(QubitId(1), QubitId(3))]); // second Bell across A-B
        let mask = vec![true, true, false, false];
        let s2_pce = compute_s2_pce(stn.mps(), stn.tableau(), &mask).unwrap();
        let s2_sv = stn.renyi_s2(2);
        eprintln!("entangled PCE: {s2_pce:.6}, SV: {s2_sv:.6}");
        assert!(
            (s2_pce - s2_sv).abs() < 1e-6,
            "PCE should match SV for entangled Clifford+T: PCE={s2_pce} SV={s2_sv}"
        );
        assert!(s2_pce > 0.5, "expected non-trivial entanglement");
    }

    #[test]
    fn test_map_pauli_after_cx() {
        // C = CX(0,1). Z_0 unchanged, Z_1 -> Z_0 Z_1, X_0 -> X_0 X_1, X_1 unchanged.
        let mut stn = StabMps::new(2);
        stn.cx(&[(QubitId(0), QubitId(1))]);
        let gz0 = map_pauli_through_tableau(stn.tableau(), 0, 'Z', 1.0);
        assert!(!gz0.x[0] && gz0.z[0]); // Z_0
        assert!(!gz0.x[1] && !gz0.z[1]);
        let gz1 = map_pauli_through_tableau(stn.tableau(), 1, 'Z', 1.0);
        assert!(!gz1.x[0] && gz1.z[0]); // Z_0
        assert!(!gz1.x[1] && gz1.z[1]); // Z_1
        let gx0 = map_pauli_through_tableau(stn.tableau(), 0, 'X', 1.0);
        assert!(gx0.x[0] && !gx0.z[0]); // X_0
        assert!(gx0.x[1] && !gx0.z[1]); // X_1
    }
}
