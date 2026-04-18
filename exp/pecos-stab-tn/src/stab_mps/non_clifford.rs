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

//! Non-Clifford gate protocol for the STN simulator.
//!
//! Applies RZ(theta) on the MPS using the rotation decomposition approach
//! from the stabilizer-TN reference implementation. Single-site cases use
//! direct 2x2 gates. Multi-site cases use either MPS addition (for all-X
//! or all-Z Pauli strings) or CNOT cascade + RX rotation + basis changes
//! (for mixed Pauli strings with Y overlaps).
//!
//! References:
//! - Masot-Llima, Garcia-Saez. arXiv:2403.08724 (STN protocol).
//! - Reference code: stabilizer-TN `update_xvec` and `apply_xvec_rot`.

use super::pauli_decomp::{ZDecomposition, decompose_z};
use crate::mps::Mps;
use nalgebra::DMatrix;
use num_complex::Complex64;
use pecos_simulators::SparseStabY;

fn z_diag() -> [Complex64; 2] {
    [Complex64::new(1.0, 0.0), Complex64::new(-1.0, 0.0)]
}

fn h_gate() -> DMatrix<Complex64> {
    let r = Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0);
    DMatrix::from_row_slice(2, 2, &[r, r, r, -r])
}

fn s_gate() -> DMatrix<Complex64> {
    DMatrix::from_row_slice(
        2,
        2,
        &[
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 1.0),
        ],
    )
}

fn sdg_gate() -> DMatrix<Complex64> {
    DMatrix::from_row_slice(
        2,
        2,
        &[
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, -1.0),
        ],
    )
}

fn rx_gate(theta: f64) -> DMatrix<Complex64> {
    let half = theta / 2.0;
    let c = Complex64::new(half.cos(), 0.0);
    let s = Complex64::new(0.0, -half.sin());
    DMatrix::from_row_slice(2, 2, &[c, s, s, c])
}

/// CNOT with first qubit (lower index) as control.
fn cnot_lo_ctrl() -> DMatrix<Complex64> {
    let o = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    DMatrix::from_row_slice(
        4,
        4,
        &[one, o, o, o, o, one, o, o, o, o, o, one, o, o, one, o],
    )
}

/// CNOT with second qubit (higher index) as control.
fn cnot_hi_ctrl() -> DMatrix<Complex64> {
    let o = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    DMatrix::from_row_slice(
        4,
        4,
        &[one, o, o, o, o, o, o, one, o, o, one, o, o, one, o, o],
    )
}

/// Per-site Pauli type for the rotation decomposition.
#[derive(Clone, Copy, Debug, PartialEq)]
enum PauliType {
    X,
    Z,
    Y, // Both flip AND sign on the same site. In Y convention: Y = iXZ (Hermitian).
}

/// Mutable context carried alongside the rotation decomposition.
pub struct RzContext<'a> {
    /// Per-site disentangling eigenstate flags.
    pub disent_flags: &'a mut [Option<super::SiteEigenstate>],
    /// GF(2) flip matrix for OFD diagnostics.
    pub gf2_matrix: &'a mut super::ofd::Gf2FlipMatrix,
    /// Running statistics for the STN simulator.
    pub stats: &'a mut super::StabMpsStats,
}

/// Apply RZ(theta) on qubit q using the rotation decomposition.
///
/// Ported from stabilizer-TN's `update_xvec` and `apply_xvec_rot`.
/// Takes &mut tableau because the disentangle path composes compensating
/// Cliffords via right-composition.
///
/// # Panics
///
/// Panics if any MPS gate application fails on a valid site.
pub fn apply_rz_stab_mps(
    tableau: &mut SparseStabY,
    mps: &mut Mps,
    cos_half: f64,
    sin_half: f64,
    q: usize,
    normalize: bool,
    ctx: &mut RzContext<'_>,
) {
    let RzContext {
        disent_flags,
        gf2_matrix,
        stats,
    } = ctx;
    stats.total_nonclifford += 1;
    let decomp = decompose_z(tableau.stabs(), tableau.destabs(), q);

    // OFD diagnostic: check whether this gate's flip pattern is in the span
    // of previously-recorded patterns. OFD says such gates can be implemented
    // without bond-dim growth (using already-tracked flip structure).
    // We also capture this BEFORE the branch decisions for cross-tab stats.
    let is_ofd_in_span = if let ZDecomposition::DestabilizerFlip { ref flip_sites, .. } = decomp {
        let flip_vec: Vec<usize> = flip_sites.clone();
        let in_span = gf2_matrix.is_in_span(&flip_vec);
        if in_span {
            stats.ofd_in_span += 1;
        } else {
            stats.ofd_new_dim += 1;
        }
        in_span
    } else {
        false
    };

    match decomp {
        ZDecomposition::Stabilizer {
            phase,
            ref sign_sites,
        } => {
            stats.stabilizer += 1;
            if sign_sites.is_empty() {
                let scalar = Complex64::new(cos_half, 0.0) - Complex64::new(0.0, sin_half) * phase;
                mps.scale(scalar);
                // Scalar multiply doesn't modify site states -- flags unchanged.
            } else if sign_sites.len() == 1 {
                let k = sign_sites[0];
                let c0 = Complex64::new(cos_half, 0.0) - Complex64::new(0.0, sin_half) * phase;
                let c1 = Complex64::new(cos_half, 0.0) + Complex64::new(0.0, sin_half) * phase;
                mps.apply_diagonal_one_site(k, &[c0, c1])
                    .expect("sign_site should be valid");
                disent_flags[k] = None;
            } else {
                // Multi-site Z diagonal via MPS addition (exact, no SVD until compress).
                let mut mps_z = mps.clone();
                let zd = z_diag();
                for &j in sign_sites {
                    mps_z
                        .apply_diagonal_one_site(j, &zd)
                        .expect("MPS op on valid site");
                }
                let scale2 = Complex64::new(0.0, -sin_half) * phase;
                mps_z.scale(scale2);
                mps.scale(Complex64::new(cos_half, 0.0));
                *mps = mps.add(&mps_z);
                mps.compress();
                for &j in sign_sites {
                    disent_flags[j] = None;
                }
            }
        }

        ZDecomposition::DestabilizerFlip {
            ref flip_sites,
            phase,
            ref sign_sites,
        } => {
            // Build the per-site Pauli map (ind_dict in the reference).
            // flip_sites -> X, sign_sites -> Z, both -> Y (= XZ = W)
            let mut pauli_map: Vec<(usize, PauliType)> = Vec::new();
            for &j in flip_sites {
                pauli_map.push((j, PauliType::X));
            }
            for &k in sign_sites {
                if let Some(entry) = pauli_map.iter_mut().find(|(s, _)| *s == k) {
                    entry.1 = PauliType::Y; // X + Z overlap -> Y (= W = XZ)
                } else {
                    pauli_map.push((k, PauliType::Z));
                }
            }

            let mut affected_sites: Vec<usize> = pauli_map.iter().map(|(s, _)| *s).collect();
            affected_sites.sort_unstable(); // Chain cascade requires sorted order

            if affected_sites.is_empty() {
                return;
            }

            // OFD disentangle check (Liu-Clark 2412.17209 Algorithm 1 Theorem 1):
            //   disentanglable iff some qubit i has MPS state |0⟩ AND P[i] ∈ {X, Y}.
            // Our `disent_flags[i] = Some(Z(false))` means MPS is |0⟩ at i (fresh
            // qubit never touched by a non-Clifford). We only ever have flags
            // Z(false) or None after this session's semantic cleanup -- hence
            // the simple check below.
            let mut disent_site = None;
            if affected_sites.len() > 1 {
                for &(site, pt) in &pauli_map {
                    if matches!(pt, PauliType::X | PauliType::Y)
                        && matches!(disent_flags[site], Some(super::SiteEigenstate::Z(false)))
                    {
                        disent_site = Some(site);
                        break;
                    }
                }
            }

            if let Some(rot_site) = disent_site {
                stats.multi_disent += 1;
                if is_ofd_in_span {
                    stats.ofd_in_span_disent += 1;
                }
                // Record effective single-site flip pattern with rot_site metadata
                gf2_matrix.add_row_with_meta(&[rot_site], super::ofd::RowMetadata { rot_site });

                // Compute RX angle. Reference formula:
                //   co = -i*sin_half*phase * i^(Ys+1)
                // After correction co should be real = ±sin(θ/2).
                // We want RX(rx_angle)|0⟩ = cos(θ/2)|0⟩ - i·co·|1⟩, so
                // rx_angle/2 must satisfy cos(rx_angle/2) = cos(θ/2) AND
                // sin(rx_angle/2) = co. arcsin loses the cos sign for |θ/2| > π/2,
                // so use the ±1 sign of co combined with the full angle θ.
                let y_count = pauli_map.iter().filter(|(_, p)| *p == PauliType::Y).count();
                let mut co = Complex64::new(0.0, -sin_half) * phase;
                let i_val = Complex64::new(0.0, 1.0);
                let mut factor = Complex64::new(1.0, 0.0);
                for _ in 0..=y_count {
                    factor *= i_val;
                }
                co *= factor;
                debug_assert!(
                    co.im.abs() < 1e-8,
                    "co should be real after i^(Ys+1) correction: phase={phase}, Ys={y_count}, co={co}"
                );
                // co = sin(θ/2) · s where s = ±1. Recover s from sign of co vs sin_half.
                let rx_sign: f64 = if sin_half.abs() < 1e-12
                    || (co.re - sin_half).abs() < (co.re + sin_half).abs()
                {
                    1.0
                } else {
                    -1.0
                };
                // Full angle: θ = 2·atan2(sin_half, cos_half). Use θ such that
                // sin(rx_angle/2) matches co AND cos(rx_angle/2) = cos(θ/2).
                // So rx_angle = s · θ.
                let theta = 2.0 * sin_half.atan2(cos_half);
                let rx_angle = rx_sign * theta;

                // Masot-Llima basis+CNOT pattern (inherited from stabilizer-TN
                // reference). A direct CY/CZ "CP cascade" (Liu-Clark Algorithm 1)
                // was attempted but produced subtle sign mismatches for rot_pt=Y
                // cases driven by the phase pre-correction applied to rx_angle.
                // Both patterns are mathematically equivalent; default is the
                // tested one — keeping only it to avoid dual-path maintenance.
                let rot_pt = pauli_map
                    .iter()
                    .find(|&&(s, _)| s == rot_site)
                    .expect("rot_site must be in pauli_map")
                    .1;
                if matches!(rot_pt, PauliType::Y) {
                    mps.apply_one_site_gate(rot_site, &s_gate())
                        .expect("MPS op on valid site");
                }
                mps.apply_one_site_gate(rot_site, &rx_gate(rx_angle))
                    .expect("MPS op on valid site");
                for &(site, pt) in &pauli_map {
                    match pt {
                        PauliType::Y => super::tableau_compose::right_compose_szdg(tableau, site),
                        PauliType::Z => super::tableau_compose::right_compose_h(tableau, site),
                        PauliType::X => {}
                    }
                }
                for &(other_site, _) in &pauli_map {
                    if other_site == rot_site {
                        continue;
                    }
                    super::tableau_compose::right_compose_cx(tableau, rot_site, other_site);
                }
                for &(site, pt) in &pauli_map {
                    if site == rot_site {
                        continue;
                    }
                    match pt {
                        PauliType::Y => super::tableau_compose::right_compose_sz(tableau, site),
                        PauliType::Z => super::tableau_compose::right_compose_h(tableau, site),
                        PauliType::X => {}
                    }
                }

                // Clear the flag (rot_site's MPS is no longer |0⟩).
                disent_flags[rot_site] = None;
            } else if affected_sites.len() == 1 {
                stats.single_site += 1;
                if is_ofd_in_span {
                    stats.ofd_in_span_single += 1;
                }
                // Record single-site flip pattern
                gf2_matrix.add_row_with_meta(
                    &affected_sites,
                    super::ofd::RowMetadata {
                        rot_site: affected_sites[0],
                    },
                );
                let site = affected_sites[0];
                let pt = pauli_map[0].1;
                let c = Complex64::new(cos_half, 0.0);
                let s = Complex64::new(0.0, -sin_half) * phase; // = -i*sin*phase

                let gate = match pt {
                    PauliType::X => {
                        // X = [[0,1],[1,0]]
                        DMatrix::from_row_slice(2, 2, &[c, s, s, c])
                    }
                    PauliType::Z => {
                        // Z = [[1,0],[0,-1]]
                        DMatrix::from_row_slice(
                            2,
                            2,
                            &[
                                c + s,
                                Complex64::new(0.0, 0.0),
                                Complex64::new(0.0, 0.0),
                                c - s,
                            ],
                        )
                    }
                    PauliType::Y => {
                        // co = -i·sin·phase · (-1) = i·sin·phase. Ys=1, factor=i^2=-1.
                        let mut co = Complex64::new(0.0, -sin_half) * phase;
                        co *= Complex64::new(-1.0, 0.0);
                        let rx_sign: f64 = if sin_half.abs() < 1e-12
                            || (co.re - sin_half).abs() < (co.re + sin_half).abs()
                        {
                            1.0
                        } else {
                            -1.0
                        };
                        let theta = 2.0 * sin_half.atan2(cos_half);
                        let rx_angle = rx_sign * theta;

                        &sdg_gate() * &(&rx_gate(rx_angle) * &s_gate())
                    }
                };
                mps.apply_one_site_gate(site, &gate)
                    .expect("MPS op on valid site");
                // Clear flag for the affected site
                disent_flags[site] = None;
            } else if sign_sites.is_empty() {
                stats.multi_std += 1;
                if is_ofd_in_span {
                    stats.ofd_in_span_std += 1;
                }
                // Note: std path creates MPS entanglement (not absorbed into
                // tableau). Do NOT add to gf2 basis — OFD's is_in_span should
                // only match against truly-absorbed rows.
                // All flip sites, no Z overlap: operator is cos*I + s*prod(X_j).
                // Use MPS addition (exact, no SWAP-chain SVD drift).
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
                let mut mps_x = mps.clone();
                for &j in flip_sites {
                    mps_x
                        .apply_one_site_gate(j, &x_gate)
                        .expect("MPS op on valid site");
                }
                let s = Complex64::new(0.0, -sin_half) * phase;
                mps_x.scale(s);
                mps.scale(Complex64::new(cos_half, 0.0));
                *mps = mps.add(&mps_x);
                mps.compress();
                for &j in flip_sites {
                    disent_flags[j] = None;
                }
            } else {
                stats.multi_std += 1;
                if is_ofd_in_span {
                    stats.ofd_in_span_std += 1;
                }
                // Std path creates MPS entanglement; do NOT add to gf2 basis.
                // Multi-site rotation via CNOT cascade + RX + basis changes.

                // Count Y sites for the coefficient extraction
                let y_count = pauli_map.iter().filter(|(_, p)| *p == PauliType::Y).count();

                // co = -i*sin*phase · i^(Ys+1). After correction co = ±sin(θ/2).
                let mut co = Complex64::new(0.0, -sin_half) * phase;
                let i_val = Complex64::new(0.0, 1.0);
                let mut factor = Complex64::new(1.0, 0.0);
                for _ in 0..=y_count {
                    factor *= i_val;
                }
                co *= factor;

                let rx_sign: f64 = if sin_half.abs() < 1e-12
                    || (co.re - sin_half).abs() < (co.re + sin_half).abs()
                {
                    1.0
                } else {
                    -1.0
                };
                let theta = 2.0 * sin_half.atan2(cos_half);
                let rx_angle = rx_sign * theta;

                // Choose rotation site as the median of affected sites.
                // This minimizes the total SWAP distance for the CNOT cascade.
                let rot_idx = affected_sites.len() / 2;
                let rot_site = affected_sites[rot_idx];

                // Apply basis changes: H for Z, S for Y
                for &(site, pt) in &pauli_map {
                    match pt {
                        PauliType::Z => {
                            mps.apply_one_site_gate(site, &h_gate())
                                .expect("MPS op on valid site");
                        }
                        PauliType::Y => {
                            mps.apply_one_site_gate(site, &s_gate())
                                .expect("MPS op on valid site");
                        }
                        PauliType::X => {}
                    }
                }

                // CNOT chain cascade (matches reference: stabilizer-TN apply_xvec_rot).
                // Chain through consecutive affected sites toward rot_site,
                // accumulating parity. Each CNOT has the current site as control
                // and the previous site as target (parity accumulator).
                // This produces shorter-range CNOTs than the star pattern.
                let cnot_lo = cnot_lo_ctrl();
                let cnot_hi = cnot_hi_ctrl();

                // Helper: apply CNOT with `ctrl` as control and `tgt` as target.
                let apply_cnot = |mps: &mut Mps, ctrl: usize, tgt: usize| {
                    let (lo, hi) = (ctrl.min(tgt), ctrl.max(tgt));
                    // cnot_lo = lower qubit controls; cnot_hi = higher qubit controls
                    let gate = if ctrl < tgt { &cnot_lo } else { &cnot_hi };
                    mps.apply_long_range_two_site_gate(lo, hi, gate)
                        .expect("CNOT should succeed");
                };

                // Left chain: [0] <- [1] <- ... <- [rot_idx]
                // Each step: control=current, target=previous
                let mut prev = affected_sites[0];
                for &site in &affected_sites[1..=rot_idx] {
                    apply_cnot(mps, site, prev);
                    prev = site;
                }

                // Right chain: [last] <- [last-1] <- ... <- [rot_idx]
                if rot_idx + 1 < affected_sites.len() {
                    prev = *affected_sites
                        .last()
                        .expect("affected_sites must be non-empty");
                    for &site in affected_sites[rot_idx..affected_sites.len() - 1]
                        .iter()
                        .rev()
                    {
                        apply_cnot(mps, site, prev);
                        prev = site;
                    }
                }

                // Apply RX on rotation site (parity accumulated here)
                mps.apply_one_site_gate(rot_site, &rx_gate(rx_angle))
                    .expect("MPS op on valid site");

                // Reverse CNOT cascade (undo in opposite order)
                // Right chain reverse: [rot_idx] -> [rot_idx+1] -> ... -> [last]
                if rot_idx + 1 < affected_sites.len() {
                    prev = affected_sites[rot_idx];
                    for &site in &affected_sites[rot_idx + 1..] {
                        apply_cnot(mps, prev, site);
                        prev = site;
                    }
                }

                // Left chain reverse: [rot_idx] -> [rot_idx-1] -> ... -> [0]
                prev = affected_sites[rot_idx];
                for &site in affected_sites[..rot_idx].iter().rev() {
                    apply_cnot(mps, prev, site);
                    prev = site;
                }

                // Undo basis changes
                for &(site, pt) in &pauli_map {
                    match pt {
                        PauliType::Z => {
                            mps.apply_one_site_gate(site, &h_gate())
                                .expect("MPS op on valid site");
                        }
                        PauliType::Y => {
                            mps.apply_one_site_gate(site, &sdg_gate())
                                .expect("MPS op on valid site");
                        }
                        PauliType::X => {}
                    }
                }
                // MPS modified at all affected sites -- clear flags.
                for &site in &affected_sites {
                    disent_flags[site] = None;
                }
            }
        }
    }

    // Flags are cleared in each branch above, tracking which sites had MPS
    // modifications. See branch-specific `disent_flags[...] = None` calls.

    if normalize {
        mps.normalize();
    }
}
