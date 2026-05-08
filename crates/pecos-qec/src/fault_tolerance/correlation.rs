// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! Detector correlation analysis for DEM validation.
//!
//! Computes k-body detector firing rates from sampled syndromes and
//! compares them between simulation and DEM outputs. This captures
//! both marginal detection rates and the correlated error structure
//! that determines decoding quality.
//!
//! # Flip frequency matrix
//!
//! The pairwise flip frequency matrix `M` for `n` detectors:
//! - `M[i][i]` = P(detector i fires)   (marginal rate)
//! - `M[i][j]` = 0.5 * P(i AND j fire) (half joint rate, i != j)
//!
//! # Higher-order correlations
//!
//! K-body rates map each k-subset of detectors to its joint firing
//! probability. For order 1 these are marginals; for order 2, pairwise;
//! for order 3, triple correlations that test whether the DEM's
//! independent error decomposition is adequate.

use std::collections::{BTreeMap, BTreeSet};

type CorrelationEntry<'a> = (&'a Vec<u32>, f64, f64);

fn count_as_f64<T>(items: &[T]) -> f64 {
    items.iter().fold(0.0, |count, _| count + 1.0)
}

/// Flat `n x n` detector flip frequency matrix.
///
/// Stored row-major. Use `index(i, j, n) = i * n + j`.
#[must_use]
pub fn flip_matrix_from_fired(fired_per_shot: &[Vec<u32>], num_detectors: usize) -> Vec<f64> {
    let n = num_detectors;
    let shots = fired_per_shot.len();
    if shots == 0 {
        return vec![0.0; n * n];
    }

    let inv = 1.0 / count_as_f64(fired_per_shot);
    let half_inv = 0.5 * inv;
    let mut m = vec![0.0; n * n];

    for fired in fired_per_shot {
        for (ai, &a) in fired.iter().enumerate() {
            let a = a as usize;
            if a >= n {
                continue;
            }
            m[a * n + a] += inv;
            for &b in &fired[ai + 1..] {
                let b = b as usize;
                if b >= n {
                    continue;
                }
                m[a * n + b] += half_inv;
                m[b * n + a] += half_inv;
            }
        }
    }

    m
}

/// Per-round flip frequency matrices.
///
/// Returns one flat `k x k` matrix per round, where `k = dets_per_round`.
#[must_use]
pub fn flip_matrices_by_round(
    fired_per_shot: &[Vec<u32>],
    num_detectors: usize,
    dets_per_round: usize,
) -> Vec<Vec<f64>> {
    let k = dets_per_round;
    let num_rounds = num_detectors.div_ceil(k);
    let shots = fired_per_shot.len();
    if shots == 0 {
        return vec![vec![0.0; k * k]; num_rounds];
    }

    let inv = 1.0 / count_as_f64(fired_per_shot);
    let half_inv = 0.5 * inv;
    let mut matrices = vec![vec![0.0; k * k]; num_rounds];

    for fired in fired_per_shot {
        // Bin by round
        let mut round_local: Vec<Vec<u32>> = vec![Vec::new(); num_rounds];
        for &d in fired {
            let r = d as usize / k;
            let local = d as usize % k;
            if r >= num_rounds {
                continue;
            }
            if let Ok(local) = u32::try_from(local) {
                round_local[r].push(local);
            }
        }

        for (r, local_ids) in round_local.iter().enumerate() {
            let mat = &mut matrices[r];
            for (ai, &a) in local_ids.iter().enumerate() {
                let a = a as usize;
                mat[a * k + a] += inv;
                for &b in &local_ids[ai + 1..] {
                    let b = b as usize;
                    mat[a * k + b] += half_inv;
                    mat[b * k + a] += half_inv;
                }
            }
        }
    }

    matrices
}

/// K-body detector firing rates up to `max_order`.
///
/// Returns a map from sorted detector index tuples to joint firing
/// probability. Keys are ordered ascending.
#[must_use]
pub fn k_body_rates(
    fired_per_shot: &[Vec<u32>],
    num_detectors: usize,
    max_order: usize,
) -> BTreeMap<Vec<u32>, f64> {
    let shots = fired_per_shot.len();
    if shots == 0 {
        return BTreeMap::new();
    }

    let inv = 1.0 / count_as_f64(fired_per_shot);
    let mut rates: BTreeMap<Vec<u32>, f64> = BTreeMap::new();

    for fired in fired_per_shot {
        let valid_fired: Vec<u32> = fired
            .iter()
            .copied()
            .filter(|&d| (d as usize) < num_detectors)
            .collect();
        let n = valid_fired.len().min(max_order);
        for order in 1..=n {
            for_each_combination(&valid_fired, order, |combo| {
                *rates.entry(combo.to_vec()).or_insert(0.0) += inv;
            });
        }
    }

    rates
}

/// Per-round k-body rates. Detector indices in the returned maps are
/// round-local (0..dets_per_round-1).
#[must_use]
pub fn k_body_rates_by_round(
    fired_per_shot: &[Vec<u32>],
    num_detectors: usize,
    dets_per_round: usize,
    max_order: usize,
) -> Vec<BTreeMap<Vec<u32>, f64>> {
    let k = dets_per_round;
    let num_rounds = num_detectors.div_ceil(k);
    let shots = fired_per_shot.len();
    if shots == 0 {
        return vec![BTreeMap::new(); num_rounds];
    }

    let inv = 1.0 / count_as_f64(fired_per_shot);
    let mut round_rates: Vec<BTreeMap<Vec<u32>, f64>> = vec![BTreeMap::new(); num_rounds];

    for fired in fired_per_shot {
        let mut round_local: Vec<Vec<u32>> = vec![Vec::new(); num_rounds];
        for &d in fired {
            let r = d as usize / k;
            let local = d as usize % k;
            if r >= num_rounds {
                continue;
            }
            if let Ok(local) = u32::try_from(local) {
                round_local[r].push(local);
            }
        }

        for (r, local_ids) in round_local.iter().enumerate() {
            let n = local_ids.len().min(max_order);
            let rr = &mut round_rates[r];
            for order in 1..=n {
                for_each_combination(local_ids, order, |combo| {
                    *rr.entry(combo.to_vec()).or_insert(0.0) += inv;
                });
            }
        }
    }

    round_rates
}

/// Compare k-body rates between two sets, grouped by order.
///
/// Returns a map from order to `(max_rel_error, rms_rel_error, worst_event)`.
#[must_use]
pub fn compare_k_body(
    sim: &BTreeMap<Vec<u32>, f64>,
    dem: &BTreeMap<Vec<u32>, f64>,
    min_rate: f64,
) -> BTreeMap<usize, (f64, f64, Vec<u32>)> {
    let all_keys: BTreeSet<&Vec<u32>> = sim.keys().chain(dem.keys()).collect();

    let mut by_order: BTreeMap<usize, Vec<CorrelationEntry<'_>>> = BTreeMap::new();
    for &key in &all_keys {
        let s = sim.get(key).copied().unwrap_or(0.0);
        let d = dem.get(key).copied().unwrap_or(0.0);
        by_order.entry(key.len()).or_default().push((key, s, d));
    }

    let mut result = BTreeMap::new();
    for (order, entries) in &by_order {
        let mut max_err = 0.0_f64;
        let mut worst: Vec<u32> = Vec::new();
        let mut sum_sq = 0.0;
        let mut count = 0.0;

        for &(key, s, d) in entries {
            if s > min_rate {
                let rel = (d / s - 1.0).abs();
                if rel > max_err {
                    max_err = rel;
                    worst.clone_from(key);
                }
                sum_sq += rel * rel;
                count += 1.0;
            }
        }

        let rms = if count > 0.0 {
            (sum_sq / count).sqrt()
        } else {
            0.0
        };
        result.insert(*order, (max_err, rms, worst));
    }

    result
}

/// Compare two flat flip matrices. Returns `(max_rel_err, frob_rel_err, worst_i, worst_j)`.
#[must_use]
pub fn compare_flip_matrices(
    sim: &[f64],
    dem: &[f64],
    n: usize,
    min_rate: f64,
) -> (f64, f64, usize, usize) {
    let mut max_err = 0.0_f64;
    let mut worst_i = 0;
    let mut worst_j = 0;
    let mut sum_sq_diff = 0.0;
    let mut sum_sq_sim = 0.0;

    for i in 0..n {
        for j in 0..n {
            let idx = i * n + j;
            let s = sim[idx];
            let d = dem[idx];
            let diff = d - s;
            sum_sq_diff += diff * diff;
            sum_sq_sim += s * s;
            if s > min_rate {
                let rel = diff.abs() / s;
                if rel > max_err {
                    max_err = rel;
                    worst_i = i;
                    worst_j = j;
                }
            }
        }
    }

    let frob = sum_sq_diff.sqrt() / sum_sq_sim.sqrt().max(1e-30);
    (max_err, frob, worst_i, worst_j)
}

// ---------------------------------------------------------------------------
// Hybrid DEM: fit mechanism probabilities to target marginals
// ---------------------------------------------------------------------------

/// A DEM mechanism: probability + detector/DEM-output sets.
#[derive(Debug, Clone)]
pub struct DemMechanism {
    pub probability: f64,
    pub detectors: Vec<u32>,
    pub observables: Vec<u32>,
}

/// Fit DEM mechanism probabilities to match target detector marginals.
///
/// Given a set of mechanisms (each with a detector set and initial
/// probability) and target per-detector marginal rates, adjusts the
/// mechanism probabilities so the DEM's independent-error marginals
/// match the targets as closely as possible.
///
/// Uses iterative proportional fitting on the exact DEM marginal equation:
///
/// ```text
/// p_d = 1/2 - 1/2 * prod_{m: d in S_m} (1 - 2*q_m)
/// ```
///
/// Each iteration computes current marginals, then scales each mechanism
/// by the geometric mean of (target/current) ratios for the detectors
/// it affects. Mechanisms with no detector overlap are untouched.
///
/// Returns the fitted mechanisms and per-detector residual errors.
#[must_use]
pub fn fit_dem_to_marginals(
    mechanisms: &[DemMechanism],
    target_marginals: &[f64],
    max_iterations: usize,
    tolerance: f64,
) -> (Vec<DemMechanism>, Vec<f64>) {
    let num_dets = target_marginals.len();
    let n_mech = mechanisms.len();

    // Build sparse incidence: for each detector, which mechanisms touch it
    let mut det_to_mechs: Vec<Vec<usize>> = vec![Vec::new(); num_dets];
    for (m, mech) in mechanisms.iter().enumerate() {
        for &d in &mech.detectors {
            if (d as usize) < num_dets {
                det_to_mechs[d as usize].push(m);
            }
        }
    }

    let mut q: Vec<f64> = mechanisms.iter().map(|m| m.probability).collect();

    for _iter in 0..max_iterations {
        // Compute current marginals from mechanism probabilities
        let mut current = vec![0.0_f64; num_dets];
        for d in 0..num_dets {
            let mut prod = 1.0;
            for &m in &det_to_mechs[d] {
                prod *= 1.0 - 2.0 * q[m];
            }
            current[d] = (1.0 - prod) / 2.0;
        }

        // Compute per-detector ratios
        let mut ratios = vec![1.0_f64; num_dets];
        for d in 0..num_dets {
            if current[d] > 1e-20 {
                ratios[d] = target_marginals[d] / current[d];
            } else if target_marginals[d] > 1e-20 {
                ratios[d] = 10.0; // large but bounded nudge
            }
        }

        // Scale each mechanism by geometric mean of its detector ratios
        let mut max_change = 0.0_f64;
        for m in 0..n_mech {
            let dets = &mechanisms[m].detectors;
            if dets.is_empty() {
                continue;
            }
            let mut log_ratio = 0.0;
            let mut count = 0;
            for &d in dets {
                if (d as usize) < num_dets {
                    log_ratio += ratios[d as usize].max(1e-10).ln();
                    count += 1;
                }
            }
            if count == 0 {
                continue;
            }
            let scale = (log_ratio / f64::from(count)).exp();
            let new_q = (q[m] * scale).clamp(0.0, 0.499);
            max_change = max_change.max((new_q - q[m]).abs());
            q[m] = new_q;
        }

        if max_change < tolerance {
            break;
        }
    }

    // Compute final residuals
    let mut residuals = vec![0.0; num_dets];
    for d in 0..num_dets {
        let mut prod = 1.0;
        for &m in &det_to_mechs[d] {
            prod *= 1.0 - 2.0 * q[m];
        }
        let fitted = (1.0 - prod) / 2.0;
        residuals[d] = (fitted - target_marginals[d]).abs();
    }

    let fitted: Vec<DemMechanism> = mechanisms
        .iter()
        .zip(q.iter())
        .map(|(mech, &prob)| DemMechanism {
            probability: prob,
            detectors: mech.detectors.clone(),
            observables: mech.observables.clone(),
        })
        .collect();

    (fitted, residuals)
}

/// Format fitted mechanisms as a standard DEM string.
#[must_use]
pub fn mechanisms_to_dem_string(mechanisms: &[DemMechanism]) -> String {
    let mut lines = Vec::new();
    for mech in mechanisms {
        if mech.probability > 1e-15 {
            let mut tokens = Vec::new();
            for &d in &mech.detectors {
                tokens.push(format!("D{d}"));
            }
            for &o in &mech.observables {
                tokens.push(format!("L{o}"));
            }
            if !tokens.is_empty() {
                lines.push(format!(
                    "error({:.10e}) {}",
                    mech.probability,
                    tokens.join(" ")
                ));
            }
        }
    }
    lines.join("\n")
}

// --- Internal helpers ---

/// Iterate over all k-combinations of `items`, calling `f` with each sorted combination.
fn for_each_combination(items: &[u32], k: usize, mut f: impl FnMut(&[u32])) {
    if k == 0 || items.len() < k {
        return;
    }
    let mut combo = vec![0u32; k];
    combination_recurse(items, k, 0, 0, &mut combo, &mut f);
}

fn combination_recurse(
    items: &[u32],
    k: usize,
    start: usize,
    depth: usize,
    combo: &mut [u32],
    f: &mut impl FnMut(&[u32]),
) {
    if depth == k {
        f(&combo[..k]);
        return;
    }
    let remaining = k - depth;
    if start + remaining > items.len() {
        return;
    }
    for i in start..=items.len() - remaining {
        combo[depth] = items[i];
        combination_recurse(items, k, i + 1, depth + 1, combo, f);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flip_matrix_single_detector() {
        // 100 shots, detector 0 fires 30 times
        let fired: Vec<Vec<u32>> = (0..100)
            .map(|i| if i < 30 { vec![0] } else { vec![] })
            .collect();
        let m = flip_matrix_from_fired(&fired, 2);
        assert!((m[0] - 0.3).abs() < 1e-10); // M[0,0] = 0.3
        assert!(m[1].abs() < 1e-10); // M[0,1] = 0
        assert!(m[3].abs() < 1e-10); // M[1,1] = 0
    }

    #[test]
    fn test_flip_matrix_correlated_pair() {
        // 100 shots, detectors 0 and 1 always fire together
        let fired: Vec<Vec<u32>> = (0..100)
            .map(|i| if i < 20 { vec![0, 1] } else { vec![] })
            .collect();
        let m = flip_matrix_from_fired(&fired, 2);
        assert!((m[0] - 0.2).abs() < 1e-10); // M[0,0]
        assert!((m[3] - 0.2).abs() < 1e-10); // M[1,1]
        assert!((m[1] - 0.1).abs() < 1e-10); // M[0,1] = 0.5 * 0.2
        assert!((m[2] - 0.1).abs() < 1e-10); // M[1,0] = 0.5 * 0.2
    }

    #[test]
    fn test_k_body_rates_basic() {
        let fired = vec![vec![0, 1, 2], vec![0, 1], vec![0], vec![]];
        let rates = k_body_rates(&fired, 3, 3);
        assert!((rates[&vec![0]] - 0.75).abs() < 1e-10);
        assert!((rates[&vec![1]] - 0.5).abs() < 1e-10);
        assert!((rates[&vec![0, 1]] - 0.5).abs() < 1e-10);
        assert!((rates[&vec![0, 1, 2]] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_compare_k_body_basic() {
        let mut sim = BTreeMap::new();
        sim.insert(vec![0], 0.1);
        sim.insert(vec![1], 0.2);
        sim.insert(vec![0, 1], 0.01);

        let mut dem = BTreeMap::new();
        dem.insert(vec![0], 0.1);
        dem.insert(vec![1], 0.2);
        dem.insert(vec![0, 1], 0.012);

        let result = compare_k_body(&sim, &dem, 0.005);
        // 1-body: exact match
        assert!(result[&1].0 < 1e-10);
        // 2-body: 20% relative error on (0,1)
        assert!((result[&2].0 - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_by_round_splits_correctly() {
        // 4 detectors, 2 per round -> 2 rounds
        let fired = vec![vec![0, 2], vec![1, 3]];
        let mats = flip_matrices_by_round(&fired, 4, 2);
        assert_eq!(mats.len(), 2);
        // Round 0: det 0 in shot 0, det 1 in shot 1
        assert!((mats[0][0] - 0.5).abs() < 1e-10); // M[0,0]
        assert!((mats[0][3] - 0.5).abs() < 1e-10); // M[1,1]
        // Round 1: det 0(=global 2) in shot 0, det 1(=global 3) in shot 1
        assert!((mats[1][0] - 0.5).abs() < 1e-10);
        assert!((mats[1][3] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_fit_dem_to_marginals_exact() {
        // Two mechanisms: M0 flips {D0}, M1 flips {D0, D1}
        // Target: P(D0) = 0.15, P(D1) = 0.05
        let mechs = vec![
            DemMechanism {
                probability: 0.1,
                detectors: vec![0],
                observables: vec![],
            },
            DemMechanism {
                probability: 0.05,
                detectors: vec![0, 1],
                observables: vec![],
            },
        ];
        let target = vec![0.15, 0.05];
        let (fitted, residuals) = fit_dem_to_marginals(&mechs, &target, 100, 1e-12);

        // M1 flips only D1, so q1 must satisfy (1-2*q1)/2 ≈ 0.05 → q1 ≈ 0.05
        // Then q0 must satisfy 1/2(1-(1-2*q0)(1-2*q1)) = 0.15
        assert!(residuals[0] < 1e-6, "D0 residual: {}", residuals[0]);
        assert!(residuals[1] < 1e-6, "D1 residual: {}", residuals[1]);
        assert!(fitted[1].probability > 0.04 && fitted[1].probability < 0.06);
    }

    #[test]
    fn test_fit_dem_preserves_structure() {
        let mechs = vec![
            DemMechanism {
                probability: 0.01,
                detectors: vec![0],
                observables: vec![0],
            },
            DemMechanism {
                probability: 0.02,
                detectors: vec![1],
                observables: vec![],
            },
        ];
        let target = vec![0.05, 0.08];
        let (fitted, _) = fit_dem_to_marginals(&mechs, &target, 100, 1e-12);

        // Structure preserved
        assert_eq!(fitted[0].detectors, vec![0]);
        assert_eq!(fitted[0].observables, vec![0]);
        assert_eq!(fitted[1].detectors, vec![1]);
    }

    #[test]
    fn test_mechanisms_to_dem_string() {
        let mechs = vec![
            DemMechanism {
                probability: 0.01,
                detectors: vec![0, 1],
                observables: vec![],
            },
            DemMechanism {
                probability: 0.001,
                detectors: vec![2],
                observables: vec![0],
            },
        ];
        let s = mechanisms_to_dem_string(&mechs);
        assert!(s.contains("D0 D1"));
        assert!(s.contains("D2 L0"));
    }
}
