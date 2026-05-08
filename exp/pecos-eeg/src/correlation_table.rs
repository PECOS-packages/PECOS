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

//! Exact detector correlation tables from backward Heisenberg walks.
//!
//! Computes exact k-body joint detection rates using product stabilizer
//! walks. No DEM approximation — captures all coherent interference.
//!
//! For k detectors with stabilizers S1..Sk, the joint detection probability
//! is computed via inclusion-exclusion:
//!
//!   P(D1 AND D2 AND ... AND Dk) = 1/2^k * sum_{T ⊆ {1..k}} (-1)^{k-|T|} <prod_{i in T} Si>
//!
//! where <prod Si> is computed by a Heisenberg walk with the product stabilizer.
//!
//! Each walk gives one expectation value. The number of walks needed:
//! - Order 1 (marginals): C(n,1) = n
//! - Order 2 (pairwise): C(n,2) new walks
//! - Order 3 (triples): C(n,3) new walks
//! - Total up to order k: sum_{j=1}^{k} C(n,j)

use crate::Bm;
use crate::dem_mapping::{Detector, Observable};
use crate::noise::NoiseSpec;
use crate::stabilizer::StabilizerGroup;
use pecos_core::Gate;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::collections::BTreeMap;
use std::fmt::Write as _;

/// Exact k-body correlation table for detectors and observables.
///
/// Contains two types of correlations:
/// - **Detector rates**: P(D_i1, ..., D_ik) — joint detection probabilities
/// - **Detector-observable rates**: P(D_i1, ..., D_ik, L_j) — how detection
///   patterns relate to logical observable flips (what decoders need)
pub struct CorrelationTable {
    /// Detector-only joint rates: sorted det_indices -> probability
    pub rates: BTreeMap<Vec<usize>, f64>,
    /// Detector-observable joint rates: (sorted det_indices, obs_id) -> probability.
    /// P(D_i1 AND ... AND D_ik AND L_j) for each observable j.
    pub observable_rates: BTreeMap<(Vec<usize>, usize), f64>,
    /// Maximum correlation order computed (for detectors)
    pub max_order: usize,
    /// Number of detectors
    pub num_detectors: usize,
    /// Number of observables
    pub num_observables: usize,
    /// Number of Heisenberg walks performed
    pub num_walks: usize,
}

/// Inputs for exact correlation table construction.
#[derive(Clone, Copy)]
pub struct CorrelationTableInput<'a> {
    /// Circuit gates.
    pub gates: &'a [Gate],
    /// Noise model used for exact correlation targets.
    pub noise: &'a dyn NoiseSpec,
    /// Detector definitions.
    pub detectors: &'a [Detector],
    /// Observable definitions.
    pub observables: &'a [Observable],
    /// Initial stabilizer group.
    pub initial_stab: &'a StabilizerGroup,
    /// Number of circuit qubits.
    pub num_qubits: usize,
    /// Maximum detector/observable correlation order.
    pub max_order: usize,
    /// Drop probabilities below this threshold.
    pub prune_threshold: f64,
}

impl CorrelationTable {
    /// Build a graphlike DEM string from the correlation table.
    ///
    /// Uses pairwise correlations as edge probabilities for MWPM decoders.
    /// Each pairwise correlation P(Di AND Dj) - P(Di)*P(Dj) becomes an edge.
    /// Observable assignment uses P(Di AND Lk) rates.
    ///
    /// This bypasses the DEM independent error model — edge weights come
    /// directly from exact Heisenberg correlations including all coherent
    /// interference effects.
    #[must_use]
    pub fn to_matching_dem(&self) -> String {
        let mut lines = Vec::new();

        // Get marginals
        let marginals: BTreeMap<usize, f64> = self
            .rates
            .iter()
            .filter(|(k, _)| k.len() == 1)
            .map(|(k, &v)| (k[0], v))
            .collect();

        // Observable marginals per detector: P(Di AND Lk)
        let mut det_obs: BTreeMap<(usize, usize), f64> = BTreeMap::new();
        for ((det_ids, obs_id), &prob) in &self.observable_rates {
            if det_ids.len() == 1 {
                det_obs.insert((det_ids[0], *obs_id), prob);
            }
        }

        // Pairwise edges: excess correlation = P(Di,Dj) - P(Di)*P(Dj)
        for (key, &joint_prob) in &self.rates {
            if key.len() != 2 {
                continue;
            }
            let (di, dj) = (key[0], key[1]);

            let pi = marginals.get(&di).copied().unwrap_or(0.0);
            let pj = marginals.get(&dj).copied().unwrap_or(0.0);
            let p_excess = joint_prob - pi * pj;

            if p_excess <= 1e-15 {
                continue;
            } // no positive correlation
            let p_edge = p_excess.min(0.499); // clamp for valid weight

            // Determine observable assignment: which Lk is most correlated
            // with this pair? Use P(Di AND Dj AND Lk) if available,
            // otherwise no observable.
            let mut obs_list = Vec::new();
            for obs_id in 0..self.num_observables {
                let pair_key = (vec![di, dj], obs_id);
                if let Some(&p_trio) = self.observable_rates.get(&pair_key) {
                    // If the trio rate is significant relative to the pair rate,
                    // this observable is correlated with this edge
                    if p_trio > joint_prob * 0.1 {
                        obs_list.push(obs_id);
                    }
                }
            }

            let mut targets = format!("D{di} D{dj}");
            for o in &obs_list {
                let _ = write!(targets, " L{o}");
            }
            lines.push(format!("error({p_edge:.6e}) {targets}"));
        }

        // Boundary edges: P(Di AND Lk) - P(Di)*P(Lk)
        // Approximation: use P(Di AND Lk) directly as boundary edge probability
        // (represents probability Di fires due to a logical error chain)
        for obs_id in 0..self.num_observables {
            for di in 0..self.num_detectors {
                let p_det_obs = det_obs.get(&(di, obs_id)).copied().unwrap_or(0.0);

                // Check if this detector has significant correlation with the observable
                // that isn't already explained by pairwise edges
                if p_det_obs <= 1e-15 {
                    continue;
                }

                // Boundary probability: P(Di fires AND it's a logical error)
                let p_boundary = p_det_obs.min(0.499);
                if p_boundary <= 1e-15 {
                    continue;
                }

                lines.push(format!("error({p_boundary:.6e}) D{di} L{obs_id}"));
            }
        }

        lines.join("\n")
    }
}

/// Compute exact correlation table up to `max_order` using Heisenberg walks.
///
/// Each entry gives the exact joint detection probability for a subset of
/// detectors, including all coherent interference effects.
#[must_use]
pub fn compute_correlation_table(input: CorrelationTableInput<'_>) -> CorrelationTable {
    let CorrelationTableInput {
        gates,
        noise,
        detectors,
        observables,
        initial_stab,
        num_qubits,
        max_order,
        prune_threshold,
    } = input;

    let n = detectors.len();
    let n_obs = observables.len();
    let has_stochastic = true; // conservative; could check noise params

    // Build noise map once, shared across all walks
    let gate_index = crate::expand::GateIndex::build(gates, num_qubits);
    let noise_map = if has_stochastic {
        Some(crate::heisenberg::build_noise_map(
            gates,
            noise,
            &gate_index.expansion_gates,
        ))
    } else {
        None
    };

    // Helper: run one Heisenberg walk with a given stabilizer.
    // Uses sparse traversal (heap + gate index) with optional noise map.
    let walk = |stab: &Bm| -> f64 {
        crate::heisenberg::heisenberg_sparse(
            gates,
            stab,
            noise,
            initial_stab,
            prune_threshold,
            &gate_index,
            noise_map.as_deref(),
        )
    };

    let mut rates = BTreeMap::new();

    // Cache walk results for product stabilizers: key = sorted det indices
    let mut walk_cache: BTreeMap<Vec<usize>, f64> = BTreeMap::new();

    // Collect all (indices, product_stabilizer) pairs for parallel walks
    let mut walk_items: Vec<(Vec<usize>, Bm)> = Vec::new();
    for order in 1..=max_order.min(n) {
        for_each_combination_idx(n, order, |indices| {
            let mut product = detectors[indices[0]].stabilizer.clone();
            for &idx in &indices[1..] {
                product = product.multiply(&detectors[idx].stabilizer);
            }
            let det_ids: Vec<usize> = indices.iter().map(|&i| detectors[i].id).collect();
            walk_items.push((det_ids, product));
        });
    }

    // Run all walks in parallel
    let walk_results: Vec<(Vec<usize>, f64)> = walk_items
        .par_iter()
        .map(|(det_ids, product)| {
            let p_walk = walk(product);
            (det_ids.clone(), p_walk)
        })
        .collect();

    let num_walks = walk_results.len();
    for (det_ids, p_walk) in walk_results {
        walk_cache.insert(det_ids, p_walk);
    }

    // Convert walk results to joint detection probabilities via inclusion-exclusion.
    // P(D_{i1}, ..., D_{ik}) = 1/2^k * sum_{T ⊆ {i1..ik}} (-1)^{k-|T|} * <prod_{j in T} S_j>
    // where <prod S> = 1 - 2 * walk_result for that product.
    for order in 1..=max_order.min(n) {
        for_each_combination_idx(n, order, |indices| {
            let det_ids: Vec<usize> = indices.iter().map(|&i| detectors[i].id).collect();
            let k = order;

            let mut prob = 0.0_f64;
            let inv_2k = 1.0 / (1u64 << k) as f64;

            // Iterate over all subsets T of {0..k-1}
            for mask in 0..(1u64 << k) {
                let subset_size = mask.count_ones() as usize;
                let sign = if subset_size.is_multiple_of(2) {
                    1.0
                } else {
                    -1.0
                };

                if subset_size == 0 {
                    // Empty subset: <I> = 1, contribution = (-1)^k * 1
                    prob += sign;
                } else {
                    // Build the subset's detector IDs
                    let subset_det_ids: Vec<usize> = (0..k)
                        .filter(|&bit| mask & (1u64 << bit) != 0)
                        .map(|bit| detectors[indices[bit]].id)
                        .collect();

                    // Look up the walk result for this subset
                    if let Some(&p_walk) = walk_cache.get(&subset_det_ids) {
                        // <prod S> = 1 - 2 * p_walk
                        let expectation = 1.0 - 2.0 * p_walk;
                        prob += sign * expectation;
                    }
                }
            }

            prob *= inv_2k;
            if prob.abs() > 1e-15 {
                rates.insert(det_ids, prob.max(0.0));
            }
        });
    }

    // Compute detector-observable cross-correlations.
    // For each observable L_j and each detector subset {D_i1,...,D_ik}:
    //   P(D_i1,...,D_ik, L_j) via inclusion-exclusion with the observable
    //   stabilizer included in the product.
    //
    // For single detector + observable:
    //   P(Di, Lj) = 1/4 (1 - <Si> - <Lj> + <Si*Lj>)
    //
    // We compute P(Lj) and P(Di, Lj) for each detector-observable pair.
    let mut observable_rates: BTreeMap<(Vec<usize>, usize), f64> = BTreeMap::new();

    // Collect observable walk items for parallel execution
    // Items: (obs_id, Option<det_id>, product_stabilizer)
    let mut obs_walk_items: Vec<(usize, Option<usize>, Bm)> = Vec::new();
    for obs in observables {
        obs_walk_items.push((obs.id, None, obs.pauli.clone()));
        for det in detectors {
            let product = det.stabilizer.multiply(&obs.pauli);
            obs_walk_items.push((obs.id, Some(det.id), product));
        }
    }

    let obs_walk_results: Vec<(usize, Option<usize>, f64)> = obs_walk_items
        .par_iter()
        .map(|(obs_id, det_id, product)| (*obs_id, *det_id, walk(product)))
        .collect();

    let num_walks = num_walks + obs_walk_results.len();

    // Process observable walk results: marginals first, then pairwise
    let mut obs_marginals: BTreeMap<usize, f64> = BTreeMap::new();
    for &(obs_id, det_id, p_walk) in &obs_walk_results {
        if det_id.is_none() {
            obs_marginals.insert(obs_id, p_walk);
            observable_rates.insert((vec![], obs_id), p_walk);
        }
    }
    for &(obs_id, det_id, p_walk) in &obs_walk_results {
        if let Some(d_id) = det_id {
            let p_di = walk_cache.get(&vec![d_id]).copied().unwrap_or(0.0);
            let p_obs = obs_marginals.get(&obs_id).copied().unwrap_or(0.0);
            let p_joint = (p_di + p_obs - p_walk) / 2.0;
            if p_joint.abs() > 1e-15 {
                observable_rates.insert((vec![d_id], obs_id), p_joint.max(0.0));
            }
        }
    }

    CorrelationTable {
        rates,
        observable_rates,
        max_order: max_order.min(n),
        num_detectors: n,
        num_observables: n_obs,
        num_walks,
    }
}

/// Iterate over all k-combinations of indices 0..n, calling f with each sorted combination.
fn for_each_combination_idx(n: usize, k: usize, mut f: impl FnMut(&[usize])) {
    if k == 0 || n < k {
        return;
    }
    let mut combo = vec![0usize; k];
    combination_recurse_idx(n, k, 0, 0, &mut combo, &mut f);
}

fn combination_recurse_idx(
    n: usize,
    k: usize,
    start: usize,
    depth: usize,
    combo: &mut [usize],
    f: &mut impl FnMut(&[usize]),
) {
    if depth == k {
        f(&combo[..k]);
        return;
    }
    let remaining = k - depth;
    if start + remaining > n {
        return;
    }
    for i in start..=(n - remaining) {
        combo[depth] = i;
        combination_recurse_idx(n, k, i + 1, depth + 1, combo, f);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_combination_idx() {
        let mut results = Vec::new();
        for_each_combination_idx(4, 2, |combo| {
            results.push(combo.to_vec());
        });
        assert_eq!(results.len(), 6); // C(4,2) = 6
        assert_eq!(results[0], vec![0, 1]);
        assert_eq!(results[5], vec![2, 3]);
    }
}
