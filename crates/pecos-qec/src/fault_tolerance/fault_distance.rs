// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the
// License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
// either express or implied. See the License for the specific language governing permissions and
// limitations under the License.

//! Fault-distance searches for detector error models.
//!
//! The general-purpose search uses the published Connected Cluster approach
//! ([arXiv:2603.22532](https://arxiv.org/abs/2603.22532)): a minimum-weight
//! undetectable observable-flipping set is connected in the graph whose nodes are fault
//! mechanisms and whose edges join mechanisms that share a detector. Growing only connected
//! clusters avoids the blind subset enumeration used by the reference implementation.

use super::dem_builder::{DetectorErrorModel, FaultMechanism};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;

/// Result of a fault-distance calculation, including one minimum-size witness.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FaultDistanceResult {
    /// Minimum number of fault mechanisms in an undetectable logical error.
    pub distance: usize,
    /// Witnessing indices into [`DetectorErrorModel::to_mechanisms`], sorted ascending.
    pub mechanism_indices: Vec<usize>,
}

/// Error returned when a requested fault-distance algorithm does not apply.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FaultDistanceError {
    /// The graphlike search was given mechanisms that flip more than two detectors.
    HyperedgesPresent {
        /// Number of hyperedge mechanisms in the detector error model.
        count: usize,
    },
}

impl fmt::Display for FaultDistanceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HyperedgesPresent { count } => write!(
                f,
                "graphlike fault-distance search requires every mechanism to flip at most 2 detectors; found {count} hyperedge mechanism(s)"
            ),
        }
    }
}

impl std::error::Error for FaultDistanceError {}

fn mechanisms_from_dem(dem: &DetectorErrorModel) -> Vec<FaultMechanism> {
    let (mechanisms, _coordinates) = dem.to_mechanisms();
    mechanisms
        .into_iter()
        .map(|(_probability, detectors, observables)| {
            // Fault distance is unit-weight: mechanism probabilities are deliberately ignored.
            FaultMechanism::from_unsorted(detectors, observables)
        })
        .collect()
}

fn detector_incidence(mechanisms: &[FaultMechanism]) -> BTreeMap<u32, Vec<usize>> {
    let mut incidence: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for (mechanism_index, mechanism) in mechanisms.iter().enumerate() {
        for &detector in &mechanism.detectors {
            incidence.entry(detector).or_default().push(mechanism_index);
        }
    }
    incidence
}

/// Removes mechanisms that contain a detector unique among the remaining mechanisms.
///
/// Once such a mechanism is removed, its other detectors may become unique, so the queue runs to
/// a fixpoint. The returned mask uses the original mechanism indices.
fn peel_unique_detector_mechanisms(
    mechanisms: &[FaultMechanism],
    incidence: &BTreeMap<u32, Vec<usize>>,
) -> Vec<bool> {
    let mut active = vec![true; mechanisms.len()];
    let mut remaining_counts: BTreeMap<u32, usize> = incidence
        .iter()
        .map(|(&detector, mechanism_indices)| (detector, mechanism_indices.len()))
        .collect();
    let mut unique_detectors: VecDeque<u32> = remaining_counts
        .iter()
        .filter_map(|(&detector, &count)| (count == 1).then_some(detector))
        .collect();

    while let Some(detector) = unique_detectors.pop_front() {
        if remaining_counts.get(&detector) != Some(&1) {
            continue;
        }
        let Some(mechanism_index) = incidence[&detector]
            .iter()
            .copied()
            .find(|&index| active[index])
        else {
            continue;
        };

        active[mechanism_index] = false;
        for &affected_detector in &mechanisms[mechanism_index].detectors {
            let count = remaining_counts
                .get_mut(&affected_detector)
                .expect("every mechanism detector is present in the incidence map");
            *count -= 1;
            if *count == 1 {
                unique_detectors.push_back(affected_detector);
            }
        }
    }

    active
}

struct ConnectedClusterSearch<'a> {
    mechanisms: &'a [FaultMechanism],
    incidence: &'a BTreeMap<u32, Vec<usize>>,
    active: &'a [bool],
    target_weight: usize,
    best: Option<Vec<usize>>,
}

impl ConnectedClusterSearch<'_> {
    fn run(mut self) -> Option<Vec<usize>> {
        for seed in 0..self.mechanisms.len() {
            if !self.active[seed] {
                continue;
            }

            let mut cluster = vec![seed];
            let mut members = BTreeSet::from([seed]);
            let effect = self.mechanisms[seed].clone();
            if self.target_weight == 1 {
                self.consider_witness(&cluster, &effect);
                continue;
            }

            let extension = self.neighbors(seed, seed, &members, &BTreeSet::new());
            self.extend(
                seed,
                &mut cluster,
                &mut members,
                &effect,
                extension,
                BTreeSet::new(),
            );
        }
        self.best
    }

    fn extend(
        &mut self,
        seed: usize,
        cluster: &mut Vec<usize>,
        members: &mut BTreeSet<usize>,
        effect: &FaultMechanism,
        mut extension: BTreeSet<usize>,
        mut excluded: BTreeSet<usize>,
    ) {
        while let Some(candidate) = extension.pop_first() {
            cluster.push(candidate);
            members.insert(candidate);
            let next_effect = effect.xor(&self.mechanisms[candidate]);

            if cluster.len() == self.target_weight {
                self.consider_witness(cluster, &next_effect);
            } else {
                let mut next_extension = extension.clone();
                next_extension.extend(self.neighbors(candidate, seed, members, &excluded));
                self.extend(
                    seed,
                    cluster,
                    members,
                    &next_effect,
                    next_extension,
                    excluded.clone(),
                );
            }

            members.remove(&candidate);
            cluster.pop();
            excluded.insert(candidate);
        }
    }

    fn neighbors(
        &self,
        mechanism_index: usize,
        seed: usize,
        members: &BTreeSet<usize>,
        excluded: &BTreeSet<usize>,
    ) -> BTreeSet<usize> {
        self.mechanisms[mechanism_index]
            .detectors
            .iter()
            .flat_map(|detector| &self.incidence[detector])
            .copied()
            .filter(|&neighbor| {
                neighbor > seed
                    && self.active[neighbor]
                    && !members.contains(&neighbor)
                    && !excluded.contains(&neighbor)
            })
            .collect()
    }

    fn consider_witness(&mut self, cluster: &[usize], effect: &FaultMechanism) {
        if !effect.detectors.is_empty() || effect.dem_outputs.is_empty() {
            return;
        }
        let mut candidate = cluster.to_vec();
        candidate.sort_unstable();
        if self
            .best
            .as_ref()
            .is_none_or(|current| candidate < *current)
        {
            self.best = Some(candidate);
        }
    }
}

#[derive(Clone, Copy)]
struct GraphEdge {
    neighbor: usize,
    mechanism_index: usize,
}

fn update_best(best: &mut Option<FaultDistanceResult>, mut mechanism_indices: Vec<usize>) {
    mechanism_indices.sort_unstable();
    let candidate = FaultDistanceResult {
        distance: mechanism_indices.len(),
        mechanism_indices,
    };
    if best.as_ref().is_none_or(|current| {
        (candidate.distance, &candidate.mechanism_indices)
            < (current.distance, &current.mechanism_indices)
    }) {
        *best = Some(candidate);
    }
}

/// Computes the exact fault distance of a graphlike detector error model.
///
/// A graphlike mechanism flips at most two detectors. The search returns an error when any
/// hyperedge is present; it never drops unsupported mechanisms or silently changes algorithms.
/// This check is unconditional and happens before any distance-one shortcut, so a DEM containing
/// a hyperedge is rejected predictably even when it also contains a detector-free logical
/// mechanism. Probabilities are ignored because distance counts mechanisms with unit weight.
///
/// A mechanism with no detectors and at least one observable is handled first because it proves
/// distance one without requiring any graph construction.
///
/// # Errors
///
/// Returns [`FaultDistanceError::HyperedgesPresent`] with the number of mechanisms that flip more
/// than two detectors.
pub fn graphlike_fault_distance(
    dem: &DetectorErrorModel,
) -> Result<Option<FaultDistanceResult>, FaultDistanceError> {
    let mechanisms = mechanisms_from_dem(dem);
    let hyperedge_count = mechanisms
        .iter()
        .filter(|mechanism| mechanism.is_hyperedge())
        .count();
    if hyperedge_count != 0 {
        return Err(FaultDistanceError::HyperedgesPresent {
            count: hyperedge_count,
        });
    }

    if let Some(mechanism_index) = mechanisms
        .iter()
        .position(|mechanism| mechanism.detectors.is_empty() && !mechanism.dem_outputs.is_empty())
    {
        return Ok(Some(FaultDistanceResult {
            distance: 1,
            mechanism_indices: vec![mechanism_index],
        }));
    }

    let detector_ids: BTreeSet<u32> = mechanisms
        .iter()
        .flat_map(|mechanism| mechanism.detectors.iter().copied())
        .collect();
    let detector_nodes: BTreeMap<u32, usize> = detector_ids
        .into_iter()
        .enumerate()
        .map(|(node, detector)| (detector, node))
        .collect();
    let boundary = detector_nodes.len();
    let mut adjacency = vec![Vec::new(); boundary + 1];

    for (mechanism_index, mechanism) in mechanisms.iter().enumerate() {
        let endpoints = match mechanism.detectors.as_slice() {
            [] => continue,
            [detector] => (detector_nodes[detector], boundary),
            [first, second] => (detector_nodes[first], detector_nodes[second]),
            _ => unreachable!("hyperedges were rejected before graph construction"),
        };
        adjacency[endpoints.0].push(GraphEdge {
            neighbor: endpoints.1,
            mechanism_index,
        });
        adjacency[endpoints.1].push(GraphEdge {
            neighbor: endpoints.0,
            mechanism_index,
        });
    }

    let observables: BTreeSet<u32> = mechanisms
        .iter()
        .flat_map(|mechanism| mechanism.dem_outputs.iter().copied())
        .collect();
    let num_states = adjacency.len() * 2;
    let mut best = None;

    for observable in observables {
        for start_node in 0..adjacency.len() {
            let start = start_node * 2;
            let target = start + 1;

            // `pecos-num::Graph` stores f64 weights, while its path APIs return either distances
            // or a path, not both. These nodes are also synthetic parity states, so local
            // unweighted BFS gives the exact distance and witness directly without materializing
            // a weighted graph. Every detector and the boundary must be a root: detector-rooted
            // searches find odd-parity cycles in components with no boundary edge.
            let mut distance = vec![usize::MAX; num_states];
            let mut predecessor: Vec<Option<(usize, usize)>> = vec![None; num_states];
            let mut queue = VecDeque::from([start]);
            distance[start] = 0;

            while let Some(state) = queue.pop_front() {
                if state == target {
                    break;
                }
                let node = state / 2;
                let parity = state % 2;
                for edge in &adjacency[node] {
                    let toggles_observable = mechanisms[edge.mechanism_index]
                        .dem_outputs
                        .binary_search(&observable)
                        .is_ok();
                    let next_parity = parity ^ usize::from(toggles_observable);
                    let next_state = edge.neighbor * 2 + next_parity;
                    if distance[next_state] == usize::MAX {
                        distance[next_state] = distance[state] + 1;
                        predecessor[next_state] = Some((state, edge.mechanism_index));
                        queue.push_back(next_state);
                    }
                }
            }

            if distance[target] == usize::MAX {
                continue;
            }

            let mut witness = Vec::with_capacity(distance[target]);
            let mut state = target;
            while state != start {
                let Some((previous, mechanism_index)) = predecessor[state] else {
                    break;
                };
                witness.push(mechanism_index);
                state = previous;
            }
            if state != start {
                continue;
            }
            debug_assert_eq!(witness.len(), distance[target]);
            update_best(&mut best, witness);
        }
    }

    Ok(best)
}

fn first_witness_of_weight(mechanisms: &[FaultMechanism], weight: usize) -> Option<Vec<usize>> {
    if weight == 0 || weight > mechanisms.len() {
        return None;
    }

    let mut indices: Vec<usize> = (0..weight).collect();
    loop {
        let effect = indices
            .iter()
            .fold(FaultMechanism::new(), |effect, &index| {
                effect.xor(&mechanisms[index])
            });
        if effect.detectors.is_empty() && !effect.dem_outputs.is_empty() {
            return Some(indices);
        }

        let position = (0..weight)
            .rev()
            .find(|&position| indices[position] < mechanisms.len() - weight + position)?;
        indices[position] += 1;
        for next in (position + 1)..weight {
            indices[next] = indices[next - 1] + 1;
        }
    }
}

/// Exhaustively computes fault distance up to `max_weight` for any detector error model.
///
/// This simple reference implementation is used to validate
/// [`connected_cluster_fault_distance`]. It supports hyperedges and ignores mechanism
/// probabilities. It examines mechanism subsets in increasing size and returns the first
/// detector-free subset whose XOR flips at least one observable. Its cost is combinatorial: in the
/// worst case it checks
/// `sum(binomial(num_mechanisms, weight), weight=1..max_weight)` subsets, so callers must choose an
/// explicit search budget.
#[must_use]
pub fn exhaustive_fault_distance(
    dem: &DetectorErrorModel,
    max_weight: usize,
) -> Option<FaultDistanceResult> {
    let mechanisms = mechanisms_from_dem(dem);
    for weight in 1..=max_weight.min(mechanisms.len()) {
        if let Some(mechanism_indices) = first_witness_of_weight(&mechanisms, weight) {
            return Some(FaultDistanceResult {
                distance: weight,
                mechanism_indices,
            });
        }
    }
    None
}

/// Computes exact fault distance up to `max_weight` using connected-cluster pruning.
///
/// This is the preferred general-purpose search for real detector error models. It supports
/// hyperedges and ignores mechanism probabilities. Before searching, it repeatedly peels every
/// mechanism containing a detector that occurs in no other remaining mechanism, because such a
/// detector cannot cancel in an undetectable set.
///
/// Connected subsets are enumerated without duplicates using a deterministic Redelmeier-style
/// scheme. The smallest mechanism index is the cluster seed, extensions are restricted to larger
/// indices, and a candidate skipped at one recursion level is excluded from later sibling
/// branches. Thus every connected set has exactly one seed and one construction branch. Search is
/// by increasing weight, with the lexicographically smallest original-index witness retained at
/// each weight, matching [`exhaustive_fault_distance`].
#[must_use]
pub fn connected_cluster_fault_distance(
    dem: &DetectorErrorModel,
    max_weight: usize,
) -> Option<FaultDistanceResult> {
    let mechanisms = mechanisms_from_dem(dem);
    let incidence = detector_incidence(&mechanisms);
    let active = peel_unique_detector_mechanisms(&mechanisms, &incidence);

    for weight in 1..=max_weight.min(mechanisms.len()) {
        let mechanism_indices = ConnectedClusterSearch {
            mechanisms: &mechanisms,
            incidence: &incidence,
            active: &active,
            target_weight: weight,
            best: None,
        }
        .run();
        if let Some(mechanism_indices) = mechanism_indices {
            return Some(FaultDistanceResult {
                distance: weight,
                mechanism_indices,
            });
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dem_from_effects(effects: &[(Vec<u32>, Vec<u32>)]) -> DetectorErrorModel {
        let mut dem = DetectorErrorModel::new();
        for (detectors, observables) in effects {
            dem.add_direct_contribution(
                FaultMechanism::from_unsorted(
                    detectors.iter().copied(),
                    observables.iter().copied(),
                ),
                0.01,
            );
        }
        dem
    }

    #[test]
    fn distance_one_detector_free_mechanism() {
        let dem = dem_from_effects(&[(vec![], vec![0])]);
        let expected = FaultDistanceResult {
            distance: 1,
            mechanism_indices: vec![0],
        };

        assert_eq!(graphlike_fault_distance(&dem), Ok(Some(expected.clone())));
        assert_eq!(exhaustive_fault_distance(&dem, 1), Some(expected.clone()));
        assert_eq!(connected_cluster_fault_distance(&dem, 1), Some(expected));
    }

    #[test]
    fn repetition_code_triad_has_distance_three_and_same_witness() {
        let dem = dem_from_effects(&[(vec![0, 1], vec![0]), (vec![0], vec![]), (vec![1], vec![])]);
        let expected = FaultDistanceResult {
            distance: 3,
            mechanism_indices: vec![0, 1, 2],
        };

        assert_eq!(graphlike_fault_distance(&dem), Ok(Some(expected.clone())));
        assert_eq!(exhaustive_fault_distance(&dem, 3), Some(expected.clone()));
        assert_eq!(connected_cluster_fault_distance(&dem, 3), Some(expected));
    }

    #[test]
    fn detector_only_cycle_has_distance_three_and_same_witness() {
        // There are no boundary edges. All three mechanisms form the unique detector-free
        // logical cycle: D0 D1 L0 ^ D1 D2 ^ D0 D2 = L0.
        let dem = dem_from_effects(&[
            (vec![0, 1], vec![0]),
            (vec![1, 2], vec![]),
            (vec![0, 2], vec![]),
        ]);
        let expected = FaultDistanceResult {
            distance: 3,
            mechanism_indices: vec![0, 1, 2],
        };

        assert_eq!(graphlike_fault_distance(&dem), Ok(Some(expected.clone())));
        assert_eq!(exhaustive_fault_distance(&dem, 3), Some(expected.clone()));
        assert_eq!(connected_cluster_fault_distance(&dem, 3), Some(expected));
    }

    #[test]
    fn no_undetectable_logical_error_returns_none() {
        let dem = dem_from_effects(&[(vec![0], vec![0]), (vec![1], vec![])]);

        assert_eq!(graphlike_fault_distance(&dem), Ok(None));
        assert_eq!(exhaustive_fault_distance(&dem, 8), None);
        assert_eq!(connected_cluster_fault_distance(&dem, 8), None);
    }

    #[test]
    fn search_budget_below_distance_returns_none() {
        let dem = dem_from_effects(&[(vec![0, 1], vec![0]), (vec![0], vec![]), (vec![1], vec![])]);

        assert_eq!(exhaustive_fault_distance(&dem, 2), None);
        assert_eq!(connected_cluster_fault_distance(&dem, 2), None);
    }

    #[test]
    fn graphlike_rejects_hyperedges_and_exhaustive_uses_them() {
        // The only logical witness is all three mechanisms. It necessarily includes the
        // hyperedge D0 D1 D2 L0, whose detectors cancel against D0 D1 and D2.
        let dem = dem_from_effects(&[
            (vec![0, 1, 2], vec![0]),
            (vec![0, 1], vec![]),
            (vec![2], vec![]),
        ]);

        assert_eq!(
            graphlike_fault_distance(&dem),
            Err(FaultDistanceError::HyperedgesPresent { count: 1 })
        );
        let expected = FaultDistanceResult {
            distance: 3,
            mechanism_indices: vec![0, 1, 2],
        };
        assert_eq!(exhaustive_fault_distance(&dem, 3), Some(expected.clone()));
        assert_eq!(connected_cluster_fault_distance(&dem, 3), Some(expected));
        assert_eq!(exhaustive_fault_distance(&dem, 2), None);
        assert_eq!(connected_cluster_fault_distance(&dem, 2), None);
    }

    #[test]
    fn searches_every_observable() {
        // L0 has no detector-free witness. The two D0 mechanisms cancel their detector and flip
        // L1, so a search restricted to L0 would incorrectly return None.
        let dem = dem_from_effects(&[(vec![1], vec![0]), (vec![0], vec![1]), (vec![0], vec![])]);
        let expected = FaultDistanceResult {
            distance: 2,
            mechanism_indices: vec![0, 1],
        };

        assert_eq!(graphlike_fault_distance(&dem), Ok(Some(expected.clone())));
        assert_eq!(exhaustive_fault_distance(&dem, 3), Some(expected.clone()));
        assert_eq!(connected_cluster_fault_distance(&dem, 3), Some(expected));
    }

    #[test]
    fn all_searches_agree_on_seeded_small_random_dems_with_hyperedges() {
        use rand::rngs::SmallRng;
        use rand::{RngExt, SeedableRng};

        const NUM_CASES: usize = 512;
        const MAX_MECHANISMS: usize = 6;
        const NUM_DETECTORS: u32 = 4;
        const NUM_OBSERVABLES: u32 = 2;

        let mut rng = SmallRng::seed_from_u64(0x000D_157A_11CE_5EED);
        for case_index in 0..NUM_CASES {
            let graphlike_case = case_index % 2 == 0;
            let num_mechanisms = if graphlike_case {
                rng.random_range(0..=MAX_MECHANISMS)
            } else {
                rng.random_range(1..=MAX_MECHANISMS)
            };
            let mut effects = Vec::with_capacity(num_mechanisms);
            for mechanism_index in 0..num_mechanisms {
                let detectors = if !graphlike_case && mechanism_index == 0 {
                    vec![0, 1, 2]
                } else {
                    let detector_limit = if graphlike_case { 2 } else { 4 };
                    let mut detectors = Vec::with_capacity(detector_limit);
                    for detector in 0..NUM_DETECTORS {
                        if detectors.len() < detector_limit && rng.random_bool(0.5) {
                            detectors.push(detector);
                        }
                    }
                    detectors
                };
                let observables = (0..NUM_OBSERVABLES)
                    .filter(|_| rng.random_bool(0.5))
                    .collect();
                effects.push((detectors, observables));
            }

            let dem = dem_from_effects(&effects);
            let exhaustive = exhaustive_fault_distance(&dem, MAX_MECHANISMS);
            let connected = connected_cluster_fault_distance(&dem, MAX_MECHANISMS);

            assert_eq!(
                connected, exhaustive,
                "general searches differed for seeded case {case_index}: {effects:?}"
            );

            if graphlike_case {
                let graphlike = graphlike_fault_distance(&dem)
                    .expect("the graphlike half of the generator has no hyperedges");
                assert_eq!(
                    graphlike.is_some(),
                    exhaustive.is_some(),
                    "solution existence differed for seeded case {case_index}: {effects:?}"
                );
                assert_eq!(
                    graphlike.as_ref().map(|result| result.distance),
                    exhaustive.as_ref().map(|result| result.distance),
                    "distance differed for seeded case {case_index}: {effects:?}"
                );
            } else {
                assert!(
                    matches!(
                        graphlike_fault_distance(&dem),
                        Err(FaultDistanceError::HyperedgesPresent { .. })
                    ),
                    "hyperedge case {case_index} did not contain a hyperedge: {effects:?}"
                );
            }
        }
    }

    #[test]
    fn peeling_unique_detectors_reaches_a_fixpoint_without_changing_distance() {
        // D0 is initially unique. Peeling its mechanism makes D1 unique, which makes D2 unique.
        // The two D10 mechanisms remain and form the only logical witness.
        let dem = dem_from_effects(&[
            (vec![0, 1], vec![]),
            (vec![1, 2], vec![]),
            (vec![2], vec![]),
            (vec![10], vec![0]),
            (vec![10], vec![]),
        ]);
        let mechanisms = mechanisms_from_dem(&dem);
        let incidence = detector_incidence(&mechanisms);
        let active = peel_unique_detector_mechanisms(&mechanisms, &incidence);

        for (mechanism, &is_active) in mechanisms.iter().zip(&active) {
            assert_eq!(is_active, mechanism.detectors.as_slice() == [10]);
        }

        let exhaustive = exhaustive_fault_distance(&dem, mechanisms.len());
        let connected = connected_cluster_fault_distance(&dem, mechanisms.len());
        assert_eq!(connected, exhaustive);
        let witness = connected.expect("the shared D10 pair is a logical witness");
        assert!(
            witness
                .mechanism_indices
                .iter()
                .all(|&index| { mechanisms[index].detectors.as_slice() == [10] })
        );
    }

    #[test]
    fn peeling_preserves_a_witness_whose_detectors_are_all_shared() {
        let dem = dem_from_effects(&[(vec![0], vec![0]), (vec![0], vec![])]);
        let expected = exhaustive_fault_distance(&dem, 2);

        assert_eq!(
            connected_cluster_fault_distance(&dem, 2),
            expected,
            "both mechanisms sharing D0 must survive peeling"
        );
        assert_eq!(expected.as_ref().map(|result| result.distance), Some(2));
    }
}
