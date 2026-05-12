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

//! Maximum likelihood lookup table decoder.
//!
//! Builds a decoder by enumerating fault combinations up to a given weight,
//! computing their probabilities from a noise model, and for each syndrome
//! pattern choosing the most likely observable outcome.
//!
//! # Example
//!
//! ```
//! use pecos_qec::fault_tolerance::lookup_decoder::LookupDecoder;
//! use pecos_qec::fault_tolerance::dem_builder::NoiseConfig;
//! use pecos_qec::fault_tolerance::propagator::dag::DagFaultInfluenceMap;
//!
//! let map = DagFaultInfluenceMap::with_capacity(0);
//! let noise = NoiseConfig::uniform(0.001);
//!
//! let decoder = LookupDecoder::build(&map, &noise, 3);
//! let result = decoder.decode(&[]);
//! assert!(result.known_syndrome);
//! assert!(result.corrections.is_empty());
//! ```

use super::dem_builder::NoiseConfig;
use super::propagator::dag::{DagFaultInfluenceMap, DagSpacetimeLocation, GateFaultLocation};
use pecos_core::gate_type::GateType;
use std::collections::BTreeMap;

/// Maximum likelihood lookup table decoder.
///
/// Maps syndrome patterns (sets of fired detectors) to the most likely
/// observable correction (which observables to flip).
#[derive(Debug, Clone)]
pub struct LookupDecoder {
    /// Syndrome -> most likely observable flip pattern.
    table: BTreeMap<Vec<u32>, Vec<bool>>,
    /// Standard observable `L<n>` IDs in correction-vector order.
    observable_ids: Vec<u32>,
    /// Maximum fault weight enumerated.
    max_weight: usize,
    /// Total probability mass accounted for (weight 0 through `max_weight`).
    accounted_probability: f64,
}

/// Result of decoding a syndrome.
#[derive(Debug, Clone)]
pub struct DecoderResult {
    /// Which observables should be flipped (ML correction).
    pub corrections: Vec<bool>,
    /// Whether this syndrome was seen during enumeration.
    pub known_syndrome: bool,
    /// Whether any detector fired (non-empty syndrome).
    /// For detection codes (d=2), discard shots where this is true.
    pub detected: bool,
}

impl LookupDecoder {
    /// Build a lookup decoder by enumerating faults up to the given weight.
    ///
    /// Uses the standard per-gate circuit noise model: each gate faults with
    /// probability p, and each non-identity Pauli is equally likely (p/3 for
    /// 1-qubit, p/15 for 2-qubit). Idle gates with T1/T2 use biased noise.
    #[must_use]
    pub fn build(map: &DagFaultInfluenceMap, noise: &NoiseConfig, max_weight: usize) -> Self {
        let observable_ids = observable_ids(map);
        let num_observables = observable_ids.len();

        let loc_probs = compute_location_probs(&map.locations, noise);
        let locs = map.gate_fault_locations();

        // Pre-compute events and no-fault probabilities per gate location.
        //
        // For PZ/MZ gates, Z faults are physically no-ops. Their probability
        // is absorbed into the no-fault probability so the total sums to 1.0.
        //
        // The combo probability uses a ratio approach:
        //   base_prob = product of no_fault(i) over all locations
        //   combo_prob = base_prob * product of (event_prob / no_fault) for participating locs
        let mut loc_no_fault_probs: Vec<f64> = Vec::with_capacity(locs.len());
        let mut loc_events: Vec<Vec<EventData>> = Vec::with_capacity(locs.len());

        for loc in &locs {
            let p = gate_location_prob(loc, &loc_probs, &map.locations);

            let events = loc.all_events();
            let num_physical_events = events.len();

            // For idle gates with T1/T2, compute biased per-Pauli probabilities
            let idle_pauli_probs = if loc.gate_type == GateType::Idle {
                let duration = map
                    .locations
                    .iter()
                    .find(|l| l.node == loc.node && l.before == loc.before)
                    .map_or(1, |l| l.idle_duration.max(1));
                // Duration values are small integers; precision loss is not a concern.
                #[allow(clippy::cast_precision_loss)]
                Some(noise.idle_pauli_probs(duration as f64))
            } else {
                None
            };

            let n_qubits = loc.num_qubits();
            let custom_weights = if idle_pauli_probs.is_some() {
                None
            } else if n_qubits == 1 {
                noise.p1_weights.as_ref()
            } else {
                noise.p2_weights.as_ref()
            };

            let event_probs: Vec<f64> = if let Some(pp) = &idle_pauli_probs {
                events
                    .iter()
                    .map(|event| {
                        let pauli = event
                            .pauli
                            .paulis()
                            .first()
                            .map_or(pecos_core::Pauli::I, |&(pa, _)| pa);
                        match pauli {
                            pecos_core::Pauli::X => pp.px,
                            pecos_core::Pauli::Y => pp.py,
                            pecos_core::Pauli::Z => pp.pz,
                            pecos_core::Pauli::I => 0.0,
                        }
                    })
                    .collect()
            } else if let Some(weights) = custom_weights {
                events
                    .iter()
                    .map(|event| p * weights.weight_for(&event.pauli))
                    .collect()
            } else {
                // Event count is a small integer; precision loss is not a concern.
                #[allow(clippy::cast_precision_loss)]
                let per_event = if num_physical_events > 0 {
                    p / num_physical_events as f64
                } else {
                    0.0
                };
                vec![per_event; events.len()]
            };

            // No-fault = 1 - sum(event probs), absorbing filtered Paulis
            let total_event_prob: f64 = event_probs.iter().sum();
            let no_fault = (1.0 - total_event_prob).max(0.0);

            let event_data: Vec<EventData> = events
                .into_iter()
                .zip(event_probs)
                .map(|(event, prob)| {
                    let ratio = if no_fault > 0.0 {
                        prob / no_fault
                    } else {
                        prob
                    };
                    EventData {
                        prob: ratio,
                        detectors: event.detectors,
                        observable_ids: event
                            .dem_outputs
                            .iter()
                            .filter_map(|&idx| map.observable_id_for_internal_dem_output(idx))
                            .collect(),
                    }
                })
                .collect();

            loc_no_fault_probs.push(no_fault);
            loc_events.push(event_data);
        }

        // Accumulate syndrome probabilities separately from per-observable
        // correction weights. This keeps probability accounting well-defined
        // even for detector-only maps with zero observables.
        let mut syndrome_probabilities: BTreeMap<Vec<u32>, f64> = BTreeMap::new();

        // Accumulate: syndrome -> per-observable (flip_prob, noflip_prob)
        let mut syndrome_data: BTreeMap<Vec<u32>, Vec<(f64, f64)>> = BTreeMap::new();

        // Base probability: all locations no-fault
        let base_prob: f64 = loc_no_fault_probs.iter().product();

        // Weight 0: no faults. Empty syndrome, no observable flips.
        {
            syndrome_probabilities.insert(Vec::new(), base_prob);
            let entry = syndrome_data
                .entry(Vec::new())
                .or_insert_with(|| vec![(0.0, 0.0); num_observables]);
            for w in entry.iter_mut() {
                w.1 += base_prob; // no flip
            }
        }

        // Weight 1..max_weight
        // Start with base_prob (all no-fault). Each event replaces a location's
        // no-fault factor with the event factor via the pre-computed ratio.
        let combo_state = ComboState {
            prob: base_prob,
            detectors: Vec::new(),
            observable_ids: Vec::new(),
        };

        for weight in 1..=max_weight {
            enumerate_combos(
                &loc_events,
                weight,
                0,
                &combo_state,
                &observable_ids,
                &mut syndrome_probabilities,
                &mut syndrome_data,
            );
        }

        let accounted_probability: f64 = syndrome_probabilities.values().sum();

        // Build ML decision table
        let table = syndrome_data
            .into_iter()
            .map(|(syndrome, weights)| {
                let corrections: Vec<bool> = weights
                    .iter()
                    .map(|&(flip, noflip)| flip > noflip)
                    .collect();
                (syndrome, corrections)
            })
            .collect();

        Self {
            table,
            observable_ids,
            max_weight,
            accounted_probability,
        }
    }

    /// Decode a syndrome given as detector indices.
    #[must_use]
    pub fn decode(&self, syndrome: &[u32]) -> DecoderResult {
        let mut key: Vec<u32> = syndrome.to_vec();
        key.sort_unstable();
        let detected = !key.is_empty();

        if let Some(corrections) = self.table.get(&key) {
            DecoderResult {
                corrections: corrections.clone(),
                known_syndrome: true,
                detected,
            }
        } else {
            DecoderResult {
                corrections: vec![false; self.observable_ids.len()],
                known_syndrome: false,
                detected,
            }
        }
    }

    /// Decode from a boolean detector vector.
    #[must_use]
    pub fn decode_from_bools(&self, detectors: &[bool]) -> DecoderResult {
        let syndrome: Vec<u32> = detectors
            .iter()
            .enumerate()
            .filter_map(|(i, &fired)| {
                if fired {
                    #[allow(clippy::cast_possible_truncation)]
                    Some(i as u32)
                } else {
                    None
                }
            })
            .collect();
        self.decode(&syndrome)
    }

    /// Number of distinct syndrome patterns in the table.
    #[must_use]
    pub fn num_syndromes(&self) -> usize {
        self.table.len()
    }

    /// Maximum fault weight that was enumerated.
    #[must_use]
    pub fn max_weight(&self) -> usize {
        self.max_weight
    }

    /// Number of observable channels.
    #[must_use]
    pub fn num_observables(&self) -> usize {
        self.observable_ids.len()
    }

    /// Standard observable `L<n>` IDs in correction-vector order.
    #[must_use]
    pub fn observable_ids(&self) -> &[u32] {
        &self.observable_ids
    }

    /// Estimated upper bound on the probability mass NOT accounted for
    /// due to weight truncation.
    ///
    /// This bounds the total probability of fault combinations with
    /// weight > `max_weight`. For low noise rates, this is small.
    /// If it's large (> 0.01), consider increasing `max_weight`.
    #[must_use]
    pub fn truncation_bound(&self) -> f64 {
        (1.0 - self.accounted_probability).max(0.0)
    }

    /// Total probability mass accounted for (weights 0 through `max_weight`).
    ///
    /// Should be close to 1.0 for low noise rates with sufficient `max_weight`.
    #[must_use]
    pub fn accounted_probability(&self) -> f64 {
        self.accounted_probability
    }
}

// ============================================================================
// Internal types and helpers
// ============================================================================

struct EventData {
    prob: f64,
    detectors: Vec<u32>,
    observable_ids: Vec<u32>,
}

#[derive(Clone)]
struct ComboState {
    prob: f64,
    detectors: Vec<u32>,
    observable_ids: Vec<u32>,
}

impl ComboState {
    /// Compose with a new event: XOR detectors/observable IDs, multiply prob.
    fn compose(&self, event: &EventData) -> Self {
        let mut detectors = self.detectors.clone();
        xor_into(&mut detectors, &event.detectors);
        let mut observable_ids = self.observable_ids.clone();
        xor_into(&mut observable_ids, &event.observable_ids);
        Self {
            prob: self.prob * event.prob,
            detectors,
            observable_ids,
        }
    }
}

/// Symmetric difference (XOR) of sorted u32 vecs.
fn xor_into(acc: &mut Vec<u32>, other: &[u32]) {
    if other.is_empty() {
        return;
    }
    if acc.is_empty() {
        acc.extend_from_slice(other);
        return;
    }
    let mut result = Vec::with_capacity(acc.len() + other.len());
    let (mut i, mut j) = (0, 0);
    while i < acc.len() && j < other.len() {
        match acc[i].cmp(&other[j]) {
            std::cmp::Ordering::Less => {
                result.push(acc[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                result.push(other[j]);
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                i += 1;
                j += 1;
            }
        }
    }
    result.extend_from_slice(&acc[i..]);
    result.extend_from_slice(&other[j..]);
    *acc = result;
}

/// Recursive combination enumeration with probability tracking.
fn enumerate_combos(
    loc_events: &[Vec<EventData>],
    remaining: usize,
    start_loc: usize,
    state: &ComboState,
    observable_ids: &[u32],
    syndrome_probabilities: &mut BTreeMap<Vec<u32>, f64>,
    syndrome_data: &mut BTreeMap<Vec<u32>, Vec<(f64, f64)>>,
) {
    if remaining == 0 {
        let mut syndrome = state.detectors.clone();
        syndrome.sort_unstable();

        *syndrome_probabilities
            .entry(syndrome.clone())
            .or_insert(0.0) += state.prob;

        let entry = syndrome_data
            .entry(syndrome)
            .or_insert_with(|| vec![(0.0, 0.0); observable_ids.len()]);

        for (&observable_id, weights) in observable_ids.iter().zip(entry.iter_mut()) {
            let flipped = state.observable_ids.contains(&observable_id);
            if flipped {
                weights.0 += state.prob;
            } else {
                weights.1 += state.prob;
            }
        }
        return;
    }

    for loc_idx in start_loc..loc_events.len() {
        for event in &loc_events[loc_idx] {
            let next_state = state.compose(event);
            enumerate_combos(
                loc_events,
                remaining - 1,
                loc_idx + 1,
                &next_state,
                observable_ids,
                syndrome_probabilities,
                syndrome_data,
            );
        }
    }
}

/// Compute per-location error probabilities from noise config.
fn compute_location_probs(locations: &[DagSpacetimeLocation], noise: &NoiseConfig) -> Vec<f64> {
    super::dem_builder::sampler::compute_location_probs_from_noise(locations, noise)
}

/// Get the per-qubit error probability for a gate fault location.
///
/// Since `gate_fault_locations` groups per-qubit locations, all per-qubit
/// locations within a gate have the same gate-type-based probability.
fn gate_location_prob(
    loc: &GateFaultLocation<'_>,
    loc_probs: &[f64],
    all_locations: &[DagSpacetimeLocation],
) -> f64 {
    // Find any per-qubit location in the influence map for this gate
    for (i, l) in all_locations.iter().enumerate() {
        if l.node == loc.node && l.before == loc.before {
            return loc_probs[i];
        }
    }
    0.0
}

fn observable_ids(map: &DagFaultInfluenceMap) -> Vec<u32> {
    map.observable_ids().into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fault_tolerance::InfluenceBuilder;
    use pecos_core::pauli::X;
    use pecos_quantum::DagCircuit;

    #[test]
    fn observable_indices_use_compact_l_namespace_with_tracked_paulis() {
        let mut dag = DagCircuit::new();
        dag.pz(&[0]);
        dag.tracked_pauli_labeled("track_x", X(0));
        let meas = dag.mz(&[0]);
        dag.observable_labeled("obs0", &[meas[0]]);

        let map = InfluenceBuilder::new(&dag)
            .with_circuit_annotations(&dag)
            .build();
        assert_eq!(map.num_tracked_paulis(), 1);
        assert_eq!(map.num_observables(), 1);

        let decoder = LookupDecoder::build(&map, &NoiseConfig::uniform(0.01), 1);
        assert_eq!(decoder.observable_ids(), &[0]);
    }

    #[test]
    fn detector_only_decoder_accounts_probability_without_observables() {
        let mut dag = DagCircuit::new();
        dag.pz(&[0, 1]);
        dag.cx(&[(0, 1)]);
        let meas = dag.mz(&[1]);
        dag.detector(&[meas[0]]);

        let map = InfluenceBuilder::new(&dag).build();
        assert_eq!(map.num_observables(), 0);

        let decoder = LookupDecoder::build(&map, &NoiseConfig::uniform(0.01), 0);
        assert_eq!(decoder.num_observables(), 0);
        assert!(decoder.accounted_probability() > 0.0);
        assert!(decoder.truncation_bound() < 1.0);
    }
}
