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

//! Targeted fault-catalog lookup decoder.
//!
//! Answers one detector syndrome at a time by searching the fault catalog,
//! instead of precomputing a full lookup table for every syndrome.
//!
//! Uses odds-space weights for efficient comparison:
//! - `base_probability = product of all (1 - p_i)`
//! - `odds_weight(alt) = alt.absolute_probability / (1 - p_i)`
//! - `configuration_probability = base_probability * product(selected odds_weights)`

use super::fault_sampler::FaultCatalog;
use std::collections::{BTreeMap, BTreeSet, HashMap};

/// A flattened fault entry for index lookup.
#[derive(Clone, Debug)]
struct FaultEntry {
    /// Index of the physical fault location in the catalog.
    location_index: usize,
    /// Detector effect as a sorted set of detector indices (XOR parity).
    detector_bits: BTreeSet<usize>,
    /// Observable/logical effect as a sorted set.
    logical_bits: BTreeSet<usize>,
    /// Odds-space weight: `absolute_probability / no_fault_probability`.
    odds_weight: f64,
}

#[derive(Clone, Copy)]
struct SearchState<'a> {
    start_entry: usize,
    needed: &'a BTreeSet<usize>,
    used_locations: &'a BTreeSet<usize>,
    logical_parity: &'a BTreeSet<usize>,
    odds_product: f64,
    depth: usize,
}

/// Result of decoding a single syndrome.
#[derive(Clone, Debug)]
pub struct DecodeResult {
    /// The queried syndrome.
    pub syndrome: Vec<usize>,
    /// Accumulated odds-space weights by logical class.
    /// Multiply by `base_probability` for absolute probabilities.
    pub logical_weights: BTreeMap<Vec<usize>, f64>,
    /// The logical class with the highest weight.
    pub best_logical: Vec<usize>,
}

/// Targeted fault-catalog lookup decoder.
///
/// Searches the fault catalog for explanations of a given detector syndrome,
/// accumulating odds-space weights by logical class up to `max_faults`
/// simultaneous fault locations.
pub struct TargetedLookupDecoder {
    max_faults: usize,
    base_prob: f64,
    entries: Vec<FaultEntry>,
    /// Index: `detector_bits` -> list of entry indices.
    by_detector: HashMap<BTreeSet<usize>, Vec<usize>>,
}

impl TargetedLookupDecoder {
    /// Build a decoder from a fault catalog.
    #[must_use]
    pub fn new(catalog: &FaultCatalog) -> Self {
        let base_prob: f64 = catalog
            .locations
            .iter()
            .map(|loc| loc.no_fault_probability)
            .product();

        let mut entries = Vec::new();
        for (loc_idx, loc) in catalog.locations.iter().enumerate() {
            for alt in &loc.faults {
                let odds = if loc.no_fault_probability > 0.0 {
                    alt.absolute_probability / loc.no_fault_probability
                } else {
                    f64::INFINITY
                };
                if odds == 0.0 {
                    continue;
                }
                entries.push(FaultEntry {
                    location_index: loc_idx,
                    detector_bits: alt.affected_detectors.iter().copied().collect(),
                    logical_bits: alt.affected_observables.iter().copied().collect(),
                    odds_weight: odds,
                });
            }
        }

        let mut by_detector: HashMap<BTreeSet<usize>, Vec<usize>> = HashMap::new();
        for (i, entry) in entries.iter().enumerate() {
            by_detector
                .entry(entry.detector_bits.clone())
                .or_default()
                .push(i);
        }

        Self {
            max_faults: 1,
            base_prob,
            entries,
            by_detector,
        }
    }

    /// Set the maximum number of simultaneous fault locations to consider.
    #[must_use]
    pub fn max_faults(mut self, max_faults: usize) -> Self {
        self.max_faults = max_faults;
        self
    }

    /// The all-no-fault probability: product of `(1 - p_i)` for all locations.
    #[must_use]
    pub fn base_probability(&self) -> f64 {
        self.base_prob
    }

    /// Decode a syndrome: find all explanations up to `max_faults` and accumulate
    /// odds-space weights by logical class.
    #[must_use]
    pub fn decode(&self, syndrome: &[usize]) -> DecodeResult {
        let target: BTreeSet<usize> = syndrome.iter().copied().collect();
        let mut logical_weights: BTreeMap<Vec<usize>, f64> = BTreeMap::new();

        // k=0: empty syndrome -> empty logical with weight 1
        if target.is_empty() {
            *logical_weights.entry(Vec::new()).or_default() += 1.0;
        }

        // k=1: direct lookup
        if self.max_faults >= 1
            && let Some(indices) = self.by_detector.get(&target)
        {
            for &i in indices {
                let e = &self.entries[i];
                let logical: Vec<usize> = e.logical_bits.iter().copied().collect();
                *logical_weights.entry(logical).or_default() += e.odds_weight;
            }
        }

        // k=2: complement lookup
        if self.max_faults >= 2 {
            self.search_k2(&target, &mut logical_weights);
        }

        // k>=3: recursive exact search
        if self.max_faults >= 3 {
            for k in 3..=self.max_faults {
                self.search_generic(k, &target, &mut logical_weights);
            }
        }

        let best_logical = logical_weights
            .iter()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(logical, _)| logical.clone())
            .unwrap_or_default();

        DecodeResult {
            syndrome: syndrome.to_vec(),
            logical_weights,
            best_logical,
        }
    }

    /// k=2 complement lookup: for each entry `a`, compute
    /// `needed_b = target XOR a.detectors`,
    /// then look up entries with that detector effect.
    fn search_k2(&self, target: &BTreeSet<usize>, logical_weights: &mut BTreeMap<Vec<usize>, f64>) {
        for (i, a) in self.entries.iter().enumerate() {
            let needed_b = xor_sets(target, &a.detector_bits);
            if let Some(b_indices) = self.by_detector.get(&needed_b) {
                for &j in b_indices {
                    if j <= i {
                        continue; // avoid double-counting (ordered pairs)
                    }
                    let b = &self.entries[j];
                    if a.location_index == b.location_index {
                        continue; // same physical location
                    }
                    let logical = xor_sets(&a.logical_bits, &b.logical_bits);
                    let logical_vec: Vec<usize> = logical.into_iter().collect();
                    *logical_weights.entry(logical_vec).or_default() +=
                        a.odds_weight * b.odds_weight;
                }
            }
        }
    }

    /// Generic exact search for k >= 3. Recursive depth-first with location exclusion.
    fn search_generic(
        &self,
        k: usize,
        target: &BTreeSet<usize>,
        logical_weights: &mut BTreeMap<Vec<usize>, f64>,
    ) {
        let used_locations = BTreeSet::new();
        let logical_parity = BTreeSet::new();
        let state = SearchState {
            start_entry: 0,
            needed: target,
            used_locations: &used_locations,
            logical_parity: &logical_parity,
            odds_product: 1.0,
            depth: 0,
        };
        self.search_recursive(k, state, logical_weights);
    }

    fn search_recursive(
        &self,
        k: usize,
        state: SearchState<'_>,
        logical_weights: &mut BTreeMap<Vec<usize>, f64>,
    ) {
        if state.depth == k {
            if state.needed.is_empty() {
                let logical_vec: Vec<usize> = state.logical_parity.iter().copied().collect();
                *logical_weights.entry(logical_vec).or_default() += state.odds_product;
            }
            return;
        }

        let remaining = k - state.depth;
        for i in state.start_entry..self.entries.len() {
            // Check if enough entries remain
            if self.entries.len() - i < remaining {
                break;
            }

            let entry = &self.entries[i];

            // Skip if this location is already used
            if state.used_locations.contains(&entry.location_index) {
                continue;
            }

            let new_needed = xor_sets(state.needed, &entry.detector_bits);
            let new_logical = xor_sets(state.logical_parity, &entry.logical_bits);
            let new_odds = state.odds_product * entry.odds_weight;

            let mut new_used = state.used_locations.clone();
            new_used.insert(entry.location_index);

            let next_state = SearchState {
                start_entry: i + 1,
                needed: &new_needed,
                used_locations: &new_used,
                logical_parity: &new_logical,
                odds_product: new_odds,
                depth: state.depth + 1,
            };
            self.search_recursive(k, next_state, logical_weights);
        }
    }
}

/// XOR two sorted sets (symmetric difference).
fn xor_sets(a: &BTreeSet<usize>, b: &BTreeSet<usize>) -> BTreeSet<usize> {
    a.symmetric_difference(b).copied().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fault_tolerance::fault_sampler::{
        FaultAlternative, FaultCatalog, FaultKind, FaultLocation, StochasticNoiseParams,
        build_fault_catalog,
    };
    use pecos_core::{QubitId, gate_type::GateType};
    use pecos_quantum::TickCircuit;

    /// Build a tiny circuit: H(0) CX(0,1) H(0) MZ(0) MZ(1)
    /// with detector D0 = m0 XOR m1 and observable L0 = m1.
    fn tiny_circuit() -> TickCircuit {
        let mut tc = TickCircuit::new();
        tc.tick().h(&[QubitId(0)]);
        tc.tick().cx(&[(QubitId(0), QubitId(1))]);
        tc.tick().h(&[QubitId(0)]);
        tc.tick().mz(&[QubitId(0)]);
        tc.tick().mz(&[QubitId(1)]);
        tc.set_meta(
            "num_measurements",
            pecos_quantum::Attribute::String("2".to_string()),
        );
        tc.set_meta(
            "detectors",
            pecos_quantum::Attribute::String(r#"[{"records": [-2, -1]}]"#.to_string()),
        );
        tc.set_meta(
            "observables",
            pecos_quantum::Attribute::String(r#"[{"records": [-1]}]"#.to_string()),
        );
        tc
    }

    fn tiny_catalog() -> FaultCatalog {
        let tc = tiny_circuit();
        let noise = StochasticNoiseParams {
            p1: 0.003,
            p2: 0.01,
            p_meas: 0.005,
            p_prep: 0.005,
        };
        build_fault_catalog(&tc, &noise).unwrap()
    }

    /// Brute-force reference: enumerate all configurations up to `max_faults`,
    /// accumulate odds weights by (syndrome, logical).
    fn brute_force_weights(
        catalog: &FaultCatalog,
        max_faults: usize,
    ) -> BTreeMap<Vec<usize>, BTreeMap<Vec<usize>, f64>> {
        let base: f64 = catalog
            .locations
            .iter()
            .map(|l| l.no_fault_probability)
            .product();
        let mut result: BTreeMap<Vec<usize>, BTreeMap<Vec<usize>, f64>> = BTreeMap::new();
        for k in 0..=max_faults {
            for event in catalog.fault_configurations(k) {
                let odds = if base > 0.0 {
                    event.configuration_probability / base
                } else {
                    0.0
                };
                *result
                    .entry(event.affected_detectors)
                    .or_default()
                    .entry(event.affected_observables)
                    .or_default() += odds;
            }
        }
        result
    }

    #[test]
    fn test_k0_empty_syndrome() {
        let catalog = tiny_catalog();
        let decoder = TargetedLookupDecoder::new(&catalog).max_faults(0);
        let result = decoder.decode(&[]);
        assert_eq!(result.best_logical, Vec::<usize>::new());
        assert!((result.logical_weights[&vec![]] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_unparameterized_catalog_has_no_positive_fault_weights() {
        let catalog = FaultCatalog::from_circuit(&tiny_circuit()).unwrap();
        let decoder = TargetedLookupDecoder::new(&catalog).max_faults(2);

        let empty = decoder.decode(&[]);
        assert_eq!(empty.logical_weights.len(), 1);
        assert!((empty.logical_weights[&vec![]] - 1.0).abs() < 1e-12);

        let non_empty = decoder.decode(&[0]);
        assert!(
            non_empty.logical_weights.is_empty(),
            "zero-probability structural faults must not create zero-weight classes"
        );
    }

    #[test]
    fn test_zero_probability_alternatives_are_ignored() {
        let catalog = FaultCatalog {
            locations: vec![
                FaultLocation {
                    tick: 0,
                    gate_index: 0,
                    gate_type: GateType::H,
                    qubits: vec![0],
                    channel: crate::fault_tolerance::fault_sampler::FaultChannel::P1,
                    channel_probability: 0.0,
                    no_fault_probability: 1.0,
                    num_alternatives: 1,
                    faults: vec![FaultAlternative {
                        kind: FaultKind::Pauli,
                        pauli: None,
                        affected_measurements: Vec::new(),
                        affected_detectors: vec![0],
                        affected_observables: vec![9],
                        affected_tracked_paulis: Vec::new(),
                        conditional_probability: 1.0,
                        absolute_probability: 0.0,
                    }],
                },
                FaultLocation {
                    tick: 1,
                    gate_index: 0,
                    gate_type: GateType::MZ,
                    qubits: vec![0],
                    channel: crate::fault_tolerance::fault_sampler::FaultChannel::PMeas,
                    channel_probability: 0.1,
                    no_fault_probability: 0.9,
                    num_alternatives: 1,
                    faults: vec![FaultAlternative {
                        kind: FaultKind::MeasurementFlip,
                        pauli: None,
                        affected_measurements: vec![0],
                        affected_detectors: vec![0],
                        affected_observables: Vec::new(),
                        affected_tracked_paulis: Vec::new(),
                        conditional_probability: 1.0,
                        absolute_probability: 0.1,
                    }],
                },
            ],
        };

        let result = TargetedLookupDecoder::new(&catalog)
            .max_faults(1)
            .decode(&[0]);

        assert!(!result.logical_weights.contains_key(&vec![9]));
        assert_eq!(result.logical_weights.len(), 1);
        assert!((result.logical_weights[&vec![]] - (0.1 / 0.9)).abs() < 1e-12);
    }

    #[test]
    fn test_decode_ignores_tracked_pauli_effects() {
        let catalog = FaultCatalog {
            locations: vec![
                FaultLocation {
                    tick: 0,
                    gate_index: 0,
                    gate_type: GateType::H,
                    qubits: vec![0],
                    channel: crate::fault_tolerance::fault_sampler::FaultChannel::P1,
                    channel_probability: 0.2,
                    no_fault_probability: 0.8,
                    num_alternatives: 1,
                    faults: vec![FaultAlternative {
                        kind: FaultKind::Pauli,
                        pauli: None,
                        affected_measurements: Vec::new(),
                        affected_detectors: vec![0],
                        affected_observables: vec![1],
                        affected_tracked_paulis: vec![0],
                        conditional_probability: 1.0,
                        absolute_probability: 0.2,
                    }],
                },
                FaultLocation {
                    tick: 1,
                    gate_index: 0,
                    gate_type: GateType::H,
                    qubits: vec![1],
                    channel: crate::fault_tolerance::fault_sampler::FaultChannel::P1,
                    channel_probability: 0.1,
                    no_fault_probability: 0.9,
                    num_alternatives: 1,
                    faults: vec![FaultAlternative {
                        kind: FaultKind::Pauli,
                        pauli: None,
                        affected_measurements: Vec::new(),
                        affected_detectors: vec![0],
                        affected_observables: Vec::new(),
                        affected_tracked_paulis: vec![3],
                        conditional_probability: 1.0,
                        absolute_probability: 0.1,
                    }],
                },
            ],
        };

        let result = TargetedLookupDecoder::new(&catalog)
            .max_faults(1)
            .decode(&[0]);

        assert_eq!(result.best_logical, vec![1]);
        assert_eq!(result.logical_weights.len(), 2);
        assert!((result.logical_weights[&vec![1]] - (0.2 / 0.8)).abs() < 1e-12);
        assert!((result.logical_weights[&vec![]] - (0.1 / 0.9)).abs() < 1e-12);
    }

    #[test]
    fn test_unexplainable_syndrome_returns_empty_weights() {
        let mut tc = TickCircuit::new();
        tc.tick().mz(&[QubitId(0)]);
        tc.tick().mz(&[QubitId(1)]);
        tc.set_meta(
            "num_measurements",
            pecos_quantum::Attribute::String("2".into()),
        );
        tc.set_meta(
            "detectors",
            pecos_quantum::Attribute::String(r#"[{"records":[-2]},{"records":[-1]}]"#.into()),
        );
        tc.set_meta("observables", pecos_quantum::Attribute::String("[]".into()));

        let noise = StochasticNoiseParams {
            p1: 0.0,
            p2: 0.0,
            p_meas: 0.01,
            p_prep: 0.0,
        };
        let catalog = build_fault_catalog(&tc, &noise).unwrap();

        let zero_fault_decoder = TargetedLookupDecoder::new(&catalog).max_faults(0);
        let zero_fault_result = zero_fault_decoder.decode(&[0]);
        assert!(
            zero_fault_result.logical_weights.is_empty(),
            "non-empty syndrome cannot be explained by zero faults"
        );

        let one_fault_decoder = TargetedLookupDecoder::new(&catalog).max_faults(1);
        let one_fault_result = one_fault_decoder.decode(&[0, 1]);
        assert!(
            one_fault_result.logical_weights.is_empty(),
            "syndrome [0, 1] requires two distinct measurement faults"
        );
    }

    #[test]
    fn test_k1_matches_brute_force() {
        let catalog = tiny_catalog();
        let decoder = TargetedLookupDecoder::new(&catalog).max_faults(1);
        let bf = brute_force_weights(&catalog, 1);

        for (syndrome, bf_logicals) in &bf {
            let result = decoder.decode(syndrome);
            for (logical, &bf_weight) in bf_logicals {
                let dec_weight = result.logical_weights.get(logical).copied().unwrap_or(0.0);
                assert!(
                    (dec_weight - bf_weight).abs() < 1e-12,
                    "k=1 mismatch for syndrome={syndrome:?} logical={logical:?}: \
                     decoder={dec_weight} brute_force={bf_weight}"
                );
            }
        }
    }

    #[test]
    fn test_k2_matches_brute_force() {
        let catalog = tiny_catalog();
        let decoder = TargetedLookupDecoder::new(&catalog).max_faults(2);
        let bf = brute_force_weights(&catalog, 2);

        for (syndrome, bf_logicals) in &bf {
            let result = decoder.decode(syndrome);
            for (logical, &bf_weight) in bf_logicals {
                let dec_weight = result.logical_weights.get(logical).copied().unwrap_or(0.0);
                assert!(
                    (dec_weight - bf_weight).abs() / bf_weight.max(1e-15) < 1e-8,
                    "k=2 mismatch for syndrome={syndrome:?} logical={logical:?}: \
                     decoder={dec_weight:.6e} brute_force={bf_weight:.6e}"
                );
            }
        }
    }

    #[test]
    fn test_new_clifford_gate_circuit_matches_brute_force() {
        let mut tc = TickCircuit::new();
        tc.tick().sx(&[QubitId(0)]);
        tc.tick().cy(&[(QubitId(0), QubitId(1))]);
        tc.tick().sxx(&[(QubitId(0), QubitId(1))]);
        tc.tick().swap(&[(QubitId(0), QubitId(1))]);
        tc.tick().mz(&[QubitId(0)]);
        tc.tick().mz(&[QubitId(1)]);
        tc.set_meta(
            "num_measurements",
            pecos_quantum::Attribute::String("2".into()),
        );
        tc.set_meta(
            "detectors",
            pecos_quantum::Attribute::String(
                r#"[{"records":[-2]},{"records":[-1]},{"records":[-2,-1]}]"#.into(),
            ),
        );
        tc.set_meta(
            "observables",
            pecos_quantum::Attribute::String(r#"[{"records":[-1]}]"#.into()),
        );

        let noise = StochasticNoiseParams {
            p1: 0.003,
            p2: 0.01,
            p_meas: 0.0,
            p_prep: 0.0,
        };
        let catalog = build_fault_catalog(&tc, &noise).unwrap();
        assert!(
            catalog
                .locations
                .iter()
                .any(|loc| loc.gate_type == GateType::SX)
        );
        assert!(
            catalog
                .locations
                .iter()
                .any(|loc| loc.gate_type == GateType::CY)
        );
        assert!(
            catalog
                .locations
                .iter()
                .any(|loc| loc.gate_type == GateType::SXX)
        );
        assert!(
            catalog
                .locations
                .iter()
                .any(|loc| loc.gate_type == GateType::SWAP)
        );

        let decoder = TargetedLookupDecoder::new(&catalog).max_faults(2);
        let bf = brute_force_weights(&catalog, 2);
        for (syndrome, bf_logicals) in &bf {
            let result = decoder.decode(syndrome);
            for (logical, &bf_weight) in bf_logicals {
                let dec_weight = result.logical_weights.get(logical).copied().unwrap_or(0.0);
                assert!(
                    (dec_weight - bf_weight).abs() / bf_weight.max(1e-15) < 1e-8,
                    "new-gate decoder mismatch for syndrome={syndrome:?} logical={logical:?}: \
                     decoder={dec_weight:.6e} brute_force={bf_weight:.6e}"
                );
            }
        }
    }

    #[test]
    fn test_k3_matches_brute_force() {
        // Very small circuit to keep k=3 tractable
        let mut tc = TickCircuit::new();
        tc.tick().h(&[QubitId(0)]);
        tc.tick().mz(&[QubitId(0)]);
        tc.set_meta(
            "num_measurements",
            pecos_quantum::Attribute::String("1".into()),
        );
        tc.set_meta(
            "detectors",
            pecos_quantum::Attribute::String(r#"[{"records":[-1]}]"#.into()),
        );
        tc.set_meta("observables", pecos_quantum::Attribute::String("[]".into()));

        let noise = StochasticNoiseParams {
            p1: 0.01,
            p2: 0.0,
            p_meas: 0.01,
            p_prep: 0.01,
        };
        let catalog = build_fault_catalog(&tc, &noise).unwrap();
        let decoder = TargetedLookupDecoder::new(&catalog).max_faults(3);
        let bf = brute_force_weights(&catalog, 3);

        for (syndrome, bf_logicals) in &bf {
            let result = decoder.decode(syndrome);
            for (logical, &bf_weight) in bf_logicals {
                let dec_weight = result.logical_weights.get(logical).copied().unwrap_or(0.0);
                assert!(
                    (dec_weight - bf_weight).abs() / bf_weight.max(1e-15) < 1e-8,
                    "k=3 mismatch for syndrome={syndrome:?} logical={logical:?}: \
                     decoder={dec_weight:.6e} brute_force={bf_weight:.6e}"
                );
            }
        }
    }

    #[test]
    fn test_cancellation() {
        // Construct a catalog where syndrome {0} is explained by
        // {0,1} XOR {1} (two faults cancelling detector 1).
        let mut tc = TickCircuit::new();
        tc.tick().h(&[QubitId(0)]);
        tc.tick().cx(&[(QubitId(0), QubitId(1))]);
        tc.tick().h(&[QubitId(0)]);
        tc.tick().mz(&[QubitId(0)]);
        tc.tick().mz(&[QubitId(1)]);
        tc.set_meta(
            "num_measurements",
            pecos_quantum::Attribute::String("2".into()),
        );
        tc.set_meta(
            "detectors",
            pecos_quantum::Attribute::String(r#"[{"records":[-2]},{"records":[-1]}]"#.into()),
        );
        tc.set_meta("observables", pecos_quantum::Attribute::String("[]".into()));

        let noise = StochasticNoiseParams {
            p1: 0.01,
            p2: 0.01,
            p_meas: 0.01,
            p_prep: 0.0,
        };
        let catalog = build_fault_catalog(&tc, &noise).unwrap();
        let decoder = TargetedLookupDecoder::new(&catalog).max_faults(2);

        // Check that syndrome [0] has k=2 explanations
        let result = decoder.decode(&[0]);
        let bf = brute_force_weights(&catalog, 2);
        if let Some(bf_logicals) = bf.get(&vec![0]) {
            for (logical, &bf_weight) in bf_logicals {
                let dec_weight = result.logical_weights.get(logical).copied().unwrap_or(0.0);
                assert!(
                    (dec_weight - bf_weight).abs() / bf_weight.max(1e-15) < 1e-8,
                    "Cancellation test mismatch"
                );
            }
        }
    }

    #[test]
    fn test_same_location_exclusion() {
        let catalog = tiny_catalog();
        let decoder = TargetedLookupDecoder::new(&catalog).max_faults(2);

        // Brute force already enforces location exclusion.
        // Verify decoder matches for all syndromes.
        let bf = brute_force_weights(&catalog, 2);
        for (syndrome, bf_logicals) in &bf {
            let result = decoder.decode(syndrome);
            for (logical, &bf_weight) in bf_logicals {
                let dec_weight = result.logical_weights.get(logical).copied().unwrap_or(0.0);
                assert!(
                    (dec_weight - bf_weight).abs() / bf_weight.max(1e-15) < 1e-8,
                    "Location exclusion mismatch at syndrome={syndrome:?}"
                );
            }
        }
    }

    #[test]
    fn test_empty_syndrome_with_silent_alternatives() {
        // Empty-detector alternatives contribute to empty syndrome
        let catalog = tiny_catalog();
        let decoder = TargetedLookupDecoder::new(&catalog).max_faults(1);
        let result = decoder.decode(&[]);

        // k=0 contributes weight 1.0 for empty logical.
        // k=1 empty-detector alternatives also contribute.
        assert!(
            result.logical_weights.contains_key(&vec![]),
            "Empty logical should appear for empty syndrome"
        );
        assert!(
            *result.logical_weights.get(&vec![]).unwrap() >= 1.0,
            "Empty-syndrome weight should be >= 1.0 (k=0 contributes 1)"
        );
    }

    #[test]
    fn test_odds_to_absolute_probability() {
        let catalog = tiny_catalog();
        let decoder = TargetedLookupDecoder::new(&catalog).max_faults(1);
        let result = decoder.decode(&[0]);

        // Sum of odds weights * base_probability = sum of configuration_probabilities
        let total_odds: f64 = result.logical_weights.values().sum();
        let total_abs = total_odds * decoder.base_probability();
        assert!(total_abs > 0.0, "Should have nonzero absolute probability");
        assert!(total_abs < 1.0, "Total probability should be < 1");
    }

    #[test]
    fn test_k2_no_double_counting() {
        let catalog = tiny_catalog();
        let decoder = TargetedLookupDecoder::new(&catalog).max_faults(2);
        let bf = brute_force_weights(&catalog, 2);

        // Check EVERY brute-force entry matches decoder exactly
        for (syndrome, bf_logicals) in &bf {
            let result = decoder.decode(syndrome);
            let total_bf: f64 = bf_logicals.values().sum();
            let total_dec: f64 = result.logical_weights.values().sum();
            assert!(
                (total_dec - total_bf).abs() / total_bf.max(1e-15) < 1e-8,
                "k=2 total weight mismatch at syndrome={syndrome:?}: \
                 decoder={total_dec:.6e} brute_force={total_bf:.6e}"
            );
        }
    }
}
