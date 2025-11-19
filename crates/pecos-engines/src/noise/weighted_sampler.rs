// Copyright 2025 The PECOS Developers
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

use std::collections::BTreeMap;

use crate::Gate;
use crate::noise::noise_rng::NoiseRng;
use crate::noise::utils::{SingleQubitNoiseResult, TwoQubitNoiseResult};
use rand::distr::weighted::WeightedIndex;

/// Tolerance for weight normalization - total weights should be within this amount of 1.0
const NORMALIZATION_TOLERANCE: f64 = 1e-5;
/// Small margin for floating-point equality comparisons
const FLOAT_EPSILON: f64 = 1e-10;

/// A sampler that selects keys with probability proportional to their weights
///
/// Uses `BTreeMap` for deterministic key ordering, ensuring consistent behavior
/// when using the same seed across multiple runs or threads.
#[derive(Debug, Clone)]
pub struct WeightedSampler<K: Clone + Ord> {
    keys: Vec<K>,
    distribution: WeightedIndex<f64>,
    weighted_map: BTreeMap<K, f64>,
}

impl<K: Clone + std::fmt::Debug + std::hash::Hash + Eq + Ord> WeightedSampler<K> {
    /// Create a new weighted sampler from a map of keys to weights
    ///
    /// The weights are normalized to sum to 1.0 with a default tolerance of 1e-10
    ///
    /// # Panics
    /// - If the weighted map is empty
    /// - If the total weight is not positive
    /// - If the total weight deviates from 1.0 by more than the tolerance
    /// - If the weighted index distribution cannot be created
    #[must_use]
    pub fn new(weighted_map: &BTreeMap<K, f64>) -> Self {
        Self::new_with_tolerance(weighted_map, NORMALIZATION_TOLERANCE)
    }

    /// Create a new weighted sampler with a specific tolerance for weight normalization
    ///
    /// # Panics
    /// - If the weighted map is empty
    /// - If the total weight is not positive
    /// - If the total weight deviates from 1.0 by more than the tolerance
    #[must_use]
    pub fn new_with_tolerance(weighted_map: &BTreeMap<K, f64>, tolerance: f64) -> Self {
        let (normalized_weighted_map, normalized_weights) =
            Self::validate_and_normalize(weighted_map, tolerance);

        // BTreeMap already provides deterministic ordering of keys
        let keys: Vec<K> = weighted_map.keys().cloned().collect();

        // Create the distribution using deterministically ordered weights
        let distribution = WeightedIndex::new(&normalized_weights)
            .expect("WeightedSampler: failed to create weighted distribution");

        WeightedSampler {
            keys,
            distribution,
            weighted_map: normalized_weighted_map,
        }
    }

    /// Validates that the weights are positive and approximately sum to 1.0
    /// Returns a normalized `BTreeMap` and a Vec of normalized weights for creating the distribution
    fn validate_and_normalize(
        weighted_map: &BTreeMap<K, f64>,
        tolerance: f64,
    ) -> (BTreeMap<K, f64>, Vec<f64>) {
        assert!(
            !weighted_map.is_empty(),
            "WeightedSampler: weighted_map cannot be empty"
        );

        let total_weight: f64 = weighted_map.values().sum();

        assert!(
            total_weight > 0.0,
            "WeightedSampler: total weight must be positive, got {total_weight}"
        );

        // Check if weights are within tolerance of 1.0
        assert!(
            (total_weight - 1.0).abs() <= tolerance, // Use <= instead of !( > )
            "WeightedSampler: total weight {total_weight} deviates from 1.0 by more than tolerance {tolerance}"
        );

        // Determine if we need to normalize (only normalize if not already very close to 1.0)
        let needs_normalization = (total_weight - 1.0).abs() > FLOAT_EPSILON;

        // Collect normalized weights for the distribution
        let normalized_weights: Vec<f64> = if needs_normalization {
            weighted_map.values().map(|&w| w / total_weight).collect()
        } else {
            weighted_map.values().copied().collect()
        };

        // Create normalized BTreeMap
        let mut normalized_map = BTreeMap::new();
        for (key, &value) in weighted_map {
            let normalized_value = if needs_normalization {
                value / total_weight
            } else {
                value
            };
            normalized_map.insert(key.clone(), normalized_value);
        }

        (normalized_map, normalized_weights)
    }

    /// Sample a key from the distribution
    ///
    /// # Panics
    /// - If the keys vector is empty (should never happen if constructed properly)
    /// - If the distribution sampling fails
    #[must_use]
    pub fn sample(&self, rng: &mut NoiseRng) -> K {
        let index = rng.sample(&self.distribution);
        self.keys[index].clone()
    }

    /// Get a reference to the normalized weighted map
    #[must_use]
    pub fn get_weighted_map(&self) -> &BTreeMap<K, f64> {
        &self.weighted_map
    }
}

/// Create a Pauli gate based on the Pauli operator character
/// Returns None for identity ('I') operations
fn create_pauli_gate(op: char, qubit: usize) -> Option<Gate> {
    match op {
        'X' => Some(Gate::x(&[qubit])),
        'Y' => Some(Gate::y(&[qubit])),
        'Z' => Some(Gate::z(&[qubit])),
        'I' => None, // Identity - no operation
        _ => panic!("Invalid Pauli operator '{op}'"),
    }
}

/// Samples single qubit noise operations (Pauli gates or leakage)
#[derive(Clone, Debug)]
pub struct SingleQubitWeightedSampler {
    sampler: WeightedSampler<String>,
}

impl SingleQubitWeightedSampler {
    /// Create a new single qubit sampler from a weighted map
    ///
    /// Valid keys are: "X", "Y", "Z", "L" (for leakage)
    ///
    /// # Panics
    /// - If the weighted map contains invalid keys
    /// - If the weighted map is empty
    /// - If the total weight is not positive
    /// - If the total weight deviates from 1.0 by more than the tolerance
    #[must_use]
    pub fn new(weighted_map: &BTreeMap<String, f64>) -> Self {
        Self::validate_pauli_leakage_keys(weighted_map);

        Self {
            sampler: WeightedSampler::new(weighted_map),
        }
    }

    fn validate_pauli_leakage_keys(weighted_map: &BTreeMap<String, f64>) {
        const VALID_KEYS: [&str; 4] = ["X", "Y", "Z", "L"];

        for key in weighted_map.keys() {
            assert!(
                VALID_KEYS.contains(&key.as_str()),
                "SingleQubitWeightedSampler: invalid key '{key}' - must be one of X, Y, Z, or L"
            );
        }
    }

    /// Get a reference to the normalized weighted map
    #[must_use]
    pub fn get_weighted_map(&self) -> &BTreeMap<String, f64> {
        self.sampler.get_weighted_map()
    }

    /// Sample a raw key from the distribution
    #[must_use]
    pub fn sample_keys(&self, rng: &mut NoiseRng) -> String {
        self.sampler.sample(rng)
    }

    /// Sample a gate operation for the given qubit
    ///
    /// # Panics
    /// - If the sampled key is invalid (this should never happen if the sampler was created properly)
    #[must_use]
    pub fn sample_gates(&self, rng: &mut NoiseRng, qubit: usize) -> SingleQubitNoiseResult {
        let key = self.sample_keys(rng);

        // Check for leakage first
        if key == "L" {
            return SingleQubitNoiseResult {
                gate: None,
                qubit_leaked: true,
            };
        }

        // For Pauli gates, create appropriate gate
        let gate = match key.as_str() {
            "X" => Gate::x(&[qubit]),
            "Y" => Gate::y(&[qubit]),
            "Z" => Gate::z(&[qubit]),
            _ => panic!(
                "SingleQubitWeightedSampler: invalid key '{key}' - must be one of \"X\", \"Y\", \"Z\", or \"L\""
            ),
        };

        SingleQubitNoiseResult {
            gate: Some(gate),
            qubit_leaked: false,
        }
    }
}

/// Samples two-qubit noise operations (pairs of Pauli gates or leakage)
#[derive(Clone, Debug)]
pub struct TwoQubitWeightedSampler {
    sampler: WeightedSampler<String>,
}

impl TwoQubitWeightedSampler {
    /// Create a new two-qubit sampler from a weighted map
    ///
    /// Valid keys are two-character strings where each character is one of:
    /// "X", "Y", "Z", "I" (identity), or "L" (leakage)
    /// Note: "II" is not allowed as it represents no operation
    ///
    /// # Panics
    /// - If the weighted map contains invalid keys
    /// - If the weighted map is empty
    /// - If the total weight is not positive
    /// - If the total weight deviates from 1.0 by more than the tolerance
    #[must_use]
    pub fn new(weighted_map: &BTreeMap<String, f64>) -> Self {
        Self::validate_two_qubit_keys(weighted_map);

        Self {
            sampler: WeightedSampler::new(weighted_map),
        }
    }

    fn validate_two_qubit_keys(weighted_map: &BTreeMap<String, f64>) {
        const VALID_CHARS: [char; 5] = ['X', 'Y', 'Z', 'I', 'L'];

        for key in weighted_map.keys() {
            // Key must be exactly 2 characters long
            assert_eq!(
                key.len(),
                2,
                "TwoQubitWeightedSampler: invalid key '{key}' - must be exactly 2 characters"
            );

            // Check each character is valid
            for c in key.chars() {
                assert!(
                    VALID_CHARS.contains(&c),
                    "TwoQubitWeightedSampler: invalid character '{c}' in key '{key}' - must be one of X, Y, Z, I, or L"
                );
            }

            // Special case: "II" is not allowed
            assert_ne!(
                key.as_str(),
                "II",
                "TwoQubitWeightedSampler: key 'II' is not allowed as it represents no operation"
            );
        }
    }

    /// Get a reference to the normalized weighted map
    #[must_use]
    pub fn get_weighted_map(&self) -> &BTreeMap<String, f64> {
        self.sampler.get_weighted_map()
    }

    /// Sample a raw key from the distribution
    #[must_use]
    pub fn sample_keys(&self, rng: &mut NoiseRng) -> String {
        self.sampler.sample(rng)
    }

    /// Sample gate operations for the given qubit pair
    ///
    /// # Panics
    /// - If the sampled key is invalid (this should never happen if the sampler was created properly)
    #[must_use]
    pub fn sample_gates(
        &self,
        rng: &mut NoiseRng,
        qubit0: usize,
        qubit1: usize,
    ) -> TwoQubitNoiseResult {
        // Sample a key and extract the characters
        let key_str = self.sample_keys(rng);
        let chars: Vec<char> = key_str.chars().collect();

        // Determine leakage status
        let qubit0_leaked = chars[0] == 'L';
        let qubit1_leaked = chars[1] == 'L';

        // If both qubits leaked, no gates needed
        if qubit0_leaked && qubit1_leaked {
            return TwoQubitNoiseResult::with_leakage(true, true, None);
        }

        // Build gates for non-leaked qubits only
        let mut gates = Vec::new();

        // Convert the first operation if not leaked
        if !qubit0_leaked && let Some(gate) = create_pauli_gate(chars[0], qubit0) {
            gates.push(gate);
        }

        // Convert the second operation if not leaked
        if !qubit1_leaked && let Some(gate) = create_pauli_gate(chars[1], qubit1) {
            gates.push(gate);
        }

        // Only return gates if we have some
        let gates_option = if gates.is_empty() { None } else { Some(gates) };

        TwoQubitNoiseResult::with_leakage(qubit0_leaked, qubit1_leaked, gates_option)
    }
}

/// Samples crosstalk noise transitions
#[derive(Clone, Debug)]
pub struct CrosstalkWeightedSampler {
    sampler_from_0: WeightedSampler<String>,
    sampler_from_1: WeightedSampler<String>,
}

impl CrosstalkWeightedSampler {
    /// Create a new crosstalk sampler from a weighted map
    ///
    /// Valid keys are: "0->0", "0->1", "0->L", "1->1", "1->0", "1->L"
    ///
    /// # Panics
    /// - If the weighted map contains invalid keys
    /// - If the weighted map is empty
    /// - If the total weight of each sampler is not positive
    /// - If the total weight of each sampler deviates from 1.0 by more than the tolerance
    #[must_use]
    pub fn new(weighted_map: &BTreeMap<String, f64>) -> Self {
        const KEYS_FROM_0: [&str; 3] = ["0->0", "0->1", "0->L"];
        const KEYS_FROM_1: [&str; 3] = ["1->1", "1->0", "1->L"];
        Self::validate_crosstalk_keys(weighted_map);

        // Separate the 0->* components from the 1->* components
        let weighted_map_from_0 = KEYS_FROM_0
            .into_iter()
            .filter_map(|key| weighted_map.get(key).map(|&val| (key.to_string(), val)))
            .collect();
        let weighted_map_from_1 = KEYS_FROM_1
            .into_iter()
            .filter_map(|key| weighted_map.get(key).map(|&val| (key.to_string(), val)))
            .collect();

        Self {
            sampler_from_0: WeightedSampler::new(&weighted_map_from_0),
            sampler_from_1: WeightedSampler::new(&weighted_map_from_1),
        }
    }

    fn validate_crosstalk_keys(weighted_map: &BTreeMap<String, f64>) {
        const VALID_KEYS: [&str; 6] = ["0->0", "0->1", "0->L", "1->1", "1->0", "1->L"];

        for key in weighted_map.keys() {
            assert!(
                VALID_KEYS.contains(&key.as_str()),
                "CrosstalkWeightedSampler: invalid key '{key}' - must be one of {VALID_KEYS:?}",
            );
        }
    }

    /// Get a reference to the normalized weighted map, for keys 0->* or 1->*
    /// # Panics
    /// - If `from_state` is not either 0 or 1.
    #[must_use]
    pub fn get_weighted_map(&self, from_state: u32) -> &BTreeMap<String, f64> {
        assert!(from_state == 0 || from_state == 1);
        if from_state == 0 {
            self.sampler_from_0.get_weighted_map()
        } else {
            self.sampler_from_1.get_weighted_map()
        }
    }

    /// Sample a raw key from the distribution, for keys 0->* or 1->*.
    /// # Panics
    /// - If `from_state` is not either 0 or 1.
    #[must_use]
    pub fn sample_keys(&self, rng: &mut NoiseRng, from_state: u32) -> String {
        assert!(from_state == 0 || from_state == 1);
        if from_state == 0 {
            self.sampler_from_0.sample(rng)
        } else {
            self.sampler_from_1.sample(rng)
        }
    }

    /// Sample a gate operation for the given qubit
    ///
    /// # Panics
    /// - If the sampled key is invalid (this should never happen if the sampler was created properly)
    #[must_use]
    pub fn sample_gates(
        &self,
        rng: &mut NoiseRng,
        qubit: usize,
        from_state: u32,
    ) -> SingleQubitNoiseResult {
        let key = self.sample_keys(rng, from_state);

        match key.as_str() {
            "0->0" | "1->1" => SingleQubitNoiseResult {
                gate: None,
                qubit_leaked: false,
            },
            "0->1" | "1->0" => SingleQubitNoiseResult {
                gate: Some(Gate::x(&[qubit])),
                qubit_leaked: false,
            },
            "0->L" | "1->L" => SingleQubitNoiseResult {
                gate: None,
                qubit_leaked: true,
            },
            _ => panic!("CrosstalkWeightedSampler: invalid key '{key}'"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::noise::noise_rng::NoiseRng;
    use rand_chacha::ChaCha8Rng;

    const SAMPLE_SIZE: usize = 100;

    #[test]
    fn test_different_sampler_instances_same_results() {
        // Create two weighted samplers with the same weights
        let mut weights1 = BTreeMap::new();
        weights1.insert("A".to_string(), 0.3);
        weights1.insert("B".to_string(), 0.7);

        // Make a separate instance with the same data
        let mut weights2 = BTreeMap::new();
        weights2.insert("A".to_string(), 0.3);
        weights2.insert("B".to_string(), 0.7);

        let sampler1 = WeightedSampler::new(&weights1);
        let sampler2 = WeightedSampler::new(&weights2);

        // Use the same seed for both RNGs
        let mut rng1 = NoiseRng::<ChaCha8Rng>::with_seed(42);
        let mut rng2 = NoiseRng::<ChaCha8Rng>::with_seed(42);

        // Sample from both samplers
        let results1: Vec<String> = (0..SAMPLE_SIZE)
            .map(|_| sampler1.sample(&mut rng1))
            .collect();
        let results2: Vec<String> = (0..SAMPLE_SIZE)
            .map(|_| sampler2.sample(&mut rng2))
            .collect();

        // Results should be identical with same seed
        assert_eq!(
            results1, results2,
            "Different sampler instances with same weights should produce identical results with same seed"
        );
    }

    #[test]
    fn test_deterministic_ordering_with_shuffled_keys() {
        // Create two weighted samplers with the same weights but different insertion order
        let mut weights1 = BTreeMap::new();
        weights1.insert("A".to_string(), 0.3);
        weights1.insert("B".to_string(), 0.2);
        weights1.insert("C".to_string(), 0.5);

        // Insert in different order
        let mut weights2 = BTreeMap::new();
        weights2.insert("C".to_string(), 0.5);
        weights2.insert("A".to_string(), 0.3);
        weights2.insert("B".to_string(), 0.2);

        let sampler1 = WeightedSampler::new(&weights1);
        let sampler2 = WeightedSampler::new(&weights2);

        // Use the same seed for both RNGs
        let mut rng1 = NoiseRng::<ChaCha8Rng>::with_seed(42);
        let mut rng2 = NoiseRng::<ChaCha8Rng>::with_seed(42);

        // Sample from both samplers
        let results1: Vec<String> = (0..SAMPLE_SIZE)
            .map(|_| sampler1.sample(&mut rng1))
            .collect();
        let results2: Vec<String> = (0..SAMPLE_SIZE)
            .map(|_| sampler2.sample(&mut rng2))
            .collect();

        // Results should be identical despite different insertion order
        assert_eq!(
            results1, results2,
            "Samplers with differently ordered but equivalent maps should produce identical results"
        );
    }

    #[test]
    fn test_deterministic_sampling_basic() {
        // Test basic deterministic sampling with same seed
        let mut weights = BTreeMap::new();
        weights.insert("A".to_string(), 0.3);
        weights.insert("B".to_string(), 0.7);

        let sampler = WeightedSampler::new(&weights);

        // Create two RNGs with the same seed
        let mut rng1 = NoiseRng::<ChaCha8Rng>::with_seed(42);
        let mut rng2 = NoiseRng::<ChaCha8Rng>::with_seed(42);

        // Sample from both RNGs
        let results1: Vec<String> = (0..SAMPLE_SIZE)
            .map(|_| sampler.sample(&mut rng1))
            .collect();
        let results2: Vec<String> = (0..SAMPLE_SIZE)
            .map(|_| sampler.sample(&mut rng2))
            .collect();

        // Verify exact sequence match
        assert_eq!(
            results1, results2,
            "Sampling results should be identical with same seed"
        );
    }

    #[test]
    fn test_deterministic_sampling_multiple_seeds() {
        // Test deterministic sampling with multiple different seeds
        let mut weights = BTreeMap::new();
        weights.insert("A".to_string(), 0.3);
        weights.insert("B".to_string(), 0.7);

        let sampler = WeightedSampler::new(&weights);

        // Test multiple seed pairs
        let seed_pairs = [(42, 42), (123, 123), (999, 999), (0, 0)];

        for (seed1, seed2) in seed_pairs {
            let mut rng1 = NoiseRng::<ChaCha8Rng>::with_seed(seed1);
            let mut rng2 = NoiseRng::<ChaCha8Rng>::with_seed(seed2);

            let results1: Vec<String> = (0..SAMPLE_SIZE)
                .map(|_| sampler.sample(&mut rng1))
                .collect();
            let results2: Vec<String> = (0..SAMPLE_SIZE)
                .map(|_| sampler.sample(&mut rng2))
                .collect();

            assert_eq!(
                results1, results2,
                "Sampling results should be identical with same seed pair ({seed1}, {seed2})"
            );
        }
    }

    #[test]
    fn test_deterministic_sampling_different_seeds() {
        // Test that different seeds produce different sequences
        let mut weights = BTreeMap::new();
        weights.insert("A".to_string(), 0.3);
        weights.insert("B".to_string(), 0.7);

        let sampler = WeightedSampler::new(&weights);

        // Test multiple different seed pairs
        let seed_pairs = [(42, 43), (123, 124), (999, 1000), (0, 1)];

        for (seed1, seed2) in seed_pairs {
            let mut rng1 = NoiseRng::<ChaCha8Rng>::with_seed(seed1);
            let mut rng2 = NoiseRng::<ChaCha8Rng>::with_seed(seed2);

            let results1: Vec<String> = (0..SAMPLE_SIZE)
                .map(|_| sampler.sample(&mut rng1))
                .collect();
            let results2: Vec<String> = (0..SAMPLE_SIZE)
                .map(|_| sampler.sample(&mut rng2))
                .collect();

            assert_ne!(
                results1, results2,
                "Sampling results should differ with different seed pair ({seed1}, {seed2})"
            );
        }
    }

    #[test]
    fn test_deterministic_sampling_single_qubit() {
        // Test deterministic sampling with single qubit sampler
        let mut weights = BTreeMap::new();
        weights.insert("X".to_string(), 0.25);
        weights.insert("Y".to_string(), 0.25);
        weights.insert("Z".to_string(), 0.25);
        weights.insert("L".to_string(), 0.25);

        let sampler = SingleQubitWeightedSampler::new(&weights);

        // Create two RNGs with the same seed
        let mut rng1 = NoiseRng::<ChaCha8Rng>::with_seed(42);
        let mut rng2 = NoiseRng::<ChaCha8Rng>::with_seed(42);

        // Sample from both RNGs
        let results1: Vec<SingleQubitNoiseResult> = (0..SAMPLE_SIZE)
            .map(|_| sampler.sample_gates(&mut rng1, 0))
            .collect();
        let results2: Vec<SingleQubitNoiseResult> = (0..SAMPLE_SIZE)
            .map(|_| sampler.sample_gates(&mut rng2, 0))
            .collect();

        // Verify exact sequence match
        for (i, (r1, r2)) in results1.iter().zip(results2.iter()).enumerate() {
            assert_eq!(
                r1.qubit_leaked, r2.qubit_leaked,
                "Leakage mismatch at index {i}"
            );
            match (&r1.gate, &r2.gate) {
                (Some(g1), Some(g2)) => assert_eq!(
                    g1.gate_type, g2.gate_type,
                    "Gate type mismatch at index {i}"
                ),
                (None, None) => (),
                _ => panic!("Gate presence mismatch at index {i}"),
            }
        }
    }

    #[test]
    fn test_deterministic_sampling_two_qubit() {
        // Test deterministic sampling with two qubit sampler
        let mut weights = BTreeMap::new();
        weights.insert("XX".to_string(), 0.2);
        weights.insert("YY".to_string(), 0.2);
        weights.insert("ZZ".to_string(), 0.2);
        weights.insert("XL".to_string(), 0.2);
        weights.insert("LX".to_string(), 0.2);

        let sampler = TwoQubitWeightedSampler::new(&weights);

        // Create two RNGs with the same seed
        let mut rng1 = NoiseRng::<ChaCha8Rng>::with_seed(42);
        let mut rng2 = NoiseRng::<ChaCha8Rng>::with_seed(42);

        // Sample from both RNGs
        let results1: Vec<TwoQubitNoiseResult> = (0..SAMPLE_SIZE)
            .map(|_| sampler.sample_gates(&mut rng1, 0, 1))
            .collect();
        let results2: Vec<TwoQubitNoiseResult> = (0..SAMPLE_SIZE)
            .map(|_| sampler.sample_gates(&mut rng2, 0, 1))
            .collect();

        // Verify exact sequence match
        for (i, (r1, r2)) in results1.iter().zip(results2.iter()).enumerate() {
            assert_eq!(
                r1.qubit0_leaked, r2.qubit0_leaked,
                "Qubit 0 leakage mismatch at index {i}"
            );
            assert_eq!(
                r1.qubit1_leaked, r2.qubit1_leaked,
                "Qubit 1 leakage mismatch at index {i}"
            );
            match (&r1.gates, &r2.gates) {
                (Some(g1), Some(g2)) => {
                    assert_eq!(g1.len(), g2.len(), "Gate count mismatch at index {i}");
                    for (j, (gate1, gate2)) in g1.iter().zip(g2.iter()).enumerate() {
                        assert_eq!(
                            gate1.gate_type, gate2.gate_type,
                            "Gate type mismatch at index {i} for gate {j}"
                        );
                    }
                }
                (None, None) => (),
                _ => panic!("Gate presence mismatch at index {i}"),
            }
        }
    }

    #[test]
    fn test_deterministic_sampling_crosstalk() {
        // Test deterministic sampling with single qubit sampler
        let mut weights = BTreeMap::new();
        weights.insert("0->1".to_string(), 0.5);
        weights.insert("0->L".to_string(), 0.5);
        weights.insert("1->0".to_string(), 0.5);
        weights.insert("1->L".to_string(), 0.5);

        let sampler = CrosstalkWeightedSampler::new(&weights);

        // Create two RNGs with the same seed
        let mut rng1 = NoiseRng::<ChaCha8Rng>::with_seed(42);
        let mut rng2 = NoiseRng::<ChaCha8Rng>::with_seed(42);

        // Sample from both RNGs
        let results1: Vec<SingleQubitNoiseResult> = (0..SAMPLE_SIZE)
            .map(|_| sampler.sample_gates(&mut rng1, 0, 0))
            .collect();
        let results2: Vec<SingleQubitNoiseResult> = (0..SAMPLE_SIZE)
            .map(|_| sampler.sample_gates(&mut rng2, 0, 0))
            .collect();

        // Verify exact sequence match
        for (i, (r1, r2)) in results1.iter().zip(results2.iter()).enumerate() {
            assert_eq!(
                r1.qubit_leaked, r2.qubit_leaked,
                "Leakage mismatch at index {i}"
            );
            match (&r1.gate, &r2.gate) {
                (Some(g1), Some(g2)) => assert_eq!(
                    g1.gate_type, g2.gate_type,
                    "Gate type mismatch at index {i}"
                ),
                (None, None) => (),
                _ => panic!("Gate presence mismatch at index {i}"),
            }
        }
    }

    #[test]
    fn test_deterministic_sampling_reset() {
        // Test that resetting the RNG and using the same seed produces the same sequence
        let mut weights = BTreeMap::new();
        weights.insert("A".to_string(), 0.3);
        weights.insert("B".to_string(), 0.7);

        let sampler = WeightedSampler::new(&weights);
        let seed = 42;

        // First sequence
        let mut rng = NoiseRng::<ChaCha8Rng>::with_seed(seed);
        let results1: Vec<String> = (0..SAMPLE_SIZE).map(|_| sampler.sample(&mut rng)).collect();

        // Reset RNG with same seed
        rng = NoiseRng::<ChaCha8Rng>::with_seed(seed);
        let results2: Vec<String> = (0..SAMPLE_SIZE).map(|_| sampler.sample(&mut rng)).collect();

        // Verify exact sequence match
        assert_eq!(
            results1, results2,
            "Sampling results should be identical after RNG reset with same seed"
        );
    }

    #[test]
    fn test_deterministic_sampling_consecutive() {
        // Test that consecutive samples from the same RNG are deterministic
        let mut weights = BTreeMap::new();
        weights.insert("A".to_string(), 0.3);
        weights.insert("B".to_string(), 0.7);

        let sampler = WeightedSampler::new(&weights);
        let mut rng = NoiseRng::<ChaCha8Rng>::with_seed(42);

        // Take two consecutive samples
        let result1 = sampler.sample(&mut rng);
        let result2 = sampler.sample(&mut rng);

        // Reset RNG and take the same two samples
        rng = NoiseRng::<ChaCha8Rng>::with_seed(42);
        let result3 = sampler.sample(&mut rng);
        let result4 = sampler.sample(&mut rng);

        // Verify the sequences match
        assert_eq!(result1, result3, "First sample should be deterministic");
        assert_eq!(result2, result4, "Second sample should be deterministic");
    }

    #[test]
    fn test_deterministic_sampling_interleaved() {
        // Test that interleaved sampling from different samplers is deterministic
        let mut weights1 = BTreeMap::new();
        weights1.insert("A".to_string(), 0.3);
        weights1.insert("B".to_string(), 0.7);

        let mut weights2 = BTreeMap::new();
        weights2.insert("X".to_string(), 0.4);
        weights2.insert("Y".to_string(), 0.6);

        let sampler1 = WeightedSampler::new(&weights1);
        let sampler2 = WeightedSampler::new(&weights2);

        let mut rng1 = NoiseRng::<ChaCha8Rng>::with_seed(42);
        let mut rng2 = NoiseRng::<ChaCha8Rng>::with_seed(42);

        // Interleaved sampling
        let results1: Vec<String> = (0..SAMPLE_SIZE)
            .map(|_| {
                if rng1.random_float() < 0.5 {
                    sampler1.sample(&mut rng1)
                } else {
                    sampler2.sample(&mut rng2)
                }
            })
            .collect();

        // Reset RNGs and repeat
        rng1 = NoiseRng::<ChaCha8Rng>::with_seed(42);
        rng2 = NoiseRng::<ChaCha8Rng>::with_seed(42);

        let results2: Vec<String> = (0..SAMPLE_SIZE)
            .map(|_| {
                if rng1.random_float() < 0.5 {
                    sampler1.sample(&mut rng1)
                } else {
                    sampler2.sample(&mut rng2)
                }
            })
            .collect();

        assert_eq!(
            results1, results2,
            "Interleaved sampling should be deterministic"
        );
    }

    #[test]
    fn test_deterministic_sampling_edge_cases() {
        // Test edge cases for sampling
        let mut weights = BTreeMap::new();
        weights.insert("A".to_string(), 1.0); // Single outcome with probability 1.0

        let sampler = WeightedSampler::new(&weights);
        let mut rng1 = NoiseRng::<ChaCha8Rng>::with_seed(42);
        let mut rng2 = NoiseRng::<ChaCha8Rng>::with_seed(42);

        // Should always get "A" regardless of RNG state
        let results1: Vec<String> = (0..SAMPLE_SIZE)
            .map(|_| sampler.sample(&mut rng1))
            .collect();
        let results2: Vec<String> = (0..SAMPLE_SIZE)
            .map(|_| sampler.sample(&mut rng2))
            .collect();

        assert_eq!(
            results1, results2,
            "Sampling should be deterministic even with single outcome"
        );
        assert!(
            results1.iter().all(|x| x == "A"),
            "All results should be 'A'"
        );
    }

    #[test]
    fn test_deterministic_sampling_single_qubit_edge_cases() {
        // Test edge cases for single qubit sampling
        let mut weights = BTreeMap::new();
        weights.insert("L".to_string(), 1.0); // Always leak

        let sampler = SingleQubitWeightedSampler::new(&weights);
        let mut rng1 = NoiseRng::<ChaCha8Rng>::with_seed(42);
        let mut rng2 = NoiseRng::<ChaCha8Rng>::with_seed(42);

        let results1: Vec<SingleQubitNoiseResult> = (0..SAMPLE_SIZE)
            .map(|_| sampler.sample_gates(&mut rng1, 0))
            .collect();
        let results2: Vec<SingleQubitNoiseResult> = (0..SAMPLE_SIZE)
            .map(|_| sampler.sample_gates(&mut rng2, 0))
            .collect();

        // Verify exact sequence match
        for (i, (r1, r2)) in results1.iter().zip(results2.iter()).enumerate() {
            assert_eq!(
                r1.qubit_leaked, r2.qubit_leaked,
                "Leakage mismatch at index {i}"
            );
            assert!(r1.qubit_leaked, "All results should indicate leakage");
            assert!(r1.gate.is_none(), "No gates should be present");
        }
    }

    #[test]
    fn test_deterministic_sampling_two_qubit_edge_cases() {
        // Test edge cases for two qubit sampling
        let mut weights = BTreeMap::new();
        weights.insert("LL".to_string(), 1.0); // Always leak both qubits

        let sampler = TwoQubitWeightedSampler::new(&weights);
        let mut rng1 = NoiseRng::<ChaCha8Rng>::with_seed(42);
        let mut rng2 = NoiseRng::<ChaCha8Rng>::with_seed(42);

        let results1: Vec<TwoQubitNoiseResult> = (0..SAMPLE_SIZE)
            .map(|_| sampler.sample_gates(&mut rng1, 0, 1))
            .collect();
        let results2: Vec<TwoQubitNoiseResult> = (0..SAMPLE_SIZE)
            .map(|_| sampler.sample_gates(&mut rng2, 0, 1))
            .collect();

        // Verify exact sequence match
        for (i, (r1, r2)) in results1.iter().zip(results2.iter()).enumerate() {
            assert_eq!(
                r1.qubit0_leaked, r2.qubit0_leaked,
                "Qubit 0 leakage mismatch at index {i}"
            );
            assert_eq!(
                r1.qubit1_leaked, r2.qubit1_leaked,
                "Qubit 1 leakage mismatch at index {i}"
            );
            assert!(
                r1.qubit0_leaked && r1.qubit1_leaked,
                "Both qubits should leak"
            );
            assert!(r1.gates.is_none(), "No gates should be present");
        }
    }

    #[test]
    fn test_deterministic_sampling_crosstalk_edge_cases() {
        // Test edge cases for single qubit sampling
        let mut weights = BTreeMap::new();
        weights.insert("0->L".to_string(), 1.0); // Always leak
        weights.insert("1->L".to_string(), 1.0); // Always leak

        let sampler = CrosstalkWeightedSampler::new(&weights);
        let mut rng1 = NoiseRng::<ChaCha8Rng>::with_seed(42);
        let mut rng2 = NoiseRng::<ChaCha8Rng>::with_seed(42);

        let results1: Vec<SingleQubitNoiseResult> = (0..SAMPLE_SIZE)
            .map(|_| sampler.sample_gates(&mut rng1, 0, 1))
            .collect();
        let results2: Vec<SingleQubitNoiseResult> = (0..SAMPLE_SIZE)
            .map(|_| sampler.sample_gates(&mut rng2, 0, 1))
            .collect();

        // Verify exact sequence match
        for (i, (r1, r2)) in results1.iter().zip(results2.iter()).enumerate() {
            assert_eq!(
                r1.qubit_leaked, r2.qubit_leaked,
                "Leakage mismatch at index {i}"
            );
            assert!(r1.qubit_leaked, "All results should indicate leakage");
            assert!(r1.gate.is_none(), "No gates should be present");
        }
    }
}
