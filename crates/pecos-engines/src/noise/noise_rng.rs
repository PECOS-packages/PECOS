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

//! Random number generator wrapper for noise models.
//!
//! This module provides a common interface for random number generation
//! in noise models through the `NoiseRng` wrapper.

use crate::Gate;
use pecos_rng::rng_ext::RngProbabilityExt;
use pecos_rng::{PecosRng, Rng, RngExt, SeedableRng};
use rand::prelude::Distribution;
use std::ops::Range;

/// Wrapper for random number generator used by noise models
///
/// Provides a common interface to random number generator functionality
/// for all noise models.
#[derive(Debug, Clone)]
pub struct NoiseRng<R: Rng + Clone = PecosRng> {
    rng: R,
}

impl<R: Rng + Clone> NoiseRng<R> {
    /// Create a new `NoiseRng` with the given RNG
    pub fn new(rng: R) -> Self {
        Self { rng }
    }

    /// Create a new `NoiseRng` with a seeded RNG
    #[must_use]
    pub fn with_seed(seed: u64) -> Self
    where
        R: SeedableRng,
    {
        Self {
            rng: R::seed_from_u64(seed),
        }
    }

    /// Generate a random float in the range [0, 1)
    pub fn random_float(&mut self) -> f64 {
        self.rng.random::<f64>()
    }

    /// Determines if an event occurs with the given probability
    ///
    /// # Arguments
    ///
    /// * `probability` - The probability of the event occurring, in the range [0, 1]
    ///
    /// # Returns
    ///
    /// `true` if the event occurs, `false` otherwise
    pub fn occurs(&mut self, probability: f64) -> bool {
        debug_assert!((0.0..=1.0).contains(&probability));
        let threshold = self.rng.probability_threshold(probability);
        self.rng.check_probability(threshold)
    }

    /// Generate a random integer in the given range
    pub fn random_int(&mut self, range: Range<usize>) -> usize {
        self.rng.random_range(range)
    }

    /// Sample a value from any distribution
    ///
    /// # Arguments
    ///
    /// * `distribution` - The distribution to sample from
    ///
    /// # Returns
    ///
    /// A value sampled from the distribution
    pub fn sample<T, D: Distribution<T>>(&mut self, distribution: &D) -> T {
        distribution.sample(&mut self.rng)
    }

    /// Sample from a weighted distribution
    ///
    /// # Arguments
    ///
    /// * `distribution` - The weighted distribution to sample from
    ///
    /// # Returns
    ///
    /// The index of the sampled item
    pub fn sample_from_distribution<D, T>(&mut self, distribution: &D) -> T
    where
        D: Distribution<T>,
    {
        self.sample(distribution)
    }

    /// Generate a random u32 in the given range
    pub fn random_u32(&mut self, range: Range<u32>) -> u32 {
        self.rng.random_range(range)
    }

    /// Get a reference to the inner RNG
    pub fn inner(&self) -> &R {
        &self.rng
    }

    /// Get a mutable reference to the inner RNG
    pub fn inner_mut(&mut self) -> &mut R {
        &mut self.rng
    }

    /// Generate a random Pauli gate (X, Y, Z) or none with equal probability
    ///
    /// # Arguments
    ///
    /// * `qubit` - The qubit to apply the Pauli gate to
    ///
    /// # Returns
    ///
    /// A `GateCommand` representing the Pauli operation, or `None` if no operation
    pub fn random_pauli_or_none(&mut self, qubit: usize) -> Option<Gate> {
        // Use optimized random_index_4 which efficiently selects from 0-3
        // 0: No operation (identity)
        // 1: X gate
        // 2: Y gate
        // 3: Z gate
        match self.rng.random_index_4() {
            0 => None,
            1 => Some(Gate::x(&[qubit])),
            2 => Some(Gate::y(&[qubit])),
            _ => Some(Gate::z(&[qubit])),
        }
    }
}

impl<R: Rng + Clone + SeedableRng> Default for NoiseRng<R> {
    fn default() -> Self {
        // Using make_rng() to seed the RNG from the OS
        Self {
            rng: rand::make_rng(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::distr::Uniform;
    use rand::distr::weighted::WeightedIndex;

    const SAMPLE_SIZE: usize = 100;
    // Epsilon for float comparisons
    const FLOAT_EPSILON: f64 = f64::EPSILON;

    // Helper function to compare floats with an epsilon
    fn float_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < FLOAT_EPSILON
    }

    #[test]
    fn test_noise_rng_random_float() {
        let mut rng = NoiseRng::<PecosRng>::with_seed(42);
        let value = rng.random_float();
        assert!((0.0..=1.0).contains(&value));

        // Test with multiple calls to ensure we get different values
        let values: Vec<f64> = (0..10).map(|_| rng.random_float()).collect();

        // Don't use a HashSet for floats, instead check that at least some values are different
        let mut all_same = true;
        for i in 1..values.len() {
            if (values[0] - values[i]).abs() > f64::EPSILON {
                all_same = false;
                break;
            }
        }
        assert!(!all_same, "Random values should vary");
    }

    #[test]
    fn test_noise_rng_occurs() {
        let mut rng = NoiseRng::<PecosRng>::with_seed(42);

        // With probability 0, should never occur
        for _ in 0..100 {
            assert!(!rng.occurs(0.0));
        }

        // With probability 1, should always occur
        for _ in 0..100 {
            assert!(rng.occurs(1.0));
        }

        // With probability 0.5, should occur roughly half the time
        let occurs_count = (0..1000).filter(|_| rng.occurs(0.5)).count();
        assert!(occurs_count > 400 && occurs_count < 600);
    }

    #[test]
    fn test_noise_rng_random_int() {
        let mut rng = NoiseRng::<PecosRng>::with_seed(42);

        // Test with a range of 0..3
        for _ in 0..100 {
            let value = rng.random_int(0..3);
            assert!(value < 3);
        }

        // Check distribution with a larger number of samples
        let counts = (0..1000)
            .map(|_| rng.random_int(0..3))
            .fold([0, 0, 0], |mut acc, val| {
                acc[val] += 1;
                acc
            });

        // Each value should appear roughly 1/3 of the time
        for count in &counts {
            assert!(*count > 250 && *count < 400);
        }
    }

    #[test]
    fn test_random_pauli_or_none() {
        let mut rng = NoiseRng::<PecosRng>::with_seed(42);

        // Count occurrences of each gate type
        let mut none_count = 0;
        let mut x_count = 0;
        let mut y_count = 0;
        let mut z_count = 0;

        // Generate a large number of samples to test distribution
        for _ in 0..1000 {
            match rng.random_pauli_or_none(0) {
                None => none_count += 1,
                Some(gate) => match gate.gate_type {
                    crate::byte_message::GateType::X => x_count += 1,
                    crate::byte_message::GateType::Y => y_count += 1,
                    crate::byte_message::GateType::Z => z_count += 1,
                    _ => panic!("Unexpected gate type: {:?}", gate.gate_type),
                },
            }
        }

        // Each outcome should occur roughly 1/4 of the time (250 times)
        // Allow a reasonable margin of error (±50)
        assert!(
            none_count > 200 && none_count < 300,
            "None count: {none_count}"
        );
        assert!(x_count > 200 && x_count < 300, "X count: {x_count}");
        assert!(y_count > 200 && y_count < 300, "Y count: {y_count}");
        assert!(z_count > 200 && z_count < 300, "Z count: {z_count}");
    }

    #[test]
    fn test_seed_determinism_basic() {
        // Test that the same seed produces the same sequence of random numbers
        let mut rng1 = NoiseRng::<PecosRng>::with_seed(42);
        let mut rng2 = NoiseRng::<PecosRng>::with_seed(42);

        for _ in 0..SAMPLE_SIZE {
            assert!(
                float_eq(rng1.random_float(), rng2.random_float()),
                "Random floats should be identical with same seed"
            );
        }
    }

    #[test]
    fn test_seed_determinism_multiple_seeds() {
        // Test multiple seed pairs to ensure determinism
        let seed_pairs = [(42, 42), (123, 123), (999, 999), (0, 0)];

        for (seed1, seed2) in seed_pairs {
            let mut rng1 = NoiseRng::<PecosRng>::with_seed(seed1);
            let mut rng2 = NoiseRng::<PecosRng>::with_seed(seed2);

            for _ in 0..SAMPLE_SIZE {
                assert!(
                    float_eq(rng1.random_float(), rng2.random_float()),
                    "Random floats should be identical with seed pair ({seed1}, {seed2})"
                );
            }
        }
    }

    #[test]
    fn test_seed_determinism_different_seeds() {
        // Test that different seeds produce different sequences
        let seed_pairs = [(42, 43), (123, 124), (999, 1000), (0, 1)];

        for (seed1, seed2) in seed_pairs {
            let mut rng1 = NoiseRng::<PecosRng>::with_seed(seed1);
            let mut rng2 = NoiseRng::<PecosRng>::with_seed(seed2);

            let mut found_difference = false;
            for _ in 0..SAMPLE_SIZE {
                if !float_eq(rng1.random_float(), rng2.random_float()) {
                    found_difference = true;
                    break;
                }
            }
            assert!(
                found_difference,
                "Different seeds ({seed1}, {seed2}) should produce different sequences"
            );
        }
    }

    #[test]
    fn test_seed_determinism_reset() {
        // Test that resetting with the same seed produces the same sequence
        let seed = 42;
        let mut rng = NoiseRng::<PecosRng>::with_seed(seed);

        // First sequence
        let results1: Vec<f64> = (0..SAMPLE_SIZE).map(|_| rng.random_float()).collect();

        // Reset and get second sequence
        rng = NoiseRng::<PecosRng>::with_seed(seed);
        let results2: Vec<f64> = (0..SAMPLE_SIZE).map(|_| rng.random_float()).collect();

        // Compare the floats with epsilon tolerance
        for i in 0..results1.len() {
            assert!(
                float_eq(results1[i], results2[i]),
                "Random sequences should be identical after reset with same seed"
            );
        }
    }

    #[test]
    fn test_seed_determinism_distribution() {
        // Test that the same seed produces the same sequence for different distributions
        let seed = 42;
        let mut rng1 = NoiseRng::<PecosRng>::with_seed(seed);
        let mut rng2 = NoiseRng::<PecosRng>::with_seed(seed);

        // Test uniform distribution
        let uniform = Uniform::new(0.0, 1.0).unwrap();
        for _ in 0..SAMPLE_SIZE {
            let sample1 = rng1.sample(&uniform);
            let sample2 = rng2.sample(&uniform);
            assert!(
                float_eq(sample1, sample2),
                "Uniform distribution samples should be identical with same seed"
            );
        }

        // Reset RNGs
        rng1 = NoiseRng::<PecosRng>::with_seed(seed);
        rng2 = NoiseRng::<PecosRng>::with_seed(seed);

        // Test weighted index distribution
        let weights = vec![0.3, 0.7];
        let weighted = WeightedIndex::new(&weights).unwrap();
        for _ in 0..SAMPLE_SIZE {
            assert_eq!(
                rng1.sample(&weighted),
                rng2.sample(&weighted),
                "Weighted distribution samples should be identical with same seed"
            );
        }
    }

    #[test]
    fn test_seed_determinism_interleaved() {
        // Test that interleaved operations maintain determinism
        let seed = 42;
        let mut rng1 = NoiseRng::<PecosRng>::with_seed(seed);
        let mut rng2 = NoiseRng::<PecosRng>::with_seed(seed);

        let uniform = Uniform::new(0.0, 1.0).unwrap();
        let weights = vec![0.3, 0.7];
        let weighted = WeightedIndex::new(&weights).unwrap();

        for _ in 0..SAMPLE_SIZE {
            // Interleave different types of random operations
            assert!(
                float_eq(rng1.random_float(), rng2.random_float()),
                "Random floats should be identical"
            );

            let sample1 = rng1.sample(&uniform);
            let sample2 = rng2.sample(&uniform);
            assert!(
                float_eq(sample1, sample2),
                "Uniform samples should be identical"
            );

            assert_eq!(
                rng1.sample(&weighted),
                rng2.sample(&weighted),
                "Weighted samples should be identical"
            );
        }
    }
}
