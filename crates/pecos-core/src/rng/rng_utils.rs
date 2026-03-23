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

use crate::rng::choices::Choices;
use pecos_random::rng_ext::RngProbabilityExt;
use rand::prelude::*;

/// A utility struct that provides methods for common random operations
///
/// This struct wraps a random number generator and provides methods for
/// common operations like coin flips, weighted choices, and generating
/// collections of random values.
pub struct RandomUtils<R: Rng + RngProbabilityExt> {
    rng: R,
}

impl<R: Rng + RngProbabilityExt> RandomUtils<R> {
    /// Create a new `RandomUtils` with the given random number generator
    ///
    /// # Arguments
    /// * `rng` - The random number generator to use
    #[inline]
    pub fn new(rng: R) -> Self {
        Self { rng }
    }

    /// Get a mutable reference to the internal RNG
    ///
    /// This is useful when you need to perform operations not directly
    /// provided by the `RandomUtils` methods.
    #[inline]
    pub fn rng_mut(&mut self) -> &mut R {
        &mut self.rng
    }

    /// Choose between options given weighted probabilities.
    ///
    /// # Arguments
    /// * `choices` - The choices to select from with their associated weights
    ///
    /// # Returns
    /// A reference to the selected item
    #[inline]
    pub fn choose_weighted<'a, T>(&mut self, choices: &'a Choices<T>) -> &'a T {
        choices.sample(&mut self.rng)
    }

    /// Simulates a fair coin flip with 50% probability of true/false
    ///
    /// # Returns
    /// `true` or `false` with equal probability
    #[inline]
    pub fn coin_flip(&mut self) -> bool {
        self.rng.coin_flip()
    }

    /// Simulates a biased coin flip with specified probability of true
    ///
    /// # Arguments
    /// * `p` - The probability of returning `true` (between 0.0 and 1.0)
    ///
    /// # Returns
    /// `true` with probability `p`, `false` with probability `1-p`
    #[inline]
    pub fn biased_coin_flip(&mut self, p: f64) -> bool {
        let threshold = self.rng.probability_threshold(p);
        self.rng.check_probability(threshold)
    }

    /// Generates a vector of bools, where true has an independent probability of `p`.
    ///
    /// # Arguments
    /// * `p` - The probability of each element being `true` (between 0.0 and 1.0)
    /// * `n` - The number of bools to generate
    ///
    /// # Returns
    /// A vector of `n` bools where each is independently `true` with probability `p`
    #[inline]
    pub fn gen_bools(&mut self, p: f64, n: usize) -> Vec<bool> {
        self.rng.gen_bools(p, n)
    }

    /// Select a random index based on a weighted probability distribution
    ///
    /// # Arguments
    /// * `weights` - A slice of weights (values should be non-negative)
    ///
    /// # Returns
    /// A randomly selected index where the probability of each index is proportional to its weight
    ///
    /// # Panics
    /// This function will panic if:
    /// - The weights slice is empty
    /// - All weights are zero
    /// - Any weight is negative
    #[inline]
    pub fn weighted_index(&mut self, weights: &[f64]) -> usize {
        assert!(!weights.is_empty(), "Cannot select from empty weights");

        let total: f64 = weights.iter().sum();
        assert!(total > 0.0, "Sum of weights must be positive");

        let mut target = self.rng.random_range(0.0..total);

        for (i, &weight) in weights.iter().enumerate() {
            assert!(weight >= 0.0, "Weights must be non-negative");
            target -= weight;
            if target <= 0.0 {
                return i;
            }
        }

        // This should never happen due to floating-point arithmetic
        weights.len() - 1
    }
}

// Keep backwards compatibility with the original functions
// by providing standalone versions that delegate to RngProbabilityExt

/// Choose between options given weighted probabilities.
#[inline]
pub fn choose_weighted<'a, T, R: Rng>(rng: &mut R, choices: &'a Choices<T>) -> &'a T {
    choices.sample(rng)
}

/// Gives true and false each with probability of 50%
///
/// Note: Consider using `RngProbabilityExt::coin_flip()` directly instead.
#[inline]
pub fn coin_flip<R: Rng + RngProbabilityExt>(rng: &mut R) -> bool {
    rng.coin_flip()
}

/// Generates a vector of bools, where true has an independent probability of `p`.
///
/// Note: Consider using `RngProbabilityExt::gen_bools()` directly instead.
#[inline]
pub fn gen_bools<R: Rng + RngProbabilityExt>(rng: &mut R, p: f64, n: usize) -> Vec<bool> {
    rng.gen_bools(p, n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn test_random_utils_struct() {
        // Create a seeded RNG for deterministic tests
        let rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let mut random_utils = RandomUtils::new(rng);

        // Test coin_flip
        let flips: Vec<bool> = (0..100).map(|_| random_utils.coin_flip()).collect();
        let true_count = flips.iter().filter(|&&b| b).count();
        // With a fair coin, we expect roughly 50 trues, but there's randomness
        assert!(true_count > 30 && true_count < 70);

        // Test biased_coin_flip
        let biased_flips: Vec<bool> = (0..100)
            .map(|_| random_utils.biased_coin_flip(0.7))
            .collect();
        let biased_true_count = biased_flips.iter().filter(|&&b| b).count();
        // With p=0.7, we expect roughly 70 trues, but there's randomness
        assert!(biased_true_count > 50 && biased_true_count < 90);

        // Test gen_bools
        let bools = random_utils.gen_bools(0.3, 50);
        assert_eq!(bools.len(), 50);
        let bools_true_count = bools.iter().filter(|&&b| b).count();
        // With p=0.3, we expect roughly 15 trues, but there's randomness
        assert!(bools_true_count > 5 && bools_true_count < 25);

        // Test weighted_index
        let weights = [1.0, 3.0, 6.0];
        let mut counts = [0, 0, 0];
        for _ in 0..1000 {
            counts[random_utils.weighted_index(&weights)] += 1;
        }
        // The middle value should be about 3x the first, and the last about 6x the first
        assert!(counts[1] > counts[0] * 2);
        assert!(counts[2] > counts[1]);
    }

    #[test]
    #[should_panic(expected = "Cannot select from empty weights")]
    fn test_weighted_index_empty() {
        let rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let mut random_utils = RandomUtils::new(rng);
        random_utils.weighted_index(&[]);
    }

    #[test]
    #[should_panic(expected = "Sum of weights must be positive")]
    fn test_weighted_index_all_zeros() {
        let rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let mut random_utils = RandomUtils::new(rng);
        random_utils.weighted_index(&[0.0, 0.0, 0.0]);
    }
}
