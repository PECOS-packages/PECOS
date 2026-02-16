// Copyright 2025 The PECOS Developers
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

//! Extension trait for probability-based RNG operations.
//!
//! This module provides [`RngProbabilityExt`], a trait that adds fixed-point
//! probability checking methods to any RNG implementing [`Rng`].
//!
//! # Usage
//!
//! ```
//! use pecos_rng::{PecosRng, SeedableRng};
//! use pecos_rng::rng_ext::RngProbabilityExt;
//!
//! let mut rng = PecosRng::seed_from_u64(42);
//!
//! // Precompute threshold for 0.1% error rate
//! let threshold = rng.probability_threshold(0.001);
//!
//! // Check if event occurs
//! if rng.check_probability(threshold) {
//!     println!("Error occurred!");
//! }
//!
//! // Count occurrences over many trials
//! let errors = rng.count_occurrences(threshold, 10_000);
//! ```
//!
//! # Performance
//!
//! The default implementations work with any [`Rng`] type. However,
//! [`PecosRng`](crate::PecosRng) provides optimized implementations that use
//! SIMD to process multiple values at once, achieving ~20% better performance.

use rand_core::Rng;

/// Extension trait providing fixed-point probability operations for RNGs.
///
/// This trait provides methods for efficient probability checking using
/// precomputed u64 thresholds instead of f64 comparisons. This is faster
/// for repeated probability checks with the same probability value.
///
/// # Default Implementations
///
/// All methods have default implementations that work with any [`Rng`].
/// RNG implementations can override these with optimized versions.
///
/// # Example
///
/// ```
/// use pecos_rng::{PecosRng, SeedableRng};
/// use pecos_rng::rng_ext::RngProbabilityExt;
///
/// let mut rng = PecosRng::seed_from_u64(42);
/// let threshold = rng.probability_threshold(0.01);
///
/// // Efficient probability checking
/// for _ in 0..1000 {
///     if rng.check_probability(threshold) {
///         // Handle event
///     }
/// }
/// ```
pub trait RngProbabilityExt: Rng {
    /// Convert a probability to a u64 threshold for use with [`check_probability`](Self::check_probability).
    ///
    /// This allows precomputing the threshold once and reusing it for many
    /// probability checks, avoiding f64 operations on each check.
    ///
    /// # Arguments
    ///
    /// * `p` - Probability value in the range [0.0, 1.0]
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_rng::{PecosRng, SeedableRng};
    /// use pecos_rng::rng_ext::RngProbabilityExt;
    ///
    /// let mut rng = PecosRng::seed_from_u64(42);
    /// let threshold = rng.probability_threshold(0.001);
    /// ```
    #[inline]
    #[must_use]
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    fn probability_threshold(&self, p: f64) -> u64 {
        (p * (u64::MAX as f64)) as u64
    }

    /// Check if a random event occurs with the given precomputed probability threshold.
    ///
    /// This is faster than comparing `random::<f64>() < p` because it avoids
    /// f64 conversion on each call. The threshold should be computed once
    /// using [`probability_threshold`](Self::probability_threshold).
    ///
    /// # Arguments
    ///
    /// * `threshold` - Precomputed threshold from [`probability_threshold`](Self::probability_threshold)
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_rng::{PecosRng, SeedableRng};
    /// use pecos_rng::rng_ext::RngProbabilityExt;
    ///
    /// let mut rng = PecosRng::seed_from_u64(42);
    /// let threshold = rng.probability_threshold(0.1);
    ///
    /// if rng.check_probability(threshold) {
    ///     println!("Event occurred!");
    /// }
    /// ```
    #[inline]
    fn check_probability(&mut self, threshold: u64) -> bool {
        self.next_u64() < threshold
    }

    /// Count how many events occur out of `count` checks with the given probability threshold.
    ///
    /// This is useful for noise models that need to know how many errors occurred
    /// without storing individual results.
    ///
    /// # Arguments
    ///
    /// * `threshold` - Precomputed threshold from [`probability_threshold`](Self::probability_threshold)
    /// * `count` - Number of probability checks to perform
    ///
    /// # Returns
    ///
    /// The number of events that occurred (where random value < threshold).
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_rng::{PecosRng, SeedableRng};
    /// use pecos_rng::rng_ext::RngProbabilityExt;
    ///
    /// let mut rng = PecosRng::seed_from_u64(42);
    /// let threshold = rng.probability_threshold(0.001);
    ///
    /// let error_count = rng.count_occurrences(threshold, 10_000);
    /// println!("Errors: {error_count}");
    /// ```
    #[inline]
    fn count_occurrences(&mut self, threshold: u64, count: usize) -> usize {
        let mut total = 0usize;
        for _ in 0..count {
            if self.next_u64() < threshold {
                total += 1;
            }
        }
        total
    }

    /// Return indices where probability check succeeded (for sparse error application).
    ///
    /// This is optimized for the common noise model pattern where:
    /// - Many items need probability checks (e.g., all gates in a circuit)
    /// - The probability is low (e.g., 0.1% error rate)
    /// - You need to apply something only at the indices where events occurred
    ///
    /// Instead of checking each item individually with branching, this method
    /// returns a sparse list of indices where the check succeeded.
    ///
    /// # Arguments
    ///
    /// * `threshold` - Precomputed threshold from [`probability_threshold`](Self::probability_threshold)
    /// * `count` - Number of items to check
    ///
    /// # Returns
    ///
    /// A vector of indices (0..count) where the random value was less than threshold.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_rng::{PecosRng, SeedableRng};
    /// use pecos_rng::rng_ext::RngProbabilityExt;
    ///
    /// let mut rng = PecosRng::seed_from_u64(42);
    /// let threshold = rng.probability_threshold(0.001); // 0.1% error rate
    ///
    /// // Check 10,000 gates, get back ~10 error indices
    /// let error_indices = rng.check_probability_indices(threshold, 10_000);
    ///
    /// for idx in error_indices {
    ///     println!("Error at gate {}", idx);
    /// }
    /// ```
    fn check_probability_indices(&mut self, threshold: u64, count: usize) -> Vec<usize> {
        // Pre-allocate based on expected number of hits
        // For p=0.001 and count=10000, expect ~10 hits
        // Use 2x expected to avoid reallocation in most cases
        #[allow(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss
        )]
        let expected_hits = ((count as f64) * (threshold as f64 / u64::MAX as f64) * 2.0) as usize;
        let mut indices = Vec::with_capacity(expected_hits.max(16));

        for i in 0..count {
            if self.next_u64() < threshold {
                indices.push(i);
            }
        }

        indices
    }

    // ========================================================================
    // Optimized index selection for noise models
    // ========================================================================

    /// Generate a random index in the range 0..3 (for single-qubit Pauli selection).
    ///
    /// This is optimized for the common noise model pattern of selecting X, Y, or Z.
    /// Uses the multiply-shift method for unbiased selection without rejection sampling.
    ///
    /// # Returns
    ///
    /// A value in 0, 1, or 2 with equal probability.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_rng::{PecosRng, SeedableRng};
    /// use pecos_rng::rng_ext::RngProbabilityExt;
    ///
    /// let mut rng = PecosRng::seed_from_u64(42);
    /// let pauli = match rng.random_index_3() {
    ///     0 => "X",
    ///     1 => "Y",
    ///     _ => "Z",
    /// };
    /// ```
    #[inline]
    #[allow(clippy::cast_possible_truncation)]
    fn random_index_3(&mut self) -> u8 {
        // Use multiply-shift method for unbiased selection:
        // Multiply by 3 and take top bits to divide [0, 2^64) into 3 equal parts
        ((u128::from(self.next_u64()) * 3) >> 64) as u8
    }

    /// Generate a random index in the range 0..4 (for Pauli + identity selection).
    ///
    /// This is optimized for patterns like `random_pauli_or_none` which selects
    /// from I, X, Y, Z with equal probability.
    ///
    /// # Returns
    ///
    /// A value in 0, 1, 2, or 3 with equal probability.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_rng::{PecosRng, SeedableRng};
    /// use pecos_rng::rng_ext::RngProbabilityExt;
    ///
    /// let mut rng = PecosRng::seed_from_u64(42);
    /// let gate = match rng.random_index_4() {
    ///     0 => None,           // Identity
    ///     1 => Some("X"),
    ///     2 => Some("Y"),
    ///     _ => Some("Z"),
    /// };
    /// ```
    #[inline]
    fn random_index_4(&mut self) -> u8 {
        // Exactly 2 bits needed, no rejection required
        (self.next_u64() & 0b11) as u8
    }

    /// Generate a random index in the range 0..15 (for two-qubit Pauli selection).
    ///
    /// This is optimized for the common noise model pattern of selecting one of 15
    /// non-identity two-qubit Pauli errors (IX, IY, IZ, XI, XX, XY, XZ, YI, ..., ZZ).
    /// Uses the multiply-shift method for unbiased selection without rejection sampling.
    ///
    /// # Returns
    ///
    /// A value in 0..14 with equal probability.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_rng::{PecosRng, SeedableRng};
    /// use pecos_rng::rng_ext::RngProbabilityExt;
    ///
    /// let mut rng = PecosRng::seed_from_u64(42);
    /// let fault_type = rng.random_index_15();
    /// // 0=IX, 1=IY, 2=IZ, 3=XI, 4=XX, ..., 14=ZZ
    /// ```
    #[inline]
    #[allow(clippy::cast_possible_truncation)]
    fn random_index_15(&mut self) -> u8 {
        // Use multiply-shift method for unbiased selection:
        // Multiply by 15 and take top bits to divide [0, 2^64) into 15 equal parts
        ((u128::from(self.next_u64()) * 15) >> 64) as u8
    }

    /// Generate a random index in the range 0..16 (for two-qubit Pauli + identity).
    ///
    /// This selects from all 16 two-qubit Paulis including identity (II).
    ///
    /// # Returns
    ///
    /// A value in 0..15 with equal probability.
    #[inline]
    fn random_index_16(&mut self) -> u8 {
        // Exactly 4 bits needed, no rejection required
        (self.next_u64() & 0b1111) as u8
    }

    // ========================================================================
    // Fused noise sampling (probability check + Pauli selection)
    // ========================================================================

    /// Sample a single-qubit noise event.
    ///
    /// This combines probability checking and Pauli selection into a single
    /// operation, which is the common pattern in noise models.
    ///
    /// When no error occurs (the common case), this uses only 1 RNG call.
    /// When an error occurs, it uses 2 RNG calls total.
    ///
    /// # Arguments
    ///
    /// * `threshold` - Precomputed probability threshold from [`probability_threshold`](Self::probability_threshold)
    ///
    /// # Returns
    ///
    /// * `None` - No error occurred
    /// * `Some(0)` - X error
    /// * `Some(1)` - Y error
    /// * `Some(2)` - Z error
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_rng::{PecosRng, SeedableRng};
    /// use pecos_rng::rng_ext::RngProbabilityExt;
    ///
    /// let mut rng = PecosRng::seed_from_u64(42);
    /// let threshold = rng.probability_threshold(0.001);
    ///
    /// match rng.noise_sample_1q(threshold) {
    ///     Some(0) => println!("X error"),
    ///     Some(1) => println!("Y error"),
    ///     Some(2) => println!("Z error"),
    ///     None => println!("No error"),
    ///     _ => unreachable!(),
    /// }
    /// ```
    #[inline]
    fn noise_sample_1q(&mut self, threshold: u64) -> Option<u8> {
        if self.next_u64() >= threshold {
            return None;
        }
        Some(self.random_index_3())
    }

    /// Sample a two-qubit noise event.
    ///
    /// This combines probability checking and two-qubit Pauli selection into
    /// a single operation, which is the common pattern in noise models.
    ///
    /// When no error occurs (the common case), this uses only 1 RNG call.
    /// When an error occurs, it uses 2 RNG calls total.
    ///
    /// # Arguments
    ///
    /// * `threshold` - Precomputed probability threshold from [`probability_threshold`](Self::probability_threshold)
    ///
    /// # Returns
    ///
    /// * `None` - No error occurred
    /// * `Some(0..14)` - Two-qubit Pauli error (IX, IY, IZ, XI, XX, ..., ZZ)
    ///
    /// # Pauli encoding
    ///
    /// | Value | Pauli |
    /// |-------|-------|
    /// | 0     | IX    |
    /// | 1     | IY    |
    /// | 2     | IZ    |
    /// | 3     | XI    |
    /// | 4     | XX    |
    /// | 5     | XY    |
    /// | 6     | XZ    |
    /// | 7     | YI    |
    /// | 8     | YX    |
    /// | 9     | YY    |
    /// | 10    | YZ    |
    /// | 11    | ZI    |
    /// | 12    | ZX    |
    /// | 13    | ZY    |
    /// | 14    | ZZ    |
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_rng::{PecosRng, SeedableRng};
    /// use pecos_rng::rng_ext::RngProbabilityExt;
    ///
    /// let mut rng = PecosRng::seed_from_u64(42);
    /// let threshold = rng.probability_threshold(0.001);
    ///
    /// if let Some(fault_type) = rng.noise_sample_2q(threshold) {
    ///     println!("Two-qubit error type: {}", fault_type);
    /// }
    /// ```
    #[inline]
    fn noise_sample_2q(&mut self, threshold: u64) -> Option<u8> {
        if self.next_u64() >= threshold {
            return None;
        }
        Some(self.random_index_15())
    }

    // ========================================================================
    // Boolean generation utilities
    // ========================================================================

    /// Generate a random boolean with 50% probability.
    ///
    /// This is an optimized coin flip that uses the sign bit of a random i32,
    /// avoiding floating-point operations entirely.
    ///
    /// # Returns
    ///
    /// `true` or `false` with equal probability.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_rng::{PecosRng, SeedableRng};
    /// use pecos_rng::rng_ext::RngProbabilityExt;
    ///
    /// let mut rng = PecosRng::seed_from_u64(42);
    /// if rng.coin_flip() {
    ///     println!("Heads!");
    /// } else {
    ///     println!("Tails!");
    /// }
    /// ```
    #[inline]
    #[allow(clippy::cast_possible_wrap)]
    fn coin_flip(&mut self) -> bool {
        // Use sign bit of random value - faster than threshold comparison
        (self.next_u32() as i32) < 0
    }

    /// Generate a vector of random booleans with the given probability.
    ///
    /// Each boolean is independently `true` with probability `p`.
    /// This is useful for generating error patterns in noise models.
    ///
    /// # Arguments
    ///
    /// * `p` - Probability of each element being `true` (0.0 to 1.0)
    /// * `n` - Number of booleans to generate
    ///
    /// # Returns
    ///
    /// A vector of `n` booleans where each is `true` with probability `p`.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_rng::{PecosRng, SeedableRng};
    /// use pecos_rng::rng_ext::RngProbabilityExt;
    ///
    /// let mut rng = PecosRng::seed_from_u64(42);
    /// let errors = rng.gen_bools(0.01, 100);
    /// println!("Error count: {}", errors.iter().filter(|&&b| b).count());
    /// ```
    #[inline]
    fn gen_bools(&mut self, p: f64, n: usize) -> Vec<bool> {
        let threshold = self.probability_threshold(p);
        (0..n).map(|_| self.next_u64() < threshold).collect()
    }

    // ========================================================================
    // Bulk generation utilities
    // ========================================================================

    /// Fill a slice with random u64 values.
    ///
    /// This is the primary bulk random number generation method. The default
    /// implementation uses a simple loop, but RNG implementations may provide
    /// optimized versions that use SIMD or batch generation.
    ///
    /// # Arguments
    ///
    /// * `dest` - The slice to fill with random values
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_rng::{PecosRng, SeedableRng};
    /// use pecos_rng::rng_ext::RngProbabilityExt;
    ///
    /// let mut rng = PecosRng::seed_from_u64(42);
    /// let mut data = vec![0u64; 1000];
    /// rng.fill_u64(&mut data);
    /// ```
    #[inline]
    fn fill_u64(&mut self, dest: &mut [u64]) {
        for val in dest {
            *val = self.next_u64();
        }
    }
}

// Blanket implementation for all Rng types
impl<T: Rng> RngProbabilityExt for T {}

// ============================================================================
// RngBulkExt: Optimized bulk operations
// ============================================================================

/// Extension trait for optimized bulk random number generation.
///
/// This trait provides methods for efficiently filling slices with random values.
/// Unlike [`RngProbabilityExt`] which has a blanket implementation, this trait
/// requires explicit implementation to enable optimized versions.
///
/// # Example
///
/// ```
/// use pecos_rng::{PecosRng, SeedableRng, RngBulkExt};
///
/// let mut rng = PecosRng::seed_from_u64(42);
/// let mut data = vec![0u64; 1000];
/// rng.fill_u64_bulk(&mut data);  // Uses optimized implementation
/// ```
pub trait RngBulkExt: Rng {
    /// Fill a slice with random u64 values using optimized bulk generation.
    ///
    /// This method is designed for high-performance scenarios where many
    /// random values are needed at once, such as measurement sampling.
    fn fill_u64_bulk(&mut self, dest: &mut [u64]);
}

// Optimized implementations for PECOS RNGs
impl RngBulkExt for crate::PecosRng {
    #[inline]
    fn fill_u64_bulk(&mut self, dest: &mut [u64]) {
        self.fill_u64(dest);
    }
}

impl RngBulkExt for crate::PecosQualityRng {
    #[inline]
    fn fill_u64_bulk(&mut self, dest: &mut [u64]) {
        self.fill_u64(dest);
    }
}

impl RngBulkExt for crate::PecosScalarRng {
    #[inline]
    fn fill_u64_bulk(&mut self, dest: &mut [u64]) {
        self.fill_u64(dest);
    }
}

// Default implementations for common external RNGs
impl RngBulkExt for rand::rngs::SmallRng {
    #[inline]
    fn fill_u64_bulk(&mut self, dest: &mut [u64]) {
        for val in dest {
            *val = self.next_u64();
        }
    }
}

impl RngBulkExt for rand::rngs::StdRng {
    #[inline]
    fn fill_u64_bulk(&mut self, dest: &mut [u64]) {
        for val in dest {
            *val = self.next_u64();
        }
    }
}

impl RngBulkExt for rand::rngs::ThreadRng {
    #[inline]
    fn fill_u64_bulk(&mut self, dest: &mut [u64]) {
        for val in dest {
            *val = self.next_u64();
        }
    }
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
mod tests {
    use super::*;
    use rand::rngs::SmallRng;
    use rand_core::SeedableRng;

    #[test]
    fn test_trait_works_with_smallrng() {
        let mut rng = SmallRng::seed_from_u64(42);
        let threshold = rng.probability_threshold(0.1);

        let mut count = 0;
        for _ in 0..10_000 {
            if rng.check_probability(threshold) {
                count += 1;
            }
        }

        let ratio = f64::from(count) / 10_000.0;
        assert!(
            (0.08..0.12).contains(&ratio),
            "SmallRng ratio {ratio} out of range"
        );
    }

    #[test]
    fn test_count_occurrences_with_smallrng() {
        let mut rng = SmallRng::seed_from_u64(42);
        let threshold = rng.probability_threshold(0.05);

        let count = rng.count_occurrences(threshold, 10_000);
        let ratio = count as f64 / 10_000.0;

        assert!(
            (0.04..0.06).contains(&ratio),
            "count_occurrences ratio {ratio} out of range"
        );
    }

    #[test]
    fn test_random_index_3_distribution() {
        let mut rng = SmallRng::seed_from_u64(42);
        let mut counts = [0usize; 3];

        for _ in 0..30_000 {
            let idx = rng.random_index_3();
            assert!(idx < 3, "random_index_3 returned {idx} >= 3");
            counts[idx as usize] += 1;
        }

        // Each should be ~10,000 (33.3%), allow 8-12% range
        for (i, &count) in counts.iter().enumerate() {
            let ratio = count as f64 / 30_000.0;
            assert!(
                (0.30..0.37).contains(&ratio),
                "random_index_3 bucket {i} ratio {ratio} out of range"
            );
        }
    }

    #[test]
    fn test_random_index_4_distribution() {
        let mut rng = SmallRng::seed_from_u64(42);
        let mut counts = [0usize; 4];

        for _ in 0..40_000 {
            let idx = rng.random_index_4();
            assert!(idx < 4, "random_index_4 returned {idx} >= 4");
            counts[idx as usize] += 1;
        }

        // Each should be ~10,000 (25%), allow 22-28% range
        for (i, &count) in counts.iter().enumerate() {
            let ratio = count as f64 / 40_000.0;
            assert!(
                (0.22..0.28).contains(&ratio),
                "random_index_4 bucket {i} ratio {ratio} out of range"
            );
        }
    }

    #[test]
    fn test_random_index_15_distribution() {
        let mut rng = SmallRng::seed_from_u64(42);
        let mut counts = [0usize; 15];

        for _ in 0..150_000 {
            let idx = rng.random_index_15();
            assert!(idx < 15, "random_index_15 returned {idx} >= 15");
            counts[idx as usize] += 1;
        }

        // Each should be ~10,000 (6.67%), allow 5-9% range
        for (i, &count) in counts.iter().enumerate() {
            let ratio = count as f64 / 150_000.0;
            assert!(
                (0.05..0.09).contains(&ratio),
                "random_index_15 bucket {i} ratio {ratio} out of range"
            );
        }
    }

    #[test]
    fn test_random_index_16_distribution() {
        let mut rng = SmallRng::seed_from_u64(42);
        let mut counts = [0usize; 16];

        for _ in 0..160_000 {
            let idx = rng.random_index_16();
            assert!(idx < 16, "random_index_16 returned {idx} >= 16");
            counts[idx as usize] += 1;
        }

        // Each should be ~10,000 (6.25%), allow 5-8% range
        for (i, &count) in counts.iter().enumerate() {
            let ratio = count as f64 / 160_000.0;
            assert!(
                (0.05..0.08).contains(&ratio),
                "random_index_16 bucket {i} ratio {ratio} out of range"
            );
        }
    }

    #[test]
    fn test_coin_flip_distribution() {
        let mut rng = SmallRng::seed_from_u64(42);
        let mut true_count = 0;

        for _ in 0..10_000 {
            if rng.coin_flip() {
                true_count += 1;
            }
        }

        let ratio = f64::from(true_count) / 10_000.0;
        assert!(
            (0.48..0.52).contains(&ratio),
            "coin_flip ratio {ratio} out of expected 50% range"
        );
    }

    #[test]
    fn test_gen_bools_length() {
        let mut rng = SmallRng::seed_from_u64(42);
        let bools = rng.gen_bools(0.5, 100);
        assert_eq!(bools.len(), 100);
    }

    #[test]
    fn test_gen_bools_distribution() {
        let mut rng = SmallRng::seed_from_u64(42);

        // Test with p=0.3
        let bools = rng.gen_bools(0.3, 10_000);
        let true_count = bools.iter().filter(|&&b| b).count();
        let ratio = true_count as f64 / 10_000.0;
        assert!(
            (0.27..0.33).contains(&ratio),
            "gen_bools(0.3) ratio {ratio} out of expected range"
        );

        // Test with p=0.0 (all false)
        let bools_zero = rng.gen_bools(0.0, 1000);
        assert!(
            bools_zero.iter().all(|&b| !b),
            "gen_bools(0.0) should all be false"
        );

        // Test with p=1.0 (all true)
        let bools_one = rng.gen_bools(1.0, 1000);
        assert!(
            bools_one.iter().all(|&b| b),
            "gen_bools(1.0) should all be true"
        );
    }

    #[test]
    fn test_noise_sample_1q_error_rate() {
        let mut rng = SmallRng::seed_from_u64(42);
        let trials = 100_000;
        let p = 0.01; // 1% error rate
        let threshold = rng.probability_threshold(p);

        let mut error_count = 0;
        for _ in 0..trials {
            if rng.noise_sample_1q(threshold).is_some() {
                error_count += 1;
            }
        }

        let ratio = f64::from(error_count) / f64::from(trials);
        assert!(
            (0.008..0.012).contains(&ratio),
            "noise_sample_1q error rate {ratio} out of expected 1% range"
        );
    }

    #[test]
    fn test_noise_sample_1q_pauli_distribution() {
        let mut rng = SmallRng::seed_from_u64(42);
        let threshold = rng.probability_threshold(1.0); // Always error to test Pauli distribution
        let mut counts = [0usize; 3];

        for _ in 0..30_000 {
            if let Some(pauli) = rng.noise_sample_1q(threshold) {
                assert!(pauli < 3, "noise_sample_1q returned invalid Pauli {pauli}");
                counts[pauli as usize] += 1;
            }
        }

        // Each Pauli should be ~10,000 (33.3%), allow 30-37% range
        for (i, &count) in counts.iter().enumerate() {
            let ratio = count as f64 / 30_000.0;
            assert!(
                (0.30..0.37).contains(&ratio),
                "noise_sample_1q Pauli {i} ratio {ratio} out of range"
            );
        }
    }

    #[test]
    fn test_noise_sample_2q_error_rate() {
        let mut rng = SmallRng::seed_from_u64(42);
        let trials = 100_000;
        let p = 0.005; // 0.5% error rate
        let threshold = rng.probability_threshold(p);

        let mut error_count = 0;
        for _ in 0..trials {
            if rng.noise_sample_2q(threshold).is_some() {
                error_count += 1;
            }
        }

        let ratio = f64::from(error_count) / f64::from(trials);
        assert!(
            (0.004..0.006).contains(&ratio),
            "noise_sample_2q error rate {ratio} out of expected 0.5% range"
        );
    }

    #[test]
    fn test_noise_sample_2q_pauli_distribution() {
        let mut rng = SmallRng::seed_from_u64(42);
        let threshold = rng.probability_threshold(1.0); // Always error to test Pauli distribution
        let mut counts = [0usize; 15];

        for _ in 0..150_000 {
            if let Some(pauli) = rng.noise_sample_2q(threshold) {
                assert!(pauli < 15, "noise_sample_2q returned invalid Pauli {pauli}");
                counts[pauli as usize] += 1;
            }
        }

        // Each Pauli should be ~10,000 (6.67%), allow 5-9% range
        for (i, &count) in counts.iter().enumerate() {
            let ratio = count as f64 / 150_000.0;
            assert!(
                (0.05..0.09).contains(&ratio),
                "noise_sample_2q Pauli {i} ratio {ratio} out of range"
            );
        }
    }

    #[test]
    fn test_noise_sample_zero_probability() {
        let mut rng = SmallRng::seed_from_u64(42);
        let threshold = rng.probability_threshold(0.0);

        // With p=0, should never get an error
        for _ in 0..1000 {
            assert!(
                rng.noise_sample_1q(threshold).is_none(),
                "noise_sample_1q should return None with p=0"
            );
            assert!(
                rng.noise_sample_2q(threshold).is_none(),
                "noise_sample_2q should return None with p=0"
            );
        }
    }

    #[test]
    fn test_check_probability_indices_count() {
        let mut rng = SmallRng::seed_from_u64(42);
        let p = 0.01; // 1% probability
        let threshold = rng.probability_threshold(p);
        let count = 10_000;

        let indices = rng.check_probability_indices(threshold, count);

        // Should get approximately 1% of count
        let ratio = indices.len() as f64 / count as f64;
        assert!(
            (0.008..0.012).contains(&ratio),
            "check_probability_indices ratio {ratio} out of expected 1% range"
        );

        // All indices should be valid
        for &idx in &indices {
            assert!(idx < count, "Index {idx} out of bounds");
        }

        // Indices should be sorted (they're generated in order)
        for i in 1..indices.len() {
            assert!(
                indices[i] > indices[i - 1],
                "Indices not sorted: {} <= {}",
                indices[i],
                indices[i - 1]
            );
        }
    }

    #[test]
    fn test_check_probability_indices_zero_probability() {
        let mut rng = SmallRng::seed_from_u64(42);
        let threshold = rng.probability_threshold(0.0);

        let indices = rng.check_probability_indices(threshold, 10_000);
        assert!(
            indices.is_empty(),
            "check_probability_indices should return empty with p=0"
        );
    }

    #[test]
    fn test_check_probability_indices_matches_scalar() {
        // Verify batched method gives same results as scalar loop
        let mut rng1 = SmallRng::seed_from_u64(42);
        let mut rng2 = SmallRng::seed_from_u64(42);

        let p = 0.05;
        let threshold = rng1.probability_threshold(p);
        let count = 1000;

        // Get indices using batched method
        let batched_indices = rng1.check_probability_indices(threshold, count);

        // Get indices using scalar loop
        let mut scalar_indices = Vec::new();
        for i in 0..count {
            if rng2.check_probability(threshold) {
                scalar_indices.push(i);
            }
        }

        // Should match exactly (same seed, same sequence)
        assert_eq!(
            batched_indices, scalar_indices,
            "Batched and scalar methods should produce identical results"
        );
    }
}
