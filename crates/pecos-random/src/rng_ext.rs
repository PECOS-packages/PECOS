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
//! [`RngProbabilityExt`], a trait that adds fixed-point
//! probability checking methods to any RNG implementing [`Rng`].
//!
//! # Usage
//!
//! ```
//! use pecos_random::{PecosRng, SeedableRng};
//! use pecos_random::rng_ext::RngProbabilityExt;
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
/// Methods for efficient probability checking using
/// precomputed u64 thresholds instead of f64 comparisons. Faster
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
/// use pecos_random::{PecosRng, SeedableRng};
/// use pecos_random::rng_ext::RngProbabilityExt;
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
    /// use pecos_random::{PecosRng, SeedableRng};
    /// use pecos_random::rng_ext::RngProbabilityExt;
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
    /// use pecos_random::{PecosRng, SeedableRng};
    /// use pecos_random::rng_ext::RngProbabilityExt;
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
    /// use pecos_random::{PecosRng, SeedableRng};
    /// use pecos_random::rng_ext::RngProbabilityExt;
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
    /// use pecos_random::{PecosRng, SeedableRng};
    /// use pecos_random::rng_ext::RngProbabilityExt;
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
    /// use pecos_random::{PecosRng, SeedableRng};
    /// use pecos_random::rng_ext::RngProbabilityExt;
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
    /// use pecos_random::{PecosRng, SeedableRng};
    /// use pecos_random::rng_ext::RngProbabilityExt;
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
    /// use pecos_random::{PecosRng, SeedableRng};
    /// use pecos_random::rng_ext::RngProbabilityExt;
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
    /// use pecos_random::{PecosRng, SeedableRng};
    /// use pecos_random::rng_ext::RngProbabilityExt;
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
    /// use pecos_random::{PecosRng, SeedableRng};
    /// use pecos_random::rng_ext::RngProbabilityExt;
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
    /// use pecos_random::{PecosRng, SeedableRng};
    /// use pecos_random::rng_ext::RngProbabilityExt;
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
    /// use pecos_random::{PecosRng, SeedableRng};
    /// use pecos_random::rng_ext::RngProbabilityExt;
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
    /// use pecos_random::{PecosRng, SeedableRng};
    /// use pecos_random::rng_ext::RngProbabilityExt;
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

    /// Fill a slice with random f64 values in the range [0, 1).
    ///
    /// This is useful for pre-generating random values for multi-shot
    /// measurement sampling or Monte Carlo simulations.
    ///
    /// # Arguments
    ///
    /// * `dest` - The slice to fill with random f64 values
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_random::{PecosRng, SeedableRng};
    /// use pecos_random::rng_ext::RngProbabilityExt;
    ///
    /// let mut rng = PecosRng::seed_from_u64(42);
    /// let mut randoms = vec![0.0f64; 1000];
    /// rng.fill_f64(&mut randoms);
    ///
    /// // All values should be in [0, 1)
    /// assert!(randoms.iter().all(|&x| (0.0..1.0).contains(&x)));
    /// ```
    #[inline]
    #[allow(clippy::cast_precision_loss)]
    fn fill_f64(&mut self, dest: &mut [f64]) {
        for val in dest {
            // Same conversion as next_f64(): use top 53 bits for full precision
            *val = (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
        }
    }

    // ========================================================================
    // Bernoulli sampling
    // ========================================================================

    /// Sample a Bernoulli random variable with probability `p`.
    ///
    /// Returns `true` with probability `p` and `false` with probability `1 - p`.
    /// This is a convenience method that combines threshold computation and
    /// probability checking, optimized for single-use probabilities.
    ///
    /// For repeated checks with the same probability, prefer using
    /// [`probability_threshold`](Self::probability_threshold) once and then
    /// [`check_probability`](Self::check_probability) for each check.
    ///
    /// # Arguments
    ///
    /// * `p` - Probability of returning `true`, in the range [0.0, 1.0]
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_random::{PecosRng, SeedableRng};
    /// use pecos_random::rng_ext::RngProbabilityExt;
    ///
    /// let mut rng = PecosRng::seed_from_u64(42);
    ///
    /// // Simulate a biased coin flip (30% heads)
    /// let heads = rng.bernoulli(0.3);
    ///
    /// // Equivalent to: rng.random::<f64>() < 0.3
    /// // but uses integer comparison internally
    /// ```
    #[inline]
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    fn bernoulli(&mut self, p: f64) -> bool {
        // Convert probability to threshold and check in one step
        let threshold = (p * (u64::MAX as f64)) as u64;
        self.next_u64() < threshold
    }

    // ========================================================================
    // Discrete distribution sampling
    // ========================================================================

    /// Sample from a discrete distribution using a precomputed CDF.
    ///
    /// This uses binary search for O(log n) sampling, which is significantly
    /// faster than linear scan for distributions with many outcomes (e.g.,
    /// batched quantum measurements with >4 qubits).
    ///
    /// # Arguments
    ///
    /// * `cdf` - Cumulative distribution function as a slice of cumulative
    ///   probabilities. Must be monotonically increasing with the last element
    ///   being 1.0 (or close to it due to floating point).
    ///
    /// # Returns
    ///
    /// The index of the sampled outcome (0 to cdf.len()-1).
    ///
    /// # Performance
    ///
    /// | Outcomes | Linear Scan | Binary Search | Speedup |
    /// |----------|-------------|---------------|---------|
    /// | 16       | 14 ns       | 22 ns         | 0.6x    |
    /// | 256      | 97 ns       | 39 ns         | 2.5x    |
    /// | 4096     | 753 ns      | 45 ns         | 17x     |
    /// | 65536    | 11537 ns    | 69 ns         | 167x    |
    ///
    /// Use linear scan for small distributions (<32 outcomes) and binary
    /// search for larger ones.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_random::{PecosRng, SeedableRng};
    /// use pecos_random::rng_ext::RngProbabilityExt;
    ///
    /// let mut rng = PecosRng::seed_from_u64(42);
    ///
    /// // Precompute CDF from probabilities
    /// let probs = [0.1, 0.2, 0.3, 0.4];
    /// let cdf: Vec<f64> = probs.iter()
    ///     .scan(0.0, |acc, &p| { *acc += p; Some(*acc) })
    ///     .collect();
    ///
    /// // Sample from the distribution
    /// let outcome = rng.sample_discrete_cdf(&cdf);
    /// assert!(outcome < 4);
    /// ```
    #[inline]
    #[allow(clippy::cast_precision_loss)]
    fn sample_discrete_cdf(&mut self, cdf: &[f64]) -> usize {
        debug_assert!(!cdf.is_empty(), "CDF must not be empty");
        let rand_val = (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
        match cdf.binary_search_by(|&c| {
            c.partial_cmp(&rand_val)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            Ok(i) => i,
            Err(i) => i.min(cdf.len() - 1),
        }
    }

    /// Compute a CDF from a probability distribution.
    ///
    /// This is a helper function to prepare a distribution for use with
    /// [`sample_discrete_cdf`](Self::sample_discrete_cdf).
    ///
    /// # Arguments
    ///
    /// * `probs` - Probability distribution (should sum to 1.0)
    ///
    /// # Returns
    ///
    /// A vector containing the cumulative distribution function.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_random::{PecosRng, SeedableRng};
    /// use pecos_random::rng_ext::RngProbabilityExt;
    ///
    /// let mut rng = PecosRng::seed_from_u64(42);
    ///
    /// let probs = vec![0.25, 0.25, 0.25, 0.25];
    /// let cdf = rng.compute_cdf(&probs);
    ///
    /// // Now sample efficiently
    /// let outcome = rng.sample_discrete_cdf(&cdf);
    /// ```
    #[inline]
    fn compute_cdf(&self, probs: &[f64]) -> Vec<f64> {
        let mut cdf = Vec::with_capacity(probs.len());
        let mut cumulative = 0.0;
        for &p in probs {
            cumulative += p;
            cdf.push(cumulative);
        }
        cdf
    }
}

// Blanket implementation for all Rng types
impl<T: Rng> RngProbabilityExt for T {}

// ============================================================================
// RngBulkExt: Optimized bulk operations
// ============================================================================

/// Extension trait for optimized bulk random number generation.
///
/// Efficiently fills slices with random values.
/// Unlike [`RngProbabilityExt`] which has a blanket implementation, this trait
/// requires explicit implementation to enable optimized versions.
///
/// # Example
///
/// ```
/// use pecos_random::{PecosRng, SeedableRng, RngBulkExt};
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
    use rand::RngExt;
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

    #[test]
    fn test_compute_cdf() {
        let rng = SmallRng::seed_from_u64(42);
        let probs = vec![0.25, 0.25, 0.25, 0.25];
        let cdf = rng.compute_cdf(&probs);

        assert_eq!(cdf.len(), 4);
        assert!((cdf[0] - 0.25).abs() < 1e-10);
        assert!((cdf[1] - 0.50).abs() < 1e-10);
        assert!((cdf[2] - 0.75).abs() < 1e-10);
        assert!((cdf[3] - 1.00).abs() < 1e-10);
    }

    #[test]
    fn test_sample_discrete_cdf_bounds() {
        let mut rng = SmallRng::seed_from_u64(42);
        let probs = vec![0.1, 0.2, 0.3, 0.4];
        let cdf = rng.compute_cdf(&probs);

        // Sample many times and check bounds
        for _ in 0..10_000 {
            let idx = rng.sample_discrete_cdf(&cdf);
            assert!(
                idx < 4,
                "sample_discrete_cdf returned out-of-bounds index {idx}"
            );
        }
    }

    #[test]
    fn test_sample_discrete_cdf_distribution() {
        let mut rng = SmallRng::seed_from_u64(42);
        let probs = vec![0.1, 0.2, 0.3, 0.4];
        let cdf = rng.compute_cdf(&probs);

        let mut counts = [0usize; 4];
        let trials: usize = 100_000;

        for _ in 0..trials {
            let idx = rng.sample_discrete_cdf(&cdf);
            counts[idx] += 1;
        }

        // Check distribution matches expected probabilities (allow 10% relative error)
        for (i, &expected) in probs.iter().enumerate() {
            #[allow(clippy::cast_precision_loss)]
            let observed = counts[i] as f64 / trials as f64;
            let error = (observed - expected).abs() / expected;
            assert!(
                error < 0.1,
                "sample_discrete_cdf bucket {i}: expected {expected}, got {observed} (error {error})"
            );
        }
    }

    #[test]
    fn test_sample_discrete_cdf_single_outcome() {
        let mut rng = SmallRng::seed_from_u64(42);
        let cdf = vec![1.0]; // Single outcome with probability 1

        for _ in 0..100 {
            let idx = rng.sample_discrete_cdf(&cdf);
            assert_eq!(idx, 0, "Single outcome CDF should always return 0");
        }
    }

    #[test]
    fn test_sample_discrete_cdf_matches_linear() {
        // Compare binary search to linear scan for correctness
        fn sample_linear(rng: &mut SmallRng, probs: &[f64]) -> usize {
            let rand_val: f64 = rng.random();
            let mut cumulative = 0.0;
            for (i, &p) in probs.iter().enumerate() {
                cumulative += p;
                if rand_val < cumulative {
                    return i;
                }
            }
            probs.len() - 1
        }

        let probs = vec![0.05, 0.15, 0.30, 0.25, 0.15, 0.10];

        // Run both methods with same seed and compare distribution
        let mut rng1 = SmallRng::seed_from_u64(42);
        let mut rng2 = SmallRng::seed_from_u64(42);
        let cdf = rng2.compute_cdf(&probs);

        let trials: usize = 50_000;
        let mut counts_linear = vec![0usize; probs.len()];
        let mut counts_binary = vec![0usize; probs.len()];

        for _ in 0..trials {
            counts_linear[sample_linear(&mut rng1, &probs)] += 1;
            counts_binary[rng2.sample_discrete_cdf(&cdf)] += 1;
        }

        // Both should produce similar distributions
        for i in 0..probs.len() {
            #[allow(clippy::cast_precision_loss)]
            let linear_ratio = counts_linear[i] as f64 / trials as f64;
            #[allow(clippy::cast_precision_loss)]
            let binary_ratio = counts_binary[i] as f64 / trials as f64;
            let diff = (linear_ratio - binary_ratio).abs();
            assert!(
                diff < 0.02,
                "Bucket {i}: linear={linear_ratio}, binary={binary_ratio}, diff={diff}"
            );
        }
    }

    #[test]
    fn test_fill_f64_range() {
        let mut rng = SmallRng::seed_from_u64(42);
        let mut values = vec![0.0f64; 1000];
        rng.fill_f64(&mut values);

        // All values should be in [0, 1)
        for (i, &v) in values.iter().enumerate() {
            assert!(
                (0.0..1.0).contains(&v),
                "fill_f64 value {i} = {v} out of range [0, 1)"
            );
        }

        // Should have some variance (not all the same)
        let first = values[0];
        let has_variance = values.iter().any(|&v| (v - first).abs() > 1e-10);
        assert!(has_variance, "fill_f64 should produce varying values");
    }

    #[test]
    fn test_fill_f64_deterministic() {
        let mut rng1 = SmallRng::seed_from_u64(42);
        let mut rng2 = SmallRng::seed_from_u64(42);

        let mut values1 = vec![0.0f64; 100];
        let mut values2 = vec![0.0f64; 100];

        rng1.fill_f64(&mut values1);
        rng2.fill_f64(&mut values2);

        for i in 0..values1.len() {
            assert!(
                (values1[i] - values2[i]).abs() < 1e-15,
                "fill_f64 should be deterministic with same seed"
            );
        }
    }

    #[test]
    fn test_bernoulli_always_false() {
        let mut rng = SmallRng::seed_from_u64(42);
        for _ in 0..1000 {
            assert!(
                !rng.bernoulli(0.0),
                "bernoulli(0.0) should always return false"
            );
        }
    }

    #[test]
    fn test_bernoulli_always_true() {
        let mut rng = SmallRng::seed_from_u64(42);
        for _ in 0..1000 {
            assert!(
                rng.bernoulli(1.0),
                "bernoulli(1.0) should always return true"
            );
        }
    }

    #[test]
    fn test_bernoulli_distribution() {
        let mut rng = SmallRng::seed_from_u64(42);
        let trials: u32 = 100_000;
        let p = 0.3;

        let mut true_count: u32 = 0;
        for _ in 0..trials {
            if rng.bernoulli(p) {
                true_count += 1;
            }
        }

        let observed = f64::from(true_count) / f64::from(trials);
        let error = (observed - p).abs();
        assert!(
            error < 0.01,
            "bernoulli({p}) observed ratio {observed}, error {error} too large"
        );
    }

    #[test]
    fn test_bernoulli_matches_threshold() {
        // Verify bernoulli produces same distribution as check_probability
        let mut rng1 = SmallRng::seed_from_u64(42);
        let mut rng2 = SmallRng::seed_from_u64(42);

        let p = 0.25;
        let threshold = rng2.probability_threshold(p);
        let trials: i32 = 10_000;

        let mut bernoulli_count: i32 = 0;
        let mut threshold_count: i32 = 0;

        for _ in 0..trials {
            if rng1.bernoulli(p) {
                bernoulli_count += 1;
            }
            if rng2.check_probability(threshold) {
                threshold_count += 1;
            }
        }

        // Both methods should produce similar counts
        let diff = (bernoulli_count - threshold_count).abs();
        let max_diff = trials / 50; // Allow 2% difference
        assert!(
            diff < max_diff,
            "bernoulli and check_probability differ too much: {bernoulli_count} vs {threshold_count}"
        );
    }
}
