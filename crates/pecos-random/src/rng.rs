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

//! High-performance random number generator using `RapidRng`.
//!
//! [`PecosRng`], an alternative RNG for PECOS simulations
//! based on the `RapidHash` mixing function. It provides the same interface as
//! [`PecosRng`](crate::quality_rng::PecosQualityRng) for easy comparison and drop-in replacement.
//!
//! # Usage
//!
//! ```
//! use pecos_random::rng::PecosRng;
//!
//! let mut rng = PecosRng::seed_from_u64(42);
//!
//! // Generate 4 values at once (most efficient)
//! let values = rng.next_u64x4();
//!
//! // Or fill a buffer
//! let mut buffer = vec![0u64; 1000];
//! rng.fill_u64(&mut buffer);
//! ```
//!
//! # Performance
//!
//! `PecosRng` uses the `RapidHash` mixing function which is extremely fast
//! for scalar operations. The x4 operations maintain 4 parallel RNG streams
//! to provide bulk generation capabilities.
//!
//! # Implementation
//!
//! Backed by 4 parallel [`RapidRng`] instances, providing similar bulk generation
//! capabilities to [`PecosRng`](crate::quality_rng::PecosQualityRng).

use core::convert::Infallible;
use rand_core::{SeedableRng, TryRng};
use rapidhash::rng::RapidRng;
use wide::u64x4;

/// Buffer size in chunks (each chunk = 4 u64s from the 4 parallel RNGs).
const BUFFER_CHUNKS: usize = 4;
/// Total buffer size in u64s.
const BUFFER_SIZE: usize = BUFFER_CHUNKS * 4; // 16 elements

/// A high-performance RNG using `RapidHash` mixing with 4 parallel streams.
///
/// This implementation maintains 4 independent [`RapidRng`] instances to provide
/// bulk random number generation with an interface matching [`PecosRng`](crate::quality_rng::PecosQualityRng).
///
/// # State Size
///
/// Maintains 4 independent [`RapidRng`] generators, each with 64 bits of state,
/// for a total of 256 bits of state.
///
/// # Optimized Methods
///
/// For high-frequency scalar operations, use these optimized methods:
/// - [`next_bool_fast`](Self::next_bool_fast): Extracts 64 bools from a single u64
/// - [`check_probability`](Self::check_probability): Fixed-point probability check (avoids f64 conversion)
#[derive(Clone, Debug)]
pub struct ParallelRapidRng {
    /// 4 parallel `RapidRng` generators for bulk operations.
    rngs: [RapidRng; 4],
    /// Buffer for scalar access - 16 elements to amortize generation overhead.
    buffer: [u64; BUFFER_SIZE],
    /// Index into buffer: 0-15 = valid index, 16 = buffer empty.
    buffer_idx: u8,
    /// Buffer for bit-packed bool generation.
    bool_bits: u64,
    /// Remaining bits in bool buffer (0 = empty, 1-64 = valid).
    bool_remaining: u8,
}

impl PartialEq for ParallelRapidRng {
    fn eq(&self, other: &Self) -> bool {
        // Only compare the RNG states, not the buffer (which is just a cache)
        self.rngs == other.rngs
    }
}

impl ParallelRapidRng {
    /// Create a new parallel `RapidRng` from a seed.
    ///
    /// Uses `SplitMix64` to derive 4 independent starting seeds, ensuring
    /// the parallel RNG streams are uncorrelated.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)] // BUFFER_SIZE is always <= 255
    pub fn seed_from_u64(seed: u64) -> Self {
        let mut splitmix = SplitMix64::new(seed);

        Self {
            rngs: [
                RapidRng::new(splitmix.next_u64()),
                RapidRng::new(splitmix.next_u64()),
                RapidRng::new(splitmix.next_u64()),
                RapidRng::new(splitmix.next_u64()),
            ],
            buffer: [0; BUFFER_SIZE],
            buffer_idx: BUFFER_SIZE as u8, // 16 = buffer empty
            bool_bits: 0,
            bool_remaining: 0,
        }
    }

    /// Generate 4 random u64 values simultaneously.
    ///
    /// This is the most efficient way to generate random numbers with this RNG.
    /// Each call advances all 4 internal generators and returns their outputs.
    #[inline]
    pub fn next_u64x4(&mut self) -> u64x4 {
        u64x4::new([
            self.rngs[0].next(),
            self.rngs[1].next(),
            self.rngs[2].next(),
            self.rngs[3].next(),
        ])
    }

    /// Refill the entire buffer with random values.
    #[inline]
    #[cold]
    fn refill_buffer(&mut self) {
        for i in 0..BUFFER_CHUNKS {
            let values = self.next_u64x4();
            let array: [u64; 4] = values.into();
            let offset = i * 4;
            self.buffer[offset] = array[0];
            self.buffer[offset + 1] = array[1];
            self.buffer[offset + 2] = array[2];
            self.buffer[offset + 3] = array[3];
        }
        self.buffer_idx = 0;
    }

    /// Fill a slice with random u64 values.
    ///
    /// Processes 4 values at a time, with buffering for any remainder
    /// to avoid wasting generated values.
    pub fn fill_u64(&mut self, dest: &mut [u64]) {
        // First, drain any buffered values
        let mut i = 0;
        while i < dest.len() && (self.buffer_idx as usize) < BUFFER_SIZE {
            dest[i] = self.buffer[self.buffer_idx as usize];
            self.buffer_idx += 1;
            i += 1;
        }

        // Process remaining in chunks of 4 directly (bypass buffer for efficiency)
        let remaining = &mut dest[i..];
        let mut chunks = remaining.chunks_exact_mut(4);

        for chunk in chunks.by_ref() {
            let values = self.next_u64x4();
            let array: [u64; 4] = values.into();
            chunk.copy_from_slice(&array);
        }

        // Handle remainder by refilling buffer
        let remainder = chunks.into_remainder();
        if !remainder.is_empty() {
            self.refill_buffer();
            for val in remainder {
                *val = self.buffer[self.buffer_idx as usize];
                self.buffer_idx += 1;
            }
        }
    }

    /// Generate a single random u64.
    ///
    /// Uses buffering to avoid wasting generated values - generates 16 values
    /// at once and serves them one at a time.
    #[inline]
    #[allow(clippy::cast_possible_truncation)] // BUFFER_SIZE is always <= 255
    pub fn next_u64(&mut self) -> u64 {
        if self.buffer_idx >= BUFFER_SIZE as u8 {
            self.refill_buffer();
        }
        let val = self.buffer[self.buffer_idx as usize];
        self.buffer_idx += 1;
        val
    }

    /// Generate a single random u32.
    #[inline]
    #[allow(clippy::cast_possible_truncation)]
    pub fn next_u32(&mut self) -> u32 {
        self.next_u64() as u32
    }

    /// Generate a random f64 in [0, 1).
    #[inline]
    #[allow(clippy::cast_precision_loss)]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    // ========================================================================
    // Optimized methods for high-frequency scalar operations
    // ========================================================================

    /// Generate a random bool using bit-packed extraction.
    ///
    /// This is ~16x more efficient than `random::<bool>()` because it extracts
    /// 64 bools from a single u64 instead of generating a new u64 for each bool.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_random::rng::PecosRng;
    ///
    /// let mut rng = PecosRng::seed_from_u64(42);
    /// let coin_flip = rng.next_bool_fast();
    /// ```
    #[inline]
    pub fn next_bool_fast(&mut self) -> bool {
        if self.bool_remaining == 0 {
            self.bool_bits = self.next_u64();
            self.bool_remaining = 64;
        }
        self.bool_remaining -= 1;
        (self.bool_bits >> self.bool_remaining) & 1 != 0
    }

    /// Generate a random bool with probability `p` of being true.
    ///
    /// This shadows the `rand::Rng::random_bool` trait method to provide
    /// an optimized implementation for `p = 0.5` using bit-packed extraction.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_random::rng::PecosRng;
    ///
    /// let mut rng = PecosRng::seed_from_u64(42);
    /// let coin_flip = rng.random_bool(0.5);  // Uses optimized path
    /// let biased = rng.random_bool(0.3);     // Uses standard path
    /// ```
    #[inline]
    #[allow(clippy::float_cmp)] // Exact 0.5 check is intentional for optimization
    pub fn random_bool(&mut self, p: f64) -> bool {
        if p == 0.5 {
            self.next_bool_fast()
        } else {
            self.next_f64() < p
        }
    }

    // ========================================================================
    // Fixed-point probability methods (avoid f64 conversion)
    // ========================================================================

    /// Convert a probability to a u64 threshold for use with [`check_probability`](Self::check_probability).
    ///
    /// This allows precomputing the threshold once and reusing it for many
    /// probability checks, avoiding f64 conversion on each check.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_random::rng::PecosRng;
    ///
    /// // Precompute threshold for 0.1% error rate
    /// let error_threshold = PecosRng::probability_threshold(0.001);
    ///
    /// let mut rng = PecosRng::seed_from_u64(42);
    /// for _ in 0..1000 {
    ///     if rng.check_probability(error_threshold) {
    ///         // Error occurred
    ///     }
    /// }
    /// ```
    #[inline]
    #[must_use]
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    pub fn probability_threshold(p: f64) -> u64 {
        // Full 64-bit range: 0 = 0.0, u64::MAX = ~1.0
        (p * (u64::MAX as f64)) as u64
    }

    /// Check if a random event occurs with the given precomputed probability threshold.
    ///
    /// This is faster than `random_bool(p)` for fixed probabilities because it
    /// avoids the f64 conversion on each call. The threshold should be computed
    /// once using [`probability_threshold`](Self::probability_threshold).
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_random::rng::PecosRng;
    ///
    /// let threshold = PecosRng::probability_threshold(0.001);
    /// let mut rng = PecosRng::seed_from_u64(42);
    ///
    /// let occurred = rng.check_probability(threshold);
    /// ```
    #[inline]
    pub fn check_probability(&mut self, threshold: u64) -> bool {
        self.next_u64() < threshold
    }

    /// Check 4 probabilities at once.
    ///
    /// Returns an array of 4 bools indicating which events occurred.
    /// This is useful when you need exactly 4 probability checks.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_random::rng::PecosRng;
    ///
    /// let mut rng = PecosRng::seed_from_u64(42);
    /// let threshold = PecosRng::probability_threshold(0.01);
    ///
    /// let [a, b, c, d] = rng.check_probability_x4(threshold);
    /// ```
    #[inline]
    pub fn check_probability_x4(&mut self, threshold: u64) -> [bool; 4] {
        let values = self.next_u64x4();
        let arr: [u64; 4] = values.into();
        [
            arr[0] < threshold,
            arr[1] < threshold,
            arr[2] < threshold,
            arr[3] < threshold,
        ]
    }

    /// Count how many events occur out of `count` checks with the given probability threshold.
    ///
    /// This is optimized for counting occurrences without storing individual results.
    /// Useful for noise models that just need to know how many errors occurred.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_random::rng::PecosRng;
    ///
    /// let mut rng = PecosRng::seed_from_u64(42);
    /// let threshold = PecosRng::probability_threshold(0.001);
    ///
    /// let error_count = rng.count_occurrences(threshold, 10000);
    /// ```
    #[inline]
    pub fn count_occurrences(&mut self, threshold: u64, count: usize) -> usize {
        let mut total = 0usize;

        // Process in chunks of 4
        let full_chunks = count / 4;
        for _ in 0..full_chunks {
            let values = self.next_u64x4();
            let arr: [u64; 4] = values.into();
            total += usize::from(arr[0] < threshold);
            total += usize::from(arr[1] < threshold);
            total += usize::from(arr[2] < threshold);
            total += usize::from(arr[3] < threshold);
        }

        // Handle remainder
        for _ in 0..(count % 4) {
            total += usize::from(self.next_u64() < threshold);
        }

        total
    }

    /// Return indices where probability check succeeded, using parallel RNGs.
    ///
    /// This is optimized for low-probability events (like noise in quantum circuits)
    /// where you need to know which indices had events, not just how many.
    ///
    /// Uses 4 parallel RNGs to check probabilities in batches of 4.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_random::rng::PecosRng;
    ///
    /// let mut rng = PecosRng::seed_from_u64(42);
    /// let threshold = PecosRng::probability_threshold(0.001);
    ///
    /// // Check 10,000 gates, get sparse list of error indices
    /// let error_indices = rng.check_probability_indices(threshold, 10_000);
    /// println!("Errors at: {:?}", error_indices);
    /// ```
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    pub fn check_probability_indices(&mut self, threshold: u64, count: usize) -> Vec<usize> {
        // Pre-allocate based on expected number of hits (2x for safety margin)
        let expected_hits = ((count as f64) * (threshold as f64 / u64::MAX as f64) * 2.0) as usize;
        let mut indices = Vec::with_capacity(expected_hits.max(16));

        // Process in chunks of 4 using parallel RNGs
        let full_chunks = count / 4;
        for chunk_idx in 0..full_chunks {
            let base_idx = chunk_idx * 4;
            let values = self.next_u64x4();
            let arr: [u64; 4] = values.into();

            if arr[0] < threshold {
                indices.push(base_idx);
            }
            if arr[1] < threshold {
                indices.push(base_idx + 1);
            }
            if arr[2] < threshold {
                indices.push(base_idx + 2);
            }
            if arr[3] < threshold {
                indices.push(base_idx + 3);
            }
        }

        // Handle remainder using scalar RNG
        let remainder_start = full_chunks * 4;
        for i in 0..(count % 4) {
            if self.next_u64() < threshold {
                indices.push(remainder_start + i);
            }
        }

        indices
    }
}

// ============================================================================
// rand_core trait implementations
// ============================================================================

impl TryRng for ParallelRapidRng {
    type Error = Infallible;

    #[inline]
    fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
        Ok(self.next_u32())
    }

    #[inline]
    fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
        Ok(self.next_u64())
    }

    #[inline]
    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), Self::Error> {
        // Process 8 bytes at a time using next_u64
        let mut chunks = dest.chunks_exact_mut(8);
        for chunk in chunks.by_ref() {
            let bytes = self.next_u64().to_le_bytes();
            chunk.copy_from_slice(&bytes);
        }

        // Handle remainder
        let remainder = chunks.into_remainder();
        if !remainder.is_empty() {
            let bytes = self.next_u64().to_le_bytes();
            for (i, byte) in remainder.iter_mut().enumerate() {
                *byte = bytes[i];
            }
        }
        Ok(())
    }
}

impl SeedableRng for ParallelRapidRng {
    type Seed = [u8; 32];

    fn from_seed(seed: Self::Seed) -> Self {
        // Convert seed bytes to a u64 for our seeding
        let mut seed_u64 = 0u64;
        for (i, chunk) in seed.chunks(8).enumerate() {
            if i >= 1 {
                break;
            }
            let mut bytes = [0u8; 8];
            bytes[..chunk.len()].copy_from_slice(chunk);
            seed_u64 ^= u64::from_le_bytes(bytes);
        }
        // Mix in all the seed bytes
        for chunk in seed.chunks(8) {
            let mut bytes = [0u8; 8];
            bytes[..chunk.len()].copy_from_slice(chunk);
            seed_u64 = seed_u64.wrapping_add(u64::from_le_bytes(bytes));
        }
        Self::seed_from_u64(seed_u64)
    }

    #[inline]
    fn seed_from_u64(seed: u64) -> Self {
        Self::seed_from_u64(seed)
    }
}

/// The default high-performance RNG for PECOS simulations.
///
/// `PecosRng` uses 4 parallel `RapidRng` generators with buffering for optimal
/// performance across different use patterns:
/// - Scalar operations benefit from buffering (amortized generation cost)
/// - Bulk operations use all 4 parallel RNGs simultaneously
/// - Batched probability checking (`check_probability_indices`) is 1.6x faster than scalar loops
///
/// For scalar-heavy patterns where batching isn't possible, consider [`PecosScalarRng`](crate::PecosScalarRng).
/// For maximum statistical quality, consider [`PecosQualityRng`](crate::PecosQualityRng).
///
/// # Example
///
/// ```
/// use pecos_random::PecosRng;
///
/// let mut rng = PecosRng::seed_from_u64(42);
/// let mut buffer = vec![0u64; 1000];
/// rng.fill_u64(&mut buffer);
/// ```
pub type PecosRng = ParallelRapidRng;

// ============================================================================
// SplitMix64 for seed derivation
// ============================================================================

/// `SplitMix64` - used to derive independent seeds from a single seed.
///
/// This is the recommended way to seed RNGs that need multiple independent
/// seeds derived from one.
#[derive(Clone, Copy, Debug)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Create a new `SplitMix64` with the given seed.
    #[inline]
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Generate the next u64 value.
    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_rapid_deterministic() {
        let mut rng1 = ParallelRapidRng::seed_from_u64(42);
        let mut rng2 = ParallelRapidRng::seed_from_u64(42);

        for _ in 0..100 {
            let v1: [u64; 4] = rng1.next_u64x4().into();
            let v2: [u64; 4] = rng2.next_u64x4().into();
            assert_eq!(v1, v2);
        }
    }

    #[test]
    fn test_parallel_rapid_different_seeds() {
        let mut rng1 = ParallelRapidRng::seed_from_u64(1);
        let mut rng2 = ParallelRapidRng::seed_from_u64(2);

        let v1: [u64; 4] = rng1.next_u64x4().into();
        let v2: [u64; 4] = rng2.next_u64x4().into();

        assert_ne!(v1, v2);
    }

    #[test]
    fn test_parallel_rapid_fill_u64() {
        let mut rng = ParallelRapidRng::seed_from_u64(42);

        let mut buffer = vec![0u64; 100];
        rng.fill_u64(&mut buffer);

        let non_zero = buffer.iter().filter(|&&x| x != 0).count();
        assert!(non_zero > 95, "Expected most values to be non-zero");
    }

    #[test]
    fn test_parallel_rapid_fill_remainder() {
        let mut rng = ParallelRapidRng::seed_from_u64(42);

        // Test with non-multiple-of-4 length
        let mut buffer = vec![0u64; 7];
        rng.fill_u64(&mut buffer);

        let non_zero = buffer.iter().filter(|&&x| x != 0).count();
        assert!(non_zero >= 5, "Expected most values to be non-zero");
    }

    #[test]
    fn test_parallel_rapid_produces_different_values() {
        let mut rng = ParallelRapidRng::seed_from_u64(42);

        let v1: [u64; 4] = rng.next_u64x4().into();
        let v2: [u64; 4] = rng.next_u64x4().into();

        // Each call should produce different values
        assert_ne!(v1, v2);

        // Each lane should be different (independent RNGs)
        assert_ne!(v1[0], v1[1]);
        assert_ne!(v1[1], v1[2]);
        assert_ne!(v1[2], v1[3]);
    }

    #[test]
    fn test_parallel_rapid_next_u64() {
        let mut rng = ParallelRapidRng::seed_from_u64(42);

        let v1 = rng.next_u64();
        let v2 = rng.next_u64();

        assert_ne!(v1, v2);
    }

    #[test]
    fn test_parallel_rapid_next_f64_range() {
        let mut rng = ParallelRapidRng::seed_from_u64(42);

        for _ in 0..1000 {
            let f = rng.next_f64();
            assert!((0.0..1.0).contains(&f), "f64 out of range: {f}");
        }
    }

    #[test]
    fn test_next_bool_fast_distribution() {
        let mut rng = ParallelRapidRng::seed_from_u64(42);
        let mut true_count = 0u32;
        let trials = 10_000;

        for _ in 0..trials {
            if rng.next_bool_fast() {
                true_count += 1;
            }
        }

        // Should be roughly 50% true (allow 5% deviation)
        let ratio = f64::from(true_count) / f64::from(trials);
        assert!(
            (0.45..0.55).contains(&ratio),
            "Bool distribution out of range: {ratio}"
        );
    }

    #[test]
    fn test_check_probability_distribution() {
        let mut rng = ParallelRapidRng::seed_from_u64(42);
        let trials = 100_000;

        // Test 0.1% probability
        let threshold = ParallelRapidRng::probability_threshold(0.001);
        let mut count = 0u32;
        for _ in 0..trials {
            if rng.check_probability(threshold) {
                count += 1;
            }
        }
        let ratio = f64::from(count) / f64::from(trials);
        assert!(
            (0.0005..0.0015).contains(&ratio),
            "check_probability(0.001) distribution out of range: {ratio}"
        );
    }

    #[test]
    fn test_count_occurrences_matches_scalar() {
        let mut rng1 = ParallelRapidRng::seed_from_u64(42);
        let mut rng2 = ParallelRapidRng::seed_from_u64(42);

        let threshold = ParallelRapidRng::probability_threshold(0.1);
        let count = 1000;

        // Count via bulk method
        let bulk_count = rng1.count_occurrences(threshold, count);

        // Count via scalar
        let scalar_count = (0..count)
            .filter(|_| rng2.check_probability(threshold))
            .count();

        assert_eq!(bulk_count, scalar_count);
    }

    #[test]
    fn test_buffering_uses_all_values() {
        // Get first 4 values via scalar access
        let mut rng1 = ParallelRapidRng::seed_from_u64(42);
        let v1 = rng1.next_u64();
        let v2 = rng1.next_u64();
        let v3 = rng1.next_u64();
        let v4 = rng1.next_u64();

        // Get same values via next_u64x4
        let mut rng2 = ParallelRapidRng::seed_from_u64(42);
        let batch: [u64; 4] = rng2.next_u64x4().into();

        // All 4 scalar values should match the batch
        assert_eq!(v1, batch[0], "First value mismatch");
        assert_eq!(v2, batch[1], "Second value mismatch");
        assert_eq!(v3, batch[2], "Third value mismatch");
        assert_eq!(v4, batch[3], "Fourth value mismatch");
    }
}
