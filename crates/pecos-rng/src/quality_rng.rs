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

//! High-performance random number generator for bulk generation.
//!
//! This module provides [`PecosRng`], the default RNG for PECOS simulations.
//! It is optimized for bulk random number generation, making it ideal for
//! Monte Carlo simulations and measurement sampling.
//!
//! # Usage
//!
//! For most use cases, use [`PecosRng`] which is a type alias for the current
//! best-performing RNG implementation:
//!
//! ```
//! use pecos_rng::quality_rng::PecosQualityRng;
//!
//! let mut rng = PecosQualityRng::seed_from_u64(42);
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
//! `PecosRng` is approximately:
//! - 35% faster than scalar Xoshiro256++
//! - 17% faster than `RapidRng`
//!
//! # Implementation
//!
//! Currently backed by [`SimdXoshiro256PlusPlus`], which maintains 4 parallel
//! Xoshiro256++ generators. This may change in future versions if a faster
//! implementation becomes available.

use rand_core::{RngCore, SeedableRng};
use wide::u64x4;

/// `SplitMix64` - used to derive independent seeds from a single seed.
///
/// This is the recommended way to seed Xoshiro generators and works well
/// for any RNG that needs multiple independent seeds derived from one.
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

/// Buffer size in SIMD chunks (each chunk = 4 u64s).
const BUFFER_CHUNKS: usize = 4;
/// Total buffer size in u64s.
const BUFFER_SIZE: usize = BUFFER_CHUNKS * 4; // 16 elements

/// A SIMD-accelerated Xoshiro256++ that generates 4 values per iteration.
///
/// This is a specialized implementation where the RNG state itself is stored
/// in SIMD registers, allowing all operations (XOR, shift, rotate) to be
/// performed on 4 values simultaneously.
///
/// # State Size
///
/// Maintains 4 independent Xoshiro256++ generators, each with 256 bits of state,
/// for a total of 1024 bits of state. This provides excellent resistance to
/// seed collisions and long periods for each stream.
///
/// # Performance
///
/// This implementation is significantly faster than running 4 separate
/// Xoshiro256++ instances because:
/// - State updates use SIMD instructions (4 ops for the price of ~1)
/// - Output generation is fully vectorized
/// - Memory access patterns are optimized for SIMD loads/stores
/// - 16-element buffer amortizes SIMD overhead for scalar access
///
/// # Optimized Methods
///
/// For high-frequency scalar operations, use these optimized methods:
/// - [`next_bool_fast`]: Extracts 64 bools from a single u64 (2x faster than `random_bool(0.5)`)
/// - [`check_probability`]: Fixed-point probability check (avoids f64 conversion)
#[derive(Clone, Debug)]
pub struct SimdXoshiro256PlusPlus {
    /// State: 4 parallel Xoshiro256++ generators stored in SIMD layout.
    /// s[i] contains the i-th state word from all 4 generators.
    s: [u64x4; 4],
    /// Buffer for scalar access - 16 elements to amortize SIMD overhead.
    buffer: [u64; BUFFER_SIZE],
    /// Index into buffer: 0-15 = valid index, 16 = buffer empty.
    buffer_idx: u8,
    /// Buffer for bit-packed bool generation.
    bool_bits: u64,
    /// Remaining bits in bool buffer (0 = empty, 1-64 = valid).
    bool_remaining: u8,
}

impl PartialEq for SimdXoshiro256PlusPlus {
    fn eq(&self, other: &Self) -> bool {
        // Only compare the RNG state, not the buffer (which is just a cache)
        self.s == other.s
    }
}

impl SimdXoshiro256PlusPlus {
    /// Create a new SIMD Xoshiro256++ from a seed.
    ///
    /// Uses `SplitMix64` to derive 4 independent starting states, ensuring
    /// the parallel RNG streams are uncorrelated.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)] // BUFFER_SIZE is always <= 255
    pub fn seed_from_u64(seed: u64) -> Self {
        let mut splitmix = SplitMix64::new(seed);

        // Generate 4 complete Xoshiro256++ states (4 x 4 = 16 u64 values)
        let mut states = [[0u64; 4]; 4];
        for state in &mut states {
            for word in state {
                *word = splitmix.next_u64();
            }
        }

        // Transpose: convert from [rng][word] to [word][rng] layout
        Self {
            s: [
                u64x4::new([states[0][0], states[1][0], states[2][0], states[3][0]]),
                u64x4::new([states[0][1], states[1][1], states[2][1], states[3][1]]),
                u64x4::new([states[0][2], states[1][2], states[2][2], states[3][2]]),
                u64x4::new([states[0][3], states[1][3], states[2][3], states[3][3]]),
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
        // Output: rotl(s[0] + s[3], 23) + s[0]
        let result = rotate_left_u64x4(self.s[0] + self.s[3], 23) + self.s[0];

        // State update (same as scalar Xoshiro256++)
        let t = self.s[1] << 17;

        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];

        self.s[2] ^= t;
        self.s[3] = rotate_left_u64x4(self.s[3], 45);

        result
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
    /// Processes 4 values at a time using SIMD, with buffering for
    /// any remainder to avoid wasting generated values.
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
    /// use pecos_rng::quality_rng::PecosQualityRng;
    ///
    /// let mut rng = PecosQualityRng::seed_from_u64(42);
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
    /// use pecos_rng::quality_rng::PecosQualityRng;
    ///
    /// let mut rng = PecosQualityRng::seed_from_u64(42);
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

    /// Convert a probability to a u64 threshold for use with [`check_probability`].
    ///
    /// This allows precomputing the threshold once and reusing it for many
    /// probability checks, avoiding f64 conversion on each check.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_rng::quality_rng::PecosQualityRng;
    ///
    /// // Precompute threshold for 0.1% error rate
    /// let error_threshold = PecosQualityRng::probability_threshold(0.001);
    ///
    /// let mut rng = PecosQualityRng::seed_from_u64(42);
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
        // No shift needed in check_probability - direct comparison
        (p * (u64::MAX as f64)) as u64
    }

    /// Check if a random event occurs with the given precomputed probability threshold.
    ///
    /// This is faster than `random_bool(p)` for fixed probabilities because it
    /// avoids the f64 conversion on each call. The threshold should be computed
    /// once using [`probability_threshold`].
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_rng::quality_rng::PecosQualityRng;
    ///
    /// let threshold = PecosQualityRng::probability_threshold(0.001);
    /// let mut rng = PecosQualityRng::seed_from_u64(42);
    ///
    /// let occurred = rng.check_probability(threshold);
    /// ```
    #[inline]
    pub fn check_probability(&mut self, threshold: u64) -> bool {
        // Direct comparison - no shift needed
        self.next_u64() < threshold
    }

    /// Check 4 probabilities at once using SIMD comparison.
    ///
    /// Returns an array of 4 bools indicating which events occurred.
    /// This is useful when you need exactly 4 probability checks.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_rng::quality_rng::PecosQualityRng;
    ///
    /// let mut rng = PecosQualityRng::seed_from_u64(42);
    /// let threshold = PecosQualityRng::probability_threshold(0.01);
    ///
    /// let [a, b, c, d] = rng.check_probability_x4(threshold);
    /// ```
    #[inline]
    pub fn check_probability_x4(&mut self, threshold: u64) -> [bool; 4] {
        let values = self.next_u64x4();
        let arr: [u64; 4] = values.into();
        // Compare each value against threshold
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
    /// use pecos_rng::quality_rng::PecosQualityRng;
    ///
    /// let mut rng = PecosQualityRng::seed_from_u64(42);
    /// let threshold = PecosQualityRng::probability_threshold(0.001);
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
}

// ============================================================================
// rand_core trait implementations
// ============================================================================

impl RngCore for SimdXoshiro256PlusPlus {
    #[inline]
    fn next_u32(&mut self) -> u32 {
        self.next_u32()
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.next_u64()
    }

    #[inline]
    fn fill_bytes(&mut self, dest: &mut [u8]) {
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
    }
}

impl SeedableRng for SimdXoshiro256PlusPlus {
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

/// A high-quality RNG based on SIMD-accelerated Xoshiro256++.
///
/// This RNG prioritizes statistical quality over raw speed. For most use cases,
/// prefer [`PecosRng`](crate::PecosRng) which offers better performance.
/// Use `PecosQualityRng` when you need the statistical properties of Xoshiro256++.
///
/// # Example
///
/// ```
/// use pecos_rng::quality_rng::PecosQualityRng;
///
/// let mut rng = PecosQualityRng::seed_from_u64(42);
/// let mut buffer = vec![0u64; 1000];
/// rng.fill_u64(&mut buffer);
/// ```
pub type PecosQualityRng = SimdXoshiro256PlusPlus;

/// Rotate left for u64x4 (SIMD version).
#[inline]
fn rotate_left_u64x4(x: u64x4, n: u32) -> u64x4 {
    (x << n) | (x >> (64 - n))
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_xoshiro_deterministic() {
        let mut rng1 = SimdXoshiro256PlusPlus::seed_from_u64(42);
        let mut rng2 = SimdXoshiro256PlusPlus::seed_from_u64(42);

        for _ in 0..100 {
            let v1: [u64; 4] = rng1.next_u64x4().into();
            let v2: [u64; 4] = rng2.next_u64x4().into();
            assert_eq!(v1, v2);
        }
    }

    #[test]
    fn test_simd_xoshiro_different_seeds() {
        let mut rng1 = SimdXoshiro256PlusPlus::seed_from_u64(1);
        let mut rng2 = SimdXoshiro256PlusPlus::seed_from_u64(2);

        let v1: [u64; 4] = rng1.next_u64x4().into();
        let v2: [u64; 4] = rng2.next_u64x4().into();

        assert_ne!(v1, v2);
    }

    #[test]
    fn test_simd_xoshiro_fill_u64() {
        let mut rng = SimdXoshiro256PlusPlus::seed_from_u64(42);

        let mut buffer = vec![0u64; 100];
        rng.fill_u64(&mut buffer);

        let non_zero = buffer.iter().filter(|&&x| x != 0).count();
        assert!(non_zero > 95, "Expected most values to be non-zero");
    }

    #[test]
    fn test_simd_xoshiro_fill_remainder() {
        let mut rng = SimdXoshiro256PlusPlus::seed_from_u64(42);

        // Test with non-multiple-of-4 length
        let mut buffer = vec![0u64; 7];
        rng.fill_u64(&mut buffer);

        let non_zero = buffer.iter().filter(|&&x| x != 0).count();
        assert!(non_zero >= 5, "Expected most values to be non-zero");
    }

    #[test]
    fn test_simd_xoshiro_produces_different_values() {
        let mut rng = SimdXoshiro256PlusPlus::seed_from_u64(42);

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
    fn test_simd_xoshiro_next_u64() {
        let mut rng = SimdXoshiro256PlusPlus::seed_from_u64(42);

        let v1 = rng.next_u64();
        let v2 = rng.next_u64();

        assert_ne!(v1, v2);
    }

    #[test]
    fn test_simd_xoshiro_next_f64_range() {
        let mut rng = SimdXoshiro256PlusPlus::seed_from_u64(42);

        for _ in 0..1000 {
            let f = rng.next_f64();
            assert!((0.0..1.0).contains(&f), "f64 out of range: {f}");
        }
    }

    #[test]
    fn test_splitmix64_produces_different_values() {
        let mut sm = SplitMix64::new(42);
        let v1 = sm.next_u64();
        let v2 = sm.next_u64();
        let v3 = sm.next_u64();

        assert_ne!(v1, v2);
        assert_ne!(v2, v3);
        assert_ne!(v1, v3);
    }

    #[test]
    fn test_buffering_uses_all_values() {
        // Get first 4 values via scalar access
        let mut rng1 = SimdXoshiro256PlusPlus::seed_from_u64(42);
        let v1 = rng1.next_u64();
        let v2 = rng1.next_u64();
        let v3 = rng1.next_u64();
        let v4 = rng1.next_u64();

        // Get same values via next_u64x4
        let mut rng2 = SimdXoshiro256PlusPlus::seed_from_u64(42);
        let batch: [u64; 4] = rng2.next_u64x4().into();

        // All 4 scalar values should match the batch
        assert_eq!(v1, batch[0], "First value mismatch");
        assert_eq!(v2, batch[1], "Second value mismatch");
        assert_eq!(v3, batch[2], "Third value mismatch");
        assert_eq!(v4, batch[3], "Fourth value mismatch");
    }

    #[test]
    fn test_buffering_state_advances_correctly() {
        let mut rng1 = SimdXoshiro256PlusPlus::seed_from_u64(42);
        let mut rng2 = SimdXoshiro256PlusPlus::seed_from_u64(42);

        // Call next_u64 8 times (2 batches)
        let mut scalar_values = Vec::new();
        for _ in 0..8 {
            scalar_values.push(rng1.next_u64());
        }

        // Call next_u64x4 twice
        let batch1: [u64; 4] = rng2.next_u64x4().into();
        let batch2: [u64; 4] = rng2.next_u64x4().into();

        // Values should match
        assert_eq!(&scalar_values[0..4], &batch1);
        assert_eq!(&scalar_values[4..8], &batch2);
    }

    #[test]
    fn test_fill_u64_with_buffered_remainder() {
        let mut rng1 = SimdXoshiro256PlusPlus::seed_from_u64(42);
        let mut rng2 = SimdXoshiro256PlusPlus::seed_from_u64(42);

        // Fill 7 values (leaves 1 in buffer)
        let mut buf1 = vec![0u64; 7];
        rng1.fill_u64(&mut buf1);

        // Get same values via next_u64x4
        let batch1: [u64; 4] = rng2.next_u64x4().into();
        let batch2: [u64; 4] = rng2.next_u64x4().into();

        assert_eq!(&buf1[0..4], &batch1);
        assert_eq!(&buf1[4..7], &batch2[0..3]);

        // Now get the next value from rng1 - should come from buffer
        let next_from_rng1 = rng1.next_u64();
        assert_eq!(next_from_rng1, batch2[3], "Buffered value mismatch");
    }

    // ========================================================================
    // Tests for optimized methods
    // ========================================================================

    #[test]
    fn test_next_bool_fast_distribution() {
        let mut rng = SimdXoshiro256PlusPlus::seed_from_u64(42);
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
    fn test_next_bool_fast_extracts_64_from_one_u64() {
        let mut rng = SimdXoshiro256PlusPlus::seed_from_u64(42);

        // Get the first u64 that will be used
        let mut rng2 = SimdXoshiro256PlusPlus::seed_from_u64(42);
        let first_u64 = rng2.next_u64();

        // Extract 64 bools - should all come from first_u64
        let mut extracted = 0u64;
        for i in 0..64 {
            if rng.next_bool_fast() {
                extracted |= 1 << (63 - i);
            }
        }

        assert_eq!(extracted, first_u64, "Bit extraction mismatch");
    }

    #[test]
    fn test_random_bool_half_uses_fast_path() {
        // Verify random_bool(0.5) produces same results as next_bool_fast()
        let mut rng1 = SimdXoshiro256PlusPlus::seed_from_u64(42);
        let mut rng2 = SimdXoshiro256PlusPlus::seed_from_u64(42);

        for _ in 0..1000 {
            assert_eq!(rng1.random_bool(0.5), rng2.next_bool_fast());
        }
    }

    #[test]
    fn test_random_bool_distribution() {
        let mut rng = SimdXoshiro256PlusPlus::seed_from_u64(42);
        let trials = 10_000;

        // Test p = 0.3
        let mut count = 0u32;
        for _ in 0..trials {
            if rng.random_bool(0.3) {
                count += 1;
            }
        }
        let ratio = f64::from(count) / f64::from(trials);
        assert!(
            (0.25..0.35).contains(&ratio),
            "random_bool(0.3) distribution out of range: {ratio}"
        );

        // Test p = 0.7
        count = 0;
        for _ in 0..trials {
            if rng.random_bool(0.7) {
                count += 1;
            }
        }
        let ratio = f64::from(count) / f64::from(trials);
        assert!(
            (0.65..0.75).contains(&ratio),
            "random_bool(0.7) distribution out of range: {ratio}"
        );
    }

    // ========================================================================
    // Tests for fixed-point probability methods
    // ========================================================================

    #[test]
    fn test_probability_threshold_values() {
        // Test that threshold conversion is correct
        // Using full 64-bit range: 0 = 0.0, u64::MAX = ~1.0
        assert_eq!(SimdXoshiro256PlusPlus::probability_threshold(0.0), 0);
        // 1.0 maps to u64::MAX (with some rounding)
        let threshold_1 = SimdXoshiro256PlusPlus::probability_threshold(1.0);
        assert!(
            threshold_1 >= u64::MAX - 1024,
            "1.0 should map close to u64::MAX"
        );
        // 0.5 should be roughly half of u64::MAX
        let threshold_half = SimdXoshiro256PlusPlus::probability_threshold(0.5);
        let expected_half = u64::MAX / 2;
        assert!(
            (i128::from(threshold_half) - i128::from(expected_half)).abs() < 1024,
            "0.5 should map close to u64::MAX/2"
        );
    }

    #[test]
    fn test_check_probability_distribution() {
        let mut rng = SimdXoshiro256PlusPlus::seed_from_u64(42);
        let trials = 100_000;

        // Test 0.1% probability
        let threshold = SimdXoshiro256PlusPlus::probability_threshold(0.001);
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

        // Test 30% probability
        let threshold = SimdXoshiro256PlusPlus::probability_threshold(0.3);
        count = 0;
        for _ in 0..trials {
            if rng.check_probability(threshold) {
                count += 1;
            }
        }
        let ratio = f64::from(count) / f64::from(trials);
        assert!(
            (0.29..0.31).contains(&ratio),
            "check_probability(0.3) distribution out of range: {ratio}"
        );
    }

    #[test]
    fn test_check_probability_matches_random_bool() {
        // Verify that check_probability produces equivalent results to random_bool
        // (statistically, not bit-for-bit, since they use different methods)
        let trials = 100_000;

        for &p in &[0.001, 0.1, 0.3, 0.5, 0.7, 0.9] {
            let mut rng1 = SimdXoshiro256PlusPlus::seed_from_u64(42);
            let mut rng2 = SimdXoshiro256PlusPlus::seed_from_u64(43);

            let threshold = SimdXoshiro256PlusPlus::probability_threshold(p);

            let mut count1 = 0u32;
            let mut count2 = 0u32;

            for _ in 0..trials {
                if rng1.check_probability(threshold) {
                    count1 += 1;
                }
                if rng2.next_f64() < p {
                    count2 += 1;
                }
            }

            let ratio1 = f64::from(count1) / f64::from(trials);
            let ratio2 = f64::from(count2) / f64::from(trials);

            // Both should be close to p
            assert!(
                (ratio1 - p).abs() < 0.02,
                "check_probability({p}) ratio {ratio1} too far from {p}"
            );
            assert!(
                (ratio2 - p).abs() < 0.02,
                "next_f64() < {p} ratio {ratio2} too far from {p}"
            );
        }
    }

    // ========================================================================
    // Tests for optimized probability methods
    // ========================================================================

    #[test]
    fn test_check_probability_x4_distribution() {
        let mut rng = SimdXoshiro256PlusPlus::seed_from_u64(42);
        let trials = 10_000;

        let threshold = SimdXoshiro256PlusPlus::probability_threshold(0.25);
        let mut count = 0u32;

        for _ in 0..trials {
            let results = rng.check_probability_x4(threshold);
            count += results.iter().filter(|&&x| x).count() as u32;
        }

        let ratio = f64::from(count) / f64::from(trials * 4);
        assert!(
            (0.23..0.27).contains(&ratio),
            "check_probability_x4 ratio {ratio} out of range"
        );
    }

    #[test]
    fn test_count_occurrences_distribution() {
        let mut rng = SimdXoshiro256PlusPlus::seed_from_u64(42);

        // Test 1% probability over 100,000 trials
        let threshold = SimdXoshiro256PlusPlus::probability_threshold(0.01);
        let count = rng.count_occurrences(threshold, 100_000);

        let ratio = count as f64 / 100_000.0;
        assert!(
            (0.009..0.011).contains(&ratio),
            "count_occurrences ratio {ratio} out of range"
        );
    }

    #[test]
    fn test_count_occurrences_matches_scalar() {
        let mut rng1 = SimdXoshiro256PlusPlus::seed_from_u64(42);
        let mut rng2 = SimdXoshiro256PlusPlus::seed_from_u64(42);

        let threshold = SimdXoshiro256PlusPlus::probability_threshold(0.1);
        let count = 1000;

        // Count via bulk method
        let bulk_count = rng1.count_occurrences(threshold, count);

        // Count via scalar
        let scalar_count = (0..count)
            .filter(|_| rng2.check_probability(threshold))
            .count();

        assert_eq!(bulk_count, scalar_count);
    }
}
