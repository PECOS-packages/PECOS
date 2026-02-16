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

//! Scalar-optimized RNG for low-latency single-value operations.
//!
//! [`PecosScalarRng`] is optimized for use cases where you need low-latency scalar operations
//! and cannot batch multiple operations together. It uses a dedicated scalar RNG with no
//! buffering overhead.
//!
//! # When to use `PecosScalarRng` vs `PecosRng`
//!
//! | Use Case | Recommended RNG | Reason |
//! |----------|-----------------|--------|
//! | Scalar probability checks (one at a time) | `PecosScalarRng` | 36% faster, no buffer overhead |
//! | Scalar f64 generation | `PecosScalarRng` | 32% faster, no buffer overhead |
//! | Batched probability checks | Either | Both use parallel RNGs, same performance |
//! | Tight u64 loops | `PecosRng` | Buffering amortizes RNG overhead |
//! | Bulk fill operations | Either | Both use parallel RNGs |
//!
//! # Example
//!
//! ```
//! use pecos_rng::{PecosScalarRng, RngProbabilityExt};
//!
//! let mut rng = PecosScalarRng::seed_from_u64(42);
//!
//! // Low-latency scalar probability check
//! let threshold = PecosScalarRng::probability_threshold(0.001);
//! if rng.check_probability(threshold) {
//!     println!("Event occurred!");
//! }
//! ```

use core::convert::Infallible;
use rand_core::{SeedableRng, TryRng};
use rapidhash::rng::RapidRng;
use wide::u64x4;

/// Scalar-optimized RNG with dedicated scalar RNG and parallel RNGs for bulk operations.
///
/// This RNG is optimized for scalar operations (probability checks, f64 generation) where
/// you need immediate results and cannot batch operations. It avoids the buffer overhead
/// of [`crate::PecosRng`] for scalar operations while still providing parallel RNGs
/// for bulk operations.
///
/// See the [module documentation](self) for guidance on when to use this vs `PecosRng`.
#[derive(Clone, Debug)]
pub struct PecosScalarRng {
    /// Dedicated scalar RNG - no buffering overhead.
    scalar_rng: RapidRng,
    /// 4 parallel `RapidRng` generators for bulk operations only.
    parallel_rngs: [RapidRng; 4],
    /// Buffer for bit-packed bool generation.
    bool_bits: u64,
    /// Remaining bits in bool buffer (0 = empty, 1-64 = valid).
    bool_remaining: u8,
}

impl PartialEq for PecosScalarRng {
    fn eq(&self, other: &Self) -> bool {
        self.scalar_rng == other.scalar_rng && self.parallel_rngs == other.parallel_rngs
    }
}

impl PecosScalarRng {
    /// Create a new `PecosScalarRng` from a seed.
    #[must_use]
    pub fn seed_from_u64(seed: u64) -> Self {
        let mut splitmix = SplitMix64::new(seed);

        Self {
            scalar_rng: RapidRng::new(splitmix.next_u64()),
            parallel_rngs: [
                RapidRng::new(splitmix.next_u64()),
                RapidRng::new(splitmix.next_u64()),
                RapidRng::new(splitmix.next_u64()),
                RapidRng::new(splitmix.next_u64()),
            ],
            bool_bits: 0,
            bool_remaining: 0,
        }
    }

    /// Generate 4 random u64 values simultaneously using parallel RNGs.
    #[inline]
    pub fn next_u64x4(&mut self) -> u64x4 {
        u64x4::new([
            self.parallel_rngs[0].next(),
            self.parallel_rngs[1].next(),
            self.parallel_rngs[2].next(),
            self.parallel_rngs[3].next(),
        ])
    }

    /// Fill a slice with random u64 values using parallel RNGs.
    pub fn fill_u64(&mut self, dest: &mut [u64]) {
        let mut chunks = dest.chunks_exact_mut(4);

        for chunk in chunks.by_ref() {
            let values = self.next_u64x4();
            let array: [u64; 4] = values.into();
            chunk.copy_from_slice(&array);
        }

        // Handle remainder using scalar RNG
        let remainder = chunks.into_remainder();
        for val in remainder {
            *val = self.scalar_rng.next();
        }
    }

    /// Generate a single random u64 using dedicated scalar RNG.
    ///
    /// No buffering - direct `RapidRng` access for minimal overhead.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.scalar_rng.next()
    }

    /// Generate a single random u32.
    #[inline]
    #[allow(clippy::cast_possible_truncation)]
    pub fn next_u32(&mut self) -> u32 {
        self.scalar_rng.next() as u32
    }

    /// Generate a random f64 in [0, 1).
    #[inline]
    #[allow(clippy::cast_precision_loss)]
    pub fn next_f64(&mut self) -> f64 {
        (self.scalar_rng.next() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Generate a random bool using bit-packed extraction.
    #[inline]
    pub fn next_bool_fast(&mut self) -> bool {
        if self.bool_remaining == 0 {
            self.bool_bits = self.scalar_rng.next();
            self.bool_remaining = 64;
        }
        self.bool_remaining -= 1;
        (self.bool_bits >> self.bool_remaining) & 1 != 0
    }

    /// Generate a random bool with probability `p` of being true.
    #[inline]
    #[allow(clippy::float_cmp)]
    pub fn random_bool(&mut self, p: f64) -> bool {
        if p == 0.5 {
            self.next_bool_fast()
        } else {
            self.next_f64() < p
        }
    }

    /// Convert a probability to a u64 threshold.
    #[inline]
    #[must_use]
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    pub fn probability_threshold(p: f64) -> u64 {
        (p * (u64::MAX as f64)) as u64
    }

    /// Check if a random event occurs with the given precomputed probability threshold.
    #[inline]
    pub fn check_probability(&mut self, threshold: u64) -> bool {
        self.scalar_rng.next() < threshold
    }

    /// Check 4 probabilities at once using parallel RNGs.
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

    /// Count how many events occur out of `count` checks.
    #[inline]
    pub fn count_occurrences(&mut self, threshold: u64, count: usize) -> usize {
        let mut total = 0usize;

        let full_chunks = count / 4;
        for _ in 0..full_chunks {
            let values = self.next_u64x4();
            let arr: [u64; 4] = values.into();
            total += usize::from(arr[0] < threshold);
            total += usize::from(arr[1] < threshold);
            total += usize::from(arr[2] < threshold);
            total += usize::from(arr[3] < threshold);
        }

        for _ in 0..(count % 4) {
            total += usize::from(self.scalar_rng.next() < threshold);
        }

        total
    }

    /// Return indices where probability check succeeded, using parallel RNGs.
    ///
    /// This is optimized for low-probability events (like noise in quantum circuits)
    /// where you need to know which indices had events, not just how many.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    pub fn check_probability_indices(&mut self, threshold: u64, count: usize) -> Vec<usize> {
        let expected_hits = ((count as f64) * (threshold as f64 / u64::MAX as f64) * 2.0) as usize;
        let mut indices = Vec::with_capacity(expected_hits.max(16));

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

        let remainder_start = full_chunks * 4;
        for i in 0..(count % 4) {
            if self.scalar_rng.next() < threshold {
                indices.push(remainder_start + i);
            }
        }

        indices
    }
}

// ============================================================================
// rand_core trait implementations
// ============================================================================

impl TryRng for PecosScalarRng {
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
        let mut chunks = dest.chunks_exact_mut(8);
        for chunk in chunks.by_ref() {
            let bytes = self.scalar_rng.next().to_le_bytes();
            chunk.copy_from_slice(&bytes);
        }

        let remainder = chunks.into_remainder();
        if !remainder.is_empty() {
            let bytes = self.scalar_rng.next().to_le_bytes();
            for (i, byte) in remainder.iter_mut().enumerate() {
                *byte = bytes[i];
            }
        }
        Ok(())
    }
}

impl SeedableRng for PecosScalarRng {
    type Seed = [u8; 32];

    fn from_seed(seed: Self::Seed) -> Self {
        let mut seed_u64 = 0u64;
        for (i, chunk) in seed.chunks(8).enumerate() {
            if i >= 1 {
                break;
            }
            let mut bytes = [0u8; 8];
            bytes[..chunk.len()].copy_from_slice(chunk);
            seed_u64 ^= u64::from_le_bytes(bytes);
        }
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

// ============================================================================
// SplitMix64 for seed derivation
// ============================================================================

#[derive(Clone, Copy, Debug)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    #[inline]
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

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
    fn test_deterministic() {
        let mut rng1 = PecosScalarRng::seed_from_u64(42);
        let mut rng2 = PecosScalarRng::seed_from_u64(42);

        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn test_different_seeds() {
        let mut rng1 = PecosScalarRng::seed_from_u64(1);
        let mut rng2 = PecosScalarRng::seed_from_u64(2);

        assert_ne!(rng1.next_u64(), rng2.next_u64());
    }

    #[test]
    fn test_fill_u64() {
        let mut rng = PecosScalarRng::seed_from_u64(42);

        let mut buffer = vec![0u64; 100];
        rng.fill_u64(&mut buffer);

        let non_zero = buffer.iter().filter(|&&x| x != 0).count();
        assert!(non_zero > 95, "Expected most values to be non-zero");
    }

    #[test]
    fn test_next_f64_range() {
        let mut rng = PecosScalarRng::seed_from_u64(42);

        for _ in 0..1000 {
            let f = rng.next_f64();
            assert!((0.0..1.0).contains(&f), "f64 out of range: {f}");
        }
    }

    #[test]
    fn test_bool_distribution() {
        let mut rng = PecosScalarRng::seed_from_u64(42);
        let mut true_count = 0u32;
        let trials = 10_000;

        for _ in 0..trials {
            if rng.next_bool_fast() {
                true_count += 1;
            }
        }

        let ratio = f64::from(true_count) / f64::from(trials);
        assert!(
            (0.45..0.55).contains(&ratio),
            "Bool distribution out of range: {ratio}"
        );
    }

    #[test]
    fn test_check_probability_distribution() {
        let mut rng = PecosScalarRng::seed_from_u64(42);
        let trials = 100_000;

        let threshold = PecosScalarRng::probability_threshold(0.001);
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
}
