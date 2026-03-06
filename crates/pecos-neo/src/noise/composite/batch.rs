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

//! Geometric sampling for batch noise processing.
//!
//! This module provides the core geometric distribution sampling mechanism used
//! by batch noise processors. For low-probability events (p < 0.01), geometric
//! sampling achieves O(n*p) complexity instead of O(n).
//!
//! # Performance
//!
//! For 1M qubits:
//!
//! | Probability | Time | Expected Events |
//! |-------------|------|-----------------|
//! | p = 1e-3 | ~10 µs | ~1,000 |
//! | p = 1e-4 | ~1 µs | ~100 |
//! | p = 1e-5 | ~200 ns | ~10 |
//!
//! # Design
//!
//! The core insight is that for Bernoulli trials with probability p, the number
//! of trials until the next success follows a geometric distribution. Instead of
//! checking each qubit individually, we sample how many to skip directly.
//!
//! ```text
//! Linear approach (O(n)):
//!   for each of 1M qubits:
//!     if random() < p: yield qubit    // 1M RNG calls
//!
//! Geometric approach (O(n*p)):
//!   while position < n:
//!     skip = sample_geometric(p)      // Only ~n*p RNG calls
//!     position += skip
//!     if position < n: yield qubit
//! ```
//!
//! # Usage
//!
//! For high-level batch noise processing, see [`batch_composite`](super::batch_composite).
//! This module provides the low-level sampling primitives.
//!
//! ```
//! use pecos_neo::noise::composite::batch::GeometricSampler;
//! use pecos_rng::PecosRng;
//!
//! let mut rng = PecosRng::seed_from_u64(42);
//!
//! // Create a sampler for p=1e-4 error rate
//! let sampler = GeometricSampler::new(1e-4);
//!
//! // Find affected qubits in range [0, 1_000_000)
//! let affected = sampler.sample_range(0, 1_000_000, &mut rng);
//!
//! // Process only the ~100 affected qubits
//! for qubit in affected {
//!     // Apply noise...
//! }
//! ```

use pecos_core::QubitId;
use pecos_rng::PecosRng;
use pecos_rng::rng_ext::RngProbabilityExt;
use rand::RngExt;

// ============================================================================
// Geometric Sampler - The Core Mechanism
// ============================================================================

/// High-performance geometric distribution sampler for sparse event generation.
///
/// This is the central mechanism for batch noise processing. It samples from
/// a geometric distribution to determine which indices experience events,
/// achieving O(n*p) complexity instead of O(n) for low probabilities.
///
/// # When to Use
///
/// - **Low probability events** (p < 0.01): Geometric is dramatically faster
/// - **High entity counts** (n > 1000): Amortizes setup cost
/// - **Sparse results needed**: Returns only affected indices
///
/// For p >= 0.1 or n < 100, linear scanning may be faster due to lower overhead.
#[derive(Debug, Clone, Copy)]
pub struct GeometricSampler {
    /// The probability of each event.
    probability: f64,
    /// Precomputed log(1-p) for geometric sampling.
    log_1_minus_p: f64,
    /// u64 threshold for fast probability checks.
    threshold_u64: u64,
}

impl GeometricSampler {
    /// Create a new geometric sampler for the given probability.
    ///
    /// # Arguments
    /// * `probability` - Event probability in range (0.0, 1.0)
    ///
    /// # Panics
    /// Panics if probability is not in (0.0, 1.0).
    #[must_use]
    pub fn new(probability: f64) -> Self {
        assert!(
            probability > 0.0 && probability < 1.0,
            "Probability must be in (0, 1), got {probability}"
        );

        Self {
            probability,
            log_1_minus_p: (1.0 - probability).ln(),
            threshold_u64: (probability * (u64::MAX as f64)) as u64,
        }
    }

    /// Create a sampler that handles edge cases (p=0 or p=1).
    #[must_use]
    pub fn new_checked(probability: f64) -> Option<Self> {
        if probability <= 0.0 || probability >= 1.0 {
            None
        } else {
            Some(Self::new(probability))
        }
    }

    /// Get the probability.
    #[must_use]
    pub fn probability(&self) -> f64 {
        self.probability
    }

    /// Get the u64 threshold for direct comparison.
    #[must_use]
    pub fn threshold(&self) -> u64 {
        self.threshold_u64
    }

    /// Sample the next event position starting from `start`.
    ///
    /// Returns the index of the next event, or None if it exceeds `end`.
    #[inline]
    pub fn next_event(&self, start: usize, end: usize, rng: &mut PecosRng) -> Option<usize> {
        let u: f64 = rng.random();
        let skip = if u > 0.0 {
            (u.ln() / self.log_1_minus_p).floor() as usize
        } else {
            0
        };

        let pos = start + skip;
        if pos < end { Some(pos) } else { None }
    }

    /// Sample all events in the range [0, count).
    ///
    /// Returns a vector of indices where events occurred.
    #[inline]
    pub fn sample_range(&self, start: usize, end: usize, rng: &mut PecosRng) -> Vec<usize> {
        // Pre-allocate based on expected events (2x to avoid reallocation)
        #[allow(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss
        )]
        let expected = (((end - start) as f64) * self.probability * 2.0) as usize;
        let mut result = Vec::with_capacity(expected.max(16));

        let mut idx = start;
        while idx < end {
            let u: f64 = rng.random();
            let skip = if u > 0.0 {
                (u.ln() / self.log_1_minus_p).floor() as usize
            } else {
                0
            };

            idx += skip;
            if idx < end {
                result.push(idx);
                idx += 1;
            }
        }

        result
    }

    /// Sample events and return as `QubitIds`.
    #[inline]
    pub fn sample_qubits(&self, start: usize, end: usize, rng: &mut PecosRng) -> Vec<QubitId> {
        self.sample_range(start, end, rng)
            .into_iter()
            .map(QubitId)
            .collect()
    }

    /// Sample events from a slice of qubits.
    #[inline]
    pub fn sample_from_slice(&self, qubits: &[QubitId], rng: &mut PecosRng) -> Vec<QubitId> {
        self.sample_range(0, qubits.len(), rng)
            .into_iter()
            .map(|i| qubits[i])
            .collect()
    }

    /// Check if a single event occurs (for occasional single checks).
    #[inline]
    pub fn check_single(&self, rng: &mut PecosRng) -> bool {
        rng.next_u64() < self.threshold_u64
    }
}

// ============================================================================
// Legacy Compatibility Functions
// ============================================================================

/// Filter qubits by probability using geometric sampling.
///
/// This is the recommended function for batch probability filtering.
/// For probabilities < 0.01, uses geometric sampling (O(n*p)).
/// For higher probabilities, falls back to linear scanning.
pub fn filter_by_probability(
    qubits: &[QubitId],
    probability: f64,
    rng: &mut PecosRng,
) -> Vec<QubitId> {
    if probability <= 0.0 {
        return vec![];
    }
    if probability >= 1.0 {
        return qubits.to_vec();
    }

    // Use geometric for low probabilities with sufficient count
    if probability < 0.01 && qubits.len() > 100 {
        let sampler = GeometricSampler::new(probability);
        sampler.sample_from_slice(qubits, rng)
    } else {
        // Linear fallback
        qubits
            .iter()
            .filter(|_| rng.random::<f64>() < probability)
            .copied()
            .collect()
    }
}

/// Filter a range of qubit IDs by probability.
///
/// More efficient than `filter_by_probability` when you just need a contiguous range.
pub fn filter_range_by_probability(
    start: usize,
    end: usize,
    probability: f64,
    rng: &mut PecosRng,
) -> Vec<QubitId> {
    if probability <= 0.0 {
        return vec![];
    }
    if probability >= 1.0 {
        return (start..end).map(QubitId).collect();
    }

    let count = end - start;

    // Use geometric for low probabilities with sufficient count
    if probability < 0.01 && count > 100 {
        let sampler = GeometricSampler::new(probability);
        sampler.sample_qubits(start, end, rng)
    } else {
        // Linear fallback
        (start..end)
            .filter(|_| rng.random::<f64>() < probability)
            .map(QubitId)
            .collect()
    }
}

// ============================================================================
// RNG Extension Compatibility
// ============================================================================

/// Precomputed probability threshold for fast batch checks.
#[derive(Debug, Clone, Copy)]
pub struct ProbabilityThreshold {
    /// The u64 threshold value.
    pub threshold: u64,
    /// The original probability.
    pub probability: f64,
}

impl ProbabilityThreshold {
    /// Create a new probability threshold.
    #[must_use]
    pub fn new(probability: f64, rng: &PecosRng) -> Self {
        Self {
            threshold: rng.probability_threshold(probability),
            probability,
        }
    }
}

/// Fast batch probability filter using RNG extension trait.
pub fn filter_by_probability_fast(
    num_qubits: usize,
    threshold: &ProbabilityThreshold,
    rng: &mut PecosRng,
) -> Vec<QubitId> {
    rng.check_probability_indices(threshold.threshold, num_qubits)
        .into_iter()
        .map(QubitId)
        .collect()
}

/// Batch noise sampling using RNG extension trait.
pub fn sample_noise_1q_batch(
    num_qubits: usize,
    threshold: &ProbabilityThreshold,
    rng: &mut PecosRng,
) -> Vec<(QubitId, u8)> {
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    let expected = ((num_qubits as f64) * threshold.probability * 2.0) as usize;
    let mut results = Vec::with_capacity(expected.max(16));

    for i in 0..num_qubits {
        if let Some(pauli) = rng.noise_sample_1q(threshold.threshold) {
            results.push((QubitId(i), pauli));
        }
    }

    results
}

/// Batch two-qubit noise sampling.
pub fn sample_noise_2q_batch(
    num_qubits: usize,
    threshold: &ProbabilityThreshold,
    rng: &mut PecosRng,
) -> Vec<(QubitId, u8)> {
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    let expected = ((num_qubits as f64) * threshold.probability * 2.0) as usize;
    let mut results = Vec::with_capacity(expected.max(16));

    for i in 0..num_qubits {
        if let Some(pauli) = rng.noise_sample_2q(threshold.threshold) {
            results.push((QubitId(i), pauli));
        }
    }

    results
}

// ============================================================================
// Convenience Constructor
// ============================================================================

/// Create a geometric sampler for the given probability.
#[must_use]
pub fn geometric(probability: f64) -> GeometricSampler {
    GeometricSampler::new(probability)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_geometric_sampler_basic() {
        let sampler = GeometricSampler::new(0.1);
        let mut rng = PecosRng::seed_from_u64(42);

        let results = sampler.sample_range(0, 1000, &mut rng);

        // Should be roughly 10% (100 +/- 30)
        assert!(
            results.len() > 50 && results.len() < 150,
            "Expected ~100, got {}",
            results.len()
        );

        // Results should be sorted and unique
        for i in 1..results.len() {
            assert!(results[i] > results[i - 1], "Results should be sorted");
        }
    }

    #[test]
    fn test_geometric_sampler_low_probability() {
        let sampler = GeometricSampler::new(1e-4);
        let mut rng = PecosRng::seed_from_u64(42);

        let results = sampler.sample_range(0, 100_000, &mut rng);

        // Should be roughly 10 events
        assert!(
            results.len() > 2 && results.len() < 30,
            "Expected ~10, got {}",
            results.len()
        );
    }

    #[test]
    fn test_geometric_sampler_very_low_probability() {
        let sampler = GeometricSampler::new(1e-5);
        let mut rng = PecosRng::seed_from_u64(42);

        let results = sampler.sample_range(0, 1_000_000, &mut rng);

        // Should be roughly 10 events
        assert!(
            results.len() > 2 && results.len() < 30,
            "Expected ~10, got {}",
            results.len()
        );
    }

    #[test]
    fn test_filter_by_probability_zero() {
        let qubits: Vec<_> = (0..1000).map(QubitId).collect();
        let mut rng = PecosRng::seed_from_u64(42);

        let result = filter_by_probability(&qubits, 0.0, &mut rng);
        assert!(result.is_empty());
    }

    #[test]
    fn test_filter_by_probability_one() {
        let qubits: Vec<_> = (0..1000).map(QubitId).collect();
        let mut rng = PecosRng::seed_from_u64(42);

        let result = filter_by_probability(&qubits, 1.0, &mut rng);
        assert_eq!(result.len(), 1000);
    }

    #[test]
    fn test_filter_range_by_probability() {
        let mut rng = PecosRng::seed_from_u64(42);

        let result = filter_range_by_probability(0, 10000, 0.1, &mut rng);

        // Should be roughly 10%
        assert!(
            result.len() > 800 && result.len() < 1200,
            "Expected ~1000, got {}",
            result.len()
        );
    }

    #[test]
    fn test_probability_threshold() {
        let rng = PecosRng::seed_from_u64(42);
        let threshold = ProbabilityThreshold::new(0.001, &rng);

        assert!((threshold.probability - 0.001).abs() < 1e-10);
        assert!(threshold.threshold > 0);
    }

    #[test]
    fn test_filter_by_probability_fast() {
        let mut rng = PecosRng::seed_from_u64(42);
        let threshold = ProbabilityThreshold::new(0.01, &rng);

        let result = filter_by_probability_fast(10000, &threshold, &mut rng);

        // Should be roughly 1%
        assert!(
            result.len() > 50 && result.len() < 200,
            "Expected ~100, got {}",
            result.len()
        );
    }

    #[test]
    fn test_sample_noise_1q_batch() {
        let mut rng = PecosRng::seed_from_u64(42);
        let threshold = ProbabilityThreshold::new(0.01, &rng);

        let result = sample_noise_1q_batch(10000, &threshold, &mut rng);

        // All Paulis should be valid
        for (_, pauli) in &result {
            assert!(*pauli < 3, "Invalid Pauli: {pauli}");
        }
    }

    #[test]
    fn test_sample_noise_2q_batch() {
        let mut rng = PecosRng::seed_from_u64(42);
        let threshold = ProbabilityThreshold::new(0.01, &rng);

        let result = sample_noise_2q_batch(10000, &threshold, &mut rng);

        // All Paulis should be valid
        for (_, pauli) in &result {
            assert!(*pauli < 15, "Invalid 2Q Pauli: {pauli}");
        }
    }

    #[test]
    fn test_geometric_statistical_consistency() {
        // Run multiple times and verify statistical properties
        let sampler = GeometricSampler::new(0.01);
        let mut total_events = 0;

        for seed in 0..100 {
            let mut rng = PecosRng::seed_from_u64(seed);
            let results = sampler.sample_range(0, 10000, &mut rng);
            total_events += results.len();
        }

        // Expected: 100 runs * 10000 * 0.01 = 10000 events
        // Allow 10% tolerance
        let expected = 10000;
        assert!(
            total_events > expected * 9 / 10 && total_events < expected * 11 / 10,
            "Total events {total_events} outside expected range around {expected}"
        );
    }
}
