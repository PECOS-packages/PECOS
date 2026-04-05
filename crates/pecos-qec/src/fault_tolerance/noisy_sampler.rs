//! Noisy Measurement Sampler using Precomputed Influence Maps
//!
//! This module provides efficient sampling of noisy measurement outcomes using
//! backward-propagated influence maps. Instead of simulating the full circuit
//! for each shot, we:
//!
//! 1. Precompute which fault locations affect which detectors/logicals
//! 2. For each shot, sample which faults fire
//! 3. Use O(1) lookups to find which detectors/logicals flip
//!
//! This approach is O(shots × `fault_locations`) instead of O(shots × `circuit_depth`),
//! providing significant speedup for noisy QEC simulations.
//!
//! # Example
//!
//! ```ignore
//! use pecos_qec::fault_tolerance::noisy_sampler::{NoisySampler, UniformNoiseModel};
//! use pecos_qec::fault_tolerance::propagator::DagFaultAnalyzer;
//!
//! // Build influence map (precomputation)
//! let analyzer = DagFaultAnalyzer::new(&dag);
//! let influence_map = analyzer.build_influence_map();
//!
//! // Create noise model (uniform depolarizing)
//! let noise = UniformNoiseModel::depolarizing(0.001);
//!
//! // Sample many shots
//! let mut sampler = NoisySampler::new(&influence_map, noise, 42);
//! let results = sampler.sample(10000);
//!
//! // Analyze results
//! for shot in &results {
//!     if shot.has_logical_error() {
//!         // ...
//!     }
//! }
//! ```

use super::propagator::Pauli;
use super::propagator::dag::DagFaultInfluenceMap;
use pecos_random::rng_ext::RngProbabilityExt;
use pecos_random::{PecosRng, Rng};
use std::collections::BTreeSet;

/// Result from a single shot of noisy sampling.
#[derive(Debug, Clone)]
pub struct ShotResult {
    /// Detectors that flipped (indices into `influence_map.detectors`).
    pub detector_flips: Vec<u32>,
    /// Logicals that flipped (indices).
    pub logical_flips: Vec<u32>,
    /// Number of faults that fired in this shot.
    pub fault_count: usize,
}

impl ShotResult {
    /// Create an empty result.
    #[must_use]
    pub fn new() -> Self {
        Self {
            detector_flips: Vec::new(),
            logical_flips: Vec::new(),
            fault_count: 0,
        }
    }

    /// Check if any logical error occurred.
    #[inline]
    #[must_use]
    pub fn has_logical_error(&self) -> bool {
        !self.logical_flips.is_empty()
    }

    /// Check if any syndrome was triggered.
    #[inline]
    #[must_use]
    pub fn has_syndrome(&self) -> bool {
        !self.detector_flips.is_empty()
    }

    /// Check if this is an undetectable logical error.
    #[inline]
    #[must_use]
    pub fn is_undetectable_logical_error(&self) -> bool {
        self.has_logical_error() && !self.has_syndrome()
    }
}

impl Default for ShotResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for noise models that can sample faults at each location.
pub trait NoiseModel {
    /// Sample a Pauli fault at the given location.
    ///
    /// Returns `Pauli::I` for no fault, or `X/Y/Z` for a fault.
    fn sample_fault(&mut self, loc_idx: usize, rng: &mut impl Rng) -> Pauli;

    /// Get the total error probability at a location (for statistics).
    fn error_probability(&self, loc_idx: usize) -> f64;
}

/// Uniform depolarizing noise model.
///
/// Same error probability at every location. With probability p,
/// applies X, Y, or Z with equal probability (p/3 each).
#[derive(Debug, Clone)]
pub struct UniformNoiseModel {
    /// Total error probability per location.
    p_error: f64,
    /// Threshold for error occurrence (`p_error` * `u64::MAX`).
    threshold: u64,
}

impl UniformNoiseModel {
    /// Create a uniform noise model with the given error probability.
    #[must_use]
    pub fn new(p_error: f64) -> Self {
        #[allow(
            clippy::cast_sign_loss,
            clippy::cast_possible_truncation,
            clippy::cast_precision_loss
        )]
        // probability in [0,1] so product fits in u64
        let threshold = (p_error * u64::MAX as f64) as u64;
        Self { p_error, threshold }
    }

    /// Create a depolarizing noise model (convenience alias).
    #[must_use]
    pub fn depolarizing(p_error: f64) -> Self {
        Self::new(p_error)
    }
}

impl NoiseModel for UniformNoiseModel {
    fn sample_fault(&mut self, _loc_idx: usize, rng: &mut impl Rng) -> Pauli {
        let rand = rng.next_u64();
        if rand < self.threshold {
            // Error occurred, sample which Pauli
            match (rng.next_u32() % 3) as u8 {
                0 => Pauli::X,
                1 => Pauli::Y,
                _ => Pauli::Z,
            }
        } else {
            Pauli::I
        }
    }

    fn error_probability(&self, _loc_idx: usize) -> f64 {
        self.p_error
    }
}

/// Per-location noise model with different probabilities.
///
/// Each location can have different error rates.
#[derive(Debug, Clone)]
pub struct PerLocationNoiseModel {
    /// Error probabilities per location.
    probabilities: Vec<f64>,
    /// Precomputed thresholds.
    thresholds: Vec<u64>,
}

impl PerLocationNoiseModel {
    /// Create from a vector of error probabilities (one per location).
    #[must_use]
    pub fn new(probabilities: Vec<f64>) -> Self {
        let thresholds = probabilities
            .iter()
            .map(|&p| {
                #[allow(
                    clippy::cast_sign_loss,
                    clippy::cast_possible_truncation,
                    clippy::cast_precision_loss
                )]
                // probability in [0,1] so product fits in u64
                {
                    (p * u64::MAX as f64) as u64
                }
            })
            .collect();
        Self {
            probabilities,
            thresholds,
        }
    }
}

impl NoiseModel for PerLocationNoiseModel {
    fn sample_fault(&mut self, loc_idx: usize, rng: &mut impl Rng) -> Pauli {
        let threshold = self.thresholds.get(loc_idx).copied().unwrap_or(0);
        let rand = rng.next_u64();
        if rand < threshold {
            match (rng.next_u32() % 3) as u8 {
                0 => Pauli::X,
                1 => Pauli::Y,
                _ => Pauli::Z,
            }
        } else {
            Pauli::I
        }
    }

    fn error_probability(&self, loc_idx: usize) -> f64 {
        self.probabilities.get(loc_idx).copied().unwrap_or(0.0)
    }
}

/// Noisy measurement sampler using precomputed influence maps.
///
/// This provides efficient sampling by using O(1) lookups from the
/// influence map instead of full circuit simulation.
pub struct NoisySampler<'a, N: NoiseModel> {
    /// Reference to the precomputed influence map.
    influence_map: &'a DagFaultInfluenceMap,
    /// Noise model for sampling faults.
    noise_model: N,
    /// Random number generator.
    rng: PecosRng,
    /// Number of fault locations.
    num_locations: usize,
    /// Number of detectors.
    num_detectors: usize,
    /// Number of logicals (derived from influence map).
    num_logicals: usize,
}

impl<'a, N: NoiseModel> NoisySampler<'a, N> {
    /// Create a new noisy sampler.
    ///
    /// # Arguments
    /// * `influence_map` - Precomputed influence map from backward propagation
    /// * `noise_model` - Noise model for sampling faults
    /// * `seed` - RNG seed for reproducibility
    pub fn new(influence_map: &'a DagFaultInfluenceMap, noise_model: N, seed: u64) -> Self {
        let num_locations = influence_map.locations.len();
        let num_detectors = influence_map.detectors.len();

        // Estimate num_logicals from the influence map
        // (could be stored explicitly in the map)
        let num_logicals = influence_map
            .influences
            .max_logical_index()
            .map_or(0, |i| i + 1);

        Self {
            influence_map,
            noise_model,
            rng: PecosRng::seed_from_u64(seed),
            num_locations,
            num_detectors,
            num_logicals,
        }
    }

    /// Sample a single shot.
    pub fn sample_one(&mut self) -> ShotResult {
        // Track which detectors/logicals have flipped (using XOR)
        let mut detector_flip_counts: Vec<u32> = vec![0; self.num_detectors];
        let mut logical_flip_counts: Vec<u32> = vec![0; self.num_logicals.max(1)];
        let mut fault_count = 0;

        // Sample each fault location
        for loc_idx in 0..self.num_locations {
            let pauli = self.noise_model.sample_fault(loc_idx, &mut self.rng);

            if pauli != Pauli::I {
                fault_count += 1;

                // Get affected detectors (O(1) lookup)
                let detectors = self
                    .influence_map
                    .get_detector_indices(loc_idx, pauli.as_u8());
                for &det_idx in detectors {
                    detector_flip_counts[det_idx as usize] ^= 1;
                }

                // Get affected logicals (O(1) lookup)
                let logicals = self
                    .influence_map
                    .get_logical_indices(loc_idx, pauli.as_u8());
                for &log_idx in logicals {
                    if (log_idx as usize) < logical_flip_counts.len() {
                        logical_flip_counts[log_idx as usize] ^= 1;
                    }
                }
            }
        }

        // Collect indices that ended up flipped (odd count)
        let detector_flips: Vec<u32> = detector_flip_counts
            .iter()
            .enumerate()
            .filter(|(_, c)| **c == 1)
            .map(|(i, _)| {
                #[allow(clippy::cast_possible_truncation)] // detector index fits in u32
                {
                    i as u32
                }
            })
            .collect();

        let logical_flips: Vec<u32> = logical_flip_counts
            .iter()
            .enumerate()
            .filter(|(_, c)| **c == 1)
            .map(|(i, _)| {
                #[allow(clippy::cast_possible_truncation)] // logical index fits in u32
                {
                    i as u32
                }
            })
            .collect();

        ShotResult {
            detector_flips,
            logical_flips,
            fault_count,
        }
    }

    /// Sample multiple shots.
    pub fn sample(&mut self, num_shots: usize) -> Vec<ShotResult> {
        (0..num_shots).map(|_| self.sample_one()).collect()
    }

    /// Sample and compute statistics.
    pub fn sample_statistics(&mut self, num_shots: usize) -> SamplingStatistics {
        let mut stats = SamplingStatistics::new();

        for _ in 0..num_shots {
            let result = self.sample_one();
            stats.record(&result);
        }

        stats
    }

    /// Get the number of fault locations.
    pub fn num_locations(&self) -> usize {
        self.num_locations
    }

    /// Get the number of detectors.
    pub fn num_detectors(&self) -> usize {
        self.num_detectors
    }
}

/// Statistics from sampling.
#[derive(Debug, Clone)]
pub struct SamplingStatistics {
    /// Total number of shots.
    pub total_shots: usize,
    /// Number of shots with logical errors.
    pub logical_error_count: usize,
    /// Number of shots with syndromes (detector flips).
    pub syndrome_count: usize,
    /// Number of undetectable logical errors.
    pub undetectable_count: usize,
    /// Total faults across all shots.
    pub total_faults: usize,
}

impl SamplingStatistics {
    /// Create empty statistics.
    #[must_use]
    pub fn new() -> Self {
        Self {
            total_shots: 0,
            logical_error_count: 0,
            syndrome_count: 0,
            undetectable_count: 0,
            total_faults: 0,
        }
    }

    /// Record a shot result.
    pub fn record(&mut self, result: &ShotResult) {
        self.total_shots += 1;
        self.total_faults += result.fault_count;

        if result.has_logical_error() {
            self.logical_error_count += 1;
        }
        if result.has_syndrome() {
            self.syndrome_count += 1;
        }
        if result.is_undetectable_logical_error() {
            self.undetectable_count += 1;
        }
    }

    /// Logical error rate.
    #[must_use]
    #[allow(clippy::cast_precision_loss)] // rate calculation
    pub fn logical_error_rate(&self) -> f64 {
        if self.total_shots == 0 {
            0.0
        } else {
            self.logical_error_count as f64 / self.total_shots as f64
        }
    }

    /// Syndrome rate (fraction of shots with non-trivial syndrome).
    #[must_use]
    #[allow(clippy::cast_precision_loss)] // rate calculation
    pub fn syndrome_rate(&self) -> f64 {
        if self.total_shots == 0 {
            0.0
        } else {
            self.syndrome_count as f64 / self.total_shots as f64
        }
    }

    /// Undetectable error rate.
    #[must_use]
    #[allow(clippy::cast_precision_loss)] // rate calculation
    pub fn undetectable_rate(&self) -> f64 {
        if self.total_shots == 0 {
            0.0
        } else {
            self.undetectable_count as f64 / self.total_shots as f64
        }
    }

    /// Average faults per shot.
    #[must_use]
    #[allow(clippy::cast_precision_loss)] // rate calculation
    pub fn average_faults(&self) -> f64 {
        if self.total_shots == 0 {
            0.0
        } else {
            self.total_faults as f64 / self.total_shots as f64
        }
    }
}

impl Default for SamplingStatistics {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Optimized Sampler using PecosRng batching and sparse tracking
// ============================================================================

/// Optimized noisy sampler using `PecosRng` batching and sparse flip tracking.
///
/// Key optimizations over [`NoisySampler`]:
/// 1. Uses `check_probability_indices()` to get sparse list of fault locations
/// 2. Uses `BTreeSet` for flip tracking instead of dense `Vec<u32>`
/// 3. Reuses buffers across shots to avoid allocation overhead
///
/// For low error rates (p < 0.01), this can be 2-3x faster than the standard sampler.
pub struct FastNoisySampler<'a> {
    /// Reference to the precomputed influence map.
    influence_map: &'a DagFaultInfluenceMap,
    /// Error probability.
    p_error: f64,
    /// Precomputed threshold for probability check.
    threshold: u64,
    /// Random number generator (`PecosRng` for batching).
    rng: PecosRng,
    /// Number of fault locations.
    num_locations: usize,
    /// Number of logicals.
    num_logicals: usize,
    /// Reusable buffer for detector flips (sparse).
    detector_flips_buffer: BTreeSet<u32>,
    /// Reusable buffer for logical flips (sparse).
    logical_flips_buffer: BTreeSet<u32>,
}

impl<'a> FastNoisySampler<'a> {
    /// Create a new optimized sampler.
    ///
    /// # Arguments
    /// * `influence_map` - Precomputed influence map
    /// * `p_error` - Uniform depolarizing error probability
    /// * `seed` - RNG seed
    #[must_use]
    pub fn new(influence_map: &'a DagFaultInfluenceMap, p_error: f64, seed: u64) -> Self {
        let num_locations = influence_map.locations.len();
        let num_logicals = influence_map
            .influences
            .max_logical_index()
            .map_or(0, |i| i + 1);

        let rng = PecosRng::seed_from_u64(seed);
        let threshold = rng.probability_threshold(p_error);

        Self {
            influence_map,
            p_error,
            threshold,
            rng,
            num_locations,
            num_logicals,
            detector_flips_buffer: BTreeSet::new(),
            logical_flips_buffer: BTreeSet::new(),
        }
    }

    /// Sample a single shot using sparse tracking.
    pub fn sample_one(&mut self) -> ShotResult {
        // Clear reusable buffers
        self.detector_flips_buffer.clear();
        self.logical_flips_buffer.clear();

        // Get sparse list of fault locations using batched RNG
        let fault_indices = self
            .rng
            .check_probability_indices(self.threshold, self.num_locations);

        let fault_count = fault_indices.len();

        // Process only the faulted locations
        for loc_idx in fault_indices {
            // Select Pauli type: 0=X, 1=Y, 2=Z
            let pauli_idx = self.rng.random_index_3();
            let pauli = match pauli_idx {
                0 => Pauli::X,
                1 => Pauli::Y,
                _ => Pauli::Z,
            };

            // Toggle affected detectors (XOR via HashSet toggle)
            let detectors = self
                .influence_map
                .get_detector_indices(loc_idx, pauli.as_u8());
            for &det_idx in detectors {
                if !self.detector_flips_buffer.remove(&det_idx) {
                    self.detector_flips_buffer.insert(det_idx);
                }
            }

            // Toggle affected logicals
            let logicals = self
                .influence_map
                .get_logical_indices(loc_idx, pauli.as_u8());
            for &log_idx in logicals {
                if (log_idx as usize) < self.num_logicals
                    && !self.logical_flips_buffer.remove(&log_idx)
                {
                    self.logical_flips_buffer.insert(log_idx);
                }
            }
        }

        // Collect results
        let detector_flips: Vec<u32> = self.detector_flips_buffer.iter().copied().collect();
        let logical_flips: Vec<u32> = self.logical_flips_buffer.iter().copied().collect();

        ShotResult {
            detector_flips,
            logical_flips,
            fault_count,
        }
    }

    /// Sample multiple shots.
    pub fn sample(&mut self, num_shots: usize) -> Vec<ShotResult> {
        (0..num_shots).map(|_| self.sample_one()).collect()
    }

    /// Sample and compute statistics.
    pub fn sample_statistics(&mut self, num_shots: usize) -> SamplingStatistics {
        let mut stats = SamplingStatistics::new();

        for _ in 0..num_shots {
            let result = self.sample_one();
            stats.record(&result);
        }

        stats
    }

    /// Get the error probability.
    #[must_use]
    pub fn p_error(&self) -> f64 {
        self.p_error
    }

    /// Get the number of fault locations.
    #[must_use]
    pub fn num_locations(&self) -> usize {
        self.num_locations
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock influence map for testing
    #[allow(dead_code)]
    fn create_test_influence_map() -> DagFaultInfluenceMap {
        DagFaultInfluenceMap::with_capacity(0)
    }

    #[test]
    fn test_uniform_noise_model() {
        let mut noise = UniformNoiseModel::new(0.5);
        let mut rng = PecosRng::seed_from_u64(42);

        let mut error_count = 0;
        for _ in 0..1000 {
            if noise.sample_fault(0, &mut rng) != Pauli::I {
                error_count += 1;
            }
        }

        // Should be roughly 50%
        assert!(error_count > 400 && error_count < 600);
    }

    #[test]
    fn test_per_location_noise_model() {
        let probs = vec![0.0, 0.5, 1.0];
        let mut noise = PerLocationNoiseModel::new(probs);
        let mut rng = PecosRng::seed_from_u64(42);

        // Location 0: never errors
        for _ in 0..100 {
            assert_eq!(noise.sample_fault(0, &mut rng), Pauli::I);
        }

        // Location 2: always errors
        for _ in 0..100 {
            assert_ne!(noise.sample_fault(2, &mut rng), Pauli::I);
        }
    }

    #[test]
    fn test_shot_result() {
        let mut result = ShotResult::new();
        assert!(!result.has_logical_error());
        assert!(!result.has_syndrome());

        result.logical_flips.push(0);
        assert!(result.has_logical_error());
        assert!(result.is_undetectable_logical_error());

        result.detector_flips.push(0);
        assert!(!result.is_undetectable_logical_error());
    }

    #[test]
    fn test_statistics() {
        let mut stats = SamplingStatistics::new();

        // Shot with no errors
        stats.record(&ShotResult::new());

        // Shot with syndrome only
        let mut shot_with_syndrome = ShotResult::new();
        shot_with_syndrome.detector_flips.push(0);
        stats.record(&shot_with_syndrome);

        // Shot with logical error
        let mut shot_with_logical = ShotResult::new();
        shot_with_logical.logical_flips.push(0);
        shot_with_logical.detector_flips.push(1);
        stats.record(&shot_with_logical);

        // Shot with undetectable logical
        let mut shot_undetectable = ShotResult::new();
        shot_undetectable.logical_flips.push(0);
        stats.record(&shot_undetectable);

        assert_eq!(stats.total_shots, 4);
        assert_eq!(stats.logical_error_count, 2);
        assert_eq!(stats.syndrome_count, 2);
        assert_eq!(stats.undetectable_count, 1);
    }
}
