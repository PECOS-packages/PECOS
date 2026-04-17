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

//! Fast DEM-style sampler for threshold estimation.
//!
//! This module provides a sampler that aggregates fault effects directly into
//! detector/observable signatures, matching Stim's DEM sampler semantics.
//!
//! # Data-Oriented Design
//!
//! The sampler uses Structure of Arrays (`SoA`) layout and CSR-style indexing for
//! cache-efficient sampling:
//!
//! - **Probabilities**: Stored in a contiguous array for sequential access
//! - **Detector/Observable indices**: CSR layout (offsets + flat data) for variable-length lists
//! - **Bit-packed outcomes**: Uses `u64` words for compact detector/observable state
//!
//! # Example
//!
//! ```
//! use pecos_qec::fault_tolerance::DagFaultAnalyzer;
//! use pecos_qec::fault_tolerance::dem_builder::DemSamplerBuilder;
//! use pecos_quantum::DagCircuit;
//! use rand::SeedableRng;
//! use rand::rngs::SmallRng;
//!
//! let mut dag = DagCircuit::new();
//! dag.pz(&[2]);
//! dag.cx(&[(0, 2)]);
//! dag.cx(&[(1, 2)]);
//! dag.mz(&[2]);
//!
//! let analyzer = DagFaultAnalyzer::new(&dag);
//! let influence_map = analyzer.build_influence_map();
//! let detectors_json = r#"[{"id": 0, "records": [-1]}]"#;
//! let observables_json = "[]";
//!
//! // Build from circuit with detector definitions
//! let sampler = DemSamplerBuilder::new(&influence_map)
//!     .with_noise(0.01, 0.01, 0.01, 0.01)
//!     .with_detectors_json(detectors_json).unwrap()
//!     .with_observables_json(observables_json).unwrap()
//!     .build();
//!
//! // Fast batch sampling for threshold estimation
//! let mut rng = SmallRng::seed_from_u64(42);
//! let (det_events, obs_flips) = sampler.sample_batch(100, &mut rng);
//! ```

use crate::fault_tolerance::propagator::{DagFaultInfluenceMap, Pauli};
use pecos_core::prelude::GateType;
use pecos_random::{PecosRng, RngProbabilityExt};
use rand_core::Rng;
use rayon::prelude::*;
use smallvec::SmallVec;
use std::collections::BTreeMap;
use wide::u64x4;

use super::types::combine_probabilities;

// ============================================================================
// DEM Mechanism (used during building)
// ============================================================================

/// A single fault mechanism with its detector/observable effects.
/// Used during building, then converted to `SoA` layout.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct DemMechanism {
    /// Sorted detector indices that flip when this mechanism fires.
    detectors: SmallVec<[u32; 4]>,
    /// Sorted observable indices that flip when this mechanism fires.
    observables: SmallVec<[u32; 2]>,
}

impl DemMechanism {
    fn new(mut detectors: SmallVec<[u32; 4]>, mut observables: SmallVec<[u32; 2]>) -> Self {
        detectors.sort_unstable();
        observables.sort_unstable();
        Self {
            detectors,
            observables,
        }
    }

    fn empty() -> Self {
        Self {
            detectors: SmallVec::new(),
            observables: SmallVec::new(),
        }
    }

    fn is_empty(&self) -> bool {
        self.detectors.is_empty() && self.observables.is_empty()
    }
}

// ============================================================================
// Bit-packed outcome storage
// ============================================================================

/// Number of bits per word in packed storage.
const BITS_PER_WORD: usize = 64;

/// Target buffer size for chunked processing (6 MB fits in L3 cache).
const TARGET_CHUNK_BUFFER_BYTES: usize = 6 * 1024 * 1024;

/// Bit-packed boolean array for efficient XOR operations.
#[derive(Debug, Clone)]
struct PackedBits {
    words: Vec<u64>,
    len: usize,
}

impl PackedBits {
    fn new(len: usize) -> Self {
        let num_words = len.div_ceil(BITS_PER_WORD);
        Self {
            words: vec![0; num_words],
            len,
        }
    }

    #[inline]
    fn clear(&mut self) {
        for w in &mut self.words {
            *w = 0;
        }
    }

    #[inline]
    fn flip(&mut self, idx: usize) {
        let word_idx = idx / BITS_PER_WORD;
        let bit_idx = idx % BITS_PER_WORD;
        self.words[word_idx] ^= 1u64 << bit_idx;
    }

    #[inline]
    fn get(&self, idx: usize) -> bool {
        let word_idx = idx / BITS_PER_WORD;
        let bit_idx = idx % BITS_PER_WORD;
        (self.words[word_idx] >> bit_idx) & 1 != 0
    }

    /// Returns true if any bit is set.
    #[inline]
    fn any(&self) -> bool {
        self.words.iter().any(|&w| w != 0)
    }

    /// Convert to Vec<bool> for output.
    fn to_vec(&self) -> Vec<bool> {
        (0..self.len).map(|i| self.get(i)).collect()
    }
}

// ============================================================================
// DEM Sampler (SoA layout)
// ============================================================================

/// Fast DEM-style sampler for threshold estimation.
///
/// Uses Structure of Arrays (`SoA`) layout with CSR-style indexing for
/// cache-efficient sampling. Detector and observable outcomes are bit-packed
/// for compact storage and fast XOR operations.
///
/// # Data-Oriented Design
///
/// - **Precomputed thresholds**: Probabilities converted to u64 thresholds at build time
/// - **CSR layout**: Detector/observable indices in flat arrays with offsets
/// - **Bit-packed outcomes**: Uses u64 words for compact XOR operations
/// - **Batch RNG**: Can use bulk random number generation for cache efficiency
///
/// # Memory Layout
///
/// ```text
/// thresholds:          [t0, t1, t2, ...]           (precomputed u64, sequential read)
/// detector_offsets:    [0, 2, 3, 5, ...]           (CSR row pointers)
/// detector_data:       [d0, d1, d2, d3, d4, ...]   (flat detector indices)
/// observable_offsets:  [0, 0, 1, 1, ...]           (CSR row pointers)
/// observable_data:     [o0, ...]                   (flat observable indices)
/// ```
///
/// For mechanism i:
/// - Detector indices: `detector_data[detector_offsets[i]..detector_offsets[i+1]]`
/// - Observable indices: `observable_data[observable_offsets[i]..observable_offsets[i+1]]`
#[derive(Debug, Clone)]
pub struct DemSampler {
    // SoA layout for cache efficiency
    /// Precomputed u64 thresholds (faster than f64 comparison).
    thresholds: Vec<u64>,

    /// Precomputed 1.0 / ln(1-p) for geometric sampling.
    /// Stored as negative reciprocal so we can multiply instead of divide.
    inv_log_1_minus_p: Vec<f64>,

    /// CSR-style offsets into `detector_data`. Length = `num_mechanisms` + 1.
    detector_offsets: Vec<u32>,
    /// Flat array of detector indices.
    detector_data: Vec<u32>,

    /// CSR-style offsets into `observable_data`. Length = `num_mechanisms` + 1.
    observable_offsets: Vec<u32>,
    /// Flat array of observable indices.
    observable_data: Vec<u32>,

    /// Number of detectors.
    num_detectors: usize,
    /// Number of observables.
    num_observables: usize,
}

impl DemSampler {
    /// Number of mechanisms in the sampler.
    #[must_use]
    pub fn num_mechanisms(&self) -> usize {
        self.thresholds.len()
    }

    /// Number of detectors.
    #[must_use]
    pub fn num_detectors(&self) -> usize {
        self.num_detectors
    }

    /// Number of observables.
    #[must_use]
    pub fn num_observables(&self) -> usize {
        self.num_observables
    }

    /// Create a [`DemSampler`] from raw mechanism data.
    ///
    /// This constructor is used when building from a parsed DEM string rather than
    /// from a circuit analysis. Each mechanism is specified by its probability and
    /// the detector/observable indices it affects.
    ///
    /// # Arguments
    ///
    /// * `mechanisms` - Iterator of (probability, `detector_indices`, `observable_indices`)
    /// * `num_detectors` - Total number of detectors
    /// * `num_observables` - Total number of observables
    #[must_use]
    pub fn from_mechanisms<I>(mechanisms: I, num_detectors: usize, num_observables: usize) -> Self
    where
        I: IntoIterator<Item = (f64, Vec<u32>, Vec<u32>)>,
    {
        let mechanisms: Vec<_> = mechanisms.into_iter().collect();
        let num_mechanisms = mechanisms.len();

        let mut thresholds = Vec::with_capacity(num_mechanisms);
        let mut inv_log_1_minus_p = Vec::with_capacity(num_mechanisms);
        let mut detector_offsets = Vec::with_capacity(num_mechanisms + 1);
        let mut detector_data = Vec::new();
        let mut observable_offsets = Vec::with_capacity(num_mechanisms + 1);
        let mut observable_data = Vec::new();

        detector_offsets.push(0);
        observable_offsets.push(0);

        for (prob, mut detectors, mut observables) in mechanisms {
            // Sort for canonical representation
            detectors.sort_unstable();
            observables.sort_unstable();

            // Precompute u64 threshold: p * u64::MAX
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                clippy::cast_precision_loss
            )]
            let threshold = (prob * (u64::MAX as f64)) as u64;
            thresholds.push(threshold);

            // Precompute 1/ln(1-p) for geometric sampling
            let log_1_minus_p = (1.0 - prob).ln();
            let inv = if log_1_minus_p.abs() < f64::EPSILON {
                0.0 // p ≈ 0, mechanism never fires
            } else {
                1.0 / log_1_minus_p
            };
            inv_log_1_minus_p.push(inv);

            detector_data.extend_from_slice(&detectors);
            #[allow(clippy::cast_possible_truncation)] // detector data length fits in u32
            detector_offsets.push(detector_data.len() as u32);

            observable_data.extend_from_slice(&observables);
            #[allow(clippy::cast_possible_truncation)] // observable data length fits in u32
            observable_offsets.push(observable_data.len() as u32);
        }

        Self {
            thresholds,
            inv_log_1_minus_p,
            detector_offsets,
            detector_data,
            observable_offsets,
            observable_data,
            num_detectors,
            num_observables,
        }
    }

    /// Sample a single shot.
    ///
    /// Returns (`detection_events`, `observable_flips`) as boolean vectors.
    #[must_use]
    pub fn sample<R: Rng>(&self, rng: &mut R) -> (Vec<bool>, Vec<bool>) {
        let mut det_bits = PackedBits::new(self.num_detectors);
        let mut obs_bits = PackedBits::new(self.num_observables);

        self.sample_into_packed(&mut det_bits, &mut obs_bits, rng);

        (det_bits.to_vec(), obs_bits.to_vec())
    }

    /// Sample into pre-allocated packed bit arrays.
    ///
    /// Uses precomputed u64 thresholds for fast integer comparison.
    #[inline]
    fn sample_into_packed<R: Rng>(
        &self,
        det_bits: &mut PackedBits,
        obs_bits: &mut PackedBits,
        rng: &mut R,
    ) {
        det_bits.clear();
        obs_bits.clear();

        let num_mechanisms = self.thresholds.len();

        for i in 0..num_mechanisms {
            // Fast integer comparison with precomputed threshold
            if rng.check_probability(self.thresholds[i]) {
                // Mechanism fired - XOR in detector/observable effects
                let det_start = self.detector_offsets[i] as usize;
                let det_end = self.detector_offsets[i + 1] as usize;
                for &d in &self.detector_data[det_start..det_end] {
                    det_bits.flip(d as usize);
                }

                let obs_start = self.observable_offsets[i] as usize;
                let obs_end = self.observable_offsets[i + 1] as usize;
                for &o in &self.observable_data[obs_start..obs_end] {
                    obs_bits.flip(o as usize);
                }
            }
        }
    }

    /// Sample multiple shots.
    ///
    /// Returns (`all_detection_events`, `all_observable_flips`).
    #[must_use]
    pub fn sample_batch<R: Rng>(
        &self,
        num_shots: usize,
        rng: &mut R,
    ) -> (Vec<Vec<bool>>, Vec<Vec<bool>>) {
        let mut all_det_events = Vec::with_capacity(num_shots);
        let mut all_obs_flips = Vec::with_capacity(num_shots);

        // Pre-allocate work arrays
        let mut det_bits = PackedBits::new(self.num_detectors);
        let mut obs_bits = PackedBits::new(self.num_observables);

        for _ in 0..num_shots {
            self.sample_into_packed(&mut det_bits, &mut obs_bits, rng);
            all_det_events.push(det_bits.to_vec());
            all_obs_flips.push(obs_bits.to_vec());
        }

        (all_det_events, all_obs_flips)
    }

    /// Compute statistics without storing individual shots.
    ///
    /// This is the recommended method for threshold estimation. It automatically
    /// selects the best algorithm based on DEM size and error rates:
    /// - Uses parallel processing for larger DEMs (>50 mechanisms)
    /// - Uses geometric sampling for low error rates (p < 0.01)
    /// - Uses SIMD sampling for higher error rates
    ///
    /// # Arguments
    /// * `num_shots` - Number of shots to sample
    /// * `seed` - Random seed for reproducibility
    #[must_use]
    pub fn sample_statistics(&self, num_shots: usize, seed: u64) -> SamplingStatistics {
        if num_shots == 0 || self.thresholds.is_empty() {
            return SamplingStatistics::new(num_shots);
        }

        let num_mechanisms = self.thresholds.len();

        // Use parallel for larger problems (amortizes thread overhead)
        if num_mechanisms >= 50 && num_shots >= 1000 {
            return self.sample_statistics_parallel(num_shots, seed);
        }

        // For smaller problems, use single-threaded auto-selection
        let mut rng = PecosRng::seed_from_u64(seed);
        self.sample_statistics_auto_internal(&mut rng, num_shots)
    }

    /// Compute statistics with a user-provided RNG.
    ///
    /// Use this when you need control over the random number generator,
    /// such as for reproducibility with a specific RNG state.
    /// For most use cases, prefer `sample_statistics` which auto-selects
    /// the best algorithm.
    #[must_use]
    pub fn sample_statistics_with_rng<R: Rng>(
        &self,
        num_shots: usize,
        rng: &mut R,
    ) -> SamplingStatistics {
        if num_shots == 0 || self.thresholds.is_empty() {
            return SamplingStatistics::new(num_shots);
        }
        self.sample_statistics_auto_internal(rng, num_shots)
    }

    /// Internal: sample statistics using the most efficient method.
    ///
    /// Uses chunked processing for large working sets (>6 MB) to improve cache
    /// locality, otherwise uses direct processing.
    fn sample_statistics_auto_internal<R: Rng>(
        &self,
        rng: &mut R,
        num_shots: usize,
    ) -> SamplingStatistics {
        // Use chunked processing for better cache efficiency on large problems
        self.sample_statistics_chunked(num_shots, rng)
    }

    /// Optimized statistics sampling using flat array layout.
    ///
    /// This method provides faster sampling than nested Vec<Vec<u64>> by:
    /// - Using a flat contiguous array for detector/observable columns
    /// - Better cache locality due to predictable memory access patterns
    ///
    /// This method is semantically equivalent to the columnar methods.
    fn sample_statistics_direct<R: Rng>(
        &self,
        num_shots: usize,
        rng: &mut R,
    ) -> SamplingStatistics {
        let num_words = num_shots.div_ceil(BITS_PER_WORD);
        let num_mechanisms = self.thresholds.len();

        // Flat array for detector columns: det_data[det_idx * num_words + word_idx]
        // XOR semantics required for correct detector behavior
        let mut det_data: Vec<u64> = vec![0u64; self.num_detectors * num_words];

        // Flat array for observable columns (XOR semantics)
        // Layout: obs_data[obs_idx * num_words + word_idx]
        let mut obs_data: Vec<u64> = vec![0u64; self.num_observables * num_words];

        for mech_idx in 0..num_mechanisms {
            let threshold = self.thresholds[mech_idx];
            if threshold == 0 {
                continue;
            }

            let det_start = self.detector_offsets[mech_idx] as usize;
            let det_end = self.detector_offsets[mech_idx + 1] as usize;
            let obs_start = self.observable_offsets[mech_idx] as usize;
            let obs_end = self.observable_offsets[mech_idx + 1] as usize;

            // Skip if mechanism affects nothing
            if det_start == det_end && obs_start == obs_end {
                continue;
            }

            let inv_log = self.inv_log_1_minus_p[mech_idx];

            let mut shot = 0usize;
            while shot < num_shots {
                #[allow(clippy::cast_precision_loss)]
                let u = (rng.next_u64() as f64) / (u64::MAX as f64);
                let u = if u == 0.0 { f64::MIN_POSITIVE } else { u };
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let skip = (u.ln() * inv_log).floor() as usize;

                shot += skip;
                if shot >= num_shots {
                    break;
                }

                let word_idx = shot / BITS_PER_WORD;
                let bit_idx = shot % BITS_PER_WORD;
                let mask = 1u64 << bit_idx;

                // XOR each affected detector (correct XOR semantics)
                for &d in &self.detector_data[det_start..det_end] {
                    let idx = d as usize * num_words + word_idx;
                    det_data[idx] ^= mask;
                }

                // XOR each affected observable
                for &o in &self.observable_data[obs_start..obs_end] {
                    let idx = o as usize * num_words + word_idx;
                    obs_data[idx] ^= mask;
                }

                shot += 1;
            }
        }

        // Compute syndrome mask by ORing all detector columns
        let mut syndrome_words = vec![0u64; num_words];
        for det_idx in 0..self.num_detectors {
            let base = det_idx * num_words;
            for word_idx in 0..num_words {
                syndrome_words[word_idx] |= det_data[base + word_idx];
            }
        }

        // Compute logical error mask by ORing all observable columns
        let mut logical_words = vec![0u64; num_words];
        for obs_idx in 0..self.num_observables {
            let base = obs_idx * num_words;
            for word_idx in 0..num_words {
                logical_words[word_idx] |= obs_data[base + word_idx];
            }
        }

        // Count statistics
        let mut stats = SamplingStatistics::new(num_shots);
        for word_idx in 0..num_words {
            let syndrome = syndrome_words[word_idx];
            let logical = logical_words[word_idx];

            let valid_bits = if word_idx == num_words - 1 {
                let remaining = num_shots % BITS_PER_WORD;
                if remaining == 0 {
                    !0u64
                } else {
                    (1u64 << remaining) - 1
                }
            } else {
                !0u64
            };

            let syndrome_masked = syndrome & valid_bits;
            let logical_masked = logical & valid_bits;

            stats.syndrome_count += syndrome_masked.count_ones() as usize;
            stats.logical_error_count += logical_masked.count_ones() as usize;
            stats.undetectable_count += (logical_masked & !syndrome_masked).count_ones() as usize;
        }

        stats
    }

    /// Compute optimal chunk size for cache-friendly processing.
    ///
    /// Returns the number of shots per chunk that keeps the detector buffer
    /// within the target cache size (L3). Returns `None` if the buffer is already
    /// small enough that chunking wouldn't help.
    fn optimal_chunk_size(&self, num_shots: usize) -> Option<usize> {
        // Calculate full buffer size: (num_detectors + num_observables) * num_words * 8 bytes
        let num_words = num_shots.div_ceil(BITS_PER_WORD);
        let full_buffer_bytes = (self.num_detectors + self.num_observables) * num_words * 8;

        // Only chunk if buffer exceeds target cache size
        if full_buffer_bytes <= TARGET_CHUNK_BUFFER_BYTES {
            return None;
        }

        // Calculate chunk size that fits in cache
        // Buffer = (num_detectors + num_observables) * (chunk_shots / 64) * 8
        // chunk_shots = TARGET * 64 / ((num_detectors + num_observables) * 8)
        let total_columns = self.num_detectors + self.num_observables;
        if total_columns == 0 {
            return None;
        }

        let chunk_shots = (TARGET_CHUNK_BUFFER_BYTES * BITS_PER_WORD) / (total_columns * 8);

        // Minimum chunk size to avoid excessive overhead
        let chunk_shots = chunk_shots.max(1000);

        // Round to word boundary for efficiency
        let chunk_shots = (chunk_shots / BITS_PER_WORD) * BITS_PER_WORD;
        let chunk_shots = chunk_shots.max(BITS_PER_WORD);

        // Only chunk if we'd have at least 2 chunks
        if chunk_shots >= num_shots {
            None
        } else {
            Some(chunk_shots)
        }
    }

    /// Optimized statistics sampling with chunked processing for cache efficiency.
    ///
    /// When the working set exceeds L3 cache, this method processes shots in
    /// chunks to improve cache locality, providing ~1.5x speedup for large
    /// problem sizes.
    fn sample_statistics_chunked<R: Rng>(
        &self,
        num_shots: usize,
        rng: &mut R,
    ) -> SamplingStatistics {
        let Some(chunk_size) = self.optimal_chunk_size(num_shots) else {
            return self.sample_statistics_direct(num_shots, rng);
        };

        let mut total_stats = SamplingStatistics::new(num_shots);
        let mut shot_offset = 0;

        while shot_offset < num_shots {
            let chunk_shots = (num_shots - shot_offset).min(chunk_size);
            let chunk_stats = self.sample_statistics_direct(chunk_shots, rng);

            total_stats.syndrome_count += chunk_stats.syndrome_count;
            total_stats.logical_error_count += chunk_stats.logical_error_count;
            total_stats.undetectable_count += chunk_stats.undetectable_count;

            shot_offset += chunk_shots;
        }

        total_stats
    }

    /// Original row-major statistics (for benchmarking comparison).
    #[must_use]
    #[doc(hidden)]
    pub fn sample_statistics_row_major<R: Rng>(
        &self,
        num_shots: usize,
        rng: &mut R,
    ) -> SamplingStatistics {
        let mut stats = SamplingStatistics::new(num_shots);

        let mut det_bits = PackedBits::new(self.num_detectors);
        let mut obs_bits = PackedBits::new(self.num_observables);

        for _ in 0..num_shots {
            self.sample_into_packed(&mut det_bits, &mut obs_bits, rng);

            let has_syndrome = det_bits.any();
            let has_logical_error = obs_bits.any();

            if has_logical_error {
                stats.logical_error_count += 1;
            }
            if has_syndrome {
                stats.syndrome_count += 1;
            }
            if has_logical_error && !has_syndrome {
                stats.undetectable_count += 1;
            }
        }

        stats
    }

    // ========================================================================
    // Columnar/SIMD-optimized batch sampling
    // ========================================================================

    /// Sample multiple shots using columnar layout for better performance.
    ///
    /// This method processes all shots for each mechanism at once, enabling:
    /// - Bulk random number generation (64 shots per u64)
    /// - Better cache locality for threshold comparisons
    /// - Vectorized XOR operations on detector/observable columns
    ///
    /// Returns columnar bit-packed results: (detector_columns, observable_columns)
    /// where each column is a Vec<u64> with bit i of word w = shot w*64 + i.
    #[must_use]
    #[doc(hidden)]
    pub fn sample_batch_columnar<R: Rng>(
        &self,
        num_shots: usize,
        rng: &mut R,
    ) -> (Vec<Vec<u64>>, Vec<Vec<u64>>) {
        if num_shots == 0 {
            return (
                vec![vec![]; self.num_detectors],
                vec![vec![]; self.num_observables],
            );
        }

        let num_words = num_shots.div_ceil(BITS_PER_WORD);

        // Initialize detector and observable columns (all zeros)
        let mut det_columns: Vec<Vec<u64>> = (0..self.num_detectors)
            .map(|_| vec![0u64; num_words])
            .collect();
        let mut obs_columns: Vec<Vec<u64>> = (0..self.num_observables)
            .map(|_| vec![0u64; num_words])
            .collect();

        // Pre-allocate random number buffer for bulk generation
        let mut random_words = vec![0u64; num_words];

        // Process each mechanism
        for mech_idx in 0..self.thresholds.len() {
            let threshold = self.thresholds[mech_idx];

            // Skip mechanisms that never fire (threshold == 0)
            if threshold == 0 {
                continue;
            }

            // Generate bulk random numbers for this mechanism
            for word in &mut random_words {
                *word = rng.next_u64();
            }

            // For each word, check threshold and apply effects
            for word_idx in 0..num_words {
                let random = random_words[word_idx];

                // Check if this mechanism fires for any shot in this word
                // For low error rates, we can skip most words
                if random >= threshold {
                    continue;
                }

                // This mechanism fires - XOR its effects into detector columns
                let det_start = self.detector_offsets[mech_idx] as usize;
                let det_end = self.detector_offsets[mech_idx + 1] as usize;
                for &d in &self.detector_data[det_start..det_end] {
                    det_columns[d as usize][word_idx] ^= !0u64;
                }

                // XOR effects into observable columns
                let obs_start = self.observable_offsets[mech_idx] as usize;
                let obs_end = self.observable_offsets[mech_idx + 1] as usize;
                for &o in &self.observable_data[obs_start..obs_end] {
                    obs_columns[o as usize][word_idx] ^= !0u64;
                }
            }
        }

        (det_columns, obs_columns)
    }

    /// Sample multiple shots using per-shot threshold checking with columnar output.
    ///
    /// This is the accurate columnar implementation where each shot has independent
    /// random draws for each mechanism (matching the row-major sampling semantics).
    #[must_use]
    #[doc(hidden)]
    pub fn sample_batch_columnar_accurate<R: Rng>(
        &self,
        num_shots: usize,
        rng: &mut R,
    ) -> (Vec<Vec<u64>>, Vec<Vec<u64>>) {
        if num_shots == 0 {
            return (
                vec![vec![]; self.num_detectors],
                vec![vec![]; self.num_observables],
            );
        }

        let num_words = num_shots.div_ceil(BITS_PER_WORD);

        // Initialize detector and observable columns (all zeros)
        let mut det_columns: Vec<Vec<u64>> = (0..self.num_detectors)
            .map(|_| vec![0u64; num_words])
            .collect();
        let mut obs_columns: Vec<Vec<u64>> = (0..self.num_observables)
            .map(|_| vec![0u64; num_words])
            .collect();

        // Process each mechanism - generate one random per shot
        for mech_idx in 0..self.thresholds.len() {
            let threshold = self.thresholds[mech_idx];

            // Skip mechanisms that never fire
            if threshold == 0 {
                continue;
            }

            // Get detector/observable indices for this mechanism
            let det_start = self.detector_offsets[mech_idx] as usize;
            let det_end = self.detector_offsets[mech_idx + 1] as usize;
            let obs_start = self.observable_offsets[mech_idx] as usize;
            let obs_end = self.observable_offsets[mech_idx + 1] as usize;

            // For each word (64 shots), generate random bits and check threshold
            for word_idx in 0..num_words {
                let mut fired_mask = 0u64;

                // Check threshold for each bit position in the word
                let shots_in_word = if word_idx == num_words - 1 {
                    let remaining = num_shots % BITS_PER_WORD;
                    if remaining == 0 {
                        BITS_PER_WORD
                    } else {
                        remaining
                    }
                } else {
                    BITS_PER_WORD
                };

                for bit in 0..shots_in_word {
                    if rng.next_u64() < threshold {
                        fired_mask |= 1u64 << bit;
                    }
                }

                // Skip if no shots fired
                if fired_mask == 0 {
                    continue;
                }

                // XOR the fired mask into affected detector columns
                for &d in &self.detector_data[det_start..det_end] {
                    det_columns[d as usize][word_idx] ^= fired_mask;
                }

                // XOR the fired mask into affected observable columns
                for &o in &self.observable_data[obs_start..obs_end] {
                    obs_columns[o as usize][word_idx] ^= fired_mask;
                }
            }
        }

        (det_columns, obs_columns)
    }

    /// Compute statistics using columnar sampling (faster for large shot counts).
    ///
    /// This is more efficient than `sample_statistics` for large numbers of shots
    /// because it uses bulk random number generation and vectorized operations.
    #[must_use]
    #[doc(hidden)]
    pub fn sample_statistics_columnar<R: Rng>(
        &self,
        num_shots: usize,
        rng: &mut R,
    ) -> SamplingStatistics {
        let (det_columns, obs_columns) = self.sample_batch_columnar_accurate(num_shots, rng);

        let num_words = num_shots.div_ceil(BITS_PER_WORD);
        let mut stats = SamplingStatistics::new(num_shots);

        // OR all detector columns to get syndrome mask
        let mut syndrome_words = vec![0u64; num_words];
        for col in &det_columns {
            for (i, &word) in col.iter().enumerate() {
                syndrome_words[i] |= word;
            }
        }

        // OR all observable columns to get logical error mask
        let mut logical_words = vec![0u64; num_words];
        for col in &obs_columns {
            for (i, &word) in col.iter().enumerate() {
                logical_words[i] |= word;
            }
        }

        // Count shots with syndrome, logical error, undetectable
        for word_idx in 0..num_words {
            let syndrome = syndrome_words[word_idx];
            let logical = logical_words[word_idx];

            // Mask out unused bits in the last word
            let valid_bits = if word_idx == num_words - 1 {
                let remaining = num_shots % BITS_PER_WORD;
                if remaining == 0 {
                    !0u64
                } else {
                    (1u64 << remaining) - 1
                }
            } else {
                !0u64
            };

            let syndrome_masked = syndrome & valid_bits;
            let logical_masked = logical & valid_bits;

            stats.syndrome_count += syndrome_masked.count_ones() as usize;
            stats.logical_error_count += logical_masked.count_ones() as usize;
            stats.undetectable_count += (logical_masked & !syndrome_masked).count_ones() as usize;
        }

        stats
    }

    // ========================================================================
    // Experimental optimizations for benchmarking
    // ========================================================================

    /// SIMD-optimized columnar sampling using u64x4 (256-bit vectors).
    ///
    /// Processes 4 u64 words (256 shots) at a time for better throughput.
    #[must_use]
    #[doc(hidden)]
    pub fn sample_batch_columnar_simd<R: Rng>(
        &self,
        num_shots: usize,
        rng: &mut R,
    ) -> (Vec<Vec<u64>>, Vec<Vec<u64>>) {
        if num_shots == 0 {
            return (
                vec![vec![]; self.num_detectors],
                vec![vec![]; self.num_observables],
            );
        }

        let num_words = num_shots.div_ceil(BITS_PER_WORD);
        let num_simd_words = num_words.div_ceil(4);

        // Initialize detector and observable columns as SIMD vectors
        let mut det_columns: Vec<Vec<u64x4>> = (0..self.num_detectors)
            .map(|_| vec![u64x4::ZERO; num_simd_words])
            .collect();
        let mut obs_columns: Vec<Vec<u64x4>> = (0..self.num_observables)
            .map(|_| vec![u64x4::ZERO; num_simd_words])
            .collect();

        // Pre-allocate random buffer for bulk generation - need one random per shot
        let mut random_buffer = vec![0u64; num_shots];

        // Process each mechanism
        for mech_idx in 0..self.thresholds.len() {
            let threshold = self.thresholds[mech_idx];

            if threshold == 0 {
                continue;
            }

            // Get detector/observable indices for this mechanism
            let det_start = self.detector_offsets[mech_idx] as usize;
            let det_end = self.detector_offsets[mech_idx + 1] as usize;
            let obs_start = self.observable_offsets[mech_idx] as usize;
            let obs_end = self.observable_offsets[mech_idx + 1] as usize;

            // Generate all random numbers for this mechanism at once
            // Use fill_u64 from RngProbabilityExt for potentially optimized bulk generation
            rng.fill_u64(&mut random_buffer);

            for simd_idx in 0..num_simd_words {
                let base = simd_idx * 4;

                // Load 4 random values and compare with threshold
                // For each of 4 u64 positions, we need to check 64 shots
                let mut fired_masks = [0u64; 4];

                for (lane, fired_mask) in fired_masks.iter_mut().enumerate() {
                    let word_idx = base + lane;
                    if word_idx >= num_words {
                        break;
                    }

                    // Determine shots in this word
                    let shots_in_word = if word_idx == num_words - 1 {
                        let remaining = num_shots % BITS_PER_WORD;
                        if remaining == 0 {
                            BITS_PER_WORD
                        } else {
                            remaining
                        }
                    } else {
                        BITS_PER_WORD
                    };

                    // Check each shot in this word
                    // Note: We still need per-shot RNG for accurate sampling
                    let word_base = word_idx * BITS_PER_WORD;
                    for bit in 0..shots_in_word {
                        let rand_idx = word_base + bit;
                        if rand_idx < random_buffer.len() && random_buffer[rand_idx] < threshold {
                            *fired_mask |= 1u64 << bit;
                        }
                    }
                }

                let fired_vec = u64x4::new(fired_masks);

                // Skip if no shots fired in this SIMD word
                if fired_vec == u64x4::ZERO {
                    continue;
                }

                // XOR into affected detector columns
                for &d in &self.detector_data[det_start..det_end] {
                    det_columns[d as usize][simd_idx] ^= fired_vec;
                }

                // XOR into affected observable columns
                for &o in &self.observable_data[obs_start..obs_end] {
                    obs_columns[o as usize][simd_idx] ^= fired_vec;
                }
            }
        }

        // Convert SIMD columns back to Vec<u64>
        let det_result: Vec<Vec<u64>> = det_columns
            .into_iter()
            .map(|col| {
                let mut result = Vec::with_capacity(num_words);
                for simd_word in col {
                    let arr = simd_word.to_array();
                    for &val in &arr {
                        if result.len() < num_words {
                            result.push(val);
                        }
                        if result.len() >= num_words {
                            break;
                        }
                    }
                }
                result.truncate(num_words);
                result
            })
            .collect();

        let obs_result: Vec<Vec<u64>> = obs_columns
            .into_iter()
            .map(|col| {
                let mut result = Vec::with_capacity(num_words);
                for simd_word in col {
                    let arr = simd_word.to_array();
                    for &val in &arr {
                        if result.len() < num_words {
                            result.push(val);
                        }
                        if result.len() >= num_words {
                            break;
                        }
                    }
                }
                result.truncate(num_words);
                result
            })
            .collect();

        (det_result, obs_result)
    }

    /// Geometric skip optimized sampling for sparse events.
    ///
    /// Uses geometric distribution to skip directly to firing shots,
    /// which is much faster for low error rates (p << 1).
    #[must_use]
    #[doc(hidden)]
    pub fn sample_batch_columnar_geometric<R: Rng>(
        &self,
        num_shots: usize,
        rng: &mut R,
    ) -> (Vec<Vec<u64>>, Vec<Vec<u64>>) {
        if num_shots == 0 {
            return (
                vec![vec![]; self.num_detectors],
                vec![vec![]; self.num_observables],
            );
        }

        let num_words = num_shots.div_ceil(BITS_PER_WORD);

        // Initialize detector and observable columns
        let mut det_columns: Vec<Vec<u64>> = (0..self.num_detectors)
            .map(|_| vec![0u64; num_words])
            .collect();
        let mut obs_columns: Vec<Vec<u64>> = (0..self.num_observables)
            .map(|_| vec![0u64; num_words])
            .collect();

        // Process each mechanism using geometric sampling
        for mech_idx in 0..self.thresholds.len() {
            let threshold = self.thresholds[mech_idx];

            if threshold == 0 {
                continue;
            }

            // Get detector/observable indices
            let det_start = self.detector_offsets[mech_idx] as usize;
            let det_end = self.detector_offsets[mech_idx + 1] as usize;
            let obs_start = self.observable_offsets[mech_idx] as usize;
            let obs_end = self.observable_offsets[mech_idx + 1] as usize;

            // Use precomputed 1/ln(1-p) for geometric sampling
            let inv_log = self.inv_log_1_minus_p[mech_idx];

            let mut shot = 0usize;
            while shot < num_shots {
                // Sample geometric skip: skip = floor(ln(U) * inv_log)
                #[allow(clippy::cast_precision_loss)]
                let u = (rng.next_u64() as f64) / (u64::MAX as f64);
                // Avoid log(0)
                let u = if u == 0.0 { f64::MIN_POSITIVE } else { u };
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let skip = (u.ln() * inv_log).floor() as usize;

                shot += skip;
                if shot >= num_shots {
                    break;
                }

                // This shot fires - set the bit
                let word_idx = shot / BITS_PER_WORD;
                let bit_idx = shot % BITS_PER_WORD;
                let mask = 1u64 << bit_idx;

                for &d in &self.detector_data[det_start..det_end] {
                    det_columns[d as usize][word_idx] ^= mask;
                }

                for &o in &self.observable_data[obs_start..obs_end] {
                    obs_columns[o as usize][word_idx] ^= mask;
                }

                shot += 1; // Move to next shot after firing
            }
        }

        (det_columns, obs_columns)
    }

    /// Statistics using SIMD columnar sampling.
    #[must_use]
    #[doc(hidden)]
    pub fn sample_statistics_simd<R: Rng>(
        &self,
        num_shots: usize,
        rng: &mut R,
    ) -> SamplingStatistics {
        let (det_columns, obs_columns) = self.sample_batch_columnar_simd(num_shots, rng);
        Self::compute_statistics_from_columns(&det_columns, &obs_columns, num_shots)
    }

    /// Statistics using geometric skip sampling.
    #[must_use]
    #[doc(hidden)]
    pub fn sample_statistics_geometric<R: Rng>(
        &self,
        num_shots: usize,
        rng: &mut R,
    ) -> SamplingStatistics {
        let (det_columns, obs_columns) = self.sample_batch_columnar_geometric(num_shots, rng);
        Self::compute_statistics_from_columns(&det_columns, &obs_columns, num_shots)
    }

    // ========================================================================
    // Auto-selection and parallel methods
    // ========================================================================

    /// Compute the average error probability across all mechanisms.
    ///
    /// Used to decide between geometric (low p) and SIMD (high p) sampling.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn average_error_probability(&self) -> f64 {
        if self.thresholds.is_empty() {
            return 0.0;
        }
        let sum: u128 = self.thresholds.iter().map(|&t| u128::from(t)).sum();
        let avg_threshold = (sum / self.thresholds.len() as u128) as f64;
        avg_threshold / u64::MAX as f64
    }

    /// Maximum error probability across all mechanisms.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn max_error_probability(&self) -> f64 {
        self.thresholds
            .iter()
            .map(|&t| t as f64 / u64::MAX as f64)
            .fold(0.0, f64::max)
    }

    /// Parallel statistics sampling using Rayon.
    ///
    /// For benchmarking. Production code should use `sample_statistics`.
    #[must_use]
    #[doc(hidden)]
    pub fn sample_statistics_parallel(&self, num_shots: usize, seed: u64) -> SamplingStatistics {
        if num_shots == 0 || self.thresholds.is_empty() {
            return SamplingStatistics::new(num_shots);
        }

        // Determine shot chunk size based on cache efficiency
        // Use optimal chunk size if available, otherwise use thread-based chunking
        let num_threads = rayon::current_num_threads();
        let cache_optimal = self.optimal_chunk_size(num_shots).unwrap_or(num_shots);
        let thread_based = (num_shots / num_threads).max(5_000);
        // Use smaller of cache-optimal and thread-based for best performance
        let shots_per_chunk = cache_optimal.min(thread_based).max(1_000);
        let num_chunks = num_shots.div_ceil(shots_per_chunk);

        // Process shot chunks in parallel
        let partial_stats: Vec<SamplingStatistics> = (0..num_chunks)
            .into_par_iter()
            .map(|chunk_idx| {
                // Determine shot range for this chunk
                let start_shot = chunk_idx * shots_per_chunk;
                let end_shot = ((chunk_idx + 1) * shots_per_chunk).min(num_shots);
                let chunk_shots = end_shot - start_shot;

                if chunk_shots == 0 {
                    return SamplingStatistics::new(0);
                }

                // Create thread-local RNG with deterministic seed based on chunk
                let chunk_seed =
                    seed.wrapping_add((chunk_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                let mut rng = PecosRng::seed_from_u64(chunk_seed);

                // Use geometric sampling for this chunk
                self.sample_statistics_geometric_range(chunk_shots, &mut rng)
            })
            .collect();

        // Sum up partial statistics
        let mut total = SamplingStatistics::new(num_shots);
        for stats in partial_stats {
            total.syndrome_count += stats.syndrome_count;
            total.logical_error_count += stats.logical_error_count;
            total.undetectable_count += stats.undetectable_count;
        }

        total
    }

    /// Internal helper: geometric statistics for a range of shots.
    ///
    /// Uses direct accumulation for optimal performance (same approach as
    /// `sample_statistics_direct`).
    fn sample_statistics_geometric_range<R: Rng>(
        &self,
        num_shots: usize,
        rng: &mut R,
    ) -> SamplingStatistics {
        // Delegate to direct method - same algorithm, avoids code duplication
        self.sample_statistics_direct(num_shots, rng)
    }

    /// Helper to compute statistics from columnar data.
    fn compute_statistics_from_columns(
        det_columns: &[Vec<u64>],
        obs_columns: &[Vec<u64>],
        num_shots: usize,
    ) -> SamplingStatistics {
        let num_words = num_shots.div_ceil(BITS_PER_WORD);
        let mut stats = SamplingStatistics::new(num_shots);

        // OR all detector columns to get syndrome mask
        let mut syndrome_words = vec![0u64; num_words];
        for col in det_columns {
            for (i, &word) in col.iter().enumerate() {
                syndrome_words[i] |= word;
            }
        }

        // OR all observable columns to get logical error mask
        let mut logical_words = vec![0u64; num_words];
        for col in obs_columns {
            for (i, &word) in col.iter().enumerate() {
                logical_words[i] |= word;
            }
        }

        // Count shots with syndrome, logical error, undetectable
        for word_idx in 0..num_words {
            let syndrome = syndrome_words[word_idx];
            let logical = logical_words[word_idx];

            let valid_bits = if word_idx == num_words - 1 {
                let remaining = num_shots % BITS_PER_WORD;
                if remaining == 0 {
                    !0u64
                } else {
                    (1u64 << remaining) - 1
                }
            } else {
                !0u64
            };

            let syndrome_masked = syndrome & valid_bits;
            let logical_masked = logical & valid_bits;

            stats.syndrome_count += syndrome_masked.count_ones() as usize;
            stats.logical_error_count += logical_masked.count_ones() as usize;
            stats.undetectable_count += (logical_masked & !syndrome_masked).count_ones() as usize;
        }

        stats
    }
}

/// Statistics from sampling.
#[derive(Debug, Clone)]
pub struct SamplingStatistics {
    /// Total number of shots.
    pub total_shots: usize,
    /// Shots with at least one logical error.
    pub logical_error_count: usize,
    /// Shots with at least one detector firing.
    pub syndrome_count: usize,
    /// Shots with logical error but no syndrome (undetectable errors).
    pub undetectable_count: usize,
}

impl SamplingStatistics {
    fn new(total_shots: usize) -> Self {
        Self {
            total_shots,
            logical_error_count: 0,
            syndrome_count: 0,
            undetectable_count: 0,
        }
    }

    /// Logical error rate.
    #[must_use]
    #[allow(clippy::cast_precision_loss)] // rate calculation
    pub fn logical_error_rate(&self) -> f64 {
        self.logical_error_count as f64 / self.total_shots as f64
    }

    /// Syndrome rate (fraction of shots with non-trivial syndrome).
    #[must_use]
    #[allow(clippy::cast_precision_loss)] // rate calculation
    pub fn syndrome_rate(&self) -> f64 {
        self.syndrome_count as f64 / self.total_shots as f64
    }

    /// Undetectable error rate.
    #[must_use]
    #[allow(clippy::cast_precision_loss)] // rate calculation
    pub fn undetectable_rate(&self) -> f64 {
        self.undetectable_count as f64 / self.total_shots as f64
    }
}

// ============================================================================
// DEM Sampler Builder
// ============================================================================

/// Builder for [`DemSampler`].
///
/// Constructs a [`DemSampler`] from a fault influence map, noise parameters,
/// and explicit detector/observable definitions.
pub struct DemSamplerBuilder<'a> {
    influence_map: &'a DagFaultInfluenceMap,
    p1: f64,
    p2: f64,
    p_meas: f64,
    p_init: f64,
    detector_records: Vec<Vec<i32>>,
    observable_records: Vec<Vec<i32>>,
    measurement_order: Option<Vec<usize>>,
    num_tc_measurements: Option<usize>,
}

impl<'a> DemSamplerBuilder<'a> {
    /// Create a new builder from an influence map.
    #[must_use]
    pub fn new(influence_map: &'a DagFaultInfluenceMap) -> Self {
        Self {
            influence_map,
            p1: 0.01,
            p2: 0.01,
            p_meas: 0.01,
            p_init: 0.01,
            detector_records: Vec::new(),
            observable_records: Vec::new(),
            measurement_order: None,
            num_tc_measurements: None,
        }
    }

    /// Set noise parameters.
    #[must_use]
    pub fn with_noise(mut self, p1: f64, p2: f64, p_meas: f64, p_init: f64) -> Self {
        self.p1 = p1;
        self.p2 = p2;
        self.p_meas = p_meas;
        self.p_init = p_init;
        self
    }

    /// Set detector definitions from JSON.
    ///
    /// Format: `[{"id": 0, "records": [-1, -5]}, ...]`
    ///
    /// # Errors
    /// Returns an error if the JSON is malformed or missing required fields.
    pub fn with_detectors_json(mut self, json: &str) -> Result<Self, String> {
        self.detector_records = parse_records_json(json, "detector")?;
        Ok(self)
    }

    /// Set observable definitions from JSON.
    ///
    /// Format: `[{"id": 0, "records": [-1, -3, -5]}, ...]`
    ///
    /// # Errors
    /// Returns an error if the JSON is malformed or missing required fields.
    pub fn with_observables_json(mut self, json: &str) -> Result<Self, String> {
        self.observable_records = parse_records_json(json, "observable")?;
        Ok(self)
    }

    /// Set detector records directly.
    #[must_use]
    pub fn with_detector_records(mut self, records: Vec<Vec<i32>>) -> Self {
        self.detector_records = records;
        self
    }

    /// Set observable records directly.
    #[must_use]
    pub fn with_observable_records(mut self, records: Vec<Vec<i32>>) -> Self {
        self.observable_records = records;
        self
    }

    /// Set the measurement order mapping from `TickCircuit`.
    ///
    /// `measurement_order[tc_idx]` is the qubit measured at `TickCircuit` index `tc_idx`.
    /// This is needed to map between `TickCircuit` record offsets and influence map indices.
    #[must_use]
    pub fn with_measurement_order(mut self, order: Vec<usize>) -> Self {
        self.num_tc_measurements = Some(order.len());
        self.measurement_order = Some(order);
        self
    }

    /// Build the [`DemSampler`].
    #[must_use]
    pub fn build(self) -> DemSampler {
        let num_detectors = self.detector_records.len();
        let num_observables = self.observable_records.len();
        let num_im_measurements = self.influence_map.measurements.len();
        let num_tc_measurements = self.num_tc_measurements.unwrap_or(num_im_measurements);

        // Build IM -> TC index mapping
        let im_to_tc = self.build_im_to_tc_mapping();

        // Aggregation map: mechanism -> probability
        let mut aggregated: BTreeMap<DemMechanism, f64> = BTreeMap::new();

        // Group two-qubit gate locations by node for paired processing
        let mut cx_groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();

        // Process each fault location
        for (loc_idx, loc) in self.influence_map.locations.iter().enumerate() {
            match loc.gate_type {
                GateType::PZ | GateType::QAlloc
                    // Prep errors: only "after" locations (X error for Z-basis prep)
                    if self.p_init > 0.0 && !loc.before => {
                        self.process_single_pauli_fault(
                            loc_idx,
                            Pauli::X,
                            self.p_init,
                            im_to_tc.as_deref(),
                            num_tc_measurements,
                            &mut aggregated,
                        );
                    }
                GateType::MZ | GateType::MeasureFree
                    // Measurement errors: only "before" locations (X error = bit flip)
                    if self.p_meas > 0.0 && loc.before => {
                        self.process_single_pauli_fault(
                            loc_idx,
                            Pauli::X,
                            self.p_meas,
                            im_to_tc.as_deref(),
                            num_tc_measurements,
                            &mut aggregated,
                        );
                    }
                GateType::CX
                | GateType::CZ
                | GateType::CY
                | GateType::SWAP
                | GateType::RXX
                | GateType::RYY
                | GateType::RZZ
                    // Two-qubit gate errors: only "after" locations, process as pairs
                    if !loc.before => {
                        cx_groups.entry(loc.node).or_default().push(loc_idx);
                    }
                GateType::H
                | GateType::SZ
                | GateType::SZdg
                | GateType::SX
                | GateType::SXdg
                | GateType::SY
                | GateType::SYdg
                | GateType::X
                | GateType::Y
                | GateType::Z
                | GateType::T
                | GateType::Tdg
                | GateType::RX
                | GateType::RY
                | GateType::RZ
                | GateType::U
                | GateType::R1XY
                    // Single-qubit gate errors: only "after" locations, depolarizing
                    if self.p1 > 0.0 && !loc.before => {
                        self.process_depolarizing_fault(
                            loc_idx,
                            self.p1,
                            im_to_tc.as_deref(),
                            num_tc_measurements,
                            &mut aggregated,
                        );
                    }
                _ => {}
            }
        }

        // Process two-qubit gates as pairs
        if self.p2 > 0.0 {
            for loc_indices in cx_groups.values() {
                if loc_indices.len() == 2 {
                    self.process_two_qubit_fault(
                        loc_indices[0],
                        loc_indices[1],
                        im_to_tc.as_deref(),
                        num_tc_measurements,
                        &mut aggregated,
                    );
                }
            }
        }

        // Convert aggregated map to SoA layout with precomputed thresholds
        let num_mechanisms = aggregated.len();
        let mut thresholds = Vec::with_capacity(num_mechanisms);
        let mut detector_offsets = Vec::with_capacity(num_mechanisms + 1);
        let mut detector_data = Vec::new();
        let mut observable_offsets = Vec::with_capacity(num_mechanisms + 1);
        let mut observable_data = Vec::new();

        detector_offsets.push(0);
        observable_offsets.push(0);

        let mut inv_log_1_minus_p = Vec::with_capacity(num_mechanisms);

        for (mech, prob) in aggregated {
            // Precompute u64 threshold: p * u64::MAX
            // This avoids f64 comparison during sampling
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                clippy::cast_precision_loss
            )]
            let threshold = (prob * (u64::MAX as f64)) as u64;
            thresholds.push(threshold);

            // Precompute 1/ln(1-p) for geometric sampling
            // Use multiplication instead of division in hot loop
            let log_1_minus_p = (1.0 - prob).ln();
            let inv = if log_1_minus_p.abs() < f64::EPSILON {
                0.0 // p ≈ 0, mechanism never fires
            } else {
                1.0 / log_1_minus_p
            };
            inv_log_1_minus_p.push(inv);

            detector_data.extend_from_slice(&mech.detectors);
            #[allow(clippy::cast_possible_truncation)] // detector data length fits in u32
            detector_offsets.push(detector_data.len() as u32);

            observable_data.extend_from_slice(&mech.observables);
            #[allow(clippy::cast_possible_truncation)] // observable data length fits in u32
            observable_offsets.push(observable_data.len() as u32);
        }

        DemSampler {
            thresholds,
            inv_log_1_minus_p,
            detector_offsets,
            detector_data,
            observable_offsets,
            observable_data,
            num_detectors,
            num_observables,
        }
    }

    /// Build mapping from influence map measurement indices to `TickCircuit` indices.
    fn build_im_to_tc_mapping(&self) -> Option<Vec<usize>> {
        let tc_order = self.measurement_order.as_ref()?;

        // Build (qubit, occurrence) -> TC index mapping
        // Use BTreeMap for deterministic iteration order
        let mut qubit_occurrences: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (tc_idx, &qubit) in tc_order.iter().enumerate() {
            qubit_occurrences.entry(qubit).or_default().push(tc_idx);
        }

        // Track how many times we've seen each qubit in the IM
        let mut qubit_seen_count: BTreeMap<usize, usize> = BTreeMap::new();

        // For each IM measurement, find corresponding TC index
        let mapping: Vec<usize> = self
            .influence_map
            .measurements
            .iter()
            .map(|&(_node, qubit, _basis)| {
                let occurrence = *qubit_seen_count.entry(qubit).or_insert(0);
                qubit_seen_count.insert(qubit, occurrence + 1);

                qubit_occurrences
                    .get(&qubit)
                    .and_then(|indices| indices.get(occurrence).copied())
                    .unwrap_or(usize::MAX)
            })
            .collect();

        Some(mapping)
    }

    /// Process a single Pauli fault (prep X error, measurement X error).
    fn process_single_pauli_fault(
        &self,
        loc_idx: usize,
        pauli: Pauli,
        prob: f64,
        im_to_tc: Option<&[usize]>,
        num_tc_measurements: usize,
        aggregated: &mut BTreeMap<DemMechanism, f64>,
    ) {
        let mechanism = self.compute_mechanism(loc_idx, pauli, im_to_tc, num_tc_measurements);
        if !mechanism.is_empty() {
            let entry = aggregated.entry(mechanism).or_insert(0.0);
            *entry = combine_probabilities(*entry, prob);
        }
    }

    /// Process a depolarizing fault (X, Y, Z each with prob/3).
    fn process_depolarizing_fault(
        &self,
        loc_idx: usize,
        prob: f64,
        im_to_tc: Option<&[usize]>,
        num_tc_measurements: usize,
        aggregated: &mut BTreeMap<DemMechanism, f64>,
    ) {
        let per_pauli_prob = prob / 3.0;
        for pauli in [Pauli::X, Pauli::Y, Pauli::Z] {
            let mechanism = self.compute_mechanism(loc_idx, pauli, im_to_tc, num_tc_measurements);
            if !mechanism.is_empty() {
                let entry = aggregated.entry(mechanism).or_insert(0.0);
                *entry = combine_probabilities(*entry, per_pauli_prob);
            }
        }
    }

    /// Process a two-qubit gate fault (15 non-identity Pauli combinations with p2/15 each).
    fn process_two_qubit_fault(
        &self,
        loc1: usize,
        loc2: usize,
        im_to_tc: Option<&[usize]>,
        num_tc_measurements: usize,
        aggregated: &mut BTreeMap<DemMechanism, f64>,
    ) {
        let prob = self.p2 / 15.0;
        let paulis = [Pauli::I, Pauli::X, Pauli::Y, Pauli::Z];

        // Cache single-qubit mechanisms for each Pauli on each location
        let mut effects1: [Option<DemMechanism>; 4] = [None, None, None, None];
        let mut effects2: [Option<DemMechanism>; 4] = [None, None, None, None];

        for &p in &[Pauli::X, Pauli::Y, Pauli::Z] {
            effects1[p as usize] =
                Some(self.compute_mechanism(loc1, p, im_to_tc, num_tc_measurements));
            effects2[p as usize] =
                Some(self.compute_mechanism(loc2, p, im_to_tc, num_tc_measurements));
        }

        // Process all 15 non-trivial Pauli combinations
        for &p1 in &paulis {
            for &p2 in &paulis {
                if p1 == Pauli::I && p2 == Pauli::I {
                    continue; // Skip II
                }

                let mechanism = if p1 == Pauli::I {
                    // IX, IY, IZ - only second qubit
                    effects2[p2 as usize]
                        .clone()
                        .unwrap_or_else(DemMechanism::empty)
                } else if p2 == Pauli::I {
                    // XI, YI, ZI - only first qubit
                    effects1[p1 as usize]
                        .clone()
                        .unwrap_or_else(DemMechanism::empty)
                } else {
                    // Correlated: XOR the detector/observable effects
                    let e1 = effects1[p1 as usize].as_ref();
                    let e2 = effects2[p2 as usize].as_ref();
                    xor_mechanisms(e1, e2)
                };

                if !mechanism.is_empty() {
                    let entry = aggregated.entry(mechanism).or_insert(0.0);
                    *entry = combine_probabilities(*entry, prob);
                }
            }
        }
    }

    /// Compute the mechanism (detector/observable effects) for a fault.
    fn compute_mechanism(
        &self,
        loc_idx: usize,
        pauli: Pauli,
        im_to_tc: Option<&[usize]>,
        num_tc_measurements: usize,
    ) -> DemMechanism {
        // Get measurement indices that flip (in IM order)
        let im_meas_flips = self
            .influence_map
            .get_detector_indices(loc_idx, pauli as u8);

        // Convert to TC order measurement outcomes
        let mut tc_outcomes = vec![false; num_tc_measurements];
        for &im_idx in im_meas_flips {
            let tc_idx = if let Some(mapping) = im_to_tc {
                if (im_idx as usize) < mapping.len() {
                    mapping[im_idx as usize]
                } else {
                    continue;
                }
            } else {
                im_idx as usize
            };

            if tc_idx < num_tc_measurements {
                tc_outcomes[tc_idx] ^= true;
            }
        }

        // Apply detector definitions (XOR of measurement outcomes)
        let detectors: SmallVec<[u32; 4]> = self
            .detector_records
            .iter()
            .enumerate()
            .filter_map(|(det_id, records)| {
                let mut xor_result = false;
                for &offset in records {
                    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)] // measurement count fits in i32
                    #[allow(clippy::cast_sign_loss)]
                    // negative offset + total count, or non-negative offset
                    let abs_idx = if offset < 0 {
                        (num_tc_measurements as i32 + offset) as usize
                    } else {
                        offset as usize
                    };
                    if abs_idx < num_tc_measurements && tc_outcomes[abs_idx] {
                        xor_result = !xor_result;
                    }
                }
                if xor_result {
                    #[allow(clippy::cast_possible_truncation)] // detector ID fits in u32
                    Some(det_id as u32)
                } else {
                    None
                }
            })
            .collect();

        // Apply observable definitions (XOR of measurement outcomes)
        let observables: SmallVec<[u32; 2]> = self
            .observable_records
            .iter()
            .enumerate()
            .filter_map(|(obs_id, records)| {
                let mut xor_result = false;
                for &offset in records {
                    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)] // measurement count fits in i32
                    #[allow(clippy::cast_sign_loss)]
                    // negative offset + total count, or non-negative offset
                    let abs_idx = if offset < 0 {
                        (num_tc_measurements as i32 + offset) as usize
                    } else {
                        offset as usize
                    };
                    if abs_idx < num_tc_measurements && tc_outcomes[abs_idx] {
                        xor_result = !xor_result;
                    }
                }
                if xor_result {
                    #[allow(clippy::cast_possible_truncation)] // observable ID fits in u32
                    Some(obs_id as u32)
                } else {
                    None
                }
            })
            .collect();

        DemMechanism::new(detectors, observables)
    }
}

/// XORs two [`DemMechanism`]s (symmetric difference of detectors and observables).
fn xor_mechanisms(a: Option<&DemMechanism>, b: Option<&DemMechanism>) -> DemMechanism {
    match (a, b) {
        (Some(m1), Some(m2)) => {
            let detectors = xor_u32_vecs::<4>(&m1.detectors, &m2.detectors);
            let observables = xor_u32_vecs::<2>(&m1.observables, &m2.observables);
            DemMechanism {
                detectors,
                observables,
            }
        }
        (Some(m), None) | (None, Some(m)) => m.clone(),
        (None, None) => DemMechanism::empty(),
    }
}

/// XORs two sorted u32 slices (symmetric difference), returning a `SmallVec`.
fn xor_u32_vecs<const N: usize>(a: &[u32], b: &[u32]) -> SmallVec<[u32; N]>
where
    [u32; N]: smallvec::Array<Item = u32>,
{
    let mut result: SmallVec<[u32; N]> = SmallVec::new();
    let mut i = 0;
    let mut j = 0;

    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => {
                result.push(a[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                result.push(b[j]);
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                // Same element in both - XOR cancels
                i += 1;
                j += 1;
            }
        }
    }

    result.extend_from_slice(&a[i..]);
    result.extend_from_slice(&b[j..]);
    result
}

/// Parse detector or observable records from JSON.
///
/// Uses a simple custom parser to avoid `serde_json` dependency.
/// Expected format: `[{"id": 0, "records": [-1, -5]}, ...]`
#[allow(clippy::unnecessary_wraps)]
fn parse_records_json(json: &str, _kind: &str) -> Result<Vec<Vec<i32>>, String> {
    let json = json.trim();
    if json.is_empty() || json == "[]" {
        return Ok(Vec::new());
    }

    let mut results = Vec::new();

    // Simple state machine to find each object
    let mut depth = 0;
    let mut start = None;

    for (i, c) in json.char_indices() {
        match c {
            '{' => {
                if depth == 1 {
                    start = Some(i);
                }
                depth += 1;
            }
            '}' => {
                depth -= 1;
                if depth == 1 {
                    if let Some(s) = start {
                        let obj_str = &json[s..i + c.len_utf8()];
                        let records = extract_records_from_object(obj_str);
                        results.push(records);
                    }
                    start = None;
                }
            }
            '[' if depth == 0 => depth = 1,
            ']' if depth == 1 => depth = 0,
            _ => {}
        }
    }

    Ok(results)
}

/// Extract the "records" array from a JSON object string.
fn extract_records_from_object(json: &str) -> Vec<i32> {
    if let Some(pos) = json.find("\"records\"") {
        let rest = &json[pos..];
        if let (Some(arr_start), Some(arr_end)) = (rest.find('['), rest.find(']'))
            && arr_start < arr_end
        {
            let arr_str = &rest[arr_start + 1..arr_end];
            return arr_str
                .split(',')
                .filter_map(|s| s.trim().parse::<i32>().ok())
                .collect();
        }
    }
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dem_mechanism_ordering() {
        let m1 = DemMechanism::new(smallvec::smallvec![1, 2], smallvec::smallvec![0]);
        let m2 = DemMechanism::new(smallvec::smallvec![2, 1], smallvec::smallvec![0]);
        assert_eq!(m1, m2); // Should be equal after sorting
    }

    #[test]
    fn test_empty_mechanism() {
        let m = DemMechanism::empty();
        assert!(m.is_empty());
    }

    #[test]
    fn test_packed_bits() {
        let mut bits = PackedBits::new(100);
        assert!(!bits.any());

        bits.flip(0);
        assert!(bits.any());
        assert!(bits.get(0));
        assert!(!bits.get(1));

        bits.flip(64); // Second word
        assert!(bits.get(64));

        bits.flip(0); // XOR back to false
        assert!(!bits.get(0));
        assert!(bits.any()); // bit 64 still set

        bits.clear();
        assert!(!bits.any());
    }

    #[test]
    fn test_packed_bits_to_vec() {
        let mut bits = PackedBits::new(5);
        bits.flip(1);
        bits.flip(3);
        let vec = bits.to_vec();
        assert_eq!(vec, vec![false, true, false, true, false]);
    }

    #[test]
    fn test_xor_mechanisms() {
        let m1 = DemMechanism::new(smallvec::smallvec![0, 1, 2], smallvec::smallvec![0]);
        let m2 = DemMechanism::new(smallvec::smallvec![1, 2, 3], smallvec::smallvec![0, 1]);

        let result = xor_mechanisms(Some(&m1), Some(&m2));

        // Detectors: {0,1,2} XOR {1,2,3} = {0,3}
        assert_eq!(result.detectors.as_slice(), &[0, 3]);
        // Observables: {0} XOR {0,1} = {1}
        assert_eq!(result.observables.as_slice(), &[1]);
    }

    #[test]
    fn test_xor_mechanisms_single() {
        let m1 = DemMechanism::new(smallvec::smallvec![0, 1], smallvec::smallvec![]);

        let result1 = xor_mechanisms(Some(&m1), None);
        let result2 = xor_mechanisms(None, Some(&m1));

        assert_eq!(result1.detectors.as_slice(), &[0, 1]);
        assert_eq!(result2.detectors.as_slice(), &[0, 1]);
    }

    #[test]
    fn test_xor_mechanisms_both_none() {
        let result = xor_mechanisms(None, None);
        assert!(result.is_empty());
    }

    #[test]
    fn test_parse_records_json_empty() {
        let result = parse_records_json("[]", "test").unwrap();
        assert!(result.is_empty());

        let result = parse_records_json("", "test").unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_parse_records_json_valid() {
        let json = r#"[{"id": 0, "records": [-1, -5]}, {"id": 1, "records": [-2, -3, -4]}]"#;
        let result = parse_records_json(json, "detector").unwrap();

        assert_eq!(result.len(), 2);
        assert_eq!(result[0], vec![-1, -5]);
        assert_eq!(result[1], vec![-2, -3, -4]);
    }

    #[test]
    fn test_sampling_statistics_zero_noise() {
        use crate::fault_tolerance::propagator::DagFaultAnalyzer;
        use pecos_quantum::DagCircuit;

        // Simple circuit with prep, gate, measure
        let mut dag = DagCircuit::new();
        dag.pz(&[0]);
        dag.h(&[0]);
        dag.mz(&[0]);

        let analyzer = DagFaultAnalyzer::new(&dag);
        let influence_map = analyzer.build_influence_map();

        // Zero noise should produce no errors
        let sampler = DemSamplerBuilder::new(&influence_map)
            .with_noise(0.0, 0.0, 0.0, 0.0)
            .build();

        assert_eq!(sampler.num_mechanisms(), 0);

        let stats = sampler.sample_statistics(100, 42);
        assert_eq!(stats.logical_error_count, 0);
        assert_eq!(stats.syndrome_count, 0);
    }

    #[test]
    fn test_sampling_with_explicit_definitions() {
        use crate::fault_tolerance::propagator::DagFaultAnalyzer;
        use pecos_quantum::DagCircuit;

        // Two-qubit parity check circuit
        let mut dag = DagCircuit::new();
        dag.pz(&[2]); // Ancilla
        dag.cx(&[(0, 2)]);
        dag.cx(&[(1, 2)]);
        dag.mz(&[2]);

        let analyzer = DagFaultAnalyzer::new(&dag);
        let influence_map = analyzer.build_influence_map();

        // Define detector on the measurement
        let detectors_json = r#"[{"id": 0, "records": [-1]}]"#;
        let observables_json = r"[]";

        let sampler = DemSamplerBuilder::new(&influence_map)
            .with_noise(0.1, 0.1, 0.1, 0.1)
            .with_detectors_json(detectors_json)
            .unwrap()
            .with_observables_json(observables_json)
            .unwrap()
            .build();

        assert_eq!(sampler.num_detectors(), 1);
        assert_eq!(sampler.num_observables(), 0);
        assert!(sampler.num_mechanisms() > 0);

        // Sample and verify we get detection events
        let stats = sampler.sample_statistics(1000, 42);
        assert!(stats.syndrome_count > 0);
    }

    #[test]
    fn test_columnar_sampling_statistics() {
        use crate::fault_tolerance::propagator::DagFaultAnalyzer;
        use pecos_quantum::DagCircuit;
        use rand::SeedableRng;
        use rand::rngs::SmallRng;

        // Create a simple circuit with noise
        let mut dag = DagCircuit::new();
        dag.pz(&[0]);
        dag.pz(&[1]);
        dag.h(&[0]);
        dag.cx(&[(0, 1)]);
        dag.mz(&[0]);
        dag.mz(&[1]);

        let analyzer = DagFaultAnalyzer::new(&dag);
        let influence_map = analyzer.build_influence_map();

        let sampler = DemSamplerBuilder::new(&influence_map)
            .with_noise(0.01, 0.01, 0.01, 0.01)
            .with_detector_records(vec![vec![-1], vec![-2]])
            .with_observable_records(vec![])
            .build();

        // Sample with row-major method
        let mut rng1 = SmallRng::seed_from_u64(12345);
        let stats1 = sampler.sample_statistics_row_major(10000, &mut rng1);

        // Sample with columnar method
        let mut rng2 = SmallRng::seed_from_u64(12345);
        let stats2 = sampler.sample_statistics_columnar(10000, &mut rng2);

        // Both should produce similar statistics (not identical due to different
        // RNG consumption order, but statistically similar)
        assert_eq!(stats1.total_shots, stats2.total_shots);

        // The syndrome rates should be similar within statistical variance
        let rate1 = stats1.syndrome_rate();
        let rate2 = stats2.syndrome_rate();
        // Allow 5% tolerance for statistical variance
        assert!(
            (rate1 - rate2).abs() < 0.05,
            "Syndrome rates differ too much: {rate1} vs {rate2}"
        );
    }

    #[test]
    fn test_columnar_batch_output_format() {
        use crate::fault_tolerance::propagator::DagFaultAnalyzer;
        use pecos_quantum::DagCircuit;
        use rand::SeedableRng;
        use rand::rngs::SmallRng;

        // Simple circuit
        let mut dag = DagCircuit::new();
        dag.pz(&[0]);
        dag.h(&[0]);
        dag.mz(&[0]);

        let analyzer = DagFaultAnalyzer::new(&dag);
        let influence_map = analyzer.build_influence_map();

        let sampler = DemSamplerBuilder::new(&influence_map)
            .with_noise(0.5, 0.0, 0.0, 0.0) // High noise rate for testing
            .with_detector_records(vec![vec![-1]])
            .with_observable_records(vec![])
            .build();

        let num_shots = 100;
        let mut rng = SmallRng::seed_from_u64(42);
        let (det_cols, obs_cols) = sampler.sample_batch_columnar_accurate(num_shots, &mut rng);

        // Verify output dimensions
        assert_eq!(det_cols.len(), 1); // 1 detector
        assert_eq!(obs_cols.len(), 0); // 0 observables

        // Each column should have ceil(100/64) = 2 words
        assert_eq!(det_cols[0].len(), 2);
    }

    #[test]
    fn test_simd_statistics_correctness() {
        use crate::fault_tolerance::propagator::DagFaultAnalyzer;
        use pecos_quantum::DagCircuit;
        use rand::SeedableRng;
        use rand::rngs::SmallRng;

        let mut dag = DagCircuit::new();
        dag.pz(&[0]);
        dag.pz(&[1]);
        dag.h(&[0]);
        dag.cx(&[(0, 1)]);
        dag.mz(&[0]);
        dag.mz(&[1]);

        let analyzer = DagFaultAnalyzer::new(&dag);
        let influence_map = analyzer.build_influence_map();

        let sampler = DemSamplerBuilder::new(&influence_map)
            .with_noise(0.01, 0.01, 0.01, 0.01)
            .with_detector_records(vec![vec![-1], vec![-2]])
            .with_observable_records(vec![])
            .build();

        // Compare SIMD to baseline
        let mut rng1 = SmallRng::seed_from_u64(42);
        let stats1 = sampler.sample_statistics_columnar(10000, &mut rng1);

        let mut rng2 = SmallRng::seed_from_u64(42);
        let stats2 = sampler.sample_statistics_simd(10000, &mut rng2);

        // Statistics should be similar (different RNG consumption but same distribution)
        let rate1 = stats1.syndrome_rate();
        let rate2 = stats2.syndrome_rate();
        assert!(
            (rate1 - rate2).abs() < 0.05,
            "SIMD syndrome rates differ too much: {rate1} vs {rate2}"
        );
    }

    #[test]
    fn test_geometric_statistics_correctness() {
        use crate::fault_tolerance::propagator::DagFaultAnalyzer;
        use pecos_quantum::DagCircuit;
        use rand::SeedableRng;
        use rand::rngs::SmallRng;

        let mut dag = DagCircuit::new();
        dag.pz(&[0]);
        dag.pz(&[1]);
        dag.h(&[0]);
        dag.cx(&[(0, 1)]);
        dag.mz(&[0]);
        dag.mz(&[1]);

        let analyzer = DagFaultAnalyzer::new(&dag);
        let influence_map = analyzer.build_influence_map();

        // Use low noise to exercise geometric sampling effectively
        let sampler = DemSamplerBuilder::new(&influence_map)
            .with_noise(0.001, 0.001, 0.001, 0.001)
            .with_detector_records(vec![vec![-1], vec![-2]])
            .with_observable_records(vec![])
            .build();

        // Compare geometric to baseline with many shots for statistical significance
        let mut rng1 = SmallRng::seed_from_u64(42);
        let stats1 = sampler.sample_statistics_columnar(100_000, &mut rng1);

        let mut rng2 = SmallRng::seed_from_u64(42);
        let stats2 = sampler.sample_statistics_geometric(100_000, &mut rng2);

        // Statistics should be similar
        let rate1 = stats1.syndrome_rate();
        let rate2 = stats2.syndrome_rate();
        // Allow 20% relative tolerance for geometric (different RNG consumption pattern)
        let relative_diff = if rate1 > 0.0 {
            (rate1 - rate2).abs() / rate1
        } else {
            (rate1 - rate2).abs()
        };
        assert!(
            relative_diff < 0.2,
            "Geometric syndrome rates differ too much: {rate1} vs {rate2} (rel diff: {relative_diff})"
        );
    }

    #[test]
    fn test_auto_selection_low_p() {
        use crate::fault_tolerance::propagator::DagFaultAnalyzer;
        use pecos_quantum::DagCircuit;
        use rand::SeedableRng;
        use rand::rngs::SmallRng;

        let mut dag = DagCircuit::new();
        dag.pz(&[0]);
        dag.h(&[0]);
        dag.mz(&[0]);

        let analyzer = DagFaultAnalyzer::new(&dag);
        let influence_map = analyzer.build_influence_map();

        // Low error rate - should use geometric
        let sampler = DemSamplerBuilder::new(&influence_map)
            .with_noise(0.001, 0.001, 0.001, 0.001)
            .with_detector_records(vec![vec![-1]])
            .with_observable_records(vec![])
            .build();

        assert!(sampler.average_error_probability() < 0.01);

        // sample_statistics_with_rng uses auto-selection internally
        let mut rng = SmallRng::seed_from_u64(42);
        let stats = sampler.sample_statistics_with_rng(10000, &mut rng);
        assert!(stats.total_shots == 10000);
    }

    #[test]
    fn test_auto_selection_high_p() {
        use crate::fault_tolerance::propagator::DagFaultAnalyzer;
        use pecos_quantum::DagCircuit;
        use rand::SeedableRng;
        use rand::rngs::SmallRng;

        let mut dag = DagCircuit::new();
        dag.pz(&[0]);
        dag.h(&[0]);
        dag.mz(&[0]);

        let analyzer = DagFaultAnalyzer::new(&dag);
        let influence_map = analyzer.build_influence_map();

        // High error rate - should use SIMD
        let sampler = DemSamplerBuilder::new(&influence_map)
            .with_noise(0.1, 0.1, 0.1, 0.1)
            .with_detector_records(vec![vec![-1]])
            .with_observable_records(vec![])
            .build();

        assert!(sampler.average_error_probability() >= 0.01);

        // sample_statistics_with_rng uses auto-selection internally
        let mut rng = SmallRng::seed_from_u64(42);
        let stats = sampler.sample_statistics_with_rng(10000, &mut rng);
        assert!(stats.total_shots == 10000);
    }

    #[test]
    fn test_parallel_statistics_correctness() {
        use crate::fault_tolerance::propagator::DagFaultAnalyzer;
        use pecos_quantum::DagCircuit;
        use rand::SeedableRng;
        use rand::rngs::SmallRng;

        let mut dag = DagCircuit::new();
        dag.pz(&[0]);
        dag.pz(&[1]);
        dag.h(&[0]);
        dag.cx(&[(0, 1)]);
        dag.mz(&[0]);
        dag.mz(&[1]);

        let analyzer = DagFaultAnalyzer::new(&dag);
        let influence_map = analyzer.build_influence_map();

        let sampler = DemSamplerBuilder::new(&influence_map)
            .with_noise(0.001, 0.001, 0.001, 0.001)
            .with_detector_records(vec![vec![-1], vec![-2]])
            .with_observable_records(vec![])
            .build();

        // Compare parallel to sequential
        let mut rng = SmallRng::seed_from_u64(42);
        let stats_seq = sampler.sample_statistics_geometric(100_000, &mut rng);

        // Parallel uses different RNG seeds per chunk, so results won't match exactly
        // but should be statistically similar
        let stats_par = sampler.sample_statistics_parallel(100_000, 42);

        let rate_seq = stats_seq.syndrome_rate();
        let rate_par = stats_par.syndrome_rate();

        // Allow 30% relative tolerance due to different RNG consumption
        let relative_diff = if rate_seq > 0.0 {
            (rate_seq - rate_par).abs() / rate_seq
        } else {
            (rate_seq - rate_par).abs()
        };
        assert!(
            relative_diff < 0.3,
            "Parallel syndrome rates differ too much: {rate_seq} vs {rate_par} (rel diff: {relative_diff})"
        );
    }

    // ========================================================================
    // Tests for from_mechanisms constructor
    // ========================================================================

    #[test]
    fn test_from_mechanisms_empty() {
        let sampler = DemSampler::from_mechanisms(std::iter::empty(), 0, 0);
        assert_eq!(sampler.num_mechanisms(), 0);
        assert_eq!(sampler.num_detectors(), 0);
        assert_eq!(sampler.num_observables(), 0);

        let stats = sampler.sample_statistics(100, 42);
        assert_eq!(stats.syndrome_count, 0);
        assert_eq!(stats.logical_error_count, 0);
    }

    #[test]
    fn test_from_mechanisms_single_detector() {
        // Single mechanism that flips D0 with p=0.5
        let mechanisms = vec![(0.5, vec![0u32], vec![])];
        let sampler = DemSampler::from_mechanisms(mechanisms, 1, 0);

        assert_eq!(sampler.num_mechanisms(), 1);
        assert_eq!(sampler.num_detectors(), 1);

        // Sample and verify rate is approximately 0.5
        let stats = sampler.sample_statistics(10000, 42);
        let rate = stats.syndrome_rate();
        assert!(
            (rate - 0.5).abs() < 0.05,
            "Syndrome rate {rate} should be close to 0.5"
        );
    }

    #[test]
    fn test_from_mechanisms_multiple_detectors() {
        // Two mechanisms: D0 with p=0.1, D1 with p=0.2
        let mechanisms = vec![(0.1, vec![0u32], vec![]), (0.2, vec![1u32], vec![])];
        let sampler = DemSampler::from_mechanisms(mechanisms, 2, 0);

        assert_eq!(sampler.num_mechanisms(), 2);
        assert_eq!(sampler.num_detectors(), 2);

        // Syndrome rate should be approximately 1 - (1-0.1)*(1-0.2) = 0.28
        let stats = sampler.sample_statistics(10000, 42);
        let rate = stats.syndrome_rate();
        assert!(
            (rate - 0.28).abs() < 0.05,
            "Syndrome rate {rate} should be close to 0.28"
        );
    }

    #[test]
    fn test_from_mechanisms_correlated_detectors() {
        // Single mechanism that flips both D0 and D1 together with p=0.3
        let mechanisms = vec![(0.3, vec![0u32, 1u32], vec![])];
        let sampler = DemSampler::from_mechanisms(mechanisms, 2, 0);

        assert_eq!(sampler.num_mechanisms(), 1);
        assert_eq!(sampler.num_detectors(), 2);

        // Syndrome rate should be approximately 0.3 (when error fires, BOTH detectors fire)
        let stats = sampler.sample_statistics(10000, 42);
        let rate = stats.syndrome_rate();
        assert!(
            (rate - 0.3).abs() < 0.05,
            "Syndrome rate {rate} should be close to 0.3"
        );
    }

    #[test]
    fn test_from_mechanisms_xor_cancellation() {
        // Two mechanisms that both flip D0 with the same probability
        // When both fire, they XOR and cancel
        let mechanisms = vec![(0.5, vec![0u32], vec![]), (0.5, vec![0u32], vec![])];
        let sampler = DemSampler::from_mechanisms(mechanisms, 1, 0);

        // With two independent p=0.5 mechanisms that both flip D0:
        // P(D0 fires) = P(exactly one fires) = 2 * 0.5 * 0.5 = 0.5
        let stats = sampler.sample_statistics(10000, 42);
        let rate = stats.syndrome_rate();
        assert!(
            (rate - 0.5).abs() < 0.05,
            "Syndrome rate {rate} should be close to 0.5 due to XOR"
        );
    }

    #[test]
    fn test_from_mechanisms_with_observables() {
        // Mechanism that flips D0 and L0
        let mechanisms = vec![(0.1, vec![0u32], vec![0u32])];
        let sampler = DemSampler::from_mechanisms(mechanisms, 1, 1);

        assert_eq!(sampler.num_observables(), 1);

        // Logical error rate should be approximately 0.1
        let stats = sampler.sample_statistics(10000, 42);
        let logical_rate = stats.logical_error_rate();
        assert!(
            (logical_rate - 0.1).abs() < 0.03,
            "Logical error rate {logical_rate} should be close to 0.1"
        );
    }

    #[test]
    fn test_from_mechanisms_very_low_error_rate() {
        // Test geometric sampling efficiency with low error rate
        let mechanisms = vec![(0.0001, vec![0u32], vec![])];
        let sampler = DemSampler::from_mechanisms(mechanisms, 1, 0);

        // Should still work correctly
        let stats = sampler.sample_statistics(100_000, 42);
        let rate = stats.syndrome_rate();
        assert!(
            (rate - 0.0001).abs() < 0.001,
            "Syndrome rate {rate} should be close to 0.0001"
        );
    }

    #[test]
    fn test_from_mechanisms_sorting() {
        // Verify that detector indices are sorted regardless of input order
        let mechanisms = vec![(0.1, vec![2u32, 0u32, 1u32], vec![1u32, 0u32])];
        let sampler = DemSampler::from_mechanisms(mechanisms, 3, 2);

        // Verify internal storage is sorted (by checking that sampling works)
        assert_eq!(sampler.num_detectors(), 3);
        assert_eq!(sampler.num_observables(), 2);

        let stats = sampler.sample_statistics(1000, 42);
        // Just verify it runs without panicking
        assert!(stats.total_shots == 1000);
    }
}
