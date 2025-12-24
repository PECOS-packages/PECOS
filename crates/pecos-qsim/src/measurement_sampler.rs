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

//! Efficient sampling from symbolic measurement histories.
//!
//! This module provides two sampler implementations:
//!
//! - [`SequentialMeasurementSampler`]: Processes one shot at a time (row-major computation)
//! - [`MeasurementSampler`]: Processes one measurement at a time across all shots (column-major)
//!
//! Both samplers output data in column-major format (`Vec<Vec<u64>>`) or as [`SampleResult`]
//! for efficient storage and bulk operations. The columnar approach is generally faster
//! for large numbers of shots due to better SIMD utilization and batched random number
//! generation.
//!
//! # Example
//!
//! ```rust
//! use pecos_qsim::symbolic_sparse_stab::StdSymbolicSparseStab;
//! use pecos_qsim::measurement_sampler::{SequentialMeasurementSampler, MeasurementSampler};
//!
//! // Create a Bell state and measure
//! let mut sim = StdSymbolicSparseStab::new(2);
//! sim.h(0).cx(0, 1);
//! sim.mz(0);
//! sim.mz(1);
//!
//! // Using shot-by-shot sampler
//! let sampler = SequentialMeasurementSampler::new(sim.measurement_history());
//! let result = sampler.sample(1000);
//!
//! // Using columnar sampler (faster for many shots)
//! let sampler = MeasurementSampler::new(sim.measurement_history());
//! let result = sampler.sample(1000);
//!
//! // For reproducible results, use a seed
//! let result = sampler.sample_with_seed(1000, 42);
//!
//! // Access individual bits
//! let m0_shot0 = result.get(0, 0);
//! ```

use crate::symbolic_sparse_stab::MeasurementHistory;
use pecos_core::{Bit, Bits};
use pecos_rng::{PecosRng, Rng, RngBulkExt, SeedableRng};
use wide::u64x4;

// ============================================================================
// Common types
// ============================================================================

/// Classification of a measurement for efficient sampling.
#[derive(Clone, Debug)]
pub enum MeasurementKind {
    /// Deterministic value (no dependencies, just 0 or 1)
    Fixed(bool),
    /// Random 50/50 outcome
    Random,
    /// Copy of another measurement (single dep, no flip)
    Copy(usize),
    /// Negation of another measurement (single dep, with flip)
    CopyFlipped(usize),
    /// Computed from XOR of dependencies plus optional flip
    Computed {
        /// Indices of measurements to XOR together
        deps: Vec<usize>,
        /// Whether to flip the result
        flip: bool,
    },
}

impl MeasurementKind {
    /// Create measurement kinds from a measurement history.
    ///
    /// This performs optimizations like detecting simple copies (single dependency, no flip).
    ///
    /// # Panics
    ///
    /// Panics if a deterministic measurement result with exactly one outcome has an empty
    /// outcome set. This is a logical invariant - if `outcome.len() == 1`, then
    /// `outcome.iter().next()` must succeed.
    #[must_use]
    pub fn from_history(history: &MeasurementHistory) -> Vec<Self> {
        history
            .iter()
            .map(|result| {
                if !result.is_deterministic {
                    MeasurementKind::Random
                } else if result.outcome.is_empty() {
                    MeasurementKind::Fixed(result.flip)
                } else if result.outcome.len() == 1 {
                    // Single dependency = copy or negation
                    let src = result.outcome.iter().next().unwrap();
                    if result.flip {
                        MeasurementKind::CopyFlipped(src)
                    } else {
                        MeasurementKind::Copy(src)
                    }
                } else {
                    MeasurementKind::Computed {
                        deps: result.outcome.iter().collect(),
                        flip: result.flip,
                    }
                }
            })
            .collect()
    }

    /// Generate a random measurement history for testing and benchmarking.
    ///
    /// # Parameters
    /// - `num_measurements`: Total number of measurements to generate
    /// - `prob_random`: Probability that a measurement is random (non-deterministic)
    /// - `prob_fixed`: Probability that a deterministic measurement is fixed (no deps)
    /// - `max_deps`: Maximum number of dependencies for computed measurements
    /// - `rng`: Random number generator
    ///
    /// Dependencies are always to earlier measurements (valid DAG structure).
    #[must_use]
    pub fn generate_random<R: Rng>(
        num_measurements: usize,
        prob_random: f64,
        prob_fixed: f64,
        max_deps: usize,
        rng: &mut R,
    ) -> Vec<Self> {
        let mut measurements = Vec::with_capacity(num_measurements);

        for i in 0..num_measurements {
            let kind = if rng.random::<f64>() < prob_random {
                // Random measurement
                MeasurementKind::Random
            } else if i == 0 || rng.random::<f64>() < prob_fixed {
                // Fixed value (no dependencies)
                MeasurementKind::Fixed(rng.random::<bool>())
            } else {
                // Computed from earlier measurements
                let num_deps = if max_deps == 0 {
                    0
                } else {
                    rng.random_range(1..=max_deps.min(i))
                };

                // Pick random earlier measurements as dependencies
                let mut deps: Vec<usize> = (0..i).collect();
                // Shuffle and take first num_deps
                for j in 0..num_deps.min(deps.len()) {
                    let swap_idx = rng.random_range(j..deps.len());
                    deps.swap(j, swap_idx);
                }
                deps.truncate(num_deps);
                deps.sort_unstable();

                MeasurementKind::Computed {
                    deps,
                    flip: rng.random::<bool>(),
                }
            };
            measurements.push(kind);
        }

        measurements
    }

    /// Validate a sequence of measurement kinds for correctness.
    ///
    /// Checks that:
    /// - All dependency indices are within bounds (< current index)
    /// - No duplicate dependencies within a single Computed measurement
    /// - Dependencies form a valid DAG (no forward references)
    ///
    /// # Errors
    ///
    /// Returns [`MeasurementValidationError`] if validation fails:
    /// - [`ForwardReference`](MeasurementValidationError::ForwardReference) if a measurement
    ///   depends on a later measurement
    /// - [`EmptyDependencies`](MeasurementValidationError::EmptyDependencies) if a Computed
    ///   measurement has no dependencies
    /// - [`DuplicateDependencies`](MeasurementValidationError::DuplicateDependencies) if a
    ///   measurement has duplicate dependency indices
    pub fn validate_sequence(measurements: &[Self]) -> Result<(), MeasurementValidationError> {
        for (idx, kind) in measurements.iter().enumerate() {
            match kind {
                MeasurementKind::Fixed(_) | MeasurementKind::Random => {}
                MeasurementKind::Copy(src) | MeasurementKind::CopyFlipped(src) => {
                    if *src >= idx {
                        return Err(MeasurementValidationError::ForwardReference {
                            measurement_idx: idx,
                            dependency_idx: *src,
                        });
                    }
                }
                MeasurementKind::Computed { deps, .. } => {
                    // Check for empty deps (should use Fixed instead)
                    if deps.is_empty() {
                        return Err(MeasurementValidationError::EmptyDependencies {
                            measurement_idx: idx,
                        });
                    }
                    // Check for forward references
                    for &dep in deps {
                        if dep >= idx {
                            return Err(MeasurementValidationError::ForwardReference {
                                measurement_idx: idx,
                                dependency_idx: dep,
                            });
                        }
                    }
                    // Check for duplicates
                    let mut seen = std::collections::HashSet::new();
                    for &dep in deps {
                        if !seen.insert(dep) {
                            return Err(MeasurementValidationError::DuplicateDependency {
                                measurement_idx: idx,
                                dependency_idx: dep,
                            });
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// Error type for measurement sequence validation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MeasurementValidationError {
    /// A measurement references a dependency that comes after it (invalid DAG).
    ForwardReference {
        measurement_idx: usize,
        dependency_idx: usize,
    },
    /// A Computed measurement has duplicate dependencies.
    DuplicateDependency {
        measurement_idx: usize,
        dependency_idx: usize,
    },
    /// A Computed measurement has no dependencies (should be Fixed instead).
    EmptyDependencies { measurement_idx: usize },
}

impl std::fmt::Display for MeasurementValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ForwardReference {
                measurement_idx,
                dependency_idx,
            } => {
                write!(
                    f,
                    "Measurement {measurement_idx} has forward reference to {dependency_idx}"
                )
            }
            Self::DuplicateDependency {
                measurement_idx,
                dependency_idx,
            } => {
                write!(
                    f,
                    "Measurement {measurement_idx} has duplicate dependency {dependency_idx}"
                )
            }
            Self::EmptyDependencies { measurement_idx } => {
                write!(
                    f,
                    "Measurement {measurement_idx} is Computed but has no dependencies"
                )
            }
        }
    }
}

impl std::error::Error for MeasurementValidationError {}

// ============================================================================
// Shot-by-shot sampler (row-major computation, column-major output)
// ============================================================================

/// Sequential measurement sampler that processes one complete shot at a time.
///
/// This sampler iterates through all measurements for each shot before moving
/// to the next shot. The output is stored in column-major format (`Vec<Vec<u64>>`)
/// for efficient bulk operations.
///
/// For most use cases, prefer [`MeasurementSampler`] which uses a faster
/// columnar algorithm with better SIMD utilization and batched random number generation.
#[derive(Clone, Debug)]
pub struct SequentialMeasurementSampler {
    /// Preprocessed measurement classifications
    measurements: Vec<MeasurementKind>,
}

impl SequentialMeasurementSampler {
    /// Create a new sampler from a measurement history.
    #[must_use]
    pub fn new(history: &MeasurementHistory) -> Self {
        Self {
            measurements: MeasurementKind::from_history(history),
        }
    }

    /// Create a new sampler from pre-computed measurement kinds.
    ///
    /// Useful for testing or when you want to generate random measurement
    /// histories without going through the symbolic stabilizer simulation.
    #[must_use]
    pub fn from_measurements(measurements: Vec<MeasurementKind>) -> Self {
        Self { measurements }
    }

    /// Returns the number of measurements per shot.
    #[inline]
    #[must_use]
    pub fn num_measurements(&self) -> usize {
        self.measurements.len()
    }

    /// Generate multiple shots using raw u64 column storage.
    ///
    /// Returns column-major data: `columns[measurement][word]` where
    /// bit `i` of word `w` corresponds to shot `w*64 + i`.
    #[inline]
    #[must_use]
    pub fn sample_raw<R: Rng>(&self, shots: usize, rng: &mut R) -> Vec<Vec<u64>> {
        if self.measurements.is_empty() || shots == 0 {
            return vec![Vec::new(); self.measurements.len()];
        }

        let num_words = shots.div_ceil(64);
        let num_measurements = self.measurements.len();

        // Initialize columns with zeros
        let mut columns: Vec<Vec<u64>> = vec![vec![0u64; num_words]; num_measurements];

        // Temporary storage for one shot's results
        let mut shot_results = vec![false; num_measurements];

        for shot_idx in 0..shots {
            let word_idx = shot_idx / 64;
            let bit_idx = shot_idx % 64;
            let bit_mask = 1u64 << bit_idx;

            // Compute this shot's measurements
            for (m, kind) in self.measurements.iter().enumerate() {
                let bit = match kind {
                    MeasurementKind::Fixed(value) => *value,
                    MeasurementKind::Random => rng.random::<bool>(),
                    MeasurementKind::Copy(src) => shot_results[*src],
                    MeasurementKind::CopyFlipped(src) => !shot_results[*src],
                    MeasurementKind::Computed { deps, flip } => {
                        let mut value = *flip;
                        for &dep in deps {
                            value ^= shot_results[dep];
                        }
                        value
                    }
                };
                shot_results[m] = bit;

                // Store in column
                if bit {
                    columns[m][word_idx] |= bit_mask;
                }
            }
        }

        columns
    }

    /// Sample measurement outcomes and return a [`SampleResult`].
    ///
    /// This is the primary sampling method. Uses [`PecosRng`] for high performance.
    ///
    /// # Arguments
    /// * `shots` - Number of measurement shots to generate
    ///
    /// # Returns
    /// A [`SampleResult`] containing the sampled measurement outcomes.
    #[inline]
    #[must_use]
    pub fn sample(&self, shots: usize) -> SampleResult {
        let mut rng = PecosRng::from_os_rng();
        self.sample_with_rng(shots, &mut rng)
    }

    /// Sample measurement outcomes with a specific seed for reproducibility.
    ///
    /// # Arguments
    /// * `shots` - Number of measurement shots to generate
    /// * `seed` - Seed for the random number generator
    ///
    /// # Returns
    /// A [`SampleResult`] containing the sampled measurement outcomes.
    #[inline]
    #[must_use]
    pub fn sample_with_seed(&self, shots: usize, seed: u64) -> SampleResult {
        let mut rng = PecosRng::seed_from_u64(seed);
        self.sample_with_rng(shots, &mut rng)
    }

    /// Sample measurement outcomes with a custom random number generator.
    ///
    /// # Arguments
    /// * `shots` - Number of measurement shots to generate
    /// * `rng` - Random number generator to use
    ///
    /// # Returns
    /// A [`SampleResult`] containing the sampled measurement outcomes.
    #[inline]
    #[must_use]
    pub fn sample_with_rng<R: Rng>(&self, shots: usize, rng: &mut R) -> SampleResult {
        let columns = self.sample_raw(shots, rng);
        SampleResult::new(columns, shots)
    }
}

// ============================================================================
// MeasurementSampler (column-major, SIMD-friendly, optimized for large shot counts)
// ============================================================================

/// High-performance measurement sampler using a columnar algorithm.
///
/// This is the recommended sampler for generating measurement outcomes from
/// a symbolic measurement history. It processes all shots for measurement 0,
/// then all shots for measurement 1, etc. This enables:
/// - Batched random number generation (generate 64 random bits at once)
/// - SIMD-friendly XOR operations on entire columns (operating on u64 words)
/// - Better cache locality for large shot counts
///
/// Internally uses `Vec<u64>` for columns to maximize performance.
///
/// # Example
///
/// ```
/// use pecos_qsim::prelude::*;
/// use pecos_qsim::measurement_sampler::MeasurementSampler;
///
/// // Create a Bell state and measure
/// let mut sim = StdSymbolicSparseStab::new(2);
/// sim.h(0).cx(0, 1);
/// sim.mz(0);
/// sim.mz(1);
///
/// // Sample 1000 shots from the measurement history
/// let sampler = MeasurementSampler::new(sim.measurement_history());
/// let result = sampler.sample(1000);
///
/// // Access individual outcomes
/// for shot in 0..5 {
///     println!("Shot {}: q0={}, q1={}", shot, result.get(shot, 0), result.get(shot, 1));
/// }
/// ```
#[derive(Clone, Debug)]
pub struct MeasurementSampler {
    /// Preprocessed measurement classifications
    measurements: Vec<MeasurementKind>,
}

impl MeasurementSampler {
    /// Create a new sampler from a measurement history.
    #[must_use]
    pub fn new(history: &MeasurementHistory) -> Self {
        Self {
            measurements: MeasurementKind::from_history(history),
        }
    }

    /// Create a new sampler from pre-computed measurement kinds.
    ///
    /// Useful for testing or when you want to generate random measurement
    /// histories without going through the symbolic stabilizer simulation.
    #[must_use]
    pub fn from_measurements(measurements: Vec<MeasurementKind>) -> Self {
        Self { measurements }
    }

    /// Returns the number of measurements per shot.
    #[inline]
    #[must_use]
    pub fn num_measurements(&self) -> usize {
        self.measurements.len()
    }

    /// Convert a SIMD column to a u64 column via zero-copy transmute.
    #[inline]
    fn simd_column_to_u64_vec(simd_col: Vec<u64x4>, num_words: usize) -> Vec<u64> {
        // Safety: u64x4 is repr(C) and contains exactly 4 u64s in order.
        // We're converting Vec<u64x4> to Vec<u64> with 4x the length.
        let simd_len = simd_col.len();
        let u64_capacity = simd_len * 4;

        // Convert Vec<u64x4> to Vec<u64> without copying
        let mut simd_col = std::mem::ManuallyDrop::new(simd_col);
        let ptr = simd_col.as_mut_ptr().cast::<u64>();

        // Safety: u64x4 has same alignment as u64 (or stricter), and we're
        // reinterpreting the memory as a flat array of u64s.
        let mut result = unsafe { Vec::from_raw_parts(ptr, u64_capacity, u64_capacity) };

        // Truncate to the actual number of words needed
        result.truncate(num_words);
        result
    }

    /// Sample directly to raw u64 columns.
    ///
    /// Returns a vector of columns where each column is a `Vec<u64>` representing
    /// all shots for one measurement. Bit `i` of word `w` corresponds to shot `w*64 + i`.
    ///
    /// Internally uses SIMD operations for better performance.
    #[inline]
    #[must_use]
    pub fn sample_raw<R: Rng + RngBulkExt>(&self, shots: usize, rng: &mut R) -> Vec<Vec<u64>> {
        if self.measurements.is_empty() || shots == 0 {
            return vec![Vec::new(); self.measurements.len()];
        }

        let num_words = shots.div_ceil(64);

        // Use SIMD internally, then convert to Vec<u64>
        let simd_columns = self.sample_raw_simd(shots, rng);

        // Convert each SIMD column to u64 via zero-copy transmute
        simd_columns
            .into_iter()
            .map(|col| Self::simd_column_to_u64_vec(col, num_words))
            .collect()
    }

    // ========================================================================
    // SIMD-native API for advanced users
    // ========================================================================
    //
    // These methods work with u64x4 (256-bit SIMD) columns directly.
    // Each u64x4 holds 4 u64s = 256 bits = 256 shots.
    // Use these for maximum performance when you can consume SIMD data directly.

    /// Generate a SIMD column of random bits.
    ///
    /// Uses bulk fill for better performance (~2x faster than individual calls).
    #[inline]
    fn generate_random_column_simd<R: Rng + RngBulkExt>(
        num_simd_words: usize,
        rng: &mut R,
    ) -> Vec<u64x4> {
        // Allocate the vector with zeros (will be overwritten)
        let mut column: Vec<u64x4> = vec![u64x4::splat(0); num_simd_words];

        // Safety: u64x4 is repr(C) containing 4 u64s, so we can treat it as &mut [u64]
        // This avoids the overhead of constructing u64x4 values one at a time.
        let u64_slice: &mut [u64] = unsafe {
            std::slice::from_raw_parts_mut(column.as_mut_ptr().cast::<u64>(), num_simd_words * 4)
        };

        // Use bulk fill from RngBulkExt trait (optimized for PECOS RNGs)
        rng.fill_u64_bulk(u64_slice);

        column
    }

    /// Compute a SIMD column by `XORing` dependency columns.
    #[inline]
    fn compute_xor_column_simd(
        columns: &[Vec<u64x4>],
        deps: &[usize],
        flip: bool,
        num_simd_words: usize,
    ) -> Vec<u64x4> {
        let init = if flip {
            u64x4::splat(!0u64)
        } else {
            u64x4::splat(0u64)
        };
        let mut result = vec![init; num_simd_words];

        for &dep_idx in deps {
            let dep_column = &columns[dep_idx];
            for (r, d) in result.iter_mut().zip(dep_column.iter()) {
                *r ^= *d;
            }
        }

        result
    }

    /// Sample directly to SIMD-native u64x4 columns (internal implementation).
    ///
    /// Returns a vector of columns where each column is a `Vec<u64x4>`.
    /// Each `u64x4` holds 4 u64s (256 bits = 256 shots).
    #[inline]
    fn sample_raw_simd<R: Rng + RngBulkExt>(&self, shots: usize, rng: &mut R) -> Vec<Vec<u64x4>> {
        if self.measurements.is_empty() || shots == 0 {
            return vec![Vec::new(); self.measurements.len()];
        }

        let num_words = shots.div_ceil(64);
        let num_simd_words = num_words.div_ceil(4);
        let num_measurements = self.measurements.len();

        let mut columns: Vec<Vec<u64x4>> = Vec::with_capacity(num_measurements);

        for kind in &self.measurements {
            match kind {
                MeasurementKind::Fixed(value) => {
                    let fill = if *value {
                        u64x4::splat(!0u64)
                    } else {
                        u64x4::splat(0u64)
                    };
                    columns.push(vec![fill; num_simd_words]);
                }
                MeasurementKind::Random => {
                    columns.push(Self::generate_random_column_simd(num_simd_words, rng));
                }
                MeasurementKind::Copy(src) => {
                    columns.push(columns[*src].clone());
                }
                MeasurementKind::CopyFlipped(src) => {
                    let src_col = &columns[*src];
                    let mut result = Vec::with_capacity(num_simd_words);
                    for v in src_col {
                        result.push(!*v);
                    }
                    columns.push(result);
                }
                MeasurementKind::Computed { deps, flip } => {
                    columns.push(Self::compute_xor_column_simd(
                        &columns,
                        deps,
                        *flip,
                        num_simd_words,
                    ));
                }
            }
        }

        columns
    }
}

// ============================================================================
// SampleResult - efficient storage with convenient access
// ============================================================================

/// Efficient storage for measurement samples with convenient bit access.
///
/// Stores data in column-major format (`Vec<Vec<u64>>`) for memory efficiency,
/// but provides convenient accessors like `result.get(shot, measurement)`.
///
/// # Memory Layout
///
/// Data is stored as columns where each column is a `Vec<u64>`:
/// - `columns[measurement][word]` where `word = shot / 64`
/// - Bit position within word: `shot % 64`
///
/// This is more memory efficient than `Vec<BitVec>` and allows efficient
/// bulk operations on entire columns.
///
/// # Example
///
/// ```rust
/// use pecos_qsim::measurement_sampler::{MeasurementSampler, SampleResult};
/// use pecos_qsim::symbolic_sparse_stab::StdSymbolicSparseStab;
///
/// let mut sim = StdSymbolicSparseStab::new(2);
/// sim.h(0).cx(0, 1);
/// sim.mz(0);
/// sim.mz(1);
///
/// let sampler = MeasurementSampler::new(sim.measurement_history());
/// let result = sampler.sample(1000);
///
/// // Access individual bits
/// let m0_shot0 = result.get(0, 0);
/// let m1_shot0 = result.get(0, 1);
///
/// // For Bell state, measurements should be correlated
/// assert_eq!(m0_shot0, m1_shot0);
/// ```
#[derive(Clone, Debug)]
pub struct SampleResult {
    /// Column-major storage: `columns[measurement][word]`
    columns: Vec<Vec<u64>>,
    /// Number of shots (needed because last word may be partial)
    shots: usize,
}

impl SampleResult {
    /// Create a new `SampleResult` from raw column data.
    #[must_use]
    pub fn new(columns: Vec<Vec<u64>>, shots: usize) -> Self {
        Self { columns, shots }
    }

    /// Get the measurement result for a specific shot and measurement.
    ///
    /// # Arguments
    /// * `shot` - The shot/sample index (0 to `shots()-1`)
    /// * `measurement` - The measurement index (0 to `num_measurements()-1`)
    ///
    /// # Panics
    /// Panics if `shot >= self.shots()` or `measurement >= self.num_measurements()`.
    #[inline]
    #[must_use]
    pub fn get(&self, shot: usize, measurement: usize) -> Bit {
        debug_assert!(shot < self.shots, "shot index out of bounds");
        debug_assert!(
            measurement < self.columns.len(),
            "measurement index out of bounds"
        );

        let word_idx = shot / 64;
        let bit_idx = shot % 64;
        Bit((self.columns[measurement][word_idx] >> bit_idx) & 1 != 0)
    }

    /// Get the measurement result, returning `None` if out of bounds.
    ///
    /// # Arguments
    /// * `shot` - The shot/sample index
    /// * `measurement` - The measurement index
    #[inline]
    #[must_use]
    pub fn try_get(&self, shot: usize, measurement: usize) -> Option<Bit> {
        if shot >= self.shots || measurement >= self.columns.len() {
            return None;
        }
        Some(self.get(shot, measurement))
    }

    /// Returns the number of shots.
    #[inline]
    #[must_use]
    pub fn shots(&self) -> usize {
        self.shots
    }

    /// Returns the number of measurements per shot.
    #[inline]
    #[must_use]
    pub fn num_measurements(&self) -> usize {
        self.columns.len()
    }

    /// Get a reference to the raw column data.
    ///
    /// Useful for efficient bulk operations on entire columns.
    #[inline]
    #[must_use]
    pub fn columns(&self) -> &[Vec<u64>] {
        &self.columns
    }

    /// Get a specific column (all shots for one measurement).
    #[inline]
    #[must_use]
    pub fn column(&self, measurement: usize) -> &[u64] {
        &self.columns[measurement]
    }

    /// Consume self and return the raw column data.
    #[must_use]
    pub fn into_columns(self) -> Vec<Vec<u64>> {
        self.columns
    }

    /// Count the number of 1s for a specific measurement across all shots.
    #[must_use]
    pub fn count_ones(&self, measurement: usize) -> usize {
        let col = &self.columns[measurement];
        let full_words = self.shots / 64;
        let remaining_bits = self.shots % 64;

        let mut count: usize = col[..full_words]
            .iter()
            .map(|w| w.count_ones() as usize)
            .sum();

        // Handle partial last word
        if remaining_bits > 0 && full_words < col.len() {
            let mask = (1u64 << remaining_bits) - 1;
            count += (col[full_words] & mask).count_ones() as usize;
        }

        count
    }

    /// Count the number of 0s for a specific measurement across all shots.
    #[must_use]
    pub fn count_zeros(&self, measurement: usize) -> usize {
        self.shots - self.count_ones(measurement)
    }

    /// Get all measurement results for a single shot.
    ///
    /// Returns a `Bits` collection where `result[m]` is the value for measurement `m`.
    /// The `Bits` type displays as a binary string (e.g., "01101").
    ///
    /// # Arguments
    /// * `shot` - The shot/sample index (0 to `shots()-1`)
    ///
    /// # Panics
    /// Panics if `shot >= self.shots()`.
    #[must_use]
    pub fn shot(&self, shot: usize) -> Bits {
        assert!(shot < self.shots, "shot index out of bounds");
        let word_idx = shot / 64;
        let bit_idx = shot % 64;
        let mask = 1u64 << bit_idx;

        self.columns
            .iter()
            .map(|col| Bit((col[word_idx] & mask) != 0))
            .collect()
    }

    /// Format a single shot as a binary string (e.g., "01101").
    ///
    /// Each character represents one measurement: '0' or '1'.
    ///
    /// # Arguments
    /// * `shot` - The shot/sample index (0 to `shots()-1`)
    ///
    /// # Panics
    /// Panics if `shot >= self.shots()`.
    #[must_use]
    pub fn format_shot(&self, shot: usize) -> String {
        assert!(shot < self.shots, "shot index out of bounds");
        let word_idx = shot / 64;
        let bit_idx = shot % 64;
        let mask = 1u64 << bit_idx;

        self.columns
            .iter()
            .map(|col| {
                if (col[word_idx] & mask) != 0 {
                    '1'
                } else {
                    '0'
                }
            })
            .collect()
    }

    /// Iterate over shots, yielding each shot's measurements as a `Bits` collection.
    ///
    /// Note: This allocates a new `Bits` for each shot. For bulk access,
    /// consider working with columns directly.
    pub fn iter_shots(&self) -> impl Iterator<Item = Bits> + '_ {
        (0..self.shots).map(|shot| {
            let word_idx = shot / 64;
            let bit_idx = shot % 64;
            let mask = 1u64 << bit_idx;

            self.columns
                .iter()
                .map(|col| Bit((col[word_idx] & mask) != 0))
                .collect()
        })
    }
}

impl std::ops::Index<(usize, usize)> for SampleResult {
    type Output = Bit;

    /// Index into sample results using `result[(shot, measurement)]` syntax.
    ///
    /// Returns a reference to a static `Bit::ZERO` or `Bit::ONE`.
    ///
    /// # Arguments
    /// * `shot` - The shot/sample index (0 to `shots()-1`)
    /// * `measurement` - The measurement index (0 to `num_measurements()-1`)
    ///
    /// # Panics
    /// Panics if indices are out of bounds.
    ///
    /// # Note
    /// Due to Rust's `Index` trait requirements, this returns a reference.
    /// For a direct value, use `result.get(shot, measurement)`.
    #[inline]
    fn index(&self, (shot, measurement): (usize, usize)) -> &Self::Output {
        if *self.get(shot, measurement) {
            &Bit::ONE
        } else {
            &Bit::ZERO
        }
    }
}

impl MeasurementSampler {
    // ========================================================================
    // Primary API - simple and ergonomic
    // ========================================================================

    /// Sample measurement outcomes and return a [`SampleResult`].
    ///
    /// This is the primary sampling method. Uses a fast non-cryptographic RNG
    /// internally for good performance.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_qsim::prelude::*;
    /// use pecos_qsim::measurement_sampler::MeasurementSampler;
    ///
    /// let mut sim = StdSymbolicSparseStab::new(2);
    /// sim.h(0).cx(0, 1);
    /// sim.mz(0);
    /// sim.mz(1);
    ///
    /// let sampler = MeasurementSampler::new(sim.measurement_history());
    /// let result = sampler.sample(1000);
    ///
    /// // Both qubits should always have the same outcome (Bell state)
    /// assert_eq!(result.get(0, 0), result.get(0, 1));
    /// ```
    #[inline]
    #[must_use]
    pub fn sample(&self, shots: usize) -> SampleResult {
        let mut rng = PecosRng::from_os_rng();
        self.sample_with_rng(shots, &mut rng)
    }

    /// Sample measurement outcomes with a specific seed for reproducibility.
    ///
    /// Use this when you need deterministic, reproducible results (e.g., for
    /// testing or debugging).
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_qsim::prelude::*;
    /// use pecos_qsim::measurement_sampler::MeasurementSampler;
    ///
    /// let mut sim = StdSymbolicSparseStab::new(2);
    /// sim.h(0).cx(0, 1);
    /// sim.mz(0);
    /// sim.mz(1);
    ///
    /// let sampler = MeasurementSampler::new(sim.measurement_history());
    ///
    /// // Same seed produces same results
    /// let result1 = sampler.sample_with_seed(1000, 42);
    /// let result2 = sampler.sample_with_seed(1000, 42);
    /// assert_eq!(result1.get(0, 0), result2.get(0, 0));
    /// ```
    #[inline]
    #[must_use]
    pub fn sample_with_seed(&self, shots: usize, seed: u64) -> SampleResult {
        let mut rng = PecosRng::seed_from_u64(seed);
        self.sample_with_rng(shots, &mut rng)
    }

    /// Sample measurement outcomes with a custom RNG.
    ///
    /// Use this when you need full control over the random number generator.
    #[inline]
    #[must_use]
    pub fn sample_with_rng<R: Rng + RngBulkExt>(&self, shots: usize, rng: &mut R) -> SampleResult {
        let columns = self.sample_raw(shots, rng);
        SampleResult::new(columns, shots)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbolic_sparse_stab::StdSymbolicSparseStab;

    // -------------------------------------------------------------------------
    // Tests for deterministic zero
    // -------------------------------------------------------------------------

    #[test]
    fn test_deterministic_zero_shot() {
        let mut sim = StdSymbolicSparseStab::new(1);
        sim.mz(0);

        let sampler = SequentialMeasurementSampler::new(sim.measurement_history());
        let result = sampler.sample(100);

        for shot in 0..100 {
            assert!(!*result.get(shot, 0), "Expected all measurements to be 0");
        }
    }

    #[test]
    fn test_deterministic_zero_columnar() {
        let mut sim = StdSymbolicSparseStab::new(1);
        sim.mz(0);

        let sampler = MeasurementSampler::new(sim.measurement_history());
        let result = sampler.sample(100);

        for shot in 0..100 {
            assert!(!*result.get(shot, 0), "Expected all measurements to be 0");
        }
    }

    // -------------------------------------------------------------------------
    // Tests for deterministic one
    // -------------------------------------------------------------------------

    #[test]
    fn test_deterministic_one_shot() {
        let mut sim = StdSymbolicSparseStab::new(1);
        sim.x(0);
        sim.mz(0);

        let sampler = SequentialMeasurementSampler::new(sim.measurement_history());
        let result = sampler.sample(100);

        for shot in 0..100 {
            assert!(*result.get(shot, 0), "Expected all measurements to be 1");
        }
    }

    #[test]
    fn test_deterministic_one_columnar() {
        let mut sim = StdSymbolicSparseStab::new(1);
        sim.x(0);
        sim.mz(0);

        let sampler = MeasurementSampler::new(sim.measurement_history());
        let result = sampler.sample(100);

        for shot in 0..100 {
            assert!(*result.get(shot, 0), "Expected all measurements to be 1");
        }
    }

    // -------------------------------------------------------------------------
    // Tests for random measurement
    // -------------------------------------------------------------------------

    #[test]
    fn test_random_measurement_shot() {
        let mut sim = StdSymbolicSparseStab::new(1);
        sim.h(0);
        sim.mz(0);

        let sampler = SequentialMeasurementSampler::new(sim.measurement_history());
        let result = sampler.sample(1000);

        let ones = result.count_ones(0);
        assert!(
            ones > 400 && ones < 600,
            "Expected roughly 50/50 split, got {ones} ones"
        );
    }

    #[test]
    fn test_random_measurement_columnar() {
        let mut sim = StdSymbolicSparseStab::new(1);
        sim.h(0);
        sim.mz(0);

        let sampler = MeasurementSampler::new(sim.measurement_history());
        let result = sampler.sample(1000);

        let ones = result.count_ones(0);
        assert!(
            ones > 400 && ones < 600,
            "Expected roughly 50/50 split, got {ones} ones"
        );
    }

    // -------------------------------------------------------------------------
    // Tests for Bell state correlation
    // -------------------------------------------------------------------------

    #[test]
    fn test_bell_state_correlation_shot() {
        let mut sim = StdSymbolicSparseStab::new(2);
        sim.h(0).cx(0, 1);
        sim.mz(0);
        sim.mz(1);

        let sampler = SequentialMeasurementSampler::new(sim.measurement_history());
        let result = sampler.sample(1000);

        for shot in 0..1000 {
            assert_eq!(
                result.get(shot, 0),
                result.get(shot, 1),
                "Bell state measurements must be correlated"
            );
        }

        let ones = result.count_ones(0);
        assert!(
            ones > 400 && ones < 600,
            "Expected roughly 50/50 for first qubit"
        );
    }

    #[test]
    fn test_bell_state_correlation_columnar() {
        let mut sim = StdSymbolicSparseStab::new(2);
        sim.h(0).cx(0, 1);
        sim.mz(0);
        sim.mz(1);

        let sampler = MeasurementSampler::new(sim.measurement_history());
        let result = sampler.sample(1000);

        for shot in 0..1000 {
            assert_eq!(
                result.get(shot, 0),
                result.get(shot, 1),
                "Bell state measurements must be correlated"
            );
        }

        let ones = result.count_ones(0);
        assert!(
            ones > 400 && ones < 600,
            "Expected roughly 50/50 for first qubit"
        );
    }

    // -------------------------------------------------------------------------
    // Tests for GHZ state correlation
    // -------------------------------------------------------------------------

    #[test]
    fn test_ghz_state_correlation_shot() {
        let mut sim = StdSymbolicSparseStab::new(3);
        sim.h(0).cx(0, 1).cx(1, 2);
        sim.mz(0);
        sim.mz(1);
        sim.mz(2);

        let sampler = SequentialMeasurementSampler::new(sim.measurement_history());
        let result = sampler.sample(1000);

        for shot in 0..1000 {
            assert_eq!(
                result.get(shot, 0),
                result.get(shot, 1),
                "GHZ measurements must be correlated"
            );
            assert_eq!(
                result.get(shot, 1),
                result.get(shot, 2),
                "GHZ measurements must be correlated"
            );
        }
    }

    #[test]
    fn test_ghz_state_correlation_columnar() {
        let mut sim = StdSymbolicSparseStab::new(3);
        sim.h(0).cx(0, 1).cx(1, 2);
        sim.mz(0);
        sim.mz(1);
        sim.mz(2);

        let sampler = MeasurementSampler::new(sim.measurement_history());
        let result = sampler.sample(1000);

        for shot in 0..1000 {
            assert_eq!(
                result.get(shot, 0),
                result.get(shot, 1),
                "GHZ measurements must be correlated"
            );
            assert_eq!(
                result.get(shot, 1),
                result.get(shot, 2),
                "GHZ measurements must be correlated"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Tests for empty history
    // -------------------------------------------------------------------------

    #[test]
    fn test_empty_history_shot() {
        let sim = StdSymbolicSparseStab::new(2);
        let sampler = SequentialMeasurementSampler::new(sim.measurement_history());

        assert_eq!(sampler.num_measurements(), 0);

        let result = sampler.sample(10);
        assert_eq!(result.shots(), 10);
        assert_eq!(result.num_measurements(), 0);
    }

    #[test]
    fn test_empty_history_columnar() {
        let sim = StdSymbolicSparseStab::new(2);
        let sampler = MeasurementSampler::new(sim.measurement_history());

        assert_eq!(sampler.num_measurements(), 0);

        let result = sampler.sample(10);
        assert_eq!(result.shots(), 10);
        assert_eq!(result.num_measurements(), 0);
    }

    // -------------------------------------------------------------------------
    // Tests for repetition code syndromes
    // -------------------------------------------------------------------------

    #[test]
    fn test_repetition_code_syndromes_shot() {
        let mut sim = StdSymbolicSparseStab::new(5);

        sim.h(0).cx(0, 1).cx(0, 2);
        sim.h(3).cx(0, 3).cx(1, 3).h(3);
        sim.mz(3);
        sim.h(4).cx(1, 4).cx(2, 4).h(4);
        sim.mz(4);
        sim.mz(0);
        sim.mz(1);
        sim.mz(2);

        let sampler = SequentialMeasurementSampler::new(sim.measurement_history());
        let result = sampler.sample(1000);

        for shot in 0..1000 {
            assert!(result.get(shot, 0).is_zero(), "Syndrome S0 should be 0");
            assert!(result.get(shot, 1).is_zero(), "Syndrome S1 should be 0");
            assert_eq!(
                result.get(shot, 2),
                result.get(shot, 3),
                "Data qubits should be correlated"
            );
            assert_eq!(
                result.get(shot, 3),
                result.get(shot, 4),
                "Data qubits should be correlated"
            );
        }
    }

    #[test]
    fn test_repetition_code_syndromes_columnar() {
        let mut sim = StdSymbolicSparseStab::new(5);

        sim.h(0).cx(0, 1).cx(0, 2);
        sim.h(3).cx(0, 3).cx(1, 3).h(3);
        sim.mz(3);
        sim.h(4).cx(1, 4).cx(2, 4).h(4);
        sim.mz(4);
        sim.mz(0);
        sim.mz(1);
        sim.mz(2);

        let sampler = MeasurementSampler::new(sim.measurement_history());
        let result = sampler.sample(1000);

        for shot in 0..1000 {
            assert!(result.get(shot, 0).is_zero(), "Syndrome S0 should be 0");
            assert!(result.get(shot, 1).is_zero(), "Syndrome S1 should be 0");
            assert_eq!(
                result.get(shot, 2),
                result.get(shot, 3),
                "Data qubits should be correlated"
            );
            assert_eq!(
                result.get(shot, 3),
                result.get(shot, 4),
                "Data qubits should be correlated"
            );
        }
    }

    // Test that both samplers produce statistically equivalent results
    #[test]
    fn test_samplers_equivalent() {
        let mut sim = StdSymbolicSparseStab::new(3);
        sim.h(0).cx(0, 1).cx(1, 2);
        sim.mz(0);
        sim.mz(1);
        sim.mz(2);

        let sequential_sampler = SequentialMeasurementSampler::new(sim.measurement_history());
        let sampler = MeasurementSampler::new(sim.measurement_history());

        let shot_result = sequential_sampler.sample(10000);
        let columnar_result = sampler.sample(10000);

        // Both should maintain GHZ correlations
        for shot in 0..10000 {
            assert_eq!(shot_result.get(shot, 0), shot_result.get(shot, 1));
            assert_eq!(shot_result.get(shot, 1), shot_result.get(shot, 2));
        }
        for shot in 0..10000 {
            assert_eq!(columnar_result.get(shot, 0), columnar_result.get(shot, 1));
            assert_eq!(columnar_result.get(shot, 1), columnar_result.get(shot, 2));
        }

        // Both should have roughly 50/50 distribution
        let shot_ones = shot_result.count_ones(0);
        let columnar_ones = columnar_result.count_ones(0);

        assert!(shot_ones > 4500 && shot_ones < 5500);
        assert!(columnar_ones > 4500 && columnar_ones < 5500);
    }

    // Test large shot counts (where columnar should excel)
    #[test]
    fn test_large_shot_count() {
        let mut sim = StdSymbolicSparseStab::new(10);
        for i in 0..10 {
            sim.h(i);
        }
        for i in 0..10 {
            sim.mz(i);
        }

        let sampler = MeasurementSampler::new(sim.measurement_history());
        let result = sampler.sample(100_000);

        assert_eq!(result.shots(), 100_000);
        assert_eq!(result.num_measurements(), 10);

        // Check that each measurement is roughly 50/50
        for m in 0..10 {
            let ones = result.count_ones(m);
            assert!(
                ones > 48_000 && ones < 52_000,
                "Measurement {m} should be ~50/50, got {ones} ones"
            );
        }
    }

    // Test the raw sampling API
    #[test]
    fn test_raw_sampling() {
        let mut sim = StdSymbolicSparseStab::new(2);
        sim.h(0).cx(0, 1);
        sim.mz(0);
        sim.mz(1);

        let sampler = MeasurementSampler::new(sim.measurement_history());
        let shots = 1000;
        let raw_columns = sampler.sample_raw(shots, &mut rand::rng());

        assert_eq!(raw_columns.len(), 2); // 2 measurements

        // Check correlations in raw format
        // For a Bell state, column 0 XOR column 1 should be all zeros
        for (col0_word, col1_word) in raw_columns[0].iter().zip(&raw_columns[1]) {
            assert_eq!(
                col0_word ^ col1_word,
                0,
                "Bell state columns should be identical"
            );
        }
    }

    // Test raw sampling with very large shot count
    #[test]
    fn test_raw_sampling_large() {
        let mut sim = StdSymbolicSparseStab::new(3);
        sim.h(0).cx(0, 1).cx(1, 2);
        sim.mz(0);
        sim.mz(1);
        sim.mz(2);

        let sampler = MeasurementSampler::new(sim.measurement_history());
        let shots = 1_000_000;
        let raw_columns = sampler.sample_raw(shots, &mut rand::rng());

        assert_eq!(raw_columns.len(), 3);

        // Verify GHZ correlations: all three columns should be identical
        for ((col0_word, col1_word), col2_word) in raw_columns[0]
            .iter()
            .zip(&raw_columns[1])
            .zip(&raw_columns[2])
        {
            assert_eq!(
                col0_word, col1_word,
                "GHZ columns 0 and 1 should be identical"
            );
            assert_eq!(
                col1_word, col2_word,
                "GHZ columns 1 and 2 should be identical"
            );
        }

        // Count ones to verify ~50% distribution
        let total_ones: u64 = raw_columns[0]
            .iter()
            .map(|w| u64::from(w.count_ones()))
            .sum();
        let expected = shots / 2;
        let tolerance = shots / 100; // 1% tolerance
        assert!(
            total_ones.abs_diff(expected as u64) < tolerance as u64,
            "Expected ~{expected} ones, got {total_ones}"
        );
    }

    // Test random measurement history generation
    #[test]
    fn test_random_history_generation() {
        let mut rng = rand::rng();

        // Generate a random history with:
        // - 100 measurements
        // - 30% random measurements
        // - 20% fixed (of the deterministic ones)
        // - max 3 dependencies
        let measurements = MeasurementKind::generate_random(100, 0.3, 0.2, 3, &mut rng);

        assert_eq!(measurements.len(), 100);

        // Verify dependencies are always to earlier measurements
        for (i, m) in measurements.iter().enumerate() {
            if let MeasurementKind::Computed { deps, .. } = m {
                for &dep in deps {
                    assert!(dep < i, "Dependency {dep} should be < current index {i}");
                }
                assert!(deps.len() <= 3, "Should have at most 3 dependencies");
            }
        }

        // Create samplers and verify they work
        let sequential_sampler =
            SequentialMeasurementSampler::from_measurements(measurements.clone());
        let sampler = MeasurementSampler::from_measurements(measurements);

        let shots = 1000;
        let shot_result = sequential_sampler.sample(shots);
        let columnar_result = sampler.sample(shots);

        assert_eq!(shot_result.shots(), shots);
        assert_eq!(columnar_result.shots(), shots);
        assert_eq!(shot_result.num_measurements(), 100);
        assert_eq!(columnar_result.num_measurements(), 100);
    }

    // Test that random history with mostly dependencies produces valid samples
    #[test]
    fn test_random_history_with_many_deps() {
        let mut rng = rand::rng();

        // Mostly computed measurements with up to 4 dependencies (realistic)
        let measurements = MeasurementKind::generate_random(50, 0.1, 0.1, 4, &mut rng);

        let sampler = MeasurementSampler::from_measurements(measurements);
        let raw = sampler.sample_raw(100_000, &mut rand::rng());

        // Just verify it doesn't crash and produces reasonable output
        assert_eq!(raw.len(), 50);
        for col in &raw {
            assert!(!col.is_empty());
        }
    }

    // Test handling of more than 64 measurements
    #[test]
    fn test_many_measurements() {
        let mut rng = rand::rng();

        // 200 measurements - well beyond 64
        let num_measurements = 200;
        let measurements =
            MeasurementKind::generate_random(num_measurements, 0.1, 0.1, 3, &mut rng);

        let sequential_sampler =
            SequentialMeasurementSampler::from_measurements(measurements.clone());
        let sampler = MeasurementSampler::from_measurements(measurements);

        let shots = 1000;

        // Test shot sampler
        let shot_result = sequential_sampler.sample(shots);
        assert_eq!(shot_result.shots(), shots);
        assert_eq!(shot_result.num_measurements(), num_measurements);

        // Test columnar sampler
        let columnar_result = sampler.sample(shots);
        assert_eq!(columnar_result.shots(), shots);
        assert_eq!(columnar_result.num_measurements(), num_measurements);

        // Test raw columnar output
        let raw = sampler.sample_raw(shots, &mut rand::rng());
        assert_eq!(raw.len(), num_measurements); // 200 columns
        let expected_words = shots.div_ceil(64);
        for col in &raw {
            assert_eq!(col.len(), expected_words);
        }
    }

    // Test handling of more than 64 shots with raw output
    #[test]
    fn test_many_shots_raw() {
        let mut sim = StdSymbolicSparseStab::new(5);
        sim.h(0);
        for i in 0..4 {
            sim.cx(i, i + 1);
        }
        for i in 0..5 {
            sim.mz(i);
        }

        let sampler = MeasurementSampler::new(sim.measurement_history());

        // Test various shot counts around the 64-bit boundary
        for shots in [63, 64, 65, 127, 128, 129, 1000, 10_000] {
            let raw = sampler.sample_raw(shots, &mut rand::rng());

            assert_eq!(raw.len(), 5, "Should have 5 measurement columns");

            let expected_words = shots.div_ceil(64);
            for col in &raw {
                assert_eq!(
                    col.len(),
                    expected_words,
                    "Wrong word count for {shots} shots"
                );
            }

            // Verify GHZ correlation: all columns should be identical
            for word_idx in 0..expected_words {
                let first = raw[0][word_idx];
                for col in &raw[1..] {
                    assert_eq!(
                        col[word_idx], first,
                        "GHZ correlation broken at word {word_idx}"
                    );
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // Tests for SampleResult
    // -------------------------------------------------------------------------

    #[test]
    fn test_sample_result_basic() {
        let mut sim = StdSymbolicSparseStab::new(2);
        sim.h(0).cx(0, 1);
        sim.mz(0);
        sim.mz(1);

        let sampler = MeasurementSampler::new(sim.measurement_history());
        let result = sampler.sample(1000);

        assert_eq!(result.shots(), 1000);
        assert_eq!(result.num_measurements(), 2);

        // Bell state: measurements must be correlated
        for shot in 0..1000 {
            assert_eq!(
                result.get(shot, 0),
                result.get(shot, 1),
                "Bell state measurements must be correlated at shot {shot}"
            );
        }
    }

    #[test]
    fn test_sample_result_count_ones() {
        let mut sim = StdSymbolicSparseStab::new(1);
        sim.h(0);
        sim.mz(0);

        let sampler = MeasurementSampler::new(sim.measurement_history());
        let shots = 10_000;
        let result = sampler.sample(shots);

        let ones = result.count_ones(0);
        let zeros = result.count_zeros(0);

        assert_eq!(ones + zeros, shots);
        // Should be roughly 50/50
        assert!(ones > 4500 && ones < 5500, "Expected ~50% ones, got {ones}");
    }

    #[test]
    fn test_sample_result_iter_matches_get() {
        let mut sim = StdSymbolicSparseStab::new(3);
        sim.h(0).cx(0, 1).cx(1, 2);
        sim.mz(0);
        sim.mz(1);
        sim.mz(2);

        let sampler = MeasurementSampler::new(sim.measurement_history());
        let result = sampler.sample(100);

        // Verify iter_shots matches direct access
        for (shot, row) in result.iter_shots().enumerate() {
            for m in 0..3 {
                assert_eq!(result.get(shot, m), row[m]);
            }
        }
    }

    #[test]
    fn test_sample_result_iter_shots() {
        let mut sim = StdSymbolicSparseStab::new(2);
        sim.x(0); // Deterministic 1
        sim.mz(0);
        sim.mz(1); // Deterministic 0

        let sampler = MeasurementSampler::new(sim.measurement_history());
        let result = sampler.sample(100);

        for (shot_idx, row) in result.iter_shots().enumerate() {
            assert!(row[0].is_one(), "m0 should be 1 at shot {shot_idx}");
            assert!(row[1].is_zero(), "m1 should be 0 at shot {shot_idx}");
        }
    }

    #[test]
    fn test_sample_result_try_get() {
        let mut sim = StdSymbolicSparseStab::new(1);
        sim.mz(0);

        let sampler = MeasurementSampler::new(sim.measurement_history());
        let result = sampler.sample(10);

        // Valid access
        assert!(result.try_get(0, 0).is_some());
        assert!(result.try_get(9, 0).is_some());

        // Out of bounds
        assert!(result.try_get(10, 0).is_none()); // shot out of bounds
        assert!(result.try_get(0, 1).is_none()); // measurement out of bounds
    }

    #[test]
    fn test_sample_result_shot_and_format() {
        let mut sim = StdSymbolicSparseStab::new(3);
        sim.x(0); // m0 = 1
        sim.mz(0);
        sim.mz(1); // m1 = 0
        sim.x(2);
        sim.mz(2); // m2 = 1

        let sampler = MeasurementSampler::new(sim.measurement_history());
        let result = sampler.sample(10);

        // All shots should be the same (deterministic)
        for shot_idx in 0..10 {
            // Test shot() method
            let bits = result.shot(shot_idx);
            assert_eq!(bits.len(), 3);
            assert!(bits[0].is_one(), "m0 should be 1");
            assert!(bits[1].is_zero(), "m1 should be 0");
            assert!(bits[2].is_one(), "m2 should be 1");

            // Test format_shot() method
            assert_eq!(result.format_shot(shot_idx), "101");
        }
    }

    #[test]
    fn test_sample_result_column_access() {
        let mut sim = StdSymbolicSparseStab::new(2);
        sim.h(0).cx(0, 1);
        sim.mz(0);
        sim.mz(1);

        let sampler = MeasurementSampler::new(sim.measurement_history());
        let result = sampler.sample(1000);

        let col0 = result.column(0);
        let col1 = result.column(1);

        // For Bell state, columns should be identical
        assert_eq!(col0, col1);

        // Verify columns() returns all columns
        let all_cols = result.columns();
        assert_eq!(all_cols.len(), 2);
    }

    #[test]
    fn test_sample_result_index_syntax() {
        let mut sim = StdSymbolicSparseStab::new(2);
        sim.h(0).cx(0, 1);
        sim.mz(0);
        sim.mz(1);

        let sampler = MeasurementSampler::new(sim.measurement_history());
        let result = sampler.sample(100);

        // Test index syntax result[(shot, measurement)]
        for shot in 0..100 {
            // Bell state: m0 == m1 for each shot
            assert_eq!(result[(shot, 0)], result[(shot, 1)]);

            // Should match get() method
            assert_eq!(result[(shot, 0)], result.get(shot, 0));
            assert_eq!(result[(shot, 1)], result.get(shot, 1));
        }
    }

    #[test]
    fn test_copy_flipped_optimization() {
        // Create a measurement that is the negation of another:
        // m0 = random, m1 = !m0
        let measurements = vec![MeasurementKind::Random, MeasurementKind::CopyFlipped(0)];

        let sequential_sampler =
            SequentialMeasurementSampler::from_measurements(measurements.clone());
        let sampler = MeasurementSampler::from_measurements(measurements);

        // Test shot sampler
        let shot_result = sequential_sampler.sample(1000);
        for shot in 0..1000 {
            assert_ne!(
                shot_result.get(shot, 0),
                shot_result.get(shot, 1),
                "m1 should be negation of m0"
            );
        }

        // Test columnar sampler
        let result = sampler.sample(1000);
        for shot in 0..1000 {
            assert_ne!(
                result.get(shot, 0),
                result.get(shot, 1),
                "m1 should be negation of m0 at shot {shot}"
            );
        }

        // Verify raw columns are bitwise NOT of each other
        let raw = sampler.sample_raw(1000, &mut rand::rng());
        for (w0, w1) in raw[0].iter().zip(raw[1].iter()) {
            assert_eq!(*w1, !*w0, "Column 1 should be bitwise NOT of column 0");
        }
    }

    // -------------------------------------------------------------------------
    // Tests verifying samples satisfy measurement equations
    // -------------------------------------------------------------------------

    /// Helper function to verify that all samples satisfy the measurement equations.
    ///
    /// For each shot, verifies:
    /// - Fixed(v): result == v
    /// - Random: no constraint (any value is valid)
    /// - Copy(src): result == samples[src]
    /// - CopyFlipped(src): result == !samples[src]
    /// - Computed { deps, flip }: result == flip ^ XOR(samples[d] for d in deps)
    fn verify_samples_satisfy_equations(measurements: &[MeasurementKind], result: &SampleResult) {
        for shot in 0..result.shots() {
            for (m_idx, kind) in measurements.iter().enumerate() {
                let actual = result.get(shot, m_idx);
                match kind {
                    MeasurementKind::Fixed(expected) => {
                        assert_eq!(
                            actual, *expected,
                            "Shot {shot}, measurement {m_idx}: Fixed({expected}) but got {actual}"
                        );
                    }
                    MeasurementKind::Random => {
                        // Any value is valid for random measurements
                    }
                    MeasurementKind::Copy(src) => {
                        let src_val = result.get(shot, *src);
                        assert_eq!(
                            actual, src_val,
                            "Shot {shot}, measurement {m_idx}: Copy({src}) expected {src_val} but got {actual}"
                        );
                    }
                    MeasurementKind::CopyFlipped(src) => {
                        let src_val = result.get(shot, *src);
                        let expected = !src_val;
                        assert_eq!(
                            actual, expected,
                            "Shot {shot}, measurement {m_idx}: CopyFlipped({src}) expected {expected} but got {actual}"
                        );
                    }
                    MeasurementKind::Computed { deps, flip } => {
                        let mut expected = *flip;
                        for &dep in deps {
                            expected ^= result.get(shot, dep);
                        }
                        assert_eq!(
                            actual, expected,
                            "Shot {shot}, measurement {m_idx}: Computed(deps={deps:?}, flip={flip}) expected {expected} but got {actual}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_equations_fixed_values() {
        let measurements = vec![
            MeasurementKind::Fixed(false),
            MeasurementKind::Fixed(true),
            MeasurementKind::Fixed(false),
            MeasurementKind::Fixed(true),
        ];

        let sequential_sampler =
            SequentialMeasurementSampler::from_measurements(measurements.clone());
        let sampler = MeasurementSampler::from_measurements(measurements.clone());

        let shot_result = sequential_sampler.sample(1000);
        let columnar_result = sampler.sample(1000);

        verify_samples_satisfy_equations(&measurements, &shot_result);
        verify_samples_satisfy_equations(&measurements, &columnar_result);
    }

    #[test]
    fn test_equations_copy_chain() {
        // m0 = random, m1 = m0, m2 = m1, m3 = m2
        let measurements = vec![
            MeasurementKind::Random,
            MeasurementKind::Copy(0),
            MeasurementKind::Copy(1),
            MeasurementKind::Copy(2),
        ];

        let sequential_sampler =
            SequentialMeasurementSampler::from_measurements(measurements.clone());
        let sampler = MeasurementSampler::from_measurements(measurements.clone());

        let shot_result = sequential_sampler.sample(1000);
        let columnar_result = sampler.sample(1000);

        verify_samples_satisfy_equations(&measurements, &shot_result);
        verify_samples_satisfy_equations(&measurements, &columnar_result);
    }

    #[test]
    fn test_equations_copy_flipped_chain() {
        // m0 = random, m1 = !m0, m2 = !m1 (= m0), m3 = !m2 (= !m0)
        let measurements = vec![
            MeasurementKind::Random,
            MeasurementKind::CopyFlipped(0),
            MeasurementKind::CopyFlipped(1),
            MeasurementKind::CopyFlipped(2),
        ];

        let sequential_sampler =
            SequentialMeasurementSampler::from_measurements(measurements.clone());
        let sampler = MeasurementSampler::from_measurements(measurements.clone());

        let shot_result = sequential_sampler.sample(1000);
        let columnar_result = sampler.sample(1000);

        verify_samples_satisfy_equations(&measurements, &shot_result);
        verify_samples_satisfy_equations(&measurements, &columnar_result);

        // Additionally verify the expected pattern: m0, !m0, m0, !m0
        for shot in 0..1000 {
            let m0 = shot_result.get(shot, 0);
            assert_eq!(shot_result.get(shot, 1), !m0);
            assert_eq!(shot_result.get(shot, 2), m0);
            assert_eq!(shot_result.get(shot, 3), !m0);
        }
    }

    #[test]
    fn test_equations_xor_dependencies() {
        // m0, m1, m2 = random
        // m3 = m0 ^ m1
        // m4 = m0 ^ m1 ^ m2
        // m5 = m0 ^ m1 ^ m2 ^ true (flip)
        let measurements = vec![
            MeasurementKind::Random,
            MeasurementKind::Random,
            MeasurementKind::Random,
            MeasurementKind::Computed {
                deps: vec![0, 1],
                flip: false,
            },
            MeasurementKind::Computed {
                deps: vec![0, 1, 2],
                flip: false,
            },
            MeasurementKind::Computed {
                deps: vec![0, 1, 2],
                flip: true,
            },
        ];

        let sequential_sampler =
            SequentialMeasurementSampler::from_measurements(measurements.clone());
        let sampler = MeasurementSampler::from_measurements(measurements.clone());

        let shot_result = sequential_sampler.sample(1000);
        let columnar_result = sampler.sample(1000);

        verify_samples_satisfy_equations(&measurements, &shot_result);
        verify_samples_satisfy_equations(&measurements, &columnar_result);

        // Additionally verify specific relationships
        for shot in 0..1000 {
            let m0 = shot_result.get(shot, 0);
            let m1 = shot_result.get(shot, 1);
            let m2 = shot_result.get(shot, 2);

            assert_eq!(shot_result.get(shot, 3), m0 ^ m1);
            assert_eq!(shot_result.get(shot, 4), m0 ^ m1 ^ m2);
            assert_eq!(shot_result.get(shot, 5), !(m0 ^ m1 ^ m2));
        }
    }

    #[test]
    fn test_equations_mixed_types() {
        // A realistic mix of all measurement types
        let measurements = vec![
            MeasurementKind::Fixed(false),   // m0 = 0
            MeasurementKind::Fixed(true),    // m1 = 1
            MeasurementKind::Random,         // m2 = ?
            MeasurementKind::Random,         // m3 = ?
            MeasurementKind::Copy(2),        // m4 = m2
            MeasurementKind::CopyFlipped(3), // m5 = !m3
            MeasurementKind::Computed {
                // m6 = m2 ^ m3
                deps: vec![2, 3],
                flip: false,
            },
            MeasurementKind::Computed {
                // m7 = m0 ^ m1 ^ m2 ^ true = 1 ^ m2
                deps: vec![0, 1, 2],
                flip: true,
            },
        ];

        let sequential_sampler =
            SequentialMeasurementSampler::from_measurements(measurements.clone());
        let sampler = MeasurementSampler::from_measurements(measurements.clone());

        let shot_result = sequential_sampler.sample(1000);
        let columnar_result = sampler.sample(1000);

        verify_samples_satisfy_equations(&measurements, &shot_result);
        verify_samples_satisfy_equations(&measurements, &columnar_result);
    }

    #[test]
    fn test_equations_random_generated_history() {
        // Test with randomly generated measurement histories
        let mut rng = rand::rng();

        for _ in 0..10 {
            // Generate random histories with various parameters
            let measurements = MeasurementKind::generate_random(50, 0.2, 0.1, 4, &mut rng);

            let sequential_sampler =
                SequentialMeasurementSampler::from_measurements(measurements.clone());
            let sampler = MeasurementSampler::from_measurements(measurements.clone());

            let shot_result = sequential_sampler.sample(100);
            let columnar_result = sampler.sample(100);

            verify_samples_satisfy_equations(&measurements, &shot_result);
            verify_samples_satisfy_equations(&measurements, &columnar_result);
        }
    }

    #[test]
    fn test_equations_large_dependency_chain() {
        // Test a long chain of dependencies to catch any ordering bugs
        // m0 = random
        // m1 = m0, m2 = m0 ^ m1, m3 = m0 ^ m1 ^ m2, etc.
        let mut measurements = vec![MeasurementKind::Random];
        for i in 1..20 {
            measurements.push(MeasurementKind::Computed {
                deps: (0..i).collect(),
                flip: i % 2 == 0,
            });
        }

        let sequential_sampler =
            SequentialMeasurementSampler::from_measurements(measurements.clone());
        let sampler = MeasurementSampler::from_measurements(measurements.clone());

        let shot_result = sequential_sampler.sample(100);
        let columnar_result = sampler.sample(100);

        verify_samples_satisfy_equations(&measurements, &shot_result);
        verify_samples_satisfy_equations(&measurements, &columnar_result);
    }

    #[test]
    fn test_equations_samplers_produce_same_structure() {
        // Verify both samplers produce results satisfying the same equations
        // (they won't have the same random values, but structure must match)
        let measurements = vec![
            MeasurementKind::Fixed(true),
            MeasurementKind::Random,
            MeasurementKind::Copy(1),
            MeasurementKind::CopyFlipped(1),
            MeasurementKind::Computed {
                deps: vec![1, 2],
                flip: false,
            },
        ];

        // Use seeded RNG for reproducibility within each sampler
        let mut rng = rand::rng();

        let sequential_sampler =
            SequentialMeasurementSampler::from_measurements(measurements.clone());
        let sampler = MeasurementSampler::from_measurements(measurements.clone());

        // Each sampler independently satisfies equations
        let shot_result = sequential_sampler.sample_with_rng(1000, &mut rng);
        verify_samples_satisfy_equations(&measurements, &shot_result);

        let columnar_result = sampler.sample_with_rng(1000, &mut rng);
        verify_samples_satisfy_equations(&measurements, &columnar_result);
    }

    // -------------------------------------------------------------------------
    // Tests for MeasurementKind validation
    // -------------------------------------------------------------------------

    #[test]
    fn test_validate_valid_sequence() {
        let measurements = vec![
            MeasurementKind::Fixed(true),
            MeasurementKind::Random,
            MeasurementKind::Copy(0),
            MeasurementKind::CopyFlipped(1),
            MeasurementKind::Computed {
                deps: vec![0, 1, 2],
                flip: false,
            },
        ];
        assert!(MeasurementKind::validate_sequence(&measurements).is_ok());
    }

    #[test]
    fn test_validate_empty_sequence() {
        assert!(MeasurementKind::validate_sequence(&[]).is_ok());
    }

    #[test]
    fn test_validate_forward_reference_copy() {
        let measurements = vec![
            MeasurementKind::Random,
            MeasurementKind::Copy(2), // Forward reference!
            MeasurementKind::Random,
        ];
        assert_eq!(
            MeasurementKind::validate_sequence(&measurements),
            Err(MeasurementValidationError::ForwardReference {
                measurement_idx: 1,
                dependency_idx: 2,
            })
        );
    }

    #[test]
    fn test_validate_forward_reference_copy_flipped() {
        let measurements = vec![
            MeasurementKind::CopyFlipped(0), // Self-reference is also forward!
        ];
        assert_eq!(
            MeasurementKind::validate_sequence(&measurements),
            Err(MeasurementValidationError::ForwardReference {
                measurement_idx: 0,
                dependency_idx: 0,
            })
        );
    }

    #[test]
    fn test_validate_forward_reference_computed() {
        let measurements = vec![
            MeasurementKind::Random,
            MeasurementKind::Random,
            MeasurementKind::Computed {
                deps: vec![0, 5], // 5 is out of bounds
                flip: false,
            },
        ];
        assert_eq!(
            MeasurementKind::validate_sequence(&measurements),
            Err(MeasurementValidationError::ForwardReference {
                measurement_idx: 2,
                dependency_idx: 5,
            })
        );
    }

    #[test]
    fn test_validate_duplicate_dependency() {
        let measurements = vec![
            MeasurementKind::Random,
            MeasurementKind::Random,
            MeasurementKind::Computed {
                deps: vec![0, 1, 0], // Duplicate 0!
                flip: false,
            },
        ];
        assert_eq!(
            MeasurementKind::validate_sequence(&measurements),
            Err(MeasurementValidationError::DuplicateDependency {
                measurement_idx: 2,
                dependency_idx: 0,
            })
        );
    }

    #[test]
    fn test_validate_empty_dependencies() {
        let measurements = vec![
            MeasurementKind::Random,
            MeasurementKind::Computed {
                deps: vec![], // Empty deps!
                flip: true,
            },
        ];
        assert_eq!(
            MeasurementKind::validate_sequence(&measurements),
            Err(MeasurementValidationError::EmptyDependencies { measurement_idx: 1 })
        );
    }

    #[test]
    fn test_validate_generated_histories_are_valid() {
        // Verify that generate_random always produces valid sequences
        let mut rng = rand::rng();
        for _ in 0..100 {
            let measurements = MeasurementKind::generate_random(50, 0.3, 0.2, 5, &mut rng);
            assert!(
                MeasurementKind::validate_sequence(&measurements).is_ok(),
                "Generated history should always be valid"
            );
        }
    }

    #[test]
    fn test_validation_error_display() {
        let err = MeasurementValidationError::ForwardReference {
            measurement_idx: 3,
            dependency_idx: 5,
        };
        assert_eq!(err.to_string(), "Measurement 3 has forward reference to 5");

        let err = MeasurementValidationError::DuplicateDependency {
            measurement_idx: 2,
            dependency_idx: 0,
        };
        assert_eq!(err.to_string(), "Measurement 2 has duplicate dependency 0");

        let err = MeasurementValidationError::EmptyDependencies { measurement_idx: 1 };
        assert_eq!(
            err.to_string(),
            "Measurement 1 is Computed but has no dependencies"
        );
    }

    // -------------------------------------------------------------------------
    // More elaborate correlation tests for generated histories
    // -------------------------------------------------------------------------

    /// Verifies parity constraints: for any subset of measurements that XOR to a
    /// constant (e.g., syndrome measurements), the samples should respect that.
    #[test]
    fn test_elaborate_parity_constraints() {
        // Create a history where certain combinations are constrained:
        // m0, m1, m2 = random
        // m3 = m0 ^ m1 (parity of m0, m1)
        // m4 = m1 ^ m2 (parity of m1, m2)
        // m5 = m0 ^ m2 (parity of m0, m2)
        // Then: m3 ^ m4 ^ m5 = (m0^m1) ^ (m1^m2) ^ (m0^m2) = 0 (always!)
        let measurements = vec![
            MeasurementKind::Random,
            MeasurementKind::Random,
            MeasurementKind::Random,
            MeasurementKind::Computed {
                deps: vec![0, 1],
                flip: false,
            },
            MeasurementKind::Computed {
                deps: vec![1, 2],
                flip: false,
            },
            MeasurementKind::Computed {
                deps: vec![0, 2],
                flip: false,
            },
        ];

        let sampler = MeasurementSampler::from_measurements(measurements.clone());
        let result = sampler.sample(10000);

        verify_samples_satisfy_equations(&measurements, &result);

        // Verify the derived parity constraint: m3 ^ m4 ^ m5 = false
        for shot in 0..10000 {
            let m3 = result.get(shot, 3);
            let m4 = result.get(shot, 4);
            let m5 = result.get(shot, 5);
            assert!(
                (m3 ^ m4 ^ m5).is_zero(),
                "Shot {shot}: m3^m4^m5 should always be false, got m3={m3}, m4={m4}, m5={m5}"
            );
        }

        // Also verify using raw column XOR
        let raw = sampler.sample_raw(10000, &mut rand::rng());
        for (word_idx, ((&w3, &w4), &w5)) in raw[3].iter().zip(&raw[4]).zip(&raw[5]).enumerate() {
            let xor = w3 ^ w4 ^ w5;
            assert_eq!(xor, 0, "Column XOR should be 0 at word {word_idx}");
        }
    }

    #[test]
    fn test_elaborate_chain_parity() {
        // Create a chain where each measurement depends on the previous
        // m0 = random
        // m1 = m0 ^ flip1, m2 = m1 ^ flip2, ...
        // Then m_n depends on m0 and the parity of all flips
        let n = 10;
        let mut measurements = vec![MeasurementKind::Random];
        let mut expected_total_flip = false;
        for i in 1..n {
            let flip = i % 3 == 0; // flip every 3rd
            if flip {
                expected_total_flip = !expected_total_flip;
            }
            measurements.push(MeasurementKind::Computed {
                deps: vec![i - 1],
                flip,
            });
        }

        let sampler = MeasurementSampler::from_measurements(measurements.clone());
        let result = sampler.sample(1000);

        verify_samples_satisfy_equations(&measurements, &result);

        // Verify: m_last = m0 ^ expected_total_flip
        for shot in 0..1000 {
            let m0 = result.get(shot, 0);
            let m_last = result.get(shot, n - 1);
            assert_eq!(
                m_last,
                m0 ^ expected_total_flip,
                "Shot {shot}: m_last should equal m0 ^ {expected_total_flip}"
            );
        }
    }

    #[test]
    fn test_elaborate_syndrome_pattern() {
        // Simulate a simple syndrome measurement pattern:
        // d0, d1, d2, d3 = data qubits (random)
        // s0 = d0 ^ d1 (syndrome between d0, d1)
        // s1 = d1 ^ d2 (syndrome between d1, d2)
        // s2 = d2 ^ d3 (syndrome between d2, d3)
        // In error-free case, all syndromes should be independent random values
        // But d0 ^ d1 ^ d2 ^ d3 ^ s0 ^ s1 ^ s2 has interesting properties
        let measurements = vec![
            MeasurementKind::Random, // d0
            MeasurementKind::Random, // d1
            MeasurementKind::Random, // d2
            MeasurementKind::Random, // d3
            MeasurementKind::Computed {
                deps: vec![0, 1],
                flip: false,
            }, // s0
            MeasurementKind::Computed {
                deps: vec![1, 2],
                flip: false,
            }, // s1
            MeasurementKind::Computed {
                deps: vec![2, 3],
                flip: false,
            }, // s2
        ];

        let sampler = MeasurementSampler::from_measurements(measurements.clone());
        let result = sampler.sample(10000);

        verify_samples_satisfy_equations(&measurements, &result);

        // Verify: d0 ^ d3 = s0 ^ s1 ^ s2
        // Because: s0 ^ s1 ^ s2 = (d0^d1) ^ (d1^d2) ^ (d2^d3) = d0 ^ d3
        for shot in 0..10000 {
            let d0 = result.get(shot, 0);
            let d3 = result.get(shot, 3);
            let s0 = result.get(shot, 4);
            let s1 = result.get(shot, 5);
            let s2 = result.get(shot, 6);

            assert_eq!(
                d0 ^ d3,
                s0 ^ s1 ^ s2,
                "Shot {shot}: d0^d3 should equal s0^s1^s2"
            );
        }
    }

    #[test]
    fn test_elaborate_multi_level_dependencies() {
        // Test dependencies that span multiple levels:
        // Level 0: m0, m1, m2, m3 (random)
        // Level 1: m4 = m0^m1, m5 = m2^m3
        // Level 2: m6 = m4^m5 = m0^m1^m2^m3
        // Level 3: m7 = m6 ^ flip = !(m0^m1^m2^m3)
        let measurements = vec![
            MeasurementKind::Random, // m0
            MeasurementKind::Random, // m1
            MeasurementKind::Random, // m2
            MeasurementKind::Random, // m3
            MeasurementKind::Computed {
                deps: vec![0, 1],
                flip: false,
            }, // m4 = m0^m1
            MeasurementKind::Computed {
                deps: vec![2, 3],
                flip: false,
            }, // m5 = m2^m3
            MeasurementKind::Computed {
                deps: vec![4, 5],
                flip: false,
            }, // m6 = m4^m5
            MeasurementKind::Computed {
                deps: vec![6],
                flip: true,
            }, // m7 = !m6
        ];

        let sampler = MeasurementSampler::from_measurements(measurements.clone());
        let result = sampler.sample(1000);

        verify_samples_satisfy_equations(&measurements, &result);

        // Verify multi-level dependencies
        for shot in 0..1000 {
            let m0 = result.get(shot, 0);
            let m1 = result.get(shot, 1);
            let m2 = result.get(shot, 2);
            let m3 = result.get(shot, 3);
            let m6 = result.get(shot, 6);
            let m7 = result.get(shot, 7);

            assert_eq!(m6, m0 ^ m1 ^ m2 ^ m3, "Shot {shot}: m6 = m0^m1^m2^m3");
            assert_eq!(m7, !(m0 ^ m1 ^ m2 ^ m3), "Shot {shot}: m7 = !(m0^m1^m2^m3)");
        }
    }

    #[test]
    fn test_elaborate_statistical_independence() {
        // Verify that independent random measurements are statistically uncorrelated
        // m0 = random, m1 = random (independent)
        // XOR of independent fair coins should also be fair
        let measurements = vec![MeasurementKind::Random, MeasurementKind::Random];

        let sampler = MeasurementSampler::from_measurements(measurements);
        let result = sampler.sample(100_000);

        // Count joint occurrences
        let mut count_00 = 0;
        let mut count_01 = 0;
        let mut count_10 = 0;
        let mut count_11 = 0;

        for shot in 0..100_000 {
            match (result.get(shot, 0), result.get(shot, 1)) {
                (Bit::ZERO, Bit::ZERO) => count_00 += 1,
                (Bit::ZERO, Bit::ONE) => count_01 += 1,
                (Bit::ONE, Bit::ZERO) => count_10 += 1,
                (Bit::ONE, Bit::ONE) => count_11 += 1,
            }
        }

        // Each combination should be ~25% with some tolerance
        let expected = 25_000.0;
        let tolerance = 1000.0; // ~4% tolerance

        assert!(
            (f64::from(count_00) - expected).abs() < tolerance,
            "00 count {count_00} too far from {expected}"
        );
        assert!(
            (f64::from(count_01) - expected).abs() < tolerance,
            "01 count {count_01} too far from {expected}"
        );
        assert!(
            (f64::from(count_10) - expected).abs() < tolerance,
            "10 count {count_10} too far from {expected}"
        );
        assert!(
            (f64::from(count_11) - expected).abs() < tolerance,
            "11 count {count_11} too far from {expected}"
        );
    }

    #[test]
    fn test_elaborate_perfect_correlation() {
        // Verify that Copy produces perfect correlation
        let measurements = vec![MeasurementKind::Random, MeasurementKind::Copy(0)];

        let sampler = MeasurementSampler::from_measurements(measurements);
        let result = sampler.sample(10_000);

        // Count joint occurrences - should only see 00 and 11
        let mut count_same = 0;
        let mut count_different = 0;

        for shot in 0..10_000 {
            if result.get(shot, 0) == result.get(shot, 1) {
                count_same += 1;
            } else {
                count_different += 1;
            }
        }

        assert_eq!(count_same, 10_000, "All shots should have m0 == m1");
        assert_eq!(count_different, 0, "No shots should have m0 != m1");
    }

    #[test]
    fn test_elaborate_perfect_anticorrelation() {
        // Verify that CopyFlipped produces perfect anticorrelation
        let measurements = vec![MeasurementKind::Random, MeasurementKind::CopyFlipped(0)];

        let sampler = MeasurementSampler::from_measurements(measurements);
        let result = sampler.sample(10_000);

        // Count joint occurrences - should only see 01 and 10
        let mut count_same = 0;
        let mut count_different = 0;

        for shot in 0..10_000 {
            if result.get(shot, 0) == result.get(shot, 1) {
                count_same += 1;
            } else {
                count_different += 1;
            }
        }

        assert_eq!(count_same, 0, "No shots should have m0 == m1");
        assert_eq!(count_different, 10_000, "All shots should have m0 != m1");
    }
}
