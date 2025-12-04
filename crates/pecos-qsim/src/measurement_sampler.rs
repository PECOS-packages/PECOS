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
//! - [`ShotSampler`]: Processes one shot at a time (row-major computation)
//! - [`ColumnarSampler`]: Processes one measurement at a time across all shots (column-major)
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
//! use pecos_qsim::measurement_sampler::{ShotSampler, ColumnarSampler};
//!
//! // Create a Bell state and measure
//! let mut sim = StdSymbolicSparseStab::new(2);
//! sim.h(0).cx(0, 1);
//! sim.mz(0);
//! sim.mz(1);
//!
//! // Using shot-by-shot sampler
//! let sampler = ShotSampler::new(sim.measurement_history());
//! let result = sampler.sample_to_result_with_thread_rng(1000);
//!
//! // Using columnar sampler (faster for many shots)
//! let sampler = ColumnarSampler::new(sim.measurement_history());
//! let result = sampler.sample_to_result_with_thread_rng(1000);
//!
//! // Access individual bits
//! let m0_shot0 = result.get(0, 0);
//! ```

use crate::symbolic_sparse_stab::MeasurementHistory;
use rand::Rng;

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
                    let src = *result.outcome.iter().next().unwrap();
                    if result.flip {
                        MeasurementKind::CopyFlipped(src)
                    } else {
                        MeasurementKind::Copy(src)
                    }
                } else {
                    MeasurementKind::Computed {
                        deps: result.outcome.iter().copied().collect(),
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
}

// ============================================================================
// Shot-by-shot sampler (row-major computation, column-major output)
// ============================================================================

/// Shot-by-shot sampler that processes one complete shot at a time.
///
/// This sampler iterates through all measurements for each shot before moving
/// to the next shot. The output is stored in column-major format (`Vec<Vec<u64>>`)
/// for efficient bulk operations.
///
/// For large numbers of shots, [`ColumnarSampler`] is typically faster due to
/// better SIMD utilization and batched random number generation.
#[derive(Clone, Debug)]
pub struct ShotSampler {
    /// Preprocessed measurement classifications
    measurements: Vec<MeasurementKind>,
}

impl ShotSampler {
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
    #[must_use]
    pub fn sample_raw<R: Rng>(&self, shots: usize, rng: &mut R) -> Vec<Vec<u64>> {
        if self.measurements.is_empty() || shots == 0 {
            return vec![Vec::new(); self.measurements.len()];
        }

        let num_words = (shots + 63) / 64;
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

    /// Generate multiple shots using the default RNG.
    #[must_use]
    pub fn sample_raw_with_thread_rng(&self, shots: usize) -> Vec<Vec<u64>> {
        let mut rng = rand::rng();
        self.sample_raw(shots, &mut rng)
    }

    /// Sample and return a `SampleResult` for convenient access.
    #[must_use]
    pub fn sample_to_result<R: Rng>(&self, shots: usize, rng: &mut R) -> SampleResult {
        let columns = self.sample_raw(shots, rng);
        SampleResult::new(columns, shots)
    }

    /// Sample and return a `SampleResult` using the default RNG.
    #[must_use]
    pub fn sample_to_result_with_thread_rng(&self, shots: usize) -> SampleResult {
        let mut rng = rand::rng();
        self.sample_to_result(shots, &mut rng)
    }
}

// ============================================================================
// Columnar sampler (column-major, SIMD-friendly, optimized for large shot counts)
// ============================================================================

/// Columnar sampler that processes one measurement at a time across all shots.
///
/// This sampler processes all shots for measurement 0, then all shots for
/// measurement 1, etc. This enables:
/// - Batched random number generation (generate 64 random bits at once)
/// - SIMD-friendly XOR operations on entire columns (operating on u64 words)
/// - Better cache locality for large shot counts
///
/// Internally uses `Vec<u64>` for columns to maximize performance.
/// Generally faster than [`ShotSampler`] for large numbers of shots (>100).
#[derive(Clone, Debug)]
pub struct ColumnarSampler {
    /// Preprocessed measurement classifications
    measurements: Vec<MeasurementKind>,
}

impl ColumnarSampler {
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

    /// Generate a column of random bits as Vec<u64>.
    #[inline]
    fn generate_random_column<R: Rng>(num_words: usize, rng: &mut R) -> Vec<u64> {
        let mut column = Vec::with_capacity(num_words);
        for _ in 0..num_words {
            column.push(rng.random::<u64>());
        }
        column
    }

    /// Compute a column by XORing dependency columns using u64 operations.
    #[inline]
    fn compute_xor_column(
        columns: &[Vec<u64>],
        deps: &[usize],
        flip: bool,
        num_words: usize,
    ) -> Vec<u64> {
        // Start with all zeros or all ones depending on flip
        let mut result = if flip {
            vec![!0u64; num_words]
        } else {
            vec![0u64; num_words]
        };

        // XOR each dependency column - this is very SIMD-friendly
        for &dep_idx in deps {
            let dep_column = &columns[dep_idx];
            for (r, &d) in result.iter_mut().zip(dep_column.iter()) {
                *r ^= d;
            }
        }

        result
    }

    /// Sample directly to raw u64 columns.
    ///
    /// Returns a vector of columns where each column is a `Vec<u64>` representing
    /// all shots for one measurement. Bit `i` of word `w` corresponds to shot `w*64 + i`.
    #[must_use]
    pub fn sample_raw<R: Rng>(&self, shots: usize, rng: &mut R) -> Vec<Vec<u64>> {
        if self.measurements.is_empty() || shots == 0 {
            return vec![Vec::new(); self.measurements.len()];
        }

        let num_words = (shots + 63) / 64;
        let mut columns: Vec<Vec<u64>> = Vec::with_capacity(self.measurements.len());

        for kind in &self.measurements {
            let column = match kind {
                MeasurementKind::Fixed(value) => {
                    let fill = if *value { !0u64 } else { 0u64 };
                    vec![fill; num_words]
                }
                MeasurementKind::Random => Self::generate_random_column(num_words, rng),
                MeasurementKind::Copy(src) => {
                    // Just clone the source column - no computation needed
                    columns[*src].clone()
                }
                MeasurementKind::CopyFlipped(src) => {
                    // Clone and NOT the column
                    columns[*src].iter().map(|w| !w).collect()
                }
                MeasurementKind::Computed { deps, flip } => {
                    Self::compute_xor_column(&columns, deps, *flip, num_words)
                }
            };
            columns.push(column);
        }

        columns
    }

    /// Sample directly to raw u64 columns using the default RNG.
    #[must_use]
    pub fn sample_raw_with_thread_rng(&self, shots: usize) -> Vec<Vec<u64>> {
        let mut rng = rand::rng();
        self.sample_raw(shots, &mut rng)
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
/// use pecos_qsim::measurement_sampler::{ColumnarSampler, SampleResult};
/// use pecos_qsim::symbolic_sparse_stab::StdSymbolicSparseStab;
///
/// let mut sim = StdSymbolicSparseStab::new(2);
/// sim.h(0).cx(0, 1);
/// sim.mz(0);
/// sim.mz(1);
///
/// let sampler = ColumnarSampler::new(sim.measurement_history());
/// let result = sampler.sample_to_result_with_thread_rng(1000);
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
    /// This is equivalent to `result[(shot, measurement)]` using index syntax.
    ///
    /// # Arguments
    /// * `shot` - The shot/sample index (0 to `shots()-1`)
    /// * `measurement` - The measurement index (0 to `num_measurements()-1`)
    ///
    /// # Panics
    /// Panics if `shot >= self.shots()` or `measurement >= self.num_measurements()`.
    #[inline]
    #[must_use]
    pub fn get(&self, shot: usize, measurement: usize) -> bool {
        debug_assert!(shot < self.shots, "shot index out of bounds");
        debug_assert!(measurement < self.columns.len(), "measurement index out of bounds");

        let word_idx = shot / 64;
        let bit_idx = shot % 64;
        (self.columns[measurement][word_idx] >> bit_idx) & 1 == 1
    }

    /// Get the measurement result, returning `None` if out of bounds.
    ///
    /// # Arguments
    /// * `shot` - The shot/sample index
    /// * `measurement` - The measurement index
    #[inline]
    #[must_use]
    pub fn try_get(&self, shot: usize, measurement: usize) -> Option<bool> {
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

    /// Iterate over shots, yielding each shot's measurements as a `Vec<bool>`.
    ///
    /// Note: This allocates a new `Vec<bool>` for each shot. For bulk access,
    /// consider working with columns directly.
    pub fn iter_shots(&self) -> impl Iterator<Item = Vec<bool>> + '_ {
        (0..self.shots).map(|shot| {
            let word_idx = shot / 64;
            let bit_idx = shot % 64;
            let mask = 1u64 << bit_idx;

            self.columns
                .iter()
                .map(|col| (col[word_idx] & mask) != 0)
                .collect()
        })
    }
}

impl std::ops::Index<(usize, usize)> for SampleResult {
    type Output = bool;

    /// Index into sample results using `result[(shot, measurement)]` syntax.
    ///
    /// # Arguments
    /// * `shot` - The shot/sample index (0 to `shots()-1`)
    /// * `measurement` - The measurement index (0 to `num_measurements()-1`)
    ///
    /// # Panics
    /// Panics if indices are out of bounds.
    ///
    /// # Note
    /// Due to Rust's `Index` trait requirements, this returns a reference to a
    /// static bool. For the actual value, use `result.get(shot, measurement)`.
    #[inline]
    fn index(&self, (shot, measurement): (usize, usize)) -> &Self::Output {
        if self.get(shot, measurement) {
            &true
        } else {
            &false
        }
    }
}

impl ColumnarSampler {
    /// Sample and return a `SampleResult` for convenient access.
    #[must_use]
    pub fn sample_to_result<R: Rng>(&self, shots: usize, rng: &mut R) -> SampleResult {
        let columns = self.sample_raw(shots, rng);
        SampleResult::new(columns, shots)
    }

    /// Sample and return a `SampleResult` using the default RNG.
    #[must_use]
    pub fn sample_to_result_with_thread_rng(&self, shots: usize) -> SampleResult {
        let mut rng = rand::rng();
        self.sample_to_result(shots, &mut rng)
    }
}

// ============================================================================
// Type alias for backwards compatibility
// ============================================================================

/// Alias for [`ShotSampler`] for backwards compatibility.
pub type MeasurementSampler = ShotSampler;

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

        let sampler = ShotSampler::new(sim.measurement_history());
        let result = sampler.sample_to_result_with_thread_rng(100);

        for shot in 0..100 {
            assert!(!result.get(shot, 0), "Expected all measurements to be 0");
        }
    }

    #[test]
    fn test_deterministic_zero_columnar() {
        let mut sim = StdSymbolicSparseStab::new(1);
        sim.mz(0);

        let sampler = ColumnarSampler::new(sim.measurement_history());
        let result = sampler.sample_to_result_with_thread_rng(100);

        for shot in 0..100 {
            assert!(!result.get(shot, 0), "Expected all measurements to be 0");
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

        let sampler = ShotSampler::new(sim.measurement_history());
        let result = sampler.sample_to_result_with_thread_rng(100);

        for shot in 0..100 {
            assert!(result.get(shot, 0), "Expected all measurements to be 1");
        }
    }

    #[test]
    fn test_deterministic_one_columnar() {
        let mut sim = StdSymbolicSparseStab::new(1);
        sim.x(0);
        sim.mz(0);

        let sampler = ColumnarSampler::new(sim.measurement_history());
        let result = sampler.sample_to_result_with_thread_rng(100);

        for shot in 0..100 {
            assert!(result.get(shot, 0), "Expected all measurements to be 1");
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

        let sampler = ShotSampler::new(sim.measurement_history());
        let result = sampler.sample_to_result_with_thread_rng(1000);

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

        let sampler = ColumnarSampler::new(sim.measurement_history());
        let result = sampler.sample_to_result_with_thread_rng(1000);

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

        let sampler = ShotSampler::new(sim.measurement_history());
        let result = sampler.sample_to_result_with_thread_rng(1000);

        for shot in 0..1000 {
            assert_eq!(
                result.get(shot, 0),
                result.get(shot, 1),
                "Bell state measurements must be correlated"
            );
        }

        let ones = result.count_ones(0);
        assert!(ones > 400 && ones < 600, "Expected roughly 50/50 for first qubit");
    }

    #[test]
    fn test_bell_state_correlation_columnar() {
        let mut sim = StdSymbolicSparseStab::new(2);
        sim.h(0).cx(0, 1);
        sim.mz(0);
        sim.mz(1);

        let sampler = ColumnarSampler::new(sim.measurement_history());
        let result = sampler.sample_to_result_with_thread_rng(1000);

        for shot in 0..1000 {
            assert_eq!(
                result.get(shot, 0),
                result.get(shot, 1),
                "Bell state measurements must be correlated"
            );
        }

        let ones = result.count_ones(0);
        assert!(ones > 400 && ones < 600, "Expected roughly 50/50 for first qubit");
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

        let sampler = ShotSampler::new(sim.measurement_history());
        let result = sampler.sample_to_result_with_thread_rng(1000);

        for shot in 0..1000 {
            assert_eq!(result.get(shot, 0), result.get(shot, 1), "GHZ measurements must be correlated");
            assert_eq!(result.get(shot, 1), result.get(shot, 2), "GHZ measurements must be correlated");
        }
    }

    #[test]
    fn test_ghz_state_correlation_columnar() {
        let mut sim = StdSymbolicSparseStab::new(3);
        sim.h(0).cx(0, 1).cx(1, 2);
        sim.mz(0);
        sim.mz(1);
        sim.mz(2);

        let sampler = ColumnarSampler::new(sim.measurement_history());
        let result = sampler.sample_to_result_with_thread_rng(1000);

        for shot in 0..1000 {
            assert_eq!(result.get(shot, 0), result.get(shot, 1), "GHZ measurements must be correlated");
            assert_eq!(result.get(shot, 1), result.get(shot, 2), "GHZ measurements must be correlated");
        }
    }

    // -------------------------------------------------------------------------
    // Tests for empty history
    // -------------------------------------------------------------------------

    #[test]
    fn test_empty_history_shot() {
        let sim = StdSymbolicSparseStab::new(2);
        let sampler = ShotSampler::new(sim.measurement_history());

        assert_eq!(sampler.num_measurements(), 0);

        let result = sampler.sample_to_result_with_thread_rng(10);
        assert_eq!(result.shots(), 10);
        assert_eq!(result.num_measurements(), 0);
    }

    #[test]
    fn test_empty_history_columnar() {
        let sim = StdSymbolicSparseStab::new(2);
        let sampler = ColumnarSampler::new(sim.measurement_history());

        assert_eq!(sampler.num_measurements(), 0);

        let result = sampler.sample_to_result_with_thread_rng(10);
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

        let sampler = ShotSampler::new(sim.measurement_history());
        let result = sampler.sample_to_result_with_thread_rng(1000);

        for shot in 0..1000 {
            assert!(!result.get(shot, 0), "Syndrome S0 should be 0");
            assert!(!result.get(shot, 1), "Syndrome S1 should be 0");
            assert_eq!(result.get(shot, 2), result.get(shot, 3), "Data qubits should be correlated");
            assert_eq!(result.get(shot, 3), result.get(shot, 4), "Data qubits should be correlated");
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

        let sampler = ColumnarSampler::new(sim.measurement_history());
        let result = sampler.sample_to_result_with_thread_rng(1000);

        for shot in 0..1000 {
            assert!(!result.get(shot, 0), "Syndrome S0 should be 0");
            assert!(!result.get(shot, 1), "Syndrome S1 should be 0");
            assert_eq!(result.get(shot, 2), result.get(shot, 3), "Data qubits should be correlated");
            assert_eq!(result.get(shot, 3), result.get(shot, 4), "Data qubits should be correlated");
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

        let shot_sampler = ShotSampler::new(sim.measurement_history());
        let columnar_sampler = ColumnarSampler::new(sim.measurement_history());

        let shot_result = shot_sampler.sample_to_result_with_thread_rng(10000);
        let columnar_result = columnar_sampler.sample_to_result_with_thread_rng(10000);

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

        let sampler = ColumnarSampler::new(sim.measurement_history());
        let result = sampler.sample_to_result_with_thread_rng(100_000);

        assert_eq!(result.shots(), 100_000);
        assert_eq!(result.num_measurements(), 10);

        // Check that each measurement is roughly 50/50
        for m in 0..10 {
            let ones = result.count_ones(m);
            assert!(
                ones > 48_000 && ones < 52_000,
                "Measurement {} should be ~50/50, got {} ones",
                m,
                ones
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

        let sampler = ColumnarSampler::new(sim.measurement_history());
        let shots = 1000;
        let raw_columns = sampler.sample_raw_with_thread_rng(shots);

        assert_eq!(raw_columns.len(), 2); // 2 measurements

        // Check correlations in raw format
        let num_words = (shots + 63) / 64;
        for word_idx in 0..num_words {
            // For a Bell state, column 0 XOR column 1 should be all zeros
            assert_eq!(
                raw_columns[0][word_idx] ^ raw_columns[1][word_idx],
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

        let sampler = ColumnarSampler::new(sim.measurement_history());
        let shots = 1_000_000;
        let raw_columns = sampler.sample_raw_with_thread_rng(shots);

        assert_eq!(raw_columns.len(), 3);

        // Verify GHZ correlations: all three columns should be identical
        let num_words = (shots + 63) / 64;
        for word_idx in 0..num_words {
            assert_eq!(
                raw_columns[0][word_idx], raw_columns[1][word_idx],
                "GHZ columns 0 and 1 should be identical"
            );
            assert_eq!(
                raw_columns[1][word_idx], raw_columns[2][word_idx],
                "GHZ columns 1 and 2 should be identical"
            );
        }

        // Count ones to verify ~50% distribution
        let total_ones: u64 = raw_columns[0].iter().map(|w| w.count_ones() as u64).sum();
        let expected = shots as f64 / 2.0;
        let tolerance = (shots as f64 * 0.01) as u64; // 1% tolerance
        assert!(
            (total_ones as i64 - expected as i64).unsigned_abs() < tolerance,
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
                    assert!(dep < i, "Dependency {} should be < current index {}", dep, i);
                }
                assert!(deps.len() <= 3, "Should have at most 3 dependencies");
            }
        }

        // Create samplers and verify they work
        let shot_sampler = ShotSampler::from_measurements(measurements.clone());
        let columnar_sampler = ColumnarSampler::from_measurements(measurements);

        let shots = 1000;
        let shot_result = shot_sampler.sample_to_result_with_thread_rng(shots);
        let columnar_result = columnar_sampler.sample_to_result_with_thread_rng(shots);

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

        let sampler = ColumnarSampler::from_measurements(measurements);
        let raw = sampler.sample_raw_with_thread_rng(100_000);

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
        let measurements = MeasurementKind::generate_random(num_measurements, 0.1, 0.1, 3, &mut rng);

        let shot_sampler = ShotSampler::from_measurements(measurements.clone());
        let columnar_sampler = ColumnarSampler::from_measurements(measurements);

        let shots = 1000;

        // Test shot sampler
        let shot_result = shot_sampler.sample_to_result_with_thread_rng(shots);
        assert_eq!(shot_result.shots(), shots);
        assert_eq!(shot_result.num_measurements(), num_measurements);

        // Test columnar sampler
        let columnar_result = columnar_sampler.sample_to_result_with_thread_rng(shots);
        assert_eq!(columnar_result.shots(), shots);
        assert_eq!(columnar_result.num_measurements(), num_measurements);

        // Test raw columnar output
        let raw = columnar_sampler.sample_raw_with_thread_rng(shots);
        assert_eq!(raw.len(), num_measurements); // 200 columns
        let expected_words = (shots + 63) / 64;
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

        let sampler = ColumnarSampler::new(sim.measurement_history());

        // Test various shot counts around the 64-bit boundary
        for shots in [63, 64, 65, 127, 128, 129, 1000, 10_000] {
            let raw = sampler.sample_raw_with_thread_rng(shots);

            assert_eq!(raw.len(), 5, "Should have 5 measurement columns");

            let expected_words = (shots + 63) / 64;
            for col in &raw {
                assert_eq!(col.len(), expected_words, "Wrong word count for {shots} shots");
            }

            // Verify GHZ correlation: all columns should be identical
            for word_idx in 0..expected_words {
                let first = raw[0][word_idx];
                for col in &raw[1..] {
                    assert_eq!(col[word_idx], first, "GHZ correlation broken at word {word_idx}");
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

        let sampler = ColumnarSampler::new(sim.measurement_history());
        let result = sampler.sample_to_result_with_thread_rng(1000);

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

        let sampler = ColumnarSampler::new(sim.measurement_history());
        let shots = 10_000;
        let result = sampler.sample_to_result_with_thread_rng(shots);

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

        let sampler = ColumnarSampler::new(sim.measurement_history());
        let result = sampler.sample_to_result_with_thread_rng(100);

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

        let sampler = ColumnarSampler::new(sim.measurement_history());
        let result = sampler.sample_to_result_with_thread_rng(100);

        for (shot_idx, row) in result.iter_shots().enumerate() {
            assert!(row[0], "m0 should be 1 at shot {shot_idx}");
            assert!(!row[1], "m1 should be 0 at shot {shot_idx}");
        }
    }

    #[test]
    fn test_sample_result_try_get() {
        let mut sim = StdSymbolicSparseStab::new(1);
        sim.mz(0);

        let sampler = ColumnarSampler::new(sim.measurement_history());
        let result = sampler.sample_to_result_with_thread_rng(10);

        // Valid access
        assert!(result.try_get(0, 0).is_some());
        assert!(result.try_get(9, 0).is_some());

        // Out of bounds
        assert!(result.try_get(10, 0).is_none()); // shot out of bounds
        assert!(result.try_get(0, 1).is_none()); // measurement out of bounds
    }

    #[test]
    fn test_sample_result_column_access() {
        let mut sim = StdSymbolicSparseStab::new(2);
        sim.h(0).cx(0, 1);
        sim.mz(0);
        sim.mz(1);

        let sampler = ColumnarSampler::new(sim.measurement_history());
        let result = sampler.sample_to_result_with_thread_rng(1000);

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

        let sampler = ColumnarSampler::new(sim.measurement_history());
        let result = sampler.sample_to_result_with_thread_rng(100);

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
        let measurements = vec![
            MeasurementKind::Random,
            MeasurementKind::CopyFlipped(0),
        ];

        let shot_sampler = ShotSampler::from_measurements(measurements.clone());
        let columnar_sampler = ColumnarSampler::from_measurements(measurements);

        // Test shot sampler
        let shot_result = shot_sampler.sample_to_result_with_thread_rng(1000);
        for shot in 0..1000 {
            assert_ne!(
                shot_result.get(shot, 0),
                shot_result.get(shot, 1),
                "m1 should be negation of m0"
            );
        }

        // Test columnar sampler
        let result = columnar_sampler.sample_to_result_with_thread_rng(1000);
        for shot in 0..1000 {
            assert_ne!(
                result.get(shot, 0),
                result.get(shot, 1),
                "m1 should be negation of m0 at shot {shot}"
            );
        }

        // Verify raw columns are bitwise NOT of each other
        let raw = columnar_sampler.sample_raw_with_thread_rng(1000);
        for (w0, w1) in raw[0].iter().zip(raw[1].iter()) {
            assert_eq!(*w1, !*w0, "Column 1 should be bitwise NOT of column 0");
        }
    }
}
