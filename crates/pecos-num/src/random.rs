// Copyright 2025 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Random number generation compatible with numpy.random.
//!
//! This module provides drop-in replacements for commonly used numpy.random functions,
//! with the same API and statistical properties. Functions use the Rust standard
//! library's random number generation.
//!
//! # Design Philosophy
//!
//! 1. **Phase 1 (Current)**: Drop-in replacements with identical APIs
//!    - Expected speedup: 1.2-2x from reduced Python overhead
//!    - Focus: Correctness and compatibility
//!
//! 2. **Phase 2 (Future)**: Fused operations for performance
//!    - Expected speedup: 5-10x for error generation patterns
//!    - Focus: Eliminating intermediate allocations and Python loops
//!
//! # Example
//!
//! ```
//! use pecos_num::random::random;
//!
//! // Generate 100 random floats in [0.0, 1.0)
//! let random_values = random(100);
//! assert_eq!(random_values.len(), 100);
//! ```

use ndarray::Array1;
use pecos_rng::PecosRng;
use rand::RngExt;
use rand::distr::uniform::SampleUniform;
use rand::seq::SliceRandom;
use std::cell::RefCell;

// Thread-local seeded RNG for reproducibility
thread_local! {
    static SEEDED_RNG: RefCell<Option<PecosRng>> = const { RefCell::new(None) };
}

/// Execute a closure with the appropriate RNG.
///
/// If a seed has been set, uses the seeded RNG and advances its state.
/// Otherwise, uses a fresh entropy-based RNG.
fn with_rng<F, R>(f: F) -> R
where
    F: FnOnce(&mut PecosRng) -> R,
{
    SEEDED_RNG.with(|cell| {
        let mut rng_opt = cell.borrow_mut();
        if let Some(ref mut rng) = *rng_opt {
            // Use seeded RNG and advance its state
            f(rng)
        } else {
            // Use fresh RNG seeded from thread_rng
            let mut thread_rng = rand::rng();
            let seed = thread_rng.random();
            let mut rng = PecosRng::seed_from_u64(seed);
            f(&mut rng)
        }
    })
}

/// Set the random seed for reproducible results.
///
/// This sets a thread-local seed, similar to `numpy.random.seed()`.
/// All subsequent random number generation in the current thread will
/// use this seed, producing reproducible sequences.
///
/// # Arguments
///
/// * `seed` - The seed value (u64)
///
/// # Example
///
/// ```
/// use pecos_num::random::{seed, random};
///
/// seed(42);
/// let values1 = random(10);
///
/// seed(42);
/// let values2 = random(10);
///
/// // Same seed produces same sequence
/// assert_eq!(values1, values2);
/// ```
///
/// # Thread Safety
///
/// Each thread has its own seed. Setting the seed in one thread does not
/// affect random number generation in other threads.
pub fn seed(seed_value: u64) {
    SEEDED_RNG.with(|cell| {
        *cell.borrow_mut() = Some(PecosRng::seed_from_u64(seed_value));
    });
}

/// Generate random floats from a uniform distribution over [0.0, 1.0).
///
/// This is a drop-in replacement for `numpy.random.random(size)`.
///
/// # Arguments
///
/// * `size` - Number of random values to generate
///
/// # Returns
///
/// Returns an array of `size` random floats, each uniformly distributed in [0.0, 1.0).
///
/// # Examples
///
/// ```
/// use pecos_num::random::random;
///
/// // Generate 5 random values
/// let values = random(5);
/// assert_eq!(values.len(), 5);
///
/// // All values should be in [0.0, 1.0)
/// for &v in &values {
///     assert!(v >= 0.0 && v < 1.0);
/// }
/// ```
///
/// # Performance
///
/// Uses `rand::rng()` which is:
/// - Thread-local (no synchronization overhead)
/// - High-quality PRNG (PCG or similar)
/// - Fast (~1-2ns per number on modern CPUs)
///
/// Expected to be 1.2-1.5x faster than `numpy.random.random()` due to:
/// - Reduced Python/FFI overhead
/// - Efficient Rust memory allocation
/// - No GIL contention
#[must_use]
pub fn random(size: usize) -> Array1<f64> {
    with_rng(|rng| Array1::from_vec((0..size).map(|_| rng.random::<f64>()).collect()))
}

/// Generate random integers from a uniform distribution.
///
/// This is a drop-in replacement for `numpy.random.randint(low, high, size)`.
///
/// # Arguments
///
/// * `low` - Lowest (signed) integer to be drawn from the distribution
/// * `high` - If provided, one above the largest integer to be drawn. If None, range is [0, low)
/// * `size` - Output shape. If None, returns a single integer.
///
/// # Returns
///
/// - If `size` is None: returns a single random integer in the range [low, high)
/// - If `size` is Some(n): returns an array of n random integers
///
/// # Examples
///
/// ```
/// use pecos_num::random::{randint_scalar, randint};
///
/// // Single random integer in [0, 10)
/// let value = randint_scalar(10, None);
/// assert!(value >= 0 && value < 10);
///
/// // Single random integer in [5, 10)
/// let value = randint_scalar(5, Some(10));
/// assert!(value >= 5 && value < 10);
///
/// // Array of random integers in [0, 5)
/// let values = randint(5, None, 100);
/// assert_eq!(values.len(), 100);
/// for &v in &values {
///     assert!(v >= 0 && v < 5);
/// }
/// ```
///
/// # Performance
///
/// Uses `rand::rng()` with uniform distribution sampling, expected to be
/// 1.2-1.5x faster than `numpy.random.randint()` due to reduced Python overhead.
pub fn randint_scalar<T>(low: T, high: Option<T>) -> T
where
    T: SampleUniform + PartialOrd + Default + Copy,
{
    with_rng(|rng| {
        let (start, end) = match high {
            Some(h) => (low, h),
            None => (T::default(), low),
        };
        rng.random_range(start..end)
    })
}

/// Generate an array of random integers from a uniform distribution.
///
/// This is a drop-in replacement for `numpy.random.randint(low, high, size)`.
///
/// # Arguments
///
/// * `low` - Lowest (signed) integer to be drawn from the distribution
/// * `high` - If provided, one above the largest integer to be drawn. If None, range is [0, low)
/// * `size` - Number of random integers to generate
///
/// # Returns
///
/// Returns an array of `size` random integers in the range [low, high) or [0, low).
///
/// # Examples
///
/// ```
/// use pecos_num::random::randint;
///
/// // Array of 10 random integers in [0, 100)
/// let values = randint(100, None, 10);
/// assert_eq!(values.len(), 10);
/// for &v in &values {
///     assert!(v >= 0 && v < 100);
/// }
///
/// // Array of 10 random integers in [50, 100)
/// let values = randint(50, Some(100), 10);
/// for &v in &values {
///     assert!(v >= 50 && v < 100);
/// }
/// ```
#[must_use]
pub fn randint<T>(low: T, high: Option<T>, size: usize) -> Array1<T>
where
    T: SampleUniform + PartialOrd + Default + Copy,
{
    with_rng(|rng| {
        let (start, end) = match high {
            Some(h) => (low, h),
            None => (T::default(), low),
        };

        Array1::from_vec((0..size).map(|_| rng.random_range(start..end)).collect())
    })
}

/// Generate a random sample from a given array.
///
/// This is a drop-in replacement for `numpy.random.choice(a, size, replace=True)`.
///
/// # Arguments
///
/// * `array` - Array to sample from
/// * `size` - Number of samples to draw
/// * `replace` - Whether to sample with replacement (True) or without (False)
///
/// # Returns
///
/// Returns a vector of `size` random samples from the input array.
///
/// # Panics
///
/// Panics if:
/// - `array` is empty
/// - `replace=false` and `size > array.len()`
///
/// # Examples
///
/// ```
/// use pecos_num::random::choice;
///
/// let items = vec!["X", "Y", "Z"];
///
/// // Sample with replacement (can repeat)
/// let samples = choice(&items, 5, true);
/// assert_eq!(samples.len(), 5);
///
/// // Sample without replacement (no repeats)
/// let samples = choice(&items, 2, false);
/// assert_eq!(samples.len(), 2);
/// // samples contains 2 different elements from items
/// ```
///
/// # Performance
///
/// Expected to be 1.3-2x faster than `numpy.random.choice()` due to:
/// - Reduced Python overhead
/// - Efficient Rust slice sampling
/// - No intermediate array conversions
pub fn choice<T: Clone>(array: &[T], size: usize, replace: bool) -> Vec<T> {
    assert!(!array.is_empty(), "Cannot sample from empty array");

    if !replace {
        assert!(
            size <= array.len(),
            "Cannot take larger sample than population when replace=false"
        );
    }

    with_rng(|rng| {
        if replace {
            // Sample with replacement - use random index
            (0..size)
                .map(|_| {
                    let idx = rng.random_range(0..array.len());
                    array[idx].clone()
                })
                .collect()
        } else {
            // Sample without replacement using partial_shuffle
            let mut indices: Vec<usize> = (0..array.len()).collect();
            let (selected, _) = indices.partial_shuffle(rng, size);
            selected.iter().map(|&i| array[i].clone()).collect()
        }
    })
}

/// Generate a single random sample from a given array.
///
/// This is a convenience function for selecting a single element.
///
/// # Arguments
///
/// * `array` - Array to sample from
///
/// # Returns
///
/// Returns a single random element from the input array.
///
/// # Panics
///
/// Panics if `array` is empty.
///
/// # Examples
///
/// ```
/// use pecos_num::random::choice_scalar;
///
/// let items = vec!["X", "Y", "Z"];
/// let sample = choice_scalar(&items);
/// assert!(items.contains(&sample));
/// ```
pub fn choice_scalar<T: Clone>(array: &[T]) -> T {
    assert!(!array.is_empty(), "Cannot sample from empty array");
    with_rng(|rng| {
        let idx = rng.random_range(0..array.len());
        array[idx].clone()
    })
}

/// Fused operation: Check if any random value is less than threshold.
///
/// This is a fused version of `np.any(np.random.random(size) < threshold)`.
/// Instead of allocating an array and then reducing it, this directly generates
/// random values and short-circuits on the first match.
///
/// # Arguments
///
/// * `size` - Number of random values to potentially generate
/// * `threshold` - Threshold to compare against (typically a probability)
///
/// # Returns
///
/// Returns `true` if any generated random value is less than `threshold`,
/// `false` otherwise.
///
/// # Performance
///
/// This is significantly faster than the unfused numpy version because:
/// - No array allocation (saves memory bandwidth)
/// - Short-circuit evaluation (returns immediately on first match)
/// - No Python overhead for array creation and reduction
///
/// Expected speedup: 2-3x for typical error model use cases.
///
/// # Examples
///
/// ```
/// use pecos_num::random::{compare_any, seed};
///
/// // Seed for reproducibility
/// seed(42);
///
/// // Check if any of 100 random values < 0.01 (1% error rate)
/// let has_error = compare_any(100, 0.01);
/// ```
#[must_use]
pub fn compare_any(size: usize, threshold: f64) -> bool {
    with_rng(|rng| {
        for _ in 0..size {
            if rng.random::<f64>() < threshold {
                return true;
            }
        }
        false
    })
}

/// Fused operation: Get indices where random values are less than threshold.
///
/// This is a fused version of the pattern:
/// ```python
/// rand_nums = np.random.random(size) <= threshold
/// indices = [i for i, r in enumerate(rand_nums) if r]
/// ```
///
/// Instead of allocating a boolean array and then filtering it, this directly
/// generates random values and collects matching indices.
///
/// # Arguments
///
/// * `size` - Number of random values to generate
/// * `threshold` - Threshold to compare against (typically a probability)
///
/// # Returns
///
/// Returns a vector of indices where the random value was less than `threshold`.
///
/// # Performance
///
/// This is faster than the unfused numpy version because:
/// - No intermediate boolean array allocation
/// - Direct collection of matching indices
/// - No Python overhead for array operations
///
/// Expected speedup: 1.5-2x for typical error model use cases.
///
/// # Examples
///
/// ```
/// use pecos_num::random::{compare_indices, seed};
///
/// // Seed for reproducibility
/// seed(42);
///
/// // Get indices of qubits with errors (1% error rate)
/// let error_indices = compare_indices(100, 0.01);
/// println!("Errors at indices: {:?}", error_indices);
/// ```
#[must_use]
pub fn compare_indices(size: usize, threshold: f64) -> Vec<usize> {
    with_rng(|rng| {
        (0..size)
            .filter(|_| rng.random::<f64>() < threshold)
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_size() {
        // Test various sizes
        assert_eq!(random(0).len(), 0);
        assert_eq!(random(1).len(), 1);
        assert_eq!(random(10).len(), 10);
        assert_eq!(random(1000).len(), 1000);
    }

    #[test]
    fn test_random_range() {
        // All values should be in [0.0, 1.0)
        let values = random(1000);
        for &v in &values {
            assert!(v >= 0.0, "Value {v} is less than 0.0");
            assert!(v < 1.0, "Value {v} is not less than 1.0");
        }
    }

    #[test]
    fn test_random_statistical_properties() {
        // Test that mean is approximately 0.5 for uniform [0, 1)
        seed(12345);
        let n = 10000;
        let values = random(n);

        let mean = values.mean().unwrap();
        let variance = values.var(0.0);

        // For uniform [0, 1): theoretical mean = 0.5, variance = 1/12 ≈ 0.0833
        // With n=10000, we expect mean within ~0.01 of 0.5 with high probability
        assert!(
            (mean - 0.5).abs() < 0.01,
            "Mean {mean} is too far from expected 0.5"
        );

        // Variance should be close to 1/12 ≈ 0.0833
        let expected_variance = 1.0 / 12.0;
        assert!(
            (variance - expected_variance).abs() < 0.01,
            "Variance {variance} is too far from expected {expected_variance}"
        );
    }

    #[test]
    fn test_random_independence() {
        // Generate two sequences and ensure they're different
        let seq1 = random(100);
        let seq2 = random(100);

        // Count how many are equal (should be very few for f64)
        // Exact comparison is intentional - we want to detect duplicate generation
        #[allow(clippy::float_cmp)]
        let equal_count = seq1
            .iter()
            .zip(seq2.iter())
            .filter(|&(&a, &b)| a == b)
            .count();

        // With f64 precision, probability of exact match is ~0
        assert!(
            equal_count < 5,
            "Too many equal values ({equal_count}/100), sequences may not be independent"
        );
    }

    #[test]
    fn test_random_distribution_uniformity() {
        // Chi-square test for uniformity
        // Divide [0, 1) into 10 bins and check counts
        seed(54321);
        let n = 10000;
        let values = random(n);
        let num_bins = 10;
        let mut bins = vec![0; num_bins];

        for &v in &values {
            // Casts are safe: num_bins=10 fits in u32, bin index always < num_bins
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let bin = (v * f64::from(num_bins as u32)).floor() as usize;
            let bin = bin.min(num_bins - 1); // Handle edge case where v = 1.0
            bins[bin] += 1;
        }

        // Expected count per bin
        // Cast is safe: 10000 points / 10 bins = 1000, well within f64 range
        #[allow(clippy::cast_precision_loss)]
        let expected = n as f64 / num_bins as f64;

        // Chi-square statistic
        let chi_square: f64 = bins
            .iter()
            .map(|&count| {
                // Cast is safe: counts are < n = 10000, well within f64 range
                #[allow(clippy::cast_precision_loss)]
                let count_f64 = f64::from(count);
                let diff = count_f64 - expected;
                diff * diff / expected
            })
            .sum();

        // For 10 bins (9 degrees of freedom), critical value at p=0.01 is ~21.67
        // If chi_square > critical value, distribution is likely not uniform
        assert!(
            chi_square < 21.67,
            "Chi-square {chi_square} exceeds critical value, distribution may not be uniform"
        );
    }

    // Tests for randint_scalar and randint
    #[test]
    fn test_randint_scalar_range_default_low() {
        // Test [0, n) behavior when high is None
        for _ in 0..100 {
            let val = randint_scalar(10, None);
            assert!((0..10).contains(&val), "Value {val} outside range [0, 10)");
        }
    }

    #[test]
    fn test_randint_scalar_range_with_high() {
        // Test [low, high) behavior
        for _ in 0..100 {
            let val = randint_scalar(5, Some(15));
            assert!((5..15).contains(&val), "Value {val} outside range [5, 15)");
        }
    }

    #[test]
    fn test_randint_array_size() {
        let values = randint(100, None, 50);
        assert_eq!(values.len(), 50);
    }

    #[test]
    fn test_randint_array_range() {
        let values = randint(10, Some(20), 1000);
        for &v in &values {
            assert!((10..20).contains(&v), "Value {v} outside range [10, 20)");
        }
    }

    #[test]
    fn test_randint_negative_range() {
        // Test with negative integers
        let values = randint(-10, Some(10), 1000);
        for &v in &values {
            assert!((-10..10).contains(&v), "Value {v} outside range [-10, 10)");
        }
    }

    #[test]
    fn test_randint_statistical_uniformity() {
        // Chi-square test for uniformity of randint
        // Use unsigned types since we're dealing with array sizes/indices
        seed(11111);
        let n = 10000;
        let range_size: u32 = 10;
        let values = randint(range_size, None, n);

        let mut counts = vec![0; range_size as usize];
        for &v in &values {
            counts[v as usize] += 1;
        }

        // Cast is safe: 10000 / 10 = 1000, well within f64 range
        #[allow(clippy::cast_precision_loss)]
        let expected = n as f64 / f64::from(range_size);

        let chi_square: f64 = counts
            .iter()
            .map(|&count| {
                // Cast is safe: counts are < n = 10000
                #[allow(clippy::cast_precision_loss)]
                let count_f64 = f64::from(count);
                let diff = count_f64 - expected;
                diff * diff / expected
            })
            .sum();

        // For 10 values (9 degrees of freedom), critical value at p=0.01 is ~21.67
        assert!(
            chi_square < 21.67,
            "Chi-square {chi_square} exceeds critical value"
        );
    }

    // Tests for choice and choice_scalar
    #[test]
    fn test_choice_scalar() {
        let items = vec!["X", "Y", "Z"];
        for _ in 0..100 {
            let sample = choice_scalar(&items);
            assert!(items.contains(&sample));
        }
    }

    #[test]
    fn test_choice_with_replacement() {
        let items = vec![1, 2, 3, 4, 5];
        let samples = choice(&items, 20, true);
        assert_eq!(samples.len(), 20);
        // All samples should be in the original array
        for &sample in &samples {
            assert!(items.contains(&sample));
        }
    }

    #[test]
    fn test_choice_without_replacement() {
        let items = vec![1, 2, 3, 4, 5];
        let samples = choice(&items, 3, false);
        assert_eq!(samples.len(), 3);

        // All samples should be in the original array
        for &sample in &samples {
            assert!(items.contains(&sample));
        }

        // All samples should be unique
        let mut sorted_samples = samples.clone();
        sorted_samples.sort_unstable();
        sorted_samples.dedup();
        assert_eq!(sorted_samples.len(), 3, "Samples should be unique");
    }

    #[test]
    #[should_panic(expected = "Cannot sample from empty array")]
    fn test_choice_empty_array() {
        let empty: Vec<i32> = vec![];
        choice(&empty, 5, true);
    }

    #[test]
    #[should_panic(expected = "Cannot take larger sample than population")]
    fn test_choice_without_replacement_too_large() {
        let items = vec![1, 2, 3];
        choice(&items, 5, false);
    }

    #[test]
    fn test_choice_uniformity() {
        // Test that choice samples uniformly
        seed(22222);
        let items = vec![0, 1, 2, 3, 4];
        let n = 10000;
        let samples = choice(&items, n, true);

        let mut counts = vec![0; items.len()];
        for &sample in &samples {
            counts[sample] += 1;
        }

        // Cast is safe: 10000 / 5 = 2000, well within f64 range
        #[allow(clippy::cast_precision_loss)]
        let expected = n as f64 / items.len() as f64;

        let chi_square: f64 = counts
            .iter()
            .map(|&count| {
                // Cast is safe: counts are < n = 10000
                #[allow(clippy::cast_precision_loss)]
                let count_f64 = f64::from(count);
                let diff = count_f64 - expected;
                diff * diff / expected
            })
            .sum();

        // For 5 values (4 degrees of freedom), critical value at p=0.01 is ~13.28
        assert!(
            chi_square < 13.28,
            "Chi-square {chi_square} exceeds critical value"
        );
    }

    #[test]
    fn test_seed_reproducibility_random() {
        // Test that seeding produces reproducible sequences
        seed(42);
        let values1 = random(10);

        seed(42);
        let values2 = random(10);

        assert_eq!(values1, values2, "Same seed should produce same sequence");
    }

    #[test]
    fn test_seed_reproducibility_randint() {
        // Test that seeding works for randint
        seed(123);
        let values1 = randint(0, Some(100), 10);

        seed(123);
        let values2 = randint(0, Some(100), 10);

        assert_eq!(values1, values2, "Same seed should produce same sequence");
    }

    #[test]
    fn test_seed_reproducibility_choice() {
        // Test that seeding works for choice
        let items = vec![1, 2, 3, 4, 5];

        seed(456);
        let samples1 = choice(&items, 10, true);

        seed(456);
        let samples2 = choice(&items, 10, true);

        assert_eq!(samples1, samples2, "Same seed should produce same sequence");
    }

    #[test]
    fn test_different_seeds_different_sequences() {
        // Test that different seeds produce different sequences
        seed(42);
        let values1 = random(100);

        seed(43);
        let values2 = random(100);

        // With 100 random floats, probability of collision is negligible
        assert_ne!(
            values1, values2,
            "Different seeds should produce different sequences"
        );
    }

    #[test]
    fn test_seed_advances_state() {
        // Test that RNG state advances between calls
        seed(789);
        let val1 = random(1);
        let val2 = random(1);

        // These should be different (not re-seeded)
        // Exact comparison is intentional - testing RNG state advancement
        #[allow(clippy::float_cmp)]
        {
            assert_ne!(val1[0], val2[0], "RNG state should advance between calls");
        }

        // Re-seed and verify we get the same first value
        seed(789);
        let val3 = random(1);
        assert_eq!(val1, val3, "Re-seeding should reset sequence");
    }

    // Tests for fused operations

    #[test]
    fn test_compare_any_basic() {
        // Test with threshold=1.0 - should always be true
        assert!(compare_any(10, 1.0), "All random values should be < 1.0");

        // Test with threshold=0.0 - should always be false
        assert!(!compare_any(10, 0.0), "No random values should be < 0.0");
    }

    #[test]
    fn test_compare_any_reproducibility() {
        // Test reproducibility with seeding
        seed(12345);
        let result1 = compare_any(100, 0.5);

        seed(12345);
        let result2 = compare_any(100, 0.5);

        assert_eq!(result1, result2, "Same seed should produce same result");
    }

    #[test]
    fn test_compare_any_statistical() {
        // For large n and p=0.5, probability of at least one hit approaches 1
        seed(999);
        let large_n_result = compare_any(1000, 0.5);
        assert!(
            large_n_result,
            "With n=1000 and p=0.5, should almost certainly get at least one hit"
        );

        // For small p, should mostly return false
        seed(888);
        // Cast is safe: i in 0..100 is always positive
        #[allow(clippy::cast_sign_loss)]
        let small_p_count: usize = (0..100)
            .filter(|&i| {
                seed(888 + i as u64);
                compare_any(10, 0.01)
            })
            .count();

        // Expect ~10% of trials to have at least one hit (binomial)
        // P(at least one) = 1 - (1-0.01)^10 ≈ 0.096
        assert!(
            small_p_count < 30,
            "Expected <30 hits out of 100 trials with p=0.01, got {small_p_count}"
        );
    }

    #[test]
    fn test_compare_indices_basic() {
        // Test with threshold=1.0 - should return all indices
        let result = compare_indices(10, 1.0);
        assert_eq!(
            result.len(),
            10,
            "All indices should match with threshold=1.0"
        );
        assert_eq!(result, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);

        // Test with threshold=0.0 - should return empty
        let result = compare_indices(10, 0.0);
        assert_eq!(
            result.len(),
            0,
            "No indices should match with threshold=0.0"
        );
    }

    #[test]
    fn test_compare_indices_reproducibility() {
        // Test reproducibility with seeding
        seed(54321);
        let result1 = compare_indices(100, 0.1);

        seed(54321);
        let result2 = compare_indices(100, 0.1);

        assert_eq!(result1, result2, "Same seed should produce same indices");
    }

    #[test]
    fn test_compare_indices_statistical() {
        // For p=0.5, expect approximately 50% of indices
        seed(777);
        let result = compare_indices(10000, 0.5);

        let count = result.len();
        let expected = 5000;
        let tolerance = 200; // Allow ±200 for statistical variation

        assert!(
            count > expected - tolerance && count < expected + tolerance,
            "Expected ~{expected} indices (±{tolerance}), got {count}"
        );

        // Verify all indices are valid and in range
        for &idx in &result {
            assert!(idx < 10000, "Index {idx} out of range");
        }

        // Verify indices are in ascending order (as they're generated sequentially)
        for i in 1..result.len() {
            assert!(
                result[i] > result[i - 1],
                "Indices should be in ascending order"
            );
        }
    }

    #[test]
    fn test_compare_indices_vs_compare_any_consistency() {
        // If compare_indices returns non-empty, compare_any should return true
        seed(111);
        let indices = compare_indices(100, 0.1);

        seed(111);
        let has_any = compare_any(100, 0.1);

        if !indices.is_empty() {
            assert!(has_any, "If indices non-empty, compare_any should be true");
        }
    }
}
