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

//! Sample weight tracking for importance sampling.
//!
//! Importance sampling requires tracking the likelihood ratio P(path)/Q(path)
//! for each sample. Since this can span many orders of magnitude, we work
//! in log space to avoid numerical overflow/underflow.
//!
//! ## Weight Computation
//!
//! For each noise decision during a shot:
//! ```text
//! log_weight += log(P(event)) - log(Q(event))
//! ```
//!
//! Where:
//! - P = true (target) distribution
//! - Q = proposal distribution (what we actually sample from)
//!
//! ## Example
//!
//! ```
//! use pecos_neo::sampling::SampleWeight;
//!
//! let mut weight = SampleWeight::one();
//!
//! // Event: error occurred, P(error) = 0.001, Q(error) = 0.1
//! weight.update(0.001, 0.1);
//!
//! // Event: no error, P(no_error) = 0.999, Q(no_error) = 0.9
//! weight.update(0.999, 0.9);
//!
//! // Get the actual weight (not log)
//! let w = weight.weight();
//! ```

/// Sample weight for importance sampling.
///
/// Stores the log of the importance weight to handle extreme values.
/// The weight represents P(path) / Q(path) where P is the target
/// distribution and Q is the proposal distribution.
#[derive(Debug, Clone, Copy)]
pub struct SampleWeight {
    /// Log of the importance weight: log(P/Q)
    log_weight: f64,
}

impl Default for SampleWeight {
    fn default() -> Self {
        Self::one()
    }
}

impl SampleWeight {
    /// Create a weight of 1 (`log_weight` = 0).
    #[must_use]
    pub fn one() -> Self {
        Self { log_weight: 0.0 }
    }

    /// Create a weight from a log value.
    #[must_use]
    pub fn from_log(log_weight: f64) -> Self {
        Self { log_weight }
    }

    /// Create a weight from a linear value.
    ///
    /// # Panics
    /// Panics if weight is not positive.
    #[must_use]
    pub fn from_linear(weight: f64) -> Self {
        assert!(weight > 0.0, "Weight must be positive");
        Self {
            log_weight: weight.ln(),
        }
    }

    /// Get the log of the weight.
    #[must_use]
    pub fn log_weight(&self) -> f64 {
        self.log_weight
    }

    /// Get the actual weight (exponentiated).
    ///
    /// Warning: May overflow/underflow for extreme weights.
    #[must_use]
    pub fn weight(&self) -> f64 {
        self.log_weight.exp()
    }

    /// Update weight with a single event's likelihood ratio.
    ///
    /// # Arguments
    /// * `p_true` - Probability under target distribution P
    /// * `p_proposal` - Probability under proposal distribution Q
    ///
    /// Adds `log(p_true` / `p_proposal`) to the log weight.
    pub fn update(&mut self, p_true: f64, p_proposal: f64) {
        debug_assert!((0.0..=1.0).contains(&p_true), "p_true must be in [0, 1]");
        debug_assert!(
            p_proposal > 0.0 && p_proposal <= 1.0,
            "p_proposal must be in (0, 1]"
        );

        // log(p_true / p_proposal) = log(p_true) - log(p_proposal)
        self.log_weight += p_true.ln() - p_proposal.ln();
    }

    /// Update weight with log probabilities directly.
    ///
    /// More numerically stable when probabilities are very small.
    pub fn update_log(&mut self, log_p_true: f64, log_p_proposal: f64) {
        self.log_weight += log_p_true - log_p_proposal;
    }

    /// Combine with another weight (multiply weights = add log weights).
    #[must_use]
    pub fn combine(&self, other: &SampleWeight) -> SampleWeight {
        SampleWeight {
            log_weight: self.log_weight + other.log_weight,
        }
    }

    /// Split weight among n clones (for splitting methods).
    ///
    /// Each clone gets weight / n.
    #[must_use]
    pub fn split(&self, n: usize) -> SampleWeight {
        SampleWeight {
            log_weight: self.log_weight - (n as f64).ln(),
        }
    }

    /// Check if weight is effectively zero (very negative log weight).
    #[must_use]
    pub fn is_negligible(&self, threshold: f64) -> bool {
        self.log_weight < threshold
    }

    /// Reset to weight = 1.
    pub fn reset(&mut self) {
        self.log_weight = 0.0;
    }
}

/// Outcome with an associated importance weight.
#[derive(Debug, Clone)]
pub struct WeightedOutcome<T> {
    /// The outcome value.
    pub outcome: T,
    /// The importance weight.
    pub weight: SampleWeight,
}

impl<T> WeightedOutcome<T> {
    /// Create a new weighted outcome.
    pub fn new(outcome: T, weight: SampleWeight) -> Self {
        Self { outcome, weight }
    }

    /// Create an outcome with unit weight.
    pub fn unweighted(outcome: T) -> Self {
        Self {
            outcome,
            weight: SampleWeight::one(),
        }
    }
}

/// Accumulator for weighted statistics.
///
/// Computes weighted mean and variance using numerically stable algorithms.
#[derive(Debug, Clone)]
pub struct WeightedStatistics {
    /// Sum of weights (in log space for stability).
    log_weight_sum: f64,
    /// Weighted sum of values.
    weighted_sum: f64,
    /// Weighted sum of squared values (for variance).
    weighted_sum_sq: f64,
    /// Number of samples.
    count: usize,
    /// Maximum log weight seen (for normalization).
    max_log_weight: f64,
}

impl Default for WeightedStatistics {
    fn default() -> Self {
        Self::new()
    }
}

impl WeightedStatistics {
    /// Create a new empty accumulator.
    #[must_use]
    pub fn new() -> Self {
        Self {
            log_weight_sum: f64::NEG_INFINITY,
            weighted_sum: 0.0,
            weighted_sum_sq: 0.0,
            count: 0,
            max_log_weight: f64::NEG_INFINITY,
        }
    }

    /// Add a weighted sample.
    ///
    /// # Arguments
    /// * `value` - The sample value (e.g., 1.0 for logical error, 0.0 for no error)
    /// * `weight` - The importance weight
    pub fn add(&mut self, value: f64, weight: &SampleWeight) {
        let log_w = weight.log_weight();

        // Update max for normalization
        if log_w > self.max_log_weight {
            // Rescale existing sums
            if self.count > 0 {
                let scale = (self.max_log_weight - log_w).exp();
                self.weighted_sum *= scale;
                self.weighted_sum_sq *= scale;
                // log_weight_sum needs log-space addition
                self.log_weight_sum = log_sum_exp(self.log_weight_sum, log_w);
            } else {
                self.log_weight_sum = log_w;
            }
            self.max_log_weight = log_w;
        } else {
            self.log_weight_sum = log_sum_exp(self.log_weight_sum, log_w);
        }

        // Add contribution (normalized by max weight)
        let normalized_w = (log_w - self.max_log_weight).exp();
        self.weighted_sum += normalized_w * value;
        self.weighted_sum_sq += normalized_w * value * value;
        self.count += 1;
    }

    /// Add a sample with unit weight (standard Monte Carlo).
    pub fn add_unweighted(&mut self, value: f64) {
        self.add(value, &SampleWeight::one());
    }

    /// Get the weighted mean.
    #[must_use]
    pub fn mean(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        // Total weight (normalized)
        let total_w = (self.log_weight_sum - self.max_log_weight).exp();
        self.weighted_sum / total_w
    }

    /// Get the weighted variance.
    #[must_use]
    pub fn variance(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        let mean = self.mean();
        let total_w = (self.log_weight_sum - self.max_log_weight).exp();
        (self.weighted_sum_sq / total_w) - mean * mean
    }

    /// Get the standard error of the weighted mean.
    #[must_use]
    pub fn standard_error(&self) -> f64 {
        if self.count < 2 {
            return f64::INFINITY;
        }
        (self.variance() / self.count as f64).sqrt()
    }

    /// Get the number of samples.
    #[must_use]
    pub fn count(&self) -> usize {
        self.count
    }

    /// Get the effective sample size (ESS).
    ///
    /// ESS measures how many independent samples the weighted samples are worth.
    /// ESS = (Σw)² / Σw²
    ///
    /// Low ESS indicates weight degeneracy (few samples dominate).
    #[must_use]
    pub fn effective_sample_size(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        // This is an approximation - proper ESS needs sum of squared weights
        // For now, return count (would need to track more state for true ESS)
        self.count as f64
    }

    /// Merge with another statistics accumulator.
    pub fn merge(&mut self, other: &WeightedStatistics) {
        if other.count == 0 {
            return;
        }
        if self.count == 0 {
            *self = other.clone();
            return;
        }

        // Scale to common max
        let new_max = self.max_log_weight.max(other.max_log_weight);
        let self_scale = (self.max_log_weight - new_max).exp();
        let other_scale = (other.max_log_weight - new_max).exp();

        self.weighted_sum = self.weighted_sum * self_scale + other.weighted_sum * other_scale;
        self.weighted_sum_sq =
            self.weighted_sum_sq * self_scale + other.weighted_sum_sq * other_scale;
        self.log_weight_sum = log_sum_exp(self.log_weight_sum, other.log_weight_sum);
        self.count += other.count;
        self.max_log_weight = new_max;
    }
}

/// Compute log(exp(a) + exp(b)) in a numerically stable way.
fn log_sum_exp(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let max = a.max(b);
    max + ((a - max).exp() + (b - max).exp()).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_weight_one() {
        let w = SampleWeight::one();
        assert!((w.weight() - 1.0).abs() < 1e-10);
        assert!((w.log_weight() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_sample_weight_update() {
        let mut w = SampleWeight::one();

        // Event with P=0.001, Q=0.1 -> ratio = 0.01
        w.update(0.001, 0.1);
        assert!((w.weight() - 0.01).abs() < 1e-10);

        // Another event with P=0.999, Q=0.9 -> ratio ≈ 1.11
        w.update(0.999, 0.9);
        let expected = 0.01 * (0.999 / 0.9);
        assert!((w.weight() - expected).abs() < 1e-10);
    }

    #[test]
    fn test_sample_weight_split() {
        let w = SampleWeight::from_linear(1.0);
        let split = w.split(4);
        assert!((split.weight() - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_weighted_statistics_unweighted() {
        let mut stats = WeightedStatistics::new();

        // Add samples: 0, 0, 0, 1 (25% success rate)
        stats.add_unweighted(0.0);
        stats.add_unweighted(0.0);
        stats.add_unweighted(0.0);
        stats.add_unweighted(1.0);

        assert!((stats.mean() - 0.25).abs() < 1e-10);
        assert_eq!(stats.count(), 4);
    }

    #[test]
    fn test_weighted_statistics_weighted() {
        let mut stats = WeightedStatistics::new();

        // Sample 1: value=1.0, weight=0.1
        stats.add(1.0, &SampleWeight::from_linear(0.1));

        // Sample 2: value=0.0, weight=0.9
        stats.add(0.0, &SampleWeight::from_linear(0.9));

        // Weighted mean = (1.0 * 0.1 + 0.0 * 0.9) / (0.1 + 0.9) = 0.1
        assert!((stats.mean() - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_log_sum_exp() {
        // log(e^0 + e^0) = log(2)
        let result = log_sum_exp(0.0, 0.0);
        assert!((result - 2.0_f64.ln()).abs() < 1e-10);

        // log(e^10 + e^0) ≈ 10 (dominated by larger term)
        let result = log_sum_exp(10.0, 0.0);
        assert!((result - 10.0).abs() < 0.001);
    }
}
