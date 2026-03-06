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

//! Validation utilities for noise parameters.
//!
//! This module provides runtime validation for noise model parameters,
//! helping catch common mistakes early with clear error messages.
//!
//! # Example
//!
//! ```
//! use pecos_neo::noise::validation::*;
//!
//! // Validate a probability
//! let p = validate_probability(0.5, "error_rate").unwrap();
//!
//! // Validate weights
//! validate_weights(&[0.25, 0.25, 0.5], "pauli_weights").unwrap();
//! ```

use std::fmt;

/// Error type for validation failures.
#[derive(Debug, Clone)]
pub struct ValidationError {
    /// What was being validated.
    pub field: String,
    /// What went wrong.
    pub message: String,
    /// The invalid value (if applicable).
    pub value: Option<String>,
    /// Suggestion for fixing the error.
    pub suggestion: Option<String>,
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Invalid {}: {}", self.field, self.message)?;
        if let Some(ref val) = self.value {
            write!(f, " (got: {val})")?;
        }
        if let Some(ref suggestion) = self.suggestion {
            write!(f, ". {suggestion}")?;
        }
        Ok(())
    }
}

impl std::error::Error for ValidationError {}

impl ValidationError {
    /// Create a new validation error.
    pub fn new(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            message: message.into(),
            value: None,
            suggestion: None,
        }
    }

    /// Add the invalid value to the error.
    #[must_use]
    pub fn with_value(mut self, value: impl ToString) -> Self {
        self.value = Some(value.to_string());
        self
    }

    /// Add a suggestion for fixing the error.
    #[must_use]
    pub fn with_suggestion(mut self, suggestion: impl Into<String>) -> Self {
        self.suggestion = Some(suggestion.into());
        self
    }
}

/// Result type for validation operations.
pub type ValidationResult<T> = Result<T, ValidationError>;

/// Validate that a value is a valid probability (0.0 to 1.0).
///
/// # Errors
///
/// Returns an error if the probability is outside [0.0, 1.0] or is NaN.
pub fn validate_probability(p: f64, field: &str) -> ValidationResult<f64> {
    if p.is_nan() {
        return Err(ValidationError::new(field, "probability cannot be NaN")
            .with_value(p)
            .with_suggestion("Check for division by zero or invalid computations"));
    }
    if p < 0.0 {
        return Err(
            ValidationError::new(field, "probability cannot be negative")
                .with_value(p)
                .with_suggestion("Use a value between 0.0 and 1.0"),
        );
    }
    if p > 1.0 {
        return Err(ValidationError::new(field, "probability cannot exceed 1.0")
            .with_value(p)
            .with_suggestion("Use a value between 0.0 and 1.0"));
    }
    Ok(p)
}

/// Validate that a value is a valid probability, clamping to [0.0, 1.0].
///
/// This is a lenient version that clamps instead of returning an error.
/// Useful when you want to accept slightly out-of-range values due to
/// floating point errors.
#[must_use]
pub fn clamp_probability(p: f64) -> f64 {
    if p.is_nan() { 0.0 } else { p.clamp(0.0, 1.0) }
}

/// Validate that weights are non-negative.
///
/// # Errors
///
/// Returns an error if any weight is negative or NaN.
pub fn validate_weights(weights: &[f64], field: &str) -> ValidationResult<()> {
    for (i, &w) in weights.iter().enumerate() {
        if w.is_nan() {
            return Err(
                ValidationError::new(field, format!("weight[{i}] cannot be NaN")).with_value(w),
            );
        }
        if w < 0.0 {
            return Err(
                ValidationError::new(field, format!("weight[{i}] cannot be negative"))
                    .with_value(w)
                    .with_suggestion("Use non-negative weights"),
            );
        }
    }
    Ok(())
}

/// Validate that weights sum to a positive value.
///
/// # Errors
///
/// Returns an error if the sum is zero or negative.
pub fn validate_weights_sum(weights: &[f64], field: &str) -> ValidationResult<()> {
    validate_weights(weights, field)?;
    let sum: f64 = weights.iter().sum();
    if sum <= 0.0 {
        return Err(
            ValidationError::new(field, "weights must sum to a positive value")
                .with_value(sum)
                .with_suggestion("Ensure at least one weight is positive"),
        );
    }
    Ok(())
}

/// Validate a rate (non-negative value).
///
/// # Errors
///
/// Returns an error if the rate is negative or NaN.
pub fn validate_rate(rate: f64, field: &str) -> ValidationResult<f64> {
    if rate.is_nan() {
        return Err(ValidationError::new(field, "rate cannot be NaN").with_value(rate));
    }
    if rate < 0.0 {
        return Err(ValidationError::new(field, "rate cannot be negative")
            .with_value(rate)
            .with_suggestion("Use a non-negative rate"));
    }
    Ok(rate)
}

/// Validate a positive value (strictly greater than zero).
///
/// # Errors
///
/// Returns an error if the value is not positive.
pub fn validate_positive(value: f64, field: &str) -> ValidationResult<f64> {
    if value.is_nan() {
        return Err(ValidationError::new(field, "value cannot be NaN").with_value(value));
    }
    if value <= 0.0 {
        return Err(ValidationError::new(field, "value must be positive")
            .with_value(value)
            .with_suggestion("Use a value greater than 0"));
    }
    Ok(value)
}

/// Check if a probability is effectively zero (within epsilon).
#[must_use]
pub fn is_probability_zero(p: f64) -> bool {
    p.abs() < f64::EPSILON
}

/// Check if a probability is effectively one (within epsilon).
#[must_use]
pub fn is_probability_one(p: f64) -> bool {
    (p - 1.0).abs() < f64::EPSILON
}

/// Warnings for common noise model mistakes.
#[derive(Debug, Clone)]
pub struct NoiseWarning {
    /// Warning message.
    pub message: String,
    /// Context or location.
    pub context: Option<String>,
}

impl fmt::Display for NoiseWarning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Warning: {}", self.message)?;
        if let Some(ref ctx) = self.context {
            write!(f, " (in {ctx})")?;
        }
        Ok(())
    }
}

impl NoiseWarning {
    /// Create a new warning.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            context: None,
        }
    }

    /// Add context to the warning.
    #[must_use]
    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.context = Some(context.into());
        self
    }
}

/// Warn if probability is very high (might be a mistake).
#[must_use]
pub fn warn_high_probability(p: f64, field: &str, threshold: f64) -> Option<NoiseWarning> {
    if p > threshold {
        Some(NoiseWarning::new(format!(
            "{field} probability {p} is unusually high (> {threshold})"
        )))
    } else {
        None
    }
}

/// Warn if probability is very low but non-zero (might be inefficient).
#[must_use]
pub fn warn_low_probability(p: f64, field: &str, threshold: f64) -> Option<NoiseWarning> {
    if p > 0.0 && p < threshold {
        Some(NoiseWarning::new(format!(
            "{field} probability {p} is very low (< {threshold}), consider using 0 if not needed"
        )))
    } else {
        None
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_probability() {
        assert!(validate_probability(0.0, "test").is_ok());
        assert!(validate_probability(0.5, "test").is_ok());
        assert!(validate_probability(1.0, "test").is_ok());

        assert!(validate_probability(-0.1, "test").is_err());
        assert!(validate_probability(1.1, "test").is_err());
        assert!(validate_probability(f64::NAN, "test").is_err());
    }

    #[test]
    fn test_clamp_probability() {
        assert_eq!(clamp_probability(0.5), 0.5);
        assert_eq!(clamp_probability(-0.1), 0.0);
        assert_eq!(clamp_probability(1.5), 1.0);
        assert_eq!(clamp_probability(f64::NAN), 0.0);
    }

    #[test]
    fn test_validate_weights() {
        assert!(validate_weights(&[0.5, 0.3, 0.2], "test").is_ok());
        assert!(validate_weights(&[0.0, 1.0], "test").is_ok());

        assert!(validate_weights(&[-0.1, 0.5], "test").is_err());
        assert!(validate_weights(&[f64::NAN], "test").is_err());
    }

    #[test]
    fn test_validate_weights_sum() {
        assert!(validate_weights_sum(&[0.5, 0.5], "test").is_ok());
        assert!(validate_weights_sum(&[0.0, 0.0], "test").is_err());
    }

    #[test]
    fn test_validation_error_display() {
        let err = ValidationError::new("probability", "value out of range")
            .with_value(1.5)
            .with_suggestion("Use a value between 0 and 1");

        let msg = err.to_string();
        assert!(msg.contains("probability"));
        assert!(msg.contains("1.5"));
        assert!(msg.contains("Use a value"));
    }

    #[test]
    fn test_warn_high_probability() {
        assert!(warn_high_probability(0.5, "error", 0.9).is_none());
        assert!(warn_high_probability(0.95, "error", 0.9).is_some());
    }
}
