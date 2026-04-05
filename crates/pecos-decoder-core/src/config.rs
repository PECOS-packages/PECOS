//! Common configuration traits and types for PECOS decoders
//!
//! Standardized configuration patterns that decoders
//! can implement for consistent API design.

use crate::errors::ConfigError;

/// Common configuration patterns shared across decoders
pub trait DecoderConfig {
    /// Number of nodes/vertices in the decoder graph
    fn node_count(&self) -> Option<usize> {
        None
    }

    /// Number of observable outcomes
    fn observable_count(&self) -> usize;

    /// Random seed for deterministic behavior
    fn seed(&self) -> Option<u64> {
        None
    }

    /// Validate the configuration
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] if the configuration is invalid, such as:
    /// - Required fields are missing
    /// - Values are out of acceptable ranges
    /// - Incompatible options are set
    fn validate(&self) -> Result<(), ConfigError> {
        Ok(())
    }
}

/// Performance-related configuration options
pub trait PerformanceConfig {
    /// Maximum number of iterations (for iterative decoders)
    fn max_iterations(&self) -> Option<usize> {
        None
    }

    /// Level of parallelism to use
    fn parallelism(&self) -> Option<usize> {
        None
    }

    /// Enable verbose/debug output
    fn verbose(&self) -> bool {
        false
    }

    /// Memory limit hint (in bytes)
    fn memory_limit(&self) -> Option<usize> {
        None
    }
}

/// Configuration for batch processing
pub trait BatchConfig {
    /// Whether input is bit-packed
    fn bit_packed_input(&self) -> bool {
        false
    }

    /// Whether output should be bit-packed
    fn bit_packed_output(&self) -> bool {
        false
    }

    /// Whether to return weights/costs
    fn return_weights(&self) -> bool {
        true
    }

    /// Batch size hint for optimization
    fn batch_size_hint(&self) -> Option<usize> {
        None
    }
}

/// Standard solver types across different decoders
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SolverType {
    /// Serial/sequential processing
    #[default]
    Serial,
    /// Parallel processing
    Parallel,
    /// Legacy implementation (for compatibility)
    Legacy,
    /// Adaptive (runtime selection)
    Adaptive,
}

impl SolverType {
    /// Check if this solver type supports parallelism
    #[must_use]
    pub fn is_parallel(&self) -> bool {
        matches!(self, SolverType::Parallel | SolverType::Adaptive)
    }
}

/// Common decoding methods/algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodingMethod {
    /// Belief propagation
    BeliefPropagation,
    /// Union-find based methods
    UnionFind,
    /// Minimum weight matching
    MinimumWeight,
    /// Maximum likelihood
    MaximumLikelihood,
    /// Hybrid approach
    Hybrid,
}

/// Configuration builder trait for fluent API
pub trait ConfigBuilder: Sized {
    /// The configuration type being built
    type Config;

    /// Build the configuration
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] if:
    /// - Required fields are missing
    /// - Any configuration values are invalid
    /// - The combination of settings is incompatible
    fn build(self) -> Result<Self::Config, ConfigError>;

    /// Set the random seed
    #[must_use]
    fn with_seed(self, seed: u64) -> Self;

    /// Set the number of observables
    #[must_use]
    fn with_observables(self, count: usize) -> Self;
}

/// Utility functions for configuration validation
pub mod validation {
    use super::ConfigError;

    /// Validate that a value is within range
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError::OutOfRange`] if the value is not within [min, max]
    pub fn validate_range<T: PartialOrd + std::fmt::Display>(
        value: &T,
        min: &T,
        max: &T,
        field_name: &str,
    ) -> Result<(), ConfigError> {
        if value < min || value > max {
            return Err(ConfigError::OutOfRange {
                field: field_name.to_string(),
                value: value.to_string(),
                min: min.to_string(),
                max: max.to_string(),
            });
        }
        Ok(())
    }

    /// Validate that a required field is present
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError::MissingField`] if the value is None
    pub fn validate_required<T>(value: Option<T>, field_name: &str) -> Result<T, ConfigError> {
        value.ok_or_else(|| ConfigError::MissingField(field_name.to_string()))
    }

    /// Validate probability values (0.0 to 1.0)
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError::OutOfRange`] if the value is not in [0.0, 1.0]
    pub fn validate_probability(value: f64, field_name: &str) -> Result<(), ConfigError> {
        if !(0.0..=1.0).contains(&value) {
            return Err(ConfigError::OutOfRange {
                field: field_name.to_string(),
                value: value.to_string(),
                min: "0.0".to_string(),
                max: "1.0".to_string(),
            });
        }
        Ok(())
    }

    /// Validate positive integer
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError::InvalidValue`] if the value is zero
    pub fn validate_positive(value: usize, field_name: &str) -> Result<(), ConfigError> {
        if value == 0 {
            return Err(ConfigError::InvalidValue {
                field: field_name.to_string(),
                value: value.to_string(),
            });
        }
        Ok(())
    }
}
