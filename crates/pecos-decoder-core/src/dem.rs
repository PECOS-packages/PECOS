//! Common traits and types for Detector Error Model (DEM) based decoders
//!
//! This module provides standardized interfaces for decoders that work
//! with Stim's detector error model format.

use crate::errors::DecoderError;

/// Trait for decoders that can be constructed from detector error models
pub trait DemDecoder: super::Decoder {
    /// Configuration type for DEM construction
    type DemConfig: Default;

    /// Create decoder from a DEM string
    ///
    /// # Errors
    ///
    /// Returns [`DecoderError`] if:
    /// - The DEM string is malformed or invalid
    /// - The detector/observable indices are out of bounds
    /// - The decoder cannot be constructed from the given DEM
    fn from_dem(dem: &str) -> Result<Self, DecoderError>
    where
        Self: Sized,
    {
        Self::from_dem_with_config(dem, Default::default())
    }

    /// Create decoder from a DEM string with configuration
    ///
    /// # Errors
    ///
    /// Returns [`DecoderError`] if:
    /// - The DEM string is malformed or invalid
    /// - The configuration is invalid
    /// - The decoder cannot be constructed with the given parameters
    fn from_dem_with_config(dem: &str, config: Self::DemConfig) -> Result<Self, DecoderError>
    where
        Self: Sized;

    /// Create decoder from a DEM file
    ///
    /// # Errors
    ///
    /// Returns [`DecoderError`] if:
    /// - The file cannot be read (I/O error)
    /// - The file contents are not valid DEM format
    /// - The decoder cannot be constructed from the DEM
    fn from_dem_file(path: &str) -> Result<Self, DecoderError>
    where
        Self: Sized,
    {
        let dem = std::fs::read_to_string(path).map_err(DecoderError::IoError)?;
        Self::from_dem(&dem)
    }

    /// Create decoder from a DEM file with configuration
    ///
    /// # Errors
    ///
    /// Returns [`DecoderError`] if:
    /// - The file cannot be read (I/O error)
    /// - The file contents are not valid DEM format
    /// - The configuration is invalid
    /// - The decoder cannot be constructed with the given parameters
    fn from_dem_file_with_config(path: &str, config: Self::DemConfig) -> Result<Self, DecoderError>
    where
        Self: Sized,
    {
        let dem = std::fs::read_to_string(path).map_err(DecoderError::IoError)?;
        Self::from_dem_with_config(&dem, config)
    }

    /// Get the number of detectors in the model
    fn detector_count(&self) -> usize;

    /// Get the number of observables in the model
    fn observable_count(&self) -> usize;
}

/// Common configuration for DEM-based decoders
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DemConfig {
    /// Random seed for deterministic behavior
    pub seed: Option<u64>,
    /// Whether to use a compressed representation
    pub compressed: bool,
    /// Custom detector coordinates (if any)
    pub detector_coordinates: Option<Vec<Vec<f64>>>,
    /// Maximum number of errors to consider per detector
    pub max_errors_per_detector: Option<usize>,
}

/// Utilities for working with detector error models
pub mod utils {
    use super::DecoderError;

    /// Parse basic DEM metadata without full parsing
    ///
    /// Returns (`detector_count`, `observable_count`)
    ///
    /// # Errors
    ///
    /// Returns [`DecoderError`] if the DEM format is invalid
    pub fn parse_dem_metadata(dem: &str) -> Result<(usize, usize), DecoderError> {
        let mut max_detector = None;
        let mut observables = std::collections::HashSet::new();

        for line in dem.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.is_empty() {
                continue;
            }

            // Handle commands with probability parameters like "error(0.01)"
            let command = if parts[0].starts_with("error(") {
                "error"
            } else {
                parts[0]
            };

            match command {
                "error" => {
                    // Parse error line for detector and observable indices
                    for part in &parts[1..] {
                        if let Some(d_str) = part.strip_prefix('D') {
                            if let Ok(d) = d_str.parse::<usize>() {
                                max_detector = Some(max_detector.map_or(d, |m: usize| m.max(d)));
                            }
                        } else if let Some(l_str) = part.strip_prefix('L')
                            && let Ok(l) = l_str.parse::<usize>()
                        {
                            observables.insert(l);
                        }
                    }
                }
                "detector" => {
                    // Parse detector declarations
                    for part in &parts[1..] {
                        if let Some(d_str) = part.strip_prefix('D')
                            && let Ok(d) = d_str.parse::<usize>()
                        {
                            max_detector = Some(max_detector.map_or(d, |m: usize| m.max(d)));
                        }
                    }
                }
                _ => {}
            }
        }

        let detector_count = max_detector.map_or(0, |m| m + 1);
        let observable_count = observables.len();

        Ok((detector_count, observable_count))
    }

    /// Validate DEM format
    ///
    /// # Errors
    ///
    /// Returns [`DecoderError`] if:
    /// - The DEM is empty
    /// - The DEM contains invalid commands or syntax
    /// - Detector/observable indices are invalid
    pub fn validate_dem(dem: &str) -> Result<(), DecoderError> {
        if dem.trim().is_empty() {
            return Err(DecoderError::InvalidConfiguration(
                "DEM cannot be empty".to_string(),
            ));
        }

        // Basic validation - check for valid DEM commands
        let valid_commands = ["error", "detector", "logical_observable", "repeat"];

        for line in dem.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let first_word = line.split_whitespace().next().unwrap_or("");
            // Handle commands with probability parameters like "error(0.01)"
            let command = if first_word.starts_with("error(") {
                "error"
            } else {
                first_word
            };
            if !valid_commands.contains(&command) {
                return Err(DecoderError::InvalidConfiguration(format!(
                    "Invalid DEM command: {first_word}"
                )));
            }
        }

        Ok(())
    }
}

/// Information about a detector error model
#[derive(Debug, Clone, PartialEq)]
pub struct DemInfo {
    /// Number of detectors
    pub detector_count: usize,
    /// Number of logical observables
    pub observable_count: usize,
    /// Number of error mechanisms
    pub error_count: usize,
    /// Detector coordinates (if specified)
    pub detector_coordinates: Option<Vec<Vec<f64>>>,
}

/// Builder pattern for DEM configuration
pub struct DemConfigBuilder {
    config: DemConfig,
}

impl DemConfigBuilder {
    /// Create a new builder
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: DemConfig::default(),
        }
    }

    /// Set the random seed
    #[must_use]
    pub fn seed(mut self, seed: u64) -> Self {
        self.config.seed = Some(seed);
        self
    }

    /// Enable compression
    #[must_use]
    pub fn compressed(mut self, compressed: bool) -> Self {
        self.config.compressed = compressed;
        self
    }

    /// Set detector coordinates
    #[must_use]
    pub fn detector_coordinates(mut self, coords: Vec<Vec<f64>>) -> Self {
        self.config.detector_coordinates = Some(coords);
        self
    }

    /// Set maximum errors per detector
    #[must_use]
    pub fn max_errors_per_detector(mut self, max: usize) -> Self {
        self.config.max_errors_per_detector = Some(max);
        self
    }

    /// Build the configuration
    #[must_use]
    pub fn build(self) -> DemConfig {
        self.config
    }
}

impl Default for DemConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dem_validation() {
        let valid_dem = r"
            error(0.01) D0
            error(0.01) D1 D2
            error(0.01) D0 L0
        ";
        assert!(utils::validate_dem(valid_dem).is_ok());

        let invalid_dem = r"
            invalid_command D0
        ";
        assert!(utils::validate_dem(invalid_dem).is_err());
    }

    #[test]
    fn test_dem_metadata_parsing() {
        let dem = r"
            error(0.01) D0
            error(0.01) D1 D2
            error(0.01) D3 L0
            error(0.01) D4 L1
        ";

        let (detectors, observables) = utils::parse_dem_metadata(dem).unwrap();
        assert_eq!(detectors, 5); // D0 through D4
        assert_eq!(observables, 2); // L0 and L1
    }

    #[test]
    fn test_dem_config_builder() {
        let config = DemConfigBuilder::new()
            .seed(42)
            .compressed(true)
            .max_errors_per_detector(2)
            .build();

        assert_eq!(config.seed, Some(42));
        assert!(config.compressed);
        assert_eq!(config.max_errors_per_detector, Some(2));
    }
}
