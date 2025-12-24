//! High-level Chromobius decoder interface

use super::bridge::ffi;
use cxx::UniquePtr;
use ndarray::ArrayView1;
use pecos_decoder_core::{Decoder, DecodingResultTrait};
use std::error::Error;
use std::fmt;

/// Error types for Chromobius operations
#[derive(Debug)]
pub enum ChromobiusError {
    /// Invalid configuration parameter
    InvalidConfig(String),
    /// Decoder initialization failed
    InitializationFailed(String),
    /// Decoding operation failed
    DecodingFailed(String),
    /// Invalid input data
    InvalidInput(String),
}

impl fmt::Display for ChromobiusError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ChromobiusError::InvalidConfig(msg) => write!(f, "Invalid configuration: {msg}"),
            ChromobiusError::InitializationFailed(msg) => {
                write!(f, "Initialization failed: {msg}")
            }
            ChromobiusError::DecodingFailed(msg) => write!(f, "Decoding failed: {msg}"),
            ChromobiusError::InvalidInput(msg) => write!(f, "Invalid input: {msg}"),
        }
    }
}

impl Error for ChromobiusError {}

/// Configuration for Chromobius decoder
#[derive(Debug, Clone, Copy)]
pub struct ChromobiusConfig {
    /// Controls whether or not errors that required the introduction of a
    /// remnant atomic error in order to decompose should be discarded or not.
    pub drop_mobius_errors_involving_remnant_errors: bool,
}

impl Default for ChromobiusConfig {
    fn default() -> Self {
        Self {
            drop_mobius_errors_involving_remnant_errors: true,
        }
    }
}

/// Result of a Chromobius decoding operation
#[derive(Debug, Clone)]
pub struct DecodingResult {
    /// Observables mask (bitwise representation of flipped observables)
    pub observables: u64,
    /// Weight of the solution (if requested)
    pub weight: Option<f32>,
}

impl DecodingResultTrait for DecodingResult {
    fn is_successful(&self) -> bool {
        // Chromobius doesn't have a low-confidence flag like Tesseract
        true
    }

    fn cost(&self) -> Option<f64> {
        self.weight.map(f64::from)
    }
}

/// Chromobius color code decoder
///
/// Chromobius is a mobius decoder that approximates the color code decoding
/// problem as a minimum weight matching problem, using `PyMatching` internally.
pub struct ChromobiusDecoder {
    inner: UniquePtr<ffi::ChromobiusDecoderWrapper>,
    num_detectors: usize,
    num_observables: usize,
}

impl ChromobiusDecoder {
    /// Create a new Chromobius decoder
    ///
    /// # Arguments
    /// * `dem_string` - Detector Error Model in Stim format with color/basis annotations
    /// * `config` - Decoder configuration
    ///
    /// # Example
    /// ```rust
    /// # #[cfg(feature = "chromobius")]
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// use pecos_decoders::chromobius::{ChromobiusDecoder, ChromobiusConfig};
    ///
    /// // DEM with color/basis annotations in 4th coordinate
    /// // 0: basis=X, color=R
    /// // 1: basis=X, color=G
    /// // 2: basis=X, color=B
    /// // 3: basis=Z, color=R
    /// // 4: basis=Z, color=G
    /// // 5: basis=Z, color=B
    /// let dem = r#"
    /// error(0.1) D0 D1
    /// error(0.1) D1 D2 L0
    /// detector(0, 0, 0, 0) D0
    /// detector(1, 0, 0, 1) D1
    /// detector(2, 0, 0, 2) D2
    /// "#.trim();
    /// let config = ChromobiusConfig::default();
    /// let decoder = ChromobiusDecoder::new(dem, config)?;
    /// println!("Created decoder with {} detectors", decoder.num_detectors());
    /// # Ok(())
    /// # }
    /// # #[cfg(not(feature = "chromobius"))]
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// #     Ok(()) // No-op when chromobius feature is disabled
    /// # }
    /// # example().unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`ChromobiusError::InitializationFailed`] if:
    /// - The DEM string is malformed
    /// - The DEM contains unsupported error mechanisms
    /// - Memory allocation fails
    pub fn new(dem_string: &str, config: ChromobiusConfig) -> Result<Self, ChromobiusError> {
        let inner = ffi::create_chromobius_decoder(
            dem_string,
            config.drop_mobius_errors_involving_remnant_errors,
        )
        .map_err(|e| ChromobiusError::InitializationFailed(e.what().to_string()))?;

        let num_detectors = ffi::chromobius_get_num_detectors(&inner);
        let num_observables = ffi::chromobius_get_num_observables(&inner);

        Ok(Self {
            inner,
            num_detectors,
            num_observables,
        })
    }

    /// Decode detection events to find the flipped observables
    ///
    /// # Arguments
    /// * `detection_events` - Bit-packed detection events
    ///
    /// # Returns
    /// The decoded observables mask
    ///
    /// # Errors
    ///
    /// Returns [`ChromobiusError::DecodingFailed`] if decoding fails.
    pub fn decode_detection_events(
        &mut self,
        detection_events: &[u8],
    ) -> Result<DecodingResult, ChromobiusError> {
        let observables = ffi::decode_detection_events(self.inner.pin_mut(), detection_events)
            .map_err(|e| ChromobiusError::DecodingFailed(e.what().to_string()))?;

        Ok(DecodingResult {
            observables,
            weight: None,
        })
    }

    /// Decode detection events and get the weight of the solution
    ///
    /// # Arguments
    /// * `detection_events` - Bit-packed detection events
    ///
    /// # Returns
    /// The decoded observables mask and weight
    ///
    /// # Errors
    ///
    /// Returns [`ChromobiusError::DecodingFailed`] if decoding fails.
    pub fn decode_detection_events_with_weight(
        &mut self,
        detection_events: &[u8],
    ) -> Result<DecodingResult, ChromobiusError> {
        let mut weight = 0.0f32;
        let observables = ffi::decode_detection_events_with_weight(
            self.inner.pin_mut(),
            detection_events,
            &mut weight,
        )
        .map_err(|e| ChromobiusError::DecodingFailed(e.what().to_string()))?;

        Ok(DecodingResult {
            observables,
            weight: Some(weight),
        })
    }

    /// Get the number of detectors in the error model
    #[must_use]
    pub fn num_detectors(&self) -> usize {
        self.num_detectors
    }

    /// Get the number of observables in the error model
    #[must_use]
    pub fn num_observables(&self) -> usize {
        self.num_observables
    }
}

impl Decoder for ChromobiusDecoder {
    type Result = DecodingResult;
    type Error = ChromobiusError;

    fn decode(&mut self, input: &ArrayView1<u8>) -> Result<Self::Result, Self::Error> {
        // Chromobius expects bit-packed detection events
        let detection_events = input.as_slice().ok_or_else(|| {
            ChromobiusError::InvalidInput("Input array is not contiguous".to_string())
        })?;

        let result = self.decode_detection_events(detection_events)?;

        Ok(result)
    }

    fn check_count(&self) -> usize {
        self.num_detectors
    }

    fn bit_count(&self) -> usize {
        // For Chromobius, this would be the number of possible error locations
        // But it's not directly exposed, so we return detectors as a proxy
        self.num_detectors
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chromobius_config_default() {
        let config = ChromobiusConfig::default();
        assert!(config.drop_mobius_errors_involving_remnant_errors);
    }
}
