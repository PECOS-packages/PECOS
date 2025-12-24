//! High-level Tesseract decoder interface

use super::bridge::ffi;
use cxx::UniquePtr;
use ndarray::{Array1, ArrayView1};
use pecos_decoder_core::{Decoder, DecodingResultTrait};
use std::error::Error;
use std::fmt;

/// Error types for Tesseract operations
#[derive(Debug)]
pub enum TesseractError {
    /// Invalid configuration parameter
    InvalidConfig(String),
    /// Decoder initialization failed
    InitializationFailed(String),
    /// Decoding operation failed
    DecodingFailed(String),
    /// Invalid input data
    InvalidInput(String),
}

impl fmt::Display for TesseractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TesseractError::InvalidConfig(msg) => write!(f, "Invalid configuration: {msg}"),
            TesseractError::InitializationFailed(msg) => {
                write!(f, "Initialization failed: {msg}")
            }
            TesseractError::DecodingFailed(msg) => write!(f, "Decoding failed: {msg}"),
            TesseractError::InvalidInput(msg) => write!(f, "Invalid input: {msg}"),
        }
    }
}

impl Error for TesseractError {}

/// Configuration for Tesseract decoder
#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)]
pub struct TesseractConfig {
    /// Maximum number of detectors to consider in beam search
    pub det_beam: u16,
    /// Enable beam climbing heuristic
    pub beam_climbing: bool,
    /// Avoid revisiting detectors during search
    pub no_revisit_dets: bool,
    /// Limit to at most two errors per detector
    pub at_most_two_errors_per_detector: bool,
    /// Enable verbose output
    pub verbose: bool,
    /// Priority queue size limit
    pub pqlimit: usize,
    /// Detector penalty factor
    pub det_penalty: f64,
}

impl Default for TesseractConfig {
    fn default() -> Self {
        Self {
            det_beam: u16::MAX, // Infinite beam by default
            beam_climbing: false,
            no_revisit_dets: false,
            at_most_two_errors_per_detector: false,
            verbose: false,
            pqlimit: usize::MAX,
            det_penalty: 0.0,
        }
    }
}

impl TesseractConfig {
    /// Create a new configuration with optimized settings for performance
    #[must_use]
    pub fn fast() -> Self {
        Self {
            det_beam: 100,
            beam_climbing: true,
            no_revisit_dets: true,
            at_most_two_errors_per_detector: true,
            verbose: false,
            pqlimit: 1_000_000,
            det_penalty: 0.1,
        }
    }

    /// Create a new configuration with settings optimized for accuracy
    #[must_use]
    pub fn accurate() -> Self {
        Self {
            det_beam: u16::MAX,
            beam_climbing: false,
            no_revisit_dets: false,
            at_most_two_errors_per_detector: false,
            verbose: false,
            pqlimit: usize::MAX,
            det_penalty: 0.0,
        }
    }

    /// Convert to FFI representation
    #[must_use]
    pub fn to_ffi_repr(&self) -> ffi::TesseractConfigRepr {
        ffi::TesseractConfigRepr {
            det_beam: self.det_beam,
            beam_climbing: self.beam_climbing,
            no_revisit_dets: self.no_revisit_dets,
            at_most_two_errors_per_detector: self.at_most_two_errors_per_detector,
            verbose: self.verbose,
            pqlimit: self.pqlimit,
            det_penalty: self.det_penalty,
        }
    }
}

/// Result of a Tesseract decoding operation
#[derive(Debug, Clone)]
pub struct DecodingResult {
    /// Indices of predicted errors
    pub predicted_errors: Array1<usize>,
    /// Observables mask (bitwise XOR of all error observables)
    pub observables_mask: u64,
    /// Total cost of the solution (sum of error likelihood costs)
    pub cost: f64,
    /// Whether this is a low-confidence prediction
    pub low_confidence: bool,
}

impl DecodingResultTrait for DecodingResult {
    fn is_successful(&self) -> bool {
        !self.low_confidence
    }

    fn cost(&self) -> Option<f64> {
        Some(self.cost)
    }
}

/// Tesseract search-based decoder for quantum error correction
///
/// The Tesseract decoder uses A* search with pruning heuristics to find
/// the most likely error configuration consistent with observed syndromes.
/// It's particularly effective for LDPC quantum codes.
pub struct TesseractDecoder {
    inner: UniquePtr<ffi::TesseractDecoderWrapper>,
    config: TesseractConfig,
    num_detectors: usize,
    num_errors: usize,
    num_observables: usize,
}

impl TesseractDecoder {
    /// Create a new Tesseract decoder
    ///
    /// # Arguments
    /// * `dem_string` - Detector Error Model in Stim format
    /// * `config` - Decoder configuration
    ///
    /// # Example
    /// ```rust
    /// # #[cfg(feature = "tesseract")]
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// use pecos_decoders::tesseract::{TesseractDecoder, TesseractConfig};
    ///
    /// let dem = "error(0.1) D0 D1\nerror(0.05) D2 L0";
    /// let config = TesseractConfig::default();
    /// let decoder = TesseractDecoder::new(dem, config)?;
    /// println!("Created decoder with {} detectors", decoder.num_detectors());
    /// # Ok(())
    /// # }
    /// # #[cfg(not(feature = "tesseract"))]
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// #     Ok(()) // No-op when tesseract feature is disabled
    /// # }
    /// # example().unwrap();
    /// ```
    pub fn new(dem_string: &str, config: TesseractConfig) -> Result<Self, TesseractError> {
        let config_repr = config.to_ffi_repr();

        let inner = ffi::create_tesseract_decoder(dem_string, &config_repr)
            .map_err(|e| TesseractError::InitializationFailed(e.what().to_string()))?;

        let num_detectors = ffi::get_num_detectors(&inner);
        let num_errors = ffi::get_num_errors(&inner);
        let num_observables = ffi::get_num_observables(&inner);

        Ok(Self {
            inner,
            config,
            num_detectors,
            num_errors,
            num_observables,
        })
    }

    /// Decode detection events to find the most likely error configuration
    ///
    /// # Arguments
    /// * `detections` - Array of detection event indices
    ///
    /// # Returns
    /// The decoded error configuration and associated metadata
    pub fn decode_detections(
        &mut self,
        detections: &ArrayView1<u64>,
    ) -> Result<DecodingResult, TesseractError> {
        let detections_slice = detections.as_slice().ok_or_else(|| {
            TesseractError::InvalidInput("Detection array is not contiguous".to_string())
        })?;

        let result = ffi::decode_detections(self.inner.pin_mut(), detections_slice)
            .map_err(|e| TesseractError::DecodingFailed(e.what().to_string()))?;

        Ok(DecodingResult {
            predicted_errors: Array1::from_vec(result.predicted_errors),
            observables_mask: result.observables_mask,
            cost: result.cost,
            low_confidence: result.low_confidence,
        })
    }

    /// Decode detection events using a specific detector ordering
    ///
    /// # Arguments
    /// * `detections` - Array of detection event indices
    /// * `det_order` - Index of the detector ordering to use
    ///
    /// # Returns
    /// The decoded error configuration using the specified ordering
    pub fn decode_with_order(
        &mut self,
        detections: &ArrayView1<u64>,
        det_order: usize,
    ) -> Result<DecodingResult, TesseractError> {
        let detections_slice = detections.as_slice().ok_or_else(|| {
            TesseractError::InvalidInput("Detection array is not contiguous".to_string())
        })?;

        let result =
            ffi::decode_detections_with_order(self.inner.pin_mut(), detections_slice, det_order)
                .map_err(|e| TesseractError::DecodingFailed(e.what().to_string()))?;

        Ok(DecodingResult {
            predicted_errors: Array1::from_vec(result.predicted_errors),
            observables_mask: result.observables_mask,
            cost: result.cost,
            low_confidence: result.low_confidence,
        })
    }

    /// Get the observables mask for a set of error indices
    #[must_use]
    pub fn mask_from_errors(&self, error_indices: &[usize]) -> u64 {
        ffi::mask_from_errors(&self.inner, error_indices)
    }

    /// Get the total cost for a set of error indices
    #[must_use]
    pub fn cost_from_errors(&self, error_indices: &[usize]) -> f64 {
        ffi::cost_from_errors(&self.inner, error_indices)
    }

    /// Get information about a specific error
    #[must_use]
    pub fn get_error_info(&self, error_idx: usize) -> Option<ErrorInfo> {
        if error_idx >= self.num_errors {
            return None;
        }

        Some(ErrorInfo {
            probability: ffi::get_error_probability(&self.inner, error_idx),
            cost: ffi::get_error_cost(&self.inner, error_idx),
            detectors: ffi::get_error_detectors(&self.inner, error_idx),
            observables: ffi::get_error_observables(&self.inner, error_idx),
        })
    }

    // Getter methods

    /// Get the number of detectors in the error model
    #[must_use]
    pub fn num_detectors(&self) -> usize {
        self.num_detectors
    }

    /// Get the number of errors in the error model
    #[must_use]
    pub fn num_errors(&self) -> usize {
        self.num_errors
    }

    /// Get the number of observables in the error model
    #[must_use]
    pub fn num_observables(&self) -> usize {
        self.num_observables
    }

    /// Get the decoder configuration
    #[must_use]
    pub fn config(&self) -> &TesseractConfig {
        &self.config
    }

    /// Get the detector beam size
    #[must_use]
    pub fn det_beam(&self) -> u16 {
        ffi::get_det_beam(&self.inner)
    }

    /// Check if beam climbing is enabled
    #[must_use]
    pub fn beam_climbing(&self) -> bool {
        ffi::get_beam_climbing(&self.inner)
    }

    /// Check if detector revisiting is disabled
    #[must_use]
    pub fn no_revisit_dets(&self) -> bool {
        ffi::get_no_revisit_dets(&self.inner)
    }

    /// Check if at-most-two-errors-per-detector is enabled
    #[must_use]
    pub fn at_most_two_errors_per_detector(&self) -> bool {
        ffi::get_at_most_two_errors_per_detector(&self.inner)
    }

    /// Check if verbose mode is enabled
    #[must_use]
    pub fn verbose(&self) -> bool {
        ffi::get_verbose(&self.inner)
    }

    /// Get the priority queue limit
    #[must_use]
    pub fn pqlimit(&self) -> usize {
        ffi::get_pqlimit(&self.inner)
    }

    /// Get the detector penalty factor
    #[must_use]
    pub fn det_penalty(&self) -> f64 {
        ffi::get_det_penalty(&self.inner)
    }
}

impl Decoder for TesseractDecoder {
    type Result = DecodingResult;
    type Error = TesseractError;

    fn decode(&mut self, input: &ArrayView1<u8>) -> Result<Self::Result, Self::Error> {
        // Convert u8 detections to u64 indices
        let detections: Vec<u64> = input
            .iter()
            .enumerate()
            .filter_map(|(i, &val)| if val != 0 { Some(i as u64) } else { None })
            .collect();

        let detections_array = Array1::from_vec(detections);
        let result = self.decode_detections(&detections_array.view())?;

        Ok(result)
    }

    fn check_count(&self) -> usize {
        self.num_detectors
    }

    fn bit_count(&self) -> usize {
        self.num_errors
    }
}

/// Information about a specific error in the error model
#[derive(Debug, Clone)]
pub struct ErrorInfo {
    /// Probability of this error occurring
    pub probability: f64,
    /// Likelihood cost (-log(probability))
    pub cost: f64,
    /// Detector indices affected by this error
    pub detectors: Vec<i32>,
    /// Observable mask for this error
    pub observables: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tesseract_config_default() {
        let config = TesseractConfig::default();
        assert_eq!(config.det_beam, u16::MAX);
        assert!(!config.beam_climbing);
        assert!(!config.verbose);
    }

    #[test]
    fn test_tesseract_config_fast() {
        let config = TesseractConfig::fast();
        assert_eq!(config.det_beam, 100);
        assert!(config.beam_climbing);
        assert!(config.no_revisit_dets);
        assert!(config.at_most_two_errors_per_detector);
    }

    #[test]
    fn test_tesseract_config_accurate() {
        let config = TesseractConfig::accurate();
        assert_eq!(config.det_beam, u16::MAX);
        assert!(!config.beam_climbing);
        assert!(!config.no_revisit_dets);
        assert!(!config.at_most_two_errors_per_detector);
    }
}
