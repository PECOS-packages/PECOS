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
    /// Shared DEM or syndrome validation failed
    Dem(pecos_decoder_core::DecoderError),
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
            TesseractError::Dem(error) => error.fmt(f),
            TesseractError::InvalidConfig(msg) => write!(f, "Invalid configuration: {msg}"),
            TesseractError::InitializationFailed(msg) => {
                write!(f, "Initialization failed: {msg}")
            }
            TesseractError::DecodingFailed(msg) => write!(f, "Decoding failed: {msg}"),
            TesseractError::InvalidInput(msg) => write!(f, "Invalid input: {msg}"),
        }
    }
}

impl Error for TesseractError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Dem(error) => Some(error),
            _ => None,
        }
    }
}

impl From<TesseractError> for pecos_decoder_core::DecoderError {
    fn from(error: TesseractError) -> Self {
        match error {
            TesseractError::Dem(error) => error,
            TesseractError::InvalidConfig(message) | TesseractError::InvalidInput(message) => {
                Self::InvalidConfiguration(message)
            }
            error => Self::InternalError(error.to_string()),
        }
    }
}

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
    /// Enable verbose output
    pub verbose: bool,
    /// Merge error mechanisms with identical detector and observable
    /// symptoms before decoding (upstream default)
    pub merge_errors: bool,
    /// Priority queue size limit
    pub pqlimit: usize,
    /// Detector penalty factor
    pub det_penalty: f64,
}

/// Upstream Tesseract's `DEFAULT_DET_BEAM`. An unbounded beam is a separate
/// opt-in constant upstream (`INF_DET_BEAM`); using it by default saturates
/// `pqlimit` on circuit-scale detector error models, which returns a truncated
/// (wrong) answer slowly: measured on a 936-detector BB144 model, an unbounded
/// beam took ~14 s/shot and failed 16 of 60 shots, versus ~0.29 s/shot and 2 of
/// 1000 with this beam.
const DEFAULT_DET_BEAM: u16 = 5;

/// Upstream Tesseract's `INF_DET_BEAM`: search without a detector beam bound.
const INFINITE_DET_BEAM: u16 = u16::MAX;

impl Default for TesseractConfig {
    fn default() -> Self {
        Self {
            det_beam: DEFAULT_DET_BEAM,
            beam_climbing: false,
            no_revisit_dets: true,
            verbose: false,
            merge_errors: true,
            pqlimit: 200_000,
            det_penalty: 0.0,
        }
    }
}

impl TesseractConfig {
    /// Validate configuration values before passing them through FFI.
    ///
    /// # Errors
    ///
    /// Returns [`TesseractError::InvalidConfig`] when a numeric tuning parameter
    /// is outside its supported range.
    pub fn validate(&self) -> Result<(), TesseractError> {
        if self.det_beam == 0 {
            return Err(TesseractError::InvalidConfig(
                "det_beam must be greater than 0".to_string(),
            ));
        }
        if self.pqlimit == 0 {
            return Err(TesseractError::InvalidConfig(
                "pqlimit must be greater than 0".to_string(),
            ));
        }
        if !self.det_penalty.is_finite() || self.det_penalty < 0.0 {
            return Err(TesseractError::InvalidConfig(
                "det_penalty must be finite and non-negative".to_string(),
            ));
        }
        Ok(())
    }

    /// Create a new configuration with optimized settings for performance
    #[must_use]
    pub fn fast() -> Self {
        Self {
            det_beam: DEFAULT_DET_BEAM,
            beam_climbing: true,
            no_revisit_dets: true,
            verbose: false,
            merge_errors: true,
            pqlimit: 200_000,
            det_penalty: 0.1,
        }
    }

    /// Create a new configuration with settings optimized for accuracy
    #[must_use]
    pub fn accurate() -> Self {
        Self {
            det_beam: INFINITE_DET_BEAM,
            beam_climbing: false,
            no_revisit_dets: false,
            verbose: false,
            merge_errors: true,
            pqlimit: 1_000_000,
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
            verbose: self.verbose,
            merge_errors: self.merge_errors,
            pqlimit: self.pqlimit,
            det_penalty: self.det_penalty,
        }
    }
}

/// Result of a Tesseract decoding operation
#[derive(Debug, Clone)]
pub struct DecodingResult {
    /// Indices into the flattened detector error model of the predicted
    /// error mechanisms; valid arguments for
    /// [`TesseractDecoder::get_error_info`] and friends
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
    ///
    /// # Errors
    ///
    /// Returns [`TesseractError::Dem`] for tokenizer errors.
    /// Returns [`TesseractError::InitializationFailed`] if:
    /// - The backend rejects the DEM structure
    /// - The DEM contains unsupported error mechanisms
    /// - Memory allocation fails
    pub fn new(dem_string: &str, config: TesseractConfig) -> Result<Self, TesseractError> {
        pecos_decoder_core::dem::grammar::validate_dem_text(dem_string)
            .map_err(TesseractError::Dem)?;
        config.validate()?;
        let config_repr = config.to_ffi_repr();

        let inner = ffi::create_tesseract_decoder(dem_string, &config_repr)
            .map_err(|e| TesseractError::InitializationFailed(e.what().to_string()))?;

        let num_detectors = ffi::get_num_detectors(&inner);
        let num_errors = ffi::get_num_errors(&inner);
        let num_observables = ffi::get_num_observables(&inner);

        // Tesseract reports its predicted observables as a u64 mask
        // (`DecodingResult::observables_mask`), so it supports at most 64
        // observables. Reject wider DEMs as an error rather than silently
        // truncating observables 64.. into the u64.
        if num_observables > 64 {
            return Err(TesseractError::InvalidConfig(format!(
                "this matching decoder packs observables into a u64 and supports at most 64 \
                 observables, but the DEM has {num_observables}; use the 'pymatching' decoder \
                 or LogicalSubgraphDecoder for wider observable sets"
            )));
        }

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
    ///
    /// # Errors
    ///
    /// Returns [`TesseractError::Dem`] carrying
    /// [`DecoderError::InvalidSyndrome`](pecos_decoder_core::DecoderError::InvalidSyndrome)
    /// if the detection array is not contiguous, repeats a detector, or names a
    /// detector at or beyond `num_detectors`, or [`TesseractError::DecodingFailed`]
    /// if the C++ decoder fails.
    pub fn decode_detections(
        &mut self,
        detections: &ArrayView1<u64>,
    ) -> Result<DecodingResult, TesseractError> {
        let detections_slice = checked_detections(detections, self.num_detectors)?;

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
    ///
    /// # Errors
    ///
    /// Same conditions as [`decode_detections`](Self::decode_detections).
    pub fn decode_with_order(
        &mut self,
        detections: &ArrayView1<u64>,
        det_order: usize,
    ) -> Result<DecodingResult, TesseractError> {
        let detections_slice = checked_detections(detections, self.num_detectors)?;

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

    /// Get the combined observables mask for a set of flattened-DEM error indices
    ///
    /// # Errors
    ///
    /// Returns [`TesseractError::InvalidInput`] if an index is out of range
    /// or names a mechanism the decoder does not retain (merged into an
    /// identical mechanism, or removed for zero probability).
    pub fn mask_from_errors(&self, error_indices: &[usize]) -> Result<u64, TesseractError> {
        ffi::mask_from_errors(&self.inner, error_indices)
            .map_err(|e| TesseractError::InvalidInput(e.what().to_string()))
    }

    /// Get the total likelihood cost for a set of flattened-DEM error indices
    ///
    /// # Errors
    ///
    /// Same conditions as [`mask_from_errors`](Self::mask_from_errors).
    pub fn cost_from_errors(&self, error_indices: &[usize]) -> Result<f64, TesseractError> {
        ffi::cost_from_errors(&self.inner, error_indices)
            .map_err(|e| TesseractError::InvalidInput(e.what().to_string()))
    }

    /// Get information about a flattened-DEM error mechanism
    ///
    /// # Errors
    ///
    /// Same conditions as [`mask_from_errors`](Self::mask_from_errors).
    pub fn get_error_info(&self, dem_error_idx: usize) -> Result<ErrorInfo, TesseractError> {
        let invalid = |e: cxx::Exception| TesseractError::InvalidInput(e.what().to_string());
        Ok(ErrorInfo {
            probability: ffi::get_error_probability(&self.inner, dem_error_idx).map_err(invalid)?,
            cost: ffi::get_error_cost(&self.inner, dem_error_idx).map_err(invalid)?,
            detectors: ffi::get_error_detectors(&self.inner, dem_error_idx).map_err(invalid)?,
            observables: ffi::get_error_observables(&self.inner, dem_error_idx).map_err(invalid)?,
        })
    }

    // Getter methods

    /// Get the number of detectors in the error model
    #[must_use]
    pub fn num_detectors(&self) -> usize {
        self.num_detectors
    }

    /// Get the number of error mechanisms in the flattened error model
    ///
    /// Merged and zero-probability mechanisms keep their index, so this is
    /// the index space of [`DecodingResult::predicted_errors`] and of
    /// [`get_error_info`](Self::get_error_info).
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

    /// Check if verbose mode is enabled
    #[must_use]
    pub fn verbose(&self) -> bool {
        ffi::get_verbose(&self.inner)
    }

    /// Check if indistinguishable error mechanisms were merged
    #[must_use]
    pub fn merge_errors(&self) -> bool {
        ffi::get_merge_errors(&self.inner)
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

/// Borrow a sparse detection array after checking it names fired detectors.
///
/// Sparse detections name a set of fired detectors. Upstream's A* sets each
/// detector bit once but counts every occurrence in its per-error detector
/// tallies, and the trellis kernel XORs each occurrence, so a repeated index
/// silently changes the answer in both; reject it at the boundary, along with
/// any index outside the model.
///
/// # Errors
///
/// Returns an [`invalid_detections`] error if the array is not contiguous,
/// names a detector at or beyond `num_detectors`, or repeats a detector.
pub(crate) fn checked_detections<'a>(
    detections: &'a ArrayView1<u64>,
    num_detectors: usize,
) -> Result<&'a [u64], TesseractError> {
    let detections = detections
        .as_slice()
        .ok_or_else(|| invalid_detections("Detection array is not contiguous".to_string()))?;
    if let Some(&bad) = detections.iter().find(|&&d| d >= num_detectors as u64) {
        return Err(invalid_detections(format!(
            "detector {bad} is out of range for a model with {num_detectors} detectors"
        )));
    }
    let mut sorted = detections.to_vec();
    sorted.sort_unstable();
    if let Some([detector, ..]) = sorted.windows(2).find(|pair| pair[0] == pair[1]) {
        return Err(invalid_detections(format!(
            "detector {detector} is repeated; detections must name each fired detector once"
        )));
    }
    Ok(detections)
}

/// A caller-supplied detection array that is not a valid syndrome.
pub(crate) fn invalid_detections(message: String) -> TesseractError {
    TesseractError::Dem(pecos_decoder_core::DecoderError::InvalidSyndrome(message))
}

/// Validate the dense dimension before converting a syndrome to sparse detections.
pub(crate) fn validate_syndrome_length(
    actual: usize,
    expected: usize,
) -> Result<(), pecos_decoder_core::DecoderError> {
    if actual != expected {
        return Err(pecos_decoder_core::DecoderError::InvalidDimensions { expected, actual });
    }
    Ok(())
}

/// Indices of the nonzero entries of a dense syndrome, as sparse detections.
pub(crate) fn fired_detectors<'a>(syndrome: impl IntoIterator<Item = &'a u8>) -> Array1<u64> {
    Array1::from_iter(
        syndrome
            .into_iter()
            .enumerate()
            .filter(|(_, val)| **val != 0)
            .map(|(i, _)| i as u64),
    )
}

impl pecos_decoder_core::ObservableDecoder for TesseractDecoder {
    fn decode_obs(
        &mut self,
        syndrome: &[u8],
    ) -> Result<pecos_decoder_core::obs_mask::ObsMask, pecos_decoder_core::DecoderError> {
        validate_syndrome_length(syndrome.len(), self.num_detectors)?;
        let result = self
            .decode_detections(&fired_detectors(syndrome.iter()).view())
            .map_err(pecos_decoder_core::DecoderError::from)?;
        Ok(pecos_decoder_core::obs_mask::ObsMask::from_u64(
            result.observables_mask,
        ))
    }
}

impl Decoder for TesseractDecoder {
    type Result = DecodingResult;
    type Error = TesseractError;

    fn decode(&mut self, input: &ArrayView1<u8>) -> Result<Self::Result, Self::Error> {
        validate_syndrome_length(input.len(), self.num_detectors).map_err(TesseractError::Dem)?;
        self.decode_detections(&fired_detectors(input.iter()).view())
    }

    fn check_count(&self) -> usize {
        self.num_detectors
    }

    fn bit_count(&self) -> usize {
        self.num_errors
    }
}

/// Information about a retained error mechanism, looked up by flattened-DEM index
#[derive(Debug, Clone)]
pub struct ErrorInfo {
    /// Probability of this error occurring (merged when identical mechanisms were combined)
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
    use pecos_decoder_core::{DecoderError, ObservableDecoder};

    const TWO_DETECTOR_DEM: &str = "error(0.1) D0 L0\nerror(0.1) D0 D1\nerror(0.1) D1";

    #[test]
    fn observable_decode_rejects_wrong_lengths() {
        let mut decoder = TesseractDecoder::new(TWO_DETECTOR_DEM, TesseractConfig::default())
            .expect("valid two-detector DEM");
        for length in [0, 1, 3] {
            let syndrome = vec![0; length];
            assert!(matches!(
                decoder.decode_obs(&syndrome),
                Err(DecoderError::InvalidDimensions { expected: 2, actual }) if actual == length
            ));
        }
        assert!(decoder.decode_obs(&[1, 0]).is_ok());
    }

    #[test]
    fn sparse_decode_rejects_out_of_range_detector_before_backend() {
        let mut decoder = TesseractDecoder::new(TWO_DETECTOR_DEM, TesseractConfig::default())
            .expect("valid two-detector DEM");
        let detections = Array1::from_vec(vec![0_u64, 2]);
        let expected = "detector 2 is out of range for a model with 2 detectors";
        assert!(matches!(
            decoder.decode_detections(&detections.view()),
            Err(TesseractError::Dem(DecoderError::InvalidSyndrome(message))) if message == expected
        ));
        assert!(matches!(
            decoder.decode_with_order(&detections.view(), 0),
            Err(TesseractError::Dem(DecoderError::InvalidSyndrome(message))) if message == expected
        ));
        assert!(
            decoder
                .decode_detections(&Array1::from_vec(vec![1_u64]).view())
                .is_ok()
        );
    }

    #[test]
    fn dense_decode_rejects_wrong_lengths() {
        let mut decoder = TesseractDecoder::new(TWO_DETECTOR_DEM, TesseractConfig::default())
            .expect("valid two-detector DEM");
        for length in [0, 1, 3] {
            let syndrome = Array1::<u8>::zeros(length);
            assert!(matches!(
                Decoder::decode(&mut decoder, &syndrome.view()),
                Err(TesseractError::Dem(DecoderError::InvalidDimensions { expected: 2, actual }))
                    if actual == length
            ));
        }
        assert!(Decoder::decode(&mut decoder, &Array1::from_vec(vec![1, 0]).view()).is_ok());
    }

    #[test]
    fn dem_validation_precedes_backend_parsing() {
        for text in [
            "error(0.1) D0 D0",
            "error(0.1) D0 D1 ^ D1 D0",
            "repeat 2 {\nerror(0.1) D0 D0\n}",
            "@bad",
        ] {
            let expected = pecos_decoder_core::dem::grammar::validate_dem_text(text).unwrap_err();
            let error = TesseractDecoder::new(text, TesseractConfig::default())
                .err()
                .unwrap();
            assert!(matches!(
                error,
                TesseractError::Dem(pecos_decoder_core::DecoderError::InvalidDemSyntax(_))
            ));
            assert_eq!(error.to_string(), expected.to_string());
        }
    }

    #[test]
    fn test_tesseract_config_default() {
        let config = TesseractConfig::default();
        // Must match upstream Tesseract's DEFAULT_DET_BEAM. An unbounded beam
        // here saturates pqlimit on circuit-scale models and returns a
        // truncated answer slowly, so this value is load-bearing, not cosmetic.
        assert_eq!(config.det_beam, DEFAULT_DET_BEAM);
        assert_ne!(config.det_beam, INFINITE_DET_BEAM);
        assert!(!config.beam_climbing);
        assert!(!config.verbose);
        assert!(config.merge_errors);
    }

    #[test]
    fn test_tesseract_config_fast() {
        let config = TesseractConfig::fast();
        assert_eq!(config.det_beam, 5);
        assert!(config.beam_climbing);
        assert!(config.no_revisit_dets);
    }

    #[test]
    fn test_tesseract_config_accurate() {
        let config = TesseractConfig::accurate();
        assert_eq!(config.det_beam, INFINITE_DET_BEAM);
        assert!(!config.beam_climbing);
        assert!(!config.no_revisit_dets);
    }

    #[test]
    fn test_tesseract_config_validation_names_invalid_parameter() {
        let mut config = TesseractConfig {
            det_beam: 0,
            ..TesseractConfig::default()
        };
        assert!(
            config
                .validate()
                .unwrap_err()
                .to_string()
                .contains("det_beam")
        );

        config = TesseractConfig {
            pqlimit: 0,
            ..TesseractConfig::default()
        };
        assert!(
            config
                .validate()
                .unwrap_err()
                .to_string()
                .contains("pqlimit")
        );

        config = TesseractConfig {
            det_penalty: f64::NAN,
            ..TesseractConfig::default()
        };
        assert!(
            config
                .validate()
                .unwrap_err()
                .to_string()
                .contains("det_penalty")
        );

        config = TesseractConfig {
            det_penalty: -0.1,
            ..TesseractConfig::default()
        };
        assert!(
            config
                .validate()
                .unwrap_err()
                .to_string()
                .contains("det_penalty")
        );

        config = TesseractConfig {
            pqlimit: 0,
            ..TesseractConfig::default()
        };
        let error = TesseractDecoder::new("error(0.1) D0\ndetector D0", config)
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains("pqlimit"));
    }
}
