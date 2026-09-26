//! Tesseract's trellis-mode decoder.
//!
//! Upstream Tesseract ships a second decoder next to its A* search: a beam
//! trellis that walks the error mechanisms in model order, keeps at most
//! `beam_width` partial-syndrome states per layer, and sums the probability
//! mass of every surviving explanation instead of returning one min-cost
//! path. The predicted observable is the one with more mass. This module
//! wraps that decoder unchanged so it can serve as an oracle and baseline;
//! the search itself stays upstream's.
//!
//! Upstream limits: at most one observable, at most 64 observables in the
//! DEM's declared width, and an active-detector frontier of at most 256
//! detectors. Each is rejected at construction with upstream's own message.

use super::bridge::ffi;
use super::decoder::{TesseractError, fired_detectors, reject_repeated_detections};
use cxx::UniquePtr;
use ndarray::ArrayView1;
use pecos_decoder_core::{Decoder, DecodingResultTrait};

/// Rule for ranking beam states when truncating a trellis layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TesseractTrellisRankingMode {
    /// Keep the states carrying the most probability mass (upstream default).
    #[default]
    MassOnly,
    /// Discount each state's mass by a detector-cost estimate of the mass
    /// still needed to clear its remaining active detectors.
    FutureDetcostRanked,
    /// Like [`FutureDetcostRanked`](Self::FutureDetcostRanked), but the
    /// estimate only counts detectors that are active in the state.
    FutureActiveDetcostRanked,
}

impl TesseractTrellisRankingMode {
    fn to_ffi_repr(self) -> ffi::TesseractTrellisRankingModeRepr {
        match self {
            Self::MassOnly => ffi::TesseractTrellisRankingModeRepr::MassOnly,
            Self::FutureDetcostRanked => ffi::TesseractTrellisRankingModeRepr::FutureDetcostRanked,
            Self::FutureActiveDetcostRanked => {
                ffi::TesseractTrellisRankingModeRepr::FutureActiveDetcostRanked
            }
        }
    }
}

/// Configuration for Tesseract's trellis-mode decoder. Defaults are
/// upstream's `TesseractTrellisConfig` defaults.
#[derive(Debug, Clone, PartialEq)]
pub struct TesseractTrellisConfig {
    /// Maximum number of partial-syndrome states kept per trellis layer.
    pub beam_width: usize,
    /// After the `beam_width` cut, keep only the highest-scoring states
    /// whose cumulative mass reaches `1 - beam_eps` of the layer's total
    /// mass; zero keeps every state up to `beam_width`. Must be in `[0, 1)`:
    /// at 1 or above the target mass is nonpositive and the beam collapses
    /// to a single state per layer.
    pub beam_eps: f64,
    /// Scale applied to the future detector-cost estimate in the ranked
    /// modes; ignored under [`TesseractTrellisRankingMode::MassOnly`].
    pub future_detcost_scale: f64,
    /// Print per-shot beam statistics to stdout.
    pub verbose: bool,
    /// Merge error mechanisms with identical detector and observable
    /// symptoms before decoding.
    pub merge_errors: bool,
    /// Beam ranking rule.
    pub ranking_mode: TesseractTrellisRankingMode,
}

impl Default for TesseractTrellisConfig {
    fn default() -> Self {
        Self {
            beam_width: 1024,
            beam_eps: 0.0,
            future_detcost_scale: 2.0,
            verbose: false,
            merge_errors: true,
            ranking_mode: TesseractTrellisRankingMode::MassOnly,
        }
    }
}

impl TesseractTrellisConfig {
    /// Validate configuration values before passing them through FFI.
    ///
    /// # Errors
    ///
    /// Returns [`TesseractError::InvalidConfig`] when a numeric tuning
    /// parameter is outside its supported range.
    pub fn validate(&self) -> Result<(), TesseractError> {
        if self.beam_width == 0 {
            return Err(TesseractError::InvalidConfig(
                "beam_width must be greater than 0".to_string(),
            ));
        }
        if !(0.0..1.0).contains(&self.beam_eps) {
            return Err(TesseractError::InvalidConfig(
                "beam_eps must be in [0, 1)".to_string(),
            ));
        }
        if !self.future_detcost_scale.is_finite() || self.future_detcost_scale < 0.0 {
            return Err(TesseractError::InvalidConfig(
                "future_detcost_scale must be finite and non-negative".to_string(),
            ));
        }
        Ok(())
    }

    /// Convert to FFI representation
    #[must_use]
    pub fn to_ffi_repr(&self) -> ffi::TesseractTrellisConfigRepr {
        ffi::TesseractTrellisConfigRepr {
            beam_width: self.beam_width,
            beam_eps: self.beam_eps,
            future_detcost_scale: self.future_detcost_scale,
            verbose: self.verbose,
            merge_errors: self.merge_errors,
            ranking_mode: self.ranking_mode.to_ffi_repr(),
        }
    }
}

/// Result of one trellis-mode decode.
#[derive(Debug, Clone)]
pub struct TesseractTrellisResult {
    /// Predicted observable flips as a bit mask (bit 0 is `L0`).
    pub observables_mask: u64,
    /// Fraction of the surviving probability mass that flips the observable;
    /// NaN when `low_confidence` is set.
    pub observable_probability: f64,
    /// No surviving state explained the syndrome (a fired detector that no
    /// mechanism touches, or the beam emptied), so the prediction is a guess.
    pub low_confidence: bool,
    /// Beam states expanded over the shot.
    pub num_states_expanded: usize,
    /// Beam states kept after truncation, summed over layers.
    pub num_states_merged: usize,
    /// Largest beam kept at any layer.
    pub max_beam_size_seen: usize,
    /// Widest active-detector frontier reached.
    pub max_frontier_width_seen: usize,
}

impl DecodingResultTrait for TesseractTrellisResult {
    fn is_successful(&self) -> bool {
        !self.low_confidence
    }

    fn cost(&self) -> Option<f64> {
        None
    }
}

/// Tesseract's trellis-mode decoder.
pub struct TesseractTrellisDecoder {
    inner: UniquePtr<ffi::TesseractTrellisDecoderWrapper>,
    config: TesseractTrellisConfig,
    num_detectors: usize,
    num_errors: usize,
    num_observables: usize,
}

impl TesseractTrellisDecoder {
    /// Create a trellis-mode decoder from a detector error model.
    ///
    /// # Errors
    ///
    /// Returns [`TesseractError::Dem`] for tokenizer errors,
    /// [`TesseractError::InvalidConfig`] for an out-of-range tuning
    /// parameter and [`TesseractError::InitializationFailed`] when the DEM is
    /// malformed or outside upstream's supported shape (more than one
    /// observable, or a frontier wider than the compiled kernel).
    pub fn new(dem_string: &str, config: TesseractTrellisConfig) -> Result<Self, TesseractError> {
        pecos_decoder_core::dem::grammar::validate_dem_text(dem_string)
            .map_err(TesseractError::Dem)?;
        config.validate()?;
        let inner = ffi::create_tesseract_trellis_decoder(dem_string, &config.to_ffi_repr())
            .map_err(|e| TesseractError::InitializationFailed(e.what().to_string()))?;
        let num_detectors = ffi::trellis_num_detectors(&inner);
        let num_errors = ffi::trellis_num_errors(&inner);
        let num_observables = ffi::trellis_num_observables(&inner);
        Ok(Self {
            inner,
            config,
            num_detectors,
            num_errors,
            num_observables,
        })
    }

    /// Decode sparse detection events (indices of fired detectors).
    ///
    /// A fired detector that no mechanism touches is upstream's
    /// low-confidence case and decodes. A repeated index or a detector
    /// outside the model is an input error, as for the A* decoder.
    ///
    /// # Errors
    ///
    /// Returns [`TesseractError::InvalidInput`] if the detection array is not
    /// contiguous, repeats a detector, or names a detector at or beyond
    /// `num_detectors`, or [`TesseractError::DecodingFailed`] if the C++
    /// decoder fails.
    pub fn decode_detections(
        &mut self,
        detections: &ArrayView1<u64>,
    ) -> Result<TesseractTrellisResult, TesseractError> {
        let detections_slice = detections.as_slice().ok_or_else(|| {
            TesseractError::InvalidInput("Detection array is not contiguous".to_string())
        })?;
        if let Some(&bad) = detections_slice
            .iter()
            .find(|&&d| d >= self.num_detectors as u64)
        {
            return Err(TesseractError::InvalidInput(format!(
                "detector {bad} is out of range for a model with {} detectors",
                self.num_detectors
            )));
        }
        reject_repeated_detections(detections_slice)?;
        let result = ffi::trellis_decode_detections(self.inner.pin_mut(), detections_slice)
            .map_err(|e| TesseractError::DecodingFailed(e.what().to_string()))?;
        Ok(TesseractTrellisResult {
            observables_mask: result.observables_mask,
            observable_probability: result.observable_probability,
            low_confidence: result.low_confidence,
            num_states_expanded: result.num_states_expanded,
            num_states_merged: result.num_states_merged,
            max_beam_size_seen: result.max_beam_size_seen,
            max_frontier_width_seen: result.max_frontier_width_seen,
        })
    }

    /// Number of detectors in the error model.
    #[must_use]
    pub fn num_detectors(&self) -> usize {
        self.num_detectors
    }

    /// Number of error mechanisms in the flattened error model, the same
    /// count the A* wrap reports for the same model.
    #[must_use]
    pub fn num_errors(&self) -> usize {
        self.num_errors
    }

    /// Number of observables in the error model.
    #[must_use]
    pub fn num_observables(&self) -> usize {
        self.num_observables
    }

    /// The decoder configuration.
    #[must_use]
    pub fn config(&self) -> &TesseractTrellisConfig {
        &self.config
    }
}

impl pecos_decoder_core::ObservableDecoder for TesseractTrellisDecoder {
    fn decode_obs(
        &mut self,
        syndrome: &[u8],
    ) -> Result<pecos_decoder_core::obs_mask::ObsMask, pecos_decoder_core::DecoderError> {
        let result = self
            .decode_detections(&fired_detectors(syndrome.iter()).view())
            .map_err(|e| pecos_decoder_core::DecoderError::DecodingFailed(e.to_string()))?;
        Ok(pecos_decoder_core::obs_mask::ObsMask::from_u64(
            result.observables_mask,
        ))
    }
}

impl Decoder for TesseractTrellisDecoder {
    type Result = TesseractTrellisResult;
    type Error = TesseractError;

    fn decode(&mut self, input: &ArrayView1<u8>) -> Result<Self::Result, Self::Error> {
        self.decode_detections(&fired_detectors(input.iter()).view())
    }

    fn check_count(&self) -> usize {
        self.num_detectors
    }

    fn bit_count(&self) -> usize {
        self.num_errors
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dem_validation_precedes_backend_parsing() {
        for text in [
            "error(0.1) D0 D0",
            "error(0.1) D0 D1 ^ D1 D0",
            "repeat 2 {\nerror(0.1) D0 D0\n}",
            "@bad",
        ] {
            let expected = pecos_decoder_core::dem::grammar::validate_dem_text(text).unwrap_err();
            let error = TesseractTrellisDecoder::new(text, TesseractTrellisConfig::default())
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
    fn default_matches_upstream_trellis_config() {
        let config = TesseractTrellisConfig::default();
        assert_eq!(config.beam_width, 1024);
        assert_eq!(config.beam_eps.to_bits(), 0.0_f64.to_bits());
        assert_eq!(config.future_detcost_scale.to_bits(), 2.0_f64.to_bits());
        assert!(!config.verbose);
        assert!(config.merge_errors);
        assert_eq!(config.ranking_mode, TesseractTrellisRankingMode::MassOnly);
    }

    #[test]
    fn validation_names_the_invalid_parameter() {
        for (config, name) in [
            (
                TesseractTrellisConfig {
                    beam_width: 0,
                    ..TesseractTrellisConfig::default()
                },
                "beam_width",
            ),
            (
                TesseractTrellisConfig {
                    beam_eps: -0.5,
                    ..TesseractTrellisConfig::default()
                },
                "beam_eps",
            ),
            (
                TesseractTrellisConfig {
                    beam_eps: f64::NAN,
                    ..TesseractTrellisConfig::default()
                },
                "beam_eps",
            ),
            (
                TesseractTrellisConfig {
                    beam_eps: 1.0,
                    ..TesseractTrellisConfig::default()
                },
                "beam_eps",
            ),
            (
                TesseractTrellisConfig {
                    future_detcost_scale: f64::INFINITY,
                    ..TesseractTrellisConfig::default()
                },
                "future_detcost_scale",
            ),
        ] {
            let message = config.validate().unwrap_err().to_string();
            assert!(message.contains(name), "{message}");
            let message = TesseractTrellisDecoder::new("error(0.1) D0 L0", config)
                .err()
                .unwrap()
                .to_string();
            assert!(message.contains(name), "{message}");
        }
    }
}
