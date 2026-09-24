//! FFI bridge to Tesseract C++ library
//!
//! Low-level FFI bindings to the Tesseract C++ library.
//! Users should prefer the high-level [`TesseractDecoder`](crate::TesseractDecoder)
//! and [`TesseractTrellisDecoder`](crate::TesseractTrellisDecoder) APIs.

#[cxx::bridge]
pub(crate) mod ffi {
    // Struct representations for C++ interop
    #[derive(Debug)]
    pub struct TesseractConfigRepr {
        pub det_beam: u16,
        pub beam_climbing: bool,
        pub no_revisit_dets: bool,
        pub verbose: bool,
        pub merge_errors: bool,
        pub pqlimit: usize,
        pub det_penalty: f64,
    }

    #[derive(Debug)]
    pub struct DecodingResultRepr {
        pub predicted_errors: Vec<usize>,
        pub observables_mask: u64,
        pub cost: f64,
        pub low_confidence: bool,
    }

    /// Beam-ranking rule of the trellis decoder (upstream
    /// `TesseractTrellisRankingMode`, same variant order).
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    #[repr(u8)]
    pub enum TesseractTrellisRankingModeRepr {
        MassOnly = 0,
        FutureDetcostRanked = 1,
        FutureActiveDetcostRanked = 2,
    }

    #[derive(Debug)]
    pub struct TesseractTrellisConfigRepr {
        pub beam_width: usize,
        pub beam_eps: f64,
        pub future_detcost_scale: f64,
        pub verbose: bool,
        pub merge_errors: bool,
        pub ranking_mode: TesseractTrellisRankingModeRepr,
    }

    #[derive(Debug)]
    pub struct TesseractTrellisResultRepr {
        pub observables_mask: u64,
        pub observable_probability: f64,
        pub low_confidence: bool,
        pub num_states_expanded: usize,
        pub num_states_merged: usize,
        pub max_beam_size_seen: usize,
        pub max_frontier_width_seen: usize,
    }

    unsafe extern "C++" {
        include!("tesseract_bridge.h");

        type TesseractDecoderWrapper;

        /// Create a Tesseract decoder from a detector error model string.
        ///
        /// # Errors
        ///
        /// Returns a CXX exception if the DEM string is malformed or
        /// memory allocation fails.
        fn create_tesseract_decoder(
            dem_string: &str,
            config: &TesseractConfigRepr,
        ) -> Result<UniquePtr<TesseractDecoderWrapper>>;

        /// Decode detection events to find the most likely error configuration.
        ///
        /// # Errors
        ///
        /// Returns a CXX exception if decoding fails.
        fn decode_detections(
            decoder: Pin<&mut TesseractDecoderWrapper>,
            detections: &[u64],
        ) -> Result<DecodingResultRepr>;

        /// Decode detection events using a specific detector ordering.
        ///
        /// # Errors
        ///
        /// Returns a CXX exception if decoding fails.
        fn decode_detections_with_order(
            decoder: Pin<&mut TesseractDecoderWrapper>,
            detections: &[u64],
            det_order: usize,
        ) -> Result<DecodingResultRepr>;

        /// Get the number of detectors in the error model.
        fn get_num_detectors(decoder: &TesseractDecoderWrapper) -> usize;

        /// Get the number of error mechanisms in the flattened error model.
        /// This is the index space of predicted errors and of every
        /// per-error accessor below.
        fn get_num_errors(decoder: &TesseractDecoderWrapper) -> usize;

        /// Get the number of observables in the error model.
        fn get_num_observables(decoder: &TesseractDecoderWrapper) -> usize;

        /// Get the detector beam size.
        fn get_det_beam(decoder: &TesseractDecoderWrapper) -> u16;

        /// Check if beam climbing is enabled.
        fn get_beam_climbing(decoder: &TesseractDecoderWrapper) -> bool;

        /// Check if detector revisiting is disabled.
        fn get_no_revisit_dets(decoder: &TesseractDecoderWrapper) -> bool;

        /// Check if verbose mode is enabled.
        fn get_verbose(decoder: &TesseractDecoderWrapper) -> bool;

        /// Check if indistinguishable error mechanisms were merged.
        fn get_merge_errors(decoder: &TesseractDecoderWrapper) -> bool;

        /// Get the priority queue limit.
        fn get_pqlimit(decoder: &TesseractDecoderWrapper) -> usize;

        /// Get the detector penalty factor.
        fn get_det_penalty(decoder: &TesseractDecoderWrapper) -> f64;

        /// Get the probability of a flattened-DEM error.
        ///
        /// # Errors
        ///
        /// Returns a CXX exception if the index is out of range or the
        /// mechanism was merged away or removed for zero probability.
        fn get_error_probability(
            decoder: &TesseractDecoderWrapper,
            dem_error_idx: usize,
        ) -> Result<f64>;

        /// Get the likelihood cost of a flattened-DEM error.
        ///
        /// # Errors
        ///
        /// Same conditions as `get_error_probability`.
        fn get_error_cost(decoder: &TesseractDecoderWrapper, dem_error_idx: usize) -> Result<f64>;

        /// Get the detectors affected by a flattened-DEM error.
        ///
        /// # Errors
        ///
        /// Same conditions as `get_error_probability`.
        fn get_error_detectors(
            decoder: &TesseractDecoderWrapper,
            dem_error_idx: usize,
        ) -> Result<Vec<i32>>;

        /// Get the observables mask for a flattened-DEM error.
        ///
        /// # Errors
        ///
        /// Same conditions as `get_error_probability`.
        fn get_error_observables(
            decoder: &TesseractDecoderWrapper,
            dem_error_idx: usize,
        ) -> Result<u64>;

        /// Get the combined observables mask for a set of flattened-DEM errors.
        ///
        /// # Errors
        ///
        /// Returns a CXX exception if any index is out of range or names a
        /// mechanism the decoder does not retain.
        fn mask_from_errors(
            decoder: &TesseractDecoderWrapper,
            error_indices: &[usize],
        ) -> Result<u64>;

        /// Get the total cost for a set of flattened-DEM errors.
        ///
        /// # Errors
        ///
        /// Same conditions as `mask_from_errors`.
        fn cost_from_errors(
            decoder: &TesseractDecoderWrapper,
            error_indices: &[usize],
        ) -> Result<f64>;

        type TesseractTrellisDecoderWrapper;

        /// Create a Tesseract trellis-mode decoder from a detector error model string.
        ///
        /// # Errors
        ///
        /// Returns a CXX exception if the DEM string is malformed, the model
        /// has more than one observable, the active-detector frontier is wider
        /// than upstream's compiled kernel supports, or memory allocation fails.
        fn create_tesseract_trellis_decoder(
            dem_string: &str,
            config: &TesseractTrellisConfigRepr,
        ) -> Result<UniquePtr<TesseractTrellisDecoderWrapper>>;

        /// Decode detection events by summing probability mass over the trellis.
        ///
        /// # Errors
        ///
        /// Returns a CXX exception if decoding fails.
        fn trellis_decode_detections(
            decoder: Pin<&mut TesseractTrellisDecoderWrapper>,
            detections: &[u64],
        ) -> Result<TesseractTrellisResultRepr>;

        /// Get the number of detectors in the error model.
        fn trellis_num_detectors(decoder: &TesseractTrellisDecoderWrapper) -> usize;

        /// Get the number of (merged, nonzero) errors in the error model.
        fn trellis_num_errors(decoder: &TesseractTrellisDecoderWrapper) -> usize;

        /// Get the number of observables in the error model.
        fn trellis_num_observables(decoder: &TesseractTrellisDecoderWrapper) -> usize;
    }
}
