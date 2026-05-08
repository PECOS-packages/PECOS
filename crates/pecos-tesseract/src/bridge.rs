//! FFI bridge to Tesseract C++ library
//!
//! Low-level FFI bindings to the Tesseract C++ library.
//! Users should prefer the high-level [`TesseractDecoder`](crate::TesseractDecoder) API.

#[cxx::bridge]
pub(crate) mod ffi {
    // Struct representations for C++ interop
    #[derive(Debug)]
    pub struct TesseractConfigRepr {
        pub det_beam: u16,
        pub beam_climbing: bool,
        pub no_revisit_dets: bool,
        pub verbose: bool,
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

        /// Get the number of errors in the error model.
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

        /// Get the priority queue limit.
        fn get_pqlimit(decoder: &TesseractDecoderWrapper) -> usize;

        /// Get the detector penalty factor.
        fn get_det_penalty(decoder: &TesseractDecoderWrapper) -> f64;

        /// Get the probability of a specific error.
        fn get_error_probability(decoder: &TesseractDecoderWrapper, error_idx: usize) -> f64;

        /// Get the cost of a specific error.
        fn get_error_cost(decoder: &TesseractDecoderWrapper, error_idx: usize) -> f64;

        /// Get the detectors affected by a specific error.
        fn get_error_detectors(decoder: &TesseractDecoderWrapper, error_idx: usize) -> Vec<i32>;

        /// Get the observables mask for a specific error.
        fn get_error_observables(decoder: &TesseractDecoderWrapper, error_idx: usize) -> u64;

        /// Get the combined observables mask for a set of errors.
        fn mask_from_errors(decoder: &TesseractDecoderWrapper, error_indices: &[usize]) -> u64;

        /// Get the total cost for a set of errors.
        fn cost_from_errors(decoder: &TesseractDecoderWrapper, error_indices: &[usize]) -> f64;
    }
}
