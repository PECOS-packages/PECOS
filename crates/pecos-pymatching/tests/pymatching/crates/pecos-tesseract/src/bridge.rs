//! FFI bridge to Tesseract C++ library

#[cxx::bridge]
pub mod ffi {
    // Struct representations for C++ interop
    #[derive(Debug)]
    pub struct TesseractConfigRepr {
        pub det_beam: u16,
        pub beam_climbing: bool,
        pub no_revisit_dets: bool,
        pub at_most_two_errors_per_detector: bool,
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

        // Tesseract decoder type
        type TesseractDecoderWrapper;

        // Constructor
        fn create_tesseract_decoder(
            dem_string: &str,
            config: &TesseractConfigRepr,
        ) -> Result<UniquePtr<TesseractDecoderWrapper>>;

        // Decoding methods
        fn decode_detections(
            decoder: Pin<&mut TesseractDecoderWrapper>,
            detections: &[u64],
        ) -> Result<DecodingResultRepr>;

        fn decode_detections_with_order(
            decoder: Pin<&mut TesseractDecoderWrapper>,
            detections: &[u64],
            det_order: usize,
        ) -> Result<DecodingResultRepr>;

        // Information getters
        fn get_num_detectors(decoder: &TesseractDecoderWrapper) -> usize;
        fn get_num_errors(decoder: &TesseractDecoderWrapper) -> usize;
        fn get_num_observables(decoder: &TesseractDecoderWrapper) -> usize;

        // Configuration getters
        fn get_det_beam(decoder: &TesseractDecoderWrapper) -> u16;
        fn get_beam_climbing(decoder: &TesseractDecoderWrapper) -> bool;
        fn get_no_revisit_dets(decoder: &TesseractDecoderWrapper) -> bool;
        fn get_at_most_two_errors_per_detector(decoder: &TesseractDecoderWrapper) -> bool;
        fn get_verbose(decoder: &TesseractDecoderWrapper) -> bool;
        fn get_pqlimit(decoder: &TesseractDecoderWrapper) -> usize;
        fn get_det_penalty(decoder: &TesseractDecoderWrapper) -> f64;

        // Error analysis
        fn get_error_probability(decoder: &TesseractDecoderWrapper, error_idx: usize) -> f64;
        fn get_error_cost(decoder: &TesseractDecoderWrapper, error_idx: usize) -> f64;
        fn get_error_detectors(decoder: &TesseractDecoderWrapper, error_idx: usize) -> Vec<i32>;
        fn get_error_observables(decoder: &TesseractDecoderWrapper, error_idx: usize) -> u64;

        // Utility functions
        fn mask_from_errors(decoder: &TesseractDecoderWrapper, error_indices: &[usize]) -> u64;

        fn cost_from_errors(decoder: &TesseractDecoderWrapper, error_indices: &[usize]) -> f64;
    }
}
