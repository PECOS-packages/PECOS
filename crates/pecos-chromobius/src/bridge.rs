//! CXX FFI bridge for Chromobius decoder
//!
//! This module provides the low-level FFI bindings to the Chromobius C++ library.
//! Users should prefer the high-level [`ChromobiusDecoder`](crate::ChromobiusDecoder) API.

#[cxx::bridge]
pub(crate) mod ffi {
    unsafe extern "C++" {
        include!("chromobius_bridge.h");

        type ChromobiusDecoderWrapper;

        /// Create a Chromobius decoder from a detector error model string.
        ///
        /// # Errors
        ///
        /// Returns a CXX exception if the DEM string is malformed or contains
        /// unsupported error mechanisms.
        fn create_chromobius_decoder(
            dem_string: &str,
            drop_mobius_errors_involving_remnant_errors: bool,
        ) -> Result<UniquePtr<ChromobiusDecoderWrapper>>;

        /// Decode bit-packed detection events and return the observables mask.
        ///
        /// # Errors
        ///
        /// Returns a CXX exception if decoding fails.
        fn decode_detection_events(
            decoder: Pin<&mut ChromobiusDecoderWrapper>,
            bit_packed_detection_events: &[u8],
        ) -> Result<u64>;

        /// Decode bit-packed detection events, returning observables mask and weight.
        ///
        /// # Errors
        ///
        /// Returns a CXX exception if decoding fails.
        fn decode_detection_events_with_weight(
            decoder: Pin<&mut ChromobiusDecoderWrapper>,
            bit_packed_detection_events: &[u8],
            weight_out: &mut f32,
        ) -> Result<u64>;

        /// Get the number of detectors in the error model.
        fn chromobius_get_num_detectors(decoder: &ChromobiusDecoderWrapper) -> usize;

        /// Get the number of observables in the error model.
        fn chromobius_get_num_observables(decoder: &ChromobiusDecoderWrapper) -> usize;
    }
}
