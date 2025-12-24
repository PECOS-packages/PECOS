//! CXX FFI bridge for Chromobius decoder

#[cxx::bridge]
pub mod ffi {
    unsafe extern "C++" {
        include!("chromobius_bridge.h");

        type ChromobiusDecoderWrapper;

        fn create_chromobius_decoder(
            dem_string: &str,
            drop_mobius_errors_involving_remnant_errors: bool,
        ) -> Result<UniquePtr<ChromobiusDecoderWrapper>>;

        fn decode_detection_events(
            decoder: Pin<&mut ChromobiusDecoderWrapper>,
            bit_packed_detection_events: &[u8],
        ) -> Result<u64>;

        fn decode_detection_events_with_weight(
            decoder: Pin<&mut ChromobiusDecoderWrapper>,
            bit_packed_detection_events: &[u8],
            weight_out: &mut f32,
        ) -> Result<u64>;

        fn chromobius_get_num_detectors(decoder: &ChromobiusDecoderWrapper) -> usize;
        fn chromobius_get_num_observables(decoder: &ChromobiusDecoderWrapper) -> usize;
    }
}
