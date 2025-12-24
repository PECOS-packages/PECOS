//! Basic tests for Chromobius decoder integration

use ndarray::Array1;
use pecos_chromobius::{ChromobiusConfig, ChromobiusDecoder};
use pecos_decoder_core::Decoder;

#[test]
fn test_chromobius_decoder_creation() {
    // Simple DEM with color/basis annotations
    // Format: detector(x,y,z,color_basis) where color_basis:
    // 0: basis=X, color=R
    // 1: basis=X, color=G
    // 2: basis=X, color=B
    // 3: basis=Z, color=R
    // 4: basis=Z, color=G
    // 5: basis=Z, color=B
    let dem = r"
error(0.1) D0 D1
error(0.1) D1 D2 L0
detector(0, 0, 0, 0) D0
detector(1, 0, 0, 1) D1
detector(2, 0, 0, 2) D2
    "
    .trim();

    let config = ChromobiusConfig::default();
    let decoder = ChromobiusDecoder::new(dem, config);

    assert!(
        decoder.is_ok(),
        "Failed to create decoder: {:?}",
        decoder.err()
    );

    let decoder = decoder.unwrap();
    assert_eq!(decoder.num_detectors(), 3);
    assert_eq!(decoder.num_observables(), 1);
}

#[test]
fn test_chromobius_basic_decoding() {
    // Simple error model
    let dem = r"
error(0.1) D0 D1
error(0.1) D1 D2 L0
detector(0, 0, 0, 0) D0
detector(1, 0, 0, 1) D1
detector(2, 0, 0, 2) D2
    "
    .trim();

    let config = ChromobiusConfig::default();
    let mut decoder = ChromobiusDecoder::new(dem, config).unwrap();

    // Create bit-packed detection events
    // For 3 detectors, we need 1 byte (8 bits)
    // Set detector 0 and 1 active
    let detection_events = vec![0b0000_0011_u8];

    let result = decoder.decode_detection_events(&detection_events);
    assert!(result.is_ok(), "Decoding failed: {:?}", result.err());

    let result = result.unwrap();
    // Check that we got some observable prediction
    println!("Decoded observables: 0x{:x}", result.observables);
}

#[test]
fn test_chromobius_with_weight() {
    let dem = r"
error(0.1) D0 D1
error(0.1) D1 D2 L0
detector(0, 0, 0, 0) D0
detector(1, 0, 0, 1) D1
detector(2, 0, 0, 2) D2
    "
    .trim();

    let config = ChromobiusConfig::default();
    let mut decoder = ChromobiusDecoder::new(dem, config).unwrap();

    // Create detection events
    let detection_events = vec![0b0000_0011_u8];

    let result = decoder.decode_detection_events_with_weight(&detection_events);
    assert!(
        result.is_ok(),
        "Decoding with weight failed: {:?}",
        result.err()
    );

    let result = result.unwrap();
    assert!(result.weight.is_some());
    println!(
        "Decoded observables: 0x{:x}, weight: {:?}",
        result.observables, result.weight
    );
}

#[test]
fn test_chromobius_empty_syndrome() {
    let dem = r"
error(0.1) D0 D1 L0
detector(0, 0, 0, 0) D0
detector(1, 0, 0, 1) D1
    "
    .trim();

    let config = ChromobiusConfig::default();
    let mut decoder = ChromobiusDecoder::new(dem, config).unwrap();

    // Empty detection events
    let detection_events = vec![0u8];

    let result = decoder.decode_detection_events(&detection_events).unwrap();
    // With no detections, should predict no observables flipped
    assert_eq!(result.observables, 0);
}

#[test]
fn test_chromobius_config() {
    let mut config = ChromobiusConfig::default();
    assert!(config.drop_mobius_errors_involving_remnant_errors);

    config.drop_mobius_errors_involving_remnant_errors = false;
    let dem = r"
error(0.1) D0 D1
detector(0, 0, 0, 0) D0
detector(1, 0, 0, 1) D1
    "
    .trim();
    let decoder = ChromobiusDecoder::new(dem, config);
    assert!(decoder.is_ok());
}

#[test]
fn test_chromobius_decoder_trait() {
    let dem = r"
error(0.1) D0 D1
error(0.1) D1 D2 L0
detector(0, 0, 0, 0) D0
detector(1, 0, 0, 1) D1
detector(2, 0, 0, 2) D2
    "
    .trim();

    let config = ChromobiusConfig::default();
    let mut decoder = ChromobiusDecoder::new(dem, config).unwrap();

    // Test the Decoder trait methods
    assert_eq!(decoder.check_count(), 3); // num detectors
    assert_eq!(decoder.bit_count(), 3); // num detectors (as proxy)

    // Test decode method from trait
    let input = Array1::from_vec(vec![0b0000_0011_u8]);
    let result = decoder.decode(&input.view());
    assert!(result.is_ok());
}
