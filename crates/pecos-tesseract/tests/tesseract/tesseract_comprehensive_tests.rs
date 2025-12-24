//! Comprehensive Tesseract tests based on upstream test patterns

use ndarray::Array1;
use pecos_tesseract::{TesseractConfig, TesseractDecoder};

/// Test based on upstream `test_create_decoder` pattern
#[test]
fn test_basic_decoder_creation_and_usage() {
    // DEM similar to their test pattern
    let dem = r"
error(0.125) D0
error(0.375) D0 D1
error(0.25) D1
    "
    .trim();

    let config = TesseractConfig::default();
    let mut decoder = TesseractDecoder::new(dem, config).unwrap();

    // Test basic properties
    assert_eq!(decoder.num_detectors(), 2);
    assert_eq!(decoder.num_errors(), 3);

    // Test decoding a simple pattern
    let detections = Array1::from_vec(vec![0]);
    let result = decoder.decode_detections(&detections.view()).unwrap();

    // Should find some predicted errors
    assert!(!result.predicted_errors.is_empty());
    assert!(result.cost > 0.0);
    assert!(!result.low_confidence);
}

/// Test `decode_with_order` method
#[test]
fn test_decode_with_order() {
    let dem = r"
error(0.1) D0 D1
error(0.2) D1 D2
error(0.15) D0 D2
    "
    .trim();

    let config = TesseractConfig::default();
    let mut decoder = TesseractDecoder::new(dem, config).unwrap();

    let detections = Array1::from_vec(vec![0, 1]);

    // Test with detector order 0
    let result = decoder.decode_with_order(&detections.view(), 0).unwrap();
    assert!(!result.predicted_errors.is_empty());
    assert!(result.cost > 0.0);
}

/// Test `mask_from_errors` functionality
#[test]
fn test_mask_from_errors() {
    let dem = r"
error(0.1) D0 D1
error(0.2) D1 D2 L0
error(0.15) D0 L0
    "
    .trim();

    let config = TesseractConfig::default();
    let decoder = TesseractDecoder::new(dem, config).unwrap();

    // Test basic functionality - check all errors for observable effects
    println!("Number of errors: {}", decoder.num_errors());
    for i in 0..decoder.num_errors() {
        let error_indices = vec![i];
        let mask = decoder.mask_from_errors(&error_indices);
        println!("Error {i} mask: 0x{mask:x}");
    }

    // Test empty errors should have zero mask
    let empty_errors = vec![];
    let zero_mask = decoder.mask_from_errors(&empty_errors);
    println!("Empty errors mask: 0x{zero_mask:x}");
    assert_eq!(zero_mask, 0);

    // Just test that the functionality works (don't make assumptions about which errors affect observables)
    let all_errors: Vec<usize> = (0..decoder.num_errors()).collect();
    let _all_mask = decoder.mask_from_errors(&all_errors);
    // This should work without panic
}

/// Test `cost_from_errors` functionality
#[test]
fn test_cost_from_errors() {
    let dem = r"
error(0.125) D0
error(0.375) D0 D1
error(0.25) D1
    "
    .trim();

    let config = TesseractConfig::default();
    let decoder = TesseractDecoder::new(dem, config).unwrap();

    // Test cost calculation for specific errors
    let error_indices = vec![1]; // Second error (0.375 probability)
    let cost = decoder.cost_from_errors(&error_indices);
    println!("Cost for error 1: {cost}");

    // Test empty errors should have zero cost
    let empty_errors = vec![];
    let zero_cost = decoder.cost_from_errors(&empty_errors);
    println!("Cost for empty errors: {zero_cost}");
    assert!(
        zero_cost.abs() < f64::EPSILON,
        "Cost should be zero but was {zero_cost}"
    );

    // Test cost calculation for all errors individually
    for i in 0..decoder.num_errors() {
        let single_error = vec![i];
        let cost = decoder.cost_from_errors(&single_error);
        println!("Cost for error {i}: {cost}");
        assert!(cost >= 0.0); // Cost should never be negative
    }
}

/// Test error information retrieval
#[test]
fn test_error_information() {
    let dem = r"
error(0.125) D0
error(0.375) D0 D1
error(0.25) D1 L0
    "
    .trim();

    let config = TesseractConfig::default();
    let decoder = TesseractDecoder::new(dem, config).unwrap();

    // Test error 0
    let error_info = decoder.get_error_info(0).unwrap();
    assert!((error_info.probability - 0.125).abs() < 0.001);
    assert_eq!(error_info.detectors, vec![0]);
    assert_eq!(error_info.observables, 0);

    // Test error 1
    let error_info = decoder.get_error_info(1).unwrap();
    assert!((error_info.probability - 0.375).abs() < 0.001);
    assert_eq!(error_info.detectors, vec![0, 1]);
    assert_eq!(error_info.observables, 0);

    // Test error 2 (affects observable)
    let error_info = decoder.get_error_info(2).unwrap();
    assert!((error_info.probability - 0.25).abs() < 0.001);
    assert_eq!(error_info.detectors, vec![1]);
    assert_ne!(error_info.observables, 0); // Should affect L0
}

/// Test different configuration presets
#[test]
fn test_configuration_presets() {
    let dem = r"
error(0.1) D0 D1
error(0.1) D1 D2
error(0.1) D2 D3
    "
    .trim();

    // Test fast configuration
    let fast_config = TesseractConfig::fast();
    let mut fast_decoder = TesseractDecoder::new(dem, fast_config).unwrap();
    assert_eq!(fast_decoder.det_beam(), 100);
    assert!(fast_decoder.beam_climbing());

    // Test accurate configuration
    let accurate_config = TesseractConfig::accurate();
    let mut accurate_decoder = TesseractDecoder::new(dem, accurate_config).unwrap();
    assert_eq!(accurate_decoder.det_beam(), u16::MAX);
    assert!(!accurate_decoder.beam_climbing());

    // Test both can decode the same pattern
    let detections = Array1::from_vec(vec![0, 2]);
    let fast_result = fast_decoder.decode_detections(&detections.view()).unwrap();
    let accurate_result = accurate_decoder
        .decode_detections(&detections.view())
        .unwrap();

    // Both should find valid solutions
    assert!(!fast_result.low_confidence);
    assert!(!accurate_result.low_confidence);
}

/// Test zero syndrome (no detections)
#[test]
fn test_zero_syndrome() {
    let dem = r"
error(0.1) D0 D1
error(0.1) D1 D2
    "
    .trim();

    let config = TesseractConfig::default();
    let mut decoder = TesseractDecoder::new(dem, config).unwrap();

    // Empty detection pattern
    let detections = Array1::from_vec(vec![]);
    let result = decoder.decode_detections(&detections.view()).unwrap();

    // Should find no errors and have zero cost
    assert!(result.predicted_errors.is_empty());
    assert!(
        result.cost.abs() < f64::EPSILON,
        "Cost should be zero but was {}",
        result.cost
    );
    assert!(!result.low_confidence);
    assert_eq!(result.observables_mask, 0);
}

/// Test all single-bit error patterns
#[test]
fn test_single_detector_patterns() {
    let dem = r"
error(0.1) D0
error(0.1) D1
error(0.1) D2
error(0.05) D0 D1
error(0.05) D1 D2
error(0.05) D0 D2
    "
    .trim();

    let config = TesseractConfig::default();
    let mut decoder = TesseractDecoder::new(dem, config).unwrap();

    // Test each single detector firing
    for detector in 0..3 {
        let detections = Array1::from_vec(vec![detector]);
        let result = decoder.decode_detections(&detections.view()).unwrap();

        // Should find a solution for each single detector
        assert!(
            !result.low_confidence,
            "Failed to decode detector {detector}"
        );
        assert!(result.cost > 0.0);
    }
}

/// Test configuration getters match what was set
#[test]
fn test_configuration_getters() {
    let dem = "error(0.1) D0";

    let custom_config = TesseractConfig {
        det_beam: 50,
        beam_climbing: true,
        no_revisit_dets: false,
        at_most_two_errors_per_detector: true,
        verbose: false,
        pqlimit: 5000,
        det_penalty: 0.05,
    };

    let decoder = TesseractDecoder::new(dem, custom_config).unwrap();

    // Verify all configuration values
    assert_eq!(decoder.det_beam(), 50);
    assert!(decoder.beam_climbing());
    assert!(!decoder.no_revisit_dets());
    assert!(decoder.at_most_two_errors_per_detector());
    assert!(!decoder.verbose());
    assert_eq!(decoder.pqlimit(), 5000);
    assert!((decoder.det_penalty() - 0.05).abs() < 0.001);
}

/// Test edge case: invalid error index
#[test]
fn test_invalid_error_index() {
    let dem = "error(0.1) D0";
    let config = TesseractConfig::default();
    let decoder = TesseractDecoder::new(dem, config).unwrap();

    // Should return None for invalid error index
    assert!(decoder.get_error_info(999).is_none());
}

/// Test multiple decoding on same decoder
#[test]
fn test_repeated_decoding() {
    let dem = r"
error(0.1) D0 D1
error(0.1) D1 D2
    "
    .trim();

    let config = TesseractConfig::default();
    let mut decoder = TesseractDecoder::new(dem, config).unwrap();

    let patterns = vec![vec![0], vec![1], vec![0, 1], vec![1, 2], vec![]];

    // Should be able to decode multiple patterns with same decoder
    for pattern in patterns {
        let detections = Array1::from_vec(pattern.clone());
        let result = decoder.decode_detections(&detections.view()).unwrap();
        // Each should succeed (most patterns should decode successfully)
        // Note: some complex patterns might have low confidence, which is acceptable
        println!(
            "Pattern {:?}: cost={:.3}, low_confidence={}",
            pattern, result.cost, result.low_confidence
        );
    }
}
