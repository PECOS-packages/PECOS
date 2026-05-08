//! Tesseract decoder integration tests
//!
//! This file includes all Tesseract-specific tests.

use pecos_tesseract::TesseractConfig;

#[test]
fn test_tesseract_config_default() {
    let config = TesseractConfig::default();
    assert_eq!(config.det_beam, u16::MAX);
    assert!(!config.beam_climbing);
    assert!(config.no_revisit_dets);
    assert!(!config.verbose);
    assert_eq!(config.pqlimit, 200_000);
    assert!(
        config.det_penalty.abs() < f64::EPSILON,
        "det_penalty should be 0.0 but was {}",
        config.det_penalty
    );
}

#[test]
fn test_tesseract_config_fast() {
    let config = TesseractConfig::fast();
    assert_eq!(config.det_beam, 5);
    assert!(config.beam_climbing);
    assert!(config.no_revisit_dets);
    assert!(!config.verbose);
    assert_eq!(config.pqlimit, 200_000);
    assert!(
        (config.det_penalty - 0.1).abs() < f64::EPSILON,
        "det_penalty should be 0.1 but was {}",
        config.det_penalty
    );
}

#[test]
fn test_tesseract_config_accurate() {
    let config = TesseractConfig::accurate();
    assert_eq!(config.det_beam, u16::MAX);
    assert!(!config.beam_climbing);
    assert!(!config.no_revisit_dets);
    assert!(!config.verbose);
    assert_eq!(config.pqlimit, 1_000_000);
    assert!(
        config.det_penalty.abs() < f64::EPSILON,
        "det_penalty should be 0.0 but was {}",
        config.det_penalty
    );
}

#[test]
fn test_tesseract_config_to_ffi_repr() {
    let config = TesseractConfig {
        det_beam: 5,
        beam_climbing: true,
        no_revisit_dets: false,
        verbose: true,
        pqlimit: 5000,
        det_penalty: 0.05,
    };

    let ffi_repr = config.to_ffi_repr();
    assert_eq!(ffi_repr.det_beam, 5);
    assert!(ffi_repr.beam_climbing);
    assert!(!ffi_repr.no_revisit_dets);
    assert!(ffi_repr.verbose);
    assert_eq!(ffi_repr.pqlimit, 5000);
    assert!(
        (ffi_repr.det_penalty - 0.05).abs() < f64::EPSILON,
        "det_penalty should be 0.05 but was {}",
        ffi_repr.det_penalty
    );
}
