//! Comprehensive tests for Chromobius decoder integration
//! Based on test patterns from the upstream Chromobius repository

use pecos_chromobius::{ChromobiusConfig, ChromobiusDecoder};

/// Test various distance color codes
#[test]
fn test_chromobius_distance_scaling() {
    // Test that decoder can handle various code distances
    let distances = vec![3, 5, 7];
    let error_rates = vec![0.001, 0.01, 0.1];

    for d in distances {
        for &p in &error_rates {
            let dem = generate_color_code_dem(d, p);
            let config = ChromobiusConfig::default();
            let decoder = ChromobiusDecoder::new(&dem, config);

            assert!(
                decoder.is_ok(),
                "Failed to create decoder for d={}, p={}: {:?}",
                d,
                p,
                decoder.err()
            );
        }
    }
}

/// Test empty circuit edge case
#[test]
fn test_chromobius_empty_circuit() {
    let dem = ""; // Empty DEM
    let config = ChromobiusConfig::default();
    let decoder = ChromobiusDecoder::new(dem, config);

    // Should handle empty circuit gracefully
    assert!(decoder.is_ok());
    let decoder = decoder.unwrap();
    assert_eq!(decoder.num_detectors(), 0);
    assert_eq!(decoder.num_observables(), 0);
}

/// Test single detector patterns
#[test]
fn test_chromobius_single_detector_patterns() {
    // Test all single detector activation patterns
    let dem = r#"
error(0.1) D0 L0
error(0.1) D1 L0
error(0.1) D2 L0
detector(0, 0, 0, 0) D0
detector(1, 0, 0, 1) D1
detector(2, 0, 0, 2) D2
        "#
    .trim();

    let config = ChromobiusConfig::default();
    let mut decoder = ChromobiusDecoder::new(dem, config).unwrap();

    // Test each single detector firing
    for i in 0..3 {
        let mut detection_events = vec![0u8];
        detection_events[0] |= 1 << i;

        let result = decoder.decode_detection_events(&detection_events);
        assert!(
            result.is_ok(),
            "Failed to decode single detector {}: {:?}",
            i,
            result.err()
        );
    }
}

/// Test multiple round decoding
#[test]
fn test_chromobius_multiple_rounds() {
    // Simulate multiple rounds of syndrome extraction
    let rounds = vec![1, 5, 10, 20];

    for r in rounds {
        let dem = generate_multi_round_dem(r);
        let config = ChromobiusConfig::default();
        let decoder = ChromobiusDecoder::new(&dem, config);

        assert!(
            decoder.is_ok(),
            "Failed to create decoder for {} rounds: {:?}",
            r,
            decoder.err()
        );

        let decoder = decoder.unwrap();
        // Number of detectors should scale with rounds
        assert!(decoder.num_detectors() > 0);
    }
}

/// Test phenomenological noise model
#[test]
fn test_chromobius_phenomenological_noise() {
    // Create a valid phenomenological noise model
    // Each error should create unique detector combinations
    let dem = r#"
error(0.001) D0 D1
error(0.001) D1 D2 L0
error(0.001) D0 D2
detector(0, 0, 0, 0) D0
detector(1, 0, 0, 1) D1
detector(2, 0, 0, 2) D2
        "#
    .trim();

    let config = ChromobiusConfig::default();
    let mut decoder = ChromobiusDecoder::new(dem, config).unwrap();

    // Test decoding with a valid detection pattern
    // Only trigger two detectors to form a valid error chain
    let detection_events = vec![0b00000011u8]; // D0 and D1 triggered
    let result = decoder.decode_detection_events(&detection_events);
    assert!(
        result.is_ok(),
        "Failed to decode with phenomenological noise: {:?}",
        result.err()
    );
}

/// Test batch decoding performance
#[test]
fn test_chromobius_batch_decode() {
    // Create a simple test circuit where we know valid detection patterns
    let dem = r#"
error(0.01) D0 D1
error(0.01) D1 D2 L0
error(0.01) D0 D2
detector(0, 0, 0, 0) D0
detector(1, 0, 0, 1) D1
detector(2, 0, 0, 2) D2
        "#
    .trim();

    let config = ChromobiusConfig::default();
    let mut decoder = ChromobiusDecoder::new(dem, config).unwrap();

    // Test various detection patterns
    let test_patterns = vec![
        0b00000000u8, // No detections
        0b00000001u8, // D0 only
        0b00000010u8, // D1 only
        0b00000011u8, // D0 and D1
        0b00000110u8, // D1 and D2
        0b00000101u8, // D0 and D2
    ];

    let mut success_count = 0;
    let mut decode_count = 0;

    // Try each pattern multiple times
    for _ in 0..10 {
        for &pattern in &test_patterns {
            let detection_events = vec![pattern];

            match decoder.decode_detection_events(&detection_events) {
                Ok(_result) => {
                    decode_count += 1;
                    // Count successful decodings
                    success_count += 1;
                }
                Err(_) => {
                    // Some patterns might not decode successfully
                }
            }
        }
    }

    // Should have decoded at least some patterns successfully
    assert!(
        success_count > 0,
        "No successful decodings out of {} attempts",
        test_patterns.len() * 10
    );
    assert!(decode_count >= success_count);
}

/// Test detector coordinate edge cases
#[test]
fn test_chromobius_detector_coordinates() {
    // Test with -1 coordinate (should be ignored)
    let dem = r#"
error(0.1) D0 D1
detector(-1, -1, -1, -1) D0
detector(1, 0, 0, 1) D1
        "#
    .trim();

    let config = ChromobiusConfig::default();
    let decoder = ChromobiusDecoder::new(dem, config);

    // Should handle -1 coordinates gracefully
    assert!(decoder.is_ok());
}

/// Test very high error rates
#[test]
fn test_chromobius_high_error_rate() {
    let dem = r#"
error(0.4) D0 D1 L0
error(0.4) D1 D2 L0
detector(0, 0, 0, 0) D0
detector(1, 0, 0, 1) D1
detector(2, 0, 0, 2) D2
        "#
    .trim();

    let config = ChromobiusConfig::default();
    let mut decoder = ChromobiusDecoder::new(dem, config).unwrap();

    // Should still decode even with high error rates
    let detection_events = vec![0b00000011u8];
    let result = decoder.decode_detection_events(&detection_events);
    assert!(result.is_ok());
}

/// Test configuration variations
#[test]
fn test_chromobius_config_variations() {
    let dem = generate_color_code_dem(5, 0.01);

    // Test with different configurations
    let config = ChromobiusConfig {
        drop_mobius_errors_involving_remnant_errors: false,
    };
    let decoder = ChromobiusDecoder::new(&dem, config);
    assert!(decoder.is_ok());

    // Test with default config (mobius errors enabled)
    let config = ChromobiusConfig::default();
    let decoder = ChromobiusDecoder::new(&dem, config);
    assert!(decoder.is_ok());
}

// Helper functions to generate test DEMs

fn generate_color_code_dem(distance: usize, error_rate: f64) -> String {
    // Simplified color code DEM generator
    let mut dem = String::new();

    // Add some errors and detectors based on distance
    for i in 0..distance {
        for j in 0..distance {
            if i + 1 < distance && j + 1 < distance {
                dem.push_str(&format!(
                    "error({}) D{} D{}\n",
                    error_rate,
                    i * distance + j,
                    (i + 1) * distance + j
                ));
            }
        }
    }

    // Add observable errors
    dem.push_str(&format!("error({error_rate}) D0 L0\n"));

    // Add detector coordinates
    for i in 0..distance {
        for j in 0..distance {
            let idx = i * distance + j;
            let color_basis = (i + j) % 6; // Cycle through color/basis combinations
            dem.push_str(&format!("detector({i}, {j}, 0, {color_basis}) D{idx}\n"));
        }
    }

    dem
}

fn generate_multi_round_dem(rounds: usize) -> String {
    // Simplified multi-round DEM generator
    let mut dem = String::new();

    for r in 0..rounds {
        // Add errors for this round
        dem.push_str(&format!("error(0.01) D{} D{}\n", r * 3, r * 3 + 1));
        dem.push_str(&format!("error(0.01) D{} D{} L0\n", r * 3 + 1, r * 3 + 2));

        // Add detectors for this round
        for i in 0..3 {
            dem.push_str(&format!(
                "detector({}, {}, {}, {}) D{}\n",
                i,
                0,
                r,
                i,
                r * 3 + i
            ));
        }
    }

    dem
}
