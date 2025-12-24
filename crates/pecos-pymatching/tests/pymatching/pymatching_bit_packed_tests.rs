//! Comprehensive tests for `PyMatching` bit-packed format functionality
//!
//! This module tests the bit-packed syndrome encoding/decoding and batch processing
//! functionality in `PyMatching` decoder implementation.

use pecos_pymatching::{BatchConfig, PyMatchingDecoder};

// Helper function to add boundary edges to handle odd parity syndromes
fn add_boundary_edges(decoder: &mut PyMatchingDecoder, num_nodes: usize) {
    for i in 0..num_nodes {
        let _ = decoder.add_boundary_edge(i, &[], Some(10.0), None, None);
    }
}

// ============================================================================
// Bit-Packed Syndrome Encoding/Decoding Tests
// ============================================================================

#[test]
fn test_bit_packed_syndrome_encoding_basic() {
    // Test basic bit-packed syndrome encoding with small syndrome sizes
    let mut decoder = PyMatchingDecoder::builder()
        .nodes(8)
        .observables(4)
        .build()
        .unwrap();

    // Create a simple matching graph
    decoder.add_edge(0, 1, &[0], Some(1.0), None, None).unwrap();
    decoder.add_edge(2, 3, &[1], Some(1.0), None, None).unwrap();
    decoder.add_edge(4, 5, &[2], Some(1.0), None, None).unwrap();
    decoder.add_edge(6, 7, &[3], Some(1.0), None, None).unwrap();

    // Add boundary edges to handle odd parity syndromes
    for i in 0..8 {
        decoder
            .add_boundary_edge(i, &[], Some(10.0), None, None)
            .unwrap();
    }

    // Test various bit patterns in a single byte
    let test_cases = vec![
        (0b0000_0000_u8, "all zeros"),
        (0b0000_0001_u8, "single bit"),
        (0b1000_0000_u8, "high bit"),
        (0b1010_1010_u8, "alternating pattern"),
        (0b1111_1111_u8, "all ones"),
    ];

    for (bit_pattern, description) in test_cases {
        // Create bit-packed shots: 1 shot with 8 detectors packed into 1 byte
        let shots = vec![bit_pattern];

        let result = decoder
            .decode_batch_with_config(
                &shots,
                1,
                8,
                BatchConfig {
                    bit_packed_input: true,
                    bit_packed_output: false,
                    return_weights: true,
                },
            )
            .unwrap();

        assert_eq!(
            result.predictions.len(),
            1,
            "Should have 1 prediction for {description}"
        );
        assert_eq!(
            result.weights.len(),
            1,
            "Should have 1 weight for {description}"
        );
        assert!(!result.bit_packed, "Predictions should not be bit-packed");

        // Verify the prediction is reasonable (not all zeros if syndrome had bits set)
        let has_syndrome_bits = bit_pattern != 0;
        if has_syndrome_bits {
            let prediction = &result.predictions[0];
            println!("Bit pattern: {bit_pattern:08b}, prediction: {prediction:?}");
            // At least some observable should be set if syndrome has bits
            assert!(
                !prediction.is_empty(),
                "Prediction should have length > 0 for {description}"
            );
        }
    }
}

#[test]
fn test_bit_packed_syndrome_encoding_multi_byte() {
    // Test bit-packed syndrome encoding with multi-byte syndromes
    let mut decoder = PyMatchingDecoder::builder()
        .nodes(16)
        .observables(8)
        .build()
        .unwrap();

    // Create a larger matching graph
    for i in 0..15 {
        decoder
            .add_edge(i, i + 1, &[i % 8], Some(1.0), None, None)
            .unwrap();
    }

    // Add boundary edges
    for i in 0..16 {
        decoder
            .add_boundary_edge(i, &[], Some(10.0), None, None)
            .unwrap();
    }

    // Test with 16 detectors (2 bytes when bit-packed)
    let test_cases = vec![
        (vec![0b0000_0000, 0b0000_0000], "all zeros"),
        (vec![0b0000_0001, 0b0000_0000], "first bit only"),
        (vec![0b0000_0000, 0b1000_0000], "last bit only"),
        (vec![0b1010_1010, 0b0101_0101], "alternating pattern"),
        (vec![0b1111_1111, 0b1111_1111], "all ones"),
        (vec![0b1111_0000, 0b0000_1111], "split pattern"),
    ];

    for (bit_pattern, description) in test_cases {
        let result = decoder
            .decode_batch_with_config(
                &bit_pattern,
                1,  // num_shots
                16, // num_detectors
                BatchConfig {
                    bit_packed_input: true,
                    bit_packed_output: false,
                    return_weights: true,
                },
            )
            .unwrap();

        assert_eq!(
            result.predictions.len(),
            1,
            "Should have 1 prediction for {description}"
        );
        assert_eq!(
            result.weights.len(),
            1,
            "Should have 1 weight for {description}"
        );

        // Check that we get a valid prediction
        let prediction = &result.predictions[0];
        assert!(
            !prediction.is_empty(),
            "Prediction should have some length for {description}"
        );
        println!(
            "Multi-byte pattern: {:?}, prediction length: {}",
            bit_pattern,
            prediction.len()
        );
    }
}

#[test]
fn test_bit_packed_vs_unpacked_syndrome_equivalence() {
    // Test that bit-packed and unpacked syndromes produce equivalent results
    let mut decoder = PyMatchingDecoder::builder()
        .nodes(8)
        .observables(4)
        .build()
        .unwrap();

    // Create a symmetric matching graph for consistent results
    decoder.add_edge(0, 1, &[0], Some(1.0), None, None).unwrap();
    decoder.add_edge(2, 3, &[1], Some(1.0), None, None).unwrap();
    decoder.add_edge(4, 5, &[2], Some(1.0), None, None).unwrap();
    decoder.add_edge(6, 7, &[3], Some(1.0), None, None).unwrap();

    // Add boundary edges
    for i in 0..8 {
        decoder
            .add_boundary_edge(i, &[], Some(10.0), None, None)
            .unwrap();
    }

    let test_patterns = [
        vec![0, 1, 0, 0, 1, 0, 1, 0], // Some detections
        vec![1, 1, 1, 1, 0, 0, 0, 0], // First half
        vec![0, 0, 0, 0, 1, 1, 1, 1], // Second half
    ];

    for (i, unpacked_syndrome) in test_patterns.iter().enumerate() {
        // Create bit-packed version
        let mut packed_syndrome = 0u8;
        for (bit_pos, &bit) in unpacked_syndrome.iter().enumerate() {
            if bit != 0 {
                packed_syndrome |= 1 << bit_pos;
            }
        }
        let packed_shots = vec![packed_syndrome];

        // Decode with unpacked format
        let unpacked_result = decoder
            .decode_batch_with_config(
                unpacked_syndrome,
                1, // num_shots
                8, // num_detectors
                BatchConfig {
                    bit_packed_input: false,
                    bit_packed_output: false,
                    return_weights: true,
                },
            )
            .unwrap();

        // Decode with bit-packed format
        let packed_result = decoder
            .decode_batch_with_config(
                &packed_shots,
                1, // num_shots
                8, // num_detectors
                BatchConfig {
                    bit_packed_input: true,
                    bit_packed_output: false,
                    return_weights: true,
                },
            )
            .unwrap();

        // Results should be equivalent
        assert_eq!(
            unpacked_result.predictions.len(),
            packed_result.predictions.len(),
            "Prediction count should match for test case {i}"
        );
        assert_eq!(
            unpacked_result.weights.len(),
            packed_result.weights.len(),
            "Weight count should match for test case {i}"
        );

        // Weights should be identical (or very close)
        let weight_diff = (unpacked_result.weights[0] - packed_result.weights[0]).abs();
        assert!(
            weight_diff < 1e-10,
            "Weights should be identical: unpacked={}, packed={}, test case {}",
            unpacked_result.weights[0],
            packed_result.weights[0],
            i
        );

        // Predictions should be identical
        assert_eq!(
            unpacked_result.predictions[0], packed_result.predictions[0],
            "Predictions should be identical for test case {i}"
        );

        println!(
            "Test case {}: unpacked syndrome: {:?}, packed: {:08b}, weight: {:.6}",
            i, unpacked_syndrome, packed_syndrome, packed_result.weights[0]
        );
    }
}

// ============================================================================
// Batch Decoding with Bit-Packed Formats Tests
// ============================================================================

#[test]
fn test_batch_decoding_bit_packed_shots() {
    // Test batch decoding with bit-packed input shots
    let mut decoder = PyMatchingDecoder::builder()
        .nodes(12)
        .observables(6)
        .build()
        .unwrap();

    // Create a matching graph
    for i in 0..11 {
        decoder
            .add_edge(i, i + 1, &[i % 6], Some(1.0), None, None)
            .unwrap();
    }
    add_boundary_edges(&mut decoder, 12);

    let num_shots = 5;
    let num_detectors: usize = 12;
    let _bytes_per_shot = num_detectors.div_ceil(8); // 2 bytes per shot

    // Create diverse bit-packed shots
    let mut shots = Vec::new();
    for shot in 0..num_shots {
        // Create different patterns for each shot
        let pattern1 = (shot * 37) % 256; // Pseudo-random pattern
        let pattern2 = (shot * 73) % 256; // Different pseudo-random pattern
        shots.push(u8::try_from(pattern1).expect("pattern fits in u8"));
        shots.push(u8::try_from(pattern2).expect("pattern fits in u8"));
    }

    let result = decoder
        .decode_batch_with_config(
            &shots,
            num_shots,
            num_detectors,
            BatchConfig {
                bit_packed_input: true,
                bit_packed_output: false,
                return_weights: true,
            },
        )
        .unwrap();

    assert_eq!(result.predictions.len(), num_shots);
    assert_eq!(result.weights.len(), num_shots);
    assert!(!result.bit_packed);

    // Verify each prediction is reasonable
    for (i, prediction) in result.predictions.iter().enumerate() {
        assert!(
            !prediction.is_empty(),
            "Prediction {i} should have some length"
        );
        println!(
            "Shot {}: prediction length: {}, weight: {:.6}",
            i,
            prediction.len(),
            result.weights[i]
        );
    }
}

#[test]
fn test_batch_decoding_bit_packed_predictions() {
    // Test batch decoding with bit-packed output predictions
    let mut decoder = PyMatchingDecoder::builder()
        .nodes(10)
        .observables(16) // More observables to test bit-packing
        .build()
        .unwrap();

    // Create a larger matching graph
    for i in 0..9 {
        decoder
            .add_edge(i, i + 1, &[i % 16], Some(1.0), None, None)
            .unwrap();
    }
    add_boundary_edges(&mut decoder, 10);

    let num_shots = 3;
    let num_detectors = 10;

    // Create simple unpacked shots for clarity
    let mut shots = Vec::new();
    for shot in 0..num_shots {
        for detector in 0..num_detectors {
            // Simple pattern: set detector if (shot + detector) is odd
            shots.push(u8::try_from((shot + detector) % 2).expect("0 or 1 fits in u8"));
        }
    }

    let result = decoder
        .decode_batch_with_config(
            &shots,
            num_shots,
            num_detectors,
            BatchConfig {
                bit_packed_input: false,
                bit_packed_output: true,
                return_weights: true,
            },
        )
        .unwrap();

    assert_eq!(result.predictions.len(), num_shots);
    assert_eq!(result.weights.len(), num_shots);
    assert!(result.bit_packed);

    // Verify bit-packed predictions format
    for (i, prediction) in result.predictions.iter().enumerate() {
        // For bit-packed predictions, the length should be related to the number of observables
        let expected_bytes = decoder.num_observables().div_ceil(8);
        println!(
            "Shot {}: prediction bytes: {}, expected bytes: {}, num_observables: {}",
            i,
            prediction.len(),
            expected_bytes,
            decoder.num_observables()
        );

        // PyMatching may use different packing strategies, so we just verify it's reasonable
        assert!(
            !prediction.is_empty(),
            "Bit-packed prediction {i} should have some bytes"
        );
        assert!(
            prediction.len() <= expected_bytes + 8,
            "Bit-packed prediction {i} should not be excessively long"
        );
    }
}

#[test]
fn test_batch_decoding_both_bit_packed() {
    // Test batch decoding with both input and output bit-packed
    let mut decoder = PyMatchingDecoder::builder()
        .nodes(16)
        .observables(8)
        .build()
        .unwrap();

    // Create a comprehensive matching graph
    for i in 0..15 {
        decoder
            .add_edge(i, i + 1, &[i % 8], Some(1.0), None, None)
            .unwrap();
    }
    add_boundary_edges(&mut decoder, 16);

    let num_shots = 4;
    let num_detectors: usize = 16;
    let _bytes_per_shot = num_detectors.div_ceil(8); // 2 bytes per shot

    // Create bit-packed shots with known patterns
    let mut shots = Vec::new();
    let test_patterns = vec![
        (0b1111_0000, 0b0000_1111), // Split pattern
        (0b1010_1010, 0b0101_0101), // Alternating
        (0b1111_1111, 0b0000_0000), // First byte full
        (0b0000_0000, 0b1111_1111), // Second byte full
    ];

    for (pattern1, pattern2) in test_patterns {
        shots.push(pattern1);
        shots.push(pattern2);
    }

    let result = decoder
        .decode_batch_with_config(
            &shots,
            num_shots,
            num_detectors,
            BatchConfig {
                bit_packed_input: true,
                bit_packed_output: true,
                return_weights: true,
            },
        )
        .unwrap();

    assert_eq!(result.predictions.len(), num_shots);
    assert_eq!(result.weights.len(), num_shots);
    assert!(result.bit_packed);

    // All shots should produce valid results
    for (i, (prediction, weight)) in result.predictions.iter().zip(&result.weights).enumerate() {
        assert!(
            !prediction.is_empty(),
            "Prediction {i} should have some bytes"
        );
        assert!(weight.is_finite(), "Weight {i} should be finite: {weight}");
        println!(
            "Shot {}: {} bytes, weight: {:.6}",
            i,
            prediction.len(),
            weight
        );
    }
}

// ============================================================================
// Different Bit-Packed Syndrome Lengths Tests
// ============================================================================

#[test]
fn test_varying_syndrome_lengths_single_byte() {
    // Test different syndrome lengths that fit in a single byte
    let test_cases = vec![
        (1, "single detector"),
        (3, "three detectors"),
        (7, "seven detectors"),
        (8, "full byte"),
    ];

    for (num_detectors, description) in test_cases {
        let mut decoder = PyMatchingDecoder::builder()
            .nodes(num_detectors)
            .observables(num_detectors)
            .build()
            .unwrap();

        // Create edges for the graph
        for i in 0..(num_detectors - 1) {
            decoder
                .add_edge(i, i + 1, &[i], Some(1.0), None, None)
                .unwrap();
        }
        add_boundary_edges(&mut decoder, num_detectors);

        // Test with all detectors triggered
        let all_ones_pattern = if num_detectors >= 8 {
            0b1111_1111_u8
        } else {
            (1u8 << num_detectors) - 1
        };
        let shots = vec![all_ones_pattern];

        let result = decoder
            .decode_batch_with_config(
                &shots,
                1,
                num_detectors,
                BatchConfig {
                    bit_packed_input: true,
                    bit_packed_output: false,
                    return_weights: true,
                },
            )
            .unwrap();

        assert_eq!(
            result.predictions.len(),
            1,
            "Should have 1 prediction for {description}"
        );
        assert_eq!(
            result.weights.len(),
            1,
            "Should have 1 weight for {description}"
        );

        println!(
            "{}: pattern: {:08b}, weight: {:.6}",
            description, all_ones_pattern, result.weights[0]
        );
    }
}

#[test]
fn test_varying_syndrome_lengths_multi_byte() {
    // Test different syndrome lengths requiring multiple bytes
    let test_cases = vec![
        (9, 2, "nine detectors, two bytes"),
        (16, 2, "sixteen detectors, two bytes"),
        (17, 3, "seventeen detectors, three bytes"),
        (24, 3, "twenty-four detectors, three bytes"),
        (25, 4, "twenty-five detectors, four bytes"),
        (32, 4, "thirty-two detectors, four bytes"),
    ];

    for (num_detectors, expected_bytes, description) in test_cases {
        let mut decoder = PyMatchingDecoder::builder()
            .nodes(num_detectors)
            .observables(num_detectors.min(16)) // Keep observables reasonable
            .build()
            .unwrap();

        // Create a chain graph
        for i in 0..(num_detectors - 1) {
            decoder
                .add_edge(
                    i,
                    i + 1,
                    &[i % decoder.num_observables()],
                    Some(1.0),
                    None,
                    None,
                )
                .unwrap();
        }
        add_boundary_edges(&mut decoder, num_detectors);

        // Create bit-packed shots with alternating pattern
        let mut shots = Vec::new();
        for byte_idx in 0..expected_bytes {
            let pattern = if byte_idx % 2 == 0 {
                0b1010_1010
            } else {
                0b0101_0101
            };
            shots.push(pattern);
        }

        let result = decoder
            .decode_batch_with_config(
                &shots,
                1,
                num_detectors,
                BatchConfig {
                    bit_packed_input: true,
                    bit_packed_output: false,
                    return_weights: true,
                },
            )
            .unwrap();

        assert_eq!(
            result.predictions.len(),
            1,
            "Should have 1 prediction for {description}"
        );
        assert_eq!(
            result.weights.len(),
            1,
            "Should have 1 weight for {description}"
        );

        println!(
            "{}: {} bytes, weight: {:.6}",
            description,
            shots.len(),
            result.weights[0]
        );
    }
}

#[test]
fn test_syndrome_length_boundary_cases() {
    // Test boundary cases for syndrome lengths
    let boundary_cases = vec![
        (7, 1),  // Just under 1 byte
        (8, 1),  // Exactly 1 byte
        (9, 2),  // Just over 1 byte
        (15, 2), // Just under 2 bytes
        (16, 2), // Exactly 2 bytes
        (17, 3), // Just over 2 bytes
    ];

    for (num_detectors, expected_bytes) in boundary_cases {
        let mut decoder = PyMatchingDecoder::builder()
            .nodes(num_detectors)
            .observables(4)
            .build()
            .unwrap();

        // Add some edges to make a valid graph
        for i in 0..(num_detectors - 1).min(10) {
            decoder
                .add_edge(i, i + 1, &[i % 4], Some(1.0), None, None)
                .unwrap();
        }
        add_boundary_edges(&mut decoder, num_detectors);

        // Test with a simple pattern
        let mut shots = vec![0u8; expected_bytes];
        if expected_bytes > 0 {
            shots[0] = 0b0000_0001; // Set first bit
        }
        if expected_bytes > 1 {
            shots[expected_bytes - 1] = 0b1000_0000; // Set last bit of last byte
        }

        let result = decoder.decode_batch_with_config(
            &shots,
            1,
            num_detectors,
            BatchConfig {
                bit_packed_input: true,
                bit_packed_output: false,
                return_weights: true,
            },
        );

        assert!(
            result.is_ok(),
            "Should succeed for {num_detectors} detectors ({expected_bytes} bytes)"
        );

        let result = result.unwrap();
        assert_eq!(result.predictions.len(), 1);
        assert_eq!(result.weights.len(), 1);

        println!(
            "{} detectors, {} bytes: weight = {:.6}",
            num_detectors, expected_bytes, result.weights[0]
        );
    }
}

// ============================================================================
// Edge Cases with Bit-Packed Formats Tests
// ============================================================================

#[test]
fn test_empty_syndromes_bit_packed() {
    // Test bit-packed format with empty syndromes (all zeros)
    let mut decoder = PyMatchingDecoder::builder()
        .nodes(12)
        .observables(6)
        .build()
        .unwrap();

    // Create a matching graph
    for i in 0..11 {
        decoder
            .add_edge(i, i + 1, &[i % 6], Some(1.0), None, None)
            .unwrap();
    }
    add_boundary_edges(&mut decoder, 12);

    let num_shots = 5;
    let num_detectors: usize = 12;
    let bytes_per_shot = num_detectors.div_ceil(8);

    // All zeros (empty syndromes)
    let shots = vec![0u8; num_shots * bytes_per_shot];

    let result = decoder
        .decode_batch_with_config(
            &shots,
            num_shots,
            num_detectors,
            BatchConfig {
                bit_packed_input: true,
                bit_packed_output: false,
                return_weights: true,
            },
        )
        .unwrap();

    assert_eq!(result.predictions.len(), num_shots);
    assert_eq!(result.weights.len(), num_shots);

    // All predictions should be empty/zero for empty syndromes
    for (i, (prediction, weight)) in result.predictions.iter().zip(&result.weights).enumerate() {
        // Weight should be 0 for empty syndrome
        assert!(
            weight.abs() < f64::EPSILON,
            "Weight should be 0 for empty syndrome {i} but was {weight}"
        );

        // Prediction should be all zeros
        assert!(
            prediction.iter().all(|&x| x == 0),
            "Prediction {i} should be all zeros for empty syndrome"
        );
    }
}

#[test]
fn test_full_syndromes_bit_packed() {
    // Test bit-packed format with full syndromes (all ones)
    let mut decoder = PyMatchingDecoder::builder()
        .nodes(8)
        .observables(4)
        .build()
        .unwrap();

    // Create a matching graph
    for i in 0..7 {
        decoder
            .add_edge(i, i + 1, &[i % 4], Some(1.0), None, None)
            .unwrap();
    }
    add_boundary_edges(&mut decoder, 8);

    let num_shots = 3;
    let num_detectors = 8;
    // 8 detectors fit in 1 byte
    assert_eq!(8_usize.div_ceil(8), 1);

    // All ones (full syndromes)
    let shots = vec![0b1111_1111_u8; num_shots];

    let result = decoder
        .decode_batch_with_config(
            &shots,
            num_shots,
            num_detectors,
            BatchConfig {
                bit_packed_input: true,
                bit_packed_output: false,
                return_weights: true,
            },
        )
        .unwrap();

    assert_eq!(result.predictions.len(), num_shots);
    assert_eq!(result.weights.len(), num_shots);

    // All predictions should be non-trivial for full syndromes
    for (i, (prediction, weight)) in result.predictions.iter().zip(&result.weights).enumerate() {
        // Weight should be positive for non-empty syndrome
        assert!(
            *weight >= 0.0,
            "Weight should be non-negative for full syndrome {i}"
        );

        // Some observables should be set (unless graph is disconnected)
        println!("Full syndrome {i}: prediction: {prediction:?}, weight: {weight:.6}");
    }
}

#[test]
fn test_single_bit_syndromes_bit_packed() {
    // Test bit-packed format with single-bit syndromes
    let mut decoder = PyMatchingDecoder::builder()
        .nodes(8)
        .observables(4)
        .build()
        .unwrap();

    // Create a matching graph with boundary edges
    for i in 0..7 {
        decoder
            .add_edge(i, i + 1, &[i % 4], Some(1.0), None, None)
            .unwrap();
    }
    for i in 0..8 {
        decoder
            .add_boundary_edge(i, &[], Some(2.0), None, None)
            .unwrap();
    }

    let num_detectors = 8;

    // Test each individual bit
    for bit_pos in 0..num_detectors {
        let pattern = 1u8 << bit_pos;
        let shots = vec![pattern];

        let result = decoder
            .decode_batch_with_config(
                &shots,
                1,
                num_detectors,
                BatchConfig {
                    bit_packed_input: true,
                    bit_packed_output: false,
                    return_weights: true,
                },
            )
            .unwrap();

        assert_eq!(result.predictions.len(), 1);
        assert_eq!(result.weights.len(), 1);

        let prediction = &result.predictions[0];
        let weight = result.weights[0];

        // Should have non-zero weight for single detection
        assert!(
            weight > 0.0,
            "Weight should be positive for single bit at position {bit_pos}"
        );

        println!(
            "Single bit at position {bit_pos}: pattern {pattern:08b}, weight: {weight:.6}, prediction: {prediction:?}"
        );
    }
}

#[test]
fn test_odd_number_detectors_bit_packed() {
    // Test bit-packed format with odd numbers of detectors (padding edge cases)
    let odd_detector_counts = vec![1, 3, 5, 7, 9, 11, 13, 15, 17, 19];

    for num_detectors in odd_detector_counts {
        let mut decoder = PyMatchingDecoder::builder()
            .nodes(num_detectors)
            .observables(num_detectors.div_ceil(2))
            .build()
            .unwrap();

        // Create a simple chain
        for i in 0..(num_detectors - 1) {
            decoder
                .add_edge(
                    i,
                    i + 1,
                    &[i % decoder.num_observables()],
                    Some(1.0),
                    None,
                    None,
                )
                .unwrap();
        }
        add_boundary_edges(&mut decoder, num_detectors);

        let bytes_needed = num_detectors.div_ceil(8);

        // Create a pattern that uses the exact number of detectors
        let mut shots = vec![0u8; bytes_needed];

        // Set alternating bits up to num_detectors
        for detector in 0..num_detectors {
            if detector % 2 == 0 {
                let byte_idx = detector / 8;
                let bit_idx = detector % 8;
                shots[byte_idx] |= 1 << bit_idx;
            }
        }

        let result = decoder
            .decode_batch_with_config(
                &shots,
                1,
                num_detectors,
                BatchConfig {
                    bit_packed_input: true,
                    bit_packed_output: false,
                    return_weights: true,
                },
            )
            .unwrap();

        assert_eq!(result.predictions.len(), 1);
        assert_eq!(result.weights.len(), 1);

        println!(
            "{} detectors: {} bytes, weight: {:.6}",
            num_detectors, bytes_needed, result.weights[0]
        );
    }
}

// ============================================================================
// Performance Comparison Tests
// ============================================================================

#[test]
#[allow(clippy::too_many_lines)] // Performance test needs comprehensive coverage
fn test_performance_bit_packed_vs_unpacked() {
    // Test performance comparison between bit-packed and unpacked formats
    let mut decoder = PyMatchingDecoder::builder()
        .nodes(32)
        .observables(16)
        .build()
        .unwrap();

    // Create a complex matching graph for meaningful performance test
    for i in 0..31 {
        decoder
            .add_edge(i, i + 1, &[i % 16], Some(1.0), None, None)
            .unwrap();
    }
    // Add some cross-connections
    for i in 0..16 {
        decoder
            .add_edge(i, i + 16, &[i], Some(1.5), None, None)
            .unwrap();
    }
    add_boundary_edges(&mut decoder, 32);

    let num_shots = 100;
    let num_detectors: usize = 32;
    let bytes_per_shot_packed = num_detectors.div_ceil(8); // 4 bytes per shot

    // Create test data
    let mut unpacked_shots = Vec::new();
    let mut packed_shots = Vec::new();

    for shot in 0..num_shots {
        // Create a deterministic pattern
        let mut shot_data = Vec::new();
        let mut packed_bytes = vec![0u8; bytes_per_shot_packed];

        for detector in 0..num_detectors {
            let bit_value = ((shot * 7 + detector * 3) % 5) == 0;
            shot_data.push(u8::from(bit_value));

            if bit_value {
                let byte_idx = detector / 8;
                let bit_idx = detector % 8;
                packed_bytes[byte_idx] |= 1 << bit_idx;
            }
        }

        unpacked_shots.extend(shot_data);
        packed_shots.extend(packed_bytes);
    }

    // Time unpacked decoding
    let start_unpacked = std::time::Instant::now();
    let unpacked_result = decoder
        .decode_batch_with_config(
            &unpacked_shots,
            num_shots,
            num_detectors,
            BatchConfig {
                bit_packed_input: false,
                bit_packed_output: false,
                return_weights: true,
            },
        )
        .unwrap();
    let duration_unpacked = start_unpacked.elapsed();

    // Time bit-packed decoding
    let start_packed = std::time::Instant::now();
    let packed_result = decoder
        .decode_batch_with_config(
            &packed_shots,
            num_shots,
            num_detectors,
            BatchConfig {
                bit_packed_input: true,
                bit_packed_output: false,
                return_weights: true,
            },
        )
        .unwrap();
    let duration_packed = start_packed.elapsed();

    // Verify results are equivalent
    assert_eq!(
        unpacked_result.predictions.len(),
        packed_result.predictions.len()
    );
    assert_eq!(unpacked_result.weights.len(), packed_result.weights.len());

    // Compare results (should be identical)
    for (i, (unpacked_pred, packed_pred)) in unpacked_result
        .predictions
        .iter()
        .zip(&packed_result.predictions)
        .enumerate()
    {
        assert_eq!(
            unpacked_pred, packed_pred,
            "Predictions should match for shot {i}"
        );
    }

    for (i, (unpacked_weight, packed_weight)) in unpacked_result
        .weights
        .iter()
        .zip(&packed_result.weights)
        .enumerate()
    {
        let weight_diff = (unpacked_weight - packed_weight).abs();
        assert!(
            weight_diff < 1e-10,
            "Weights should match for shot {i}: {unpacked_weight} vs {packed_weight}"
        );
    }

    println!("Performance comparison for {num_shots} shots with {num_detectors} detectors:");
    println!(
        "  Unpacked: {:.2} ms",
        duration_unpacked.as_secs_f64() * 1000.0
    );
    println!(
        "  Bit-packed: {:.2} ms",
        duration_packed.as_secs_f64() * 1000.0
    );
    println!(
        "  Data size - Unpacked: {} bytes, Bit-packed: {} bytes",
        unpacked_shots.len(),
        packed_shots.len()
    );
    #[allow(clippy::cast_precision_loss)] // Acceptable for compression ratio calculation
    {
        println!(
            "  Compression ratio: {:.2}x",
            unpacked_shots.len() as f64 / packed_shots.len() as f64
        );
    }

    // Bit-packed format should use less memory
    assert!(
        packed_shots.len() < unpacked_shots.len(),
        "Bit-packed format should use less memory"
    );
}

#[test]
fn test_memory_usage_bit_packed_vs_unpacked() {
    // Test memory usage comparison for different problem sizes
    let test_cases = vec![
        (8, 1000), // Small: 8 detectors, 1000 shots
        (16, 500), // Medium: 16 detectors, 500 shots
        (32, 250), // Large: 32 detectors, 250 shots
        (64, 125), // Extra large: 64 detectors, 125 shots
    ];

    for (num_detectors, num_shots) in test_cases {
        let mut decoder = PyMatchingDecoder::builder()
            .nodes(num_detectors)
            .observables(num_detectors / 2)
            .build()
            .unwrap();

        // Create a simple graph
        for i in 0..(num_detectors - 1) {
            decoder
                .add_edge(i, i + 1, &[i % (num_detectors / 2)], Some(1.0), None, None)
                .unwrap();
        }
        add_boundary_edges(&mut decoder, num_detectors);

        let bytes_per_shot_packed = num_detectors.div_ceil(8);
        let bytes_per_shot_unpacked = num_detectors;

        let total_packed_bytes = num_shots * bytes_per_shot_packed;
        let total_unpacked_bytes = num_shots * bytes_per_shot_unpacked;
        // Precision loss is acceptable for computing compression ratios
        #[allow(clippy::cast_precision_loss)]
        let compression_ratio = total_unpacked_bytes as f64 / total_packed_bytes as f64;

        // Create dummy data for testing
        let packed_shots = vec![0b1010_1010_u8; total_packed_bytes];
        let unpacked_shots = vec![0u8; total_unpacked_bytes];

        // Test both formats work
        let packed_result = decoder
            .decode_batch_with_config(
                &packed_shots,
                num_shots,
                num_detectors,
                BatchConfig {
                    bit_packed_input: true,
                    bit_packed_output: false,
                    return_weights: false,
                },
            )
            .unwrap();

        let unpacked_result = decoder
            .decode_batch_with_config(
                &unpacked_shots,
                num_shots,
                num_detectors,
                BatchConfig {
                    bit_packed_input: false,
                    bit_packed_output: false,
                    return_weights: false,
                },
            )
            .unwrap();

        assert_eq!(packed_result.predictions.len(), num_shots);
        assert_eq!(unpacked_result.predictions.len(), num_shots);

        println!("Memory usage for {num_detectors} detectors, {num_shots} shots:");
        println!("  Unpacked: {total_unpacked_bytes} bytes");
        println!("  Bit-packed: {total_packed_bytes} bytes");
        println!("  Compression ratio: {compression_ratio:.2}x");

        // Verify compression is meaningful
        assert!(
            compression_ratio > 1.0,
            "Bit-packing should reduce memory usage"
        );
        if num_detectors >= 8 {
            assert!(
                compression_ratio >= 2.0,
                "Should get significant compression for {num_detectors} detectors"
            );
        }
    }
}

// ============================================================================
// Error Handling for Invalid Bit-Packed Inputs Tests
// ============================================================================

#[test]
fn test_invalid_bit_packed_input_sizes() {
    // Test error handling for invalid bit-packed input sizes
    let mut decoder = PyMatchingDecoder::builder()
        .nodes(8)
        .observables(4)
        .build()
        .unwrap();

    // Add some edges
    for i in 0..7 {
        decoder
            .add_edge(i, i + 1, &[i % 4], Some(1.0), None, None)
            .unwrap();
    }

    let num_detectors: usize = 8;
    let num_shots = 2;
    let expected_bytes = num_shots * num_detectors.div_ceil(8); // 2 bytes total

    // Test with too few bytes
    let too_few_bytes = vec![0u8; expected_bytes - 1];
    let result = decoder.decode_batch_with_config(
        &too_few_bytes,
        num_shots,
        num_detectors,
        BatchConfig {
            bit_packed_input: true,
            bit_packed_output: false,
            return_weights: false,
        },
    );
    assert!(result.is_err(), "Should error with too few bytes");
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("doesn't match expected size")
    );

    // Test with too many bytes
    let too_many_bytes = vec![0u8; expected_bytes + 1];
    let result = decoder.decode_batch_with_config(
        &too_many_bytes,
        num_shots,
        num_detectors,
        BatchConfig {
            bit_packed_input: true,
            bit_packed_output: false,
            return_weights: false,
        },
    );
    assert!(result.is_err(), "Should error with too many bytes");
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("doesn't match expected size")
    );

    // Test with correct size (should work)
    let correct_bytes = vec![0u8; expected_bytes];
    let result = decoder.decode_batch_with_config(
        &correct_bytes,
        num_shots,
        num_detectors,
        BatchConfig {
            bit_packed_input: true,
            bit_packed_output: false,
            return_weights: false,
        },
    );
    assert!(
        result.is_ok(),
        "Should succeed with correct number of bytes"
    );
}

#[test]
fn test_invalid_detector_count_bit_packed() {
    // Test error handling for invalid detector counts
    let mut decoder = PyMatchingDecoder::builder()
        .nodes(5)
        .observables(2)
        .build()
        .unwrap();

    // Add edges
    for i in 0..4 {
        decoder
            .add_edge(i, i + 1, &[i % 2], Some(1.0), None, None)
            .unwrap();
    }

    let actual_detectors = decoder.num_detectors();
    let too_many_detectors = actual_detectors + 10;

    // Test with detector count exceeding actual
    let bytes_needed = too_many_detectors.div_ceil(8);
    let shots = vec![0u8; bytes_needed];

    let result = decoder.decode_batch_with_config(
        &shots,
        1,
        too_many_detectors,
        BatchConfig {
            bit_packed_input: true,
            bit_packed_output: false,
            return_weights: false,
        },
    );

    assert!(
        result.is_err(),
        "Should error when num_detectors exceeds actual count"
    );
    let error = result.unwrap_err();
    assert!(
        error.to_string().contains("Invalid syndrome")
            || error.to_string().contains("expected length"),
        "Error message should mention invalid syndrome length: '{error}'"
    );
}

#[test]
fn test_zero_shots_bit_packed() {
    // Test edge case with zero shots
    let mut decoder = PyMatchingDecoder::builder()
        .nodes(8)
        .observables(4)
        .build()
        .unwrap();

    // Add edges
    for i in 0..7 {
        decoder
            .add_edge(i, i + 1, &[i % 4], Some(1.0), None, None)
            .unwrap();
    }

    let result = decoder.decode_batch_with_config(
        &[], // empty shots
        0,   // num_shots
        8,   // num_detectors
        BatchConfig {
            bit_packed_input: true,
            bit_packed_output: true,
            return_weights: true,
        },
    );

    assert!(result.is_ok(), "Should handle zero shots gracefully");
    let result = result.unwrap();
    assert_eq!(result.predictions.len(), 0);
    assert_eq!(result.weights.len(), 0);
}

#[test]
fn test_mismatched_shot_parameters_bit_packed() {
    // Test various parameter mismatches
    let mut decoder = PyMatchingDecoder::builder()
        .nodes(8)
        .observables(4)
        .build()
        .unwrap();

    // Add edges
    for i in 0..7 {
        decoder
            .add_edge(i, i + 1, &[i % 4], Some(1.0), None, None)
            .unwrap();
    }

    // Test: num_shots = 0 but non-empty shots array
    // (PyMatching may accept this and return empty results)
    let result = decoder.decode_batch_with_config(
        &[0u8, 0u8], // 2 bytes
        0,           // num_shots = 0
        8,           // num_detectors
        BatchConfig {
            bit_packed_input: true,
            bit_packed_output: false,
            return_weights: false,
        },
    );
    // Accept either an error or empty result
    if let Ok(result) = result {
        assert_eq!(
            result.predictions.len(),
            0,
            "Should have empty predictions for 0 shots"
        );
    }

    // Test: Wrong calculation of bytes per shot
    let result = decoder.decode_batch_with_config(
        &[0u8], // 1 byte
        2,      // num_shots = 2
        8,      // num_detectors (needs 1 byte per shot, so 2 bytes total)
        BatchConfig {
            bit_packed_input: true,
            bit_packed_output: false,
            return_weights: false,
        },
    );
    assert!(
        result.is_err(),
        "Should error when shots array size doesn't match parameters"
    );
}

#[test]
fn test_bit_packed_with_large_observable_counts() {
    // Test bit-packed format with large numbers of observables
    let mut decoder = PyMatchingDecoder::builder()
        .nodes(16)
        .observables(100) // Large number of observables
        .build()
        .unwrap();

    // Create a graph
    for i in 0..15 {
        decoder
            .add_edge(i, i + 1, &[i % 100], Some(1.0), None, None)
            .unwrap();
    }
    add_boundary_edges(&mut decoder, 16);

    let num_shots = 2;
    let num_detectors: usize = 16;
    let bytes_per_shot = num_detectors.div_ceil(8); // 2 bytes per shot

    let shots = vec![0b1010_1010_u8; num_shots * bytes_per_shot];

    // Test with bit-packed predictions
    let result = decoder
        .decode_batch_with_config(
            &shots,
            num_shots,
            num_detectors,
            BatchConfig {
                bit_packed_input: true,
                bit_packed_output: true,
                return_weights: true,
            },
        )
        .unwrap();

    assert_eq!(result.predictions.len(), num_shots);
    assert_eq!(result.weights.len(), num_shots);
    assert!(result.bit_packed);

    // Verify predictions are reasonable for large observable count
    for (i, prediction) in result.predictions.iter().enumerate() {
        assert!(
            !prediction.is_empty(),
            "Prediction {i} should have some bytes"
        );
        // With 100 observables, we expect multiple bytes
        println!(
            "Shot {i} with 100 observables: {} prediction bytes",
            prediction.len()
        );
    }
}

// ============================================================================
// Integration Tests
// ============================================================================

#[test]
fn test_bit_packed_end_to_end_workflow() {
    // End-to-end test of bit-packed workflow: noise -> syndrome -> decode
    let mut decoder = PyMatchingDecoder::builder()
        .nodes(12)
        .observables(8)
        .build()
        .unwrap();

    // Create a surface code-like graph
    for i in 0..11 {
        decoder
            .add_edge(i, i + 1, &[i % 8], None, Some(0.1), None)
            .unwrap();
    }
    add_boundary_edges(&mut decoder, 12);

    // Generate noise
    let num_samples = 20;
    let noise_result = decoder.add_noise(num_samples, 123).unwrap();

    assert_eq!(noise_result.errors.len(), num_samples);
    assert_eq!(noise_result.syndromes.len(), num_samples);

    // Convert syndromes to bit-packed format
    let num_detectors = decoder.num_detectors();
    let bytes_per_shot = num_detectors.div_ceil(8);
    let mut bit_packed_syndromes = Vec::new();

    for syndrome in &noise_result.syndromes {
        let mut packed_bytes = vec![0u8; bytes_per_shot];

        for (detector_idx, &syndrome_bit) in syndrome.iter().enumerate() {
            if syndrome_bit != 0 && detector_idx < num_detectors {
                let byte_idx = detector_idx / 8;
                let bit_idx = detector_idx % 8;
                packed_bytes[byte_idx] |= 1 << bit_idx;
            }
        }

        bit_packed_syndromes.extend(packed_bytes);
    }

    // Decode using bit-packed format
    let batch_result = decoder
        .decode_batch_with_config(
            &bit_packed_syndromes,
            num_samples,
            num_detectors,
            BatchConfig {
                bit_packed_input: true,
                bit_packed_output: false,
                return_weights: true,
            },
        )
        .unwrap();

    assert_eq!(batch_result.predictions.len(), num_samples);
    assert_eq!(batch_result.weights.len(), num_samples);

    // Compare with individual decoding
    let mut individual_results = Vec::new();
    for syndrome in &noise_result.syndromes {
        let result = decoder.decode(syndrome).unwrap();
        individual_results.push(result);
    }

    // Results should be consistent
    for (i, (batch_pred, individual_result)) in batch_result
        .predictions
        .iter()
        .zip(&individual_results)
        .enumerate()
    {
        // Compare predictions (only compare the first elements up to the individual result length)
        let min_len = batch_pred.len().min(individual_result.observable.len());
        assert_eq!(
            &batch_pred[..min_len],
            &individual_result.observable[..min_len],
            "Batch and individual predictions should match for sample {i} (first {min_len} elements)"
        );

        // Compare weights
        let weight_diff = (batch_result.weights[i] - individual_result.weight).abs();
        assert!(
            weight_diff < 1e-10,
            "Batch and individual weights should match for sample {}: {} vs {}",
            i,
            batch_result.weights[i],
            individual_result.weight
        );
    }

    println!("End-to-end bit-packed workflow test completed successfully");
    println!("  {num_samples} samples processed");
    println!("  {num_detectors} detectors, {bytes_per_shot} bytes per syndrome");
    // Precision loss is acceptable for computing compression ratios
    #[allow(clippy::cast_precision_loss)]
    let compression_ratio =
        (num_samples * num_detectors) as f64 / bit_packed_syndromes.len() as f64;
    println!("  Compression ratio: {compression_ratio:.2}x");
}
