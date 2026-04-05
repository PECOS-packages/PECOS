//! Integration tests for the pecos-relay-bp decoder wrapper.
//!
//! Tests verify correctness (syndrome consistency), configuration variations,
//! decoder reuse, batch decoding, error handling, and trait implementations.

use ndarray::{Array1, Array2, ArrayView2};
use pecos_relay_bp::{
    DecodingResult, MinSumBpBuilder, MinSumBpDecoder, MinSumConfig, RelayBpBuilder, RelayBpDecoder,
    RelayConfig, StoppingCriterion,
};

// ============================================================================
// Test helpers
// ============================================================================

/// Compute syndrome = H * error (mod 2) using PECOS ndarray types.
fn compute_syndrome(h: &ArrayView2<u8>, error: &Array1<u8>) -> Array1<u8> {
    let nrows = h.nrows();
    let mut syndrome = Array1::zeros(nrows);
    for i in 0..nrows {
        let mut val: u8 = 0;
        for j in 0..h.ncols() {
            val ^= h[[i, j]] & error[j];
        }
        syndrome[i] = val;
    }
    syndrome
}

/// [3,1,3] repetition code: H = [[1,1,0],[0,1,1]]
fn repetition_3_matrix() -> Array2<u8> {
    Array2::from_shape_vec((2, 3), vec![1, 1, 0, 0, 1, 1]).unwrap()
}

/// [5,1,5] repetition code
fn repetition_5_matrix() -> Array2<u8> {
    #[rustfmt::skip]
    let data = vec![
        1, 1, 0, 0, 0,
        0, 1, 1, 0, 0,
        0, 0, 1, 1, 0,
        0, 0, 0, 1, 1,
    ];
    Array2::from_shape_vec((4, 5), data).unwrap()
}

/// [7,4,3] Hamming code parity check matrix
fn hamming_7_4_matrix() -> Array2<u8> {
    #[rustfmt::skip]
    let data = vec![
        1, 0, 1, 0, 1, 0, 1,
        0, 1, 1, 0, 0, 1, 1,
        0, 0, 0, 1, 1, 1, 1,
    ];
    Array2::from_shape_vec((3, 7), data).unwrap()
}

fn default_priors(n: usize) -> Vec<f64> {
    vec![0.05; n]
}

// ============================================================================
// Syndrome correctness: verify H * decoded_error mod 2 == syndrome
// ============================================================================

#[test]
fn min_sum_decoding_satisfies_syndrome_repetition_3() {
    let h = repetition_3_matrix();
    let config = MinSumConfig::new(default_priors(3));
    let mut decoder = MinSumBpDecoder::new(&h.view(), &config).unwrap();

    // All single-bit error patterns for the repetition code
    let errors: Vec<Array1<u8>> = vec![
        Array1::from_vec(vec![1, 0, 0]),
        Array1::from_vec(vec![0, 1, 0]),
        Array1::from_vec(vec![0, 0, 1]),
    ];

    for error in &errors {
        let syndrome = compute_syndrome(&h.view(), error);
        let result = decoder.decode(&syndrome.view()).unwrap();

        // The decoded error must produce the same syndrome
        let recomputed = compute_syndrome(&h.view(), &result.decoding);
        assert_eq!(
            syndrome, recomputed,
            "Syndrome mismatch for error {error:?}: expected {syndrome:?}, got {recomputed:?}"
        );
        assert!(result.converged);
    }
}

#[test]
fn relay_decoding_satisfies_syndrome_repetition_3() {
    let h = repetition_3_matrix();
    let ms_config = MinSumConfig::new(default_priors(3));
    let relay_config = RelayConfig::default();
    let mut decoder = RelayBpDecoder::new(&h.view(), &ms_config, &relay_config).unwrap();

    let errors: Vec<Array1<u8>> = vec![
        Array1::from_vec(vec![1, 0, 0]),
        Array1::from_vec(vec![0, 1, 0]),
        Array1::from_vec(vec![0, 0, 1]),
    ];

    for error in &errors {
        let syndrome = compute_syndrome(&h.view(), error);
        let result = decoder.decode(&syndrome.view()).unwrap();

        let recomputed = compute_syndrome(&h.view(), &result.decoding);
        assert_eq!(
            syndrome, recomputed,
            "Syndrome mismatch for error {error:?}"
        );
        assert!(result.converged);
    }
}

#[test]
fn min_sum_decoding_satisfies_syndrome_repetition_5() {
    let h = repetition_5_matrix();
    let config = MinSumConfig::new(default_priors(5));
    let mut decoder = MinSumBpDecoder::new(&h.view(), &config).unwrap();

    // Test all single-bit errors
    for bit in 0..5 {
        let mut error = Array1::zeros(5);
        error[bit] = 1;
        let syndrome = compute_syndrome(&h.view(), &error);
        let result = decoder.decode(&syndrome.view()).unwrap();

        let recomputed = compute_syndrome(&h.view(), &result.decoding);
        assert_eq!(
            syndrome, recomputed,
            "Syndrome mismatch for single-bit error on bit {bit}"
        );
        assert!(result.converged);
    }
}

#[test]
fn min_sum_decoding_satisfies_syndrome_hamming_7_4() {
    // Note: BP may not converge for all patterns on the Hamming code due to
    // short cycles in its Tanner graph. We verify syndrome consistency only
    // when the decoder reports convergence, and require that at least some
    // single-bit errors are decoded correctly.
    let h = hamming_7_4_matrix();
    let config = MinSumConfig::new(default_priors(7));
    let mut decoder = MinSumBpDecoder::new(&h.view(), &config).unwrap();

    assert_eq!(decoder.check_count(), 3);
    assert_eq!(decoder.bit_count(), 7);

    let mut converged_count = 0;
    for bit in 0..7 {
        let mut error = Array1::zeros(7);
        error[bit] = 1;
        let syndrome = compute_syndrome(&h.view(), &error);
        let result = decoder.decode(&syndrome.view()).unwrap();

        if result.converged {
            let recomputed = compute_syndrome(&h.view(), &result.decoding);
            assert_eq!(
                syndrome, recomputed,
                "Hamming code syndrome mismatch for error on bit {bit}"
            );
            converged_count += 1;
        }
    }
    // BP should converge for most single-bit errors
    assert!(
        converged_count >= 4,
        "Expected at least 4/7 converged, got {converged_count}"
    );
}

#[test]
fn relay_decoding_satisfies_syndrome_hamming_7_4() {
    let h = hamming_7_4_matrix();
    let ms_config = MinSumConfig::new(default_priors(7));
    let relay_config = RelayConfig {
        num_sets: 50,
        ..Default::default()
    };
    let mut decoder = RelayBpDecoder::new(&h.view(), &ms_config, &relay_config).unwrap();

    let mut converged_count = 0;
    for bit in 0..7 {
        let mut error = Array1::zeros(7);
        error[bit] = 1;
        let syndrome = compute_syndrome(&h.view(), &error);
        let result = decoder.decode(&syndrome.view()).unwrap();

        if result.converged {
            let recomputed = compute_syndrome(&h.view(), &result.decoding);
            assert_eq!(
                syndrome, recomputed,
                "Relay Hamming code syndrome mismatch for error on bit {bit}"
            );
            converged_count += 1;
        }
    }
    assert!(
        converged_count >= 4,
        "Expected at least 4/7 converged, got {converged_count}"
    );
}

// ============================================================================
// Zero syndrome
// ============================================================================

#[test]
fn zero_syndrome_produces_zero_decoding() {
    let h = repetition_5_matrix();
    let config = MinSumConfig::new(default_priors(5));
    let mut decoder = MinSumBpDecoder::new(&h.view(), &config).unwrap();

    let syndrome = Array1::zeros(4);
    let result = decoder.decode(&syndrome.view()).unwrap();

    assert!(
        result.decoding.iter().all(|&x| x == 0),
        "Zero syndrome should produce zero decoding, got {:?}",
        result.decoding
    );
    assert!(result.converged);
}

#[test]
fn relay_zero_syndrome_produces_zero_decoding() {
    let h = hamming_7_4_matrix();
    let ms_config = MinSumConfig::new(default_priors(7));
    let relay_config = RelayConfig::default();
    let mut decoder = RelayBpDecoder::new(&h.view(), &ms_config, &relay_config).unwrap();

    let syndrome = Array1::zeros(3);
    let result = decoder.decode(&syndrome.view()).unwrap();

    assert!(
        result.decoding.iter().all(|&x| x == 0),
        "Zero syndrome should produce zero decoding, got {:?}",
        result.decoding
    );
}

// ============================================================================
// Sequential decode stability: same decoder, multiple calls
// ============================================================================

#[test]
fn min_sum_sequential_decodes_are_independent() {
    let h = repetition_3_matrix();
    let config = MinSumConfig::new(default_priors(3));
    let mut decoder = MinSumBpDecoder::new(&h.view(), &config).unwrap();

    // Decode several different syndromes in sequence
    let syndromes = vec![
        Array1::from_vec(vec![1u8, 0]),
        Array1::from_vec(vec![0u8, 1]),
        Array1::from_vec(vec![1u8, 1]),
        Array1::from_vec(vec![0u8, 0]),
        Array1::from_vec(vec![1u8, 0]), // repeat first
    ];

    let mut results: Vec<DecodingResult> = Vec::new();
    for s in &syndromes {
        let result = decoder.decode(&s.view()).unwrap();
        let recomputed = compute_syndrome(&h.view(), &result.decoding);
        assert_eq!(*s, recomputed, "Syndrome mismatch on sequential decode");
        results.push(result);
    }

    // The first and last syndromes are identical, so their decodings must
    // produce the same syndrome (they may differ if there are degenerate
    // solutions, but syndrome consistency must hold).
    let s_first = compute_syndrome(&h.view(), &results[0].decoding);
    let s_last = compute_syndrome(&h.view(), &results[4].decoding);
    assert_eq!(s_first, s_last);
}

#[test]
fn relay_sequential_decodes_are_independent() {
    let h = repetition_3_matrix();
    let ms_config = MinSumConfig::new(default_priors(3));
    let relay_config = RelayConfig {
        num_sets: 20,
        seed: 42,
        ..Default::default()
    };
    let mut decoder = RelayBpDecoder::new(&h.view(), &ms_config, &relay_config).unwrap();

    for _ in 0..10 {
        let syndrome = Array1::from_vec(vec![1u8, 0]);
        let result = decoder.decode(&syndrome.view()).unwrap();
        let recomputed = compute_syndrome(&h.view(), &result.decoding);
        assert_eq!(syndrome, recomputed);
    }
}

// ============================================================================
// Configuration variations
// ============================================================================

#[test]
fn min_sum_with_alpha_scaling() {
    let h = hamming_7_4_matrix();
    let mut config = MinSumConfig::new(default_priors(7));
    config.alpha = Some(0.0);
    config.alpha_iteration_scaling_factor = 0.0;
    let mut decoder = MinSumBpDecoder::new(&h.view(), &config).unwrap();

    let mut error = Array1::zeros(7);
    error[3] = 1;
    let syndrome = compute_syndrome(&h.view(), &error);
    let result = decoder.decode(&syndrome.view()).unwrap();

    let recomputed = compute_syndrome(&h.view(), &result.decoding);
    assert_eq!(syndrome, recomputed);
}

#[test]
fn min_sum_with_memory_bp() {
    let h = hamming_7_4_matrix();
    let mut config = MinSumConfig::new(default_priors(7));
    config.gamma0 = Some(0.15);
    let mut decoder = MinSumBpDecoder::new(&h.view(), &config).unwrap();

    let mut error = Array1::zeros(7);
    error[0] = 1;
    let syndrome = compute_syndrome(&h.view(), &error);
    let result = decoder.decode(&syndrome.view()).unwrap();

    let recomputed = compute_syndrome(&h.view(), &result.decoding);
    assert_eq!(syndrome, recomputed);
}

#[test]
fn min_sum_low_max_iter() {
    let h = repetition_3_matrix();
    let mut config = MinSumConfig::new(default_priors(3));
    config.max_iter = 1;
    let mut decoder = MinSumBpDecoder::new(&h.view(), &config).unwrap();

    // Even with 1 iteration, BP on a simple code should work
    let syndrome = Array1::from_vec(vec![1u8, 0]);
    let result = decoder.decode(&syndrome.view()).unwrap();
    assert!(result.iterations <= 1);
}

#[test]
fn relay_with_stopping_criterion_all() {
    let h = repetition_3_matrix();
    let ms_config = MinSumConfig::new(default_priors(3));
    let relay_config = RelayConfig {
        num_sets: 5,
        stopping_criterion: StoppingCriterion::All,
        seed: 123,
        ..Default::default()
    };
    let mut decoder = RelayBpDecoder::new(&h.view(), &ms_config, &relay_config).unwrap();

    let syndrome = Array1::from_vec(vec![1u8, 0]);
    let result = decoder.decode(&syndrome.view()).unwrap();
    let recomputed = compute_syndrome(&h.view(), &result.decoding);
    assert_eq!(syndrome, recomputed);
}

#[test]
fn relay_with_stopping_criterion_pre_iter() {
    let h = repetition_3_matrix();
    let ms_config = MinSumConfig::new(default_priors(3));
    let relay_config = RelayConfig {
        pre_iter: 50,
        num_sets: 10,
        stopping_criterion: StoppingCriterion::PreIter,
        ..Default::default()
    };
    let mut decoder = RelayBpDecoder::new(&h.view(), &ms_config, &relay_config).unwrap();

    let syndrome = Array1::from_vec(vec![0u8, 1]);
    let result = decoder.decode(&syndrome.view()).unwrap();
    let recomputed = compute_syndrome(&h.view(), &result.decoding);
    assert_eq!(syndrome, recomputed);
}

#[test]
fn relay_with_custom_gamma_interval() {
    let h = hamming_7_4_matrix();
    let ms_config = MinSumConfig::new(default_priors(7));
    let relay_config = RelayConfig {
        gamma_dist_interval: (-0.5, 1.0),
        num_sets: 30,
        seed: 7,
        ..Default::default()
    };
    let mut decoder = RelayBpDecoder::new(&h.view(), &ms_config, &relay_config).unwrap();

    let mut error = Array1::zeros(7);
    error[5] = 1;
    let syndrome = compute_syndrome(&h.view(), &error);
    let result = decoder.decode(&syndrome.view()).unwrap();

    let recomputed = compute_syndrome(&h.view(), &result.decoding);
    assert_eq!(syndrome, recomputed);
}

#[test]
fn different_error_priors() {
    let h = repetition_3_matrix();
    // Non-uniform priors: middle bit has higher error probability
    let config = MinSumConfig::new(vec![0.01, 0.3, 0.01]);
    let mut decoder = MinSumBpDecoder::new(&h.view(), &config).unwrap();

    // Syndrome [1, 1] is ambiguous (could be bit 0+2 or bit 1 alone).
    // With high prior on bit 1, the decoder should prefer the single-bit solution.
    let syndrome = Array1::from_vec(vec![1u8, 1]);
    let result = decoder.decode(&syndrome.view()).unwrap();

    let recomputed = compute_syndrome(&h.view(), &result.decoding);
    assert_eq!(syndrome, recomputed);

    // With high prior on bit 1, expect bit 1 alone to be decoded
    assert_eq!(result.decoding[1], 1, "Expected bit 1 decoded as error");
    assert_eq!(
        result.decoding.iter().filter(|&&x| x == 1).count(),
        1,
        "Expected single-bit decoding"
    );
}

// ============================================================================
// Batch decoding correctness
// ============================================================================

#[test]
fn batch_decoding_matches_sequential() {
    use pecos_decoder_core::BatchDecoder;

    let h = repetition_3_matrix();
    let config = MinSumConfig::new(default_priors(3));
    let mut decoder = MinSumBpDecoder::new(&h.view(), &config).unwrap();

    let syndromes = [
        Array1::from_vec(vec![1u8, 0]),
        Array1::from_vec(vec![0u8, 1]),
        Array1::from_vec(vec![1u8, 1]),
        Array1::from_vec(vec![0u8, 0]),
    ];

    let views: Vec<_> = syndromes.iter().map(|s| s.view()).collect();
    let batch_results = decoder.decode_batch(&views).unwrap();

    assert_eq!(batch_results.len(), 4);

    // Each batch result must satisfy syndrome consistency
    for (s, r) in syndromes.iter().zip(&batch_results) {
        let recomputed = compute_syndrome(&h.view(), &r.decoding);
        assert_eq!(*s, recomputed, "Batch decode syndrome mismatch");
    }
}

// ============================================================================
// Builder API
// ============================================================================

#[test]
fn relay_builder_all_options() {
    let h = hamming_7_4_matrix();
    let mut decoder = RelayBpBuilder::new(&h.view())
        .error_priors(&default_priors(7))
        .max_iter(150)
        .alpha(Some(0.0))
        .alpha_iteration_scaling_factor(0.0)
        .gamma0(Some(0.1))
        .pre_iter(60)
        .num_sets(50)
        .set_max_iter(40)
        .gamma_dist_interval((-0.3, 0.7))
        .stopping_criterion(StoppingCriterion::NConv { stop_after: 2 })
        .seed(99)
        .build()
        .unwrap();

    let mut error = Array1::zeros(7);
    error[2] = 1;
    let syndrome = compute_syndrome(&h.view(), &error);
    let result = decoder.decode(&syndrome.view()).unwrap();
    let recomputed = compute_syndrome(&h.view(), &result.decoding);
    assert_eq!(syndrome, recomputed);
}

#[test]
fn min_sum_builder_all_options() {
    let h = hamming_7_4_matrix();
    let mut decoder = MinSumBpBuilder::new(&h.view())
        .error_priors(&default_priors(7))
        .max_iter(100)
        .alpha(Some(0.5))
        .alpha_iteration_scaling_factor(0.9)
        .gamma0(Some(0.2))
        .build()
        .unwrap();

    let mut error = Array1::zeros(7);
    error[4] = 1;
    let syndrome = compute_syndrome(&h.view(), &error);
    let result = decoder.decode(&syndrome.view()).unwrap();
    let recomputed = compute_syndrome(&h.view(), &result.decoding);
    assert_eq!(syndrome, recomputed);
}

// ============================================================================
// CheckMatrixDecoder trait: dense and sparse construction
// ============================================================================

#[test]
fn check_matrix_decoder_from_dense() {
    use pecos_decoder_core::{CheckMatrixConfig, CheckMatrixDecoder, Decoder};

    let h = hamming_7_4_matrix();
    let config = CheckMatrixConfig {
        weights: Some(default_priors(7)),
        ..Default::default()
    };

    let mut decoder = MinSumBpDecoder::from_dense_matrix_with_config(&h.view(), config).unwrap();
    assert_eq!(decoder.check_count(), 3);
    assert_eq!(decoder.bit_count(), 7);

    // Use bit 0 which BP handles reliably on this code
    let mut error = Array1::zeros(7);
    error[0] = 1;
    let syndrome = compute_syndrome(&h.view(), &error);
    let result = <MinSumBpDecoder as Decoder>::decode(&mut decoder, &syndrome.view()).unwrap();
    let recomputed = compute_syndrome(&h.view(), &result.decoding);
    assert_eq!(syndrome, recomputed);
}

#[test]
fn check_matrix_decoder_from_sparse() {
    use pecos_decoder_core::{CheckMatrixConfig, CheckMatrixDecoder, Decoder};

    let h = hamming_7_4_matrix();

    // Convert to COO sparse format
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    for ((r, c), &v) in h.indexed_iter() {
        if v != 0 {
            rows.push(r);
            cols.push(c);
        }
    }
    let shape = (h.nrows(), h.ncols());

    let config = CheckMatrixConfig {
        weights: Some(default_priors(7)),
        ..Default::default()
    };

    let mut decoder =
        MinSumBpDecoder::from_sparse_matrix_with_config(rows, cols, shape, config).unwrap();
    assert_eq!(decoder.check_count(), 3);
    assert_eq!(decoder.bit_count(), 7);

    let mut error = Array1::zeros(7);
    error[1] = 1;
    let syndrome = compute_syndrome(&h.view(), &error);
    let result = <MinSumBpDecoder as Decoder>::decode(&mut decoder, &syndrome.view()).unwrap();
    let recomputed = compute_syndrome(&h.view(), &result.decoding);
    assert_eq!(syndrome, recomputed);
}

#[test]
fn check_matrix_decoder_default_priors() {
    use pecos_decoder_core::{CheckMatrixConfig, CheckMatrixDecoder, Decoder};

    // Use default config (no weights specified -> defaults to 0.1)
    let h = repetition_3_matrix();
    let config = CheckMatrixConfig::default();

    let mut decoder = MinSumBpDecoder::from_dense_matrix_with_config(&h.view(), config).unwrap();

    let syndrome = Array1::from_vec(vec![1u8, 0]);
    let result = <MinSumBpDecoder as Decoder>::decode(&mut decoder, &syndrome.view()).unwrap();
    let recomputed = compute_syndrome(&h.view(), &result.decoding);
    assert_eq!(syndrome, recomputed);
}

// ============================================================================
// Error handling
// ============================================================================

#[test]
fn invalid_syndrome_length_min_sum() {
    let h = repetition_3_matrix();
    let config = MinSumConfig::new(default_priors(3));
    let mut decoder = MinSumBpDecoder::new(&h.view(), &config).unwrap();

    // Too long
    let syndrome = Array1::from_vec(vec![1u8, 0, 1]);
    assert!(decoder.decode(&syndrome.view()).is_err());

    // Too short
    let syndrome = Array1::from_vec(vec![1u8]);
    assert!(decoder.decode(&syndrome.view()).is_err());
}

#[test]
fn invalid_syndrome_length_relay() {
    let h = repetition_3_matrix();
    let ms_config = MinSumConfig::new(default_priors(3));
    let relay_config = RelayConfig::default();
    let mut decoder = RelayBpDecoder::new(&h.view(), &ms_config, &relay_config).unwrap();

    let syndrome = Array1::from_vec(vec![1u8, 0, 1, 0]);
    assert!(decoder.decode(&syndrome.view()).is_err());
}

#[test]
fn builder_missing_error_priors() {
    let h = repetition_3_matrix();
    assert!(RelayBpBuilder::new(&h.view()).build().is_err());
    assert!(MinSumBpBuilder::new(&h.view()).build().is_err());
}

#[test]
fn invalid_matrix_empty() {
    let h = Array2::<u8>::zeros((0, 0));
    let config = MinSumConfig::new(vec![]);
    assert!(MinSumBpDecoder::new(&h.view(), &config).is_err());
}

// ============================================================================
// DecodingResult trait
// ============================================================================

#[test]
fn decoding_result_to_standard_conversion() {
    use pecos_decoder_core::DecodingResultTrait;

    let result = DecodingResult {
        decoding: Array1::from_vec(vec![1, 0, 0]),
        converged: true,
        iterations: 7,
    };

    assert!(result.is_successful());
    assert_eq!(result.iterations(), Some(7));
    assert_eq!(result.cost(), None);

    let std = result.to_standard();
    assert_eq!(std.observable, vec![1, 0, 0]);
    assert_eq!(std.converged, Some(true));
    assert_eq!(std.iterations, Some(7));
}

#[test]
fn decoding_result_failed() {
    use pecos_decoder_core::DecodingResultTrait;

    let result = DecodingResult {
        decoding: Array1::from_vec(vec![0, 0, 0]),
        converged: false,
        iterations: 200,
    };

    assert!(!result.is_successful());
}

// ============================================================================
// Exact decoding match (inspired by relay-bp's own tests)
//
// With very low error priors (0.003), the decoder has strong prior information
// and should recover the exact error pattern on simple codes, not just a
// syndrome-equivalent one.
// ============================================================================

#[test]
fn min_sum_exact_decoding_repetition_3_low_priors() {
    let h = repetition_3_matrix();
    let config = MinSumConfig::new(vec![0.003, 0.003, 0.003]);
    let mut decoder = MinSumBpDecoder::new(&h.view(), &config).unwrap();

    // All 4 possible syndrome patterns for the [3,1,3] rep code
    let cases: Vec<(Array1<u8>, Array1<u8>)> = vec![
        (
            Array1::from_vec(vec![0, 0]),
            Array1::from_vec(vec![0, 0, 0]),
        ),
        (
            Array1::from_vec(vec![1, 0]),
            Array1::from_vec(vec![1, 0, 0]),
        ),
        (
            Array1::from_vec(vec![1, 1]),
            Array1::from_vec(vec![0, 1, 0]),
        ),
        (
            Array1::from_vec(vec![0, 1]),
            Array1::from_vec(vec![0, 0, 1]),
        ),
    ];

    for (syndrome, expected_error) in &cases {
        let result = decoder.decode(&syndrome.view()).unwrap();
        assert!(
            result.converged,
            "Failed to converge for syndrome {syndrome:?}"
        );
        assert_eq!(
            result.decoding, *expected_error,
            "Exact decoding mismatch for syndrome {:?}: got {:?}, expected {:?}",
            syndrome, result.decoding, expected_error
        );
    }
}

#[test]
fn relay_exact_decoding_repetition_3_low_priors() {
    let h = repetition_3_matrix();
    let ms_config = MinSumConfig::new(vec![0.003, 0.003, 0.003]);
    let relay_config = RelayConfig {
        num_sets: 20,
        seed: 42,
        ..Default::default()
    };
    let mut decoder = RelayBpDecoder::new(&h.view(), &ms_config, &relay_config).unwrap();

    let cases: Vec<(Array1<u8>, Array1<u8>)> = vec![
        (
            Array1::from_vec(vec![0, 0]),
            Array1::from_vec(vec![0, 0, 0]),
        ),
        (
            Array1::from_vec(vec![1, 0]),
            Array1::from_vec(vec![1, 0, 0]),
        ),
        (
            Array1::from_vec(vec![1, 1]),
            Array1::from_vec(vec![0, 1, 0]),
        ),
        (
            Array1::from_vec(vec![0, 1]),
            Array1::from_vec(vec![0, 0, 1]),
        ),
    ];

    for (syndrome, expected_error) in &cases {
        let result = decoder.decode(&syndrome.view()).unwrap();
        assert!(
            result.converged,
            "Relay failed to converge for syndrome {syndrome:?}"
        );
        assert_eq!(
            result.decoding, *expected_error,
            "Relay exact decoding mismatch for syndrome {:?}: got {:?}, expected {:?}",
            syndrome, result.decoding, expected_error
        );
    }
}

#[test]
fn min_sum_exact_decoding_repetition_5_low_priors() {
    let h = repetition_5_matrix();
    let config = MinSumConfig::new(vec![0.003; 5]);
    let mut decoder = MinSumBpDecoder::new(&h.view(), &config).unwrap();

    // All single-bit error patterns
    for bit in 0..5 {
        let mut expected_error = Array1::zeros(5);
        expected_error[bit] = 1;
        let syndrome = compute_syndrome(&h.view(), &expected_error);
        let result = decoder.decode(&syndrome.view()).unwrap();
        assert!(result.converged, "Failed to converge for bit {bit}");
        assert_eq!(
            result.decoding, expected_error,
            "Exact decoding mismatch for single-bit error on bit {bit}"
        );
    }
}

// ============================================================================
// Relay with num_sets=0: pure BP passthrough mode
//
// When num_sets=0 and stopping_criterion=PreIter, relay runs only the
// initial BP phase (no relay legs). This mirrors relay-bp's own
// `min_sum_decode_repetition_code` test.
// ============================================================================

#[test]
fn relay_num_sets_zero_is_pure_bp_passthrough() {
    let h = repetition_3_matrix();
    let ms_config = MinSumConfig::new(vec![0.003, 0.003, 0.003]);
    let relay_config = RelayConfig {
        pre_iter: 10,
        num_sets: 0,
        stopping_criterion: StoppingCriterion::PreIter,
        ..Default::default()
    };
    let mut decoder = RelayBpDecoder::new(&h.view(), &ms_config, &relay_config).unwrap();

    let cases: Vec<(Array1<u8>, Array1<u8>)> = vec![
        (
            Array1::from_vec(vec![0, 0]),
            Array1::from_vec(vec![0, 0, 0]),
        ),
        (
            Array1::from_vec(vec![1, 0]),
            Array1::from_vec(vec![1, 0, 0]),
        ),
        (
            Array1::from_vec(vec![1, 1]),
            Array1::from_vec(vec![0, 1, 0]),
        ),
        (
            Array1::from_vec(vec![0, 1]),
            Array1::from_vec(vec![0, 0, 1]),
        ),
    ];

    for (syndrome, expected_error) in &cases {
        let result = decoder.decode(&syndrome.view()).unwrap();
        assert!(
            result.converged,
            "Passthrough failed for syndrome {syndrome:?}"
        );
        assert_eq!(
            result.decoding, *expected_error,
            "Passthrough exact mismatch for syndrome {syndrome:?}"
        );
    }
}

// ============================================================================
// MemBP variant (gamma0 parameter)
//
// Tests the memory-enhanced BP variant where gamma0 controls the strength
// of memory effects across iterations.
// ============================================================================

#[test]
fn min_sum_membp_exact_decoding_repetition_3() {
    let h = repetition_3_matrix();
    let mut config = MinSumConfig::new(vec![0.003, 0.003, 0.003]);
    config.gamma0 = Some(0.15);
    let mut decoder = MinSumBpDecoder::new(&h.view(), &config).unwrap();

    let cases: Vec<(Array1<u8>, Array1<u8>)> = vec![
        (
            Array1::from_vec(vec![0, 0]),
            Array1::from_vec(vec![0, 0, 0]),
        ),
        (
            Array1::from_vec(vec![1, 0]),
            Array1::from_vec(vec![1, 0, 0]),
        ),
        (
            Array1::from_vec(vec![1, 1]),
            Array1::from_vec(vec![0, 1, 0]),
        ),
        (
            Array1::from_vec(vec![0, 1]),
            Array1::from_vec(vec![0, 0, 1]),
        ),
    ];

    for (syndrome, expected_error) in &cases {
        let result = decoder.decode(&syndrome.view()).unwrap();
        assert!(result.converged, "MemBP failed for syndrome {syndrome:?}");
        assert_eq!(
            result.decoding, *expected_error,
            "MemBP exact mismatch for syndrome {syndrome:?}"
        );
    }
}

#[test]
fn relay_with_membp_exact_decoding_repetition_3() {
    let h = repetition_3_matrix();
    let mut ms_config = MinSumConfig::new(vec![0.003, 0.003, 0.003]);
    ms_config.gamma0 = Some(0.9);
    let relay_config = RelayConfig {
        num_sets: 20,
        seed: 42,
        ..Default::default()
    };
    let mut decoder = RelayBpDecoder::new(&h.view(), &ms_config, &relay_config).unwrap();

    let cases: Vec<(Array1<u8>, Array1<u8>)> = vec![
        (
            Array1::from_vec(vec![0, 0]),
            Array1::from_vec(vec![0, 0, 0]),
        ),
        (
            Array1::from_vec(vec![1, 0]),
            Array1::from_vec(vec![1, 0, 0]),
        ),
        (
            Array1::from_vec(vec![1, 1]),
            Array1::from_vec(vec![0, 1, 0]),
        ),
        (
            Array1::from_vec(vec![0, 1]),
            Array1::from_vec(vec![0, 0, 1]),
        ),
    ];

    for (syndrome, expected_error) in &cases {
        let result = decoder.decode(&syndrome.view()).unwrap();
        assert!(
            result.converged,
            "Relay+MemBP failed for syndrome {syndrome:?}"
        );
        assert_eq!(
            result.decoding, *expected_error,
            "Relay+MemBP exact mismatch for syndrome {syndrome:?}"
        );
    }
}

// ============================================================================
// Alpha scaling with exact decoding
//
// Tests alpha (min-sum scaling factor) combined with iteration scaling,
// mirroring relay-bp's use of alpha=0 with alpha_iteration_scaling_factor.
// ============================================================================

#[test]
fn min_sum_alpha_zero_exact_decoding_repetition_3() {
    let h = repetition_3_matrix();
    let mut config = MinSumConfig::new(vec![0.003, 0.003, 0.003]);
    config.alpha = Some(0.0);
    config.alpha_iteration_scaling_factor = 0.0;
    let mut decoder = MinSumBpDecoder::new(&h.view(), &config).unwrap();

    // alpha=0 with scaling_factor=0 should still converge on the simple rep code
    for (syndrome, expected_error) in [(vec![1u8, 0], vec![1u8, 0, 0]), (vec![0, 1], vec![0, 0, 1])]
    {
        let s = Array1::from_vec(syndrome);
        let e = Array1::from_vec(expected_error);
        let result = decoder.decode(&s.view()).unwrap();
        assert!(result.converged);
        assert_eq!(result.decoding, e);
    }
}

// ============================================================================
// Seed reproducibility
// ============================================================================

#[test]
fn relay_seed_gives_deterministic_results() {
    let h = hamming_7_4_matrix();
    let priors = default_priors(7);

    let mut error = Array1::zeros(7);
    error[3] = 1;
    let syndrome = compute_syndrome(&h.view(), &error);

    // Run twice with same seed
    let mut results = Vec::new();
    for _ in 0..2 {
        let ms_config = MinSumConfig::new(priors.clone());
        let relay_config = RelayConfig {
            seed: 42,
            num_sets: 30,
            ..Default::default()
        };
        let mut decoder = RelayBpDecoder::new(&h.view(), &ms_config, &relay_config).unwrap();
        let result = decoder.decode(&syndrome.view()).unwrap();
        results.push(result);
    }

    assert_eq!(
        results[0].decoding, results[1].decoding,
        "Same seed should give deterministic results"
    );
    assert_eq!(results[0].iterations, results[1].iterations);
}
