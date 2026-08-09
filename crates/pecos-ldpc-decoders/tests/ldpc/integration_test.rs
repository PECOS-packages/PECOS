//! Integration tests for LDPC decoders

use ndarray::{Array1, arr1, arr2};
use pecos_ldpc_decoders::{
    BpLsdDecoder, BpMethod, BpOsdDecoder, BpSchedule, InputVectorType, OsdMethod,
    SoftInfoBpDecoder, SparseMatrix,
};

fn repetition_code(n: usize) -> SparseMatrix {
    // Create repetition code parity check matrix
    let mut row_indices = Vec::new();
    let mut col_indices = Vec::new();

    // Each check connects adjacent bits
    for i in 0..n - 1 {
        let i_u32 = u32::try_from(i).expect("index too large");
        row_indices.push(i_u32);
        col_indices.push(i_u32);
        row_indices.push(i_u32);
        col_indices.push(u32::try_from(i + 1).expect("index too large"));
    }

    SparseMatrix::from_coo(n - 1, n, row_indices, col_indices).unwrap()
}

#[test]
fn test_sparse_matrix_creation() {
    let dense = arr2(&[[1, 0, 1], [0, 1, 1], [1, 1, 0]]);

    let sparse = SparseMatrix::from_dense(&dense.view());
    assert_eq!(sparse.rows, 3);
    assert_eq!(sparse.cols, 3);
    assert_eq!(sparse.nnz(), 6);

    let reconstructed = sparse.to_dense();
    assert_eq!(reconstructed, dense);
}

#[test]
fn test_bp_tuning_validation_names_invalid_parameter() {
    let pcm = repetition_code(3);
    let too_large = usize::try_from(i32::MAX).unwrap() + 1;
    let make_osd = |max_iter, scaling, osd_order| {
        BpOsdDecoder::new(
            &pcm,
            Some(0.1),
            None,
            max_iter,
            BpMethod::ProductSum,
            BpSchedule::Parallel,
            scaling,
            OsdMethod::OsdCs,
            osd_order,
            InputVectorType::Syndrome,
            None,
            None,
            None,
        )
    };

    assert!(
        make_osd(too_large, 1.0, 0)
            .err()
            .unwrap()
            .to_string()
            .contains("max_iter")
    );
    assert!(
        make_osd(10, f64::NAN, 0)
            .err()
            .unwrap()
            .to_string()
            .contains("ms_scaling_factor")
    );
    assert!(
        make_osd(10, -0.1, 0)
            .err()
            .unwrap()
            .to_string()
            .contains("ms_scaling_factor")
    );
    assert!(
        make_osd(10, 1.0, too_large)
            .err()
            .unwrap()
            .to_string()
            .contains("osd_order")
    );

    let lsd_error = BpLsdDecoder::new(
        &pcm,
        Some(0.1),
        None,
        10,
        BpMethod::ProductSum,
        BpSchedule::Parallel,
        1.0,
        OsdMethod::OsdCs,
        too_large,
        0,
        InputVectorType::Syndrome,
        None,
        None,
        None,
    )
    .err()
    .unwrap()
    .to_string();
    assert!(lsd_error.contains("lsd_order"));
}

#[test]
fn test_repetition_code_decoder() {
    let pcm = repetition_code(5);
    let error_rate = 0.1;

    // Create decoder
    let mut decoder = BpOsdDecoder::new(
        &pcm,
        Some(error_rate),
        None,
        20,
        BpMethod::MinimumSum,
        BpSchedule::Parallel,
        1.0,
        OsdMethod::Off,
        0,
        InputVectorType::Syndrome,
        None, // omp_thread_count
        None, // serial_schedule_order
        None, // random_schedule_seed
    )
    .expect("Failed to create decoder");

    // Test with all-zero syndrome (no error)
    let syndrome = Array1::zeros(4);
    let result = decoder.decode(&syndrome.view()).expect("Decoding failed");

    assert!(result.converged);
    assert_eq!(result.decoding, Array1::<u8>::zeros(5));
}

#[test]
fn test_simple_error_correction() {
    let pcm = repetition_code(3);
    let error_rate = 0.1;

    let mut decoder = BpOsdDecoder::new(
        &pcm,
        Some(error_rate),
        None,
        10,
        BpMethod::ProductSum,
        BpSchedule::Parallel,
        1.0,
        OsdMethod::Osd0,
        0,
        InputVectorType::Syndrome,
        None, // omp_thread_count
        None, // serial_schedule_order
        None, // random_schedule_seed
    )
    .expect("Failed to create decoder");

    // Syndrome for single error in first position
    let syndrome = arr1(&[1, 0]);
    let result = decoder.decode(&syndrome.view()).expect("Decoding failed");

    // Should decode to error in first position
    assert_eq!(result.decoding[0], 1);
    assert_eq!(result.decoding[1], 0);
    assert_eq!(result.decoding[2], 0);
}

#[test]
fn test_bplsd_decoder() {
    let pcm = repetition_code(5);
    let error_rate = 0.1;

    let mut decoder = BpLsdDecoder::new(
        &pcm,
        Some(error_rate),
        None,
        10,
        BpMethod::MinimumSum,
        BpSchedule::Parallel,
        0.625,
        OsdMethod::OsdCs,
        0,
        1,
        InputVectorType::Syndrome,
        None, // omp_thread_count
        None, // serial_schedule_order
        None, // random_schedule_seed
    )
    .expect("Failed to create decoder");

    // Test with simple syndrome
    let syndrome = arr1(&[1, 0, 0, 0]);
    let result = decoder.decode(&syndrome.view()).expect("Decoding failed");

    // Check that we get a valid decoding
    assert_eq!(result.decoding.len(), 5);
}

#[test]
fn test_error_channel() {
    let pcm = repetition_code(3);

    // Different error rates for each bit
    let error_channel = vec![0.05, 0.1, 0.15];

    let mut decoder = BpOsdDecoder::new(
        &pcm,
        None,
        Some(&error_channel),
        10,
        BpMethod::MinimumSum,
        BpSchedule::Parallel,
        1.0,
        OsdMethod::Off,
        0,
        InputVectorType::Syndrome,
        None, // omp_thread_count
        None, // serial_schedule_order
        None, // random_schedule_seed
    )
    .expect("Failed to create decoder");

    let syndrome = Array1::zeros(2);
    let result = decoder.decode(&syndrome.view()).expect("Decoding failed");

    assert!(result.converged);
}

#[test]
fn test_invalid_syndrome_length() {
    let pcm = repetition_code(5);
    let error_rate = 0.1;

    let mut decoder = BpOsdDecoder::new(
        &pcm,
        Some(error_rate),
        None,
        10,
        BpMethod::MinimumSum,
        BpSchedule::Parallel,
        1.0,
        OsdMethod::Off,
        0,
        InputVectorType::Syndrome,
        None, // omp_thread_count
        None, // serial_schedule_order
        None, // random_schedule_seed
    )
    .expect("Failed to create decoder");

    // Wrong syndrome length
    let syndrome = Array1::zeros(10);
    let result = decoder.decode(&syndrome.view());

    assert!(result.is_err());
}

#[test]
fn test_decoder_getters() {
    // Test BP+OSD decoder getters
    let pcm = repetition_code(8);
    let error_rate = 0.05;

    let mut decoder = BpOsdDecoder::new(
        &pcm,
        Some(error_rate),
        None,
        100,
        BpMethod::MinimumSum,
        BpSchedule::Parallel,
        1.0,
        OsdMethod::OsdE,
        2,
        InputVectorType::Syndrome,
        None, // omp_thread_count
        None, // serial_schedule_order
        None, // random_schedule_seed
    )
    .expect("Failed to create decoder");

    // Test basic getters before decoding
    assert_eq!(decoder.check_count(), 7);
    assert_eq!(decoder.bit_count(), 8);
    assert_eq!(decoder.max_iter(), 100);
    assert!((decoder.ms_scaling_factor() - 1.0).abs() < f64::EPSILON);
    assert_eq!(decoder.osd_order(), 2);

    // Test enum getters
    match decoder.bp_method() {
        BpMethod::MinimumSum => (),
        BpMethod::ProductSum => panic!("Expected MinimumSum"),
    }

    match decoder.bp_schedule() {
        BpSchedule::Parallel => (),
        _ => panic!("Expected Parallel"),
    }

    match decoder.osd_method() {
        OsdMethod::OsdE => (),
        _ => panic!("Expected OsdE"),
    }

    // Test channel probs getter
    let channel_probs = decoder.channel_probs();
    assert_eq!(channel_probs.len(), 8);
    assert!(channel_probs.iter().all(|&p| (p - 0.05).abs() < 1e-10));

    // Decode something to test post-decode getters
    let syndrome = Array1::zeros(7);
    let result = decoder.decode(&syndrome.view()).expect("Decoding failed");

    assert!(result.converged);
    assert!(decoder.converged());
    assert_eq!(decoder.iterations(), result.iterations);

    // Test BP decoding getter
    let bp_decoding = decoder.bp_decoding();
    assert_eq!(bp_decoding.len(), 8);

    // Test input vector type getter
    match decoder.input_vector_type() {
        InputVectorType::Syndrome => (),
        _ => panic!("Expected Syndrome"),
    }

    // Test BP+LSD decoder getters
    let mut lsd_decoder = BpLsdDecoder::new(
        &pcm,
        Some(error_rate),
        None,
        50,
        BpMethod::ProductSum,
        BpSchedule::Parallel,
        0.625,
        OsdMethod::OsdCs,
        3,
        2,
        InputVectorType::Syndrome,
        None, // omp_thread_count
        None, // serial_schedule_order
        None, // random_schedule_seed
    )
    .expect("Failed to create LSD decoder");

    // Test LSD-specific getters
    assert_eq!(lsd_decoder.check_count(), 7);
    assert_eq!(lsd_decoder.bit_count(), 8);
    assert_eq!(lsd_decoder.max_iter(), 50);
    assert!((lsd_decoder.ms_scaling_factor() - 0.625).abs() < f64::EPSILON);
    assert_eq!(lsd_decoder.lsd_order(), 3);
    assert_eq!(lsd_decoder.bits_per_step(), 2);

    // Test enum getters
    match lsd_decoder.bp_method() {
        BpMethod::ProductSum => (),
        BpMethod::MinimumSum => panic!("Expected ProductSum"),
    }

    match lsd_decoder.lsd_method() {
        OsdMethod::OsdCs => (),
        _ => panic!("Expected OsdCs"),
    }

    // Decode to test post-decode getters
    let result = lsd_decoder
        .decode(&syndrome.view())
        .expect("Decoding failed");
    assert!(lsd_decoder.converged());
    assert_eq!(lsd_decoder.iterations(), result.iterations);

    // Test input vector type getter
    match lsd_decoder.input_vector_type() {
        InputVectorType::Syndrome => (),
        _ => panic!("Expected Syndrome"),
    }
}

#[test]
fn test_input_vector_types() {
    let pcm = repetition_code(5);
    let error_rate = 0.1;

    // Test with syndrome input
    let mut decoder_syndrome = BpOsdDecoder::new(
        &pcm,
        Some(error_rate),
        None,
        10,
        BpMethod::MinimumSum,
        BpSchedule::Parallel,
        1.0,
        OsdMethod::Off,
        0,
        InputVectorType::Syndrome,
        None, // omp_thread_count
        None, // serial_schedule_order
        None, // random_schedule_seed
    )
    .expect("Failed to create decoder");

    let syndrome = arr1(&[1, 0, 0, 0]);
    let result = decoder_syndrome
        .decode(&syndrome.view())
        .expect("Decoding should succeed");
    assert_eq!(result.decoding.len(), 5);

    // Test with AUTO input - should work with syndrome
    let mut decoder_auto = BpOsdDecoder::new(
        &pcm,
        Some(error_rate),
        None,
        10,
        BpMethod::MinimumSum,
        BpSchedule::Parallel,
        1.0,
        OsdMethod::Off,
        0,
        InputVectorType::Auto,
        None, // omp_thread_count
        None, // serial_schedule_order
        None, // random_schedule_seed
    )
    .expect("Failed to create decoder");

    let result = decoder_auto
        .decode(&syndrome.view())
        .expect("Auto decode should work with syndrome");
    assert_eq!(result.decoding.len(), 5);

    // Test that OSD requires syndrome input
    let decoder_osd_result = BpOsdDecoder::new(
        &pcm,
        Some(error_rate),
        None,
        10,
        BpMethod::MinimumSum,
        BpSchedule::Parallel,
        1.0,
        OsdMethod::OsdE,
        2,
        InputVectorType::ReceivedVector,
        None, // omp_thread_count
        None, // serial_schedule_order
        None, // random_schedule_seed
    );
    assert!(decoder_osd_result.is_err());

    // Test that LSD requires syndrome input
    let lsd_decoder_result = BpLsdDecoder::new(
        &pcm,
        Some(error_rate),
        None,
        10,
        BpMethod::MinimumSum,
        BpSchedule::Parallel,
        1.0,
        OsdMethod::OsdCs,
        2,
        1,
        InputVectorType::ReceivedVector,
        None, // omp_thread_count
        None, // serial_schedule_order
        None, // random_schedule_seed
    );
    assert!(lsd_decoder_result.is_err());
}

#[test]
fn test_schedule_and_thread_control() {
    let pcm = repetition_code(8);
    let error_rate = 0.05;

    // Test with custom thread count
    let decoder = BpOsdDecoder::new(
        &pcm,
        Some(error_rate),
        None,
        20,
        BpMethod::MinimumSum,
        BpSchedule::Parallel,
        1.0,
        OsdMethod::Off,
        0,
        InputVectorType::Syndrome,
        Some(4), // 4 threads
        None,
        None,
    )
    .expect("Failed to create decoder");

    assert_eq!(decoder.omp_thread_count(), 4);
    assert_eq!(decoder.random_schedule_seed(), -1); // Default

    // Test with serial schedule order
    let custom_schedule = vec![7, 6, 5, 4, 3, 2, 1, 0];
    let decoder_serial = BpOsdDecoder::new(
        &pcm,
        Some(error_rate),
        None,
        20,
        BpMethod::MinimumSum,
        BpSchedule::Serial,
        1.0,
        OsdMethod::Off,
        0,
        InputVectorType::Syndrome,
        Some(1), // Serial should use 1 thread
        Some(&custom_schedule),
        None,
    )
    .expect("Failed to create decoder");

    assert_eq!(decoder_serial.bp_schedule(), BpSchedule::Serial);
    assert_eq!(decoder_serial.omp_thread_count(), 1);

    // Test with random schedule
    let decoder_random = BpOsdDecoder::new(
        &pcm,
        Some(error_rate),
        None,
        20,
        BpMethod::MinimumSum,
        BpSchedule::Serial,
        1.0,
        OsdMethod::Off,
        0,
        InputVectorType::Syndrome,
        Some(1),
        None,
        Some(42), // Random seed
    )
    .expect("Failed to create decoder");

    assert_eq!(decoder_random.random_schedule_seed(), 42);

    // Test adaptive iterations
    let decoder_adaptive = BpOsdDecoder::new(
        &pcm,
        Some(error_rate),
        None,
        0, // Adaptive - should become 8 (num cols)
        BpMethod::MinimumSum,
        BpSchedule::Parallel,
        1.0,
        OsdMethod::Off,
        0,
        InputVectorType::Syndrome,
        None,
        None,
        None,
    )
    .expect("Failed to create decoder");

    assert_eq!(decoder_adaptive.max_iter(), 8);
}

#[test]
fn test_lsd_statistics() {
    let pcm = repetition_code(8);
    let error_rate = 0.1;

    let mut decoder = BpLsdDecoder::new(
        &pcm,
        Some(error_rate),
        None,
        20,
        BpMethod::MinimumSum,
        BpSchedule::Parallel,
        1.0,
        OsdMethod::OsdCs,
        2,
        1,
        InputVectorType::Syndrome,
        None,
        None,
        None,
    )
    .expect("Failed to create LSD decoder");

    // Initially stats should be disabled
    assert!(!decoder.do_stats());

    // Enable statistics
    decoder.set_do_stats(true);
    assert!(decoder.do_stats());

    // Decode a syndrome with statistics enabled
    let syndrome = arr1(&[1, 0, 0, 0, 1, 0, 0]);
    let _result = decoder.decode(&syndrome.view()).expect("Decoding failed");

    // Get statistics as JSON
    let stats_json = decoder
        .get_statistics_json()
        .expect("Failed to get statistics");
    println!("LSD Statistics: {stats_json}");

    // Verify the JSON contains expected fields
    assert!(stats_json.contains("elapsed_time_mu"));
    assert!(stats_json.contains("lsd_method"));
    assert!(stats_json.contains("lsd_order"));
    assert!(stats_json.contains("individual_cluster_stats"));

    // Disable statistics
    decoder.set_do_stats(false);
    assert!(!decoder.do_stats());
}

#[test]
fn test_soft_info_bp_decoder() {
    let pcm = repetition_code(8);
    let error_rate = 0.1;

    let mut decoder = SoftInfoBpDecoder::new(
        &pcm,
        Some(error_rate),
        None,
        20,
        BpMethod::MinimumSum,
        1.0,
        None,
        None,
        None,
    )
    .expect("Failed to create Soft Info BP decoder");

    // Verify getter methods
    assert_eq!(decoder.check_count(), 7);
    assert_eq!(decoder.bit_count(), 8);
    assert_eq!(decoder.max_iter(), 20);
    assert_eq!(decoder.bp_method(), BpMethod::MinimumSum);
    assert!((decoder.ms_scaling_factor() - 1.0).abs() < f64::EPSILON);
    assert_eq!(decoder.omp_thread_count(), 1);
    assert_eq!(decoder.random_schedule_seed(), -1);

    // Test soft syndrome decoding
    // Create a soft syndrome (log-likelihood ratios)
    let soft_syndrome = vec![
        -2.0, // Strong evidence for 1
        0.5,  // Weak evidence for 0
        -0.3, // Very weak evidence for 1
        1.5,  // Moderate evidence for 0
        -3.0, // Very strong evidence for 1
        0.1,  // Very weak evidence for 0
        0.0,  // No evidence either way
    ];

    let cutoff = 1.0;
    let sigma = 1.0;

    let result = decoder
        .decode(&soft_syndrome, cutoff, sigma)
        .expect("Decoding failed");

    println!("Soft Info BP decoding result:");
    println!("  Converged: {}", result.converged);
    println!("  Iterations: {}", result.iterations);
    println!("  Decoding: {:?}", result.decoding);

    // Check that we get log prob ratios
    let llrs = decoder.log_prob_ratios();
    assert_eq!(llrs.len(), 8);

    // Additional tests with different parameters
    let decoder2 = SoftInfoBpDecoder::new(
        &pcm,
        None,
        Some(&[0.05; 8]), // Use error channel
        0,                // Adaptive iterations
        BpMethod::ProductSum,
        0.8,
        Some(2), // 2 threads
        None,
        Some(123), // Random seed
    )
    .expect("Failed to create decoder");

    assert_eq!(decoder2.max_iter(), 8); // Adaptive should use n=8
    assert_eq!(decoder2.bp_method(), BpMethod::ProductSum);
    assert!((decoder2.ms_scaling_factor() - 0.8).abs() < f64::EPSILON);
    assert_eq!(decoder2.omp_thread_count(), 2);
    assert_eq!(decoder2.random_schedule_seed(), 123);

    // Test with custom serial schedule order
    let schedule_order: Vec<i32> = vec![7, 6, 5, 4, 3, 2, 1, 0];
    let mut decoder3 = SoftInfoBpDecoder::new(
        &pcm,
        Some(error_rate),
        None,
        10,
        BpMethod::MinimumSum,
        1.0,
        None,
        Some(&schedule_order),
        None,
    )
    .expect("Failed to create decoder");

    // Decode with the custom schedule
    let result3 = decoder3
        .decode(&soft_syndrome, cutoff, sigma)
        .expect("Decoding failed");
    println!("Custom schedule decoding converged: {}", result3.converged);
}
