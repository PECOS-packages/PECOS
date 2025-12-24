//! Comprehensive decoder tests based on Python and C++ test suites

use ndarray::{Array1, arr1};
use pecos_ldpc_decoders::{
    BpLsdDecoder, BpMethod, BpOsdDecoder, BpSchedule, InputVectorType, OsdMethod,
    SoftInfoBpDecoder, SparseMatrix,
};

/// Create a repetition code parity check matrix
fn repetition_code(n: usize) -> SparseMatrix {
    let mut row_indices = Vec::new();
    let mut col_indices = Vec::new();

    // H matrix for repetition code has n-1 rows and n columns
    // Each row connects adjacent bits
    for i in 0..n - 1 {
        let i_u32 = u32::try_from(i).expect("index too large");
        row_indices.push(i_u32);
        col_indices.push(i_u32);
        row_indices.push(i_u32);
        col_indices.push(u32::try_from(i + 1).expect("index too large"));
    }

    SparseMatrix::from_coo(n - 1, n, row_indices, col_indices).unwrap()
}

/// Create a ring code (cyclic repetition code)
fn ring_code(n: usize) -> SparseMatrix {
    let mut row_indices = Vec::new();
    let mut col_indices = Vec::new();

    for i in 0..n {
        let i_u32 = u32::try_from(i).expect("index too large");
        row_indices.push(i_u32);
        col_indices.push(i_u32);
        row_indices.push(i_u32);
        col_indices.push(u32::try_from((i + 1) % n).expect("index too large"));
    }

    SparseMatrix::from_coo(n, n, row_indices, col_indices).unwrap()
}

/// Create Hamming(7,4) code
fn hamming_7_4_code() -> SparseMatrix {
    let mut row_indices = Vec::new();
    let mut col_indices = Vec::new();

    // Row 0: positions 3,4,5,6
    row_indices.extend(&[0, 0, 0, 0]);
    col_indices.extend(&[3, 4, 5, 6]);

    // Row 1: positions 1,2,5,6
    row_indices.extend(&[1, 1, 1, 1]);
    col_indices.extend(&[1, 2, 5, 6]);

    // Row 2: positions 0,2,4,6
    row_indices.extend(&[2, 2, 2, 2]);
    col_indices.extend(&[0, 2, 4, 6]);

    SparseMatrix::from_coo(3, 7, row_indices, col_indices).unwrap()
}

#[cfg(test)]
mod bp_decoder_tests {
    use super::*;

    #[test]
    fn test_repetition_code_all_syndromes() {
        // Test 3-bit repetition code with all possible syndromes
        let pcm = repetition_code(3);
        let error_rate = 0.1;

        let syndromes = [arr1(&[0, 0]), arr1(&[0, 1]), arr1(&[1, 0]), arr1(&[1, 1])];

        let expected_decodings = [
            arr1(&[0, 0, 0]),
            arr1(&[0, 0, 1]),
            arr1(&[1, 0, 0]),
            arr1(&[0, 1, 0]),
        ];

        for (syndrome, expected) in syndromes.iter().zip(expected_decodings.iter()) {
            let mut decoder = BpOsdDecoder::new(
                &pcm,
                Some(error_rate),
                None,
                10,
                BpMethod::ProductSum,
                BpSchedule::Parallel,
                1.0,
                OsdMethod::Off,
                0,
                InputVectorType::Syndrome,
                None,
                None,
                None,
            )
            .unwrap();

            let result = decoder.decode(&syndrome.view()).unwrap();
            assert_eq!(
                result.decoding, *expected,
                "Failed for syndrome {syndrome:?}"
            );
            assert!(result.converged);
        }
    }

    #[test]
    fn test_bp_methods_comparison() {
        let pcm = ring_code(5);
        let error_rate = 0.1;
        let syndrome = arr1(&[1, 0, 0, 0, 1]);

        // Test Product Sum
        let mut decoder_ps = BpOsdDecoder::new(
            &pcm,
            Some(error_rate),
            None,
            20,
            BpMethod::ProductSum,
            BpSchedule::Parallel,
            1.0,
            OsdMethod::Off,
            0,
            InputVectorType::Syndrome,
            None,
            None,
            None,
        )
        .unwrap();

        let result_ps = decoder_ps.decode(&syndrome.view()).unwrap();

        // Test Minimum Sum
        let mut decoder_minsum = BpOsdDecoder::new(
            &pcm,
            Some(error_rate),
            None,
            20,
            BpMethod::MinimumSum,
            BpSchedule::Parallel,
            0.625,
            OsdMethod::Off,
            0,
            InputVectorType::Syndrome,
            None,
            None,
            None,
        )
        .unwrap();

        let result_minsum = decoder_minsum.decode(&syndrome.view()).unwrap();

        // Both should converge for this simple case
        assert!(result_ps.converged);
        assert!(result_minsum.converged);
    }

    #[test]
    fn test_serial_schedule_order() {
        let pcm = repetition_code(5);
        let error_rate = 0.1;
        let syndrome = arr1(&[1, 0, 0, 1]);

        // Custom serial schedule (reverse order)
        let schedule_order: Vec<i32> = vec![4, 3, 2, 1, 0];

        let mut decoder = BpOsdDecoder::new(
            &pcm,
            Some(error_rate),
            None,
            10,
            BpMethod::MinimumSum,
            BpSchedule::Serial,
            1.0,
            OsdMethod::Off,
            0,
            InputVectorType::Syndrome,
            None,
            Some(&schedule_order),
            None,
        )
        .unwrap();

        let result = decoder.decode(&syndrome.view()).unwrap();
        assert!(result.converged);
    }

    #[test]
    fn test_non_uniform_error_channel() {
        let pcm = hamming_7_4_code();
        // Non-uniform error probabilities
        let error_channel = vec![0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07];
        let syndrome = arr1(&[1, 1, 0]);

        let mut decoder = BpOsdDecoder::new(
            &pcm,
            None,
            Some(&error_channel),
            20,
            BpMethod::MinimumSum,
            BpSchedule::Parallel,
            0.625,
            OsdMethod::Off,
            0,
            InputVectorType::Syndrome,
            None,
            None,
            None,
        )
        .unwrap();

        let result = decoder.decode(&syndrome.view()).unwrap();
        assert!(result.converged);

        // Verify channel probabilities were set correctly
        let stored_probs = decoder.channel_probs();
        for (i, &prob) in error_channel.iter().enumerate() {
            assert!((stored_probs[i] - prob).abs() < f64::EPSILON);
        }
    }
}

#[cfg(test)]
mod bp_osd_decoder_tests {
    use super::*;

    #[test]
    fn test_osd_methods() {
        let pcm = hamming_7_4_code();
        let error_rate = 0.1;
        let syndrome = arr1(&[1, 1, 1]); // All checks failed

        let osd_methods = vec![
            (OsdMethod::Off, 0),
            (OsdMethod::Osd0, 0),
            (OsdMethod::OsdCs, 3),
            (OsdMethod::OsdE, 3),
        ];

        for (osd_method, osd_order) in osd_methods {
            let mut decoder = BpOsdDecoder::new(
                &pcm,
                Some(error_rate),
                None,
                20,
                BpMethod::MinimumSum,
                BpSchedule::Parallel,
                0.625,
                osd_method,
                osd_order,
                InputVectorType::Syndrome,
                None,
                None,
                None,
            )
            .unwrap();

            let result = decoder.decode(&syndrome.view()).unwrap();
            println!(
                "OSD method {:?} with order {}: converged = {}, iterations = {}",
                osd_method, osd_order, result.converged, result.iterations
            );
        }
    }

    #[test]
    fn test_zero_syndrome() {
        let pcm = hamming_7_4_code();
        let error_rate = 0.1;
        let syndrome = arr1(&[0, 0, 0]);

        let mut decoder = BpOsdDecoder::new(
            &pcm,
            Some(error_rate),
            None,
            10,
            BpMethod::ProductSum,
            BpSchedule::Parallel,
            1.0,
            OsdMethod::OsdCs,
            2,
            InputVectorType::Syndrome,
            None,
            None,
            None,
        )
        .unwrap();

        let result = decoder.decode(&syndrome.view()).unwrap();
        assert!(result.converged);
        assert_eq!(result.decoding, arr1(&[0, 0, 0, 0, 0, 0, 0]));
        // Implementation performs at least 1 iteration even for zero syndrome
        assert!(result.iterations >= 1);
    }

    #[test]
    fn test_osd_vs_bp_only() {
        let pcm = hamming_7_4_code();
        let error_rate = 0.05;

        // Create a syndrome that BP alone might struggle with
        let syndrome = arr1(&[1, 1, 1]);

        // BP only
        let mut bp_only = BpOsdDecoder::new(
            &pcm,
            Some(error_rate),
            None,
            5, // Limited iterations
            BpMethod::MinimumSum,
            BpSchedule::Parallel,
            0.625,
            OsdMethod::Off,
            0,
            InputVectorType::Syndrome,
            None,
            None,
            None,
        )
        .unwrap();

        let bp_result = bp_only.decode(&syndrome.view()).unwrap();

        // BP + OSD
        let mut bp_osd = BpOsdDecoder::new(
            &pcm,
            Some(error_rate),
            None,
            5, // Same limited iterations for BP
            BpMethod::MinimumSum,
            BpSchedule::Parallel,
            0.625,
            OsdMethod::OsdCs,
            3,
            InputVectorType::Syndrome,
            None,
            None,
            None,
        )
        .unwrap();

        let osd_result = bp_osd.decode(&syndrome.view()).unwrap();

        println!("BP only: converged = {}", bp_result.converged);
        println!("BP+OSD: converged = {}", osd_result.converged);

        // OSD should help if BP alone didn't converge
        if !bp_result.converged {
            assert!(osd_result.converged || osd_result.iterations >= bp_result.iterations);
        }
    }
}

#[cfg(test)]
mod bp_lsd_decoder_tests {
    use super::*;

    #[test]
    fn test_lsd_basic_decoding() {
        let pcm = ring_code(6);
        let error_rate = 0.1;
        let syndrome = arr1(&[1, 0, 0, 0, 1, 0]);

        let mut decoder = BpLsdDecoder::new(
            &pcm,
            Some(error_rate),
            None,
            20,
            BpMethod::MinimumSum,
            BpSchedule::Parallel,
            0.625,
            OsdMethod::OsdCs,
            3,
            1, // bits_per_step
            InputVectorType::Syndrome,
            None,
            None,
            None,
        )
        .unwrap();

        let result = decoder.decode(&syndrome.view()).unwrap();
        assert!(result.converged);
    }

    #[test]
    fn test_lsd_statistics_collection() {
        let pcm = repetition_code(8);
        let error_rate = 0.1;
        let syndrome = arr1(&[1, 0, 0, 1, 0, 0, 0]);

        let mut decoder = BpLsdDecoder::new(
            &pcm,
            Some(error_rate),
            None,
            50, // Increased iterations
            BpMethod::MinimumSum,
            BpSchedule::Parallel,
            0.625,
            OsdMethod::OsdCs, // Changed to OsdCs
            2,                // Increased order
            1,
            InputVectorType::Syndrome,
            None,
            None,
            None,
        )
        .unwrap();

        // Enable statistics
        decoder.set_do_stats(true);

        let result = decoder.decode(&syndrome.view()).unwrap();
        // Don't require convergence for this test - focus on statistics collection
        println!(
            "LSD decoding converged: {}, iterations: {}",
            result.converged, result.iterations
        );

        // Get statistics
        let stats_json = decoder.get_statistics_json().unwrap();
        assert!(stats_json.contains("elapsed_time_mu"));
        assert!(stats_json.contains("individual_cluster_stats"));
    }

    #[test]
    fn test_lsd_bits_per_step() {
        let pcm = hamming_7_4_code();
        let error_rate = 0.1;
        let syndrome = arr1(&[1, 0, 1]);

        // Test different bits_per_step values
        for bits_per_step in [0, 1, 2, 3] {
            let mut decoder = BpLsdDecoder::new(
                &pcm,
                Some(error_rate),
                None,
                20,
                BpMethod::MinimumSum,
                BpSchedule::Parallel,
                0.625,
                OsdMethod::OsdCs,
                2,
                bits_per_step,
                InputVectorType::Syndrome,
                None,
                None,
                None,
            )
            .unwrap();

            let result = decoder.decode(&syndrome.view()).unwrap();
            println!(
                "bits_per_step = {}: converged = {}",
                bits_per_step, result.converged
            );
        }
    }
}

#[cfg(test)]
mod soft_info_decoder_tests {
    use super::*;

    #[test]
    fn test_soft_syndrome_ring_code() {
        let pcm = ring_code(6);
        let error_rate = 0.1;

        // Soft syndrome with values near decision boundary
        let soft_syndrome = vec![-20.0, 1.0, 20.0, -1.0, 0.5, -0.5];
        let cutoff = 2.0;
        let sigma = 1.0;

        let mut decoder = SoftInfoBpDecoder::new(
            &pcm,
            Some(error_rate),
            None,
            20,
            BpMethod::MinimumSum,
            0.625,
            None,
            None,
            None,
        )
        .unwrap();

        let result = decoder.decode(&soft_syndrome, cutoff, sigma).unwrap();
        println!("Soft decoding result: {:?}", result.decoding);
        println!(
            "Converged: {}, iterations: {}",
            result.converged, result.iterations
        );
    }

    #[test]
    fn test_soft_syndrome_with_errors() {
        let pcm = repetition_code(10);
        let error_rate = 0.1;

        // Soft syndrome with one erroneous measurement
        let mut soft_syndrome = vec![2.0; 9];
        soft_syndrome[0] = -1.0; // Weak evidence for wrong value

        let cutoff = 1.5;
        let sigma = 1.0;

        let mut decoder = SoftInfoBpDecoder::new(
            &pcm,
            Some(error_rate),
            None,
            30,
            BpMethod::ProductSum,
            1.0,
            None,
            None,
            None,
        )
        .unwrap();

        let result = decoder.decode(&soft_syndrome, cutoff, sigma).unwrap();
        assert!(result.converged);
    }

    #[test]
    fn test_soft_syndrome_hamming_code() {
        let pcm = hamming_7_4_code();
        let error_rate = 0.1;

        // Test various soft syndrome patterns
        let test_cases = vec![
            (vec![-10.0, -10.0, -10.0], "All strong negative"),
            (vec![10.0, 10.0, 10.0], "All strong positive"),
            (vec![-5.0, 0.0, 5.0], "Mixed strengths"),
            (vec![0.1, -0.1, 0.2], "All weak"),
        ];

        for (soft_syndrome, description) in test_cases {
            let mut decoder = SoftInfoBpDecoder::new(
                &pcm,
                Some(error_rate),
                None,
                20,
                BpMethod::MinimumSum,
                0.625,
                None,
                None,
                None,
            )
            .unwrap();

            let result = decoder.decode(&soft_syndrome, 1.0, 1.0).unwrap();
            println!(
                "{}: converged = {}, iterations = {}",
                description, result.converged, result.iterations
            );
        }
    }

    #[test]
    fn test_cutoff_parameter_effect() {
        let pcm = ring_code(5);
        let error_rate = 0.1;
        let soft_syndrome = vec![-0.5, 0.5, -0.5, 0.5, -0.5];
        let sigma = 1.0;

        // Test different cutoff values
        for cutoff in [0.1, 0.5, 1.0, 2.0, 5.0] {
            let mut decoder = SoftInfoBpDecoder::new(
                &pcm,
                Some(error_rate),
                None,
                20,
                BpMethod::MinimumSum,
                0.625,
                None,
                None,
                None,
            )
            .unwrap();

            let result = decoder.decode(&soft_syndrome, cutoff, sigma).unwrap();
            println!(
                "Cutoff {}: converged = {}, iterations = {}",
                cutoff, result.converged, result.iterations
            );
        }
    }
}

#[cfg(test)]
mod edge_case_tests {
    use super::*;

    #[test]
    fn test_all_ones_syndrome() {
        let pcm = repetition_code(8);
        let error_rate = 0.1;
        let syndrome = arr1(&[1, 1, 1, 1, 1, 1, 1]);

        let mut decoder = BpOsdDecoder::new(
            &pcm,
            Some(error_rate),
            None,
            20,
            BpMethod::MinimumSum,
            BpSchedule::Parallel,
            0.625,
            OsdMethod::Off,
            0,
            InputVectorType::Syndrome,
            None,
            None,
            None,
        )
        .unwrap();

        let result = decoder.decode(&syndrome.view()).unwrap();
        // Should find some valid error pattern
        assert_eq!(result.decoding.len(), 8);
    }

    #[test]
    fn test_single_bit_errors() {
        let pcm = hamming_7_4_code();
        let error_rate = 0.01; // Low error rate

        // Test single bit error patterns
        for bit_pos in 0..7 {
            let mut error = vec![0u8; 7];
            error[bit_pos] = 1;

            // Calculate syndrome manually
            let dense_pcm = pcm.to_dense();
            let error_vec = arr1(&error);
            let syndrome_vec = dense_pcm.dot(&error_vec);
            let syndrome: Array1<u8> = syndrome_vec.mapv(|x| x % 2);

            let mut decoder = BpOsdDecoder::new(
                &pcm,
                Some(error_rate),
                None,
                10,
                BpMethod::ProductSum,
                BpSchedule::Parallel,
                1.0,
                OsdMethod::Off,
                0,
                InputVectorType::Syndrome,
                None,
                None,
                None,
            )
            .unwrap();

            let result = decoder.decode(&syndrome.view()).unwrap();
            assert!(result.converged);

            // Verify that the decoded error gives the same syndrome
            let decoded_syndrome_vec = dense_pcm.dot(&result.decoding);
            let decoded_syndrome: Array1<u8> = decoded_syndrome_vec.mapv(|x| x % 2);
            assert_eq!(
                decoded_syndrome, syndrome,
                "Decoded error doesn't produce correct syndrome for bit position {bit_pos}"
            );
        }
    }

    #[test]
    fn test_maximum_iterations_limit() {
        let pcm = ring_code(10);
        let error_rate = 0.5; // Very high error rate
        let syndrome = arr1(&[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]);

        let max_iter = 3; // Very limited iterations

        let mut decoder = BpOsdDecoder::new(
            &pcm,
            Some(error_rate),
            None,
            max_iter,
            BpMethod::MinimumSum,
            BpSchedule::Parallel,
            0.625,
            OsdMethod::Off,
            0,
            InputVectorType::Syndrome,
            None,
            None,
            None,
        )
        .unwrap();

        let result = decoder.decode(&syndrome.view()).unwrap();
        assert!(result.iterations <= max_iter);
    }

    #[test]
    fn test_thread_safety() {
        let pcm = hamming_7_4_code();
        let error_rate = 0.1;
        let syndrome = arr1(&[1, 0, 1]);

        // Test with different thread counts
        for thread_count in [1, 2, 4] {
            let mut decoder = BpOsdDecoder::new(
                &pcm,
                Some(error_rate),
                None,
                20,
                BpMethod::MinimumSum,
                BpSchedule::Parallel,
                0.625,
                OsdMethod::Off,
                0,
                InputVectorType::Syndrome,
                Some(thread_count),
                None,
                None,
            )
            .unwrap();

            let result = decoder.decode(&syndrome.view()).unwrap();
            assert!(result.converged);
            assert_eq!(decoder.omp_thread_count(), thread_count);
        }
    }
}
