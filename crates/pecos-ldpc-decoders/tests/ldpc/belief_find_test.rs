//! Tests for `BeliefFind` decoder

use ndarray::{Array1, arr1, arr2};
use pecos_ldpc_decoders::{BeliefFindDecoder, BpMethod, BpSchedule, SparseMatrix, UfMethod};

/// Create a repetition code parity check matrix
fn repetition_code(n: usize) -> SparseMatrix {
    let mut row_indices = Vec::new();
    let mut col_indices = Vec::new();

    // H matrix for repetition code has n-1 rows and n columns
    for i in 0..n - 1 {
        let i_u32 = u32::try_from(i).expect("index too large");
        row_indices.push(i_u32);
        col_indices.push(i_u32);
        row_indices.push(i_u32);
        col_indices.push(u32::try_from(i + 1).expect("index too large"));
    }

    SparseMatrix::from_coo(n - 1, n, row_indices, col_indices).unwrap()
}

/// Create a simple LDPC code for testing
fn simple_ldpc_code() -> SparseMatrix {
    // 4x6 LDPC code
    let row_indices = vec![0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3];
    let col_indices = vec![0, 2, 4, 1, 3, 5, 0, 3, 5, 1, 2, 4];
    SparseMatrix::from_coo(4, 6, row_indices, col_indices).unwrap()
}

/// Create a Hamming(7,4) code
fn hamming_code() -> SparseMatrix {
    let dense = arr2(&[
        [1, 0, 1, 0, 1, 0, 1],
        [0, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1],
    ]);

    SparseMatrix::from_dense(&dense.view())
}

#[cfg(test)]
mod belief_find_tests {
    use super::*;

    #[test]
    fn test_belief_find_basic() {
        let pcm = hamming_code();
        let mut decoder = BeliefFindDecoder::new(
            &pcm,
            Some(0.1),
            None,
            10, // max_iter
            BpMethod::MinimumSum,
            0.625,
            BpSchedule::Parallel,
            None, // omp_thread_count
            None, // serial_schedule_order
            None, // random_schedule_seed
            UfMethod::Inversion,
            0, // bits_per_step (0 = all)
        )
        .unwrap();

        // Test a simple syndrome that BP should handle
        let syndrome = arr1(&[1, 1, 0]);
        let result = decoder.decode(&syndrome.view()).unwrap();

        println!("BeliefFind decoder result (BP should converge):");
        println!("  Decoding: {:?}", result.decoding);
        println!("  Converged: {}", result.converged);
        println!("  Iterations: {}", result.iterations);

        // Verify the decoding produces the correct syndrome
        let dense_pcm = pcm.to_dense();
        let decoded_syndrome_vec = dense_pcm.dot(&result.decoding);
        let decoded_syndrome: Array1<u8> = decoded_syndrome_vec.mapv(|x| x % 2);
        assert_eq!(decoded_syndrome, syndrome);
    }

    #[test]
    fn test_belief_find_fallback_to_uf_fixed() {
        // Use a smaller, simpler code to avoid hanging
        let pcm = repetition_code(5);

        // Create a decoder that will definitely fall back to UF
        // Use parameters that make BP very likely to fail
        let mut decoder = BeliefFindDecoder::new(
            &pcm,
            Some(0.3), // High error rate
            None,
            2, // Very few BP iterations
            BpMethod::MinimumSum,
            0.5, // Lower scaling factor
            BpSchedule::Parallel,
            None,
            None,
            None,
            UfMethod::Peeling, // Use peeling which is typically faster
            0,
        )
        .unwrap();

        // Create a syndrome that's challenging for BP
        let syndrome = arr1(&[1, 1, 0, 1]);
        let result = decoder.decode(&syndrome.view()).unwrap();

        println!("BeliefFind decoder result (forced UF fallback):");
        println!("  Decoding: {:?}", result.decoding);
        println!("  Iterations: {} (BP only)", result.iterations);
        println!("  Converged: {}", result.converged);

        // Verify we get a valid decoding
        let dense_pcm = pcm.to_dense();
        let decoded_syndrome_vec = dense_pcm.dot(&result.decoding);
        let decoded_syndrome: Array1<u8> = decoded_syndrome_vec.mapv(|x| x % 2);
        assert_eq!(decoded_syndrome, syndrome);
    }

    #[test]
    fn test_belief_find_timeout_protection() {
        // Test with a small timeout to ensure decoder doesn't hang
        let pcm = hamming_code();

        // Create decoder with potentially problematic parameters
        let mut decoder = BeliefFindDecoder::new(
            &pcm,
            Some(0.45), // Very high error rate
            None,
            5, // Limited iterations
            BpMethod::MinimumSum,
            0.625,
            BpSchedule::Parallel,
            None,
            None,
            None,
            UfMethod::Inversion,
            0,
        )
        .unwrap();

        // Test multiple syndromes to check for hanging
        let test_syndromes = vec![arr1(&[1, 0, 1]), arr1(&[1, 1, 1]), arr1(&[0, 1, 1])];

        for syndrome in test_syndromes {
            let start = std::time::Instant::now();
            let result = decoder.decode(&syndrome.view()).unwrap();
            let elapsed = start.elapsed();

            // Ensure decoding doesn't take too long (should be < 100ms for small codes)
            assert!(
                elapsed.as_millis() < 100,
                "Decoding took too long: {:?}ms",
                elapsed.as_millis()
            );

            // Verify valid decoding
            let dense_pcm = pcm.to_dense();
            let decoded_syndrome_vec = dense_pcm.dot(&result.decoding);
            let decoded_syndrome: Array1<u8> = decoded_syndrome_vec.mapv(|x| x % 2);
            assert_eq!(decoded_syndrome, syndrome);
        }
    }

    #[test]
    fn test_belief_find_with_peeling() {
        let pcm = repetition_code(7);
        let mut decoder = BeliefFindDecoder::new(
            &pcm,
            Some(0.2),
            None,
            2, // Low iterations to potentially trigger UF
            BpMethod::ProductSum,
            1.0,
            BpSchedule::Serial,
            None,
            None,
            None,
            UfMethod::Peeling, // Use peeling method
            0,
        )
        .unwrap();

        let syndrome = arr1(&[1, 0, 1, 0, 0, 1]);
        let result = decoder.decode(&syndrome.view()).unwrap();

        println!("BeliefFind with peeling result:");
        println!("  Decoding: {:?}", result.decoding);

        // Verify syndrome
        let dense_pcm = pcm.to_dense();
        let decoded_syndrome_vec = dense_pcm.dot(&result.decoding);
        let decoded_syndrome: Array1<u8> = decoded_syndrome_vec.mapv(|x| x % 2);
        assert_eq!(decoded_syndrome, syndrome);
    }

    #[test]
    fn test_belief_find_getters() {
        let pcm = hamming_code();
        let decoder = BeliefFindDecoder::new(
            &pcm,
            Some(0.15),
            None,
            20,
            BpMethod::MinimumSum,
            0.7,
            BpSchedule::SerialRelative,
            Some(4),
            None,
            Some(42),
            UfMethod::Inversion,
            5,
        )
        .unwrap();

        assert_eq!(decoder.check_count(), 3);
        assert_eq!(decoder.bit_count(), 7);
        assert_eq!(decoder.max_iter(), 20);
        assert_eq!(decoder.bp_method(), BpMethod::MinimumSum);
        assert!((decoder.ms_scaling_factor() - 0.7).abs() < f64::EPSILON);
        assert_eq!(decoder.bp_schedule(), BpSchedule::SerialRelative);
        assert_eq!(decoder.uf_method(), UfMethod::Inversion);
        assert_eq!(decoder.bits_per_step(), 5);
        assert_eq!(decoder.omp_thread_count(), 4);
    }

    #[test]
    fn test_belief_find_zero_syndrome() {
        let pcm = simple_ldpc_code();
        let mut decoder = BeliefFindDecoder::new(
            &pcm,
            Some(0.1),
            None,
            10,
            BpMethod::MinimumSum,
            0.625,
            BpSchedule::Parallel,
            None,
            None,
            None,
            UfMethod::Inversion,
            0,
        )
        .unwrap();

        // Zero syndrome should converge immediately
        let syndrome = arr1(&[0, 0, 0, 0]);
        let result = decoder.decode(&syndrome.view()).unwrap();

        assert!(result.converged);
        assert_eq!(result.decoding, arr1(&[0, 0, 0, 0, 0, 0]));
        // Note: Due to implementation, even zero syndrome does at least 1 iteration
        assert!(result.iterations >= 1);
    }

    #[test]
    fn test_belief_find_simple_fallback() {
        // Test with a very small code where we can control the behavior
        let pcm = SparseMatrix::from_coo(
            2,
            3, // 2x3 code
            vec![0, 0, 1, 1],
            vec![0, 1, 1, 2],
        )
        .unwrap();

        let mut decoder = BeliefFindDecoder::new(
            &pcm,
            Some(0.4), // High error rate
            None,
            1, // Minimal BP iterations
            BpMethod::MinimumSum,
            0.625,
            BpSchedule::Parallel,
            None,
            None,
            None,
            UfMethod::Inversion,
            0,
        )
        .unwrap();

        // This syndrome has a simple solution
        let syndrome = arr1(&[1, 1]);
        let result = decoder.decode(&syndrome.view()).unwrap();

        // Should produce a valid decoding
        let dense_pcm = pcm.to_dense();
        let decoded_syndrome_vec = dense_pcm.dot(&result.decoding);
        let decoded_syndrome: Array1<u8> = decoded_syndrome_vec.mapv(|x| x % 2);
        assert_eq!(decoded_syndrome, syndrome);

        println!(
            "Simple fallback test passed with decoding: {:?}",
            result.decoding
        );
    }
}
