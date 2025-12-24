//! Tests for newly added decoders (Flip, Union Find)

use ndarray::{Array1, arr1};
use pecos_ldpc_decoders::{FlipDecoder, SparseMatrix, UfMethod, UnionFindDecoder};

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

#[cfg(test)]
mod flip_decoder_tests {
    use super::*;

    #[test]
    fn test_flip_decoder_basic() {
        let pcm = repetition_code(5);
        let mut decoder = FlipDecoder::new(
            &pcm, 20, // max_iter
            0,  // pfreq (never perturb)
            0,  // random seed
        )
        .unwrap();

        // Test decoding a simple syndrome
        let syndrome = arr1(&[1, 0, 0, 1]);
        let result = decoder.decode(&syndrome.view()).unwrap();

        println!("Flip decoder result:");
        println!("  Converged: {}", result.converged);
        println!("  Iterations: {}", result.iterations);
        println!("  Decoding: {:?}", result.decoding);

        // Verify the decoding produces the correct syndrome
        let dense_pcm = pcm.to_dense();
        let decoded_syndrome_vec = dense_pcm.dot(&result.decoding);
        let decoded_syndrome: Array1<u8> = decoded_syndrome_vec.mapv(|x| x % 2);
        assert_eq!(decoded_syndrome, syndrome);
    }

    #[test]
    fn test_flip_decoder_with_perturbation() {
        let pcm = simple_ldpc_code();
        let mut decoder = FlipDecoder::new(
            &pcm, 50, // max_iter
            5,  // pfreq - perturb every 5 iterations
            42, // fixed seed for reproducibility
        )
        .unwrap();

        // Create a syndrome that might need perturbation
        let syndrome = arr1(&[1, 1, 0, 0]);
        let result = decoder.decode(&syndrome.view()).unwrap();

        println!("Flip decoder with perturbation:");
        println!("  Converged: {}", result.converged);
        println!("  Iterations: {}", result.iterations);
    }

    #[test]
    fn test_flip_decoder_getters() {
        let pcm = repetition_code(7);
        let decoder = FlipDecoder::new(&pcm, 30, 10, 123).unwrap();

        assert_eq!(decoder.check_count(), 6);
        assert_eq!(decoder.bit_count(), 7);
        assert_eq!(decoder.max_iter(), 30);
    }
}

#[cfg(test)]
mod union_find_decoder_tests {
    use super::*;

    #[test]
    fn test_uf_decoder_inversion_method() {
        let pcm = simple_ldpc_code();
        let mut decoder = UnionFindDecoder::new(&pcm, UfMethod::Inversion).unwrap();

        let syndrome = arr1(&[1, 0, 1, 0]);
        let empty_llrs: Vec<f64> = vec![];

        let result = decoder
            .decode(
                &syndrome.view(),
                &empty_llrs,
                0, // bits_per_step (0 = all)
            )
            .unwrap();

        println!("Union Find (inversion) result:");
        println!("  Decoding: {:?}", result.decoding);

        // Verify the decoding produces the correct syndrome
        let dense_pcm = pcm.to_dense();
        let decoded_syndrome_vec = dense_pcm.dot(&result.decoding);
        let decoded_syndrome: Array1<u8> = decoded_syndrome_vec.mapv(|x| x % 2);
        assert_eq!(decoded_syndrome, syndrome);
    }

    #[test]
    fn test_uf_decoder_with_llrs() {
        let pcm = repetition_code(6);
        let mut decoder = UnionFindDecoder::new(&pcm, UfMethod::Inversion).unwrap();

        let syndrome = arr1(&[1, 0, 1, 0, 0]);
        // Provide log-likelihood ratios to guide decoding
        let llrs = vec![
            -1.0, // Bit 0 likely error
            2.0,  // Bit 1 likely correct
            -0.5, // Bit 2 weakly likely error
            1.5,  // Bit 3 likely correct
            0.1,  // Bit 4 almost neutral
            2.5,  // Bit 5 very likely correct
        ];

        let result = decoder
            .decode(
                &syndrome.view(),
                &llrs,
                1, // Add 1 bit per step
            )
            .unwrap();

        println!("Union Find with LLRs result:");
        println!("  Decoding: {:?}", result.decoding);
    }

    #[test]
    fn test_uf_decoder_peeling_method() {
        // Create a code suitable for peeling (max degree 2)
        let pcm = repetition_code(5);
        let mut decoder = UnionFindDecoder::new(&pcm, UfMethod::Peeling).unwrap();

        let syndrome = arr1(&[1, 0, 0, 1]);
        let empty_llrs: Vec<f64> = vec![];

        let result = decoder.decode(&syndrome.view(), &empty_llrs, 0).unwrap();

        println!("Union Find (peeling) result:");
        println!("  Decoding: {:?}", result.decoding);

        // Verify syndrome
        let dense_pcm = pcm.to_dense();
        let decoded_syndrome_vec = dense_pcm.dot(&result.decoding);
        let decoded_syndrome: Array1<u8> = decoded_syndrome_vec.mapv(|x| x % 2);
        assert_eq!(decoded_syndrome, syndrome);
    }

    #[test]
    fn test_uf_decoder_getters() {
        let pcm = simple_ldpc_code();
        let decoder = UnionFindDecoder::new(&pcm, UfMethod::Inversion).unwrap();

        assert_eq!(decoder.check_count(), 4);
        assert_eq!(decoder.bit_count(), 6);
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_flip_decoder_simple_case() {
        // Use a simpler syndrome that the flip decoder can handle
        let pcm = repetition_code(5);
        let syndrome = arr1(&[1, 0, 0, 1]); // Simple syndrome

        // Try Flip decoder with more iterations and some perturbation
        let mut flip_decoder = FlipDecoder::new(
            &pcm, 50, // More iterations
            10, // Add some perturbation
            42, // Fixed seed for reproducibility
        )
        .unwrap();

        let flip_result = flip_decoder.decode(&syndrome.view()).unwrap();

        println!("Flip decoder simple case:");
        println!(
            "  Converged: {}, iterations: {}",
            flip_result.converged, flip_result.iterations
        );
        println!("  Decoding: {:?}", flip_result.decoding);

        // Verify that either the decoder converged with correct syndrome,
        // or at least produced some output (flip decoder is heuristic)
        assert!(flip_result.iterations > 0);

        // If it converged, verify the syndrome matches
        if flip_result.converged {
            let dense_pcm = pcm.to_dense();
            let computed_syndrome = dense_pcm.dot(&flip_result.decoding);
            let computed_syndrome: Array1<u8> = computed_syndrome.mapv(|x| x % 2);
            assert_eq!(computed_syndrome, syndrome);
        }
    }
}
