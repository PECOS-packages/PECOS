use ndarray::arr1;
use pecos_ldpc_decoders::{
    BpLsdDecoder, BpMethod, BpOsdDecoder, BpSchedule, InputVectorType, OsdMethod, SparseMatrix,
};

#[test]
fn product_sum_tiny_prior_osd() {
    let pcm = SparseMatrix::from_coo(2, 3, vec![0, 0, 1, 1], vec![0, 1, 1, 2]).unwrap();
    for schedule in [BpSchedule::Parallel, BpSchedule::Serial] {
        let mut decoder = BpOsdDecoder::new(
            &pcm,
            Some(1e-17),
            None,
            10,
            BpMethod::ProductSum,
            schedule,
            1.0,
            OsdMethod::Osd0,
            0,
            InputVectorType::Syndrome,
            None,
            None,
            None,
        )
        .unwrap();
        let result = decoder.decode(&arr1(&[1, 0]).view()).unwrap();
        assert!(
            decoder
                .log_prob_ratios()
                .unwrap()
                .iter()
                .all(|x| x.is_finite())
        );
        assert_eq!(result.decoding[0] ^ result.decoding[1], 1);
        assert_eq!(result.decoding[1] ^ result.decoding[2], 0);
    }
}

#[test]
fn product_sum_tiny_prior_lsd() {
    let pcm = SparseMatrix::from_coo(2, 3, vec![0, 0, 1, 1], vec![0, 1, 1, 2]).unwrap();
    for schedule in [BpSchedule::Parallel, BpSchedule::Serial] {
        let mut decoder = BpLsdDecoder::new(
            &pcm,
            Some(1e-17),
            None,
            10,
            BpMethod::ProductSum,
            schedule,
            1.0,
            OsdMethod::Osd0,
            0,
            1,
            InputVectorType::Syndrome,
            None,
            None,
            None,
        )
        .unwrap();
        let result = decoder.decode(&arr1(&[1, 0]).view()).unwrap();
        assert!(
            decoder
                .log_prob_ratios()
                .unwrap()
                .iter()
                .all(|x| x.is_finite())
        );
        assert_eq!(result.decoding[0] ^ result.decoding[1], 1);
        assert_eq!(result.decoding[1] ^ result.decoding[2], 0);
    }
}
