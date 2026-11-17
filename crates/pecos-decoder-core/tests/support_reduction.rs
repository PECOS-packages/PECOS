use ndarray::{ArrayView1, arr2};
use pecos_decoder_core::{
    CheckMatrixObservableDecoder, Decoder, DecoderError, DecodingResultTrait, DemCheckMatrix,
    ObservableDecoder,
};

#[test]
fn exact_support_reduction() {
    let matrix = DemCheckMatrix {
        check_matrix: arr2(&[
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0],
        ]),
        observable_matrix: arr2(&[[0, 1, 1, 0, 0], [0, 0, 1, 1, 0]]),
        error_priors: vec![-0.0, 1.0, 1.0, 0.2, 0.0],
        num_detectors: 4,
        num_observables: 2,
        num_mechanisms: 5,
    };
    let (reduced, support) = matrix.reduce_support();
    assert_eq!(reduced.error_priors, vec![0.2]);
    assert_eq!(reduced.check_matrix, arr2(&[[0], [0], [1], [0]]));
    assert_eq!(reduced.observable_matrix, arr2(&[[0], [1]]));
    assert_eq!(support.syndrome_bias, vec![0, 1, 0, 0]);
    assert_eq!(support.observable_bias.to_u64(), Some(2));
    assert_eq!(support.dead_detectors, vec![0, 1, 3]);
}

#[test]
fn exact_support_merge_and_override() {
    for (count, expected) in [(2, 0), (3, 1)] {
        let text = "error(1) D0 L0\n".repeat(count);
        let (matrix, support) = DemCheckMatrix::from_dem_str(&text)
            .unwrap()
            .merge_duplicate_symptoms()
            .reduce_support();
        assert_eq!(matrix.num_mechanisms, 0);
        assert_eq!(matrix.check_matrix.dim(), (1, 0));
        assert_eq!(matrix.observable_matrix.dim(), (1, 0));
        assert_eq!(support.syndrome_bias, vec![expected]);
        assert_eq!(support.observable_bias.to_u64(), Some(u64::from(expected)));
        assert_eq!(support.dead_detectors, vec![0]);
    }
    for rate in [0.0, 1.0] {
        let mut matrix =
            DemCheckMatrix::from_dem_str("error(0.1) D0 L0\nerror(0.2) D1 L1").unwrap();
        matrix.error_priors.fill(rate);
        let (matrix, support) = matrix.merge_duplicate_symptoms().reduce_support();
        assert_eq!(matrix.num_mechanisms, 0);
        assert_eq!(
            support.syndrome_bias,
            vec![u8::from(rate.to_bits() == 1.0_f64.to_bits()); 2]
        );
        assert_eq!(support.dead_detectors, vec![0, 1]);
        assert!(matches!(
            matrix.ensure_nonempty_support(),
            Err(DecoderError::InvalidConfiguration(_))
        ));
    }
}

struct Echo;
struct Correction(Vec<u8>);
impl DecodingResultTrait for Correction {
    fn is_successful(&self) -> bool {
        true
    }
    fn correction(&self) -> &[u8] {
        &self.0
    }
}
impl Decoder for Echo {
    type Result = Correction;
    type Error = DecoderError;
    fn decode(&mut self, input: &ArrayView1<u8>) -> Result<Correction, DecoderError> {
        Ok(Correction(vec![input[1]]))
    }
    fn check_count(&self) -> usize {
        3
    }
    fn bit_count(&self) -> usize {
        1
    }
}

#[test]
fn exact_support_wrapper_methods_and_validation() {
    let matrix =
        DemCheckMatrix::from_dem_str("error(1) D0 L0\nerror(0.1) D1 L1\ndetector D2").unwrap();
    let (matrix, support) = matrix.reduce_support();
    let mut decoder = CheckMatrixObservableDecoder::new(Echo, matrix.clone()).with_support(support);
    assert_eq!(decoder.decode_to_observables(&[1, 1, 0]).unwrap(), 3);
    assert_eq!(
        decoder
            .decode_batch_to_observables(&[1, 1, 0, 1, 0, 0], 2, 3)
            .unwrap()
            .iter()
            .map(pecos_decoder_core::obs_mask::ObsMask::to_u64)
            .collect::<Vec<_>>(),
        vec![Some(3), Some(1)]
    );
    assert_eq!(
        decoder
            .decode_outcome(&[1, 0, 0])
            .unwrap()
            .prediction
            .unwrap()
            .to_u64(),
        Some(1)
    );
    assert!(matches!(
        decoder.decode_batch_to_observables(&[1, 0, 0, 1, 0, 1], 2, 3),
        Err(DecoderError::InvalidSyndrome(_))
    ));
    let mut plain = CheckMatrixObservableDecoder::new(Echo, matrix);
    // new() must not infer dead rows or biases.
    assert_eq!(plain.decode_to_observables(&[1, 0, 1]).unwrap(), 0);
    assert!(matches!(
        plain.decode_obs(&[0]),
        Err(DecoderError::InvalidDimensions {
            expected: 3,
            actual: 1
        })
    ));
    assert!(
        matches!(plain.decode_obs(&[0, 2, 0]), Err(DecoderError::InvalidSyndrome(message)) if message.contains("index 1") && message.contains('2'))
    );
}
