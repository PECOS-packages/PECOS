// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

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
    assert_eq!(support.syndrome_bias(), vec![0, 1, 0, 0]);
    assert_eq!(support.observable_bias().to_u64(), Some(2));
    assert_eq!(support.dead_detectors(), vec![0, 1, 3]);
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
        assert_eq!(
            support.syndrome_bias(),
            if expected == 0 {
                vec![]
            } else {
                vec![expected]
            }
        );
        assert_eq!(
            support.observable_bias().to_u64(),
            Some(u64::from(expected))
        );
        assert_eq!(support.dead_detectors(), vec![0]);
    }
    for rate in [0.0, 1.0] {
        let mut matrix =
            DemCheckMatrix::from_dem_str("error(0.1) D0 L0\nerror(0.2) D1 L1").unwrap();
        matrix.error_priors.fill(rate);
        let (matrix, support) = matrix.merge_duplicate_symptoms().reduce_support();
        assert_eq!(matrix.num_mechanisms, 0);
        assert_eq!(
            support.syndrome_bias(),
            if rate >= 1.0 { vec![1; 2] } else { vec![] }
        );
        assert_eq!(support.dead_detectors(), vec![0, 1]);
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
    let mut decoder = CheckMatrixObservableDecoder::new(Echo, matrix.clone())
        .with_support(support)
        .unwrap();
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

#[test]
fn certain_columns_xor_shared_detectors() {
    let (_, support) = DemCheckMatrix::from_dem_str("error(1) D0 D1\nerror(1) D1 D2")
        .unwrap()
        .reduce_support();
    assert_eq!(support.syndrome_bias(), vec![1, 0, 1]);
}

#[test]
fn support_without_certain_columns_has_no_syndrome_bias() {
    let (_, support) = DemCheckMatrix::from_dem_str("error(0) D0\nerror(0.1) D1")
        .unwrap()
        .reduce_support();
    assert_eq!(support.syndrome_bias(), []);
    assert_eq!(support.dead_detectors(), &[0]);
}

#[test]
fn mismatched_support_is_rejected_without_panicking() {
    let matrix = DemCheckMatrix::from_dem_str("error(0.1) D0").unwrap();
    for text in ["error(1) D2", "error(0) D2"] {
        let (_, support) = DemCheckMatrix::from_dem_str(text).unwrap().reduce_support();
        assert!(matches!(
            CheckMatrixObservableDecoder::new(Echo, matrix.clone()).with_support(support.clone()),
            Err(DecoderError::InvalidConfiguration(message))
                if message.contains("matrix detector count 1")
                    && message.contains("attach the support returned by `reduce_support` for this matrix")
        ));
        // Direct callers of the shared helper are also checked before indexing.
        assert!(matches!(
            support.bias_syndrome(&mut [0], 1),
            Err(DecoderError::InvalidConfiguration(_))
        ));
    }
}

#[test]
fn replacing_public_dem_resizes_buffer_and_checks_support() {
    let small = DemCheckMatrix::from_dem_str("error(0.1) D0 L0").unwrap();
    let large = DemCheckMatrix::from_dem_str("error(0.1) D1 L0\ndetector D2").unwrap();
    let mut plain = CheckMatrixObservableDecoder::new(Echo, small.clone());
    plain.dem = large.clone();
    assert_eq!(plain.decode_to_observables(&[0, 1, 0]).unwrap(), 1);
    plain.dem = DemCheckMatrix::from_dem_str("error(0.1) D1 L0").unwrap();
    assert_eq!(plain.decode_to_observables(&[0, 1]).unwrap(), 1);
    for text in [
        "error(1) D2\nerror(0.1) D1 L0",
        "error(0.1) D1 L0\ndetector D2",
    ] {
        let (matrix, support) = DemCheckMatrix::from_dem_str(text).unwrap().reduce_support();
        let mut decoder = CheckMatrixObservableDecoder::new(Echo, matrix)
            .with_support(support)
            .unwrap();
        decoder.dem = small.clone();
        assert!(matches!(
            decoder.decode_obs(&[0]),
            Err(DecoderError::InvalidConfiguration(_))
        ));
    }
}

#[test]
fn bias_length_mismatch_without_dead_detectors_is_configuration_error() {
    let (_, support) = DemCheckMatrix::from_dem_str("error(1) D0\nerror(0.1) D0")
        .unwrap()
        .reduce_support();
    assert_eq!(support.syndrome_bias(), &[1]);
    assert_eq!(support.dead_detectors(), []);
    let matrix = DemCheckMatrix::from_dem_str("error(0.1) D0 D1").unwrap();
    assert!(matches!(
        CheckMatrixObservableDecoder::new(Echo, matrix).with_support(support.clone()),
        Err(DecoderError::InvalidConfiguration(message))
            if message == "support bias length 1 does not match matrix detector count 2; attach the support returned by `reduce_support` for this matrix"
    ));
    assert!(matches!(
        support.bias_syndrome(&mut [0, 0], 2),
        Err(DecoderError::InvalidConfiguration(_))
    ));
}
