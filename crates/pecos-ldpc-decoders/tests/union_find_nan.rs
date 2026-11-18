//! NaN weights must never reach the union-find sort, from callers or from BP.

use ndarray::arr1;
use pecos_ldpc_decoders::{
    BeliefFindDecoder, BpMethod, BpOsdDecoder, BpSchedule, LdpcError, OsdMethod, SparseMatrix,
    UfMethod, UnionFindDecoder,
};

fn repetition_pcm() -> SparseMatrix {
    SparseMatrix::from_coo(2, 3, vec![0, 0, 1, 1], vec![0, 1, 1, 2]).unwrap()
}

#[test]
fn min_sum_subnormal_prior_produces_nan() {
    let pcm = repetition_pcm();
    let mut decoder = BpOsdDecoder::builder(&pcm)
        .error_rate(f64::from_bits(1))
        .max_iter(1)
        .bp_method(BpMethod::MinimumSum)
        .bp_schedule(BpSchedule::Parallel)
        .ms_scaling_factor(2.0)
        .osd_method(OsdMethod::Off)
        .build()
        .unwrap();
    let result = decoder.decode(&arr1(&[1, 0]).view()).unwrap();
    let llrs = decoder.log_prob_ratios().unwrap();
    assert!(!result.converged);
    assert!(llrs[0].is_nan());
}

#[test]
fn union_find_rejects_nan_weights() {
    let pcm = repetition_pcm();
    for method in [UfMethod::Inversion, UfMethod::Peeling] {
        for index in 0..pcm.cols {
            let mut decoder = UnionFindDecoder::new(&pcm, method).unwrap();
            let mut llrs = [1.0; 3];
            llrs[index] = f64::NAN;
            let error = decoder.decode(&arr1(&[1, 0]).view(), &llrs, 1).unwrap_err();
            assert!(
                matches!(error, LdpcError::InvalidInput(ref message)
                if message.contains("NaN") && message.contains(&format!("index {index}"))),
                "{error}"
            );
        }
    }
}

#[test]
fn belief_find_rejects_bp_nan() {
    let pcm = repetition_pcm();
    for method in [UfMethod::Inversion, UfMethod::Peeling] {
        let mut decoder = BeliefFindDecoder::builder(&pcm)
            .error_rate(f64::from_bits(1))
            .max_iter(1)
            .bp_method(BpMethod::MinimumSum)
            .bp_schedule(BpSchedule::Parallel)
            .ms_scaling_factor(2.0)
            .uf_method(method)
            .build()
            .unwrap();
        let error = decoder.decode(&arr1(&[1, 0]).view()).unwrap_err();
        assert!(
            matches!(error, LdpcError::Ldpc(ref message)
            if message.contains("BP produced a NaN log-probability ratio")
                && message.contains("bit 0")),
            "{error}"
        );
    }
}

#[test]
fn union_find_accepts_infinite_weights() {
    let pcm = repetition_pcm();
    for method in [UfMethod::Inversion, UfMethod::Peeling] {
        let mut decoder = UnionFindDecoder::new(&pcm, method).unwrap();
        let result = decoder
            .decode(
                &arr1(&[1, 0]).view(),
                &[f64::NEG_INFINITY, 1.0, f64::INFINITY],
                1,
            )
            .unwrap();
        assert_eq!(result.decoding[0] ^ result.decoding[1], 1);
        assert_eq!(result.decoding[1] ^ result.decoding[2], 0);
    }
}
