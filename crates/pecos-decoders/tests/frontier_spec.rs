//! Feature-independent specs and native Frontier construction.
use pecos_decoders::spec::{DecodeModel, DecoderSpec, FrontierConfig};

#[test]
fn frontier_spec_is_feature_independent() {
    let spec = DecoderSpec::parse("frontier").unwrap();
    assert_eq!(spec, DecoderSpec::Frontier(FrontierConfig::default()));
    assert!(!spec.execution_traits().history_dependent);
    assert!(!spec.execution_traits().wall_clock_dependent);
    assert!(!spec.requires_graphlike_model());
}

#[cfg(not(feature = "frontier"))]
#[test]
fn missing_feature_is_actionable() {
    let result = DecoderSpec::parse("frontier")
        .unwrap()
        .build(&DecodeModel::SingleDem("error(0.1) D0 L0".into()));
    assert!(matches!(
        result,
        Err(pecos_decoders::DecoderError::BackendUnavailable {
            family: "frontier",
            required_feature: "frontier",
        })
    ));
}

#[cfg(feature = "frontier")]
#[test]
fn raw_hyperedges_and_wide_observables() {
    let spec = DecoderSpec::parse("frontier").unwrap();
    let mut decoder = spec
        .build(&DecodeModel::SingleDem("error(0.1) D0 D1 D2 L70\n".into()))
        .unwrap();
    assert_eq!(decoder.num_detectors(), Some(3));
    let predictions = decoder
        .decode_batch_to_observables(&[1, 1, 1, 0, 0, 0], 2, 3)
        .unwrap();
    assert_eq!(predictions[0].words(), &[0, 1 << 6]);
    assert!(predictions[1].words().iter().all(|&word| word == 0));
}
