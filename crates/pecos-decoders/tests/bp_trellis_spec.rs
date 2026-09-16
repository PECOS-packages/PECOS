//! Feature-independent specs and native BP-Trellis construction.
use pecos_decoders::spec::{BpTrellisConfig, DecodeModel, DecoderSpec};

#[test]
fn bp_trellis_spec_is_feature_independent() {
    let spec = DecoderSpec::parse("bp_trellis").unwrap();
    assert_eq!(spec, DecoderSpec::BpTrellis(BpTrellisConfig::default()));
    assert!(!spec.execution_traits().history_dependent);
    assert!(!spec.execution_traits().wall_clock_dependent);
    assert!(!spec.requires_graphlike_model());
}

#[cfg(not(feature = "bp-trellis"))]
#[test]
fn missing_feature_is_actionable() {
    let result = DecoderSpec::parse("bp_trellis")
        .unwrap()
        .build(&DecodeModel::SingleDem("error(0.1) D0 L0".into()));
    assert!(matches!(
        result,
        Err(pecos_decoders::DecoderError::BackendUnavailable {
            family: "bp_trellis",
            required_feature: "bp-trellis",
        })
    ));
}

#[cfg(feature = "bp-trellis")]
#[test]
fn raw_hyperedges_and_wide_observables() {
    let spec = DecoderSpec::parse("bp_trellis").unwrap();
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
