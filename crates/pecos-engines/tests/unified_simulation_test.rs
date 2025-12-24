//! Tests for the unified simulation API with focus on seeding behavior
//!
//! These tests verify that the reusable simulation pattern works correctly
//! with different seeding strategies.

use pecos_engines::{BiasedDepolarizingNoise, DepolarizingNoise, PassThroughNoise};

// For now, we'll use simpler tests that don't require a full mock engine implementation.
// The integration tests with real engines (QASM, LLVM, Selene) provide the actual
// behavioral verification.

#[test]
fn test_sim_builder_api() {
    use pecos_engines::SimConfig;
    use pecos_engines::noise::IntoNoiseModel;

    // Test that SimConfig has expected defaults
    let config = SimConfig::default();
    assert_eq!(config.workers, 1);
    assert!(config.seed.is_none());
    assert!(!config.verbose);

    // Test noise conversions work with IntoNoiseModel trait
    let _: Box<dyn pecos_engines::noise::NoiseModel> = PassThroughNoise.into_noise_model();
    let _: Box<dyn pecos_engines::noise::NoiseModel> =
        DepolarizingNoise { p: 0.01 }.into_noise_model();
    let _: Box<dyn pecos_engines::noise::NoiseModel> =
        BiasedDepolarizingNoise { p: 0.01 }.into_noise_model();
}

#[test]
fn test_columnar_conversion() {
    use pecos_engines::{
        shot_results::{Data, Shot, ShotVec},
        shots_to_columnar,
    };
    use std::collections::BTreeMap;

    // Test empty shot vec
    let empty = ShotVec::new();
    let columnar = shots_to_columnar(&empty);
    assert!(columnar.is_empty());

    // Test with data
    let mut shot1 = BTreeMap::new();
    shot1.insert("q0".to_string(), Data::U32(0));
    shot1.insert("q1".to_string(), Data::U32(1));

    let mut shot2 = BTreeMap::new();
    shot2.insert("q0".to_string(), Data::U32(1));
    shot2.insert("q1".to_string(), Data::U32(0));

    let shot_vector = ShotVec {
        shots: vec![Shot { data: shot1 }, Shot { data: shot2 }],
    };

    let columnar = shots_to_columnar(&shot_vector);
    assert_eq!(columnar.len(), 2);
    assert_eq!(columnar["q0"], vec![0, 1]);
    assert_eq!(columnar["q1"], vec![1, 0]);
}
