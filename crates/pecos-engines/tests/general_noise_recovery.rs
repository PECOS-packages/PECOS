//! Regression tests for parse rejection and abandoned-continuation reset.
use pecos_core::RngManageable;
use pecos_engines::noise::GeneralNoiseModel;
use pecos_engines::quantum::StateVecEngine;
use pecos_engines::{ByteMessage, ControlEngine, Engine, EngineStage, QuantumSystem};
use std::collections::BTreeMap;

fn program(ops: &str) -> ByteMessage {
    let mut b = ByteMessage::quantum_operations_builder();
    for op in ops.chars() {
        match op {
            'M' => {
                b.mz(&[0]);
            }
            'L' => {
                b.measure_leakages(&[0]);
            }
            'P' => {
                b.pz(&[0]);
            }
            'X' => {
                b.x(&[0]);
            }
            'I' => {
                b.idle(0.25, &[0]);
            }
            _ => panic!("invalid fixture"),
        }
    }
    b.build()
}

use pecos_engines::byte_message::protocol::BatchHeader;

fn unsupported_version() -> ByteMessage {
    let mut bytes = program("XX").as_bytes().to_vec();
    // Valid gate prefix followed by an unknown record: rejection must precede X.
    let header_len = size_of::<BatchHeader>();
    let record_len = (bytes.len() - header_len) / 2;
    bytes[header_len + record_len] = 250;
    bytes[4] = 2; // A probe only; no v2 format is defined by this test.
    ByteMessage::new(&bytes)
}

#[test]
fn general_noise_returns_version_error_without_execution_or_rng_consumption() {
    let mut noise = GeneralNoiseModel::builder().build();
    let mut expected_rng = noise.rng().clone();
    assert!(noise.start(unsupported_version()).is_err());
    let mut actual_rng = noise.rng().clone();
    assert_eq!(actual_rng.next_u64(), expected_rng.next_u64());
    let mut system = QuantumSystem::new(Box::new(noise), Box::new(StateVecEngine::new(1)));
    assert!(system.process(unsupported_version()).is_err());
    assert_eq!(
        system.process(program("M")).unwrap().outcomes().unwrap(),
        [0]
    );
}

fn crosstalk_model() -> GeneralNoiseModel {
    GeneralNoiseModel::builder()
        .with_p_meas_crosstalk_local(1.0)
        .with_p_meas_crosstalk_model(&BTreeMap::from([
            ("0->1".to_owned(), 1.0),
            ("1->0".to_owned(), 1.0),
        ]))
        .with_seed(19)
        .build()
}

fn pending_crosstalk(model: &mut GeneralNoiseModel) {
    let mut b = ByteMessage::quantum_operations_builder();
    b.pz(&[0, 1]);
    b.mz(&[0]);
    b.meas_crosstalk_local_payload(&[1]);
    let EngineStage::NeedsProcessing(commands) = model.start(b.build()).unwrap() else {
        panic!("expected simulator work");
    };
    let mut sim = StateVecEngine::new(2);
    let outcomes = sim.process(commands).unwrap();
    let EngineStage::NeedsProcessing(effects) = model.continue_processing(outcomes).unwrap() else {
        panic!("expected crosstalk continuation");
    };
    assert!(!effects.quantum_ops().unwrap().is_empty());
}

#[test]
fn reset_after_continuation_error_discards_pending_results() {
    let mut model = crosstalk_model();
    pending_crosstalk(&mut model);
    // The simulator/controller exchange fails while a user outcome is retained.
    let unexpected_outcome = ByteMessage::outcomes_builder().add_outcomes(&[0]).build();
    assert!(model.continue_processing(unexpected_outcome).is_err());
    model.reset().unwrap();
    let mut system = QuantumSystem::new(Box::new(model), Box::new(StateVecEngine::new(2)));
    assert_eq!(
        system.process(program("M")).unwrap().outcomes().unwrap(),
        [0]
    );
}

#[test]
fn resetting_clone_does_not_clear_original_pending_results() {
    let mut original = crosstalk_model();
    pending_crosstalk(&mut original);
    let mut cloned = original.clone();
    cloned.reset().unwrap();
    let empty_outcomes = ByteMessage::outcomes_builder().build();
    let EngineStage::Complete(result) = original.continue_processing(empty_outcomes).unwrap()
    else {
        panic!("expected completion");
    };
    assert_eq!(result.outcomes().unwrap(), [0]);
    assert!(cloned.start(program("M")).is_ok());
}
