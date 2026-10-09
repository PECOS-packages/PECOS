//! Regression tests for parse rejection and abandoned-continuation reset.
use pecos_core::{RngManageable, errors::PecosError};
use pecos_engines::byte_message::protocol::BatchHeader;
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
    assert!(matches!(
        noise.start(unsupported_version()),
        Err(PecosError::Input(_))
    ));
    let mut actual_rng = noise.rng().clone();
    assert_eq!(actual_rng.next_u64(), expected_rng.next_u64());
    let mut system = QuantumSystem::new(Box::new(noise), Box::new(StateVecEngine::new(1)));
    assert!(matches!(
        system.process(unsupported_version()),
        Err(PecosError::Input(_))
    ));
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
    assert!(
        !effects.quantum_ops().unwrap().is_empty(),
        "crosstalk continuation must contain quantum operations"
    );
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

#[test]
fn malformed_measurement_reply_propagates_input_error() {
    let mut model = GeneralNoiseModel::builder().build();
    model.start(program("M")).unwrap();
    let mut bytes = ByteMessage::outcomes_builder()
        .add_outcomes(&[0])
        .build()
        .into_bytes();
    bytes[20..24].copy_from_slice(&3_u32.to_le_bytes());
    bytes.pop();
    let len = u32::try_from(bytes.len()).unwrap();
    bytes[12..16].copy_from_slice(&len.to_le_bytes());
    match model.continue_processing(ByteMessage::new(&bytes)) {
        Err(PecosError::Input(message)) => assert_eq!(
            message,
            "Message 0: Outcome payload size must be 4, found 3"
        ),
        _ => panic!("malformed reply must propagate the outcome parse error"),
    }
}

#[test]
fn measurement_reply_empty_and_nonempty() {
    let mut model = GeneralNoiseModel::builder().build();
    assert!(
        model
            .apply_noise_on_continue_processing(ByteMessage::create_empty())
            .unwrap()
            .is_empty()
            .unwrap()
    );
    model.start(program("M")).unwrap();
    match model.continue_processing(ByteMessage::create_empty()) {
        Err(PecosError::Processing(message)) => {
            assert_eq!(message, "missing pending measurement outcomes");
        }
        _ => panic!("empty reply must report pending outcomes"),
    }
    let outcome = ByteMessage::outcomes_builder().add_outcomes(&[1]).build();
    let EngineStage::Complete(result) = model.continue_processing(outcome).unwrap() else {
        panic!("expected measurement completion");
    };
    assert_eq!(result.outcomes().unwrap(), [1]);
}
