//! Executable review probes, NOT a runtime-event implementation or wire format.
//! These independently invented fixtures test assumptions in the proposal.

use pecos_core::RngManageable;
use pecos_core::errors::PecosError;
use pecos_engines::byte_message::protocol::BatchHeader;
use pecos_engines::noise::{GeneralNoiseModel, NoiseModel, PassThroughNoiseModel};
use pecos_engines::quantum::{QuantumEngine, SparseStabEngine, StabVecEngine, StateVecEngine};
use pecos_engines::{ByteMessage, ControlEngine, Engine, EngineStage, QuantumSystem};
use pecos_random::PecosRng;
use std::any::Any;

fn gates(names: &[char]) -> ByteMessage {
    let mut builder = ByteMessage::quantum_operations_builder();
    for name in names {
        match name {
            'H' => {
                builder.h(&[0]);
            }
            'X' => {
                builder.x(&[0]);
            }
            'Z' => {
                builder.z(&[0]);
            }
            'M' => {
                builder.mz(&[0]);
            }
            _ => panic!("invalid test gate"),
        }
    }
    builder.build()
}

// A test-only decoder/executor sketch. There is deliberately no production
// MessageType or reserved Selene tag. This does NOT exercise the FFI producer.
// This intentionally incorrect segmenting sketch is NOT a capability adapter:
// the completion-noise counterexample below rules it out for production.
fn execute_probe(
    system: &mut QuantumSystem,
    before: ByteMessage,
    tag: &str,
    after: ByteMessage,
) -> Result<Vec<u32>, PecosError> {
    if !matches!(tag, "probe.flip" | "probe.phase" | "probe.ack") {
        return Err(PecosError::Input(
            "unsupported mandatory probe event".into(),
        ));
    }
    // process() must finish ALL continuations, including measurement noise.
    let mut outcomes = system.process(before)?.outcomes()?;
    let effect = match tag {
        "probe.flip" => Some('X'),
        "probe.phase" => Some('Z'),
        _ => None,
    };
    if let Some(effect) = effect {
        // Demonstrate the proposed non-recursive effect execution boundary.
        system.quantum_engine_mut().process(gates(&[effect]))?;
    }
    outcomes.extend(system.process(after)?.outcomes()?);
    Ok(outcomes)
}

#[test]
fn probe_effect_executes_between_gates_and_measurements() {
    let mut system = QuantumSystem::new_without_noise(Box::new(StateVecEngine::new(1)));
    assert_eq!(
        execute_probe(
            &mut system,
            gates(&['H']),
            "probe.phase",
            gates(&['H', 'M'])
        )
        .unwrap(),
        [1]
    );
    system.reset().unwrap();
    // Moving the effect to lowering time is observably different.
    assert_eq!(
        system
            .process(gates(&['Z', 'H', 'H', 'M']))
            .unwrap()
            .outcomes()
            .unwrap(),
        [0]
    );
    system.reset().unwrap();
    assert_eq!(
        execute_probe(&mut system, gates(&['M']), "probe.flip", gates(&['M'])).unwrap(),
        [0, 1]
    );
}

#[test]
fn unsupported_probe_is_rejected_before_execution() {
    let mut system = QuantumSystem::new_without_noise(Box::new(StateVecEngine::new(1)));
    assert!(
        execute_probe(
            &mut system,
            gates(&['X']),
            "probe.unsupported",
            gates(&['M'])
        )
        .is_err()
    );
    assert_eq!(
        system.process(gates(&['M'])).unwrap().outcomes().unwrap(),
        [0]
    );
}

#[test]
fn unknown_v1_record_is_silently_skipped_so_it_cannot_be_mandatory() {
    let mut bytes = gates(&['X']).as_bytes().to_vec();
    // Mutate an existing record instead of defining a new protocol message.
    bytes[size_of::<BatchHeader>()] = 250;
    let message = ByteMessage::new(&bytes);
    assert!(message.quantum_ops().unwrap().is_empty());
    let mut system = QuantumSystem::new_without_noise(Box::new(StateVecEngine::new(1)));
    system.process(message).unwrap();
    assert_eq!(
        system.process(gates(&['M'])).unwrap().outcomes().unwrap(),
        [0]
    );
}

fn unsupported_version() -> ByteMessage {
    let mut bytes = gates(&['X', 'Z']).as_bytes().to_vec();
    // Valid gate prefix followed by an unknown record: rejection must precede X.
    let header_len = size_of::<BatchHeader>();
    let record_len = (bytes.len() - header_len) / 2;
    bytes[header_len + record_len] = 250;
    bytes[4] = 2; // A probe only; no v2 format is defined by this test.
    ByteMessage::new(&bytes)
}

#[test]
fn unsupported_version_is_rejected_by_gate_consumers() {
    let engines: Vec<Box<dyn QuantumEngine>> = vec![
        Box::new(StateVecEngine::new(1)),
        Box::new(SparseStabEngine::new(1)),
        Box::new(StabVecEngine::new(1)),
    ];
    for engine in engines {
        let mut system = QuantumSystem::new_without_noise(engine);
        assert!(system.process(unsupported_version()).is_err());
        assert_eq!(
            system.process(gates(&['M'])).unwrap().outcomes().unwrap(),
            [0]
        );
    }
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
        system.process(gates(&['M'])).unwrap().outcomes().unwrap(),
        [0]
    );
}

/// A valid existing `NoiseModel` that applies one synthetic X at input completion.
/// No contract currently says that a model must be invariant under segmentation.
#[derive(Clone)]
struct CompletionFlip {
    base: PassThroughNoiseModel,
    pending: Option<ByteMessage>,
}

impl RngManageable for CompletionFlip {
    type Rng = PecosRng;
    fn rng(&self) -> &PecosRng {
        self.base.rng()
    }
    fn rng_mut(&mut self) -> &mut PecosRng {
        self.base.rng_mut()
    }
    fn set_rng(&mut self, rng: PecosRng) {
        self.base.set_rng(rng);
    }
}

impl NoiseModel for CompletionFlip {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl ControlEngine for CompletionFlip {
    type Input = ByteMessage;
    type Output = ByteMessage;
    type EngineInput = ByteMessage;
    type EngineOutput = ByteMessage;

    fn start(
        &mut self,
        input: ByteMessage,
    ) -> Result<EngineStage<ByteMessage, ByteMessage>, PecosError> {
        assert!(self.pending.is_none());
        Ok(EngineStage::NeedsProcessing(input))
    }

    fn continue_processing(
        &mut self,
        result: ByteMessage,
    ) -> Result<EngineStage<ByteMessage, ByteMessage>, PecosError> {
        if let Some(outcomes) = self.pending.take() {
            Ok(EngineStage::Complete(outcomes))
        } else {
            self.pending = Some(result);
            Ok(EngineStage::NeedsProcessing(gates(&['X'])))
        }
    }

    fn reset(&mut self) -> Result<(), PecosError> {
        self.pending = None;
        Ok(())
    }
}

#[test]
fn splitting_around_acknowledgement_duplicates_completion_noise() {
    let mut system = QuantumSystem::new(
        Box::new(CompletionFlip {
            base: PassThroughNoiseModel::new(),
            pending: None,
        }),
        Box::new(StateVecEngine::new(1)),
    );
    system.process(gates(&['Z', 'Z'])).unwrap();
    assert_eq!(
        system
            .quantum_engine_mut()
            .process(gates(&['M']))
            .unwrap()
            .outcomes()
            .unwrap(),
        [1]
    );
    system.reset().unwrap();
    // Same gates, with a metadata-only boundary: two completed segments cause
    // two X faults. Waiting for every continuation is necessary but insufficient.
    execute_probe(&mut system, gates(&['Z']), "probe.ack", gates(&['Z'])).unwrap();
    assert_eq!(
        system
            .quantum_engine_mut()
            .process(gates(&['M']))
            .unwrap()
            .outcomes()
            .unwrap(),
        [0]
    );
}
