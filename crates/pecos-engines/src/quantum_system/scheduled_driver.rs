//! Internal driver tests, not admission of spectator effects on v6/v7 input.
use super::*;
use crate::{ControlEngine, EngineStage, noise::GeneralNoiseModel};
use pecos_core::RngManageable;
use pecos_random::PecosRng;
use std::{
    any::Any,
    collections::{BTreeMap, VecDeque},
};

#[test]
fn byte_limit_rejects_one_oversized_command_before_execution() {
    let mut system = QuantumSystem::new_without_noise(Box::new(crate::StabilizerEngine::new(33)));
    // A valid legacy grouped gate has one command but a large payload. Such
    // output is outside the checked profiles' <=2-target emission contract.
    let commands = ByteMessage::quantum_operations_builder()
        .x(&(0..33).collect::<Vec<_>>())
        .build();
    assert_eq!(commands.quantum_ops().unwrap().len(), 1);
    assert!(commands.as_bytes().len() > 16 + 128);
    let result = system.drive_scheduled(
        crate::EngineStage::NeedsProcessing(commands),
        ScheduledExecutionLimits::single_pass(1).unwrap(),
    );
    assert!(
        result
            .err()
            .expect("byte limit must reject")
            .to_string()
            .contains("scheduled noise expansion limit")
    );
    let output = system
        .drive_legacy(ByteMessage::quantum_operations_builder().mz(&[0]).build())
        .unwrap();
    assert_eq!(output.outcomes().unwrap(), vec![0]);
}

#[test]
fn expanded_commands_reject_before_simulator_mutation() {
    let mut system = QuantumSystem::new_without_noise(Box::new(crate::StateVecEngine::new(1)));
    for budget in [0, 2, usize::MAX] {
        let mut commands = ByteMessage::quantum_operations_builder();
        commands.x(&[0]).x(&[0]).x(&[0]);
        let result = ScheduledExecutionLimits::single_pass(budget).and_then(|limits| {
            system.drive_scheduled(
                crate::EngineStage::NeedsProcessing(commands.build()),
                limits,
            )
        });
        assert!(result.is_err());
    }
    let result = system
        .drive_legacy(ByteMessage::quantum_operations_builder().mz(&[0]).build())
        .unwrap();
    assert_eq!(result.outcomes().unwrap(), vec![0]);
}

#[test]
fn internal_measurements_resume_once_and_do_not_escape() {
    let model = GeneralNoiseModel::builder()
        .with_p_meas_crosstalk(1.0)
        .with_p_meas_crosstalk_model(&BTreeMap::from([
            ("0->1".into(), 1.0),
            ("1->0".into(), 1.0),
        ]))
        .build();
    let mut system = QuantumSystem::new(Box::new(model), Box::new(crate::StateVecEngine::new(1)));
    let input = ByteMessage::quantum_operations_builder()
        .pz(&[0])
        .meas_crosstalk_local_payload(&[0])
        .mz(&[0])
        .meas_crosstalk_local_payload(&[0])
        .mz(&[0])
        .x(&[0])
        .build();
    let stage = system
        .noise_model
        .as_any_mut()
        .downcast_mut::<GeneralNoiseModel>()
        .unwrap()
        .start_scheduled(&input)
        .unwrap();
    let output = system
        .drive_scheduled(
            stage,
            ScheduledExecutionLimits {
                operations: 32,
                bytes: 4096,
                simulator_calls: 8,
            },
        )
        .unwrap();
    assert_eq!(output.outcomes().unwrap(), vec![1, 0]);
    // Completion consumes internal results while preserving the quantum state.
    // The trailing X leaves |1>; if completion cleared the simulator, the next
    // batch would read 0 and this assertion would fail.
    let stage = system
        .noise_model
        .as_any_mut()
        .downcast_mut::<GeneralNoiseModel>()
        .unwrap()
        .start_scheduled(&ByteMessage::quantum_operations_builder().mz(&[0]).build())
        .unwrap();
    assert_eq!(
        system
            .drive_scheduled(stage, ScheduledExecutionLimits::single_pass(16).unwrap())
            .unwrap()
            .outcomes()
            .unwrap(),
        vec![1]
    );
}

// Scripted outputs let the tests exhaust each budget independently. The real
// simulator still executes the admitted prefix, so rejected suffix mutations
// are observable. This controller is never admitted by the public scheduled API.
#[derive(Clone)]
struct Script {
    pending: VecDeque<ByteMessage>,
    endless: bool,
    replies: usize,
    starts: usize,
    rng: PecosRng,
}
impl Script {
    fn new(pending: Vec<ByteMessage>, endless: bool) -> Self {
        Self {
            pending: pending.into(),
            endless,
            replies: 0,
            starts: 0,
            rng: PecosRng::seed_from_u64(0),
        }
    }
}
impl ControlEngine for Script {
    type Input = ByteMessage;
    type Output = ByteMessage;
    type EngineInput = ByteMessage;
    type EngineOutput = ByteMessage;
    fn start(
        &mut self,
        input: ByteMessage,
    ) -> Result<EngineStage<ByteMessage, ByteMessage>, PecosError> {
        self.starts += 1;
        Ok(EngineStage::NeedsProcessing(input))
    }
    fn continue_processing(
        &mut self,
        reply: ByteMessage,
    ) -> Result<EngineStage<ByteMessage, ByteMessage>, PecosError> {
        self.replies += 1;
        Ok(if let Some(commands) = self.pending.pop_front() {
            EngineStage::NeedsProcessing(commands)
        } else if self.endless {
            EngineStage::NeedsProcessing(ByteMessage::quantum_operations_builder().build())
        } else {
            EngineStage::Complete(reply)
        })
    }
    fn reset(&mut self) -> Result<(), PecosError> {
        Ok(())
    }
}
impl NoiseModel for Script {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
impl RngManageable for Script {
    type Rng = PecosRng;
    fn rng(&self) -> &PecosRng {
        &self.rng
    }
    fn rng_mut(&mut self) -> &mut PecosRng {
        &mut self.rng
    }
    fn set_rng(&mut self, rng: PecosRng) {
        self.rng = rng;
    }
}
fn scripted(pending: Vec<ByteMessage>, endless: bool) -> QuantumSystem {
    QuantumSystem::new(
        Box::new(Script::new(pending, endless)),
        Box::new(crate::StateVecEngine::new(1)),
    )
}
fn script(system: &QuantumSystem) -> &Script {
    system.noise_model.as_any().downcast_ref().unwrap()
}
fn measure(system: &mut QuantumSystem) -> Vec<u32> {
    system
        .quantum_engine
        .process(ByteMessage::quantum_operations_builder().mz(&[0]).build())
        .unwrap()
        .outcomes()
        .unwrap()
}

#[test]
fn cumulative_commands_reject_suffix_before_mutation() {
    let x = ByteMessage::quantum_operations_builder().x(&[0]).build();
    let mut system = scripted(vec![x.clone()], false);
    let stage = system.noise_model.start(x).unwrap();
    let error = system
        .drive_scheduled(
            stage,
            ScheduledExecutionLimits {
                operations: 1,
                bytes: 4096,
                simulator_calls: 8,
            },
        )
        .err()
        .expect("expected rejection");
    assert!(
        error
            .to_string()
            .contains("(operations): request needs 1, 0 of 1 remain"),
        "{error}"
    );
    assert_eq!(script(&system).starts, 1);
    assert_eq!(script(&system).replies, 1);
    assert_eq!(measure(&mut system), vec![1]);
}

#[test]
fn cumulative_bytes_reject_suffix_before_mutation() {
    let x = ByteMessage::quantum_operations_builder().x(&[0]).build();
    let len = x.as_bytes().len();
    let bytes = 2 * len - 1;
    let mut system = scripted(vec![x.clone()], false);
    let stage = system.noise_model.start(x).unwrap();
    let error = system
        .drive_scheduled(
            stage,
            ScheduledExecutionLimits {
                operations: 2,
                bytes,
                simulator_calls: 8,
            },
        )
        .err()
        .expect("expected rejection");
    assert!(
        error.to_string().contains(&format!(
            "(bytes): request needs {len}, {} of {bytes} remain",
            len - 1
        )),
        "{error}"
    );
    assert_eq!(script(&system).replies, 1);
    assert_eq!(measure(&mut system), vec![1]);
}

#[test]
fn exact_limits_allow_completion_without_restarting() {
    let x = ByteMessage::quantum_operations_builder().x(&[0]).build();
    let mz = ByteMessage::quantum_operations_builder().mz(&[0]).build();
    let bytes = x.as_bytes().len() + mz.as_bytes().len();
    let mut system = scripted(vec![mz], false);
    let stage = system.noise_model.start(x).unwrap();
    assert_eq!(
        system
            .drive_scheduled(
                stage,
                ScheduledExecutionLimits {
                    operations: 2,
                    bytes,
                    simulator_calls: 2,
                }
            )
            .unwrap()
            .outcomes()
            .unwrap(),
        vec![1]
    );
    assert_eq!(script(&system).starts, 1);
    assert_eq!(script(&system).replies, 2);
}

#[test]
fn empty_continuations_exhaust_calls_and_headers() {
    for (calls, bytes, expected) in [(2, 1024, "call"), (8, 32, "bytes")] {
        let mut system = scripted(vec![], true);
        let stage = system
            .noise_model
            .start(ByteMessage::quantum_operations_builder().build())
            .unwrap();
        let error = system
            .drive_scheduled(
                stage,
                ScheduledExecutionLimits {
                    operations: 0,
                    bytes,
                    simulator_calls: calls,
                },
            )
            .err()
            .expect("expected rejection");
        assert!(error.to_string().contains(expected), "{error}");
        assert_eq!(script(&system).starts, 1);
        assert_eq!(script(&system).replies, 2);
    }
}

#[test]
fn existing_profiles_still_reject_a_second_call() {
    let x = ByteMessage::quantum_operations_builder().x(&[0]).build();
    let mut system = scripted(vec![x.clone()], false);
    let stage = system.noise_model.start(x).unwrap();
    let error = system
        .drive_scheduled(stage, ScheduledExecutionLimits::single_pass(32).unwrap())
        .err()
        .expect("expected rejection");
    assert!(
        error
            .to_string()
            .contains("call limit: batch allows 1 call(s)"),
        "{error}"
    );
    assert_eq!(script(&system).replies, 1);
    assert_eq!(measure(&mut system), vec![1]);
}

#[test]
fn zero_calls_reject_before_execution_and_complete_needs_no_call() {
    let mut system = scripted(vec![], false);
    let stage = system
        .noise_model
        .start(ByteMessage::quantum_operations_builder().x(&[0]).build())
        .unwrap();
    assert!(
        system
            .drive_scheduled(
                stage,
                ScheduledExecutionLimits {
                    operations: 16,
                    bytes: 4096,
                    simulator_calls: 0,
                }
            )
            .err()
            .expect("expected rejection")
            .to_string()
            .contains("call limit")
    );
    assert_eq!(script(&system).replies, 0);
    assert_eq!(measure(&mut system), vec![0]);
    let output = ByteMessage::outcomes_builder().add_outcomes(&[1]).build();
    assert_eq!(
        system
            .drive_scheduled(
                EngineStage::Complete(output),
                ScheduledExecutionLimits {
                    operations: 0,
                    bytes: 0,
                    simulator_calls: 0,
                }
            )
            .unwrap()
            .outcomes()
            .unwrap(),
        vec![1]
    );
    assert_eq!(script(&system).replies, 0);
}

#[test]
fn bounded_driver_matches_unsplit_controller_rng_and_results() {
    let model = GeneralNoiseModel::builder()
        .with_p_meas_crosstalk(0.7)
        .with_p_meas_crosstalk_model(&BTreeMap::from([
            ("0->1".into(), 0.3),
            ("0->L".into(), 0.7),
            ("1->0".into(), 0.6),
            ("1->L".into(), 0.4),
        ]))
        .build();
    for seed in 0..64 {
        let mut bounded = QuantumSystem::new(
            Box::new(model.clone()),
            Box::new(crate::StateVecEngine::new(2)),
        );
        let mut reference = QuantumSystem::new(
            Box::new(model.clone()),
            Box::new(crate::StateVecEngine::new(2)),
        );
        bounded.set_seed(seed);
        reference.set_seed(seed);
        // The middle batch retains leakage and quantum state; only the first
        // and third batches prepare/reset. Internal results never reach the user.
        for batch in 0..3 {
            let mut builder = ByteMessage::quantum_operations_builder();
            if batch != 1 {
                builder.pz(&[0, 1]);
            }
            let input = builder
                .h(&[0])
                .meas_crosstalk_local_payload(&[0])
                .mz(&[0])
                .meas_crosstalk_global_payload(&[0])
                .mz(&[1])
                .build();
            let stage = bounded
                .noise_model
                .as_any_mut()
                .downcast_mut::<GeneralNoiseModel>()
                .unwrap()
                .start_scheduled(&input)
                .unwrap();
            let actual = bounded
                .drive_scheduled(
                    stage,
                    ScheduledExecutionLimits {
                        operations: 128,
                        bytes: 16384,
                        simulator_calls: 16,
                    },
                )
                .unwrap();
            let expected = reference.drive_legacy(input).unwrap();
            assert_eq!(
                actual.outcomes().unwrap(),
                expected.outcomes().unwrap(),
                "outcomes, seed {seed}, batch {batch}"
            );
            assert_eq!(
                bounded.noise_model.rng().clone().next_u64(),
                reference.noise_model.rng().clone().next_u64(),
                "noise RNG, seed {seed}, batch {batch}"
            );
            assert_eq!(
                bounded
                    .quantum_engine
                    .as_any()
                    .downcast_ref::<crate::StateVecEngine>()
                    .unwrap()
                    .simulator()
                    .rng()
                    .clone()
                    .next_u64(),
                reference
                    .quantum_engine
                    .as_any()
                    .downcast_ref::<crate::StateVecEngine>()
                    .unwrap()
                    .simulator()
                    .rng()
                    .clone()
                    .next_u64(),
                "simulator RNG, seed {seed}, batch {batch}"
            );
        }
    }
}

#[test]
fn malformed_continuation_stops_before_dispatch_or_further_reply() {
    // This is a deliberately invalid header, not an admitted scheduled record.
    let malformed = ByteMessage::new(&[0xff; 16]);
    assert!(malformed.quantum_ops().is_err());
    let mut system = scripted(vec![malformed], false);
    let stage = system
        .noise_model
        .start(ByteMessage::quantum_operations_builder().x(&[0]).build())
        .unwrap();
    assert!(
        system
            .drive_scheduled(
                stage,
                ScheduledExecutionLimits {
                    operations: 16,
                    bytes: 4096,
                    simulator_calls: 8,
                }
            )
            .is_err()
    );
    assert_eq!(script(&system).starts, 1);
    assert_eq!(script(&system).replies, 1);
    assert_eq!(measure(&mut system), vec![1]);
}

#[test]
fn non_clifford_continuation_rejects_before_stabilizer_mutation() {
    // The stabilizer would apply X before failing on RZ, leaving a partial batch.
    let mut continuation = ByteMessage::quantum_operations_builder();
    continuation
        .x(&[0])
        .rz(pecos_core::Angle64::QUARTER_TURN / 2u64, &[0]);
    let mut system = QuantumSystem::new(
        Box::new(Script::new(vec![continuation.build()], false)),
        Box::new(crate::StabilizerEngine::new(1)),
    );
    let stage = system
        .noise_model
        .start(ByteMessage::quantum_operations_builder().x(&[0]).build())
        .unwrap();
    let error = system
        .drive_scheduled(
            stage,
            ScheduledExecutionLimits {
                operations: 16,
                bytes: 4096,
                simulator_calls: 8,
            },
        )
        .err()
        .expect("expected rejection");
    assert!(error.to_string().contains("Clifford"), "{error}");
    assert_eq!(script(&system).replies, 1);
    assert_eq!(measure(&mut system), vec![1]);
}

#[test]
fn byte_limit_precedes_malformed_continuation_parsing() {
    let malformed = ByteMessage::new(&[0xff; 16]);
    let x = ByteMessage::quantum_operations_builder().x(&[0]).build();
    let bytes = x.as_bytes().len() + malformed.as_bytes().len() - 1;
    let mut system = scripted(vec![malformed], false);
    let stage = system.noise_model.start(x).unwrap();
    let error = system
        .drive_scheduled(
            stage,
            ScheduledExecutionLimits {
                operations: 16,
                bytes,
                simulator_calls: 8,
            },
        )
        .err()
        .expect("expected rejection");
    assert!(
        error.to_string().contains("expansion limit (bytes)"),
        "{error}"
    );
    assert_eq!(script(&system).replies, 1);
    assert_eq!(measure(&mut system), vec![1]);
}
