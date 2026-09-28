use super::*;
use pecos_qis::runtime::{ClassicalState, Result as RuntimeResult, RuntimeError};
use pecos_qis::scheduled::ScheduledMeasurement;
use pecos_qis_ffi_types::{OperationCollector, QuantumOp};
use std::collections::VecDeque;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

#[derive(Clone, Default)]
struct Fixture {
    state: ClassicalState,
    batches: VecDeque<Vec<ScheduledBatch>>,
    fail_feedback: bool,
    panic_feedback: bool,
    fail_reset: Arc<AtomicBool>,
    reset_count: Arc<std::sync::atomic::AtomicUsize>,
}
impl QisRuntime for Fixture {
    fn load_interface(&mut self, _: OperationCollector) -> RuntimeResult<()> {
        unreachable!()
    }
    fn execute_until_quantum(&mut self) -> RuntimeResult<Option<Vec<QuantumOp>>> {
        unreachable!()
    }
    fn provide_measurements(&mut self, m: BTreeMap<usize, bool>) -> RuntimeResult<()> {
        self.state.measurements.extend(m);
        Ok(())
    }
    fn provide_measurement_outcomes(&mut self, m: BTreeMap<usize, u32>) -> RuntimeResult<()> {
        assert!(!self.panic_feedback, "synthetic feedback panic");
        if self.fail_feedback {
            return Err(RuntimeError::ExecutionError(
                "synthetic feedback failure".into(),
            ));
        }
        self.provide_measurements(m.into_iter().map(|(id, v)| (id, v != 0)).collect())
    }
    fn get_classical_state(&self) -> &ClassicalState {
        &self.state
    }
    fn get_classical_state_mut(&mut self) -> &mut ClassicalState {
        &mut self.state
    }
    fn is_complete(&self) -> bool {
        self.batches.is_empty()
    }
    fn num_qubits(&self) -> usize {
        4
    }
    fn reset(&mut self) -> RuntimeResult<()> {
        self.reset_count.fetch_add(1, Ordering::SeqCst);
        if self.fail_reset.load(Ordering::SeqCst) {
            return Err(RuntimeError::ExecutionError(
                "synthetic reset failure".into(),
            ));
        }
        self.state = ClassicalState::default();
        Ok(())
    }
    fn lower_scheduled_operations(
        &mut self,
        _: &[Operation],
    ) -> RuntimeResult<Vec<ScheduledBatch>> {
        Ok(self.batches.pop_front().unwrap_or_default())
    }
    fn drain_pending_scheduled_operations(&mut self) -> RuntimeResult<Vec<ScheduledBatch>> {
        Ok(self.batches.pop_front().unwrap_or_default())
    }
}
fn context(shot: usize) -> ShotContext {
    ShotContext {
        run: 19,
        worker: 2,
        shot,
    }
}
fn pulse() -> RuntimeScheduledOp {
    RuntimeScheduledOp::Rxy {
        qubit_id: 0,
        theta: std::f64::consts::PI,
        phi: 0.0,
    }
}
fn batch(index: usize, operations: Vec<RuntimeScheduledOp>) -> ScheduledBatch {
    ScheduledBatch {
        runtime_shot_id: 7,
        batch_index: index,
        start_time_nanos: (index as u64) * 10,
        duration_nanos: 10,
        operations,
        measurements: vec![],
    }
}
fn fixture_executor(f: Fixture) -> ScheduledExecutor {
    let mut executor = ScheduledExecutor::new(Box::new(f), 4).unwrap();
    executor.start_shot(context(3), 7, 123).unwrap();
    executor
}
fn measurement(batch: &mut ScheduledBatch, result: usize) {
    batch.measurements.push(ScheduledMeasurement {
        operation_index: batch.operations.len(),
        runtime_result: 901,
        program_result: result,
        leakage_aware: false,
    });
    batch.operations.push(RuntimeScheduledOp::Measure {
        qubit_id: 0,
        result_id: 901,
    });
}

#[test]
fn unknown_event_in_later_batch_rejects_before_any_quantum_mutation() {
    let f = Fixture {
        batches: VecDeque::from([vec![
            batch(0, vec![pulse()]),
            batch(
                1,
                vec![RuntimeScheduledOp::Custom {
                    tag: 7301,
                    data: vec![17],
                }],
            ),
        ]]),
        ..Default::default()
    };
    let mut executor = fixture_executor(f);
    assert!(
        executor
            .submit(&[])
            .unwrap_err()
            .to_string()
            .contains("custom events")
    );
    let state = executor
        .quantum
        .process(ByteMessage::quantum_operations_builder().mz(&[0]).build())
        .unwrap();
    assert_eq!(state.outcomes().unwrap(), [0]);
    assert!(executor.submit(&[]).is_err());
    assert!(executor.finish_shot().is_err());
    assert!(executor.start_shot(context(4), 8, 123).is_err());
    executor.reset().unwrap();
    executor.start_shot(context(4), 8, 123).unwrap();
}

#[test]
fn entire_input_admission_rejects_bad_identity_timing_targets_and_mappings() {
    for case in 0..7 {
        let mut second = batch(1, vec![pulse()]);
        match case {
            0 => second.batch_index = 9,
            1 => second.runtime_shot_id = 100,
            2 => second.start_time_nanos = 5,
            3 => second.duration_nanos = u64::MAX,
            4 => second.operations[0] = RuntimeScheduledOp::Reset { qubit_id: 999 },
            5 => {
                second.operations[0] = RuntimeScheduledOp::Rz {
                    qubit_id: 0,
                    theta: f64::NAN,
                }
            }
            _ => second.measurements.push(ScheduledMeasurement {
                operation_index: 0,
                runtime_result: 901,
                program_result: 0,
                leakage_aware: false,
            }),
        }
        let mut executor = fixture_executor(Fixture {
            batches: VecDeque::from([vec![batch(0, vec![pulse()]), second]]),
            ..Default::default()
        });
        assert!(executor.submit(&[]).is_err(), "case {case}");
        let state = executor
            .quantum
            .process(ByteMessage::quantum_operations_builder().mz(&[0]).build())
            .unwrap();
        assert_eq!(state.outcomes().unwrap(), [0]);
    }
}

#[test]
fn measurements_use_program_ids_and_feedback_precedes_completion() {
    let mut b = batch(0, vec![pulse()]);
    measurement(&mut b, 71);
    let mut executor = fixture_executor(Fixture {
        batches: VecDeque::from([vec![b.clone()]]),
        ..Default::default()
    });
    let output = executor.submit(&[]).unwrap();
    assert_eq!(output.batches, [b]);
    assert_eq!(output.context, context(3));
    assert_eq!(output.measurements, BTreeMap::from([(71, 1)]));
    let (_, shot) = executor.finish_shot().unwrap();
    assert_eq!(shot.measurements, BTreeMap::from([(71, true)]));
    assert!(executor.submit(&[]).is_err());
}

#[test]
fn feedback_failure_and_panic_poison_the_owner() {
    for panic_feedback in [false, true] {
        let mut b = batch(0, vec![pulse()]);
        measurement(&mut b, 71);
        let mut executor = fixture_executor(Fixture {
            batches: VecDeque::from([vec![b]]),
            fail_feedback: !panic_feedback,
            panic_feedback,
            ..Default::default()
        });
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| executor.submit(&[])));
        assert!(result.is_err() || result.unwrap().is_err());
        assert!(executor.finish_shot().is_err());
        assert!(executor.start_shot(context(4), 8, 0).is_err());
    }
}

#[test]
fn failed_reset_cannot_restore_authorization() {
    let f = Fixture::default();
    let fail = f.fail_reset.clone();
    let mut executor = fixture_executor(f);
    fail.store(true, Ordering::SeqCst);
    assert!(executor.reset().is_err());
    assert!(executor.submit(&[]).is_err());
    fail.store(false, Ordering::SeqCst);
    executor.reset().unwrap();
    executor.start_shot(context(4), 8, 123).unwrap();
}

#[test]
fn persistent_state_and_batch_order_survive_multiple_inputs() {
    let mut measure = batch(1, vec![]);
    measurement(&mut measure, 8);
    let mut executor = fixture_executor(Fixture {
        batches: VecDeque::from([vec![batch(0, vec![pulse()])], vec![measure]]),
        ..Default::default()
    });
    assert!(executor.submit(&[]).unwrap().measurements.is_empty());
    assert_eq!(executor.submit(&[]).unwrap().measurements[&8], 1);
}

#[cfg(feature = "selene")]
#[test]
fn real_native_runtimes_execute_large_schedules_and_deliver_measurements() {
    for runtime in [
        pecos_qis::selene_simple_runtime().unwrap(),
        pecos_qis::selene_soft_rz_runtime().unwrap(),
    ] {
        let mut executor = ScheduledExecutor::new(Box::new(runtime), 4).unwrap();
        for shot_id in 0..2 {
            executor
                .start_shot(context(shot_id), shot_id as u64, 123)
                .unwrap();
            let mut ops = (0..200)
                .map(|_| QuantumOp::RXY(std::f64::consts::PI, 0.0, 0).into())
                .collect::<Vec<Operation>>();
            ops.push(QuantumOp::Measure(0, 41).into());
            let initial = executor.submit(&ops).unwrap();
            let (terminal, shot) = executor.finish_shot().unwrap();
            let results = initial
                .measurements
                .into_iter()
                .chain(terminal.measurements)
                .collect::<BTreeMap<_, _>>();
            assert_eq!(results, BTreeMap::from([(41, 0)]));
            assert_eq!(shot.measurements.get(&41), Some(&false));
        }
    }
}

#[cfg(feature = "selene")]
#[test]
fn two_qubit_native_rotations_and_leakage_aware_readout_execute() {
    let runtime = pecos_qis::selene_simple_runtime().unwrap();
    let mut executor = ScheduledExecutor::new(Box::new(runtime), 2).unwrap();
    executor.start_shot(context(0), 10, 123).unwrap();
    let output = executor
        .submit(&[
            Operation::AllocateQubit { id: 7 },
            Operation::AllocateQubit { id: 3 },
            QuantumOp::RXYXY2Q(std::f64::consts::PI, 0.0, 7, 3).into(),
            QuantumOp::RZZ(0.73, 7, 3).into(),
            QuantumOp::MeasureLeaked(7, 100).into(),
            QuantumOp::MeasureLeaked(3, 200).into(),
        ])
        .unwrap();
    let (terminal, _) = executor.finish_shot().unwrap();
    let outcomes = output
        .measurements
        .into_iter()
        .chain(terminal.measurements)
        .collect::<BTreeMap<_, _>>();
    assert_eq!(outcomes, BTreeMap::from([(100, 1), (200, 1)]));
}

#[cfg(feature = "selene")]
#[test]
fn matched_seeded_measurements_are_invariant_to_submission_boundaries() {
    fn execute(split: bool, seed: u64) -> BTreeMap<usize, u32> {
        let mut executor =
            ScheduledExecutor::new(Box::new(pecos_qis::selene_simple_runtime().unwrap()), 2)
                .unwrap();
        executor
            .start_shot(context(usize::try_from(seed).unwrap()), seed, seed)
            .unwrap();
        let prep: Vec<Operation> = vec![
            QuantumOp::RXY(std::f64::consts::FRAC_PI_2, 0.0, 0).into(),
            QuantumOp::RXY(std::f64::consts::FRAC_PI_2, 0.0, 1).into(),
        ];
        let meas: Vec<Operation> = vec![
            QuantumOp::Measure(0, 40).into(),
            QuantumOp::Measure(1, 80).into(),
        ];
        let output = if split {
            assert!(executor.submit(&prep).unwrap().measurements.is_empty());
            executor.submit(&meas).unwrap()
        } else {
            executor
                .submit(&prep.into_iter().chain(meas).collect::<Vec<_>>())
                .unwrap()
        };
        let (tail, _) = executor.finish_shot().unwrap();
        output
            .measurements
            .into_iter()
            .chain(tail.measurements)
            .collect()
    }
    for seed in 0..8 {
        assert_eq!(execute(false, seed), execute(true, seed));
    }
}

#[cfg(feature = "selene")]
#[test]
fn independent_owners_do_not_share_state_or_host_context() {
    let mut a =
        ScheduledExecutor::new(Box::new(pecos_qis::selene_simple_runtime().unwrap()), 2).unwrap();
    let mut b =
        ScheduledExecutor::new(Box::new(pecos_qis::selene_simple_runtime().unwrap()), 2).unwrap();
    a.start_shot(
        ShotContext {
            run: 1,
            worker: 0,
            shot: 0,
        },
        0,
        123,
    )
    .unwrap();
    b.start_shot(
        ShotContext {
            run: 1,
            worker: 1,
            shot: 0,
        },
        0,
        123,
    )
    .unwrap();
    a.submit(&[QuantumOp::RXY(std::f64::consts::PI, 0.0, 0).into()])
        .unwrap();
    let ao = a.submit(&[QuantumOp::Measure(0, 3).into()]).unwrap();
    let bo = b.submit(&[QuantumOp::Measure(0, 3).into()]).unwrap();
    assert_eq!(ao.measurements[&3], 1);
    assert_eq!(bo.measurements[&3], 0);
    assert_ne!(ao.context, bo.context);
    a.finish_shot().unwrap();
    b.finish_shot().unwrap();
}

#[test]
fn measurement_identity_guards_reject_each_malformed_mapping() {
    for case in 0..4 {
        let mut b = batch(0, vec![]);
        measurement(&mut b, 40);
        b.operations.push(RuntimeScheduledOp::Measure {
            qubit_id: 1,
            result_id: 902,
        });
        b.measurements.push(ScheduledMeasurement {
            operation_index: 1,
            runtime_result: 902,
            program_result: 41,
            leakage_aware: false,
        });
        match case {
            0 => b.measurements[1].program_result = 40,
            1 => {
                b.operations[1] = RuntimeScheduledOp::Measure {
                    qubit_id: 1,
                    result_id: 901,
                };
                b.measurements[1].runtime_result = 901;
            }
            2 => b.measurements[1].runtime_result = 999,
            _ => {
                b.operations[1] = RuntimeScheduledOp::MeasureLeaked {
                    qubit_id: 1,
                    result_id: 902,
                }
            }
        }
        let mut executor = fixture_executor(Fixture {
            batches: VecDeque::from([vec![b]]),
            ..Default::default()
        });
        assert!(
            executor
                .submit(&[])
                .unwrap_err()
                .to_string()
                .contains("invalid or duplicate scheduled measurement identity"),
            "case {case}"
        );
        assert!(executor.poisoned);
    }
}

#[test]
fn two_qubit_rotations_require_distinct_targets() {
    for op in [
        RuntimeScheduledOp::Rzz {
            qubit_id_1: 0,
            qubit_id_2: 0,
            theta: 0.5,
        },
        RuntimeScheduledOp::Rpp {
            qubit_id_1: 1,
            qubit_id_2: 1,
            theta: 0.5,
            phi: 0.25,
        },
    ] {
        let executor = fixture_executor(Fixture::default());
        assert!(
            executor
                .admit(&[batch(0, vec![op])])
                .err()
                .unwrap()
                .to_string()
                .contains("invalid scheduled gate")
        );
    }
}

#[test]
fn leakage_aware_metadata_selects_the_encoded_measurement_command() {
    use pecos_engines::GateType;
    for raw_leaked in [false, true] {
        let executor = fixture_executor(Fixture::default());
        let mut b = batch(0, vec![]);
        measurement(&mut b, 40);
        b.measurements[0].leakage_aware = true;
        if raw_leaked {
            b.operations[0] = RuntimeScheduledOp::MeasureLeaked {
                qubit_id: 0,
                result_id: 901,
            };
        }
        let AdmittedSchedule { commands, .. } = executor.admit(&[b]).unwrap();
        let gates = commands.quantum_ops().unwrap();
        assert_eq!(gates.len(), 1);
        assert_eq!(gates[0].gate_type, GateType::MeasureLeaked);
    }
    let executor = fixture_executor(Fixture::default());
    let mut b = batch(0, vec![]);
    measurement(&mut b, 40);
    let AdmittedSchedule { commands, .. } = executor.admit(&[b]).unwrap();
    assert_eq!(commands.quantum_ops().unwrap()[0].gate_type, GateType::MZ);
}

#[derive(Clone, Debug)]
struct WrongOutcomes(Vec<usize>);
impl Engine for WrongOutcomes {
    type Input = ByteMessage;
    type Output = ByteMessage;
    fn process(&mut self, _: ByteMessage) -> Result<ByteMessage, PecosError> {
        Ok(ByteMessage::outcomes_builder()
            .add_outcomes(&self.0)
            .build())
    }
    fn reset(&mut self) -> Result<(), PecosError> {
        Ok(())
    }
}
impl pecos_engines::quantum::QuantumEngine for WrongOutcomes {
    fn set_seed(&mut self, _: u64) {}
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

#[test]
fn missing_and_extra_quantum_outcomes_fail_before_feedback() {
    for outcomes in [vec![], vec![0, 1]] {
        let mut b = batch(0, vec![]);
        measurement(&mut b, 40);
        let mut executor = fixture_executor(Fixture {
            batches: VecDeque::from([vec![b]]),
            ..Default::default()
        });
        executor.quantum = QuantumSystem::new_without_noise(Box::new(WrongOutcomes(outcomes)));
        assert!(
            executor
                .submit(&[])
                .unwrap_err()
                .to_string()
                .contains("measurement count mismatch")
        );
        assert!(
            executor
                .runtime
                .get_classical_state()
                .measurements
                .is_empty()
        );
        assert!(executor.finish_shot().is_err());
    }
}

#[test]
fn clean_shots_reuse_runtime_but_reset_quantum_state() {
    let mut first = batch(0, vec![pulse()]);
    measurement(&mut first, 40);
    let mut second = batch(0, vec![]);
    measurement(&mut second, 41);
    second.runtime_shot_id = 8;
    let f = Fixture {
        batches: VecDeque::from([vec![first], vec![], vec![second], vec![]]),
        ..Default::default()
    };
    let resets = f.reset_count.clone();
    let mut executor = fixture_executor(f);
    assert_eq!(resets.load(Ordering::SeqCst), 1);
    assert_eq!(executor.submit(&[]).unwrap().measurements[&40], 1);
    executor.finish_shot().unwrap();
    executor.start_shot(context(4), 8, 123).unwrap();
    assert_eq!(resets.load(Ordering::SeqCst), 1);
    assert_eq!(executor.submit(&[]).unwrap().measurements[&41], 0);
    executor.finish_shot().unwrap();
    executor.reset().unwrap();
    assert_eq!(resets.load(Ordering::SeqCst), 2);
}

// Ramsey experiment: only the idle timestamp changes, not the ideal gates.
fn ramsey_schedule(gap: u64) -> Vec<ScheduledBatch> {
    let mut first = batch(
        0,
        vec![RuntimeScheduledOp::Rxy {
            qubit_id: 0,
            theta: std::f64::consts::FRAC_PI_2,
            phi: std::f64::consts::FRAC_PI_2,
        }],
    );
    first.duration_nanos = 100;
    let mut last = batch(
        1,
        vec![RuntimeScheduledOp::Rxy {
            qubit_id: 0,
            theta: -std::f64::consts::FRAC_PI_2,
            phi: std::f64::consts::FRAC_PI_2,
        }],
    );
    last.start_time_nanos = 100 + gap;
    measurement(&mut last, 40);
    vec![first, last]
}
fn run_idle_schedule_state(
    batches: Vec<ScheduledBatch>,
    profile: IdleZNoise,
    split: bool,
    seed: u64,
) -> (u32, u64) {
    let chunks = if split {
        batches.into_iter().map(|b| vec![b]).collect::<Vec<_>>()
    } else {
        vec![batches]
    };
    let count = chunks.len();
    let fixture = Fixture {
        batches: chunks.into(),
        ..Default::default()
    };
    let mut executor = ScheduledExecutor::with_idle_z(Box::new(fixture), 2, profile).unwrap();
    executor.start_shot(context(0), 7, seed).unwrap();
    let mut outcomes = BTreeMap::new();
    for _ in 0..count {
        outcomes.extend(executor.submit(&[]).unwrap().measurements);
    }
    outcomes.extend(executor.finish_shot().unwrap().0.measurements);
    let model = executor
        .quantum
        .noise_model()
        .as_any()
        .downcast_ref::<pecos_engines::noise::GeneralNoiseModel>()
        .unwrap();
    let mut rng = pecos_core::RngManageable::rng(model).clone();
    (outcomes[&40], rng.next_u64())
}
fn run_idle_schedule(
    batches: Vec<ScheduledBatch>,
    profile: IdleZNoise,
    split: bool,
    seed: u64,
) -> u32 {
    run_idle_schedule_state(batches, profile, split, seed).0
}

#[test]
fn coherent_idle_uses_native_nanoseconds_and_changes_ramsey_outcome() {
    let profile = IdleZNoise {
        coherent: std::f64::consts::PI,
        ..Default::default()
    };
    assert_eq!(run_idle_schedule(ramsey_schedule(0), profile, false, 2), 0);
    assert_eq!(
        run_idle_schedule(ramsey_schedule(1_000_000_000), profile, false, 2),
        1
    );
    // Two operations on one target within the final batch share ONE idle site.
    let mut executor =
        ScheduledExecutor::with_idle_z(Box::new(Fixture::default()), 2, profile).unwrap();
    executor.start_shot(context(0), 7, 2).unwrap();
    let message = executor
        .admit(&ramsey_schedule(1_000_000_000))
        .unwrap()
        .commands;
    let gates = message.quantum_ops().unwrap();
    let idles = gates
        .iter()
        .filter(|g| g.gate_type == pecos_engines::GateType::Idle)
        .collect::<Vec<_>>();
    assert_eq!(idles.len(), 1);
    assert_eq!(idles[0].idle_duration().to_bits(), 1.0_f64.to_bits());
}

#[test]
fn empty_batches_and_submissions_do_not_split_nonlinear_idle_intervals() {
    let profile = IdleZNoise {
        sine: std::f64::consts::PI,
        ..Default::default()
    };
    let mut batches = ramsey_schedule(1_000_000_000);
    batches[1].batch_index = 2;
    let mut empty = batch(1, vec![]);
    empty.start_time_nanos = 500_000_100;
    empty.duration_nanos = 0;
    batches.insert(1, empty);
    // sin²(pi*1)=0; splitting at the empty batch would introduce two
    // sin²(pi/2)=1 faults and consume different random draws. Inspect sites too.
    for split in [false, true] {
        assert_eq!(run_idle_schedule(batches.clone(), profile, split, 2), 0);
    }
    let mut executor =
        ScheduledExecutor::with_idle_z(Box::new(Fixture::default()), 2, profile).unwrap();
    executor.start_shot(context(0), 7, 2).unwrap();
    let gates = executor
        .admit(&batches)
        .unwrap()
        .commands
        .quantum_ops()
        .unwrap();
    let durations = gates
        .iter()
        .filter(|g| g.gate_type == pecos_engines::GateType::Idle)
        .map(pecos_engines::Gate::idle_duration)
        .collect::<Vec<_>>();
    assert_eq!(durations, vec![1.0]);
}

#[test]
fn stochastic_idle_is_seeded_and_invariant_to_extraction_grouping() {
    let profile = IdleZNoise {
        linear: 0.12,
        sine: 0.71,
        coherent: 0.1,
    };
    let mut values = Vec::new();
    for seed in 0..16 {
        let together =
            run_idle_schedule_state(ramsey_schedule(1_000_000_000), profile, false, seed);
        let split = run_idle_schedule_state(ramsey_schedule(1_000_000_000), profile, true, seed);
        assert_eq!(together, split, "seed {seed}");
        values.push(together.0);
    }
    assert!(
        values.contains(&0) && values.contains(&1),
        "probe must exercise stochastic outcomes"
    );
}

#[test]
fn idle_profile_rejects_invalid_rates_and_overflow_before_quantum_mutation() {
    for rate in [f64::NAN, f64::INFINITY, -0.1] {
        for profile in [
            IdleZNoise {
                linear: rate,
                ..Default::default()
            },
            IdleZNoise {
                sine: rate,
                ..Default::default()
            },
            IdleZNoise {
                coherent: rate,
                ..Default::default()
            },
        ] {
            assert!(
                ScheduledExecutor::with_idle_z(Box::new(Fixture::default()), 2, profile).is_err()
            );
        }
    }
    let profile = IdleZNoise {
        coherent: f64::MAX,
        ..Default::default()
    };
    let mut final_batch = batch(1, vec![pulse()]);
    final_batch.start_time_nanos = 3_000_000_000;
    let fixture = Fixture {
        batches: VecDeque::from([vec![batch(0, vec![pulse()]), final_batch]]),
        ..Default::default()
    };
    let mut executor = ScheduledExecutor::with_idle_z(Box::new(fixture), 2, profile).unwrap();
    executor.start_shot(context(0), 7, 2).unwrap();
    assert!(
        executor
            .submit(&[])
            .unwrap_err()
            .to_string()
            .contains("duration product")
    );
    assert!(executor.poisoned);
    let result = executor
        .quantum
        .process(ByteMessage::quantum_operations_builder().mz(&[0]).build())
        .unwrap();
    assert_eq!(result.outcomes().unwrap(), vec![0]);
}

#[test]
fn idle_profile_rejects_opaque_events_before_earlier_gates_execute() {
    let fixture = Fixture {
        batches: VecDeque::from([vec![
            batch(0, vec![pulse()]),
            batch(
                1,
                vec![RuntimeScheduledOp::Custom {
                    tag: 7301,
                    data: vec![],
                }],
            ),
        ]]),
        ..Default::default()
    };
    let mut executor = ScheduledExecutor::with_idle_z(
        Box::new(fixture),
        2,
        IdleZNoise {
            linear: 0.1,
            ..Default::default()
        },
    )
    .unwrap();
    executor.start_shot(context(0), 7, 2).unwrap();
    assert!(executor.submit(&[]).is_err());
    assert!(executor.finish_shot().is_err());
    let result = executor
        .quantum
        .process(ByteMessage::quantum_operations_builder().mz(&[0]).build())
        .unwrap();
    assert_eq!(result.outcomes().unwrap(), vec![0]);
    executor.reset().unwrap();
    executor.start_shot(context(1), 8, 2).unwrap();
    executor.finish_shot().unwrap();
}

#[derive(Clone, Debug)]
struct CountQuantum {
    inner: StateVecEngine,
    calls: Arc<std::sync::atomic::AtomicUsize>,
}
impl Engine for CountQuantum {
    type Input = ByteMessage;
    type Output = ByteMessage;
    fn process(&mut self, input: ByteMessage) -> Result<ByteMessage, PecosError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.inner.process(input)
    }
    fn reset(&mut self) -> Result<(), PecosError> {
        self.inner.reset()
    }
}
impl pecos_engines::quantum::QuantumEngine for CountQuantum {
    fn set_seed(&mut self, seed: u64) {
        pecos_engines::quantum::QuantumEngine::set_seed(&mut self.inner, seed);
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

#[test]
fn both_profiles_preserve_native_processing_boundaries() {
    for idle in [false, true] {
        let mut batches = ramsey_schedule(1_000_000_000);
        batches[1].batch_index = 2;
        let mut empty = batch(1, vec![]);
        empty.duration_nanos = 0;
        batches.insert(1, empty);
        let fixture = Fixture {
            batches: VecDeque::from([batches]),
            ..Default::default()
        };
        let mut executor = if idle {
            ScheduledExecutor::with_idle_z(
                Box::new(fixture),
                2,
                IdleZNoise {
                    coherent: 0.3,
                    ..Default::default()
                },
            )
            .unwrap()
        } else {
            ScheduledExecutor::new(Box::new(fixture), 2).unwrap()
        };
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        executor.quantum = QuantumSystem::new(
            dyn_clone::clone_box(executor.quantum.noise_model()),
            Box::new(CountQuantum {
                inner: StateVecEngine::new(2),
                calls: calls.clone(),
            }),
        );
        executor.start_shot(context(0), 7, 2).unwrap();
        executor.submit(&[]).unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }
}

#[cfg(feature = "selene")]
#[test]
fn public_runtimes_zero_timing_preserves_ideal_feedback_across_shots() {
    for runtime in [
        pecos_qis::selene_simple_runtime().unwrap(),
        pecos_qis::selene_soft_rz_runtime().unwrap(),
    ] {
        let mut executor = ScheduledExecutor::with_idle_z(
            Box::new(runtime),
            2,
            IdleZNoise {
                linear: 0.1,
                sine: 0.3,
                coherent: 0.2,
            },
        )
        .unwrap();
        let mut results = Vec::new();
        for shot in 0..2 {
            executor.start_shot(context(shot), shot as u64, 13).unwrap();
            let first = executor
                .submit(&[
                    Operation::AllocateQubit { id: 7 },
                    QuantumOp::RXY(std::f64::consts::FRAC_PI_2, 0.0, 7).into(),
                ])
                .unwrap();
            let second = executor
                .submit(&[QuantumOp::Measure(7, 40).into()])
                .unwrap();
            let (last, runtime_shot) = executor.finish_shot().unwrap();
            // At the pinned public runtime revision these batches carry no
            // physical timing. This test must not be counted as idle-noise evidence.
            let batches = first
                .batches
                .into_iter()
                .chain(second.batches)
                .chain(last.batches)
                .collect::<Vec<_>>();
            assert!(!batches.is_empty());
            assert!(
                batches
                    .iter()
                    .all(|b| b.start_time_nanos == 0 && b.duration_nanos == 0)
            );
            let encoded = executor.admit_at(&batches, 0, vec![0; 2]).unwrap().commands;
            assert!(
                encoded
                    .quantum_ops()
                    .unwrap()
                    .iter()
                    .all(|g| g.gate_type != pecos_engines::GateType::Idle)
            );
            let measured = first
                .measurements
                .into_iter()
                .chain(second.measurements)
                .chain(last.measurements)
                .collect::<BTreeMap<_, _>>();
            assert_eq!(runtime_shot.measurements[&40], measured[&40] != 0);
            results.push(measured[&40]);
        }
        assert_eq!(results[0], results[1]);
    }
}

#[test]
fn per_batch_execution_still_rejects_cross_batch_duplicate_measurement_ids() {
    for idle in [false, true] {
        for duplicate_program in [false, true] {
            let mut first = batch(0, vec![pulse()]);
            measurement(&mut first, 40);
            let mut second = batch(1, vec![]);
            measurement(&mut second, if duplicate_program { 40 } else { 41 });
            if duplicate_program {
                second.operations[0] = RuntimeScheduledOp::Measure {
                    qubit_id: 0,
                    result_id: 902,
                };
                second.measurements[0].runtime_result = 902;
            }
            let fixture = Fixture {
                batches: VecDeque::from([vec![first, second]]),
                ..Default::default()
            };
            let mut executor = if idle {
                ScheduledExecutor::with_idle_z(
                    Box::new(fixture),
                    2,
                    IdleZNoise {
                        coherent: 0.1,
                        ..Default::default()
                    },
                )
                .unwrap()
            } else {
                ScheduledExecutor::new(Box::new(fixture), 2).unwrap()
            };
            executor.start_shot(context(0), 7, 2).unwrap();
            assert!(
                executor
                    .submit(&[])
                    .unwrap_err()
                    .to_string()
                    .contains("duplicate scheduled measurement identity")
            );
            assert!(executor.poisoned);
            let result = executor
                .quantum
                .process(ByteMessage::quantum_operations_builder().mz(&[0]).build())
                .unwrap();
            assert_eq!(result.outcomes().unwrap(), vec![0]);
        }
    }
}
