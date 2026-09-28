use super::*;
use crate::runtime::{ClassicalState, Result as RuntimeResult, RuntimeError};
use crate::scheduled::ScheduledMeasurement;
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
fn fixture_executor(f: Fixture) -> NoiselessScheduledExecutor {
    let mut executor = NoiselessScheduledExecutor::new(Box::new(f), 4).unwrap();
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
        crate::selene_simple_runtime().unwrap(),
        crate::selene_soft_rz_runtime().unwrap(),
    ] {
        let mut executor = NoiselessScheduledExecutor::new(Box::new(runtime), 4).unwrap();
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
    let runtime = crate::selene_simple_runtime().unwrap();
    let mut executor = NoiselessScheduledExecutor::new(Box::new(runtime), 2).unwrap();
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
            NoiselessScheduledExecutor::new(Box::new(crate::selene_simple_runtime().unwrap()), 2)
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
        NoiselessScheduledExecutor::new(Box::new(crate::selene_simple_runtime().unwrap()), 2)
            .unwrap();
    let mut b =
        NoiselessScheduledExecutor::new(Box::new(crate::selene_simple_runtime().unwrap()), 2)
            .unwrap();
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
