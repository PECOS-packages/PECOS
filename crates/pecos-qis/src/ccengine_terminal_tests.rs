//! Completion through a real Helios worker, native runtime and quantum engine.
use super::*;
use pecos_core::{QubitId, gate_type::GateType};
use pecos_engines::{Gate, StateVecEngine};

const PULSE_TAIL: &str = "
    call void @__quantum__qis__x__body(i64 1)
    call void @__quantum__qis__rz__body(double 0.3, i64 1)
";

fn program(tail: &str, explicit_handles: bool, read_result: bool) -> String {
    let allocations = if explicit_handles {
        "%q0 = call i64 @__quantum__rt__qubit_allocate()\n%q1 = call i64 @__quantum__rt__qubit_allocate()"
    } else {
        ""
    };
    let read = if read_result {
        "%read = call i1 @___read_future_bool(i64 0)"
    } else {
        ""
    };
    format!(
        r#"
        @key = private constant [12 x i8] c"source_label"
        @label = private constant [4 x i8] c"tail"
        define i64 @qmain(i64 %shot) #0 {{
            {allocations}
            call void @__quantum__qis__x__body(i64 0)
            %m = call i32 @__quantum__qis__m__body(i64 0, i64 0)
            {read}
            {tail}
            ret i64 0
        }}
        declare i64 @__quantum__rt__qubit_allocate()
        declare void @__quantum__qis__x__body(i64)
        declare void @__quantum__qis__rz__body(double, i64)
        declare void @__quantum__qis__reset__body(i64)
        declare i32 @__quantum__qis__m__body(i64, i64)
        declare i1 @___read_future_bool(i64)
        declare void @pecos_qis_trace_metadata_direct(ptr, i64, ptr, i64)
        attributes #0 = {{ "EntryPoint" }}
        "#
    )
}

struct CompletedRun {
    batches: Vec<Vec<Gate>>,
    terminal_batches: Vec<Vec<Gate>>,
    trace: Vec<OperationTraceChunk>,
    probabilities: Vec<f64>,
}

fn run(soft: bool, source: &str, tracing: bool) -> CompletedRun {
    // Executor tests mutate process-wide runtime/cache paths, and Helios uses
    // the shared FFI library. Use their lock for the whole worker lifetime.
    let _env_lock = crate::test_env::ENV_MUTEX
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let runtime = if soft {
        crate::selene_soft_rz_runtime().unwrap()
    } else {
        crate::selene_simple_runtime().unwrap()
    };
    let mut engine = QisEngine::new(
        Box::new(crate::QisHeliosInterface::new()),
        Box::new(runtime),
    );
    engine.set_num_qubits_hint(2);
    let trace = OperationTraceStore::default();
    if tracing {
        engine.set_operation_trace_collector(Arc::clone(&trace));
    }
    engine
        .load_program(source.as_bytes(), ProgramFormat::LlvmIrText)
        .unwrap();
    let mut quantum = StateVecEngine::new(2);
    let mut stage = engine.start(()).unwrap();
    let mut batches = Vec::new();
    let mut terminal_batches = Vec::new();
    // Empty batches mean the program worker is still running; only real
    // batches count toward the bound, and waiting is bounded by time.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    while batches.len() < 32 && std::time::Instant::now() < deadline {
        match stage {
            EngineStage::NeedsProcessing(commands) if commands.is_empty().unwrap() => {
                std::thread::yield_now();
                let measurements = quantum.process(commands).unwrap();
                stage = engine.continue_processing(measurements).unwrap();
            }
            EngineStage::NeedsProcessing(commands) => {
                let gates = commands.quantum_ops().unwrap();
                if engine
                    .dynamic_state
                    .as_ref()
                    .unwrap()
                    .terminal_lowering_flushed
                {
                    terminal_batches.push(gates.clone());
                }
                batches.push(gates);
                let measurements = quantum.process(commands).unwrap();
                stage = engine.continue_processing(measurements).unwrap();
            }
            EngineStage::Complete(shot) => {
                assert_eq!(shot.data.get("measurement_0"), Some(&Data::U32(1)));
                // A completed shot cannot flush or finalize its plugin twice.
                assert!(matches!(
                    engine
                        .continue_processing(ByteMessage::outcomes_builder().build())
                        .unwrap(),
                    EngineStage::Complete(_)
                ));
                let probabilities = quantum
                    .simulator_mut()
                    .state()
                    .iter()
                    .map(num_complex::Complex64::norm_sqr)
                    .collect();
                return CompletedRun {
                    batches,
                    terminal_batches,
                    trace: trace.lock().unwrap().clone(),
                    probabilities,
                };
            }
        }
    }
    panic!("ControlEngine failed to reach Complete");
}

#[test]
fn soft_rz_control_engine_flushes_native_tail() {
    for explicit in [false, true] {
        for read_result in [false, true] {
            for tracing in [false, true] {
                let output = run(true, &program(PULSE_TAIL, explicit, read_result), tracing);
                assert_eq!(output.terminal_batches.len(), 1);
                let tail = &output.terminal_batches[0];
                // Qubit 1's lifetime starts in the tail, so its prep is queued
                // with the pulse. Measurement keeps qubit 0 on slot 0, static or
                // explicit, so qubit 1 always has slot 1.
                assert_eq!(
                    tail.iter().map(|gate| gate.gate_type).collect::<Vec<_>>(),
                    [GateType::PZ, GateType::RXY1Q]
                );
                assert!(
                    tail.iter()
                        .all(|gate| gate.qubits.as_slice() == [QubitId(1)])
                );
                assert!(output.probabilities[3] > 1.0 - 1e-10);
            }
        }
    }
}

#[test]
fn terminal_lowering_keeps_source_labels() {
    let tail = format!(
        "call void @pecos_qis_trace_metadata_direct(ptr @key, i64 12, ptr @label, i64 4)\n{PULSE_TAIL}"
    );
    let output = run(true, &program(&tail, true, true), true);
    assert_eq!(output.terminal_batches.len(), 1);
    let terminal: Vec<_> = output
        .trace
        .iter()
        .filter(|chunk| chunk.stage == "terminal_flush")
        .collect();
    assert_eq!(terminal.len(), 1);
    assert!(terminal[0].lowered_quantum_ops_complete);
    assert_eq!(terminal[0].lowered_quantum_ops.len(), 1);
    let gate = &terminal[0].lowered_quantum_ops[0];
    assert_eq!(gate.gate_type, "RXY1Q");
    assert_eq!(gate.qubits, [1]);
    assert_eq!(gate.metadata["source_label"], "tail");
    assert_eq!(output.trace.last().unwrap().stage, "trace_complete");
}

#[test]
fn terminal_measurement_adds_no_empty_batch() {
    for soft in [false, true] {
        for read_result in [false, true] {
            let output = run(soft, &program("", true, read_result), true);
            assert_eq!(output.terminal_batches, Vec::<Vec<Gate>>::new());
            assert!(
                !output
                    .trace
                    .iter()
                    .any(|chunk| chunk.stage == "terminal_flush")
            );
            assert_eq!(
                output
                    .batches
                    .iter()
                    .filter(|gates| !gates.is_empty())
                    .count(),
                1
            );
            assert_eq!(output.trace.last().unwrap().stage, "trace_complete");
        }
    }
}

#[test]
fn simple_runtime_has_no_deferred_terminal_pulse() {
    for read_result in [false, true] {
        let output = run(false, &program(PULSE_TAIL, true, read_result), true);
        assert_eq!(output.terminal_batches, Vec::<Vec<Gate>>::new());
        assert!(output.probabilities[3] > 1.0 - 1e-10);
        assert_eq!(
            output
                .batches
                .iter()
                .flatten()
                .filter(|gate| gate.gate_type == GateType::RXY1Q)
                .count(),
            2
        );
    }
}

#[test]
fn soft_rz_flushes_prep_before_an_absorbed_rz_tail() {
    let tail = "call void @__quantum__qis__reset__body(i64 1)\ncall void @__quantum__qis__rz__body(double 0.3, i64 1)";
    let output = run(true, &program(tail, true, false), true);
    assert_eq!(output.terminal_batches.len(), 1);
    assert_eq!(output.terminal_batches[0].len(), 1);
    assert_eq!(output.terminal_batches[0][0].gate_type, GateType::PZ);
    assert_eq!(
        output.terminal_batches[0][0].qubits.as_slice(),
        &[QubitId(1)]
    );
}

fn reset_fixture(mode: ScheduledTransport) -> QisEngine {
    let mut engine = QisEngine::new(
        Box::new(crate::QisHeliosInterface::new()),
        Box::new(crate::selene_simple_runtime().unwrap()),
    );
    engine.scheduled_transport = mode;
    engine.set_num_qubits_hint(2);
    engine
        .load_program(
            program(PULSE_TAIL, true, true).as_bytes(),
            ProgramFormat::LlvmIrText,
        )
        .unwrap();
    engine
}

fn assert_waiting(engine: &mut QisEngine, stage: &EngineStage<ByteMessage, Shot>) {
    let EngineStage::NeedsProcessing(commands) = stage else {
        panic!("worker must wait for measurement")
    };
    assert_ne!(commands.as_bytes(), [0_u8; 0]);
    assert_eq!(engine.measurement_mapping, [0]);
    assert_eq!(engine.wait_for_result_needed(0), Some(0));
    assert!(engine.interface.is_none());
    assert!(!engine.check_worker_complete());
}

fn finish_reset_shot(
    engine: &mut QisEngine,
    quantum: &mut impl Engine<Input = ByteMessage, Output = ByteMessage>,
    mut stage: EngineStage<ByteMessage, Shot>,
) {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        assert!(Instant::now() < deadline, "second shot did not finish");
        match stage {
            EngineStage::NeedsProcessing(commands) => {
                let reply = quantum.process(commands).unwrap();
                stage = engine.continue_processing(reply).unwrap();
            }
            EngineStage::Complete(shot) => {
                assert_eq!(shot.data.get("measurement_0"), Some(&Data::U32(1)));
                assert!(engine.interface.is_some());
                assert!(engine.dynamic_state.as_ref().unwrap().finalized);
                return;
            }
        }
    }
}

#[test]
fn reset_discards_real_worker_execution_failure_and_reuses_interface() {
    let _env_lock = crate::test_env::ENV_MUTEX.lock().unwrap();
    let mut engine = reset_fixture(ScheduledTransport::Off);
    let source = program(
        &format!(
            "br i1 %read, label %success, label %failure\n\
             failure:\n\
             call void @panic(i32 1042, ptr null)\n\
             ret i64 0\n\
             success:\n{PULSE_TAIL}"
        ),
        true,
        true,
    ) + "\ndeclare void @panic(i32, ptr)\n";
    engine
        .load_program(source.as_bytes(), ProgramFormat::LlvmIrText)
        .unwrap();
    let stage = engine.start(()).unwrap();
    assert_waiting(&mut engine, &stage);
    // Let the abandoned shot fail without consuming its worker result through
    // continue_processing/check_worker_complete. Reset must reclaim it itself.
    engine.set_dynamic_result(0, 0).unwrap();
    engine.signal_dynamic_result_ready().unwrap();
    let wait_started = Instant::now();
    assert_eq!(engine.wait_for_result_needed(2_000), None);
    assert!(wait_started.elapsed() < Duration::from_secs(2));
    let worker_id = engine
        .persistent_worker
        .as_ref()
        .unwrap()
        .handle
        .as_ref()
        .unwrap()
        .thread()
        .id();
    engine.reset_all().unwrap();
    assert!(engine.interface.is_some());
    assert!(engine.reset_failure.is_none());
    assert!(engine.dynamic_state.is_none());
    let stage = engine.start(()).unwrap();
    assert_eq!(
        engine
            .persistent_worker
            .as_ref()
            .unwrap()
            .handle
            .as_ref()
            .unwrap()
            .thread()
            .id(),
        worker_id
    );
    finish_reset_shot(&mut engine, &mut StateVecEngine::new(2), stage);
}

#[test]
fn midshot_reset_reuses_real_worker_through_every_entry_point() {
    type ResetFn = fn(&mut QisEngine) -> Result<(), PecosError>;
    let _env_lock = crate::test_env::ENV_MUTEX.lock().unwrap();
    let resets: [ResetFn; 3] = [Engine::reset, ClassicalEngine::reset, ControlEngine::reset];
    for reset in resets {
        let mut engine = reset_fixture(ScheduledTransport::Off);
        let trace = OperationTraceStore::default();
        engine.set_operation_trace_collector(Arc::clone(&trace));
        let stage = engine.start(()).unwrap();
        assert_waiting(&mut engine, &stage);
        let aborted_shot = engine.trace_shot_index;
        let worker_id = engine
            .persistent_worker
            .as_ref()
            .unwrap()
            .handle
            .as_ref()
            .unwrap()
            .thread()
            .id();
        reset(&mut engine).unwrap();
        assert!(engine.interface.is_some());
        assert!(engine.dynamic_state.is_none());
        assert!(
            !trace
                .lock()
                .unwrap()
                .iter()
                .any(|chunk| chunk.stage == "trace_complete")
        );
        reset(&mut engine).unwrap();
        let stage = engine.start(()).unwrap();
        assert_eq!(
            engine
                .persistent_worker
                .as_ref()
                .unwrap()
                .handle
                .as_ref()
                .unwrap()
                .thread()
                .id(),
            worker_id
        );
        finish_reset_shot(&mut engine, &mut StateVecEngine::new(2), stage);
        let trace = trace.lock().unwrap();
        let terminal: Vec<_> = trace
            .iter()
            .filter(|chunk| chunk.stage == "trace_complete")
            .collect();
        assert_eq!(terminal.len(), 1);
        assert_eq!(terminal[0].shot_index, aborted_shot + 1);
    }
}

#[test]
fn midshot_reset_targets_only_its_engine_on_shared_host_thread() {
    let _env_lock = crate::test_env::ENV_MUTEX.lock().unwrap();
    let mut first = reset_fixture(ScheduledTransport::Off);
    let first_stage = first.start(()).unwrap();
    assert_waiting(&mut first, &first_stage);
    let mut second = reset_fixture(ScheduledTransport::Off);
    let second_stage = second.start(()).unwrap();
    assert_waiting(&mut second, &second_stage);
    // Host TLS now belongs to second. Reset must still cancel first's worker.
    ControlEngine::reset(&mut first).unwrap();
    assert!(first.interface.is_some());
    assert_waiting(&mut second, &second_stage);
    finish_reset_shot(&mut second, &mut StateVecEngine::new(2), second_stage);
    let stage = first.start(()).unwrap();
    finish_reset_shot(&mut first, &mut StateVecEngine::new(2), stage);
}

#[test]
fn midshot_reset_from_thread_without_registered_context() {
    let _env_lock = crate::test_env::ENV_MUTEX.lock().unwrap();
    let mut engine = reset_fixture(ScheduledTransport::Off);
    let stage = engine.start(()).unwrap();
    assert_waiting(&mut engine, &stage);
    let mut engine = std::thread::spawn(move || {
        // New thread has never registered an FFI context.
        ControlEngine::reset(&mut engine).unwrap();
        assert!(engine.interface.is_some());
        engine
    })
    .join()
    .unwrap();
    let stage = engine.start(()).unwrap();
    finish_reset_shot(&mut engine, &mut StateVecEngine::new(2), stage);
}

#[test]
fn scheduled_midshot_reset_discards_aborted_shot_and_reuses_worker() {
    use pecos_engines::noise::IntoNoiseModel;
    use pecos_engines::quantum_system::QuantumSystem;
    use pecos_engines::runtime_frame::ShotContext;
    use pecos_engines::scheduled_events::{
        ScheduledBatchAdapter, ScheduledEventBatch, ScheduledEventIdleZ, ScheduledEventOp,
        ScheduledGateBuffer,
    };
    use pecos_engines::scheduled_frame::ScheduledIdleZ;

    struct GatesOnly;
    impl ScheduledBatchAdapter for GatesOnly {
        fn validate(&self, batch: &ScheduledEventBatch) -> Result<(), PecosError> {
            if batch
                .operations
                .iter()
                .any(|op| matches!(op, ScheduledEventOp::Custom { .. }))
            {
                return Err(PecosError::Input("unexpected custom event".into()));
            }
            Ok(())
        }
        fn translate(
            &mut self,
            batch: &ScheduledEventBatch,
            out: &mut ScheduledGateBuffer<'_>,
        ) -> Result<(), PecosError> {
            for op in &batch.operations {
                match op {
                    ScheduledEventOp::Gate(gate) => out.push(gate.as_ref().clone())?,
                    ScheduledEventOp::Custom { .. } => {
                        return Err(PecosError::Input("unexpected custom event".into()));
                    }
                }
            }
            Ok(())
        }
    }

    let _env_lock = crate::test_env::ENV_MUTEX.lock().unwrap();
    for mode in [ScheduledTransport::V3, ScheduledTransport::V4] {
        let mut engine = reset_fixture(mode);
        let stage = engine.start(()).unwrap();
        assert_waiting(&mut engine, &stage);
        // Scheduled start deliberately rejects operation tracing. Install an
        // observer only for reset, so accidental certification cannot go unseen.
        let trace = OperationTraceStore::default();
        engine.set_operation_trace_collector(Arc::clone(&trace));
        ControlEngine::reset(&mut engine).unwrap();
        assert!(engine.interface.is_some());
        assert_eq!(trace.lock().unwrap().len(), 0);
        // Probe admission before start() performs its own reset: otherwise that
        // reset could mask an abandoned scheduled shot here.
        engine
            .runtime
            .shot_start(u64::try_from(engine.trace_shot_index + 1).unwrap(), None)
            .unwrap();
        engine.operation_trace_collector = None;
        let profile = ScheduledIdleZ::new(2, 0.0, 0.0, 0.0).unwrap();
        let noise = if mode == ScheduledTransport::V4 {
            ScheduledEventIdleZ::new(profile, |_| Ok(Box::new(GatesOnly))).into_noise_model()
        } else {
            profile.into_noise_model()
        };
        let mut quantum = QuantumSystem::new(noise, Box::new(StateVecEngine::new(2)));
        let stage = engine.start(()).unwrap();
        quantum
            .begin_shot(ShotContext {
                run: 1,
                worker: 0,
                shot: engine.trace_shot_index,
            })
            .unwrap();
        finish_reset_shot(&mut engine, &mut quantum, stage);
        assert_eq!(trace.lock().unwrap().len(), 0);
    }
}

#[test]
fn drop_cancels_real_worker_waiting_for_measurement() {
    let _env_lock = crate::test_env::ENV_MUTEX.lock().unwrap();
    let mut engine = reset_fixture(ScheduledTransport::Off);
    let counts = Arc::clone(&engine.worker_counts);
    let stage = engine.start(()).unwrap();
    assert_waiting(&mut engine, &stage);
    assert_eq!(counts.live.load(Ordering::SeqCst), 1);
    drop(engine);
    counts.assert_joined();
    // Cancellation returns the interface before the result receiver is dropped.
    assert_eq!(counts.returned.load(Ordering::SeqCst), 1);
}

#[test]
fn drop_cancels_real_worker_from_another_thread() {
    let _env_lock = crate::test_env::ENV_MUTEX.lock().unwrap();
    let mut engine = reset_fixture(ScheduledTransport::Off);
    let counts = Arc::clone(&engine.worker_counts);
    let stage = engine.start(()).unwrap();
    assert_waiting(&mut engine, &stage);
    std::thread::spawn(move || {
        // This thread has never registered a Helios execution context.
        drop(engine);
        counts.assert_joined();
        assert_eq!(counts.returned.load(Ordering::SeqCst), 1);
    })
    .join()
    .unwrap();
}

#[test]
fn drop_joins_idle_real_worker_after_completed_shot() {
    let _env_lock = crate::test_env::ENV_MUTEX.lock().unwrap();
    let mut engine = reset_fixture(ScheduledTransport::Off);
    let counts = Arc::clone(&engine.worker_counts);
    let stage = engine.start(()).unwrap();
    finish_reset_shot(&mut engine, &mut StateVecEngine::new(2), stage);
    assert_eq!(counts.live.load(Ordering::SeqCst), 1);
    drop(engine);
    counts.assert_joined();
}

#[test]
fn drop_joins_real_worker_with_queued_unconsumed_result() {
    let _env_lock = crate::test_env::ENV_MUTEX.lock().unwrap();
    let mut engine = reset_fixture(ScheduledTransport::Off);
    let counts = Arc::clone(&engine.worker_counts);
    let stage = engine.start(()).unwrap();
    assert_waiting(&mut engine, &stage);
    engine.set_dynamic_result(0, 1).unwrap();
    engine.signal_dynamic_result_ready().unwrap();
    let deadline = Instant::now() + Duration::from_secs(2);
    while counts.returned.load(Ordering::SeqCst) == 0 {
        assert!(Instant::now() < deadline, "worker did not queue its result");
        std::thread::yield_now();
    }
    assert!(engine.interface.is_none());
    assert!(!engine.dynamic_state.as_ref().unwrap().execution_complete);
    // A queued, unconsumed result must not prevent the worker from being joined.
    drop(engine);
    counts.assert_joined();
}

fn assert_monte_carlo_workers_joined(erroring: bool) {
    use pecos_engines::monte_carlo::MonteCarloEngine;

    let _env_lock = crate::test_env::ENV_MUTEX.lock().unwrap();
    let mut engine = reset_fixture(ScheduledTransport::Off);
    let source = if erroring {
        program("call void @panic(i32 1042, ptr null)", true, true)
            + "\ndeclare void @panic(i32, ptr)\n"
    } else {
        program(PULSE_TAIL, true, true)
    };
    engine
        .load_program(source.as_bytes(), ProgramFormat::LlvmIrText)
        .unwrap();
    engine.set_dynamic_config(Box::new(crate::helios_interface_builder()), &source);
    let counts = Arc::clone(&engine.worker_counts);
    let mut monte_carlo = MonteCarloEngine::builder()
        .with_classical_engine(Box::new(engine))
        .with_quantum_engine(Box::new(StateVecEngine::new(2)))
        .build();
    for run in 1..=3 {
        let previous = counts.started.load(Ordering::SeqCst);
        let result = monte_carlo.run_with_workers(8, 2);
        if erroring {
            assert!(result.is_err());
            // Rayon may stop admitting work after the first error.
            assert!(counts.started.load(Ordering::SeqCst) > previous);
        } else {
            assert_eq!(result.unwrap().len(), 8);
            assert_eq!(counts.started.load(Ordering::SeqCst), run * 2);
        }
        counts.assert_joined();
    }
    drop(monte_carlo);
    counts.assert_joined();
}

#[test]
fn drop_joins_monte_carlo_workers_after_repeated_runs() {
    assert_monte_carlo_workers_joined(false);
}

#[test]
fn drop_joins_monte_carlo_workers_after_erroring_runs() {
    assert_monte_carlo_workers_joined(true);
}

#[derive(Clone, Debug)]
struct FailOnMeasurement;

impl Engine for FailOnMeasurement {
    type Input = ByteMessage;
    type Output = ByteMessage;

    fn process(&mut self, commands: ByteMessage) -> Result<ByteMessage, PecosError> {
        let gates = commands.quantum_ops()?;
        if gates.is_empty() {
            return Ok(ByteMessage::outcomes_builder().build());
        }
        assert!(gates.iter().any(|gate| gate.gate_type == GateType::MZ));
        Err(PecosError::Processing(
            "injected measurement failure".into(),
        ))
    }
    fn reset(&mut self) -> Result<(), PecosError> {
        Ok(())
    }
}

impl pecos_engines::quantum::QuantumEngine for FailOnMeasurement {
    fn set_seed(&mut self, _: u64) {}
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

#[test]
fn drop_cancels_monte_carlo_workers_after_quantum_failure() {
    use pecos_engines::monte_carlo::MonteCarloEngine;

    let _env_lock = crate::test_env::ENV_MUTEX.lock().unwrap();
    let mut engine = reset_fixture(ScheduledTransport::Off);
    let source = program(PULSE_TAIL, true, true);
    engine.set_dynamic_config(Box::new(crate::helios_interface_builder()), &source);
    let counts = Arc::clone(&engine.worker_counts);
    let mut monte_carlo = MonteCarloEngine::builder()
        .with_classical_engine(Box::new(engine))
        .with_quantum_engine(Box::new(FailOnMeasurement))
        .build();
    // The quantum side fails before supplying the measurement, leaving the
    // program blocked on its read when each MonteCarlo clone is dropped.
    let started = Instant::now();
    let error = monte_carlo.run_with_workers(8, 2).unwrap_err();
    let elapsed = started.elapsed();
    assert!(error.to_string().contains("injected measurement failure"));
    let live = counts.live.load(Ordering::SeqCst);
    assert!(
        elapsed < Duration::from_secs(5) && live == 0,
        "run took {elapsed:?} and left {live} live workers"
    );
    counts.assert_joined();
}

const DYNAMIC_READ_PROBE: &str = r#"declare void @__quantum__qis__x__body(i64)
declare i32 @__quantum__qis__m__body(i64, i64)
declare i32 @__quantum__rt__result_get_one(i64)
define void @main() #0 {
entry:
  call void @__quantum__qis__x__body(i64 0)
  %m0 = call i32 @__quantum__qis__m__body(i64 0, i64 0)
  %one = call i32 @__quantum__rt__result_get_one(i64 0)
  %is1 = icmp ne i32 %one, 0
  br i1 %is1, label %flip, label %done
flip:
  call void @__quantum__qis__x__body(i64 1)
  br label %done
done:
  %m1 = call i32 @__quantum__qis__m__body(i64 1, i64 1)
  ret void
}
attributes #0 = { "EntryPoint" }
"#;

fn dynamic_read_engine(source: &str) -> QisEngine {
    let mut engine = QisEngine::new(
        Box::new(crate::QisHeliosInterface::new()),
        Box::new(crate::selene_simple_runtime().unwrap()),
    );
    engine.set_num_qubits_hint(2);
    engine
        .load_program(source.as_bytes(), ProgramFormat::LlvmIrText)
        .unwrap();
    engine
}

fn drive_dynamic_read_shot(engine: &mut QisEngine) -> Result<Shot, PecosError> {
    let mut quantum = StateVecEngine::new(2);
    let mut stage = engine.start(())?;
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        assert!(
            Instant::now() < deadline,
            "dynamic read failed to end its shot"
        );
        match stage {
            EngineStage::NeedsProcessing(commands) => {
                for result_id in &engine.measurement_mapping {
                    assert!(
                        !engine.measurement_results.contains_key(result_id),
                        "new measurement retained a cached result for {result_id}"
                    );
                }
                stage = engine.continue_processing(quantum.process(commands)?)?;
            }
            EngineStage::Complete(shot) => return Ok(shot),
        }
    }
}

#[test]
fn result_get_one_feedback_uses_the_host_measurement() {
    let _env_lock = crate::test_env::ENV_MUTEX.lock().unwrap();
    let mut engine = dynamic_read_engine(DYNAMIC_READ_PROBE);
    for _ in 0..3 {
        let shot = drive_dynamic_read_shot(&mut engine).unwrap();
        assert_eq!(shot.data.get("measurement_1"), Some(&Data::U32(1)));
    }
}

#[test]
fn same_result_slot_feedback_reads_the_new_measurement() {
    let _env_lock = crate::test_env::ENV_MUTEX.lock().unwrap();
    for (ty, reader, first, second) in [
        ("i1", "___read_future_bool", "%first", "%second"),
        (
            "i32",
            "__quantum__rt__result_get_one",
            "%first_bool",
            "%second_bool",
        ),
    ] {
        let conversions = if ty == "i32" {
            "%first_bool = icmp ne i32 %first, 0\n%second_bool = icmp ne i32 %second, 0"
        } else {
            ""
        };
        let source = format!(
            r#"
            declare void @__quantum__qis__x__body(i64)
            declare i32 @__quantum__qis__m__body(i64, i64)
            declare {ty} @{reader}(i64)
            declare void @panic(i32, ptr)
            define void @main() #0 {{
                call void @__quantum__qis__x__body(i64 0)
                %m0 = call i32 @__quantum__qis__m__body(i64 0, i64 0)
                %first = call {ty} @{reader}(i64 0)
                call void @__quantum__qis__x__body(i64 0)
                %m1 = call i32 @__quantum__qis__m__body(i64 0, i64 0)
                %second = call {ty} @{reader}(i64 0)
                {conversions}
                %second_zero = xor i1 {second}, true
                %correct = and i1 {first}, %second_zero
                br i1 %correct, label %done, label %failure
            failure:
                call void @panic(i32 1058, ptr null)
                ret void
            done:
                ret void
            }}
            attributes #0 = {{ "EntryPoint" }}
        "#
        );
        let mut engine = dynamic_read_engine(&source);
        let shot = drive_dynamic_read_shot(&mut engine).unwrap();
        assert_eq!(shot.data.get("measurement_0"), Some(&Data::U32(0)));
    }
}

#[test]
fn never_measured_failure_surfaces_from_start_and_continue() {
    let _env_lock = crate::test_env::ENV_MUTEX.lock().unwrap();
    for (ty, reader) in [
        ("i1", "___read_future_bool"),
        ("i32", "__quantum__rt__result_get_one"),
    ] {
        for after_measurement in [false, true] {
            let prefix = if after_measurement {
                "%m = call i32 @__quantum__qis__m__body(i64 0, i64 0)\n%read = call i1 @___read_future_bool(i64 0)"
            } else {
                ""
            };
            // Duplicate declarations of the bool reader are avoided for LLVM.
            let extra_declaration = if ty == "i1" {
                ""
            } else {
                "declare i1 @___read_future_bool(i64)"
            };
            let source = format!(
                r#"
                declare i32 @__quantum__qis__m__body(i64, i64)
                declare {ty} @{reader}(i64)
                {extra_declaration}
                define void @main() #0 {{
                    {prefix}
                    %never = call {ty} @{reader}(i64 7)
                    ret void
                }}
                attributes #0 = {{ "EntryPoint" }}
            "#
            );
            let mut engine = dynamic_read_engine(&source);
            let started = Instant::now();
            let error = drive_dynamic_read_shot(&mut engine).unwrap_err();
            assert!(started.elapsed() < Duration::from_secs(5));
            assert!(
                error
                    .to_string()
                    .contains("QIS measurement result 7 unavailable: never measured"),
                "{error}"
            );
            assert!(
                engine
                    .dynamic_state
                    .as_ref()
                    .unwrap()
                    .terminal_error
                    .is_some()
            );
            assert!(!error.to_string().contains("cancelled by reset"));
        }
    }
}

#[derive(Clone, Copy)]
enum DrainPadding {
    Idle,
    Custom,
    Empty,
}

type MeasurementDeliveries = std::sync::Arc<std::sync::Mutex<Vec<(usize, bool)>>>;

/// Scheduler fixture that accepts source operations and defers their release
/// to a barrier/drain, optionally releasing measurement 0 early or padding drains.
#[derive(Clone, Default)]
struct DeferredReadRuntime {
    state: crate::runtime::ClassicalState,
    pending: Vec<QuantumOp>,
    discard: Option<usize>,
    batch_index: usize,
    drain_padding: Option<DrainPadding>,
    release_measurement_zero: bool,
    feedback_dependency: Option<usize>,
    generated_measurement: Option<usize>,
    drop_first_measurement: bool,
    deliveries: Option<MeasurementDeliveries>,
}
impl DeferredReadRuntime {
    fn accept(&mut self, operations: &[Operation]) {
        for op in operations {
            if let Operation::Quantum(op) = op {
                if self.drop_first_measurement
                    && matches!(op, QuantumOp::Measure(..) | QuantumOp::MeasureLeaked(..))
                {
                    self.drop_first_measurement = false;
                    continue;
                }
                if matches!(op, QuantumOp::Measure(_, id) if Some(*id) == self.discard) {
                    continue;
                }
                self.pending.push(op.clone());
            }
        }
    }
    fn ready_operations(&mut self) -> Vec<QuantumOp> {
        if self.release_measurement_zero
            && let Some(position) = self.pending.iter().position(|op| {
                matches!(
                    op,
                    QuantumOp::Measure(_, 0) | QuantumOp::MeasureLeaked(_, 0)
                )
            })
        {
            self.pending.drain(..=position).collect()
        } else {
            vec![]
        }
    }

    fn released_operations(&mut self) -> Vec<QuantumOp> {
        if let Some(result_id) = self.generated_measurement {
            return vec![QuantumOp::Measure(0, result_id)];
        }
        if let Some(dependency) = self.feedback_dependency
            && !self.state.measurements.contains_key(&dependency)
        {
            if let Some(position) = self
                .pending
                .iter()
                .position(|op| matches!(op, QuantumOp::Measure(_, result) if *result == dependency))
            {
                self.pending.drain(..=position).collect()
            } else {
                vec![]
            }
        } else {
            std::mem::take(&mut self.pending)
        }
    }

    fn scheduled_drain(&mut self) -> Vec<crate::scheduled::ScheduledBatch> {
        use crate::scheduled::{RuntimeScheduledOp as Op, ScheduledBatch, ScheduledMeasurement};
        let released = self.released_operations();
        if released.is_empty() && self.drain_padding.is_none() {
            return vec![];
        }
        let mut operations = Vec::new();
        let mut measurements = Vec::new();
        for op in released {
            let scheduled = match op {
                QuantumOp::X(q) => Op::Rxy {
                    qubit_id: q as u64,
                    theta: std::f64::consts::PI,
                    phi: 0.0,
                },
                QuantumOp::Reset(q) => Op::Reset { qubit_id: q as u64 },
                QuantumOp::Measure(q, r) | QuantumOp::MeasureLeaked(q, r) => {
                    let leaked = matches!(op, QuantumOp::MeasureLeaked(..));
                    measurements.push(ScheduledMeasurement {
                        operation_index: operations.len(),
                        runtime_result: r as u64,
                        program_result: r,
                        leakage_aware: leaked,
                    });
                    if leaked {
                        Op::MeasureLeaked {
                            qubit_id: q as u64,
                            result_id: r as u64,
                        }
                    } else {
                        Op::Measure {
                            qubit_id: q as u64,
                            result_id: r as u64,
                        }
                    }
                }
                op => panic!("unexpected test operation {op:?}"),
            };
            operations.push(scheduled);
        }
        match self.drain_padding {
            Some(DrainPadding::Idle) => operations.push(Op::Rz {
                qubit_id: 0,
                theta: 0.0,
            }),
            Some(DrainPadding::Custom) => operations.push(Op::Custom {
                tag: 1058,
                data: vec![],
            }),
            Some(DrainPadding::Empty) | None => {}
        }
        let batch_index = self.batch_index;
        self.batch_index += 1;
        vec![ScheduledBatch {
            runtime_shot_id: self.state.shot_id.unwrap(),
            batch_index,
            start_time_nanos: 0,
            duration_nanos: u64::from(self.drain_padding.is_some()),
            operations,
            measurements,
        }]
    }
}
impl crate::runtime::QisRuntime for DeferredReadRuntime {
    fn load_interface(&mut self, _: OperationList) -> crate::runtime::Result<()> {
        Ok(())
    }
    fn reset(&mut self) -> crate::runtime::Result<()> {
        self.state = crate::runtime::ClassicalState::default();
        self.pending.clear();
        self.batch_index = 0;
        Ok(())
    }
    fn execute_until_quantum(&mut self) -> crate::runtime::Result<Option<Vec<QuantumOp>>> {
        Ok(None)
    }
    fn provide_measurements(
        &mut self,
        values: BTreeMap<usize, bool>,
    ) -> crate::runtime::Result<()> {
        if let Some(deliveries) = &self.deliveries {
            deliveries
                .lock()
                .unwrap()
                .extend(values.iter().map(|(&id, &value)| (id, value)));
        }
        self.state.measurements.extend(values);
        Ok(())
    }
    fn get_classical_state(&self) -> &crate::runtime::ClassicalState {
        &self.state
    }
    fn get_classical_state_mut(&mut self) -> &mut crate::runtime::ClassicalState {
        &mut self.state
    }
    fn is_complete(&self) -> bool {
        true
    }
    fn num_qubits(&self) -> usize {
        2
    }
    fn supports_operation_lowering(&self) -> bool {
        true
    }
    fn lower_operations(
        &mut self,
        operations: &[Operation],
    ) -> crate::runtime::Result<Vec<QuantumOp>> {
        self.accept(operations);
        if operations.contains(&Operation::Barrier) {
            let mut emitted = self.released_operations();
            if let Some(DrainPadding::Idle) = self.drain_padding {
                emitted.push(QuantumOp::Idle(1e-9, 0));
            }
            Ok(emitted)
        } else {
            Ok(self.ready_operations())
        }
    }
    fn drain_pending_operations(&mut self) -> crate::runtime::Result<Vec<QuantumOp>> {
        Ok(std::mem::take(&mut self.pending))
    }
    fn lower_scheduled_operations(
        &mut self,
        operations: &[Operation],
    ) -> crate::runtime::Result<Vec<crate::scheduled::ScheduledBatch>> {
        self.accept(operations);
        let ready = self.ready_operations();
        if ready.is_empty() {
            return Ok(vec![]);
        }
        let held = std::mem::replace(&mut self.pending, ready);
        let batches = self.scheduled_drain();
        self.pending = held;
        Ok(batches)
    }
    fn drain_pending_scheduled_operations(
        &mut self,
    ) -> crate::runtime::Result<Vec<crate::scheduled::ScheduledBatch>> {
        Ok(self.scheduled_drain())
    }
}

fn assert_unsatisfiable_read_fails(
    mode: ScheduledTransport,
    runtime: DeferredReadRuntime,
    after_first_read: bool,
) {
    let prefix = if after_first_read {
        "%m0 = call i32 @__quantum__qis__m__body(i64 0, i64 0)\n%r0 = call i1 @___read_future_bool(i64 0)"
    } else {
        ""
    };
    let source = format!(
        r#"
        declare i32 @__quantum__qis__m__body(i64, i64)
        declare i1 @___read_future_bool(i64)
        define void @main() #0 {{
            {prefix}
            %m = call i32 @__quantum__qis__m__body(i64 0, i64 7)
            %r = call i1 @___read_future_bool(i64 7)
            ret void
        }}
        attributes #0 = {{ "EntryPoint" }}
    "#
    );
    let mut engine = QisEngine::new(
        Box::new(crate::QisHeliosInterface::new()),
        Box::new(runtime),
    );
    engine.scheduled_transport = mode;
    engine.set_num_qubits_hint(2);
    engine
        .load_program(source.as_bytes(), ProgramFormat::LlvmIrText)
        .unwrap();
    // Discarded measurement 7 may still emit a lifetime prep; simulate
    // that work, then require the next poll to fail instead of spinning.
    let mut stage = engine.start(());
    let mut polls = 0;
    let started = Instant::now();
    while let Ok(EngineStage::NeedsProcessing(_)) = &stage {
        polls += 1;
        if polls > 3 {
            break;
        }
        assert!(!engine.measurement_mapping.contains(&7));
        let outcomes = ByteMessage::outcomes_builder()
            .add_outcomes(&vec![0; engine.measurement_mapping.len()])
            .build();
        // Scheduled commands are intentionally not decoded by a flat engine.
        stage = engine.continue_processing(outcomes);
    }
    let retry = engine.continue_processing(ByteMessage::outcomes_builder().build());
    engine.reset_all().unwrap();
    assert!(started.elapsed() < Duration::from_secs(5));
    let error = stage
        .err()
        .expect("an unsatisfiable measured read must fail promptly");
    assert!(
        error
            .to_string()
            .contains("runtime released no measurement for worker-requested result 7"),
        "{error}"
    );
    assert_eq!(
        retry.err().expect("failure must stay latched").to_string(),
        error.to_string()
    );
}

#[test]
fn unsatisfiable_measured_request_fails_from_start_and_continue() {
    let _env_lock = crate::test_env::ENV_MUTEX.lock().unwrap();
    for mode in [
        ScheduledTransport::Off,
        ScheduledTransport::V3,
        ScheduledTransport::V4,
    ] {
        for after_first_read in [false, true] {
            assert_unsatisfiable_read_fails(
                mode,
                DeferredReadRuntime {
                    discard: Some(7),
                    ..DeferredReadRuntime::default()
                },
                after_first_read,
            );
        }
    }
}

#[test]
fn non_measurement_drains_cannot_keep_an_unsatisfiable_read_alive() {
    let _env_lock = crate::test_env::ENV_MUTEX.lock().unwrap();
    for (mode, padding) in [
        (ScheduledTransport::Off, DrainPadding::Idle),
        (ScheduledTransport::V3, DrainPadding::Idle),
        (ScheduledTransport::V3, DrainPadding::Empty),
        (ScheduledTransport::V4, DrainPadding::Idle),
        (ScheduledTransport::V4, DrainPadding::Custom),
        (ScheduledTransport::V4, DrainPadding::Empty),
    ] {
        for after_first_read in [false, true] {
            assert_unsatisfiable_read_fails(
                mode,
                DeferredReadRuntime {
                    discard: Some(7),
                    drain_padding: Some(padding),
                    ..DeferredReadRuntime::default()
                },
                after_first_read,
            );
        }
    }
}

fn deferred_read_quantum(mode: ScheduledTransport) -> pecos_engines::quantum_system::QuantumSystem {
    use pecos_engines::noise::IntoNoiseModel;
    use pecos_engines::quantum_system::QuantumSystem;
    use pecos_engines::runtime_frame::ShotContext;
    use pecos_engines::scheduled_events::{
        ScheduledBatchAdapter, ScheduledEventBatch, ScheduledEventIdleZ, ScheduledEventOp,
        ScheduledGateBuffer,
    };
    use pecos_engines::scheduled_frame::ScheduledIdleZ;
    struct Gates;
    impl ScheduledBatchAdapter for Gates {
        fn validate(&self, _: &ScheduledEventBatch) -> Result<(), PecosError> {
            Ok(())
        }
        fn translate(
            &mut self,
            batch: &ScheduledEventBatch,
            output: &mut ScheduledGateBuffer<'_>,
        ) -> Result<(), PecosError> {
            for op in &batch.operations {
                let ScheduledEventOp::Gate(gate) = op else {
                    panic!("unexpected custom event")
                };
                output.push(gate.as_ref().clone())?;
            }
            Ok(())
        }
    }
    let noise = if mode == ScheduledTransport::V4 {
        ScheduledEventIdleZ::new(ScheduledIdleZ::new(2, 0.0, 0.0, 0.0).unwrap(), |_| {
            Ok(Box::new(Gates))
        })
        .into_noise_model()
    } else if mode == ScheduledTransport::V3 {
        ScheduledIdleZ::new(2, 0.0, 0.0, 0.0)
            .unwrap()
            .into_noise_model()
    } else {
        Box::new(pecos_engines::noise::PassThroughNoiseModel::default())
    };
    let mut quantum = QuantumSystem::new(noise, Box::new(StateVecEngine::new(2)));
    quantum
        .begin_shot(ShotContext {
            run: 0,
            worker: 0,
            shot: 0,
        })
        .unwrap();
    quantum
}

fn finish_deferred_read_shot(
    engine: &mut QisEngine,
    quantum: &mut pecos_engines::quantum_system::QuantumSystem,
    mut stage: EngineStage<ByteMessage, Shot>,
) -> Result<Shot, PecosError> {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if Instant::now() >= deadline {
            return Err(PecosError::Generic(
                "deferred measurement never completed".into(),
            ));
        }
        match stage {
            EngineStage::NeedsProcessing(commands) => {
                // An empty host poll can precede delivery of the worker's
                // completion message; it is independent of transport.
                let outcomes = if commands.as_bytes() == ByteMessage::builder().build().as_bytes() {
                    std::thread::yield_now();
                    ByteMessage::outcomes_builder().build()
                } else {
                    quantum.process(commands)?
                };
                stage = engine.continue_processing(outcomes)?;
            }
            EngineStage::Complete(shot) => return Ok(shot),
        }
    }
}

#[test]
fn deferred_measurement_reads_and_final_tail_share_the_runtime_drain() {
    let _env_lock = crate::test_env::ENV_MUTEX.lock().unwrap();
    let source = r#"
        declare void @__quantum__qis__x__body(i64)
        declare i32 @__quantum__qis__m__body(i64, i64)
        declare i1 @___read_future_bool(i64)
        declare void @panic(i32, ptr)
        define void @main() #0 {
            call void @__quantum__qis__x__body(i64 0)
            %m0 = call i32 @__quantum__qis__m__body(i64 0, i64 0)
            %r0 = call i1 @___read_future_bool(i64 0)
            call void @__quantum__qis__x__body(i64 0)
            %m1 = call i32 @__quantum__qis__m__body(i64 0, i64 1)
            %r1 = call i1 @___read_future_bool(i64 1)
            %r1_zero = xor i1 %r1, true
            %correct = and i1 %r0, %r1_zero
            br i1 %correct, label %done, label %failure
        failure:
            call void @panic(i32 1058, ptr null)
            ret void
        done:
            call void @__quantum__qis__x__body(i64 1)
            %m2 = call i32 @__quantum__qis__m__body(i64 1, i64 2)
            ret void
        }
        attributes #0 = { "EntryPoint" }
    "#;
    for mode in [
        ScheduledTransport::Off,
        ScheduledTransport::V3,
        ScheduledTransport::V4,
    ] {
        let mut engine = QisEngine::new(
            Box::new(crate::QisHeliosInterface::new()),
            Box::new(DeferredReadRuntime::default()),
        );
        engine.scheduled_transport = mode;
        engine.set_num_qubits_hint(2);
        engine
            .load_program(source.as_bytes(), ProgramFormat::LlvmIrText)
            .unwrap();
        let mut quantum = deferred_read_quantum(mode);
        let result = engine
            .start(())
            .and_then(|stage| finish_deferred_read_shot(&mut engine, &mut quantum, stage));
        engine.reset_all().unwrap();
        let shot = result.unwrap();
        assert_eq!(shot.data.get("measurement_0"), Some(&Data::U32(1)));
        assert_eq!(shot.data.get("measurement_1"), Some(&Data::U32(0)));
        // Final tail must still flush after the two mid-shot read flushes.
        assert_eq!(shot.data.get("measurement_2"), Some(&Data::U32(1)));
    }
}

#[test]
fn earlier_measurement_does_not_signal_ready_for_the_outstanding_read() {
    let _env_lock = crate::test_env::ENV_MUTEX.lock().unwrap();
    let source = r#"
        declare void @__quantum__qis__x__body(i64)
        declare i32 @__quantum__qis__m__body(i64, i64)
        declare i1 @___read_future_bool(i64)
        declare void @panic(i32, ptr)
        define void @main() #0 {
            call void @__quantum__qis__x__body(i64 0)
            %m0 = call i32 @__quantum__qis__m__body(i64 0, i64 0)
            call void @__quantum__qis__x__body(i64 1)
            %m1 = call i32 @__quantum__qis__m__body(i64 1, i64 1)
            %r1 = call i1 @___read_future_bool(i64 1)
            %r0 = call i1 @___read_future_bool(i64 0)
            %correct = and i1 %r0, %r1
            br i1 %correct, label %done, label %failure
        failure:
            call void @panic(i32 1058, ptr null)
            ret void
        done:
            ret void
        }
        attributes #0 = { "EntryPoint" }
    "#;
    for mode in [
        ScheduledTransport::Off,
        ScheduledTransport::V3,
        ScheduledTransport::V4,
    ] {
        let mut engine = QisEngine::new(
            Box::new(crate::QisHeliosInterface::new()),
            Box::new(DeferredReadRuntime {
                release_measurement_zero: true,
                ..DeferredReadRuntime::default()
            }),
        );
        engine.scheduled_transport = mode;
        engine.set_num_qubits_hint(2);
        engine
            .load_program(source.as_bytes(), ProgramFormat::LlvmIrText)
            .unwrap();
        let mut quantum = deferred_read_quantum(mode);
        let result = (|| -> Result<Shot, PecosError> {
            let EngineStage::NeedsProcessing(earlier) = engine.start(())? else {
                panic!("measurement 0 must be released before measurement 1");
            };
            assert_eq!(engine.measurement_mapping, [0]);
            assert_eq!(engine.wait_for_result_needed(0), Some(1));
            let reply = quantum.process(earlier)?;
            assert_eq!(reply.outcomes()?, [1]);
            let EngineStage::NeedsProcessing(requested) = engine.continue_processing(reply)? else {
                panic!("the outstanding read must drain its held measurement");
            };
            assert_eq!(engine.measurement_mapping, [1]);
            assert_eq!(engine.wait_for_result_needed(0), Some(1));
            let reply = quantum.process(requested)?;
            assert_eq!(reply.outcomes()?, [1]);
            let stage = engine.continue_processing(reply)?;
            finish_deferred_read_shot(&mut engine, &mut quantum, stage)
        })();
        engine.reset_all().unwrap();
        let shot = result.unwrap();
        assert_eq!(shot.data.get("measurement_0"), Some(&Data::U32(1)));
        assert_eq!(shot.data.get("measurement_1"), Some(&Data::U32(1)));
    }
}

#[test]
fn deferred_same_slot_delivery_cannot_satisfy_a_later_measurement_read() {
    assert_deferred_same_slot_delivery(true);
}

#[test]
fn deferred_same_slot_deliveries_complete_before_any_read() {
    assert_deferred_same_slot_delivery(false);
}

fn assert_deferred_same_slot_delivery(read: bool) {
    let _env_lock = crate::test_env::ENV_MUTEX.lock().unwrap();
    let tail = if read {
        r"
            %value = call i1 @___read_future_bool(i64 0)
            br i1 %value, label %done, label %failure
        failure:
            call void @panic(i32 1058, ptr null)
            ret void
        done:
        "
    } else {
        ""
    };
    let source = format!(
        r#"
        declare void @__quantum__qis__x__body(i64)
        declare i32 @__quantum__qis__m__body(i64, i64)
        declare i1 @___read_future_bool(i64)
        declare void @panic(i32, ptr)
        define void @main() #0 {{
            %first = call i32 @__quantum__qis__m__body(i64 0, i64 0)
            call void @__quantum__qis__x__body(i64 0)
            %second = call i32 @__quantum__qis__m__body(i64 0, i64 0)
            {tail}
            ret void
        }}
        attributes #0 = {{ "EntryPoint" }}
    "#
    );
    for mode in [
        ScheduledTransport::Off,
        ScheduledTransport::V3,
        ScheduledTransport::V4,
    ] {
        let deliveries = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut engine = QisEngine::new(
            Box::new(crate::QisHeliosInterface::new()),
            Box::new(DeferredReadRuntime {
                release_measurement_zero: true,
                deliveries: Some(std::sync::Arc::clone(&deliveries)),
                ..DeferredReadRuntime::default()
            }),
        );
        engine.scheduled_transport = mode;
        engine.set_num_qubits_hint(2);
        engine
            .load_program(source.as_bytes(), ProgramFormat::LlvmIrText)
            .unwrap();
        let mut quantum = deferred_read_quantum(mode);
        let result = (|| -> Result<Shot, PecosError> {
            if !read {
                // Completion can precede the worker's final channel send. Drive
                // empty host polls and terminal drains through the public API;
                // the simulator outcomes, feedback record and final Shot below
                // establish the no-read path's observable behaviour.
                let stage = engine.start(())?;
                return finish_deferred_read_shot(&mut engine, &mut quantum, stage);
            }
            let EngineStage::NeedsProcessing(first) = engine.start(())? else {
                panic!("the first measurement must be emitted on its own");
            };
            assert_eq!(engine.measurement_mapping, [0]);
            assert_eq!(engine.pending_measurements[&0].outstanding, 2);
            assert_eq!(engine.pending_measurements[&0].unemitted, 1);
            let first_reply = quantum.process(first)?;
            assert_eq!(first_reply.outcomes()?, [0]);
            let EngineStage::NeedsProcessing(second) = engine.continue_processing(first_reply)?
            else {
                panic!("the second measurement must still be delivered");
            };
            assert!(!engine.measurement_results.contains_key(&0));
            assert_eq!(engine.pending_measurements[&0].outstanding, 1);
            assert_eq!(engine.pending_measurements[&0].unemitted, 0);
            assert_eq!(engine.wait_for_result_needed(0), Some(0));
            assert_eq!(engine.wait_for_result_needed(0), Some(0));
            let second_reply = quantum.process(second)?;
            assert_eq!(second_reply.outcomes()?, [1]);
            let stage = engine.continue_processing(second_reply)?;
            assert!(!engine.pending_measurements.contains_key(&0));
            finish_deferred_read_shot(&mut engine, &mut quantum, stage)
        })();
        engine.reset_all().unwrap();
        if read {
            assert_eq!(engine.pending_measurements.len(), 0);
        }
        let shot = result.unwrap();
        assert_eq!(shot.data.get("measurement_0"), Some(&Data::U32(1)));
        // Earlier outcomes remain in the delivery record even though they cannot
        // be used as the current slot value or satisfy a dynamic read.
        assert_eq!(*deliveries.lock().unwrap(), [(0, false), (0, true)]);
    }
}

#[test]
fn forced_drain_measurement_feedback_releases_the_requested_measurement() {
    let _env_lock = crate::test_env::ENV_MUTEX.lock().unwrap();
    let source = r#"
        declare void @__quantum__qis__x__body(i64)
        declare i32 @__quantum__qis__m__body(i64, i64)
        declare i1 @___read_future_bool(i64)
        declare void @panic(i32, ptr)
        define void @main() #0 {
            call void @__quantum__qis__x__body(i64 0)
            %m0 = call i32 @__quantum__qis__m__body(i64 0, i64 0)
            call void @__quantum__qis__x__body(i64 1)
            %m1 = call i32 @__quantum__qis__m__body(i64 1, i64 1)
            %r1 = call i1 @___read_future_bool(i64 1)
            %r0 = call i1 @___read_future_bool(i64 0)
            %correct = and i1 %r0, %r1
            br i1 %correct, label %done, label %failure
        failure:
            call void @panic(i32 1058, ptr null)
            ret void
        done:
            ret void
        }
        attributes #0 = { "EntryPoint" }
    "#;
    for mode in [
        ScheduledTransport::Off,
        ScheduledTransport::V3,
        ScheduledTransport::V4,
    ] {
        let mut engine = QisEngine::new(
            Box::new(crate::QisHeliosInterface::new()),
            Box::new(DeferredReadRuntime {
                feedback_dependency: Some(0),
                ..DeferredReadRuntime::default()
            }),
        );
        engine.scheduled_transport = mode;
        engine.set_num_qubits_hint(2);
        engine
            .load_program(source.as_bytes(), ProgramFormat::LlvmIrText)
            .unwrap();
        let mut quantum = deferred_read_quantum(mode);
        let result = (|| -> Result<Shot, PecosError> {
            // Import emits nothing; the forced drain releases only dependency 0.
            let EngineStage::NeedsProcessing(dependency) = engine.start(())? else {
                panic!("the drain must return a measurement that can provide feedback");
            };
            assert_eq!(engine.measurement_mapping, [0]);
            assert_eq!(engine.wait_for_result_needed(0), Some(1));
            let reply = quantum.process(dependency)?;
            assert_eq!(reply.outcomes()?, [1]);
            let EngineStage::NeedsProcessing(requested) = engine.continue_processing(reply)? else {
                panic!("feedback must make the requested measurement available");
            };
            assert_eq!(engine.measurement_mapping, [1]);
            assert_eq!(engine.wait_for_result_needed(0), Some(1));
            let reply = quantum.process(requested)?;
            assert_eq!(reply.outcomes()?, [1]);
            let stage = engine.continue_processing(reply)?;
            finish_deferred_read_shot(&mut engine, &mut quantum, stage)
        })();
        engine.reset_all().unwrap();
        let shot = result.unwrap();
        assert_eq!(shot.data.get("measurement_0"), Some(&Data::U32(1)));
        assert_eq!(shot.data.get("measurement_1"), Some(&Data::U32(1)));
    }
}

#[test]
fn pending_measurement_counts_preserve_leakage_and_reset_each_shot() {
    for mode in [
        ScheduledTransport::Off,
        ScheduledTransport::V3,
        ScheduledTransport::V4,
    ] {
        let mut engine = QisEngine::with_runtime(Box::new(DeferredReadRuntime {
            release_measurement_zero: true,
            ..DeferredReadRuntime::default()
        }));
        engine.scheduled_transport = mode;
        engine.runtime.shot_start(0, None).unwrap();
        engine.dynamic_state = Some(DynamicExecutionState {
            sync_handle: None,
            execution_complete: true,
            terminal_error: None,
            finalized: false,
            terminal_lowering_flushed: false,
        });
        engine
            .lower_operations_terminal(&[
                QuantumOp::MeasureLeaked(0, 0).into(),
                QuantumOp::MeasureLeaked(0, 0).into(),
            ])
            .unwrap();
        assert_eq!(engine.measurement_mapping, [0]);
        let first = engine.map_measurements(&[0]).unwrap();
        assert_eq!(engine.store_measurement_updates(&first).unwrap(), []);
        assert!(!engine.measurement_results.contains_key(&0));
        assert!(engine.drain_commands(false).unwrap().is_some());
        assert_eq!(engine.measurement_mapping, [0]);
        let second = engine.map_measurements(&[2]).unwrap();
        assert_eq!(engine.store_measurement_updates(&second).unwrap(), [(0, 2)]);
        assert_eq!(engine.measurement_results.get(&0), Some(&2));
        assert_eq!(engine.pending_measurements.len(), 0);
        // A queued, undelivered slot must disappear on reset and on Clone's
        // fresh shot, even when it has not emitted a simulator command yet.
        engine
            .lower_operations_terminal(&[QuantumOp::MeasureLeaked(0, 8).into()])
            .unwrap();
        assert_eq!(engine.pending_measurements[&8].outstanding, 1);
        assert_eq!(engine.pending_measurements[&8].unemitted, 1);
        assert_eq!(engine.clone().pending_measurements.len(), 0);
        // Only the drain was synthetic; no worker owns an interface here.
        engine.dynamic_state = None;
        engine.reset_all().unwrap();
        assert_eq!(engine.pending_measurements.len(), 0);
    }
}

#[test]
fn unimported_measurement_emissions_latch_a_terminal_error() {
    for mode in [
        ScheduledTransport::Off,
        ScheduledTransport::V3,
        ScheduledTransport::V4,
    ] {
        let mut engine = QisEngine::with_runtime(Box::new(DeferredReadRuntime {
            discard: Some(7),
            generated_measurement: Some(9),
            ..DeferredReadRuntime::default()
        }));
        engine.scheduled_transport = mode;
        engine.runtime.shot_start(0, None).unwrap();
        engine.dynamic_state = Some(DynamicExecutionState {
            sync_handle: None,
            execution_complete: true,
            terminal_error: None,
            finalized: false,
            terminal_lowering_flushed: false,
        });
        engine
            .lower_operations_terminal(&[QuantumOp::Measure(0, 7).into()])
            .unwrap();
        let error = engine
            .drain_commands(false)
            .err()
            .expect("an unimported emission must fail");
        assert!(
            error
                .to_string()
                .contains("runtime emitted a measurement for result 9 that was never imported"),
            "{error}"
        );
        assert_eq!(
            engine
                .continue_processing(ByteMessage::outcomes_builder().build())
                .err()
                .unwrap()
                .to_string(),
            error.to_string()
        );
        assert!(!engine.measurement_results.contains_key(&9));
    }
}

#[test]
fn generated_measurement_drains_cannot_keep_an_unsatisfiable_read_alive() {
    let _env_lock = crate::test_env::ENV_MUTEX.lock().unwrap();
    let source = r#"
        declare i32 @__quantum__qis__m__body(i64, i64)
        declare i1 @___read_future_bool(i64)
        define void @main() #0 {
            %measurement = call i32 @__quantum__qis__m__body(i64 0, i64 7)
            %read = call i1 @___read_future_bool(i64 7)
            ret void
        }
        attributes #0 = { "EntryPoint" }
    "#;
    for mode in [
        ScheduledTransport::Off,
        ScheduledTransport::V3,
        ScheduledTransport::V4,
    ] {
        let mut engine = QisEngine::new(
            Box::new(crate::QisHeliosInterface::new()),
            Box::new(DeferredReadRuntime {
                discard: Some(7),
                generated_measurement: Some(9),
                ..DeferredReadRuntime::default()
            }),
        );
        engine.scheduled_transport = mode;
        engine.set_num_qubits_hint(2);
        engine
            .load_program(source.as_bytes(), ProgramFormat::LlvmIrText)
            .unwrap();
        let mut quantum = deferred_read_quantum(mode);
        let started = Instant::now();
        let mut stage = engine.start(());
        // A mutant may return generated measurements forever. Bound host rounds,
        // rather than waiting for a timeout in either side of the protocol.
        for _ in 0..3 {
            let Ok(EngineStage::NeedsProcessing(commands)) = stage else {
                break;
            };
            let reply = quantum.process(commands).unwrap();
            stage = engine.continue_processing(reply);
        }
        let retry = engine.continue_processing(ByteMessage::outcomes_builder().build());
        engine.reset_all().unwrap();
        assert!(started.elapsed() < Duration::from_secs(5));
        let error = stage
            .err()
            .expect("generated drains must fail rather than loop");
        assert!(
            error
                .to_string()
                .contains("runtime emitted a measurement for result 9 that was never imported"),
            "{error}"
        );
        assert_eq!(
            retry.err().expect("failure must stay latched").to_string(),
            error.to_string()
        );
    }
}

fn dropped_measurement_shot(mode: ScheduledTransport) -> (QisEngine, Result<Shot, PecosError>) {
    let source = r#"
        declare i32 @__quantum__qis__m__body(i64, i64)
        declare void @__quantum__qis__x__body(i64)
        define void @main() #0 {
            %first = call i32 @__quantum__qis__m__body(i64 0, i64 3)
            call void @__quantum__qis__x__body(i64 0)
            %second = call i32 @__quantum__qis__m__body(i64 0, i64 3)
            %lost = call i32 @__quantum__qis__m__body(i64 1, i64 7)
            ret void
        }
        attributes #0 = { "EntryPoint" }
    "#;
    let mut engine = QisEngine::new(
        Box::new(crate::QisHeliosInterface::new()),
        Box::new(DeferredReadRuntime {
            discard: Some(7),
            drop_first_measurement: true,
            ..DeferredReadRuntime::default()
        }),
    );
    engine.scheduled_transport = mode;
    engine.set_num_qubits_hint(2);
    engine
        .load_program(source.as_bytes(), ProgramFormat::LlvmIrText)
        .unwrap();
    let mut quantum = deferred_read_quantum(mode);
    let result = engine
        .start(())
        .and_then(|stage| finish_deferred_read_shot(&mut engine, &mut quantum, stage));
    (engine, result)
}

#[test]
fn dropped_imported_measurements_prevent_shot_certification() {
    let _env_lock = crate::test_env::ENV_MUTEX.lock().unwrap();
    for mode in [
        ScheduledTransport::Off,
        ScheduledTransport::V3,
        ScheduledTransport::V4,
    ] {
        let (mut engine, result) = dropped_measurement_shot(mode);
        let retry = engine.continue_processing(ByteMessage::outcomes_builder().build());
        let recorded = engine.get_results();
        let finalized = engine.dynamic_state.as_ref().unwrap().finalized;
        engine.reset_all().unwrap();
        let error = result.expect_err("a lost measurement must fail certification");
        assert!(
            error
                .to_string()
                .contains("runtime dropped imported measurements for result slots [3, 7]"),
            "{error}"
        );
        assert!(!finalized);
        assert_eq!(retry.err().unwrap().to_string(), error.to_string());
        assert_eq!(recorded.err().unwrap().to_string(), error.to_string());
    }
}

#[test]
fn start_discards_measurement_credits_from_the_previous_shot() {
    let _env_lock = crate::test_env::ENV_MUTEX.lock().unwrap();
    for mode in [
        ScheduledTransport::Off,
        ScheduledTransport::V3,
        ScheduledTransport::V4,
    ] {
        let (mut engine, previous) = dropped_measurement_shot(mode);
        assert!(previous.is_err());
        assert!(engine.interface.is_some());
        // Start the next shot directly, with no Engine::reset that could hide
        // failure to clear credits in start(). The previous worker is complete.
        engine
            .load_program(
                program("", false, true).as_bytes(),
                ProgramFormat::LlvmIrText,
            )
            .unwrap();
        let mut quantum = deferred_read_quantum(mode);
        let result = engine
            .start(())
            .and_then(|stage| finish_deferred_read_shot(&mut engine, &mut quantum, stage));
        engine.reset_all().unwrap();
        let shot = result.expect("the next start must discard the abandoned measurement credits");
        assert_eq!(shot.data.get("measurement_0"), Some(&Data::U32(1)));
    }
}

fn finish_interleaved_stage(
    engine: &mut QisEngine,
    quantum: &mut StateVecEngine,
    stage: EngineStage<ByteMessage, Shot>,
) -> Shot {
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut stage = stage;
    loop {
        assert!(Instant::now() < deadline);
        match stage {
            EngineStage::NeedsProcessing(commands) => {
                stage = engine
                    .continue_processing(quantum.process(commands).unwrap())
                    .unwrap();
            }
            EngineStage::Complete(shot) => return shot,
        }
    }
}

#[test]
fn dynamic_engines_interleave_on_one_host_thread() {
    let _env_lock = crate::test_env::ENV_MUTEX.lock().unwrap();
    let mut a = dynamic_read_engine(DYNAMIC_READ_PROBE);
    let mut b = dynamic_read_engine(
        &DYNAMIC_READ_PROBE.replace("call void @__quantum__qis__x__body(i64 0)", ""),
    );
    let mut qa = StateVecEngine::new(2);
    let mut qb = StateVecEngine::new(2);
    let sa = a.start(()).unwrap();
    let sb = b.start(()).unwrap();
    assert_waiting(&mut a, &sa);
    assert_waiting(&mut b, &sb);
    let EngineStage::NeedsProcessing(ca) = sa else {
        panic!("A must wait")
    };
    let EngineStage::NeedsProcessing(cb) = sb else {
        panic!("B must wait")
    };
    let sa = a.continue_processing(qa.process(ca).unwrap()).unwrap();
    let sb = b.continue_processing(qb.process(cb).unwrap()).unwrap();
    let ra = finish_interleaved_stage(&mut a, &mut qa, sa);
    let rb = finish_interleaved_stage(&mut b, &mut qb, sb);
    assert_eq!(ra.data.get("measurement_1"), Some(&Data::U32(1)));
    assert_eq!(rb.data.get("measurement_1"), Some(&Data::U32(0)));
}

#[test]
fn migrated_shot_replacement_leaves_original_host_tls_empty() {
    let _env_lock = crate::test_env::ENV_MUTEX.lock().unwrap();
    let (send, recv) = mpsc::channel();
    let (replaced, replacement) = mpsc::channel();
    let x = std::thread::spawn(move || {
        let mut engine = dynamic_read_engine(DYNAMIC_READ_PROBE);
        let path = engine
            .interface
            .as_ref()
            .unwrap()
            .get_qis_ffi_lib_path()
            .unwrap();
        // Use the same runtime instance as the engine, with its compatibility ABI.
        let lib = unsafe { libloading::Library::new(path).unwrap() };
        let pending: libloading::Symbol<unsafe extern "C" fn() -> *mut OperationList> =
            unsafe { lib.get(b"pecos_get_pending_operations\0").unwrap() };
        let stage = engine.start(()).unwrap();
        // Check while the context is still live so a registration regression
        // fails before allowing another thread to free a dangling TLS pointer.
        assert!(
            unsafe { pending() }.is_null(),
            "host X registered the shot context"
        );
        send.send((engine, stage)).unwrap();
        replacement.recv().unwrap();
        assert!(
            unsafe { pending() }.is_null(),
            "host X retained the freed context"
        );
    });
    let (mut engine, stage) = recv.recv().unwrap();
    let mut quantum = StateVecEngine::new(2);
    let shot = finish_interleaved_stage(&mut engine, &mut quantum, stage);
    assert_eq!(shot.data.get("measurement_1"), Some(&Data::U32(1)));
    let old = engine
        .interface
        .as_ref()
        .unwrap()
        .get_execution_context_ptr()
        .unwrap() as usize;
    let stage = engine.start(()).unwrap();
    // Enabling the next shot replaced the old interface context and sync handle.
    let shot = finish_interleaved_stage(&mut engine, &mut StateVecEngine::new(2), stage);
    assert_eq!(shot.data.get("measurement_1"), Some(&Data::U32(1)));
    assert_ne!(
        old,
        engine
            .interface
            .as_ref()
            .unwrap()
            .get_execution_context_ptr()
            .unwrap() as usize
    );
    replaced.send(()).unwrap();
    x.join().unwrap();
}

#[test]
fn selene_leaked_lazy_measure_reads_two_under_leakage_noise() {
    use pecos_engines::{QuantumSystem, noise::general::GeneralNoiseModel};
    let _env_lock = crate::test_env::ENV_MUTEX.lock().unwrap();
    // Both SeleneFutureResult and SeleneU64Result use a hidden return pointer
    // on Windows, and two registers on the other supported 64-bit targets.
    #[cfg(not(windows))]
    let (declarations, calls) = (
        "declare {i32, i64} @selene_qubit_lazy_measure_leaked(ptr, i64)\ndeclare {i32, i64} @selene_future_read_u64(ptr, i64)",
        "%future = call {i32, i64} @selene_qubit_lazy_measure_leaked(ptr null, i64 0)\n%id = extractvalue {i32, i64} %future, 1\n%read = call {i32, i64} @selene_future_read_u64(ptr null, i64 %id)",
    );
    #[cfg(windows)]
    let (declarations, calls) = (
        "declare void @selene_qubit_lazy_measure_leaked(ptr sret({i32, i64}), ptr, i64)\ndeclare void @selene_future_read_u64(ptr sret({i32, i64}), ptr, i64)",
        "%out = alloca {i32, i64}\ncall void @selene_qubit_lazy_measure_leaked(ptr sret({i32, i64}) %out, ptr null, i64 0)\n%future = load {i32, i64}, ptr %out\n%id = extractvalue {i32, i64} %future, 1\ncall void @selene_future_read_u64(ptr sret({i32, i64}) %out, ptr null, i64 %id)\n%read = load {i32, i64}, ptr %out",
    );
    let source = format!(
        r"
        {declarations}
        declare void @__quantum__qis__x__body(i64)
        define i64 @qmain(i64 %shot) {{
            call void @__quantum__qis__x__body(i64 0)
            {calls}
            %value = extractvalue {{i32, i64}} %read, 1
            %bad = icmp ne i64 %value, 2
            %status = zext i1 %bad to i64
            ret i64 %status
        }}
    "
    );
    let mut engine = dynamic_read_engine(&source);
    let noise = GeneralNoiseModel::builder()
        .with_p1(1.0)
        .with_p1_emission_ratio(1.0)
        .with_p1_emission_model(&BTreeMap::from([("L".to_string(), 1.0)]))
        .build();
    let mut quantum = QuantumSystem::new(Box::new(noise), Box::new(StateVecEngine::new(2)));
    let stage = engine.start(()).unwrap();
    let shot = finish_deferred_read_shot(&mut engine, &mut quantum, stage).unwrap();
    assert_eq!(shot.data.get("measurement_0"), Some(&Data::U32(2)));
}

#[test]
fn scheduled_remeasured_slot_reads_and_records_latest_outcome() {
    let _env_lock = crate::test_env::ENV_MUTEX.lock().unwrap();
    let source = r"
        declare void @__quantum__qis__x__body(i64)
        declare i32 @__quantum__qis__m__body(i64, i64)
        declare i1 @___read_future_bool(i64)
        define i64 @qmain(i64 %shot) {
            %first = call i32 @__quantum__qis__m__body(i64 0, i64 7)
            call void @__quantum__qis__x__body(i64 0)
            %second = call i32 @__quantum__qis__m__body(i64 0, i64 7)
            %read = call i1 @___read_future_bool(i64 7)
            %bad = xor i1 %read, true
            %status = zext i1 %bad to i64
            ret i64 %status
        }
    ";
    for mode in [ScheduledTransport::V3, ScheduledTransport::V4] {
        let mut engine = dynamic_read_engine(source);
        engine.scheduled_transport = mode;
        for _ in 0..2 {
            let stage = engine.start(()).unwrap();
            let shot =
                finish_deferred_read_shot(&mut engine, &mut deferred_read_quantum(mode), stage)
                    .unwrap();
            assert_eq!(shot.data.get("measurement_7"), Some(&Data::U32(1)));
        }
    }
}

#[test]
fn soft_rz_large_scheduled_gate_only_tail_completes() {
    let _env_lock = crate::test_env::ENV_MUTEX.lock().unwrap();
    let source = br"
        declare void @__quantum__qis__x__body(i64)
        define i64 @qmain(i64 %shot) {
        entry:
            br label %loop
        loop:
            %i = phi i64 [0, %entry], [%next, %loop]
            call void @__quantum__qis__x__body(i64 0)
            %next = add i64 %i, 1
            %more = icmp ult i64 %next, 8192
            br i1 %more, label %loop, label %done
        done:
            ret i64 0
        }
    ";
    for mode in [ScheduledTransport::V3, ScheduledTransport::V4] {
        let mut engine = QisEngine::new(
            Box::new(crate::QisHeliosInterface::new()),
            Box::new(crate::selene_soft_rz_runtime().unwrap()),
        );
        engine.set_num_qubits_hint(2);
        engine.scheduled_transport = mode;
        engine
            .load_program(source, ProgramFormat::LlvmIrText)
            .unwrap();
        let mut quantum = deferred_read_quantum(mode);
        let stage = engine.start(()).unwrap();
        finish_deferred_read_shot(&mut engine, &mut quantum, stage).unwrap();
        assert_eq!(engine.scheduled_drain_round, 2);
        assert!(engine.get_results().is_ok());
    }
}
