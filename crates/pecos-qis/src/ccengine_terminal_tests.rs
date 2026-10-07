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
