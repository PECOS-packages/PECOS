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
                assert_eq!(tail.len(), 1);
                assert_eq!(tail[0].gate_type, GateType::RXY1Q);
                // Legacy static handles release on measurement; the new logical
                // handle then reuses slot 0. Explicit lifetimes keep slot 1.
                assert_eq!(tail[0].qubits.as_slice(), &[QubitId(usize::from(explicit))]);
                if explicit {
                    assert!(output.probabilities[3] > 1.0 - 1e-10);
                }
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
