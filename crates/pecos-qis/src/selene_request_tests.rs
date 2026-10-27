//! Timing-aware runtime fixture. Delay interpretation belongs entirely to the plugin.
use super::*;
use pecos_engines::{ClassicalEngine, ControlEngine, Engine, EngineStage, StateVecEngine};
use std::cell::{Cell, RefCell};

type BarrierCall = (Option<Vec<u64>>, u64);
thread_local! {
    static ORIGINAL: Cell<Option<RuntimePluginDescriptorV1>> = const { Cell::new(None) };
    static CALLS: RefCell<Vec<BarrierCall>> = const { RefCell::new(Vec::new()) };
}
unsafe extern "C" fn local(
    instance: RuntimeInstance,
    qubits: *const u64,
    count: u64,
    delay: u64,
) -> i32 {
    let qubits = unsafe { std::slice::from_raw_parts(qubits, usize::try_from(count).unwrap()) };
    CALLS.with_borrow_mut(|calls| calls.push((Some(qubits.to_vec()), delay)));
    // This fixture defines a local barrier to advance the device clock. PECOS
    // must pass the scope unchanged even though this plugin chooses global time.
    unsafe { ORIGINAL.get().unwrap().simulate_delay_fn.unwrap()(instance, delay) }
}
unsafe extern "C" fn global(instance: RuntimeInstance, delay: u64) -> i32 {
    CALLS.with_borrow_mut(|calls| calls.push((None, delay)));
    unsafe { ORIGINAL.get().unwrap().simulate_delay_fn.unwrap()(instance, delay) }
}

fn with_runtime(run: impl FnOnce(SeleneRuntime)) {
    let executable = std::env::current_exe().unwrap();
    let source = crate::selene_runtimes::find_library_in_dir(
        executable.parent().unwrap(),
        pecos_qis_test_runtime::LIBRARY_NAME,
    )
    .unwrap();
    let directory = tempfile::tempdir().unwrap();
    let plugin = directory.path().join(source.file_name().unwrap());
    std::fs::copy(source, &plugin).unwrap();
    let public = crate::selene_runtimes::selene_simple_runtime().unwrap();
    let library = unsafe { libloading::Library::new(&public.plugin_path).unwrap() };
    let mut descriptor =
        Box::new(unsafe { SeleneRuntime::runtime_plugin_descriptor(&library).unwrap() });
    ORIGINAL.set(Some(*descriptor));
    descriptor.local_barrier_fn = local;
    descriptor.global_barrier_fn = global;
    descriptor.custom_call_fn = None;
    let fixture = unsafe { libloading::Library::new(&plugin).unwrap() };
    unsafe {
        fixture
            .get::<unsafe extern "C" fn(*mut c_void)>(b"set_descriptor")
            .unwrap()((&raw mut *descriptor).cast());
    }
    CALLS.with_borrow_mut(Vec::clear);
    let mut runtime = SeleneRuntime::new(plugin);
    runtime.init_args.clone_from(&public.init_args);
    runtime.set_num_qubits(2);
    run(runtime);
}

#[test]
fn requested_delays_keep_native_scope_units_and_timestamps() {
    for route in 0..3 {
        with_runtime(|mut runtime| {
            runtime.shot_start(0, Some(7)).unwrap();
            let requests = [
                Operation::AllocateQubit { id: 7 },
                Operation::AllocateQubit { id: 9 },
                Operation::RuntimeRequest {
                    request: RuntimeRequest::LocalBarrier {
                        qubits: vec![9, 7],
                        sleep_ns: 23,
                    },
                    result_id: Some(0),
                },
                QuantumOp::Idle(11e-9, 7).into(),
                Operation::RuntimeRequest {
                    request: RuntimeRequest::GlobalBarrier { sleep_ns: 37 },
                    result_id: Some(1),
                },
                QuantumOp::RXY(0.5, 0.0, 7).into(),
            ];
            match route {
                0 => assert_eq!(
                    runtime.lower_operations(&requests).unwrap(),
                    [QuantumOp::Idle(71e-9, 0), QuantumOp::RXY(0.5, 0.0, 0)]
                ),
                1 => assert_eq!(
                    runtime
                        .lower_operations_with_metadata(&requests)
                        .unwrap()
                        .into_iter()
                        .map(|op| op.op)
                        .collect::<Vec<_>>(),
                    [QuantumOp::Idle(71e-9, 0), QuantumOp::RXY(0.5, 0.0, 0)]
                ),
                _ => {
                    let batches = runtime.lower_scheduled_operations(&requests).unwrap();
                    assert_eq!(batches.len(), 1);
                    assert_eq!(batches[0].start_time_nanos, 71);
                    assert_eq!(
                        batches[0].operations,
                        [RuntimeScheduledOp::Rxy {
                            qubit_id: 0,
                            theta: 0.5,
                            phi: 0.0
                        }]
                    );
                }
            }
            assert_eq!(
                CALLS.with_borrow(Clone::clone),
                [(Some(vec![1, 0]), 23), (Some(vec![0]), 11), (None, 37)]
            );
            assert_eq!(runtime.read_result(0).unwrap(), Some(0));
            assert_eq!(runtime.read_result(1).unwrap(), Some(0));
        });
    }
}

#[test]
fn missing_custom_call_support_fails_loudly() {
    with_runtime(|mut runtime| {
        let error = runtime
            .lower_operations(&[Operation::RuntimeRequest {
                request: RuntimeRequest::Custom {
                    tag: 17,
                    data: vec![0, 255],
                },
                result_id: Some(0),
            }])
            .unwrap_err()
            .to_string();
        assert!(error.contains("custom_call is not supported"), "{error}");
        assert!(runtime.lower_operations(&[]).is_err());
    });
}

#[test]
fn program_barriers_reach_runtime_and_produce_timed_trace_idles() {
    let _lock = crate::test_env::ENV_MUTEX.lock().unwrap();
    with_runtime(|runtime| {
        let source = r"
            @qubits = private constant [2 x i64] [i64 1, i64 0]
            declare i64 @__quantum__rt__qubit_allocate()
            declare i32 @selene_local_barrier(ptr, ptr, i64, i64)
            declare i32 @selene_global_barrier(ptr, i64)
            declare void @__quantum__qis__r1xy__body(double, double, i64)
            define i64 @qmain(i64 %shot) {
                %a = call i64 @__quantum__rt__qubit_allocate()
                %b = call i64 @__quantum__rt__qubit_allocate()
                %l = call i32 @selene_local_barrier(ptr null, ptr @qubits, i64 2, i64 23)
                %g = call i32 @selene_global_barrier(ptr null, i64 37)
                call void @__quantum__qis__r1xy__body(double 0.5, double 0.0, i64 %a)
                ret i64 0
            }
        ";
        let trace = crate::OperationTraceStore::default();
        let mut engine = crate::QisEngine::new(
            Box::new(crate::QisHeliosInterface::new()),
            Box::new(runtime),
        );
        engine.set_num_qubits_hint(2);
        engine.set_operation_trace_collector(Arc::clone(&trace));
        engine
            .load_program(source.as_bytes(), crate::ProgramFormat::LlvmIrText)
            .unwrap();
        let mut quantum = StateVecEngine::new(2);
        let mut stage = engine.start(()).unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            assert!(std::time::Instant::now() < deadline);
            match stage {
                EngineStage::NeedsProcessing(commands) => {
                    stage = engine
                        .continue_processing(quantum.process(commands).unwrap())
                        .unwrap();
                }
                EngineStage::Complete(_) => break,
            }
        }
        assert_eq!(
            CALLS.with_borrow(Clone::clone),
            [(Some(vec![1, 0]), 23), (None, 37)]
        );
        let traces = trace.lock().unwrap();
        let gates: Vec<_> = traces
            .iter()
            .flat_map(|chunk| &chunk.lowered_quantum_ops)
            .collect();
        assert!(
            gates
                .iter()
                .any(|gate| gate.gate_type == "Idle" && gate.params == [60e-9]),
            "{gates:?}"
        );
        assert_eq!(traces.last().unwrap().stage, "trace_complete");
    });
}
