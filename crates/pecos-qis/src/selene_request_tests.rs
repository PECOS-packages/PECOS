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
    with_config(|_| {}, run);
}

fn with_config(
    configure: impl FnOnce(&mut RuntimePluginDescriptorV1),
    run: impl FnOnce(SeleneRuntime),
) {
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
    configure(&mut descriptor);
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
            declare i32 @selene_simulate_delay(ptr, i64)
            declare void @__quantum__qis__r1xy__body(double, double, i64)
            define i64 @qmain(i64 %shot) {
                %a = call i64 @__quantum__rt__qubit_allocate()
                %b = call i64 @__quantum__rt__qubit_allocate()
                %l = call i32 @selene_local_barrier(ptr null, ptr @qubits, i64 2, i64 23)
                %g = call i32 @selene_global_barrier(ptr null, i64 37)
                %d = call i32 @selene_simulate_delay(ptr null, i64 101)
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
                .any(|gate| gate.gate_type == "Idle" && gate.params == [161e-9]),
            "{gates:?}"
        );
        assert_eq!(traces.last().unwrap().stage, "trace_complete");
    });
}

#[test]
fn classical_delay_missing_callback_and_native_errors_fail() {
    unsafe extern "C" fn fail(_: RuntimeInstance, _: u64) -> i32 {
        84
    }
    for missing in [false, true] {
        with_config(
            |d| d.simulate_delay_fn = if missing { None } else { Some(fail) },
            |mut runtime| {
                let error = runtime
                    .lower_operations(&[Operation::RuntimeRequest {
                        request: RuntimeRequest::SimulateDelay { nanoseconds: 101 },
                        result_id: Some(0),
                    }])
                    .unwrap_err()
                    .to_string();
                assert!(
                    error.contains(if missing {
                        "simulate_delay is not supported"
                    } else {
                        "simulate_delay failed with errno 84"
                    }),
                    "{error}"
                );
                assert!(runtime.lower_operations(&[]).is_err());
                assert!(CALLS.with_borrow(Vec::is_empty));
            },
        );
    }
}

#[test]
fn classical_delay_preserves_full_width_nanoseconds_without_a_barrier() {
    thread_local! { static DELAY: Cell<u64> = const { Cell::new(0) }; }
    unsafe extern "C" fn capture(_: RuntimeInstance, delay: u64) -> i32 {
        DELAY.set(delay);
        0
    }
    with_config(
        |d| d.simulate_delay_fn = Some(capture),
        |mut runtime| {
            for value in [0, 1, u64::MAX] {
                runtime
                    .lower_operations(&[Operation::RuntimeRequest {
                        request: RuntimeRequest::SimulateDelay { nanoseconds: value },
                        result_id: None,
                    }])
                    .unwrap();
                assert_eq!(DELAY.get(), value);
            }
            assert!(CALLS.with_borrow(Vec::is_empty));
        },
    );
}

// Compare handle-bearing custom calls with direct Selene.
#[test]
fn native_custom_qubit_handle_identity() {
    use selene_core::runtime::{
        RuntimeInterface, RuntimeInterfaceFactory, plugin::RuntimePluginInterface,
    };
    unsafe extern "C" fn use_qubit(
        instance: RuntimeInstance,
        _: u64,
        bytes: *const c_void,
        size: usize,
        out: *mut u64,
    ) -> i32 {
        assert_eq!(size, 8);
        let qubit = unsafe { bytes.cast::<u64>().read_unaligned() };
        unsafe {
            *out = qubit;
            (ORIGINAL.get().unwrap().rxy_gate_fn)(instance, qubit, 0.0, 0.0)
        }
    }
    let _lock = crate::test_env::ENV_MUTEX.lock().unwrap();
    with_config(
        |d| d.custom_call_fn = Some(use_qubit),
        |runtime| {
            let mut direct = RuntimePluginInterface::new_from_file(&runtime.plugin_path)
                .unwrap()
                .init(2, 0.into(), &runtime.init_args)
                .unwrap();
            direct.shot_start(0, 7).unwrap();
            let first = direct.qalloc().unwrap();
            direct.qfree(first).unwrap();
            let second = direct.qalloc().unwrap();
            assert_eq!(second, 0);
            assert_eq!(direct.custom_call(0, &second.to_ne_bytes()).unwrap(), 0);
            while direct.get_next_operations().unwrap().is_some() {}
            let source = r"
            declare {i32, i64} @selene_qalloc(ptr)
            declare i32 @selene_qfree(ptr, i64)
            declare {i32, i64} @selene_custom_runtime_call(ptr, i64, ptr, i64)
            define i64 @qmain(i64 %shot) {
                %a = call {i32, i64} @selene_qalloc(ptr null)
                %q = extractvalue {i32, i64} %a, 1
                %free = call i32 @selene_qfree(ptr null, i64 %q)
                %b = call {i32, i64} @selene_qalloc(ptr null)
                %r = extractvalue {i32, i64} %b, 1
                %data = alloca i64
                store i64 %r, ptr %data
                %custom = call {i32, i64} @selene_custom_runtime_call(ptr null, i64 0, ptr %data, i64 8)
                ret i64 0
            }
        ";
            run_native_source(runtime, source).unwrap();
        },
    );
}

// Compare handle-bearing custom calls with direct Selene.
#[test]
fn native_custom_returned_qubit_identity() {
    use selene_core::runtime::{
        RuntimeInterface, RuntimeInterfaceFactory, plugin::RuntimePluginInterface,
    };
    unsafe extern "C" fn use_qubit(
        instance: RuntimeInstance,
        _: u64,
        bytes: *const c_void,
        size: usize,
        out: *mut u64,
    ) -> i32 {
        assert_eq!(size, 8);
        let qubit = unsafe { bytes.cast::<u64>().read_unaligned() };
        unsafe {
            *out = qubit;
            (ORIGINAL.get().unwrap().rxy_gate_fn)(instance, qubit, 0.0, 0.0)
        }
    }
    let _lock = crate::test_env::ENV_MUTEX.lock().unwrap();
    with_config(
        |d| d.custom_call_fn = Some(use_qubit),
        |runtime| {
            let mut direct = RuntimePluginInterface::new_from_file(&runtime.plugin_path)
                .unwrap()
                .init(2, 0.into(), &runtime.init_args)
                .unwrap();
            direct.shot_start(0, 7).unwrap();
            let first = direct.qalloc().unwrap();
            direct.qfree(first).unwrap();
            let second = direct.qalloc().unwrap();
            assert_eq!(second, 0);
            assert_eq!(direct.custom_call(0, &second.to_ne_bytes()).unwrap(), 0);
            while direct.get_next_operations().unwrap().is_some() {}
            let source = r"
            declare {i32, i64} @selene_qalloc(ptr)
            declare i32 @selene_qfree(ptr, i64)
            declare i32 @selene_rxy(ptr, i64, double, double)
            declare {i32, i64} @selene_custom_runtime_call(ptr, i64, ptr, i64)
            define i64 @qmain(i64 %shot) {
                %a = call {i32, i64} @selene_qalloc(ptr null)
                %q = extractvalue {i32, i64} %a, 1
                %free = call i32 @selene_qfree(ptr null, i64 %q)
                %b = call {i32, i64} @selene_qalloc(ptr null)
                %r = extractvalue {i32, i64} %b, 1
                %data = alloca i64
                store i64 0, ptr %data
                %custom = call {i32, i64} @selene_custom_runtime_call(ptr null, i64 0, ptr %data, i64 8)
                %handle = extractvalue {i32, i64} %custom, 1
                %gate = call i32 @selene_rxy(ptr null, i64 %handle, double 0.0, double 0.0)
                ret i64 0
            }
        ";
            run_native_source(runtime, source).unwrap();
        },
    );
}

#[test]
fn native_custom_future_handle_identity() {
    use selene_core::runtime::{
        RuntimeInterface, RuntimeInterfaceFactory, plugin::RuntimePluginInterface,
    };
    thread_local! { static FUTURE: Cell<u64> = const { Cell::new(100) }; }
    unsafe extern "C" fn measure(_: RuntimeInstance, _: u64, out: *mut u64) -> i32 {
        unsafe {
            *out = FUTURE.get();
        }
        0
    }
    unsafe extern "C" fn read(_: RuntimeInstance, future: u64, out: *mut i8) -> i32 {
        if future != FUTURE.get() {
            return 96;
        }
        unsafe {
            *out = 1;
        }
        0
    }
    unsafe extern "C" fn custom(
        _: RuntimeInstance,
        tag: u64,
        bytes: *const c_void,
        size: usize,
        out: *mut u64,
    ) -> i32 {
        if tag == 0 {
            if size != 8 {
                return 98;
            }
            let supplied = unsafe { bytes.cast::<u64>().read_unaligned() };
            if supplied != FUTURE.get() {
                return 97;
            }
        }
        unsafe {
            *out = FUTURE.get();
        }
        0
    }
    let _lock = crate::test_env::ENV_MUTEX.lock().unwrap();
    for native in [100, (1_u64 << 63) + 7, u64::MAX - 1, u64::MAX] {
        for returned in [false, true] {
            FUTURE.set(native);
            with_config(
                |d| {
                    d.measure_fn = measure;
                    d.get_bool_result_fn = read;
                    d.custom_call_fn = Some(custom);
                },
                |runtime| {
                    let mut direct = RuntimePluginInterface::new_from_file(&runtime.plugin_path)
                        .unwrap()
                        .init(2, 0.into(), &runtime.init_args)
                        .unwrap();
                    direct.shot_start(0, 7).unwrap();
                    let q = direct.qalloc().unwrap();
                    let future = direct.measure(q).unwrap();
                    while direct.get_next_operations().unwrap().is_some() {}
                    let response = direct
                        .custom_call(u64::from(returned), &future.to_ne_bytes())
                        .unwrap();
                    assert_eq!(response, native);
                    assert_eq!(direct.get_bool_result(response).unwrap(), Some(true));
                    let tag = u64::from(returned);
                    let read_back = if returned {
                        "%handle = extractvalue {i32, i64} %custom, 1\n%read = call {i32, i8} @selene_future_read_bool(ptr null, i64 %handle)"
                    } else {
                        ""
                    };
                    let source = format!(
                        r"
                    declare {{i32, i64}} @selene_qalloc(ptr)
                    declare {{i32, i64}} @selene_qubit_lazy_measure(ptr, i64)
                    declare {{i32, i64}} @selene_custom_runtime_call(ptr, i64, ptr, i64)
                    declare {{i32, i8}} @selene_future_read_bool(ptr, i64)
                    define i64 @qmain(i64 %shot) {{
                        %a = call {{i32, i64}} @selene_qalloc(ptr null)
                        %q = extractvalue {{i32, i64}} %a, 1
                        %m = call {{i32, i64}} @selene_qubit_lazy_measure(ptr null, i64 %q)
                        %future = extractvalue {{i32, i64}} %m, 1
                        %data = alloca i64
                        store i64 %future, ptr %data
                        %custom = call {{i32, i64}} @selene_custom_runtime_call(ptr null, i64 {tag}, ptr %data, i64 8)
                        {read_back}
                        ret i64 0
                    }}
                "
                    );
                    run_native_source(runtime, &source).unwrap();
                },
            );
        }
    }
}

#[test]
fn native_allocation_failure_sentinel_is_returned_to_program() {
    let _lock = crate::test_env::ENV_MUTEX.lock().unwrap();
    with_config(
        |_| {},
        |runtime| {
            let source = r"
            declare i64 @___qalloc()
            declare void @___qfree(i64)
            define i64 @qmain(i64 %shot) {
                %a = call i64 @___qalloc()
                %b = call i64 @___qalloc()
                %full = call i64 @___qalloc()
                %failed = icmp eq i64 %full, -1
                call void @___qfree(i64 %a)
                %reused = call i64 @___qalloc()
                %same = icmp eq i64 %a, %reused
                %ok = and i1 %failed, %same
                %status = select i1 %ok, i64 0, i64 1
                ret i64 %status
            }
        ";
            run_native_source(runtime, source).unwrap();
        },
    );
}

fn run_native_source(
    runtime: SeleneRuntime,
    source: &str,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let mut engine = crate::QisEngine::new(
        Box::new(crate::QisHeliosInterface::new()),
        Box::new(runtime),
    );
    engine.set_num_qubits_hint(2);
    engine.load_program(source.as_bytes(), crate::ProgramFormat::LlvmIrText)?;
    let mut quantum = StateVecEngine::new(2);
    let mut stage = engine.start(())?;
    loop {
        match stage {
            EngineStage::NeedsProcessing(commands) => {
                stage = engine.continue_processing(quantum.process(commands)?)?;
            }
            EngineStage::Complete(_) => return Ok(()),
        }
    }
}

#[test]
fn native_guppy_futures_preserve_unsigned_handles_and_lazy_force() {
    thread_local! {
        static READY: Cell<bool> = const { Cell::new(false) };
        static FORCES: Cell<usize> = const { Cell::new(0) };
        static COUNTS: RefCell<Vec<bool>> = const { RefCell::new(Vec::new()) };
    }
    const FUTURE: u64 = (1_u64 << 63) + 7;
    unsafe extern "C" fn measure(_: RuntimeInstance, _: u64, out: *mut u64) -> i32 {
        unsafe {
            *out = FUTURE;
        }
        0
    }
    unsafe extern "C" fn read_value(_: RuntimeInstance, future: u64, out: *mut u64) -> i32 {
        if future != FUTURE {
            return 96;
        }
        unsafe {
            *out = if READY.get() { 2 } else { u64::MAX };
        }
        0
    }
    unsafe extern "C" fn force(_: RuntimeInstance, future: u64) -> i32 {
        if future != FUTURE {
            return 97;
        }
        FORCES.set(FORCES.get() + 1);
        READY.set(true);
        0
    }
    unsafe extern "C" fn increment(_: RuntimeInstance, future: u64) -> i32 {
        if future != FUTURE {
            return 98;
        }
        COUNTS.with_borrow_mut(|counts| counts.push(true));
        0
    }
    unsafe extern "C" fn decrement(_: RuntimeInstance, future: u64) -> i32 {
        if future != FUTURE {
            return 99;
        }
        COUNTS.with_borrow_mut(|counts| counts.push(false));
        0
    }
    let _lock = crate::test_env::ENV_MUTEX.lock().unwrap();
    for read in [false, true] {
        READY.set(false);
        FORCES.set(0);
        COUNTS.with_borrow_mut(Vec::clear);
        with_config(
            |d| {
                d.measure_leaked_fn = measure;
                d.get_u64_result_fn = read_value;
                d.force_result_fn = force;
                d.increment_future_refcount_fn = increment;
                d.decrement_future_refcount_fn = decrement;
            },
            |runtime| {
                let read_ir = if read {
                    "%value = call i64 @___read_future_uint(i64 %future)\n%ok = icmp eq i64 %value, 2\n%status = select i1 %ok, i64 0, i64 1"
                } else {
                    "%ok = icmp eq i64 %future, 9223372036854775815\n%status = select i1 %ok, i64 0, i64 1"
                };
                let source = format!(
                    r"
                declare i64 @___qalloc()
                declare i64 @___lazy_measure_leaked(i64)
                declare i64 @___read_future_uint(i64)
                declare void @___inc_future_refcount(i64)
                declare void @___dec_future_refcount(i64)
                define i64 @qmain(i64 %shot) {{
                    %q = call i64 @___qalloc()
                    %future = call i64 @___lazy_measure_leaked(i64 %q)
                    call void @___inc_future_refcount(i64 %future)
                    {read_ir}
                    call void @___dec_future_refcount(i64 %future)
                    ret i64 %status
                }}
            "
                );
                run_native_source(runtime, &source).unwrap();
            },
        );
        assert_eq!(FORCES.get(), usize::from(read));
        assert_eq!(COUNTS.with_borrow(Clone::clone), [true, false]);
    }
}

#[test]
fn native_custom_call_events_are_executed_or_rejected_before_return() {
    thread_local! { static EVENT: Cell<Option<u64>> = const { Cell::new(None) }; }
    unsafe extern "C" fn custom(
        instance: RuntimeInstance,
        _: u64,
        data: *const c_void,
        size: usize,
        out: *mut u64,
    ) -> i32 {
        if size != 8 {
            return 90;
        }
        let qubit = unsafe { data.cast::<u64>().read_unaligned() };
        let errno = unsafe { ORIGINAL.get().unwrap().simulate_delay_fn.unwrap()(instance, 101) };
        if errno != 0 {
            return errno;
        }
        EVENT.set(Some(qubit));
        unsafe {
            *out = qubit;
        }
        0
    }
    unsafe extern "C" fn next(
        instance: RuntimeInstance,
        output: SeleneRuntimeGetOperationHandle,
    ) -> i32 {
        if let Some(qubit) = EVENT.take() {
            let bytes = qubit.to_le_bytes();
            unsafe {
                (output.interface.custom)(
                    output.instance,
                    7301,
                    bytes.as_ptr().cast(),
                    bytes.len(),
                );
                (output.interface.set_batch_time)(output.instance, 101, 0);
            }
            0
        } else {
            unsafe {
                (ORIGINAL.get().unwrap().get_next_operations_fn)(
                    instance,
                    std::mem::transmute::<
                        SeleneRuntimeGetOperationHandle,
                        selene_core::operation::plugin::RuntimeGetOperationHandle,
                    >(output),
                )
            }
        }
    }
    struct PhysicalX;
    impl RuntimeBatchAdapter for PhysicalX {
        fn normalize(&mut self, batch: &RuntimeBatch) -> Result<Vec<RuntimeScheduledOp>> {
            batch
                .operations
                .iter()
                .map(|op| match op {
                    RuntimeScheduledOp::Custom { tag: 7301, data } if data.len() == 8 => {
                        let qubit_id = u64::from_le_bytes(data.as_slice().try_into().unwrap());
                        Ok(RuntimeScheduledOp::Rxy {
                            qubit_id,
                            theta: std::f64::consts::PI,
                            phi: 0.0,
                        })
                    }
                    RuntimeScheduledOp::Custom { .. } => Err(RuntimeError::ExecutionError(
                        "unsupported synthetic event".into(),
                    )),
                    gate => Ok(gate.clone()),
                })
                .collect()
        }
    }
    let _lock = crate::test_env::ENV_MUTEX.lock().unwrap();
    for normalize in [false, true] {
        EVENT.set(None);
        with_config(
            |d| {
                d.custom_call_fn = Some(custom);
                d.get_next_operations_fn = unsafe {
                    std::mem::transmute::<
                        unsafe extern "C" fn(
                            RuntimeInstance,
                            SeleneRuntimeGetOperationHandle,
                        ) -> i32,
                        unsafe extern "C" fn(
                            RuntimeInstance,
                            selene_core::operation::plugin::RuntimeGetOperationHandle,
                        ) -> i32,
                    >(next)
                };
            },
            |mut runtime| {
                if normalize {
                    runtime.set_batch_adapter_factory(|| Ok(Box::new(PhysicalX)));
                }
                let source = r"
                declare i64 @___qalloc()
                declare void @___reset(i64)
                declare i64 @___lazy_measure(i64)
                declare i1 @___read_future_bool(i64)
                declare {i32, i64} @selene_custom_runtime_call(ptr, i64, ptr, i64)
                define i64 @qmain(i64 %shot) {
                    %q = call i64 @___qalloc()
                    call void @___reset(i64 %q)
                    %data = alloca i64
                    store i64 %q, ptr %data
                    %custom = call {i32, i64} @selene_custom_runtime_call(ptr null, i64 42, ptr %data, i64 8)
                    %returned = extractvalue {i32, i64} %custom, 1
                    %future = call i64 @___lazy_measure(i64 %returned)
                    %one = call i1 @___read_future_bool(i64 %future)
                    %status = select i1 %one, i64 0, i64 1
                    ret i64 %status
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
                let result = (|| -> std::result::Result<(), Box<dyn std::error::Error>> {
                    let mut stage = engine.start(())?;
                    loop {
                        match stage {
                            EngineStage::NeedsProcessing(commands) => {
                                stage = engine.continue_processing(quantum.process(commands)?)?;
                            }
                            EngineStage::Complete(_) => return Ok(()),
                        }
                    }
                })();
                if normalize {
                    result.unwrap();
                    let trace = trace.lock().unwrap();
                    let gates: Vec<_> = trace
                        .iter()
                        .flat_map(|chunk| &chunk.lowered_quantum_ops)
                        .collect();
                    assert_eq!(
                        gates
                            .iter()
                            .filter(|g| g.gate_type == "Idle")
                            .map(|g| &g.params)
                            .collect::<Vec<_>>(),
                        [&vec![101e-9]]
                    );
                    assert_eq!(
                        gates
                            .iter()
                            .filter(|g| g.gate_type == "MZ")
                            .map(|g| &g.measurement_result_ids)
                            .collect::<Vec<_>>(),
                        [&vec![0]]
                    );
                    assert_eq!(trace.last().unwrap().stage, "trace_complete");
                } else {
                    let error = result.unwrap_err().to_string();
                    assert!(error.contains("tag 7301"), "{error}");
                    assert!(
                        !trace
                            .lock()
                            .unwrap()
                            .iter()
                            .any(|chunk| chunk.stage == "trace_complete")
                    );
                }
            },
        );
    }
}

#[test]
fn selene_rpp_abi_preserves_native_targets_and_angles() {
    let _lock = crate::test_env::ENV_MUTEX.lock().unwrap();
    with_config(
        |_| {},
        |runtime| {
            let source = r"
            declare i64 @___qalloc()
            declare i32 @selene_rpp(ptr, i64, i64, double, double)
            declare i1 @___measure(i64)
            define i64 @qmain(i64 %shot) {
                %a = call i64 @___qalloc()
                %b = call i64 @___qalloc()
                %gate = call i32 @selene_rpp(ptr null, i64 %a, i64 %b, double 0x400921FB54442D18, double 0.0)
                %one = call i1 @___measure(i64 %a)
                %two = call i1 @___measure(i64 %b)
                %ok = and i1 %one, %two
                %status = select i1 %ok, i64 0, i64 1
                ret i64 %status
            }
        ";
            run_native_source(runtime, source).unwrap();
        },
    );
}

#[test]
fn custom_created_measurements_use_engine_feedback_without_source_slots() {
    use pecos_engines::ClassicalControlEngineBuilder;
    thread_local! { static DELIVERIES: RefCell<Vec<(u64, bool)>> = const { RefCell::new(Vec::new()) }; }
    unsafe extern "C" fn custom(
        instance: RuntimeInstance,
        _: u64,
        _: *const c_void,
        _: usize,
        out: *mut u64,
    ) -> i32 {
        let original = ORIGINAL.get().unwrap();
        let mut first = 0;
        let code = unsafe { (original.measure_fn)(instance, 0, &raw mut first) };
        if code != 0 {
            return code;
        }
        unsafe { (original.measure_fn)(instance, 1, out) }
    }
    unsafe extern "C" fn feedback(instance: RuntimeInstance, future: u64, value: bool) -> i32 {
        DELIVERIES.with_borrow_mut(|values| values.push((future, value)));
        unsafe { (ORIGINAL.get().unwrap().set_bool_result_fn)(instance, future, value) }
    }
    unsafe extern "C" fn deferred_measure(_: RuntimeInstance, _: u64, out: *mut u64) -> i32 {
        unsafe {
            *out = 0;
        }
        0
    }
    unsafe extern "C" fn discard(_: RuntimeInstance, _: u64) -> i32 {
        0
    }
    let source = r"
        declare i64 @___qalloc()
        declare i64 @___lazy_measure(i64)
        declare void @___dec_future_refcount(i64)
        declare void @___reset(i64)
        declare void @___rxy(i64, double, double)
        declare {i32, i64} @selene_custom_runtime_call(ptr, i64, ptr, i64)
        declare i1 @___read_future_bool(i64)
        define i64 @qmain(i64 %shot) {
            %a = call i64 @___qalloc()
            %b = call i64 @___qalloc()
            call void @___reset(i64 %a)
            call void @___reset(i64 %b)
            call void @___rxy(i64 %b, double 3.141592653589793, double 0.0)
            %call = call {i32, i64} @selene_custom_runtime_call(ptr null, i64 0, ptr null, i64 0)
            %future = extractvalue {i32, i64} %call, 1
            %value = call i1 @___read_future_bool(i64 %future)
            %failed = xor i1 %value, true
            %status = zext i1 %failed to i64
            ret i64 %status
        }
    ";
    let _lock = crate::test_env::ENV_MUTEX.lock().unwrap();
    for (trace_enabled, deferred) in [(false, false), (true, false), (false, true), (true, true)] {
        let source = if deferred {
            source.replace("%call = call", "%old = call i64 @___lazy_measure(i64 %a)\n            call void @___dec_future_refcount(i64 %old)\n            %call = call")
        } else {
            source.to_string()
        };
        DELIVERIES.with_borrow_mut(Vec::clear);
        with_config(
            |d| {
                if deferred {
                    d.measure_fn = deferred_measure;
                    d.decrement_future_refcount_fn = discard;
                }
                d.custom_call_fn = Some(custom);
                d.set_bool_result_fn = feedback;
            },
            |runtime| {
                let trace = crate::OperationTraceStore::default();
                // Exercise the public builder: configuring a program must not run
                // custom calls in a preliminary static execution.
                let mut builder = crate::qis_engine()
                    .interface(crate::helios_interface_builder())
                    .runtime(runtime)
                    .try_program(pecos_programs::Qis::from_string(&source))
                    .unwrap();
                if trace_enabled {
                    builder = builder.trace_operations_in_memory_to(Arc::clone(&trace));
                }
                let mut engine = builder.build().unwrap();
                engine.set_num_qubits_hint(2);
                let mut quantum = StateVecEngine::new(2);
                let outcome = (|| -> std::result::Result<_, pecos_core::errors::PecosError> {
                    let mut stage = engine.start(())?;
                    loop {
                        match stage {
                            EngineStage::NeedsProcessing(commands) => {
                                stage = engine.continue_processing(quantum.process(commands)?)?;
                            }
                            EngineStage::Complete(shot) => return Ok(shot),
                        }
                    }
                })();
                if trace_enabled {
                    assert!(
                        outcome
                            .unwrap_err()
                            .to_string()
                            .contains("runtime-owned measurement")
                    );
                    assert!(
                        !trace
                            .lock()
                            .unwrap()
                            .iter()
                            .any(|c| c.stage == "trace_complete")
                    );
                } else {
                    let shot = outcome.unwrap();
                    assert!(
                        shot.data.is_empty(),
                        "internal outcomes must not become source results"
                    );
                    assert_eq!(
                        DELIVERIES.with_borrow(Clone::clone),
                        [(0, false), (1, true)]
                    );
                }
            },
        );
    }
}

#[test]
fn opaque_future_reuse_does_not_inherit_source_provenance() {
    use pecos_engines::ClassicalControlEngineBuilder;
    thread_local! { static REUSED: Cell<bool> = const { Cell::new(false) }; }
    unsafe extern "C" fn custom(
        _: RuntimeInstance,
        _: u64,
        _: *const c_void,
        _: usize,
        out: *mut u64,
    ) -> i32 {
        REUSED.set(true);
        unsafe {
            *out = 0;
        }
        0
    }
    unsafe extern "C" fn read(instance: RuntimeInstance, future: u64, out: *mut i8) -> i32 {
        if REUSED.get() {
            unsafe {
                *out = 1;
            }
            0
        } else {
            unsafe { (ORIGINAL.get().unwrap().get_bool_result_fn)(instance, future, out) }
        }
    }
    let source = r#"
        @name = private constant [8 x i8] c"detector"
        declare i64 @___qalloc()
        declare void @___reset(i64)
        declare i64 @___lazy_measure(i64)
        declare void @___dec_future_refcount(i64)
        declare {i32, i64} @selene_custom_runtime_call(ptr, i64, ptr, i64)
        declare i1 @___read_future_bool(i64)
        declare void @print_bool_selene(ptr, i64, i1)
        define i64 @qmain(i64 %shot) {
            %q = call i64 @___qalloc()
            call void @___reset(i64 %q)
            %old = call i64 @___lazy_measure(i64 %q)
            call void @___dec_future_refcount(i64 %old)
            %call = call {i32, i64} @selene_custom_runtime_call(ptr null, i64 0, ptr null, i64 0)
            %new = extractvalue {i32, i64} %call, 1
            %value = call i1 @___read_future_bool(i64 %new)
            call void @print_bool_selene(ptr @name, i64 8, i1 %value)
            ret i64 0
        }
    "#;
    let _lock = crate::test_env::ENV_MUTEX.lock().unwrap();
    for (trace_enabled, output) in [
        (false, "bool"),
        (true, "bool"),
        (true, "repeated"),
        (true, "integer"),
        (false, "qir"),
        (true, "qir"),
    ] {
        let source = if output == "qir" {
            source
                .replace("declare i64 @___qalloc()", "declare i32 @__quantum__qis__m__body(i64, i64)\ndeclare i32 @__quantum__rt__result_get_one(i64)")
                .replace("%q = call i64 @___qalloc()\n            call void @___reset(i64 %q)\n            %old = call i64 @___lazy_measure(i64 %q)\n            call void @___dec_future_refcount(i64 %old)", "%old = call i32 @__quantum__qis__m__body(i64 0, i64 7)")
                .replace("%value = call i1 @___read_future_bool(i64 %new)", "%raw = call i32 @__quantum__rt__result_get_one(i64 7)\n            %value = icmp ne i32 %raw, 0")
        } else if output == "repeated" {
            source.replace(
                "%value = call",
                "%unused = call i1 @___read_future_bool(i64 %new)\n            %value = call",
            )
        } else if output == "integer" {
            source.replace("declare void @print_bool_selene(ptr, i64, i1)", "declare void @print_int_selene(ptr, i64, i64)")
                .replace("call void @print_bool_selene(ptr @name, i64 8, i1 %value)", "%integer = zext i1 %value to i64\n            call void @print_int_selene(ptr @name, i64 8, i64 %integer)")
        } else {
            source.to_string()
        };
        REUSED.set(false);
        with_config(
            |d| {
                d.custom_call_fn = Some(custom);
                d.get_bool_result_fn = read;
            },
            |runtime| {
                let trace = crate::OperationTraceStore::default();
                let mut builder = crate::qis_engine()
                    .interface(crate::helios_interface_builder())
                    .runtime(runtime)
                    .try_program(pecos_programs::Qis::from_string(&source))
                    .unwrap();
                if trace_enabled {
                    builder = builder.trace_operations_in_memory_to(Arc::clone(&trace));
                }
                let mut engine = builder.build().unwrap();
                engine.set_num_qubits_hint(2);
                let mut quantum = StateVecEngine::new(2);
                let outcome = (|| -> std::result::Result<_, pecos_core::errors::PecosError> {
                    let mut stage = engine.start(())?;
                    loop {
                        match stage {
                            EngineStage::NeedsProcessing(commands) => {
                                stage = engine.continue_processing(quantum.process(commands)?)?;
                            }
                            EngineStage::Complete(shot) => return Ok(shot),
                        }
                    }
                })();
                if output == "qir" {
                    assert!(
                        outcome
                            .unwrap_err()
                            .to_string()
                            .contains("no native measurement for program result 0")
                    );
                } else if trace_enabled {
                    assert!(
                        outcome
                            .unwrap_err()
                            .to_string()
                            .contains("unknown native future provenance")
                    );
                    assert!(
                        !trace
                            .lock()
                            .unwrap()
                            .iter()
                            .any(|c| c.stage == "trace_complete")
                    );
                } else {
                    assert_eq!(
                        outcome.unwrap().data.get("detector"),
                        Some(&pecos_engines::shot_results::Data::U32(1))
                    );
                }
            },
        );
    }
}

#[test]
fn decoder_request_never_forces_an_unread_native_future_or_inserts_a_barrier() {
    use crate::runtime_decoder::RuntimeDecoderSession;
    use pecos_core::errors::PecosError;
    use std::sync::Arc;
    thread_local! {
        static FORCES: Cell<usize> = const { Cell::new(0) };
    }
    unsafe extern "C" fn measure(_: RuntimeInstance, _: u64, out: *mut u64) -> i32 {
        unsafe {
            *out = 777;
        }
        0 // No physical batch is available until the runtime chooses to emit it.
    }
    unsafe extern "C" fn force(_: RuntimeInstance, _: u64) -> i32 {
        FORCES.set(FORCES.get() + 1);
        95 // Any decoder-induced forcing must fail the test.
    }
    struct Decoder;
    impl RuntimeDecoderSession for Decoder {
        fn decode(
            &mut self,
            window: u64,
            instance: u64,
            syndrome: &[u8],
        ) -> std::result::Result<u64, PecosError> {
            assert_eq!((window, instance, syndrome), (9, 0, &[0][..]));
            assert_eq!(FORCES.get(), 0);
            assert!(CALLS.with_borrow(Vec::is_empty));
            Ok(1)
        }
    }
    let _lock = crate::test_env::ENV_MUTEX.lock().unwrap();
    FORCES.set(0);
    with_config(
        |d| {
            d.measure_fn = measure;
            d.force_result_fn = force;
        },
        |runtime| {
            let source = r"
        declare i64 @___qalloc()
        declare i64 @___lazy_measure(i64)
        declare i64 @pecos_runtime_decode(i64, i64, ptr, i64)
        define i64 @qmain(i64 %shot) {
            %q = call i64 @___qalloc()
            %future = call i64 @___lazy_measure(i64 %q)
            %data = alloca i8
            store i8 0, ptr %data
            %frame = call i64 @pecos_runtime_decode(i64 9, i64 0, ptr %data, i64 1)
            %status = sub i64 1, %frame
            ret i64 %status
        }";
            let mut engine = crate::QisEngine::new(
                Box::new(crate::QisHeliosInterface::new()),
                Box::new(runtime),
            );
            engine.decoder_factory = Some(Arc::new(|| Ok(Box::new(Decoder))));
            engine.set_num_qubits_hint(2);
            engine
                .load_program(source.as_bytes(), crate::ProgramFormat::LlvmIrText)
                .unwrap();
            let mut quantum = StateVecEngine::new(2);
            let mut stage = engine.start(()).unwrap();
            while let EngineStage::NeedsProcessing(commands) = stage {
                stage = engine
                    .continue_processing(quantum.process(commands).unwrap())
                    .unwrap();
            }
        },
    );
    assert_eq!(FORCES.get(), 0);
    assert!(CALLS.with_borrow(Vec::is_empty));
}
