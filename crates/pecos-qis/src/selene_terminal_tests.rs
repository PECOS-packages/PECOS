//! A native descriptor that emits its only extra batch from `shot_end`.
use super::*;
use crate::scheduled_transport::ScheduledTransport;
use pecos_core::errors::PecosError;
use pecos_engines::noise::IntoNoiseModel;
use pecos_engines::quantum_system::QuantumSystem;
use pecos_engines::runtime_frame::ShotContext;
use pecos_engines::scheduled_events::{
    ScheduledBatchAdapter, ScheduledEventBatch, ScheduledEventIdleZ, ScheduledEventOp,
    ScheduledGateBuffer,
};
use pecos_engines::scheduled_frame::ScheduledIdleZ;
use pecos_engines::{
    ByteMessage, ClassicalControlEngineBuilder, ClassicalEngine, ControlEngine, Engine, EngineStage,
};

use std::sync::{
    Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

type EndFn = unsafe extern "C" fn(*mut c_void) -> i32;
type StartFn = unsafe extern "C" fn(*mut c_void, u64, u64) -> i32;
static REAL_START: Mutex<Option<StartFn>> = Mutex::new(None);
static STARTS: Mutex<Vec<(u64, u64)>> = Mutex::new(Vec::new());
unsafe extern "C" fn start(instance: *mut c_void, shot: u64, seed: u64) -> i32 {
    STARTS.lock().unwrap().push((shot, seed));
    unsafe { REAL_START.lock().unwrap().unwrap()(instance, shot, seed) }
}
// The upstream repr(C) handle is non_exhaustive; the callback uses the local
// ABI mirror, as the production descriptor bridge does.
type NextFn = unsafe extern "C" fn(*mut c_void, SeleneRuntimeGetOperationHandle) -> i32;
static REAL_END: Mutex<Option<EndFn>> = Mutex::new(None);
static REAL_NEXT: Mutex<Option<NextFn>> = Mutex::new(None);
static ENDED: AtomicBool = AtomicBool::new(false);
static EMITTED: AtomicBool = AtomicBool::new(false);
static CALLS: AtomicUsize = AtomicUsize::new(0);
static BARRIER_AFTER_END: AtomicBool = AtomicBool::new(false);
static MODE: AtomicUsize = AtomicUsize::new(0);
unsafe extern "C" fn end(instance: *mut c_void) -> i32 {
    CALLS.fetch_add(1, Ordering::SeqCst);
    let errno = unsafe { REAL_END.lock().unwrap().unwrap()(instance) };
    ENDED.store(true, Ordering::SeqCst);
    errno
}
unsafe extern "C" fn next(instance: *mut c_void, out: SeleneRuntimeGetOperationHandle) -> i32 {
    if ENDED.load(Ordering::SeqCst) {
        if MODE.load(Ordering::SeqCst) == 4 {
            return 0;
        }
        if MODE.load(Ordering::SeqCst) == 2 {
            return 19;
        }
        if !EMITTED.swap(true, Ordering::SeqCst) {
            unsafe {
                (out.interface.set_batch_time)(out.instance, 123, 17);
                if MODE.load(Ordering::SeqCst) == 1 {
                    (out.interface.custom)(out.instance, 7301, std::ptr::null(), 0);
                } else if MODE.load(Ordering::SeqCst) == 3 {
                    for _ in 0..=MAX_OPERATIONS {
                        (out.interface.rxy)(out.instance, 0, 0.0, 0.0);
                    }
                } else {
                    (out.interface.rxy)(out.instance, 0, std::f64::consts::PI, 0.0);
                }
            }
        }
        return 0;
    }
    unsafe { REAL_NEXT.lock().unwrap().unwrap()(instance, out) }
}
unsafe extern "C" fn barrier(instance: *mut c_void, duration: u64) -> i32 {
    if ENDED.load(Ordering::SeqCst) {
        BARRIER_AFTER_END.store(true, Ordering::SeqCst);
    }
    unsafe { REAL_BARRIER.lock().unwrap().unwrap()(instance, duration) }
}
type BarrierFn = unsafe extern "C" fn(*mut c_void, u64) -> i32;
static REAL_BARRIER: Mutex<Option<BarrierFn>> = Mutex::new(None);

fn install_descriptor() -> (
    libloading::Library,
    libloading::Library,
    Box<RuntimePluginDescriptorV1>,
    std::path::PathBuf,
    SeleneRuntime,
) {
    let executable = std::env::current_exe().unwrap();
    let path = crate::selene_runtimes::find_library_in_dir(
        executable.parent().unwrap(),
        pecos_qis_test_runtime::LIBRARY_NAME,
    )
    .unwrap();
    let public = crate::selene_runtimes::selene_simple_runtime().unwrap();
    let library = unsafe { libloading::Library::new(&public.plugin_path).unwrap() };
    let mut descriptor =
        Box::new(unsafe { SeleneRuntime::runtime_plugin_descriptor(&library).unwrap() });
    *REAL_END.lock().unwrap() = Some(descriptor.shot_end_fn);
    *REAL_START.lock().unwrap() = Some(descriptor.shot_start_fn);
    descriptor.shot_start_fn = start;
    let real_next: NextFn = unsafe { std::mem::transmute(descriptor.get_next_operations_fn) };
    *REAL_NEXT.lock().unwrap() = Some(real_next);
    *REAL_BARRIER.lock().unwrap() = Some(descriptor.global_barrier_fn);
    descriptor.shot_end_fn = end;
    descriptor.get_next_operations_fn = unsafe {
        std::mem::transmute::<
            NextFn,
            unsafe extern "C" fn(
                *mut c_void,
                selene_core::operation::plugin::RuntimeGetOperationHandle,
            ) -> i32,
        >(next)
    };
    descriptor.global_barrier_fn = barrier;
    let fixture = unsafe { libloading::Library::new(&path).unwrap() };
    unsafe {
        let set = fixture
            .get::<unsafe extern "C" fn(*mut c_void)>(b"set_descriptor")
            .unwrap();
        set((&raw mut *descriptor).cast());
    }
    (library, fixture, descriptor, path, public)
}

fn reset_shim(mode: usize) {
    STARTS.lock().unwrap().clear();
    ENDED.store(false, Ordering::SeqCst);
    EMITTED.store(false, Ordering::SeqCst);
    CALLS.store(0, Ordering::SeqCst);
    BARRIER_AFTER_END.store(false, Ordering::SeqCst);
    MODE.store(mode, Ordering::SeqCst);
}

#[test]
fn native_end_tail_preserves_identity_timing_and_strict_rejection() {
    const CHILD: &str = "PECOS_TEST_NATIVE_END_TAIL";
    if std::env::var_os(CHILD).is_none() {
        crate::test_env::run_test_in_child(
            "selene_runtime::terminal_tests::native_end_tail_preserves_identity_timing_and_strict_rejection",
            &[(CHILD, "1".as_ref())],
        );
        return;
    }
    let (_library, _fixture, _descriptor, path, public) = install_descriptor();
    for scheduled in [false, true] {
        for (mode, legacy) in (0..4).map(|mode| (mode, false)).chain([(0, true)]) {
            let mut runtime = SeleneRuntime::new(&path);
            runtime.init_args.clone_from(&public.init_args);
            runtime.set_num_qubits(1);
            for shot in [73, 74] {
                runtime.reset().unwrap();
                reset_shim(mode);
                runtime.shot_start(shot, Some(9)).unwrap();
                let operations = [Operation::AllocateQubit { id: 0 }];
                if scheduled {
                    runtime.lower_scheduled_operations(&operations).unwrap();
                    runtime.drain_pending_scheduled_operations().unwrap();
                } else {
                    runtime.lower_operations(&operations).unwrap();
                    runtime.drain_pending_operations().unwrap();
                }
                let index = runtime.runtime_batch_index;
                let result = if legacy {
                    runtime.shot_end().map(|_| ())
                } else if scheduled {
                    runtime.finish_scheduled_operations().map(|batches| {
                        assert_eq!(batches.len(), 1);
                        let batch = &batches[0];
                        assert_eq!((batch.runtime_shot_id, batch.batch_index), (shot, index));
                        assert_eq!((batch.start_time_nanos, batch.duration_nanos), (123, 17));
                        assert!(matches!(
                            batch.operations.as_slice(),
                            [RuntimeScheduledOp::Rxy { qubit_id: 0, .. }]
                        ));
                    })
                } else {
                    runtime.finish_operations().map(|ops| {
                        assert!(ops.iter().any(|op| matches!(op.op, QuantumOp::RXY(..))));
                    })
                };
                assert_eq!(result.is_err(), mode != 0 || legacy);
                if mode == 3 {
                    // Must fail on the extraction bound, not on any later check.
                    let message = result.as_ref().unwrap_err().to_string();
                    assert!(message.contains("budget exceeded"), "{message}");
                }
                assert_eq!(CALLS.load(Ordering::SeqCst), 1);
                if mode != 0 || legacy {
                    assert!(runtime.shot_end().is_err());
                    assert!(runtime.clone().shot_end().is_err());
                } else {
                    // No native shot remains, so a repeated end releases nothing.
                    assert!(runtime.shot_end().is_ok());
                }
                assert_eq!(CALLS.load(Ordering::SeqCst), 1);
                assert!(!BARRIER_AFTER_END.load(Ordering::SeqCst));
            }
        }
    }
}

struct GatesOnly;
impl ScheduledBatchAdapter for GatesOnly {
    fn validate(&self, batch: &ScheduledEventBatch) -> std::result::Result<(), PecosError> {
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
    ) -> std::result::Result<(), PecosError> {
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

fn quantum_system(transport: ScheduledTransport) -> QuantumSystem {
    let profile = ScheduledIdleZ::new(1, 0.0, 0.0, 0.0).unwrap();
    let noise = match transport {
        ScheduledTransport::Off => {
            Box::new(pecos_engines::noise::PassThroughNoiseModel::builder().build())
                as Box<dyn pecos_engines::noise::NoiseModel>
        }
        ScheduledTransport::V6 => profile.into_noise_model().unwrap(),
        ScheduledTransport::V7 => ScheduledEventIdleZ::new(profile, |_| Ok(Box::new(GatesOnly)))
            .into_noise_model()
            .unwrap(),
    };
    QuantumSystem::new(noise, Box::new(pecos_engines::StateVecEngine::new(1)))
}

fn finish_engine_shot(
    engine: &mut crate::QisEngine,
    quantum: &mut QuantumSystem,
) -> std::result::Result<(), PecosError> {
    let mut stage = engine.start(())?;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    loop {
        assert!(std::time::Instant::now() < deadline, "shot did not finish");
        match stage {
            EngineStage::NeedsProcessing(commands) => {
                let reply = quantum.process(commands).unwrap();
                stage = engine.continue_processing(reply)?;
            }
            EngineStage::Complete(shot) => {
                assert_eq!(
                    shot.data.get("measurement_0"),
                    Some(&pecos_engines::shot_results::Data::U32(1))
                );
                return Ok(());
            }
        }
    }
}

fn assert_completed_observable(
    quantum: &mut QuantumSystem,
    trace: &crate::OperationTraceStore,
    transport: ScheduledTransport,
    mode: usize,
) {
    let values = quantum
        .quantum_engine_mut()
        .process(ByteMessage::quantum_operations_builder().mz(&[0]).build())
        .unwrap();
    assert_eq!(
        values.outcomes().unwrap(),
        vec![u32::from(mode == 4)],
        "{transport:?}, mode {mode}"
    );
    if transport == ScheduledTransport::Off {
        let trace = trace.lock().unwrap();
        assert_eq!(
            trace
                .iter()
                .filter(|chunk| chunk.stage == "shot_end")
                .count(),
            usize::from(mode == 0)
        );
        assert_eq!(trace.last().unwrap().stage, "trace_complete");
        assert_eq!(
            trace
                .iter()
                .filter(|chunk| chunk.stage == "trace_complete")
                .count(),
            1
        );
    }
}

#[test]
fn native_end_tail_executes_before_engine_completion() {
    const CHILD: &str = "PECOS_TEST_NATIVE_END_TAIL_ENGINE";
    if std::env::var_os(CHILD).is_none() {
        crate::test_env::run_test_in_child(
            "selene_runtime::terminal_tests::native_end_tail_executes_before_engine_completion",
            &[(CHILD, "1".as_ref())],
        );
        return;
    }
    let _env_lock = crate::test_env::ENV_MUTEX.lock().unwrap();
    let (_library, _fixture, _descriptor, path, public) = install_descriptor();
    let program = pecos_programs::Qis::from_string(
        r#"
        define i64 @qmain(i64 %shot) #0 {
            ; Guppy RNG(42) uses this stream, independently of runtime IDs/seeds.
            call void @random_seed(i64 42)
            %random = call i32 @random_int()
            %valid = icmp eq i32 %random, 1085446021
            br i1 %valid, label %gates, label %bad_rng
        bad_rng:
            ret i64 77
        gates:
            call void @__quantum__qis__x__body(i64 0)
            %m = call i32 @__quantum__qis__m__body(i64 0, i64 0)
            ret i64 0
        }
        declare void @random_seed(i64)
        declare i32 @random_int()
        declare void @__quantum__qis__x__body(i64)
        declare i32 @__quantum__qis__m__body(i64, i64)
        attributes #0 = { "EntryPoint" }
        "#,
    );
    for transport in [
        ScheduledTransport::Off,
        ScheduledTransport::V6,
        ScheduledTransport::V7,
    ] {
        let mut runtime = SeleneRuntime::new(&path);
        runtime.init_args.clone_from(&public.init_args);
        // The only program qubit is allocated first, so native slot 0 is qubit 0.
        runtime.set_num_qubits(1);
        let trace = crate::OperationTraceStore::default();
        let mut builder = crate::qis_engine()
            .interface(crate::helios_interface_builder())
            .program(program.clone())
            .runtime(runtime);
        builder = match transport {
            ScheduledTransport::Off => builder.trace_operations_in_memory_to(Arc::clone(&trace)),
            ScheduledTransport::V6 => builder.scheduled_batches(true),
            ScheduledTransport::V7 => builder.scheduled_event_batches(true),
        };
        let mut engine = builder.build().unwrap();
        engine.set_seed(123);
        let mut expected_rng = pecos_random::PecosRng::seed_from_u64(123);
        let mut quantum = quantum_system(transport);
        for (shot, mode) in [0, 0, 4, 4, 1, 0, 2, 0].into_iter().enumerate() {
            reset_shim(mode);
            trace.lock().unwrap().clear();
            quantum.reset().unwrap();
            quantum
                .begin_shot(ShotContext {
                    run: 1,
                    worker: 0,
                    shot,
                })
                .unwrap();
            let result = finish_engine_shot(&mut engine, &mut quantum);
            assert_eq!(
                *STARTS.lock().unwrap(),
                [(u64::try_from(shot).unwrap(), expected_rng.next_u64())],
                "{transport:?}, mode {mode}: native ID and seed",
            );
            assert_eq!(
                CALLS.load(Ordering::SeqCst),
                1,
                "{transport:?}, mode {mode}"
            );
            let repeated = engine.continue_processing(ByteMessage::outcomes_builder().build());
            if mode == 1 || mode == 2 {
                // Pin the cause: the shim's tail must be what failed the shot.
                let cause = if mode == 1 {
                    "custom event"
                } else {
                    "errno 19"
                };
                let message = result.unwrap_err().to_string();
                assert!(message.contains(cause), "{transport:?}: {message}");
                assert!(repeated.is_err());
            } else {
                result.unwrap();
                assert!(matches!(repeated.unwrap(), EngineStage::Complete(_)));
                assert_completed_observable(&mut quantum, &trace, transport, mode);
            }
            assert_eq!(CALLS.load(Ordering::SeqCst), 1);
            assert!(!BARRIER_AFTER_END.load(Ordering::SeqCst));
        }
    }
}
