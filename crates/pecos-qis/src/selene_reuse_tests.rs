use super::tests::LoweringRoute;
use super::*;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

static REAL: Mutex<Option<RuntimePluginDescriptorV1>> = Mutex::new(None);
static INITS: Mutex<Vec<(usize, u64)>> = Mutex::new(Vec::new());
static STARTS: Mutex<Vec<(usize, u64, u64)>> = Mutex::new(Vec::new());
static ENDS: Mutex<Vec<usize>> = Mutex::new(Vec::new());
static EXITS: Mutex<Vec<usize>> = Mutex::new(Vec::new());
static EMITTED: AtomicBool = AtomicBool::new(false);
static POLLS: AtomicUsize = AtomicUsize::new(0);
static BARRIERS: AtomicUsize = AtomicUsize::new(0);

unsafe extern "C" fn recording_next(
    instance: *mut c_void,
    out: SeleneRuntimeGetOperationHandle,
) -> i32 {
    POLLS.fetch_add(1, Ordering::SeqCst);
    let next: unsafe extern "C" fn(*mut c_void, SeleneRuntimeGetOperationHandle) -> i32 =
        unsafe { std::mem::transmute(REAL.lock().unwrap().unwrap().get_next_operations_fn) };
    unsafe { next(instance, out) }
}

unsafe extern "C" fn recording_barrier(instance: *mut c_void, duration: u64) -> i32 {
    BARRIERS.fetch_add(1, Ordering::SeqCst);
    unsafe { (REAL.lock().unwrap().unwrap().global_barrier_fn)(instance, duration) }
}

unsafe extern "C" fn recording_init(
    instance: *mut *mut c_void,
    capacity: u64,
    start: u64,
    argc: u32,
    argv: *const *const std::ffi::c_char,
) -> i32 {
    let init = REAL.lock().unwrap().unwrap().init_fn;
    let errno = unsafe { init(instance, capacity, start, argc, argv) };
    INITS
        .lock()
        .unwrap()
        .push((unsafe { *instance } as usize, capacity));
    errno
}

unsafe extern "C" fn recording_start(instance: *mut c_void, shot: u64, seed: u64) -> i32 {
    STARTS.lock().unwrap().push((instance as usize, shot, seed));
    EMITTED.store(false, Ordering::SeqCst);
    unsafe { (REAL.lock().unwrap().unwrap().shot_start_fn)(instance, shot, seed) }
}

unsafe extern "C" fn recording_end(instance: *mut c_void) -> i32 {
    ENDS.lock().unwrap().push(instance as usize);
    unsafe { (REAL.lock().unwrap().unwrap().shot_end_fn)(instance) }
}

unsafe extern "C" fn recording_exit(instance: *mut c_void) -> i32 {
    EXITS.lock().unwrap().push(instance as usize);
    unsafe { (REAL.lock().unwrap().unwrap().exit_fn.unwrap())(instance) }
}

unsafe extern "C" fn clock_behind(_: *mut c_void, out: SeleneRuntimeGetOperationHandle) -> i32 {
    if !EMITTED.swap(true, Ordering::SeqCst) {
        unsafe {
            (out.interface.set_batch_time)(out.instance, 0, 10);
            (out.interface.rz)(out.instance, 0, 0.25);
        }
    }
    0
}

fn fixture(test: &str, env: &str, behind: bool, run: impl FnOnce(&Path, &SeleneRuntime)) {
    fixture_with_runtime(
        test,
        env,
        behind,
        &crate::selene_runtimes::selene_simple_runtime().unwrap(),
        run,
    );
}

fn fixture_with_runtime(
    test: &str,
    env: &str,
    behind: bool,
    public: &SeleneRuntime,
    run: impl FnOnce(&Path, &SeleneRuntime),
) {
    if std::env::var_os(env).is_none() {
        crate::test_env::run_test_in_child(
            &format!("selene_runtime::reuse_tests::{test}"),
            &[(env, "1".as_ref())],
        );
        return;
    }
    let executable = std::env::current_exe().unwrap();
    let plugin = crate::selene_runtimes::find_library_in_dir(
        executable.parent().unwrap(),
        pecos_qis_test_runtime::LIBRARY_NAME,
    )
    .unwrap();
    let library = unsafe { libloading::Library::new(&public.plugin_path).unwrap() };
    let mut descriptor =
        Box::new(unsafe { SeleneRuntime::runtime_plugin_descriptor(&library).unwrap() });
    *REAL.lock().unwrap() = Some(*descriptor);
    descriptor.init_fn = recording_init;
    descriptor.shot_start_fn = recording_start;
    descriptor.shot_end_fn = recording_end;
    descriptor.exit_fn = Some(recording_exit);
    descriptor.global_barrier_fn = recording_barrier;
    descriptor.get_next_operations_fn = unsafe {
        std::mem::transmute::<
            unsafe extern "C" fn(*mut c_void, SeleneRuntimeGetOperationHandle) -> i32,
            unsafe extern "C" fn(
                *mut c_void,
                selene_core::runtime::plugin::RuntimeGetOperationHandle,
            ) -> i32,
        >(if behind { clock_behind } else { recording_next })
    };
    let fixture = unsafe { libloading::Library::new(&plugin).unwrap() };
    unsafe {
        fixture
            .get::<unsafe extern "C" fn(*mut c_void)>(b"set_descriptor")
            .unwrap()((&raw mut *descriptor).cast());
    }
    run(&plugin, public);
}

fn runtime(plugin: &Path, public: &SeleneRuntime) -> SeleneRuntime {
    INITS.lock().unwrap().clear();
    STARTS.lock().unwrap().clear();
    ENDS.lock().unwrap().clear();
    EXITS.lock().unwrap().clear();
    SeleneRuntime::with_plugin_config(plugin, public.init_args.clone(), Vec::new())
}

fn clean_shot(runtime: &mut SeleneRuntime, route: LoweringRoute, shot: u64, capacity: usize) {
    runtime.reset().unwrap();
    runtime.shot_start(shot, Some(shot + 100)).unwrap();
    runtime.set_num_qubits(capacity);
    route
        .lower(runtime, &[QuantumOp::RZ(0.25, capacity - 1).into()])
        .unwrap();
    route.drain(runtime);
    runtime.shot_end().unwrap();
}

#[test]
fn clean_shots_initialize_once() {
    fixture(
        "clean_shots_initialize_once",
        "PECOS_REUSE_INIT",
        false,
        |plugin, public| {
            for route in LoweringRoute::ALL {
                let mut runtime = runtime(plugin, public);
                for shot in 0..3 {
                    clean_shot(&mut runtime, route, shot, 2);
                }
                assert_eq!(INITS.lock().unwrap().len(), 1, "{route:?}");
                assert!(EXITS.lock().unwrap().is_empty());
                let instance = INITS.lock().unwrap()[0].0;
                assert_eq!(
                    *STARTS.lock().unwrap(),
                    vec![(instance, 0, 100), (instance, 1, 101), (instance, 2, 102)]
                );
                assert_eq!(*ENDS.lock().unwrap(), vec![instance; 3]);
            }
        },
    );
}

fn timed_runtime() -> SeleneRuntime {
    let public = crate::selene_runtimes::selene_simple_runtime().unwrap();
    SeleneRuntime::with_plugin_config(
        public.plugin_path.clone(),
        [
            ("rxy", 10),
            ("rzz", 20),
            ("rz", 5),
            ("rpp", 7),
            ("measure", 30),
            ("reset", 3),
            ("measure-leaked", 30),
        ]
        .map(|(gate, duration)| format!("--duration-ns-{gate}={duration}"))
        .into(),
        Vec::new(),
    )
}

#[derive(Debug, PartialEq)]
enum Output {
    Flat(Vec<QuantumOp>),
    Scheduled(Vec<(u64, u64, Vec<RuntimeScheduledOp>)>),
}

fn timed_shot(runtime: &mut SeleneRuntime, route: LoweringRoute, shot: u64) -> Output {
    runtime.set_num_qubits(2);
    runtime.shot_start(shot, Some(19)).unwrap();
    let ops = [
        QuantumOp::RXY(0.5, 0.25, 0).into(),
        QuantumOp::RZ(0.25, 1).into(),
        QuantumOp::RZZ(0.5, 0, 1).into(),
        QuantumOp::Reset(0).into(),
        QuantumOp::Measure(1, 0).into(),
    ];
    let output = match route {
        LoweringRoute::Flat => Output::Flat(runtime.lower_operations(&ops).unwrap()),
        LoweringRoute::Metadata => Output::Flat(
            runtime
                .lower_operations_with_metadata(&ops)
                .unwrap()
                .into_iter()
                .map(|op| op.op)
                .collect(),
        ),
        LoweringRoute::Scheduled => Output::Scheduled(
            runtime
                .lower_scheduled_operations(&ops)
                .unwrap()
                .into_iter()
                .map(|b| (b.start_time_nanos, b.duration_nanos, b.operations))
                .collect(),
        ),
    };
    runtime
        .provide_measurements(BTreeMap::from([(0, true)]))
        .unwrap();
    route.drain(runtime);
    runtime.shot_end().unwrap();
    output
}

#[test]
fn repeated_shots_have_fresh_timing() {
    fixture(
        "repeated_shots_have_fresh_timing",
        "PECOS_REUSE_TIMING",
        false,
        |plugin, public| {
            for route in LoweringRoute::ALL {
                let fresh = timed_shot(&mut timed_runtime(), route, 0);
                if let Output::Flat(ops) = &fresh {
                    assert!(ops.iter().any(|op| matches!(op, QuantumOp::Idle(10e-9, 1))));
                }
                for reset in [true, false] {
                    let mut runtime = runtime(plugin, public);
                    runtime.init_args = timed_runtime().init_args.clone();
                    for shot in 0..3 {
                        if reset {
                            runtime.reset().unwrap();
                        }
                        assert_eq!(
                            timed_shot(&mut runtime, route, shot),
                            fresh,
                            "{route:?}, reset={reset}, shot={shot}"
                        );
                        assert_eq!(INITS.lock().unwrap().len(), 1, "{route:?}, reset={reset}");
                    }
                }
            }
        },
    );
}

#[test]
fn clock_behind_origin_is_rejected() {
    fixture(
        "clock_behind_origin_is_rejected",
        "PECOS_REUSE_CLOCK",
        true,
        |plugin, public| {
            for route in LoweringRoute::ALL {
                let mut runtime = runtime(plugin, public);
                clean_shot(&mut runtime, route, 0, 1);
                runtime.reset().unwrap();
                runtime.shot_start(1, None).unwrap();
                let error = route
                    .lower(&mut runtime, &[QuantumOp::RZ(0.25, 0).into()])
                    .unwrap_err();
                assert_eq!(
                    error.to_string(),
                    format!(
                        "Execution error: runtime {} batch start 0 is before shot origin 10: runtime clock is behind the clock carried from the previous shot; reset required",
                        plugin.display()
                    )
                );
            }
        },
    );
}

#[test]
fn capacity_replacement_transfers_eager_shot_start() {
    fixture(
        "capacity_replacement_transfers_eager_shot_start",
        "PECOS_REUSE_CAPACITY",
        false,
        |plugin, public| {
            for route in LoweringRoute::ALL {
                for capacity in [1, 3] {
                    if capacity == 1 && !matches!(route, LoweringRoute::Scheduled) {
                        continue;
                    }
                    let mut runtime = runtime(plugin, public);
                    clean_shot(&mut runtime, route, 10, 2);
                    clean_shot(&mut runtime, route, 11, capacity);
                    let inits = INITS.lock().unwrap().clone();
                    assert_eq!(
                        inits.iter().map(|(_, n)| *n).collect::<Vec<_>>(),
                        vec![2, capacity as u64]
                    );
                    assert_eq!(
                        *STARTS.lock().unwrap(),
                        vec![
                            (inits[0].0, 10, 110),
                            (inits[0].0, 11, 111),
                            (inits[1].0, 11, 111)
                        ]
                    );
                    assert_eq!(*ENDS.lock().unwrap(), vec![inits[0].0, inits[1].0]);
                    assert_eq!(*EXITS.lock().unwrap(), vec![inits[0].0]);
                }
            }
        },
    );
}

#[test]
fn unclean_reset_exits_and_reinitializes() {
    fixture(
        "unclean_reset_exits_and_reinitializes",
        "PECOS_REUSE_ABORT",
        false,
        |plugin, public| {
            for route in LoweringRoute::ALL {
                let mut runtime = runtime(plugin, public);
                runtime.set_num_qubits(1);
                runtime.shot_start(0, None).unwrap();
                route
                    .lower(&mut runtime, &[Operation::AllocateQubit { id: 0 }])
                    .unwrap();
                clean_shot(&mut runtime, route, 1, 1);
                let inits = INITS.lock().unwrap().clone();
                assert_eq!(inits.len(), 2);
                assert_ne!(inits[0].0, inits[1].0);
                assert_eq!(*EXITS.lock().unwrap(), vec![inits[0].0]);
                assert_eq!(*ENDS.lock().unwrap(), vec![inits[1].0]);
            }
        },
    );
}

#[test]
fn empty_clean_shot_preserves_empty_output() {
    for route in LoweringRoute::ALL {
        let mut runtime = timed_runtime();
        clean_shot(&mut runtime, route, 0, 2);
        runtime.reset().unwrap();
        runtime.num_qubits_hint = None;
        runtime.shot_start(1, None).unwrap();
        if matches!(route, LoweringRoute::Scheduled) {
            assert_eq!(
                runtime.finish_scheduled_operations().unwrap(),
                Vec::<ScheduledBatch>::new()
            );
        } else {
            assert_eq!(
                runtime.lower_operations(&[]).unwrap(),
                Vec::<QuantumOp>::new()
            );
            assert_eq!(
                runtime.drain_pending_operations().unwrap(),
                Vec::<QuantumOp>::new()
            );
            assert_eq!(
                runtime.finish_operations().unwrap(),
                Vec::<LoweredQuantumOp>::new()
            );
        }
        assert!(runtime.shot_end().unwrap().measurements.is_empty());
    }
}

#[test]
fn empty_input_preserves_shot_engagement() {
    fixture(
        "empty_input_preserves_shot_engagement",
        "PECOS_REUSE_ENGAGEMENT",
        false,
        |plugin, public| {
            for kept in [false, true] {
                for engaged in [false, true] {
                    for path in 0..4 {
                        let mut runtime = runtime(plugin, public);
                        if kept {
                            clean_shot(&mut runtime, LoweringRoute::Flat, 0, 1);
                        }
                        runtime.reset().unwrap();
                        runtime.shot_start(1, None).unwrap();
                        if engaged {
                            runtime.set_num_qubits(1);
                            runtime.prepare_runtime_input(&[]).unwrap();
                        }
                        assert!(!runtime.native_shot.work_submitted);
                        POLLS.store(0, Ordering::SeqCst);
                        BARRIERS.store(0, Ordering::SeqCst);
                        match path {
                            0 => assert_eq!(
                                runtime.drain_native_pending_operations().unwrap(),
                                Vec::<QuantumOp>::new()
                            ),
                            1 => assert_eq!(
                                runtime.lower_runtime_barrier().unwrap(),
                                Vec::<QuantumOp>::new()
                            ),
                            2 => assert_eq!(
                                runtime.drain_runtime_operations_bounded(false).unwrap(),
                                Vec::<QuantumOp>::new()
                            ),
                            3 => {
                                assert_eq!(
                                    runtime.finish_scheduled_operations().unwrap(),
                                    Vec::<ScheduledBatch>::new()
                                );
                                assert_eq!(runtime.scheduled_mode, engaged.then_some(true));
                            }
                            _ => unreachable!(),
                        }
                        assert_eq!(
                            POLLS.load(Ordering::SeqCst),
                            usize::from(engaged && path != 1)
                        );
                        assert_eq!(
                            BARRIERS.load(Ordering::SeqCst),
                            usize::from(engaged && path < 2)
                        );
                    }
                }
            }
        },
    );
}

#[test]
fn capacity_retirement_preserves_collected_shot_state() {
    fixture(
        "capacity_retirement_preserves_collected_shot_state",
        "PECOS_REUSE_BOOKKEEPING",
        true,
        |plugin, public| {
            let mut runtime = runtime(plugin, public);
            runtime.set_num_qubits(1);
            runtime.shot_start(0, None).unwrap();
            // A plugin may emit work on engagement even without any native submission.
            assert_eq!(
                runtime.lower_operations(&[]).unwrap(),
                vec![QuantumOp::RZ(0.25, 0)]
            );
            assert!(!runtime.native_shot.work_submitted);
            assert_eq!(runtime.last_gate_time_end_nanos, vec![10]);
            runtime.set_num_qubits(2);
            runtime.prepare_runtime_input(&[]).unwrap();
            assert_eq!(runtime.initialized_num_qubits, Some(2));
            assert_eq!(runtime.last_gate_time_end_nanos, vec![10]);
            assert_eq!(runtime.runtime_batch_index, 1);
        },
    );
}

#[test]
fn scheduled_capacity_replacement_on_drain_and_finish() {
    fixture(
        "scheduled_capacity_replacement_on_drain_and_finish",
        "PECOS_REUSE_FINISH_CAPACITY",
        false,
        |plugin, public| {
            for finish in [false, true] {
                let mut runtime = runtime(plugin, public);
                runtime.set_num_qubits(1);
                runtime.shot_start(7, Some(13)).unwrap();
                if finish {
                    runtime.prepare_runtime_input(&[]).unwrap();
                } else {
                    runtime.lower_scheduled_operations(&[]).unwrap();
                }
                assert!(!runtime.native_shot.work_submitted);
                runtime.set_num_qubits(2);
                if !finish {
                    runtime.drain_pending_scheduled_operations().unwrap();
                }
                runtime.finish_scheduled_operations().unwrap();
                let inits = INITS.lock().unwrap().clone();
                assert_eq!(
                    inits.iter().map(|(_, n)| *n).collect::<Vec<_>>(),
                    vec![1, 2]
                );
                assert_eq!(
                    *STARTS.lock().unwrap(),
                    vec![(inits[0].0, 7, 13), (inits[1].0, 7, 13)]
                );
                assert_eq!(*ENDS.lock().unwrap(), vec![inits[1].0]);
                assert!(runtime.active_shot.is_none());
            }
        },
    );
}

fn timed_soft_runtime(batch_size: usize) -> SeleneRuntime {
    let mut runtime = crate::selene_runtimes::selene_soft_rz_runtime().unwrap();
    runtime.init_args = vec![
        "--duration-ns-rxy=10".into(),
        "--duration-ns-rzz=20".into(),
        "--duration-ns-measure=30".into(),
        "--duration-ns-reset=3".into(),
        "--duration-ns-measure-leaked=30".into(),
        format!("--max-batch-size={batch_size}"),
    ];
    runtime
}

#[test]
fn shotless_reset_retires_unsettled_instance() {
    fixture(
        "shotless_reset_retires_unsettled_instance",
        "PECOS_REUSE_SHOTLESS_RESET",
        false,
        |plugin, public| {
            let mut runtime = runtime(plugin, public);
            runtime.set_num_qubits(1);
            let first = [
                Operation::AllocateQubit { id: 0 },
                QuantumOp::RXY(0.25, 0.0, 0).into(),
            ];
            runtime.lower_operations(&first).unwrap();
            runtime.reset().unwrap();
            let second = runtime.lower_operations(&[
                Operation::AllocateQubit { id: 0 },
                QuantumOp::RXY(0.5, 0.0, 0).into(),
            ]);
            assert!(
                second.is_ok(),
                "reset left an allocation or queued work: {second:?}"
            );
            assert_eq!(second.unwrap(), vec![QuantumOp::RXY(0.5, 0.0, 0)]);
            assert_eq!(INITS.lock().unwrap().len(), 2);
            assert_eq!(EXITS.lock().unwrap().len(), 1);
        },
    );
}

#[test]
fn undrained_finish_retires_instance_before_reuse() {
    fixture_with_runtime(
        "undrained_finish_retires_instance_before_reuse",
        "PECOS_REUSE_UNDRAINED_FINISH",
        false,
        &timed_soft_runtime(4),
        |plugin, public| {
            for reset in [true, false] {
                let mut runtime = runtime(plugin, public);
                runtime.set_num_qubits(1);
                runtime.shot_start(0, None).unwrap();
                let ops = [
                    Operation::AllocateQubit { id: 0 },
                    QuantumOp::RXY(0.25, 0.0, 0).into(),
                ];
                assert_eq!(
                    runtime.lower_operations(&ops).unwrap(),
                    Vec::<QuantumOp>::new()
                );
                assert_eq!(
                    runtime.finish_operations().unwrap(),
                    Vec::<LoweredQuantumOp>::new()
                );
                if reset {
                    runtime.reset().unwrap();
                }
                runtime.shot_start(1, None).unwrap();
                let mut actual = runtime.lower_operations(&ops).unwrap();
                actual.extend(runtime.drain_pending_operations().unwrap());
                let mut fresh = timed_soft_runtime(4);
                fresh.set_num_qubits(1);
                fresh.shot_start(1, None).unwrap();
                let mut expected = fresh.lower_operations(&ops).unwrap();
                expected.extend(fresh.drain_pending_operations().unwrap());
                assert_eq!(
                    actual, expected,
                    "the discarded batch must not shift the next shot's clock"
                );
                assert_eq!(INITS.lock().unwrap().len(), 2);
                assert_eq!(EXITS.lock().unwrap().len(), 1);
            }
        },
    );
}

#[test]
fn shotless_lowering_after_clean_reset_rebases_time() {
    let mut runtime = timed_runtime();
    runtime.set_num_qubits(1);
    runtime.shot_start(0, None).unwrap();
    let ops = [QuantumOp::RXY(0.25, 0.0, 0).into()];
    runtime.lower_operations(&ops).unwrap();
    runtime.drain_pending_operations().unwrap();
    runtime.finish_operations().unwrap();
    let instance = runtime.instance;
    runtime.reset().unwrap();
    assert_eq!(runtime.instance, instance);
    let actual = runtime.lower_operations(&ops).unwrap();
    let mut fresh = timed_runtime();
    fresh.set_num_qubits(1);
    assert_eq!(actual, fresh.lower_operations(&ops).unwrap());
}

fn collect_timed_program(runtime: &mut SeleneRuntime, scheduled: bool, two_drains: bool) -> Output {
    let first = [
        QuantumOp::RXY(0.25, 0.0, 0).into(),
        QuantumOp::RXY(0.5, 0.0, 1).into(),
    ];
    let second = [
        QuantumOp::RZZ(0.3, 0, 1).into(),
        QuantumOp::RXY(0.1, 0.0, 1).into(),
    ];
    let combined = [first.as_slice(), second.as_slice()].concat();
    let chunks = if two_drains {
        vec![first.as_slice(), second.as_slice()]
    } else {
        vec![combined.as_slice()]
    };
    let mut flat = Vec::new();
    let mut batches = Vec::new();
    for chunk in chunks {
        if scheduled {
            batches.extend(runtime.lower_scheduled_operations(chunk).unwrap());
            batches.extend(runtime.drain_pending_scheduled_operations().unwrap());
        } else {
            flat.extend(runtime.lower_operations(chunk).unwrap());
            flat.extend(runtime.drain_pending_operations().unwrap());
        }
    }
    if scheduled {
        batches.extend(runtime.finish_scheduled_operations().unwrap());
        Output::Scheduled(
            batches
                .into_iter()
                .map(|b| (b.start_time_nanos, b.duration_nanos, b.operations))
                .collect(),
        )
    } else {
        flat.extend(
            runtime
                .finish_operations()
                .unwrap()
                .into_iter()
                .map(|op| op.op),
        );
        Output::Flat(flat)
    }
}

#[test]
fn soft_rz_reused_shots_preserve_batched_timing() {
    for batch_size in [1, 4] {
        for scheduled in [false, true] {
            let mut fresh = timed_soft_runtime(batch_size);
            fresh.set_num_qubits(2);
            fresh.shot_start(0, None).unwrap();
            let expected = collect_timed_program(&mut fresh, scheduled, false);
            if let Output::Scheduled(batches) = &expected {
                assert_eq!((batches[0].0, batches[0].1), (0, 10));
                assert_eq!(batches[0].2.len(), if batch_size == 1 { 1 } else { 2 });
            }
            let mut runtime = timed_soft_runtime(batch_size);
            runtime.set_num_qubits(2);
            let mut instance = None;
            for shot in 0..3 {
                runtime.reset().unwrap();
                runtime.shot_start(shot, None).unwrap();
                assert_eq!(
                    collect_timed_program(&mut runtime, scheduled, false),
                    expected,
                    "batch size {batch_size}, scheduled={scheduled}, shot={shot}"
                );
                assert_eq!(*instance.get_or_insert(runtime.instance), runtime.instance);
            }
        }
    }
}

#[test]
fn two_drains_per_shot_preserve_timing() {
    for variant in 0..3 {
        let make_runtime = || match variant {
            0 => timed_runtime(),
            1 => timed_soft_runtime(1),
            _ => timed_soft_runtime(4),
        };
        for scheduled in [false, true] {
            let mut fresh = make_runtime();
            fresh.set_num_qubits(2);
            fresh.shot_start(0, None).unwrap();
            let expected = collect_timed_program(&mut fresh, scheduled, true);
            let mut runtime = make_runtime();
            runtime.set_num_qubits(2);
            let mut instance = None;
            for shot in 0..3 {
                runtime.reset().unwrap();
                runtime.shot_start(shot, None).unwrap();
                assert_eq!(
                    collect_timed_program(&mut runtime, scheduled, true),
                    expected,
                    "variant {variant}, scheduled={scheduled}, shot={shot}"
                );
                assert_eq!(*instance.get_or_insert(runtime.instance), runtime.instance);
            }
        }
    }
}

#[test]
fn batch_adapters_restart_native_clocks_with_their_shot_context() {
    struct Identity;
    impl RuntimeBatchAdapter for Identity {
        fn normalize(&mut self, batch: &RuntimeBatch) -> Result<Vec<RuntimeScheduledOp>> {
            Ok(batch.operations.clone())
        }
    }
    fixture(
        "batch_adapters_restart_native_clocks_with_their_shot_context",
        "PECOS_ADAPTER_FRESH_CLOCK",
        true,
        |plugin, public| {
            for explicit_reset in [false, true] {
                let mut runtime = runtime(plugin, public);
                runtime.set_batch_adapter_factory(|| Ok(Box::new(Identity)));
                runtime.set_num_qubits(1);
                for shot in 0..3 {
                    if explicit_reset {
                        runtime.reset().unwrap();
                    }
                    runtime.shot_start(shot, None).unwrap();
                    let ops = runtime
                        .lower_operations(&[QuantumOp::RZ(0.25, 0).into()])
                        .unwrap();
                    assert_eq!(ops, [QuantumOp::RZ(0.25, 0)]);
                    assert!(runtime.drain_pending_operations().unwrap().is_empty());
                    runtime.shot_end().unwrap();
                }
                assert_eq!(INITS.lock().unwrap().len(), 3);
                assert_eq!(EXITS.lock().unwrap().len(), 2);
            }
        },
    );
}
