//! A native descriptor that emits its only extra batch from `shot_end`.
use super::*;
use std::sync::{
    Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

type EndFn = unsafe extern "C" fn(*mut c_void) -> i32;
// The upstream repr(C) handle is non_exhaustive; the local ABI mirror is used
// by the callback, as in the production descriptor bridge.
#[allow(improper_ctypes_definitions)]
type NativeNextFn = unsafe extern "C" fn(
    *mut c_void,
    selene_core::operation::plugin::RuntimeGetOperationHandle,
) -> i32;
type NextFn = unsafe extern "C" fn(*mut c_void, SeleneRuntimeGetOperationHandle) -> i32;
static REAL_END: Mutex<Option<EndFn>> = Mutex::new(None);
static REAL_NEXT: Mutex<Option<NextFn>> = Mutex::new(None);
static ENDED: AtomicBool = AtomicBool::new(false);
static EMITTED: AtomicBool = AtomicBool::new(false);
static CALLS: AtomicUsize = AtomicUsize::new(0);
static MODE: AtomicUsize = AtomicUsize::new(0);
unsafe extern "C" fn end(instance: *mut c_void) -> i32 {
    CALLS.fetch_add(1, Ordering::SeqCst);
    let errno = unsafe { REAL_END.lock().unwrap().unwrap()(instance) };
    ENDED.store(true, Ordering::SeqCst);
    errno
}
unsafe extern "C" fn next(instance: *mut c_void, out: SeleneRuntimeGetOperationHandle) -> i32 {
    if ENDED.load(Ordering::SeqCst) {
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
    assert!(
        !ENDED.load(Ordering::SeqCst),
        "barrier submitted after native end"
    );
    unsafe { REAL_BARRIER.lock().unwrap().unwrap()(instance, duration) }
}
type BarrierFn = unsafe extern "C" fn(*mut c_void, u64) -> i32;
static REAL_BARRIER: Mutex<Option<BarrierFn>> = Mutex::new(None);

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
    *REAL_NEXT.lock().unwrap() = Some(unsafe {
        std::mem::transmute::<NativeNextFn, NextFn>(descriptor.get_next_operations_fn)
    });
    *REAL_BARRIER.lock().unwrap() = Some(descriptor.global_barrier_fn);
    descriptor.shot_end_fn = end;
    descriptor.get_next_operations_fn =
        unsafe { std::mem::transmute::<NextFn, NativeNextFn>(next) };
    descriptor.global_barrier_fn = barrier;
    let fixture = unsafe { libloading::Library::new(&path).unwrap() };
    unsafe {
        let set = fixture
            .get::<unsafe extern "C" fn(*mut c_void)>(b"set_descriptor")
            .unwrap();
        set((&raw mut *descriptor).cast());
    }
    for scheduled in [false, true] {
        for (mode, legacy) in (0..4).map(|mode| (mode, false)).chain([(0, true)]) {
            let mut runtime = SeleneRuntime::new(&path);
            runtime.init_args.clone_from(&public.init_args);
            runtime.set_num_qubits(1);
            for shot in [73, 74] {
                runtime.reset().unwrap();
                ENDED.store(false, Ordering::SeqCst);
                EMITTED.store(false, Ordering::SeqCst);
                CALLS.store(0, Ordering::SeqCst);
                MODE.store(mode, Ordering::SeqCst);
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
                assert_eq!(CALLS.load(Ordering::SeqCst), 1);
                if mode != 0 || legacy {
                    assert!(runtime.shot_end().is_err());
                    assert!(runtime.clone().shot_end().is_err());
                    assert_eq!(CALLS.load(Ordering::SeqCst), 1);
                }
            }
        }
    }
}
