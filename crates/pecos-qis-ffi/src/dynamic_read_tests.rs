//! Dynamic reader contracts, isolated from other tests' TLS and contexts.
use crate::ffi::*;
use crate::*;
use std::sync::mpsc;
use std::time::Duration;

struct Context(*mut ExecutionContext);
impl Context {
    fn new() -> Self {
        let context = Self(pecos_create_execution_context());
        unsafe { pecos_register_execution_context(context.0) };
        reset_interface();
        clear_quantum_executor();
        pecos_enable_dynamic_mode();
        TRANSFERS.set(0);
        unsafe { pecos_set_program_panic_handler(Some(inspect_transfer)) };
        context
    }
    fn get(&self) -> &ExecutionContext {
        unsafe { &*self.0 }
    }
    fn error(&self) -> ProgramError {
        self.get().program_error.lock().unwrap().clone().unwrap()
    }
}
impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            pecos_set_program_panic_handler(None);
            pecos_register_execution_context(std::ptr::null_mut());
            pecos_destroy_execution_context(self.0);
        }
    }
}

thread_local! { static TRANSFERS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }
unsafe extern "C-unwind" fn inspect_transfer() {
    let ctx = unsafe { &*get_execution_context().unwrap() };
    assert!(!matches!(
        ctx.sync_state.try_lock(),
        Err(std::sync::TryLockError::WouldBlock)
    ));
    assert!(ctx.program_error.try_lock().is_ok());
    // Poisoned mutexes still own their guard in the error. Drop it here.
    assert!(!matches!(
        ctx.pending_ops.try_lock(),
        Err(std::sync::TryLockError::WouldBlock)
    ));
    assert!(!matches!(
        ctx.measurement_results.try_lock(),
        Err(std::sync::TryLockError::WouldBlock)
    ));
    assert!(ctx.measured_results.try_lock().is_ok());
    assert!(matches!(
        *ctx.program_error.lock().unwrap(),
        Some(ProgramError::ResultUnavailable { .. })
    ));
    with_interface(|_| ());
    EXECUTOR.with(|executor| assert!(executor.try_borrow_mut().is_ok()));
    TRANSFERS.set(TRANSFERS.get() + 1);
}

type Reader = fn(i64) -> u64;
fn readers() -> [Reader; 5] {
    [
        |id| u64::from(unsafe { ___read_future_bool(id) }),
        |id| unsafe { ___read_future_uint(id) },
        |id| u64::try_from(unsafe { __quantum__rt__result_get_one(id) }).unwrap(),
        |id| u64::from(unsafe { selene::selene_future_read_bool(std::ptr::null_mut(), id.cast_unsigned()) }.value),
        |id| unsafe { selene::selene_future_read_u64(std::ptr::null_mut(), id.cast_unsigned()) }.value,
    ]
}

#[test]
fn never_measured_fails_without_publishing_a_request() {
    if !test_env::run_test_in_child(
        "dynamic_read_tests::never_measured_fails_without_publishing_a_request",
    ) {
        return;
    }
    for reader in readers() {
        let ctx = Context::new();
        assert_eq!(reader(7), 0);
        let expected = ProgramError::ResultUnavailable {
            result_id: 7,
            reason: "never measured".into(),
        };
        assert_eq!(ctx.error(), expected);
        assert_eq!(
            expected.to_string(),
            "QIS measurement result 7 unavailable: never measured"
        );
        assert_eq!(pecos_check_need_result(), u64::MAX);
        assert!(!ctx.get().sync_state.lock().unwrap().need_result);
        assert_eq!(TRANSFERS.get(), 1);
        assert_eq!(COLLECTION_MODE_READ_COUNT.get(), 0);
    }
}

/// Run a reader on its own thread; the host responds only after observing `need_result`.
fn host_read(
    reader: Reader,
    outcome: Option<u64>,
    result_id: u64,
) -> (u64, usize, u32, ProgramError) {
    let ctx = Context::new();
    let address = ctx.0 as usize;
    let worker = std::thread::spawn(move || {
        unsafe {
            pecos_register_execution_context(address as *mut ExecutionContext);
            pecos_set_program_panic_handler(Some(inspect_transfer));
            __quantum__qis__m__body(0, result_id.cast_signed());
        }
        let value = reader(result_id.cast_signed());
        let observation = (value, TRANSFERS.get(), COLLECTION_MODE_READ_COUNT.get());
        unsafe {
            pecos_set_program_panic_handler(None);
            pecos_register_execution_context(std::ptr::null_mut());
        }
        observation
    });
    assert_eq!(pecos_wait_for_need_result(2_000), result_id);
    if let Some(outcome) = outcome {
        pecos_set_measurement_outcome(result_id, outcome);
    }
    pecos_signal_result_ready();
    let (value, transfers, counter) = worker.join().unwrap();
    (value, transfers, counter, ctx.error())
}

#[test]
fn every_reader_rejects_ready_without_an_outcome() {
    if !test_env::run_test_in_child(
        "dynamic_read_tests::every_reader_rejects_ready_without_an_outcome",
    ) {
        return;
    }
    for reader in readers() {
        let (value, transfers, counter, error) = host_read(reader, None, 7);
        assert_eq!((value, transfers, counter), (0, 1, 0));
        assert_eq!(
            error,
            ProgramError::ResultUnavailable {
                result_id: 7,
                reason: "ready without a stored outcome".into()
            }
        );
    }
    let (_, transfers, counter, error) = host_read(
        |_| u64::from(unsafe { selene::selene_qubit_measure(std::ptr::null_mut(), 0) }.value),
        None,
        0,
    );
    assert_eq!((transfers, counter), (1, 0));
    assert_eq!(
        error,
        ProgramError::ResultUnavailable {
            result_id: 0,
            reason: "ready without a stored outcome".into(),
        }
    );
}

#[test]
fn poisoned_stores_and_export_fail_without_publishing() {
    if !test_env::run_test_in_child(
        "dynamic_read_tests::poisoned_stores_and_export_fail_without_publishing",
    ) {
        return;
    }
    for (store, reason) in [
        (0, "poisoned pending operations"),
        (1, "poisoned measurement outcomes"),
        (2, "poisoned synchronization state"),
    ] {
        let ctx = Context::new();
        unsafe { __quantum__qis__m__body(0, 7) };
        let address = ctx.0 as usize;
        std::thread::spawn(move || {
            let ctx = unsafe { &*(address as *mut ExecutionContext) };
            match store {
                0 => {
                    let _guard = ctx.pending_ops.lock().unwrap();
                    panic!("poison pending ops");
                }
                1 => {
                    let _guard = ctx.measurement_results.lock().unwrap();
                    panic!("poison outcomes");
                }
                _ => {
                    let _guard = ctx.sync_state.lock().unwrap();
                    panic!("poison sync");
                }
            }
        })
        .join()
        .unwrap_err();
        assert_eq!(unsafe { ___read_future_uint(7) }, 0);
        assert_eq!(
            ctx.error(),
            ProgramError::ResultUnavailable {
                result_id: 7,
                reason: reason.into()
            }
        );
        assert_eq!(
            ctx.get().waiting_for_result.load(Ordering::SeqCst),
            u64::MAX
        );
        assert_eq!(COLLECTION_MODE_READ_COUNT.get(), 0);
        assert_eq!(TRANSFERS.get(), 1);
        if store != 2 {
            assert!(!ctx.get().sync_state.lock().unwrap().need_result);
        }
    }
}

#[test]
fn selene_preserves_leakage_and_bool_rejects_it_before_wait() {
    if !test_env::run_test_in_child(
        "dynamic_read_tests::selene_preserves_leakage_and_bool_rejects_it_before_wait",
    ) {
        return;
    }
    let ctx = Context::new();
    unsafe { ___lazy_measure_leaked(0) };
    pecos_set_measurement_outcome(0, 2);
    assert_eq!(
        unsafe { selene::selene_future_read_u64(std::ptr::null_mut(), 0) }.value,
        2
    );
    assert!(!unsafe { selene::selene_future_read_bool(std::ptr::null_mut(), 0) }.value);
    assert_eq!(
        ctx.error(),
        ProgramError::ResultUnavailable {
            result_id: 0,
            reason: "outcome is not representable as bool".into()
        }
    );
    assert_eq!(pecos_check_need_result(), u64::MAX);
    assert_eq!(TRANSFERS.get(), 1);
    assert_eq!(COLLECTION_MODE_READ_COUNT.get(), 0);
}

#[test]
fn synchronous_selene_measure_waits_for_host() {
    if !test_env::run_test_in_child("dynamic_read_tests::synchronous_selene_measure_waits_for_host")
    {
        return;
    }
    let ctx = Context::new();
    let address = ctx.0 as usize;
    let worker = std::thread::spawn(move || {
        unsafe { pecos_register_execution_context(address as *mut ExecutionContext) };
        let value = unsafe { selene::selene_qubit_measure(std::ptr::null_mut(), 0) }.value;
        unsafe { pecos_register_execution_context(std::ptr::null_mut()) };
        value
    });
    assert_eq!(pecos_wait_for_need_result(2_000), 0);
    pecos_set_measurement_outcome(0, 1);
    pecos_signal_result_ready();
    assert!(worker.join().unwrap());
    assert!(ctx.get().program_error.lock().unwrap().is_none());
}

#[test]
fn measurements_invalidate_slots_and_track_every_entry_point() {
    if !test_env::run_test_in_child(
        "dynamic_read_tests::measurements_invalidate_slots_and_track_every_entry_point",
    ) {
        return;
    }
    let ctx = Context::new();
    let ids = unsafe {
        [
            ___lazy_measure(0).cast_unsigned(),
            ___lazy_measure_leaked(0).cast_unsigned(),
            selene::selene_qubit_lazy_measure(std::ptr::null_mut(), 0).reference,
            selene::selene_qubit_lazy_measure_leaked(std::ptr::null_mut(), 0).reference,
        ]
    };
    for id in ids {
        assert_eq!(lookup_measurement_outcome(id), Ok(None));
        pecos_set_measurement_outcome(id, 1);
        assert_eq!(unsafe { ___read_future_uint(id.cast_signed()) }, 1);
        unsafe { __quantum__qis__m__body(0, id.cast_signed()) };
        assert_eq!(lookup_measurement_outcome(id), Ok(None));
    }
    reset_interface();
    assert!(ctx.get().measured_results.lock().unwrap().is_empty());
    assert!(ctx.get().measurement_results.lock().unwrap().is_empty());
}

#[test]
fn wait_has_no_deadline_and_reader_completes_only_with_host_outcome() {
    fn is_ready(outcome: &ResultWaitOutcome) -> bool {
        match outcome {
            ResultWaitOutcome::Ready => true,
            ResultWaitOutcome::Cancelled
            | ResultWaitOutcome::WorkerComplete
            | ResultWaitOutcome::Failed(_) => false,
        }
    }
    if !test_env::run_test_in_child(
        "dynamic_read_tests::wait_has_no_deadline_and_reader_completes_only_with_host_outcome",
    ) {
        return;
    }
    // Compile-time signature check and exhaustive match intentionally rule out
    // a timeout argument and a timeout variant without sleeping for 30 seconds.
    let _: fn(u64) -> ResultWaitOutcome = wait_for_result_ready;
    assert!(is_ready(&ResultWaitOutcome::Ready));
    let ctx = Context::new();
    let address = ctx.0 as usize;
    let (done, completed) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        unsafe {
            pecos_register_execution_context(address as *mut ExecutionContext);
            __quantum__qis__m__body(0, 7);
        }
        let value = unsafe { ___read_future_uint(7) };
        unsafe { pecos_register_execution_context(std::ptr::null_mut()) };
        done.send(value).unwrap();
    });
    assert_eq!(pecos_wait_for_need_result(2_000), 7);
    assert_eq!(
        completed.recv_timeout(Duration::from_millis(30)),
        Err(mpsc::RecvTimeoutError::Timeout)
    );
    pecos_set_measurement_outcome(7, 2);
    pecos_signal_result_ready();
    assert_eq!(completed.recv_timeout(Duration::from_secs(2)).unwrap(), 2);
    worker.join().unwrap();
    assert!(ctx.get().program_error.lock().unwrap().is_none());
}

#[test]
fn missing_transfer_handler_records_error_and_returns_abi_default() {
    if !test_env::run_test_in_child(
        "dynamic_read_tests::missing_transfer_handler_records_error_and_returns_abi_default",
    ) {
        return;
    }
    for reader in readers() {
        let ctx = Context::new();
        unsafe { pecos_set_program_panic_handler(None) };
        assert_eq!(reader(7), 0);
        assert_eq!(
            ctx.error(),
            ProgramError::ResultUnavailable {
                result_id: 7,
                reason: "never measured".into()
            }
        );
        assert_eq!(TRANSFERS.get(), 0);
        assert_eq!(COLLECTION_MODE_READ_COUNT.get(), 0);
        assert_eq!(pecos_check_need_result(), u64::MAX);
        assert!(!ctx.get().sync_state.lock().unwrap().need_result);
    }
}
