//! Each test runs in a child process and owns its context and TLS registration.
use crate::ffi::*;
use crate::*;
use std::sync::atomic::Ordering;

struct Context(*mut ExecutionContext);
impl Context {
    fn new() -> Self {
        let ctx = Self(pecos_create_execution_context());
        // SAFETY: The owned context outlives this thread's registration.
        unsafe { pecos_register_execution_context(ctx.0) };
        reset_interface();
        pecos_enable_dynamic_mode();
        ctx
    }
    fn get(&self) -> &ExecutionContext {
        // SAFETY: This fixture owns the allocation until drop, after worker joins.
        unsafe { &*self.0 }
    }
    fn cancel(&self) {
        // SAFETY: The fixture keeps the context alive throughout the call.
        assert_eq!(unsafe { pecos_abort_dynamic_execution(self.0) }, 0);
    }
}
impl Drop for Context {
    fn drop(&mut self) {
        // SAFETY: All workers have joined and this is the last registration.
        unsafe {
            pecos_set_program_panic_handler(None);
            pecos_register_execution_context(std::ptr::null_mut());
            pecos_destroy_execution_context(self.0);
        }
    }
}

#[test]
fn cancellation_before_wait_survives_output_resets() {
    if !crate::test_env::run_test_in_child(
        "cancellation_tests::cancellation_before_wait_survives_output_resets",
    ) {
        return;
    }
    let ctx = Context::new();
    ctx.cancel();
    reset_interface();
    pecos_clear_program_error();
    assert!(ctx.get().cancel_requested.load(Ordering::Acquire));
    let start = std::time::Instant::now();
    assert!(!wait_for_result_ready(7, 1_000));
    assert!(start.elapsed() < std::time::Duration::from_millis(500));
}

// Wake a stalled reader on failure so it can be joined before freeing its context.
// Return the original timeout: this cleanup must never make a broken wake pass.
fn receive_cancelled_reader<T>(
    ctx: &Context,
    completed: &std::sync::mpsc::Receiver<T>,
) -> Result<T, std::sync::mpsc::RecvTimeoutError> {
    let result = completed.recv_timeout(std::time::Duration::from_secs(1));
    if result.is_err() {
        ctx.get().sync_state.lock().unwrap().worker_complete = true;
        ctx.get().sync_condvar.notify_all();
    }
    result
}

#[test]
fn abort_export_wakes_waiting_reader() {
    if !crate::test_env::run_test_in_child("cancellation_tests::abort_export_wakes_waiting_reader")
    {
        return;
    }
    let ctx = Context::new();
    std::thread::scope(|scope| {
        let address = ctx.0 as usize;
        let (done, completed) = std::sync::mpsc::channel();
        let worker = scope.spawn(move || {
            // SAFETY: Parent keeps the context alive until this worker joins.
            unsafe { pecos_register_execution_context(address as *mut ExecutionContext) };
            // SAFETY: Valid result ID; no guard means cancellation returns zero.
            let result = unsafe { ___read_future_uint(7) };
            // SAFETY: Clear this worker's registration before its context is freed.
            unsafe { pecos_register_execution_context(std::ptr::null_mut()) };
            done.send(result).unwrap();
        });
        assert_eq!(pecos_wait_for_need_result(2_000), 7);
        // Let the reader settle into its wait, and prove it has not returned.
        assert_eq!(
            completed.recv_timeout(std::time::Duration::from_millis(20)),
            Err(std::sync::mpsc::RecvTimeoutError::Timeout)
        );
        ctx.cancel();
        let result = receive_cancelled_reader(&ctx, &completed);
        worker.join().unwrap();
        assert_eq!(result.unwrap(), 0);
        assert_eq!(
            *ctx.get().program_error.lock().unwrap(),
            Some(ProgramError::Cancelled)
        );
    });
}

#[test]
fn cancellation_after_wait_precedes_ready_result_and_collection_fallback() {
    if !crate::test_env::run_test_in_child(
        "cancellation_tests::cancellation_after_wait_precedes_ready_result_and_collection_fallback",
    ) {
        return;
    }
    let readers: [fn(i64) -> u64; 2] = [
        |id| {
            // SAFETY: Valid result ID with a live registered context.
            u64::from(unsafe { ___read_future_bool(id) })
        },
        |id| {
            // SAFETY: Valid result ID with a live registered context.
            unsafe { ___read_future_uint(id) }
        },
    ];
    for (reader, ready) in readers
        .into_iter()
        .flat_map(|reader| [(reader, false), (reader, true)])
    {
        let ctx = Context::new();
        std::thread::scope(|scope| {
            let address = ctx.0 as usize;
            let (done, completed) = std::sync::mpsc::channel();
            let worker = scope.spawn(move || {
                // SAFETY: Parent owns the context until join. The returning handler
                // only inspects borrows on this thread and is cleared before exit.
                unsafe {
                    pecos_register_execution_context(address as *mut ExecutionContext);
                    pecos_set_program_panic_handler(Some(inspect_transfer));
                }
                assert_eq!(COLLECTION_MODE_READ_COUNT.get(), 0);
                let result = reader(7);
                let observations = (result, TRANSFERS.get(), COLLECTION_MODE_READ_COUNT.get());
                // SAFETY: Clear the thread's handler and registration before exit.
                unsafe {
                    pecos_set_program_panic_handler(None);
                    pecos_register_execution_context(std::ptr::null_mut());
                }
                done.send(observations).unwrap();
            });
            assert_eq!(pecos_wait_for_need_result(2_000), 7);
            assert_eq!(
                completed.recv_timeout(std::time::Duration::from_millis(20)),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout)
            );
            if ready {
                pecos_set_measurement_outcome(7, 1);
            }
            {
                let mut state = ctx.get().sync_state.lock().unwrap();
                // Publish both under the wait's mutex to exercise a deterministic
                // ready-result/cancel race. The export's wake is tested separately.
                state.result_ready = ready;
                state.cancellation = CancellationState::Requested;
                ctx.get().cancel_requested.store(true, Ordering::Release);
            }
            ctx.get().sync_condvar.notify_all();
            let observations = receive_cancelled_reader(&ctx, &completed);
            worker.join().unwrap();
            assert_eq!(observations.unwrap(), (0, 1, 0));
            assert_eq!(
                *ctx.get().program_error.lock().unwrap(),
                Some(ProgramError::Cancelled)
            );
        });
    }
}

thread_local! { static TRANSFERS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }
unsafe extern "C-unwind" fn inspect_transfer() {
    let ctx = get_execution_context().unwrap();
    // SAFETY: Called while the test's context is registered and live.
    let ctx = unsafe { &*ctx };
    assert!(ctx.sync_state.try_lock().is_ok());
    assert!(ctx.program_error.try_lock().is_ok());
    with_interface(|_| ());
    EXECUTOR.with(|executor| assert!(executor.try_borrow_mut().is_ok()));
    TRANSFERS.set(TRANSFERS.get() + 1);
}

#[test]
fn cancellation_precedes_every_cached_reader_and_releases_borrows() {
    if !crate::test_env::run_test_in_child(
        "cancellation_tests::cancellation_precedes_every_cached_reader_and_releases_borrows",
    ) {
        return;
    }
    let ctx = Context::new();
    with_interface(|iface| iface.store_result(7, true));
    pecos_set_measurement_outcome(7, 2);
    ctx.cancel();
    TRANSFERS.set(0);
    // SAFETY: The handler only inspects released borrows and returns, and remains
    // installed on this test thread until Context::drop clears it.
    unsafe {
        pecos_set_program_panic_handler(Some(inspect_transfer));
        assert!(!___read_future_bool(7));
        assert_eq!(___read_future_uint(7), 0);
        assert_eq!(__quantum__rt__result_get_one(7), 0);
        assert!(!selene::selene_future_read_bool(std::ptr::null_mut(), 7).value);
        assert_eq!(
            selene::selene_future_read_u64(std::ptr::null_mut(), 7).value,
            0
        );
    }
    assert_eq!(pecos_wait_for_need_result(0), u64::MAX);
    assert_eq!(TRANSFERS.get(), 5);
    assert_eq!(
        *ctx.get().program_error.lock().unwrap(),
        Some(ProgramError::Cancelled)
    );
}

#[test]
fn cancellation_after_callback_returns_releases_executor_borrow() {
    if !crate::test_env::run_test_in_child(
        "cancellation_tests::cancellation_after_callback_returns_releases_executor_borrow",
    ) {
        return;
    }
    let ctx = Context::new();
    let address = ctx.0 as usize;
    set_quantum_executor(move |_| {
        // SAFETY: Callback runs before the owning fixture is dropped.
        assert_eq!(
            unsafe { pecos_abort_dynamic_execution(address as *mut ExecutionContext) },
            0
        );
        BTreeMap::from([(7, true)])
    });
    TRANSFERS.set(0);
    // SAFETY: The returning handler inspects borrows on this thread and is
    // cleared by Context::drop. The result ID is valid.
    unsafe {
        pecos_set_program_panic_handler(Some(inspect_transfer));
        assert_eq!(__quantum__rt__result_get_one(7), 0);
    }
    assert_eq!(TRANSFERS.get(), 1);
    clear_quantum_executor();
    assert_eq!(
        *ctx.get().program_error.lock().unwrap(),
        Some(ProgramError::Cancelled)
    );
}

#[test]
fn cancellation_checkpoints_prevent_operation_submission() {
    if !crate::test_env::run_test_in_child(
        "cancellation_tests::cancellation_checkpoints_prevent_operation_submission",
    ) {
        return;
    }
    let ctx = Context::new();
    ctx.cancel();
    // SAFETY: All IDs are valid; no transfer handler is installed.
    unsafe {
        __quantum__qis__y__body(0);
        __quantum__qis__cx__body(0, 1);
        __quantum__qis__ccx__body(0, 1, 2);
        __quantum__qis__rx__body(0.1, 0);
        __quantum__qis__rzz__body(0.1, 0, 1);
        __quantum__qis__h__body(0);
        __quantum__qis__x__body(0);
        __quantum__qis__r1xy__body(0.1, 0.2, 0);
        ___rpp(0, 1, 0.1, 0.2);
        pecos_qis_runtime_barrier_qubit_hugr(1);
        pecos_qis_runtime_barrier_qubits2_hugr(1, 2);
        assert_eq!(__quantum__rt__qubit_allocate(), 0);
        assert_eq!(__quantum__qis__m__body(0, 7), 0);
        assert_eq!(___lazy_measure(0), 0);
        assert_eq!(___lazy_measure_leaked(0), 0);
    }
    with_interface(|iface| assert_eq!(iface.operations.len(), 0));
    assert_eq!(
        *ctx.get().program_error.lock().unwrap(),
        Some(ProgramError::Cancelled)
    );
}

#[test]
fn abort_is_explicit_idempotent_and_reports_poison() {
    if !crate::test_env::run_test_in_child(
        "cancellation_tests::abort_is_explicit_idempotent_and_reports_poison",
    ) {
        return;
    }
    let first = Context::new();
    let second = Context::new();
    first.cancel();
    // SAFETY: Unregister only this thread; both fixtures retain their contexts.
    unsafe { pecos_register_execution_context(std::ptr::null_mut()) };
    first.cancel();
    assert!(!second.get().cancel_requested.load(Ordering::Acquire));
    let address = second.0 as usize;
    std::thread::spawn(move || {
        // SAFETY: The parent keeps the context alive until join.
        let ctx = unsafe { &*(address as *mut ExecutionContext) };
        let _guard = ctx.sync_state.lock().unwrap();
        panic!("poison synchronization state");
    })
    .join()
    .unwrap_err();
    // SAFETY: Null is explicitly supported; second is still owned here.
    unsafe {
        assert_ne!(pecos_abort_dynamic_execution(std::ptr::null_mut()), 0);
        assert_ne!(pecos_abort_dynamic_execution(second.0), 0);
    }
}
