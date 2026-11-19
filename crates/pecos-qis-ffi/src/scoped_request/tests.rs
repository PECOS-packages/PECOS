use super::*;
use crate::*;
use std::panic::{AssertUnwindSafe, catch_unwind};

thread_local! {
    static REACHED: Cell<bool> = const { Cell::new(false) };
    static DROPPED: Cell<bool> = const { Cell::new(false) };
    static OBSERVED: Cell<u64> = const { Cell::new(0) };
}

struct Context(*mut ExecutionContext);
impl Context {
    fn new() -> Self {
        let context = Self(pecos_create_execution_context());
        unsafe { pecos_register_execution_context(context.0) };
        reset_interface();
        clear_quantum_executor();
        assert_eq!(
            unsafe { pecos_enable_dynamic_mode_with_context(context.0) },
            0
        );
        REACHED.set(false);
        DROPPED.set(false);
        context
    }
}
impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            pecos_register_execution_context(std::ptr::null_mut());
            pecos_destroy_execution_context(self.0);
        }
    }
}

#[derive(Clone, Copy)]
enum Mode {
    Good,
    ErrorAfterReady,
    MissingReady,
    MissingOutcome,
    Cancel,
    Panic,
    Exit,
    ExitCancel,
    Reentry,
    Nested,
    NestedScoped,
    Rebind,
}
struct Host {
    context: *mut ExecutionContext,
    mode: Mode,
    calls: usize,
    caller: std::thread::ThreadId,
}
struct Witness;
impl Drop for Witness {
    fn drop(&mut self) {
        DROPPED.set(true);
    }
}

unsafe extern "C" fn nested_entry() {
    OBSERVED.set(999);
}

unsafe extern "C" fn service(userdata: *mut c_void, id: u64) -> i32 {
    // This is the host trampoline: it catches panics before the C ABI boundary.
    let attempt = catch_unwind(AssertUnwindSafe(|| {
        let _witness = Witness;
        let host = unsafe { &mut *userdata.cast::<Host>() };
        host.calls += 1;
        assert_eq!(std::thread::current().id(), host.caller);
        assert_eq!(get_execution_context(), Some(host.context));
        assert!(ffi::PROGRAM_PANIC_TRANSFER.get().is_none());
        let ctx = unsafe { &*host.context };
        assert!(ctx.sync_state.try_lock().is_ok());
        with_interface(|_| ());
        let mut ops = std::ptr::null_mut();
        assert_eq!(
            unsafe { pecos_get_pending_operations_with_context(host.context, &raw mut ops) },
            0
        );
        assert!(!ops.is_null());
        let operations = unsafe { &*ops };
        assert_eq!(operations.operations.len(), 1);
        assert!(matches!(
            operations.operations[0],
            Operation::Quantum(QuantumOp::Measure(0, 7))
        ));
        unsafe { pecos_free_operations(ops) };
        match host.mode {
            Mode::Panic => std::panic::panic_any(73_u32),
            Mode::Exit | Mode::ExitCancel => unsafe {
                ffi::pecos_record_program_panic(7, std::ptr::null(), 0);
            },
            Mode::Reentry => {
                assert!(matches!(
                    wait_for_result_ready(id),
                    ResultWaitOutcome::Failed("reentrant scoped request")
                ));
            }
            Mode::Nested => {
                assert_eq!(ctx.program_allocations.lock().unwrap().len(), 1);
                assert_eq!(
                    unsafe { crate::selene::pecos_call_void_main_with_setjmp(nested_entry) },
                    1
                );
                assert_ne!(OBSERVED.get(), 999);
                assert_eq!(ctx.program_allocations.lock().unwrap().len(), 1);
            }
            Mode::NestedScoped => {
                assert_eq!(
                    unsafe {
                        pecos_call_void_main_with_request_callback_v1(
                            host.context,
                            nested_entry,
                            service,
                            std::ptr::null_mut(),
                        )
                    },
                    1
                );
                assert_ne!(OBSERVED.get(), 999);
                assert_eq!(ctx.program_allocations.lock().unwrap().len(), 1);
            }
            Mode::Rebind => unsafe { pecos_register_execution_context(std::ptr::null_mut()) },
            _ => {}
        }
        if !matches!(host.mode, Mode::MissingOutcome) {
            assert_eq!(
                unsafe { pecos_set_measurement_outcome_with_context(host.context, id, 1) },
                0
            );
        }
        if !matches!(host.mode, Mode::MissingReady) {
            assert_eq!(
                unsafe { pecos_signal_result_ready_with_context(host.context) },
                0
            );
        }
        if matches!(host.mode, Mode::ExitCancel) {
            let address = host.context as usize;
            assert_eq!(
                std::thread::spawn(move || unsafe {
                    pecos_abort_dynamic_execution(address as *mut ExecutionContext)
                })
                .join()
                .unwrap(),
                0
            );
        } else if matches!(host.mode, Mode::Cancel) {
            assert_eq!(unsafe { pecos_abort_dynamic_execution(host.context) }, 0);
        }
        i32::from(matches!(host.mode, Mode::ErrorAfterReady))
    }));
    match attempt {
        Ok(status) => status,
        Err(payload) => {
            assert_eq!(payload.downcast_ref::<u32>(), Some(&73));
            1
        }
    }
}

// The program-side frames own no destructors or Rust borrows across a read;
// a failed read may transfer to the C guard, exactly as a compiled program does.
unsafe extern "C" fn entry() {
    // Nested entry must not run shot-start cleanup on this outer allocation.
    let _allocation = unsafe { ffi::heap_alloc(8) };
    unsafe { ffi::__quantum__qis__m__body(0, 7) };
    OBSERVED.set(u64::from(unsafe { ffi::___read_future_bool(7) }));
    REACHED.set(true);
}
unsafe extern "C" fn qentry(_: u64) -> u64 {
    unsafe { entry() };
    0
}

fn invoke(ctx: &Context, mode: Mode, is_void: bool) -> (u64, usize) {
    let mut host = Host {
        context: ctx.0,
        mode,
        calls: 0,
        caller: std::thread::current().id(),
    };
    let userdata = (&raw mut host).cast();
    let status = unsafe {
        if is_void {
            pecos_call_void_main_with_request_callback_v1(ctx.0, entry, service, userdata)
        } else {
            pecos_call_qmain_with_request_callback_v1(ctx.0, qentry, service, userdata)
        }
    };
    assert!(!is_registered());
    assert!(ffi::PROGRAM_PANIC_TRANSFER.get().is_none());
    assert!(!unsafe { pecos_execution_guard_is_active() });
    assert!(DROPPED.get());
    assert!(
        unsafe { &*ctx.0 }
            .program_allocations
            .lock()
            .unwrap()
            .is_empty()
    );
    (status, host.calls)
}

#[test]
fn both_entries_service_requests_without_a_worker_and_release_registration() {
    if !test_env::run_test_in_child(
        "scoped_request::tests::both_entries_service_requests_without_a_worker_and_release_registration",
    ) {
        return;
    }
    for is_void in [false, true] {
        let ctx = Context::new();
        for _ in 0..2 {
            assert_eq!(invoke(&ctx, Mode::Good, is_void), (0, 1));
            assert!(REACHED.get());
            assert_eq!(OBSERVED.get(), 1);
        }
    }
}

#[test]
fn scoped_contexts_are_isolated_across_threads() {
    if !test_env::run_test_in_child(
        "scoped_request::tests::scoped_contexts_are_isolated_across_threads",
    ) {
        return;
    }
    let threads: Vec<_> = (0..4)
        .map(|index| {
            std::thread::spawn(move || {
                let ctx = Context::new();
                for _ in 0..4 {
                    assert_eq!(invoke(&ctx, Mode::Good, index % 2 == 0), (0, 1));
                    assert_eq!(OBSERVED.get(), 1);
                }
            })
        })
        .collect();
    for thread in threads {
        thread.join().unwrap();
    }
}

#[test]
fn callback_failures_cancel_context_after_cleanup_and_do_not_resume_program() {
    if !test_env::run_test_in_child(
        "scoped_request::tests::callback_failures_cancel_context_after_cleanup_and_do_not_resume_program",
    ) {
        return;
    }
    for is_void in [false, true] {
        for mode in [
            Mode::ErrorAfterReady,
            Mode::MissingReady,
            Mode::MissingOutcome,
            Mode::Cancel,
            Mode::Panic,
            Mode::Exit,
            Mode::ExitCancel,
            Mode::Reentry,
            Mode::Nested,
            Mode::NestedScoped,
            Mode::Rebind,
        ] {
            let ctx = Context::new();
            assert_eq!(invoke(&ctx, mode, is_void), (1, 1));
            assert!(!REACHED.get());
            let ctx_ref = unsafe { &*ctx.0 };
            assert!(ctx_ref.cancel_requested.load(Ordering::Acquire));
            assert!(!ctx_ref.sync_state.lock().unwrap().need_result);
            let error = ctx_ref.program_error.lock().unwrap();
            assert!(error.is_some());
            if matches!(mode, Mode::Exit | Mode::ExitCancel) {
                assert!(matches!(*error, Some(ProgramError::Exit { code: 7, .. })));
            }
            drop(error);
            assert_eq!(
                unsafe {
                    pecos_call_void_main_with_request_callback_v1(
                        ctx.0,
                        nested_entry,
                        service,
                        std::ptr::null_mut(),
                    )
                },
                1
            );
        }
        let fresh = Context::new();
        assert_eq!(invoke(&fresh, Mode::Good, is_void), (0, 1));
    }
}

#[test]
fn admission_rejects_missing_context_non_dynamic_and_rust_executor() {
    if !test_env::run_test_in_child(
        "scoped_request::tests::admission_rejects_missing_context_non_dynamic_and_rust_executor",
    ) {
        return;
    }
    assert_eq!(
        unsafe {
            pecos_call_void_main_with_request_callback_v1(
                std::ptr::null_mut(),
                nested_entry,
                service,
                std::ptr::null_mut(),
            )
        },
        1
    );
    let ctx = Context::new();
    unsafe { &*ctx.0 }
        .dynamic_mode_active
        .store(false, Ordering::SeqCst);
    assert_eq!(
        unsafe {
            pecos_call_void_main_with_request_callback_v1(
                ctx.0,
                nested_entry,
                service,
                std::ptr::null_mut(),
            )
        },
        1
    );
    unsafe { &*ctx.0 }
        .dynamic_mode_active
        .store(true, Ordering::SeqCst);
    set_quantum_executor(|_| std::collections::BTreeMap::new());
    assert_eq!(
        unsafe {
            pecos_call_void_main_with_request_callback_v1(
                ctx.0,
                nested_entry,
                service,
                std::ptr::null_mut(),
            )
        },
        1
    );
    clear_quantum_executor();
    assert_eq!(invoke(&ctx, Mode::Good, true), (0, 1));
}

unsafe extern "C" fn terminal_error() {
    // Like a typed-output mismatch: record an error, then return normally.
    unsafe { ffi::pecos_record_program_panic(1001, std::ptr::null(), 0) };
}

#[test]
fn terminal_error_without_a_request_fails_but_ordinary_completion_succeeds() {
    if !test_env::run_test_in_child(
        "scoped_request::tests::terminal_error_without_a_request_fails_but_ordinary_completion_succeeds",
    ) {
        return;
    }
    let ctx = Context::new();
    assert_eq!(
        unsafe {
            pecos_call_void_main_with_request_callback_v1(
                ctx.0,
                terminal_error,
                service,
                std::ptr::null_mut(),
            )
        },
        1
    );
    assert!(unsafe { &*ctx.0 }.cancel_requested.load(Ordering::Acquire));
    assert!(!is_registered());
    drop(ctx);
    let fresh = Context::new();
    assert_eq!(
        unsafe {
            pecos_call_void_main_with_request_callback_v1(
                fresh.0,
                nested_entry,
                service,
                std::ptr::null_mut(),
            )
        },
        0
    );
    assert!(
        !unsafe { &*fresh.0 }
            .cancel_requested
            .load(Ordering::Acquire)
    );
}
