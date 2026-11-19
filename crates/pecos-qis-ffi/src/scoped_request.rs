//! Scoped, same-thread service of dynamic requests. The staged path is unchanged.
use crate::{ExecutionContext, ProgramError, ResultWaitOutcome, ffi, get_execution_context};
use std::cell::Cell;
use std::ffi::c_void;
use std::sync::atomic::Ordering;

/// Hosts catch Rust panics in their trampoline and retain the original payload.
/// No unwind or nonlocal transfer may cross this C ABI callback.
type RequestCallback = unsafe extern "C" fn(*mut c_void, u64) -> i32;

#[derive(Clone, Copy)]
struct Binding {
    context: *mut ExecutionContext,
    callback: RequestCallback,
    userdata: *mut c_void,
}

thread_local! {
    static BINDING: Cell<Option<Binding>> = const { Cell::new(None) };
    static IN_CALLBACK: Cell<bool> = const { Cell::new(false) };
    static FAILED: Cell<Option<&'static str>> = const { Cell::new(None) };
}

fn fail(reason: &'static str) {
    if FAILED.get().is_none() {
        FAILED.set(Some(reason));
    }
}

pub(super) fn is_registered() -> bool {
    BINDING.get().is_some()
}

pub(super) fn allow_context(context: *mut ExecutionContext) -> bool {
    if BINDING
        .get()
        .is_some_and(|binding| binding.context != context)
    {
        fail("context changed during scoped execution");
        return false;
    }
    true
}

pub(super) fn admit_request(context: *mut ExecutionContext) -> Result<(), &'static str> {
    if let Some(binding) = BINDING.get() {
        if IN_CALLBACK.get() {
            fail("reentrant scoped request");
        } else if binding.context != context {
            fail("scoped request context mismatch");
        }
        if let Some(reason) = FAILED.get() {
            return Err(reason);
        }
    }
    Ok(())
}

struct CallbackGuard(Option<unsafe extern "C-unwind" fn()>);
impl Drop for CallbackGuard {
    fn drop(&mut self) {
        IN_CALLBACK.set(false);
        ffi::PROGRAM_PANIC_TRANSFER.set(self.0);
    }
}

pub(super) fn dispatch(result_id: u64) -> ResultWaitOutcome {
    let binding = BINDING.get().expect("admitted scoped request");
    let status = {
        // No TLS borrow or context lock is held across the host call. Disable
        // longjmp until the callback's Rust frames and destructors have returned.
        let _guard = CallbackGuard(ffi::PROGRAM_PANIC_TRANSFER.replace(None));
        IN_CALLBACK.set(true);
        // SAFETY: The scoped entry keeps the callback and userdata live. Its
        // contract forbids unwinding, transfers, and destruction of the context.
        unsafe { (binding.callback)(binding.userdata, result_id) }
    };
    if status != 0 {
        fail("scoped request callback failed");
    }
    // SAFETY: The entry's caller keeps this context alive for the entire scope.
    let ctx = unsafe { &*binding.context };
    let Ok(mut state) = ctx.sync_state.lock() else {
        fail("poisoned synchronization state");
        return ResultWaitOutcome::Failed("poisoned synchronization state");
    };
    let cancelled = state.cancellation == crate::CancellationState::Requested;
    let ready = state.result_ready && !state.need_result;
    // Retire this publication even on failure. There is no staged consumer in
    // this mode, and no subsequent callback may inherit an old readiness flag.
    state.need_result = false;
    state.result_ready = false;
    if cancelled {
        return ResultWaitOutcome::Cancelled;
    }
    if state.worker_complete {
        fail("scoped callback completed the worker");
    }
    if !ready {
        fail("scoped callback returned without readiness");
    }
    drop(state);
    match ctx.program_error.lock() {
        Ok(error) if error.is_none() => {}
        _ => fail("program error during scoped callback"),
    }
    match FAILED.get() {
        Some(reason) => ResultWaitOutcome::Failed(reason),
        None => ResultWaitOutcome::Ready,
    }
}

struct Registration;
impl Drop for Registration {
    fn drop(&mut self) {
        BINDING.set(None);
        FAILED.set(None);
        IN_CALLBACK.set(false);
    }
}

unsafe extern "C" {
    fn pecos_execution_guard_is_active() -> bool;
}

fn enter(
    context: *mut ExecutionContext,
    callback: RequestCallback,
    userdata: *mut c_void,
) -> Option<Registration> {
    // This check precedes lifecycle resets and protects the C recovery target.
    if is_registered() || unsafe { pecos_execution_guard_is_active() } {
        reject_nested();
        return None;
    }
    if context.is_null() || get_execution_context() != Some(context) {
        return None;
    }
    // The older Rust-only executor takes precedence over dynamic reads. It
    // cannot coexist with the scoped C ABI protocol.
    if crate::EXECUTOR.with(|executor| executor.try_borrow().map_or(true, |e| e.is_some())) {
        return None;
    }
    // SAFETY: The exported entry requires a live registered context.
    let ctx = unsafe { &*context };
    if !ctx.dynamic_mode_active.load(Ordering::SeqCst)
        || ctx.cancel_requested.load(Ordering::Acquire)
        || ffi::PROGRAM_PANIC_TRANSFER.get().is_some()
    {
        return None;
    }
    let Ok(state) = ctx.sync_state.lock() else {
        return None;
    };
    if state.worker_complete || state.need_result {
        return None;
    }
    FAILED.set(None);
    BINDING.set(Some(Binding {
        context,
        callback,
        userdata,
    }));
    Some(Registration)
}

fn finish(context: *mut ExecutionContext, result: u64) -> u64 {
    // A program can record a terminal error after its final request and return
    // normally. Cancellation also wins over a previously recorded ordinary Exit.
    // SAFETY: The entry's caller keeps the context live for this entire scope.
    let ctx = unsafe { &*context };
    if ctx.cancel_requested.load(Ordering::Acquire) {
        fail("scoped execution cancelled");
    }
    match ctx.program_error.lock() {
        Ok(error) if matches!(*error, None | Some(ProgramError::Exit { .. })) => {}
        _ => fail("scoped program failed"),
    }
    let result = if FAILED.get().is_some() { 1 } else { result };
    if result != 0 {
        // Failed scopes retire their context. Never fall back to a staged wait
        // or reuse partial collector/runtime state; recovery needs a fresh owner.
        unsafe { crate::pecos_abort_dynamic_execution(context) };
    }
    result
}

/// Called by the C guard before touching lifecycle or its existing jump buffer.
#[unsafe(no_mangle)]
pub extern "C" fn pecos_reject_nested_execution() {
    reject_nested();
}

fn reject_nested() {
    if is_registered() {
        fail("nested program execution");
    }
    if let Some(context) = get_execution_context() {
        // SAFETY: A registered context remains live throughout execution.
        unsafe { &*context }.record_program_error(ProgramError::InvalidInput {
            entry: "execution guard".into(),
            detail: "nested program execution".into(),
        });
    }
}

/// Run qmain with same-thread service of synchronous result/runtime requests.
///
/// Returns the guarded program status, or 1 for admission, callback, cancellation
/// or recorded terminal failure.
/// The callback receives (userdata, result ID), drains the existing pending-op
/// APIs, stores the outcome and signals readiness before returning zero. Any
/// nonzero callback status is terminal. Registration is removed on every return.
/// A failed scope cancels its context permanently; retry requires a fresh context.
/// Context registration, mode configuration and destruction remain with the host.
/// The ordinary C shot lifecycle still runs, including its per-shot RNG reset.
///
/// # Safety
/// The context must be live, dynamic, registered in this exact library on this
/// thread, and exclusively executing this program (no staged consumer). The entry
/// and callback must be valid for the call. Userdata may borrow caller state for
/// this scope only.
/// The callback must catch its own Rust panics and return nonzero; it must not
/// unwind, longjmp, destroy/reset the context, install a transfer handler, or
/// execute program instructions, including cached reads. Recursive waits and
/// nested guards fail closed, but are not general protection for arbitrary
/// program API reentry. Only pending-operation drain/free, outcome storage and
/// readiness signaling APIs may be used by the host callback. Other threads may
/// request cancellation, but must not otherwise mutate the context. The host must
/// validate native Boolean responses as 0/1 before storing them. The program
/// obeys the ordinary C recovery discipline. This mode cannot detach an
/// unresponsive program; callers must opt into that cancellation limitation.
/// Rust program entries must use `C-unwind` for Windows recovery and own no
/// destructors across a transfer. This does not permit the host callback to unwind.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pecos_call_qmain_with_request_callback_v1(
    context: *mut ExecutionContext,
    qmain: unsafe extern "C-unwind" fn(u64) -> u64,
    callback: RequestCallback,
    userdata: *mut c_void,
) -> u64 {
    let Some(_registration) = enter(context, callback, userdata) else {
        return 1;
    };
    // SAFETY: The caller provides a valid program. This scope is outside the
    // C recovery target, so its registration destructor is never skipped.
    let result = unsafe { crate::selene::pecos_call_qmain_with_setjmp(qmain) };
    finish(context, result)
}

/// Run a void main entry using the scoped request callback protocol.
///
/// # Safety
/// Identical context, callback, lifetime, recovery and cancellation requirements
/// to [`pecos_call_qmain_with_request_callback_v1`], with a void entry signature.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pecos_call_void_main_with_request_callback_v1(
    context: *mut ExecutionContext,
    main: unsafe extern "C-unwind" fn(),
    callback: RequestCallback,
    userdata: *mut c_void,
) -> u64 {
    let Some(_registration) = enter(context, callback, userdata) else {
        return 1;
    };
    // SAFETY: Matching void entry/guard ABI; registration lives outside setjmp.
    let result = unsafe { crate::selene::pecos_call_void_main_with_setjmp(main) };
    finish(context, result)
}

#[cfg(test)]
mod tests;
