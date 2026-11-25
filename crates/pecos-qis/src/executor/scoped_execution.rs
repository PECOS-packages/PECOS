//! Scoped trampoline. This state is stack-owned, never cloned or retained.
use super::{ExecutionContext, ExecutionEntryPoint, InterfaceError, Library, Symbol};
use std::ffi::c_void;

thread_local! { static EXECUTING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }

pub(super) struct ExecutionScope;
impl ExecutionScope {
    pub(super) fn enter() -> Result<Self, InterfaceError> {
        if EXECUTING.replace(true) {
            return Err(InterfaceError::ExecutionError(
                "Nested program execution is unsupported".into(),
            ));
        }
        Ok(Self)
    }
}
impl Drop for ExecutionScope {
    fn drop(&mut self) {
        EXECUTING.set(false);
    }
}

type Callback = unsafe extern "C" fn(*mut c_void, u64) -> i32;
type QGuard = unsafe extern "C" fn(
    *mut ExecutionContext,
    unsafe extern "C-unwind" fn(u64) -> u64,
    Callback,
    *mut c_void,
) -> u64;
type VGuard = unsafe extern "C" fn(
    *mut ExecutionContext,
    unsafe extern "C-unwind" fn(),
    Callback,
    *mut c_void,
) -> u64;

pub(super) struct RequestHost<'a> {
    request: &'a mut dyn FnMut(u64) -> Result<(), InterfaceError>,
    pub(super) error: Option<InterfaceError>,
    pub(super) panic: Option<Box<dyn std::any::Any + Send>>,
}

impl<'a> RequestHost<'a> {
    pub(super) fn new(request: &'a mut dyn FnMut(u64) -> Result<(), InterfaceError>) -> Self {
        Self {
            request,
            error: None,
            panic: None,
        }
    }
}

unsafe extern "C" fn service(data: *mut c_void, id: u64) -> i32 {
    // SAFETY: call() lends the live host exclusively to the synchronous guard;
    // the FFI boundary rejects reentry before invoking us again.
    let host = unsafe { &mut *data.cast::<RequestHost<'_>>() };
    if host.error.is_some() || host.panic.is_some() {
        return 1;
    }
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (host.request)(id))) {
        Ok(Ok(())) => 0,
        Ok(Err(error)) => {
            host.error = Some(error);
            1
        }
        Err(payload) => {
            host.panic = Some(payload);
            1
        }
    }
}

pub(super) fn call(
    lib: &Library,
    context: *mut ExecutionContext,
    entry: &ExecutionEntryPoint<'_>,
    host: &mut RequestHost<'_>,
) -> Result<(&'static str, u64), InterfaceError> {
    let userdata = std::ptr::from_mut(host).cast::<c_void>();
    let missing = |e| InterfaceError::ExecutionError(format!("Missing scoped entry: {e}"));
    // SAFETY: Both symbols come from the admitted FFI library. The caller has
    // registered its live context in that library and owns the program entry.
    // The guard returns before host/userdata can move or be released.
    unsafe {
        match entry {
            ExecutionEntryPoint::Qmain { func, .. } => {
                let guard: Symbol<QGuard> = lib
                    .get(b"pecos_call_qmain_with_request_callback_v1\0")
                    .map_err(missing)?;
                Ok(("qmain", guard(context, **func, service, userdata)))
            }
            ExecutionEntryPoint::VoidMain { func, .. } => {
                let guard: Symbol<VGuard> = lib
                    .get(b"pecos_call_void_main_with_request_callback_v1\0")
                    .map_err(missing)?;
                Ok(("main", guard(context, **func, service, userdata)))
            }
        }
    }
}
