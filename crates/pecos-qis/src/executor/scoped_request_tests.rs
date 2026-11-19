//! Verify the scoped protocol through the selected cdylib and compiled LLVM.
//! This does not enable scoped execution in `QisEngine` or bypass its future parity gate.
use super::*;
use crate::selene_test_ir;
use pecos_qis_ffi_types::{Operation, RUNTIME_REQUEST_RESULT_ID, RuntimeRequest, SeleneRequest};
use std::ffi::c_void;

type Callback = unsafe extern "C" fn(*mut c_void, u64) -> i32;
type QGuard = unsafe extern "C" fn(
    *mut ExecutionContext,
    unsafe extern "C" fn(u64) -> u64,
    Callback,
    *mut c_void,
) -> u64;
type VGuard = unsafe extern "C" fn(
    *mut ExecutionContext,
    unsafe extern "C" fn(),
    Callback,
    *mut c_void,
) -> u64;

struct Host {
    context: *mut ExecutionContext,
    drain: GetPendingOperationsFn,
    free: unsafe extern "C" fn(*mut OperationCollector),
    store: SetMeasurementOutcomeFn,
    ready: SignalResultReadyFn,
    caller: std::thread::ThreadId,
    calls: usize,
    fail: bool,
    panic: Option<Box<dyn std::any::Any + Send>>,
}

unsafe extern "C" fn service(data: *mut c_void, result_id: u64) -> i32 {
    let host = unsafe { &mut *data.cast::<Host>() };
    let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        assert_eq!(std::thread::current().id(), host.caller);
        assert_eq!(result_id, RUNTIME_REQUEST_RESULT_ID as u64);
        let mut collector = std::ptr::null_mut();
        assert_eq!(unsafe { (host.drain)(host.context, &raw mut collector) }, 0);
        assert!(!collector.is_null());
        let ops = unsafe { &*collector }.operations.clone();
        unsafe { (host.free)(collector) };
        assert_eq!(
            ops,
            vec![Operation::RuntimeRequest {
                request: RuntimeRequest::Native(SeleneRequest::Qalloc),
                result_id: Some(RUNTIME_REQUEST_RESULT_ID),
            }]
        );
        host.calls += 1;
        assert_eq!(
            unsafe { (host.store)(host.context, result_id, (1_u64 << 60) + host.calls as u64) },
            0
        );
        assert_eq!(unsafe { (host.ready)(host.context) }, 0);
        i32::from(host.fail)
    }));
    match attempt {
        Ok(status) => status,
        Err(payload) => {
            host.panic = Some(payload);
            1
        }
    }
}

pub(super) fn run_compiled_entries() {
    const CHILD: &str = "PECOS_TEST_SCOPED_CALLBACK_CHILD";
    if std::env::var_os(CHILD).is_none() {
        crate::test_env::run_test_in_child(
            "executor::tests::compiled_entries_use_scoped_requests_in_the_selected_ffi",
            &[(CHILD, "1".as_ref())],
        );
        return;
    }
    let ffi = QisHeliosInterface::get_qis_ffi_lib_singleton().unwrap();
    let qalloc = selene_test_ir::declaration("selene_qalloc", "ptr");
    let first = selene_test_ir::call("a", "selene_qalloc", "ptr null");
    let second = selene_test_ir::call("b", "selene_qalloc", "ptr null");
    // The second request reuses the reserved response slot; the branch verifies
    // both the full-width value and invalidation of the first response.
    let body = format!(
        r"
        {first}
        %av = extractvalue {{i32, i64}} %a, 1
        %ok = icmp eq i64 %av, 1152921504606846977
        br i1 %ok, label %next, label %bad
        next:
        {second}
        %bv = extractvalue {{i32, i64}} %b, 1
        store i64 %bv, ptr @observed
        br label %done
        bad:
        store i64 99, ptr @observed
        br label %done
        done:
    "
    );
    for is_void in [false, true] {
        let signature = if is_void {
            "void @main()"
        } else {
            "i64 @qmain(i64 %arg)"
        };
        let ret = if is_void { "ret void" } else { "ret i64 0" };
        let program = format!(
            r"
            @observed = global i64 0
            {qalloc}
            define {signature} {{ {body} {ret} }}
            define i64 @read_observed() {{ %v = load i64, ptr @observed
                ret i64 %v }}
            define void @clear_observed() {{ store i64 0, ptr @observed
                ret void }}
        "
        );
        let mut interface = QisHeliosInterface::new();
        interface
            .load_program(program.as_bytes(), ProgramFormat::LlvmIrText)
            .unwrap();
        let library = QisHeliosInterface::get_or_cache_program_lib(
            interface.executable_path.as_ref().unwrap(),
        )
        .unwrap();
        unsafe {
            let register: Symbol<RegisterExecutionContextFn> =
                ffi.get(b"pecos_register_execution_context\0").unwrap();
            let enable: Symbol<EnableDynamicModeFn> = ffi
                .get(b"pecos_enable_dynamic_mode_with_context\0")
                .unwrap();
            let native: Symbol<unsafe extern "C" fn(*mut ExecutionContext, bool) -> i32> =
                ffi.get(b"pecos_set_native_selene_handles\0").unwrap();
            let read: Symbol<unsafe extern "C" fn() -> u64> =
                library.get(b"read_observed\0").unwrap();
            let clear: Symbol<unsafe extern "C" fn()> = library.get(b"clear_observed\0").unwrap();
            // Failure after readiness must stop the program; a fresh context on
            // the same thread must subsequently succeed with the same library.
            for fail in [false, true, false] {
                clear();
                let context = QisHeliosInterface::create_execution_context(ffi.inner()).unwrap();
                register(context.0);
                let _registration = ExecutionRegistration(*register);
                let reset: Symbol<ResetInterfaceFn> =
                    ffi.get(b"pecos_qis_reset_interface\0").unwrap();
                reset();
                assert_eq!(enable(context.0), 0);
                assert_eq!(native(context.0, true), 0);
                let mut host = Host {
                    context: context.0,
                    drain: *ffi
                        .get(b"pecos_get_pending_operations_with_context\0")
                        .unwrap(),
                    free: *ffi.get(b"pecos_free_operations\0").unwrap(),
                    store: *ffi
                        .get(b"pecos_set_measurement_outcome_with_context\0")
                        .unwrap(),
                    ready: *ffi
                        .get(b"pecos_signal_result_ready_with_context\0")
                        .unwrap(),
                    caller: std::thread::current().id(),
                    calls: 0,
                    fail,
                    panic: None,
                };
                let userdata = (&raw mut host).cast();
                let result = if is_void {
                    let guard: Symbol<VGuard> = ffi
                        .get(b"pecos_call_void_main_with_request_callback_v1\0")
                        .unwrap();
                    guard(
                        context.0,
                        *library.get(b"main\0").unwrap(),
                        service,
                        userdata,
                    )
                } else {
                    let guard: Symbol<QGuard> = ffi
                        .get(b"pecos_call_qmain_with_request_callback_v1\0")
                        .unwrap();
                    guard(
                        context.0,
                        *library.get(b"qmain\0").unwrap(),
                        service,
                        userdata,
                    )
                };
                if let Some(payload) = host.panic {
                    std::panic::resume_unwind(payload);
                }
                assert_eq!(result, u64::from(fail));
                assert_eq!(host.calls, if fail { 1 } else { 2 });
                assert_eq!(read(), if fail { 0 } else { (1_u64 << 60) + 2 });
            }
        }
    }
}
