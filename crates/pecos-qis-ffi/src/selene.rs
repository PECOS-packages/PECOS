//! Selene ABI adapters sharing the QIS runtime's context and recovery guard.
//!
//! All transfer-capable frames use C-unwind for Windows longjmp. Adapters keep
//! only plain ABI values live across calls: Unix longjmp skips destructors.
//! Instance pointers and string ownership flags are ignored, as in the Selene ABI.

use crate::ffi;
use pecos_qis_ffi_types::{Operation, RUNTIME_REQUEST_RESULT_ID, RuntimeRequest};
use std::ffi::{c_char, c_int};
use std::sync::atomic::Ordering;

/// Opaque Selene instance; operation collection uses thread-local state.
#[repr(C)]
pub struct SeleneInstance {
    pub dummy: c_int,
}

/// Selene void result in the C ABI.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SeleneVoidResult {
    pub error_code: u32,
}

/// Selene u64 result in the C ABI.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SeleneU64Result {
    pub error_code: u32,
    pub value: u64,
}

/// Selene u32 result in the C ABI.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SeleneU32Result {
    pub error_code: u32,
    pub value: u32,
}

/// Selene f64 result in the C ABI.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SeleneF64Result {
    pub error_code: u32,
    pub value: f64,
}

/// Selene bool result in the C ABI.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SeleneBoolResult {
    pub error_code: u32,
    pub value: bool,
}

/// Selene future result in the C ABI.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SeleneFutureResult {
    pub error_code: u32,
    pub reference: u64,
}

/// String passed by value; `owned` does not transfer ownership to this runtime.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SeleneString {
    pub data: *const c_char,
    pub length: u64,
    pub owned: bool,
}

// C converts uint64_t lengths to size_t modulo the pointer width.
fn c_size_t(length: u64) -> usize {
    usize::try_from(length & (usize::MAX as u64)).expect("masked to size_t")
}

/// Allocate a qubit, preserving the negative-ID allocation error.
///
/// # Safety
/// Instance pointers are ignored. Execution must obey the non-nested guard
/// and recovery discipline documented in this module.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn selene_qalloc(_instance: *mut SeleneInstance) -> SeleneU64Result {
    let id = unsafe { ffi::__quantum__rt__qubit_allocate() };
    if id < 0 {
        return SeleneU64Result {
            error_code: 100_000,
            value: 0,
        };
    }
    SeleneU64Result {
        error_code: 0,
        value: id.cast_unsigned(),
    }
}

/// Forward qfree to the QIS operation collector.
///
/// # Safety
/// Instance pointers are ignored. Execution must obey the non-nested guard
/// and recovery discipline documented in this module.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn selene_qfree(
    _instance: *mut SeleneInstance,
    q: u64,
) -> SeleneVoidResult {
    unsafe { ffi::__quantum__rt__qubit_release(q.cast_signed()) };
    SeleneVoidResult { error_code: 0 }
}

/// Forward rxy to the QIS operation collector.
///
/// # Safety
/// Instance pointers are ignored. Execution must obey the non-nested guard
/// and recovery discipline documented in this module.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn selene_rxy(
    _instance: *mut SeleneInstance,
    q: u64,
    theta: f64,
    phi: f64,
) -> SeleneVoidResult {
    unsafe { ffi::__quantum__qis__r1xy__body(theta, phi, q.cast_signed()) };
    SeleneVoidResult { error_code: 0 }
}

/// Forward rz to the QIS operation collector.
///
/// # Safety
/// Instance pointers are ignored. Execution must obey the non-nested guard
/// and recovery discipline documented in this module.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn selene_rz(
    _instance: *mut SeleneInstance,
    q: u64,
    theta: f64,
) -> SeleneVoidResult {
    unsafe { ffi::__quantum__qis__rz__body(theta, q.cast_signed()) };
    SeleneVoidResult { error_code: 0 }
}

/// Forward rzz to the QIS operation collector.
///
/// # Safety
/// Instance pointers are ignored. Execution must obey the non-nested guard
/// and recovery discipline documented in this module.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn selene_rzz(
    _instance: *mut SeleneInstance,
    q1: u64,
    q2: u64,
    theta: f64,
) -> SeleneVoidResult {
    unsafe { ffi::__quantum__qis__rzz__body(theta, q1.cast_signed(), q2.cast_signed()) };
    SeleneVoidResult { error_code: 0 }
}

/// Forward `qubit_reset` to the QIS operation collector.
///
/// # Safety
/// Instance pointers are ignored. Execution must obey the non-nested guard
/// and recovery discipline documented in this module.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn selene_qubit_reset(
    _instance: *mut SeleneInstance,
    q: u64,
) -> SeleneVoidResult {
    unsafe { ffi::__quantum__qis__reset__body(q.cast_signed()) };
    SeleneVoidResult { error_code: 0 }
}

/// Allocate a result and measure immediately.
///
/// # Safety
/// Instance pointers are ignored. Execution must obey the non-nested guard
/// and recovery discipline documented in this module.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn selene_qubit_measure(
    _instance: *mut SeleneInstance,
    q: u64,
) -> SeleneBoolResult {
    let result = unsafe { ffi::__quantum__rt__result_allocate() };
    let placeholder = unsafe { ffi::__quantum__qis__m__body(q.cast_signed(), result) };
    let value = if crate::is_dynamic_mode_active() {
        i32::from(unsafe { ffi::___read_future_bool(result) })
    } else {
        placeholder
    };
    SeleneBoolResult {
        error_code: 0,
        value: value != 0,
    }
}

/// Measure and return the allocated result reference.
///
/// # Safety
/// Instance pointers are ignored. Execution must obey the non-nested guard
/// and recovery discipline documented in this module.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn selene_qubit_lazy_measure(
    _instance: *mut SeleneInstance,
    q: u64,
) -> SeleneFutureResult {
    let result = unsafe { ffi::__quantum__rt__result_allocate() };
    unsafe { ffi::__quantum__qis__m__body(q.cast_signed(), result) };
    SeleneFutureResult {
        error_code: 0,
        reference: result.cast_unsigned(),
    }
}

/// Allocate and queue a leakage-aware measurement future.
///
/// # Safety
/// Instance pointers are ignored. Execution must obey the non-nested guard
/// and recovery discipline documented in this module.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn selene_qubit_lazy_measure_leaked(
    _instance: *mut SeleneInstance,
    q: u64,
) -> SeleneFutureResult {
    let result = unsafe { ffi::___lazy_measure_leaked(q.cast_signed()) };
    SeleneFutureResult {
        error_code: 0,
        reference: result.cast_unsigned(),
    }
}

/// Read a measurement future through the dynamic execution interface.
///
/// # Safety
/// Instance pointers are ignored. Execution must obey the non-nested guard
/// and recovery discipline documented in this module.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn selene_future_read_bool(
    _instance: *mut SeleneInstance,
    r: u64,
) -> SeleneBoolResult {
    let value = unsafe { ffi::__quantum__rt__result_get_one(r.cast_signed()) };
    SeleneBoolResult {
        error_code: 0,
        value: value != 0,
    }
}

/// Read a measurement future through the dynamic execution interface.
///
/// # Safety
/// Instance pointers are ignored. Execution must obey the non-nested guard
/// and recovery discipline documented in this module.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn selene_future_read_u64(
    _instance: *mut SeleneInstance,
    r: u64,
) -> SeleneU64Result {
    let value = if crate::is_dynamic_mode_active() {
        unsafe { ffi::___read_future_uint(r.cast_signed()) }
    } else {
        i64::from(unsafe { ffi::__quantum__rt__result_get_one(r.cast_signed()) }).cast_unsigned()
    };
    SeleneU64Result {
        error_code: 0,
        value,
    }
}

/// Forward the future reference-count increment to the runtime.
///
/// # Safety
/// Instance pointers are ignored. Execution must obey the non-nested guard
/// and recovery discipline documented in this module.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn selene_refcount_increment(
    _instance: *mut SeleneInstance,
    r: u64,
) -> SeleneVoidResult {
    let queued = queue_runtime_request(
        usize::try_from(r)
            .map(|result_id| RuntimeRequest::IncrementFutureRefcount { result_id })
            .map_err(|_| "future reference exceeds usize".into()),
    );
    SeleneVoidResult {
        error_code: unsafe { finish_runtime_request(queued, "selene_refcount_increment") }
            .error_code,
    }
}

/// Forward the future reference-count decrement to the runtime.
///
/// # Safety
/// Instance pointers are ignored. Execution must obey the non-nested guard
/// and recovery discipline documented in this module.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn selene_refcount_decrement(
    _instance: *mut SeleneInstance,
    r: u64,
) -> SeleneVoidResult {
    let queued = queue_runtime_request(
        usize::try_from(r)
            .map(|result_id| RuntimeRequest::DecrementFutureRefcount { result_id })
            .map_err(|_| "future reference exceeds usize".into()),
    );
    SeleneVoidResult {
        error_code: unsafe { finish_runtime_request(queued, "selene_refcount_decrement") }
            .error_code,
    }
}

/// Record typed named output.
///
/// # Safety
/// `tag.data` must reference `tag.length` readable bytes (or be null).
/// Callers must follow the execution recovery discipline.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn selene_print_bool(
    _instance: *mut SeleneInstance,
    tag: SeleneString,
    value: bool,
) -> SeleneVoidResult {
    unsafe { ffi::print_bool_selene(tag.data.cast(), tag.length.cast_signed(), value) };
    SeleneVoidResult { error_code: 0 }
}

/// Record typed named output.
///
/// # Safety
/// `tag.data` must reference `tag.length` readable bytes (or be null).
/// `ptr` must reference `length` aligned, valid elements, or be null for empty input.
/// Callers must follow the execution recovery discipline.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn selene_print_bool_array(
    _instance: *mut SeleneInstance,
    tag: SeleneString,
    ptr: *const bool,
    length: u64,
) -> SeleneVoidResult {
    unsafe { ffi::print_bool_arr_selene(tag.data.cast(), tag.length.cast_signed(), ptr, length) };
    SeleneVoidResult { error_code: 0 }
}

/// Record typed named output.
///
/// # Safety
/// `tag.data` must reference `tag.length` readable bytes (or be null).
/// Callers must follow the execution recovery discipline.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn selene_print_i64(
    _instance: *mut SeleneInstance,
    tag: SeleneString,
    value: i64,
) -> SeleneVoidResult {
    unsafe { ffi::print_int_selene(tag.data.cast(), tag.length.cast_signed(), value) };
    SeleneVoidResult { error_code: 0 }
}

/// Record typed named output.
///
/// # Safety
/// `tag.data` must reference `tag.length` readable bytes (or be null).
/// `ptr` must reference `length` aligned, valid elements, or be null for empty input.
/// Callers must follow the execution recovery discipline.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn selene_print_i64_array(
    _instance: *mut SeleneInstance,
    tag: SeleneString,
    ptr: *const i64,
    length: u64,
) -> SeleneVoidResult {
    unsafe { ffi::print_int_arr_selene(tag.data.cast(), tag.length.cast_signed(), ptr, length) };
    SeleneVoidResult { error_code: 0 }
}

/// Record typed named output.
///
/// # Safety
/// `tag.data` must reference `tag.length` readable bytes (or be null).
/// Callers must follow the execution recovery discipline.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn selene_print_u64(
    _instance: *mut SeleneInstance,
    tag: SeleneString,
    value: u64,
) -> SeleneVoidResult {
    unsafe { ffi::print_uint_selene(tag.data.cast(), tag.length.cast_signed(), value) };
    SeleneVoidResult { error_code: 0 }
}

/// Record typed named output.
///
/// # Safety
/// `tag.data` must reference `tag.length` readable bytes (or be null).
/// `ptr` must reference `length` aligned, valid elements, or be null for empty input.
/// Callers must follow the execution recovery discipline.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn selene_print_u64_array(
    _instance: *mut SeleneInstance,
    tag: SeleneString,
    ptr: *const u64,
    length: u64,
) -> SeleneVoidResult {
    unsafe { ffi::print_uint_arr_selene(tag.data.cast(), tag.length.cast_signed(), ptr, length) };
    SeleneVoidResult { error_code: 0 }
}

/// Record typed named output.
///
/// # Safety
/// `tag.data` must reference `tag.length` readable bytes (or be null).
/// Callers must follow the execution recovery discipline.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn selene_print_f64(
    _instance: *mut SeleneInstance,
    tag: SeleneString,
    value: f64,
) -> SeleneVoidResult {
    unsafe { ffi::print_float_selene(tag.data.cast(), tag.length.cast_signed(), value) };
    SeleneVoidResult { error_code: 0 }
}

/// Record typed named output.
///
/// # Safety
/// `tag.data` must reference `tag.length` readable bytes (or be null).
/// `ptr` must reference `length` aligned, valid elements, or be null for empty input.
/// Callers must follow the execution recovery discipline.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn selene_print_f64_array(
    _instance: *mut SeleneInstance,
    tag: SeleneString,
    ptr: *const f64,
    length: u64,
) -> SeleneVoidResult {
    unsafe { ffi::print_float_arr_selene(tag.data.cast(), tag.length.cast_signed(), ptr, length) };
    SeleneVoidResult { error_code: 0 }
}

/// Record termination and transfer to the current execution guard, if present.
///
/// # Safety
/// The message must be null or reference `length` readable bytes.
/// Callers must release owned values and borrows before a possible transfer.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn selene_print_panic(
    _instance: *mut SeleneInstance,
    message: SeleneString,
    error_code: u32,
) -> SeleneVoidResult {
    unsafe {
        ffi::pecos_record_program_panic(
            error_code.cast_signed(),
            message.data.cast(),
            c_size_t(message.length),
        );
    };
    // Recording has released its owned values and locks before transfer.
    if let Some(transfer) = ffi::PROGRAM_PANIC_TRANSFER.get() {
        unsafe { transfer() };
    }
    SeleneVoidResult { error_code: 0 }
}

/// State dumping requires simulator support and is currently unsupported.
///
/// # Safety
/// Instance pointers are ignored. Execution must obey the non-nested guard
/// and recovery discipline documented in this module.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn selene_dump_state(
    _instance: *mut SeleneInstance,
    _message: SeleneString,
    _qubits: *const u64,
    _qubits_length: u64,
) -> SeleneVoidResult {
    unsafe {
        ffi::fatal_ffi_input(
            "selene_dump_state",
            "state dumping is unsupported by the PECOS execution interface".into(),
        );
    };
    SeleneVoidResult { error_code: 1 }
}

/// Store the program's time cursor, as in Selene; this does not schedule work.
///
/// # Safety
/// Instance pointers are ignored. Execution must obey the non-nested guard
/// and recovery discipline documented in this module.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn selene_set_tc(
    _instance: *mut SeleneInstance,
    time_cursor: u64,
) -> SeleneVoidResult {
    if let Some(ctx) = crate::get_execution_context() {
        unsafe { &*ctx }
            .time_cursor
            .store(time_cursor, Ordering::SeqCst);
        SeleneVoidResult { error_code: 0 }
    } else {
        SeleneVoidResult { error_code: 1 }
    }
}

// Copy ABI input before waiting or transferring to the C guard. No borrowed
// buffers, owned payloads, or locks may remain on transfer-capable frames.
unsafe fn copy_input<T: Copy>(data: *const T, length: u64) -> Result<Vec<T>, String> {
    let count = usize::try_from(length).map_err(|_| "input length exceeds usize")?;
    if count > (isize::MAX as usize) / std::mem::size_of::<T>() {
        return Err("input length exceeds isize::MAX bytes".into());
    }
    if count == 0 {
        return Ok(Vec::new());
    }
    if data.is_null() || !data.is_aligned() {
        return Err("nonempty input has a null or unaligned pointer".into());
    }
    Ok(unsafe { std::slice::from_raw_parts(data, count) }.to_vec())
}

fn queue_runtime_request(request: Result<RuntimeRequest, String>) -> Result<Option<usize>, String> {
    let request = request?;
    let dynamic = crate::is_dynamic_mode_active();
    if !dynamic && matches!(request, RuntimeRequest::Custom { .. }) {
        return Err(
            "custom runtime calls require dynamic execution to obtain the runtime's response"
                .into(),
        );
    }
    let result_id = if dynamic {
        let ctx = crate::get_execution_context().ok_or("no execution context")?;
        // Requests are synchronous, so one separate response slot is enough.
        // Never allocate a program result: that would change Guppy measurement
        // identities and the DEM/decoder binding merely by making a host call.
        *unsafe { &*ctx }
            .runtime_response
            .lock()
            .map_err(|_| "poisoned runtime response")? = None;
        Some(RUNTIME_REQUEST_RESULT_ID)
    } else {
        None
    };
    crate::with_interface(|interface| {
        interface.queue_operation(Operation::RuntimeRequest { request, result_id });
    });
    Ok(result_id)
}

// This frame owns only scalar state when it transfers. Runtime request reads
// deliberately do not record measurement provenance for named outputs.
unsafe fn finish_runtime_request(
    queued: Result<Option<usize>, String>,
    entry: &str,
) -> SeleneU64Result {
    let id = match queued {
        Ok(Some(id)) => id,
        Ok(None) => {
            return SeleneU64Result {
                error_code: 0,
                value: 0,
            };
        }
        Err(reason) => {
            unsafe { ffi::fatal_ffi_input(entry, reason) };
            return SeleneU64Result {
                error_code: 1,
                value: 0,
            };
        }
    };
    if let Some(value) = crate::read_dynamic_result(id as u64, false) {
        return SeleneU64Result {
            error_code: 0,
            value,
        };
    }
    if let Some(transfer) = ffi::PROGRAM_PANIC_TRANSFER.get() {
        unsafe { transfer() };
    }
    SeleneU64Result {
        error_code: 1,
        value: 0,
    }
}

/// Notify the runtime of elapsed classical time, in integer nanoseconds.
///
/// # Safety
/// Instance pointers are ignored. Execution must obey the non-nested guard
/// and recovery discipline documented in this module.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn selene_simulate_delay(
    _instance: *mut SeleneInstance,
    nanoseconds: u64,
) -> SeleneVoidResult {
    let queued = queue_runtime_request(Ok(RuntimeRequest::SimulateDelay { nanoseconds }));
    SeleneVoidResult {
        error_code: unsafe { finish_runtime_request(queued, "selene_simulate_delay") }.error_code,
    }
}

/// Forward a scoped barrier and its nanosecond delay to the runtime.
///
/// # Safety
/// A nonempty qubit buffer must contain the specified number of readable IDs.
/// Instance pointers are ignored. Execution must obey the non-nested guard
/// and recovery discipline documented in this module.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn selene_local_barrier(
    _instance: *mut SeleneInstance,
    qubit_ids: *const u64,
    qubit_ids_length: u64,
    sleep_time: u64,
) -> SeleneVoidResult {
    let queued = queue_runtime_request(
        unsafe { copy_input(qubit_ids, qubit_ids_length) }.and_then(|ids| {
            let qubits = ids
                .into_iter()
                .map(|id| usize::try_from(id).map_err(|_| "qubit ID exceeds usize".into()))
                .collect::<Result<Vec<_>, String>>()?;
            Ok(RuntimeRequest::LocalBarrier {
                qubits,
                sleep_ns: sleep_time,
            })
        }),
    );
    SeleneVoidResult {
        error_code: unsafe { finish_runtime_request(queued, "selene_local_barrier") }.error_code,
    }
}

/// Forward a global barrier and its nanosecond delay to the runtime.
///
/// # Safety
/// Instance pointers are ignored. Execution must obey the non-nested guard
/// and recovery discipline documented in this module.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn selene_global_barrier(
    _instance: *mut SeleneInstance,
    sleep_time: u64,
) -> SeleneVoidResult {
    let queued = queue_runtime_request(Ok(RuntimeRequest::GlobalBarrier {
        sleep_ns: sleep_time,
    }));
    SeleneVoidResult {
        error_code: unsafe { finish_runtime_request(queued, "selene_global_barrier") }.error_code,
    }
}

/// Instance teardown is a no-op; return success.
///
/// # Safety
/// Instance pointers are ignored. Execution must obey the non-nested guard
/// and recovery discipline documented in this module.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn selene_exit(_instance: *mut SeleneInstance) -> SeleneVoidResult {
    SeleneVoidResult { error_code: 0 }
}

/// Read the program time cursor, independently of the runtime clock.
///
/// # Safety
/// Instance pointers are ignored. Execution must obey the non-nested guard
/// and recovery discipline documented in this module.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn selene_get_tc(_instance: *mut SeleneInstance) -> SeleneU64Result {
    if let Some(ctx) = crate::get_execution_context() {
        SeleneU64Result {
            error_code: 0,
            value: unsafe { &*ctx }.time_cursor.load(Ordering::SeqCst),
        }
    } else {
        SeleneU64Result {
            error_code: 1,
            value: 0,
        }
    }
}

/// Return the host-assigned job-wide program shot index.
///
/// # Safety
/// Instance pointers are ignored. Execution must obey the non-nested guard
/// and recovery discipline documented in this module.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn selene_get_current_shot(
    _instance: *mut SeleneInstance,
) -> SeleneU64Result {
    SeleneU64Result {
        error_code: 0,
        value: unsafe { ffi::get_current_shot() },
    }
}

/// Return one shot per PECOS program invocation.
///
/// # Safety
/// Instance pointers are ignored. Execution must obey the non-nested guard
/// and recovery discipline documented in this module.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn selene_shot_count(
    _instance: *mut SeleneInstance,
) -> SeleneU64Result {
    SeleneU64Result {
        error_code: 0,
        value: 1,
    }
}

/// Reset the random stream and release outstanding program allocations.
/// The host executes one program shot per invocation and owns its global index;
/// the compiled wrapper's local shot argument must not overwrite that identity.
///
/// # Safety
/// Instance pointers are ignored. Execution must obey the non-nested guard
/// and recovery discipline documented in this module.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn selene_on_shot_start(
    _instance: *mut SeleneInstance,
    _shot_index: u64,
) -> SeleneVoidResult {
    ffi::pecos_reset_program_rng();
    crate::pecos_cleanup_program_allocations();
    SeleneVoidResult { error_code: 0 }
}

/// Release outstanding program allocations at shot end.
///
/// # Safety
/// Instance pointers are ignored. Execution must obey the non-nested guard
/// and recovery discipline documented in this module.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn selene_on_shot_end(
    _instance: *mut SeleneInstance,
) -> SeleneVoidResult {
    crate::pecos_cleanup_program_allocations();
    SeleneVoidResult { error_code: 0 }
}

/// Return a static dummy instance; the configuration path is ignored.
///
/// # Safety
/// `instance` must point to writable storage for one instance pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn selene_load_config(
    instance: *mut *mut SeleneInstance,
    _config_file: *const c_char,
) -> SeleneVoidResult {
    static mut INSTANCE: SeleneInstance = SeleneInstance { dummy: 0 };
    unsafe { *instance = &raw mut INSTANCE };
    SeleneVoidResult { error_code: 0 }
}

// Use the C stream, including its Windows text mode and raw-byte behavior.
// Rust's stderr writer can require UTF-8 when attached to a Windows console.
#[cfg(unix)]
unsafe fn stderr_stream() -> *mut libc::FILE {
    unsafe extern "C" {
        #[cfg_attr(target_vendor = "apple", link_name = "__stderrp")]
        static stderr: *mut libc::FILE;
    }
    unsafe { stderr }
}

#[cfg(windows)]
unsafe fn stderr_stream() -> *mut libc::FILE {
    unsafe extern "C" {
        fn __acrt_iob_func(index: u32) -> *mut libc::FILE;
    }
    unsafe { __acrt_iob_func(2) }
}

/// Write the diagnostic to stderr without recording termination or transferring.
///
/// # Safety
/// The message must be readable up to `length` bytes or the first NUL.
/// For unsupported lengths above `i32::MAX`, it must be NUL-terminated.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn selene_print_exit(
    _instance: *mut SeleneInstance,
    message: SeleneString,
    error_code: u32,
) -> SeleneVoidResult {
    // A negative precision reads to the first NUL for unsupported long inputs.
    // Never construct a Rust slice extending past an early NUL.
    let precision = i32::try_from(message.length).unwrap_or(-1);
    unsafe {
        libc::fprintf(
            stderr_stream(),
            c"EXIT [%u]: %.*s\n".as_ptr(),
            error_code,
            precision,
            message.data,
        );
    }
    SeleneVoidResult { error_code: 0 }
}

/// Update the shared program random stream.
///
/// # Safety
/// Instance pointers are ignored. Execution must obey the non-nested guard
/// and recovery discipline documented in this module.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn selene_random_seed(
    _instance: *mut SeleneInstance,
    seed: u64,
) -> SeleneVoidResult {
    unsafe { ffi::random_seed_selene(seed) };
    SeleneVoidResult { error_code: 0 }
}

/// Update the shared program random stream.
///
/// # Safety
/// Instance pointers are ignored. Execution must obey the non-nested guard
/// and recovery discipline documented in this module.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn selene_random_advance(
    _instance: *mut SeleneInstance,
    delta: u64,
) -> SeleneVoidResult {
    unsafe { ffi::random_advance_selene(delta) };
    SeleneVoidResult { error_code: 0 }
}

/// Draw from the shared program random stream.
///
/// # Safety
/// Instance pointers are ignored. Execution must obey the non-nested guard
/// and recovery discipline documented in this module.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn selene_random_u32(
    _instance: *mut SeleneInstance,
) -> SeleneU32Result {
    let value = unsafe { ffi::random_u32_selene() };
    SeleneU32Result {
        error_code: 0,
        value,
    }
}

/// Draw from the shared program random stream.
///
/// # Safety
/// Instance pointers are ignored. Execution must obey the non-nested guard
/// and recovery discipline documented in this module.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn selene_random_u32_bounded(
    _instance: *mut SeleneInstance,
    bound: u32,
) -> SeleneU32Result {
    let value = unsafe { ffi::random_u32_bounded_selene(bound) };
    SeleneU32Result {
        error_code: 0,
        value,
    }
}

/// Draw from the shared program random stream.
///
/// # Safety
/// Instance pointers are ignored. Execution must obey the non-nested guard
/// and recovery discipline documented in this module.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn selene_random_f64(
    _instance: *mut SeleneInstance,
) -> SeleneF64Result {
    let value = unsafe { ffi::random_f64_selene() };
    SeleneF64Result {
        error_code: 0,
        value,
    }
}

/// Forward opaque custom data and return the runtime response after processing its work.
///
/// # Safety
/// A nonempty data buffer must contain the specified number of readable bytes.
/// Instance pointers are ignored. Execution must obey the non-nested guard
/// and recovery discipline documented in this module.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn selene_custom_runtime_call(
    _instance: *mut SeleneInstance,
    tag: u64,
    data: *const u8,
    data_length: u64,
) -> SeleneU64Result {
    let queued = queue_runtime_request(
        unsafe { copy_input(data, data_length) }.map(|data| RuntimeRequest::Custom { tag, data }),
    );
    unsafe { finish_runtime_request(queued, "selene_custom_runtime_call") }
}

unsafe extern "C-unwind" {
    fn pecos_guard_qmain_with_setjmp(qmain: unsafe extern "C" fn(u64) -> u64) -> u64;
    fn pecos_guard_void_main_with_setjmp(main: unsafe extern "C" fn()) -> u64;
}

/// Run a program with the C recovery guard and shot lifecycle.
///
/// # Safety
/// The entry point must be valid and obey the recovery discipline. Nested
/// guards and re-entrant calls holding context locks are unsupported.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pecos_call_qmain_with_setjmp(
    qmain: unsafe extern "C" fn(u64) -> u64,
) -> u64 {
    // This frame is outside the setjmp/transfer interval. The C-unwind import
    // lets an escaping program panic reach this C boundary, where it aborts.
    unsafe { pecos_guard_qmain_with_setjmp(qmain) }
}

/// Run a program with the C recovery guard and shot lifecycle.
///
/// # Safety
/// The entry point must be valid and obey the recovery discipline. Nested
/// guards and re-entrant calls holding context locks are unsupported.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pecos_call_void_main_with_setjmp(main: unsafe extern "C" fn()) -> u64 {
    // This frame is outside the setjmp/transfer interval. The C-unwind import
    // lets an escaping program panic reach this C boundary, where it aborts.
    unsafe { pecos_guard_void_main_with_setjmp(main) }
}

#[cfg(all(test, target_pointer_width = "64"))]
mod tests {
    use super::*;
    use std::mem::{align_of, offset_of, size_of};

    struct Context(crate::ExecutionContext);
    impl Context {
        fn register(&mut self) {
            unsafe { crate::pecos_register_execution_context(&raw mut self.0) };
            crate::with_interface(pecos_qis_ffi_types::OperationCollector::reset);
        }
    }
    impl Drop for Context {
        fn drop(&mut self) {
            unsafe { crate::pecos_register_execution_context(std::ptr::null_mut()) };
        }
    }

    #[test]
    fn time_cursor_is_context_local_and_resets_between_shots() {
        let mut first = Context(crate::ExecutionContext::new());
        let mut second = Context(crate::ExecutionContext::new());
        first.register();
        unsafe {
            assert_eq!(selene_set_tc(std::ptr::null_mut(), u64::MAX).error_code, 0);
            assert_eq!(selene_get_tc(std::ptr::null_mut()).value, u64::MAX);
            second.register();
            assert_eq!(selene_get_tc(std::ptr::null_mut()).value, 0);
            first.register();
            assert_eq!(selene_get_tc(std::ptr::null_mut()).value, u64::MAX);
            first.0.reset_outputs();
            assert_eq!(selene_get_tc(std::ptr::null_mut()).value, 0);
        }
    }

    #[test]
    fn barriers_preserve_scope_and_integer_nanoseconds_in_collection() {
        let mut ctx = Context(crate::ExecutionContext::new());
        ctx.register();
        let qubits = [3, 1];
        unsafe {
            assert_eq!(
                selene_local_barrier(std::ptr::null_mut(), qubits.as_ptr(), 2, 23).error_code,
                0
            );
            assert_eq!(
                selene_global_barrier(std::ptr::null_mut(), u64::MAX).error_code,
                0
            );
        }
        assert_eq!(
            crate::with_interface(pecos_qis_ffi_types::OperationCollector::take_operations),
            [
                Operation::RuntimeRequest {
                    request: RuntimeRequest::LocalBarrier {
                        qubits: vec![3, 1],
                        sleep_ns: 23
                    },
                    result_id: None
                },
                Operation::RuntimeRequest {
                    request: RuntimeRequest::GlobalBarrier { sleep_ns: u64::MAX },
                    result_id: None
                },
            ]
        );
    }

    #[test]
    fn delay_and_refcounts_are_collected_in_source_order() {
        let mut ctx = Context(crate::ExecutionContext::new());
        ctx.register();
        unsafe {
            assert_eq!(
                selene_simulate_delay(std::ptr::null_mut(), u64::MAX).error_code,
                0
            );
            assert_eq!(
                selene_refcount_increment(std::ptr::null_mut(), 7).error_code,
                0
            );
            assert_eq!(
                selene_refcount_decrement(std::ptr::null_mut(), 7).error_code,
                0
            );
        }
        let expected = [
            RuntimeRequest::SimulateDelay {
                nanoseconds: u64::MAX,
            },
            RuntimeRequest::IncrementFutureRefcount { result_id: 7 },
            RuntimeRequest::DecrementFutureRefcount { result_id: 7 },
        ]
        .map(|request| Operation::RuntimeRequest {
            request,
            result_id: None,
        });
        assert_eq!(
            crate::with_interface(pecos_qis_ffi_types::OperationCollector::take_operations),
            expected
        );
    }

    #[test]
    fn unsupported_diagnostics_and_static_custom_calls_fail() {
        let mut ctx = Context(crate::ExecutionContext::new());
        ctx.register();
        unsafe {
            assert_ne!(
                selene_dump_state(
                    std::ptr::null_mut(),
                    SeleneString {
                        data: std::ptr::null(),
                        length: 0,
                        owned: false
                    },
                    std::ptr::null(),
                    0
                )
                .error_code,
                0
            );
            assert!(
                ctx.0
                    .program_error
                    .lock()
                    .unwrap()
                    .as_ref()
                    .unwrap()
                    .to_string()
                    .contains("state dumping is unsupported")
            );
            ctx.0.reset_outputs();
            assert_ne!(
                selene_custom_runtime_call(std::ptr::null_mut(), 0, std::ptr::null(), 0).error_code,
                0
            );
            assert!(
                ctx.0
                    .program_error
                    .lock()
                    .unwrap()
                    .as_ref()
                    .unwrap()
                    .to_string()
                    .contains("require dynamic execution")
            );
        }
    }

    #[test]
    fn request_responses_do_not_consume_or_reuse_measurement_slots() {
        let mut ctx = Context(crate::ExecutionContext::new());
        ctx.register();
        ctx.0.dynamic_mode_active.store(true, Ordering::SeqCst);
        crate::queue_measurement(0, 0, false);
        let id = queue_runtime_request(Ok(RuntimeRequest::GlobalBarrier { sleep_ns: 0 })).unwrap();
        assert_eq!(id, Some(RUNTIME_REQUEST_RESULT_ID));
        assert_eq!(
            crate::with_interface(pecos_qis_ffi_types::OperationCollector::allocate_result),
            0
        );
        assert_eq!(*ctx.0.measured_results.lock().unwrap(), [0].into());
        unsafe {
            crate::pecos_set_measurement_outcome_with_context(
                &raw mut ctx.0,
                RUNTIME_REQUEST_RESULT_ID as u64,
                u64::MAX,
            )
        };
        assert_eq!(*ctx.0.runtime_response.lock().unwrap(), Some(u64::MAX));
        assert!(ctx.0.measurement_results.lock().unwrap().is_empty());
        queue_runtime_request(Ok(RuntimeRequest::GlobalBarrier { sleep_ns: 0 })).unwrap();
        assert_eq!(*ctx.0.runtime_response.lock().unwrap(), None);
        assert_eq!(ctx.0.pending_result_reads.lock().unwrap().len(), 0);
    }

    #[test]
    fn invalid_request_buffers_fail_without_queuing() {
        let mut ctx = Context(crate::ExecutionContext::new());
        ctx.register();
        unsafe {
            assert_ne!(
                selene_local_barrier(std::ptr::null_mut(), std::ptr::null(), 1, 0).error_code,
                0
            );
            assert_ne!(
                selene_custom_runtime_call(std::ptr::null_mut(), 0, std::ptr::null(), u64::MAX)
                    .error_code,
                0
            );
        }
        assert_eq!(
            crate::with_interface(pecos_qis_ffi_types::OperationCollector::take_operations),
            []
        );
    }

    #[test]
    fn c_abi_layouts() {
        // Fixed values independently asserted by the C executor fixture.
        macro_rules! layout {
            ($ty:ty, $size:expr, $align:expr, $field:ident, $offset:expr) => {
                assert_eq!(size_of::<$ty>(), $size);
                assert_eq!(align_of::<$ty>(), $align);
                assert_eq!(offset_of!($ty, $field), $offset);
            };
        }
        layout!(SeleneInstance, 4, 4, dummy, 0);
        layout!(SeleneVoidResult, 4, 4, error_code, 0);
        layout!(SeleneU64Result, 16, 8, value, 8);
        layout!(SeleneU32Result, 8, 4, value, 4);
        layout!(SeleneF64Result, 16, 8, value, 8);
        layout!(SeleneBoolResult, 8, 4, value, 4);
        layout!(SeleneFutureResult, 16, 8, reference, 8);
        assert_eq!(offset_of!(SeleneU64Result, error_code), 0);
        assert_eq!(offset_of!(SeleneU32Result, error_code), 0);
        assert_eq!(offset_of!(SeleneF64Result, error_code), 0);
        assert_eq!(offset_of!(SeleneBoolResult, error_code), 0);
        assert_eq!(offset_of!(SeleneFutureResult, error_code), 0);
        layout!(SeleneString, 24, 8, data, 0);
        assert_eq!(offset_of!(SeleneString, length), 8);
        assert_eq!(offset_of!(SeleneString, owned), 16);
    }
}
