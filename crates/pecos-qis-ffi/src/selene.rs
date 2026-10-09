//! Selene ABI adapters sharing the QIS runtime's context and recovery guard.
//!
//! All transfer-capable frames use C-unwind for Windows longjmp. Adapters keep
//! only plain ABI values live across calls: Unix longjmp skips destructors.
//! Instance pointers and string ownership flags are ignored, as in the Selene ABI.

use crate::ffi;
use std::ffi::{c_char, c_int};

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

/// Reference counting is a no-op in operation collection mode.
///
/// # Safety
/// Instance pointers are ignored. Execution must obey the non-nested guard
/// and recovery discipline documented in this module.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn selene_refcount_increment(
    _instance: *mut SeleneInstance,
    _r: u64,
) -> SeleneVoidResult {
    SeleneVoidResult { error_code: 0 }
}

/// Reference counting is a no-op in operation collection mode.
///
/// # Safety
/// Instance pointers are ignored. Execution must obey the non-nested guard
/// and recovery discipline documented in this module.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn selene_refcount_decrement(
    _instance: *mut SeleneInstance,
    _r: u64,
) -> SeleneVoidResult {
    SeleneVoidResult { error_code: 0 }
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

/// State dumping is unsupported; return success.
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
    SeleneVoidResult { error_code: 0 }
}

/// Time cursors are unused; return success.
///
/// # Safety
/// Instance pointers are ignored. Execution must obey the non-nested guard
/// and recovery discipline documented in this module.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn selene_set_tc(
    _instance: *mut SeleneInstance,
    _time_cursor: u64,
) -> SeleneVoidResult {
    SeleneVoidResult { error_code: 0 }
}

/// Local barriers are unused; return success.
///
/// # Safety
/// Instance pointers are ignored. Execution must obey the non-nested guard
/// and recovery discipline documented in this module.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn selene_local_barrier(
    _instance: *mut SeleneInstance,
    _qubit_ids: *const u64,
    _qubit_ids_length: u64,
    _sleep_time: u64,
) -> SeleneVoidResult {
    SeleneVoidResult { error_code: 0 }
}

/// Global barriers are unused; return success.
///
/// # Safety
/// Instance pointers are ignored. Execution must obey the non-nested guard
/// and recovery discipline documented in this module.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn selene_global_barrier(
    _instance: *mut SeleneInstance,
    _sleep_time: u64,
) -> SeleneVoidResult {
    SeleneVoidResult { error_code: 0 }
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

/// Return the operation collection mode constant.
///
/// # Safety
/// Instance pointers are ignored. Execution must obey the non-nested guard
/// and recovery discipline documented in this module.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn selene_get_tc(_instance: *mut SeleneInstance) -> SeleneU64Result {
    SeleneU64Result {
        error_code: 0,
        value: 0,
    }
}

/// Return the operation collection mode constant.
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
        value: 0,
    }
}

/// Return the operation collection mode constant.
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

/// Custom runtime calls are unsupported; return zero.
///
/// # Safety
/// Instance pointers are ignored. Execution must obey the non-nested guard
/// and recovery discipline documented in this module.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn selene_custom_runtime_call(
    _instance: *mut SeleneInstance,
    _tag: u64,
    _data: *const u8,
    _data_length: u64,
) -> SeleneU64Result {
    SeleneU64Result {
        error_code: 0,
        value: 0,
    }
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
