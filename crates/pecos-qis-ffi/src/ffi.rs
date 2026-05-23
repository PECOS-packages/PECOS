//! FFI exports for linking with QIS LLVM IR programs
//!
//! Minimal set of FFI functions needed to link QIS programs
//! with Rust. These functions simply collect operations into the thread-local interface
//! without performing any simulation or complex state management.

use crate::{Operation, QuantumOp, with_interface};
use log::debug;
use std::cell::Cell;

// Thread-local counter to prevent infinite loops in collection mode.
// After MAX_COLLECTION_READS, `___read_future_bool` returns true to break out of
// loops like "repeat_until_one" (while not result: ... result = measure(q)).
thread_local! {
    static COLLECTION_MODE_READ_COUNT: Cell<u32> = const { Cell::new(0) };
}

/// Maximum number of measurement reads in collection mode before returning true.
/// This prevents infinite loops when collecting operations for programs with
/// "repeat until success" patterns.
const MAX_COLLECTION_READS: u32 = 100;

/// Helper to convert i64 to usize
#[inline]
fn i64_to_usize(value: i64) -> usize {
    usize::try_from(value).expect("Invalid ID: value must be non-negative and fit in usize")
}

// --- Gate FFI Macros ---
//
// These macros generate the boilerplate for FFI gate functions.
// Each macro handles a different gate signature pattern.

/// Single-qubit gate: `fn name(qubit: i64)` -> `QuantumOp::Op(qubit_id)`
macro_rules! ffi_gate_1q {
    ($name:ident, $op:ident) => {
        /// # Safety
        /// Called from C/LLVM code. Qubit must be a valid non-negative ID.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name(qubit: i64) {
            let qubit_id = i64_to_usize(qubit);
            with_interface(|interface| {
                interface.queue_operation(QuantumOp::$op(qubit_id).into());
            });
        }
    };
}

/// Two-qubit gate: `fn name(q1: i64, q2: i64)` -> `QuantumOp::Op(q1_id, q2_id)`
macro_rules! ffi_gate_2q {
    ($name:ident, $op:ident) => {
        /// # Safety
        /// Called from C/LLVM code. Qubit IDs must be valid non-negative values.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name(q1: i64, q2: i64) {
            let q1_id = i64_to_usize(q1);
            let q2_id = i64_to_usize(q2);
            with_interface(|interface| {
                interface.queue_operation(QuantumOp::$op(q1_id, q2_id).into());
            });
        }
    };
}

/// Three-qubit gate: `fn name(q1: i64, q2: i64, q3: i64)` -> `QuantumOp::Op(q1_id, q2_id, q3_id)`
macro_rules! ffi_gate_3q {
    ($name:ident, $op:ident) => {
        /// # Safety
        /// Called from C/LLVM code. Qubit IDs must be valid non-negative values.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name(q1: i64, q2: i64, q3: i64) {
            let q1_id = i64_to_usize(q1);
            let q2_id = i64_to_usize(q2);
            let q3_id = i64_to_usize(q3);
            with_interface(|interface| {
                interface.queue_operation(QuantumOp::$op(q1_id, q2_id, q3_id).into());
            });
        }
    };
}

/// Rotation + single-qubit: `fn name(theta: f64, qubit: i64)` -> `QuantumOp::Op(theta, qubit_id)`
macro_rules! ffi_gate_rot_1q {
    ($name:ident, $op:ident) => {
        /// # Safety
        /// Called from C/LLVM code. Qubit must be a valid non-negative ID.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name(theta: f64, qubit: i64) {
            let qubit_id = i64_to_usize(qubit);
            with_interface(|interface| {
                interface.queue_operation(QuantumOp::$op(theta, qubit_id).into());
            });
        }
    };
}

/// Rotation + two-qubit: `fn name(theta: f64, q1: i64, q2: i64)` -> `QuantumOp::Op(theta, q1_id, q2_id)`
macro_rules! ffi_gate_rot_2q {
    ($name:ident, $op:ident) => {
        /// # Safety
        /// Called from C/LLVM code. Qubit IDs must be valid non-negative values.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name(theta: f64, q1: i64, q2: i64) {
            let q1_id = i64_to_usize(q1);
            let q2_id = i64_to_usize(q2);
            with_interface(|interface| {
                interface.queue_operation(QuantumOp::$op(theta, q1_id, q2_id).into());
            });
        }
    };
}

// --- Single-Qubit Gates ---

/// Hadamard gate operation
///
/// # Safety
/// This function is safe to call from C/LLVM code. The qubit parameter must be a valid
/// non-negative qubit ID that fits in usize. Invalid IDs will cause a panic.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __quantum__qis__h__body(qubit: i64) {
    debug!("[FFI] __quantum__qis__h__body called with qubit={qubit}");
    let qubit_id = i64_to_usize(qubit);
    with_interface(|interface| {
        debug!("[FFI] H gate: queuing operation for qubit {qubit_id}");
        interface.queue_operation(QuantumOp::H(qubit_id).into());
        debug!(
            "[FFI] H gate: operation queued, interface now has {} operations",
            interface.operations.len()
        );
    });
    debug!("[FFI] __quantum__qis__h__body completed");
}

/// Pauli-X gate operation
///
/// # Safety
/// This function is safe to call from C/LLVM code. The qubit parameter must be a valid
/// non-negative qubit ID that fits in usize. Invalid IDs will cause a panic.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __quantum__qis__x__body(qubit: i64) {
    debug!("[FFI] __quantum__qis__x__body called with qubit={qubit}");
    let qubit_id = i64_to_usize(qubit);
    with_interface(|interface| {
        debug!("[FFI] X gate: queuing operation for qubit {qubit_id}");
        interface.queue_operation(QuantumOp::X(qubit_id).into());
        debug!(
            "[FFI] X gate: operation queued, interface now has {} operations",
            interface.operations.len()
        );
    });
    debug!("[FFI] __quantum__qis__x__body completed");
}

ffi_gate_1q!(__quantum__qis__y__body, Y);
ffi_gate_1q!(__quantum__qis__z__body, Z);
ffi_gate_1q!(__quantum__qis__s__body, S);
ffi_gate_1q!(__quantum__qis__sdg__body, Sdg);
ffi_gate_1q!(__quantum__qis__t__body, T);
ffi_gate_1q!(__quantum__qis__tdg__body, Tdg);

// --- Two-Qubit Gates ---

ffi_gate_2q!(__quantum__qis__cx__body, CX);

ffi_gate_2q!(__quantum__qis__cnot__body, CX);

ffi_gate_2q!(__quantum__qis__cy__body, CY);
ffi_gate_2q!(__quantum__qis__cz__body, CZ);
ffi_gate_2q!(__quantum__qis__ch__body, CH);

// --- Rotation Gates ---

ffi_gate_rot_1q!(__quantum__qis__rx__body, RX);
ffi_gate_rot_1q!(__quantum__qis__ry__body, RY);
ffi_gate_rot_1q!(__quantum__qis__rz__body, RZ);

ffi_gate_rot_2q!(__quantum__qis__rzz__body, RZZ);

/// # Safety
/// Called from C/LLVM code. Qubit must be a valid non-negative ID.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __quantum__qis__r1xy__body(theta: f64, phi: f64, qubit: i64) {
    let qubit_id = i64_to_usize(qubit);
    with_interface(|interface| {
        interface.queue_operation(QuantumOp::RXY(theta, phi, qubit_id).into());
    });
}

ffi_gate_rot_2q!(__quantum__qis__crz__body, CRZ);

// --- Three-Qubit Gates ---

ffi_gate_3q!(__quantum__qis__ccx__body, CCX);

// --- ZZ Interaction ---

ffi_gate_2q!(__quantum__qis__zz__body, ZZ);

// --- Measurement and Reset ---

/// Measure a qubit and store result
///
/// # Safety
/// This function is safe to call from C/LLVM code. The qubit and result parameters must be valid
/// non-negative IDs that fit in usize. Invalid IDs will cause a panic.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __quantum__qis__m__body(qubit: i64, result: i64) -> i32 {
    let qubit_id = i64_to_usize(qubit);
    let result_id = i64_to_usize(result);
    with_interface(|interface| {
        interface.queue_operation(QuantumOp::Measure(qubit_id, result_id).into());
    });
    // Return 0 for now - actual result will be available after runtime execution
    0
}

ffi_gate_1q!(__quantum__qis__reset__body, Reset);

// --- Allocation and Deallocation ---

/// Allocate a new qubit
///
/// # Safety
/// This function is safe to call from C/LLVM code.
///
/// # Panics
/// Panics if the allocated qubit ID is too large to fit in i64.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __quantum__rt__qubit_allocate() -> i64 {
    with_interface(|interface| {
        let id = interface.allocate_qubit();
        interface.queue_operation(Operation::AllocateQubit { id });
        i64::try_from(id).expect("Qubit ID too large for i64")
    })
}

/// Release (deallocate) a qubit
///
/// # Safety
/// This function is safe to call from C/LLVM code. The qubit parameter must be a valid
/// non-negative qubit ID that fits in usize. Invalid IDs will cause a panic.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __quantum__rt__qubit_release(qubit: i64) {
    let qubit_id = i64_to_usize(qubit);
    with_interface(|interface| {
        interface.queue_operation(Operation::ReleaseQubit { id: qubit_id });
    });
}

/// Allocate a new result storage
///
/// # Safety
/// This function is safe to call from C/LLVM code.
///
/// # Panics
/// Panics if the allocated result ID is too large to fit in i64.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __quantum__rt__result_allocate() -> i64 {
    with_interface(|interface| {
        let id = interface.allocate_result();
        interface.queue_operation(Operation::AllocateResult { id });
        i64::try_from(id).expect("Result ID too large for i64")
    })
}

// --- Result Retrieval ---

/// Get measurement result (returns 1 if result is One, 0 otherwise)
///
/// This function supports dynamic circuits: if the result is not yet available and
/// a quantum executor callback has been registered, it will execute pending quantum
/// operations to obtain the measurement result.
///
/// # Safety
/// This function is safe to call from C/LLVM code. The result parameter must be a valid
/// non-negative result ID that fits in usize. Invalid IDs will cause a panic.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __quantum__rt__result_get_one(result: i64) -> i32 {
    log::debug!("__quantum__rt__result_get_one called with result={result}");
    let result_id = i64_to_usize(result);

    // First check if result is already available
    let existing_result = with_interface(|interface| interface.get_result(result_id));

    if let Some(value) = existing_result {
        return i32::from(value);
    }

    // Result not available - try to execute pending operations
    // This enables dynamic circuits where conditionals depend on measurements
    if crate::execute_pending_and_get_results() {
        log::debug!("Executed pending operations, checking result again");
        // Execution happened, try to get the result again
        with_interface(|interface| {
            interface.get_result(result_id).map_or_else(
                || {
                    log::warn!(
                        "Measurement result {result_id} still not available after executing pending operations"
                    );
                    0
                },
                i32::from,
            )
        })
    } else {
        // No executor set - return default (static circuit behavior)
        log::debug!("No quantum executor set, returning default 0 for result {result_id}");
        0
    }
}

// --- Utility Functions ---

/// Log a message from quantum program
///
/// # Safety
/// This function is safe to call from C/LLVM code. The msg pointer may be null or must point
/// to a valid null-terminated C string. Invalid pointers will cause undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __quantum__rt__message(msg: *const std::ffi::c_char) {
    if !msg.is_null() {
        let c_str = unsafe { std::ffi::CStr::from_ptr(msg) };
        if let Ok(rust_str) = c_str.to_str() {
            log::trace!("QIS Message: {rust_str}");
        }
    }
}

/// Record data from quantum program
///
/// # Safety
/// This function is safe to call from C/LLVM code. The data pointer may be null or must point
/// to a valid null-terminated C string. Invalid pointers will cause undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __quantum__rt__record(data: *const std::ffi::c_char) {
    if !data.is_null() {
        let c_str = unsafe { std::ffi::CStr::from_ptr(data) };
        if let Ok(rust_str) = c_str.to_str() {
            log::trace!("QIS Record: {rust_str}");
        }
    }
}

// --- Selene-style FFI Functions ---
//
// These functions match the naming convention used by Selene's hugr-qis compiler.
// They provide the same functionality as the QIS-style functions above but with
// different names to support Selene-generated LLVM IR.

ffi_gate_1q!(___reset, Reset);

/// RXY rotation (Selene-style)
///
/// # Safety
/// This function is safe to call from C/LLVM code. The qubit parameter must be a valid
/// non-negative qubit ID that fits in usize. Invalid IDs will cause a panic.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ___rxy(qubit: i64, theta: f64, phi: f64) {
    // Delegate to the QIS-style function
    unsafe { __quantum__qis__r1xy__body(theta, phi, qubit) };
}

/// RZ rotation (Selene-style)
///
/// # Safety
/// This function is safe to call from C/LLVM code. The qubit parameter must be a valid
/// non-negative qubit ID that fits in usize. Invalid IDs will cause a panic.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ___rz(qubit: i64, theta: f64) {
    // Delegate to the QIS-style function
    unsafe { __quantum__qis__rz__body(theta, qubit) };
}

/// RZZ two-qubit rotation (Selene-style)
///
/// # Safety
/// This function is safe to call from C/LLVM code. The qubit parameters must be valid
/// non-negative qubit IDs that fit in usize. Invalid IDs will cause a panic.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ___rzz(qubit1: i64, qubit2: i64, theta: f64) {
    // Delegate to the QIS-style function
    unsafe { __quantum__qis__rzz__body(theta, qubit1, qubit2) };
}

/// Qubit allocation (Selene-style)
///
/// # Safety
/// This function is safe to call from C/LLVM code.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ___qalloc() -> i64 {
    // Delegate to the QIS-style function
    unsafe { __quantum__rt__qubit_allocate() }
}

/// Qubit deallocation (Selene-style)
///
/// # Safety
/// This function is safe to call from C/LLVM code. The qubit parameter must be a valid
/// non-negative qubit ID that fits in usize. Invalid IDs will cause a panic.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ___qfree(qubit: i64) {
    // Delegate to the QIS-style function
    unsafe { __quantum__rt__qubit_release(qubit) };
}

/// Setup function (called at program start)
///
/// # Safety
/// This function is safe to call from C/LLVM code.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn setup(_arg: i64) {
    // Nothing to do for now - the thread-local interface is automatically initialized
}

ffi_gate_1q!(___h, H);
ffi_gate_2q!(___cx, CX);

/// Lazy measurement function (Selene/HUGR-LLVM style)
///
/// This function performs a lazy measurement: it allocates a result ID, queues the measurement
/// operation, and returns the result ID. The actual measurement result will be available after
/// runtime execution via `__quantum__rt__result_get_one` or `___read_future_bool`.
///
/// # Safety
/// This function is safe to call from C/LLVM code. The qubit parameter must be a valid
/// non-negative qubit ID that fits in usize. Invalid IDs will cause a panic.
///
/// # Returns
/// Returns the allocated result ID as i64.
///
/// # Panics
/// Panics if the allocated result ID is too large to fit in i64.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ___lazy_measure(qubit: i64) -> i64 {
    let qubit_id = i64_to_usize(qubit);
    with_interface(|interface| {
        // Allocate a result ID for this measurement
        let result_id = interface.allocate_result();
        // Queue the allocation operation
        interface.queue_operation(Operation::AllocateResult { id: result_id });
        // Queue the measurement operation
        interface.queue_operation(QuantumOp::Measure(qubit_id, result_id).into());
        // Return the result ID
        i64::try_from(result_id).expect("Result ID too large for i64")
    })
}

/// Read a future boolean value (Guppy/HUGR-LLVM style)
///
/// This function retrieves a measurement result from a future/deferred measurement.
/// The `future_id` is the result ID returned by `___lazy_measure`.
///
/// For dynamic circuits: If the result is not yet available and dynamic mode is active,
/// this function will signal the main thread and block until the result is available.
/// The main thread should simulate the pending operations and provide the result.
///
/// Requires an execution context to be registered for dynamic circuit support.
///
/// # Safety
/// This function is safe to call from C/LLVM code. The `future_id` parameter must be a valid
/// result ID previously returned by `___lazy_measure`. Invalid IDs will cause a panic.
///
/// # Returns
/// Returns the boolean measurement result (true = 1, false = 0).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ___read_future_bool(future_id: i64) -> bool {
    log::debug!("___read_future_bool called with future_id={future_id}");
    let result_id = i64_to_usize(future_id);

    // Check if result is already available in thread-local storage
    let existing_result = with_interface(|interface| interface.get_result(result_id));
    log::debug!("___read_future_bool: existing_result={existing_result:?}");

    if let Some(result) = existing_result {
        return result;
    }

    // Check if dynamic mode is active (requires execution context)
    if crate::is_dynamic_mode_active() {
        // First check if result is already available in execution context
        // This can happen when multiple measurements are batched together
        if let Some(result) = crate::get_measurement_result(result_id as u64) {
            log::debug!(
                "___read_future_bool: result already in context for result_id={result_id}: {result}"
            );
            return result;
        }

        log::debug!(
            "___read_future_bool: dynamic mode active, signaling need for result_id={result_id}"
        );

        // Wait for the main thread to provide the result
        // This uses the per-execution context for synchronization
        if crate::wait_for_result_ready(result_id as u64, 30000) {
            // Result should now be available in the execution context
            // The main thread stores results there to cross the thread boundary
            let result = crate::get_measurement_result(result_id as u64);
            log::debug!("___read_future_bool: got result after waiting: {result:?}");
            return result.unwrap_or(false);
        }
        log::debug!("___read_future_bool: timeout waiting for result");
    }

    // Collection mode (non-dynamic): track read count to prevent infinite loops.
    // For programs with "repeat until success" loops like:
    //   while not result:
    //       q = qubit()
    //       result = measure(q)
    // Each iteration creates a new result_id, so we track total reads.
    // After MAX_COLLECTION_READS, we return true to break the loop.
    let read_count = COLLECTION_MODE_READ_COUNT.with(|c| {
        let count = c.get() + 1;
        c.set(count);
        count
    });

    if read_count >= MAX_COLLECTION_READS {
        log::debug!(
            "___read_future_bool: collection mode read count ({read_count}) >= threshold, returning true to break loop"
        );
        true
    } else {
        // Default: return false (allows first iterations of loops to proceed)
        false
    }
}

/// Reset the collection mode read counter.
///
/// This should be called at the start of each new execution to reset the loop
/// termination counter used in `___read_future_bool`.
pub fn reset_collection_read_count() {
    COLLECTION_MODE_READ_COUNT.with(|c| c.set(0));
}

/// Increment the reference count of a future (Guppy/HUGR-LLVM style)
///
/// This function is called when a future value is copied or shared.
/// In the minimal interface, this is a no-op since we don't do reference counting.
///
/// # Safety
/// This function is safe to call from C/LLVM code. The `future_id` parameter is ignored.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ___inc_future_refcount(_future_id: i64) {
    // No-op in the minimal interface - we don't do reference counting
    // The runtime will clean up measurement results when the shot completes
}

/// Decrement the reference count of a future (Guppy/HUGR-LLVM style)
///
/// This function is called when a future value is no longer needed.
/// In the minimal interface, this is a no-op since we don't do reference counting.
///
/// # Safety
/// This function is safe to call from C/LLVM code. The `future_id` parameter is ignored.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ___dec_future_refcount(_future_id: i64) {
    // No-op in the minimal interface - we don't do reference counting
    // The runtime will clean up measurement results when the shot completes
}

/// Teardown function (called at program end)
///
/// # Safety
/// This function is safe to call from C/LLVM code.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn teardown() -> i64 {
    // Return success
    0
}

/// Panic function (called on program errors)
///
/// # Safety
/// This function is safe to call from C/LLVM code. The message pointer may be null or must point
/// to a valid null-terminated C string. Invalid pointers will cause undefined behavior.
///
/// # Panics
/// This function intentionally panics to propagate errors from the quantum program.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn panic(code: i32, message: *const std::ffi::c_char) {
    let msg = if message.is_null() {
        "Unknown error".to_string()
    } else {
        unsafe {
            let cstr = std::ffi::CStr::from_ptr(message);
            cstr.to_string_lossy().to_string()
        }
    };
    std::panic!("QIS program panic: code={code}, message={msg}");
}

/// Record measurement result output (for compatibility with QIR)
/// This is typically used to record measurement results to classical registers
///
/// # Safety
/// This function is safe to call from C/LLVM code. The `result_ptr` parameter is an i8* pointer
/// that represents a result ID (typically from inttoptr i64 conversion in LLVM IR).
/// The `register_name` pointer may be null or must point to a valid null-terminated C string.
/// Invalid pointers will cause undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __quantum__rt__result_record_output(
    result_ptr: *const std::ffi::c_void,
    register_name: *const std::ffi::c_char,
) {
    // Extract the result ID from the pointer
    // HUGR generates: %result_ptr = inttoptr i64 %result_id to i8*
    let result_id = result_ptr as usize;

    // Convert the C string to a Rust String
    let register_name_str = if register_name.is_null() {
        "unknown".to_string()
    } else {
        let c_str = unsafe { std::ffi::CStr::from_ptr(register_name) };
        c_str.to_str().unwrap_or("unknown").to_string()
    };

    log::trace!(
        "Recording output mapping: result_id={result_id} -> register_name='{register_name_str}'"
    );

    // Queue the operation to record this output mapping
    with_interface(|interface| {
        interface.queue_operation(Operation::RecordOutput {
            result_id,
            register_name: register_name_str,
        });
    });
}

// --- QIS measurement functions ---

// QIS measurement functions - mz is measurement in Z basis
/// Measure a qubit in the Z basis
///
/// # Safety
/// This function is safe to call from C/LLVM code. The qubit parameter must be a valid
/// non-negative qubit ID that fits in usize. Invalid IDs will cause a panic.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __quantum__qis__mz__body(qubit: i64) -> i32 {
    // Call our standard measurement function with result ID = qubit ID
    unsafe { __quantum__qis__m__body(qubit, qubit) }
}

// --- Result printing functions ---

/// Print a boolean result with a label
///
/// This function is called by QIS programs to output measurement results
/// with labels like "`measurement_0`", "`measurement_1`", etc.
///
/// # Arguments
/// * `label_ptr` - Pointer to the label struct: `{len: u8, data: [u8; len]}`
/// * `label_len` - Length of the label string (same as the len byte in the struct)
/// * `value` - Boolean value to print
///
/// # Note
/// The tket2 LLVM codegen emits strings as `{u8 len, u8[] data}` structs.
/// The `label_ptr` points to this struct, and `label_len` is the length value.
/// We need to skip the first byte (the length) to get to the actual string data.
///
/// # Safety
/// This function is safe to call from C/LLVM code. The `label_ptr` must point to a valid
/// string struct with at least `label_len + 1` bytes. Invalid pointers will cause undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn print_bool(label_ptr: *const u8, label_len: i64, value: bool) {
    let thread_id = std::thread::current().id();
    let Ok(label_len_usize) = usize::try_from(label_len) else {
        log::error!("print_bool: invalid label length {label_len}");
        return;
    };

    // The tket2 string format is: {len: u8, data: [u8; len]}
    // label_ptr points to the len byte, so we need to skip it to get the actual data
    let data_ptr = unsafe { label_ptr.add(1) };
    let label_slice = unsafe { std::slice::from_raw_parts(data_ptr, label_len_usize) };

    let Ok(label) = std::str::from_utf8(label_slice) else {
        log::error!("print_bool: invalid UTF-8 in label");
        return;
    };

    // Strip the USER:BOOL: or USER:BOOLARR: prefix if present
    let name = if let Some(stripped) = label.strip_prefix("USER:BOOL:") {
        stripped
    } else if let Some(stripped) = label.strip_prefix("USER:BOOLARR:") {
        stripped
    } else {
        label
    };

    // Get execution context and store the result
    let ctx_ptr = crate::get_execution_context();
    log::debug!(
        "print_bool: thread {thread_id:?}, name='{name}', value={value}, context={ctx_ptr:?}"
    );

    if let Some(ctx) = ctx_ptr {
        // SAFETY: Context is valid for duration of execution
        let ctx = unsafe { &*ctx };
        ctx.store_named_bool(name, value);
    } else {
        log::warn!(
            "print_bool: NO EXECUTION CONTEXT on thread {thread_id:?} for '{name}' = {value}"
        );
    }
}

/// Dense 1D array struct matching the LLVM ABI from tket2
///
/// This struct is passed by pointer from LLVM-compiled code.
/// The layout matches what `struct_1d_arr_t` in tket-qsystem creates:
/// - x: array length (i32)
/// - y: always 1 (i32)
/// - data: pointer to the data array
/// - mask: pointer to mask array (unused for dense arrays)
#[repr(C)]
pub struct Dense1DArrayBool {
    pub x: i32,
    pub y: i32,
    pub data: *const bool,
    pub mask: *const bool,
}

/// Print a boolean array result with a label
///
/// This function is called by Guppy-generated QIS programs to output arrays of
/// measurement results (e.g., syndrome arrays, final measurements).
///
/// # Arguments
/// * `label_ptr` - Pointer to the label struct: `{len: u8, data: [u8; len]}`
/// * `label_len` - Length of the label string (same as the len byte in the struct)
/// * `arr` - Pointer to the `Dense1DArrayBool` struct containing the array data
///
/// # Note
/// The tket2 LLVM codegen emits strings as `{u8 len, u8[] data}` structs.
/// The `label_ptr` points to this struct, and `label_len` is the length value.
/// We need to skip the first byte (the length) to get to the actual string data.
///
/// # Safety
/// This function is safe to call from C/LLVM code. The `label_ptr` must point to a valid
/// string struct with at least `label_len + 1` bytes. The `arr` must point to a valid
/// `Dense1DArrayBool` struct. Invalid pointers will cause undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn print_bool_arr(
    label_ptr: *const u8,
    label_len: i64,
    arr: *const Dense1DArrayBool,
) {
    // Validate label length
    let Ok(label_len_usize) = usize::try_from(label_len) else {
        log::error!("print_bool_arr: invalid label length {label_len}");
        return;
    };

    // Check that arr pointer is valid
    if arr.is_null() {
        log::error!("print_bool_arr: null array pointer");
        return;
    }

    // The tket2 string format is: {len: u8, data: [u8; len]}
    // label_ptr points to the len byte, so we need to skip it to get the actual data
    let data_ptr = unsafe { label_ptr.add(1) };
    let label_slice = unsafe { std::slice::from_raw_parts(data_ptr, label_len_usize) };
    let Ok(label) = std::str::from_utf8(label_slice) else {
        log::error!("print_bool_arr: invalid UTF-8 in label");
        return;
    };

    // Read the array struct
    let arr_struct = unsafe { &*arr };
    let Ok(arr_len) = usize::try_from(arr_struct.x) else {
        log::error!("print_bool_arr: invalid array length {}", arr_struct.x);
        return;
    };

    // Validate data pointer
    if arr_struct.data.is_null() {
        log::error!("print_bool_arr: null data pointer in array struct");
        return;
    }

    // Convert the array to a Rust slice
    let arr_slice = unsafe { std::slice::from_raw_parts(arr_struct.data, arr_len) };

    // Log the array for debugging
    log::debug!("print_bool_arr called: {label} = {arr_slice:?}");

    // Strip the USER:BOOLARR: prefix if present
    let name = if let Some(stripped) = label.strip_prefix("USER:BOOLARR:") {
        stripped
    } else if let Some(stripped) = label.strip_prefix("USER:BOOL:") {
        stripped
    } else {
        label
    };

    // Store in the execution context's named results
    if let Some(ctx) = crate::get_execution_context() {
        // SAFETY: Context is valid for duration of execution
        let ctx = unsafe { &*ctx };
        ctx.store_named_array(name, arr_slice);
    }
}

// --- Selene-compatible print functions ---
//
// The Selene runtime uses a different string format (selene_string_t has direct
// data pointer, not tket2's {len: u8, data: [u8]} format). These functions are
// called from the selene_shim.c and expect direct string data.

/// Print a boolean result with a label (Selene-compatible format)
///
/// This function is called from the Selene shim with direct string data pointer.
/// Unlike `print_bool`, this does NOT skip the first byte (no tket2 format).
///
/// # Safety
/// This function is safe to call from C/LLVM code. The `label_ptr` must point to valid
/// string data of at least `label_len` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn print_bool_selene(label_ptr: *const u8, label_len: i64, value: bool) {
    let thread_id = std::thread::current().id();
    let Ok(label_len_usize) = usize::try_from(label_len) else {
        log::error!("print_bool_selene: invalid label length {label_len}");
        return;
    };

    // Direct string data - no need to skip any bytes
    let label_slice = unsafe { std::slice::from_raw_parts(label_ptr, label_len_usize) };

    let Ok(label) = std::str::from_utf8(label_slice) else {
        log::error!("print_bool_selene: invalid UTF-8 in label");
        return;
    };

    // Strip the USER:BOOL: or USER:BOOLARR: prefix if present
    let name = if let Some(stripped) = label.strip_prefix("USER:BOOL:") {
        stripped
    } else if let Some(stripped) = label.strip_prefix("USER:BOOLARR:") {
        stripped
    } else {
        label
    };

    // Get execution context and store the result
    let ctx_ptr = crate::get_execution_context();
    log::debug!(
        "print_bool_selene: thread {thread_id:?}, name='{name}', value={value}, context={ctx_ptr:?}"
    );

    if let Some(ctx) = ctx_ptr {
        let ctx = unsafe { &*ctx };
        ctx.store_named_bool(name, value);
    } else {
        log::warn!(
            "print_bool_selene: NO EXECUTION CONTEXT on thread {thread_id:?} for '{name}' = {value}"
        );
    }
}

/// Print a boolean array result with a label (Selene-compatible format)
///
/// This function is called from the Selene shim with direct string/array pointers.
/// Unlike `print_bool_arr`, this does NOT expect tket2 format or `Dense1DArrayBool` struct.
///
/// # Safety
/// This function is safe to call from C/LLVM code. The `label_ptr` must point to valid
/// string data of at least `label_len` bytes. The `arr_ptr` must point to valid bool
/// data of at least `arr_len` elements.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn print_bool_arr_selene(
    label_ptr: *const u8,
    label_len: i64,
    arr_ptr: *const bool,
    arr_len: u64,
) {
    let Ok(label_len_usize) = usize::try_from(label_len) else {
        log::error!("print_bool_arr_selene: invalid label length {label_len}");
        return;
    };

    let Ok(arr_len_usize) = usize::try_from(arr_len) else {
        log::error!("print_bool_arr_selene: invalid array length {arr_len}");
        return;
    };

    if arr_ptr.is_null() {
        log::error!("print_bool_arr_selene: null array pointer");
        return;
    }

    // Direct string data - no need to skip any bytes
    let label_slice = unsafe { std::slice::from_raw_parts(label_ptr, label_len_usize) };
    let Ok(label) = std::str::from_utf8(label_slice) else {
        log::error!("print_bool_arr_selene: invalid UTF-8 in label");
        return;
    };

    // Direct array data
    let arr_slice = unsafe { std::slice::from_raw_parts(arr_ptr, arr_len_usize) };

    // Strip the USER:BOOLARR: or USER:BOOL: prefix if present
    let name = if let Some(stripped) = label.strip_prefix("USER:BOOLARR:") {
        stripped
    } else if let Some(stripped) = label.strip_prefix("USER:BOOL:") {
        stripped
    } else {
        label
    };

    // Store in the execution context's named results
    if let Some(ctx) = crate::get_execution_context() {
        let ctx = unsafe { &*ctx };
        ctx.store_named_array(name, arr_slice);
    }
}

// --- Interface Management (C exports for dlsym access) ---

/// Reset the thread-local interface
/// Exported as C function so it can be called via dlsym from the cdylib
///
/// # Safety
/// This function is safe to call from C/LLVM code.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pecos_qis_reset_interface() {
    crate::reset_interface();
}

/// Take the current `OperationCollector`, leaving an empty collector behind.
/// Exported as C function so it can be called via dlsym from the cdylib
///
/// # Safety
/// This function is safe to call from C/LLVM code. The returned pointer must be freed using
/// `pecos_qis_free_operations` to avoid memory leaks.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pecos_qis_get_operations() -> *mut crate::OperationCollector {
    let operations = crate::take_interface();
    Box::into_raw(Box::new(operations))
}

/// Free an `OperationCollector` returned by `pecos_qis_get_operations`
///
/// # Safety
/// This function is safe to call from C/LLVM code. The ptr must be either null or a valid
/// pointer previously returned by `pecos_qis_get_operations` that has not yet been freed.
/// Double-freeing will cause undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pecos_qis_free_operations(ptr: *mut crate::OperationCollector) {
    if !ptr.is_null() {
        unsafe {
            drop(Box::from_raw(ptr));
        }
    }
}

/// Set measurement results in the thread-local interface
/// Takes a pointer to an array of (`result_id`, value) pairs and the array length
/// This allows pre-populating measurement outcomes for conditional execution
///
/// # Safety
/// This function is safe to call from C/LLVM code. The `pairs_ptr` may be null or must point to a
/// valid array of at least count elements. Invalid pointers or counts will cause undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pecos_qis_set_measurements(pairs_ptr: *const (usize, bool), count: usize) {
    if pairs_ptr.is_null() {
        return;
    }

    let pairs = unsafe { std::slice::from_raw_parts(pairs_ptr, count) };

    with_interface(|interface| {
        interface.set_measurement_results(pairs.iter().copied());
    });
}

// --- Heap Management Functions (Selene compatibility) ---

/// Allocate heap memory
///
/// This is used by Guppy/HUGR for array allocation and other heap operations.
/// Following Selene's approach, we use libc malloc/free which handle size tracking.
///
/// # Safety
/// This function is safe to call from C/LLVM code. Returns a null pointer for zero-sized allocations.
///
/// # Panics
/// Panics if malloc fails to allocate the requested memory.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn heap_alloc(size: u64) -> *mut u8 {
    if size == 0 {
        // Return null for zero-sized allocations (standard malloc behavior)
        return std::ptr::null_mut();
    }

    // Use libc malloc which tracks allocation sizes internally
    // Convert u64 to size_t, handling potential overflow
    let Ok(size_t) = libc::size_t::try_from(size) else {
        // Size too large for this platform
        std::panic!("heap_alloc: size {size} too large for platform");
    };
    let ptr = unsafe { libc::malloc(size_t).cast::<u8>() };

    assert!(
        !ptr.is_null(),
        "heap_alloc: failed to allocate {size} bytes"
    );

    ptr
}

/// Free heap memory
///
/// This is used by Guppy/HUGR to deallocate arrays and other heap objects.
/// Following Selene's approach, we use libc free which matches malloc.
///
/// # Safety
/// This function is safe to call from C/LLVM code. The ptr must be either null or a valid pointer
/// previously returned by `heap_alloc` that has not yet been freed. Double-freeing will cause
/// undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn heap_free(ptr: *mut u8) {
    if ptr.is_null() {
        // Ignore null pointer frees (standard free behavior)
        return;
    }

    // Use libc free which pairs with malloc
    unsafe { libc::free(ptr.cast::<libc::c_void>()) };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Operation, QuantumOp, reset_interface, with_interface};

    /// Helper to reset and get a clean interface for testing
    fn setup_test() {
        reset_interface();
    }

    // --- Single-qubit gate tests ---

    #[test]
    fn test_h_gate() {
        setup_test();
        unsafe { __quantum__qis__h__body(0) };

        with_interface(|iface| {
            assert_eq!(iface.operations.len(), 1);
            assert_eq!(iface.operations[0], Operation::Quantum(QuantumOp::H(0)));
        });
    }

    #[test]
    fn test_x_gate() {
        setup_test();
        unsafe { __quantum__qis__x__body(1) };

        with_interface(|iface| {
            assert_eq!(iface.operations.len(), 1);
            assert_eq!(iface.operations[0], Operation::Quantum(QuantumOp::X(1)));
        });
    }

    #[test]
    fn test_y_gate() {
        setup_test();
        unsafe { __quantum__qis__y__body(2) };

        with_interface(|iface| {
            assert_eq!(iface.operations[0], Operation::Quantum(QuantumOp::Y(2)));
        });
    }

    #[test]
    fn test_z_gate() {
        setup_test();
        unsafe { __quantum__qis__z__body(3) };

        with_interface(|iface| {
            assert_eq!(iface.operations[0], Operation::Quantum(QuantumOp::Z(3)));
        });
    }

    #[test]
    fn test_s_gate() {
        setup_test();
        unsafe { __quantum__qis__s__body(0) };

        with_interface(|iface| {
            assert_eq!(iface.operations[0], Operation::Quantum(QuantumOp::S(0)));
        });
    }

    #[test]
    fn test_sdg_gate() {
        setup_test();
        unsafe { __quantum__qis__sdg__body(0) };

        with_interface(|iface| {
            assert_eq!(iface.operations[0], Operation::Quantum(QuantumOp::Sdg(0)));
        });
    }

    #[test]
    fn test_t_gate() {
        setup_test();
        unsafe { __quantum__qis__t__body(0) };

        with_interface(|iface| {
            assert_eq!(iface.operations[0], Operation::Quantum(QuantumOp::T(0)));
        });
    }

    #[test]
    fn test_tdg_gate() {
        setup_test();
        unsafe { __quantum__qis__tdg__body(0) };

        with_interface(|iface| {
            assert_eq!(iface.operations[0], Operation::Quantum(QuantumOp::Tdg(0)));
        });
    }

    // --- Two-qubit gate tests ---

    #[test]
    fn test_cx_gate() {
        setup_test();
        unsafe { __quantum__qis__cx__body(0, 1) };

        with_interface(|iface| {
            assert_eq!(iface.operations[0], Operation::Quantum(QuantumOp::CX(0, 1)));
        });
    }

    #[test]
    fn test_cnot_gate() {
        setup_test();
        unsafe { __quantum__qis__cnot__body(2, 3) };

        with_interface(|iface| {
            // CNOT is an alias for CX
            assert_eq!(iface.operations[0], Operation::Quantum(QuantumOp::CX(2, 3)));
        });
    }

    #[test]
    fn test_cy_gate() {
        setup_test();
        unsafe { __quantum__qis__cy__body(0, 1) };

        with_interface(|iface| {
            assert_eq!(iface.operations[0], Operation::Quantum(QuantumOp::CY(0, 1)));
        });
    }

    #[test]
    fn test_cz_gate() {
        setup_test();
        unsafe { __quantum__qis__cz__body(0, 1) };

        with_interface(|iface| {
            assert_eq!(iface.operations[0], Operation::Quantum(QuantumOp::CZ(0, 1)));
        });
    }

    #[test]
    fn test_ch_gate() {
        setup_test();
        unsafe { __quantum__qis__ch__body(0, 1) };

        with_interface(|iface| {
            assert_eq!(iface.operations[0], Operation::Quantum(QuantumOp::CH(0, 1)));
        });
    }

    // --- Rotation gate tests ---

    #[test]
    fn test_rx_gate() {
        setup_test();
        let theta = std::f64::consts::PI / 2.0;
        unsafe { __quantum__qis__rx__body(theta, 0) };

        with_interface(|iface| {
            assert_eq!(
                iface.operations[0],
                Operation::Quantum(QuantumOp::RX(theta, 0))
            );
        });
    }

    #[test]
    fn test_ry_gate() {
        setup_test();
        let theta = std::f64::consts::PI / 4.0;
        unsafe { __quantum__qis__ry__body(theta, 1) };

        with_interface(|iface| {
            assert_eq!(
                iface.operations[0],
                Operation::Quantum(QuantumOp::RY(theta, 1))
            );
        });
    }

    #[test]
    fn test_rz_gate() {
        setup_test();
        let theta = std::f64::consts::PI;
        unsafe { __quantum__qis__rz__body(theta, 2) };

        with_interface(|iface| {
            assert_eq!(
                iface.operations[0],
                Operation::Quantum(QuantumOp::RZ(theta, 2))
            );
        });
    }

    #[test]
    fn test_rzz_gate() {
        setup_test();
        let theta = 1.5;
        unsafe { __quantum__qis__rzz__body(theta, 0, 1) };

        with_interface(|iface| {
            assert_eq!(
                iface.operations[0],
                Operation::Quantum(QuantumOp::RZZ(theta, 0, 1))
            );
        });
    }

    #[test]
    fn test_r1xy_gate() {
        setup_test();
        let theta = 1.0;
        let phi = 0.5;
        unsafe { __quantum__qis__r1xy__body(theta, phi, 0) };

        with_interface(|iface| {
            assert_eq!(
                iface.operations[0],
                Operation::Quantum(QuantumOp::RXY(theta, phi, 0))
            );
        });
    }

    #[test]
    fn test_crz_gate() {
        setup_test();
        let theta = 2.0;
        unsafe { __quantum__qis__crz__body(theta, 0, 1) };

        with_interface(|iface| {
            assert_eq!(
                iface.operations[0],
                Operation::Quantum(QuantumOp::CRZ(theta, 0, 1))
            );
        });
    }

    // --- Three-qubit gate tests ---

    #[test]
    fn test_ccx_gate() {
        setup_test();
        unsafe { __quantum__qis__ccx__body(0, 1, 2) };

        with_interface(|iface| {
            assert_eq!(
                iface.operations[0],
                Operation::Quantum(QuantumOp::CCX(0, 1, 2))
            );
        });
    }

    // --- ZZ interaction tests ---

    #[test]
    fn test_zz_gate() {
        setup_test();
        unsafe { __quantum__qis__zz__body(0, 1) };

        with_interface(|iface| {
            assert_eq!(iface.operations[0], Operation::Quantum(QuantumOp::ZZ(0, 1)));
        });
    }

    // --- Measurement and reset tests ---

    #[test]
    fn test_measurement() {
        setup_test();
        let result = unsafe { __quantum__qis__m__body(0, 0) };

        assert_eq!(result, 0); // Default return value

        with_interface(|iface| {
            assert_eq!(
                iface.operations[0],
                Operation::Quantum(QuantumOp::Measure(0, 0))
            );
        });
    }

    #[test]
    fn test_mz_measurement() {
        setup_test();
        let result = unsafe { __quantum__qis__mz__body(5) };

        assert_eq!(result, 0);

        with_interface(|iface| {
            // mz uses qubit ID as result ID
            assert_eq!(
                iface.operations[0],
                Operation::Quantum(QuantumOp::Measure(5, 5))
            );
        });
    }

    #[test]
    fn test_reset() {
        setup_test();
        unsafe { __quantum__qis__reset__body(3) };

        with_interface(|iface| {
            assert_eq!(iface.operations[0], Operation::Quantum(QuantumOp::Reset(3)));
        });
    }

    // --- Allocation tests ---

    #[test]
    fn test_qubit_allocate() {
        setup_test();
        let q0 = unsafe { __quantum__rt__qubit_allocate() };
        let q1 = unsafe { __quantum__rt__qubit_allocate() };

        assert_eq!(q0, 0);
        assert_eq!(q1, 1);

        with_interface(|iface| {
            assert_eq!(iface.operations.len(), 2);
            assert_eq!(iface.operations[0], Operation::AllocateQubit { id: 0 });
            assert_eq!(iface.operations[1], Operation::AllocateQubit { id: 1 });
        });
    }

    #[test]
    fn test_qubit_release() {
        setup_test();
        unsafe { __quantum__rt__qubit_release(5) };

        with_interface(|iface| {
            assert_eq!(iface.operations[0], Operation::ReleaseQubit { id: 5 });
        });
    }

    #[test]
    fn test_result_allocate() {
        setup_test();
        let r0 = unsafe { __quantum__rt__result_allocate() };
        let r1 = unsafe { __quantum__rt__result_allocate() };

        assert_eq!(r0, 0);
        assert_eq!(r1, 1);

        with_interface(|iface| {
            assert_eq!(iface.operations.len(), 2);
            assert_eq!(iface.operations[0], Operation::AllocateResult { id: 0 });
            assert_eq!(iface.operations[1], Operation::AllocateResult { id: 1 });
        });
    }

    // --- Selene-style function tests ---

    #[test]
    fn test_selene_reset() {
        setup_test();
        unsafe { ___reset(4) };

        with_interface(|iface| {
            assert_eq!(iface.operations[0], Operation::Quantum(QuantumOp::Reset(4)));
        });
    }

    #[test]
    fn test_selene_rxy() {
        setup_test();
        unsafe { ___rxy(0, 1.5, 0.5) };

        with_interface(|iface| {
            assert_eq!(
                iface.operations[0],
                Operation::Quantum(QuantumOp::RXY(1.5, 0.5, 0))
            );
        });
    }

    #[test]
    fn test_selene_rz() {
        setup_test();
        unsafe { ___rz(1, 2.0) };

        with_interface(|iface| {
            assert_eq!(
                iface.operations[0],
                Operation::Quantum(QuantumOp::RZ(2.0, 1))
            );
        });
    }

    #[test]
    fn test_selene_rzz() {
        setup_test();
        unsafe { ___rzz(0, 1, 1.5) };

        with_interface(|iface| {
            assert_eq!(
                iface.operations[0],
                Operation::Quantum(QuantumOp::RZZ(1.5, 0, 1))
            );
        });
    }

    #[test]
    fn test_selene_qalloc() {
        setup_test();
        let q = unsafe { ___qalloc() };

        assert_eq!(q, 0);

        with_interface(|iface| {
            assert_eq!(iface.operations[0], Operation::AllocateQubit { id: 0 });
        });
    }

    #[test]
    fn test_selene_qfree() {
        setup_test();
        unsafe { ___qfree(3) };

        with_interface(|iface| {
            assert_eq!(iface.operations[0], Operation::ReleaseQubit { id: 3 });
        });
    }

    #[test]
    fn test_selene_h() {
        setup_test();
        unsafe { ___h(2) };

        with_interface(|iface| {
            assert_eq!(iface.operations[0], Operation::Quantum(QuantumOp::H(2)));
        });
    }

    #[test]
    fn test_selene_cx() {
        setup_test();
        unsafe { ___cx(0, 1) };

        with_interface(|iface| {
            assert_eq!(iface.operations[0], Operation::Quantum(QuantumOp::CX(0, 1)));
        });
    }

    // --- Lazy measure and future tests ---

    #[test]
    fn test_lazy_measure() {
        setup_test();
        let result_id = unsafe { ___lazy_measure(0) };

        assert_eq!(result_id, 0);

        with_interface(|iface| {
            assert_eq!(iface.operations.len(), 2);
            assert_eq!(iface.operations[0], Operation::AllocateResult { id: 0 });
            assert_eq!(
                iface.operations[1],
                Operation::Quantum(QuantumOp::Measure(0, 0))
            );
        });
    }

    #[test]
    fn test_read_future_bool_with_stored_result() {
        setup_test();

        // Store a result first
        with_interface(|iface| {
            iface.store_result(0, true);
        });

        let result = unsafe { ___read_future_bool(0) };
        assert!(result);
    }

    #[test]
    fn test_read_future_bool_default() {
        setup_test();

        // No result stored, no dynamic mode - should return false
        let result = unsafe { ___read_future_bool(99) };
        assert!(!result);
    }

    #[test]
    fn test_future_refcount_noops() {
        // These are no-ops but should not crash
        unsafe {
            ___inc_future_refcount(0);
            ___dec_future_refcount(0);
            ___inc_future_refcount(999);
            ___dec_future_refcount(999);
        }
    }

    // --- Setup/teardown tests ---

    #[test]
    fn test_setup() {
        // Should not crash
        unsafe { setup(0) };
        unsafe { setup(42) };
    }

    #[test]
    fn test_teardown() {
        let result = unsafe { teardown() };
        assert_eq!(result, 0);
    }

    // --- Result retrieval tests ---

    #[test]
    fn test_result_get_one_with_stored_result() {
        setup_test();

        with_interface(|iface| {
            iface.store_result(0, true);
        });

        let result = unsafe { __quantum__rt__result_get_one(0) };
        assert_eq!(result, 1);
    }

    #[test]
    fn test_result_get_one_default() {
        setup_test();

        // No result stored - returns 0
        let result = unsafe { __quantum__rt__result_get_one(99) };
        assert_eq!(result, 0);
    }

    // --- Heap allocation tests ---

    #[test]
    fn test_heap_alloc_and_free() {
        let ptr = unsafe { heap_alloc(100) };
        assert!(!ptr.is_null());

        // SAFETY: `ptr` was just allocated by `heap_alloc(100)` and asserted non-null
        // above; the test scope owns it exclusively until `heap_free` below.
        unsafe {
            std::ptr::write(ptr, 42u8);
            assert_eq!(std::ptr::read(ptr), 42u8);
        }

        // Free should not crash
        unsafe { heap_free(ptr) };
    }

    #[test]
    fn test_heap_alloc_zero_size() {
        let ptr = unsafe { heap_alloc(0) };
        assert!(ptr.is_null());
    }

    #[test]
    fn test_heap_free_null() {
        // Should not crash
        unsafe { heap_free(std::ptr::null_mut()) };
    }

    // --- Message and record tests ---

    #[test]
    fn test_message_null() {
        // Should not crash with null pointer
        unsafe { __quantum__rt__message(std::ptr::null()) };
    }

    #[test]
    fn test_message_valid() {
        let msg = std::ffi::CString::new("Test message").unwrap();
        unsafe { __quantum__rt__message(msg.as_ptr()) };
    }

    #[test]
    fn test_record_null() {
        // Should not crash with null pointer
        unsafe { __quantum__rt__record(std::ptr::null()) };
    }

    #[test]
    fn test_record_valid() {
        let data = std::ffi::CString::new("Test data").unwrap();
        unsafe { __quantum__rt__record(data.as_ptr()) };
    }

    // --- Result record output tests ---

    #[test]
    fn test_result_record_output() {
        setup_test();

        let register_name = std::ffi::CString::new("c0").unwrap();
        unsafe {
            __quantum__rt__result_record_output(
                5 as *const std::ffi::c_void,
                register_name.as_ptr(),
            );
        };

        with_interface(|iface| {
            assert_eq!(iface.operations.len(), 1);
            assert_eq!(
                iface.operations[0],
                Operation::RecordOutput {
                    result_id: 5,
                    register_name: "c0".to_string()
                }
            );
        });
    }

    #[test]
    fn test_result_record_output_null_name() {
        setup_test();

        unsafe {
            __quantum__rt__result_record_output(3 as *const std::ffi::c_void, std::ptr::null());
        };

        with_interface(|iface| {
            assert_eq!(
                iface.operations[0],
                Operation::RecordOutput {
                    result_id: 3,
                    register_name: "unknown".to_string()
                }
            );
        });
    }

    // --- Interface management tests (C exports) ---

    #[test]
    fn test_pecos_qis_reset_interface() {
        // Add some operations
        unsafe { __quantum__qis__h__body(0) };

        // Reset
        unsafe { pecos_qis_reset_interface() };

        with_interface(|iface| {
            assert!(iface.operations.is_empty());
        });
    }

    #[test]
    fn test_pecos_qis_get_and_free_operations() {
        setup_test();
        unsafe { __quantum__qis__h__body(0) };

        let ptr = unsafe { pecos_qis_get_operations() };
        assert!(!ptr.is_null());

        // Verify contents
        let collector = unsafe { &*ptr };
        assert_eq!(collector.operations.len(), 1);

        // Free
        unsafe { pecos_qis_free_operations(ptr) };

        with_interface(|iface| {
            assert!(iface.operations.is_empty());
        });
    }

    #[test]
    fn test_pecos_qis_free_operations_null() {
        // Should not crash
        unsafe { pecos_qis_free_operations(std::ptr::null_mut()) };
    }

    #[test]
    fn test_pecos_qis_set_measurements() {
        setup_test();

        let pairs: [(usize, bool); 2] = [(0, true), (1, false)];
        unsafe { pecos_qis_set_measurements(pairs.as_ptr(), pairs.len()) };

        with_interface(|iface| {
            assert_eq!(iface.get_result(0), Some(true));
            assert_eq!(iface.get_result(1), Some(false));
        });
    }

    #[test]
    fn test_pecos_qis_set_measurements_null() {
        // Should not crash
        unsafe { pecos_qis_set_measurements(std::ptr::null(), 5) };
    }

    // --- Multiple operations sequence test ---

    #[test]
    fn test_bell_state_circuit() {
        setup_test();

        // Bell state: allocate 2 qubits, H on first, CNOT, measure both
        let q0 = unsafe { __quantum__rt__qubit_allocate() };
        let q1 = unsafe { __quantum__rt__qubit_allocate() };
        unsafe { __quantum__qis__h__body(q0) };
        unsafe { __quantum__qis__cx__body(q0, q1) };
        let _r0 = unsafe { __quantum__rt__result_allocate() };
        let _r1 = unsafe { __quantum__rt__result_allocate() };
        unsafe { __quantum__qis__m__body(q0, 0) };
        unsafe { __quantum__qis__m__body(q1, 1) };

        with_interface(|iface| {
            assert_eq!(iface.operations.len(), 8);
            assert_eq!(iface.operations[0], Operation::AllocateQubit { id: 0 });
            assert_eq!(iface.operations[1], Operation::AllocateQubit { id: 1 });
            assert_eq!(iface.operations[2], Operation::Quantum(QuantumOp::H(0)));
            assert_eq!(iface.operations[3], Operation::Quantum(QuantumOp::CX(0, 1)));
            assert_eq!(iface.operations[4], Operation::AllocateResult { id: 0 });
            assert_eq!(iface.operations[5], Operation::AllocateResult { id: 1 });
            assert_eq!(
                iface.operations[6],
                Operation::Quantum(QuantumOp::Measure(0, 0))
            );
            assert_eq!(
                iface.operations[7],
                Operation::Quantum(QuantumOp::Measure(1, 1))
            );
        });
    }
}
