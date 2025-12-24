//! FFI exports for linking with QIS LLVM IR programs
//!
//! This module provides the minimal set of FFI functions needed to link QIS programs
//! with Rust. These functions simply collect operations into the thread-local interface
//! without performing any simulation or complex state management.

use crate::{Operation, QuantumOp, with_interface};
use log::debug;

/// Helper to convert i64 to usize
#[inline]
fn i64_to_usize(value: i64) -> usize {
    usize::try_from(value).expect("Invalid ID: value must be non-negative and fit in usize")
}

// =============================================================================
// Single-Qubit Gates
// =============================================================================

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

/// Pauli-Y gate operation
///
/// # Safety
/// This function is safe to call from C/LLVM code. The qubit parameter must be a valid
/// non-negative qubit ID that fits in usize. Invalid IDs will cause a panic.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __quantum__qis__y__body(qubit: i64) {
    let qubit_id = i64_to_usize(qubit);
    with_interface(|interface| {
        interface.queue_operation(QuantumOp::Y(qubit_id).into());
    });
}

/// Pauli-Z gate operation
///
/// # Safety
/// This function is safe to call from C/LLVM code. The qubit parameter must be a valid
/// non-negative qubit ID that fits in usize. Invalid IDs will cause a panic.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __quantum__qis__z__body(qubit: i64) {
    let qubit_id = i64_to_usize(qubit);
    with_interface(|interface| {
        interface.queue_operation(QuantumOp::Z(qubit_id).into());
    });
}

/// S gate (phase) operation
///
/// # Safety
/// This function is safe to call from C/LLVM code. The qubit parameter must be a valid
/// non-negative qubit ID that fits in usize. Invalid IDs will cause a panic.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __quantum__qis__s__body(qubit: i64) {
    let qubit_id = i64_to_usize(qubit);
    with_interface(|interface| {
        interface.queue_operation(QuantumOp::S(qubit_id).into());
    });
}

/// S-dagger gate (inverse phase) operation
///
/// # Safety
/// This function is safe to call from C/LLVM code. The qubit parameter must be a valid
/// non-negative qubit ID that fits in usize. Invalid IDs will cause a panic.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __quantum__qis__sdg__body(qubit: i64) {
    let qubit_id = i64_to_usize(qubit);
    with_interface(|interface| {
        interface.queue_operation(QuantumOp::Sdg(qubit_id).into());
    });
}

/// T gate (π/8 phase) operation
///
/// # Safety
/// This function is safe to call from C/LLVM code. The qubit parameter must be a valid
/// non-negative qubit ID that fits in usize. Invalid IDs will cause a panic.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __quantum__qis__t__body(qubit: i64) {
    let qubit_id = i64_to_usize(qubit);
    with_interface(|interface| {
        interface.queue_operation(QuantumOp::T(qubit_id).into());
    });
}

/// T-dagger gate (inverse π/8 phase) operation
///
/// # Safety
/// This function is safe to call from C/LLVM code. The qubit parameter must be a valid
/// non-negative qubit ID that fits in usize. Invalid IDs will cause a panic.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __quantum__qis__tdg__body(qubit: i64) {
    let qubit_id = i64_to_usize(qubit);
    with_interface(|interface| {
        interface.queue_operation(QuantumOp::Tdg(qubit_id).into());
    });
}

// =============================================================================
// Two-Qubit Gates
// =============================================================================

/// Controlled-X (CNOT) gate operation
///
/// # Safety
/// This function is safe to call from C/LLVM code. The control and target parameters must be
/// valid non-negative qubit IDs that fit in usize. Invalid IDs will cause a panic.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __quantum__qis__cx__body(control: i64, target: i64) {
    let control_id = i64_to_usize(control);
    let target_id = i64_to_usize(target);
    with_interface(|interface| {
        interface.queue_operation(QuantumOp::CX(control_id, target_id).into());
    });
}

/// CNOT gate operation (alias for CX)
///
/// # Safety
/// This function is safe to call from C/LLVM code. The control and target parameters must be
/// valid non-negative qubit IDs that fit in usize. Invalid IDs will cause a panic.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __quantum__qis__cnot__body(control: i64, target: i64) {
    // CNOT is an alias for CX
    unsafe { __quantum__qis__cx__body(control, target) };
}

/// Controlled-Y gate operation
///
/// # Safety
/// This function is safe to call from C/LLVM code. The control and target parameters must be
/// valid non-negative qubit IDs that fit in usize. Invalid IDs will cause a panic.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __quantum__qis__cy__body(control: i64, target: i64) {
    let control_id = i64_to_usize(control);
    let target_id = i64_to_usize(target);
    with_interface(|interface| {
        interface.queue_operation(QuantumOp::CY(control_id, target_id).into());
    });
}

/// Controlled-Z gate operation
///
/// # Safety
/// This function is safe to call from C/LLVM code. The control and target parameters must be
/// valid non-negative qubit IDs that fit in usize. Invalid IDs will cause a panic.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __quantum__qis__cz__body(control: i64, target: i64) {
    let control_id = i64_to_usize(control);
    let target_id = i64_to_usize(target);
    with_interface(|interface| {
        interface.queue_operation(QuantumOp::CZ(control_id, target_id).into());
    });
}

/// Controlled-H gate operation
///
/// # Safety
/// This function is safe to call from C/LLVM code. The control and target parameters must be
/// valid non-negative qubit IDs that fit in usize. Invalid IDs will cause a panic.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __quantum__qis__ch__body(control: i64, target: i64) {
    let control_id = i64_to_usize(control);
    let target_id = i64_to_usize(target);
    with_interface(|interface| {
        interface.queue_operation(QuantumOp::CH(control_id, target_id).into());
    });
}

// =============================================================================
// Rotation Gates
// =============================================================================

/// Rotation around X-axis
///
/// # Safety
/// This function is safe to call from C/LLVM code. The qubit parameter must be a valid
/// non-negative qubit ID that fits in usize. Invalid IDs will cause a panic.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __quantum__qis__rx__body(theta: f64, qubit: i64) {
    let qubit_id = i64_to_usize(qubit);
    with_interface(|interface| {
        interface.queue_operation(QuantumOp::RX(theta, qubit_id).into());
    });
}

/// Rotation around Y-axis
///
/// # Safety
/// This function is safe to call from C/LLVM code. The qubit parameter must be a valid
/// non-negative qubit ID that fits in usize. Invalid IDs will cause a panic.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __quantum__qis__ry__body(theta: f64, qubit: i64) {
    let qubit_id = i64_to_usize(qubit);
    with_interface(|interface| {
        interface.queue_operation(QuantumOp::RY(theta, qubit_id).into());
    });
}

/// Rotation around Z-axis
///
/// # Safety
/// This function is safe to call from C/LLVM code. The qubit parameter must be a valid
/// non-negative qubit ID that fits in usize. Invalid IDs will cause a panic.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __quantum__qis__rz__body(theta: f64, qubit: i64) {
    let qubit_id = i64_to_usize(qubit);
    with_interface(|interface| {
        interface.queue_operation(QuantumOp::RZ(theta, qubit_id).into());
    });
}

/// ZZ rotation gate
///
/// # Safety
/// This function is safe to call from C/LLVM code. The qubit parameters must be valid
/// non-negative qubit IDs that fit in usize. Invalid IDs will cause a panic.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __quantum__qis__rzz__body(theta: f64, qubit1: i64, qubit2: i64) {
    let qubit1_id = i64_to_usize(qubit1);
    let qubit2_id = i64_to_usize(qubit2);
    with_interface(|interface| {
        interface.queue_operation(QuantumOp::RZZ(theta, qubit1_id, qubit2_id).into());
    });
}

/// Single-qubit rotation in XY plane
///
/// # Safety
/// This function is safe to call from C/LLVM code. The qubit parameter must be a valid
/// non-negative qubit ID that fits in usize. Invalid IDs will cause a panic.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __quantum__qis__r1xy__body(theta: f64, phi: f64, qubit: i64) {
    let qubit_id = i64_to_usize(qubit);
    with_interface(|interface| {
        interface.queue_operation(QuantumOp::RXY(theta, phi, qubit_id).into());
    });
}

/// Controlled rotation around Z-axis
///
/// # Safety
/// This function is safe to call from C/LLVM code. The control and target parameters must be
/// valid non-negative qubit IDs that fit in usize. Invalid IDs will cause a panic.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __quantum__qis__crz__body(theta: f64, control: i64, target: i64) {
    let control_id = i64_to_usize(control);
    let target_id = i64_to_usize(target);
    with_interface(|interface| {
        interface.queue_operation(QuantumOp::CRZ(theta, control_id, target_id).into());
    });
}

// =============================================================================
// Three-Qubit Gates
// =============================================================================

/// Toffoli (CCX) gate operation
///
/// # Safety
/// This function is safe to call from C/LLVM code. All qubit parameters must be valid
/// non-negative qubit IDs that fit in usize. Invalid IDs will cause a panic.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __quantum__qis__ccx__body(control1: i64, control2: i64, target: i64) {
    let control1_id = i64_to_usize(control1);
    let control2_id = i64_to_usize(control2);
    let target_id = i64_to_usize(target);
    with_interface(|interface| {
        interface.queue_operation(QuantumOp::CCX(control1_id, control2_id, target_id).into());
    });
}

// =============================================================================
// ZZ Interaction
// =============================================================================

/// ZZ interaction gate
///
/// # Safety
/// This function is safe to call from C/LLVM code. The qubit parameters must be valid
/// non-negative qubit IDs that fit in usize. Invalid IDs will cause a panic.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __quantum__qis__zz__body(qubit1: i64, qubit2: i64) {
    let qubit1_id = i64_to_usize(qubit1);
    let qubit2_id = i64_to_usize(qubit2);
    with_interface(|interface| {
        interface.queue_operation(QuantumOp::ZZ(qubit1_id, qubit2_id).into());
    });
}

// =============================================================================
// Measurement and Reset
// =============================================================================

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

/// Reset a qubit to |0⟩ state
///
/// # Safety
/// This function is safe to call from C/LLVM code. The qubit parameter must be a valid
/// non-negative qubit ID that fits in usize. Invalid IDs will cause a panic.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __quantum__qis__reset__body(qubit: i64) {
    let qubit_id = i64_to_usize(qubit);
    with_interface(|interface| {
        interface.queue_operation(QuantumOp::Reset(qubit_id).into());
    });
}

// =============================================================================
// Allocation and Deallocation
// =============================================================================

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

// =============================================================================
// Result Retrieval (placeholder - actual implementation in runtime)
// =============================================================================

/// Get measurement result (returns 1 if result is One, 0 otherwise)
///
/// # Safety
/// This function is safe to call from C/LLVM code. The result parameter must be a valid
/// non-negative result ID that fits in usize. Invalid IDs will cause a panic.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __quantum__rt__result_get_one(result: i64) -> i32 {
    let result_id = i64_to_usize(result);
    with_interface(|interface| {
        // In the minimal interface, we just return a placeholder
        // The actual result will be available after runtime execution
        interface.get_result(result_id).map_or(0, i32::from)
    })
}

// =============================================================================
// Utility Functions
// =============================================================================

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

// =============================================================================
// Selene-style FFI Functions
//
// These functions match the naming convention used by Selene's hugr-qis compiler.
// They provide the same functionality as the QIS-style functions above but with
// different names to support Selene-generated LLVM IR.
// =============================================================================

/// Reset operation (Selene-style)
///
/// # Safety
/// This function is safe to call from C/LLVM code. The qubit parameter must be a valid
/// non-negative qubit ID that fits in usize. Invalid IDs will cause a panic.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ___reset(qubit: i64) {
    // Delegate to the QIS-style function
    unsafe { __quantum__qis__reset__body(qubit) };
}

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

/// H gate function (Selene-style)
///
/// # Safety
/// This function is safe to call from C/LLVM code. The qubit parameter must be a valid
/// non-negative qubit ID that fits in usize. Invalid IDs will cause a panic.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ___h(qubit: i64) {
    // Delegate to the QIS-style function
    unsafe { __quantum__qis__h__body(qubit) };
}

/// CX gate function (Selene-style)
///
/// # Safety
/// This function is safe to call from C/LLVM code. The control and target parameters must be
/// valid non-negative qubit IDs that fit in usize. Invalid IDs will cause a panic.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ___cx(control: i64, target: i64) {
    // Delegate to the QIS-style function
    unsafe { __quantum__qis__cx__body(control, target) };
}

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
/// # Safety
/// This function is safe to call from C/LLVM code. The `future_id` parameter must be a valid
/// result ID previously returned by `___lazy_measure`. Invalid IDs will cause a panic.
///
/// # Returns
/// Returns the boolean measurement result (true = 1, false = 0).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ___read_future_bool(future_id: i64) -> bool {
    let result_id = i64_to_usize(future_id);
    with_interface(|interface| {
        // Get the measurement result from the interface
        // Returns false if the result is not yet available
        interface.get_result(result_id).unwrap_or(false)
    })
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

// =============================================================================
// QIS measurement functions
// =============================================================================

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

// =============================================================================
// Result printing functions
// =============================================================================

/// Print a boolean result with a label
///
/// This function is called by QIS programs to output measurement results
/// with labels like "`measurement_0`", "`measurement_1`", etc.
///
/// # Arguments
/// * `label_ptr` - Pointer to the label string
/// * `label_len` - Length of the label string
/// * `value` - Boolean value to print
///
/// # Safety
/// This function is safe to call from C/LLVM code. The `label_ptr` must point to a valid byte
/// array of at least `label_len` bytes. Invalid pointers or lengths will cause undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn print_bool(label_ptr: *const u8, label_len: i64, value: bool) {
    // Convert the C string to a Rust string for debugging
    let Ok(label_len) = usize::try_from(label_len) else {
        log::error!("print_bool: invalid label length {label_len}");
        return;
    };
    let label_slice = unsafe { std::slice::from_raw_parts(label_ptr, label_len) };

    // For now, just log the print operation - this prevents segfaults
    // while allowing the program to run
    if let Ok(label_str) = std::str::from_utf8(label_slice) {
        // Log the measurement for debugging
        log::debug!("print_bool called: {label_str} = {value}");
    }

    // TODO: Properly integrate with measurement storage system
    // The current QisInterface uses numeric IDs, but Guppy uses string names
    // This mismatch needs to be resolved in a future update
}

// =============================================================================
// Interface Management (C exports for dlsym access)
// =============================================================================

/// Reset the thread-local interface
/// Exported as C function so it can be called via dlsym from the cdylib
///
/// # Safety
/// This function is safe to call from C/LLVM code.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pecos_qis_reset_interface() {
    crate::reset_interface();
}

/// Get a clone of the current `OperationCollector`
/// Exported as C function so it can be called via dlsym from the cdylib
///
/// # Safety
/// This function is safe to call from C/LLVM code. The returned pointer must be freed using
/// `pecos_qis_free_operations` to avoid memory leaks.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pecos_qis_get_operations() -> *mut crate::OperationCollector {
    let operations = with_interface(|interface| interface.clone());
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

// =============================================================================
// Heap Management Functions (Selene compatibility)
// =============================================================================

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
