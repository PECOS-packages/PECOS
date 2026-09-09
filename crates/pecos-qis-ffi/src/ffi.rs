//! FFI exports for linking with QIS LLVM IR programs
//!
//! Minimal set of FFI functions needed to link QIS programs
//! with Rust. These functions simply collect operations into the thread-local interface
//! without performing any simulation or complex state management.

pub use crate::random::*;

use crate::{Operation, QuantumOp, TraceMetadata, with_interface};
use log::debug;
use std::cell::Cell;

/// C ABI return value for helpers that consume and return two qubits.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct QubitPair {
    pub first: i64,
    pub second: i64,
}

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

// Validate IDs before borrowing the interface. A transfer must never skip a
// live RefMut/MutexGuard. With no handler, return the ABI's zero/empty value.
macro_rules! checked_ffi_id {
    ($entry:expr, $value:expr, $ty:ty) => {
        match <$ty>::try_from($value) {
            Ok(value) => value,
            Err(_) => {
                unsafe {
                    fatal_ffi_input(
                        $entry,
                        format!("ID={} cannot be represented as {}", $value, stringify!($ty)),
                    )
                };
                return Default::default();
            }
        }
    };
}

const PACKED_TRACE_METADATA_JSON_KEY: &str = "__pecos_trace_metadata_json_v1__";

unsafe fn read_tket_string_arg<'a>(
    func_name: &str,
    arg_name: &str,
    ptr: *const u8,
    len: i64,
) -> Option<&'a str> {
    let Ok(len) = usize::try_from(len) else {
        unsafe { fatal_ffi_input(func_name, format!("{arg_name} length={len}")) };
        return None;
    };
    if ptr.is_null() {
        unsafe { fatal_ffi_input(func_name, format!("{arg_name} pointer=null, length={len}")) };
        return None;
    }

    if len > isize::MAX as usize {
        unsafe {
            fatal_ffi_input(
                func_name,
                format!("{arg_name} length={len} exceeds isize::MAX"),
            );
        };
        return None;
    }
    // The tket2 string format is: {len: u8, data: [u8; len]}.
    // The pointer references the length byte, so skip it to read the payload.
    let data_ptr = unsafe { ptr.add(1) };
    let bytes = unsafe { std::slice::from_raw_parts(data_ptr, len) };
    if let Ok(value) = std::str::from_utf8(bytes) {
        Some(value)
    } else {
        unsafe { fatal_ffi_input(func_name, format!("invalid UTF-8 in {arg_name}: {bytes:?}")) };
        None
    }
}

unsafe fn read_direct_string_arg<'a>(
    func_name: &str,
    arg_name: &str,
    ptr: *const u8,
    len: i64,
) -> Option<&'a str> {
    let Ok(len) = usize::try_from(len) else {
        unsafe { fatal_ffi_input(func_name, format!("{arg_name} length={len}")) };
        return None;
    };
    if ptr.is_null() {
        unsafe { fatal_ffi_input(func_name, format!("{arg_name} pointer=null, length={len}")) };
        return None;
    }

    if len > isize::MAX as usize {
        unsafe {
            fatal_ffi_input(
                func_name,
                format!("{arg_name} length={len} exceeds isize::MAX"),
            );
        };
        return None;
    }
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
    if let Ok(value) = std::str::from_utf8(bytes) {
        Some(value)
    } else {
        unsafe { fatal_ffi_input(func_name, format!("invalid UTF-8 in {arg_name}: {bytes:?}")) };
        None
    }
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
            let qubit_id = checked_ffi_id!(stringify!($name), qubit, usize);
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
            let q1_id = checked_ffi_id!(stringify!($name), q1, usize);
            let q2_id = checked_ffi_id!(stringify!($name), q2, usize);
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
            let q1_id = checked_ffi_id!(stringify!($name), q1, usize);
            let q2_id = checked_ffi_id!(stringify!($name), q2, usize);
            let q3_id = checked_ffi_id!(stringify!($name), q3, usize);
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
            let qubit_id = checked_ffi_id!(stringify!($name), qubit, usize);
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
            let q1_id = checked_ffi_id!(stringify!($name), q1, usize);
            let q2_id = checked_ffi_id!(stringify!($name), q2, usize);
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
/// non-negative qubit ID that fits in usize. Invalid IDs produce a program error.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __quantum__qis__h__body(qubit: i64) {
    debug!("[FFI] __quantum__qis__h__body called with qubit={qubit}");
    let qubit_id = checked_ffi_id!(stringify!(__quantum__qis__h__body), qubit, usize);
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
/// non-negative qubit ID that fits in usize. Invalid IDs produce a program error.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __quantum__qis__x__body(qubit: i64) {
    debug!("[FFI] __quantum__qis__x__body called with qubit={qubit}");
    let qubit_id = checked_ffi_id!(stringify!(__quantum__qis__x__body), qubit, usize);
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
    let qubit_id = checked_ffi_id!(stringify!(__quantum__qis__r1xy__body), qubit, usize);
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
/// non-negative IDs that fit in usize. Invalid IDs produce a program error.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __quantum__qis__m__body(qubit: i64, result: i64) -> i32 {
    let qubit_id = checked_ffi_id!(stringify!(__quantum__qis__m__body), qubit, usize);
    let result_id = checked_ffi_id!(stringify!(__quantum__qis__m__body), result, usize);
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
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __quantum__rt__qubit_allocate() -> i64 {
    let allocated_id = with_interface(|interface| {
        let id = interface.allocate_qubit();
        interface.queue_operation(Operation::AllocateQubit { id });
        id
    });
    checked_ffi_id!("__quantum__rt__qubit_allocate", allocated_id, i64)
}

/// Release (deallocate) a qubit
///
/// # Safety
/// This function is safe to call from C/LLVM code. The qubit parameter must be a valid
/// non-negative qubit ID that fits in usize. Invalid IDs produce a program error.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __quantum__rt__qubit_release(qubit: i64) {
    let qubit_id = checked_ffi_id!(stringify!(__quantum__rt__qubit_release), qubit, usize);
    with_interface(|interface| {
        interface.queue_operation(Operation::ReleaseQubit { id: qubit_id });
    });
}

/// Allocate a new result storage
///
/// # Safety
/// This function is safe to call from C/LLVM code.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __quantum__rt__result_allocate() -> i64 {
    let allocated_id = with_interface(|interface| {
        let id = interface.allocate_result();
        interface.queue_operation(Operation::AllocateResult { id });
        id
    });
    checked_ffi_id!("__quantum__rt__result_allocate", allocated_id, i64)
}

// --- Result Retrieval ---

fn record_result_read(result_id: usize) {
    if let Some(ctx) = crate::get_execution_context() {
        // SAFETY: Context is valid for duration of execution.
        unsafe { &*ctx }.record_result_read(result_id);
    }
}

/// Get measurement result (returns 1 if result is One, 0 otherwise)
///
/// This function supports dynamic circuits: if the result is not yet available and
/// a quantum executor callback has been registered, it will execute pending quantum
/// operations to obtain the measurement result.
///
/// # Safety
/// This function is safe to call from C/LLVM code. The result parameter must be a valid
/// non-negative result ID that fits in usize. Invalid IDs produce a program error.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __quantum__rt__result_get_one(result: i64) -> i32 {
    log::debug!("__quantum__rt__result_get_one called with result={result}");
    let result_id = checked_ffi_id!(stringify!(__quantum__rt__result_get_one), result, usize);

    // First check if result is already available
    let existing_result = with_interface(|interface| interface.get_result(result_id));

    if let Some(value) = existing_result {
        record_result_read(result_id);
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
                |value| {
                    record_result_read(result_id);
                    i32::from(value)
                },
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

fn trace_metadata_from_key_value(
    func_name: &str,
    key: String,
    value: String,
) -> Option<TraceMetadata> {
    if key != PACKED_TRACE_METADATA_JSON_KEY {
        let mut metadata = TraceMetadata::new();
        metadata.insert(key, value);
        return Some(metadata);
    }

    match serde_json::from_str::<TraceMetadata>(&value) {
        Ok(metadata) => Some(metadata),
        Err(err) => {
            log::error!("{func_name}: invalid packed trace metadata JSON: {err}");
            None
        }
    }
}

fn queue_trace_metadata(func_name: &str, key: String, value: String, qubit: Option<usize>) {
    let Some(metadata) = trace_metadata_from_key_value(func_name, key, value) else {
        return;
    };
    with_interface(|interface| {
        interface.queue_operation(Operation::TraceMetadata { metadata, qubit });
    });
}

/// Attach source/runtime metadata to the next lowerable quantum operation.
///
/// This function uses the tket2 string ABI: each string pointer references a
/// `{len: u8, data: [u8; len]}` payload and the length argument gives the data
/// length. Metadata is intentionally represented as ordinary key/value strings
/// so callers can add generic provenance without PECOS knowing about a specific
/// runtime or hardware target.
///
/// # Safety
/// The key and value pointers must be valid tket2 string structs with at least
/// `len + 1` bytes. Invalid pointers cause undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pecos_qis_trace_metadata(
    key_ptr: *const u8,
    key_len: i64,
    value_ptr: *const u8,
    value_len: i64,
) {
    let Some(key) =
        (unsafe { read_tket_string_arg("pecos_qis_trace_metadata", "key", key_ptr, key_len) })
    else {
        return;
    };
    let Some(value) = (unsafe {
        read_tket_string_arg("pecos_qis_trace_metadata", "value", value_ptr, value_len)
    }) else {
        return;
    };
    queue_trace_metadata(
        "pecos_qis_trace_metadata",
        key.to_owned(),
        value.to_owned(),
        None,
    );
}

/// Attach source/runtime metadata to the next lowerable quantum operation.
///
/// This variant matches the HUGR lowering ABI for Guppy string arguments: each
/// argument is passed as a pointer to a tket2 string payload whose first byte is
/// the string length.
///
/// # Safety
/// The key and value pointers must be valid tket2 string structs. Invalid
/// pointers cause undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pecos_qis_trace_metadata_hugr(key_ptr: *const u8, value_ptr: *const u8) {
    if key_ptr.is_null() {
        log::error!("pecos_qis_trace_metadata_hugr: null key pointer");
        return;
    }
    if value_ptr.is_null() {
        log::error!("pecos_qis_trace_metadata_hugr: null value pointer");
        return;
    }

    let key_len = i64::from(unsafe { *key_ptr });
    let value_len = i64::from(unsafe { *value_ptr });
    let Some(key) =
        (unsafe { read_tket_string_arg("pecos_qis_trace_metadata_hugr", "key", key_ptr, key_len) })
    else {
        return;
    };
    let Some(value) = (unsafe {
        read_tket_string_arg(
            "pecos_qis_trace_metadata_hugr",
            "value",
            value_ptr,
            value_len,
        )
    }) else {
        return;
    };
    queue_trace_metadata(
        "pecos_qis_trace_metadata_hugr",
        key.to_owned(),
        value.to_owned(),
        None,
    );
}

/// Attach source/runtime metadata to the next operation on a specific qubit.
///
/// Returning the qubit handle gives Guppy/HUGR a data dependency that preserves
/// the metadata call immediately before the gate it annotates.
///
/// # Safety
/// The key and value pointers must be valid tket2 string structs. Invalid
/// pointers cause undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pecos_qis_trace_metadata_qubit_hugr(
    qubit: i64,
    key_ptr: *const u8,
    value_ptr: *const u8,
) -> i64 {
    if key_ptr.is_null() {
        log::error!("pecos_qis_trace_metadata_qubit_hugr: null key pointer");
        return qubit;
    }
    if value_ptr.is_null() {
        log::error!("pecos_qis_trace_metadata_qubit_hugr: null value pointer");
        return qubit;
    }

    let key_len = i64::from(unsafe { *key_ptr });
    let value_len = i64::from(unsafe { *value_ptr });
    let Some(key) = (unsafe {
        read_tket_string_arg(
            "pecos_qis_trace_metadata_qubit_hugr",
            "key",
            key_ptr,
            key_len,
        )
    }) else {
        return qubit;
    };
    let Some(value) = (unsafe {
        read_tket_string_arg(
            "pecos_qis_trace_metadata_qubit_hugr",
            "value",
            value_ptr,
            value_len,
        )
    }) else {
        return qubit;
    };
    let qubit_id = checked_ffi_id!(
        stringify!(pecos_qis_trace_metadata_qubit_hugr),
        qubit,
        usize
    );
    queue_trace_metadata(
        "pecos_qis_trace_metadata_qubit_hugr",
        key.to_owned(),
        value.to_owned(),
        Some(qubit_id),
    );
    qubit
}

/// Insert a runtime scheduling barrier after prior operations touching this qubit.
///
/// Returning the qubit handle gives Guppy/HUGR a data dependency that keeps the
/// barrier between the preceding operation on this qubit and the following
/// operation that consumes the returned handle. The barrier itself is a
/// runtime-level batch/drain marker; it does not emit a quantum gate.
///
/// # Safety
/// Called from C/LLVM code. Qubit must be a valid non-negative ID.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pecos_qis_runtime_barrier_qubit_hugr(qubit: i64) -> i64 {
    let _ = checked_ffi_id!(
        stringify!(pecos_qis_runtime_barrier_qubit_hugr),
        qubit,
        usize
    );
    with_interface(|interface| {
        interface.queue_operation(Operation::Barrier);
    });
    qubit
}

/// Insert a runtime scheduling barrier after prior operations touching either qubit.
///
/// The returned qubit pair gives Guppy/HUGR data dependencies on both inputs. A
/// caller can place this helper immediately before a hosted local pulse so that
/// the local pulse cannot be scheduled before the host qubit is ready.
///
/// # Safety
/// Called from C/LLVM code. Qubits must be valid non-negative IDs.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pecos_qis_runtime_barrier_qubits2_hugr(
    first: i64,
    second: i64,
) -> QubitPair {
    let _ = checked_ffi_id!(
        stringify!(pecos_qis_runtime_barrier_qubits2_hugr),
        first,
        usize
    );
    let _ = checked_ffi_id!(
        stringify!(pecos_qis_runtime_barrier_qubits2_hugr),
        second,
        usize
    );
    with_interface(|interface| {
        interface.queue_operation(Operation::Barrier);
    });
    QubitPair { first, second }
}

/// Attach source/runtime metadata to the next lowerable quantum operation.
///
/// This variant uses direct string data pointers instead of the tket2 string
/// struct layout. It is useful for runtime shims that already carry plain
/// pointer/length pairs.
///
/// # Safety
/// The key and value pointers must reference valid UTF-8 data of the provided
/// lengths. Invalid pointers cause undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pecos_qis_trace_metadata_direct(
    key_ptr: *const u8,
    key_len: i64,
    value_ptr: *const u8,
    value_len: i64,
) {
    let Some(key) = (unsafe {
        read_direct_string_arg("pecos_qis_trace_metadata_direct", "key", key_ptr, key_len)
    }) else {
        return;
    };
    let Some(value) = (unsafe {
        read_direct_string_arg(
            "pecos_qis_trace_metadata_direct",
            "value",
            value_ptr,
            value_len,
        )
    }) else {
        return;
    };
    queue_trace_metadata(
        "pecos_qis_trace_metadata_direct",
        key.to_owned(),
        value.to_owned(),
        None,
    );
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
/// non-negative qubit ID that fits in usize. Invalid IDs produce a program error.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ___rxy(qubit: i64, theta: f64, phi: f64) {
    // Delegate to the QIS-style function
    unsafe { __quantum__qis__r1xy__body(theta, phi, qubit) };
}

/// RZ rotation (Selene-style)
///
/// # Safety
/// This function is safe to call from C/LLVM code. The qubit parameter must be a valid
/// non-negative qubit ID that fits in usize. Invalid IDs produce a program error.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ___rz(qubit: i64, theta: f64) {
    // Delegate to the QIS-style function
    unsafe { __quantum__qis__rz__body(theta, qubit) };
}

/// RZZ two-qubit rotation (Selene-style)
///
/// # Safety
/// This function is safe to call from C/LLVM code. The qubit parameters must be valid
/// non-negative qubit IDs that fit in usize. Invalid IDs produce a program error.
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
/// non-negative qubit ID that fits in usize. Invalid IDs produce a program error.
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
/// non-negative qubit ID that fits in usize. Invalid IDs produce a program error.
///
/// # Returns
/// Returns the allocated result ID as i64.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ___lazy_measure(qubit: i64) -> i64 {
    let qubit_id = checked_ffi_id!(stringify!(___lazy_measure), qubit, usize);
    let allocated_id = with_interface(|interface| {
        // Allocate a result ID for this measurement
        let result_id = interface.allocate_result();
        // Queue the allocation operation
        interface.queue_operation(Operation::AllocateResult { id: result_id });
        // Queue the measurement operation
        interface.queue_operation(QuantumOp::Measure(qubit_id, result_id).into());
        // Return the result ID
        result_id
    });
    checked_ffi_id!("___lazy_measure", allocated_id, i64)
}

/// Lazy leakage-aware measurement function (Selene/HUGR-LLVM style).
///
/// # Safety
/// The same requirements as [`___lazy_measure`] apply.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ___lazy_measure_leaked(qubit: i64) -> i64 {
    let qubit_id = checked_ffi_id!(stringify!(___lazy_measure_leaked), qubit, usize);
    let allocated_id = with_interface(|interface| {
        let result_id = interface.allocate_result();
        interface.queue_operation(Operation::AllocateResult { id: result_id });
        interface.queue_operation(QuantumOp::MeasureLeaked(qubit_id, result_id).into());
        result_id
    });
    checked_ffi_id!("___lazy_measure_leaked", allocated_id, i64)
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
/// result ID previously returned by `___lazy_measure`. Invalid IDs produce a program error.
///
/// # Returns
/// Returns the boolean measurement result (true = 1, false = 0).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ___read_future_bool(future_id: i64) -> bool {
    log::debug!("___read_future_bool called with future_id={future_id}");
    let result_id = checked_ffi_id!(stringify!(___read_future_bool), future_id, usize);

    // Check if result is already available in thread-local storage
    let existing_result = with_interface(|interface| interface.get_result(result_id));
    log::debug!("___read_future_bool: existing_result={existing_result:?}");

    if let Some(result) = existing_result {
        record_result_read(result_id);
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
            record_result_read(result_id);
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
            if result.is_some() {
                record_result_read(result_id);
            }
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

/// Read an integer-valued measurement future (Selene/HUGR-LLVM style).
///
/// Leakage-aware Guppy measurements use an unsigned future with outcomes 0, 1,
/// or 2 (leaked).
///
/// # Safety
/// The same requirements as [`___read_future_bool`] apply.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ___read_future_uint(future_id: i64) -> u64 {
    log::debug!("___read_future_uint called with future_id={future_id}");
    let result_id = checked_ffi_id!(stringify!(___read_future_uint), future_id, usize);

    if crate::is_dynamic_mode_active() {
        if let Some(result) = crate::get_measurement_outcome(result_id as u64) {
            record_result_read(result_id);
            return result;
        }
        if crate::wait_for_result_ready(result_id as u64, 30_000) {
            let result = crate::get_measurement_outcome(result_id as u64);
            if result.is_some() {
                record_result_read(result_id);
            }
            return result.unwrap_or(0);
        }
    }

    // Static collection cannot synthesize a leak. Reuse the Boolean collection
    // behavior so bounded probing and repeat-until-success termination remain
    // consistent with ordinary measurements.
    u64::from(unsafe { ___read_future_bool(future_id) })
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

thread_local! {
    // Installed only while this thread is inside the C shim's setjmp wrapper.
    static PROGRAM_PANIC_TRANSFER: std::cell::Cell<Option<unsafe extern "C" fn()>> = const {
        std::cell::Cell::new(None)
    };
}

/// Register the shim's C longjmp function, or clear it after leaving the guard.
///
/// # Safety
/// A non-null handler must be safe to invoke on this thread and remain valid
/// until replaced. A transferring handler must target a live setjmp. Only the
/// execution wrapper may install it.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pecos_set_program_panic_handler(handler: Option<unsafe extern "C" fn()>) {
    PROGRAM_PANIC_TRANSFER.set(handler);
}

/// Get the current handler so an execution wrapper can save and restore it.
#[unsafe(no_mangle)]
pub extern "C" fn pecos_get_program_panic_handler() -> Option<unsafe extern "C" fn()> {
    PROGRAM_PANIC_TRANSFER.get()
}

/// Whether this thread has a live execution guard. The C transfer checks this.
#[unsafe(no_mangle)]
pub extern "C" fn pecos_program_panic_handler_is_installed() -> bool {
    PROGRAM_PANIC_TRANSFER.get().is_some()
}

fn record_invalid_input(entry: &str, detail: String) {
    if let Some(ctx) = crate::get_execution_context() {
        unsafe { &*ctx }.record_program_error(crate::ProgramError::InvalidInput {
            entry: entry.to_string(),
            detail,
        });
    } else {
        log::error!("QIS invalid FFI input in {entry}: {detail}: no execution context registered");
    }
}

/// Record first, then transfer only after all recording temporaries are dropped.
///
/// # Safety
/// Callers must release every owned value and mutex/TLS borrow before invoking
/// this path. The skipped Rust frames must not own live destructors.
pub(super) unsafe fn fatal_ffi_input(entry: &str, detail: String) {
    record_invalid_input(entry, detail);
    if let Some(transfer) = PROGRAM_PANIC_TRANSFER.get() {
        unsafe { transfer() };
    }
}

fn checked_slice_len<T>(len: u64) -> Result<usize, String> {
    let count = usize::try_from(len).map_err(|_| format!("length={len} does not fit usize"))?;
    if count > (isize::MAX as usize) / std::mem::size_of::<T>() {
        return Err(format!(
            "length={len} with element size={} exceeds isize::MAX bytes",
            std::mem::size_of::<T>()
        ));
    }
    Ok(count)
}

/// Record a panic with plain string data, also used by the Selene shim.
///
/// # Safety
/// `message` must be null or reference `len` readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pecos_record_program_panic(code: i32, message: *const u8, len: usize) {
    if len > isize::MAX as usize {
        unsafe {
            fatal_ffi_input(
                "pecos_record_program_panic",
                format!("message length={len} exceeds isize::MAX"),
            );
        };
        return;
    }
    let message = if message.is_null() {
        "Unknown error (null panic message)".to_string()
    } else {
        String::from_utf8_lossy(unsafe { std::slice::from_raw_parts(message, len) }).into_owned()
    };
    if let Some(ctx) = crate::get_execution_context() {
        // guppylang std/platform.py's exit/panic convention lowers to exit
        // codes 0..=1000 and panic codes signal + 1000. Apply it at the C ABI
        // to every producer, independently of the source function's name.
        // Guppy documents supported signals 1..=1000. panic(msg, 0) lowers
        // to 1000 (an exit here), but signal 0 is outside that contract.
        let termination = if (0..=1000).contains(&code) {
            crate::ProgramError::Exit { code, message }
        } else {
            crate::ProgramError::Panic { code, message }
        };
        if matches!(termination, crate::ProgramError::Exit { .. }) {
            log::debug!("{termination}");
        }
        unsafe { &*ctx }.record_program_error(termination);
    } else {
        log::error!(
            "Cannot record program termination: code={code}, message={message}: no execution context registered"
        );
    }
}

/// Panic function called on program errors, with tket's length-prefixed message.
///
/// Guppylang emits calls to this direct symbol, rather than routing through the
/// Selene shim. For division by zero its LLVM IR contains:
/// ```llvm
/// @"e_Attempted .0BD5FABD.0" = private constant [33 x i8] c" EXIT:INT:Attempted division by 0"
/// tail call void @panic(i32 1002, ptr nonnull @"e_Attempted .0BD5FABD.0")
/// ```
/// The leading space is the length byte 0x20: the following 32 bytes are the
/// `EXIT:INT:Attempted division by 0` payload, without a C-string terminator.
///
/// Recording finishes before the C transfer is invoked: this deliberately
/// skipped Rust frame owns only raw pointers, integers and a function pointer,
/// and no destructor or TLS borrow remains live across longjmp. Outside an
/// execution guard there is no program to jump out of; the error stays recorded.
///
/// # Safety
/// `message` must be null or point to a length byte followed by that many bytes.
/// Program execution must take place inside the shim's setjmp wrapper.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn panic(code: i32, message: *const std::ffi::c_char) {
    if message.is_null() {
        unsafe { pecos_record_program_panic(code, std::ptr::null(), 0) };
    } else {
        let ptr = message.cast::<u8>();
        unsafe { pecos_record_program_panic(code, ptr.add(1), usize::from(*ptr)) };
    }
    if let Some(transfer) = PROGRAM_PANIC_TRANSFER.get() {
        unsafe { transfer() };
    }
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
/// non-negative qubit ID that fits in usize. Invalid IDs produce a program error.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __quantum__qis__mz__body(qubit: i64) -> i32 {
    // Call our standard measurement function with result ID = qubit ID
    unsafe { __quantum__qis__m__body(qubit, qubit) }
}

// --- Result printing functions ---

/// Dense 1D array matching tket's `{i32 x, i32 y, ptr data, ptr mask}` ABI.
#[repr(C)]
pub struct Dense1DArray<T> {
    pub x: i32,
    pub y: i32,
    pub data: *const T,
    pub mask: *const bool,
}

pub type Dense1DArrayBool = Dense1DArray<bool>;
pub type Dense1DArrayInt = Dense1DArray<i64>;
pub type Dense1DArrayUint = Dense1DArray<u64>;
pub type Dense1DArrayFloat = Dense1DArray<f64>;

/// Both string ABIs converge here after the direct form skips its length byte.
unsafe fn record_named_output(
    label_ptr: *const u8,
    label_len: i64,
    values: crate::NamedResult,
    scalar_prefix: &str,
    array_prefix: &str,
    entry: &str,
    is_scalar: bool,
) {
    let Ok(len) = usize::try_from(label_len) else {
        drop(values);
        unsafe {
            fatal_ffi_input(
                entry,
                format!("label length={label_len} does not fit usize"),
            );
        };
        return;
    };
    if label_ptr.is_null() {
        drop(values);
        unsafe { fatal_ffi_input(entry, format!("label pointer=null, length={label_len}")) };
        return;
    }
    if len > isize::MAX as usize {
        drop(values);
        unsafe {
            fatal_ffi_input(
                entry,
                format!("label length={label_len} exceeds isize::MAX"),
            );
        };
        return;
    }
    // SAFETY: The caller provides label_len readable bytes of string data.
    let bytes = unsafe { std::slice::from_raw_parts(label_ptr, len) };
    let Ok(label) = std::str::from_utf8(bytes) else {
        drop(values);
        unsafe { fatal_ffi_input(entry, format!("invalid UTF-8 label={bytes:?}")) };
        return;
    };
    let name = label
        .strip_prefix(scalar_prefix)
        .or_else(|| label.strip_prefix(array_prefix))
        .unwrap_or(label);
    if let Some(ctx) = crate::get_execution_context() {
        // SAFETY: The registered context lives for the duration of execution.
        unsafe { &*ctx }.store_named_result(name, values, is_scalar);
    } else {
        log::error!("Cannot record named result '{name}': no execution context registered");
    }
}

unsafe fn result_array<'a, T>(ptr: *const T, len: u64) -> Result<&'a [T], String> {
    let len = checked_slice_len::<T>(len)?;
    if len == 0 {
        // Compatibility exception: the parent rejected null data even at zero
        // length. Accepting it now records an empty Bool result and empty trace.
        // This is the sole exception to bool byte-compatibility; it drains nothing.
        return Ok(&[]);
    }
    if ptr.is_null() || !ptr.is_aligned() {
        return Err(format!(
            "array pointer={ptr:?}, length={len} is null or misaligned"
        ));
    }
    // SAFETY: The caller supplies len initialized elements; alignment and the
    // isize::MAX byte bound were checked before constructing the slice.
    Ok(unsafe { std::slice::from_raw_parts(ptr, len) })
}

// As with the gate export macros, keep ABI variants together so every type
// receives identical validation, label handling, and storage behavior.
macro_rules! named_result_exports {
    ($scalar:ident, $array:ident, $selene_scalar:ident, $selene_array:ident,
     $ty:ty, $dense:ident, $variant:ident, $prefix:literal) => {
        /// Record a scalar using tket's length-prefixed string format.
        ///
        /// # Safety
        /// `label_ptr` must reference a tket string with at least `label_len + 1`
        /// bytes. The first byte is the length, and the data starts at byte one.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $scalar(label_ptr: *const u8, label_len: i64, value: $ty) {
            let data = if label_ptr.is_null() {
                label_ptr
            } else {
                unsafe { label_ptr.add(1) }
            };
            unsafe { $selene_scalar(data, label_len, value) };
        }

        /// Record an array using tket's string and dense array formats.
        ///
        /// # Safety
        /// `label_ptr` must reference `label_len + 1` bytes. `arr` must point to
        /// a valid dense array with `x` initialized elements (data may be null
        /// when `x` is zero).
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $array(label_ptr: *const u8, label_len: i64, arr: *const $dense) {
            if arr.is_null() || !arr.is_aligned() {
                unsafe {
                    fatal_ffi_input(
                        stringify!($array),
                        format!("array pointer={arr:?} is null or misaligned"),
                    )
                };
                return;
            }
            let arr = unsafe { &*arr };
            let Ok(len) = u64::try_from(arr.x) else {
                unsafe { fatal_ffi_input(stringify!($array), format!("array length={}", arr.x)) };
                return;
            };
            let data = if label_ptr.is_null() {
                label_ptr
            } else {
                unsafe { label_ptr.add(1) }
            };
            unsafe { $selene_array(data, label_len, arr.data, len) };
        }

        /// Record a scalar using Selene's plain string data pointer.
        ///
        /// # Safety
        /// `label_ptr` must reference at least `label_len` bytes of string data.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $selene_scalar(label_ptr: *const u8, label_len: i64, value: $ty) {
            unsafe {
                record_named_output(
                    label_ptr,
                    label_len,
                    crate::NamedResult::$variant(vec![value]),
                    concat!("USER:", $prefix, ":"),
                    concat!("USER:", $prefix, "ARR:"),
                    stringify!($selene_scalar),
                    true,
                )
            };
        }

        /// Record an array using Selene's plain string and array pointers.
        ///
        /// # Safety
        /// `label_ptr` must reference `label_len` bytes. `arr_ptr` must reference
        /// `arr_len` initialized elements, or may be null for an empty array.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $selene_array(
            label_ptr: *const u8,
            label_len: i64,
            arr_ptr: *const $ty,
            arr_len: u64,
        ) {
            match unsafe { result_array(arr_ptr, arr_len) } {
                Ok(values) => {
                    unsafe {
                        record_named_output(
                            label_ptr,
                            label_len,
                            crate::NamedResult::$variant(values.to_vec()),
                            concat!("USER:", $prefix, ":"),
                            concat!("USER:", $prefix, "ARR:"),
                            stringify!($selene_array),
                            false,
                        )
                    };
                }
                Err(detail) => unsafe { fatal_ffi_input(stringify!($selene_array), detail) },
            }
        }
    };
}

named_result_exports!(
    print_bool,
    print_bool_arr,
    print_bool_selene,
    print_bool_arr_selene,
    bool,
    Dense1DArrayBool,
    Bool,
    "BOOL"
);
named_result_exports!(
    print_int,
    print_int_arr,
    print_int_selene,
    print_int_arr_selene,
    i64,
    Dense1DArrayInt,
    I64,
    "INT"
);
named_result_exports!(
    print_uint,
    print_uint_arr,
    print_uint_selene,
    print_uint_arr_selene,
    u64,
    Dense1DArrayUint,
    U64,
    "UINT"
);
named_result_exports!(
    print_float,
    print_float_arr,
    print_float_selene,
    print_float_arr_selene,
    f64,
    Dense1DArrayFloat,
    F64,
    "FLOAT"
);

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
    if count == 0 {
        return;
    }
    if pairs_ptr.is_null() {
        unsafe {
            fatal_ffi_input(
                "pecos_qis_set_measurements",
                format!("pointer=null, count={count}"),
            );
        };
        return;
    }

    if count > (isize::MAX as usize) / std::mem::size_of::<(usize, bool)>()
        || !pairs_ptr.is_aligned()
    {
        unsafe {
            fatal_ffi_input(
                "pecos_qis_set_measurements",
                format!("pointer={pairs_ptr:?}, count={count} violates slice bounds or alignment"),
            );
        };
        return;
    }
    let pairs = unsafe { std::slice::from_raw_parts(pairs_ptr, count) };

    with_interface(|interface| {
        interface.set_measurement_results(pairs.iter().copied());
    });
}

// --- Heap Management Functions (Selene compatibility) ---

/// Allocate program-owned libc memory, reclaimed at shot end if still live.
///
/// # Safety
/// The execution context must be registered. Zero size returns null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn heap_alloc(size: u64) -> *mut u8 {
    if size == 0 {
        return std::ptr::null_mut();
    }
    match allocate_program_memory(size) {
        Ok(ptr) => ptr,
        Err(detail) => {
            unsafe { fatal_ffi_input("heap_alloc", detail) };
            std::ptr::null_mut()
        }
    }
}

fn allocate_program_memory(size: u64) -> Result<*mut u8, String> {
    let size_t = checked_slice_len::<u8>(size)?;
    let ctx = crate::get_execution_context()
        .ok_or_else(|| format!("size={size}: no execution context registered"))?;
    let ctx = unsafe { &*ctx };
    let mut allocations = ctx
        .program_allocations
        .lock()
        .map_err(|e| format!("size={size}: allocation lock error: {e}"))?;
    let ptr = unsafe { libc::malloc(size_t).cast::<u8>() };
    if ptr.is_null() {
        return Err(format!("size={size}: malloc returned null"));
    }
    allocations.insert(ptr as usize);
    crate::LIVE_PROGRAM_ALLOCATIONS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    crate::TOTAL_PROGRAM_ALLOCATIONS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    Ok(ptr)
}

/// Free a live program allocation and remove it from shot ownership.
///
/// # Safety
/// The pointer must be null or an allocation owned by the registered context.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn heap_free(ptr: *mut u8) {
    if ptr.is_null() {
        return;
    }
    if let Err(detail) = free_program_memory(ptr) {
        unsafe { fatal_ffi_input("heap_free", detail) };
    }
}

fn free_program_memory(ptr: *mut u8) -> Result<(), String> {
    let ctx = crate::get_execution_context()
        .ok_or_else(|| format!("pointer={ptr:?}: no execution context registered"))?;
    let ctx = unsafe { &*ctx };
    let removed = ctx
        .program_allocations
        .lock()
        .map_err(|e| format!("pointer={ptr:?}: allocation lock error: {e}"))?
        .remove(&(ptr as usize));
    if !removed {
        return Err(format!(
            "pointer={ptr:?} is not a live allocation owned by this context"
        ));
    }
    unsafe { ctx.free_program_allocation(ptr.cast()) };
    Ok(())
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
    fn test_rxy1q_gate() {
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

    #[test]
    fn test_trace_metadata_direct() {
        setup_test();
        let key = b"source_label";
        let value = b"szz_prefix:H:data_0";
        unsafe {
            pecos_qis_trace_metadata_direct(
                key.as_ptr(),
                i64::try_from(key.len()).unwrap(),
                value.as_ptr(),
                i64::try_from(value.len()).unwrap(),
            );
        }

        with_interface(|iface| {
            assert_eq!(iface.operations.len(), 1);
            let Operation::TraceMetadata { metadata, qubit } = &iface.operations[0] else {
                panic!("expected trace metadata operation");
            };
            assert_eq!(*qubit, None);
            assert_eq!(
                metadata.get("source_label").map(String::as_str),
                Some("szz_prefix:H:data_0")
            );
        });
    }

    #[test]
    fn test_trace_metadata_tket_string_layout() {
        setup_test();
        let key = [
            11_u8, b's', b'o', b'u', b'r', b'c', b'e', b'_', b'k', b'i', b'n', b'd',
        ];
        let value = [
            10_u8, b's', b'z', b'z', b'_', b'p', b'r', b'e', b'f', b'i', b'x',
        ];
        unsafe {
            pecos_qis_trace_metadata(key.as_ptr(), 11, value.as_ptr(), 10);
        }

        with_interface(|iface| {
            assert_eq!(iface.operations.len(), 1);
            let Operation::TraceMetadata { metadata, qubit } = &iface.operations[0] else {
                panic!("expected trace metadata operation");
            };
            assert_eq!(*qubit, None);
            assert_eq!(
                metadata.get("source_kind").map(String::as_str),
                Some("szz_prefix")
            );
        });
    }

    #[test]
    fn test_trace_metadata_hugr_string_layout() {
        setup_test();
        let key = [
            11_u8, b's', b'o', b'u', b'r', b'c', b'e', b'_', b'k', b'i', b'n', b'd',
        ];
        let value = [
            10_u8, b's', b'z', b'z', b'_', b'p', b'r', b'e', b'f', b'i', b'x',
        ];
        unsafe {
            pecos_qis_trace_metadata_hugr(key.as_ptr(), value.as_ptr());
        }

        with_interface(|iface| {
            assert_eq!(iface.operations.len(), 1);
            let Operation::TraceMetadata { metadata, qubit } = &iface.operations[0] else {
                panic!("expected trace metadata operation");
            };
            assert_eq!(*qubit, None);
            assert_eq!(
                metadata.get("source_kind").map(String::as_str),
                Some("szz_prefix")
            );
        });
    }

    #[test]
    fn test_trace_metadata_qubit_hugr_returns_qubit_and_queues_metadata() {
        setup_test();
        let key = [
            11_u8, b's', b'o', b'u', b'r', b'c', b'e', b'_', b'k', b'i', b'n', b'd',
        ];
        let value = [
            10_u8, b's', b'z', b'z', b'_', b'p', b'r', b'e', b'f', b'i', b'x',
        ];
        let returned =
            unsafe { pecos_qis_trace_metadata_qubit_hugr(17, key.as_ptr(), value.as_ptr()) };
        assert_eq!(returned, 17);

        with_interface(|iface| {
            assert_eq!(iface.operations.len(), 1);
            let Operation::TraceMetadata { metadata, qubit } = &iface.operations[0] else {
                panic!("expected trace metadata operation");
            };
            assert_eq!(*qubit, Some(17));
            assert_eq!(
                metadata.get("source_kind").map(String::as_str),
                Some("szz_prefix")
            );
        });
    }

    #[test]
    fn test_trace_metadata_qubit_hugr_expands_packed_json_metadata() {
        setup_test();
        let mut key = Vec::with_capacity(PACKED_TRACE_METADATA_JSON_KEY.len() + 1);
        key.push(u8::try_from(PACKED_TRACE_METADATA_JSON_KEY.len()).unwrap());
        key.extend_from_slice(PACKED_TRACE_METADATA_JSON_KEY.as_bytes());
        let value = br#"{"host_id":"probe:host","source_kind":"szz_host"}"#;
        let mut packed = Vec::with_capacity(value.len() + 1);
        packed.push(u8::try_from(value.len()).unwrap());
        packed.extend_from_slice(value);

        let returned =
            unsafe { pecos_qis_trace_metadata_qubit_hugr(19, key.as_ptr(), packed.as_ptr()) };
        assert_eq!(returned, 19);

        with_interface(|iface| {
            assert_eq!(iface.operations.len(), 1);
            let Operation::TraceMetadata { metadata, qubit } = &iface.operations[0] else {
                panic!("expected trace metadata operation");
            };
            assert_eq!(*qubit, Some(19));
            assert_eq!(
                metadata.get("source_kind").map(String::as_str),
                Some("szz_host")
            );
            assert_eq!(
                metadata.get("host_id").map(String::as_str),
                Some("probe:host")
            );
            assert!(!metadata.contains_key(PACKED_TRACE_METADATA_JSON_KEY));
        });
    }

    #[test]
    fn test_runtime_barrier_qubit_hugr_returns_qubit_and_queues_barrier() {
        setup_test();
        let returned = unsafe { pecos_qis_runtime_barrier_qubit_hugr(17) };
        assert_eq!(returned, 17);

        with_interface(|iface| {
            assert_eq!(iface.operations, vec![Operation::Barrier]);
        });
    }

    #[test]
    fn test_runtime_barrier_qubits2_hugr_returns_qubits_and_queues_barrier() {
        setup_test();
        let returned = unsafe { pecos_qis_runtime_barrier_qubits2_hugr(17, 23) };
        assert_eq!(
            returned,
            QubitPair {
                first: 17,
                second: 23,
            },
        );

        with_interface(|iface| {
            assert_eq!(iface.operations, vec![Operation::Barrier]);
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
    fn test_lazy_measure_leaked_uses_an_allocated_measurement_result() {
        setup_test();
        let result_id = unsafe { ___lazy_measure_leaked(0) };

        assert_eq!(result_id, 0);
        with_interface(|iface| {
            assert_eq!(iface.operations.len(), 2);
            assert_eq!(iface.operations[0], Operation::AllocateResult { id: 0 });
            assert_eq!(
                iface.operations[1],
                Operation::Quantum(QuantumOp::MeasureLeaked(0, 0))
            );
        });
    }

    #[test]
    fn test_read_future_uint_preserves_the_ideal_measurement_value() {
        setup_test();
        with_interface(|iface| iface.store_result(4, true));

        assert_eq!(unsafe { ___read_future_uint(4) }, 1);
    }

    #[test]
    fn test_read_future_uint_preserves_leakage_outcome() {
        setup_test();
        let ctx = crate::pecos_create_execution_context();
        let context = unsafe { &*ctx };
        context
            .dynamic_mode_active
            .store(true, std::sync::atomic::Ordering::SeqCst);
        unsafe { crate::pecos_register_execution_context(ctx) };
        crate::pecos_set_measurement_outcome(4, 2);

        assert_eq!(unsafe { ___read_future_uint(4) }, 2);

        unsafe {
            crate::pecos_register_execution_context(std::ptr::null_mut());
            crate::pecos_destroy_execution_context(ctx);
        }
    }

    #[test]
    fn test_print_int_records_integer_detector_literals() {
        let ctx = crate::pecos_create_execution_context();
        unsafe { crate::pecos_register_execution_context(ctx) };
        let label = b"\x08DETECTOR";

        unsafe { print_int(label.as_ptr(), 8, 0) };
        assert_eq!(
            unsafe { &*ctx }.get_named_results()["DETECTOR"],
            crate::NamedResult::I64(vec![0])
        );

        unsafe {
            crate::pecos_register_execution_context(std::ptr::null_mut());
            crate::pecos_destroy_execution_context(ctx);
        }
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
    fn test_named_result_trace_consumes_recorded_result_reads() {
        let ctx = crate::ExecutionContext::new();

        ctx.record_result_read(7);
        ctx.store_named_bool("m", true);
        ctx.record_result_read(8);
        ctx.record_result_read(9);
        ctx.store_named_array("arr", &[false, true]);

        let traces = ctx.get_named_result_traces();
        assert_eq!(traces.len(), 2);
        assert_eq!(traces[0].name, "m");
        assert_eq!(traces[0].values, vec![true]);
        assert_eq!(traces[0].result_ids, vec![7]);
        assert_eq!(traces[1].name, "arr");
        assert_eq!(traces[1].values, vec![false, true]);
        assert_eq!(traces[1].result_ids, vec![8, 9]);
    }

    #[test]
    fn test_named_result_trace_rejects_ambiguous_recorded_reads() {
        let ctx = crate::ExecutionContext::new();

        ctx.record_result_read(7);
        ctx.record_result_read(8);
        ctx.store_named_bool("computed", true);

        let traces = ctx.get_named_result_traces();
        assert_eq!(traces.len(), 1);
        assert!(traces[0].result_ids.is_empty());

        ctx.record_result_read(9);
        ctx.store_named_bool("next", false);
        let traces = ctx.get_named_result_traces();
        assert_eq!(traces[1].result_ids, vec![9]);
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
        let context = crate::pecos_create_execution_context();
        unsafe { crate::pecos_register_execution_context(context) };
        let ptr = std::ptr::NonNull::new(unsafe { heap_alloc(100) })
            .expect("heap_alloc(100) should return non-null");

        // SAFETY: `ptr` owns this allocation exclusively for the test scope;
        // `heap_alloc` returns a properly-aligned `u8` allocation.
        unsafe {
            std::ptr::write(ptr.as_ptr(), 42u8);
            assert_eq!(std::ptr::read(ptr.as_ptr()), 42u8);
            heap_free(ptr.as_ptr());
            crate::pecos_destroy_execution_context(context);
        }
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
