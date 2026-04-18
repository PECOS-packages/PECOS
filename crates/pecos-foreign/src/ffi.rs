//! C-ABI entry points for the universal shared library.
//!
//! These functions are the complete public API for foreign languages.
//! Any language that can call C functions can use PECOS by linking against
//! `libpecos_ffi.so` and including `pecos_foreign.h`.
//!
//! ## Decoder plugin functions
//! - [`pecos_foreign_decoder_create`] -- wrap a foreign decoder vtable
//! - [`pecos_foreign_decoder_decode`] -- decode a syndrome
//! - [`pecos_foreign_decoder_check_count`] / [`pecos_foreign_decoder_bit_count`]
//! - [`pecos_foreign_decoder_free`] -- destroy
//!
//! ## Simulator plugin functions
//! - [`pecos_foreign_simulator_create`] -- wrap a foreign simulator vtable
//! - [`pecos_foreign_simulator_supports_rotations`] -- query capability
//! - [`pecos_foreign_simulator_free`] -- destroy
//!
//! ## Engine / circuit functions are in [`crate::engine`].
//! ## Version constants are in [`crate::version`].

use crate::decoder::{ForeignDecoder, ForeignDecoderVTable, ForeignDecodingResultRaw};
use crate::simulator::{ForeignSimulator, ForeignSimulatorVTable};

// ============================================================================
// Decoder bridge
// ============================================================================

/// Create a `ForeignDecoder` from an opaque handle and vtable.
///
/// Returns an opaque pointer, or null if the version check fails.
/// Caller must eventually call `pecos_foreign_decoder_free`.
///
/// # Safety
///
/// - `handle` must be a valid pointer to a foreign decoder instance
/// - `vtable` must point to a valid `ForeignDecoderVTable` (same as `PecosDecoderVTable` in C header)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pecos_foreign_decoder_create(
    handle: *mut (),
    vtable: *const ForeignDecoderVTable,
) -> *mut ForeignDecoder {
    let vtable_copy = unsafe { *vtable };
    let Some(decoder) = (unsafe { ForeignDecoder::new(handle, vtable_copy) }) else {
        return std::ptr::null_mut();
    };
    Box::into_raw(Box::new(decoder))
}

/// Get the check count from a foreign decoder.
///
/// # Safety
/// `decoder` must be a valid pointer from `pecos_foreign_decoder_create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pecos_foreign_decoder_check_count(
    decoder: *const ForeignDecoder,
) -> usize {
    use crate::pecos_decoder_core::Decoder;
    let d = unsafe { &*decoder };
    d.check_count()
}

/// Get the bit count from a foreign decoder.
///
/// # Safety
/// `decoder` must be a valid pointer from `pecos_foreign_decoder_create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pecos_foreign_decoder_bit_count(decoder: *const ForeignDecoder) -> usize {
    use crate::pecos_decoder_core::Decoder;
    let d = unsafe { &*decoder };
    d.bit_count()
}

/// Decode a syndrome using a foreign decoder.
///
/// Writes into `result_out`. Returns 0 on success, non-zero on error.
///
/// # Safety
/// All pointers must be valid. `input_ptr` must point to `input_len` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pecos_foreign_decoder_decode(
    decoder: *mut ForeignDecoder,
    input_ptr: *const u8,
    input_len: usize,
    result_out: *mut ForeignDecodingResultRaw,
) -> i32 {
    use crate::pecos_decoder_core::Decoder;
    use ndarray::ArrayView1;

    let d = unsafe { &mut *decoder };
    let input_slice = unsafe { std::slice::from_raw_parts(input_ptr, input_len) };
    let view = ArrayView1::from(input_slice);

    match d.decode(&view) {
        Ok(result) => {
            let out = unsafe { &mut *result_out };
            let mut obs = result.observable.into_boxed_slice();
            out.observable_len = obs.len();
            out.observable_ptr = if obs.is_empty() {
                std::ptr::null_mut()
            } else {
                let ptr = obs.as_mut_ptr();
                std::mem::forget(obs);
                ptr
            };
            out.weight = result.weight;
            out.converged = match result.converged {
                Some(true) => 1,
                Some(false) => 0,
                None => -1,
            };
            out.error_ptr = std::ptr::null();
            out.error_len = 0;
            0
        }
        Err(e) => {
            let out = unsafe { &mut *result_out };
            let msg = e.0.into_bytes().into_boxed_slice();
            out.error_len = msg.len();
            out.error_ptr = if msg.is_empty() {
                std::ptr::null()
            } else {
                let ptr = msg.as_ptr();
                std::mem::forget(msg);
                ptr
            };
            out.observable_ptr = std::ptr::null_mut();
            out.observable_len = 0;
            -1
        }
    }
}

/// Free observable bytes from `pecos_foreign_decoder_decode`.
///
/// # Safety
/// `ptr` must be from a decode result, or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pecos_foreign_decoder_free_observable(ptr: *mut u8, len: usize) {
    if !ptr.is_null() && len > 0 {
        unsafe {
            let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, len));
        }
    }
}

/// Free error string from `pecos_foreign_decoder_decode`.
///
/// # Safety
/// `ptr` must be from a decode error, or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pecos_foreign_decoder_free_error(ptr: *const u8, len: usize) {
    if !ptr.is_null() && len > 0 {
        unsafe {
            let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr.cast_mut(), len));
        }
    }
}

/// Destroy a foreign decoder.
///
/// # Safety
/// `decoder` must be from `pecos_foreign_decoder_create`. Call at most once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pecos_foreign_decoder_free(decoder: *mut ForeignDecoder) {
    if !decoder.is_null() {
        unsafe {
            let _ = Box::from_raw(decoder);
        }
    }
}

// ============================================================================
// Simulator bridge
// ============================================================================

/// Create a `ForeignSimulator` from an opaque handle and vtable.
///
/// Returns an opaque pointer. Caller must eventually call `pecos_foreign_simulator_free`.
///
/// # Safety
/// `handle` and `vtable` must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pecos_foreign_simulator_create(
    handle: *mut (),
    vtable: *const ForeignSimulatorVTable,
    num_qubits: usize,
) -> *mut ForeignSimulator {
    let vtable_copy = unsafe { *vtable };
    let Some(sim) = (unsafe { ForeignSimulator::new(handle, vtable_copy, num_qubits) }) else {
        return std::ptr::null_mut();
    };
    Box::into_raw(Box::new(sim))
}

/// Whether a foreign simulator supports rotation gates.
///
/// # Safety
/// `sim` must be from `pecos_foreign_simulator_create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pecos_foreign_simulator_supports_rotations(
    sim: *const ForeignSimulator,
) -> bool {
    let s = unsafe { &*sim };
    s.supports_rotations()
}

/// Destroy a foreign simulator.
///
/// # Safety
/// `sim` must be from `pecos_foreign_simulator_create`. Call at most once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pecos_foreign_simulator_free(sim: *mut ForeignSimulator) {
    if !sim.is_null() {
        unsafe {
            let _ = Box::from_raw(sim);
        }
    }
}

// ============================================================================
// Version query
// ============================================================================

/// Get the expected decoder vtable ABI version.
#[unsafe(no_mangle)]
pub extern "C" fn pecos_decoder_vtable_version() -> u32 {
    crate::version::DECODER_VTABLE_VERSION
}

/// Get the expected simulator vtable ABI version.
#[unsafe(no_mangle)]
pub extern "C" fn pecos_simulator_vtable_version() -> u32 {
    crate::version::SIMULATOR_VTABLE_VERSION
}
