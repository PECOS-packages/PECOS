//! C-ABI bridge for foreign decoders.
//!
//! Exposes functions that Go (or any C-ABI language) calls to register a decoder
//! with PECOS and then use it.

use pecos_foreign::{ForeignDecoder, ForeignDecoderVTable, ForeignDecodingResultRaw};

/// C-compatible vtable passed from Go. Must match the Go `PecosDecoderVTable` struct layout.
#[repr(C)]
pub struct CDecoderVTable {
    pub version: u32,
    pub decode: unsafe extern "C" fn(
        handle: *mut (),
        input_ptr: *const u8,
        input_len: usize,
        result_out: *mut ForeignDecodingResultRaw,
    ) -> i32,
    pub check_count: unsafe extern "C" fn(handle: *const ()) -> usize,
    pub bit_count: unsafe extern "C" fn(handle: *const ()) -> usize,
    pub free_result: unsafe extern "C" fn(ptr: *mut u8, len: usize),
    pub free_error: unsafe extern "C" fn(ptr: *const u8, len: usize),
    pub destroy: unsafe extern "C" fn(handle: *mut ()),
}

/// Create a `ForeignDecoder` from a Go-provided handle and vtable.
///
/// Returns an opaque pointer to a boxed `ForeignDecoder`. The caller (Go) can
/// later pass this to `pecos_foreign_decoder_decode` etc. to use it, and must
/// call `pecos_foreign_decoder_free` to destroy it.
///
/// # Safety
///
/// - `handle` must be a valid decoder handle from Go's registry
/// - `vtable` must point to a valid, fully-populated `CDecoderVTable`
/// - All function pointers in the vtable must remain valid until `destroy` is called
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pecos_foreign_decoder_create(
    handle: *mut (),
    vtable: *const CDecoderVTable,
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
///
/// `decoder` must be a valid pointer from `pecos_foreign_decoder_create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pecos_foreign_decoder_check_count(
    decoder: *const ForeignDecoder,
) -> usize {
    use pecos_foreign::pecos_decoder_core::Decoder;
    let d = unsafe { &*decoder };
    d.check_count()
}

/// Get the bit count from a foreign decoder.
///
/// # Safety
///
/// `decoder` must be a valid pointer from `pecos_foreign_decoder_create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pecos_foreign_decoder_bit_count(decoder: *const ForeignDecoder) -> usize {
    use pecos_foreign::pecos_decoder_core::Decoder;
    let d = unsafe { &*decoder };
    d.bit_count()
}

/// Decode a syndrome using a foreign decoder.
///
/// Writes the result into `result_out`. Returns 0 on success, non-zero on error.
///
/// # Safety
///
/// - `decoder` must be a valid pointer from `pecos_foreign_decoder_create`
/// - `input_ptr` must point to `input_len` valid bytes
/// - `result_out` must point to a valid `ForeignDecodingResultRaw`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pecos_foreign_decoder_decode(
    decoder: *mut ForeignDecoder,
    input_ptr: *const u8,
    input_len: usize,
    result_out: *mut ForeignDecodingResultRaw,
) -> i32 {
    use ndarray::ArrayView1;
    use pecos_foreign::pecos_decoder_core::Decoder;

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
            let msg = e.0;
            let c_msg = msg.into_bytes().into_boxed_slice();
            out.error_len = c_msg.len();
            out.error_ptr = if c_msg.is_empty() {
                std::ptr::null()
            } else {
                let ptr = c_msg.as_ptr();
                std::mem::forget(c_msg);
                ptr
            };
            out.observable_ptr = std::ptr::null_mut();
            out.observable_len = 0;
            -1
        }
    }
}

/// Free the observable bytes returned by `pecos_foreign_decoder_decode`.
///
/// # Safety
///
/// `ptr` must be a pointer previously returned in a `ForeignDecodingResultRaw`
/// from `pecos_foreign_decoder_decode`, or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pecos_foreign_decoder_free_observable(ptr: *mut u8, len: usize) {
    if !ptr.is_null() && len > 0 {
        // SAFETY: ptr was allocated by Box::into_raw on a boxed slice.
        unsafe {
            let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, len));
        }
    }
}

/// Free an error string returned by `pecos_foreign_decoder_decode`.
///
/// # Safety
///
/// `ptr` must be a pointer previously returned in a `ForeignDecodingResultRaw`
/// error field from `pecos_foreign_decoder_decode`, or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pecos_foreign_decoder_free_error(ptr: *const u8, len: usize) {
    if !ptr.is_null() && len > 0 {
        // SAFETY: ptr was allocated by Box::into_raw on a boxed slice.
        unsafe {
            let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr.cast_mut(), len));
        }
    }
}

/// Destroy a foreign decoder created by `pecos_foreign_decoder_create`.
///
/// This calls the vtable's `destroy` function and frees the Rust allocation.
///
/// # Safety
///
/// `decoder` must be a valid pointer from `pecos_foreign_decoder_create`.
/// Must not be called more than once for the same pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pecos_foreign_decoder_free(decoder: *mut ForeignDecoder) {
    if !decoder.is_null() {
        // SAFETY: We own this box, and Drop calls vtable.destroy.
        unsafe {
            let _ = Box::from_raw(decoder);
        }
    }
}
