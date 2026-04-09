//! Foreign decoder plugin interface.
//!
//! A foreign language implements a decoder by providing:
//! - An opaque handle (`*mut ()`) to its decoder instance
//! - A vtable of C-ABI function pointers ([`ForeignDecoderVTable`])
//!
//! The Rust [`ForeignDecoder`] wraps these into a [`pecos_decoder_core::Decoder`]
//! implementation that PECOS can use identically to any native Rust decoder.

use ndarray::ArrayView1;
use pecos_decoder_core::{Decoder, DecodingResultTrait};
use std::fmt;

/// Result returned by a foreign decoder over the C ABI.
///
/// The foreign `decode` function writes into this struct.
/// The caller (Rust) owns the `observable_ptr` memory after the call
/// and must free it via the vtable's `free_result` function.
#[repr(C)]
pub struct ForeignDecodingResultRaw {
    /// Pointer to the observable bytes (owned by foreign code until handed over).
    pub observable_ptr: *mut u8,
    /// Length of the observable array.
    pub observable_len: usize,
    /// Weight/cost of the decoding solution.
    pub weight: f64,
    /// Whether the decoder converged (0 = false, 1 = true, -1 = unknown).
    pub converged: i8,
    /// Error message pointer (null if no error). UTF-8 bytes of `error_len` length
    /// (not necessarily null-terminated). Owned by foreign code -- Rust copies it then calls `free_error`.
    pub error_ptr: *const u8,
    /// Length of error message (excluding null terminator).
    pub error_len: usize,
}

/// The vtable that foreign code must populate.
///
/// All function pointers use the C calling convention and take the opaque
/// decoder handle as their first argument.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ForeignDecoderVTable {
    /// ABI version. Must equal [`crate::version::DECODER_VTABLE_VERSION`].
    /// Checked on construction; mismatches are rejected with a clear error.
    pub version: u32,

    /// Decode a syndrome.
    ///
    /// # Arguments
    /// - `handle`: the opaque decoder pointer
    /// - `input_ptr`: pointer to syndrome bytes
    /// - `input_len`: number of syndrome bytes
    /// - `result_out`: pointer to a `ForeignDecodingResultRaw` that the callee fills in
    ///
    /// Returns 0 on success, non-zero on error (with `result_out.error_ptr` set).
    pub decode: unsafe extern "C" fn(
        handle: *mut (),
        input_ptr: *const u8,
        input_len: usize,
        result_out: *mut ForeignDecodingResultRaw,
    ) -> i32,

    /// Return the number of checks (rows in parity check matrix).
    pub check_count: unsafe extern "C" fn(handle: *const ()) -> usize,

    /// Return the number of bits (columns in parity check matrix).
    pub bit_count: unsafe extern "C" fn(handle: *const ()) -> usize,

    /// Free the observable array from a decoding result.
    ///
    /// Called by Rust after copying the data out of `ForeignDecodingResultRaw`.
    /// If `ptr` is null, this must be a no-op.
    pub free_result: unsafe extern "C" fn(ptr: *mut u8, len: usize),

    /// Free an error message string.
    ///
    /// Called by Rust after copying the error. If `ptr` is null, this must be a no-op.
    pub free_error: unsafe extern "C" fn(ptr: *const u8, len: usize),

    /// Destroy the decoder. Called once when `ForeignDecoder` is dropped.
    ///
    /// If `handle` is null, this must be a no-op.
    pub destroy: unsafe extern "C" fn(handle: *mut ()),
}

// SAFETY: The foreign decoder handle is opaque and accessed only through the vtable
// function pointers. We require that the foreign implementation is thread-safe
// (documented in the C header). This is the same contract as `Send` for any
// FFI wrapper (see cuQuantum, QuEST wrappers in this repo).
unsafe impl Send for ForeignDecoder {}

/// A decoder implemented in a foreign language via C ABI function pointers.
///
/// This wraps an opaque foreign decoder handle + vtable into a Rust type
/// that implements [`Decoder`].
pub struct ForeignDecoder {
    handle: *mut (),
    vtable: ForeignDecoderVTable,
}

impl ForeignDecoder {
    /// Create a new `ForeignDecoder` from an opaque handle and vtable.
    ///
    /// Returns `None` if the vtable version does not match the expected ABI version.
    ///
    /// # Safety
    ///
    /// The caller must guarantee:
    /// - `handle` is a valid pointer to a foreign decoder instance
    /// - All function pointers in `vtable` are valid and follow the documented contracts
    /// - The foreign decoder lives until `destroy` is called
    /// - The foreign decoder is safe to call from any thread (Send)
    pub unsafe fn new(handle: *mut (), vtable: ForeignDecoderVTable) -> Option<Self> {
        if vtable.version != crate::version::DECODER_VTABLE_VERSION {
            log::error!(
                "Foreign decoder ABI version mismatch: plugin has v{}, PECOS expects v{}",
                vtable.version,
                crate::version::DECODER_VTABLE_VERSION,
            );
            return None;
        }
        Some(Self { handle, vtable })
    }
}

impl Drop for ForeignDecoder {
    fn drop(&mut self) {
        // SAFETY: We own the handle and destroy is called exactly once.
        unsafe {
            (self.vtable.destroy)(self.handle);
        }
    }
}

/// Error from a foreign decoder.
#[derive(Debug)]
pub struct ForeignDecoderError(pub String);

impl fmt::Display for ForeignDecoderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "foreign decoder error: {}", self.0)
    }
}

impl std::error::Error for ForeignDecoderError {}

/// Decoded result from a foreign decoder, converted to Rust-owned data.
#[derive(Debug, Clone)]
pub struct ForeignDecodingResult {
    pub observable: Vec<u8>,
    pub weight: f64,
    pub converged: Option<bool>,
}

impl DecodingResultTrait for ForeignDecodingResult {
    fn is_successful(&self) -> bool {
        self.converged.unwrap_or(true)
    }

    fn cost(&self) -> Option<f64> {
        Some(self.weight)
    }
}

impl Decoder for ForeignDecoder {
    type Result = ForeignDecodingResult;
    type Error = ForeignDecoderError;

    fn decode(&mut self, input: &ArrayView1<u8>) -> Result<Self::Result, Self::Error> {
        let input_slice = input
            .as_slice()
            .expect("input ArrayView1 should be contiguous");

        let mut raw = ForeignDecodingResultRaw {
            observable_ptr: std::ptr::null_mut(),
            observable_len: 0,
            weight: 0.0,
            converged: -1,
            error_ptr: std::ptr::null(),
            error_len: 0,
        };

        // SAFETY: We pass valid pointers; the foreign code fills `raw`.
        let rc = unsafe {
            (self.vtable.decode)(
                self.handle,
                input_slice.as_ptr(),
                input_slice.len(),
                &raw mut raw,
            )
        };

        if rc != 0 {
            let err_msg = if raw.error_ptr.is_null() || raw.error_len == 0 {
                "unknown error".to_string()
            } else {
                // SAFETY: Foreign code guarantees error_ptr is valid UTF-8 for error_len bytes.
                let msg = unsafe {
                    let slice = std::slice::from_raw_parts(raw.error_ptr, raw.error_len);
                    String::from_utf8_lossy(slice).into_owned()
                };
                // Free the foreign error string.
                unsafe { (self.vtable.free_error)(raw.error_ptr, raw.error_len) };
                msg
            };
            return Err(ForeignDecoderError(err_msg));
        }

        // Copy observable data into Rust-owned Vec, then free the foreign allocation.
        let observable = if raw.observable_ptr.is_null() || raw.observable_len == 0 {
            vec![]
        } else {
            // SAFETY: Foreign code guarantees observable_ptr is valid for observable_len bytes.
            let data = unsafe {
                std::slice::from_raw_parts(raw.observable_ptr, raw.observable_len).to_vec()
            };
            unsafe { (self.vtable.free_result)(raw.observable_ptr, raw.observable_len) };
            data
        };

        let converged = match raw.converged {
            0 => Some(false),
            1 => Some(true),
            _ => None,
        };

        Ok(ForeignDecodingResult {
            observable,
            weight: raw.weight,
            converged,
        })
    }

    fn check_count(&self) -> usize {
        // SAFETY: handle is valid, check_count is a pure query.
        unsafe { (self.vtable.check_count)(self.handle) }
    }

    fn bit_count(&self) -> usize {
        // SAFETY: handle is valid, bit_count is a pure query.
        unsafe { (self.vtable.bit_count)(self.handle) }
    }
}
