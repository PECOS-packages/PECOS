//! Integration test for `ForeignDecoder`.
//!
//! Implements a trivial "XOR decoder" as C-ABI functions in Rust, wraps it
//! into a `ForeignDecoder`, and verifies it works through the Decoder trait.

use ndarray::array;
use pecos_decoder_core::Decoder;
use pecos_foreign::{ForeignDecoder, ForeignDecoderVTable, ForeignDecodingResultRaw};

// -- "Foreign" decoder state (what Go/Julia/C would hold) --

struct XorDecoderState {
    checks: usize,
    bits: usize,
}

// -- C-ABI callback implementations --

unsafe extern "C" fn xor_decode(
    handle: *mut (),
    input_ptr: *const u8,
    input_len: usize,
    result_out: *mut ForeignDecodingResultRaw,
) -> i32 {
    let state = unsafe { &*(handle.cast::<XorDecoderState>()) };
    let input = unsafe { std::slice::from_raw_parts(input_ptr, input_len) };

    if input_len != state.checks {
        // Return error
        let msg = format!("expected {} bytes, got {}", state.checks, input_len);
        let boxed = msg.into_bytes().into_boxed_slice();
        let out = unsafe { &mut *result_out };
        out.error_len = boxed.len();
        out.error_ptr = Box::into_raw(boxed).cast::<u8>();
        out.observable_ptr = std::ptr::null_mut();
        out.observable_len = 0;
        return -1;
    }

    // XOR all syndrome bytes into observable[0]
    let mut observable = vec![0u8; state.bits];
    let xor: u8 = input.iter().fold(0, |acc, &b| acc ^ b);
    if !observable.is_empty() {
        observable[0] = xor;
    }

    let boxed = observable.into_boxed_slice();
    let out = unsafe { &mut *result_out };
    out.observable_len = boxed.len();
    out.observable_ptr = if boxed.is_empty() {
        std::ptr::null_mut()
    } else {
        Box::into_raw(boxed).cast::<u8>()
    };
    out.weight = 1.0;
    out.converged = 1;
    out.error_ptr = std::ptr::null();
    out.error_len = 0;
    0
}

unsafe extern "C" fn xor_check_count(handle: *const ()) -> usize {
    let state = unsafe { &*(handle.cast::<XorDecoderState>()) };
    state.checks
}

unsafe extern "C" fn xor_bit_count(handle: *const ()) -> usize {
    let state = unsafe { &*(handle.cast::<XorDecoderState>()) };
    state.bits
}

unsafe extern "C" fn xor_free_result(ptr: *mut u8, len: usize) {
    if !ptr.is_null() && len > 0 {
        unsafe {
            let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, len));
        }
    }
}

unsafe extern "C" fn xor_free_error(ptr: *const u8, len: usize) {
    if !ptr.is_null() && len > 0 {
        unsafe {
            let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr.cast_mut(), len));
        }
    }
}

unsafe extern "C" fn xor_destroy(handle: *mut ()) {
    if !handle.is_null() {
        unsafe {
            let _ = Box::from_raw(handle.cast::<XorDecoderState>());
        }
    }
}

fn make_xor_decoder(checks: usize, bits: usize) -> ForeignDecoder {
    let state = Box::new(XorDecoderState { checks, bits });
    let handle = Box::into_raw(state).cast::<()>();

    let vtable = ForeignDecoderVTable {
        version: pecos_foreign::version::DECODER_VTABLE_VERSION,
        decode: xor_decode,
        check_count: xor_check_count,
        bit_count: xor_bit_count,
        free_result: xor_free_result,
        free_error: xor_free_error,
        destroy: xor_destroy,
    };

    unsafe { ForeignDecoder::new(handle, vtable) }.expect("vtable version should match")
}

#[test]
fn test_foreign_decoder_basic() {
    let mut decoder = make_xor_decoder(4, 2);

    assert_eq!(decoder.check_count(), 4);
    assert_eq!(decoder.bit_count(), 2);

    let syndrome = array![0u8, 1, 0, 1];
    let result = decoder.decode(&syndrome.view()).unwrap();

    // XOR of [0, 1, 0, 1] = 0
    assert_eq!(result.observable, vec![0, 0]);
    assert!((result.weight - 1.0).abs() < f64::EPSILON);
    assert_eq!(result.converged, Some(true));
}

#[test]
fn test_foreign_decoder_nonzero_result() {
    let mut decoder = make_xor_decoder(3, 1);

    let syndrome = array![1u8, 0, 0];
    let result = decoder.decode(&syndrome.view()).unwrap();

    // XOR of [1, 0, 0] = 1
    assert_eq!(result.observable, vec![1]);
}

#[test]
fn test_foreign_decoder_error() {
    let mut decoder = make_xor_decoder(4, 2);

    // Wrong input length
    let syndrome = array![0u8, 1];
    let err = decoder.decode(&syndrome.view()).unwrap_err();

    assert!(err.0.contains("expected 4 bytes, got 2"), "got: {}", err.0);
}

#[test]
fn test_foreign_decoder_trait_object() {
    // Verify ForeignDecoder works as a trait object -- this is the whole point
    let decoder = make_xor_decoder(3, 1);
    let mut boxed: Box<dyn Decoder<Result = _, Error = _>> = Box::new(decoder);

    let syndrome = array![1u8, 1, 1];
    let result = boxed.decode(&syndrome.view()).unwrap();

    // XOR of [1, 1, 1] = 1
    assert_eq!(result.observable, vec![1]);
}

#[test]
fn test_foreign_decoder_version_mismatch() {
    let state = Box::new(XorDecoderState { checks: 1, bits: 1 });
    let handle = Box::into_raw(state).cast::<()>();

    let vtable = ForeignDecoderVTable {
        version: 9999, // wrong version
        decode: xor_decode,
        check_count: xor_check_count,
        bit_count: xor_bit_count,
        free_result: xor_free_result,
        free_error: xor_free_error,
        destroy: xor_destroy,
    };

    // Should return None on version mismatch
    let result = unsafe { ForeignDecoder::new(handle, vtable) };
    assert!(result.is_none(), "wrong version should return None");

    // Clean up the leaked state since ForeignDecoder didn't take ownership
    unsafe {
        let _ = Box::from_raw(handle.cast::<XorDecoderState>());
    }
}
