// Copyright 2025 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
// the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

/*!
C-compatible FFI exports for PECOS Go bindings.

This crate provides C-compatible functions that can be called from Go via cgo.
*/

use pecos::QubitId;
use std::ffi::CString;
use std::os::raw::c_char;

/// Get the PECOS version information
///
/// # Panics
///
/// This function will panic if the version string contains a null byte.
#[unsafe(no_mangle)]
pub extern "C" fn pecos_version() -> *const c_char {
    let version = CString::new("PECOS v0.1.0 (Go bindings)")
        .expect("Version string should not contain null bytes");
    version.into_raw()
}

/// Create a `QubitId` and return its index
#[unsafe(no_mangle)]
pub extern "C" fn create_qubit_id(index: i64) -> i64 {
    if index < 0 {
        return -1;
    }

    // Safe conversion: we've already checked that index >= 0
    match usize::try_from(index) {
        Ok(idx) => {
            let qubit_id = QubitId::new(idx);
            // This cast is safe because QubitId indices fit in i64
            i64::try_from(qubit_id.index()).unwrap_or(i64::MAX)
        }
        Err(_) => -1, // Index too large for usize
    }
}

/// Convert a qubit index to its string representation
///
/// # Panics
///
/// This function will panic if the resulting string contains a null byte.
#[unsafe(no_mangle)]
pub extern "C" fn qubit_id_to_string(index: i64) -> *const c_char {
    let result = if index < 0 {
        CString::new("Invalid qubit index").expect("Error string should not contain null bytes")
    } else {
        match usize::try_from(index) {
            Ok(idx) => {
                let qubit_id = QubitId::new(idx);
                CString::new(format!("QubitId({qubit_id})"))
                    .expect("QubitId string should not contain null bytes")
            }
            Err(_) => CString::new("Invalid qubit index")
                .expect("Error string should not contain null bytes"),
        }
    };

    result.into_raw()
}

/// Simple addition function to test FFI
#[unsafe(no_mangle)]
pub extern "C" fn add_two_numbers(a: i64, b: i64) -> i64 {
    a + b
}

/// Free a string allocated by Rust (important for memory management)
///
/// # Safety
///
/// This function is unsafe because:
/// - The caller must ensure `s` is a valid pointer allocated by `CString::into_raw()`
/// - The pointer must not be used after calling this function
/// - The pointer must not be freed more than once
#[unsafe(no_mangle)]
pub unsafe extern "C" fn free_rust_string(s: *mut c_char) {
    if s.is_null() {
        return;
    }
    // SAFETY: The caller guarantees that `s` is a valid pointer from CString::into_raw()
    unsafe {
        // Reconstruct the CString and drop it
        let _ = CString::from_raw(s);
    }
}
