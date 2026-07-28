//! Fuzz target for the Zlup parser.
//!
//! This target fuzzes the parser with arbitrary byte sequences to find:
//! - Panics or crashes
//! - Infinite loops (detected via timeout)
//! - Memory issues
//!
//! Run with:
//! ```bash
//! cargo +nightly fuzz run fuzz_parser
//! ```

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Convert bytes to string, skipping invalid UTF-8
    if let Ok(source) = std::str::from_utf8(data) {
        // The parser should never panic on any input
        let _ = zlup::parse(source);
    }
});
