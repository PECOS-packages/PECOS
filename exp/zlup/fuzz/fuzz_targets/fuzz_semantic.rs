//! Fuzz target for the Zlup semantic analyzer.
//!
//! This target fuzzes the semantic analyzer with arbitrary programs to find:
//! - Panics or crashes during type checking
//! - Infinite loops in type inference
//! - Memory issues in symbol table operations
//!
//! Run with:
//! ```bash
//! cargo +nightly fuzz run fuzz_semantic
//! ```

#![no_main]

use libfuzzer_sys::fuzz_target;
use zlup::semantic::SemanticAnalyzer;

fuzz_target!(|data: &[u8]| {
    // Convert bytes to string, skipping invalid UTF-8
    if let Ok(source) = std::str::from_utf8(data) {
        // First try to parse - skip if parsing fails
        if let Ok(program) = zlup::parse(source) {
            // The semantic analyzer should never panic on any valid AST
            let mut analyzer = SemanticAnalyzer::new();
            let _ = analyzer.analyze(&program);
        }
    }
});
