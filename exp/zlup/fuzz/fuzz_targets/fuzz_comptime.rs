//! Fuzz target for the Zlup compile-time evaluator.
//!
//! This target fuzzes the comptime evaluator with arbitrary expressions to find:
//! - Panics during evaluation
//! - Overflow issues not caught by checked arithmetic
//! - Infinite recursion
//!
//! Run with:
//! ```bash
//! cargo +nightly fuzz run fuzz_comptime
//! ```

#![no_main]

use libfuzzer_sys::fuzz_target;
use zlup::comptime::ComptimeEvaluator;

fuzz_target!(|data: &[u8]| {
    // Convert bytes to string, skipping invalid UTF-8
    if let Ok(source) = std::str::from_utf8(data) {
        // Wrap in a simple expression context
        let wrapped = format!("x := {};", source);

        // Try to parse as a binding with an expression
        if let Ok(program) = zlup::parse(&wrapped) {
            // Try to evaluate any expressions in the program
            let mut evaluator = ComptimeEvaluator::new();

            // Extract expressions from bindings and try to evaluate them
            for decl in &program.declarations {
                if let zlup::ast::TopLevelDecl::Binding(binding) = decl {
                    if let Some(expr) = &binding.value {
                        // The evaluator should never panic
                        let _ = evaluator.eval_expr(expr);
                    }
                }
            }
        }
    }
});
