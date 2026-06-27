//! # Zluppy
//!
//! **EXPERIMENTAL** - A Zig/SLR/NASA Power of 10 reflection of Guppy's approach to
//! quantum programming.
//!
//! ## Philosophy
//!
//! Zluppy solves the same problems as Guppy but through a different lens:
//!
//! - **Zig's philosophy**: Explicit over implicit, simple over complex, compile-time
//!   metaprogramming instead of runtime magic
//! - **SLR's allocator model**: Hierarchical resource management - allocators own
//!   qubits, children borrow from parents, lifetimes are structural
//! - **NASA Power of 10**: Bounded loops, fixed resource limits, no dynamic allocation
//!   after initialization, assertions everywhere, predictable execution
//!
//! Where Guppy uses linear types, we use allocators. Where Guppy embeds in Python,
//! we stand alone. Same problems, simpler idioms, low-level but clean.
//!
//! ## Design Goals
//!
//! - Standalone language (not Python-embedded)
//! - Explicit resource management via allocators
//! - Compiles to SLR-AST for Python/PECOS integration
//! - Compiles to HUGR for hardware/experiment targeting
//! - Compiles to PHIR for simulator targeting
//! - Bounded, predictable execution (NASA Power of 10)
//! - Full comptime metaprogramming (Zig-style)
//!
//! ## Compilation Targets
//!
//! Both HUGR and PHIR are MLIR-inspired IRs:
//! - **HUGR**: Hierarchical Unified Graph Representation - for experiments/hardware
//! - **PHIR**: Program Hierarchical IR - for simulator targeting
//!
//! ```text
//! ┌─────────────┐
//! │ Zluppy      │
//! │ (.zlp)      │
//! └──────┬──────┘
//!        │
//!        ▼
//! ┌─────────────┐
//! │ Zluppy AST  │
//! └──────┬──────┘
//!        │
//!    ┌───┼───┐
//!    │   │   │
//!    ▼   ▼   ▼
//! ┌────┐ ┌────┐ ┌────┐
//! │SLR │ │HUGR│ │PHIR│
//! │AST │ │    │ │    │
//! └─┬──┘ └─┬──┘ └─┬──┘
//!   │      │      │
//!   ▼      ▼      ▼
//! ┌────┐ ┌────┐ ┌────┐
//! │Guppy│ │Exp │ │Sim │
//! │QASM│ │HW  │ │    │
//! └────┘ └────┘ └────┘
//! ```
//!
//! ## Example
//!
//! ```zluppy
//! const std = @import("std");
//!
//! pub fn main() -> unit {
//!     var base = qalloc(10);
//!     var q = base.child(2);
//!
//!     pz q;
//!
//!     // Bell state
//!     h(q[0]);
//!     cx(q[0], q[1]);
//!
//!     const results = measure(q);
//!     return unit;
//! }
//! ```
//!
//! ## Status
//!
//! This is an **experimental** language for research purposes. The API and syntax
//! are subject to change without notice.

// Experimental crate - suppress docs and dead code warnings during development
#![allow(missing_docs)]
#![warn(clippy::all)]
#![allow(dead_code)]

pub mod analysis;
pub mod ast;
pub mod build;
pub mod codegen;
pub mod comptime;
pub mod config;
pub mod docgen;
pub mod formatter;
pub mod linter;
pub mod logging;
pub mod module;
pub mod optimize;
pub mod parser;
pub mod pretty;
pub mod rational;
pub mod semantic;
pub mod test_runner;

/// Crate version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

// Re-export commonly used analysis types for convenience
pub use analysis::{
    AllocatorAnalysis, AllocatorInfo, DepEdge, DepKind, DependencyGraph, OperationTagger,
    ParallelismSummary, Resource, TaggedOp, analyze_parallelism,
};

/// Parse a Zluppy source file into an AST.
///
/// # Errors
///
/// Returns an error if the source contains syntax errors.
pub fn parse(source: &str) -> Result<ast::Program, parser::ParseError> {
    parser::parse(source)
}

/// Parse a Zluppy source file with filename for error reporting.
///
/// # Errors
///
/// Returns an error if the source contains syntax errors.
pub fn parse_file(
    source: &str,
    filename: impl Into<String>,
) -> Result<ast::Program, parser::ParseError> {
    parser::parse_file(source, filename)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn test_analysis_reexports() {
        // Verify the analysis re-exports are accessible
        let source = r#"
            fn main() -> unit {
                mut q := qalloc(2);
                h q[0];
                return;
            }
        "#;

        let program = parse(source).unwrap();

        // Test re-exported types and functions
        let allocator_analysis = AllocatorAnalysis::analyze(&program);
        assert!(allocator_analysis.allocators.contains_key("q"));

        let summaries = analyze_parallelism(&program);
        assert_eq!(summaries.len(), 1);
        assert_eq!(summaries[0].function_name, "main");

        // Test Resource enum
        let _r1 = Resource::allocator("q");
        let _r2 = Resource::qubit("q", 0);
        let _r3 = Resource::variable("x");

        // Test OperationTagger and DependencyGraph
        let tagger = OperationTagger::tag(&program);
        let graph = DependencyGraph::build(tagger.operations);
        let _layers = graph.parallel_layers();
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod comprehensive_tests;
