//! Code generation backends for Zluppy.
//!
//! Zluppy compiles to multiple targets:
//! - **HUGR**: Hierarchical Unified Graph Representation for experiments/hardware
//! - **SLR-AST**: JSON bridge to Python/PECOS for integration
//! - **PHIR/JSON**: JSON serialization of PECOS High-level IR for simulator targeting
//! - **QASM**: OpenQASM 2.0 for hardware execution
//!
//! ## Design Philosophy
//!
//! Same problems as Guppy, simpler idioms:
//! - Explicit over implicit
//! - Low-level but safe
//! - Predictable, bounded output

#[cfg(feature = "hugr")]
pub mod hugr;
pub mod phir;
pub mod qasm;
pub mod slr;

#[cfg(feature = "hugr")]
pub use hugr::{CodegenMode, HugrCodegen};
pub use phir::{PhirJsonCodegen, PhirJsonError, PhirJsonProgram};
pub use qasm::{QasmCodegen, QasmError};
pub use slr::SlrCodegen;
