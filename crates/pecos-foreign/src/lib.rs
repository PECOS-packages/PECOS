//! Foreign language plugin interface for PECOS.
//!
//! This crate lets authors in other languages (Go, C, C++, Julia, Python, etc.)
//! implement PECOS traits via a stable C ABI. The foreign code fills in a small
//! vtable of function pointers, and the Rust side wraps it into a real trait object.
//!
//! # Currently supported
//!
//! - [`Decoder`](pecos_decoder_core::Decoder) via [`ForeignDecoder`]
//!
//! # How it works
//!
//! 1. Foreign code creates its decoder and returns an opaque `*mut ()` handle.
//! 2. Foreign code fills a [`ForeignDecoderVTable`] with function pointers.
//! 3. Rust wraps this into a [`ForeignDecoder`] which implements [`Decoder`].
//! 4. PECOS uses it like any other decoder -- no special casing needed.

pub mod conformance;
pub mod decoder;
pub mod discovery;
pub mod engine;
pub mod ffi;
#[cfg(feature = "neo")]
pub mod gate_support;
pub mod simulator;
pub mod version;

pub use decoder::{
    ForeignDecoder, ForeignDecoderVTable, ForeignDecodingResult, ForeignDecodingResultRaw,
};
pub use simulator::{ForeignMeasurementResult, ForeignSimulator, ForeignSimulatorVTable};

// Re-export so downstream FFI crates can use traits without adding
// extra direct dependencies.
pub use pecos_decoder_core;
pub use pecos_simulators;
