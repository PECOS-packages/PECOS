//! # zlup-ffi
//!
//! FFI traits and types for integrating Rust code with Zlup.
//!
//! This crate provides:
//! - Traits that decoders and simulation backends should implement
//! - FFI-safe types for crossing the Zlup/Rust boundary
//! - (With `macros` feature) `#[zlup_export]` proc macro for generating C ABI wrappers
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use zlup_ffi::prelude::*;
//!
//! pub struct MyDecoder { /* ... */ }
//!
//! impl Decoder for MyDecoder {
//!     type Syndrome = u64;
//!     type Correction = u64;
//!
//!     fn decode(&self, syndrome: u64) -> u64 {
//!         // Your decoding logic here
//!         0
//!     }
//! }
//! ```
//!
//! See the [Zlup Rust Integration Guide](https://github.com/PECOS-packages/PECOS) for details.

#![warn(missing_docs)]

pub mod types;
pub mod traits;

/// Prelude module - import everything commonly needed
pub mod prelude {
    pub use crate::traits::*;
    pub use crate::types::*;
}

// TODO: Re-export proc macros when zlup-ffi-macros crate is implemented
// #[cfg(feature = "macros")]
// pub use zlup_ffi_macros::zlup_export;
