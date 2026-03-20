//! Relay BP decoder module
//!
//! This module provides Rust bindings for the Relay BP belief propagation
//! decoder for quantum LDPC error correction codes.
//!
//! Two decoders are exposed:
//! - [`RelayBpDecoder`] -- full relay ensemble with disordered memory strengths
//! - [`MinSumBpDecoder`] -- plain min-sum belief propagation

pub mod builder;
pub mod config;
pub(crate) mod convert;
pub mod core_traits;
pub mod decoder;
pub mod errors;

// Re-export main types
pub use builder::{MinSumBpBuilder, RelayBpBuilder};
pub use config::{MinSumConfig, RelayConfig, StoppingCriterion};
pub use decoder::{DecodingResult, MinSumBpDecoder, RelayBpDecoder};
pub use errors::RelayBpError;
