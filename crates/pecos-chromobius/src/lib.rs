//! Chromobius color code decoder for PECOS

// Internal crate - don't require exhaustive docs
#![allow(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::must_use_candidate
)]

pub mod bridge;
pub mod decoder;

pub use self::decoder::{ChromobiusConfig, ChromobiusDecoder, ChromobiusError, DecodingResult};
