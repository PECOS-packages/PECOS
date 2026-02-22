//! Fusion Blossom decoder module
//!
//! This module provides Rust bindings for the Fusion Blossom minimum-weight perfect matching
//! decoder for quantum error correction.

// Allow casts between float/int for weight conversions (inherent to MWPM algorithm)
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

pub mod builder;
pub mod core_traits;
pub mod decoder;
pub mod errors;

// Re-export main types
pub use builder::FusionBlossomBuilder;
pub use decoder::{
    DecodingOptions, DecodingResult, FusionBlossomConfig, FusionBlossomDecoder,
    PerfectMatchingInfo, SolverType, StandardCode, SyndromeData,
};
pub use errors::FusionBlossomError;
