//! Chromobius color code decoder for PECOS
//!
//! This crate provides Rust bindings for the Chromobius decoder, which is designed
//! for decoding color codes in quantum error correction. Chromobius uses a Mobius
//! matching approach to efficiently decode color code syndromes.

pub mod bridge;
pub mod decoder;

pub use self::decoder::{ChromobiusConfig, ChromobiusDecoder, ChromobiusError, DecodingResult};
