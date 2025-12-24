//! Chromobius color code decoder for PECOS

pub mod bridge;
pub mod decoder;

pub use self::decoder::{ChromobiusConfig, ChromobiusDecoder, ChromobiusError, DecodingResult};
