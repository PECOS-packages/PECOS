//! Tesseract decoder wrapper for PECOS
//!
//! Rust bindings for the Tesseract search-based decoder
//! for quantum error correction. Tesseract is designed for LDPC quantum codes
//! and uses A* search with pruning heuristics to find the most likely error
//! configuration consistent with observed syndromes. Upstream also ships a
//! trellis-mode decoder that sums probability mass over a beam of partial
//! syndromes; [`TesseractTrellisDecoder`] wraps it as its own decoder.
//!
//! ## Key Features
//! - A* search with Dijkstra algorithm for high performance
//! - Trellis-mode beam decoder with per-shot observable probability
//! - Support for Stim circuits and Detector Error Models (DEM)
//! - Parallel decoding with multithreading
//! - Beam search for efficiency optimization
//! - Comprehensive heuristics for performance tuning

pub mod bridge;
pub mod decoder;
pub mod trellis;

// Re-export main types for convenience
pub use self::decoder::{DecodingResult, TesseractConfig, TesseractDecoder};
pub use self::trellis::{
    TesseractTrellisConfig, TesseractTrellisDecoder, TesseractTrellisRankingMode,
    TesseractTrellisResult,
};
