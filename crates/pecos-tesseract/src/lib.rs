//! Tesseract decoder wrapper for PECOS
//!
//! This crate provides Rust bindings for the Tesseract search-based decoder
//! for quantum error correction. Tesseract is designed for LDPC quantum codes
//! and uses A* search with pruning heuristics to find the most likely error
//! configuration consistent with observed syndromes.
//!
//! ## Key Features
//! - A* search with Dijkstra algorithm for high performance
//! - Support for Stim circuits and Detector Error Models (DEM)
//! - Parallel decoding with multithreading
//! - Beam search for efficiency optimization
//! - Comprehensive heuristics for performance tuning

pub mod bridge;
pub mod decoder;

// Re-export main types for convenience
pub use self::decoder::{DecodingResult, TesseractConfig, TesseractDecoder};
