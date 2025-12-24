//! `PyMatching` decoder module
//!
//! This module provides Rust bindings for the `PyMatching` minimum-weight perfect matching
//! decoder for quantum error correction.

// Allow casts between float/int for weight conversions (inherent to MWPM algorithm)
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

pub mod bridge;
pub mod builder;
pub mod core_traits;
pub mod decoder;
pub mod errors;
pub mod iterators;
pub mod zero_copy;

pub mod petgraph;

// Re-export main types
pub use builder::PyMatchingBuilder;
pub use decoder::{
    BatchConfig, BatchDecodingResult, CheckMatrix, CheckMatrixConfig, DecodingResult, EdgeConfig,
    EdgeData, MatchedPair, MatchedPairsDict, MergeStrategy, NoiseResult, PyMatchingConfig,
    PyMatchingDecoder,
};
pub use errors::{CheckMatrixError, PyMatchingError};
pub use iterators::{BoundaryIterator, EdgeIterator};
pub use zero_copy::DecodeBuffer;

pub use petgraph::{
    PyMatchingEdge, PyMatchingNode, pymatching_from_petgraph, pymatching_from_petgraph_weighted,
    pymatching_to_petgraph,
};
