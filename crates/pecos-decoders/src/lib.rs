//! Unified decoder library for PECOS
//!
//! This is a meta-crate that provides a unified interface to all PECOS decoders.
//! Enable the appropriate features to include specific decoder families.
//!
//! ## Features
//!
//! - `ldpc` - LDPC decoders (BP-OSD, BP-LSD, Union-Find, etc.)
//! - `fusion-blossom` - Fusion Blossom MWPM decoder (pure Rust)
//! - `pymatching` - `PyMatching` MWPM decoder (C++ FFI)
//! - `tesseract` - Tesseract search-based decoder (C++ FFI)
//! - `chromobius` - Chromobius color code decoder (C++ FFI)
//! - `all` - Enable all decoders

// Re-export core traits
pub use pecos_decoder_core::{
    BatchDecoder, CssDecoder, Decoder, DecoderError, DecodingResultTrait, SoftDecoder,
};

// Re-export LDPC decoders when feature is enabled
#[cfg(feature = "ldpc")]
pub use pecos_ldpc_decoders::{
    BeliefFindDecoder,
    BpLsdDecoder,
    // Types
    BpMethod,
    // Decoders
    BpOsdDecoder,
    BpSchedule,
    ClusterStatistics,
    CssCode,
    DecodingResult as LdpcDecodingResult,
    FlipDecoder,
    InputVectorType,
    // Errors
    LdpcError,
    LsdStatistics,
    MbpDecoder,
    OsdMethod,
    SoftInfoBpDecoder,
    SparseMatrix,
    UfMethod,
    UnionFindDecoder,
};

// Re-export Fusion Blossom decoder when feature is enabled
#[cfg(feature = "fusion-blossom")]
pub use pecos_fusion_blossom::{
    DecodingOptions as FusionBlossomDecodingOptions, DecodingResult as FusionBlossomDecodingResult,
    FusionBlossomConfig, FusionBlossomDecoder, FusionBlossomError, PerfectMatchingInfo, SolverType,
    StandardCode, SyndromeData,
};

// Re-export PyMatching decoder when feature is enabled
#[cfg(feature = "pymatching")]
pub use pecos_pymatching::{
    BatchConfig, BatchDecodingResult, BoundaryIterator, CheckMatrix, CheckMatrixConfig,
    CheckMatrixError, DecodeBuffer, DecodingResult as PyMatchingDecodingResult, EdgeConfig,
    EdgeData, EdgeIterator, MatchedPair, MatchedPairsDict, MergeStrategy, NoiseResult,
    PyMatchingBuilder, PyMatchingConfig, PyMatchingDecoder, PyMatchingEdge, PyMatchingError,
    PyMatchingNode,
};

// Re-export Tesseract decoder when feature is enabled
#[cfg(feature = "tesseract")]
pub use pecos_tesseract::{
    DecodingResult as TesseractDecodingResult, TesseractConfig, TesseractDecoder,
};

// Re-export Chromobius decoder when feature is enabled
#[cfg(feature = "chromobius")]
pub use pecos_chromobius::{
    ChromobiusConfig, ChromobiusDecoder, ChromobiusError,
    DecodingResult as ChromobiusDecodingResult,
};
