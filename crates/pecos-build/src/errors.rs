//! Error types for dependency management

use thiserror::Error;

/// Error type for dependency operations
#[derive(Error, Debug)]
pub enum Error {
    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Environment variable error
    #[error("Environment variable error: {0}")]
    EnvVar(#[from] std::env::VarError),

    /// Download error
    #[error("Download error: {0}")]
    Download(String),

    /// HTTP request error
    #[error("HTTP error: {0}")]
    Http(String),

    /// Archive extraction error
    #[error("Archive extraction error: {0}")]
    Archive(String),

    /// SHA256 verification error
    #[error("SHA256 mismatch: expected {expected}, got {actual}")]
    Sha256Mismatch { expected: String, actual: String },

    /// Home directory error
    #[error("Home directory error: {0}")]
    HomeDir(String),

    /// LLVM error
    #[error("LLVM error: {0}")]
    Llvm(String),

    /// CUDA error
    #[error("CUDA error: {0}")]
    Cuda(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    Config(String),

    /// Extension error
    #[error("Extension error: {0}")]
    Extension(String),

    /// Selene plugin error
    #[error("Selene error: {0}")]
    Selene(String),
}

/// Result type alias for dependency operations
pub type Result<T> = std::result::Result<T, Error>;
