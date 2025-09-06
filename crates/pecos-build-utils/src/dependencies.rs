//! Shared dependency constants for all decoders
//!
//! This module centralizes all external dependency versions and checksums
//! to ensure consistency across the workspace and avoid duplication.

/// Stim library constants
/// Used by Tesseract, Chromobius, and PyMatching decoders
pub const STIM_COMMIT: &str = "bd60b73525fd5a9b30839020eb7554ad369e4337";
pub const STIM_SHA256: &str = "2a4be24295ce3018d79e08369b31e401a2d33cd8b3a75675d57dac3afd9de37d";

/// PyMatching library constants
/// Used by PyMatching and Chromobius decoders
pub const PYMATCHING_COMMIT: &str = "2b72b2c558eec678656da20ab6c358aa123fb664";
pub const PYMATCHING_SHA256: &str =
    "1470520b66ad7899f85020664aeeadfc6e2967f0b5e19ad205829968b845cd70";

/// LDPC library constants
/// Used by LDPC decoders
pub const LDPC_COMMIT: &str = "31cf9f33872f32579af1efbe1e84552d42b03ea8";
pub const LDPC_SHA256: &str = "43ea9bfe543233c5f65e2dfb7966229df803040b4b26e25e99c3068eb23a797a";

/// Tesseract library constants
/// Used by Tesseract decoder
pub const TESSERACT_COMMIT: &str = "1d81f0b385b6a9de49ae361d08bd6b5dbcec1773";
pub const TESSERACT_SHA256: &str =
    "0b5d8bfa63bab68ab4882510a96d7e238d598d2ba0e669a8903af142ce276892";

/// Chromobius library constants
/// Used by Chromobius decoder
pub const CHROMOBIUS_COMMIT: &str = "35e289570fdc1d71e73582e1fd4e0c8e29298ef5";
pub const CHROMOBIUS_SHA256: &str =
    "da73d819e67572065fd715db45fabb342c2a2a1e961d2609df4f9864b9836054";

/// QuEST library constants
/// Used by QuEST quantum simulator wrapper
pub const QUEST_COMMIT: &str = "v4.0.0";
pub const QUEST_SHA256: &str = "e6a922a9dc1d6ee7c4d2591a277646dca2ce2fd90eecf36fd66970cb24bbfb67";

/// Qulacs library constants
/// Used by Qulacs quantum simulator
pub const QULACS_VERSION: &str = "0.6.12";
pub const QULACS_SHA256: &str = "b9e5422e0bb2b07725b0c62f7827326b5a1486facb30cf68d12b4ef119c485e9";

/// Eigen library constants
/// Used by Qulacs quantum simulator
pub const EIGEN_VERSION: &str = "3.4.0";
pub const EIGEN_SHA256: &str = "8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72";

/// Boost library constants
/// Used by Qulacs quantum simulator (for property_tree and dynamic_bitset)
pub const BOOST_VERSION: &str = "1.83.0";
pub const BOOST_SHA256: &str = "6478edfe2f3305127cffe8caf73ea0176c53769f4bf1585be237eb30798c3b8e";

/// Helper functions to create DownloadInfo structs for each dependency
use crate::DownloadInfo;

/// Create DownloadInfo for Stim with decoder-specific cache naming
pub fn stim_download_info(decoder_name: &str) -> DownloadInfo {
    DownloadInfo {
        url: format!("https://github.com/quantumlib/Stim/archive/{STIM_COMMIT}.tar.gz"),
        sha256: STIM_SHA256,
        name: format!("stim-{}-{}", decoder_name, &STIM_COMMIT[..8]),
    }
}

/// Create DownloadInfo for PyMatching
pub fn pymatching_download_info() -> DownloadInfo {
    DownloadInfo {
        url: format!(
            "https://github.com/oscarhiggott/PyMatching/archive/{PYMATCHING_COMMIT}.tar.gz"
        ),
        sha256: PYMATCHING_SHA256,
        name: format!("PyMatching-{}", &PYMATCHING_COMMIT[..8]),
    }
}

/// Create DownloadInfo for LDPC
pub fn ldpc_download_info() -> DownloadInfo {
    DownloadInfo {
        url: format!("https://github.com/quantumgizmos/ldpc/archive/{LDPC_COMMIT}.tar.gz"),
        sha256: LDPC_SHA256,
        name: format!("ldpc-{}", &LDPC_COMMIT[..8]),
    }
}

/// Create DownloadInfo for Tesseract
pub fn tesseract_download_info() -> DownloadInfo {
    DownloadInfo {
        url: format!(
            "https://github.com/quantumlib/tesseract-decoder/archive/{TESSERACT_COMMIT}.tar.gz"
        ),
        sha256: TESSERACT_SHA256,
        name: format!("tesseract-{}", &TESSERACT_COMMIT[..8]),
    }
}

/// Create DownloadInfo for Chromobius
pub fn chromobius_download_info() -> DownloadInfo {
    DownloadInfo {
        url: format!("https://github.com/quantumlib/chromobius/archive/{CHROMOBIUS_COMMIT}.tar.gz"),
        sha256: CHROMOBIUS_SHA256,
        name: format!("chromobius-{}", &CHROMOBIUS_COMMIT[..8]),
    }
}

/// Create DownloadInfo for QuEST
pub fn quest_download_info() -> DownloadInfo {
    DownloadInfo {
        url: format!("https://github.com/QuEST-Kit/QuEST/archive/refs/tags/{QUEST_COMMIT}.tar.gz"),
        sha256: QUEST_SHA256,
        name: format!("quest-{}", QUEST_COMMIT),
    }
}

/// Create DownloadInfo for Qulacs
pub fn qulacs_download_info() -> DownloadInfo {
    DownloadInfo {
        url: format!("https://github.com/qulacs/qulacs/archive/v{QULACS_VERSION}.tar.gz"),
        sha256: QULACS_SHA256,
        name: format!("qulacs-{}", QULACS_VERSION),
    }
}

/// Create DownloadInfo for Eigen
pub fn eigen_download_info() -> DownloadInfo {
    DownloadInfo {
        url: format!(
            "https://gitlab.com/libeigen/eigen/-/archive/{}/eigen-{}.tar.gz",
            EIGEN_VERSION, EIGEN_VERSION
        ),
        sha256: EIGEN_SHA256,
        name: format!("eigen-{}", EIGEN_VERSION),
    }
}

/// Create DownloadInfo for Boost
pub fn boost_download_info() -> DownloadInfo {
    let version_underscore = BOOST_VERSION.replace('.', "_");
    DownloadInfo {
        url: format!(
            "https://archives.boost.io/release/{}/source/boost_{}.tar.bz2",
            BOOST_VERSION, version_underscore
        ),
        sha256: BOOST_SHA256,
        name: format!("boost-{}", BOOST_VERSION),
    }
}
