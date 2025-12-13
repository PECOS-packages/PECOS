//! External dependency definitions
//!
//! This module contains version and checksum constants for external C++ dependencies.
//! These constants are used by `Manifest::default_pecos()` to generate default manifests.
//!
//! Build scripts should NOT use these constants directly. Instead, use the manifest:
//! ```ignore
//! let manifest = Manifest::find_and_load()?;
//! let info = manifest.get_download_info("quest")?;
//! let data = download_cached(&info)?;
//! ```

// =============================================================================
// Stim - Stabilizer simulator
// =============================================================================

/// Stim library commit hash
pub const STIM_COMMIT: &str = "bd60b73525fd5a9b30839020eb7554ad369e4337";
/// Stim archive SHA256 checksum
pub const STIM_SHA256: &str = "2a4be24295ce3018d79e08369b31e401a2d33cd8b3a75675d57dac3afd9de37d";

// =============================================================================
// PyMatching - MWPM decoder
// =============================================================================

/// `PyMatching` library commit hash
pub const PYMATCHING_COMMIT: &str = "2b72b2c558eec678656da20ab6c358aa123fb664";
/// `PyMatching` archive SHA256 checksum
pub const PYMATCHING_SHA256: &str =
    "1470520b66ad7899f85020664aeeadfc6e2967f0b5e19ad205829968b845cd70";

// =============================================================================
// LDPC - Low-density parity-check decoders
// =============================================================================

/// LDPC library commit hash
pub const LDPC_COMMIT: &str = "31cf9f33872f32579af1efbe1e84552d42b03ea8";
/// LDPC archive SHA256 checksum
pub const LDPC_SHA256: &str = "43ea9bfe543233c5f65e2dfb7966229df803040b4b26e25e99c3068eb23a797a";

// =============================================================================
// Tesseract - Tesseract decoder
// =============================================================================

/// Tesseract library commit hash
pub const TESSERACT_COMMIT: &str = "1d81f0b385b6a9de49ae361d08bd6b5dbcec1773";
/// Tesseract archive SHA256 checksum
pub const TESSERACT_SHA256: &str =
    "0b5d8bfa63bab68ab4882510a96d7e238d598d2ba0e669a8903af142ce276892";

// =============================================================================
// Chromobius - Color code decoder
// =============================================================================

/// Chromobius library commit hash
pub const CHROMOBIUS_COMMIT: &str = "35e289570fdc1d71e73582e1fd4e0c8e29298ef5";
/// Chromobius archive SHA256 checksum
pub const CHROMOBIUS_SHA256: &str =
    "da73d819e67572065fd715db45fabb342c2a2a1e961d2609df4f9864b9836054";

// =============================================================================
// QuEST - Quantum simulator
// =============================================================================

/// `QuEST` library version tag
pub const QUEST_VERSION: &str = "v4.1.0";
/// `QuEST` archive SHA256 checksum
pub const QUEST_SHA256: &str = "85aa95bba6457c4f4e93221f4c417d988588891a1f7cb211c307dfe81a10cadd";

// =============================================================================
// Qulacs - Quantum simulator
// =============================================================================

/// Qulacs library version
pub const QULACS_VERSION: &str = "0.6.12";
/// Qulacs archive SHA256 checksum
pub const QULACS_SHA256: &str = "b9e5422e0bb2b07725b0c62f7827326b5a1486facb30cf68d12b4ef119c485e9";

// =============================================================================
// Eigen - C++ linear algebra library
// =============================================================================

/// Eigen library version
pub const EIGEN_VERSION: &str = "3.4.0";
/// Eigen archive SHA256 checksum
pub const EIGEN_SHA256: &str = "8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72";

// =============================================================================
// Boost - C++ libraries
// =============================================================================

/// Boost library version
pub const BOOST_VERSION: &str = "1.83.0";
/// Boost archive SHA256 checksum
pub const BOOST_SHA256: &str = "6478edfe2f3305127cffe8caf73ea0176c53769f4bf1585be237eb30798c3b8e";

// =============================================================================
// Dependency listing
// =============================================================================

/// Information about an available dependency
#[derive(Debug, Clone)]
pub struct DependencyInfo {
    /// Name of the dependency
    pub name: &'static str,
    /// Version or commit
    pub version: &'static str,
    /// Description
    pub description: &'static str,
}

/// List all available dependencies
#[must_use]
pub fn list_dependencies() -> Vec<DependencyInfo> {
    vec![
        DependencyInfo {
            name: "stim",
            version: &STIM_COMMIT[..8],
            description: "Stabilizer simulator for QEC",
        },
        DependencyInfo {
            name: "pymatching",
            version: &PYMATCHING_COMMIT[..8],
            description: "MWPM decoder",
        },
        DependencyInfo {
            name: "ldpc",
            version: &LDPC_COMMIT[..8],
            description: "LDPC decoders",
        },
        DependencyInfo {
            name: "tesseract",
            version: &TESSERACT_COMMIT[..8],
            description: "Tesseract decoder",
        },
        DependencyInfo {
            name: "chromobius",
            version: &CHROMOBIUS_COMMIT[..8],
            description: "Color code decoder",
        },
        DependencyInfo {
            name: "quest",
            version: QUEST_VERSION,
            description: "QuEST quantum simulator",
        },
        DependencyInfo {
            name: "qulacs",
            version: QULACS_VERSION,
            description: "Qulacs quantum simulator",
        },
        DependencyInfo {
            name: "eigen",
            version: EIGEN_VERSION,
            description: "C++ linear algebra library",
        },
        DependencyInfo {
            name: "boost",
            version: BOOST_VERSION,
            description: "C++ Boost libraries",
        },
    ]
}
