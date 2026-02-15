//! Unified engine builder for HUGR that integrates with the common simulation API
//!
//! This module provides the engine builder that implements the `ClassicalControlEngineBuilder`
//! trait from pecos-engines, enabling the unified simulation API.
//!
//! # Example
//!
//! ```
//! use pecos_guppy_hugr::hugr_engine;
//! use pecos_engines::{ClassicalControlEngineBuilder, ClassicalEngine};
//!
//! // Build engine from a HUGR file
//! let hugr_path = concat!(
//!     env!("CARGO_MANIFEST_DIR"),
//!     "/../pecos/tests/test_data/hugr/single_hadamard.hugr"
//! );
//! let engine = hugr_engine()
//!     .hugr_file(hugr_path)
//!     .build()
//!     .expect("Failed to build engine");
//!
//! // The engine is now ready to use with a quantum backend
//! assert!(engine.num_qubits() >= 1);
//! ```
//!
//! For full simulation with quantum execution, see the `hugr_sim` function.

use crate::engine::GuppyHugrEngine;
use pecos_core::errors::PecosError;
use pecos_engines::ClassicalControlEngineBuilder;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tket::hugr::Hugr;

#[cfg(feature = "wasm")]
use pecos_wasm::ForeignObject;

/// Wrapper for `ForeignObject` that implements Clone using `clone_box()`
#[cfg(feature = "wasm")]
struct CloneableForeignObject(Box<dyn ForeignObject>);

#[cfg(feature = "wasm")]
impl Clone for CloneableForeignObject {
    fn clone(&self) -> Self {
        CloneableForeignObject(self.0.clone_box())
    }
}

/// Builder for HUGR engines that integrates with the unified simulation API
#[derive(Default)]
pub struct GuppyHugrEngineBuilder {
    /// The HUGR source (either bytes, file path, or direct Hugr)
    source: Option<HugrSource>,
    /// Optional foreign object for WASM calls
    #[cfg(feature = "wasm")]
    foreign_object: Option<CloneableForeignObject>,
}

impl Clone for GuppyHugrEngineBuilder {
    fn clone(&self) -> Self {
        Self {
            source: self.source.clone(),
            #[cfg(feature = "wasm")]
            foreign_object: self.foreign_object.clone(),
        }
    }
}

#[derive(Clone)]
enum HugrSource {
    /// HUGR as bytes
    Bytes(Vec<u8>),
    /// Path to HUGR file
    File(PathBuf),
    /// Direct Hugr object (wrapped in Arc for Clone)
    Direct(Arc<Hugr>),
}

impl GuppyHugrEngineBuilder {
    /// Create a new HUGR engine builder
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the HUGR source from bytes
    #[must_use]
    pub fn hugr_bytes(mut self, bytes: impl Into<Vec<u8>>) -> Self {
        self.source = Some(HugrSource::Bytes(bytes.into()));
        self
    }

    /// Set the HUGR source from a file path
    #[must_use]
    pub fn hugr_file(mut self, path: impl AsRef<Path>) -> Self {
        self.source = Some(HugrSource::File(path.as_ref().to_path_buf()));
        self
    }

    /// Set the HUGR source from a Hugr object directly
    #[must_use]
    pub fn hugr(mut self, hugr: Hugr) -> Self {
        self.source = Some(HugrSource::Direct(Arc::new(hugr)));
        self
    }

    /// Check if this builder has a HUGR source configured
    #[must_use]
    pub fn has_source(&self) -> bool {
        self.source.is_some()
    }

    /// Set a foreign object for WASM function calls
    #[cfg(feature = "wasm")]
    #[must_use]
    pub fn foreign_object(mut self, foreign_obj: Box<dyn ForeignObject>) -> Self {
        self.foreign_object = Some(CloneableForeignObject(foreign_obj));
        self
    }
}

impl ClassicalControlEngineBuilder for GuppyHugrEngineBuilder {
    type Engine = GuppyHugrEngine;

    /// Build the HUGR engine
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No HUGR source was specified
    /// - Failed to read HUGR file from disk
    /// - Failed to parse HUGR content
    fn build(self) -> Result<Self::Engine, PecosError> {
        #[allow(unused_mut)]
        let mut engine = match self.source {
            Some(HugrSource::Bytes(bytes)) => GuppyHugrEngine::from_bytes(&bytes)?,
            Some(HugrSource::File(path)) => GuppyHugrEngine::from_file(&path)?,
            Some(HugrSource::Direct(hugr)) => {
                // Clone the Hugr from the Arc
                GuppyHugrEngine::from_hugr((*hugr).clone())
            }
            None => {
                return Err(PecosError::Input(
                    "No HUGR source specified. Use .hugr(), .hugr_bytes(), or .hugr_file()"
                        .to_string(),
                ));
            }
        };

        // Set the foreign object if provided (WASM feature)
        #[cfg(feature = "wasm")]
        if let Some(foreign_obj) = self.foreign_object {
            engine.set_foreign_object(foreign_obj.0);
        }

        Ok(engine)
    }
}

impl std::fmt::Debug for GuppyHugrEngineBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GuppyHugrEngineBuilder")
            .field("has_source", &self.source.is_some())
            .field("has_foreign_object", &self.foreign_object.is_some())
            .finish()
    }
}

/// Create a new HUGR engine builder
///
/// This is the entry point for the unified simulation API with HUGR programs.
///
/// # Examples
///
/// Build an engine from a HUGR file:
///
/// ```
/// use pecos_guppy_hugr::hugr_engine;
/// use pecos_engines::{ClassicalControlEngineBuilder, ClassicalEngine};
///
/// let hugr_path = concat!(
///     env!("CARGO_MANIFEST_DIR"),
///     "/../pecos/tests/test_data/hugr/bell_state.hugr"
/// );
///
/// // Build the engine
/// let engine = hugr_engine()
///     .hugr_file(hugr_path)
///     .build()
///     .expect("Failed to build engine");
///
/// // Verify the engine loaded correctly
/// assert!(engine.num_qubits() >= 2);  // Bell state uses 2 qubits
/// ```
///
/// Build an engine from HUGR bytes:
///
/// ```
/// use pecos_guppy_hugr::hugr_engine;
/// use pecos_engines::{ClassicalControlEngineBuilder, ClassicalEngine};
///
/// let hugr_path = concat!(
///     env!("CARGO_MANIFEST_DIR"),
///     "/../pecos/tests/test_data/hugr/single_hadamard.hugr"
/// );
/// let hugr_bytes = std::fs::read(hugr_path).unwrap();
///
/// let engine = hugr_engine()
///     .hugr_bytes(hugr_bytes)  // Takes ownership of bytes
///     .build()
///     .expect("Failed to build engine");
///
/// assert!(engine.num_qubits() >= 1);
/// ```
#[must_use]
pub fn hugr_engine() -> GuppyHugrEngineBuilder {
    GuppyHugrEngineBuilder::new()
}

/// Create a new HUGR simulation builder directly from a file path
///
/// This is a convenience function for quick simulations. It combines
/// `hugr_engine().hugr_file(path).to_sim()` into a single call.
///
/// # Examples
///
/// ```no_run
/// use pecos_guppy_hugr::hugr_sim;
///
/// // Quick simulation from file (requires quantum backend at runtime)
/// let hugr_path = concat!(
///     env!("CARGO_MANIFEST_DIR"),
///     "/../pecos/tests/test_data/hugr/bell_state.hugr"
/// );
/// let results = hugr_sim(hugr_path)
///     .seed(42)
///     .run(100)
///     .unwrap();
///
/// // Each shot contains measurement results
/// for shot in &results.shots {
///     println!("Measurements: {:?}", shot.data);
/// }
/// ```
#[must_use]
pub fn hugr_sim(path: impl AsRef<Path>) -> pecos_engines::SimBuilder {
    hugr_engine().hugr_file(path).to_sim()
}

#[cfg(test)]
mod tests {
    use super::*;
    use pecos_engines::{ClassicalControlEngineBuilder, ClassicalEngine};

    #[test]
    fn test_builder_no_source() {
        let result = hugr_engine().build();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("No HUGR source"));
    }

    #[test]
    fn test_builder_from_file() {
        let hugr_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../pecos/tests/test_data/hugr/single_hadamard.hugr"
        );

        let engine = hugr_engine().hugr_file(hugr_path).build();
        assert!(engine.is_ok(), "Failed to build engine: {:?}", engine.err());

        let engine = engine.unwrap();
        assert!(engine.num_qubits() >= 1);
    }

    #[test]
    fn test_builder_from_bytes() {
        let hugr_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../pecos/tests/test_data/hugr/single_hadamard.hugr"
        );

        let bytes = std::fs::read(hugr_path).expect("Failed to read HUGR file");
        let engine = hugr_engine().hugr_bytes(bytes).build();
        assert!(engine.is_ok(), "Failed to build engine: {:?}", engine.err());
    }

    #[test]
    fn test_builder_from_hugr_direct() {
        use crate::load_hugr_from_file;

        let hugr_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../pecos/tests/test_data/hugr/single_hadamard.hugr"
        );

        // Load the Hugr directly
        let hugr = load_hugr_from_file(hugr_path).expect("Failed to load HUGR");

        // Use the builder with direct Hugr
        let engine = hugr_engine().hugr(hugr).build();
        assert!(engine.is_ok(), "Failed to build engine: {:?}", engine.err());

        let engine = engine.unwrap();
        assert!(engine.num_qubits() >= 1);
    }

    #[test]
    fn test_sim_builder_integration() {
        let hugr_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../pecos/tests/test_data/hugr/single_hadamard.hugr"
        );

        // Test the full sim builder chain
        let sim = hugr_sim(hugr_path).seed(42);

        // Run the simulation
        let results = sim.run(10);
        assert!(results.is_ok(), "Simulation failed: {:?}", results.err());

        let shots = results.unwrap();
        assert_eq!(shots.shots.len(), 10, "Should have 10 shots");
    }

    #[test]
    fn test_sim_with_bell_state() {
        let hugr_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../pecos/tests/test_data/hugr/bell_state.hugr"
        );

        let results = hugr_sim(hugr_path).seed(42).run(100);
        assert!(results.is_ok(), "Simulation failed: {:?}", results.err());

        let shots = results.unwrap();
        assert_eq!(shots.shots.len(), 100, "Should have 100 shots");

        // Verify Bell state correlation: all shots should have same value for both qubits
        for shot in &shots.shots {
            if let Some(measurements) = shot.data.get("measurements")
                && let Some(values) = measurements.as_u32_vec()
                && values.len() >= 2
            {
                assert_eq!(
                    values[0], values[1],
                    "Bell state qubits should be correlated"
                );
            }
        }
    }
}
