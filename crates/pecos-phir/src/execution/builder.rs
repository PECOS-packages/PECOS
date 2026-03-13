/*!
Builder for `PhirEngine`

Provides a `PhirEngineBuilder` implementing `ClassicalControlEngineBuilder`
for integration with the PECOS simulation infrastructure.
*/

use crate::error::Result;
use crate::phir::Module;
use pecos_core::errors::PecosError;
use pecos_engines::engine_builder::ClassicalControlEngineBuilder;

use super::engine::PhirEngine;

/// Builder for creating a `PhirEngine` from a PHIR `Module`
#[derive(Debug, Clone)]
pub struct PhirEngineBuilder {
    module: Option<Module>,
}

impl PhirEngineBuilder {
    /// Create a new builder
    #[must_use]
    pub fn new() -> Self {
        Self { module: None }
    }

    /// Set the PHIR module to execute
    #[must_use]
    pub fn program(mut self, module: Module) -> Self {
        self.module = Some(module);
        self
    }

    /// Set the module from QIS LLVM IR text (parses and converts to `QuantumOps`)
    ///
    /// # Errors
    ///
    /// Returns an error if parsing or conversion fails
    pub fn from_qis_llvm_ir(self, llvm_ir: &str) -> Result<Self> {
        let module = crate::parse_qis_to_quantum(llvm_ir)?;
        Ok(Self {
            module: Some(module),
        })
    }

    /// Set the module from HUGR bytes (envelope format from Guppy)
    ///
    /// # Errors
    ///
    /// Returns an error if HUGR parsing or conversion fails
    #[cfg(feature = "hugr")]
    pub fn from_hugr_bytes(self, hugr_bytes: &[u8]) -> Result<Self> {
        let module = crate::hugr_parser::parse_hugr_bytes_to_phir(hugr_bytes)?;
        Ok(Self {
            module: Some(module),
        })
    }

    /// Set the module from a RON string
    ///
    /// # Errors
    ///
    /// Returns an error if RON deserialization fails
    pub fn from_ron(self, ron_str: &str) -> Result<Self> {
        let module = crate::ron_support::from_ron(ron_str)?;
        Ok(Self {
            module: Some(module),
        })
    }

    /// Set the module from a RON file
    ///
    /// # Errors
    ///
    /// Returns an error if file reading or RON deserialization fails
    pub fn from_ron_file(self, path: impl AsRef<std::path::Path>) -> Result<Self> {
        let module = crate::ron_support::from_ron_file(path)?;
        Ok(Self {
            module: Some(module),
        })
    }

    /// Serialize the current module to a RON string
    ///
    /// # Errors
    ///
    /// Returns an error if no module is set or serialization fails
    pub fn to_ron(&self) -> Result<String> {
        let module = self.module.as_ref().ok_or_else(|| {
            crate::PhirError::internal("PhirEngineBuilder: no module to serialize")
        })?;
        crate::ron_support::to_ron(module)
    }

    /// Save the current module to a RON file
    ///
    /// # Errors
    ///
    /// Returns an error if no module is set, serialization fails, or file writing fails
    pub fn save_ron(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        let module = self.module.as_ref().ok_or_else(|| {
            crate::PhirError::internal("PhirEngineBuilder: no module to serialize")
        })?;
        crate::ron_support::to_ron_file(module, path)
    }
}

impl Default for PhirEngineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ClassicalControlEngineBuilder for PhirEngineBuilder {
    type Engine = PhirEngine;

    fn build(self) -> std::result::Result<Self::Engine, PecosError> {
        let module = self.module.ok_or_else(|| {
            PecosError::Input("PhirEngineBuilder: no module provided".to_string())
        })?;

        PhirEngine::new(module).map_err(|e| PecosError::Input(format!("PhirEngine build: {e}")))
    }
}

/// Create a new `PhirEngineBuilder`
#[must_use]
pub fn phir_engine() -> PhirEngineBuilder {
    PhirEngineBuilder::new()
}
