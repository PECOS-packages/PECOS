use pecos_core::errors::PecosError;
use pecos_engines::ClassicalControlEngine;
use std::path::Path;

/// Trait that defines the common interface for all PHIR versions
pub trait PhirImplementation {
    /// The program type for this version
    type Program;
    /// The engine type for this version
    type Engine: ClassicalControlEngine + 'static;

    /// Parse a PHIR program from JSON
    ///
    /// # Errors
    ///
    /// Returns an error if the JSON cannot be parsed.
    fn parse_program(json: &str) -> Result<Self::Program, PecosError>;

    /// Create a new engine from a program
    ///
    /// # Errors
    ///
    /// Returns an error if the engine cannot be created.
    fn create_engine(program: Self::Program) -> Result<Self::Engine, PecosError>;

    /// Load a PHIR program from a file and create an engine
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or the engine cannot be created.
    fn setup_engine(path: &Path) -> Result<Box<dyn ClassicalControlEngine>, PecosError> {
        let content = std::fs::read_to_string(path).map_err(PecosError::IO)?;
        let program = Self::parse_program(&content)?;
        let engine = Self::create_engine(program)?;
        Ok(Box::new(engine))
    }
}
