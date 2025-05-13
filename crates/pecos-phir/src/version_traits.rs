use pecos_core::errors::PecosError;
use pecos_engines::ClassicalEngine;
use std::path::Path;

/// Trait that defines the common interface for all PHIR versions
pub trait PHIRImplementation {
    /// The program type for this version
    type Program;
    /// The engine type for this version
    type Engine: ClassicalEngine + 'static;

    /// Parse a PHIR program from JSON
    fn parse_program(json: &str) -> Result<Self::Program, PecosError>;

    /// Create a new engine from a program
    fn create_engine(program: Self::Program) -> Result<Self::Engine, PecosError>;

    /// Load a PHIR program from a file and create an engine
    fn setup_engine(path: &Path) -> Result<Box<dyn ClassicalEngine>, PecosError> {
        let content = std::fs::read_to_string(path).map_err(PecosError::IO)?;
        let program = Self::parse_program(&content)?;
        let engine = Self::create_engine(program)?;
        Ok(Box::new(engine))
    }
}
