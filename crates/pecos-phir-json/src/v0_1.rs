pub mod ast;
pub mod classical_interpreter;
pub mod engine;
pub mod foreign_objects;
pub mod name_resolver;
pub mod operations;
pub mod phir_converter;
pub mod wasm_foreign_object;

// Our improved implementations
pub mod block_executor;
pub mod block_iterative_executor;
pub mod enhanced_results;
pub mod environment;
pub mod expression;

// The following modules have been removed as their functionality
// has been integrated into operations.rs and engine.rs

use crate::version_traits::PhirImplementation;
use pecos_core::errors::PecosError;
use pecos_engines::ClassicalControlEngine;
use std::path::Path;

/// Implementation of PHIR-JSON v0.1
pub struct V0_1;

impl PhirImplementation for V0_1 {
    type Program = ast::PHIRProgram;
    type Engine = engine::PhirJsonEngine;

    fn parse_program(json: &str) -> Result<Self::Program, PecosError> {
        let program: Self::Program = serde_json::from_str(json).map_err(|e| {
            PecosError::Input(format!(
                "Failed to parse PHIR-JSON program: Invalid JSON format: {e}"
            ))
        })?;

        if program.format != "PHIR/JSON" {
            return Err(PecosError::Input(format!(
                "Invalid PHIR-JSON program format: found '{}', expected 'PHIR/JSON'",
                program.format
            )));
        }

        if program.version != "0.1.0" {
            return Err(PecosError::Input(format!(
                "Unsupported PHIR-JSON version: found '{}', only version '0.1.0' is supported",
                program.version
            )));
        }

        Ok(program)
    }

    fn create_engine(program: Self::Program) -> Result<Self::Engine, PecosError> {
        Self::Engine::from_program(program)
    }
}

/// Enhanced implementation of PHIR-JSON v0.1 that uses our improved components
/// Note: We've now integrated the enhancements directly into the regular `PhirJsonEngine`,
/// so this is now just an alias for `V0_1` to maintain backward compatibility.
pub struct EnhancedV0_1;

impl PhirImplementation for EnhancedV0_1 {
    type Program = ast::PHIRProgram;
    type Engine = engine::PhirJsonEngine; // Using the regular PhirJsonEngine now that it's been enhanced

    fn parse_program(json: &str) -> Result<Self::Program, PecosError> {
        // Use the same parsing logic as V0_1
        V0_1::parse_program(json)
    }

    fn create_engine(program: Self::Program) -> Result<Self::Engine, PecosError> {
        // Create engine using the regular PhirJsonEngine which now has our enhancements
        engine::PhirJsonEngine::from_program(program)
    }
}

/// Shorthand function to set up a v0.1 PHIR-JSON engine from a file path
///
/// # Errors
/// Returns an error if the file cannot be read or parsed.
pub fn setup_phir_json_v0_1_engine(
    program_path: &Path,
) -> Result<Box<dyn ClassicalControlEngine>, PecosError> {
    V0_1::setup_engine(program_path)
}

/// Shorthand function to set up an enhanced v0.1 PHIR-JSON engine from a file path
///
/// # Errors
/// Returns an error if the file cannot be read or parsed.
pub fn setup_enhanced_phir_json_v0_1_engine(
    program_path: &Path,
) -> Result<Box<dyn ClassicalControlEngine>, PecosError> {
    EnhancedV0_1::setup_engine(program_path)
}

/// Shorthand function to set up an enhanced v0.1 PHIR-JSON engine from a file path with WebAssembly support
///
/// # Errors
/// Returns an error if the files cannot be read, parsed, or WebAssembly initialization fails.
#[cfg(feature = "wasm")]
pub fn setup_enhanced_phir_json_v0_1_engine_with_wasm(
    program_path: &Path,
    wasm_path: &Path,
) -> Result<Box<dyn ClassicalControlEngine>, PecosError> {
    use crate::v0_1::wasm_foreign_object::WasmtimeForeignObject;

    // Create WebAssembly foreign object
    let foreign_object = WasmtimeForeignObject::new(wasm_path)?;
    let foreign_object = Box::new(foreign_object);

    // Create engine
    let content = std::fs::read_to_string(program_path).map_err(PecosError::IO)?;
    let program = EnhancedV0_1::parse_program(&content)?;
    let mut engine = EnhancedV0_1::create_engine(program)?;

    // Set foreign object
    engine.set_foreign_object(foreign_object);

    Ok(Box::new(engine))
}

/// Fallback function when WebAssembly support is disabled
///
/// # Errors
/// Always returns an error indicating WebAssembly support is not enabled.
#[cfg(not(feature = "wasm"))]
pub fn setup_enhanced_phir_json_v0_1_engine_with_wasm(
    _program_path: &Path,
    _wasm_path: &Path,
) -> Result<Box<dyn ClassicalControlEngine>, PecosError> {
    Err(PecosError::Feature(
        "WebAssembly support is not enabled. Rebuild with the 'wasm' feature to enable it."
            .to_string(),
    ))
}

/// Shorthand function to set up a v0.1 PHIR-JSON engine from a file path with WebAssembly support
///
/// # Errors
/// Returns an error if the files cannot be read, parsed, or WebAssembly initialization fails.
#[cfg(feature = "wasm")]
pub fn setup_phir_json_v0_1_engine_with_wasm(
    program_path: &Path,
    wasm_path: &Path,
) -> Result<Box<dyn ClassicalControlEngine>, PecosError> {
    use crate::v0_1::wasm_foreign_object::WasmtimeForeignObject;

    // Create WebAssembly foreign object
    let foreign_object = WasmtimeForeignObject::new(wasm_path)?;
    let foreign_object = Box::new(foreign_object);

    // Create engine
    let content = std::fs::read_to_string(program_path).map_err(PecosError::IO)?;
    let program = V0_1::parse_program(&content)?;
    let mut engine = V0_1::create_engine(program)?;

    // Set foreign object
    engine.set_foreign_object(foreign_object);

    Ok(Box::new(engine))
}

/// Fallback function when WebAssembly support is disabled
///
/// # Errors
/// Always returns an error indicating WebAssembly support is not enabled.
#[cfg(not(feature = "wasm"))]
pub fn setup_phir_json_v0_1_engine_with_wasm(
    _program_path: &Path,
    _wasm_path: &Path,
) -> Result<Box<dyn ClassicalControlEngine>, PecosError> {
    Err(PecosError::Feature(
        "WebAssembly support is not enabled. Rebuild with the 'wasm' feature to enable it."
            .to_string(),
    ))
}
