pub mod ast;
pub mod engine;
pub mod foreign_objects;
pub mod operations;
pub mod wasm_foreign_object;

use crate::version_traits::PHIRImplementation;
use pecos_core::errors::PecosError;
use pecos_engines::ClassicalEngine;
use std::path::Path;

/// Implementation of PHIR v0.1
pub struct V0_1;

impl PHIRImplementation for V0_1 {
    type Program = ast::PHIRProgram;
    type Engine = engine::PHIREngine;

    fn parse_program(json: &str) -> Result<Self::Program, PecosError> {
        let program: Self::Program = serde_json::from_str(json).map_err(|e| {
            PecosError::Input(format!(
                "Failed to parse PHIR program: Invalid JSON format: {e}"
            ))
        })?;

        if program.format != "PHIR/JSON" {
            return Err(PecosError::Input(format!(
                "Invalid PHIR program format: found '{}', expected 'PHIR/JSON'",
                program.format
            )));
        }

        if program.version != "0.1.0" {
            return Err(PecosError::Input(format!(
                "Unsupported PHIR version: found '{}', only version '0.1.0' is supported",
                program.version
            )));
        }

        // Validate that at least one Result command exists
        let has_result_command = program.ops.iter().any(|op| {
            if let ast::Operation::ClassicalOp { cop, .. } = op {
                cop == "Result"
            } else {
                false
            }
        });

        if !has_result_command {
            return Err(PecosError::Input(
                "Invalid PHIR program structure: Program must contain at least one Result command to specify outputs"
                    .to_string(),
            ));
        }

        Ok(program)
    }

    fn create_engine(program: Self::Program) -> Self::Engine {
        Self::Engine::from_program(program)
    }
}

/// Shorthand function to set up a v0.1 PHIR engine from a file path
pub fn setup_phir_v0_1_engine(program_path: &Path) -> Result<Box<dyn ClassicalEngine>, PecosError> {
    V0_1::setup_engine(program_path)
}

/// Shorthand function to set up a v0.1 PHIR engine from a file path with WebAssembly support
#[cfg(feature = "wasm")]
pub fn setup_phir_v0_1_engine_with_wasm(
    program_path: &Path,
    wasm_path: &Path,
) -> Result<Box<dyn ClassicalEngine>, PecosError> {
    use crate::v0_1::wasm_foreign_object::WasmtimeForeignObject;
    use std::sync::Arc;

    // Create WebAssembly foreign object
    let foreign_object = WasmtimeForeignObject::new(wasm_path)?;
    let foreign_object = Arc::new(foreign_object);

    // Create engine
    let content = std::fs::read_to_string(program_path).map_err(PecosError::IO)?;
    let program = V0_1::parse_program(&content)?;
    let mut engine = V0_1::create_engine(program);

    // Set foreign object
    engine.set_foreign_object(foreign_object);

    Ok(Box::new(engine))
}

#[cfg(not(feature = "wasm"))]
pub fn setup_phir_v0_1_engine_with_wasm(
    _program_path: &Path,
    _wasm_path: &Path,
) -> Result<Box<dyn ClassicalEngine>, PecosError> {
    Err(PecosError::Feature(
        "WebAssembly support is not enabled. Rebuild with the 'wasm' feature to enable it."
            .to_string(),
    ))
}
