pub mod ast;
pub mod engine;
pub mod operations;

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
