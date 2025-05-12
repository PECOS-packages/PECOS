pub mod common;
pub mod version_traits;

// Version-specific implementations
#[cfg(feature = "v0_1")]
pub mod v0_1;

// Re-exports for backward compatibility
#[cfg(feature = "v0_1")]
pub use v0_1::ast::{Operation, PHIRProgram};
#[cfg(feature = "v0_1")]
pub use v0_1::engine::PHIREngine;
#[cfg(feature = "v0_1")]
pub use v0_1::setup_phir_v0_1_engine;

use common::{PHIRVersion, detect_version};
use log::debug;
use pecos_core::errors::PecosError;
use pecos_engines::ClassicalEngine;
use std::path::Path;

/// Sets up a PHIR engine automatically detecting the version from the program file.
///
/// This function reads the PHIR program from the provided path, detects its version,
/// and creates the appropriate engine implementation.
///
/// # Parameters
///
/// - `program_path`: A reference to the path of the PHIR program file
///
/// # Returns
///
/// Returns a `Box<dyn ClassicalEngine>` containing the PHIR engine matching the detected version
///
/// # Errors
///
/// - Returns an error if the file cannot be read
/// - Returns an error if the JSON parsing fails
/// - Returns an error if the version is not supported
/// - Returns an error if the format is invalid
pub fn setup_phir_engine(program_path: &Path) -> Result<Box<dyn ClassicalEngine>, PecosError> {
    debug!("Setting up PHIR engine for: {}", program_path.display());

    // Read the program file
    let content = std::fs::read_to_string(program_path).map_err(PecosError::IO)?;

    // Detect the version
    let version = detect_version(&content)?;

    // Create the appropriate engine based on the detected version
    match version {
        #[cfg(feature = "v0_1")]
        PHIRVersion::V0_1 => setup_phir_v0_1_engine(program_path),
        #[allow(unreachable_patterns)]
        _ => Err(PecosError::Input(format!(
            "Unsupported PHIR version: {version:?}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pecos_engines::byte_message::ByteMessage;
    use std::fs::File;
    use std::io::Write;
    use tempfile::tempdir;

    #[cfg(feature = "v0_1")]
    #[test]
    fn test_phir_engine_basic() -> Result<(), PecosError> {
        let dir = tempdir().map_err(PecosError::IO)?;
        let program_path = dir.path().join("test.json");

        // Create a test program
        let program = r#"{
    "format": "PHIR/JSON",
    "version": "0.1.0",
    "metadata": {"test": "true"},
    "ops": [
        {
            "data": "qvar_define",
            "data_type": "qubits",
            "variable": "q",
            "size": 2
        },
        {
            "data": "cvar_define",
            "data_type": "i64",
            "variable": "m",
            "size": 2
        },
        {
            "data": "cvar_define",
            "data_type": "i64",
            "variable": "result",
            "size": 2
        },
        {
            "qop": "H",
            "args": [["q", 0]]
        },
        {
            "qop": "Measure",
            "args": [["q", 0]],
            "returns": [["m", 0]]
        },
        {"cop": "Result", "args": [["m", 0]], "returns": [["result", 0]]}
    ]
}"#;

        let mut file = File::create(&program_path).map_err(PecosError::IO)?;
        file.write_all(program.as_bytes()).map_err(PecosError::IO)?;

        // Test with automatic version detection
        let mut engine = setup_phir_engine(&program_path)?;

        // Generate commands and verify they're correctly generated
        let command_message = engine.generate_commands()?;

        // Parse the message back to confirm it has the correct operations
        let parsed_commands = command_message.parse_quantum_operations().map_err(|e| {
            PecosError::Input(format!(
                "PHIR test failed: Unable to validate generated quantum operations: {e}"
            ))
        })?;
        assert_eq!(parsed_commands.len(), 2);

        // Create a measurement message and test handling
        // result_id=0, outcome=1
        let message = ByteMessage::builder()
            .add_measurement_results(&[1], &[0])
            .build();

        engine.handle_measurements(message)?;

        // Get results and verify
        let results = engine.get_results()?;

        // The Result operation maps "m" to "result", so "result" should be in the output
        assert!(
            results.registers.contains_key("result"),
            "result register should be in results"
        );
        assert_eq!(
            results.registers["result"], 1,
            "result register should have value 1"
        );
        assert_eq!(
            results.registers.len(),
            1,
            "There should be exactly one register in the results"
        );

        Ok(())
    }

    #[cfg(feature = "v0_1")]
    #[test]
    fn test_explicit_v0_1_engine() -> Result<(), PecosError> {
        let dir = tempdir().map_err(PecosError::IO)?;
        let program_path = dir.path().join("test_v0_1.json");

        // Create a test program
        let program = r#"{
    "format": "PHIR/JSON",
    "version": "0.1.0",
    "metadata": {"test": "true"},
    "ops": [
        {
            "data": "qvar_define",
            "data_type": "qubits",
            "variable": "q",
            "size": 1
        },
        {
            "data": "cvar_define",
            "data_type": "i64",
            "variable": "result",
            "size": 1
        },
        {
            "qop": "H",
            "args": [["q", 0]]
        },
        {
            "qop": "Measure",
            "args": [["q", 0]],
            "returns": [["result", 0]]
        },
        {
            "cop": "Result",
            "args": [["result", 0]],
            "returns": [["output", 0]]
        }
    ]
}"#;

        let mut file = File::create(&program_path).map_err(PecosError::IO)?;
        file.write_all(program.as_bytes()).map_err(PecosError::IO)?;

        // Test with explicit v0.1 engine
        let engine = setup_phir_v0_1_engine(&program_path)?;

        // Check engine type using Any for runtime type checking
        let engine_any = engine.as_any();
        assert!(
            engine_any.is::<v0_1::engine::PHIREngine>(),
            "Engine should be v0_1::engine::PHIREngine"
        );

        Ok(())
    }
}
