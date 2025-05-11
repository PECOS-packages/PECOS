use log::debug;
use pecos_core::errors::PecosError;
use pecos_engines::ClassicalEngine;
use std::path::Path;

/// Sets up a basic QASM engine.
///
/// This function creates a QASM engine from the provided path.
///
/// # Parameters
///
/// - `program_path`: A reference to the path of the QASM program file
/// - `seed`: Optional seed value for deterministic execution
///
/// # Returns
///
/// Returns a `Box<dyn ClassicalEngine>` containing the QASM engine
///
/// # Errors
///
/// This function may return the following errors:
/// - `PecosError::IO`: If the QASM file cannot be read
/// - `PecosError::Processing`: If the QASM engine creation fails or if parsing fails
pub fn setup_qasm_engine(
    program_path: &Path,
    seed: Option<u64>,
) -> Result<Box<dyn ClassicalEngine>, PecosError> {
    debug!("Setting up QASM engine for: {}", program_path.display());

    // Use the QASMEngine from the pecos-qasm crate
    let engine = if let Some(seed_value) = seed {
        // Use the seed-specific constructor
        pecos_qasm::QASMEngine::with_seed(program_path, seed_value).map_err(|e| {
            PecosError::Processing(format!(
                "QASM engine setup failed: Could not create seeded engine: {e}"
            ))
        })?
    } else {
        // Use the standard constructor
        let mut engine = pecos_qasm::QASMEngine::new().map_err(|e| {
            PecosError::Processing(format!(
                "QASM engine setup failed: Could not create engine: {e}"
            ))
        })?;

        // Parse the QASM file
        let qasm = std::fs::read_to_string(program_path).map_err(|e| {
            PecosError::IO(std::io::Error::new(
                e.kind(),
                format!(
                    "QASM engine setup failed: Could not read QASM file {}: {}",
                    program_path.display(),
                    e
                ),
            ))
        })?;

        engine.from_str(&qasm).map_err(|e| {
            PecosError::Processing(format!(
                "QASM engine setup failed: Could not parse QASM file {}: {}",
                program_path.display(),
                e
            ))
        })?;

        engine
    };

    Ok(Box::new(engine))
}
