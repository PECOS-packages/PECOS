use log::debug;
use pecos_core::errors::PecosError;
use pecos_engines::ClassicalEngine;
use std::path::Path;

// Import the QirEngine from pecos-qir
use pecos_qir::QirEngine;

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

    // Note: The seed parameter is unused as QASMEngine doesn't handle randomness.
    // Randomness is managed by the QuantumEngine in MonteCarloEngine.
    // The seed parameter is kept for API consistency with other engines.
    let _ = seed;

    // Use the QASMEngine from the pecos-qasm crate
    let engine = pecos_qasm::QASMEngine::from_file(program_path).map_err(|e| {
        PecosError::Processing(format!(
            "QASM engine setup failed: Could not create engine: {e}"
        ))
    })?;

    Ok(Box::new(engine))
}

/// Sets up a basic QIR engine.
///
/// This function creates a QIR engine from the provided path.
///
/// # Parameters
///
/// - `program_path`: A reference to the path of the QIR program file
/// - `shots`: Optional number of shots to assign to the engine
///
/// # Returns
///
/// Returns a `Box<dyn ClassicalEngine>` containing the QIR engine
///
/// # Errors
///
/// This function may return the following errors:
/// - `PecosError::Compilation`: If the QIR file cannot be compiled
/// - `PecosError::Processing`: If the QIR engine fails to process commands
pub fn setup_qir_engine(
    program_path: &Path,
    shots: Option<usize>,
) -> Result<Box<dyn ClassicalEngine>, PecosError> {
    debug!("Setting up QIR engine for: {}", program_path.display());

    // Create a QirEngine from the path
    let mut engine = QirEngine::new(program_path.to_path_buf());

    // Set the number of shots assigned to this engine if specified
    if let Some(num_shots) = shots {
        engine.set_assigned_shots(num_shots)?;
    }

    // Pre-compile the QIR library for efficient cloning
    engine.pre_compile()?;

    Ok(Box::new(engine))
}
