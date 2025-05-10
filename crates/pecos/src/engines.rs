use log::debug;
use pecos_engines::ClassicalEngine;
use std::error::Error;
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
pub fn setup_qasm_engine(
    program_path: &Path,
    seed: Option<u64>,
) -> Result<Box<dyn ClassicalEngine>, Box<dyn Error>> {
    debug!("Setting up QASM engine for: {}", program_path.display());

    // Use the QASMEngine from the pecos-qasm crate
    let engine = if let Some(seed_value) = seed {
        // Use the seed-specific constructor
        pecos_qasm::QASMEngine::with_seed(program_path, seed_value)?
    } else {
        // Use the standard constructor
        let mut engine = pecos_qasm::QASMEngine::new()?;
        // Parse the QASM file
        let qasm = std::fs::read_to_string(program_path)
            .map_err(|e| Box::<dyn Error>::from(format!("Failed to read QASM file: {e}")))?;
        engine.from_str(&qasm)?;
        engine
    };

    Ok(Box::new(engine))
}
