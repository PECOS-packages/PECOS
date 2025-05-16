use log::debug;
use pecos::prelude::*;
use std::path::Path;

/// Sets up a classical engine for the CLI based on the program type
///
/// This function handles all engine types including QIR, PHIR, and QASM.
pub fn setup_cli_engine(
    program_path: &Path,
    shots: Option<usize>,
) -> Result<Box<dyn ClassicalEngine>, PecosError> {
    debug!("Setting up engine for path: {}", program_path.display());

    // Create build directory for engine outputs
    let build_dir = program_path
        .parent()
        .ok_or_else(|| {
            PecosError::Input(format!(
                "Cannot determine parent directory for path: {}",
                program_path.display()
            ))
        })?
        .join("build");
    debug!("Build directory: {}", build_dir.display());
    std::fs::create_dir_all(&build_dir).map_err(PecosError::IO)?;

    // The detect_program_type function now includes proper context in errors
    let program_type = detect_program_type(program_path)?;

    match program_type {
        ProgramType::QIR => {
            debug!("Setting up QIR engine");
            setup_qir_engine(program_path, shots)
        }
        ProgramType::PHIR => {
            debug!("Setting up PHIR engine");
            setup_phir_engine(program_path)
        }
        ProgramType::QASM => {
            debug!("Setting up QASM engine");

            // Create a new QASMEngine from the path
            // Let MonteCarloEngine handle all seeding and randomness
            let engine = QASMEngine::from_file(program_path)?;

            Ok(Box::new(engine))
        }
    }
}
