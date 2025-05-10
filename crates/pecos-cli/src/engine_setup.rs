use log::debug;
use pecos::prelude::*;
use std::error::Error;
use std::path::Path;

/// Sets up a classical engine for the CLI based on the program type
///
/// This function handles all engine types including QIR, PHIR, and QASM.
pub fn setup_cli_engine(
    program_path: &Path,
    shots: Option<usize>,
) -> Result<Box<dyn ClassicalEngine>, Box<dyn Error>> {
    debug!("Setting up engine for path: {}", program_path.display());

    // Create build directory for engine outputs
    let build_dir = program_path.parent().unwrap().join("build");
    debug!("Build directory: {}", build_dir.display());
    std::fs::create_dir_all(&build_dir)?;

    match detect_program_type(program_path)? {
        ProgramType::QIR => {
            debug!("Setting up QIR engine");
            let mut engine = QirEngine::new(program_path.to_path_buf());

            // Set the number of shots assigned to this engine if specified
            if let Some(num_shots) = shots {
                engine.set_assigned_shots(num_shots)?;
            }

            // Pre-compile the QIR library for efficient cloning
            engine.pre_compile()?;

            Ok(Box::new(engine))
        }
        ProgramType::PHIR => {
            debug!("Setting up PHIR engine");
            let engine = PHIREngine::new(program_path)?;
            Ok(Box::new(engine))
        }
        ProgramType::QASM => {
            debug!("Setting up QASM engine");

            // Create a new QASMEngine from the path
            // Let MonteCarloEngine handle all seeding and randomness
            let engine = QASMEngine::with_file(program_path)?;

            Ok(Box::new(engine))
        }
    }
}
