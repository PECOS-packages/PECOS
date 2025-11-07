use log::debug;
use pecos::DynamicEngineBuilder;
#[cfg(feature = "phir")]
use pecos::phir_json_engine;
use pecos::prelude::*;
#[cfg(all(feature = "llvm", feature = "selene"))]
use pecos::{helios_interface_builder, qis_engine, selene_simple_runtime};
use std::path::Path;

/// Sets up a classical engine for the CLI based on the program type
///
/// This function handles all engine types including QIR, PHIR, and QASM.
pub fn setup_cli_engine(
    program_path: &Path,
    _shots: Option<usize>,
    _use_jit: bool,
) -> Result<Box<dyn ClassicalControlEngine>, PecosError> {
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

            #[cfg(all(feature = "llvm", feature = "selene"))]
            {
                let qis_program = QisProgram::from_file(program_path)?;

                // Use Selene runtime and Helios interface (default and only option)
                debug!("Using Selene runtime and Helios interface for QIR engine");
                let selene_runtime = selene_simple_runtime().map_err(|e| {
                    PecosError::Generic(format!("Failed to load Selene runtime: {e}"))
                })?;
                let helios_builder = helios_interface_builder();
                let engine = qis_engine()
                    .runtime(selene_runtime)
                    .interface(helios_builder)
                    .try_program(qis_program)?
                    .build()?;

                Ok(Box::new(engine))
            }
            #[cfg(all(feature = "llvm", not(feature = "selene")))]
            {
                Err(PecosError::Input(
                    "Selene support is required for QIR programs but not compiled in.\n\
                     Please rebuild with --features selene"
                        .to_string(),
                ))
            }
            #[cfg(not(feature = "llvm"))]
            {
                Err(PecosError::Input(
                    "LLVM support not compiled in".to_string(),
                ))
            }
        }
        ProgramType::PHIR => {
            debug!("Setting up PHIR-JSON engine");
            setup_phir_json_engine(program_path)
        }
        ProgramType::QASM => {
            debug!("Setting up QASM engine");
            setup_qasm_engine(program_path, None)
        }
    }
}

/// Sets up a classical engine builder for the CLI based on the program type
///
/// This function returns a `DynamicEngineBuilder` that can be used with `sim_builder`
pub fn setup_cli_engine_builder(
    program_path: &Path,
    _use_jit: bool,
) -> Result<DynamicEngineBuilder, PecosError> {
    debug!(
        "Setting up engine builder for path: {}",
        program_path.display()
    );

    let program_type = detect_program_type(program_path)?;

    match program_type {
        ProgramType::QIR => {
            debug!("Setting up QIR engine builder");
            #[cfg(all(feature = "llvm", feature = "selene"))]
            {
                let qis_program = QisProgram::from_file(program_path)?;

                // Use Selene runtime and Helios interface (default and only option)
                debug!("Using Selene runtime and Helios interface for QIR engine builder");
                let selene_runtime = selene_simple_runtime().map_err(|e| {
                    PecosError::Generic(format!("Failed to load Selene runtime: {e}"))
                })?;
                let helios_builder = helios_interface_builder();
                let engine_builder = qis_engine()
                    .runtime(selene_runtime)
                    .interface(helios_builder)
                    .try_program(qis_program)?;

                Ok(DynamicEngineBuilder::new(engine_builder))
            }
            #[cfg(all(feature = "llvm", not(feature = "selene")))]
            {
                Err(PecosError::Input(
                    "Selene support is required for QIR programs but not compiled in.\n\
                     Please rebuild with --features selene"
                        .to_string(),
                ))
            }
            #[cfg(not(feature = "llvm"))]
            {
                Err(PecosError::Input(
                    "LLVM support not compiled in".to_string(),
                ))
            }
        }
        ProgramType::PHIR => {
            debug!("Setting up PHIR-JSON engine builder");
            #[cfg(feature = "phir")]
            {
                Ok(DynamicEngineBuilder::new(
                    phir_json_engine().file(program_path)?,
                ))
            }
            #[cfg(not(feature = "phir"))]
            {
                Err(PecosError::Input(
                    "PHIR support not compiled in".to_string(),
                ))
            }
        }
        ProgramType::QASM => {
            debug!("Setting up QASM engine builder");
            #[cfg(feature = "qasm")]
            {
                use pecos::qasm_engine;
                let qasm_content = std::fs::read_to_string(program_path)
                    .map_err(|e| PecosError::Input(format!("Failed to read QASM file: {e}")))?;
                Ok(DynamicEngineBuilder::new(qasm_engine().qasm(qasm_content)))
            }
            #[cfg(not(feature = "qasm"))]
            {
                Err(PecosError::Input(
                    "QASM support not compiled in".to_string(),
                ))
            }
        }
    }
}
