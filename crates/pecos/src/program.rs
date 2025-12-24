use log::debug;
use pecos_core::errors::PecosError;
use pecos_engines::ClassicalControlEngine;
use pecos_phir_json::setup_phir_json_engine;
use pecos_qasm::setup_qasm_engine;
use std::path::{Path, PathBuf};

/// Represents the types of programs that PECOS can execute
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgramType {
    /// Quantum Instruction Set (QIS) program in LLVM IR format
    QIS,
    /// PECOS High-level Intermediate Representation (PHIR)
    PHIR,
    /// Quantum Assembly Language (QASM)
    QASM,
}

/// Detects the type of program based on its file extension and content.
///
/// This function examines the file extension and content to determine if the file
/// corresponds to a QIS, PHIR, or QASM program type.
///
/// # Parameters
///
/// - `path`: A reference to the path of the file to be analyzed.
///
/// # Returns
///
/// Returns a `ProgramType` indicating the detected type if successful, or a `PecosError`
/// if format detection fails.
///
/// # Errors
///
/// This function may return the following errors:
/// - `PecosError::IO`: If the file cannot be opened or read.
/// - `PecosError::Input`: If the JSON content cannot be parsed or if the file does not
///   conform to a supported format (e.g., invalid JSON format for PHIR or
///   unsupported file extension).
pub fn detect_program_type(path: &Path) -> Result<ProgramType, PecosError> {
    // Check if it ends with .phir.json
    if path.to_str().is_some_and(|s| s.ends_with(".phir.json")) {
        return Ok(ProgramType::PHIR);
    }

    match path.extension().and_then(|ext| ext.to_str()) {
        Some("json") => {
            // Read JSON and verify format for backward compatibility
            let content = std::fs::read_to_string(path).map_err(PecosError::IO)?;
            let json: serde_json::Value = serde_json::from_str(&content).map_err(|e| {
                PecosError::Input(format!(
                    "Failed to detect program type: File contains invalid JSON: {e}"
                ))
            })?;

            if let Some("PHIR/JSON") = json.get("format").and_then(|f| f.as_str()) {
                Ok(ProgramType::PHIR)
            } else {
                Err(PecosError::Input(
                    "Failed to detect program type: JSON file is missing required 'format' field or has incorrect format value. Expected 'PHIR/JSON'.".into()
                ))
            }
        }
        Some("ll") => Ok(ProgramType::QIS),
        Some("qasm") => Ok(ProgramType::QASM),
        _ => Err(PecosError::Input(format!(
            "Failed to detect program type: Unsupported file extension '{}'. Expected file extensions: .ll (QIS), .phir.json (PHIR-JSON), .json (PHIR-JSON with format check), or .qasm (QASM).",
            path.extension()
                .and_then(|ext| ext.to_str())
                .unwrap_or("none")
        ))),
    }
}

/// Resolves the absolute path of the provided program.
///
/// This function takes a program path (either absolute or relative),
/// resolves it to an absolute path, and checks if the file exists.
///
/// # Parameters
///
/// - `program`: A string slice containing the path to the program file.
///
/// # Returns
///
/// Returns a `PathBuf` containing the canonicalized absolute path if successful,
/// or a `PecosError` if the file cannot be found or resolved.
///
/// # Errors
///
/// This function can return the following errors:
/// - `PecosError::IO`: If the current working directory cannot be obtained or
///   if the canonicalization of the path fails.
/// - `PecosError::Resource`: If the program file does not exist.
pub fn get_program_path(program: &str) -> Result<PathBuf, PecosError> {
    debug!("Resolving program path");

    // Get the current directory for relative path resolution
    let current_dir = std::env::current_dir().map_err(PecosError::IO)?;
    debug!("Current directory: {}", current_dir.display());

    // Resolve the path
    let path = if Path::new(program).is_absolute() {
        PathBuf::from(program)
    } else {
        current_dir.join(program)
    };

    // Check if file exists
    if !path.exists() {
        return Err(PecosError::Resource(format!(
            "Failed to locate program: File not found at path '{}'. Please check the file path and permissions.",
            path.display()
        )));
    }

    // Canonicalize the path (convert to absolute path, resolving symlinks)
    path.canonicalize()
        .map_err(|e| PecosError::IO(std::io::Error::new(
            e.kind(),
            format!("Failed to resolve program path to absolute path: '{}' - {}. The path may contain symlinks that cannot be resolved.", path.display(), e)
        )))
}

/// Sets up a `ClassicalEngine` appropriate for the given program type.
///
/// This function examines the program type and creates the corresponding
/// engine (QIS, PHIR, or QASM) for the provided program path.
///
/// # Parameters
///
/// - `program_type`: The type of program to create an engine for
/// - `program_path`: A reference to the path of the program file
/// - `seed`: Optional seed for deterministic simulation
///
/// # Returns
///
/// Returns a boxed `ClassicalEngine` if successful, or a `PecosError`
/// if engine setup fails.
///
/// # Errors
///
/// This function may return the following errors:
/// - `std::io::Error`: If the program file cannot be read
/// - `PecosError`: If engine setup fails
pub fn setup_engine_for_program(
    program_type: ProgramType,
    program_path: &Path,
    seed: Option<u64>,
) -> Result<Box<dyn ClassicalControlEngine>, PecosError> {
    debug!(
        "Setting up engine for {:?} program: {}",
        program_type,
        program_path.display(),
    );

    match program_type {
        ProgramType::QIS => {
            // QIS programs require a runtime implementation (e.g., Selene)
            // Users should use explicit builder API to select their runtime
            Err(PecosError::Processing(
                "QIS program execution requires explicit runtime selection.\n\
                \n\
                Please use the builder API with Selene or Native runtime:\n\
                \n\
                use pecos_qis_core::{{qis_engine, setup_qis_engine_with_runtime}};\n\
                use pecos_qis_selene::selene_simple_runtime;\n\
                \n\
                // Option 1: Use setup function\n\
                let engine = setup_qis_engine_with_runtime(path, selene_simple_runtime()?);\n\
                \n\
                // Option 2: Use builder\n\
                let engine = qis_engine()\n\
                    .runtime(selene_simple_runtime()?)\n\
                    .program(program)\n\
                    .build()?;"
                    .to_string(),
            ))
        }
        ProgramType::PHIR => setup_phir_json_engine(program_path),
        ProgramType::QASM => setup_qasm_engine(program_path, seed),
    }
}
