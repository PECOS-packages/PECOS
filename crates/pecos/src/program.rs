use log::debug;
use pecos_engines::ClassicalEngine;
use std::error::Error;
use std::path::{Path, PathBuf};

/// Represents the types of programs that PECOS can execute
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgramType {
    /// Quantum Intermediate Representation (QIR)
    QIR,
    /// PECOS High-level Intermediate Representation (PHIR)
    PHIR,
    /// Quantum Assembly Language (QASM)
    QASM,
}

/// Detects the type of program based on its file extension and content.
///
/// This function examines the file extension and content to determine if the file
/// corresponds to a QIR, PHIR, or QASM program type.
///
/// # Parameters
///
/// - `path`: A reference to the path of the file to be analyzed.
///
/// # Returns
///
/// Returns a `ProgramType` indicating the detected type if successful, or a boxed error
/// if format detection fails.
///
/// # Errors
///
/// This function may return the following errors:
/// - `std::io::Error`: If the file cannot be opened or read.
/// - `serde_json::Error`: If the JSON content cannot be parsed when detecting a PHIR program.
/// - `Box<dyn std::error::Error>`: If the file does not conform to a supported format
///   (e.g., invalid JSON format for PHIR or unsupported file extension).
pub fn detect_program_type(path: &Path) -> Result<ProgramType, Box<dyn Error>> {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some("json") => {
            // Read JSON and verify format
            let content = std::fs::read_to_string(path)?;
            let json: serde_json::Value = serde_json::from_str(&content)?;

            if let Some("PHIR/JSON") = json.get("format").and_then(|f| f.as_str()) {
                Ok(ProgramType::PHIR)
            } else {
                Err("Invalid JSON format - expected PHIR/JSON".into())
            }
        }
        Some("ll") => Ok(ProgramType::QIR),
        Some("qasm") => Ok(ProgramType::QASM),
        _ => Err("Unsupported file format. Expected .ll, .json, or .qasm".into()),
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
/// or an error if the file cannot be found or resolved.
///
/// # Errors
///
/// This function can return the following errors:
/// - `std::io::Error`: If the current working directory cannot be obtained.
/// - `Box<dyn std::error::Error>`: If the program file does not exist, or if the
///   canonicalization of the file path fails.
pub fn get_program_path(program: &str) -> Result<PathBuf, Box<dyn Error>> {
    debug!("Resolving program path");

    // Get the current directory for relative path resolution
    let current_dir = std::env::current_dir()?;
    debug!("Current directory: {}", current_dir.display());

    // Resolve the path
    let path = if Path::new(program).is_absolute() {
        PathBuf::from(program)
    } else {
        current_dir.join(program)
    };

    // Check if file exists
    if !path.exists() {
        return Err(format!("Program file not found: {}", path.display()).into());
    }

    Ok(path.canonicalize()?)
}

/// Sets up a `ClassicalEngine` appropriate for the given program type.
///
/// This function examines the program type and creates the corresponding
/// engine (QIR, PHIR, or QASM) for the provided program path.
///
/// # Parameters
///
/// - `program_type`: The type of program to create an engine for
/// - `program_path`: A reference to the path of the program file
/// - `seed`: Optional seed for deterministic simulation
///
/// # Returns
///
/// Returns a boxed `ClassicalEngine` if successful, or a boxed error
/// if engine setup fails.
///
/// # Errors
///
/// This function may return the following errors:
/// - `std::io::Error`: If the program file cannot be read
/// - `Box<dyn std::error::Error>`: If engine setup fails
pub fn setup_engine_for_program(
    program_type: ProgramType,
    program_path: &Path,
    seed: Option<u64>,
) -> Result<Box<dyn ClassicalEngine>, Box<dyn Error>> {
    debug!(
        "Setting up engine for {:?} program: {}",
        program_type,
        program_path.display()
    );

    match program_type {
        ProgramType::QIR => pecos_engines::setup_qir_engine(program_path, None),
        ProgramType::PHIR => pecos_engines::setup_phir_engine(program_path),
        ProgramType::QASM => crate::engines::setup_qasm_engine(program_path, seed),
    }
}
