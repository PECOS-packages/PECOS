/*!
MLIR Toolchain Integration

This module provides integration with MLIR tools (mlir-opt, mlir-translate)
to lower MLIR text to LLVM IR.
*/

use pecos_core::errors::PecosError;
use std::io::Write;
use std::process::{Command, Stdio};
use tempfile::NamedTempFile;

/// Configuration for MLIR toolchain
#[derive(Debug, Clone)]
pub struct MlirToolchainConfig {
    /// Path to mlir-opt binary
    pub mlir_opt_path: Option<String>,
    /// Path to mlir-translate binary
    pub mlir_translate_path: Option<String>,
    /// Additional passes for mlir-opt
    pub optimization_passes: Vec<String>,
    /// Keep intermediate files for debugging
    pub keep_intermediate_files: bool,
}

impl Default for MlirToolchainConfig {
    fn default() -> Self {
        Self {
            mlir_opt_path: None,
            mlir_translate_path: None,
            optimization_passes: vec![
                // For MLIR-14, we use different pass names
                // Convert standard operations to LLVM
                "--convert-std-to-llvm".to_string(),
                // Convert arithmetic operations to LLVM (if available)
                "--convert-arith-to-llvm".to_string(),
                // Final cleanup (if available)
                "--reconcile-unrealized-casts".to_string(),
            ],
            keep_intermediate_files: false,
        }
    }
}

/// Process MLIR text through the toolchain to produce LLVM IR
///
/// # Errors
///
/// Returns `PecosError` if:
/// - Failed to create or write temporary files
/// - MLIR tools are not found or fail to execute
/// - MLIR optimization or translation fails
///
/// Convert MLIR text to LLVM IR using external MLIR tools
///
/// # Panics
///
/// Panics if the internal regex pattern for matching the main function is invalid.
/// This should never happen in practice as the pattern is hardcoded and tested.
pub fn mlir_to_llvm_ir(
    mlir_text: &str,
    config: &MlirToolchainConfig,
) -> Result<String, PecosError> {
    use regex::Regex;
    // Write MLIR to temporary file
    let mut mlir_file = NamedTempFile::new().map_err(PecosError::IO)?;

    mlir_file
        .write_all(mlir_text.as_bytes())
        .map_err(PecosError::IO)?;

    mlir_file.flush().map_err(PecosError::IO)?;

    let mlir_path = mlir_file.path();

    // Run mlir-opt for optimization and lowering passes
    let mlir_opt = if let Some(path) = &config.mlir_opt_path {
        path.clone()
    } else {
        find_executable("mlir-opt")
            .ok_or_else(|| PecosError::Resource(
                "mlir-opt not found. Please install MLIR tools (e.g., 'sudo apt install mlir-14-tools').".to_string()
            ))?
    };

    let mut opt_cmd = Command::new(&mlir_opt);
    opt_cmd.arg(mlir_path);

    // Add optimization passes
    for pass in &config.optimization_passes {
        opt_cmd.arg(pass);
    }

    let opt_output = opt_cmd
        .output()
        .map_err(|e| PecosError::Processing(format!("Failed to run mlir-opt: {e}")))?;

    if !opt_output.status.success() {
        let stderr = String::from_utf8_lossy(&opt_output.stderr);
        return Err(PecosError::Processing(format!("mlir-opt failed: {stderr}")));
    }

    // Run mlir-translate to convert to LLVM IR
    let mlir_translate = if let Some(path) = &config.mlir_translate_path {
        path.clone()
    } else {
        find_executable("mlir-translate")
            .ok_or_else(|| PecosError::Resource(
                "mlir-translate not found. Please install MLIR tools (e.g., 'sudo apt install mlir-14-tools').".to_string()
            ))?
    };

    let translate_output = Command::new(&mlir_translate)
        .arg("--mlir-to-llvmir")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            // Write optimized MLIR to stdin
            if let Some(stdin) = child.stdin.as_mut() {
                stdin.write_all(&opt_output.stdout)?;
            }
            child.wait_with_output()
        })
        .map_err(|e| PecosError::Processing(format!("Failed to run mlir-translate: {e}")))?;

    if !translate_output.status.success() {
        let stderr = String::from_utf8_lossy(&translate_output.stderr);
        return Err(PecosError::Processing(format!(
            "mlir-translate failed: {stderr}"
        )));
    }

    // Get LLVM IR
    let mut llvm_ir = String::from_utf8(translate_output.stdout)
        .map_err(|e| PecosError::Processing(format!("Invalid UTF-8 in LLVM IR: {e}")))?;

    // Add EntryPoint attribute to main function for PECOS runtime compatibility
    // Use regex to match any main function signature
    let main_pattern = Regex::new(r"define (\{[^}]+\}|[^ ]+) @main\(\)")
        .expect("Invalid regex pattern for main function - this is a bug");

    if let Some(captures) = main_pattern.captures(&llvm_ir) {
        let original = captures.get(0).unwrap().as_str();
        let replacement = format!("{original} #0");
        llvm_ir = llvm_ir.replace(original, &replacement);

        // Add attribute definition at the end if not present
        if !llvm_ir.contains("attributes #0") {
            llvm_ir.push_str("\nattributes #0 = { \"EntryPoint\" }\n");
        }
    }

    // Note: Qubit handles from __quantum__rt__qubit_allocate() are already 0-based
    // No additional indexing transformation needed

    Ok(llvm_ir)
}

/// Find an executable, trying versioned variants if the base name fails
fn find_executable(base_name: &str) -> Option<String> {
    // Try the base name first
    if Command::new(base_name).arg("--version").output().is_ok() {
        return Some(base_name.to_string());
    }

    // Try common versioned variants
    for version in &["18", "17", "16", "15", "14", "13", "12"] {
        let versioned = format!("{base_name}-{version}");
        if Command::new(&versioned).arg("--version").output().is_ok() {
            return Some(versioned);
        }
    }

    None
}

/// Check if MLIR tools are available
/// Check if MLIR tools are available
///
/// # Errors
///
/// Returns `PecosError` if any required MLIR tool is not found or cannot be executed
pub fn check_mlir_tools(config: &MlirToolchainConfig) -> Result<(), PecosError> {
    let mlir_opt = if let Some(path) = &config.mlir_opt_path {
        path.clone()
    } else {
        find_executable("mlir-opt")
            .ok_or_else(|| PecosError::Resource(
                "mlir-opt not found. Please install MLIR tools (e.g., 'sudo apt install mlir-14-tools').".to_string()
            ))?
    };

    let mlir_translate = if let Some(path) = &config.mlir_translate_path {
        path.clone()
    } else {
        find_executable("mlir-translate")
            .ok_or_else(|| PecosError::Resource(
                "mlir-translate not found. Please install MLIR tools (e.g., 'sudo apt install mlir-14-tools').".to_string()
            ))?
    };

    // Check mlir-opt
    Command::new(&mlir_opt)
        .arg("--version")
        .output()
        .map_err(|e| PecosError::Resource(format!("mlir-opt not accessible: {e}")))?;

    // Check mlir-translate
    Command::new(&mlir_translate)
        .arg("--version")
        .output()
        .map_err(|e| PecosError::Resource(format!("mlir-translate not accessible: {e}")))?;

    Ok(())
}

/// Process MLIR text in memory (requires custom MLIR integration)
///
/// # Errors
///
/// Currently always returns an error as in-memory processing is not yet implemented
pub fn mlir_to_llvm_ir_in_memory(
    _mlir_text: &str,
    _config: &MlirToolchainConfig,
) -> Result<String, PecosError> {
    // TODO: This would require direct MLIR C++ API integration
    // For now, we use the file-based approach above
    Err(PecosError::Feature(
        "In-memory MLIR processing not yet implemented".to_string(),
    ))
}
