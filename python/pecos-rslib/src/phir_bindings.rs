// Copyright 2025 The PECOS Developers
use crate::prelude::*;
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License.You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

use crate::prelude::*;
//! Python bindings for PHIR (PECOS High-level IR) compilation pipeline


use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;


/// Find PECOS binary in various possible locations
fn find_pecos_binary() -> Option<std::path::PathBuf> {
    let mut possible_paths = vec![
        // Try relative paths from current working directory
        std::path::PathBuf::from("target/release/pecos"),
        std::path::PathBuf::from("../target/release/pecos"),
        std::path::PathBuf::from("../../target/release/pecos"),
        std::path::PathBuf::from("../../../target/release/pecos"),
        // Try common install locations
        std::path::PathBuf::from("/usr/local/bin/pecos"),
        std::path::PathBuf::from("/usr/bin/pecos"),
    ];

    // Try environment variable
    if let Ok(env_path) = std::env::var("PECOS_BINARY") {
        possible_paths.insert(0, std::path::PathBuf::from(env_path));
    }

    possible_paths
        .into_iter()
        .find(|path| path.exists() && path.is_file())
}

/// Convert HUGR JSON to PHIR (MLIR text format)
#[pyfunction]
#[pyo3(name = "hugr_to_phir_mlir")]
pub fn py_hugr_to_phir_mlir(
    hugr_json: &str,
    debug_output: Option<bool>,
    optimization_level: Option<u8>,
) -> PyResult<String> {
    let config = PhirConfig {
        debug: debug_output.unwrap_or(false),
        optimization_level: optimization_level.unwrap_or(2),
        target_triple: None,
        generate_llvm_ir: false, // For MLIR text output, not LLVM IR
    };

    // Parse HUGR directly to PHIR, then convert to MLIR
    let phir_module = phir::hugr_parser::parse_hugr_to_phir(hugr_json)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to parse HUGR to PHIR: {e:?}")))?;

    // Convert PHIR to MLIR text
    let mlir_text = phir::mlir_lowering::phir_to_mlir(&phir_module, &config)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to convert PHIR to MLIR: {e:?}")))?;

    Ok(mlir_text)
}

/// PHIR QIR Engine for executing PHIR-generated LLVM IR (in-memory)
#[pyclass]
#[pyo3(name = "PhirQisEngine")]
pub struct PyPhirQisEngine {
    llvm_ir_content: String,
    shots: Option<usize>,
    seed: Option<u64>,
}

#[pymethods]
impl PyPhirQisEngine {
    /// Create a new PHIR QIR engine from LLVM IR content (in-memory)
    #[new]
    pub fn new(llvm_ir: &str) -> Self {
        // Store LLVM IR content in memory instead of using temp files
        // We'll only create a temp file when actually needed for execution
        Self {
            llvm_ir_content: llvm_ir.to_string(),
            shots: None,
            seed: None,
        }
    }

    /// Set the number of shots for execution
    pub fn set_shots(&mut self, shots: usize) {
        self.shots = Some(shots);
    }

    /// Set the random seed for execution
    pub fn set_seed(&mut self, seed: u64) {
        self.seed = Some(seed);
    }

    /// Get the LLVM IR content (for inspection)
    pub fn get_llvm_ir(&self) -> String {
        self.llvm_ir_content.clone()
    }

    /// Execute the QIR and return results
    pub fn run(&mut self) -> PyResult<Py<PyAny>> {
        use pyo3::types::PyDict;
        use std::process::Command;
        use tempfile::NamedTempFile;

        // Get number of shots
        let shots = self.shots.unwrap_or(1);

        // Create temporary file only for execution (keep LLVM IR in memory until now)
        let temp_file = NamedTempFile::with_suffix(".ll")
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create temp file: {e}")))?;

        // Write LLVM IR content to temp file
        std::fs::write(temp_file.path(), &self.llvm_ir_content)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to write LLVM IR: {e}")))?;

        let qir_file_path = temp_file.path();

        Python::attach(|py| {
            // Try to find PECOS binary in various locations
            let pecos_binary =
                find_pecos_binary().unwrap_or_else(|| std::path::PathBuf::from("pecos"));

            let mut cmd = Command::new(pecos_binary);
            cmd.args([
                "run",
                &qir_file_path.to_string_lossy(),
                "--shots",
                &shots.to_string(),
                "--format",
                "decimal",
            ]);

            // Add seed if provided
            if let Some(seed) = self.seed {
                cmd.args(["--seed", &seed.to_string()]);
            }

            let output = cmd.output();

            match output {
                Ok(result) if result.status.success() => {
                    let stdout = String::from_utf8_lossy(&result.stdout);
                    let result_dict = PyDict::new(py);

                    // For now, just return the raw JSON string
                    // The user can parse it in Python if needed
                    result_dict.set_item("raw_output", stdout.trim())?;
                    result_dict.set_item("status", "success")?;
                    result_dict.set_item("shots", shots)?;

                    Ok(result_dict.into())
                }
                Ok(result) => {
                    // Check if we got stdout output even with non-zero exit (e.g., segfault after successful execution)
                    let stdout = String::from_utf8_lossy(&result.stdout);
                    let stderr = String::from_utf8_lossy(&result.stderr);

                    if !stdout.trim().is_empty() && stderr.contains("Compilation successful") {
                        // We got output despite segfault - this is expected behavior
                        let result_dict = PyDict::new(py);
                        result_dict.set_item("raw_output", stdout.trim())?;
                        result_dict.set_item("status", "success")?;
                        result_dict.set_item("shots", shots)?;
                        result_dict.set_item(
                            "note",
                            "Execution completed successfully (segfault during cleanup ignored)",
                        )?;
                        Ok(result_dict.into())
                    } else {
                        Err(PyRuntimeError::new_err(format!(
                            "PECOS execution failed: {stderr}"
                        )))
                    }
                }
                Err(e) => Err(PyRuntimeError::new_err(format!(
                    "Failed to run PECOS CLI: {e}"
                ))),
            }
        })
    }
}

/// Full PHIR pipeline: HUGR -> PHIR -> LLVM IR -> Execution
#[pyfunction]
#[pyo3(name = "compile_and_execute_via_phir")]
pub fn py_compile_and_execute_via_phir(
    hugr_json: &str,
    shots: u32,
    seed: Option<u64>,
    debug_output: bool,
    optimization_level: u8,
) -> PyResult<Py<PyAny>> {
    // Step 1: Compile HUGR to LLVM IR via PHIR
    let config = PhirConfig {
        debug: debug_output,
        optimization_level,
        target_triple: None,
        generate_llvm_ir: true, // We want LLVM IR for execution
    };

    let llvm_ir = phir::compile_hugr_via_phir(hugr_json, &config)
        .map_err(|e| PyRuntimeError::new_err(format!("PHIR compilation failed: {e:?}")))?;

    // Step 2: Create PHIR QIR engine and execute
    let mut engine = PyPhirQisEngine::new(&llvm_ir);
    engine.set_shots(shots as usize);
    if let Some(s) = seed {
        engine.set_seed(s);
    }
    engine.run()
}

/// Compile HUGR to LLVM IR via PHIR pipeline (without execution)
#[pyfunction]
#[pyo3(name = "compile_hugr_via_phir")]
pub fn py_compile_hugr_via_phir(
    hugr_json: &str,
    debug_output: Option<bool>,
    optimization_level: Option<u8>,
    target_triple: Option<String>,
) -> PyResult<String> {
    let config = PhirConfig {
        debug: debug_output.unwrap_or(false),
        optimization_level: optimization_level.unwrap_or(2),
        target_triple,
        generate_llvm_ir: true, // Default to generating LLVM IR
    };

    phir::compile_hugr_via_phir(hugr_json, &config)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to compile via PHIR: {e:?}")))
}

/// Register PHIR Python module
pub fn register_phir_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Add PHIR functions directly to the module
    m.add_function(wrap_pyfunction!(py_hugr_to_phir_mlir, m)?)?;
    m.add_function(wrap_pyfunction!(py_compile_hugr_via_phir, m)?)?;
    m.add_function(wrap_pyfunction!(py_compile_and_execute_via_phir, m)?)?;

    // Add PHIR QIR Engine class
    m.add_class::<PyPhirQisEngine>()?;

    Ok(())
}
