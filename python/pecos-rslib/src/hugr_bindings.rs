/*!
use crate::prelude::*;
`PyO3` bindings for HUGR/LLVM functionality

This module exposes HUGR compilation and LLVM engine functionality to Python.
*/

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyType};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{LazyLock, Mutex};
use tempfile::TempDir;

static mut NEXT_ENGINE_ID: usize = 1;

/// Storage entry for LLVM engines - stores both the engine and the temporary directory
pub struct QisEngineEntry {
    pub engine: QisEngine,
    _temp_dir: Option<TempDir>, // Keep the temp dir alive
}

/// Global storage for LLVM engines when called from Python bindings
pub static PYTHON_LLVM_ENGINES: LazyLock<Mutex<BTreeMap<usize, QisEngineEntry>>> =
    LazyLock::new(|| Mutex::new(BTreeMap::new()));

/// Get the next available engine ID
fn get_next_engine_id() -> usize {
    unsafe {
        let id = NEXT_ENGINE_ID;
        NEXT_ENGINE_ID += 1;
        id
    }
}

/// Python wrapper for HUGR compiler
#[pyclass(name = "HugrCompiler")]
pub struct PyHugrCompiler {}

#[pymethods]
impl PyHugrCompiler {
    /// Create a new HUGR compiler
    #[new]
    fn new() -> Self {
        Self {}
    }

    /// Compile HUGR bytes to LLVM IR string
    ///
    /// # Arguments
    /// * `hugr_bytes` - HUGR data as bytes
    ///
    /// # Returns
    /// LLVM IR as a string
    #[allow(clippy::unused_self)] // PyO3 method convention
    fn compile_bytes_to_llvm(&self, hugr_bytes: &Bound<'_, PyBytes>) -> PyResult<String> {
        let bytes = hugr_bytes.as_bytes();

        // Use the pure compilation crate
        let compiler = HugrCompiler::new();

        compiler
            .compile_hugr_bytes_to_string(bytes)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Compile HUGR bytes to LLVM IR file
    ///
    /// # Arguments
    /// * `hugr_bytes` - HUGR data as bytes
    /// * `llvm_path` - Path for output LLVM IR file
    #[allow(clippy::unused_self)] // PyO3 method convention
    fn compile_bytes_to_llvm_file(
        &self,
        hugr_bytes: &Bound<'_, PyBytes>,
        llvm_path: &str,
    ) -> PyResult<()> {
        let config = HugrCompilerConfig {
            output_path: Some(PathBuf::from(llvm_path)),
        };

        let compiler = HugrCompiler::with_config(config.clone());
        let bytes = hugr_bytes.as_bytes();

        // Compile directly to the output path
        compiler
            .compile_hugr_bytes(bytes, config.output_path.as_ref().unwrap())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(())
    }

    /// Compile HUGR file to LLVM IR file
    ///
    /// # Arguments
    /// * `hugr_path` - Path to HUGR file
    /// * `llvm_path` - Path for output LLVM IR file
    #[allow(clippy::unused_self)] // PyO3 method convention
    fn compile_file_to_llvm(&self, hugr_path: &str, llvm_path: &str) -> PyResult<()> {
        let config = HugrCompilerConfig {
            output_path: Some(PathBuf::from(llvm_path)),
        };

        let compiler = HugrCompiler::with_config(config);
        compiler
            .compile_hugr(hugr_path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(())
    }

    /// Compile HUGR bytes to QIR string (deprecated, use `compile_bytes_to_llvm`)
    fn compile_bytes_to_qir(&self, hugr_bytes: &Bound<'_, PyBytes>) -> PyResult<String> {
        self.compile_bytes_to_llvm(hugr_bytes)
    }

    /// Compile HUGR file to QIR file (deprecated, use `compile_file_to_llvm`)
    fn compile_file_to_qir(&self, hugr_path: &str, qir_path: &str) -> PyResult<()> {
        self.compile_file_to_llvm(hugr_path, qir_path)
    }
}

/// Python wrapper for HUGR LLVM engine
#[pyclass(name = "HugrQisEngine")]
pub struct PyHugrQisEngine {
    engine_id: usize,
    shots: usize,
}

#[pymethods]
impl PyHugrQisEngine {
    /// Create LLVM engine from HUGR bytes
    ///
    /// # Arguments
    /// * `hugr_bytes` - HUGR data as bytes
    /// * `shots` - Number of shots to assign to the engine
    #[new]
    fn new(hugr_bytes: &Bound<'_, PyBytes>, shots: Option<usize>) -> PyResult<Self> {
        let bytes = hugr_bytes.as_bytes();
        let shots = shots.unwrap_or(1000);

        // Step 1: Compile HUGR to LLVM IR
        let compiler = HugrCompiler::new();

        let llvm_ir = compiler
            .compile_hugr_bytes_to_string(bytes)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        // Step 2: Create temporary file for LLVM IR
        let temp_dir = TempDir::new()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        let llvm_path = temp_dir.path().join("output.ll");

        std::fs::write(&llvm_path, llvm_ir)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

        // Step 3: Create LLVM engine
        let mut engine = QisEngine::new(llvm_path);
        engine.set_assigned_shots(shots);

        // Pre-compile the engine
        engine
            .pre_compile()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        // Store the engine
        let engine_id = get_next_engine_id();
        let entry = QisEngineEntry {
            engine,
            _temp_dir: Some(temp_dir),
        };

        PYTHON_LLVM_ENGINES.lock().unwrap().insert(engine_id, entry);

        Ok(Self { engine_id, shots })
    }

    /// Create LLVM engine from HUGR file
    ///
    /// # Arguments
    /// * `hugr_path` - Path to HUGR file
    /// * `shots` - Number of shots to assign to the engine
    #[classmethod]
    fn from_file(
        _cls: &Bound<'_, PyType>,
        hugr_path: &str,
        shots: Option<usize>,
    ) -> PyResult<Self> {
        let shots = shots.unwrap_or(1000);

        // Step 1: Compile HUGR to LLVM IR
        let temp_dir = TempDir::new()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        let llvm_path = temp_dir.path().join("output.ll");

        let config = HugrCompilerConfig {
            output_path: Some(llvm_path.clone()),
        };

        let compiler = HugrCompiler::with_config(config);
        compiler
            .compile_hugr(hugr_path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        // Step 2: Create LLVM engine
        let mut engine = QisEngine::new(llvm_path);
        engine.set_assigned_shots(shots);

        // Pre-compile the engine
        engine
            .pre_compile()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        // Store the engine
        let engine_id = get_next_engine_id();
        let entry = QisEngineEntry {
            engine,
            _temp_dir: Some(temp_dir),
        };

        PYTHON_LLVM_ENGINES.lock().unwrap().insert(engine_id, entry);

        Ok(Self { engine_id, shots })
    }

    /// Run the LLVM engine and return measurement results
    ///
    /// # Returns
    /// List of measurement results (0 or 1)
    fn run(&self) -> PyResult<Vec<u8>> {
        use pecos_engines::{
            ClassicalEngine, MonteCarloEngine, PassThroughNoiseModel, QuantumEngineBuilder,
            state_vector,
        };

        let mut engines = PYTHON_LLVM_ENGINES.lock().unwrap();
        let entry = engines.get_mut(&self.engine_id).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Engine {} not found",
                self.engine_id
            ))
        })?;

        // Clone the engine to use as a ClassicalEngine
        let engine_clone = entry.engine.clone();
        let num_qubits = engine_clone.num_qubits();

        // Use MonteCarloEngine with the proper architecture
        let results = MonteCarloEngine::run_with_engines(
            Box::new(engine_clone),
            Box::new(PassThroughNoiseModel::builder().build()),
            state_vector()
                .qubits(num_qubits)
                .build()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?,
            self.shots,
            1,    // workers
            None, // seed
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        // Extract measurement results - take the first measurement from each shot
        let mut measurements = Vec::with_capacity(self.shots);
        for shot in results.shots {
            // Find the first measurement value
            let measurement = shot
                .data
                .values()
                .find_map(|data| match data {
                    pecos_engines::shot_results::Data::U32(v) => Some(*v != 0),
                    pecos_engines::shot_results::Data::I64(v) => Some(*v != 0),
                    pecos_engines::shot_results::Data::U8(v) => Some(*v != 0),
                    _ => None,
                })
                .unwrap_or(false);
            measurements.push(u8::from(measurement));
        }

        Ok(measurements)
    }

    /// Reset the engine state
    fn reset(&mut self) -> PyResult<()> {
        let mut engines = PYTHON_LLVM_ENGINES.lock().unwrap();
        let entry = engines.get_mut(&self.engine_id).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Engine {} not found",
                self.engine_id
            ))
        })?;

        // Reset by creating a new engine with the same configuration
        let llvm_path = entry.engine.get_llvm_file().to_path_buf();
        let mut new_engine = QisEngine::new(llvm_path);
        new_engine.set_assigned_shots(self.shots);
        new_engine
            .pre_compile()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        entry.engine = new_engine;
        Ok(())
    }

    /// Get the number of shots assigned to this engine
    fn get_shots(&self) -> usize {
        self.shots
    }

    /// Get the engine ID
    fn get_engine_id(&self) -> usize {
        self.engine_id
    }

    /// Create QIR engine from HUGR bytes (deprecated, use new)
    #[classmethod]
    fn from_hugr_bytes(
        _cls: &Bound<'_, PyType>,
        hugr_bytes: &Bound<'_, PyBytes>,
        shots: Option<usize>,
    ) -> PyResult<Self> {
        Self::new(hugr_bytes, shots)
    }

    /// Create QIR engine from HUGR file (deprecated, use `from_file`)
    #[classmethod]
    fn from_hugr_file(
        cls: &Bound<'_, PyType>,
        hugr_path: &str,
        shots: Option<usize>,
    ) -> PyResult<Self> {
        Self::from_file(cls, hugr_path, shots)
    }
}

impl Drop for PyHugrQisEngine {
    fn drop(&mut self) {
        // Remove from storage when dropped
        let _ = PYTHON_LLVM_ENGINES.lock().unwrap().remove(&self.engine_id);
    }
}
