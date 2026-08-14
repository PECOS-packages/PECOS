//! `PyO3` wrappers for engine builders following the simulation API
//!
//! This module provides thin wrappers around the Rust engine builders,
//! maintaining the same API pattern: `engine().program(...).to_sim()`

// PyO3 convention is to return PyResult even for infallible operations
#![allow(clippy::unnecessary_wraps)]

// Import from pecos metacrate prelude
use crate::prelude::*;

// Rename quantum engine builder types for clarity (from pecos prelude)
type RustQasmEngineBuilder = pecos_qasm::QasmEngineBuilder;
type RustQisEngineBuilder = pecos_qis::QisEngineBuilder;
type RustPhirJsonEngineBuilder = pecos_phir_json::PhirJsonEngineBuilder;
type RustHugrEngineBuilder = pecos_hugr::HugrEngineBuilder;
type RustPhirEngineBuilder = pecos_phir::PhirEngineBuilder;
type RustCoinTossEngineBuilder = CoinTossEngineBuilder;
type RustStabVecEngineBuilder = StabVecEngineBuilder;
type RustDensityMatrixEngineBuilder = DensityMatrixEngineBuilder;
type RustStabilizerEngineBuilder = StabilizerEngineBuilder;
type RustSparseStabEngineBuilder = SparseStabEngineBuilder;
type RustStateVectorEngineBuilder = StateVectorEngineBuilder;

use pecos_engines::noise::{
    P2PauliLeakageStep, P2TransitionStep, PauliLeakageChannel, PauliLeakageDict,
    QubitTransitionChannel, TransitionDict, TwoQubitPauliLeakageChannel, TwoQubitTransitionChannel,
};
use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::PyBool;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

// Import existing shot result types
use crate::shot_results_bindings::PyShotVec;

// Import the unified SimBuilder from sim.rs
use crate::sim::{PySimBuilder, SimBuilderInner};

/// Python wrapper for QASM engine builder
#[pyclass(name = "QasmEngineBuilder", from_py_object)]
#[derive(Clone)]
pub struct PyQasmEngineBuilder {
    pub(crate) inner: RustQasmEngineBuilder,
}

#[pymethods]
impl PyQasmEngineBuilder {
    #[new]
    fn new() -> Self {
        Self {
            inner: pecos_qasm::qasm_engine(),
        }
    }

    /// Set the program for this engine
    #[pyo3(signature = (program))]
    fn program(&mut self, program: &PyQasm) -> PyResult<Self> {
        self.inner = self.inner.clone().program(program.inner.clone());
        Ok(self.clone())
    }

    /// Set the WebAssembly module for foreign function calls
    #[pyo3(signature = (wasm_path))]
    fn wasm(&mut self, wasm_path: &str) -> PyResult<Self> {
        self.inner = self.inner.clone().wasm(wasm_path);
        Ok(self.clone())
    }

    /// Check if this builder has a QASM source configured
    pub fn has_source(&self) -> bool {
        self.inner.has_source()
    }

    /// Get the `Qasm` from this builder (if any)
    pub fn get_program(&self) -> Option<PyQasm> {
        self.inner.get_program().map(|prog| PyQasm { inner: prog })
    }

    /// Convert to simulation builder
    fn to_sim(&self) -> PyResult<PySimBuilder> {
        Ok(PySimBuilder {
            inner: SimBuilderInner::Qasm(PyQasmSimBuilder {
                engine_builder: Arc::new(Mutex::new(Some(self.inner.clone()))),
                seed: None,
                workers: None,
                shots: None,
                quantum_engine_builder: None,
                noise_builder: None,
                explicit_num_qubits: None,
                foreign_object: None,
                stack: None,
                classical_override: false,
            }),
        })
    }
}

/// Python wrapper for QIS Engine builder (unified QIS/HUGR engine)
#[pyclass(name = "QisEngineBuilder", from_py_object)]
#[derive(Clone)]
pub struct PyQisEngineBuilder {
    pub(crate) inner: RustQisEngineBuilder,
    runtime_configured: bool,
}

#[pymethods]
impl PyQisEngineBuilder {
    #[new]
    fn new() -> Self {
        Self {
            inner: pecos_qis::qis_engine(),
            runtime_configured: false,
        }
    }

    /// Set the program for this engine
    #[pyo3(signature = (program))]
    #[allow(clippy::needless_pass_by_value)] // Py<PyAny> must be passed by value for PyO3
    fn program(&mut self, program: Py<PyAny>, py: Python) -> PyResult<Self> {
        // Check if it's a Qis
        if let Ok(qis_prog) = program.extract::<PyQis>(py) {
            self.inner = self
                .inner
                .clone()
                .try_program(qis_prog.inner)
                .map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                        "Failed to load QIS program: {e}"
                    ))
                })?;
        }
        // Check if it's a Hugr
        else if let Ok(hugr_prog) = program.extract::<PyHugr>(py) {
            self.inner = self
                .inner
                .clone()
                .try_program(hugr_prog.inner)
                .map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                        "Failed to load HUGR program: {e}"
                    ))
                })?;
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "program must be either a Qis or Hugr instance",
            ));
        }
        Ok(self.clone())
    }

    /// Use a Selene runtime built into the current PECOS/Cargo target.
    #[pyo3(signature = (runtime_name = None))]
    fn selene_runtime(&mut self, runtime_name: Option<&str>) -> PyResult<Self> {
        let runtime = match runtime_name {
            None | Some("selene_simple_runtime") => pecos_qis::selene_simple_runtime(),
            Some(name) => pecos_qis::selene_runtime_auto(name),
        }
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load Selene runtime: {e}"
            ))
        })?;
        self.inner = self.inner.clone().runtime(runtime);
        self.runtime_configured = true;
        Ok(self.clone())
    }

    /// Use a generic Selene runtime plugin by its shared library and plugin arguments.
    #[pyo3(signature = (library_file, init_args = None, library_search_dirs = None))]
    fn selene_runtime_plugin(
        &mut self,
        library_file: &str,
        init_args: Option<Vec<String>>,
        library_search_dirs: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let runtime = pecos_qis::SeleneRuntime::with_plugin_config(
            library_file,
            init_args.unwrap_or_default(),
            library_search_dirs
                .unwrap_or_default()
                .into_iter()
                .map(PathBuf::from)
                .collect(),
        );
        self.inner = self.inner.clone().runtime(runtime);
        self.runtime_configured = true;
        Ok(self.clone())
    }

    /// Set the interface builder (Helios)
    #[pyo3(signature = (_builder))]
    fn interface(&mut self, _builder: &PyQisInterfaceBuilder) -> PyResult<Self> {
        // The PyQisInterfaceBuilder contains a boxed trait object which we can't easily clone
        // Use Helios interface as the default
        log::debug!("Python interface() called, setting Helios interface");

        // Set Helios interface
        self.inner = self
            .inner
            .clone()
            .interface(pecos_qis::helios_interface_builder());

        if !self.runtime_configured {
            log::debug!(
                "No runtime configured; setting default Selene runtime for Helios interface"
            );
            let runtime = pecos_qis::selene_simple_runtime().map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Failed to load Selene runtime: {e}"
                ))
            })?;
            self.inner = self.inner.clone().runtime(runtime);
            self.runtime_configured = true;
        }

        log::debug!("Helios interface and Selene runtime configured");
        Ok(self.clone())
    }

    /// Dump Helios-collected operation chunks to the given directory as JSON.
    #[pyo3(signature = (trace_dir))]
    fn trace_operations(&mut self, trace_dir: &str) -> PyResult<Self> {
        self.inner = self.inner.clone().trace_operations_to(trace_dir);
        Ok(self.clone())
    }

    /// Convert to simulation builder
    fn to_sim(&self) -> PyResult<PySimBuilder> {
        Ok(PySimBuilder {
            inner: SimBuilderInner::QisControl(PyQisControlSimBuilder {
                engine_builder: Arc::new(Mutex::new(Some(self.inner.clone()))),
                seed: None,
                workers: None,
                shots: None,
                quantum_engine_builder: None,
                noise_builder: None,
                explicit_num_qubits: None,
                keep_intermediate_files: false,
                hugr_bytes: None,
                qis_source: None,
                operation_trace_dir: None,
            }),
        })
    }
}

/// Python wrapper for PHIR JSON engine builder
#[pyclass(name = "PhirJsonEngineBuilder", from_py_object)]
#[derive(Clone)]
pub struct PyPhirJsonEngineBuilder {
    pub(crate) inner: RustPhirJsonEngineBuilder,
}

#[pymethods]
impl PyPhirJsonEngineBuilder {
    #[new]
    fn new() -> Self {
        Self {
            inner: pecos_phir_json::phir_json_engine(),
        }
    }

    /// Set the program for this engine
    #[pyo3(signature = (program))]
    fn program(&mut self, program: &PyPhirJson) -> PyResult<Self> {
        self.inner = self.inner.clone().program(program.inner.clone());
        Ok(self.clone())
    }

    /// Set the WebAssembly module for foreign function calls
    #[pyo3(signature = (wasm_path))]
    fn wasm(&mut self, wasm_path: &str) -> PyResult<Self> {
        self.inner = self.inner.clone().wasm(wasm_path);
        Ok(self.clone())
    }

    /// Convert to simulation builder
    fn to_sim(&self) -> PyResult<PySimBuilder> {
        Ok(PySimBuilder {
            inner: SimBuilderInner::PhirJson(PyPhirJsonSimBuilder {
                engine_builder: Arc::new(Mutex::new(Some(self.inner.clone()))),
                seed: None,
                workers: None,
                shots: None,
                quantum_engine_builder: None,
                noise_builder: None,
                explicit_num_qubits: None,
            }),
        })
    }
}

/// Internal QASM simulation builder state
///
/// This stores configuration and rebuilds the Rust `SimBuilder` when needed,
/// avoiding the `FnOnce` + Sync issue while maintaining the same API
pub struct PyQasmSimBuilder {
    pub(crate) engine_builder: Arc<Mutex<Option<RustQasmEngineBuilder>>>,
    pub(crate) seed: Option<u64>,
    pub(crate) workers: Option<usize>,
    pub(crate) shots: Option<usize>,
    pub(crate) quantum_engine_builder: Option<Py<PyAny>>,
    pub(crate) noise_builder: Option<Py<PyAny>>,
    pub(crate) explicit_num_qubits: Option<usize>,
    pub(crate) foreign_object: Option<Py<PyAny>>,
    pub(crate) stack: Option<crate::sim::PySimStack>,
    /// True once `.classical()` has supplied an explicit engine builder.
    /// The neo route rejects it (the facade contract has no classical
    /// override on neo), matching the Rust `sim().stack(Neo)` behavior.
    pub(crate) classical_override: bool,
}

/// Python wrapper for built QASM simulation
#[pyclass(name = "QasmSimulation")]
pub struct PyQasmSimulation {
    pub(crate) inner: Arc<Mutex<MonteCarloEngine>>,
}

#[pymethods]
impl PyQasmSimulation {
    /// Run the simulation
    pub fn run(&self, shots: usize) -> PyResult<PyShotVec> {
        let mut engine = self.inner.lock().expect("lock poisoned");
        // Use workers from builder config or default (1)
        match engine.run(shots) {
            Ok(shot_vec) => Ok(PyShotVec::new(shot_vec)),
            Err(e) => Err(PyRuntimeError::new_err(format!("Simulation failed: {e}"))),
        }
    }

    /// Run the simulation with specified number of workers
    fn run_with_workers(&self, shots: usize, workers: usize) -> PyResult<PyShotVec> {
        let mut engine = self.inner.lock().expect("lock poisoned");
        match engine.run_with_workers(shots, workers) {
            Ok(shot_vec) => Ok(PyShotVec::new(shot_vec)),
            Err(e) => Err(PyRuntimeError::new_err(format!("Simulation failed: {e}"))),
        }
    }

    /// Reset the simulation to its initial state (quantum state back to |0⟩).
    ///
    /// Returns the simulation object for method chaining.
    fn reset(slf: PyRef<'_, Self>) -> PyResult<PyRef<'_, Self>> {
        {
            let mut engine = slf.inner.lock().expect("lock poisoned");
            engine
                .reset()
                .map_err(|e| PyRuntimeError::new_err(format!("Reset failed: {e}")))?;
        }
        Ok(slf)
    }
}

/// Python wrapper for built PHIR JSON simulation
#[pyclass(name = "PhirJsonSimulation")]
pub struct PyPhirJsonSimulation {
    pub(crate) inner: Arc<Mutex<MonteCarloEngine>>,
}

#[pymethods]
impl PyPhirJsonSimulation {
    /// Run the simulation
    pub fn run(&self, shots: usize) -> PyResult<PyShotVec> {
        let mut engine = self.inner.lock().expect("lock poisoned");
        // Use workers from builder config or default (1)
        match engine.run(shots) {
            Ok(shot_vec) => Ok(PyShotVec::new(shot_vec)),
            Err(e) => Err(PyRuntimeError::new_err(format!("Simulation failed: {e}"))),
        }
    }

    /// Run the simulation with specified number of workers
    fn run_with_workers(&self, shots: usize, workers: usize) -> PyResult<PyShotVec> {
        let mut engine = self.inner.lock().expect("lock poisoned");
        match engine.run_with_workers(shots, workers) {
            Ok(shot_vec) => Ok(PyShotVec::new(shot_vec)),
            Err(e) => Err(PyRuntimeError::new_err(format!("Simulation failed: {e}"))),
        }
    }

    /// Reset the simulation to its initial state (quantum state back to |0⟩).
    ///
    /// Returns the simulation object for method chaining.
    fn reset(slf: PyRef<'_, Self>) -> PyResult<PyRef<'_, Self>> {
        {
            let mut engine = slf.inner.lock().expect("lock poisoned");
            engine
                .reset()
                .map_err(|e| PyRuntimeError::new_err(format!("Reset failed: {e}")))?;
        }
        Ok(slf)
    }
}

/// Internal QIS Engine simulation builder state
pub struct PyQisControlSimBuilder {
    pub(crate) engine_builder: Arc<Mutex<Option<RustQisEngineBuilder>>>,
    pub(crate) seed: Option<u64>,
    pub(crate) workers: Option<usize>,
    pub(crate) shots: Option<usize>,
    pub(crate) quantum_engine_builder: Option<Py<PyAny>>,
    pub(crate) noise_builder: Option<Py<PyAny>>,
    pub(crate) explicit_num_qubits: Option<usize>,
    pub(crate) keep_intermediate_files: bool,
    pub(crate) hugr_bytes: Option<Vec<u8>>,
    /// The QIS IR source, kept so classical() can re-attach the program
    /// when a fresh engine builder replaces the program-loaded one.
    pub(crate) qis_source: Option<String>,
    pub(crate) operation_trace_dir: Option<String>,
}

/// Python wrapper for built QIS control simulation
#[pyclass(name = "QisControlSimulation")]
pub struct PyQisControlSimulation {
    pub(crate) inner: Arc<Mutex<MonteCarloEngine>>,
    /// Path to temp directory containing intermediate files (if `keep_intermediate_files` was true)
    pub(crate) temp_dir: Option<String>,
    /// Path to directory containing operation trace chunks (if enabled)
    pub(crate) operation_trace_dir: Option<String>,
}

#[pymethods]
impl PyQisControlSimulation {
    /// Run the simulation
    pub fn run(&self, shots: usize) -> PyResult<PyShotVec> {
        let mut engine = self.inner.lock().expect("lock poisoned");
        match engine.run(shots) {
            Ok(shot_vec) => Ok(PyShotVec::new(shot_vec)),
            Err(e) => Err(PyRuntimeError::new_err(format!("Simulation failed: {e}"))),
        }
    }

    /// Run the simulation with specified number of workers
    fn run_with_workers(&self, shots: usize, workers: usize) -> PyResult<PyShotVec> {
        let mut engine = self.inner.lock().expect("lock poisoned");
        match engine.run_with_workers(shots, workers) {
            Ok(shot_vec) => Ok(PyShotVec::new(shot_vec)),
            Err(e) => Err(PyRuntimeError::new_err(format!("Simulation failed: {e}"))),
        }
    }

    /// Get the temp directory path (if `keep_intermediate_files` was enabled)
    #[getter]
    fn temp_dir(&self) -> Option<String> {
        self.temp_dir.clone()
    }

    /// Get the operation trace directory (if operation tracing was enabled)
    #[getter]
    fn operation_trace_dir(&self) -> Option<String> {
        self.operation_trace_dir.clone()
    }

    /// Reset the simulation to its initial state (quantum state back to |0⟩).
    ///
    /// Returns the simulation object for method chaining.
    fn reset(slf: PyRef<'_, Self>) -> PyResult<PyRef<'_, Self>> {
        {
            let mut engine = slf.inner.lock().expect("lock poisoned");
            engine
                .reset()
                .map_err(|e| PyRuntimeError::new_err(format!("Reset failed: {e}")))?;
        }
        Ok(slf)
    }
}

/// Internal PHIR JSON simulation builder state
pub struct PyPhirJsonSimBuilder {
    pub(crate) engine_builder: Arc<Mutex<Option<RustPhirJsonEngineBuilder>>>,
    pub(crate) seed: Option<u64>,
    pub(crate) workers: Option<usize>,
    pub(crate) shots: Option<usize>,
    pub(crate) quantum_engine_builder: Option<Py<PyAny>>,
    pub(crate) noise_builder: Option<Py<PyAny>>,
    pub(crate) explicit_num_qubits: Option<usize>,
}

/// Python wrapper for PHIR engine builder (PHIR Module execution)
#[pyclass(name = "PhirEngineBuilder", from_py_object)]
#[derive(Clone)]
pub struct PyPhirEngineBuilder {
    pub(crate) inner: RustPhirEngineBuilder,
}

#[pymethods]
impl PyPhirEngineBuilder {
    #[new]
    fn new() -> Self {
        Self {
            inner: pecos_phir::phir_engine(),
        }
    }

    /// Set the program from QIS LLVM IR text
    #[pyo3(signature = (llvm_ir))]
    fn qis_llvm_ir(&self, llvm_ir: &str) -> PyResult<Self> {
        let builder =
            self.inner.clone().from_qis_llvm_ir(llvm_ir).map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to parse QIS LLVM IR: {e}"))
            })?;
        Ok(Self { inner: builder })
    }

    /// Convert to simulation builder
    fn to_sim(&self) -> PyResult<PySimBuilder> {
        Ok(PySimBuilder {
            inner: SimBuilderInner::Phir(PyPhirSimBuilder {
                engine_builder: Arc::new(Mutex::new(Some(self.inner.clone()))),
                seed: None,
                workers: None,
                shots: None,
                quantum_engine_builder: None,
                noise_builder: None,
                explicit_num_qubits: None,
            }),
        })
    }
}

/// Internal PHIR simulation builder state
pub struct PyPhirSimBuilder {
    pub(crate) engine_builder: Arc<Mutex<Option<RustPhirEngineBuilder>>>,
    pub(crate) seed: Option<u64>,
    pub(crate) workers: Option<usize>,
    pub(crate) shots: Option<usize>,
    pub(crate) quantum_engine_builder: Option<Py<PyAny>>,
    pub(crate) noise_builder: Option<Py<PyAny>>,
    pub(crate) explicit_num_qubits: Option<usize>,
}

/// Python wrapper for built PHIR simulation
#[pyclass(name = "PhirSimulation")]
pub struct PyPhirSimulation {
    pub(crate) inner: Arc<Mutex<MonteCarloEngine>>,
}

#[pymethods]
impl PyPhirSimulation {
    /// Run the simulation
    pub fn run(&self, shots: usize) -> PyResult<PyShotVec> {
        let mut engine = self.inner.lock().expect("lock poisoned");
        match engine.run(shots) {
            Ok(shot_vec) => Ok(PyShotVec::new(shot_vec)),
            Err(e) => Err(PyRuntimeError::new_err(format!("Simulation failed: {e}"))),
        }
    }

    /// Run the simulation with specified number of workers
    fn run_with_workers(&self, shots: usize, workers: usize) -> PyResult<PyShotVec> {
        let mut engine = self.inner.lock().expect("lock poisoned");
        match engine.run_with_workers(shots, workers) {
            Ok(shot_vec) => Ok(PyShotVec::new(shot_vec)),
            Err(e) => Err(PyRuntimeError::new_err(format!("Simulation failed: {e}"))),
        }
    }

    /// Reset the simulation to its initial state (quantum state back to |0>).
    ///
    /// Returns the simulation object for method chaining.
    fn reset(slf: PyRef<'_, Self>) -> PyResult<PyRef<'_, Self>> {
        {
            let mut engine = slf.inner.lock().expect("lock poisoned");
            engine
                .reset()
                .map_err(|e| PyRuntimeError::new_err(format!("Reset failed: {e}")))?;
        }
        Ok(slf)
    }
}

/// Python wrapper for HUGR engine builder (direct HUGR interpreter)
///
/// This engine directly interprets HUGR programs without LLVM compilation,
/// making it faster for simple circuits and useful for testing.
#[pyclass(name = "HugrEngineBuilder", from_py_object)]
#[derive(Clone)]
pub struct PyHugrEngineBuilder {
    pub(crate) inner: RustHugrEngineBuilder,
}

#[pymethods]
impl PyHugrEngineBuilder {
    #[new]
    fn new() -> Self {
        Self {
            inner: pecos_hugr::hugr_engine(),
        }
    }

    /// Set the HUGR source from a file path
    #[pyo3(signature = (path))]
    fn hugr_file(&self, path: &str) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().hugr_file(path),
        })
    }

    /// Set the HUGR source from bytes
    #[pyo3(signature = (bytes))]
    fn hugr_bytes(&self, bytes: Vec<u8>) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().hugr_bytes(bytes),
        })
    }

    /// Set the HUGR program
    #[pyo3(signature = (program))]
    fn program(&self, program: &PyHugr) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().hugr_bytes(program.inner.hugr.clone()),
        })
    }

    /// Check if this builder has a HUGR source configured
    pub fn has_source(&self) -> bool {
        self.inner.has_source()
    }

    /// Convert to simulation builder
    fn to_sim(&self) -> PyResult<PySimBuilder> {
        Ok(PySimBuilder {
            inner: SimBuilderInner::Hugr(PyHugrSimBuilder {
                engine_builder: Arc::new(Mutex::new(Some(self.inner.clone()))),
                seed: None,
                workers: None,
                shots: None,
                quantum_engine_builder: None,
                noise_builder: None,
                explicit_num_qubits: None,
                foreign_object: None,
                keep_intermediate_files: false,
                hugr_bytes: None,
                stack: None,
            }),
        })
    }
}

/// Internal HUGR simulation builder state
pub struct PyHugrSimBuilder {
    pub(crate) engine_builder: Arc<Mutex<Option<RustHugrEngineBuilder>>>,
    pub(crate) seed: Option<u64>,
    pub(crate) workers: Option<usize>,
    pub(crate) shots: Option<usize>,
    pub(crate) quantum_engine_builder: Option<Py<PyAny>>,
    pub(crate) noise_builder: Option<Py<PyAny>>,
    pub(crate) explicit_num_qubits: Option<usize>,
    pub(crate) foreign_object: Option<Py<PyAny>>,
    pub(crate) keep_intermediate_files: bool,
    pub(crate) hugr_bytes: Option<Vec<u8>>,
    pub(crate) stack: Option<crate::sim::PySimStack>,
}

/// Python wrapper for built HUGR simulation
#[pyclass(name = "HugrSimulation")]
pub struct PyHugrSimulation {
    pub(crate) inner: Arc<Mutex<MonteCarloEngine>>,
    /// Path to temp directory containing intermediate files (if `keep_intermediate_files` was true)
    pub(crate) temp_dir: Option<String>,
}

#[pymethods]
impl PyHugrSimulation {
    /// Run the simulation
    pub fn run(&self, shots: usize) -> PyResult<PyShotVec> {
        let mut engine = self.inner.lock().expect("lock poisoned");
        match engine.run(shots) {
            Ok(shot_vec) => Ok(PyShotVec::new(shot_vec)),
            Err(e) => Err(PyRuntimeError::new_err(format!("Simulation failed: {e}"))),
        }
    }

    /// Run the simulation with specified number of workers
    fn run_with_workers(&self, shots: usize, workers: usize) -> PyResult<PyShotVec> {
        let mut engine = self.inner.lock().expect("lock poisoned");
        match engine.run_with_workers(shots, workers) {
            Ok(shot_vec) => Ok(PyShotVec::new(shot_vec)),
            Err(e) => Err(PyRuntimeError::new_err(format!("Simulation failed: {e}"))),
        }
    }

    /// Get the temp directory path (if `keep_intermediate_files` was enabled)
    #[getter]
    fn temp_dir(&self) -> Option<String> {
        self.temp_dir.clone()
    }

    /// Reset the simulation to its initial state (quantum state back to |0⟩).
    ///
    /// Returns the simulation object for method chaining.
    fn reset(slf: PyRef<'_, Self>) -> PyResult<PyRef<'_, Self>> {
        {
            let mut engine = slf.inner.lock().expect("lock poisoned");
            engine
                .reset()
                .map_err(|e| PyRuntimeError::new_err(format!("Reset failed: {e}")))?;
        }
        Ok(slf)
    }
}

/// Python wrapper for program types
#[pyclass(name = "Qasm", from_py_object)]
#[derive(Clone)]
pub struct PyQasm {
    pub(crate) inner: Qasm,
}

#[pymethods]
impl PyQasm {
    #[staticmethod]
    fn from_string(source: String) -> Self {
        PyQasm {
            inner: Qasm::from_string(source),
        }
    }
}

#[pyclass(name = "Qis", from_py_object)]
#[derive(Clone)]
pub struct PyQis {
    pub(crate) inner: Qis,
}

#[pymethods]
impl PyQis {
    #[new]
    fn new(source: String) -> Self {
        PyQis {
            inner: Qis::from_string(source),
        }
    }

    #[staticmethod]
    fn from_string(source: String) -> Self {
        PyQis {
            inner: Qis::from_string(source),
        }
    }

    fn source(&self) -> String {
        self.inner.source().to_string()
    }

    #[staticmethod]
    fn preprocess_ir(llvm_ir: String) -> String {
        Qis::preprocess_ir(llvm_ir)
    }
}

#[pyclass(name = "Hugr", from_py_object)]
#[derive(Clone)]
pub struct PyHugr {
    pub(crate) inner: Hugr,
}

#[pymethods]
impl PyHugr {
    #[staticmethod]
    fn from_bytes(bytes: Vec<u8>) -> Self {
        PyHugr {
            inner: Hugr::from_bytes(bytes),
        }
    }

    /// Get the HUGR bytes
    fn to_bytes(&self) -> Vec<u8> {
        self.inner.hugr.clone()
    }
}

#[pyclass(name = "PhirJson", from_py_object)]
#[derive(Clone)]
pub struct PyPhirJson {
    pub(crate) inner: PhirJson,
}

#[pymethods]
impl PyPhirJson {
    #[staticmethod]
    fn from_string(source: String) -> Self {
        PyPhirJson {
            inner: PhirJson::from_string(source),
        }
    }

    #[staticmethod]
    fn from_json(source: String) -> Self {
        PyPhirJson {
            inner: PhirJson::from_json(source),
        }
    }
}

/// Create a QASM engine builder
#[pyfunction]
pub fn qasm_engine() -> PyQasmEngineBuilder {
    PyQasmEngineBuilder {
        inner: pecos_qasm::qasm_engine(),
    }
}

/// Create a QIS Engine builder (unified QIS/HUGR engine)
#[pyfunction]
pub fn qis_engine() -> PyQisEngineBuilder {
    PyQisEngineBuilder {
        inner: pecos_qis::qis_engine(),
        runtime_configured: false,
    }
}

/// Create a Selene-backed QIS Control Engine builder.
#[pyfunction]
#[pyo3(signature = (runtime_name = None))]
pub fn selene_engine(runtime_name: Option<&str>) -> PyResult<PyQisEngineBuilder> {
    let runtime = match runtime_name {
        None | Some("selene_simple_runtime") => pecos_qis::selene_simple_runtime(),
        Some(name) => pecos_qis::selene_runtime_auto(name),
    }
    .map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Failed to load Selene runtime: {e}"
        ))
    })?;
    Ok(PyQisEngineBuilder {
        inner: pecos_qis::qis_engine().runtime(runtime),
        runtime_configured: true,
    })
}

/// Create a PHIR JSON engine builder
#[pyfunction]
pub fn phir_json_engine() -> PyPhirJsonEngineBuilder {
    PyPhirJsonEngineBuilder {
        inner: pecos_phir_json::phir_json_engine(),
    }
}

/// Create a PHIR engine builder (PHIR Module execution)
#[pyfunction]
pub fn phir_engine() -> PyPhirEngineBuilder {
    PyPhirEngineBuilder {
        inner: pecos_phir::phir_engine(),
    }
}

/// Create a HUGR engine builder (direct HUGR interpreter)
///
/// This creates a builder for the direct HUGR interpreter engine,
/// which executes HUGR programs without LLVM compilation.
/// This is useful for testing and for simple circuits.
#[pyfunction]
pub fn hugr_engine() -> PyHugrEngineBuilder {
    PyHugrEngineBuilder::new()
}

/// Create a general noise model builder with no-effect defaults.
///
/// Call ``.auto()`` to opt into the legacy demonstration preset.
#[pyfunction]
pub fn general_noise() -> PyGeneralNoiseModelBuilder {
    PyGeneralNoiseModelBuilder::new()
}

/// Create a depolarizing noise model builder
#[pyfunction]
pub fn depolarizing_noise() -> PyDepolarizingNoiseModelBuilder {
    PyDepolarizingNoiseModelBuilder::new()
}

/// Create a biased depolarizing noise model builder
#[pyfunction]
pub fn biased_depolarizing_noise() -> PyBiasedDepolarizingNoiseModelBuilder {
    PyBiasedDepolarizingNoiseModelBuilder::new()
}

/// A validated relative distribution of stochastic Pauli-plus-leakage events.
#[pyclass(name = "PauliLeakageDict", from_py_object)]
#[derive(Clone)]
pub struct PyPauliLeakageDict {
    inner: PauliLeakageDict,
}

impl PyPauliLeakageDict {
    fn extract(value: &Bound<'_, PyAny>) -> PyResult<PauliLeakageDict> {
        if let Ok(events) = value.extract::<PyRef<'_, Self>>() {
            return Ok(events.inner.clone());
        }
        let events = value.extract::<BTreeMap<String, f64>>().map_err(|_| {
            PyTypeError::new_err("events must be a PauliLeakageDict or dict[str, float]")
        })?;
        Ok(PauliLeakageDict::new(&events))
    }
}

#[pymethods]
impl PyPauliLeakageDict {
    #[new]
    fn new(events: BTreeMap<String, f64>) -> Self {
        Self {
            inner: PauliLeakageDict::new(&events),
        }
    }

    #[getter]
    fn arity(&self) -> usize {
        self.inner.arity()
    }

    #[getter]
    fn events(&self) -> BTreeMap<String, f64> {
        self.inner.events().clone()
    }

    fn to_dict(&self) -> BTreeMap<String, f64> {
        self.inner.events().clone()
    }

    fn __len__(&self) -> usize {
        self.inner.events().len()
    }

    fn __getitem__(&self, event: &str) -> PyResult<f64> {
        self.inner
            .events()
            .get(event)
            .copied()
            .ok_or_else(|| PyKeyError::new_err(event.to_string()))
    }

    fn __contains__(&self, event: &str) -> bool {
        self.inner.events().contains_key(event)
    }

    fn get(&self, event: &str) -> Option<f64> {
        self.inner.events().get(event).copied()
    }

    fn keys(&self) -> Vec<String> {
        self.inner.events().keys().cloned().collect()
    }

    fn items(&self) -> Vec<(String, f64)> {
        self.inner
            .events()
            .iter()
            .map(|(event, probability)| (event.clone(), *probability))
            .collect()
    }

    fn values(&self) -> Vec<f64> {
        self.inner.events().values().copied().collect()
    }

    /// Tensor/Kronecker product; equivalent to ``self * other``.
    fn tensor(&self, other: &Self) -> Self {
        Self {
            inner: self.inner.tensor(&other.inner),
        }
    }

    fn __mul__(&self, other: &Self) -> Self {
        self.tensor(other)
    }

    fn __repr__(&self) -> String {
        format!(
            "PauliLeakageDict(arity={}, events={:?})",
            self.inner.arity(),
            self.inner.events()
        )
    }
}

/// A stochastic single-qubit Pauli-plus-leakage channel.
#[pyclass(name = "PauliLeakageChannel", from_py_object)]
#[derive(Clone)]
pub struct PyPauliLeakageChannel {
    inner: PauliLeakageChannel,
}

#[pymethods]
impl PyPauliLeakageChannel {
    #[new]
    fn new(probability: f64, events: &Bound<'_, PyAny>) -> PyResult<Self> {
        let events = PyPauliLeakageDict::extract(events)?;
        Ok(Self {
            inner: PauliLeakageChannel::from_event_dict(probability, &events),
        })
    }

    #[getter]
    fn probability(&self) -> f64 {
        self.inner.probability()
    }

    #[getter]
    fn events(&self) -> BTreeMap<String, f64> {
        self.inner.events().clone()
    }

    #[getter]
    fn event_dict(&self) -> PyPauliLeakageDict {
        PyPauliLeakageDict {
            inner: self.inner.event_dict().clone(),
        }
    }

    fn __mul__(&self, other: &Self) -> PyP2PauliLeakageStep {
        PyP2PauliLeakageStep {
            inner: P2PauliLeakageStep::tensor_product(self.inner.clone(), other.inner.clone()),
        }
    }
}

/// A stochastic joint two-qubit Pauli-plus-leakage channel.
#[pyclass(name = "TwoQubitPauliLeakageChannel", from_py_object)]
#[derive(Clone)]
pub struct PyTwoQubitPauliLeakageChannel {
    inner: TwoQubitPauliLeakageChannel,
}

#[pymethods]
impl PyTwoQubitPauliLeakageChannel {
    #[new]
    fn new(probability: f64, events: &Bound<'_, PyAny>) -> PyResult<Self> {
        let events = PyPauliLeakageDict::extract(events)?;
        Ok(Self {
            inner: TwoQubitPauliLeakageChannel::from_event_dict(probability, &events),
        })
    }

    #[getter]
    fn probability(&self) -> f64 {
        self.inner.probability()
    }

    #[getter]
    fn events(&self) -> BTreeMap<String, f64> {
        self.inner.events().clone()
    }

    #[getter]
    fn event_dict(&self) -> PyPauliLeakageDict {
        PyPauliLeakageDict {
            inner: self.inner.event_dict().clone(),
        }
    }
}

/// One independent-leg or joint Pauli-plus-leakage step at a two-qubit hook.
#[pyclass(name = "P2PauliLeakageStep", from_py_object)]
#[derive(Clone)]
pub struct PyP2PauliLeakageStep {
    inner: P2PauliLeakageStep,
}

#[pymethods]
impl PyP2PauliLeakageStep {
    #[staticmethod]
    fn independent(first: &PyPauliLeakageChannel, second: &PyPauliLeakageChannel) -> Self {
        Self {
            inner: P2PauliLeakageStep::independent(first.inner.clone(), second.inner.clone()),
        }
    }

    #[staticmethod]
    fn tensor_product(first: &PyPauliLeakageChannel, second: &PyPauliLeakageChannel) -> Self {
        Self {
            inner: P2PauliLeakageStep::tensor_product(first.inner.clone(), second.inner.clone()),
        }
    }

    #[staticmethod]
    fn same_on_each(channel: &PyPauliLeakageChannel) -> Self {
        Self {
            inner: P2PauliLeakageStep::same_on_each(channel.inner.clone()),
        }
    }

    #[staticmethod]
    fn joint(channel: &PyTwoQubitPauliLeakageChannel) -> Self {
        Self {
            inner: P2PauliLeakageStep::joint(channel.inner.clone()),
        }
    }
}

/// A validated conditional-transition mapping over strings in ``{"0", "1", "L"}^arity``.
#[pyclass(name = "TransitionDict", from_py_object)]
#[derive(Clone)]
pub struct PyTransitionDict {
    inner: TransitionDict,
}

impl PyTransitionDict {
    fn extract(value: &Bound<'_, PyAny>) -> PyResult<TransitionDict> {
        if let Ok(transitions) = value.extract::<PyRef<'_, Self>>() {
            return Ok(transitions.inner.clone());
        }
        let transitions = value
            .extract::<BTreeMap<String, BTreeMap<String, f64>>>()
            .map_err(|_| {
                PyTypeError::new_err(
                    "transitions must be a TransitionDict or dict[str, dict[str, float]]",
                )
            })?;
        Ok(TransitionDict::new(&transitions))
    }
}

#[pymethods]
impl PyTransitionDict {
    #[new]
    fn new(transitions: BTreeMap<String, BTreeMap<String, f64>>) -> Self {
        Self {
            inner: TransitionDict::new(&transitions),
        }
    }

    /// Number of effective qutrits represented by each state label.
    #[getter]
    fn arity(&self) -> usize {
        self.inner.arity()
    }

    /// Return a plain nested dictionary copy.
    #[getter]
    fn transitions(&self) -> BTreeMap<String, BTreeMap<String, f64>> {
        self.inner.transitions().clone()
    }

    fn to_dict(&self) -> BTreeMap<String, BTreeMap<String, f64>> {
        self.inner.transitions().clone()
    }

    fn __len__(&self) -> usize {
        self.inner.transitions().len()
    }

    fn __getitem__(&self, source: &str) -> PyResult<BTreeMap<String, f64>> {
        self.inner
            .transitions()
            .get(source)
            .cloned()
            .ok_or_else(|| PyKeyError::new_err(source.to_string()))
    }

    fn __contains__(&self, source: &str) -> bool {
        self.inner.transitions().contains_key(source)
    }

    fn get(&self, source: &str) -> Option<BTreeMap<String, f64>> {
        self.inner.transitions().get(source).cloned()
    }

    fn keys(&self) -> Vec<String> {
        self.inner.transitions().keys().cloned().collect()
    }

    fn items(&self) -> Vec<(String, BTreeMap<String, f64>)> {
        self.inner
            .transitions()
            .iter()
            .map(|(source, row)| (source.clone(), row.clone()))
            .collect()
    }

    fn values(&self) -> Vec<BTreeMap<String, f64>> {
        self.inner.transitions().values().cloned().collect()
    }

    /// Tensor/Kronecker product; equivalent to ``self * other``.
    fn tensor(&self, other: &Self) -> Self {
        Self {
            inner: self.inner.tensor(&other.inner),
        }
    }

    /// Sequential composition, applying ``before`` first and ``self`` second.
    fn compose(&self, before: &Self) -> Self {
        Self {
            inner: self.inner.compose(&before.inner),
        }
    }

    /// Sequential composition, applying ``self`` first and ``next`` second.
    fn then(&self, next: &Self) -> Self {
        Self {
            inner: self.inner.then(&next.inner),
        }
    }

    fn __mul__(&self, other: &Self) -> Self {
        self.tensor(other)
    }

    fn __matmul__(&self, before: &Self) -> Self {
        self.compose(before)
    }

    fn __repr__(&self) -> String {
        format!(
            "TransitionDict(arity={}, transitions={:?})",
            self.inner.arity(),
            self.inner.transitions()
        )
    }
}

/// A conditional population-transition channel on ``{"0", "1", "L"}``.
#[pyclass(name = "TransitionChannel", from_py_object)]
#[derive(Clone)]
pub struct PyTransitionChannel {
    inner: QubitTransitionChannel,
}

#[pymethods]
impl PyTransitionChannel {
    /// Construct ``transitions[source][destination] = P(destination | source)``.
    #[new]
    fn new(probability: f64, transitions: &Bound<'_, PyAny>) -> PyResult<Self> {
        let transitions = PyTransitionDict::extract(transitions)?;
        Ok(Self {
            inner: QubitTransitionChannel::from_transition_dict(probability, &transitions),
        })
    }

    /// Recover ``L`` to ``0`` with ``p_zero`` and to ``1`` otherwise.
    #[staticmethod]
    #[pyo3(signature = (probability, p_zero=0.5))]
    fn leak_recovery(probability: f64, p_zero: f64) -> Self {
        Self {
            inner: QubitTransitionChannel::leak_recovery(probability, p_zero),
        }
    }

    #[getter]
    fn probability(&self) -> f64 {
        self.inner.probability()
    }

    #[getter]
    fn transitions(&self) -> BTreeMap<String, BTreeMap<String, f64>> {
        self.inner.transitions().clone()
    }

    #[getter]
    fn transition_dict(&self) -> PyTransitionDict {
        PyTransitionDict {
            inner: self.inner.transition_dict().clone(),
        }
    }

    /// Build an independent two-leg step, preserving each channel's outer probability.
    fn __mul__(&self, other: &Self) -> PyP2TransitionStep {
        PyP2TransitionStep {
            inner: P2TransitionStep::tensor_product(self.inner.clone(), other.inner.clone()),
        }
    }
}

/// A joint conditional population-transition channel on two effective qutrits.
#[pyclass(name = "TwoQubitTransitionChannel", from_py_object)]
#[derive(Clone)]
pub struct PyTwoQubitTransitionChannel {
    inner: TwoQubitTransitionChannel,
}

#[pymethods]
impl PyTwoQubitTransitionChannel {
    /// Construct ``transitions[xy][wz] = P(wz | xy)`` for symbols in ``{"0", "1", "L"}``.
    #[new]
    fn new(probability: f64, transitions: &Bound<'_, PyAny>) -> PyResult<Self> {
        let transitions = PyTransitionDict::extract(transitions)?;
        Ok(Self {
            inner: TwoQubitTransitionChannel::from_transition_dict(probability, &transitions),
        })
    }

    #[getter]
    fn probability(&self) -> f64 {
        self.inner.probability()
    }

    #[getter]
    fn transitions(&self) -> BTreeMap<String, BTreeMap<String, f64>> {
        self.inner.transitions().clone()
    }

    #[getter]
    fn transition_dict(&self) -> PyTransitionDict {
        PyTransitionDict {
            inner: self.inner.transition_dict().clone(),
        }
    }
}

/// One ordered transition step at a two-qubit gate hook.
#[pyclass(name = "P2TransitionStep", from_py_object)]
#[derive(Clone)]
pub struct PyP2TransitionStep {
    inner: P2TransitionStep,
}

#[pymethods]
impl PyP2TransitionStep {
    /// Apply potentially different single-qubit channels independently to the two gate legs.
    #[staticmethod]
    fn independent(first: &PyTransitionChannel, second: &PyTransitionChannel) -> Self {
        Self {
            inner: P2TransitionStep::independent(first.inner.clone(), second.inner.clone()),
        }
    }

    /// Alias for ``independent`` emphasizing the product of two one-qubit channels.
    #[staticmethod]
    fn tensor_product(first: &PyTransitionChannel, second: &PyTransitionChannel) -> Self {
        Self {
            inner: P2TransitionStep::tensor_product(first.inner.clone(), second.inner.clone()),
        }
    }

    /// Apply the same single-qubit channel independently to both gate legs.
    #[staticmethod]
    fn same_on_each(channel: &PyTransitionChannel) -> Self {
        Self {
            inner: P2TransitionStep::same_on_each(channel.inner.clone()),
        }
    }

    /// Apply one correlated transition matrix to the joint two-qutrit basis state.
    #[staticmethod]
    fn joint(channel: &PyTwoQubitTransitionChannel) -> Self {
        Self {
            inner: P2TransitionStep::joint(channel.inner.clone()),
        }
    }
}

fn extract_p2_transition_channel(channel: &Bound<'_, PyAny>) -> PyResult<P2TransitionStep> {
    if let Ok(channel) = channel.extract::<PyRef<'_, PyTransitionChannel>>() {
        return Ok(P2TransitionStep::same_on_each(channel.inner.clone()));
    }
    if let Ok(channel) = channel.extract::<PyRef<'_, PyTwoQubitTransitionChannel>>() {
        return Ok(P2TransitionStep::joint(channel.inner.clone()));
    }
    Err(PyTypeError::new_err(
        "channel must be a TransitionChannel or TwoQubitTransitionChannel",
    ))
}

/// Python wrapper for `GeneralNoiseModelBuilder`
#[pyclass(name = "GeneralNoiseModelBuilder", from_py_object)]
#[derive(Clone)]
pub struct PyGeneralNoiseModelBuilder {
    pub(crate) inner: GeneralNoiseModelBuilder,
}

impl PyGeneralNoiseModelBuilder {
    pub(crate) fn validated_inner(&self) -> PyResult<GeneralNoiseModelBuilder> {
        self.inner
            .validate_configuration()
            .map_err(|message| pyo3::exceptions::PyValueError::new_err(message.to_string()))?;
        Ok(self.inner.clone())
    }
}

#[pymethods]
impl PyGeneralNoiseModelBuilder {
    #[new]
    fn new() -> Self {
        Self {
            inner: GeneralNoiseModelBuilder::new(),
        }
    }

    /// Fill unset parameters with the legacy demonstration preset.
    ///
    /// This reproduces the general noise model's historical defaults for demonstrations; it is
    /// not a calibrated device model. Explicit setters win in either call order because ``auto``
    /// fills only parameters that the caller has not set.
    ///
    /// The preset sets preparation, measurement, one-qubit, two-qubit, and linear-idle rates to
    /// 0.01, 0.01/0.01, 0.001, 0.01, and 0.001 respectively. In addition:
    ///
    /// * ``p_prep_leak_ratio = 0.5`` means half of preparation faults leak the qubit out of the
    ///   computational subspace.
    /// * ``p1_emission_ratio = p2_emission_ratio = 0.5`` means half of gate errors take the
    ///   spontaneous-emission branch, which removes the original gate and substitutes a sample
    ///   from the emission model. The preset emission models contain Pauli keys only, so these
    ///   branches cause no leakage.
    /// * ``p1_seepage_prob = p2_seepage_prob = 0.5`` applies only to qubits that are already
    ///   leaked.
    fn auto(&self) -> Self {
        Self {
            inner: self.inner.clone().auto(),
        }
    }

    /// Set single-qubit gate error probability
    fn with_p1(&self, p: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_p1(p),
        })
    }

    /// Set two-qubit gate error probability
    fn with_p2(&self, p: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_p2(p),
        })
    }

    /// Set preparation error probability
    fn with_p_prep(&self, p: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_p_prep(p),
        })
    }

    /// Set measurement error probability for |0⟩ state
    fn with_p_meas_0(&self, p: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_p_meas_0(p),
        })
    }

    /// Set measurement error probability for |1⟩ state
    fn with_p_meas_1(&self, p: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_p_meas_1(p),
        })
    }

    /// Set seed for reproducibility
    fn with_seed(&self, seed: u64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_seed(seed),
        })
    }

    /// Set global scale factor
    fn with_scale(&self, scale: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_scale(scale),
        })
    }

    /// Set leakage scale factor
    fn with_leakage_scale(&self, scale: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_leakage_scale(scale),
        })
    }

    /// Set emission scale factor
    fn with_emission_scale(&self, scale: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_emission_scale(scale),
        })
    }

    /// Set single-qubit Pauli error model
    fn with_p1_pauli_model(
        &self,
        model: std::collections::BTreeMap<String, f64>,
    ) -> PyResult<Self> {
        use std::collections::BTreeMap;
        let btree_map: BTreeMap<String, f64> = model.into_iter().collect();
        Ok(Self {
            inner: self.inner.clone().with_p1_pauli_model(&btree_map),
        })
    }

    /// Replace the ordered Pauli-plus-leakage stack before single-qubit gates.
    fn with_p1_pauli_leakage_channels_before_gate(
        &self,
        channels: Vec<PyPauliLeakageChannel>,
    ) -> PyResult<Self> {
        let channels = channels
            .into_iter()
            .map(|channel| channel.inner)
            .collect::<Vec<_>>();
        Ok(Self {
            inner: self
                .inner
                .clone()
                .with_p1_pauli_leakage_channels_before_gate(&channels),
        })
    }

    /// Append one Pauli-plus-leakage channel before single-qubit gates.
    fn add_p1_pauli_leakage_channel_before_gate(
        &self,
        channel: &PyPauliLeakageChannel,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: self
                .inner
                .clone()
                .add_p1_pauli_leakage_channel_before_gate(&channel.inner),
        })
    }

    /// Replace the ordered Pauli-plus-leakage stack after single-qubit noise sites.
    fn with_p1_pauli_leakage_channels_after_gate(
        &self,
        channels: Vec<PyPauliLeakageChannel>,
    ) -> PyResult<Self> {
        let channels = channels
            .into_iter()
            .map(|channel| channel.inner)
            .collect::<Vec<_>>();
        Ok(Self {
            inner: self
                .inner
                .clone()
                .with_p1_pauli_leakage_channels_after_gate(&channels),
        })
    }

    /// Append one Pauli-plus-leakage channel after single-qubit noise sites.
    fn add_p1_pauli_leakage_channel_after_gate(
        &self,
        channel: &PyPauliLeakageChannel,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: self
                .inner
                .clone()
                .add_p1_pauli_leakage_channel_after_gate(&channel.inner),
        })
    }

    /// Replace the ordered transition-channel stack before each single-qubit gate.
    fn with_p1_transition_channels_before_gate(
        &self,
        channels: Vec<PyTransitionChannel>,
    ) -> PyResult<Self> {
        let channels = channels
            .into_iter()
            .map(|channel| channel.inner)
            .collect::<Vec<_>>();
        Ok(Self {
            inner: self
                .inner
                .clone()
                .with_p1_transition_channels_before_gate(&channels),
        })
    }

    /// Append one transition channel before each single-qubit gate.
    fn add_p1_transition_channel_before_gate(
        &self,
        channel: &PyTransitionChannel,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: self
                .inner
                .clone()
                .add_p1_transition_channel_before_gate(&channel.inner),
        })
    }

    /// Replace the ordered transition-channel stack after each single-qubit noise site.
    fn with_p1_transition_channels_after_gate(
        &self,
        channels: Vec<PyTransitionChannel>,
    ) -> PyResult<Self> {
        let channels = channels
            .into_iter()
            .map(|channel| channel.inner)
            .collect::<Vec<_>>();
        Ok(Self {
            inner: self
                .inner
                .clone()
                .with_p1_transition_channels_after_gate(&channels),
        })
    }

    /// Append one transition channel after each single-qubit noise site.
    fn add_p1_transition_channel_after_gate(
        &self,
        channel: &PyTransitionChannel,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: self
                .inner
                .clone()
                .add_p1_transition_channel_after_gate(&channel.inner),
        })
    }

    /// Set two-qubit Pauli error model
    fn with_p2_pauli_model(
        &self,
        model: std::collections::BTreeMap<String, f64>,
    ) -> PyResult<Self> {
        use std::collections::BTreeMap;
        let btree_map: BTreeMap<String, f64> = model.into_iter().collect();
        Ok(Self {
            inner: self.inner.clone().with_p2_pauli_model(&btree_map),
        })
    }

    /// Replace the ordered Pauli-plus-leakage step stack before two-qubit gates.
    fn with_p2_pauli_leakage_steps_before_gate(
        &self,
        steps: Vec<PyP2PauliLeakageStep>,
    ) -> PyResult<Self> {
        let steps = steps.into_iter().map(|step| step.inner).collect::<Vec<_>>();
        Ok(Self {
            inner: self
                .inner
                .clone()
                .with_p2_pauli_leakage_steps_before_gate(&steps),
        })
    }

    /// Append one Pauli-plus-leakage step before two-qubit gates.
    fn add_p2_pauli_leakage_step_before_gate(&self, step: &PyP2PauliLeakageStep) -> PyResult<Self> {
        Ok(Self {
            inner: self
                .inner
                .clone()
                .add_p2_pauli_leakage_step_before_gate(&step.inner),
        })
    }

    /// Append the same independently drawn single-qubit channel before both gate legs.
    fn add_p2_pauli_leakage_channel_before_gate(
        &self,
        channel: &PyPauliLeakageChannel,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: self
                .inner
                .clone()
                .add_p2_pauli_leakage_channel_before_gate(&channel.inner),
        })
    }

    /// Replace the ordered Pauli-plus-leakage step stack after two-qubit noise sites.
    fn with_p2_pauli_leakage_steps_after_gate(
        &self,
        steps: Vec<PyP2PauliLeakageStep>,
    ) -> PyResult<Self> {
        let steps = steps.into_iter().map(|step| step.inner).collect::<Vec<_>>();
        Ok(Self {
            inner: self
                .inner
                .clone()
                .with_p2_pauli_leakage_steps_after_gate(&steps),
        })
    }

    /// Append one Pauli-plus-leakage step after two-qubit noise sites.
    fn add_p2_pauli_leakage_step_after_gate(&self, step: &PyP2PauliLeakageStep) -> PyResult<Self> {
        Ok(Self {
            inner: self
                .inner
                .clone()
                .add_p2_pauli_leakage_step_after_gate(&step.inner),
        })
    }

    /// Append the same independently drawn single-qubit channel after both gate legs.
    fn add_p2_pauli_leakage_channel_after_gate(
        &self,
        channel: &PyPauliLeakageChannel,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: self
                .inner
                .clone()
                .add_p2_pauli_leakage_channel_after_gate(&channel.inner),
        })
    }

    /// Replace the ordered transition-step stack before each two-qubit gate.
    fn with_p2_transition_steps_before_gate(
        &self,
        steps: Vec<PyP2TransitionStep>,
    ) -> PyResult<Self> {
        let steps = steps.into_iter().map(|step| step.inner).collect::<Vec<_>>();
        Ok(Self {
            inner: self
                .inner
                .clone()
                .with_p2_transition_steps_before_gate(&steps),
        })
    }

    /// Append an independent-leg or joint transition step before each two-qubit gate.
    fn add_p2_transition_step_before_gate(&self, step: &PyP2TransitionStep) -> PyResult<Self> {
        Ok(Self {
            inner: self
                .inner
                .clone()
                .add_p2_transition_step_before_gate(&step.inner),
        })
    }

    /// Append a single-qubit-per-leg or joint channel before each two-qubit gate.
    fn add_p2_transition_channel_before_gate(&self, channel: &Bound<'_, PyAny>) -> PyResult<Self> {
        let step = extract_p2_transition_channel(channel)?;
        Ok(Self {
            inner: self.inner.clone().add_p2_transition_step_before_gate(&step),
        })
    }

    /// Replace the ordered transition-step stack after each two-qubit noise site.
    fn with_p2_transition_steps_after_gate(
        &self,
        steps: Vec<PyP2TransitionStep>,
    ) -> PyResult<Self> {
        let steps = steps.into_iter().map(|step| step.inner).collect::<Vec<_>>();
        Ok(Self {
            inner: self
                .inner
                .clone()
                .with_p2_transition_steps_after_gate(&steps),
        })
    }

    /// Append an independent-leg or joint transition step after each two-qubit noise site.
    fn add_p2_transition_step_after_gate(&self, step: &PyP2TransitionStep) -> PyResult<Self> {
        Ok(Self {
            inner: self
                .inner
                .clone()
                .add_p2_transition_step_after_gate(&step.inner),
        })
    }

    /// Append a single-qubit-per-leg or joint channel after each two-qubit site.
    fn add_p2_transition_channel_after_gate(&self, channel: &Bound<'_, PyAny>) -> PyResult<Self> {
        let step = extract_p2_transition_channel(channel)?;
        Ok(Self {
            inner: self.inner.clone().add_p2_transition_step_after_gate(&step),
        })
    }

    /// Set average single-qubit gate error probability
    fn with_average_p1(&self, p: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_average_p1(p),
        })
    }

    /// Set average two-qubit gate error probability
    fn with_average_p2(&self, p: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_average_p2(p),
        })
    }

    /// Set measurement error probability (symmetric)
    fn with_p_meas(&self, p: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_p_meas(p),
        })
    }

    /// Set measurement error probability (asymmetric)
    fn with_measurement_probability(&self, p0: f64, p1: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_p_meas_0(p0).with_p_meas_1(p1),
        })
    }

    /// Add a noiseless gate
    fn with_noiseless_gate(&self, gate_name: &str) -> PyResult<Self> {
        // Make it case-insensitive
        let gate_type = match gate_name.to_uppercase().as_str() {
            "I" => GateType::I,
            "X" => GateType::X,
            "Y" => GateType::Y,
            "Z" => GateType::Z,
            "S" | "SZ" => GateType::SZ,       // S gate is SZ in GateType
            "SDG" | "SZDG" => GateType::SZdg, // S dagger
            "H" => GateType::H,
            "RX" => GateType::RX,
            "RY" => GateType::RY,
            "RZ" => GateType::RZ,
            "T" => GateType::T,
            "TDG" => GateType::Tdg,
            "U" => GateType::U,
            "R1XY" => GateType::R1XY,
            "CX" => GateType::CX,
            "SZZ" => GateType::SZZ,
            "SZZDG" => GateType::SZZdg,
            "RZZ" => GateType::RZZ,
            "MEASURE" => GateType::MZ,
            "PREP" => GateType::PZ,
            "IDLE" => GateType::Idle,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid gate type: {gate_name}"
                )));
            }
        };
        Ok(Self {
            inner: self.inner.clone().with_noiseless_gate(gate_type),
        })
    }

    /// Set seepage probability
    fn with_seepage_prob(&self, prob: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_seepage_prob(prob),
        })
    }

    /// Set the DEM-style linear idle-noise family.
    ///
    /// ``rate`` is the total event rate per time unit. The X/Y/Z/L ``model`` must be a
    /// normalized distribution because this family splits one total linear rate across axes.
    ///
    /// By contrast, ``with_p_idle_sin_squared`` uses radians per time unit and unnormalized
    /// relative multipliers because sine laws do not add linearly: each axis has its own
    /// independent rate. It applies no unit conversion.
    ///
    /// All engines idle-noise families are off by default, so translating a DEM configuration only
    /// requires setting the requested families. Engines keeps its existing linear sampling
    /// structure: one event followed by a categorical axis choice, versus the DEM's independent
    /// per-axis mechanisms. The difference is second order in the rates; this setter aligns units
    /// and the axis alphabet, not the sampling structure.
    fn with_p_idle_linear(
        &self,
        rate: f64,
        model: std::collections::BTreeMap<String, f64>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_p_idle_linear(rate, &model),
        })
    }

    /// Set the DEM-style stochastic sine-squared idle-noise family.
    ///
    /// ``rate`` is radians per time unit and no unit conversion is applied. For each X/Y/Z/L axis
    /// P, multiplier ``n_P``, and duration ``d``, engines independently samples
    /// ``P(P) = sin^2(rate * n_P * d)``.
    ///
    /// The model is intentionally unnormalized because sine laws do not add linearly: each axis
    /// carries its own independent rate. ``with_p_idle_linear`` instead requires a normalized
    /// distribution because it splits one total linear rate across axes.
    ///
    /// The removed cycles-per-time spelling migrates exactly as follows at its former default
    /// factor of one:
    ///
    /// ``with_p_idle_quadratic_rate(r)  ==  with_p_idle_sin_squared(r * PI, {"Z": 1.0})``
    ///
    /// All engines idle-noise families are off by default, so translating a DEM configuration
    /// only requires setting the requested families.
    ///
    /// Engines deliberately retains its existing linear sampling structure: one event followed
    /// by a categorical axis choice, versus the DEM's independent per-axis mechanisms. The
    /// difference is second order in the rates; this setter aligns units and the axis alphabet,
    /// not the sampling structure.
    fn with_p_idle_sin_squared(
        &self,
        rate: f64,
        model: std::collections::BTreeMap<String, f64>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_p_idle_sin_squared(rate, &model),
        })
    }

    /// Set the DEM-style coherent idle-noise family.
    ///
    /// ``rate`` is radians per time unit and no unit conversion is applied. For each RX/RY/RZ
    /// generator P, multiplier ``n_P``, and duration ``d``, engines deterministically applies a
    /// rotation with angle ``rate * n_P * d``. Coherent evolution is not sampled and consumes no
    /// random draw.
    ///
    /// The model is intentionally unnormalized because its values are relative rate multipliers,
    /// not probabilities to be split from one total event rate. It defaults to
    /// ``{"RX": 1.0, "RY": 1.0, "RZ": 1.0}``. Leakage and all other keys are rejected because
    /// leakage is not a rotation.
    ///
    /// Consumption is consumer-dependent: the standard DEM builder rejects coherent idle noise;
    /// the EEG route in ``exp/pecos-eeg`` represents it with an RZ generator; and a simulator
    /// applies it only when its rotation executor is installed. PECOS #437 documents how a
    /// missing executor could otherwise silently drop it.
    #[pyo3(signature = (rate, model=None))]
    fn with_p_idle_coherent(
        &self,
        rate: &Bound<'_, PyAny>,
        model: Option<std::collections::BTreeMap<String, f64>>,
    ) -> PyResult<Self> {
        if rate.is_instance_of::<PyBool>() {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "coherent idling rate must be a finite, non-negative float, not bool",
            ));
        }
        let rate = rate.extract::<f64>().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(
                "coherent idling rate must be a finite, non-negative float",
            )
        })?;
        let model = model.unwrap_or_else(|| {
            std::collections::BTreeMap::from([
                ("RX".to_string(), 1.0),
                ("RY".to_string(), 1.0),
                ("RZ".to_string(), 1.0),
            ])
        });
        Ok(Self {
            inner: self.inner.clone().with_p_idle_coherent(rate, &model),
        })
    }

    /// Set idle scale factor
    fn with_idle_scale(&self, scale: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_idle_scale(scale),
        })
    }

    /// Set the preparation leakage ratio
    fn with_prep_leak_ratio(&self, ratio: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_prep_leak_ratio(ratio),
        })
    }

    /// Set the probability of crosstalk during initialization operations
    fn with_p_prep_crosstalk(&self, prob: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_p_prep_crosstalk(prob),
        })
    }

    /// Set the scaling factor for initialization errors
    fn with_prep_scale(&self, scale: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_prep_scale(scale),
        })
    }

    /// Set the scaling factor for initialization crosstalk probability
    fn with_p_prep_crosstalk_scale(&self, scale: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_p_prep_crosstalk_scale(scale),
        })
    }

    /// Set the emission-to-absorption ratio for single-qubit gates
    fn with_p1_emission_ratio(&self, ratio: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_p1_emission_ratio(ratio),
        })
    }

    /// Set the emission model for single-qubit gates
    fn with_p1_emission_model(
        &self,
        model: std::collections::BTreeMap<String, f64>,
    ) -> PyResult<Self> {
        use std::collections::BTreeMap;
        let btree_map: BTreeMap<String, f64> = model.into_iter().collect();
        Ok(Self {
            inner: self.inner.clone().with_p1_emission_model(&btree_map),
        })
    }

    /// Set the seepage probability for single-qubit gates
    fn with_p1_seepage_prob(&self, prob: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_p1_seepage_prob(prob),
        })
    }

    /// Set the scaling factor for single-qubit gate errors
    fn with_p1_scale(&self, scale: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_p1_scale(scale),
        })
    }

    /// Set angle-dependent parameters for two-qubit gates
    fn with_p2_angle_params(&self, a: f64, b: f64, c: f64, d: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_p2_angle_params(a, b, c, d),
        })
    }

    /// Set angle-dependent power for two-qubit gates
    fn with_p2_angle_power(&self, power: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_p2_angle_power(power),
        })
    }

    /// Set the emission-to-absorption ratio for two-qubit gates
    fn with_p2_emission_ratio(&self, ratio: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_p2_emission_ratio(ratio),
        })
    }

    /// Set the emission model for two-qubit gates
    fn with_p2_emission_model(
        &self,
        model: std::collections::BTreeMap<String, f64>,
    ) -> PyResult<Self> {
        use std::collections::BTreeMap;
        let btree_map: BTreeMap<String, f64> = model.into_iter().collect();
        Ok(Self {
            inner: self.inner.clone().with_p2_emission_model(&btree_map),
        })
    }

    /// Set the seepage probability for two-qubit gates
    fn with_p2_seepage_prob(&self, prob: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_p2_seepage_prob(prob),
        })
    }

    /// Set the duration of the idle-noise site applied to each qubit after a two-qubit gate.
    ///
    /// A duration of `0.0` disables these sites. Nonzero sites receive all configured idle
    /// families over the given duration: linear stochastic noise, independent per-axis
    /// sine-squared noise, and coherent rotations.
    ///
    /// Anyone who previously wrote `with_p2_idle(0.01)` and no linear rate now gets no after-2q
    /// idle noise; the equivalent is
    /// `with_p_idle_linear(0.01, {"Z": 1.0}).with_idle_after_2q(1.0)`.
    fn with_idle_after_2q(&self, duration: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_idle_after_2q(duration),
        })
    }

    /// Set the scaling factor for two-qubit gate errors
    fn with_p2_scale(&self, scale: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_p2_scale(scale),
        })
    }

    /// Set the probability of crosstalk during measurement operations
    fn with_p_meas_crosstalk(&self, prob: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_p_meas_crosstalk(prob),
        })
    }

    /// Set the probability of global crosstalk during measurement operations
    fn with_p_meas_crosstalk_global(&self, prob: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_p_meas_crosstalk_global(prob),
        })
    }

    /// Set the probability of local crosstalk during measurement operations
    fn with_p_meas_crosstalk_local(&self, prob: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_p_meas_crosstalk_local(prob),
        })
    }

    /// Set the transition model for measurement crosstalk
    fn with_p_meas_crosstalk_model(
        &self,
        model: std::collections::BTreeMap<String, f64>,
    ) -> PyResult<Self> {
        use std::collections::BTreeMap;
        let btree_map: BTreeMap<String, f64> = model.into_iter().collect();
        Ok(Self {
            inner: self.inner.clone().with_p_meas_crosstalk_model(&btree_map),
        })
    }

    /// Set the scaling factor for measurement errors
    fn with_meas_scale(&self, scale: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_meas_scale(scale),
        })
    }

    /// Set the scaling factor for measurement crosstalk probability
    fn with_p_meas_crosstalk_scale(&self, scale: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_p_meas_crosstalk_scale(scale),
        })
    }
}

/// Python wrapper for `DepolarizingNoiseModelBuilder`
#[pyclass(name = "DepolarizingNoiseModelBuilder", from_py_object)]
#[derive(Clone)]
pub struct PyDepolarizingNoiseModelBuilder {
    pub(crate) inner: DepolarizingNoiseModelBuilder,
}

#[pymethods]
impl PyDepolarizingNoiseModelBuilder {
    #[new]
    fn new() -> Self {
        Self {
            inner: DepolarizingNoiseModelBuilder::new(),
        }
    }

    /// Set preparation error probability
    fn with_p_prep(&self, p: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_p_prep(p),
        })
    }

    /// Set measurement error probability
    fn with_p_meas(&self, p: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_p_meas(p),
        })
    }

    /// Set single-qubit gate error probability
    fn with_p1(&self, p: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_p1(p),
        })
    }

    /// Set two-qubit gate error probability
    fn with_p2(&self, p: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_p2(p),
        })
    }

    /// Set uniform probability for all error types
    fn with_uniform_probability(&self, p: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_uniform_probability(p),
        })
    }

    /// Set seed for reproducibility
    fn with_seed(&self, seed: u64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_seed(seed),
        })
    }
}

/// Python wrapper for `BiasedDepolarizingNoiseModelBuilder`
#[pyclass(name = "BiasedDepolarizingNoiseModelBuilder", from_py_object)]
#[derive(Clone)]
pub struct PyBiasedDepolarizingNoiseModelBuilder {
    pub(crate) inner: BiasedDepolarizingNoiseModelBuilder,
}

#[pymethods]
impl PyBiasedDepolarizingNoiseModelBuilder {
    #[new]
    fn new() -> Self {
        Self {
            inner: BiasedDepolarizingNoiseModelBuilder::new(),
        }
    }

    /// Set preparation error probability
    fn with_p_prep(&self, p: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_p_prep(p),
        })
    }

    /// Set measurement 0->1 flip probability
    fn with_p_meas_0(&self, p: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_p_meas_0(p),
        })
    }

    /// Set measurement 1->0 flip probability
    fn with_p_meas_1(&self, p: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_p_meas_1(p),
        })
    }

    /// Set single-qubit gate error probability
    fn with_p1(&self, p: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_p1(p),
        })
    }

    /// Set two-qubit gate error probability
    fn with_p2(&self, p: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_p2(p),
        })
    }

    /// Set uniform probability for all error types
    fn with_uniform_probability(&self, p: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_uniform_probability(p),
        })
    }

    /// Set seed for reproducibility
    fn with_seed(&self, seed: u64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_seed(seed),
        })
    }
}

/// Python wrapper for `StateVectorEngineBuilder`
#[pyclass(name = "StateVectorEngineBuilder", from_py_object)]
#[derive(Clone)]
pub struct PyStateVectorEngineBuilder {
    pub(crate) inner: Option<RustStateVectorEngineBuilder>,
}

#[pymethods]
impl PyStateVectorEngineBuilder {
    #[new]
    fn new() -> Self {
        Self {
            inner: Some(pecos_engines::state_vector()),
        }
    }

    /// Set the number of qubits
    fn qubits(slf: Py<Self>, num_qubits: usize, py: Python) -> PyResult<Py<Self>> {
        let mut borrowed = slf.borrow_mut(py);
        if let Some(inner) = borrowed.inner.take() {
            borrowed.inner = Some(inner.qubits(num_qubits));
            drop(borrowed);
            Ok(slf)
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Builder has already been consumed",
            ))
        }
    }
}

/// Python wrapper for `SparseStabEngineBuilder`
#[pyclass(name = "SparseStabEngineBuilder", from_py_object)]
#[derive(Clone)]
pub struct PySparseStabEngineBuilder {
    pub(crate) inner: Option<RustSparseStabEngineBuilder>,
}

#[pymethods]
impl PySparseStabEngineBuilder {
    #[new]
    fn new() -> Self {
        Self {
            inner: Some(pecos_engines::sparse_stab()),
        }
    }

    /// Set the number of qubits
    fn qubits(slf: Py<Self>, num_qubits: usize, py: Python) -> PyResult<Py<Self>> {
        let mut borrowed = slf.borrow_mut(py);
        if let Some(inner) = borrowed.inner.take() {
            borrowed.inner = Some(inner.qubits(num_qubits));
            drop(borrowed);
            Ok(slf)
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Builder has already been consumed",
            ))
        }
    }
}

/// Create a state vector quantum engine builder
#[pyfunction]
pub fn state_vector() -> PyStateVectorEngineBuilder {
    PyStateVectorEngineBuilder::new()
}

/// Create a sparse stabilizer quantum engine builder
#[pyfunction]
pub fn sparse_stab() -> PySparseStabEngineBuilder {
    PySparseStabEngineBuilder::new()
}

/// Python wrapper for `StabilizerEngineBuilder` (recommended stabilizer backend).
#[pyclass(name = "StabilizerEngineBuilder", from_py_object)]
#[derive(Clone)]
pub struct PyStabilizerEngineBuilder {
    pub(crate) inner: Option<RustStabilizerEngineBuilder>,
}

#[pymethods]
impl PyStabilizerEngineBuilder {
    #[new]
    fn new() -> Self {
        Self {
            inner: Some(pecos_engines::stabilizer()),
        }
    }

    /// Set the number of qubits
    fn qubits(slf: Py<Self>, num_qubits: usize, py: Python) -> PyResult<Py<Self>> {
        let mut borrowed = slf.borrow_mut(py);
        if let Some(inner) = borrowed.inner.take() {
            borrowed.inner = Some(inner.qubits(num_qubits));
            drop(borrowed);
            Ok(slf)
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Builder has already been consumed",
            ))
        }
    }
}

/// Create a stabilizer quantum engine builder (recommended).
#[pyfunction]
pub fn stabilizer() -> PyStabilizerEngineBuilder {
    PyStabilizerEngineBuilder::new()
}

/// Python wrapper for `StabVecEngineBuilder`
#[pyclass(name = "StabVecEngineBuilder", from_py_object)]
#[derive(Clone)]
pub struct PyStabVecEngineBuilder {
    pub(crate) inner: Option<RustStabVecEngineBuilder>,
}

#[pymethods]
impl PyStabVecEngineBuilder {
    #[new]
    fn new() -> Self {
        Self {
            inner: Some(pecos_engines::stab_vec()),
        }
    }

    /// Set the number of qubits
    fn qubits(slf: Py<Self>, num_qubits: usize, py: Python) -> PyResult<Py<Self>> {
        let mut borrowed = slf.borrow_mut(py);
        if let Some(inner) = borrowed.inner.take() {
            borrowed.inner = Some(inner.qubits(num_qubits));
            drop(borrowed);
            Ok(slf)
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Builder has already been consumed",
            ))
        }
    }
}

/// Create a Clifford+RZ quantum engine builder
#[pyfunction]
pub fn stab_vec() -> PyStabVecEngineBuilder {
    PyStabVecEngineBuilder::new()
}

/// Python wrapper for `DensityMatrixEngineBuilder`
#[pyclass(name = "DensityMatrixEngineBuilder", from_py_object)]
#[derive(Clone)]
pub struct PyDensityMatrixEngineBuilder {
    pub(crate) inner: Option<RustDensityMatrixEngineBuilder>,
}

#[pymethods]
impl PyDensityMatrixEngineBuilder {
    #[new]
    fn new() -> Self {
        Self {
            inner: Some(pecos_engines::density_matrix()),
        }
    }

    /// Set the number of qubits
    fn qubits(slf: Py<Self>, num_qubits: usize, py: Python) -> PyResult<Py<Self>> {
        let mut borrowed = slf.borrow_mut(py);
        if let Some(inner) = borrowed.inner.take() {
            borrowed.inner = Some(inner.qubits(num_qubits));
            drop(borrowed);
            Ok(slf)
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Builder has already been consumed",
            ))
        }
    }
}

/// Create a density matrix quantum engine builder
#[pyfunction]
pub fn density_matrix() -> PyDensityMatrixEngineBuilder {
    PyDensityMatrixEngineBuilder::new()
}

/// Python wrapper for `CoinTossEngineBuilder`
#[pyclass(name = "CoinTossEngineBuilder", from_py_object)]
#[derive(Clone)]
pub struct PyCoinTossEngineBuilder {
    pub(crate) inner: Option<RustCoinTossEngineBuilder>,
}

#[pymethods]
impl PyCoinTossEngineBuilder {
    #[new]
    fn new() -> Self {
        Self {
            inner: Some(pecos_engines::coin_toss()),
        }
    }

    fn qubits(slf: Py<Self>, num_qubits: usize, py: Python) -> PyResult<Py<Self>> {
        let mut borrowed = slf.borrow_mut(py);
        if let Some(inner) = borrowed.inner.take() {
            borrowed.inner = Some(inner.qubits(num_qubits));
            drop(borrowed);
            Ok(slf)
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Builder has already been consumed",
            ))
        }
    }
}

/// Create a coin toss quantum engine builder
#[pyfunction]
pub fn coin_toss() -> PyCoinTossEngineBuilder {
    PyCoinTossEngineBuilder::new()
}

/// Create a `SimBuilder` from scratch without a program
#[pyfunction]
pub fn sim_builder() -> PySimBuilder {
    PySimBuilder {
        inner: SimBuilderInner::Empty,
    }
}

/// Python wrapper for `QisInterfaceBuilder`
/// Since we can't directly expose trait objects to Python, we'll use an opaque wrapper
///
/// This is deprecated - interface builders have moved to implementation crates
#[pyclass(name = "QisInterfaceBuilder")]
pub struct PyQisInterfaceBuilder {
    // Store the actual Rust builder internally
    // Field is intentionally unused as this is a deprecated stub
    #[allow(dead_code)]
    inner: Box<dyn QisInterfaceBuilder>,
}

/// Create a Helios interface builder
#[pyfunction]
pub fn qis_helios_interface() -> PyResult<PyQisInterfaceBuilder> {
    // Use the Helios interface builder from pecos
    Ok(PyQisInterfaceBuilder {
        inner: Box::new(pecos_qis::helios_interface_builder()),
    })
}

/// Create a Selene Helios interface builder (alias for `qis_helios_interface`)
///
/// This is the reference implementation that uses the Selene compiler to compile
/// QIS programs to native code via the Helios interface.
#[pyfunction]
pub fn qis_selene_helios_interface() -> PyResult<PyQisInterfaceBuilder> {
    // Both qis_helios_interface and qis_selene_helios_interface use the same
    // Helios interface builder from pecos-qis
    qis_helios_interface()
}

/// Register the engine builder module with `PyO3`
pub fn register_engine_builders(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Engine builders
    m.add_class::<PyQasmEngineBuilder>()?;
    m.add_class::<PyQisEngineBuilder>()?;
    m.add_class::<PyPhirJsonEngineBuilder>()?;
    m.add_class::<PyPhirEngineBuilder>()?;
    m.add_class::<PyHugrEngineBuilder>()?;

    // Simulation builders are now handled by the unified PySimBuilder in sim.rs

    // Built simulations
    m.add_class::<PyQasmSimulation>()?;
    m.add_class::<PyPhirJsonSimulation>()?;
    m.add_class::<PyPhirSimulation>()?;
    m.add_class::<PyQisControlSimulation>()?;
    m.add_class::<PyHugrSimulation>()?;

    // Program types
    m.add_class::<PyQasm>()?;
    m.add_class::<PyHugr>()?;
    m.add_class::<PyPhirJson>()?;

    // Noise builders
    m.add_class::<PyPauliLeakageDict>()?;
    m.add_class::<PyPauliLeakageChannel>()?;
    m.add_class::<PyTwoQubitPauliLeakageChannel>()?;
    m.add_class::<PyP2PauliLeakageStep>()?;
    m.add_class::<PyTransitionDict>()?;
    m.add_class::<PyTransitionChannel>()?;
    m.add_class::<PyTwoQubitTransitionChannel>()?;
    m.add_class::<PyP2TransitionStep>()?;
    m.add_class::<PyGeneralNoiseModelBuilder>()?;
    m.add_class::<PyDepolarizingNoiseModelBuilder>()?;
    m.add_class::<PyBiasedDepolarizingNoiseModelBuilder>()?;

    // Quantum engine builders
    m.add_class::<PyStateVectorEngineBuilder>()?;
    m.add_class::<PySparseStabEngineBuilder>()?;
    m.add_class::<PyStabVecEngineBuilder>()?;
    m.add_class::<PyDensityMatrixEngineBuilder>()?;
    m.add_class::<PyStabilizerEngineBuilder>()?;
    m.add_class::<PyCoinTossEngineBuilder>()?;

    // Interface builder wrapper
    m.add_class::<PyQisInterfaceBuilder>()?;

    // Engine functions
    m.add_function(wrap_pyfunction!(self::qasm_engine, m)?)?;
    m.add_function(wrap_pyfunction!(self::qis_engine, m)?)?;
    m.add_function(wrap_pyfunction!(self::selene_engine, m)?)?;
    m.add_function(wrap_pyfunction!(self::phir_json_engine, m)?)?;
    m.add_function(wrap_pyfunction!(self::hugr_engine, m)?)?;

    // Interface builder functions
    m.add_function(wrap_pyfunction!(self::qis_helios_interface, m)?)?;
    m.add_function(wrap_pyfunction!(self::qis_selene_helios_interface, m)?)?;

    // SimBuilder function
    m.add_function(wrap_pyfunction!(self::sim_builder, m)?)?;

    // Noise builder functions
    m.add_function(wrap_pyfunction!(self::general_noise, m)?)?;
    m.add_function(wrap_pyfunction!(self::depolarizing_noise, m)?)?;
    m.add_function(wrap_pyfunction!(self::biased_depolarizing_noise, m)?)?;

    // Quantum engine builder functions
    m.add_function(wrap_pyfunction!(self::state_vector, m)?)?;
    m.add_function(wrap_pyfunction!(self::sparse_stab, m)?)?;
    m.add_function(wrap_pyfunction!(self::stabilizer, m)?)?;
    m.add_function(wrap_pyfunction!(self::stab_vec, m)?)?;
    m.add_function(wrap_pyfunction!(self::density_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(self::coin_toss, m)?)?;

    Ok(())
}
