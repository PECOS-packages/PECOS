//! Simulation API that mirrors the Rust pecos crate
//!
//! This module provides a `sim(program)` function that auto-detects the program type
//! and creates the appropriate simulation builder, following the same pattern as the
//! Rust `pecos::sim()` function.

// Import from pecos metacrate prelude
use pecos::prelude::*;

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use std::sync::{Arc, Mutex};

use crate::engine_builders::{
    PyHugrProgram, PyPhirJsonEngineBuilder, PyPhirJsonProgram, PyPhirJsonSimBuilder,
    PyQasmEngineBuilder, PyQasmProgram, PyQasmSimBuilder, PyQisControlSimBuilder,
    PyQisEngineBuilder, PyQisProgram,
};

/// Check if a Python object is a Guppy function
fn is_guppy_function(py: Python, obj: &Py<PyAny>) -> PyResult<bool> {
    // Check if guppylang module is available
    let Ok(_guppylang) = py.import(pyo3::intern!(py, "guppylang")) else {
        // GuppyLang not installed
        return Ok(false);
    };

    // Check if the object has guppy-related attributes
    let obj_bound = obj.bind(py);

    // Check multiple possible guppy attributes
    let has_guppy_attr = obj_bound.hasattr(pyo3::intern!(py, "__guppy"))?
        || obj_bound.hasattr(pyo3::intern!(py, "_guppy_compiled"))?
        || obj_bound.hasattr(pyo3::intern!(py, "compile"))?;

    // Additional check: see if the string representation contains GuppyFunctionDefinition
    if !has_guppy_attr {
        let obj_str = obj_bound.str()?.to_string();
        return Ok(obj_str.contains("GuppyFunctionDefinition"));
    }

    Ok(has_guppy_attr)
}

/// Create a simulation builder from a program
///
/// This function auto-detects the program type and creates the appropriate
/// simulation builder. It mirrors the behavior of the Rust `pecos::sim()` function.
///
/// # Supported program types:
/// - `QasmProgram` - Uses QASM engine
/// - `QisProgram` - Uses QIS control engine
/// - `HugrProgram` - Uses QIS control engine (via conversion to QIS)
/// - `PhirJsonProgram` - Uses PHIR JSON engine
/// - Guppy functions - Will be compiled to HUGR on Python side, then use QIS control engine
///
/// # Returns
/// A `PySimBuilder` configured for the detected program type
#[pyfunction]
#[allow(clippy::needless_pass_by_value)] // Py<PyAny> must be passed by value for PyO3
#[allow(clippy::too_many_lines)] // Complex function handling multiple program types
pub fn sim(py: Python, program: Py<PyAny>) -> PyResult<PySimBuilder> {
    log::debug!("sim() function called");

    // Check if it's a Guppy function - if so, it needs to be compiled to HUGR on Python side
    if is_guppy_function(py, &program)? {
        log::debug!("Detected Guppy function, will need compilation to HUGR on Python side");
        // Return a special marker that Python will recognize to trigger Guppy compilation
        // For now, we'll just return an error to let Python handle it
        return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Guppy functions must be compiled to HUGR on Python side before simulation",
        ));
    }

    // Try to extract each program type and create the appropriate builder
    if let Ok(qasm_prog) = program.extract::<PyQasmProgram>(py) {
        // Create QASM engine builder with program
        let engine_builder = pecos::qasm_engine().program(qasm_prog.inner);
        Ok(PySimBuilder {
            inner: SimBuilderInner::Qasm(PyQasmSimBuilder {
                engine_builder: Arc::new(Mutex::new(Some(engine_builder))),
                seed: None,
                workers: None,
                quantum_engine_builder: None,
                noise_builder: None,
                explicit_num_qubits: None,
            }),
        })
    } else if let Ok(qis_prog) = program.extract::<PyQisProgram>(py) {
        // Use the QIS control engine with Selene simple runtime (default)
        log::debug!("Extracted QisProgram successfully");

        // Get Selene simple runtime
        log::debug!("Getting Selene simple runtime...");
        let selene_runtime = selene_simple_runtime().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Selene simple runtime not available: {e}\n\
                    \n\
                    The default runtime for QIS programs is Selene simple.\n\
                    Please ensure Selene is built:\n\
                    cd ../selene && cargo build --release"
            ))
        })?;

        log::debug!("Creating QIS engine with Helios interface...");
        let helios_builder = helios_interface_builder();
        let builder = pecos::qis_engine();
        let builder = builder.runtime(selene_runtime);
        let builder = builder.interface(helios_builder);

        log::debug!("Loading QIS program into engine...");
        let engine_builder =
            builder
                .try_program(qis_prog.inner.clone())
                .map_err(|e: PecosError| {
                    log::error!("Failed to load QIS program: {e}");
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                        "Failed to load QIS program with Selene runtime and Helios interface: {e}"
                    ))
                })?;
        log::info!("QIS program loaded successfully");
        Ok(PySimBuilder {
            inner: SimBuilderInner::QisControl(PyQisControlSimBuilder {
                engine_builder: Arc::new(Mutex::new(Some(engine_builder))),
                seed: None,
                workers: None,
                quantum_engine_builder: None,
                noise_builder: None,
                explicit_num_qubits: None,
            }),
        })
    } else if let Ok(hugr_prog) = program.extract::<PyHugrProgram>(py) {
        // Compile HUGR to LLVM first
        log::debug!(
            "HUGR program detected (size: {} bytes), compiling to LLVM...",
            hugr_prog.inner.hugr.len()
        );

        // Compile HUGR to LLVM IR
        let llvm_ir = compile_hugr_bytes_to_string(&hugr_prog.inner.hugr).map_err(|e| {
            log::error!("HUGR compilation failed: {e}");
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "HUGR compilation failed: {e}"
            ))
        })?;
        log::info!(
            "HUGR compilation succeeded (LLVM IR size: {} bytes)",
            llvm_ir.len()
        );

        // Create QIS program from the compiled LLVM IR
        log::debug!("Creating QIS program from compiled LLVM IR...");
        let qis_prog = QisProgram::from_string(llvm_ir);

        // Get Selene simple runtime
        log::debug!("Getting Selene simple runtime...");
        let selene_runtime = selene_simple_runtime().map_err(|e| {
            log::error!("Selene simple runtime not available: {e}");
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Selene simple runtime not available: {e}\n\
                    \n\
                    The default runtime for HUGR programs is Selene simple.\n\
                    Please ensure Selene is built:\n\
                    cd ../selene && cargo build --release"
            ))
        })?;

        // Use QIS control engine with Helios interface
        log::debug!("Creating QIS engine with Helios interface for HUGR program...");
        let engine_builder = pecos::qis_engine()
            .runtime(selene_runtime)
            .interface(helios_interface_builder())
            .try_program(qis_prog)
            .map_err(|e| {
                log::error!("Failed to load compiled HUGR program: {e}");
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Failed to load compiled HUGR program: {e}"
                ))
            })?;
        log::info!("HUGR program loaded successfully");

        Ok(PySimBuilder {
            inner: SimBuilderInner::QisControl(PyQisControlSimBuilder {
                engine_builder: Arc::new(Mutex::new(Some(engine_builder))),
                seed: None,
                workers: None,
                quantum_engine_builder: None,
                noise_builder: None,
                explicit_num_qubits: None,
            }),
        })
    } else if let Ok(phir_prog) = program.extract::<PyPhirJsonProgram>(py) {
        // Create PHIR JSON engine builder with program
        let engine_builder = pecos::phir_json_engine().program(phir_prog.inner);
        Ok(PySimBuilder {
            inner: SimBuilderInner::PhirJson(PyPhirJsonSimBuilder {
                engine_builder: Arc::new(Mutex::new(Some(engine_builder))),
                seed: None,
                workers: None,
                quantum_engine_builder: None,
                noise_builder: None,
                explicit_num_qubits: None,
            }),
        })
    } else {
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "program must be a QasmProgram, QisProgram, HugrProgram, or PhirJsonProgram instance",
        ))
    }
}

/// Create an empty simulation builder
///
/// This creates a builder without a program, which must have a classical engine
/// set explicitly using `.classical()`.
#[pyfunction]
pub fn sim_builder() -> PySimBuilder {
    PySimBuilder {
        inner: SimBuilderInner::Empty,
    }
}

/// Python simulation builder
///
/// This builder follows the same fluent API as the Rust `SimBuilder`,
/// allowing method chaining to configure the simulation.
#[pyclass(name = "SimBuilder", module = "_pecos_rslib")]
#[derive(Clone)]
pub struct PySimBuilder {
    pub(crate) inner: SimBuilderInner,
}

pub(crate) enum SimBuilderInner {
    Qasm(PyQasmSimBuilder),
    QisControl(PyQisControlSimBuilder), // Unified QIS/HUGR engine
    PhirJson(PyPhirJsonSimBuilder),
    Empty, // For creating SimBuilder without a program
}

#[pymethods]
#[allow(clippy::unnecessary_wraps)] // PyO3 convention to return PyResult
impl PySimBuilder {
    /// Override the auto-selected classical engine
    #[pyo3(signature = (engine_builder))]
    #[allow(clippy::too_many_lines)] // Complex engine builder dispatch logic
    #[allow(clippy::needless_pass_by_value)] // Py<PyAny> must be passed by value for PyO3
    fn classical(&mut self, engine_builder: Py<PyAny>) -> PyResult<Self> {
        Python::attach(|py| {
            match &mut self.inner {
                SimBuilderInner::Qasm(sim_builder) => {
                    if let Ok(mut qasm_engine) = engine_builder.extract::<PyQasmEngineBuilder>(py) {
                        // Transfer program from existing engine to new engine if needed
                        let existing_engine_lock = sim_builder.engine_builder.lock().unwrap();
                        if let Some(existing_engine) = existing_engine_lock.as_ref()
                            && existing_engine.has_source()
                            && !qasm_engine.inner.has_source()
                            && let Some(program) = existing_engine.get_program()
                        {
                            // Transfer the program to the new engine
                            qasm_engine.inner = qasm_engine.inner.program(program);
                        }
                        drop(existing_engine_lock);

                        sim_builder.engine_builder = Arc::new(Mutex::new(Some(qasm_engine.inner)));
                        Ok(PySimBuilder {
                            inner: self.inner.clone(),
                        })
                    } else {
                        Err(PyTypeError::new_err(
                            "For QASM programs, classical() requires a QasmEngineBuilder",
                        ))
                    }
                }
                SimBuilderInner::QisControl(sim_builder) => {
                    if let Ok(qis_engine) = engine_builder.extract::<PyQisEngineBuilder>(py) {
                        sim_builder.engine_builder = Arc::new(Mutex::new(Some(qis_engine.inner)));
                        Ok(PySimBuilder {
                            inner: self.inner.clone(),
                        })
                    } else {
                        Err(PyTypeError::new_err(
                            "For QIS Engine programs, classical() requires a QisEngineBuilder",
                        ))
                    }
                }
                SimBuilderInner::PhirJson(sim_builder) => {
                    if let Ok(phir_engine) = engine_builder.extract::<PyPhirJsonEngineBuilder>(py) {
                        sim_builder.engine_builder = Arc::new(Mutex::new(Some(phir_engine.inner)));
                        Ok(PySimBuilder {
                            inner: self.inner.clone(),
                        })
                    } else {
                        Err(PyTypeError::new_err(
                            "For PHIR JSON programs, classical() requires a PhirJsonEngineBuilder",
                        ))
                    }
                }
                SimBuilderInner::Empty => {
                    // Handle custom engines being set on empty builder
                    Err(PyTypeError::new_err(
                        "Cannot set classical engine on empty builder - create with appropriate program type",
                    ))
                }
            }
        })
    }

    /// Set random seed
    fn seed(&mut self, seed: u64) -> PyResult<Self> {
        match &mut self.inner {
            SimBuilderInner::Qasm(builder) => builder.seed = Some(seed),
            SimBuilderInner::QisControl(builder) => builder.seed = Some(seed),
            SimBuilderInner::PhirJson(builder) => builder.seed = Some(seed),
            SimBuilderInner::Empty => {} // No-op for empty builder
        }
        Ok(PySimBuilder {
            inner: self.inner.clone(),
        })
    }

    /// Set number of worker threads
    fn workers(&mut self, workers: usize) -> PyResult<Self> {
        match &mut self.inner {
            SimBuilderInner::Qasm(builder) => builder.workers = Some(workers),
            SimBuilderInner::QisControl(builder) => builder.workers = Some(workers),
            SimBuilderInner::PhirJson(builder) => builder.workers = Some(workers),
            SimBuilderInner::Empty => {} // No-op for empty builder
        }
        Ok(PySimBuilder {
            inner: self.inner.clone(),
        })
    }

    /// Use automatic worker count based on available CPUs
    fn auto_workers(&mut self) -> PyResult<Self> {
        let workers = std::thread::available_parallelism()
            .map(std::num::NonZero::get)
            .unwrap_or(4);
        self.workers(workers)
    }

    /// Set quantum simulator/engine
    fn quantum(&mut self, engine: Py<PyAny>) -> PyResult<Self> {
        match &mut self.inner {
            SimBuilderInner::Qasm(builder) => builder.quantum_engine_builder = Some(engine),
            SimBuilderInner::QisControl(builder) => builder.quantum_engine_builder = Some(engine),
            SimBuilderInner::PhirJson(builder) => builder.quantum_engine_builder = Some(engine),
            SimBuilderInner::Empty => {} // No-op for empty builder
        }
        Ok(PySimBuilder {
            inner: self.inner.clone(),
        })
    }

    /// Set the number of qubits
    fn qubits(&mut self, num_qubits: usize) -> PyResult<Self> {
        match &mut self.inner {
            SimBuilderInner::Qasm(builder) => builder.explicit_num_qubits = Some(num_qubits),
            SimBuilderInner::QisControl(builder) => builder.explicit_num_qubits = Some(num_qubits),
            SimBuilderInner::PhirJson(builder) => builder.explicit_num_qubits = Some(num_qubits),
            SimBuilderInner::Empty => {} // No-op for empty builder
        }
        Ok(PySimBuilder {
            inner: self.inner.clone(),
        })
    }

    /// Set noise model builder
    fn noise(&mut self, noise_builder: Py<PyAny>) -> PyResult<Self> {
        match &mut self.inner {
            SimBuilderInner::Qasm(builder) => builder.noise_builder = Some(noise_builder),
            SimBuilderInner::QisControl(builder) => builder.noise_builder = Some(noise_builder),
            SimBuilderInner::PhirJson(builder) => builder.noise_builder = Some(noise_builder),
            SimBuilderInner::Empty => {} // No-op for empty builder
        }
        Ok(PySimBuilder {
            inner: self.inner.clone(),
        })
    }

    /// Run the simulation
    #[allow(clippy::too_many_lines)] // Complex simulation dispatch with multiple engine types
    fn run(&self, shots: usize) -> PyResult<crate::shot_results_bindings::PyShotVec> {
        use crate::engine_builders::{
            PyBiasedDepolarizingNoiseModelBuilder, PyDepolarizingNoiseModelBuilder,
            PyGeneralNoiseModelBuilder,
        };
        use crate::engine_builders::{PySparseStabilizerEngineBuilder, PyStateVectorEngineBuilder};
        use crate::shot_results_bindings::PyShotVec;
        use pyo3::exceptions::PyRuntimeError;

        log::debug!("PySimBuilder::run() called with {shots} shots");

        match &self.inner {
            SimBuilderInner::Qasm(builder) => {
                let mut builder_lock = builder.engine_builder.lock().unwrap();
                let engine_builder = builder_lock
                    .take()
                    .ok_or_else(|| PyRuntimeError::new_err("Builder already consumed"))?;

                // Create the Rust SimBuilder
                let mut sim_builder = engine_builder.to_sim();

                // Apply configuration
                if let Some(seed) = builder.seed {
                    sim_builder = sim_builder.seed(seed);
                }
                if let Some(workers) = builder.workers {
                    sim_builder = sim_builder.workers(workers);
                }
                if let Some(n) = builder.explicit_num_qubits {
                    sim_builder = sim_builder.qubits(n);
                }

                // Apply quantum engine builder if present
                if let Some(ref qe_py) = builder.quantum_engine_builder {
                    sim_builder = Python::attach(|py| -> PyResult<_> {
                        if let Ok(mut state_vec) = qe_py.extract::<PyStateVectorEngineBuilder>(py) {
                            if let Some(inner) = state_vec.inner.take() {
                                Ok(sim_builder.quantum(inner))
                            } else {
                                Err(PyErr::new::<PyRuntimeError, _>(
                                    "Quantum engine builder has already been consumed",
                                ))
                            }
                        } else if let Ok(mut sparse_stab) =
                            qe_py.extract::<PySparseStabilizerEngineBuilder>(py)
                        {
                            if let Some(inner) = sparse_stab.inner.take() {
                                Ok(sim_builder.quantum(inner))
                            } else {
                                Err(PyErr::new::<PyRuntimeError, _>(
                                    "Quantum engine builder has already been consumed",
                                ))
                            }
                        } else {
                            Ok(sim_builder)
                        }
                    })?;
                }

                // Apply noise builder if present
                if let Some(ref noise_py) = builder.noise_builder {
                    sim_builder = Python::attach(|py| -> PyResult<_> {
                        if let Ok(general) = noise_py.extract::<PyGeneralNoiseModelBuilder>(py) {
                            Ok(sim_builder.noise(general.inner.clone()))
                        } else if let Ok(depolarizing) =
                            noise_py.extract::<PyDepolarizingNoiseModelBuilder>(py)
                        {
                            Ok(sim_builder.noise(depolarizing.inner.clone()))
                        } else if let Ok(biased) =
                            noise_py.extract::<PyBiasedDepolarizingNoiseModelBuilder>(py)
                        {
                            Ok(sim_builder.noise(biased.inner.clone()))
                        } else {
                            Ok(sim_builder)
                        }
                    })?;
                }

                // Run directly
                match sim_builder.run(shots) {
                    Ok(shot_vec) => Ok(PyShotVec::new(shot_vec)),
                    Err(e) => Err(PyRuntimeError::new_err(format!("Simulation failed: {e}"))),
                }
            }
            SimBuilderInner::QisControl(builder) => {
                // Implementation for QIS Engine
                let mut builder_lock = builder.engine_builder.lock().unwrap();
                let engine_builder = builder_lock
                    .take()
                    .ok_or_else(|| PyRuntimeError::new_err("Builder already consumed"))?;

                // Use the Rust sim_builder API directly (from pecos prelude)
                let mut sim_builder = pecos::sim_builder().classical(engine_builder);

                if let Some(seed) = builder.seed {
                    sim_builder = sim_builder.seed(seed);
                }
                if let Some(workers) = builder.workers {
                    sim_builder = sim_builder.workers(workers);
                }
                // QIS programs require explicit qubit specification since they don't inherently specify qubit count
                let n = builder.explicit_num_qubits.ok_or_else(|| {
                    PyRuntimeError::new_err(
                        "QIS/HUGR programs require explicit qubit specification. \
                        Please call .qubits(N) to specify the number of qubits.\n\
                        \n\
                        Example:\n\
                        sim(qis_program).qubits(10).run(100)\n\
                        \n\
                        Unlike QASM programs which declare qubit registers explicitly, \
                        QIS/HUGR programs need the qubit count to be specified for proper simulation."
                    )
                })?;
                sim_builder = sim_builder.qubits(n);
                // Apply quantum engine if present
                if let Some(ref qe_py) = builder.quantum_engine_builder {
                    sim_builder = Python::attach(|py| -> PyResult<_> {
                        if let Ok(mut state_vec) = qe_py.extract::<PyStateVectorEngineBuilder>(py) {
                            if let Some(inner) = state_vec.inner.take() {
                                Ok(sim_builder.quantum(inner))
                            } else {
                                Err(PyErr::new::<PyRuntimeError, _>(
                                    "Quantum engine builder has already been consumed",
                                ))
                            }
                        } else if let Ok(mut sparse_stab) =
                            qe_py.extract::<PySparseStabilizerEngineBuilder>(py)
                        {
                            if let Some(inner) = sparse_stab.inner.take() {
                                Ok(sim_builder.quantum(inner))
                            } else {
                                Err(PyErr::new::<PyRuntimeError, _>(
                                    "Quantum engine builder has already been consumed",
                                ))
                            }
                        } else {
                            Ok(sim_builder)
                        }
                    })?;
                }

                // Apply noise builder if present
                if let Some(ref noise_py) = builder.noise_builder {
                    sim_builder = Python::attach(|py| -> PyResult<_> {
                        if let Ok(general) = noise_py.extract::<PyGeneralNoiseModelBuilder>(py) {
                            Ok(sim_builder.noise(general.inner.clone()))
                        } else if let Ok(depolarizing) =
                            noise_py.extract::<PyDepolarizingNoiseModelBuilder>(py)
                        {
                            Ok(sim_builder.noise(depolarizing.inner.clone()))
                        } else if let Ok(biased) =
                            noise_py.extract::<PyBiasedDepolarizingNoiseModelBuilder>(py)
                        {
                            Ok(sim_builder.noise(biased.inner.clone()))
                        } else {
                            Ok(sim_builder)
                        }
                    })?;
                }

                match sim_builder.run(shots) {
                    Ok(shot_vec) => Ok(PyShotVec::new(shot_vec)),
                    Err(e) => Err(PyRuntimeError::new_err(format!("Simulation failed: {e}"))),
                }
            }
            SimBuilderInner::PhirJson(builder) => {
                // Similar implementation for PHIR JSON
                let mut builder_lock = builder.engine_builder.lock().unwrap();
                let engine_builder = builder_lock
                    .take()
                    .ok_or_else(|| PyRuntimeError::new_err("Builder already consumed"))?;

                let mut sim_builder = engine_builder.to_sim();

                if let Some(seed) = builder.seed {
                    sim_builder = sim_builder.seed(seed);
                }
                if let Some(workers) = builder.workers {
                    sim_builder = sim_builder.workers(workers);
                }
                if let Some(n) = builder.explicit_num_qubits {
                    sim_builder = sim_builder.qubits(n);
                }

                // TODO: Add quantum and noise builder support for PHIR JSON

                match sim_builder.run(shots) {
                    Ok(shot_vec) => Ok(PyShotVec::new(shot_vec)),
                    Err(e) => Err(PyRuntimeError::new_err(format!("Simulation failed: {e}"))),
                }
            }
            SimBuilderInner::Empty => Err(PyRuntimeError::new_err(
                "Cannot run empty builder - no program specified",
            )),
        }
    }

    /// Build the simulation (for multiple runs)
    #[allow(clippy::too_many_lines)] // Complex builder pattern with multiple engine types
    fn build(&self) -> PyResult<Py<PyAny>> {
        use crate::engine_builders::{
            PyBiasedDepolarizingNoiseModelBuilder, PyDepolarizingNoiseModelBuilder,
            PyGeneralNoiseModelBuilder,
        };
        use crate::engine_builders::{PyPhirJsonSimulation, PyQasmSimulation};
        use crate::engine_builders::{PySparseStabilizerEngineBuilder, PyStateVectorEngineBuilder};
        use pyo3::exceptions::PyRuntimeError;

        Python::attach(|py| {
            match &self.inner {
                SimBuilderInner::Qasm(builder) => {
                    let mut builder_lock = builder.engine_builder.lock().unwrap();
                    let engine_builder = builder_lock
                        .take()
                        .ok_or_else(|| PyRuntimeError::new_err("Builder already consumed"))?;

                    // Create the Rust SimBuilder
                    let mut sim_builder = engine_builder.to_sim();

                    // Apply configuration
                    if let Some(seed) = builder.seed {
                        sim_builder = sim_builder.seed(seed);
                    }
                    if let Some(workers) = builder.workers {
                        sim_builder = sim_builder.workers(workers);
                    }
                    if let Some(n) = builder.explicit_num_qubits {
                        sim_builder = sim_builder.qubits(n);
                    }

                    // Apply quantum engine builder if present
                    if let Some(ref qe_py) = builder.quantum_engine_builder {
                        sim_builder = Python::attach(|py| -> PyResult<_> {
                            if let Ok(mut state_vec) =
                                qe_py.extract::<PyStateVectorEngineBuilder>(py)
                            {
                                if let Some(inner) = state_vec.inner.take() {
                                    Ok(sim_builder.quantum(inner))
                                } else {
                                    Err(PyErr::new::<PyRuntimeError, _>(
                                        "Quantum engine builder has already been consumed",
                                    ))
                                }
                            } else if let Ok(mut sparse_stab) =
                                qe_py.extract::<PySparseStabilizerEngineBuilder>(py)
                            {
                                if let Some(inner) = sparse_stab.inner.take() {
                                    Ok(sim_builder.quantum(inner))
                                } else {
                                    Err(PyErr::new::<PyRuntimeError, _>(
                                        "Quantum engine builder has already been consumed",
                                    ))
                                }
                            } else {
                                Ok(sim_builder)
                            }
                        })?;
                    }

                    // Apply noise builder if present
                    if let Some(ref noise_py) = builder.noise_builder {
                        sim_builder = Python::attach(|py| -> PyResult<_> {
                            if let Ok(general) = noise_py.extract::<PyGeneralNoiseModelBuilder>(py)
                            {
                                Ok(sim_builder.noise(general.inner.clone()))
                            } else if let Ok(depolarizing) =
                                noise_py.extract::<PyDepolarizingNoiseModelBuilder>(py)
                            {
                                Ok(sim_builder.noise(depolarizing.inner.clone()))
                            } else if let Ok(biased) =
                                noise_py.extract::<PyBiasedDepolarizingNoiseModelBuilder>(py)
                            {
                                Ok(sim_builder.noise(biased.inner.clone()))
                            } else {
                                Ok(sim_builder)
                            }
                        })?;
                    }

                    // Build the MonteCarloEngine
                    let engine = sim_builder.build().map_err(|e| {
                        PyRuntimeError::new_err(format!("Failed to build simulation: {e}"))
                    })?;

                    Ok(Py::new(
                        py,
                        PyQasmSimulation {
                            inner: Arc::new(Mutex::new(engine)),
                        },
                    )?
                    .into_any())
                }
                SimBuilderInner::PhirJson(builder) => {
                    // Similar implementation for PHIR JSON
                    let mut builder_lock = builder.engine_builder.lock().unwrap();
                    let engine_builder = builder_lock
                        .take()
                        .ok_or_else(|| PyRuntimeError::new_err("Builder already consumed"))?;

                    let mut sim_builder = engine_builder.to_sim();

                    if let Some(seed) = builder.seed {
                        sim_builder = sim_builder.seed(seed);
                    }
                    if let Some(workers) = builder.workers {
                        sim_builder = sim_builder.workers(workers);
                    }
                    if let Some(n) = builder.explicit_num_qubits {
                        sim_builder = sim_builder.qubits(n);
                    }

                    // TODO: Add quantum and noise builder support for PHIR JSON

                    let engine = sim_builder.build().map_err(|e| {
                        PyRuntimeError::new_err(format!("Failed to build simulation: {e}"))
                    })?;

                    Ok(Py::new(
                        py,
                        PyPhirJsonSimulation {
                            inner: Arc::new(Mutex::new(engine)),
                        },
                    )?
                    .into_any())
                }
                // QisControl doesn't have build() method in current implementation
                SimBuilderInner::QisControl(_) => Err(PyRuntimeError::new_err(
                    "QIS Engine simulation does not support build() yet - use run() directly",
                )),
                SimBuilderInner::Empty => Err(PyRuntimeError::new_err(
                    "Cannot build empty builder - no program specified",
                )),
            }
        })
    }
}

// Clone implementations for the inner types
impl Clone for SimBuilderInner {
    fn clone(&self) -> Self {
        Python::attach(|py| match self {
            SimBuilderInner::Qasm(builder) => SimBuilderInner::Qasm(PyQasmSimBuilder {
                engine_builder: builder.engine_builder.clone(),
                seed: builder.seed,
                workers: builder.workers,
                quantum_engine_builder: builder
                    .quantum_engine_builder
                    .as_ref()
                    .map(|obj| obj.clone_ref(py)),
                noise_builder: builder.noise_builder.as_ref().map(|obj| obj.clone_ref(py)),
                explicit_num_qubits: builder.explicit_num_qubits,
            }),
            SimBuilderInner::QisControl(builder) => {
                SimBuilderInner::QisControl(PyQisControlSimBuilder {
                    engine_builder: builder.engine_builder.clone(),
                    seed: builder.seed,
                    workers: builder.workers,
                    quantum_engine_builder: builder
                        .quantum_engine_builder
                        .as_ref()
                        .map(|obj| obj.clone_ref(py)),
                    noise_builder: builder.noise_builder.as_ref().map(|obj| obj.clone_ref(py)),
                    explicit_num_qubits: builder.explicit_num_qubits,
                })
            }
            SimBuilderInner::PhirJson(builder) => SimBuilderInner::PhirJson(PyPhirJsonSimBuilder {
                engine_builder: builder.engine_builder.clone(),
                seed: builder.seed,
                workers: builder.workers,
                quantum_engine_builder: builder
                    .quantum_engine_builder
                    .as_ref()
                    .map(|obj| obj.clone_ref(py)),
                noise_builder: builder.noise_builder.as_ref().map(|obj| obj.clone_ref(py)),
                explicit_num_qubits: builder.explicit_num_qubits,
            }),
            SimBuilderInner::Empty => SimBuilderInner::Empty,
        })
    }
}

/// Register the sim module with `PyO3`
pub fn register_sim_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySimBuilder>()?;
    m.add_function(wrap_pyfunction!(self::sim, m)?)?;
    m.add_function(wrap_pyfunction!(self::sim_builder, m)?)?;
    Ok(())
}
