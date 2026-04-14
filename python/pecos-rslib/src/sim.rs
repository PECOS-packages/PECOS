//! Simulation API that mirrors the Rust pecos crate
//!
//! This module provides a `sim(program)` function that auto-detects the program type
//! and creates the appropriate simulation builder, following the same pattern as the
//! Rust `pecos::sim()` function.

// Import from pecos metacrate prelude
use crate::prelude::*;

// Import QASM WASM support
use pecos_qasm::QasmEngineWasm;

use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::prelude::*;
use std::sync::{Arc, Mutex};

use crate::engine_builders::{
    PyHugr, PyHugrEngineBuilder, PyHugrSimBuilder, PyPhirEngineBuilder, PyPhirJson,
    PyPhirJsonEngineBuilder, PyPhirJsonSimBuilder, PyPhirSimBuilder, PyQasm, PyQasmEngineBuilder,
    PyQasmSimBuilder, PyQis, PyQisControlSimBuilder, PyQisEngineBuilder,
};
use crate::wasm_foreign_object_bindings::PyWasmForeignObject;

fn unwrap_engine_builder_proxy(py: Python, engine_builder: Py<PyAny>) -> PyResult<Py<PyAny>> {
    match engine_builder
        .bind(py)
        .getattr(pyo3::intern!(py, "_builder"))
    {
        Ok(inner) => Ok(inner.into_any().unbind()),
        Err(err) if err.is_instance_of::<pyo3::exceptions::PyAttributeError>(py) => {
            Ok(engine_builder)
        }
        Err(err) => Err(err),
    }
}

fn clone_py_any_option(py: Python, value: Option<&Py<PyAny>>) -> Option<Py<PyAny>> {
    value.map(|inner| inner.clone_ref(py))
}

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
/// - `Qasm` - Uses QASM engine
/// - `Qis` - Uses QIS control engine
/// - `Hugr` - Uses QIS control engine (via conversion to QIS)
/// - `PhirJson` - Uses PHIR JSON engine
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
    if let Ok(qasm_prog) = program.extract::<PyQasm>(py) {
        // Create QASM engine builder with program
        let engine_builder = pecos_qasm::qasm_engine().program(qasm_prog.inner);
        Ok(PySimBuilder {
            inner: SimBuilderInner::Qasm(PyQasmSimBuilder {
                engine_builder: Arc::new(Mutex::new(Some(engine_builder))),
                seed: None,
                workers: None,
                quantum_engine_builder: None,
                noise_builder: None,
                explicit_num_qubits: None,
                foreign_object: None,
            }),
        })
    } else if let Ok(qis_prog) = program.extract::<PyQis>(py) {
        // Use the QIS control engine with Selene simple runtime (default)
        log::debug!("Extracted Qis successfully");

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
        let builder = pecos_qis::qis_engine();
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
                keep_intermediate_files: false,
                hugr_bytes: None, // QIS programs don't have HUGR bytes
                operation_trace_dir: None,
            }),
        })
    } else if let Ok(hugr_prog) = program.extract::<PyHugr>(py) {
        // Use direct HUGR interpreter (faster and supports loops better than LLVM path)
        log::debug!(
            "HUGR program detected (size: {} bytes), using direct interpreter",
            hugr_prog.inner.hugr.len()
        );

        // Create HUGR engine builder with the HUGR bytes
        let hugr_bytes = hugr_prog.inner.hugr.clone();
        let engine_builder = pecos_hugr::hugr_engine().hugr_bytes(hugr_bytes.clone());
        log::info!("HUGR program loaded successfully via direct interpreter");

        Ok(PySimBuilder {
            inner: SimBuilderInner::Hugr(crate::engine_builders::PyHugrSimBuilder {
                engine_builder: Arc::new(Mutex::new(Some(engine_builder))),
                seed: None,
                workers: None,
                quantum_engine_builder: None,
                noise_builder: None,
                explicit_num_qubits: None,
                foreign_object: None,
                keep_intermediate_files: false,
                hugr_bytes: Some(hugr_bytes),
            }),
        })
    } else if let Ok(phir_prog) = program.extract::<PyPhirJson>(py) {
        // Create PHIR JSON engine builder with program
        let engine_builder = pecos_phir_json::phir_json_engine().program(phir_prog.inner);
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
            "program must be a Qasm, Qis, Hugr, or PhirJson instance",
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
#[pyclass(name = "SimBuilder", module = "pecos_rslib", from_py_object)]
#[derive(Clone)]
pub struct PySimBuilder {
    pub(crate) inner: SimBuilderInner,
}

pub(crate) enum SimBuilderInner {
    Qasm(PyQasmSimBuilder),
    QisControl(PyQisControlSimBuilder), // Unified QIS/HUGR engine via LLVM
    Hugr(PyHugrSimBuilder),             // Direct HUGR interpreter
    PhirJson(PyPhirJsonSimBuilder),
    Phir(PyPhirSimBuilder),
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
            let engine_builder = unwrap_engine_builder_proxy(py, engine_builder)?;
            match &mut self.inner {
                SimBuilderInner::Qasm(sim_builder) => {
                    if let Ok(mut qasm_engine) = engine_builder.extract::<PyQasmEngineBuilder>(py) {
                        // Transfer program from existing engine to new engine if needed
                        let existing_engine_lock =
                            sim_builder.engine_builder.lock().expect("lock poisoned");
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
                SimBuilderInner::Hugr(sim_builder) => {
                    if let Ok(hugr_engine) = engine_builder.extract::<PyHugrEngineBuilder>(py) {
                        sim_builder.engine_builder = Arc::new(Mutex::new(Some(hugr_engine.inner)));
                        Ok(PySimBuilder {
                            inner: self.inner.clone(),
                        })
                    } else if let Ok(qis_engine) = engine_builder.extract::<PyQisEngineBuilder>(py)
                    {
                        if sim_builder.foreign_object.is_some() {
                            return Err(PyTypeError::new_err(
                                "For HUGR programs, classical(QisEngineBuilder) is not compatible with foreign_object()",
                            ));
                        }

                        let hugr_bytes = sim_builder.hugr_bytes.clone().ok_or_else(|| {
                            PyRuntimeError::new_err(
                                "HUGR program bytes are not available to switch this simulation onto the QIS/Helios path",
                            )
                        })?;
                        let qis_engine = qis_engine
                            .inner
                            .clone()
                            .try_program(Hugr::from_bytes(hugr_bytes.clone()))
                            .map_err(|e| {
                                PyRuntimeError::new_err(format!(
                                    "Failed to load HUGR program into QIS engine: {e}"
                                ))
                            })?;

                        Ok(PySimBuilder {
                            inner: SimBuilderInner::QisControl(PyQisControlSimBuilder {
                                engine_builder: Arc::new(Mutex::new(Some(qis_engine))),
                                seed: sim_builder.seed,
                                workers: sim_builder.workers,
                                quantum_engine_builder: clone_py_any_option(
                                    py,
                                    sim_builder.quantum_engine_builder.as_ref(),
                                ),
                                noise_builder: clone_py_any_option(
                                    py,
                                    sim_builder.noise_builder.as_ref(),
                                ),
                                explicit_num_qubits: sim_builder.explicit_num_qubits,
                                keep_intermediate_files: sim_builder.keep_intermediate_files,
                                hugr_bytes: Some(hugr_bytes),
                                operation_trace_dir: None,
                            }),
                        })
                    } else {
                        Err(PyTypeError::new_err(
                            "For direct HUGR programs, classical() requires a HugrEngineBuilder or QisEngineBuilder",
                        ))
                    }
                }
                SimBuilderInner::Phir(sim_builder) => {
                    if let Ok(phir_eng) = engine_builder.extract::<PyPhirEngineBuilder>(py) {
                        sim_builder.engine_builder = Arc::new(Mutex::new(Some(phir_eng.inner)));
                        Ok(PySimBuilder {
                            inner: self.inner.clone(),
                        })
                    } else {
                        Err(PyTypeError::new_err(
                            "For PHIR programs, classical() requires a PhirEngineBuilder",
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
            SimBuilderInner::Hugr(builder) => builder.seed = Some(seed),
            SimBuilderInner::PhirJson(builder) => builder.seed = Some(seed),
            SimBuilderInner::Phir(builder) => builder.seed = Some(seed),
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
            SimBuilderInner::Hugr(builder) => builder.workers = Some(workers),
            SimBuilderInner::PhirJson(builder) => builder.workers = Some(workers),
            SimBuilderInner::Phir(builder) => builder.workers = Some(workers),
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
            SimBuilderInner::Hugr(builder) => builder.quantum_engine_builder = Some(engine),
            SimBuilderInner::PhirJson(builder) => builder.quantum_engine_builder = Some(engine),
            SimBuilderInner::Phir(builder) => builder.quantum_engine_builder = Some(engine),
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
            SimBuilderInner::Hugr(builder) => builder.explicit_num_qubits = Some(num_qubits),
            SimBuilderInner::PhirJson(builder) => builder.explicit_num_qubits = Some(num_qubits),
            SimBuilderInner::Phir(builder) => builder.explicit_num_qubits = Some(num_qubits),
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
            SimBuilderInner::Hugr(builder) => builder.noise_builder = Some(noise_builder),
            SimBuilderInner::PhirJson(builder) => builder.noise_builder = Some(noise_builder),
            SimBuilderInner::Phir(builder) => builder.noise_builder = Some(noise_builder),
            SimBuilderInner::Empty => {} // No-op for empty builder
        }
        Ok(PySimBuilder {
            inner: self.inner.clone(),
        })
    }

    /// Set foreign object for WASM function calls
    ///
    /// The foreign object provides external function implementations that can be
    /// called from within HUGR or QASM programs (e.g., WASM modules).
    fn foreign_object(&mut self, foreign_obj: Py<PyAny>) -> PyResult<Self> {
        match &mut self.inner {
            SimBuilderInner::Hugr(builder) => {
                builder.foreign_object = Some(foreign_obj);
            }
            SimBuilderInner::Qasm(builder) => {
                builder.foreign_object = Some(foreign_obj);
            }
            SimBuilderInner::QisControl(_)
            | SimBuilderInner::PhirJson(_)
            | SimBuilderInner::Phir(_)
            | SimBuilderInner::Empty => {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "foreign_object() is only supported for HUGR and QASM programs",
                ));
            }
        }
        Ok(PySimBuilder {
            inner: self.inner.clone(),
        })
    }

    /// Enable verbose output (no-op for now, reserved for future use)
    fn verbose(&mut self, _verbose: bool) -> PyResult<Self> {
        // Currently a no-op - placeholder for future verbose output support
        Ok(PySimBuilder {
            inner: self.inner.clone(),
        })
    }

    /// Enable debug mode (no-op for now, reserved for future use)
    fn debug(&mut self, _debug: bool) -> PyResult<Self> {
        // Currently a no-op - placeholder for future debug mode support
        Ok(PySimBuilder {
            inner: self.inner.clone(),
        })
    }

    /// Enable optimization (no-op for now, reserved for future use)
    fn optimize(&mut self, _optimize: bool) -> PyResult<Self> {
        // Currently a no-op - placeholder for future optimization support
        Ok(PySimBuilder {
            inner: self.inner.clone(),
        })
    }

    /// Keep intermediate compilation files (HUGR bytes and LLVM IR)
    ///
    /// When enabled, the built simulation will have a `temp_dir` attribute
    /// pointing to a directory containing:
    /// - `program.hugr` - The HUGR bytes (if available)
    /// - `program.ll` - The compiled LLVM IR
    fn keep_intermediate_files(&mut self, keep: bool) -> PyResult<Self> {
        match &mut self.inner {
            SimBuilderInner::QisControl(builder) => {
                builder.keep_intermediate_files = keep;
            }
            SimBuilderInner::Hugr(builder) => {
                builder.keep_intermediate_files = keep;
            }
            SimBuilderInner::Qasm(_)
            | SimBuilderInner::PhirJson(_)
            | SimBuilderInner::Phir(_)
            | SimBuilderInner::Empty => {
                // These engine types don't support keep_intermediate_files yet
                // Just ignore silently for now
            }
        }
        Ok(PySimBuilder {
            inner: self.inner.clone(),
        })
    }

    /// Dump Helios-collected operation chunks to the given directory as JSON.
    fn trace_operations(&mut self, trace_dir: &str) -> PyResult<Self> {
        match &mut self.inner {
            SimBuilderInner::QisControl(builder) => {
                builder.operation_trace_dir = Some(trace_dir.to_string());
            }
            SimBuilderInner::Qasm(_)
            | SimBuilderInner::Hugr(_)
            | SimBuilderInner::PhirJson(_)
            | SimBuilderInner::Phir(_)
            | SimBuilderInner::Empty => {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "trace_operations() is only supported for QIS control simulations",
                ));
            }
        }
        Ok(PySimBuilder {
            inner: self.inner.clone(),
        })
    }

    /// Capture one in-memory QIS operation trace shot and return it as Python data.
    ///
    /// This is the preferred programmatic tracing path for QIS-control simulations.
    /// It collects the structured trace in memory first, and any JSON dumping
    /// configured via `trace_operations(...)` becomes an optional mirror/export.
    fn capture_operation_trace(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        use crate::engine_builders::{
            PyBiasedDepolarizingNoiseModelBuilder, PyDepolarizingNoiseModelBuilder,
            PyGeneralNoiseModelBuilder,
        };
        use crate::engine_builders::{
            PyCliffordRzEngineBuilder, PyCoinTossEngineBuilder, PyDensityMatrixEngineBuilder,
            PySparseStabEngineBuilder, PyStabilizerEngineBuilder, PyStateVectorEngineBuilder,
        };

        match &self.inner {
            SimBuilderInner::QisControl(builder) => {
                let mut builder_lock = builder.engine_builder.lock().expect("lock poisoned");
                let engine_builder = builder_lock
                    .take()
                    .ok_or_else(|| PyRuntimeError::new_err("Builder already consumed"))?;
                let collector: pecos_qis::OperationTraceStore = Arc::new(Mutex::new(Vec::new()));
                let engine_builder =
                    engine_builder.trace_operations_in_memory_to(collector.clone());
                let engine_builder = if let Some(ref trace_dir) = builder.operation_trace_dir {
                    engine_builder.trace_operations_to(trace_dir)
                } else {
                    engine_builder
                };

                let mut sim_builder = pecos_engines::sim_builder().classical(engine_builder);

                if let Some(seed) = builder.seed {
                    sim_builder = sim_builder.seed(seed);
                }
                if let Some(workers) = builder.workers {
                    sim_builder = sim_builder.workers(workers);
                }
                let n = builder.explicit_num_qubits.ok_or_else(|| {
                    PyRuntimeError::new_err(
                        "QIS/HUGR programs require explicit qubit specification. \
                        Please call .qubits(N) before capture_operation_trace().",
                    )
                })?;
                sim_builder = sim_builder.qubits(n);

                if let Some(ref qe_py) = builder.quantum_engine_builder {
                    sim_builder = if let Ok(mut state_vec) =
                        qe_py.extract::<PyStateVectorEngineBuilder>(py)
                    {
                        if let Some(inner) = state_vec.inner.take() {
                            sim_builder.quantum(inner)
                        } else {
                            return Err(PyErr::new::<PyRuntimeError, _>(
                                "Quantum engine builder has already been consumed",
                            ));
                        }
                    } else if let Ok(mut sparse_stab) =
                        qe_py.extract::<PySparseStabEngineBuilder>(py)
                    {
                        if let Some(inner) = sparse_stab.inner.take() {
                            sim_builder.quantum(inner)
                        } else {
                            return Err(PyErr::new::<PyRuntimeError, _>(
                                "Quantum engine builder has already been consumed",
                            ));
                        }
                    } else if let Ok(mut clifford_rz) =
                        qe_py.extract::<PyCliffordRzEngineBuilder>(py)
                    {
                        if let Some(inner) = clifford_rz.inner.take() {
                            sim_builder.quantum(inner)
                        } else {
                            return Err(PyErr::new::<PyRuntimeError, _>(
                                "Quantum engine builder has already been consumed",
                            ));
                        }
                    } else if let Ok(mut density_mat) =
                        qe_py.extract::<PyDensityMatrixEngineBuilder>(py)
                    {
                        if let Some(inner) = density_mat.inner.take() {
                            sim_builder.quantum(inner)
                        } else {
                            return Err(PyErr::new::<PyRuntimeError, _>(
                                "Quantum engine builder has already been consumed",
                            ));
                        }
                    } else if let Ok(mut stab) = qe_py.extract::<PyStabilizerEngineBuilder>(py) {
                        if let Some(inner) = stab.inner.take() {
                            sim_builder.quantum(inner)
                        } else {
                            return Err(PyErr::new::<PyRuntimeError, _>(
                                "Quantum engine builder has already been consumed",
                            ));
                        }
                    } else if let Ok(mut ct) = qe_py.extract::<PyCoinTossEngineBuilder>(py) {
                        if let Some(inner) = ct.inner.take() {
                            sim_builder.quantum(inner)
                        } else {
                            return Err(PyErr::new::<PyRuntimeError, _>(
                                "Quantum engine builder has already been consumed",
                            ));
                        }
                    } else {
                        sim_builder
                    };
                }

                if let Some(ref noise_py) = builder.noise_builder {
                    sim_builder =
                        if let Ok(general) = noise_py.extract::<PyGeneralNoiseModelBuilder>(py) {
                            sim_builder.noise(general.inner.clone())
                        } else if let Ok(depolarizing) =
                            noise_py.extract::<PyDepolarizingNoiseModelBuilder>(py)
                        {
                            sim_builder.noise(depolarizing.inner.clone())
                        } else if let Ok(biased) =
                            noise_py.extract::<PyBiasedDepolarizingNoiseModelBuilder>(py)
                        {
                            sim_builder.noise(biased.inner.clone())
                        } else {
                            sim_builder
                        };
                }

                sim_builder.run(1).map_err(|e| {
                    PyRuntimeError::new_err(format!("Trace capture simulation failed: {e}"))
                })?;

                let trace = collector.lock().expect("lock poisoned").clone();
                let trace_json = serde_json::to_string(&trace).map_err(|e| {
                    PyRuntimeError::new_err(format!("Failed to serialize in-memory trace: {e}"))
                })?;
                let json = py.import(pyo3::intern!(py, "json"))?;
                Ok(json
                    .call_method1(pyo3::intern!(py, "loads"), (trace_json,))?
                    .into())
            }
            SimBuilderInner::Qasm(_)
            | SimBuilderInner::Hugr(_)
            | SimBuilderInner::PhirJson(_)
            | SimBuilderInner::Phir(_)
            | SimBuilderInner::Empty => Err(PyTypeError::new_err(
                "capture_operation_trace() is only supported for QIS control simulations",
            )),
        }
    }

    /// Run the simulation
    #[allow(clippy::too_many_lines)] // Complex simulation dispatch with multiple engine types
    fn run(&self, shots: usize) -> PyResult<crate::shot_results_bindings::PyShotVec> {
        use crate::engine_builders::{
            PyBiasedDepolarizingNoiseModelBuilder, PyDepolarizingNoiseModelBuilder,
            PyGeneralNoiseModelBuilder,
        };
        use crate::engine_builders::{
            PyCliffordRzEngineBuilder, PyCoinTossEngineBuilder, PyDensityMatrixEngineBuilder,
            PySparseStabEngineBuilder, PyStabilizerEngineBuilder, PyStateVectorEngineBuilder,
        };
        use crate::shot_results_bindings::PyShotVec;
        use pyo3::exceptions::PyRuntimeError;

        log::debug!("PySimBuilder::run() called with {shots} shots");

        match &self.inner {
            SimBuilderInner::Qasm(builder) => {
                let mut builder_lock = builder.engine_builder.lock().expect("lock poisoned");
                let engine_builder = builder_lock
                    .take()
                    .ok_or_else(|| PyRuntimeError::new_err("Builder already consumed"))?;

                // Apply foreign object if present
                let engine_builder = if let Some(ref fo_py) = builder.foreign_object {
                    Python::attach(|py| -> PyResult<_> {
                        let fo_bound = fo_py.bind(py);
                        let wasm_obj: PyRef<'_, PyWasmForeignObject> =
                            fo_bound.cast::<PyWasmForeignObject>()?.borrow();
                        // Get WASM bytes and create QasmEngineWasm
                        let wasm_bytes = wasm_obj.inner.wasm_bytes().to_vec();
                        let qasm_wasm = QasmEngineWasm::from_bytes(wasm_bytes);
                        Ok(engine_builder.wasm(qasm_wasm))
                    })?
                } else {
                    engine_builder
                };

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
                            qe_py.extract::<PySparseStabEngineBuilder>(py)
                        {
                            if let Some(inner) = sparse_stab.inner.take() {
                                Ok(sim_builder.quantum(inner))
                            } else {
                                Err(PyErr::new::<PyRuntimeError, _>(
                                    "Quantum engine builder has already been consumed",
                                ))
                            }
                        } else if let Ok(mut clifford_rz) =
                            qe_py.extract::<PyCliffordRzEngineBuilder>(py)
                        {
                            if let Some(inner) = clifford_rz.inner.take() {
                                Ok(sim_builder.quantum(inner))
                            } else {
                                Err(PyErr::new::<PyRuntimeError, _>(
                                    "Quantum engine builder has already been consumed",
                                ))
                            }
                        } else if let Ok(mut density_mat) =
                            qe_py.extract::<PyDensityMatrixEngineBuilder>(py)
                        {
                            if let Some(inner) = density_mat.inner.take() {
                                Ok(sim_builder.quantum(inner))
                            } else {
                                Err(PyErr::new::<PyRuntimeError, _>(
                                    "Quantum engine builder has already been consumed",
                                ))
                            }
                        } else if let Ok(mut stab) = qe_py.extract::<PyStabilizerEngineBuilder>(py)
                        {
                            if let Some(inner) = stab.inner.take() {
                                Ok(sim_builder.quantum(inner))
                            } else {
                                Err(PyErr::new::<PyRuntimeError, _>(
                                    "Quantum engine builder has already been consumed",
                                ))
                            }
                        } else if let Ok(mut ct) = qe_py.extract::<PyCoinTossEngineBuilder>(py) {
                            if let Some(inner) = ct.inner.take() {
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
                let mut builder_lock = builder.engine_builder.lock().expect("lock poisoned");
                let engine_builder = builder_lock
                    .take()
                    .ok_or_else(|| PyRuntimeError::new_err("Builder already consumed"))?;
                let engine_builder = if let Some(ref trace_dir) = builder.operation_trace_dir {
                    engine_builder.trace_operations_to(trace_dir)
                } else {
                    engine_builder
                };

                // Use the Rust sim_builder API directly (from pecos prelude)
                let mut sim_builder = pecos_engines::sim_builder().classical(engine_builder);

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
                            qe_py.extract::<PySparseStabEngineBuilder>(py)
                        {
                            if let Some(inner) = sparse_stab.inner.take() {
                                Ok(sim_builder.quantum(inner))
                            } else {
                                Err(PyErr::new::<PyRuntimeError, _>(
                                    "Quantum engine builder has already been consumed",
                                ))
                            }
                        } else if let Ok(mut clifford_rz) =
                            qe_py.extract::<PyCliffordRzEngineBuilder>(py)
                        {
                            if let Some(inner) = clifford_rz.inner.take() {
                                Ok(sim_builder.quantum(inner))
                            } else {
                                Err(PyErr::new::<PyRuntimeError, _>(
                                    "Quantum engine builder has already been consumed",
                                ))
                            }
                        } else if let Ok(mut density_mat) =
                            qe_py.extract::<PyDensityMatrixEngineBuilder>(py)
                        {
                            if let Some(inner) = density_mat.inner.take() {
                                Ok(sim_builder.quantum(inner))
                            } else {
                                Err(PyErr::new::<PyRuntimeError, _>(
                                    "Quantum engine builder has already been consumed",
                                ))
                            }
                        } else if let Ok(mut stab) = qe_py.extract::<PyStabilizerEngineBuilder>(py)
                        {
                            if let Some(inner) = stab.inner.take() {
                                Ok(sim_builder.quantum(inner))
                            } else {
                                Err(PyErr::new::<PyRuntimeError, _>(
                                    "Quantum engine builder has already been consumed",
                                ))
                            }
                        } else if let Ok(mut ct) = qe_py.extract::<PyCoinTossEngineBuilder>(py) {
                            if let Some(inner) = ct.inner.take() {
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
                let mut builder_lock = builder.engine_builder.lock().expect("lock poisoned");
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
            SimBuilderInner::Phir(builder) => {
                let mut builder_lock = builder.engine_builder.lock().expect("lock poisoned");
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

                match sim_builder.run(shots) {
                    Ok(shot_vec) => Ok(PyShotVec::new(shot_vec)),
                    Err(e) => Err(PyRuntimeError::new_err(format!("Simulation failed: {e}"))),
                }
            }
            SimBuilderInner::Hugr(builder) => {
                // Direct HUGR interpreter
                let mut builder_lock = builder.engine_builder.lock().expect("lock poisoned");
                let engine_builder = builder_lock
                    .take()
                    .ok_or_else(|| PyRuntimeError::new_err("Builder already consumed"))?;

                // Apply foreign object if present
                let engine_builder = if let Some(ref fo_py) = builder.foreign_object {
                    Python::attach(|py| -> PyResult<_> {
                        let fo_bound = fo_py.bind(py);
                        let wasm_obj: PyRef<'_, PyWasmForeignObject> =
                            fo_bound.cast::<PyWasmForeignObject>()?.borrow();
                        Ok(engine_builder.foreign_object(wasm_obj.clone_boxed()))
                    })?
                } else {
                    engine_builder
                };

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
                            qe_py.extract::<PySparseStabEngineBuilder>(py)
                        {
                            if let Some(inner) = sparse_stab.inner.take() {
                                Ok(sim_builder.quantum(inner))
                            } else {
                                Err(PyErr::new::<PyRuntimeError, _>(
                                    "Quantum engine builder has already been consumed",
                                ))
                            }
                        } else if let Ok(mut clifford_rz) =
                            qe_py.extract::<PyCliffordRzEngineBuilder>(py)
                        {
                            if let Some(inner) = clifford_rz.inner.take() {
                                Ok(sim_builder.quantum(inner))
                            } else {
                                Err(PyErr::new::<PyRuntimeError, _>(
                                    "Quantum engine builder has already been consumed",
                                ))
                            }
                        } else if let Ok(mut density_mat) =
                            qe_py.extract::<PyDensityMatrixEngineBuilder>(py)
                        {
                            if let Some(inner) = density_mat.inner.take() {
                                Ok(sim_builder.quantum(inner))
                            } else {
                                Err(PyErr::new::<PyRuntimeError, _>(
                                    "Quantum engine builder has already been consumed",
                                ))
                            }
                        } else if let Ok(mut stab) = qe_py.extract::<PyStabilizerEngineBuilder>(py)
                        {
                            if let Some(inner) = stab.inner.take() {
                                Ok(sim_builder.quantum(inner))
                            } else {
                                Err(PyErr::new::<PyRuntimeError, _>(
                                    "Quantum engine builder has already been consumed",
                                ))
                            }
                        } else if let Ok(mut ct) = qe_py.extract::<PyCoinTossEngineBuilder>(py) {
                            if let Some(inner) = ct.inner.take() {
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
        use crate::engine_builders::{
            PyCliffordRzEngineBuilder, PyCoinTossEngineBuilder, PyDensityMatrixEngineBuilder,
            PySparseStabEngineBuilder, PyStabilizerEngineBuilder, PyStateVectorEngineBuilder,
        };
        use crate::engine_builders::{PyPhirJsonSimulation, PyPhirSimulation, PyQasmSimulation};
        use pyo3::exceptions::PyRuntimeError;

        Python::attach(|py| {
            match &self.inner {
                SimBuilderInner::Qasm(builder) => {
                    let mut builder_lock = builder.engine_builder.lock().expect("lock poisoned");
                    let engine_builder = builder_lock
                        .take()
                        .ok_or_else(|| PyRuntimeError::new_err("Builder already consumed"))?;

                    // Apply foreign object if present
                    let engine_builder = if let Some(ref fo_py) = builder.foreign_object {
                        let fo_bound = fo_py.bind(py);
                        let wasm_obj: PyRef<'_, PyWasmForeignObject> =
                            fo_bound.cast::<PyWasmForeignObject>()?.borrow();
                        // Get WASM bytes and create QasmEngineWasm
                        let wasm_bytes = wasm_obj.inner.wasm_bytes().to_vec();
                        let qasm_wasm = QasmEngineWasm::from_bytes(wasm_bytes);
                        engine_builder.wasm(qasm_wasm)
                    } else {
                        engine_builder
                    };

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
                                qe_py.extract::<PySparseStabEngineBuilder>(py)
                            {
                                if let Some(inner) = sparse_stab.inner.take() {
                                    Ok(sim_builder.quantum(inner))
                                } else {
                                    Err(PyErr::new::<PyRuntimeError, _>(
                                        "Quantum engine builder has already been consumed",
                                    ))
                                }
                            } else if let Ok(mut clifford_rz) =
                                qe_py.extract::<PyCliffordRzEngineBuilder>(py)
                            {
                                if let Some(inner) = clifford_rz.inner.take() {
                                    Ok(sim_builder.quantum(inner))
                                } else {
                                    Err(PyErr::new::<PyRuntimeError, _>(
                                        "Quantum engine builder has already been consumed",
                                    ))
                                }
                            } else if let Ok(mut density_mat) =
                                qe_py.extract::<PyDensityMatrixEngineBuilder>(py)
                            {
                                if let Some(inner) = density_mat.inner.take() {
                                    Ok(sim_builder.quantum(inner))
                                } else {
                                    Err(PyErr::new::<PyRuntimeError, _>(
                                        "Quantum engine builder has already been consumed",
                                    ))
                                }
                            } else if let Ok(mut stab) =
                                qe_py.extract::<PyStabilizerEngineBuilder>(py)
                            {
                                if let Some(inner) = stab.inner.take() {
                                    Ok(sim_builder.quantum(inner))
                                } else {
                                    Err(PyErr::new::<PyRuntimeError, _>(
                                        "Quantum engine builder has already been consumed",
                                    ))
                                }
                            } else if let Ok(mut ct) = qe_py.extract::<PyCoinTossEngineBuilder>(py)
                            {
                                if let Some(inner) = ct.inner.take() {
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
                    let mut builder_lock = builder.engine_builder.lock().expect("lock poisoned");
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
                SimBuilderInner::Phir(builder) => {
                    let mut builder_lock = builder.engine_builder.lock().expect("lock poisoned");
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

                    let engine = sim_builder.build().map_err(|e| {
                        PyRuntimeError::new_err(format!("Failed to build simulation: {e}"))
                    })?;

                    Ok(Py::new(
                        py,
                        PyPhirSimulation {
                            inner: Arc::new(Mutex::new(engine)),
                        },
                    )?
                    .into_any())
                }
                SimBuilderInner::QisControl(builder) => {
                    // Implementation for QIS Engine build()
                    let mut builder_lock = builder.engine_builder.lock().expect("lock poisoned");
                    let engine_builder = builder_lock
                        .take()
                        .ok_or_else(|| PyRuntimeError::new_err("Builder already consumed"))?;
                    let engine_builder = if let Some(ref trace_dir) = builder.operation_trace_dir {
                        engine_builder.trace_operations_to(trace_dir)
                    } else {
                        engine_builder
                    };

                    // Use the Rust sim_builder API directly (from pecos prelude)
                    let mut sim_builder = pecos_engines::sim_builder().classical(engine_builder);

                    if let Some(seed) = builder.seed {
                        sim_builder = sim_builder.seed(seed);
                    }
                    if let Some(workers) = builder.workers {
                        sim_builder = sim_builder.workers(workers);
                    }
                    // QIS programs require explicit qubit specification
                    let n = builder.explicit_num_qubits.ok_or_else(|| {
                        PyRuntimeError::new_err(
                            "QIS/HUGR programs require explicit qubit specification. \
                            Please call .qubits(N) to specify the number of qubits.",
                        )
                    })?;
                    sim_builder = sim_builder.qubits(n);

                    // Apply quantum engine if present
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
                                qe_py.extract::<PySparseStabEngineBuilder>(py)
                            {
                                if let Some(inner) = sparse_stab.inner.take() {
                                    Ok(sim_builder.quantum(inner))
                                } else {
                                    Err(PyErr::new::<PyRuntimeError, _>(
                                        "Quantum engine builder has already been consumed",
                                    ))
                                }
                            } else if let Ok(mut clifford_rz) =
                                qe_py.extract::<PyCliffordRzEngineBuilder>(py)
                            {
                                if let Some(inner) = clifford_rz.inner.take() {
                                    Ok(sim_builder.quantum(inner))
                                } else {
                                    Err(PyErr::new::<PyRuntimeError, _>(
                                        "Quantum engine builder has already been consumed",
                                    ))
                                }
                            } else if let Ok(mut density_mat) =
                                qe_py.extract::<PyDensityMatrixEngineBuilder>(py)
                            {
                                if let Some(inner) = density_mat.inner.take() {
                                    Ok(sim_builder.quantum(inner))
                                } else {
                                    Err(PyErr::new::<PyRuntimeError, _>(
                                        "Quantum engine builder has already been consumed",
                                    ))
                                }
                            } else if let Ok(mut stab) =
                                qe_py.extract::<PyStabilizerEngineBuilder>(py)
                            {
                                if let Some(inner) = stab.inner.take() {
                                    Ok(sim_builder.quantum(inner))
                                } else {
                                    Err(PyErr::new::<PyRuntimeError, _>(
                                        "Quantum engine builder has already been consumed",
                                    ))
                                }
                            } else if let Ok(mut ct) = qe_py.extract::<PyCoinTossEngineBuilder>(py)
                            {
                                if let Some(inner) = ct.inner.take() {
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

                    // Handle intermediate file saving if requested
                    let temp_dir = if builder.keep_intermediate_files {
                        // Create a persistent temp directory
                        let temp_dir = tempfile::Builder::new()
                            .prefix("pecos_sim_")
                            .tempdir()
                            .map_err(|e| {
                                PyRuntimeError::new_err(format!(
                                    "Failed to create temp directory: {e}"
                                ))
                            })?;

                        let temp_path = temp_dir.path();

                        // Save HUGR bytes if available
                        if let Some(ref hugr_bytes) = builder.hugr_bytes {
                            let hugr_file = temp_path.join("program.hugr");
                            std::fs::write(&hugr_file, hugr_bytes).map_err(|e| {
                                PyRuntimeError::new_err(format!("Failed to write HUGR file: {e}"))
                            })?;

                            // Also compile and save LLVM IR
                            match compile_hugr_bytes_to_string(hugr_bytes) {
                                Ok(llvm_ir) => {
                                    let ll_file = temp_path.join("program.ll");
                                    std::fs::write(&ll_file, llvm_ir).map_err(|e| {
                                        PyRuntimeError::new_err(format!(
                                            "Failed to write LLVM IR file: {e}"
                                        ))
                                    })?;
                                }
                                Err(e) => {
                                    log::warn!("Could not compile HUGR to LLVM IR for saving: {e}");
                                }
                            }
                        }

                        // Keep the directory (don't let it be deleted on drop)
                        let path_str = temp_path.to_string_lossy().to_string();
                        let _ = temp_dir.keep(); // Prevents cleanup
                        Some(path_str)
                    } else {
                        None
                    };

                    Ok(Py::new(
                        py,
                        crate::engine_builders::PyQisControlSimulation {
                            inner: Arc::new(Mutex::new(engine)),
                            temp_dir,
                            operation_trace_dir: builder.operation_trace_dir.clone(),
                        },
                    )?
                    .into_any())
                }
                SimBuilderInner::Hugr(builder) => {
                    // Direct HUGR interpreter build
                    let mut builder_lock = builder.engine_builder.lock().expect("lock poisoned");
                    let engine_builder = builder_lock
                        .take()
                        .ok_or_else(|| PyRuntimeError::new_err("Builder already consumed"))?;

                    // Apply foreign object if present
                    let engine_builder = if let Some(ref fo_py) = builder.foreign_object {
                        let fo_bound = fo_py.bind(py);
                        let wasm_obj: PyRef<'_, PyWasmForeignObject> =
                            fo_bound.cast::<PyWasmForeignObject>()?.borrow();
                        engine_builder.foreign_object(wasm_obj.clone_boxed())
                    } else {
                        engine_builder
                    };

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
                                qe_py.extract::<PySparseStabEngineBuilder>(py)
                            {
                                if let Some(inner) = sparse_stab.inner.take() {
                                    Ok(sim_builder.quantum(inner))
                                } else {
                                    Err(PyErr::new::<PyRuntimeError, _>(
                                        "Quantum engine builder has already been consumed",
                                    ))
                                }
                            } else if let Ok(mut clifford_rz) =
                                qe_py.extract::<PyCliffordRzEngineBuilder>(py)
                            {
                                if let Some(inner) = clifford_rz.inner.take() {
                                    Ok(sim_builder.quantum(inner))
                                } else {
                                    Err(PyErr::new::<PyRuntimeError, _>(
                                        "Quantum engine builder has already been consumed",
                                    ))
                                }
                            } else if let Ok(mut density_mat) =
                                qe_py.extract::<PyDensityMatrixEngineBuilder>(py)
                            {
                                if let Some(inner) = density_mat.inner.take() {
                                    Ok(sim_builder.quantum(inner))
                                } else {
                                    Err(PyErr::new::<PyRuntimeError, _>(
                                        "Quantum engine builder has already been consumed",
                                    ))
                                }
                            } else if let Ok(mut stab) =
                                qe_py.extract::<PyStabilizerEngineBuilder>(py)
                            {
                                if let Some(inner) = stab.inner.take() {
                                    Ok(sim_builder.quantum(inner))
                                } else {
                                    Err(PyErr::new::<PyRuntimeError, _>(
                                        "Quantum engine builder has already been consumed",
                                    ))
                                }
                            } else if let Ok(mut ct) = qe_py.extract::<PyCoinTossEngineBuilder>(py)
                            {
                                if let Some(inner) = ct.inner.take() {
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

                    let engine = sim_builder.build().map_err(|e| {
                        PyRuntimeError::new_err(format!("Failed to build simulation: {e}"))
                    })?;

                    // Handle intermediate file saving if requested
                    let temp_dir = if builder.keep_intermediate_files {
                        // Create a persistent temp directory
                        let temp_dir = tempfile::Builder::new()
                            .prefix("pecos_hugr_sim_")
                            .tempdir()
                            .map_err(|e| {
                                PyRuntimeError::new_err(format!(
                                    "Failed to create temp directory: {e}"
                                ))
                            })?;

                        let temp_path = temp_dir.path();

                        // Save HUGR bytes if available
                        if let Some(ref hugr_bytes) = builder.hugr_bytes {
                            let hugr_file = temp_path.join("program.hugr");
                            std::fs::write(&hugr_file, hugr_bytes).map_err(|e| {
                                PyRuntimeError::new_err(format!("Failed to write HUGR file: {e}"))
                            })?;

                            // Also compile and save LLVM IR for debugging (graceful failure)
                            match compile_hugr_bytes_to_string(hugr_bytes) {
                                Ok(llvm_ir) => {
                                    let ll_file = temp_path.join("program.ll");
                                    if let Err(e) = std::fs::write(&ll_file, llvm_ir) {
                                        log::warn!("Could not write LLVM IR file: {e}");
                                    }
                                }
                                Err(e) => {
                                    log::warn!("Could not compile HUGR to LLVM IR for saving: {e}");
                                }
                            }
                        }

                        // Keep the directory (don't let it be deleted on drop)
                        let path_str = temp_path.to_string_lossy().to_string();
                        let _ = temp_dir.keep(); // Prevents cleanup
                        Some(path_str)
                    } else {
                        None
                    };

                    Ok(Py::new(
                        py,
                        crate::engine_builders::PyHugrSimulation {
                            inner: Arc::new(Mutex::new(engine)),
                            temp_dir,
                        },
                    )?
                    .into_any())
                }
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
                foreign_object: builder.foreign_object.as_ref().map(|obj| obj.clone_ref(py)),
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
                    keep_intermediate_files: builder.keep_intermediate_files,
                    hugr_bytes: builder.hugr_bytes.clone(),
                    operation_trace_dir: builder.operation_trace_dir.clone(),
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
            SimBuilderInner::Hugr(builder) => SimBuilderInner::Hugr(PyHugrSimBuilder {
                engine_builder: builder.engine_builder.clone(),
                seed: builder.seed,
                workers: builder.workers,
                quantum_engine_builder: builder
                    .quantum_engine_builder
                    .as_ref()
                    .map(|obj| obj.clone_ref(py)),
                noise_builder: builder.noise_builder.as_ref().map(|obj| obj.clone_ref(py)),
                explicit_num_qubits: builder.explicit_num_qubits,
                foreign_object: builder.foreign_object.as_ref().map(|obj| obj.clone_ref(py)),
                keep_intermediate_files: builder.keep_intermediate_files,
                hugr_bytes: builder.hugr_bytes.clone(),
            }),
            SimBuilderInner::Phir(builder) => SimBuilderInner::Phir(PyPhirSimBuilder {
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
