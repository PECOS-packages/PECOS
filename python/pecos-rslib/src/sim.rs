//! Simulation API that mirrors the Rust pecos crate
//!
//! This module provides a `sim(program)` function that auto-detects the program type
//! and creates the appropriate simulation builder, following the same pattern as the
//! Rust `pecos::sim()` function.

// Import from pecos metacrate prelude
use crate::prelude::*;

// Import QASM WASM support
use pecos_qasm::QasmEngineWasm;

use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use std::sync::{Arc, Mutex};

use crate::engine_builders::{
    PyPhirEngineBuilder, PyPhirJson, PyPhirJsonEngineBuilder, PyPhirJsonSimBuilder,
    PyPhirSimBuilder, PyQasm, PyQasmEngineBuilder, PyQasmSimBuilder, PyQis, PyQisControlSimBuilder,
    PyQisEngineBuilder,
};
use crate::wasm_foreign_object_bindings::PyWasmForeignObject;

const UNRECOGNIZED_NOISE_BUILDER: &str = "Unrecognized noise builder type; expected \
    depolarizing_noise(), biased_depolarizing_noise(), general_noise(), scheduled_idle_z(), or scheduled_idle_noise(); \
    scheduled event noise requires QIS/HUGR engines without operation tracing";

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

/// Construct the default QIS engine.
fn default_qis_engine() -> PyResult<pecos_qis::QisEngineBuilder> {
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

    Ok(builder)
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
/// - `PhirJson` - Uses PHIR JSON engine
/// - Guppy functions - Will be lowered to QIS at the Python boundary, then use QIS control engine
///
/// # Returns
/// A `PySimBuilder` configured for the detected program type
#[pyfunction]
#[allow(clippy::needless_pass_by_value)] // Py<PyAny> must be passed by value for PyO3
#[allow(clippy::too_many_lines)] // Complex function handling multiple program types
pub fn sim(py: Python, program: Py<PyAny>) -> PyResult<PySimBuilder> {
    log::debug!("sim() function called");

    // Check if it's a Guppy function - if so, it needs to be lowered to QIS at the Python boundary
    if is_guppy_function(py, &program)? {
        log::debug!("Detected Guppy function, requires lowering to QIS at the Python boundary");
        // Return a special marker that Python will recognize to trigger Guppy compilation
        // For now, we'll just return an error to let Python handle it
        return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Guppy functions must be lowered to QIS at the Python boundary before simulation",
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
                shots: None,
                quantum_engine_builder: None,
                noise_builder: None,
                explicit_num_qubits: None,
                foreign_object: None,
                stack: None,
                classical_override: false,
            }),
        })
    } else if let Ok(qis_prog) = program.extract::<PyQis>(py) {
        // Use the QIS control engine with Selene simple runtime (default)
        log::debug!("Extracted Qis successfully");

        let builder = default_qis_engine()?;

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
                shots: None,
                quantum_engine_builder: None,
                noise_builder: None,
                explicit_num_qubits: None,
                keep_intermediate_files: false,
                qis_source: match &qis_prog.inner.content {
                    pecos_programs::QisContent::Ir(ir) => Some(ir.clone()),
                    pecos_programs::QisContent::Bitcode(_) => None,
                },
                operation_trace_dir: None,
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
                shots: None,
                quantum_engine_builder: None,
                noise_builder: None,
                explicit_num_qubits: None,
            }),
        })
    } else {
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "program must be a Qasm, Qis, or PhirJson instance",
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

/// Which simulation stack `run()` uses, mirroring the Rust facade's
/// `pecos::SimStack`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum PySimStack {
    Engines,
    Neo,
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
    QisControl(PyQisControlSimBuilder), // QIS engine via LLVM
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
                        sim_builder.classical_override = true;
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
                        // A fresh builder replaces the program-loaded one:
                        // re-attach the stored QIS source, or the built
                        // engine has no program to run.
                        let mut inner = qis_engine.inner;
                        if let Some(ref source) = sim_builder.qis_source {
                            inner = inner
                                .try_program(pecos_programs::Qis::from_string(source))
                                .map_err(|e| {
                                    PyRuntimeError::new_err(format!(
                                        "Failed to re-attach QIS program to the new engine builder: {e}"
                                    ))
                                })?;
                        }
                        sim_builder.engine_builder = Arc::new(Mutex::new(Some(inner)));
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
            SimBuilderInner::PhirJson(builder) => builder.seed = Some(seed),
            SimBuilderInner::Phir(builder) => builder.seed = Some(seed),
            SimBuilderInner::Empty => {} // No-op for empty builder
        }
        Ok(PySimBuilder {
            inner: self.inner.clone(),
        })
    }

    /// Select the simulation stack: "engines" (the default) or "neo"
    /// (experimental), mirroring the Rust facade's `.stack(SimStack)`.
    fn stack(&mut self, stack: &str) -> PyResult<Self> {
        let parsed = match stack {
            "engines" => PySimStack::Engines,
            "neo" => PySimStack::Neo,
            other => {
                return Err(PyValueError::new_err(format!(
                    "Unknown simulation stack '{other}'; expected \"engines\" or \"neo\""
                )));
            }
        };
        match &mut self.inner {
            SimBuilderInner::Qasm(builder) => builder.stack = Some(parsed),
            SimBuilderInner::QisControl(_)
            | SimBuilderInner::PhirJson(_)
            | SimBuilderInner::Phir(_) => {
                if parsed == PySimStack::Neo {
                    return Err(PyValueError::new_err(
                        "Only QASM programs are routed to the neo stack so far; \
                         this program type runs on the engines stack",
                    ));
                }
                // "engines" is already the default for every program type.
            }
            SimBuilderInner::Empty => {
                return Err(PyTypeError::new_err(
                    "Cannot select a stack on an empty builder - create with a program first",
                ));
            }
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
            SimBuilderInner::Phir(builder) => builder.workers = Some(workers),
            SimBuilderInner::Empty => {} // No-op for empty builder
        }
        Ok(PySimBuilder {
            inner: self.inner.clone(),
        })
    }

    /// Use automatic worker count based on available CPUs
    fn auto_workers(&mut self) -> PyResult<Self> {
        let workers = std::thread::available_parallelism().map_or(4, std::num::NonZero::get);
        self.workers(workers)
    }

    /// Set the number of Monte Carlo shots to run.
    ///
    /// Mirrors the Rust facade's `.shots(n)`: configure the shot count on the
    /// builder, then call `.run()` with no argument. The legacy `.run(shots)`
    /// still works and, when given, overrides this.
    fn shots(&mut self, shots: usize) -> PyResult<Self> {
        match &mut self.inner {
            SimBuilderInner::Qasm(builder) => builder.shots = Some(shots),
            SimBuilderInner::QisControl(builder) => builder.shots = Some(shots),
            SimBuilderInner::PhirJson(builder) => builder.shots = Some(shots),
            SimBuilderInner::Phir(builder) => builder.shots = Some(shots),
            SimBuilderInner::Empty => {}
        }
        Ok(PySimBuilder {
            inner: self.inner.clone(),
        })
    }

    /// Set quantum simulator/engine
    fn quantum(&mut self, engine: Py<PyAny>) -> PyResult<Self> {
        match &mut self.inner {
            SimBuilderInner::Qasm(builder) => builder.quantum_engine_builder = Some(engine),
            SimBuilderInner::QisControl(builder) => builder.quantum_engine_builder = Some(engine),
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
        if crate::scheduled_adapter::is_event_noise(&noise_builder)
            && !matches!(self.inner, SimBuilderInner::QisControl(_))
        {
            return Err(crate::scheduled_adapter::unsupported_route());
        }
        match &mut self.inner {
            SimBuilderInner::Qasm(builder) => builder.noise_builder = Some(noise_builder),
            SimBuilderInner::QisControl(builder) => builder.noise_builder = Some(noise_builder),
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
    /// called from QASM programs (e.g., WASM modules). QIS wiring is pending (issue #854), including lowered Guppy/HUGR programs.
    fn foreign_object(&mut self, foreign_obj: Py<PyAny>) -> PyResult<Self> {
        match &mut self.inner {
            SimBuilderInner::Qasm(builder) => {
                builder.foreign_object = Some(foreign_obj);
            }
            SimBuilderInner::QisControl(_) => {
                return Err(PyTypeError::new_err(
                    "foreign_object() is only supported for QASM programs; \
                     WASM foreign objects on the QIS route are not wired yet (issue #854)",
                ));
            }
            SimBuilderInner::PhirJson(_) | SimBuilderInner::Phir(_) | SimBuilderInner::Empty => {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "foreign_object() is only supported for QASM programs",
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

    /// Keep a temporary directory containing `program.ll`, the QIS source,
    /// on the built simulation.
    fn keep_intermediate_files(&mut self, keep: bool) -> PyResult<Self> {
        match &mut self.inner {
            SimBuilderInner::QisControl(builder) => {
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
    #[pyo3(signature = (shots=1))]
    fn capture_operation_trace(&self, py: Python<'_>, shots: usize) -> PyResult<Py<PyAny>> {
        use crate::engine_builders::{
            PyBiasedDepolarizingNoiseModelBuilder, PyDepolarizingNoiseModelBuilder,
            PyGeneralNoiseModelBuilder,
        };
        use crate::engine_builders::{
            PyCoinTossEngineBuilder, PyDensityMatrixEngineBuilder, PySparseStabEngineBuilder,
            PyStabVecEngineBuilder, PyStabilizerEngineBuilder, PyStateVectorEngineBuilder,
        };

        let noise = match &self.inner {
            SimBuilderInner::QisControl(b) => b.noise_builder.as_ref(),
            _ => None,
        };
        if noise.is_some_and(crate::scheduled_adapter::is_event_noise) {
            return Err(crate::scheduled_adapter::unsupported_route());
        }

        match &self.inner {
            SimBuilderInner::QisControl(builder) => {
                let builder_lock = builder.engine_builder.lock().expect("lock poisoned");
                let engine_builder = builder_lock
                    .as_ref()
                    .cloned()
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
                        "QIS programs require explicit qubit specification. \
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
                    } else if let Ok(mut stab_vec) = qe_py.extract::<PyStabVecEngineBuilder>(py) {
                        if let Some(inner) = stab_vec.inner.take() {
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
                        return Err(PyTypeError::new_err(
                            "Unrecognized quantum engine builder type; expected state_vector(), \
                             sparse_stab(), stabilizer(), stab_vec(), density_matrix(), or coin_toss()",
                        ));
                    };
                }

                if let Some(ref noise_py) = builder.noise_builder {
                    sim_builder = if let Some(scheduled) =
                        crate::engine_builders::extract_scheduled_idle(noise_py, py)
                    {
                        sim_builder.noise(scheduled)
                    } else if let Ok(general) = noise_py.extract::<PyGeneralNoiseModelBuilder>(py) {
                        sim_builder.noise(general.validated_inner()?)
                    } else if let Ok(depolarizing) =
                        noise_py.extract::<PyDepolarizingNoiseModelBuilder>(py)
                    {
                        sim_builder.noise(depolarizing.inner.clone())
                    } else if let Ok(biased) =
                        noise_py.extract::<PyBiasedDepolarizingNoiseModelBuilder>(py)
                    {
                        sim_builder.noise(biased.inner.clone())
                    } else {
                        return Err(PyTypeError::new_err(UNRECOGNIZED_NOISE_BUILDER));
                    };
                }

                if shots == 0 {
                    return Err(PyValueError::new_err(
                        "capture_operation_trace shots must be greater than zero",
                    ));
                }
                sim_builder.run(shots).map_err(|e| {
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
            | SimBuilderInner::PhirJson(_)
            | SimBuilderInner::Phir(_)
            | SimBuilderInner::Empty => Err(PyTypeError::new_err(
                "capture_operation_trace() is only supported for QIS control simulations",
            )),
        }
    }

    /// Run the simulation.
    ///
    /// `shots` may be passed here (`run(1000)`) or configured on the builder
    /// first (`.shots(1000).run()`); a `run()` argument overrides `.shots(n)`.
    /// With neither set, this fails fast rather than defaulting silently.
    #[pyo3(signature = (shots=None))]
    #[allow(clippy::too_many_lines)] // Complex simulation dispatch with multiple engine types
    fn run(&self, shots: Option<usize>) -> PyResult<crate::shot_results_bindings::PyShotVec> {
        use crate::engine_builders::{
            PyBiasedDepolarizingNoiseModelBuilder, PyDepolarizingNoiseModelBuilder,
            PyGeneralNoiseModelBuilder,
        };
        use crate::engine_builders::{
            PyCoinTossEngineBuilder, PyDensityMatrixEngineBuilder, PySparseStabEngineBuilder,
            PyStabVecEngineBuilder, PyStabilizerEngineBuilder, PyStateVectorEngineBuilder,
        };
        use crate::shot_results_bindings::PyShotVec;
        use pyo3::exceptions::PyRuntimeError;

        // Resolve the shot count: an explicit `run(shots)` argument wins, then
        // the builder's `.shots(n)`, else fail fast (no silent default).
        let configured = match &self.inner {
            SimBuilderInner::Qasm(b) => b.shots,
            SimBuilderInner::QisControl(b) => b.shots,
            SimBuilderInner::PhirJson(b) => b.shots,
            SimBuilderInner::Phir(b) => b.shots,
            SimBuilderInner::Empty => None,
        };
        let shots = shots.or(configured).ok_or_else(|| {
            PyValueError::new_err(
                "No shot count configured; pass run(shots) or set .shots(n) before .run(). \
                 Example: sim(program).shots(1000).run()",
            )
        })?;

        log::debug!("PySimBuilder::run() called with {shots} shots");

        match &self.inner {
            SimBuilderInner::Qasm(builder) => run_qasm_via_facade(builder, shots),
            SimBuilderInner::QisControl(builder) => {
                // Implementation for QIS Engine
                let builder_lock = builder.engine_builder.lock().expect("lock poisoned");
                let engine_builder = builder_lock
                    .as_ref()
                    .cloned()
                    .ok_or_else(|| PyRuntimeError::new_err("Builder already consumed"))?;
                drop(builder_lock);
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
                        "QIS programs require explicit qubit specification. \
                        Please call .qubits(N) to specify the number of qubits.\n\
                        \n\
                        Example:\n\
                        sim(qis_program).qubits(10).run(100)\n\
                        \n\
                        Unlike QASM programs which declare qubit registers explicitly, \
                        QIS programs need the qubit count to be specified for proper simulation.",
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
                        } else if let Ok(mut stab_vec) = qe_py.extract::<PyStabVecEngineBuilder>(py)
                        {
                            if let Some(inner) = stab_vec.inner.take() {
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
                            Err(PyTypeError::new_err(
                                "Unrecognized quantum engine builder type; expected state_vector(), \
                                 sparse_stab(), stabilizer(), stab_vec(), density_matrix(), or coin_toss()",
                            ))
                        }
                    })?;
                }

                // Apply noise builder if present
                if let Some(ref noise_py) = builder.noise_builder {
                    sim_builder = Python::attach(|py| -> PyResult<_> {
                        if let Some(events) =
                            crate::scheduled_adapter::extract_event_noise(noise_py, py)
                        {
                            Ok(sim_builder.noise(events))
                        } else if let Some(scheduled) =
                            crate::engine_builders::extract_scheduled_idle(noise_py, py)
                        {
                            Ok(sim_builder.noise(scheduled))
                        } else if let Ok(general) =
                            noise_py.extract::<PyGeneralNoiseModelBuilder>(py)
                        {
                            Ok(sim_builder.noise(general.validated_inner()?))
                        } else if let Ok(depolarizing) =
                            noise_py.extract::<PyDepolarizingNoiseModelBuilder>(py)
                        {
                            Ok(sim_builder.noise(depolarizing.inner.clone()))
                        } else if let Ok(biased) =
                            noise_py.extract::<PyBiasedDepolarizingNoiseModelBuilder>(py)
                        {
                            Ok(sim_builder.noise(biased.inner.clone()))
                        } else {
                            Err(PyTypeError::new_err(UNRECOGNIZED_NOISE_BUILDER))
                        }
                    })?;
                }

                match Python::attach(|py| py.detach(move || sim_builder.run(shots))) {
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
            PyCoinTossEngineBuilder, PyDensityMatrixEngineBuilder, PySparseStabEngineBuilder,
            PyStabVecEngineBuilder, PyStabilizerEngineBuilder, PyStateVectorEngineBuilder,
        };
        use crate::engine_builders::{PyPhirJsonSimulation, PyPhirSimulation, PyQasmSimulation};
        use pyo3::exceptions::PyRuntimeError;

        let neo_selected = match &self.inner {
            SimBuilderInner::Qasm(builder) => builder.stack == Some(PySimStack::Neo),
            _ => false,
        };
        if neo_selected {
            return Err(PyRuntimeError::new_err(
                "build() is not available on the neo stack (it has no reusable \
                 MonteCarloEngine); call run(shots) directly or use the engines stack",
            ));
        }

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
                            } else if let Ok(mut stab_vec) =
                                qe_py.extract::<PyStabVecEngineBuilder>(py)
                            {
                                if let Some(inner) = stab_vec.inner.take() {
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
                                Err(PyTypeError::new_err(
                                    "Unrecognized quantum engine builder type; expected state_vector(), \
                                 sparse_stab(), stabilizer(), stab_vec(), density_matrix(), or coin_toss()",
                                ))
                            }
                        })?;
                    }

                    // Apply noise builder if present
                    if let Some(ref noise_py) = builder.noise_builder {
                        sim_builder = Python::attach(|py| -> PyResult<_> {
                            if let Some(scheduled) =
                                crate::engine_builders::extract_scheduled_idle(noise_py, py)
                            {
                                Ok(sim_builder.noise(scheduled))
                            } else if let Ok(general) =
                                noise_py.extract::<PyGeneralNoiseModelBuilder>(py)
                            {
                                Ok(sim_builder.noise(general.validated_inner()?))
                            } else if let Ok(depolarizing) =
                                noise_py.extract::<PyDepolarizingNoiseModelBuilder>(py)
                            {
                                Ok(sim_builder.noise(depolarizing.inner.clone()))
                            } else if let Ok(biased) =
                                noise_py.extract::<PyBiasedDepolarizingNoiseModelBuilder>(py)
                            {
                                Ok(sim_builder.noise(biased.inner.clone()))
                            } else {
                                Err(PyTypeError::new_err(UNRECOGNIZED_NOISE_BUILDER))
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
                    let builder_lock = builder.engine_builder.lock().expect("lock poisoned");
                    let engine_builder = builder_lock
                        .as_ref()
                        .cloned()
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
                            "QIS programs require explicit qubit specification. \
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
                            } else if let Ok(mut stab_vec) =
                                qe_py.extract::<PyStabVecEngineBuilder>(py)
                            {
                                if let Some(inner) = stab_vec.inner.take() {
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
                                Err(PyTypeError::new_err(
                                    "Unrecognized quantum engine builder type; expected state_vector(), \
                                 sparse_stab(), stabilizer(), stab_vec(), density_matrix(), or coin_toss()",
                                ))
                            }
                        })?;
                    }

                    // Apply noise builder if present
                    if let Some(ref noise_py) = builder.noise_builder {
                        sim_builder = Python::attach(|py| -> PyResult<_> {
                            if let Some(events) =
                                crate::scheduled_adapter::extract_event_noise(noise_py, py)
                            {
                                Ok(sim_builder.noise(events))
                            } else if let Some(scheduled) =
                                crate::engine_builders::extract_scheduled_idle(noise_py, py)
                            {
                                Ok(sim_builder.noise(scheduled))
                            } else if let Ok(general) =
                                noise_py.extract::<PyGeneralNoiseModelBuilder>(py)
                            {
                                Ok(sim_builder.noise(general.validated_inner()?))
                            } else if let Ok(depolarizing) =
                                noise_py.extract::<PyDepolarizingNoiseModelBuilder>(py)
                            {
                                Ok(sim_builder.noise(depolarizing.inner.clone()))
                            } else if let Ok(biased) =
                                noise_py.extract::<PyBiasedDepolarizingNoiseModelBuilder>(py)
                            {
                                Ok(sim_builder.noise(biased.inner.clone()))
                            } else {
                                Err(PyTypeError::new_err(UNRECOGNIZED_NOISE_BUILDER))
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

                        // Save the QIS source this engine executes. HUGR
                        // envelopes are lowered at the Python boundary, so the
                        // QIS program is the only intermediate Rust holds; a
                        // caller that wants the envelope reads it from the
                        // Python program wrapper (`Guppy.hugr_bytes`).
                        if let Some(ref qis_source) = builder.qis_source {
                            let qis_file = temp_path.join("program.ll");
                            std::fs::write(&qis_file, qis_source).map_err(|e| {
                                PyRuntimeError::new_err(format!(
                                    "Failed to write QIS source file: {e}"
                                ))
                            })?;
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
                SimBuilderInner::Empty => Err(PyRuntimeError::new_err(
                    "Cannot build empty builder - no program specified",
                )),
            }
        })
    }
}

/// Run a QASM program through the unified `pecos::sim()` facade.
///
/// Both stacks flow through this one entry: when no stack was selected the
/// facade default governs, so a future default flip in crates/pecos carries
/// the Python surface automatically. Noise mapping for the neo stack stays
/// centralized in the facade (`map_noise_to_neo`); nothing is translated
/// here.
fn run_qasm_via_facade(
    builder: &PyQasmSimBuilder,
    shots: usize,
) -> PyResult<crate::shot_results_bindings::PyShotVec> {
    let engine_builder = builder
        .engine_builder
        .lock()
        .expect("lock poisoned")
        .take()
        .ok_or_else(|| PyRuntimeError::new_err("Builder already consumed"))?;

    // Apply a foreign object (WASM) if present, as the direct path did.
    let engine_builder = if let Some(ref fo_py) = builder.foreign_object {
        Python::attach(|py| -> PyResult<_> {
            let fo_bound = fo_py.bind(py);
            let wasm_obj: PyRef<'_, PyWasmForeignObject> =
                fo_bound.cast::<PyWasmForeignObject>()?.borrow();
            let wasm_bytes = wasm_obj.inner.wasm_bytes().to_vec();
            let qasm_wasm = QasmEngineWasm::from_bytes(wasm_bytes);
            Ok(engine_builder.wasm(qasm_wasm))
        })?
    } else {
        engine_builder
    };

    // A builder with no resolvable QASM program is invalid regardless of
    // stack or classical/WASM configuration. Resolve the program first so
    // this fundamental error is reported ahead of the neo-specific
    // rejections below — otherwise a sourceless `.classical()` + neo would
    // misreport the missing source as an unrouted classical override.
    let program = engine_builder.get_program().ok_or_else(|| {
        PyRuntimeError::new_err("No QASM source specified. Use .qasm() or .qasm_file()")
    })?;

    if builder.stack == Some(PySimStack::Neo) && engine_builder.has_wasm() {
        return Err(PyRuntimeError::new_err(
            "WASM foreign objects are not routed to the neo stack; \
             remove .wasm()/.foreign_object() or use the engines stack",
        ));
    }
    if builder.stack == Some(PySimStack::Neo) && builder.classical_override {
        // The facade contract has no classical-engine override on the neo
        // stack (the Rust sim().stack(Neo) path rejects it the same way).
        // Refuse rather than silently dropping the explicit engine and
        // running with only its program.
        return Err(PyRuntimeError::new_err(
            "Explicit .classical() engine builders are not routed to the neo stack; \
             remove .classical() or use the engines stack",
        ));
    }

    // The Python QasmEngineBuilder can only carry a program and a WASM
    // module. A plain program re-enters through the facade's auto
    // selection (identical construction); a WASM-configured engine is
    // kept verbatim via the classical override, where the facade never
    // reads the program field.
    let mut facade = if engine_builder.has_wasm() {
        pecos::sim(program).classical(engine_builder)
    } else {
        pecos::sim(program)
    };

    match builder.stack {
        None => {} // the facade default stack governs
        Some(PySimStack::Engines) => facade = facade.stack(pecos::SimStack::Engines),
        Some(PySimStack::Neo) => facade = facade.stack(pecos::SimStack::Neo),
    }
    if let Some(seed) = builder.seed {
        facade = facade.seed(seed);
    }
    if let Some(workers) = builder.workers {
        facade = facade.workers(workers);
    }
    if let Some(n) = builder.explicit_num_qubits {
        facade = facade.qubits(n);
    }
    if let Some(ref qe_py) = builder.quantum_engine_builder {
        facade = apply_quantum_to_facade(facade, qe_py)?;
    }
    if let Some(ref noise_py) = builder.noise_builder {
        facade = apply_noise_to_facade(facade, noise_py)?;
    }
    match facade.shots(shots).run() {
        Ok(shot_vec) => Ok(crate::shot_results_bindings::PyShotVec::new(shot_vec)),
        Err(e) => Err(PyRuntimeError::new_err(format!("Simulation failed: {e}"))),
    }
}

/// Extract a Python quantum-engine builder and apply it to the facade.
fn apply_quantum_to_facade(
    facade: pecos::ProgrammedSimBuilder,
    qe_py: &Py<PyAny>,
) -> PyResult<pecos::ProgrammedSimBuilder> {
    use crate::engine_builders::{
        PyCoinTossEngineBuilder, PyDensityMatrixEngineBuilder, PySparseStabEngineBuilder,
        PyStabVecEngineBuilder, PyStabilizerEngineBuilder, PyStateVectorEngineBuilder,
    };

    let consumed = || PyRuntimeError::new_err("Quantum engine builder has already been consumed");
    Python::attach(|py| -> PyResult<_> {
        if let Ok(mut state_vec) = qe_py.extract::<PyStateVectorEngineBuilder>(py) {
            Ok(facade.quantum(state_vec.inner.take().ok_or_else(consumed)?))
        } else if let Ok(mut sparse_stab) = qe_py.extract::<PySparseStabEngineBuilder>(py) {
            Ok(facade.quantum(sparse_stab.inner.take().ok_or_else(consumed)?))
        } else if let Ok(mut stab_vec) = qe_py.extract::<PyStabVecEngineBuilder>(py) {
            Ok(facade.quantum(stab_vec.inner.take().ok_or_else(consumed)?))
        } else if let Ok(mut density_mat) = qe_py.extract::<PyDensityMatrixEngineBuilder>(py) {
            Ok(facade.quantum(density_mat.inner.take().ok_or_else(consumed)?))
        } else if let Ok(mut stab) = qe_py.extract::<PyStabilizerEngineBuilder>(py) {
            Ok(facade.quantum(stab.inner.take().ok_or_else(consumed)?))
        } else if let Ok(mut ct) = qe_py.extract::<PyCoinTossEngineBuilder>(py) {
            Ok(facade.quantum(ct.inner.take().ok_or_else(consumed)?))
        } else {
            Err(PyTypeError::new_err(
                "Unrecognized quantum engine builder type; expected state_vector(), \
                 sparse_stab(), stabilizer(), stab_vec(), density_matrix(), or coin_toss()",
            ))
        }
    })
}

/// Extract a Python noise builder and apply it to the facade.
fn apply_noise_to_facade(
    facade: pecos::ProgrammedSimBuilder,
    noise_py: &Py<PyAny>,
) -> PyResult<pecos::ProgrammedSimBuilder> {
    use crate::engine_builders::{
        PyBiasedDepolarizingNoiseModelBuilder, PyDepolarizingNoiseModelBuilder,
        PyGeneralNoiseModelBuilder,
    };

    Python::attach(|py| -> PyResult<_> {
        if let Some(scheduled) = crate::engine_builders::extract_scheduled_idle(noise_py, py) {
            Ok(facade.noise(scheduled))
        } else if let Ok(general) = noise_py.extract::<PyGeneralNoiseModelBuilder>(py) {
            Ok(facade.noise(general.validated_inner()?))
        } else if let Ok(depolarizing) = noise_py.extract::<PyDepolarizingNoiseModelBuilder>(py) {
            Ok(facade.noise(depolarizing.inner.clone()))
        } else if let Ok(biased) = noise_py.extract::<PyBiasedDepolarizingNoiseModelBuilder>(py) {
            Ok(facade.noise(biased.inner.clone()))
        } else {
            Err(PyTypeError::new_err(UNRECOGNIZED_NOISE_BUILDER))
        }
    })
}

// Clone implementations for the inner types
impl Clone for SimBuilderInner {
    fn clone(&self) -> Self {
        Python::attach(|py| match self {
            SimBuilderInner::Qasm(builder) => SimBuilderInner::Qasm(PyQasmSimBuilder {
                engine_builder: builder.engine_builder.clone(),
                seed: builder.seed,
                workers: builder.workers,
                shots: builder.shots,
                quantum_engine_builder: builder
                    .quantum_engine_builder
                    .as_ref()
                    .map(|obj| obj.clone_ref(py)),
                noise_builder: builder.noise_builder.as_ref().map(|obj| obj.clone_ref(py)),
                explicit_num_qubits: builder.explicit_num_qubits,
                foreign_object: builder.foreign_object.as_ref().map(|obj| obj.clone_ref(py)),
                stack: builder.stack,
                classical_override: builder.classical_override,
            }),
            SimBuilderInner::QisControl(builder) => {
                SimBuilderInner::QisControl(PyQisControlSimBuilder {
                    engine_builder: builder.engine_builder.clone(),
                    seed: builder.seed,
                    workers: builder.workers,
                    shots: builder.shots,
                    quantum_engine_builder: builder
                        .quantum_engine_builder
                        .as_ref()
                        .map(|obj| obj.clone_ref(py)),
                    noise_builder: builder.noise_builder.as_ref().map(|obj| obj.clone_ref(py)),
                    explicit_num_qubits: builder.explicit_num_qubits,
                    keep_intermediate_files: builder.keep_intermediate_files,
                    qis_source: builder.qis_source.clone(),
                    operation_trace_dir: builder.operation_trace_dir.clone(),
                })
            }
            SimBuilderInner::PhirJson(builder) => SimBuilderInner::PhirJson(PyPhirJsonSimBuilder {
                engine_builder: builder.engine_builder.clone(),
                seed: builder.seed,
                workers: builder.workers,
                shots: builder.shots,
                quantum_engine_builder: builder
                    .quantum_engine_builder
                    .as_ref()
                    .map(|obj| obj.clone_ref(py)),
                noise_builder: builder.noise_builder.as_ref().map(|obj| obj.clone_ref(py)),
                explicit_num_qubits: builder.explicit_num_qubits,
            }),
            SimBuilderInner::Phir(builder) => SimBuilderInner::Phir(PyPhirSimBuilder {
                engine_builder: builder.engine_builder.clone(),
                seed: builder.seed,
                workers: builder.workers,
                shots: builder.shots,
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
