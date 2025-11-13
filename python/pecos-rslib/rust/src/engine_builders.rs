//! `PyO3` wrappers for engine builders following the simulation API
//!
//! This module provides thin wrappers around the Rust engine builders,
//! maintaining the same API pattern: `engine().program(...).to_sim()`

// PyO3 convention is to return PyResult even for infallible operations
#![allow(clippy::unnecessary_wraps)]

// Import from pecos metacrate prelude
use pecos::prelude::*;

// Rename quantum engine builder types for clarity (from pecos prelude)
type RustQasmEngineBuilder = pecos::QasmEngineBuilder;
type RustQisEngineBuilder = pecos::QisEngineBuilder;
type RustPhirJsonEngineBuilder = pecos::PhirJsonEngineBuilder;
type RustSparseStabilizerEngineBuilder = SparseStabilizerEngineBuilder;
type RustStateVectorEngineBuilder = StateVectorEngineBuilder;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::sync::{Arc, Mutex};

// Import existing shot result types
use crate::shot_results_bindings::PyShotVec;

// Import the unified SimBuilder from sim.rs
use crate::sim::{PySimBuilder, SimBuilderInner};

/// Python wrapper for QASM engine builder
#[pyclass(name = "QasmEngineBuilder")]
#[derive(Clone)]
pub struct PyQasmEngineBuilder {
    pub(crate) inner: RustQasmEngineBuilder,
}

#[pymethods]
impl PyQasmEngineBuilder {
    #[new]
    fn new() -> Self {
        Self {
            inner: pecos::qasm_engine(),
        }
    }

    /// Set the program for this engine
    #[pyo3(signature = (program))]
    fn program(&mut self, program: &PyQasmProgram) -> PyResult<Self> {
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

    /// Get the `QasmProgram` from this builder (if any)
    pub fn get_program(&self) -> Option<PyQasmProgram> {
        self.inner
            .get_program()
            .map(|prog| PyQasmProgram { inner: prog })
    }

    /// Convert to simulation builder
    fn to_sim(&self) -> PyResult<PySimBuilder> {
        Ok(PySimBuilder {
            inner: SimBuilderInner::Qasm(PyQasmSimBuilder {
                engine_builder: Arc::new(Mutex::new(Some(self.inner.clone()))),
                seed: None,
                workers: None,
                quantum_engine_builder: None,
                noise_builder: None,
                explicit_num_qubits: None,
            }),
        })
    }
}

/// Python wrapper for QIS Engine builder (unified QIS/HUGR engine)
#[pyclass(name = "QisEngineBuilder")]
#[derive(Clone)]
pub struct PyQisEngineBuilder {
    pub(crate) inner: RustQisEngineBuilder,
}

#[pymethods]
impl PyQisEngineBuilder {
    #[new]
    fn new() -> Self {
        Self {
            inner: pecos::qis_engine(),
        }
    }

    /// Set the program for this engine
    #[pyo3(signature = (program))]
    #[allow(clippy::needless_pass_by_value)] // Py<PyAny> must be passed by value for PyO3
    fn program(&mut self, program: Py<PyAny>, py: Python) -> PyResult<Self> {
        // Check if it's a QisProgram
        if let Ok(qis_prog) = program.extract::<PyQisProgram>(py) {
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
        // Check if it's a HugrProgram
        else if let Ok(hugr_prog) = program.extract::<PyHugrProgram>(py) {
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
                "program must be either a QisProgram or HugrProgram instance",
            ));
        }
        Ok(self.clone())
    }

    /// Use Selene simple runtime
    fn selene_runtime(&mut self) -> PyResult<Self> {
        let runtime = pecos::selene_simple_runtime().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load Selene runtime: {e}"
            ))
        })?;
        self.inner = self.inner.clone().runtime(runtime);
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
            .interface(pecos::helios_interface_builder());

        // Always set Selene runtime to work with Helios interface
        log::debug!("Setting Selene runtime for Helios interface");
        let runtime = pecos::selene_simple_runtime().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load Selene runtime: {e}"
            ))
        })?;
        self.inner = self.inner.clone().runtime(runtime);

        log::debug!("Helios interface and Selene runtime configured");
        Ok(self.clone())
    }

    /// Convert to simulation builder
    fn to_sim(&self) -> PyResult<PySimBuilder> {
        Ok(PySimBuilder {
            inner: SimBuilderInner::QisControl(PyQisControlSimBuilder {
                engine_builder: Arc::new(Mutex::new(Some(self.inner.clone()))),
                seed: None,
                workers: None,
                quantum_engine_builder: None,
                noise_builder: None,
                explicit_num_qubits: None,
            }),
        })
    }
}

/// Python wrapper for PHIR JSON engine builder
#[pyclass(name = "PhirJsonEngineBuilder")]
#[derive(Clone)]
pub struct PyPhirJsonEngineBuilder {
    pub(crate) inner: RustPhirJsonEngineBuilder,
}

#[pymethods]
impl PyPhirJsonEngineBuilder {
    #[new]
    fn new() -> Self {
        Self {
            inner: pecos::phir_json_engine(),
        }
    }

    /// Set the program for this engine
    #[pyo3(signature = (program))]
    fn program(&mut self, program: &PyPhirJsonProgram) -> PyResult<Self> {
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
    pub(crate) quantum_engine_builder: Option<Py<PyAny>>,
    pub(crate) noise_builder: Option<Py<PyAny>>,
    pub(crate) explicit_num_qubits: Option<usize>,
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
        let mut engine = self.inner.lock().unwrap();
        // Use workers from builder config or default (1)
        match engine.run(shots) {
            Ok(shot_vec) => Ok(PyShotVec::new(shot_vec)),
            Err(e) => Err(PyRuntimeError::new_err(format!("Simulation failed: {e}"))),
        }
    }

    /// Run the simulation with specified number of workers
    fn run_with_workers(&self, shots: usize, workers: usize) -> PyResult<PyShotVec> {
        let mut engine = self.inner.lock().unwrap();
        match engine.run_with_workers(shots, workers) {
            Ok(shot_vec) => Ok(PyShotVec::new(shot_vec)),
            Err(e) => Err(PyRuntimeError::new_err(format!("Simulation failed: {e}"))),
        }
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
        let mut engine = self.inner.lock().unwrap();
        // Use workers from builder config or default (1)
        match engine.run(shots) {
            Ok(shot_vec) => Ok(PyShotVec::new(shot_vec)),
            Err(e) => Err(PyRuntimeError::new_err(format!("Simulation failed: {e}"))),
        }
    }

    /// Run the simulation with specified number of workers
    fn run_with_workers(&self, shots: usize, workers: usize) -> PyResult<PyShotVec> {
        let mut engine = self.inner.lock().unwrap();
        match engine.run_with_workers(shots, workers) {
            Ok(shot_vec) => Ok(PyShotVec::new(shot_vec)),
            Err(e) => Err(PyRuntimeError::new_err(format!("Simulation failed: {e}"))),
        }
    }
}

/// Internal QIS Engine simulation builder state
pub struct PyQisControlSimBuilder {
    pub(crate) engine_builder: Arc<Mutex<Option<RustQisEngineBuilder>>>,
    pub(crate) seed: Option<u64>,
    pub(crate) workers: Option<usize>,
    pub(crate) quantum_engine_builder: Option<Py<PyAny>>,
    pub(crate) noise_builder: Option<Py<PyAny>>,
    pub(crate) explicit_num_qubits: Option<usize>,
}

/// Internal PHIR JSON simulation builder state
pub struct PyPhirJsonSimBuilder {
    pub(crate) engine_builder: Arc<Mutex<Option<RustPhirJsonEngineBuilder>>>,
    pub(crate) seed: Option<u64>,
    pub(crate) workers: Option<usize>,
    pub(crate) quantum_engine_builder: Option<Py<PyAny>>,
    pub(crate) noise_builder: Option<Py<PyAny>>,
    pub(crate) explicit_num_qubits: Option<usize>,
}

/// Python wrapper for program types
#[pyclass(name = "QasmProgram")]
#[derive(Clone)]
pub struct PyQasmProgram {
    pub(crate) inner: QasmProgram,
}

#[pymethods]
impl PyQasmProgram {
    #[staticmethod]
    fn from_string(source: String) -> Self {
        PyQasmProgram {
            inner: QasmProgram::from_string(source),
        }
    }
}

#[pyclass(name = "QisProgram")]
#[derive(Clone)]
pub struct PyQisProgram {
    pub(crate) inner: QisProgram,
}

#[pymethods]
impl PyQisProgram {
    #[new]
    fn new(source: String) -> Self {
        PyQisProgram {
            inner: QisProgram::from_string(source),
        }
    }

    #[staticmethod]
    fn from_string(source: String) -> Self {
        PyQisProgram {
            inner: QisProgram::from_string(source),
        }
    }

    fn source(&self) -> String {
        self.inner.source().to_string()
    }

    #[staticmethod]
    fn preprocess_ir(llvm_ir: String) -> String {
        QisProgram::preprocess_ir(llvm_ir)
    }
}

#[pyclass(name = "HugrProgram")]
#[derive(Clone)]
pub struct PyHugrProgram {
    pub(crate) inner: HugrProgram,
}

#[pymethods]
impl PyHugrProgram {
    #[staticmethod]
    fn from_bytes(bytes: Vec<u8>) -> Self {
        PyHugrProgram {
            inner: HugrProgram::from_bytes(bytes),
        }
    }

    /// Get the HUGR bytes
    fn to_bytes(&self) -> Vec<u8> {
        self.inner.hugr.clone()
    }
}

#[pyclass(name = "PhirJsonProgram")]
#[derive(Clone)]
pub struct PyPhirJsonProgram {
    pub(crate) inner: PhirJsonProgram,
}

#[pymethods]
impl PyPhirJsonProgram {
    #[staticmethod]
    fn from_string(source: String) -> Self {
        PyPhirJsonProgram {
            inner: PhirJsonProgram::from_string(source),
        }
    }

    #[staticmethod]
    fn from_json(source: String) -> Self {
        PyPhirJsonProgram {
            inner: PhirJsonProgram::from_json(source),
        }
    }
}

/// Create a QASM engine builder
#[pyfunction]
pub fn qasm_engine() -> PyQasmEngineBuilder {
    PyQasmEngineBuilder {
        inner: pecos::qasm_engine(),
    }
}

/// Create a QIS Engine builder (unified QIS/HUGR engine)
#[pyfunction]
pub fn qis_engine() -> PyQisEngineBuilder {
    PyQisEngineBuilder {
        inner: pecos::qis_engine(),
    }
}

/// Create Selene runtime for QIS Control Engine
#[pyfunction]
pub fn selene_runtime() -> PyResult<PyQisEngineBuilder> {
    let runtime = pecos::selene_simple_runtime().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Failed to load Selene runtime: {e}"
        ))
    })?;
    Ok(PyQisEngineBuilder {
        inner: pecos::qis_engine().runtime(runtime),
    })
}

/// Create a PHIR JSON engine builder
#[pyfunction]
pub fn phir_json_engine() -> PyPhirJsonEngineBuilder {
    PyPhirJsonEngineBuilder {
        inner: pecos::phir_json_engine(),
    }
}

/// Create a general noise model builder
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

/// Python wrapper for `GeneralNoiseModelBuilder`
#[pyclass(name = "GeneralNoiseModelBuilder")]
#[derive(Clone)]
pub struct PyGeneralNoiseModelBuilder {
    pub(crate) inner: GeneralNoiseModelBuilder,
}

#[pymethods]
impl PyGeneralNoiseModelBuilder {
    #[new]
    fn new() -> Self {
        Self {
            inner: GeneralNoiseModelBuilder::new(),
        }
    }

    /// Set single-qubit gate error probability
    fn with_p1_probability(&self, p: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_p1_probability(p),
        })
    }

    /// Set two-qubit gate error probability
    fn with_p2_probability(&self, p: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_p2_probability(p),
        })
    }

    /// Set preparation error probability
    fn with_prep_probability(&self, p: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_prep_probability(p),
        })
    }

    /// Set measurement error probability for |0⟩ state
    fn with_meas_0_probability(&self, p: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_meas_0_probability(p),
        })
    }

    /// Set measurement error probability for |1⟩ state
    fn with_meas_1_probability(&self, p: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_meas_1_probability(p),
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

    /// Set average single-qubit gate error probability
    fn with_average_p1_probability(&self, p: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_average_p1_probability(p),
        })
    }

    /// Set average two-qubit gate error probability
    fn with_average_p2_probability(&self, p: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_average_p2_probability(p),
        })
    }

    /// Set measurement error probability (symmetric)
    fn with_meas_probability(&self, p: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_meas_probability(p),
        })
    }

    /// Set preparation error probability
    fn with_preparation_probability(&self, p: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_prep_probability(p),
        })
    }

    /// Set measurement error probability (asymmetric)
    fn with_measurement_probability(&self, p0: f64, p1: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self
                .inner
                .clone()
                .with_meas_0_probability(p0)
                .with_meas_1_probability(p1),
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
            "MEASURE" => GateType::Measure,
            "PREP" => GateType::Prep,
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

    /// Set whether to use coherent dephasing for idle errors
    fn with_p_idle_coherent(&self, use_coherent: bool) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_p_idle_coherent(use_coherent),
        })
    }

    /// Set the idling noise error rate for the linear term
    fn with_p_idle_linear_rate(&self, rate: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_p_idle_linear_rate(rate),
        })
    }

    /// Set the idling noise error rate for the quadratic term
    fn with_p_idle_quadratic_rate(&self, rate: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_p_idle_quadratic_rate(rate),
        })
    }

    /// Set the stochastic model for idling that is linearly dependent on time
    fn with_p_idle_linear_model(
        &self,
        model: std::collections::BTreeMap<String, f64>,
    ) -> PyResult<Self> {
        use std::collections::BTreeMap;
        let btree_map: BTreeMap<String, f64> = model.into_iter().collect();
        Ok(Self {
            inner: self.inner.clone().with_p_idle_linear_model(&btree_map),
        })
    }

    /// Set coherent to incoherent noise conversion factor
    fn with_p_idle_coherent_to_incoherent_factor(&self, factor: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self
                .inner
                .clone()
                .with_p_idle_coherent_to_incoherent_factor(factor),
        })
    }

    /// Set the average idling noise error rate per channel for the linear term
    fn with_average_p_idle_linear_rate(&self, rate: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_average_p_idle_linear_rate(rate),
        })
    }

    /// Set the average idling noise error rate per channel for the quadratic term
    fn with_average_p_idle_quadratic_rate(&self, rate: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_average_p_idle_quadratic_rate(rate),
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

    /// Set idle probability for two-qubit gates
    fn with_p2_idle(&self, probability: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_p2_idle(probability),
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
#[pyclass(name = "DepolarizingNoiseModelBuilder")]
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
    fn with_prep_probability(&self, p: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_prep_probability(p),
        })
    }

    /// Set measurement error probability
    fn with_meas_probability(&self, p: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_meas_probability(p),
        })
    }

    /// Set single-qubit gate error probability
    fn with_p1_probability(&self, p: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_p1_probability(p),
        })
    }

    /// Set two-qubit gate error probability
    fn with_p2_probability(&self, p: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_p2_probability(p),
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

    /// Set preparation error probability (alias for `with_prep_probability`)
    fn with_preparation_probability(&self, p: f64) -> PyResult<Self> {
        self.with_prep_probability(p)
    }
}

/// Python wrapper for `BiasedDepolarizingNoiseModelBuilder`
#[pyclass(name = "BiasedDepolarizingNoiseModelBuilder")]
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
    fn with_prep_probability(&self, p: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_prep_probability(p),
        })
    }

    /// Set measurement 0->1 flip probability
    fn with_meas_0_probability(&self, p: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_meas_0_probability(p),
        })
    }

    /// Set measurement 1->0 flip probability
    fn with_meas_1_probability(&self, p: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_meas_1_probability(p),
        })
    }

    /// Set single-qubit gate error probability
    fn with_p1_probability(&self, p: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_p1_probability(p),
        })
    }

    /// Set two-qubit gate error probability
    fn with_p2_probability(&self, p: f64) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_p2_probability(p),
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
#[pyclass(name = "StateVectorEngineBuilder")]
#[derive(Clone)]
pub struct PyStateVectorEngineBuilder {
    pub(crate) inner: Option<RustStateVectorEngineBuilder>,
}

#[pymethods]
impl PyStateVectorEngineBuilder {
    #[new]
    fn new() -> Self {
        Self {
            inner: Some(pecos::state_vector()),
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

/// Python wrapper for `SparseStabilizerEngineBuilder`
#[pyclass(name = "SparseStabilizerEngineBuilder")]
#[derive(Clone)]
pub struct PySparseStabilizerEngineBuilder {
    pub(crate) inner: Option<RustSparseStabilizerEngineBuilder>,
}

#[pymethods]
impl PySparseStabilizerEngineBuilder {
    #[new]
    fn new() -> Self {
        Self {
            inner: Some(pecos::sparse_stabilizer()),
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
pub fn sparse_stabilizer() -> PySparseStabilizerEngineBuilder {
    PySparseStabilizerEngineBuilder::new()
}

/// Alias for `sparse_stabilizer`
#[pyfunction]
pub fn sparse_stab() -> PySparseStabilizerEngineBuilder {
    sparse_stabilizer()
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
        inner: Box::new(pecos::helios_interface_builder()),
    })
}

/// Interface builders have been moved to implementation crates.
/// This function is deprecated and will be removed in a future version.
#[pyfunction]
pub fn qis_selene_helios_interface() -> PyResult<PyQisInterfaceBuilder> {
    Err(PyRuntimeError::new_err(
        "qis_selene_helios_interface has been moved to pecos_qis_selene crate.\n\
        Please use the implementation crate directly.",
    ))
}

/// Register the engine builder module with `PyO3`
pub fn register_engine_builders(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Engine builders
    m.add_class::<PyQasmEngineBuilder>()?;
    m.add_class::<PyQisEngineBuilder>()?;
    m.add_class::<PyQisEngineBuilder>()?;
    m.add_class::<PyPhirJsonEngineBuilder>()?;

    // Simulation builders are now handled by the unified PySimBuilder in sim.rs

    // Built simulations
    m.add_class::<PyQasmSimulation>()?;
    m.add_class::<PyPhirJsonSimulation>()?;

    // Program types
    m.add_class::<PyQasmProgram>()?;
    m.add_class::<PyHugrProgram>()?;
    m.add_class::<PyPhirJsonProgram>()?;

    // Noise builders
    m.add_class::<PyGeneralNoiseModelBuilder>()?;
    m.add_class::<PyDepolarizingNoiseModelBuilder>()?;
    m.add_class::<PyBiasedDepolarizingNoiseModelBuilder>()?;

    // Quantum engine builders
    m.add_class::<PyStateVectorEngineBuilder>()?;
    m.add_class::<PySparseStabilizerEngineBuilder>()?;

    // Interface builder wrapper
    m.add_class::<PyQisInterfaceBuilder>()?;

    // Engine functions
    m.add_function(wrap_pyfunction!(self::qasm_engine, m)?)?;
    m.add_function(wrap_pyfunction!(self::qis_engine, m)?)?;
    m.add_function(wrap_pyfunction!(self::selene_runtime, m)?)?;
    m.add_function(wrap_pyfunction!(self::phir_json_engine, m)?)?;

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
    m.add_function(wrap_pyfunction!(self::sparse_stabilizer, m)?)?;
    m.add_function(wrap_pyfunction!(sparse_stab, m)?)?;

    Ok(())
}
