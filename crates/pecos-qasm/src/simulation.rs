//! Builder-based simulation runner for QASM
//!
//! This module provides a fluent builder API for running QASM simulations
//! with support for various noise models and quantum engines.

use crate::QASMEngine;
use pecos_core::errors::PecosError;
use pecos_engines::noise::{
    BiasedDepolarizingNoiseModelBuilder, DepolarizingNoiseModelBuilder, GeneralNoiseModelBuilder,
    NoiseModel, PassThroughNoiseModel, PassThroughNoiseModelBuilder,
};
use pecos_engines::quantum::{QuantumEngine, SparseStabEngine, StateVecEngine};
use pecos_engines::shot_results::ShotVec;
use pecos_engines::{ClassicalEngine, MonteCarloEngine};
use std::str::FromStr;

/// Noise model configuration
///
/// This enum holds builders for different noise models.
#[derive(Debug, Clone)]
pub enum NoiseModelType {
    /// No noise (ideal simulation)
    PassThrough(Box<PassThroughNoiseModelBuilder>),
    /// Depolarizing noise model
    Depolarizing(Box<DepolarizingNoiseModelBuilder>),
    /// Biased depolarizing noise model
    BiasedDepolarizing(Box<BiasedDepolarizingNoiseModelBuilder>),
    /// General noise model
    General(Box<GeneralNoiseModelBuilder>),
}

impl NoiseModelType {
    /// Create a boxed noise model instance
    #[must_use]
    pub fn create_noise_model(self) -> Box<dyn NoiseModel> {
        match self {
            Self::PassThrough(builder) => Box::new(builder.build()),
            Self::Depolarizing(builder) => Box::new(builder.build()),
            Self::BiasedDepolarizing(builder) => Box::new(builder.build()),
            Self::General(builder) => Box::new(builder.build()),
        }
    }
}

impl Default for NoiseModelType {
    fn default() -> Self {
        Self::PassThrough(Box::new(PassThroughNoiseModel::builder()))
    }
}

/// Available quantum simulation engines
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuantumEngineType {
    /// State vector simulator (full quantum state)
    StateVector,
    /// Sparse stabilizer simulator (efficient for Clifford circuits)
    SparseStabilizer,
}

impl QuantumEngineType {
    /// Create a boxed quantum engine instance
    #[must_use]
    pub fn create_quantum_engine(self, num_qubits: usize) -> Box<dyn QuantumEngine> {
        match self {
            Self::StateVector => Box::new(StateVecEngine::new(num_qubits)),
            Self::SparseStabilizer => Box::new(SparseStabEngine::new(num_qubits)),
        }
    }

    /// Create a boxed quantum engine instance with a specific seed
    #[must_use]
    pub fn create_quantum_engine_with_seed(
        self,
        num_qubits: usize,
        seed: u64,
    ) -> Box<dyn QuantumEngine> {
        match self {
            Self::StateVector => Box::new(StateVecEngine::with_seed(num_qubits, seed)),
            Self::SparseStabilizer => Box::new(SparseStabEngine::with_seed(num_qubits, seed)),
        }
    }
}

/// Bit vector format for shot results
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BitVecFormat {
    /// Store as `BigUint` (default)
    BigUint,
    /// Store as binary strings
    BinaryString,
}

// Implement From traits for converting noise builders to NoiseModelType

impl From<PassThroughNoiseModelBuilder> for NoiseModelType {
    fn from(builder: PassThroughNoiseModelBuilder) -> Self {
        NoiseModelType::PassThrough(Box::new(builder))
    }
}

impl From<DepolarizingNoiseModelBuilder> for NoiseModelType {
    fn from(builder: DepolarizingNoiseModelBuilder) -> Self {
        NoiseModelType::Depolarizing(Box::new(builder))
    }
}

impl From<BiasedDepolarizingNoiseModelBuilder> for NoiseModelType {
    fn from(builder: BiasedDepolarizingNoiseModelBuilder) -> Self {
        NoiseModelType::BiasedDepolarizing(Box::new(builder))
    }
}

impl From<GeneralNoiseModelBuilder> for NoiseModelType {
    fn from(builder: GeneralNoiseModelBuilder) -> Self {
        NoiseModelType::General(Box::new(builder))
    }
}

/// A built QASM simulation that can be run multiple times
pub struct QasmSimulation {
    engine: QASMEngine,
    seed: Option<u64>,
    workers: usize,
    noise_model: NoiseModelType,
    quantum_engine_type: QuantumEngineType,
    bit_format: BitVecFormat,
    #[cfg(feature = "wasm")]
    foreign_object: Option<Box<dyn crate::foreign_objects::ForeignObject>>,
}

impl QasmSimulation {
    /// Get the configured bit vector format
    #[must_use]
    pub fn bit_format(&self) -> BitVecFormat {
        self.bit_format
    }

    /// Run the simulation with the specified number of shots
    ///
    /// This can be called multiple times to run the same simulation
    /// with different numbers of shots.
    ///
    /// # Errors
    ///
    /// Returns an error if simulation fails.
    pub fn run(&self, shots: usize) -> Result<ShotVec, PecosError> {
        let num_qubits = self.engine.num_qubits();

        // Create fresh engine instance for this run
        #[cfg(feature = "wasm")]
        let mut engine = self.engine.clone();
        #[cfg(not(feature = "wasm"))]
        let engine = self.engine.clone();

        // Initialize and set foreign object if available
        #[cfg(feature = "wasm")]
        if let Some(ref foreign_obj) = self.foreign_object {
            let mut cloned_obj = foreign_obj.clone_box();
            cloned_obj.init()?;
            engine.set_foreign_object(cloned_obj);
        }

        // Get the noise model
        let noise_model = self.noise_model.clone().create_noise_model();

        // Run simulation
        let results = match self.quantum_engine_type {
            QuantumEngineType::StateVector => {
                if let Some(seed) = self.seed {
                    let quantum_engine = StateVecEngine::with_seed(num_qubits, seed);
                    run_qasm_shots(
                        engine,
                        quantum_engine,
                        shots,
                        noise_model,
                        self.workers,
                        Some(seed),
                    )?
                } else {
                    let quantum_engine = StateVecEngine::new(num_qubits);
                    run_qasm_shots(
                        engine,
                        quantum_engine,
                        shots,
                        noise_model,
                        self.workers,
                        None,
                    )?
                }
            }
            QuantumEngineType::SparseStabilizer => {
                if let Some(seed) = self.seed {
                    let quantum_engine = SparseStabEngine::with_seed(num_qubits, seed);
                    run_qasm_shots(
                        engine,
                        quantum_engine,
                        shots,
                        noise_model,
                        self.workers,
                        Some(seed),
                    )?
                } else {
                    let quantum_engine = SparseStabEngine::new(num_qubits);
                    run_qasm_shots(
                        engine,
                        quantum_engine,
                        shots,
                        noise_model,
                        self.workers,
                        None,
                    )?
                }
            }
        };

        Ok(results)
    }
}

/// Builder for configuring and running QASM simulations
#[derive(Debug)]
pub struct QasmSimulationBuilder {
    qasm: String,
    seed: Option<u64>,
    workers: Option<usize>,
    noise_model: Option<NoiseModelType>,
    quantum_engine_type: Option<QuantumEngineType>,
    bit_format: BitVecFormat,
    #[cfg(feature = "wasm")]
    wasm_path: Option<String>,
}

impl QasmSimulationBuilder {
    /// Create a new builder from QASM source
    #[must_use]
    pub fn new(qasm: impl Into<String>) -> Self {
        Self {
            qasm: qasm.into(),
            seed: None,
            workers: None,
            noise_model: None,
            quantum_engine_type: None,
            bit_format: BitVecFormat::BigUint,
            #[cfg(feature = "wasm")]
            wasm_path: None,
        }
    }

    /// Set the random seed
    #[must_use]
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set the number of workers
    #[must_use]
    pub fn workers(mut self, workers: usize) -> Self {
        self.workers = Some(workers);
        self
    }

    /// Use automatic worker count based on available CPUs
    #[must_use]
    pub fn auto_workers(mut self) -> Self {
        self.workers = None;
        self
    }

    /// Set the noise model
    #[must_use]
    pub fn noise<N>(mut self, noise: N) -> Self
    where
        N: Into<NoiseModelType>,
    {
        self.noise_model = Some(noise.into());
        self
    }

    /// Set the quantum engine type
    #[must_use]
    pub fn quantum_engine(mut self, engine: QuantumEngineType) -> Self {
        self.quantum_engine_type = Some(engine);
        self
    }

    /// Configure output to use binary string format
    #[must_use]
    pub fn with_binary_string_format(mut self) -> Self {
        self.bit_format = BitVecFormat::BinaryString;
        self
    }

    /// Set the path to a WebAssembly file (.wasm or .wat) for foreign function calls
    #[cfg(feature = "wasm")]
    #[must_use]
    pub fn wasm(mut self, wasm_path: impl Into<String>) -> Self {
        self.wasm_path = Some(wasm_path.into());
        self
    }

    /// Build the simulation (for reusable execution)
    ///
    /// # Errors
    ///
    /// Returns an error if the QASM cannot be parsed.
    pub fn build(self) -> Result<QasmSimulation, PecosError> {
        let engine = QASMEngine::from_str(&self.qasm)?;

        #[cfg(feature = "wasm")]
        let foreign_object = if let Some(wasm_path) = self.wasm_path {
            use crate::program::QASMProgram;
            use crate::wasm_foreign_object::WasmtimeForeignObject;
            use std::str::FromStr;

            // Create the WASM foreign object
            let wasm_obj = WasmtimeForeignObject::new(wasm_path)?;

            // Get exported functions from WASM module
            let exported_functions = wasm_obj.get_exported_functions();

            // Check if init function exists
            if !exported_functions.contains(&"init".to_string()) {
                return Err(PecosError::Input(
                    "WebAssembly module must export an 'init' function".to_string(),
                ));
            }

            // Parse the QASM program to extract function calls
            let program = QASMProgram::from_str(&self.qasm)?;
            let non_builtin_calls = program.get_non_builtin_function_calls();

            // Validate that all non-builtin function calls exist in WASM module
            for func_name in non_builtin_calls {
                if !exported_functions.contains(&func_name) {
                    return Err(PecosError::Input(format!(
                        "Function '{func_name}' is called in QASM but not exported by WebAssembly module. Available functions: {exported_functions:?}"
                    )));
                }
            }

            Some(Box::new(wasm_obj) as Box<dyn crate::foreign_objects::ForeignObject>)
        } else {
            None
        };

        Ok(QasmSimulation {
            engine,
            seed: self.seed,
            workers: self.workers.unwrap_or(1),
            noise_model: self.noise_model.unwrap_or_default(),
            quantum_engine_type: self
                .quantum_engine_type
                .unwrap_or(QuantumEngineType::SparseStabilizer),
            bit_format: self.bit_format,
            #[cfg(feature = "wasm")]
            foreign_object,
        })
    }

    /// Run the simulation directly with the specified number of shots
    ///
    /// # Errors
    ///
    /// Returns an error if simulation fails.
    pub fn run(self, shots: usize) -> Result<ShotVec, PecosError> {
        let sim = self.build()?;
        sim.run(shots)
    }
}

/// Create a new QASM simulation builder
///
/// This is the primary entry point for running QASM simulations.
///
/// # Example
///
/// ```
/// use pecos_qasm::prelude::*;
///
/// let qasm = r#"
///     OPENQASM 2.0;
///     include "qelib1.inc";
///     qreg q[2];
///     creg c[2];
///     h q[0];
///     cx q[0], q[1];
///     measure q -> c;
/// "#;
///
/// // Run with default settings (no noise)
/// let results = qasm_sim(qasm).run(100).unwrap();
///
/// // Run with noise
/// let noise = GeneralNoiseModel::builder()
///     .with_p1_probability(0.001)
///     .with_p2_probability(0.01);
///
/// let results = qasm_sim(qasm)
///     .seed(42)
///     .noise(noise)
///     .run(1000)
///     .unwrap();
/// ```
#[must_use]
pub fn qasm_sim(qasm: impl Into<String>) -> QasmSimulationBuilder {
    QasmSimulationBuilder::new(qasm)
}

// Private helper function for running shots
fn run_qasm_shots<QE: QuantumEngine + 'static>(
    engine: QASMEngine,
    quantum_engine: QE,
    shots: usize,
    noise_model: Box<dyn NoiseModel>,
    workers: usize,
    seed: Option<u64>,
) -> Result<ShotVec, PecosError> {
    MonteCarloEngine::run_with_engines(
        Box::new(engine),
        noise_model,
        Box::new(quantum_engine),
        shots,
        workers,
        seed, // pass the seed to MonteCarloEngine
    )
}
