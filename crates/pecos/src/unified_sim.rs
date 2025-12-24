//! Unified simulation API with automatic engine selection
//!
//! This module provides a convenience wrapper around the lower-level `sim_builder`
//! from pecos-engines, adding automatic engine selection based on program type.

use pecos_core::errors::PecosError;
use pecos_engines::{ClassicalControlEngineBuilder, MonteCarloEngine, SimBuilder, sim_builder};
use pecos_programs::Program;
use pecos_qasm::qasm_engine;
#[cfg(feature = "qis")]
use pecos_qis_core::qis_engine;

/// Extension trait for `SimBuilder` to add program-based methods
pub trait SimBuilderExt {
    /// Set the program and automatically select an appropriate engine
    ///
    /// This method inspects the program type and selects:
    /// - QASM programs → QASM engine
    /// - QIS programs → QIS control engine (Selene Helios interface)
    /// - HUGR programs → QIS control engine (Selene Helios interface)
    /// - WASM/WAT programs → Error (not yet supported)
    /// - PHIR JSON programs → Error (not yet supported)
    ///
    /// The engine can be overridden by calling `.classical()` after this method.
    fn program<P: Into<Program>>(self, program: P) -> ProgrammedSimBuilder;
}

impl SimBuilderExt for SimBuilder {
    fn program<P: Into<Program>>(self, program: P) -> ProgrammedSimBuilder {
        ProgrammedSimBuilder {
            base_builder: self,
            program: program.into(),
            override_classical: false,
        }
    }
}

/// A simulation builder that has a program set and can auto-select engines
pub struct ProgrammedSimBuilder {
    base_builder: SimBuilder,
    program: Program,
    override_classical: bool,
}

impl ProgrammedSimBuilder {
    /// Build the simulation with automatic engine selection
    ///
    /// This selects an engine based on the program type and builds the simulation,
    /// unless a classical engine was already explicitly set.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The program type is not yet supported (WASM, WAT, PHIR JSON, `SeleneInterface`)
    /// - Engine building fails
    pub fn build(self) -> Result<MonteCarloEngine, PecosError> {
        if self.override_classical {
            // Classical engine was already set, just build
            self.base_builder.build()
        } else {
            // Auto-select engine based on program type
            match self.program {
                Program::Qasm(qasm) => self
                    .base_builder
                    .classical(qasm_engine().program(qasm))
                    .build(),
                Program::Qis(qis) => {
                    // Use Selene runtime and Helios interface
                    #[cfg(feature = "qis")]
                    {
                        let selene_runtime = crate::selene_simple_runtime().map_err(|e| {
                            PecosError::Generic(format!("Failed to load Selene runtime: {e}"))
                        })?;
                        let helios_builder = crate::helios_interface_builder();
                        let engine_builder = qis_engine()
                            .runtime(selene_runtime)
                            .interface(helios_builder)
                            .try_program(qis)
                            .map_err(|e| {
                                PecosError::Generic(format!("Failed to load QIS program: {e}"))
                            })?;

                        self.base_builder.classical(engine_builder).build()
                    }
                    #[cfg(not(feature = "llvm"))]
                    {
                        let _ = qis; // Mark as used to avoid warning
                        Err(PecosError::Generic(
                            "QIS programs require Selene and LLVM support. Please rebuild with --features selene,llvm".to_string()
                        ))
                    }
                }
                Program::Hugr(hugr) => {
                    // Use Selene runtime and Helios interface for HUGR programs
                    #[cfg(feature = "qis")]
                    {
                        let selene_runtime = crate::selene_simple_runtime().map_err(|e| {
                            PecosError::Generic(format!("Failed to load Selene runtime: {e}"))
                        })?;
                        let helios_builder = crate::helios_interface_builder();
                        let engine_builder = qis_engine()
                            .runtime(selene_runtime)
                            .interface(helios_builder)
                            .try_program(hugr)
                            .map_err(|e| {
                                PecosError::Generic(format!("Failed to load HUGR program: {e}"))
                            })?;

                        self.base_builder.classical(engine_builder).build()
                    }
                    #[cfg(not(feature = "llvm"))]
                    {
                        let _ = hugr; // Mark as used to avoid warning
                        Err(PecosError::Generic(
                            "HUGR programs require Selene and LLVM support. Please rebuild with --features selene,llvm".to_string()
                        ))
                    }
                }
                Program::Wasm(_) => Err(PecosError::Input(
                    "WASM programs are not yet supported in unified simulation".to_string(),
                )),
                Program::Wat(_) => Err(PecosError::Input(
                    "WAT programs are not yet supported in unified simulation".to_string(),
                )),
                Program::PhirJson(_) => Err(PecosError::Input(
                    "PHIR JSON programs are not yet supported in unified simulation".to_string(),
                )),
                Program::SeleneInterface(_) => Err(PecosError::Input(
                    "SeleneInterface programs are not yet supported in unified simulation"
                        .to_string(),
                )),
            }
        }
    }

    /// Build and run the simulation with automatic engine selection
    ///
    /// This selects an engine based on the program type and runs the simulation,
    /// unless a classical engine was already explicitly set.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The program type is not yet supported (WASM, WAT, PHIR JSON, `SeleneInterface`)
    /// - Engine building or running fails
    pub fn run(self, shots: usize) -> Result<pecos_engines::shot_results::ShotVec, PecosError> {
        if self.override_classical {
            // Classical engine was already set, just run
            self.base_builder.run(shots)
        } else {
            // Auto-select engine based on program type
            match self.program {
                Program::Qasm(qasm) => self
                    .base_builder
                    .classical(qasm_engine().program(qasm))
                    .run(shots),
                Program::Qis(qis) => {
                    // Use Selene runtime and Helios interface
                    #[cfg(feature = "qis")]
                    {
                        let selene_runtime = crate::selene_simple_runtime().map_err(|e| {
                            PecosError::Generic(format!("Failed to load Selene runtime: {e}"))
                        })?;
                        let helios_builder = crate::helios_interface_builder();
                        let engine_builder = qis_engine()
                            .runtime(selene_runtime)
                            .interface(helios_builder)
                            .try_program(qis)
                            .map_err(|e| {
                                PecosError::Generic(format!("Failed to load QIS program: {e}"))
                            })?;

                        self.base_builder.classical(engine_builder).run(shots)
                    }
                    #[cfg(not(feature = "llvm"))]
                    {
                        let _ = qis; // Mark as used to avoid warning
                        Err(PecosError::Generic(
                            "QIS programs require Selene and LLVM support. Please rebuild with --features selene,llvm".to_string()
                        ))
                    }
                }
                Program::Hugr(hugr) => {
                    // Use Selene runtime and Helios interface for HUGR programs
                    #[cfg(feature = "qis")]
                    {
                        let selene_runtime = crate::selene_simple_runtime().map_err(|e| {
                            PecosError::Generic(format!("Failed to load Selene runtime: {e}"))
                        })?;
                        let helios_builder = crate::helios_interface_builder();
                        let engine_builder = qis_engine()
                            .runtime(selene_runtime)
                            .interface(helios_builder)
                            .try_program(hugr)
                            .map_err(|e| {
                                PecosError::Generic(format!("Failed to load HUGR program: {e}"))
                            })?;

                        self.base_builder.classical(engine_builder).run(shots)
                    }
                    #[cfg(not(feature = "llvm"))]
                    {
                        let _ = hugr; // Mark as used to avoid warning
                        Err(PecosError::Generic(
                            "HUGR programs require Selene and LLVM support. Please rebuild with --features selene,llvm".to_string()
                        ))
                    }
                }
                Program::Wasm(_) => Err(PecosError::Input(
                    "WASM programs are not yet supported in unified simulation".to_string(),
                )),
                Program::Wat(_) => Err(PecosError::Input(
                    "WAT programs are not yet supported in unified simulation".to_string(),
                )),
                Program::PhirJson(_) => Err(PecosError::Input(
                    "PHIR JSON programs are not yet supported in unified simulation".to_string(),
                )),
                Program::SeleneInterface(_) => Err(PecosError::Input(
                    "SeleneInterface programs are not yet supported in unified simulation"
                        .to_string(),
                )),
            }
        }
    }

    /// Override the classical engine selection
    ///
    /// This allows you to specify a different engine than the auto-selected one.
    #[must_use]
    pub fn classical<B: ClassicalControlEngineBuilder + Send + 'static>(
        mut self,
        engine_builder: B,
    ) -> Self
    where
        B::Engine: 'static,
    {
        self.base_builder = self.base_builder.classical(engine_builder);
        self.override_classical = true;
        self
    }

    /// Set the random seed (delegates to base builder)
    #[must_use]
    pub fn seed(mut self, seed: u64) -> Self {
        self.base_builder = self.base_builder.seed(seed);
        self
    }

    /// Set the number of worker threads (delegates to base builder)
    #[must_use]
    pub fn workers(mut self, workers: usize) -> Self {
        self.base_builder = self.base_builder.workers(workers);
        self
    }

    /// Use automatic worker count (delegates to base builder)
    #[must_use]
    pub fn auto_workers(mut self) -> Self {
        self.base_builder = self.base_builder.auto_workers();
        self
    }

    /// Enable verbose output (delegates to base builder)
    #[must_use]
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.base_builder = self.base_builder.verbose(verbose);
        self
    }

    /// Set the noise model (delegates to base builder)
    #[must_use]
    pub fn noise<N>(mut self, noise_builder: N) -> Self
    where
        N: pecos_engines::noise::IntoNoiseModel + Send + 'static,
    {
        self.base_builder = self.base_builder.noise(noise_builder);
        self
    }

    /// Set the quantum engine (delegates to base builder)
    #[must_use]
    pub fn quantum<Q>(mut self, quantum_builder: Q) -> Self
    where
        Q: pecos_engines::quantum_engine_builder::IntoQuantumEngineBuilder + 'static,
        Q::Builder: Send + 'static,
    {
        self.base_builder = self.base_builder.quantum(quantum_builder);
        self
    }

    /// Set the number of qubits (delegates to base builder)
    #[must_use]
    pub fn qubits(mut self, num_qubits: usize) -> Self {
        self.base_builder = self.base_builder.qubits(num_qubits);
        self
    }
}

/// Create a simulation builder with a program and automatic engine selection
///
/// This function provides the primary API for quantum simulations in PECOS.
/// It automatically selects the appropriate classical engine based on the program type.
///
/// # Automatic Engine Selection
///
/// - QASM programs → QASM engine
/// - QIS programs → QIS control engine (Selene Helios interface)
/// - HUGR programs → QIS control engine (Selene Helios interface)
/// - Other formats → Error (not yet supported)
///
/// # Examples
///
/// ```rust,no_run
/// use pecos::sim;
/// use pecos_programs::Qasm;
/// use pecos_engines::{sparse_stab, DepolarizingNoise};
///
/// // Automatic engine selection based on program type
/// let qasm_prog = Qasm::from_string("OPENQASM 2.0; qreg q[1]; h q[0];");
/// let results = sim(qasm_prog)
///     .quantum(sparse_stab())
///     .noise(DepolarizingNoise { p: 0.01 })
///     .seed(42)
///     .run(100)?;
/// # Ok::<(), pecos_core::errors::PecosError>(())
/// ```
pub fn sim<P: Into<Program>>(program: P) -> ProgrammedSimBuilder {
    ProgrammedSimBuilder {
        base_builder: sim_builder(),
        program: program.into(),
        override_classical: false,
    }
}
