//! Simple function interface for QASM simulation
//!
//! This module provides convenience functions for users who prefer
//! function calls over builder patterns.

use crate::qasm_engine;
use pecos_core::errors::PecosError;
use pecos_engines::ClassicalControlEngineBuilder;
use pecos_engines::noise::IntoNoiseModel;
use pecos_engines::quantum_engine_builder::IntoQuantumEngineBuilder;
use pecos_engines::shot_results::ShotVec;
use pecos_programs::Qasm;

/// Run a QASM simulation with a simple function interface
///
/// This is a convenience wrapper around [`qasm_engine`] for users who prefer
/// function calls over builder patterns.
///
/// For more control and a fluent API, consider using [`qasm_engine`] directly:
///
/// ```no_run
/// use pecos_qasm::qasm_engine;
/// use pecos_engines::{ClassicalControlEngineBuilder, noise::DepolarizingNoiseModel};
/// use pecos_programs::Qasm;
/// let qasm = "OPENQASM 2.0; include \"qelib1.inc\"; qreg q[1]; creg c[1]; h q[0]; measure q[0] -> c[0];";
/// let results = qasm_engine().program(Qasm::from_string(qasm)).to_sim().seed(42).run(100)?;
/// # Ok::<(), pecos_core::errors::PecosError>(())
/// ```
///
/// # Parameters
/// - `qasm`: QASM source code as a string
/// - `shots`: Number of simulation shots to run
/// - `noise`: Optional noise model builder
/// - `quantum_engine`: Optional quantum engine builder
/// - `workers`: Optional number of worker threads
/// - `seed`: Optional random seed
///
/// # Returns
/// Results from the simulation shots
///
/// # Errors
/// Returns an error if the QASM cannot be parsed or simulation fails
pub fn run_qasm<N, Q>(
    qasm: impl Into<String>,
    shots: usize,
    noise: Option<N>,
    quantum_engine: Option<Q>,
    workers: Option<usize>,
    seed: Option<u64>,
) -> Result<ShotVec, PecosError>
where
    N: IntoNoiseModel + Send + 'static,
    Q: IntoQuantumEngineBuilder + 'static,
    Q::Builder: Send + 'static,
{
    // Use the SimBuilder for conditional configuration
    let mut builder = qasm_engine().program(Qasm::from_string(qasm)).to_sim();

    if let Some(noise) = noise {
        builder = builder.noise(noise);
    }

    if let Some(e) = quantum_engine {
        builder = builder.quantum(e);
    }

    if let Some(w) = workers {
        builder = builder.workers(w);
    }

    if let Some(s) = seed {
        builder = builder.seed(s);
    }

    builder.run(shots)
}
