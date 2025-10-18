use crate::{
    prelude::QasmSimulationBuilder,
    simulation::{NoiseModelType, QuantumEngineType, qasm_sim},
};
use pecos_core::errors::PecosError;
use pecos_engines::shot_results::ShotVec;

/// Run a QASM simulation with a simple function interface
///
/// This is a convenience wrapper around [`qasm_sim`] for users who prefer
/// function calls over builder patterns. It provides the same functionality
/// in a more traditional function interface.
///
/// For more control and a fluent API, consider using [`qasm_sim`] directly:
///
/// ```
/// use pecos_qasm::prelude::*;
/// use pecos_engines::noise::DepolarizingNoiseModel;
/// let qasm = "OPENQASM 2.0; include \"qelib1.inc\"; qreg q[1]; creg c[1]; h q[0]; measure q[0] -> c[0];";
/// let results = qasm_sim(qasm).seed(42).run(100)?;
/// assert_eq!(results.len(), 100);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Parameters
///
/// * `qasm` - QASM code as a string
/// * `shots` - Number of shots to run
/// * `noise` - Noise configuration (any noise model builder)
/// * `quantum_engine` - Optional quantum engine type (defaults to appropriate engine for circuit)
/// * `workers` - Optional number of workers for parallelization (defaults to 1)
/// * `seed` - Optional seed for reproducibility
///
/// # Returns
///
/// A [`ShotVec`] containing the simulation results. This can be converted to
/// [`ShotMap`](crate::shot_results::ShotMap) for columnar access via `try_as_shot_map()`
///
/// # Example
///
/// ```
/// # use pecos_qasm::prelude::*;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
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
/// // Simple usage - ideal simulation (no noise)
/// let results = run_qasm(
///     qasm,
///     100,
///     PassThroughNoiseModel::builder(),
///     None,
///     None,
///     None
/// )?;
/// assert_eq!(results.len(), 100);
///
/// // With depolarizing noise
/// let noise = DepolarizingNoiseModel::builder()
///     .with_uniform_probability(0.01);
/// let results = run_qasm(
///     qasm,
///     1000,
///     noise,
///     Some(QuantumEngineType::StateVector),
///     Some(4),  // workers
///     Some(42), // seed
/// )?;
/// assert_eq!(results.len(), 1000);
///
/// // With custom depolarizing noise parameters
/// let custom_noise = DepolarizingNoiseModel::builder()
///     .with_prep_probability(0.001)
///     .with_meas_probability(0.01)
///     .with_p1_probability(0.005)
///     .with_p2_probability(0.02);
/// let results = run_qasm(qasm, 100, custom_noise, None, None, Some(42))?;
///
/// // Check results are Bell states
/// let shot_map = results.try_as_shot_map()?;
/// let values = shot_map.try_bits_as_u64("c")?;
/// for val in &values[..10] {  // Check first 10
///     assert!(*val == 0 || *val == 3 || *val == 1 || *val == 2); // With noise, all outcomes possible
/// }
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns a [`PecosError`] if:
/// - QASM parsing fails due to syntax errors or unsupported operations
/// - Simulation fails due to invalid quantum operations
/// - Memory allocation fails for large circuits
pub fn run_qasm<N>(
    qasm: &str,
    shots: usize,
    noise: N,
    quantum_engine: Option<QuantumEngineType>,
    workers: Option<usize>,
    seed: Option<u64>,
) -> Result<ShotVec, PecosError>
where
    N: Into<NoiseModelType>,
{
    let mut builder: QasmSimulationBuilder = qasm_sim(qasm).noise(noise);

    if let Some(e) = quantum_engine {
        builder = builder.quantum_engine(e);
    }

    if let Some(w) = workers {
        builder = builder.workers(w);
    }

    if let Some(s) = seed {
        builder = builder.seed(s);
    }

    builder.run(shots)
}
