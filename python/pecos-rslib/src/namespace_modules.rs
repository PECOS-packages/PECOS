// Namespace modules for organizational structure
// These modules provide logical groupings for related functionality

use pyo3::prelude::*;
use pyo3::types::PyModule;

/// Register the 'quantum' namespace module
/// Contains quantum circuit types and simulation backends
pub fn register_quantum_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let quantum = PyModule::new(py, "quantum")?;

    // Add circuit representation types
    quantum.add("DagCircuit", parent.getattr("DagCircuit")?)?;
    quantum.add("Gate", parent.getattr("Gate")?)?;
    quantum.add("GateType", parent.getattr("GateType")?)?;
    quantum.add("QubitId", parent.getattr("QubitId")?)?;
    quantum.add("Tick", parent.getattr("Tick")?)?;
    quantum.add("TickCircuit", parent.getattr("TickCircuit")?)?;
    quantum.add("TickHandle", parent.getattr("TickHandle")?)?;
    quantum.add("TickPrepHandle", parent.getattr("TickPrepHandle")?)?;
    quantum.add("TickMeasureHandle", parent.getattr("TickMeasureHandle")?)?;
    quantum.add(
        "DagCircuitWouldCycleError",
        parent.getattr("DagCircuitWouldCycleError")?,
    )?;

    // Add HUGR conversion functions and exception
    quantum.add(
        "HugrConversionError",
        parent.getattr("HugrConversionError")?,
    )?;
    quantum.add("QubitConflictError", parent.getattr("QubitConflictError")?)?;
    quantum.add(
        "hugr_to_dag_circuit",
        parent.getattr("hugr_to_dag_circuit")?,
    )?;
    quantum.add(
        "hugr_op_to_gate_type",
        parent.getattr("hugr_op_to_gate_type")?,
    )?;
    quantum.add(
        "gate_type_to_hugr_op",
        parent.getattr("gate_type_to_hugr_op")?,
    )?;
    quantum.add(
        "is_quantum_operation",
        parent.getattr("is_quantum_operation")?,
    )?;

    // Add factory functions (references to the engine builders)
    quantum.add("state_vector", parent.getattr("state_vector")?)?;
    quantum.add("sparse_stab", parent.getattr("sparse_stab")?)?;
    quantum.add("stabilizer", parent.getattr("stabilizer")?)?;
    quantum.add("clifford_rz", parent.getattr("clifford_rz")?)?;
    quantum.add("density_matrix", parent.getattr("density_matrix")?)?;
    quantum.add("coin_toss", parent.getattr("coin_toss")?)?;

    // Add builder classes (via getattr from parent)
    quantum.add(
        "StateVectorEngineBuilder",
        parent.getattr("StateVectorEngineBuilder")?,
    )?;
    quantum.add(
        "SparseStabEngineBuilder",
        parent.getattr("SparseStabEngineBuilder")?,
    )?;
    quantum.add(
        "StabilizerEngineBuilder",
        parent.getattr("StabilizerEngineBuilder")?,
    )?;
    quantum.add(
        "CliffordRzEngineBuilder",
        parent.getattr("CliffordRzEngineBuilder")?,
    )?;
    quantum.add(
        "DensityMatrixEngineBuilder",
        parent.getattr("DensityMatrixEngineBuilder")?,
    )?;
    quantum.add(
        "CoinTossEngineBuilder",
        parent.getattr("CoinTossEngineBuilder")?,
    )?;

    // Register in sys.modules for import statement support
    let sys = py.import("sys")?;
    let modules = sys.getattr("modules")?;
    modules.set_item("pecos_rslib.quantum", &quantum)?;

    parent.add_submodule(&quantum)?;
    Ok(())
}

/// Register the 'noise' namespace module
/// Contains noise model builders and factory functions
pub fn register_noise_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let noise = PyModule::new(py, "noise")?;

    // Add factory functions with both short and long names
    let general_fn = parent.getattr("general_noise")?;
    let depolarizing_fn = parent.getattr("depolarizing_noise")?;
    let biased_fn = parent.getattr("biased_depolarizing_noise")?;

    noise.add("general", &general_fn)?;
    noise.add("depolarizing", &depolarizing_fn)?;
    noise.add("biased_depolarizing", &biased_fn)?;
    noise.add("general_noise", &general_fn)?;
    noise.add("depolarizing_noise", &depolarizing_fn)?;
    noise.add("biased_depolarizing_noise", &biased_fn)?;

    // Add builder classes (via getattr from parent)
    noise.add(
        "GeneralNoiseModelBuilder",
        parent.getattr("GeneralNoiseModelBuilder")?,
    )?;
    noise.add(
        "DepolarizingNoiseModelBuilder",
        parent.getattr("DepolarizingNoiseModelBuilder")?,
    )?;
    noise.add(
        "BiasedDepolarizingNoiseModelBuilder",
        parent.getattr("BiasedDepolarizingNoiseModelBuilder")?,
    )?;

    // Register in sys.modules
    let sys = py.import("sys")?;
    let modules = sys.getattr("modules")?;
    modules.set_item("pecos_rslib.noise", &noise)?;

    parent.add_submodule(&noise)?;
    Ok(())
}

/// Register all namespace modules
pub fn register_namespace_modules(m: &Bound<'_, PyModule>) -> PyResult<()> {
    register_quantum_module(m)?;
    register_noise_module(m)?;
    Ok(())
}
