// Namespace modules for organizational structure
// These modules provide logical groupings for related functionality

use pyo3::prelude::*;
use pyo3::types::PyModule;

/// Register the 'quantum' namespace module
/// Contains quantum simulation backends and builders
pub fn register_quantum_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let quantum = PyModule::new(py, "quantum")?;

    // Add factory functions (references to the engine builders)
    quantum.add("state_vector", parent.getattr("state_vector")?)?;
    quantum.add("sparse_stabilizer", parent.getattr("sparse_stabilizer")?)?;
    quantum.add("sparse_stab", parent.getattr("sparse_stab")?)?;

    // Add builder classes (via getattr from parent)
    quantum.add(
        "StateVectorEngineBuilder",
        parent.getattr("StateVectorEngineBuilder")?,
    )?;
    quantum.add(
        "SparseStabilizerEngineBuilder",
        parent.getattr("SparseStabilizerEngineBuilder")?,
    )?;

    // Register in sys.modules for import statement support
    let sys = py.import("sys")?;
    let modules = sys.getattr("modules")?;
    modules.set_item("_pecos_rslib.quantum", &quantum)?;

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
    modules.set_item("_pecos_rslib.noise", &noise)?;

    parent.add_submodule(&noise)?;
    Ok(())
}

/// Register the 'llvm' namespace module
/// Contains LLVM IR generation compatible with llvmlite API
pub fn register_llvm_namespace_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let llvm = PyModule::new(py, "llvm")?;

    // Add references to ir and binding modules
    llvm.add("ir", parent.getattr("ir")?)?;
    llvm.add("binding", parent.getattr("binding")?)?;

    // Register in sys.modules
    let sys = py.import("sys")?;
    let modules = sys.getattr("modules")?;
    modules.set_item("_pecos_rslib.llvm", &llvm)?;

    parent.add_submodule(&llvm)?;
    Ok(())
}

/// Register all namespace modules
pub fn register_namespace_modules(m: &Bound<'_, PyModule>) -> PyResult<()> {
    register_quantum_module(m)?;
    register_noise_module(m)?;
    register_llvm_namespace_module(m)?;
    Ok(())
}
