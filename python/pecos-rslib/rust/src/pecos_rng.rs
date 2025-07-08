//! PyO3 bindings for PCG32 RNG functions

use pyo3::prelude::*;

/// Generate a random 32-bit unsigned integer
#[pyfunction]
fn pcg32_random() -> u32 {
    unsafe { pecos_clibs::rng::pcg32_random() }
}

/// Generate a random floating point number in [0.0, 1.0)
#[pyfunction]
fn pcg32_frandom() -> f64 {
    unsafe { pecos_clibs::rng::pcg32_frandom() }
}

/// Generate a bounded random number in [0, bound)
#[pyfunction]
fn pcg32_boundedrand(bound: u32) -> u32 {
    unsafe { pecos_clibs::rng::pcg32_boundedrand(bound) }
}

/// Seed the random number generator
#[pyfunction]
fn pcg32_srandom(seq: u64) {
    unsafe { pecos_clibs::rng::pcg32_srandom(seq) }
}

/// Register the pecos_rng submodule
pub fn register_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent_module.py(), "pecos_rng")?;
    m.add_function(wrap_pyfunction!(pcg32_random, &m)?)?;
    m.add_function(wrap_pyfunction!(pcg32_frandom, &m)?)?;
    m.add_function(wrap_pyfunction!(pcg32_boundedrand, &m)?)?;
    m.add_function(wrap_pyfunction!(pcg32_srandom, &m)?)?;
    parent_module.add_submodule(&m)?;
    Ok(())
}