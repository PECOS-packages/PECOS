extern crate pecos_sims;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

#[pyfunction]
fn add(a: usize, b: usize) -> PyResult<usize> {
    Ok(pecos_sims::add(a, b))
}

#[pymodule]
fn pecos_pyo3(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add, m)?)?;
    Ok(())
}