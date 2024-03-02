mod sim;

use pyo3::prelude::*;

#[pymodule]
fn pyo3pecos(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<sim::PyCoinToss>()?;
    Ok(())
}
