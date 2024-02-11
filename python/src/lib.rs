mod sim;

use pyo3::prelude::*;

#[pymodule]
fn pecos_pyo3(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<sim::PyCoinToss>()?;
    Ok(())
}
