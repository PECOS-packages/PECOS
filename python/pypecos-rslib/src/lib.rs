mod sim;

use pyo3::prelude::*;

#[pymodule]
fn rslib(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<sim::PyCoinToss>()?;
    Ok(())
}
