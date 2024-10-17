mod sparse_sim;
use sparse_sim::SparseSim;

use pyo3::prelude::*;

#[pymodule]
fn _pecos_rslib(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SparseSim>()?;
    Ok(())
}
