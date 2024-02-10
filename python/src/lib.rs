extern crate pecos_sim;

use pyo3::prelude::*;
use pecos_sim::CoinToss;


#[pyclass]
struct PyCoinToss {
    inner: CoinToss,
}

#[pymethods]
impl PyCoinToss {
    #[new]
    fn new(prob: f64, num_qubits: usize) -> Self {
        PyCoinToss {
            inner: CoinToss::new(num_qubits, prob)
        }
    }

    fn h(&mut self, qubit: usize) -> () {
        self.inner.h(qubit)
    }

    fn meas(&mut self, qubit: usize) -> PyResult<bool> {
        Ok(self.inner.meas(qubit))
    }
}

#[pyfunction]
fn add(a: usize, b: usize) -> PyResult<usize> {
    Ok(pecos_sim::add(a, b))
}

#[pymodule]
fn pecos_pyo3(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_class::<PyCoinToss>()?;
    Ok(())
}