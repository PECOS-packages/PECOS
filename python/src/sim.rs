extern crate pecos_sim;

use pecos_sim::CoinToss;
use pyo3::prelude::*;

#[pyclass]
pub struct PyCoinToss {
    inner: CoinToss,
}

#[pymethods]
impl PyCoinToss {
    #[new]
    pub fn new(prob: f64, num_qubits: usize) -> Self {
        PyCoinToss {
            inner: CoinToss::new(num_qubits, prob),
        }
    }

    pub fn h(&mut self, qubit: usize) {
        self.inner.h(qubit)
    }

    pub fn meas(&mut self, qubit: usize) -> PyResult<bool> {
        Ok(self.inner.meas(qubit))
    }
}