// Copyright 2025 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License.You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

use pecos::prelude::*;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// The struct represents the coin toss simulator exposed to Python
///
/// This simulator ignores all quantum gates and returns random measurement results
/// based on a configurable probability. It's useful for debugging classical logic
/// paths and testing error correction protocols with random noise.
#[pyclass(name = "CoinToss")]
pub struct PyCoinToss {
    inner: CoinToss,
}

#[pymethods]
impl PyCoinToss {
    /// Creates a new coin toss simulator with the specified number of qubits
    ///
    /// # Arguments
    /// * `num_qubits` - Number of qubits in the system
    /// * `prob` - Probability of measuring |1⟩ (default: 0.5)
    /// * `seed` - Optional seed for the random number generator
    #[new]
    #[pyo3(signature = (num_qubits, prob=0.5, seed=None))]
    pub fn new(num_qubits: usize, prob: f64, seed: Option<u64>) -> PyResult<Self> {
        if !(0.0..=1.0).contains(&prob) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Probability must be between 0.0 and 1.0, got {prob}"
            )));
        }

        let inner = match seed {
            Some(s) => CoinToss::with_prob_and_seed(num_qubits, prob, Some(s)),
            None => CoinToss::with_prob(num_qubits, prob),
        };

        Ok(PyCoinToss { inner })
    }

    /// Resets the simulator (no-op for coin toss, but maintains interface compatibility)
    fn reset(&mut self) {
        self.inner.reset();
    }

    /// Returns the number of qubits in the system
    #[getter]
    fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    /// Gets the current measurement probability
    #[getter]
    fn prob(&self) -> f64 {
        self.inner.prob()
    }

    /// Sets the measurement probability
    ///
    /// # Arguments
    /// * `prob` - New probability (must be between 0.0 and 1.0)
    #[setter]
    fn set_prob(&mut self, prob: f64) -> PyResult<()> {
        if !(0.0..=1.0).contains(&prob) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Probability must be between 0.0 and 1.0, got {prob}"
            )));
        }
        self.inner.set_prob(prob);
        Ok(())
    }

    /// Sets the seed for reproducible randomness
    ///
    /// # Arguments
    /// * `seed` - Seed value for the random number generator
    fn set_seed(&mut self, seed: u64) {
        self.inner.set_seed(seed);
    }

    /// Executes a single-qubit gate based on the provided symbol and location
    ///
    /// All gates are no-ops in the coin toss simulator.
    ///
    /// # Arguments
    /// * `symbol` - The gate symbol (e.g., "X", "H", "Z") - ignored
    /// * `location` - The qubit index to apply the gate to - ignored
    /// * `params` - Optional parameters for parameterized gates - ignored
    ///
    /// # Returns
    /// Always returns an empty dictionary since all gates are no-ops
    #[allow(clippy::unused_self)]
    fn run_gate_1(
        &mut self,
        _symbol: &str,
        _location: usize,
        _params: Option<Py<PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        // All gates are no-ops in coin toss simulator
        Python::attach(|py| Ok(PyDict::new(py).into()))
    }

    /// Executes a two-qubit gate based on the provided symbol and locations
    ///
    /// All gates are no-ops in the coin toss simulator.
    ///
    /// # Arguments
    /// * `symbol` - The gate symbol (e.g., "CX", "CZ", "SWAP") - ignored
    /// * `location_1` - First qubit index - ignored
    /// * `location_2` - Second qubit index - ignored
    /// * `params` - Optional parameters for parameterized gates - ignored
    ///
    /// # Returns
    /// Always returns an empty dictionary since all gates are no-ops
    #[allow(clippy::unused_self)]
    fn run_gate_2(
        &mut self,
        _symbol: &str,
        _location_1: usize,
        _location_2: usize,
        _params: Option<Py<PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        // All gates are no-ops in coin toss simulator
        Python::attach(|py| Ok(PyDict::new(py).into()))
    }

    /// Performs a measurement in the Z basis
    ///
    /// Returns a random result (0 or 1) based on the configured probability.
    ///
    /// # Arguments
    /// * `location` - The qubit index to measure (ignored - result is always random)
    ///
    /// # Returns
    /// Dictionary containing the measurement result: {location: outcome}
    /// where outcome is 0 or 1 based on the probability
    fn run_measure(&mut self, location: usize) -> PyResult<Py<PyAny>> {
        let result = self.inner.mz(location);
        let outcome = i32::from(result.outcome);

        Python::attach(|py| {
            let dict = PyDict::new(py);
            dict.set_item(location, outcome)?;
            Ok(dict.into())
        })
    }

    /// String representation of the simulator
    fn __repr__(&self) -> String {
        format!(
            "CoinToss(num_qubits={}, prob={})",
            self.inner.num_qubits(),
            self.inner.prob()
        )
    }

    /// String representation of the simulator
    fn __str__(&self) -> String {
        format!(
            "CoinToss simulator with {} qubits, P(|1⟩) = {:.3}",
            self.inner.num_qubits(),
            self.inner.prob()
        )
    }
}
