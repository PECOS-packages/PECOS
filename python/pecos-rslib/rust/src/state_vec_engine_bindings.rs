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

use crate::byte_message_bindings::PyByteMessage;
use crate::engine_bindings::{PyEngineCommon, PyEngineWrapper, PyQuantumEngineWrapper};
use pecos::prelude::StateVecEngine;
use pyo3::prelude::*;

/// Python wrapper for Rust StateVecEngine to execute ByteMessage circuits
#[pyclass(name = "StateVecEngineRs")]
pub struct PyStateVecEngine {
    inner: StateVecEngine,
}

// Implement the PyEngineWrapper trait for PyStateVecEngine
impl PyEngineWrapper for PyStateVecEngine {
    type EngineType = StateVecEngine;

    fn inner(&self) -> &Self::EngineType {
        &self.inner
    }

    fn inner_mut(&mut self) -> &mut Self::EngineType {
        &mut self.inner
    }
}

// Implement PyQuantumEngineWrapper for PyStateVecEngine
impl PyQuantumEngineWrapper for PyStateVecEngine {}

// Implement PyEngineCommon for PyStateVecEngine
impl PyEngineCommon for PyStateVecEngine {}

#[pymethods]
impl PyStateVecEngine {
    /// Create a new StateVecEngine with the specified number of qubits
    #[new]
    fn new(num_qubits: usize) -> Self {
        Self {
            inner: StateVecEngine::new(num_qubits),
        }
    }

    /// Reset the simulator state
    #[pyo3(text_signature = "($self)")]
    fn reset(&mut self) -> PyResult<()> {
        self.py_reset()
    }

    /// Process a ByteMessage circuit and return the measurement results
    #[pyo3(text_signature = "($self, message)")]
    fn process(&mut self, message: &PyByteMessage) -> PyResult<PyByteMessage> {
        self.py_process(message)
    }

    /// Execute a ByteMessage circuit multiple times and return the measurement results
    #[pyo3(text_signature = "($self, message, shots=1000)")]
    fn run_circuit_with_shots(
        &mut self,
        message: &PyByteMessage,
        shots: Option<usize>,
        py: Python<'_>,
    ) -> PyResult<PyObject> {
        self.py_run_circuit_with_shots(message, shots, py)
    }

    /// Set a specific seed for reproducible randomness
    #[pyo3(text_signature = "($self, seed)")]
    fn set_seed(&mut self, seed: u64) -> PyResult<()> {
        self.py_set_seed(seed)
    }
}
