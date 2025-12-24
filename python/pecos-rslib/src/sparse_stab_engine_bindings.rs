// Copyright 2025 The PECOS Developers
use pecos::prelude::*;
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
use pyo3::prelude::*;

/// Python wrapper for Rust `SparseStabEngine` to execute `ByteMessage` circuits with Clifford gates
#[pyclass(name = "SparseStabEngine")]
pub struct PySparseStabEngine {
    inner: SparseStabEngine,
}

// Implement the PyEngineWrapper trait for PySparseStabEngine
impl PyEngineWrapper for PySparseStabEngine {
    type EngineType = SparseStabEngine;

    fn inner(&self) -> &Self::EngineType {
        &self.inner
    }

    fn inner_mut(&mut self) -> &mut Self::EngineType {
        &mut self.inner
    }
}

// Implement PyQuantumEngineWrapper for PySparseStabEngine
impl PyQuantumEngineWrapper for PySparseStabEngine {}

// Implement PyEngineCommon for PySparseStabEngine
impl PyEngineCommon for PySparseStabEngine {}

#[pymethods]
impl PySparseStabEngine {
    /// Create a new `SparseStabEngine` with the specified number of qubits
    #[new]
    fn new(num_qubits: usize) -> Self {
        Self {
            inner: SparseStabEngine::new(num_qubits),
        }
    }

    /// Reset the simulator state
    #[pyo3(text_signature = "($self)")]
    fn reset(&mut self) -> PyResult<()> {
        self.py_reset()
    }

    /// Process a `ByteMessage` circuit and return the measurement results
    #[pyo3(text_signature = "($self, message)")]
    fn process(&mut self, message: &PyByteMessage) -> PyResult<PyByteMessage> {
        self.py_process(message)
    }

    /// Execute a `ByteMessage` circuit multiple times and return the measurement results
    #[pyo3(text_signature = "($self, message, shots=1000)")]
    fn run_circuit_with_shots(
        &mut self,
        message: &PyByteMessage,
        shots: Option<usize>,
        py: Python<'_>,
    ) -> PyResult<Py<PyAny>> {
        self.py_run_circuit_with_shots(message, shots, py)
    }

    /// Set a specific seed for reproducible randomness
    #[pyo3(text_signature = "($self, seed)")]
    fn set_seed(&mut self, seed: u64) -> PyResult<()> {
        self.py_set_seed(seed)
    }
}
