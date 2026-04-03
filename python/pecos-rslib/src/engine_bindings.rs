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

//! Common traits and functionality for engine bindings
//!
//! This module provides common functionality for binding Rust engines to Python.
//! It defines traits that both concrete engines should implement.

use crate::byte_message_bindings::PyByteMessage;
use crate::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyList;

/// Trait for engine wrappers to implement
///
/// This trait defines the common interface that all engine wrappers should implement.
/// It provides a way to access the inner engine.
pub trait PyEngineWrapper {
    /// The type of the inner engine
    type EngineType: Engine<Input = ByteMessage, Output = ByteMessage> + 'static;

    /// Get a reference to the inner engine
    ///
    /// This method is kept for API completeness, even though it's not currently used.
    #[allow(dead_code)]
    fn inner(&self) -> &Self::EngineType;

    /// Get a mutable reference to the inner engine
    fn inner_mut(&mut self) -> &mut Self::EngineType;
}

/// Trait for quantum engine wrappers to implement
///
/// This trait extends `PyEngineWrapper` with additional methods specific to quantum engines.
pub trait PyQuantumEngineWrapper: PyEngineWrapper
where
    Self::EngineType: QuantumEngine,
{
    /// Set a specific seed for reproducible randomness
    fn py_set_seed(&mut self, seed: u64) -> PyResult<()> {
        self.inner_mut().set_seed(seed);
        Ok(())
    }
}

/// Common implementation for all engine wrapper types
///
/// This trait provides default implementations for common methods.
pub trait PyEngineCommon: PyEngineWrapper {
    /// Reset the engine state
    fn py_reset(&mut self) -> PyResult<()> {
        self.inner_mut().reset().map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to reset engine in Python bindings: {e}"))
        })
    }

    /// Process a `ByteMessage` and return the result
    fn py_process(&mut self, message: &PyByteMessage) -> PyResult<PyByteMessage> {
        let result = self
            .inner_mut()
            .process(message.clone_inner())
            .map_err(|e| {
                PyRuntimeError::new_err(format!(
                    "Failed to process message in Python bindings: {e}"
                ))
            })?;

        Ok(PyByteMessage::from_byte_message(result))
    }

    /// Execute a circuit multiple times and return the measurement results
    fn py_run_circuit_with_shots(
        &mut self,
        message: &PyByteMessage,
        shots: Option<usize>,
        py: Python<'_>,
    ) -> PyResult<Py<PyAny>> {
        let num_shots = shots.unwrap_or(1000);
        let result_list = PyList::empty(py);

        for _ in 0..num_shots {
            // Reset the engine
            self.py_reset()?;

            // Process the circuit
            let result = self.py_process(message)?;

            // Get the measurement results
            let measurements = result.measurement_results(py)?;
            result_list.append(measurements)?;
        }

        Ok(result_list.into())
    }
}
