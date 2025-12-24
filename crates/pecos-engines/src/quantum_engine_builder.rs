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

//! Quantum Engine Builder traits and implementations
//!
//! This module provides traits and builders for creating quantum engines
//! in a flexible, extensible way that allows different crates to implement
//! their own quantum simulators.

use crate::quantum::{QuantumEngine, SparseStabEngine, StateVecEngine};
use pecos_core::errors::PecosError;

/// Trait for types that can build or configure a quantum engine
///
/// This trait enables lazy evaluation and flexible configuration of quantum engines.
/// Different crates can implement this trait to provide their own quantum simulators.
///
/// # Example
/// ```rust
/// use pecos_engines::quantum_engine_builder::{state_vector, sparse_stabilizer, QuantumEngineBuilder};
///
/// // Using built-in engines
/// let mut state_vec = state_vector();
/// state_vec.set_qubits_if_needed(10);
///
/// let mut sparse_stab = sparse_stabilizer();
/// sparse_stab.set_qubits_if_needed(5);
///
/// // You can build engines from these builders
/// let engine1 = state_vec.build().unwrap();
/// let engine2 = sparse_stab.build().unwrap();
///
/// // Engines are successfully created and ready to use
/// // They implement the QuantumEngine trait for processing quantum operations
/// ```
pub trait QuantumEngineBuilder: Send + Sync {
    /// Build the quantum engine, consuming the builder
    ///
    /// # Errors
    /// Returns an error if the engine cannot be built (e.g., missing required configuration)
    fn build(&mut self) -> Result<Box<dyn QuantumEngine>, PecosError>;

    /// Set the number of qubits if not already set
    /// This allows `SimBuilder` to provide qubits at build time if needed
    fn set_qubits_if_needed(&mut self, num_qubits: usize);
}

/// Trait for types that can be converted into a quantum engine builder
///
/// This enables the sim builder to accept various types that can produce
/// quantum engine builders for lazy evaluation.
pub trait IntoQuantumEngineBuilder: Send + Sync {
    /// The concrete builder type
    type Builder: QuantumEngineBuilder;

    /// Convert into a quantum engine builder
    fn into_quantum_engine_builder(self) -> Self::Builder;
}

/// Builder for state vector quantum engine
#[derive(Debug, Clone, Default)]
pub struct StateVectorEngineBuilder {
    /// Number of qubits (if explicitly set)
    num_qubits: Option<usize>,
    // Future: Could add configuration options here
    // e.g., gpu_enabled, precision, etc.
}

impl StateVectorEngineBuilder {
    /// Create a new state vector engine builder
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of qubits
    #[must_use]
    pub fn qubits(mut self, num_qubits: usize) -> Self {
        self.num_qubits = Some(num_qubits);
        self
    }
}

impl QuantumEngineBuilder for StateVectorEngineBuilder {
    fn build(&mut self) -> Result<Box<dyn QuantumEngine>, PecosError> {
        // Require qubits to be set
        let num_qubits = self.num_qubits.ok_or_else(|| {
            PecosError::Input("Number of qubits not specified for quantum engine".to_string())
        })?;
        Ok(Box::new(StateVecEngine::new(num_qubits)))
    }

    fn set_qubits_if_needed(&mut self, num_qubits: usize) {
        if self.num_qubits.is_none() {
            self.num_qubits = Some(num_qubits);
        }
    }
}

impl IntoQuantumEngineBuilder for StateVectorEngineBuilder {
    type Builder = Self;

    fn into_quantum_engine_builder(self) -> Self::Builder {
        self
    }
}

/// Builder for sparse stabilizer quantum engine
#[derive(Debug, Clone, Default)]
pub struct SparseStabilizerEngineBuilder {
    /// Number of qubits (if explicitly set)
    num_qubits: Option<usize>,
    // Future: Could add configuration options here
    // e.g., tableau_size_hint, optimization_flags, etc.
}

impl SparseStabilizerEngineBuilder {
    /// Create a new sparse stabilizer engine builder
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of qubits
    #[must_use]
    pub fn qubits(mut self, num_qubits: usize) -> Self {
        self.num_qubits = Some(num_qubits);
        self
    }
}

impl QuantumEngineBuilder for SparseStabilizerEngineBuilder {
    fn build(&mut self) -> Result<Box<dyn QuantumEngine>, PecosError> {
        // Require qubits to be set
        let num_qubits = self.num_qubits.ok_or_else(|| {
            PecosError::Input("Number of qubits not specified for quantum engine".to_string())
        })?;
        Ok(Box::new(SparseStabEngine::new(num_qubits)))
    }

    fn set_qubits_if_needed(&mut self, num_qubits: usize) {
        if self.num_qubits.is_none() {
            self.num_qubits = Some(num_qubits);
        }
    }
}

impl IntoQuantumEngineBuilder for SparseStabilizerEngineBuilder {
    type Builder = Self;

    fn into_quantum_engine_builder(self) -> Self::Builder {
        self
    }
}

// Removed IntoQuantumEngine implementation for enum - using builders only

/// Create a state vector quantum engine builder
#[must_use]
pub fn state_vector() -> StateVectorEngineBuilder {
    StateVectorEngineBuilder::new()
}

/// Create a sparse stabilizer quantum engine builder
#[must_use]
pub fn sparse_stabilizer() -> SparseStabilizerEngineBuilder {
    SparseStabilizerEngineBuilder::new()
}

/// Alias for `sparse_stabilizer`
#[must_use]
pub fn sparse_stab() -> SparseStabilizerEngineBuilder {
    sparse_stabilizer()
}
