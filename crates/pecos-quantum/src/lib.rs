// Copyright 2025 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Quantum computing primitives for PECOS.
//!
//! This crate provides quantum computing data structures, including:
//! - [`DagCircuit`] for representing quantum circuits as directed acyclic graphs
//! - [`TickCircuit`] for representing quantum circuits as sequences of parallel time slices
//!
//! # Example - `DagCircuit`
//!
//! ```
//! use pecos_quantum::{DagCircuit, Gate, QubitId};
//!
//! let mut circuit = DagCircuit::new();
//!
//! // Add gates
//! let h = circuit.add_gate(Gate::h(&[0]));
//! let cx = circuit.add_gate(Gate::cx(&[(0, 1)]));
//!
//! // Connect H to CX with a qubit wire
//! circuit.connect(h, cx, QubitId::from(0)).unwrap();
//!
//! // Get circuit properties
//! assert_eq!(circuit.gate_count(), 2);
//! assert_eq!(circuit.wire_count(), 1);
//! ```
//!
//! # Example - `TickCircuit`
//!
//! ```
//! use pecos_quantum::TickCircuit;
//!
//! let mut circuit = TickCircuit::new();
//!
//! // Each tick() returns a handle for adding gates
//! // Regular gates chain, but preps/measurements break the chain
//! circuit.tick().pz(&[0]);              // Tick 0: Prepare q0 (breaks chain)
//! circuit.tick().pz(&[1]);              // Tick 1: Prepare q1 (breaks chain)
//! circuit.tick().h(&[0]).x(&[1]);       // Tick 2: H and X chain together
//! circuit.tick().cx(&[(0, 1)]);         // Tick 3: CNOT
//! circuit.tick().mz(&[0]);              // Tick 4: Measure q0 (breaks chain)
//! circuit.tick().mz(&[1]);              // Tick 5: Measure q1 (breaks chain)
//!
//! assert_eq!(circuit.num_ticks(), 6);
//!
//! // All methods accept slices for bulk operations:
//! let mut circuit2 = TickCircuit::new();
//! circuit2.tick().pz(&[0, 1, 2, 3]);    // Prep multiple qubits
//! circuit2.tick().h(&[0, 1, 2, 3]);     // H on multiple qubits
//! circuit2.tick().cx(&[(0, 1), (2, 3)]); // Multiple CX gates
//! circuit2.tick().mz(&[0, 1, 2, 3]);    // Measure multiple qubits
//! ```

mod circuit;
mod circuit_display;
mod dag_circuit;
pub mod pass;
pub mod pauli_group;
pub mod pauli_sequence;
pub mod pauli_set;
pub mod stabilizer_group;
mod tick_circuit;
pub mod tick_circuit_soa;
pub mod unitary_matrix;

#[cfg(feature = "hugr")]
pub mod hugr_convert;

pub use circuit::{Circuit, CircuitMut, GateHandle, GateView};
pub use dag_circuit::{
    Attribute, DagCircuit, DagTraversalIndex, MeasureHandle, PrepHandle, TraversalWorkBuffers,
};
pub use tick_circuit::{
    CustomGateError, GateSignatureMismatchError, QubitConflictError, Tick, TickCircuit, TickHandle,
    TickMeasureHandle, TickPrepHandle,
};
pub use tick_circuit_soa::{
    CircuitIndexes, GateBatch, GateId, GateStorage, MetadataStorage, TickBatches, TickCircuitSoA,
    TickCircuitSoABuilder, TickGateGroups,
};

// Re-export commonly used types from dependencies
pub use pecos_core::gate_type::GateType;
pub use pecos_core::{ClassicalBitId, Gate, QubitId, TimeScale, TimeUnits};
pub use pecos_num::dag::DagWouldCycleError;

// Re-export operator matrix types for convenient method-style matrix conversion
pub use unitary_matrix::{ToMatrix, UnitaryMatrix};

// Pauli collection and stabilizer group types
pub use pauli_group::{PauliGroup, PauliGroupError};
pub use pauli_sequence::{F2Matrix, PauliSequence};
pub use pauli_set::PauliSet;
pub use stabilizer_group::{PauliStabilizerGroup, PauliStabilizerGroupError};

// Re-export HUGR types when the feature is enabled
#[cfg(feature = "hugr")]
pub use tket::hugr::Hugr;
