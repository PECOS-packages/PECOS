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

//! Trait abstraction for quantum circuits.
//!
//! This module provides the [`Circuit`] trait, which defines a common interface
//! for quantum circuit representations. This allows generic code to work with different
//! circuit types like [`DagCircuit`](crate::DagCircuit) and [`SimpleHugr`](crate::hugr_convert::SimpleHugr).

use std::collections::BTreeMap;

use pecos_core::{ClassicalBitId, Gate, QubitId};
use pecos_num::graph::Attribute;

/// A read-only view of a gate in a quantum circuit.
///
/// This provides access to gate information without exposing the underlying
/// storage details of the circuit implementation.
#[derive(Debug, Clone)]
pub struct GateView<'a> {
    /// The gate data.
    pub gate: &'a Gate,
    /// The node/gate index in the circuit.
    pub index: usize,
}

/// A handle to a gate in a circuit, used for referencing gates across operations.
pub type GateHandle = usize;

/// Trait for read-only access to quantum circuits.
///
/// Common interface for different quantum circuit representations,
/// allowing generic algorithms to work with any circuit type.
///
/// # Implementors
///
/// - [`DagCircuit`](crate::DagCircuit): Native DAG-based circuit representation
/// - [`SimpleHugr`](crate::hugr_convert::SimpleHugr): Validated HUGR wrapper (when `hugr` feature enabled)
pub trait Circuit {
    // ==================== Basic properties ====================

    /// Returns the number of gates in the circuit.
    fn gate_count(&self) -> usize;

    /// Returns the number of wires (edges) in the circuit.
    fn wire_count(&self) -> usize;

    /// Returns all unique qubits used in the circuit.
    fn qubits(&self) -> Vec<QubitId>;

    /// Returns the circuit width (number of unique qubits).
    fn width(&self) -> usize {
        self.qubits().len()
    }

    /// Returns the circuit depth (longest path from root to leaf).
    fn depth(&self) -> usize;

    // ==================== Gate access ====================

    /// Returns a reference to the gate at the given index.
    fn gate(&self, index: GateHandle) -> Option<&Gate>;

    /// Returns all node/gate indices in the circuit.
    fn nodes(&self) -> Vec<GateHandle>;

    /// Returns an iterator over all gates as `(index, gate)` pairs.
    fn iter_gates(&self) -> Box<dyn Iterator<Item = GateView<'_>> + '_>;

    /// Returns gates in topological order.
    fn topological_order(&self) -> Vec<GateHandle>;

    /// Returns an iterator over gates in topological order.
    fn iter_gates_topo(&self) -> Box<dyn Iterator<Item = GateView<'_>> + '_>;

    // ==================== Graph structure ====================

    /// Returns the predecessor gates (gates with wires into this gate).
    fn predecessors(&self, gate: GateHandle) -> Vec<GateHandle>;

    /// Returns the successor gates (gates with wires from this gate).
    fn successors(&self, gate: GateHandle) -> Vec<GateHandle>;

    /// Returns the root gates (gates with no incoming wires).
    fn roots(&self) -> Vec<GateHandle>;

    /// Returns the leaf gates (gates with no outgoing wires).
    fn leaves(&self) -> Vec<GateHandle>;

    // ==================== Qubit-based queries ====================

    /// Returns all gates acting on a specific qubit.
    fn gates_on_qubit(&self, qubit: QubitId) -> Vec<GateHandle>;

    /// Returns gates acting on a specific qubit in topological order.
    fn qubit_timeline(&self, qubit: QubitId) -> Vec<GateHandle>;

    // ==================== Attribute access ====================

    /// Returns the circuit-level attributes.
    fn circuit_attrs(&self) -> &BTreeMap<String, Attribute>;

    /// Returns a specific circuit-level attribute.
    fn circuit_attr(&self, key: &str) -> Option<&Attribute> {
        self.circuit_attrs().get(key)
    }

    /// Returns the attributes for a specific gate.
    fn gate_attrs(&self, gate: GateHandle) -> Option<&BTreeMap<String, Attribute>>;

    /// Returns a specific attribute for a gate.
    fn gate_attr(&self, gate: GateHandle, key: &str) -> Option<&Attribute> {
        self.gate_attrs(gate).and_then(|attrs| attrs.get(key))
    }

    // ==================== Classical bit access ====================

    /// Returns the number of classical bits in the circuit.
    fn num_cbits(&self) -> usize {
        0
    }

    /// Returns the classical bit that receives a measurement outcome for a gate.
    fn measurement_target(&self, _gate: GateHandle) -> Option<ClassicalBitId> {
        None
    }

    /// Returns the condition (classical bit, expected value) for a conditional gate.
    fn condition(&self, _gate: GateHandle) -> Option<(ClassicalBitId, bool)> {
        None
    }
}

/// Trait for mutable operations on quantum circuits.
///
/// This extends [`Circuit`] with methods for modifying the circuit.
pub trait CircuitMut: Circuit {
    /// Adds a gate to the circuit.
    ///
    /// Returns the handle for the newly added gate.
    fn add_gate(&mut self, gate: Gate) -> GateHandle;

    /// Removes a gate from the circuit.
    ///
    /// Returns the removed gate if it existed.
    fn remove_gate(&mut self, gate: GateHandle) -> Option<Gate>;

    /// Sets a circuit-level attribute.
    fn set_circuit_attr(&mut self, key: impl Into<String>, value: Attribute);

    /// Sets multiple circuit-level attributes.
    fn set_circuit_attrs(&mut self, attrs: BTreeMap<String, Attribute>);

    /// Sets an attribute on a specific gate.
    ///
    /// Returns `true` if the gate exists.
    fn set_gate_attr(&mut self, gate: GateHandle, key: impl Into<String>, value: Attribute)
    -> bool;

    /// Sets multiple attributes on a specific gate.
    ///
    /// Returns `true` if the gate exists.
    fn set_gate_attrs(&mut self, gate: GateHandle, attrs: BTreeMap<String, Attribute>) -> bool;
}
