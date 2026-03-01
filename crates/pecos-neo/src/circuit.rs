// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! Integration with pecos-quantum circuit representations.
//!
//! This module provides conversions between pecos-quantum's circuit representations
//! ([`TickCircuit`], [`DagCircuit`]) and pecos-neo's [`CommandQueue`].
//!
//! # Example
//!
//! ```no_run
//! use pecos_neo::prelude::*;
//! use pecos_quantum::TickCircuit;
//! use pecos_qsim::SparseStab;
//!
//! // Build a circuit using TickCircuit
//! let mut circuit = TickCircuit::new();
//! circuit.tick().pz(&[0, 1]);
//! circuit.tick().h(&[0]);
//! circuit.tick().cx(&[(0, 1)]);
//! circuit.tick().mz(&[0, 1]);
//!
//! // Convert to CommandQueue and execute
//! let commands = CommandQueue::from(&circuit);
//! let mut state = SparseStab::new(2);
//! let mut runner = CircuitRunner::<SparseStab>::new().with_seed(42);
//! let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();
//! ```

use crate::command::{CommandQueue, GateCommand, GateType};
use pecos_core::{Angle64, Gate, QubitId, TimeUnits};
use pecos_quantum::{DagCircuit, TickCircuit};
use smallvec::SmallVec;

// ============================================================================
// GateType Conversion
// ============================================================================

impl From<pecos_core::gate_type::GateType> for GateType {
    #[allow(clippy::match_same_arms)] // Unknown gate types explicitly map to I
    fn from(gt: pecos_core::gate_type::GateType) -> Self {
        use pecos_core::gate_type::GateType as CoreGT;
        match gt {
            CoreGT::I => Self::I,
            CoreGT::X => Self::X,
            CoreGT::Y => Self::Y,
            CoreGT::Z => Self::Z,
            CoreGT::H => Self::H,
            CoreGT::SX => Self::SX,
            CoreGT::SXdg => Self::SXdg,
            CoreGT::SY => Self::SY,
            CoreGT::SYdg => Self::SYdg,
            CoreGT::SZ => Self::SZ,
            CoreGT::SZdg => Self::SZdg,
            CoreGT::T => Self::T,
            CoreGT::Tdg => Self::Tdg,
            CoreGT::RX => Self::RX,
            CoreGT::RY => Self::RY,
            CoreGT::RZ => Self::RZ,
            CoreGT::U => Self::U,
            CoreGT::R1XY => Self::R1XY,
            CoreGT::CX => Self::CX,
            CoreGT::CY => Self::CY,
            CoreGT::CZ => Self::CZ,
            CoreGT::SZZ => Self::SZZ,
            CoreGT::SZZdg => Self::SZZdg,
            CoreGT::SWAP => Self::SWAP,
            CoreGT::CRZ => Self::CRZ,
            CoreGT::RXX => Self::RXX,
            CoreGT::RYY => Self::RYY,
            CoreGT::RZZ => Self::RZZ,
            CoreGT::CCX => Self::CCX,
            CoreGT::MZ => Self::MZ,
            CoreGT::MeasureLeaked => Self::MeasureLeaked,
            CoreGT::MeasureFree => Self::MeasureFree,
            CoreGT::PZ => Self::PZ,
            CoreGT::QAlloc => Self::QAlloc,
            CoreGT::QFree => Self::QFree,
            CoreGT::Idle => Self::Idle,
            // Any unknown gate types default to identity
            _ => Self::I,
        }
    }
}

impl From<GateType> for pecos_core::gate_type::GateType {
    fn from(gt: GateType) -> Self {
        use pecos_core::gate_type::GateType as CoreGT;
        match gt {
            GateType::I => CoreGT::I,
            GateType::X => CoreGT::X,
            GateType::Y => CoreGT::Y,
            GateType::Z => CoreGT::Z,
            GateType::H => CoreGT::H,
            GateType::SX => CoreGT::SX,
            GateType::SXdg => CoreGT::SXdg,
            GateType::SY => CoreGT::SY,
            GateType::SYdg => CoreGT::SYdg,
            GateType::SZ => CoreGT::SZ,
            GateType::SZdg => CoreGT::SZdg,
            GateType::T => CoreGT::T,
            GateType::Tdg => CoreGT::Tdg,
            GateType::RX => CoreGT::RX,
            GateType::RY => CoreGT::RY,
            GateType::RZ => CoreGT::RZ,
            GateType::U => CoreGT::U,
            GateType::R1XY => CoreGT::R1XY,
            GateType::CX => CoreGT::CX,
            GateType::CY => CoreGT::CY,
            GateType::CZ => CoreGT::CZ,
            GateType::SZZ => CoreGT::SZZ,
            GateType::SZZdg => CoreGT::SZZdg,
            GateType::SWAP => CoreGT::SWAP,
            GateType::CRZ => CoreGT::CRZ,
            GateType::RXX => CoreGT::RXX,
            GateType::RYY => CoreGT::RYY,
            GateType::RZZ => CoreGT::RZZ,
            GateType::CCX => CoreGT::CCX,
            GateType::MZ => CoreGT::MZ,
            GateType::MeasureLeaked => CoreGT::MeasureLeaked,
            GateType::MeasureFree => CoreGT::MeasureFree,
            GateType::PZ => CoreGT::PZ,
            GateType::QAlloc => CoreGT::QAlloc,
            GateType::QFree => CoreGT::QFree,
            GateType::Idle => CoreGT::Idle,
        }
    }
}

// ============================================================================
// Gate to GateCommand Conversion
// ============================================================================

impl From<&Gate> for GateCommand {
    fn from(gate: &Gate) -> Self {
        let gate_type: GateType = gate.gate_type.into();
        let qubits: SmallVec<[QubitId; 4]> = gate.qubits.iter().copied().collect();

        // Handle idle gates specially - they store duration in params
        if gate_type == GateType::Idle
            && let Some(&duration) = gate.params.first()
        {
            return GateCommand::idle(qubits[0], TimeUnits::new(duration as u64));
        }

        // Copy angles
        let angles: SmallVec<[Angle64; 2]> = gate.angles.iter().copied().collect();

        GateCommand {
            gate_type,
            qubits,
            angles,
        }
    }
}

impl From<Gate> for GateCommand {
    fn from(gate: Gate) -> Self {
        (&gate).into()
    }
}

// ============================================================================
// TickCircuit to CommandQueue Conversion
// ============================================================================

impl From<&TickCircuit> for CommandQueue {
    /// Convert a `TickCircuit` to a `CommandQueue`.
    ///
    /// Gates are added in tick order - all gates from tick 0, then tick 1, etc.
    /// Within each tick, gates are added in the order they appear.
    fn from(circuit: &TickCircuit) -> Self {
        let mut queue = CommandQueue::new();

        for tick in circuit.ticks() {
            for gate in tick.gates() {
                queue.push(gate.into());
            }
        }

        queue
    }
}

impl From<TickCircuit> for CommandQueue {
    fn from(circuit: TickCircuit) -> Self {
        (&circuit).into()
    }
}

// ============================================================================
// DagCircuit to CommandQueue Conversion
// ============================================================================

impl From<&DagCircuit> for CommandQueue {
    /// Convert a `DagCircuit` to a `CommandQueue`.
    ///
    /// Gates are added in topological order, ensuring that dependencies
    /// are respected.
    fn from(circuit: &DagCircuit) -> Self {
        let mut queue = CommandQueue::new();

        // Get gates in topological order
        for node_id in circuit.topological_order() {
            if let Some(gate) = circuit.gate(node_id) {
                queue.push(gate.into());
            }
        }

        queue
    }
}

impl From<DagCircuit> for CommandQueue {
    fn from(circuit: DagCircuit) -> Self {
        (&circuit).into()
    }
}

// ============================================================================
// CommandQueue to TickCircuit Conversion (Round-trip support)
// ============================================================================

impl From<&CommandQueue> for TickCircuit {
    /// Convert a `CommandQueue` to a `TickCircuit`.
    ///
    /// Each command becomes its own tick. For better parallelization,
    /// consider using the `CommandBuilder` to construct circuits directly,
    /// or manually building a `TickCircuit`.
    fn from(queue: &CommandQueue) -> Self {
        let mut circuit = TickCircuit::new();

        for cmd in queue.iter() {
            let gate_type: pecos_core::gate_type::GateType = cmd.gate_type.into();
            let qubits: Vec<usize> = cmd.qubits.iter().map(|q| q.0).collect();

            // Create a new tick for each command
            let mut tick = circuit.tick();

            // Handle different gate types
            match gate_type {
                pecos_core::gate_type::GateType::PZ => {
                    tick.pz(&qubits);
                }
                pecos_core::gate_type::GateType::MZ => {
                    tick.mz(&qubits);
                }
                pecos_core::gate_type::GateType::H => {
                    tick.h(&qubits);
                }
                pecos_core::gate_type::GateType::X => {
                    tick.x(&qubits);
                }
                pecos_core::gate_type::GateType::Y => {
                    tick.y(&qubits);
                }
                pecos_core::gate_type::GateType::Z => {
                    tick.z(&qubits);
                }
                pecos_core::gate_type::GateType::CX => {
                    if qubits.len() >= 2 {
                        tick.cx(&[(qubits[0], qubits[1])]);
                    }
                }
                pecos_core::gate_type::GateType::CZ => {
                    if qubits.len() >= 2 {
                        tick.cz(&[(qubits[0], qubits[1])]);
                    }
                }
                _ => {
                    // For other gate types, add as a raw gate
                    let angles: SmallVec<[Angle64; 3]> = cmd.angles.iter().copied().collect();
                    let qubit_ids: SmallVec<[QubitId; 4]> =
                        qubits.iter().map(|&q| QubitId(q)).collect();
                    let gate = Gate {
                        gate_type,
                        angles,
                        params: SmallVec::new(),
                        qubits: qubit_ids,
                    };
                    // Use try_add_gate and ignore errors (shouldn't happen with one gate per tick)
                    let _ = tick.try_add_gate(gate);
                }
            }
        }

        circuit
    }
}

impl From<CommandQueue> for TickCircuit {
    fn from(queue: CommandQueue) -> Self {
        (&queue).into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gate_type_conversion_roundtrip() {
        // Test a few gate types
        let neo_types = [
            GateType::H,
            GateType::X,
            GateType::CX,
            GateType::MZ,
            GateType::PZ,
        ];

        for &gt in &neo_types {
            let core_gt: pecos_core::gate_type::GateType = gt.into();
            let back: GateType = core_gt.into();
            assert_eq!(gt, back, "Roundtrip failed for {gt:?}");
        }
    }

    #[test]
    fn test_tick_circuit_to_command_queue() {
        // Use separate tick calls to create separate gates
        let mut circuit = TickCircuit::new();
        circuit.tick().pz(&[0]);
        circuit.tick().pz(&[1]);
        circuit.tick().h(&[0]);
        circuit.tick().cx(&[(0, 1)]);
        circuit.tick().mz(&[0]);
        circuit.tick().mz(&[1]);

        let queue = CommandQueue::from(&circuit);

        // Should have: 2 preps + 1 H + 1 CX + 2 measures = 6 commands
        assert_eq!(queue.len(), 6);

        // Check gate types
        let types: Vec<_> = queue.iter().map(|c| c.gate_type).collect();
        assert_eq!(types[0], GateType::PZ);
        assert_eq!(types[1], GateType::PZ);
        assert_eq!(types[2], GateType::H);
        assert_eq!(types[3], GateType::CX);
        assert_eq!(types[4], GateType::MZ);
        assert_eq!(types[5], GateType::MZ);
    }

    #[test]
    fn test_tick_circuit_bulk_ops() {
        // Test bulk operations - create single gate with multiple qubits
        let mut circuit = TickCircuit::new();
        circuit.tick().pz(&[0, 1]); // One prep gate with 2 qubits
        circuit.tick().h(&[0, 1]); // One H gate with 2 qubits
        circuit.tick().mz(&[0, 1]); // One measure gate with 2 qubits

        let queue = CommandQueue::from(&circuit);

        // Bulk ops create single gates with multiple qubits
        assert_eq!(queue.len(), 3);

        // First command should be Prep with 2 qubits
        let prep_cmd = queue.iter().next().unwrap();
        assert_eq!(prep_cmd.gate_type, GateType::PZ);
        assert_eq!(prep_cmd.qubits.len(), 2);
    }

    #[test]
    fn test_dag_circuit_to_command_queue() {
        let mut dag = DagCircuit::new();
        dag.pz(0);
        dag.pz(1);
        dag.h(0);
        dag.cx(0, 1);
        dag.mz(0);
        dag.mz(1);

        let queue = CommandQueue::from(&dag);

        // Should have 6 commands
        assert_eq!(queue.len(), 6);

        // First two should be preps
        assert_eq!(queue.iter().next().unwrap().gate_type, GateType::PZ);
    }

    #[test]
    fn test_command_queue_to_tick_circuit() {
        use crate::command::CommandBuilder;

        let commands = CommandBuilder::new()
            .pz(0)
            .pz(1)
            .h(0)
            .cx(0, 1)
            .mz(0)
            .mz(1)
            .build();

        let circuit = TickCircuit::from(&commands);

        // Each command becomes its own tick
        assert_eq!(circuit.num_ticks(), 6);
    }

    #[test]
    fn test_gate_conversion_with_angles() {
        use pecos_core::gate_type::GateType as CoreGT;

        let gate = Gate {
            gate_type: CoreGT::RZ,
            angles: smallvec::smallvec![Angle64::QUARTER_TURN],
            params: SmallVec::new(),
            qubits: smallvec::smallvec![QubitId(0)],
        };

        let cmd: GateCommand = (&gate).into();

        assert_eq!(cmd.gate_type, GateType::RZ);
        assert_eq!(cmd.angles.len(), 1);
        assert_eq!(cmd.angles[0], Angle64::QUARTER_TURN);
        assert_eq!(cmd.qubits.len(), 1);
        assert_eq!(cmd.qubits[0], QubitId(0));
    }
}
