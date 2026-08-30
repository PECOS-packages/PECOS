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

//! Batched circuit execution for Clifford simulators.
//!
//! This module provides efficient circuit execution using the full-fidelity
//! batched gate commands stored by `TickCircuit`. Instead of dispatching each
//! individual gate application, stored batched commands are applied as one
//! simulator call.
//!
//! # Performance Benefits
//!
//! - **Reduced dispatch overhead**: One match per gate type per tick, not per gate
//! - **Better cache utilization**: Qubits for same-type gates are contiguous
//! - **Simulator optimization**: Simulators can vectorize batch operations
//!
//! # Example
//!
//! ```
//! use pecos_simulators::{SparseStab, CircuitExecutor};
//! use pecos_quantum::TickCircuit;
//!
//! let mut circuit = TickCircuit::new();
//! circuit.tick().pz(&[0, 1, 2, 3]);
//! circuit.tick().h(&[0, 1, 2, 3]);
//! circuit.tick().cx(&[(0, 1), (2, 3)]);
//! circuit.tick().mz(&[0, 1, 2, 3]);
//!
//! let mut sim = SparseStab::new(4);
//! let executor = CircuitExecutor::new(&circuit);
//! executor.run(&mut sim).expect("Clifford circuit should execute");
//! ```

use crate::clifford_rotation::CliffordRotation;
use crate::{CliffordGateable, MeasurementResult};
use pecos_core::gate_type::GateType;
use pecos_core::{Gate, QubitId};
use pecos_quantum::TickCircuit;
use smallvec::SmallVec;

/// Convert a flat qubit slice `[c0, t0, c1, t1, ...]` to a vec of pairs.
fn flat_to_pairs(qubits: &[QubitId]) -> SmallVec<[(QubitId, QubitId); 4]> {
    qubits
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| (pair[0], pair[1]))
        .collect()
}

/// Executes a `TickCircuit` on a Clifford simulator using batched operations.
///
/// This executor leverages the full-fidelity batched gate commands in
/// `TickCircuit` for efficient execution with minimal dispatch overhead.
pub struct CircuitExecutor<'a> {
    /// The circuit to execute.
    circuit: &'a TickCircuit,
}

impl<'a> CircuitExecutor<'a> {
    /// Creates a new executor for the given circuit.
    #[inline]
    #[must_use]
    pub fn new(circuit: &'a TickCircuit) -> Self {
        Self { circuit }
    }

    /// Runs the circuit on a Clifford simulator.
    ///
    /// Returns measurement results collected during execution.
    ///
    /// # Errors
    ///
    /// Returns an error when a gate has the wrong angle arity, a rotation is
    /// non-Clifford, or the simulator does not support the gate.
    pub fn run<S: CliffordGateable>(&self, sim: &mut S) -> Result<Vec<MeasurementResult>, String> {
        let mut measurements = Vec::new();

        for (_tick_idx, tick) in self.circuit.iter_ticks() {
            for batch in tick.iter_gate_batches() {
                Self::execute_gate_batch(sim, batch.as_gate(), &mut measurements)?;
            }
        }

        Ok(measurements)
    }

    /// Executes a single full-fidelity batched gate command.
    ///
    /// This is the core dispatch function - one match per batch, not per gate.
    #[inline]
    fn execute_gate_batch<S: CliffordGateable>(
        sim: &mut S,
        batch: &Gate,
        measurements: &mut Vec<MeasurementResult>,
    ) -> Result<(), String> {
        execute_gate_command(sim, batch, measurements)
    }
}

/// Executes one full-fidelity `TickCircuit` gate command on a simulator.
#[inline]
fn execute_gate_command<S: CliffordGateable>(
    sim: &mut S,
    gate: &Gate,
    measurements: &mut Vec<MeasurementResult>,
) -> Result<(), String> {
    gate.validate()?;
    let qubits = gate.qubits.as_slice();

    match gate.gate_type {
        GateType::I => {
            sim.identity(qubits);
        }
        GateType::X => {
            sim.x(qubits);
        }
        GateType::Y => {
            sim.y(qubits);
        }
        GateType::Z => {
            sim.z(qubits);
        }
        GateType::H => {
            sim.h(qubits);
        }
        GateType::F => {
            sim.f(qubits);
        }
        GateType::Fdg => {
            sim.fdg(qubits);
        }
        GateType::SX => {
            sim.sx(qubits);
        }
        GateType::SXdg => {
            sim.sxdg(qubits);
        }
        GateType::SY => {
            sim.sy(qubits);
        }
        GateType::SYdg => {
            sim.sydg(qubits);
        }
        GateType::SZ => {
            sim.sz(qubits);
        }
        GateType::SZdg => {
            sim.szdg(qubits);
        }
        GateType::CX => {
            let pairs = flat_to_pairs(qubits);
            sim.cx(&pairs);
        }
        GateType::CY => {
            let pairs = flat_to_pairs(qubits);
            sim.cy(&pairs);
        }
        GateType::CZ => {
            let pairs = flat_to_pairs(qubits);
            sim.cz(&pairs);
        }
        GateType::SXX => {
            let pairs = flat_to_pairs(qubits);
            sim.sxx(&pairs);
        }
        GateType::SXXdg => {
            let pairs = flat_to_pairs(qubits);
            sim.sxxdg(&pairs);
        }
        GateType::SYY => {
            let pairs = flat_to_pairs(qubits);
            sim.syy(&pairs);
        }
        GateType::SYYdg => {
            let pairs = flat_to_pairs(qubits);
            sim.syydg(&pairs);
        }
        GateType::SZZ => {
            let pairs = flat_to_pairs(qubits);
            sim.szz(&pairs);
        }
        GateType::SZZdg => {
            let pairs = flat_to_pairs(qubits);
            sim.szzdg(&pairs);
        }
        GateType::SWAP => {
            let pairs = flat_to_pairs(qubits);
            sim.swap(&pairs);
        }
        GateType::PX => {
            sim.pz(qubits);
            sim.h(qubits);
        }
        GateType::PZ | GateType::QAlloc => {
            sim.pz(qubits);
        }
        GateType::MX => {
            sim.h(qubits);
            measurements.extend(sim.mz(qubits));
        }
        GateType::MZ | GateType::MeasureFree => {
            measurements.extend(sim.mz(qubits));
        }
        GateType::MPZ => {
            measurements.extend(sim.mpz(qubits));
        }
        GateType::Idle => {}
        GateType::RZ => {
            sim.try_rz(gate.angles[0], qubits)?;
        }
        GateType::RX => {
            sim.try_rx(gate.angles[0], qubits)?;
        }
        GateType::RY => {
            sim.try_ry(gate.angles[0], qubits)?;
        }
        GateType::RZZ => {
            sim.try_rzz(gate.angles[0], &flat_to_pairs(qubits))?;
        }
        GateType::RXX => {
            sim.try_rxx(gate.angles[0], &flat_to_pairs(qubits))?;
        }
        GateType::RYY => {
            sim.try_ryy(gate.angles[0], &flat_to_pairs(qubits))?;
        }
        GateType::RXY1Q => {
            sim.try_rxy1q(gate.angles[0], gate.angles[1], qubits)?;
        }
        GateType::CRZ => {
            sim.try_crz(gate.angles[0], &flat_to_pairs(qubits))?;
        }
        GateType::U => {
            sim.try_u(gate.angles[0], gate.angles[1], gate.angles[2], qubits)?;
        }
        GateType::RXXRYYRZZ => {
            sim.try_rxxryyrzz(
                gate.angles[0],
                gate.angles[1],
                gate.angles[2],
                &flat_to_pairs(qubits),
            )?;
        }
        GateType::U2q => {
            let before = [
                [gate.angles[0], gate.angles[1], gate.angles[2]],
                [gate.angles[3], gate.angles[4], gate.angles[5]],
            ];
            let interaction = [gate.angles[6], gate.angles[7], gate.angles[8]];
            let after = [
                [gate.angles[9], gate.angles[10], gate.angles[11]],
                [gate.angles[12], gate.angles[13], gate.angles[14]],
            ];
            sim.try_u2q(before, interaction, after, &flat_to_pairs(qubits))?;
        }
        other => {
            return Err(format!(
                "Unsupported gate type in circuit executor: {other:?}"
            ));
        }
    }
    Ok(())
}

// ============================================================================
// DOD/ECS-Style Execution Pipeline
// ============================================================================

/// A DOD/ECS-style execution system that processes gate batches.
///
/// A "system" in ECS terminology - a function that
/// operates on components (gate batches) to produce effects (simulator state changes).
pub trait GateSystem<S: CliffordGateable> {
    /// The gate type this system handles.
    fn gate_type(&self) -> GateType;

    /// Execute the system on a batch of qubits.
    fn execute(&self, sim: &mut S, qubits: &[QubitId]);
}

/// Registry of gate systems for dynamic dispatch.
///
/// This enables extensible gate handling without modifying the executor.
pub struct GateSystemRegistry<S: CliffordGateable> {
    systems: Vec<Box<dyn GateSystem<S>>>,
}

impl<S: CliffordGateable> Default for GateSystemRegistry<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: CliffordGateable> GateSystemRegistry<S> {
    /// Creates an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            systems: Vec::new(),
        }
    }

    /// Registers a gate system.
    pub fn register(&mut self, system: Box<dyn GateSystem<S>>) {
        self.systems.push(system);
    }

    /// Finds a system for the given gate type.
    #[must_use]
    pub fn find(&self, gate_type: GateType) -> Option<&dyn GateSystem<S>> {
        self.systems
            .iter()
            .find(|s| s.gate_type() == gate_type)
            .map(AsRef::as_ref)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{SparseStab, StabilizerTableauSimulator};
    use pecos_core::Angle64;
    use pecos_quantum::TickCircuit;

    #[test]
    fn test_circuit_executor_basic() {
        let mut circuit = TickCircuit::new();
        circuit.tick().pz(&[0, 1]);
        circuit.tick().h(&[0]);
        circuit.tick().cx(&[(0, 1)]);
        circuit.tick().mz(&[0, 1]);

        let mut sim = SparseStab::new(2);
        let executor = CircuitExecutor::new(&circuit);
        let measurements = executor.run(&mut sim).unwrap();

        // Should have 2 measurements
        assert_eq!(measurements.len(), 2);
    }

    #[test]
    fn test_circuit_executor_batched_gates() {
        // Create a circuit with multiple gates of same type per tick
        let mut circuit = TickCircuit::new();
        circuit.tick().pz(&[0, 1, 2, 3]); // 4 preps in one batch
        circuit.tick().h(&[0, 1, 2, 3]); // 4 H gates in one batch
        circuit.tick().cx(&[(0, 1), (2, 3)]); // 2 CX gates in one batch
        circuit.tick().mz(&[0, 1, 2, 3]); // 4 measurements in one batch

        let mut sim = SparseStab::new(4);
        let executor = CircuitExecutor::new(&circuit);
        let measurements = executor.run(&mut sim).unwrap();

        // Should have 4 measurements
        assert_eq!(measurements.len(), 4);
    }

    #[test]
    fn circuit_executor_rz_quarter_turn_matches_sz() {
        let mut rotation_circuit = TickCircuit::new();
        rotation_circuit.tick().h(&[0]);
        rotation_circuit.tick().rz(Angle64::QUARTER_TURN, &[0]);

        let mut named_circuit = TickCircuit::new();
        named_circuit.tick().h(&[0]);
        named_circuit.tick().sz(&[0]);

        let mut rotation_sim = SparseStab::new(1);
        CircuitExecutor::new(&rotation_circuit)
            .run(&mut rotation_sim)
            .unwrap();
        let mut named_sim = SparseStab::new(1);
        CircuitExecutor::new(&named_circuit)
            .run(&mut named_sim)
            .unwrap();

        assert_eq!(rotation_sim.full_tableau(), named_sim.full_tableau());
    }

    #[test]
    fn circuit_executor_non_clifford_rotation_returns_error() {
        let mut circuit = TickCircuit::new();
        circuit.tick().rz(Angle64::from_radians(0.5), &[0]);

        let Err(err) = CircuitExecutor::new(&circuit).run(&mut SparseStab::new(1)) else {
            panic!("non-Clifford rotations must fail");
        };
        assert!(err.contains("is not a Clifford rotation"));
    }

    #[test]
    fn circuit_executor_rotation_with_wrong_arity_returns_error() {
        let gate = Gate::new(
            GateType::RZ,
            Vec::<Angle64>::new(),
            Vec::<f64>::new(),
            vec![QubitId(0)],
        );
        let err = execute_gate_command(&mut SparseStab::new(1), &gate, &mut Vec::new())
            .expect_err("wrong rotation arity must fail");

        assert_eq!(err, "Gate RZ expected 1 angle parameters, got 0");
    }

    #[test]
    fn circuit_executor_unsupported_gate_returns_error() {
        let mut circuit = TickCircuit::new();
        circuit.tick().t(&[0]);

        let Err(err) = CircuitExecutor::new(&circuit).run(&mut SparseStab::new(1)) else {
            panic!("unsupported gates must fail without panicking");
        };
        assert!(err.contains('T'));
    }
}
