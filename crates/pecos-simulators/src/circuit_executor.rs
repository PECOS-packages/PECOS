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
//! This module provides efficient circuit execution using the batched gate groups
//! from `TickCircuitSoA`. Instead of dispatching each gate individually, gates of
//! the same type are applied as a single batch call.
//!
//! # Performance Benefits
//!
//! - **Reduced dispatch overhead**: One match per gate type per tick, not per gate
//! - **Better cache utilization**: Qubits for same-type gates are contiguous
//! - **Simulator optimization**: Simulators can vectorize batch operations
//!
//! # Example
//!
//! ```ignore
//! use pecos_simulators::{SparseStab, CircuitExecutor};
//! use pecos_quantum::TickCircuitSoA;
//!
//! let circuit = TickCircuitSoA::builder()
//!     .tick().pz(&[0, 1, 2, 3])
//!     .tick().h(&[0, 1, 2, 3])
//!     .tick().cx(&[(0, 1), (2, 3)])
//!     .tick().mz(&[0, 1, 2, 3])
//!     .build();
//!
//! let mut sim = SparseStab::new(4);
//! let executor = CircuitExecutor::new(&circuit);
//! executor.run(&mut sim);
//! ```

use crate::{CliffordGateable, MeasurementResult};
use pecos_core::QubitId;
use pecos_core::gate_type::GateType;
use pecos_quantum::{GateBatch, TickBatches, TickCircuitSoA, TickGateGroups};
use smallvec::SmallVec;

/// Convert a flat qubit slice `[c0, t0, c1, t1, ...]` to a vec of pairs.
fn flat_to_pairs(qubits: &[QubitId]) -> SmallVec<[(QubitId, QubitId); 4]> {
    qubits
        .chunks_exact(2)
        .map(|pair| (pair[0], pair[1]))
        .collect()
}

/// Executes a `TickCircuitSoA` on a Clifford simulator using batched operations.
///
/// This executor leverages the pre-grouped gate batches in `TickCircuitSoA` for
/// efficient execution with minimal dispatch overhead.
pub struct CircuitExecutor<'a> {
    /// The circuit to execute.
    circuit: &'a TickCircuitSoA,
}

impl<'a> CircuitExecutor<'a> {
    /// Creates a new executor for the given circuit.
    #[inline]
    #[must_use]
    pub fn new(circuit: &'a TickCircuitSoA) -> Self {
        Self { circuit }
    }

    /// Runs the circuit on a Clifford simulator.
    ///
    /// Returns measurement results collected during execution.
    pub fn run<S: CliffordGateable>(&self, sim: &mut S) -> Vec<MeasurementResult> {
        let mut measurements = Vec::new();

        for (_tick_idx, tick) in self.circuit.iter_ticks_batched() {
            Self::execute_tick(sim, tick, &mut measurements);
        }

        measurements
    }

    /// Runs a single tick on the simulator.
    #[inline]
    fn execute_tick<S: CliffordGateable>(
        sim: &mut S,
        tick: &TickBatches,
        measurements: &mut Vec<MeasurementResult>,
    ) {
        for batch in tick.iter() {
            Self::execute_batch(sim, batch, measurements);
        }
    }

    /// Executes a single batch of gates.
    ///
    /// This is the core dispatch function - one match per batch, not per gate.
    #[inline]
    fn execute_batch<S: CliffordGateable>(
        sim: &mut S,
        batch: &GateBatch,
        measurements: &mut Vec<MeasurementResult>,
    ) {
        execute_single_batch(sim, batch, measurements);
    }
}

/// Executes a `TickGateGroups` directly on a simulator.
///
/// This is a simpler interface when you have gate groups but not the full circuit.
pub fn execute_batched<S: CliffordGateable>(
    groups: &TickGateGroups,
    sim: &mut S,
) -> Vec<MeasurementResult> {
    let mut measurements = Vec::new();

    for (_tick_idx, tick) in groups.iter_ticks() {
        for batch in tick.iter() {
            execute_single_batch(sim, batch, &mut measurements);
        }
    }

    measurements
}

/// Executes a single batch on a simulator.
#[inline]
fn execute_single_batch<S: CliffordGateable>(
    sim: &mut S,
    batch: &GateBatch,
    measurements: &mut Vec<MeasurementResult>,
) {
    let qubits = batch.qubits();

    match batch.gate_type {
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
        GateType::PZ | GateType::QAlloc => {
            sim.pz(qubits);
        }
        GateType::MZ | GateType::MeasureFree => {
            measurements.extend(sim.mz(qubits));
        }
        GateType::Idle => {}
        other => {
            panic!("Unsupported gate type in circuit executor: {other:?}");
        }
    }
}

// ============================================================================
// DOD/ECS-Style Execution Pipeline
// ============================================================================

/// A DOD/ECS-style execution system that processes gate batches.
///
/// This trait represents a "system" in ECS terminology - a function that
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
    use crate::SparseStab;
    use pecos_quantum::TickCircuitSoA;

    #[test]
    fn test_circuit_executor_basic() {
        let mut builder = TickCircuitSoA::builder();
        builder
            .tick()
            .pz(&[0, 1])
            .tick()
            .h(&[0])
            .tick()
            .cx(&[(0, 1)])
            .tick()
            .mz(&[0, 1]);

        let circuit = builder.build();

        let mut sim = SparseStab::new(2);
        let executor = CircuitExecutor::new(&circuit);
        let measurements = executor.run(&mut sim);

        // Should have 2 measurements
        assert_eq!(measurements.len(), 2);
    }

    #[test]
    fn test_circuit_executor_batched_gates() {
        // Create a circuit with multiple gates of same type per tick
        let mut builder = TickCircuitSoA::builder();
        builder
            .tick()
            .pz(&[0, 1, 2, 3]) // 4 preps in one batch
            .tick()
            .h(&[0, 1, 2, 3]) // 4 H gates in one batch
            .tick()
            .cx(&[(0, 1), (2, 3)]) // 2 CX gates in one batch
            .tick()
            .mz(&[0, 1, 2, 3]); // 4 measurements in one batch

        let circuit = builder.build();

        let mut sim = SparseStab::new(4);
        let executor = CircuitExecutor::new(&circuit);
        let measurements = executor.run(&mut sim);

        // Should have 4 measurements
        assert_eq!(measurements.len(), 4);
    }

    #[test]
    fn test_execute_batched_function() {
        let mut builder = TickCircuitSoA::builder();
        builder
            .tick()
            .pz(&[0, 1])
            .tick()
            .h(&[0, 1])
            .tick()
            .mz(&[0, 1]);

        let circuit = builder.build();

        let mut sim = SparseStab::new(2);
        let measurements = execute_batched(&circuit.batched, &mut sim);

        assert_eq!(measurements.len(), 2);
    }
}
