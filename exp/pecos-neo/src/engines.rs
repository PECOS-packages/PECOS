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

//! Native `CommandSource` engines for circuit formats.
//!
//! These engines wrap circuit types and implement `CommandSource`, enabling them
//! to be used with `ProgramRunner` for shot execution.
//!
//! - [`CommandQueueEngine`]: Single-batch execution of a `CommandQueue`
//! - [`TickCircuitEngine`]: Tick-by-tick execution of a `TickCircuit`
//! - [`DagCircuitEngine`]: Topological-order execution of a `DagCircuit`

use crate::command::CommandQueue;
use crate::outcome::MeasurementOutcomes;
use crate::program::CommandSource;
use pecos_quantum::{DagCircuit, TickCircuit};

// ============================================================================
// CommandQueueEngine
// ============================================================================

/// Engine that executes a `CommandQueue` as a single batch.
///
/// Equivalent to `StaticProgram` but named consistently with the engine pattern.
///
/// # Example
///
/// ```
/// use pecos_neo::prelude::*;
/// use pecos_neo::engines::CommandQueueEngine;
///
/// let commands = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();
/// let engine = CommandQueueEngine::new(commands, 1);
/// ```
#[derive(Debug, Clone)]
pub struct CommandQueueEngine {
    commands: CommandQueue,
    executed: bool,
    num_qubits: usize,
}

impl CommandQueueEngine {
    /// Create a new engine from a command queue.
    #[must_use]
    pub fn new(commands: CommandQueue, num_qubits: usize) -> Self {
        Self {
            commands,
            executed: false,
            num_qubits,
        }
    }
}

impl CommandSource for CommandQueueEngine {
    fn next_commands(&mut self, _outcomes: Option<&MeasurementOutcomes>) -> Option<CommandQueue> {
        if self.executed {
            None
        } else {
            self.executed = true;
            Some(self.commands.clone())
        }
    }

    fn is_complete(&self) -> bool {
        self.executed
    }

    fn reset(&mut self) {
        self.executed = false;
    }

    fn num_qubits(&self) -> usize {
        self.num_qubits
    }
}

// ============================================================================
// TickCircuitEngine
// ============================================================================

/// Engine that executes a `TickCircuit` one tick at a time.
///
/// Each call to `next_commands()` returns one tick's worth of gates as a
/// `CommandQueue` batch. This preserves tick structure, enabling tick-level
/// noise (e.g., idle noise between ticks) in the future.
///
/// # Example
///
/// ```
/// use pecos_neo::engines::TickCircuitEngine;
/// use pecos_quantum::TickCircuit;
///
/// let mut circuit = TickCircuit::new();
/// circuit.tick().pz(&[0, 1]);
/// circuit.tick().h(&[0]);
/// circuit.tick().cx(&[(0, 1)]);
/// circuit.tick().mz(&[0, 1]);
///
/// let engine = TickCircuitEngine::new(circuit);
/// ```
#[derive(Debug, Clone)]
pub struct TickCircuitEngine {
    circuit: TickCircuit,
    current_tick: usize,
    num_qubits: usize,
}

impl TickCircuitEngine {
    /// Create a new engine from a tick circuit.
    #[must_use]
    pub fn new(circuit: TickCircuit) -> Self {
        let num_qubits = circuit.all_qubits().last().map_or(0, |q| q.0 + 1);
        Self {
            circuit,
            current_tick: 0,
            num_qubits,
        }
    }
}

impl CommandSource for TickCircuitEngine {
    fn next_commands(&mut self, _outcomes: Option<&MeasurementOutcomes>) -> Option<CommandQueue> {
        let ticks = self.circuit.ticks();
        if self.current_tick >= ticks.len() {
            return None;
        }

        let tick = &ticks[self.current_tick];
        self.current_tick += 1;

        let mut queue = CommandQueue::new();
        for gate in tick.gates() {
            queue.push(gate.into());
        }

        Some(queue)
    }

    fn is_complete(&self) -> bool {
        self.current_tick >= self.circuit.ticks().len()
    }

    fn reset(&mut self) {
        self.current_tick = 0;
    }

    fn num_qubits(&self) -> usize {
        self.num_qubits
    }
}

// ============================================================================
// DagCircuitEngine
// ============================================================================

/// Engine that executes a `DagCircuit` in topological order.
///
/// All gates are emitted in a single batch in topological order,
/// respecting gate dependencies.
///
/// # Example
///
/// ```
/// use pecos_neo::engines::DagCircuitEngine;
/// use pecos_quantum::DagCircuit;
///
/// let mut dag = DagCircuit::new();
/// dag.pz(&[0]);
/// dag.h(&[0]);
/// dag.mz(&[0]);
///
/// let engine = DagCircuitEngine::new(dag);
/// ```
#[derive(Debug, Clone)]
pub struct DagCircuitEngine {
    circuit: DagCircuit,
    executed: bool,
}

impl DagCircuitEngine {
    /// Create a new engine from a DAG circuit.
    #[must_use]
    pub fn new(circuit: DagCircuit) -> Self {
        Self {
            circuit,
            executed: false,
        }
    }
}

impl CommandSource for DagCircuitEngine {
    fn next_commands(&mut self, _outcomes: Option<&MeasurementOutcomes>) -> Option<CommandQueue> {
        if self.executed {
            return None;
        }
        self.executed = true;
        Some(CommandQueue::from(&self.circuit))
    }

    fn is_complete(&self) -> bool {
        self.executed
    }

    fn reset(&mut self) {
        self.executed = false;
    }

    fn num_qubits(&self) -> usize {
        self.circuit.qubits().last().map_or(0, |q| q.0 + 1)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command::CommandBuilder;

    #[test]
    fn test_command_queue_engine() {
        let commands = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();
        let mut engine = CommandQueueEngine::new(commands.clone(), 1);

        assert!(!engine.is_complete());
        assert_eq!(engine.num_qubits(), 1);

        let batch = engine.next_commands(None);
        assert!(batch.is_some());
        assert!(engine.is_complete());

        let batch = engine.next_commands(None);
        assert!(batch.is_none());

        engine.reset();
        assert!(!engine.is_complete());
        assert!(engine.next_commands(None).is_some());
    }

    #[test]
    fn test_tick_circuit_engine() {
        let mut circuit = TickCircuit::new();
        circuit.tick().pz(&[0usize, 1]);
        circuit.tick().h(&[0usize]);
        circuit.tick().cx(&[(0usize, 1)]);
        circuit.tick().mz(&[0usize, 1]);

        let mut engine = TickCircuitEngine::new(circuit);

        assert!(!engine.is_complete());
        assert_eq!(engine.num_qubits(), 2);

        // Should return 4 batches (one per tick)
        let mut batch_count = 0;
        while engine.next_commands(None).is_some() {
            batch_count += 1;
        }
        assert_eq!(batch_count, 4);
        assert!(engine.is_complete());

        engine.reset();
        assert!(!engine.is_complete());
    }

    #[test]
    fn test_dag_circuit_engine() {
        let mut dag = DagCircuit::new();
        dag.pz(&[0usize]);
        dag.pz(&[1usize]);
        dag.h(&[0usize]);
        dag.cx(&[(0usize, 1usize)]);
        dag.mz(&[0usize]);
        dag.mz(&[1usize]);

        let mut engine = DagCircuitEngine::new(dag);

        assert!(!engine.is_complete());
        assert_eq!(engine.num_qubits(), 2);

        let batch = engine.next_commands(None);
        assert!(batch.is_some());
        assert!(engine.is_complete());

        let batch = engine.next_commands(None);
        assert!(batch.is_none());

        engine.reset();
        assert!(!engine.is_complete());
    }
}
