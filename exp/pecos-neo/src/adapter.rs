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

//! Adapter for integrating pecos-engines classical engines with pecos-neo.
//!
//! This module provides adapters that wrap existing classical control engines
//! (like `QASMEngine` or `HugrEngine`) from pecos-engines to implement the
//! pecos-neo `CommandSource` trait, enabling integration with the DOD-style
//! simulation infrastructure.
//!
//! ## Design Philosophy
//!
//! Rather than rewriting classical engines, we provide a bridge layer that:
//! 1. Converts `ByteMessage` commands to `CommandQueue`
//! 2. Converts `MeasurementOutcomes` to `ByteMessage` for the engine
//! 3. Manages the `ControlEngine` state machine
//!
//! This allows gradual migration: existing programs work unchanged while
//! gaining access to pecos-neo's composable noise models and sampling infrastructure.
//!
//! ## Example
//!
//! ```rust,no_run
//! use std::str::FromStr;
//! use pecos_neo::adapter::ClassicalEngineAdapter;
//! use pecos_neo::prelude::*;
//! use pecos_neo::program::ProgramRunner;
//! use pecos_qasm::QASMEngine;
//! use pecos_simulators::SparseStab;
//!
//! // Load a QASM program
//! let qasm = r#"
//!     OPENQASM 2.0;
//!     qreg q[2];
//!     h q[0];
//!     cx q[0], q[1];
//!     measure q[0];
//!     measure q[1];
//! "#;
//!
//! let engine = QASMEngine::from_str(qasm).unwrap();
//!
//! // Wrap in adapter to get CommandSource interface
//! let mut program = ClassicalEngineAdapter::new(engine);
//!
//! // Run with pecos-neo infrastructure (composable noise, etc.)
//! let noise = ComposableNoiseModel::new()
//!     .add_channel(SingleQubitChannel::depolarizing(0.01));
//!
//! let mut runner = ProgramRunner::new(SparseStab::new(2))
//!     .with_noise(noise);
//!
//! let result = runner.run_shot(&mut program);
//! ```

use crate::command::{CommandQueue, GateCommand, GateType as NeoGateType};
use crate::outcome::{MeasurementOutcome, MeasurementOutcomes};
use crate::program::{CommandSource, DynProgramRunner, ProgramResult};
use pecos_core::gate_type::GateType as CoreGateType;
use pecos_core::gates::Gate;
use pecos_core::{Angle64, QubitId};

/// Convert pecos-core `GateType` to pecos-neo `GateType`.
///
/// This handles the mapping between the two gate type enums.
fn convert_gate_type(core_type: CoreGateType) -> Option<NeoGateType> {
    Some(match core_type {
        // Single-qubit Paulis
        CoreGateType::I => NeoGateType::I,
        CoreGateType::X => NeoGateType::X,
        CoreGateType::Y => NeoGateType::Y,
        CoreGateType::Z => NeoGateType::Z,

        // Single-qubit Cliffords
        CoreGateType::H => NeoGateType::H,
        CoreGateType::F => NeoGateType::F,
        CoreGateType::Fdg => NeoGateType::Fdg,
        CoreGateType::SX => NeoGateType::SX,
        CoreGateType::SXdg => NeoGateType::SXdg,
        CoreGateType::SY => NeoGateType::SY,
        CoreGateType::SYdg => NeoGateType::SYdg,
        CoreGateType::SZ => NeoGateType::SZ,
        CoreGateType::SZdg => NeoGateType::SZdg,
        CoreGateType::T => NeoGateType::T,
        CoreGateType::Tdg => NeoGateType::Tdg,

        // Single-qubit rotations
        CoreGateType::RX => NeoGateType::RX,
        CoreGateType::RY => NeoGateType::RY,
        CoreGateType::RZ => NeoGateType::RZ,
        CoreGateType::U => NeoGateType::U,
        CoreGateType::R1XY => NeoGateType::R1XY,

        // Two-qubit gates
        CoreGateType::CX => NeoGateType::CX,
        CoreGateType::CY => NeoGateType::CY,
        CoreGateType::CZ => NeoGateType::CZ,
        CoreGateType::SZZ => NeoGateType::SZZ,
        CoreGateType::SZZdg => NeoGateType::SZZdg,
        CoreGateType::SXX => NeoGateType::SXX,
        CoreGateType::SXXdg => NeoGateType::SXXdg,
        CoreGateType::SYY => NeoGateType::SYY,
        CoreGateType::SYYdg => NeoGateType::SYYdg,
        CoreGateType::SWAP => NeoGateType::SWAP,
        CoreGateType::CRZ => NeoGateType::CRZ,
        CoreGateType::RXX => NeoGateType::RXX,
        CoreGateType::RYY => NeoGateType::RYY,
        CoreGateType::RZZ => NeoGateType::RZZ,

        // Three-qubit gates
        CoreGateType::CCX => NeoGateType::CCX,

        // Measurement and preparation
        CoreGateType::MZ => NeoGateType::MZ,
        CoreGateType::MeasureLeaked => NeoGateType::MeasureLeaked,
        CoreGateType::MeasureFree => NeoGateType::MeasureFree,
        CoreGateType::PZ => NeoGateType::PZ,
        CoreGateType::QAlloc => NeoGateType::QAlloc,
        CoreGateType::QFree => NeoGateType::QFree,

        // Idle
        CoreGateType::Idle => NeoGateType::Idle,

        // Unknown or unhandled gate types
        _ => return None,
    })
}

/// Convert a pecos-core `Gate` to a pecos-neo `GateCommand`.
fn convert_gate(gate: &Gate) -> Result<GateCommand, pecos_core::errors::PecosError> {
    let neo_type = convert_gate_type(gate.gate_type).ok_or_else(|| {
        pecos_core::errors::PecosError::Input(format!(
            "pecos-neo adapter does not support gate type {:?}",
            gate.gate_type
        ))
    })?;

    let qubits = gate.qubits.iter().copied().collect();
    let angles = gate.angles.iter().copied().collect();

    Ok(GateCommand {
        gate_type: neo_type,
        qubits,
        angles,
    })
}

/// Convert a `ByteMessage` containing quantum operations to a `CommandQueue`.
///
/// # Errors
/// Returns `PecosError` if the byte message cannot be decoded or contains a
/// gate unsupported by the pecos-neo command representation.
pub fn byte_message_to_command_queue(
    message: &pecos_engines::ByteMessage,
) -> Result<CommandQueue, pecos_core::errors::PecosError> {
    let gates = message.quantum_ops()?;

    let mut queue = CommandQueue::with_capacity(gates.len());

    for gate in &gates {
        queue.push(convert_gate(gate)?);
    }

    Ok(queue)
}

/// Convert `MeasurementOutcomes` to a `ByteMessage` containing outcomes.
///
/// The outcomes are ordered by qubit ID for consistency with the engine's expectations.
#[must_use]
pub fn outcomes_to_byte_message(outcomes: &MeasurementOutcomes) -> pecos_engines::ByteMessage {
    let mut builder = pecos_engines::ByteMessage::outcomes_builder();

    // Collect outcomes in order and convert to usize values (as expected by ByteMessageBuilder)
    let outcome_values: Vec<usize> = outcomes.iter().map(|o| usize::from(o.outcome)).collect();

    builder.add_outcomes(&outcome_values);
    builder.build()
}

/// Adapter that wraps a classical control engine to implement `CommandSource`.
///
/// This allows existing engines (`QASMEngine`, `HugrEngine`, etc.) to be used
/// with pecos-neo's `ProgramRunner` and sampling infrastructure.
pub struct ClassicalEngineAdapter<E> {
    /// The wrapped classical control engine.
    engine: E,
    /// Number of qubits in the program.
    num_qubits: usize,
    /// Current state: whether we've started and what stage we're at.
    state: AdapterState,
    /// Pending commands from the engine (if any).
    pending_commands: Option<CommandQueue>,
    /// Track measurement count for outcome ordering.
    measurement_count: usize,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AdapterState {
    /// Not yet started.
    NotStarted,
    /// Started, awaiting outcome processing.
    Running,
    /// Complete.
    Complete,
}
impl<E> ClassicalEngineAdapter<E>
where
    E: pecos_engines::ClassicalControlEngine,
{
    /// Create a new adapter wrapping the given engine.
    ///
    /// # Arguments
    /// * `engine` - The classical control engine to wrap
    pub fn new(engine: E) -> Self {
        let num_qubits = engine.num_qubits();
        Self {
            engine,
            num_qubits,
            state: AdapterState::NotStarted,
            pending_commands: None,
            measurement_count: 0,
        }
    }

    /// Get a reference to the underlying engine.
    #[must_use]
    pub fn engine(&self) -> &E {
        &self.engine
    }

    /// Get a mutable reference to the underlying engine.
    pub fn engine_mut(&mut self) -> &mut E {
        &mut self.engine
    }

    /// Convert pending outcomes to `ByteMessage` and continue processing.
    fn continue_with_outcomes(
        &mut self,
        outcomes: &MeasurementOutcomes,
    ) -> Result<Option<CommandQueue>, pecos_core::errors::PecosError> {
        use pecos_engines::EngineStage;

        let outcome_message = outcomes_to_byte_message(outcomes);
        let stage =
            pecos_engines::ControlEngine::continue_processing(&mut self.engine, outcome_message)?;

        match stage {
            EngineStage::NeedsProcessing(commands) => {
                let queue = byte_message_to_command_queue(&commands)?;
                Ok(Some(queue))
            }
            EngineStage::Complete(_shot) => {
                self.state = AdapterState::Complete;
                Ok(None)
            }
        }
    }

    /// Start the engine and get initial commands.
    fn start_engine(&mut self) -> Result<Option<CommandQueue>, pecos_core::errors::PecosError> {
        use pecos_engines::EngineStage;

        let stage = pecos_engines::ControlEngine::start(&mut self.engine, ())?;
        self.state = AdapterState::Running;

        match stage {
            EngineStage::NeedsProcessing(commands) => {
                let queue = byte_message_to_command_queue(&commands)?;
                Ok(Some(queue))
            }
            EngineStage::Complete(_shot) => {
                self.state = AdapterState::Complete;
                Ok(None)
            }
        }
    }
}
impl<E> CommandSource for ClassicalEngineAdapter<E>
where
    E: pecos_engines::ClassicalControlEngine,
{
    fn next_commands(&mut self, outcomes: Option<&MeasurementOutcomes>) -> Option<CommandQueue> {
        match self.state {
            AdapterState::NotStarted => {
                // Start the engine
                match self.start_engine() {
                    Ok(cmds) => cmds,
                    Err(e) => {
                        // Log error and complete
                        eprintln!("ClassicalEngineAdapter: start error: {e}");
                        self.state = AdapterState::Complete;
                        None
                    }
                }
            }
            AdapterState::Running => {
                // Continue with outcomes from previous batch
                if let Some(outcomes) = outcomes {
                    match self.continue_with_outcomes(outcomes) {
                        Ok(cmds) => cmds,
                        Err(e) => {
                            eprintln!("ClassicalEngineAdapter: continue error: {e}");
                            self.state = AdapterState::Complete;
                            None
                        }
                    }
                } else {
                    // No outcomes but running - shouldn't happen in normal flow
                    self.state = AdapterState::Complete;
                    None
                }
            }
            AdapterState::Complete => None,
        }
    }

    fn is_complete(&self) -> bool {
        self.state == AdapterState::Complete
    }

    fn reset(&mut self) {
        // Reset both the adapter state and the underlying engine
        self.state = AdapterState::NotStarted;
        self.pending_commands = None;
        self.measurement_count = 0;

        // Try to reset the engine (ignore errors - some engines don't need reset)
        let _ = pecos_engines::ControlEngine::reset(&mut self.engine);
    }

    fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    fn shot_results(&self) -> Option<pecos_results::Shot> {
        Some(
            self.engine
                .get_results()
                .expect("classical engine failed to produce results for a completed shot"),
        )
    }
}

/// Runner adapter for executing pecos-neo command sources through a
/// `pecos_engines::QuantumEngine`.
///
/// This lets `sim_neo()` accept the same Rust quantum engine builders as the
/// stable `sim()` API without identifying backends by name. The adapter only
/// translates between the two execution protocols; unsupported behavior is
/// rejected by the surrounding builder before this runner is constructed.
pub struct QuantumEngineProgramRunner {
    engine: Box<dyn pecos_engines::QuantumEngine>,
}
impl QuantumEngineProgramRunner {
    /// Create a new runner around a quantum engine.
    #[must_use]
    pub fn new(engine: Box<dyn pecos_engines::QuantumEngine>) -> Self {
        Self { engine }
    }

    fn commands_to_message(commands: &CommandQueue) -> pecos_engines::ByteMessage {
        let gates = command_queue_to_gates(commands);
        let mut builder = pecos_engines::ByteMessage::quantum_operations_builder();
        builder.add_gate_commands(&gates);
        builder.build()
    }

    fn measured_qubits(commands: &CommandQueue) -> Vec<QubitId> {
        commands
            .iter()
            .filter(|cmd| {
                matches!(
                    cmd.gate_type,
                    NeoGateType::MZ | NeoGateType::MeasureLeaked | NeoGateType::MeasureFree
                )
            })
            .flat_map(|cmd| cmd.qubits.iter().copied())
            .collect()
    }

    fn outcomes_from_message(
        message: &pecos_engines::ByteMessage,
        measured_qubits: &[QubitId],
    ) -> Result<MeasurementOutcomes, pecos_core::errors::PecosError> {
        let values = message.outcomes()?;
        if values.len() != measured_qubits.len() {
            return Err(pecos_core::errors::PecosError::Processing(format!(
                "quantum engine returned {} measurement outcomes for {} measured qubits",
                values.len(),
                measured_qubits.len()
            )));
        }

        let mut outcomes = MeasurementOutcomes::with_capacity(values.len());
        for (&qubit, value) in measured_qubits.iter().zip(values.iter().copied()) {
            match value {
                0 => outcomes.record(MeasurementOutcome::new(qubit, false, false)),
                1 => outcomes.record(MeasurementOutcome::new(qubit, true, false)),
                2 => outcomes.record(MeasurementOutcome::leaked(qubit)),
                other => {
                    return Err(pecos_core::errors::PecosError::Processing(format!(
                        "quantum engine returned invalid measurement outcome {other}"
                    )));
                }
            }
        }

        Ok(outcomes)
    }
}
impl DynProgramRunner for QuantumEngineProgramRunner {
    fn run_shot(&mut self, source: &mut dyn CommandSource) -> ProgramResult {
        source.reset();
        self.engine
            .reset()
            .expect("quantum engine reset should not fail");

        let mut all_outcomes = MeasurementOutcomes::new();
        let mut num_batches = 0;
        let mut last_outcomes: Option<MeasurementOutcomes> = None;

        loop {
            let commands = source.next_commands(last_outcomes.as_ref());

            match commands {
                Some(cmds) if !cmds.is_empty() => {
                    let measured_qubits = Self::measured_qubits(&cmds);
                    let message = Self::commands_to_message(&cmds);
                    let response = self
                        .engine
                        .process(message)
                        .expect("quantum engine command batch should execute");
                    let outcomes = Self::outcomes_from_message(&response, &measured_qubits)
                        .expect("quantum engine outcomes should match measured qubits");

                    num_batches += 1;
                    for outcome in outcomes.iter() {
                        all_outcomes.record(*outcome);
                    }
                    last_outcomes = Some(outcomes);
                }
                _ => break,
            }

            if source.is_complete() {
                break;
            }
        }

        ProgramResult {
            outcomes: all_outcomes,
            num_batches,
        }
    }

    fn set_full_seed(&mut self, seed: u64) {
        self.engine.set_seed(seed);
    }
}

// ============================================================================
// Gate conversion utilities (always available)
// ============================================================================

/// Convert a pecos-core Gate to a `GateCommand` (no `ByteMessage` dependency).
///
/// This is useful when you have raw Gate objects and want to convert them
/// to the pecos-neo format.
///
/// # Errors
///
/// Returns `PecosError` if the gate has no pecos-neo command representation.
pub fn gate_to_command(gate: &Gate) -> Result<GateCommand, pecos_core::errors::PecosError> {
    convert_gate(gate)
}

/// Convert a slice of pecos-core Gates to a `CommandQueue`.
///
/// # Errors
///
/// Returns `PecosError` if any gate has no pecos-neo command representation.
pub fn gates_to_command_queue(
    gates: &[Gate],
) -> Result<CommandQueue, pecos_core::errors::PecosError> {
    let mut queue = CommandQueue::with_capacity(gates.len());
    for gate in gates {
        queue.push(convert_gate(gate)?);
    }
    Ok(queue)
}

/// Convert a `CommandQueue` back to a Vec of pecos-core Gates.
///
/// This is useful for interoperability with code that expects Gate objects.
#[must_use]
pub fn command_queue_to_gates(queue: &CommandQueue) -> Vec<Gate> {
    queue.iter().map(command_to_gate).collect()
}

/// Convert a `GateCommand` back to a pecos-core Gate.
fn command_to_gate(cmd: &GateCommand) -> Gate {
    let core_type = convert_neo_to_core_gate_type(cmd.gate_type);
    let qubits: Vec<QubitId> = cmd.qubits.iter().copied().collect();
    let angles: Vec<Angle64> = cmd.angles.iter().copied().collect();

    Gate::new(core_type, angles, vec![], qubits)
}

/// Convert pecos-neo `GateType` back to pecos-core `GateType`.
fn convert_neo_to_core_gate_type(neo_type: NeoGateType) -> CoreGateType {
    match neo_type {
        NeoGateType::I => CoreGateType::I,
        NeoGateType::X => CoreGateType::X,
        NeoGateType::Y => CoreGateType::Y,
        NeoGateType::Z => CoreGateType::Z,
        NeoGateType::H => CoreGateType::H,
        NeoGateType::F => CoreGateType::F,
        NeoGateType::Fdg => CoreGateType::Fdg,
        NeoGateType::SX => CoreGateType::SX,
        NeoGateType::SXdg => CoreGateType::SXdg,
        NeoGateType::SY => CoreGateType::SY,
        NeoGateType::SYdg => CoreGateType::SYdg,
        NeoGateType::SZ => CoreGateType::SZ,
        NeoGateType::SZdg => CoreGateType::SZdg,
        NeoGateType::T => CoreGateType::T,
        NeoGateType::Tdg => CoreGateType::Tdg,
        NeoGateType::RX => CoreGateType::RX,
        NeoGateType::RY => CoreGateType::RY,
        NeoGateType::RZ => CoreGateType::RZ,
        NeoGateType::U => CoreGateType::U,
        NeoGateType::R1XY => CoreGateType::R1XY,
        NeoGateType::CX => CoreGateType::CX,
        NeoGateType::CY => CoreGateType::CY,
        NeoGateType::CZ => CoreGateType::CZ,
        NeoGateType::SZZ => CoreGateType::SZZ,
        NeoGateType::SZZdg => CoreGateType::SZZdg,
        NeoGateType::SXX => CoreGateType::SXX,
        NeoGateType::SXXdg => CoreGateType::SXXdg,
        NeoGateType::SYY => CoreGateType::SYY,
        NeoGateType::SYYdg => CoreGateType::SYYdg,
        NeoGateType::SWAP => CoreGateType::SWAP,
        NeoGateType::CRZ => CoreGateType::CRZ,
        NeoGateType::RXX => CoreGateType::RXX,
        NeoGateType::RYY => CoreGateType::RYY,
        NeoGateType::RZZ => CoreGateType::RZZ,
        NeoGateType::CCX => CoreGateType::CCX,
        NeoGateType::MZ => CoreGateType::MZ,
        NeoGateType::MeasureLeaked => CoreGateType::MeasureLeaked,
        NeoGateType::MeasureFree => CoreGateType::MeasureFree,
        NeoGateType::PZ => CoreGateType::PZ,
        NeoGateType::QAlloc => CoreGateType::QAlloc,
        NeoGateType::QFree => CoreGateType::QFree,
        NeoGateType::Idle => CoreGateType::Idle,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gate_type_conversion_roundtrip() {
        // Test that gate types can be converted back and forth
        let test_types = [
            CoreGateType::H,
            CoreGateType::X,
            CoreGateType::Y,
            CoreGateType::Z,
            CoreGateType::CX,
            CoreGateType::CZ,
            CoreGateType::RZ,
            CoreGateType::MZ,
            CoreGateType::PZ,
        ];

        for core_type in test_types {
            let neo_type = convert_gate_type(core_type).expect("should convert");
            let back = convert_neo_to_core_gate_type(neo_type);
            assert_eq!(core_type, back, "roundtrip failed for {core_type:?}");
        }
    }

    #[test]
    fn test_gate_conversion() {
        // Create a simple Gate and convert it
        let gate = Gate::new(CoreGateType::H, vec![], vec![], vec![QubitId(0)]);

        let cmd = gate_to_command(&gate).expect("should convert");
        assert_eq!(cmd.gate_type, NeoGateType::H);
        assert_eq!(cmd.qubits.as_slice(), &[QubitId(0)]);
    }

    #[test]
    fn test_parameterized_gate_conversion() {
        // Create an RZ gate with angle
        let angle = Angle64::from_radians(std::f64::consts::PI / 2.0);
        let gate = Gate::new(CoreGateType::RZ, vec![angle], vec![], vec![QubitId(0)]);

        let cmd = gate_to_command(&gate).expect("should convert");
        assert_eq!(cmd.gate_type, NeoGateType::RZ);
        assert_eq!(cmd.qubits.as_slice(), &[QubitId(0)]);
        assert_eq!(cmd.angles.len(), 1);
    }

    #[test]
    fn test_gates_to_command_queue() {
        let gates = vec![
            Gate::new(CoreGateType::H, vec![], vec![], vec![QubitId(0)]),
            Gate::new(
                CoreGateType::CX,
                vec![],
                vec![],
                vec![QubitId(0), QubitId(1)],
            ),
            Gate::new(CoreGateType::MZ, vec![], vec![], vec![QubitId(0)]),
        ];

        let queue = gates_to_command_queue(&gates).expect("should convert");
        assert_eq!(queue.len(), 3);
    }

    #[test]
    fn test_command_queue_to_gates_roundtrip() {
        let original_gates = vec![
            Gate::new(CoreGateType::H, vec![], vec![], vec![QubitId(0)]),
            Gate::new(
                CoreGateType::CX,
                vec![],
                vec![],
                vec![QubitId(0), QubitId(1)],
            ),
        ];

        let queue = gates_to_command_queue(&original_gates).expect("should convert");
        let back = command_queue_to_gates(&queue);

        assert_eq!(back.len(), 2);
        assert_eq!(back[0].gate_type, CoreGateType::H);
        assert_eq!(back[1].gate_type, CoreGateType::CX);
    }

    #[test]
    fn test_gate_to_command_rejects_channel_gate() {
        let gate = Gate::channel(pecos_core::channel::Depolarizing(0.01, 0));

        let err = gate_to_command(&gate).expect_err("channel gates need typed channel handling");

        assert!(
            err.to_string()
                .contains("does not support gate type Channel")
        );
    }
}
