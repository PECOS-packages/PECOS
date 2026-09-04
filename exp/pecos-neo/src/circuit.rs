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
//! use pecos_simulators::SparseStab;
//!
//! // Build a circuit using TickCircuit
//! let mut circuit = TickCircuit::new();
//! circuit.tick().pz(&[0, 1]);
//! circuit.tick().h(&[0]);
//! circuit.tick().cx(&[(0, 1)]);
//! circuit.tick().mz(&[0, 1]);
//!
//! // Convert to CommandQueue and execute
//! let commands = CommandQueue::try_from(&circuit).unwrap();
//! let mut state = SparseStab::new(2);
//! let mut runner = CircuitRunner::<SparseStab>::new().with_seed(42);
//! let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();
//! ```

use crate::command::{CommandQueue, GateCommand, GateCommandError, GateType};
use pecos_core::gate_type::GateType as CoreGateType;
use pecos_core::{Angle64, Gate, MeasId, QubitId};
use pecos_quantum::{DagCircuit, TickCircuit, TickGateError};
use smallvec::SmallVec;
use std::fmt;

// ============================================================================
// GateType Conversion
// ============================================================================

impl GateType {
    fn try_from_core(gt: CoreGateType) -> Option<Self> {
        use pecos_core::gate_type::GateType as CoreGT;
        Some(match gt {
            CoreGT::I => Self::I,
            CoreGT::X => Self::X,
            CoreGT::Y => Self::Y,
            CoreGT::Z => Self::Z,
            CoreGT::H => Self::H,
            CoreGT::F => Self::F,
            CoreGT::Fdg => Self::Fdg,
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
            CoreGT::RXY1Q => Self::RXY1Q,
            CoreGT::CX => Self::CX,
            CoreGT::CY => Self::CY,
            CoreGT::CZ => Self::CZ,
            CoreGT::SXX => Self::SXX,
            CoreGT::SXXdg => Self::SXXdg,
            CoreGT::SYY => Self::SYY,
            CoreGT::SYYdg => Self::SYYdg,
            CoreGT::SZZ => Self::SZZ,
            CoreGT::SZZdg => Self::SZZdg,
            CoreGT::SWAP => Self::SWAP,
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
            _ => return None,
        })
    }
}

impl From<CoreGateType> for GateType {
    fn from(gt: CoreGateType) -> Self {
        Self::try_from_core(gt).unwrap_or_else(|| {
            panic!("unsupported pecos-core gate type for pecos-neo conversion: {gt:?}")
        })
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
            GateType::F => CoreGT::F,
            GateType::Fdg => CoreGT::Fdg,
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
            GateType::RXY1Q => CoreGT::RXY1Q,
            GateType::CX => CoreGT::CX,
            GateType::CY => CoreGT::CY,
            GateType::CZ => CoreGT::CZ,
            GateType::SZZ => CoreGT::SZZ,
            GateType::SZZdg => CoreGT::SZZdg,
            GateType::SXX => CoreGT::SXX,
            GateType::SXXdg => CoreGT::SXXdg,
            GateType::SYY => CoreGT::SYY,
            GateType::SYYdg => CoreGT::SYYdg,
            GateType::SWAP => CoreGT::SWAP,
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

/// Error converting between core circuits and pecos-neo commands.
#[derive(Debug, Clone, PartialEq)]
pub enum CircuitConversionError {
    /// The core gate type has no pecos-neo command representation.
    UnsupportedGateType { gate_type: CoreGateType },
    /// A core gate failed its own payload validation.
    InvalidCoreGate {
        gate_type: CoreGateType,
        message: String,
    },
    /// An Idle batch has no target qubits.
    EmptyIdleBatch,
    /// A core floating-point Idle duration is not an exactly representable
    /// pecos-neo integer duration.
    InvalidCoreIdleDuration { duration: f64 },
    /// A command could not enter the destination queue.
    InvalidCommand(GateCommandError),
    /// Measurement records could not be reserved in the destination circuit.
    MeasurementRecords(String),
    /// A validated command could not be inserted into its destination tick.
    InvalidTickGate(TickGateError),
}

impl fmt::Display for CircuitConversionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedGateType { gate_type } => {
                write!(f, "pecos-neo does not support core gate type {gate_type:?}")
            }
            Self::InvalidCoreGate { message, .. } | Self::MeasurementRecords(message) => {
                f.write_str(message)
            }
            Self::EmptyIdleBatch => f.write_str("an Idle command must target at least one qubit"),
            Self::InvalidCoreIdleDuration { duration } => write!(
                f,
                "core Idle duration {duration:?} is not an exactly representable non-negative integer"
            ),
            Self::InvalidCommand(error) => error.fmt(f),
            Self::InvalidTickGate(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for CircuitConversionError {}

fn exact_nonnegative_f64_to_u64(value: f64) -> Option<u64> {
    if !value.is_finite() || value < 0.0 {
        return None;
    }
    if value == 0.0 {
        return Some(0);
    }

    let bits = value.to_bits();
    let raw_exponent = (bits >> 52) & 0x7ff;
    if raw_exponent < 1023 {
        return None;
    }
    let exponent = raw_exponent - 1023;
    if exponent > 63 {
        return None;
    }

    let significand = (1_u64 << 52) | (bits & ((1_u64 << 52) - 1));
    if exponent >= 52 {
        significand.checked_shl(u32::try_from(exponent - 52).ok()?)
    } else {
        let shift = exponent.abs_diff(52);
        let discarded_mask = (1_u64 << shift) - 1;
        (significand & discarded_mask == 0).then_some(significand >> shift)
    }
}

impl TryFrom<&Gate> for GateCommand {
    type Error = CircuitConversionError;

    fn try_from(gate: &Gate) -> Result<Self, Self::Error> {
        gate.validate()
            .map_err(|message| CircuitConversionError::InvalidCoreGate {
                gate_type: gate.gate_type,
                message,
            })?;
        let gate_type = GateType::try_from_core(gate.gate_type).ok_or(
            CircuitConversionError::UnsupportedGateType {
                gate_type: gate.gate_type,
            },
        )?;
        let qubits: SmallVec<[QubitId; 4]> = gate.qubits.iter().copied().collect();

        if gate_type == GateType::Idle {
            if qubits.is_empty() {
                return Err(CircuitConversionError::EmptyIdleBatch);
            }
            let duration = gate.params[0];
            let duration = exact_nonnegative_f64_to_u64(duration)
                .ok_or(CircuitConversionError::InvalidCoreIdleDuration { duration })?;
            return Ok(GateCommand::with_angles(
                GateType::Idle,
                qubits,
                smallvec::smallvec![Angle64::new(duration)],
            ));
        }

        let angles: SmallVec<[Angle64; 2]> = gate.angles.iter().copied().collect();

        Ok(GateCommand {
            gate_type,
            qubits,
            angles,
        })
    }
}

impl TryFrom<Gate> for GateCommand {
    type Error = CircuitConversionError;

    fn try_from(gate: Gate) -> Result<Self, Self::Error> {
        Self::try_from(&gate)
    }
}

// ============================================================================
// TickCircuit to CommandQueue Conversion
// ============================================================================

impl TryFrom<&TickCircuit> for CommandQueue {
    type Error = CircuitConversionError;

    /// Convert a `TickCircuit` to a `CommandQueue`.
    ///
    /// Gate batches are added in tick order - all commands from tick 0, then tick 1, etc.
    /// Within each tick, commands are added in the order they appear.
    fn try_from(circuit: &TickCircuit) -> Result<Self, Self::Error> {
        let mut queue = CommandQueue::new();

        for tick in circuit.ticks() {
            for gate in tick.iter_gate_batches() {
                let command = GateCommand::try_from(gate.as_gate())?;
                queue
                    .try_push(command)
                    .map_err(CircuitConversionError::InvalidCommand)?;
            }
        }

        Ok(queue)
    }
}

impl TryFrom<TickCircuit> for CommandQueue {
    type Error = CircuitConversionError;

    fn try_from(circuit: TickCircuit) -> Result<Self, Self::Error> {
        Self::try_from(&circuit)
    }
}

// ============================================================================
// DagCircuit to CommandQueue Conversion
// ============================================================================

impl TryFrom<&DagCircuit> for CommandQueue {
    type Error = CircuitConversionError;

    /// Convert a `DagCircuit` to a `CommandQueue`.
    ///
    /// Gates are added in topological order, ensuring that dependencies
    /// are respected.
    fn try_from(circuit: &DagCircuit) -> Result<Self, Self::Error> {
        let mut queue = CommandQueue::new();

        // Get gates in topological order
        for node_id in circuit.topological_order() {
            if let Some(gate) = circuit.gate(node_id) {
                let command = GateCommand::try_from(gate)?;
                queue
                    .try_push(command)
                    .map_err(CircuitConversionError::InvalidCommand)?;
            }
        }

        Ok(queue)
    }
}

impl TryFrom<DagCircuit> for CommandQueue {
    type Error = CircuitConversionError;

    fn try_from(circuit: DagCircuit) -> Result<Self, Self::Error> {
        Self::try_from(&circuit)
    }
}

// ============================================================================
// CommandQueue to TickCircuit Conversion (Round-trip support)
// ============================================================================

impl TryFrom<&CommandQueue> for TickCircuit {
    type Error = CircuitConversionError;

    /// Convert a `CommandQueue` to a `TickCircuit`.
    ///
    /// Each command becomes its own tick. For better parallelization,
    /// consider using the `CommandBuilder` to construct circuits directly,
    /// or manually building a `TickCircuit`.
    fn try_from(queue: &CommandQueue) -> Result<Self, Self::Error> {
        let mut circuit = TickCircuit::new();

        for cmd in queue.iter() {
            let mut gate = cmd.to_core_gate();
            if gate.gate_type.consumes_measurement_record() {
                let base = circuit
                    .try_advance_meas_counter(gate.qubits.len())
                    .map_err(CircuitConversionError::MeasurementRecords)?;
                gate.meas_ids
                    .extend((base..base + gate.qubits.len()).map(MeasId::from_raw));
            }
            let mut tick = circuit.tick();
            tick.try_add_gate(gate)
                .map_err(CircuitConversionError::InvalidTickGate)?;
        }

        Ok(circuit)
    }
}

impl TryFrom<CommandQueue> for TickCircuit {
    type Error = CircuitConversionError;

    fn try_from(queue: CommandQueue) -> Result<Self, Self::Error> {
        Self::try_from(&queue)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command::GateCommandError;
    use pecos_core::Angle64;

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

        let queue = CommandQueue::try_from(&circuit).expect("circuit should convert");

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

        let queue = CommandQueue::try_from(&circuit).expect("circuit should convert");

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
        dag.pz(&[0]);
        dag.pz(&[1]);
        dag.h(&[0]);
        dag.cx(&[(0, 1)]);
        dag.mz(&[0]);
        dag.mz(&[1]);

        let queue = CommandQueue::try_from(&dag).expect("DAG should convert");

        // Should have 6 commands
        assert_eq!(queue.len(), 6);

        // First two should be preps
        assert_eq!(queue.iter().next().unwrap().gate_type, GateType::PZ);
    }

    #[test]
    fn test_command_queue_to_tick_circuit() {
        use crate::command::CommandBuilder;

        let commands = CommandBuilder::new()
            .pz(&[0])
            .pz(&[1])
            .h(&[0])
            .cx(&[(0, 1)])
            .mz(&[0])
            .mz(&[1])
            .mz_free(&[2])
            .build();

        let circuit = TickCircuit::try_from(&commands).expect("commands should convert");

        // Each command becomes its own tick
        assert_eq!(circuit.num_ticks(), 7);
        assert_eq!(circuit.num_measurements(), 3);
        let measurement_ids: Vec<_> = circuit
            .ticks()
            .iter()
            .flat_map(pecos_quantum::Tick::iter_gate_batches)
            .flat_map(|gate| gate.as_gate().meas_ids.iter().copied())
            .collect();
        assert_eq!(
            measurement_ids,
            vec![
                MeasId::from_raw(0),
                MeasId::from_raw(1),
                MeasId::from_raw(2)
            ]
        );
    }

    #[test]
    fn malformed_fixed_gate_cannot_reach_tick_circuit_conversion() {
        let mut commands = CommandQueue::new();
        let error = commands
            .try_push(GateCommand::with_angles(
                GateType::H,
                smallvec::smallvec![QubitId(0)],
                smallvec::smallvec![Angle64::QUARTER_TURN],
            ))
            .expect_err("surplus angles must be rejected before TickCircuit conversion");
        let GateCommandError::AngleArity(error) = error else {
            panic!("wrong error variant");
        };
        assert_eq!(error.gate_type, GateType::H);
        assert_eq!(error.expected, 0);
        assert_eq!(error.actual, 1);

        commands.push(GateCommand::h(QubitId(0)));
        let circuit = TickCircuit::try_from(&commands).expect("valid commands should convert");
        let gate = circuit.ticks()[0]
            .iter_gate_batches()
            .next()
            .expect("converted tick contains H")
            .as_gate();
        assert_eq!(gate.gate_type, pecos_core::gate_type::GateType::H);
        assert!(gate.angles.is_empty());
    }

    #[test]
    fn test_gate_conversion_with_angles() {
        use pecos_core::gate_type::GateType as CoreGT;

        let gate = Gate {
            gate_type: CoreGT::RZ,
            angles: smallvec::smallvec![Angle64::QUARTER_TURN],
            params: SmallVec::new(),
            qubits: smallvec::smallvec![QubitId(0)],
            meas_ids: SmallVec::new(),
            channel: None,
        };

        let cmd = GateCommand::try_from(&gate).expect("gate should convert");

        assert_eq!(cmd.gate_type, GateType::RZ);
        assert_eq!(cmd.angles.len(), 1);
        assert_eq!(cmd.angles[0], Angle64::QUARTER_TURN);
        assert_eq!(cmd.qubits.len(), 1);
        assert_eq!(cmd.qubits[0], QubitId(0));
    }

    #[test]
    fn batched_idle_round_trips_without_losing_targets_or_duration() {
        let gate = Gate::idle(23.0, vec![QubitId(1), QubitId(2)]);

        let command = GateCommand::try_from(&gate).expect("integral Idle should convert");
        assert_eq!(command.qubits.as_slice(), &[QubitId(1), QubitId(2)]);
        assert_eq!(
            command.get_idle_duration(),
            Some(pecos_core::TimeUnits::new(23))
        );

        let mut queue = CommandQueue::new();
        queue
            .try_push(command)
            .expect("Idle command should enter queue");
        let back = queue.iter().next().expect("Idle command").to_core_gate();
        assert_eq!(back.qubits.as_slice(), &[QubitId(1), QubitId(2)]);
        assert!((back.idle_duration() - 23.0).abs() < f64::EPSILON);
    }

    #[test]
    fn core_idle_conversion_rejects_empty_and_non_integer_durations() {
        let empty = Gate::idle(23.0, Vec::<QubitId>::new());
        assert_eq!(
            GateCommand::try_from(&empty),
            Err(CircuitConversionError::EmptyIdleBatch)
        );

        for duration in [23.5, -1.0, f64::INFINITY, f64::NAN] {
            let gate = Gate::idle(duration, vec![QubitId(0)]);
            let error = GateCommand::try_from(&gate)
                .expect_err("non-integral, negative, and non-finite durations must fail");
            assert!(matches!(
                error,
                CircuitConversionError::InvalidCoreIdleDuration { duration: actual }
                    if actual.to_bits() == duration.to_bits()
            ));
        }
    }
}
