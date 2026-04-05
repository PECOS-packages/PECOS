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

//! Typed gate commands for quantum circuits.
//!
//! This module provides strongly-typed representations of quantum gate operations,
//! replacing the generic `ByteMessage` format with structured data.

mod builder;
pub(crate) mod signal_store;

pub use builder::CommandBuilder;
pub use signal_store::{SignalIter, SignalStore};

use pecos_core::{Angle64, QubitId, Signal, TimeUnits};
use smallvec::SmallVec;

/// The type of a quantum gate operation.
///
/// This mirrors `pecos_core::gate_type::GateType` but is scoped to ECS usage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GateType {
    // Single-qubit Paulis
    I,
    X,
    Y,
    Z,

    // Single-qubit Cliffords
    H,
    SX,
    SXdg,
    SY,
    SYdg,
    SZ,
    SZdg,
    T,
    Tdg,

    // Single-qubit rotations
    RX,
    RY,
    RZ,
    U,
    R1XY,

    // Two-qubit gates
    CX,
    CY,
    CZ,
    SZZ,
    SZZdg,
    SWAP,
    CRZ,
    RXX,
    RYY,
    RZZ,

    // Three-qubit gates
    CCX,

    // Measurement and preparation
    MZ,
    MeasureLeaked,
    MeasureFree,
    PZ,
    QAlloc,
    QFree,

    // Idle
    Idle,
}

impl GateType {
    /// Returns the number of qubits this gate operates on.
    #[must_use]
    pub const fn quantum_arity(self) -> usize {
        match self {
            Self::I
            | Self::X
            | Self::Y
            | Self::Z
            | Self::H
            | Self::SX
            | Self::SXdg
            | Self::SY
            | Self::SYdg
            | Self::SZ
            | Self::SZdg
            | Self::T
            | Self::Tdg
            | Self::RX
            | Self::RY
            | Self::RZ
            | Self::U
            | Self::R1XY
            | Self::MZ
            | Self::MeasureLeaked
            | Self::MeasureFree
            | Self::PZ
            | Self::QAlloc
            | Self::QFree
            | Self::Idle => 1,

            Self::CX
            | Self::CY
            | Self::CZ
            | Self::SZZ
            | Self::SZZdg
            | Self::SWAP
            | Self::CRZ
            | Self::RXX
            | Self::RYY
            | Self::RZZ => 2,

            Self::CCX => 3,
        }
    }

    /// Returns the number of angle parameters this gate requires.
    #[must_use]
    pub const fn angle_arity(self) -> usize {
        match self {
            Self::RX | Self::RY | Self::RZ | Self::RXX | Self::RYY | Self::RZZ | Self::CRZ => 1,
            Self::R1XY => 2,
            Self::U => 3,
            _ => 0,
        }
    }

    /// Returns true if this is a single-qubit gate.
    #[must_use]
    pub const fn is_single_qubit(self) -> bool {
        self.quantum_arity() == 1
    }

    /// Returns true if this is a two-qubit gate.
    #[must_use]
    pub const fn is_two_qubit(self) -> bool {
        self.quantum_arity() == 2
    }

    /// Returns true if this is a measurement operation.
    #[must_use]
    pub const fn is_measurement(self) -> bool {
        matches!(self, Self::MZ | Self::MeasureLeaked | Self::MeasureFree)
    }

    /// Returns true if this is a preparation operation.
    #[must_use]
    pub const fn is_preparation(self) -> bool {
        matches!(self, Self::PZ | Self::QAlloc)
    }
}

/// A single quantum gate command.
///
/// This is a typed representation of a gate operation with its target qubits
/// and any angle parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct GateCommand {
    /// The type of gate to apply.
    pub gate_type: GateType,

    /// The target qubits for this gate.
    /// Uses `SmallVec` to avoid heap allocation for common cases (1-4 qubits).
    pub qubits: SmallVec<[QubitId; 4]>,

    /// Angle parameters for parameterized gates.
    /// Empty for non-parameterized gates.
    pub angles: SmallVec<[Angle64; 2]>,
}

impl GateCommand {
    /// Create a new gate command.
    #[must_use]
    pub fn new(gate_type: GateType, qubits: impl Into<SmallVec<[QubitId; 4]>>) -> Self {
        Self {
            gate_type,
            qubits: qubits.into(),
            angles: SmallVec::new(),
        }
    }

    /// Create a new gate command with angle parameters.
    #[must_use]
    pub fn with_angles(
        gate_type: GateType,
        qubits: impl Into<SmallVec<[QubitId; 4]>>,
        angles: impl Into<SmallVec<[Angle64; 2]>>,
    ) -> Self {
        Self {
            gate_type,
            qubits: qubits.into(),
            angles: angles.into(),
        }
    }

    /// Create an identity gate on a qubit.
    #[must_use]
    pub fn identity(qubit: QubitId) -> Self {
        Self::new(GateType::I, smallvec::smallvec![qubit])
    }

    /// Create a Pauli-X gate on a qubit.
    #[must_use]
    pub fn x(qubit: QubitId) -> Self {
        Self::new(GateType::X, smallvec::smallvec![qubit])
    }

    /// Create a Pauli-Y gate on a qubit.
    #[must_use]
    pub fn y(qubit: QubitId) -> Self {
        Self::new(GateType::Y, smallvec::smallvec![qubit])
    }

    /// Create a Pauli-Z gate on a qubit.
    #[must_use]
    pub fn z(qubit: QubitId) -> Self {
        Self::new(GateType::Z, smallvec::smallvec![qubit])
    }

    /// Create a Hadamard gate on a qubit.
    #[must_use]
    pub fn h(qubit: QubitId) -> Self {
        Self::new(GateType::H, smallvec::smallvec![qubit])
    }

    /// Create an SZ (sqrt-Z) gate on a qubit.
    #[must_use]
    pub fn sz(qubit: QubitId) -> Self {
        Self::new(GateType::SZ, smallvec::smallvec![qubit])
    }

    /// Create a CNOT gate.
    #[must_use]
    pub fn cx(control: QubitId, target: QubitId) -> Self {
        Self::new(GateType::CX, smallvec::smallvec![control, target])
    }

    /// Create a CZ gate.
    #[must_use]
    pub fn cz(qubit0: QubitId, qubit1: QubitId) -> Self {
        Self::new(GateType::CZ, smallvec::smallvec![qubit0, qubit1])
    }

    /// Create an RZ rotation gate.
    #[must_use]
    pub fn rz(qubit: QubitId, angle: Angle64) -> Self {
        Self::with_angles(
            GateType::RZ,
            smallvec::smallvec![qubit],
            smallvec::smallvec![angle],
        )
    }

    /// Create an RZZ rotation gate.
    #[must_use]
    pub fn rzz(qubit0: QubitId, qubit1: QubitId, angle: Angle64) -> Self {
        Self::with_angles(
            GateType::RZZ,
            smallvec::smallvec![qubit0, qubit1],
            smallvec::smallvec![angle],
        )
    }

    /// Create a Z-basis preparation gate.
    #[must_use]
    pub fn pz(qubit: QubitId) -> Self {
        Self::new(GateType::PZ, smallvec::smallvec![qubit])
    }

    /// Create a Z-basis measurement gate.
    #[must_use]
    pub fn mz(qubit: QubitId) -> Self {
        Self::new(GateType::MZ, smallvec::smallvec![qubit])
    }

    /// Create an idle gate with a specified duration.
    ///
    /// The duration is stored in the angles field as abstract time units.
    /// Use [`Self::get_idle_duration`] to retrieve the duration.
    ///
    /// Time units are abstract - the interpretation (nanoseconds, clock cycles, etc.)
    /// is defined by the noise model configuration.
    #[must_use]
    pub fn idle(qubit: QubitId, duration: TimeUnits) -> Self {
        // Store duration in the angles field (repurposing Angle64's u64 storage)
        Self::with_angles(
            GateType::Idle,
            smallvec::smallvec![qubit],
            smallvec::smallvec![Angle64::new(duration.as_u64())],
        )
    }

    /// Get the idle duration for an Idle gate.
    ///
    /// Returns `None` if this is not an Idle gate or has no duration.
    #[must_use]
    pub fn get_idle_duration(&self) -> Option<TimeUnits> {
        if self.gate_type == GateType::Idle {
            self.angles.first().map(|a| TimeUnits::new(a.fraction()))
        } else {
            None
        }
    }
}

/// A queue of gate commands representing a quantum circuit or layer.
///
/// In addition to gate commands, a `CommandQueue` can carry typed **signals**:
/// user-defined metadata that flows alongside gates in the command stream.
/// See [`Signal`] for details.
#[derive(Debug, Clone, Default)]
pub struct CommandQueue {
    commands: Vec<GateCommand>,
    signals: SignalStore,
}

impl CommandQueue {
    /// Create an empty command queue.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a command queue with pre-allocated capacity.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            commands: Vec::with_capacity(capacity),
            signals: SignalStore::default(),
        }
    }

    /// Add a command to the queue.
    pub fn push(&mut self, command: GateCommand) {
        self.commands.push(command);
    }

    /// Push a signal at the current position (after the last pushed command).
    ///
    /// The signal is associated with the current length of the command stream,
    /// so it will be dispatched between the commands before and after it.
    ///
    /// ```
    /// use pecos_neo::command::CommandQueue;
    /// use pecos_core::impl_signal;
    ///
    /// #[derive(Copy, Clone, Debug)]
    /// struct Temperature(pub f64);
    /// impl_signal!(Temperature);
    ///
    /// let mut queue = CommandQueue::new();
    /// queue.push(pecos_neo::command::GateCommand::h(0.into()));
    /// queue.signal(Temperature(300.0));  // positioned after the H gate
    /// queue.push(pecos_neo::command::GateCommand::h(1.into()));
    ///
    /// assert!(queue.has_signals());
    /// assert_eq!(queue.iter_signals::<Temperature>().len(), 1);
    /// ```
    pub fn signal<S: Signal>(&mut self, signal: S) {
        #[allow(clippy::cast_possible_truncation)] // command count fits in u32
        let position = self.commands.len() as u32;
        self.signals.push(position, signal);
    }

    /// Push a signal at a specific command index.
    pub fn signal_at<S: Signal>(&mut self, index: u32, signal: S) {
        self.signals.push(index, signal);
    }

    /// Check if any signals are present.
    #[must_use]
    pub fn has_signals(&self) -> bool {
        !self.signals.is_empty()
    }

    /// Iterate over signals of a specific type, yielding `(position, &S)`.
    #[must_use]
    pub fn iter_signals<S: Signal>(&self) -> SignalIter<'_, S> {
        self.signals.iter()
    }

    /// Get the signal store (used by the runner for signal dispatch).
    pub(crate) fn signals(&self) -> &SignalStore {
        &self.signals
    }

    /// Get the number of gate commands in the queue (excluding signals).
    #[must_use]
    pub fn len(&self) -> usize {
        self.commands.len()
    }

    /// Check if the queue has no gate commands.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }

    /// Iterate over gate commands (ignoring signals).
    pub fn iter(&self) -> impl Iterator<Item = &GateCommand> {
        self.commands.iter()
    }

    /// Clear all gate commands and signals from the queue.
    pub fn clear(&mut self) {
        self.commands.clear();
        self.signals.clear();
    }

    /// Get the gate commands as a slice.
    #[must_use]
    pub fn as_slice(&self) -> &[GateCommand] {
        &self.commands
    }
}

impl FromIterator<GateCommand> for CommandQueue {
    fn from_iter<I: IntoIterator<Item = GateCommand>>(iter: I) -> Self {
        Self {
            commands: iter.into_iter().collect(),
            signals: SignalStore::default(),
        }
    }
}

impl<'a> IntoIterator for &'a CommandQueue {
    type Item = &'a GateCommand;
    type IntoIter = std::slice::Iter<'a, GateCommand>;

    fn into_iter(self) -> Self::IntoIter {
        self.commands.iter()
    }
}

impl IntoIterator for CommandQueue {
    type Item = GateCommand;
    type IntoIter = std::vec::IntoIter<GateCommand>;

    fn into_iter(self) -> Self::IntoIter {
        self.commands.into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gate_command_creation() {
        let x = GateCommand::x(QubitId(0));
        assert_eq!(x.gate_type, GateType::X);
        assert_eq!(x.qubits.as_slice(), &[QubitId(0)]);
        assert!(x.angles.is_empty());

        let cx = GateCommand::cx(QubitId(0), QubitId(1));
        assert_eq!(cx.gate_type, GateType::CX);
        assert_eq!(cx.qubits.as_slice(), &[QubitId(0), QubitId(1)]);
    }

    #[test]
    fn test_command_queue() {
        let mut queue = CommandQueue::new();
        assert!(queue.is_empty());

        queue.push(GateCommand::h(QubitId(0)));
        queue.push(GateCommand::cx(QubitId(0), QubitId(1)));

        assert_eq!(queue.len(), 2);
        assert!(!queue.is_empty());
    }

    #[test]
    fn test_gate_type_arity() {
        assert_eq!(GateType::X.quantum_arity(), 1);
        assert_eq!(GateType::CX.quantum_arity(), 2);
        assert_eq!(GateType::CCX.quantum_arity(), 3);

        assert_eq!(GateType::RZ.angle_arity(), 1);
        assert_eq!(GateType::R1XY.angle_arity(), 2);
        assert_eq!(GateType::U.angle_arity(), 3);
        assert_eq!(GateType::H.angle_arity(), 0);
    }
}
