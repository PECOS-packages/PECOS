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

use pecos_core::{Angle64, Gate, QubitId, Signal, TimeUnits};
use smallvec::SmallVec;
use std::fmt;

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
    F,
    Fdg,
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
    RXY1Q,

    // Two-qubit gates
    CX,
    CY,
    CZ,
    SZZ,
    SZZdg,
    SXX,
    SXXdg,
    SYY,
    SYYdg,
    SWAP,
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
            | Self::F
            | Self::Fdg
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
            | Self::RXY1Q
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
            | Self::SXX
            | Self::SXXdg
            | Self::SYY
            | Self::SYYdg
            | Self::SWAP
            | Self::RXX
            | Self::RYY
            | Self::RZZ => 2,

            Self::CCX => 3,
        }
    }

    /// Returns the number of angle parameters this gate requires.
    #[must_use]
    pub fn angle_arity(self) -> usize {
        pecos_core::gate_type::GateType::from(self).angle_arity()
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

    /// Returns true if this is an idle operation.
    #[must_use]
    pub const fn is_idle(self) -> bool {
        matches!(self, Self::Idle)
    }

    /// Returns true if this is a resource management operation.
    #[must_use]
    pub const fn is_resource_management(self) -> bool {
        matches!(self, Self::QAlloc | Self::QFree)
    }

    /// Returns true if this is a unitary gate (not preparation, measurement,
    /// idle, or resource management).
    ///
    /// These are the gates that should receive gate depolarizing noise
    /// (p1 for single-qubit, p2 for two-qubit).
    #[must_use]
    pub const fn is_unitary_gate(self) -> bool {
        !self.is_measurement()
            && !self.is_preparation()
            && !self.is_idle()
            && !self.is_resource_management()
    }
}

/// Error returned when a command has the wrong number of angle values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GateCommandAngleArityError {
    /// The command gate type being validated.
    pub gate_type: GateType,
    /// The number of angles required by the command representation.
    pub expected: usize,
    /// The number of angles supplied by the command.
    pub actual: usize,
}

impl fmt::Display for GateCommandAngleArityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Gate {:?} expected {} angle parameters, got {}",
            self.gate_type, self.expected, self.actual
        )
    }
}

impl std::error::Error for GateCommandAngleArityError {}

/// Error returned by command validation or conversion to a core gate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GateCommandError {
    /// The command carries the wrong number of angles.
    AngleArity(GateCommandAngleArityError),
    /// The corresponding core gate payload is invalid.
    InvalidGate {
        /// The command gate type being validated.
        gate_type: GateType,
        /// The validation failure reported by core gate validation.
        message: String,
    },
    /// An Idle command has no target qubits.
    EmptyIdleBatch,
    /// An Idle duration cannot be represented exactly by the core `f64`
    /// duration field.
    IdleDurationNotRepresentable { duration: u64 },
}

impl fmt::Display for GateCommandError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AngleArity(error) => error.fmt(f),
            Self::InvalidGate { message, .. } => f.write_str(message),
            Self::EmptyIdleBatch => f.write_str("an Idle command must target at least one qubit"),
            Self::IdleDurationNotRepresentable { duration } => write!(
                f,
                "Idle duration {duration} cannot be represented exactly as a core f64 duration"
            ),
        }
    }
}

impl std::error::Error for GateCommandError {}

fn u64_is_exactly_representable_as_f64(value: u64) -> bool {
    if value == 0 {
        return true;
    }
    let significant_bits = u64::BITS - value.leading_zeros();
    significant_bits <= 53 || value.trailing_zeros() >= significant_bits - 53
}

/// Rotation angles or an idle duration for a gate command.
#[derive(Debug, Clone, PartialEq)]
pub enum GatePayload {
    /// Rotation angles, empty for non-parameterized gates.
    Angles(SmallVec<[Angle64; 2]>),
    /// An idle duration in abstract time units.
    Duration(TimeUnits),
}

/// A single quantum gate command.
///
/// This is a typed representation of a gate operation with its target qubits
/// and either rotation angles or an idle duration.
#[derive(Debug, Clone, PartialEq)]
pub struct GateCommand {
    /// The type of gate to apply.
    pub gate_type: GateType,

    /// The target qubits for this gate.
    /// Uses `SmallVec` to avoid heap allocation for common cases (1-4 qubits).
    pub qubits: SmallVec<[QubitId; 4]>,

    /// Rotation angles or an idle duration, never both.
    pub payload: GatePayload,
}

impl GateCommand {
    /// Create a new gate command.
    #[must_use]
    pub fn new(gate_type: GateType, qubits: impl Into<SmallVec<[QubitId; 4]>>) -> Self {
        Self {
            gate_type,
            qubits: qubits.into(),
            payload: GatePayload::Angles(SmallVec::new()),
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
            payload: GatePayload::Angles(angles.into()),
        }
    }

    /// Get the rotation angles. Duration payloads have no angles.
    #[must_use]
    pub fn angles(&self) -> &[Angle64] {
        match &self.payload {
            GatePayload::Angles(angles) => angles,
            GatePayload::Duration(_) => &[],
        }
    }

    /// Create an identity gate on a qubit.
    #[must_use]
    pub fn identity(qubit: QubitId) -> Self {
        Self::new(GateType::I, smallvec::smallvec![qubit])
    }

    /// The inverse (dagger) of this gate, if it is an invertible unitary;
    /// `None` otherwise (measurement, prep, idle, resource management).
    ///
    /// Used by the gate-removing emission noise model: when a gate suffers a
    /// spontaneous-emission error, undoing the gate (`G dagger`) and then
    /// applying the emission error reproduces engines' "the emission replaces
    /// the gate" semantics, since `G * G_dagger = I`.
    ///
    /// Rotation inverses use the standard conventions
    /// (`RX/RY/RZ(theta)` and `RXX/RYY/RZZ(theta)` -> negate `theta`;
    /// `RXY1Q(theta, phi)` -> `RXY1Q(-theta, phi)`; `U(theta, phi, lambda)` ->
    /// `U(-theta, -lambda, -phi)`).
    #[must_use]
    pub fn dagger(&self) -> Option<GateCommand> {
        let q = self.qubits.clone();
        let same = |t: GateType| Some(Self::new(t, q.clone()));
        let neg_first = |t: GateType| -> Option<Self> {
            let theta = *self.angles().first()?;
            Some(Self::with_angles(t, q.clone(), smallvec::smallvec![-theta]))
        };
        match self.gate_type {
            // Self-inverse gates (1q, 2q, 3q).
            GateType::I
            | GateType::X
            | GateType::Y
            | GateType::Z
            | GateType::H
            | GateType::CX
            | GateType::CY
            | GateType::CZ
            | GateType::SWAP
            | GateType::CCX => same(self.gate_type),
            // Dagger pairs.
            GateType::F => same(GateType::Fdg),
            GateType::Fdg => same(GateType::F),
            GateType::SX => same(GateType::SXdg),
            GateType::SXdg => same(GateType::SX),
            GateType::SY => same(GateType::SYdg),
            GateType::SYdg => same(GateType::SY),
            GateType::SZ => same(GateType::SZdg),
            GateType::SZdg => same(GateType::SZ),
            GateType::T => same(GateType::Tdg),
            GateType::Tdg => same(GateType::T),
            GateType::SZZ => same(GateType::SZZdg),
            GateType::SZZdg => same(GateType::SZZ),
            GateType::SXX => same(GateType::SXXdg),
            GateType::SXXdg => same(GateType::SXX),
            GateType::SYY => same(GateType::SYYdg),
            GateType::SYYdg => same(GateType::SYY),
            // Single-angle rotations (1q and 2q): negate the angle.
            GateType::RX
            | GateType::RY
            | GateType::RZ
            | GateType::RXX
            | GateType::RYY
            | GateType::RZZ => neg_first(self.gate_type),
            // RXY1Q(theta, phi) dagger = RXY1Q(-theta, phi).
            GateType::RXY1Q => {
                let theta = *self.angles().first()?;
                let phi = *self.angles().get(1)?;
                Some(Self::with_angles(
                    GateType::RXY1Q,
                    q,
                    smallvec::smallvec![-theta, phi],
                ))
            }
            // U(theta, phi, lambda) dagger = U(-theta, -lambda, -phi).
            GateType::U => {
                let theta = *self.angles().first()?;
                let phi = *self.angles().get(1)?;
                let lambda = *self.angles().get(2)?;
                Some(Self::with_angles(
                    GateType::U,
                    q,
                    smallvec::smallvec![-theta, -lambda, -phi],
                ))
            }
            _ => None,
        }
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
    /// Use [`Self::get_idle_duration`] to retrieve the duration.
    ///
    /// Time units are abstract - the interpretation (nanoseconds, clock cycles, etc.)
    /// is defined by the noise model configuration.
    #[must_use]
    pub fn idle(qubit: QubitId, duration: TimeUnits) -> Self {
        Self {
            gate_type: GateType::Idle,
            qubits: smallvec::smallvec![qubit],
            payload: GatePayload::Duration(duration),
        }
    }

    /// Get the idle duration for an Idle gate.
    ///
    /// Returns `None` if this is not an Idle gate or has no duration.
    #[must_use]
    pub fn get_idle_duration(&self) -> Option<TimeUnits> {
        match (self.gate_type, &self.payload) {
            (GateType::Idle, GatePayload::Duration(duration)) => Some(*duration),
            _ => None,
        }
    }

    /// Validate the command's payload and qubit support.
    ///
    /// # Errors
    /// Returns an error for invalid arity or qubit support.
    pub fn validate(&self) -> Result<(), GateCommandError> {
        let expected = self.gate_type.angle_arity();
        let actual = self.angles().len();
        if actual != expected {
            return Err(GateCommandError::AngleArity(GateCommandAngleArityError {
                gate_type: self.gate_type,
                expected,
                actual,
            }));
        }
        let gate = if self.gate_type == GateType::Idle {
            if self.qubits.is_empty() {
                return Err(GateCommandError::EmptyIdleBatch);
            }
            let duration =
                self.get_idle_duration()
                    .ok_or_else(|| GateCommandError::InvalidGate {
                        gate_type: self.gate_type,
                        message: "an Idle command requires a duration payload".to_string(),
                    })?;
            // Core validation checks support and parameter count. Only
            // try_to_core_gate may return this floating-point representation.
            Gate::idle(duration.as_f64(), self.qubits.clone())
        } else {
            if matches!(self.payload, GatePayload::Duration(_)) {
                return Err(GateCommandError::InvalidGate {
                    gate_type: self.gate_type,
                    message: "only an Idle command can carry a duration payload".to_string(),
                });
            }
            Gate::new(
                self.gate_type.into(),
                self.angles().to_vec(),
                Vec::new(),
                self.qubits.clone(),
            )
        };
        gate.validate()
            .map_err(|message| GateCommandError::InvalidGate {
                gate_type: self.gate_type,
                message,
            })
    }

    /// Convert to a core gate without truncating Idle durations.
    ///
    /// # Errors
    /// Returns an error for an invalid command or an inexact duration conversion.
    pub fn try_to_core_gate(&self) -> Result<Gate, GateCommandError> {
        self.validate()?;
        if let Some(time) = self.get_idle_duration() {
            let duration = time.as_u64();
            if !u64_is_exactly_representable_as_f64(duration) {
                return Err(GateCommandError::IdleDurationNotRepresentable { duration });
            }
            return Ok(Gate::idle(time.as_f64(), self.qubits.clone()));
        }
        Ok(Gate::new(
            self.gate_type.into(),
            self.angles().to_vec(),
            Vec::new(),
            self.qubits.clone(),
        ))
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

    /// Add a command without validation. Collection and builders preserve input;
    /// execution and conversion boundaries report invalid commands as errors.
    pub fn push(&mut self, command: GateCommand) {
        self.commands.push(command);
    }

    /// Add a command after checking that it converts losslessly to a core gate.
    ///
    /// # Errors
    /// Returns a command validation or representation error without changing the queue.
    pub fn try_push(&mut self, command: GateCommand) -> Result<(), GateCommandError> {
        command.try_to_core_gate()?;
        self.commands.push(command);
        Ok(())
    }

    /// Validate all commands for native execution without converting durations to f64.
    ///
    /// # Errors
    /// Returns the first invalid command payload or qubit-support error.
    pub fn validate_for_execution(&self) -> Result<(), GateCommandError> {
        for command in self {
            command.validate()?;
        }
        Ok(())
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

    pub(crate) fn clear_commands(&mut self) {
        self.commands.clear();
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
        let iter = iter.into_iter();
        let mut queue = Self::with_capacity(iter.size_hint().0);
        for command in iter {
            queue.push(command);
        }
        queue
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
        assert!(x.angles().is_empty());

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
        assert_eq!(GateType::RXY1Q.angle_arity(), 2);
        assert_eq!(GateType::U.angle_arity(), 3);
        assert_eq!(GateType::H.angle_arity(), 0);
        assert_eq!(GateType::Idle.angle_arity(), 0);
    }

    #[test]
    fn try_push_rejects_wrong_angle_arity_without_changing_queue() {
        let mut queue = CommandQueue::new();
        let error = queue
            .try_push(GateCommand::new(
                GateType::RZ,
                smallvec::smallvec![QubitId(0)],
            ))
            .expect_err("a malformed rotation must not enter a command queue");

        let GateCommandError::AngleArity(error) = error else {
            panic!("wrong error variant");
        };
        assert_eq!(error.gate_type, GateType::RZ);
        assert_eq!(error.expected, 1);
        assert_eq!(error.actual, 0);
        assert!(queue.is_empty());
    }

    #[test]
    fn try_push_rejects_surplus_angles_on_fixed_gate() {
        let mut queue = CommandQueue::new();
        let error = queue
            .try_push(GateCommand::with_angles(
                GateType::H,
                smallvec::smallvec![QubitId(0)],
                smallvec::smallvec![Angle64::QUARTER_TURN],
            ))
            .expect_err("a fixed gate must not silently discard supplied angles");

        let GateCommandError::AngleArity(error) = error else {
            panic!("wrong error variant");
        };
        assert_eq!(error.gate_type, GateType::H);
        assert_eq!(error.expected, 0);
        assert_eq!(error.actual, 1);
        assert!(queue.is_empty());
    }

    #[test]
    fn try_push_rejects_invalid_core_gate_payload_without_changing_queue() {
        let mut queue = CommandQueue::new();
        let error = queue
            .try_push(GateCommand::cx(QubitId(0), QubitId(0)))
            .expect_err("a command with duplicate operands must not enter the queue");

        let GateCommandError::InvalidGate { gate_type, message } = error else {
            panic!("wrong error variant");
        };
        assert_eq!(gate_type, GateType::CX);
        assert!(message.contains("requires distinct qubits"));
        assert!(queue.is_empty());
    }

    #[test]
    fn try_push_rejects_idle_that_core_cannot_represent_losslessly() {
        let mut queue = CommandQueue::new();
        let duration = (1_u64 << 53) + 1;
        let error = queue
            .try_push(GateCommand::idle(QubitId(0), TimeUnits::new(duration)))
            .expect_err("an inexact core f64 conversion must be rejected");

        assert_eq!(
            error,
            GateCommandError::IdleDurationNotRepresentable { duration }
        );
        assert!(queue.is_empty());

        let exactly_representable = 1_u64 << 54;
        queue
            .try_push(GateCommand::idle(
                QubitId(0),
                TimeUnits::new(exactly_representable),
            ))
            .expect("exact powers of two remain representable above 2^53");
    }

    #[test]
    fn try_push_rejects_empty_idle_batch_without_panicking() {
        let mut queue = CommandQueue::new();
        let error = queue
            .try_push(GateCommand {
                gate_type: GateType::Idle,
                qubits: SmallVec::new(),
                payload: GatePayload::Duration(TimeUnits::ZERO),
            })
            .expect_err("a zero-width Idle must be rejected before conversion");

        assert_eq!(error, GateCommandError::EmptyIdleBatch);
        assert!(queue.is_empty());
    }
}
