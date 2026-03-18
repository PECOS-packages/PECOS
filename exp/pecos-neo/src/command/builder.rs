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

//! Ergonomic builder for constructing command queues.

use super::{CommandQueue, GateCommand, GateType};
use pecos_core::{Angle64, QubitId, Signal, TimeUnits};

/// Builder for constructing command queues with a fluent API.
///
/// # Example
///
/// ```
/// use pecos_neo::command::CommandBuilder;
///
/// let commands = CommandBuilder::new()
///     .pz(0)
///     .pz(1)
///     .h(0)
///     .cx(0, 1)
///     .mz(0)
///     .mz(1)
///     .build();
/// ```
#[derive(Debug, Default)]
pub struct CommandBuilder {
    queue: CommandQueue,
}

impl CommandBuilder {
    /// Create a new empty command builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a builder with pre-allocated capacity.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            queue: CommandQueue::with_capacity(capacity),
        }
    }

    /// Build the final command queue, consuming the builder.
    #[must_use]
    pub fn build(self) -> CommandQueue {
        self.queue
    }

    /// Add a raw gate command.
    #[must_use]
    pub fn gate(mut self, command: GateCommand) -> Self {
        self.queue.push(command);
        self
    }

    /// Add a signal at the current position in the command stream.
    ///
    /// The signal is placed after the most recently added gate command.
    ///
    /// ```
    /// use pecos_neo::command::CommandBuilder;
    /// use pecos_core::impl_signal;
    ///
    /// #[derive(Copy, Clone, Debug)]
    /// struct RoundBoundary(pub i64);
    /// impl_signal!(RoundBoundary);
    ///
    /// let queue = CommandBuilder::new()
    ///     .pz(0).pz(1)
    ///     .signal(RoundBoundary(1))
    ///     .h(0).h(1)
    ///     .mz(0).mz(1)
    ///     .build();
    ///
    /// assert!(queue.has_signals());
    /// ```
    #[must_use]
    pub fn signal<S: Signal>(mut self, signal: S) -> Self {
        self.queue.signal(signal);
        self
    }

    // Single-qubit gates

    /// Add an identity gate.
    #[must_use]
    pub fn identity(self, qubit: impl Into<QubitId>) -> Self {
        self.gate(GateCommand::identity(qubit.into()))
    }

    /// Add a Pauli-X gate.
    #[must_use]
    pub fn x(self, qubit: impl Into<QubitId>) -> Self {
        self.gate(GateCommand::x(qubit.into()))
    }

    /// Add a Pauli-Y gate.
    #[must_use]
    pub fn y(self, qubit: impl Into<QubitId>) -> Self {
        self.gate(GateCommand::y(qubit.into()))
    }

    /// Add a Pauli-Z gate.
    #[must_use]
    pub fn z(self, qubit: impl Into<QubitId>) -> Self {
        self.gate(GateCommand::z(qubit.into()))
    }

    /// Add a Hadamard gate.
    #[must_use]
    pub fn h(self, qubit: impl Into<QubitId>) -> Self {
        self.gate(GateCommand::h(qubit.into()))
    }

    /// Add an SX (sqrt-X) gate.
    #[must_use]
    pub fn sx(self, qubit: impl Into<QubitId>) -> Self {
        self.gate(GateCommand::new(
            GateType::SX,
            smallvec::smallvec![qubit.into()],
        ))
    }

    /// Add an `SXdg` (sqrt-X dagger) gate.
    #[must_use]
    pub fn sxdg(self, qubit: impl Into<QubitId>) -> Self {
        self.gate(GateCommand::new(
            GateType::SXdg,
            smallvec::smallvec![qubit.into()],
        ))
    }

    /// Add an SY (sqrt-Y) gate.
    #[must_use]
    pub fn sy(self, qubit: impl Into<QubitId>) -> Self {
        self.gate(GateCommand::new(
            GateType::SY,
            smallvec::smallvec![qubit.into()],
        ))
    }

    /// Add an `SYdg` (sqrt-Y dagger) gate.
    #[must_use]
    pub fn sydg(self, qubit: impl Into<QubitId>) -> Self {
        self.gate(GateCommand::new(
            GateType::SYdg,
            smallvec::smallvec![qubit.into()],
        ))
    }

    /// Add an SZ (sqrt-Z) gate.
    #[must_use]
    pub fn sz(self, qubit: impl Into<QubitId>) -> Self {
        self.gate(GateCommand::sz(qubit.into()))
    }

    /// Add an `SZdg` (sqrt-Z dagger) gate.
    #[must_use]
    pub fn szdg(self, qubit: impl Into<QubitId>) -> Self {
        self.gate(GateCommand::new(
            GateType::SZdg,
            smallvec::smallvec![qubit.into()],
        ))
    }

    /// Add a T gate.
    #[must_use]
    pub fn t(self, qubit: impl Into<QubitId>) -> Self {
        self.gate(GateCommand::new(
            GateType::T,
            smallvec::smallvec![qubit.into()],
        ))
    }

    /// Add a Tdg (T dagger) gate.
    #[must_use]
    pub fn tdg(self, qubit: impl Into<QubitId>) -> Self {
        self.gate(GateCommand::new(
            GateType::Tdg,
            smallvec::smallvec![qubit.into()],
        ))
    }

    // Parameterized single-qubit gates

    /// Add an RX rotation gate.
    #[must_use]
    pub fn rx(self, qubit: impl Into<QubitId>, angle: impl Into<Angle64>) -> Self {
        self.gate(GateCommand::with_angles(
            GateType::RX,
            smallvec::smallvec![qubit.into()],
            smallvec::smallvec![angle.into()],
        ))
    }

    /// Add an RY rotation gate.
    #[must_use]
    pub fn ry(self, qubit: impl Into<QubitId>, angle: impl Into<Angle64>) -> Self {
        self.gate(GateCommand::with_angles(
            GateType::RY,
            smallvec::smallvec![qubit.into()],
            smallvec::smallvec![angle.into()],
        ))
    }

    /// Add an RZ rotation gate.
    #[must_use]
    pub fn rz(self, qubit: impl Into<QubitId>, angle: impl Into<Angle64>) -> Self {
        self.gate(GateCommand::rz(qubit.into(), angle.into()))
    }

    // Two-qubit gates

    /// Add a CNOT (CX) gate.
    #[must_use]
    pub fn cx(self, control: impl Into<QubitId>, target: impl Into<QubitId>) -> Self {
        self.gate(GateCommand::cx(control.into(), target.into()))
    }

    /// Add a CY gate.
    #[must_use]
    pub fn cy(self, control: impl Into<QubitId>, target: impl Into<QubitId>) -> Self {
        self.gate(GateCommand::new(
            GateType::CY,
            smallvec::smallvec![control.into(), target.into()],
        ))
    }

    /// Add a CZ gate.
    #[must_use]
    pub fn cz(self, qubit0: impl Into<QubitId>, qubit1: impl Into<QubitId>) -> Self {
        self.gate(GateCommand::cz(qubit0.into(), qubit1.into()))
    }

    /// Add an SZZ gate.
    #[must_use]
    pub fn szz(self, qubit0: impl Into<QubitId>, qubit1: impl Into<QubitId>) -> Self {
        self.gate(GateCommand::new(
            GateType::SZZ,
            smallvec::smallvec![qubit0.into(), qubit1.into()],
        ))
    }

    /// Add a SWAP gate.
    #[must_use]
    pub fn swap(self, qubit0: impl Into<QubitId>, qubit1: impl Into<QubitId>) -> Self {
        self.gate(GateCommand::new(
            GateType::SWAP,
            smallvec::smallvec![qubit0.into(), qubit1.into()],
        ))
    }

    /// Add an RZZ rotation gate.
    #[must_use]
    pub fn rzz(
        self,
        qubit0: impl Into<QubitId>,
        qubit1: impl Into<QubitId>,
        angle: impl Into<Angle64>,
    ) -> Self {
        self.gate(GateCommand::rzz(qubit0.into(), qubit1.into(), angle.into()))
    }

    /// Add an RXX rotation gate.
    #[must_use]
    pub fn rxx(
        self,
        qubit0: impl Into<QubitId>,
        qubit1: impl Into<QubitId>,
        angle: impl Into<Angle64>,
    ) -> Self {
        self.gate(GateCommand::with_angles(
            GateType::RXX,
            smallvec::smallvec![qubit0.into(), qubit1.into()],
            smallvec::smallvec![angle.into()],
        ))
    }

    /// Add an RYY rotation gate.
    #[must_use]
    pub fn ryy(
        self,
        qubit0: impl Into<QubitId>,
        qubit1: impl Into<QubitId>,
        angle: impl Into<Angle64>,
    ) -> Self {
        self.gate(GateCommand::with_angles(
            GateType::RYY,
            smallvec::smallvec![qubit0.into(), qubit1.into()],
            smallvec::smallvec![angle.into()],
        ))
    }

    // Three-qubit gates

    /// Add a Toffoli (CCX) gate.
    #[must_use]
    pub fn ccx(
        self,
        control0: impl Into<QubitId>,
        control1: impl Into<QubitId>,
        target: impl Into<QubitId>,
    ) -> Self {
        self.gate(GateCommand::new(
            GateType::CCX,
            smallvec::smallvec![control0.into(), control1.into(), target.into()],
        ))
    }

    // Preparation and measurement

    /// Add a Z-basis state preparation gate.
    #[must_use]
    pub fn pz(self, qubit: impl Into<QubitId>) -> Self {
        self.gate(GateCommand::pz(qubit.into()))
    }

    /// Add a qubit allocation.
    #[must_use]
    pub fn qalloc(self, qubit: impl Into<QubitId>) -> Self {
        self.gate(GateCommand::new(
            GateType::QAlloc,
            smallvec::smallvec![qubit.into()],
        ))
    }

    /// Add a qubit deallocation.
    #[must_use]
    pub fn qfree(self, qubit: impl Into<QubitId>) -> Self {
        self.gate(GateCommand::new(
            GateType::QFree,
            smallvec::smallvec![qubit.into()],
        ))
    }

    /// Add a Z-basis measurement.
    #[must_use]
    pub fn mz(self, qubit: impl Into<QubitId>) -> Self {
        self.gate(GateCommand::mz(qubit.into()))
    }

    /// Add a measurement that also frees the qubit.
    #[must_use]
    pub fn measure_free(self, qubit: impl Into<QubitId>) -> Self {
        self.gate(GateCommand::new(
            GateType::MeasureFree,
            smallvec::smallvec![qubit.into()],
        ))
    }

    /// Add Z-basis preparation for multiple qubits.
    #[must_use]
    pub fn pz_all(mut self, qubits: impl IntoIterator<Item = impl Into<QubitId>>) -> Self {
        for q in qubits {
            self.queue.push(GateCommand::pz(q.into()));
        }
        self
    }

    /// Add Z-basis measurements for multiple qubits.
    #[must_use]
    pub fn mz_all(mut self, qubits: impl IntoIterator<Item = impl Into<QubitId>>) -> Self {
        for q in qubits {
            self.queue.push(GateCommand::mz(q.into()));
        }
        self
    }

    /// Add an idle period for a qubit.
    ///
    /// The duration is in abstract time units. The interpretation
    /// (nanoseconds, clock cycles, etc.) is defined by the noise model.
    #[must_use]
    pub fn idle(self, qubit: impl Into<QubitId>, duration: impl Into<TimeUnits>) -> Self {
        self.gate(GateCommand::idle(qubit.into(), duration.into()))
    }

    /// Add idle periods for multiple qubits with the same duration.
    #[must_use]
    pub fn idle_all(
        mut self,
        qubits: impl IntoIterator<Item = impl Into<QubitId>>,
        duration: impl Into<TimeUnits> + Copy,
    ) -> Self {
        for q in qubits {
            self.queue
                .push(GateCommand::idle(q.into(), duration.into()));
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_fluent_api() {
        let commands = CommandBuilder::new()
            .pz(0)
            .pz(1)
            .h(0)
            .cx(0, 1)
            .mz(0)
            .mz(1)
            .build();

        assert_eq!(commands.len(), 6);

        let cmds: Vec<_> = commands.iter().collect();
        assert_eq!(cmds[0].gate_type, GateType::PZ);
        assert_eq!(cmds[1].gate_type, GateType::PZ);
        assert_eq!(cmds[2].gate_type, GateType::H);
        assert_eq!(cmds[3].gate_type, GateType::CX);
        assert_eq!(cmds[4].gate_type, GateType::MZ);
        assert_eq!(cmds[5].gate_type, GateType::MZ);
    }

    #[test]
    fn test_builder_with_angles() {
        let commands = CommandBuilder::new()
            .rz(0, Angle64::QUARTER_TURN / 2u64) // pi/4
            .rzz(0, 1, Angle64::QUARTER_TURN) // pi/2
            .build();

        assert_eq!(commands.len(), 2);

        let cmds: Vec<_> = commands.iter().collect();
        assert_eq!(cmds[0].gate_type, GateType::RZ);
        assert_eq!(cmds[0].angles.len(), 1);
        assert_eq!(cmds[1].gate_type, GateType::RZZ);
        assert_eq!(cmds[1].angles.len(), 1);
    }

    #[test]
    fn test_prep_all_measure_all() {
        let commands = CommandBuilder::new().pz_all(0..4).h(0).mz_all(0..4).build();

        assert_eq!(commands.len(), 9); // 4 preps + 1 H + 4 measures
    }
}
