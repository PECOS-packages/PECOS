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
///     .pz(&[0, 1])
///     .h(&[0])
///     .cx(&[(0, 1)])
///     .mz(&[0, 1])
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
    ///     .pz(&[0, 1])
    ///     .signal(RoundBoundary(1))
    ///     .h(&[0, 1])
    ///     .mz(&[0, 1])
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

    /// Add identity gates.
    #[must_use]
    pub fn identity(mut self, qubits: &[impl Into<QubitId> + Copy]) -> Self {
        for &q in qubits {
            self.queue.push(GateCommand::identity(q.into()));
        }
        self
    }

    /// Add Pauli-X gates.
    #[must_use]
    pub fn x(mut self, qubits: &[impl Into<QubitId> + Copy]) -> Self {
        for &q in qubits {
            self.queue.push(GateCommand::x(q.into()));
        }
        self
    }

    /// Add Pauli-Y gates.
    #[must_use]
    pub fn y(mut self, qubits: &[impl Into<QubitId> + Copy]) -> Self {
        for &q in qubits {
            self.queue.push(GateCommand::y(q.into()));
        }
        self
    }

    /// Add Pauli-Z gates.
    #[must_use]
    pub fn z(mut self, qubits: &[impl Into<QubitId> + Copy]) -> Self {
        for &q in qubits {
            self.queue.push(GateCommand::z(q.into()));
        }
        self
    }

    /// Add Hadamard gates.
    #[must_use]
    pub fn h(mut self, qubits: &[impl Into<QubitId> + Copy]) -> Self {
        for &q in qubits {
            self.queue.push(GateCommand::h(q.into()));
        }
        self
    }

    /// Add face gates.
    #[must_use]
    pub fn f(mut self, qubits: &[impl Into<QubitId> + Copy]) -> Self {
        for &q in qubits {
            self.queue
                .push(GateCommand::new(GateType::F, smallvec::smallvec![q.into()]));
        }
        self
    }

    /// Add face-dagger gates.
    #[must_use]
    pub fn fdg(mut self, qubits: &[impl Into<QubitId> + Copy]) -> Self {
        for &q in qubits {
            self.queue.push(GateCommand::new(
                GateType::Fdg,
                smallvec::smallvec![q.into()],
            ));
        }
        self
    }

    /// Add SX (sqrt-X) gates.
    #[must_use]
    pub fn sx(mut self, qubits: &[impl Into<QubitId> + Copy]) -> Self {
        for &q in qubits {
            self.queue.push(GateCommand::new(
                GateType::SX,
                smallvec::smallvec![q.into()],
            ));
        }
        self
    }

    /// Add `SXdg` (sqrt-X dagger) gates.
    #[must_use]
    pub fn sxdg(mut self, qubits: &[impl Into<QubitId> + Copy]) -> Self {
        for &q in qubits {
            self.queue.push(GateCommand::new(
                GateType::SXdg,
                smallvec::smallvec![q.into()],
            ));
        }
        self
    }

    /// Add SY (sqrt-Y) gates.
    #[must_use]
    pub fn sy(mut self, qubits: &[impl Into<QubitId> + Copy]) -> Self {
        for &q in qubits {
            self.queue.push(GateCommand::new(
                GateType::SY,
                smallvec::smallvec![q.into()],
            ));
        }
        self
    }

    /// Add `SYdg` (sqrt-Y dagger) gates.
    #[must_use]
    pub fn sydg(mut self, qubits: &[impl Into<QubitId> + Copy]) -> Self {
        for &q in qubits {
            self.queue.push(GateCommand::new(
                GateType::SYdg,
                smallvec::smallvec![q.into()],
            ));
        }
        self
    }

    /// Add SZ (sqrt-Z) gates.
    #[must_use]
    pub fn sz(mut self, qubits: &[impl Into<QubitId> + Copy]) -> Self {
        for &q in qubits {
            self.queue.push(GateCommand::sz(q.into()));
        }
        self
    }

    /// Add `SZdg` (sqrt-Z dagger) gates.
    #[must_use]
    pub fn szdg(mut self, qubits: &[impl Into<QubitId> + Copy]) -> Self {
        for &q in qubits {
            self.queue.push(GateCommand::new(
                GateType::SZdg,
                smallvec::smallvec![q.into()],
            ));
        }
        self
    }

    /// Add T gates.
    #[must_use]
    pub fn t(mut self, qubits: &[impl Into<QubitId> + Copy]) -> Self {
        for &q in qubits {
            self.queue
                .push(GateCommand::new(GateType::T, smallvec::smallvec![q.into()]));
        }
        self
    }

    /// Add Tdg (T dagger) gates.
    #[must_use]
    pub fn tdg(mut self, qubits: &[impl Into<QubitId> + Copy]) -> Self {
        for &q in qubits {
            self.queue.push(GateCommand::new(
                GateType::Tdg,
                smallvec::smallvec![q.into()],
            ));
        }
        self
    }

    // Parameterized single-qubit gates

    /// Add RX rotation gates with the same angle.
    #[must_use]
    pub fn rx(
        mut self,
        qubits: &[impl Into<QubitId> + Copy],
        angle: impl Into<Angle64> + Copy,
    ) -> Self {
        for &q in qubits {
            self.queue.push(GateCommand::with_angles(
                GateType::RX,
                smallvec::smallvec![q.into()],
                smallvec::smallvec![angle.into()],
            ));
        }
        self
    }

    /// Add RY rotation gates with the same angle.
    #[must_use]
    pub fn ry(
        mut self,
        qubits: &[impl Into<QubitId> + Copy],
        angle: impl Into<Angle64> + Copy,
    ) -> Self {
        for &q in qubits {
            self.queue.push(GateCommand::with_angles(
                GateType::RY,
                smallvec::smallvec![q.into()],
                smallvec::smallvec![angle.into()],
            ));
        }
        self
    }

    /// Add RZ rotation gates with the same angle.
    #[must_use]
    pub fn rz(
        mut self,
        qubits: &[impl Into<QubitId> + Copy],
        angle: impl Into<Angle64> + Copy,
    ) -> Self {
        for &q in qubits {
            self.queue.push(GateCommand::rz(q.into(), angle.into()));
        }
        self
    }

    /// Add R1XY rotation gates with two angles (theta, phi).
    #[must_use]
    pub fn r1xy(
        mut self,
        qubits: &[impl Into<QubitId> + Copy],
        theta: impl Into<Angle64> + Copy,
        phi: impl Into<Angle64> + Copy,
    ) -> Self {
        for &q in qubits {
            self.queue.push(GateCommand::with_angles(
                GateType::R1XY,
                smallvec::smallvec![q.into()],
                smallvec::smallvec![theta.into(), phi.into()],
            ));
        }
        self
    }

    // Two-qubit gates

    /// Add CNOT (CX) gates.
    #[must_use]
    pub fn cx(mut self, pairs: &[(impl Into<QubitId> + Copy, impl Into<QubitId> + Copy)]) -> Self {
        for &(control, target) in pairs {
            self.queue
                .push(GateCommand::cx(control.into(), target.into()));
        }
        self
    }

    /// Add CY gates.
    #[must_use]
    pub fn cy(mut self, pairs: &[(impl Into<QubitId> + Copy, impl Into<QubitId> + Copy)]) -> Self {
        for &(control, target) in pairs {
            self.queue.push(GateCommand::new(
                GateType::CY,
                smallvec::smallvec![control.into(), target.into()],
            ));
        }
        self
    }

    /// Add CZ gates.
    #[must_use]
    pub fn cz(mut self, pairs: &[(impl Into<QubitId> + Copy, impl Into<QubitId> + Copy)]) -> Self {
        for &(q0, q1) in pairs {
            self.queue.push(GateCommand::cz(q0.into(), q1.into()));
        }
        self
    }

    /// Add SZZ gates.
    #[must_use]
    pub fn szz(mut self, pairs: &[(impl Into<QubitId> + Copy, impl Into<QubitId> + Copy)]) -> Self {
        for &(q0, q1) in pairs {
            self.queue.push(GateCommand::new(
                GateType::SZZ,
                smallvec::smallvec![q0.into(), q1.into()],
            ));
        }
        self
    }

    /// Add SXX gates.
    #[must_use]
    pub fn sxx(mut self, pairs: &[(impl Into<QubitId> + Copy, impl Into<QubitId> + Copy)]) -> Self {
        for &(q0, q1) in pairs {
            self.queue.push(GateCommand::new(
                GateType::SXX,
                smallvec::smallvec![q0.into(), q1.into()],
            ));
        }
        self
    }

    /// Add `SXXdg` gates.
    #[must_use]
    pub fn sxxdg(
        mut self,
        pairs: &[(impl Into<QubitId> + Copy, impl Into<QubitId> + Copy)],
    ) -> Self {
        for &(q0, q1) in pairs {
            self.queue.push(GateCommand::new(
                GateType::SXXdg,
                smallvec::smallvec![q0.into(), q1.into()],
            ));
        }
        self
    }

    /// Add SYY gates.
    #[must_use]
    pub fn syy(mut self, pairs: &[(impl Into<QubitId> + Copy, impl Into<QubitId> + Copy)]) -> Self {
        for &(q0, q1) in pairs {
            self.queue.push(GateCommand::new(
                GateType::SYY,
                smallvec::smallvec![q0.into(), q1.into()],
            ));
        }
        self
    }

    /// Add `SYYdg` gates.
    #[must_use]
    pub fn syydg(
        mut self,
        pairs: &[(impl Into<QubitId> + Copy, impl Into<QubitId> + Copy)],
    ) -> Self {
        for &(q0, q1) in pairs {
            self.queue.push(GateCommand::new(
                GateType::SYYdg,
                smallvec::smallvec![q0.into(), q1.into()],
            ));
        }
        self
    }

    /// Add `SZZdg` gates (inverse of `SZZ`).
    #[must_use]
    pub fn szzdg(
        mut self,
        pairs: &[(impl Into<QubitId> + Copy, impl Into<QubitId> + Copy)],
    ) -> Self {
        for &(q0, q1) in pairs {
            self.queue.push(GateCommand::new(
                GateType::SZZdg,
                smallvec::smallvec![q0.into(), q1.into()],
            ));
        }
        self
    }

    /// Add SWAP gates.
    #[must_use]
    pub fn swap(
        mut self,
        pairs: &[(impl Into<QubitId> + Copy, impl Into<QubitId> + Copy)],
    ) -> Self {
        for &(q0, q1) in pairs {
            self.queue.push(GateCommand::new(
                GateType::SWAP,
                smallvec::smallvec![q0.into(), q1.into()],
            ));
        }
        self
    }

    /// Add RZZ rotation gates with the same angle.
    #[must_use]
    pub fn rzz(
        mut self,
        pairs: &[(impl Into<QubitId> + Copy, impl Into<QubitId> + Copy)],
        angle: impl Into<Angle64> + Copy,
    ) -> Self {
        for &(q0, q1) in pairs {
            self.queue
                .push(GateCommand::rzz(q0.into(), q1.into(), angle.into()));
        }
        self
    }

    /// Add RXX rotation gates with the same angle.
    #[must_use]
    pub fn rxx(
        mut self,
        pairs: &[(impl Into<QubitId> + Copy, impl Into<QubitId> + Copy)],
        angle: impl Into<Angle64> + Copy,
    ) -> Self {
        for &(q0, q1) in pairs {
            self.queue.push(GateCommand::with_angles(
                GateType::RXX,
                smallvec::smallvec![q0.into(), q1.into()],
                smallvec::smallvec![angle.into()],
            ));
        }
        self
    }

    /// Add RYY rotation gates with the same angle.
    #[must_use]
    pub fn ryy(
        mut self,
        pairs: &[(impl Into<QubitId> + Copy, impl Into<QubitId> + Copy)],
        angle: impl Into<Angle64> + Copy,
    ) -> Self {
        for &(q0, q1) in pairs {
            self.queue.push(GateCommand::with_angles(
                GateType::RYY,
                smallvec::smallvec![q0.into(), q1.into()],
                smallvec::smallvec![angle.into()],
            ));
        }
        self
    }

    // Three-qubit gates

    /// Add Toffoli (CCX) gates.
    #[must_use]
    pub fn ccx(
        mut self,
        triples: &[(
            impl Into<QubitId> + Copy,
            impl Into<QubitId> + Copy,
            impl Into<QubitId> + Copy,
        )],
    ) -> Self {
        for &(c0, c1, t) in triples {
            self.queue.push(GateCommand::new(
                GateType::CCX,
                smallvec::smallvec![c0.into(), c1.into(), t.into()],
            ));
        }
        self
    }

    // Preparation and measurement

    /// Add Z-basis state preparation gates.
    #[must_use]
    pub fn pz(mut self, qubits: &[impl Into<QubitId> + Copy]) -> Self {
        for &q in qubits {
            self.queue.push(GateCommand::pz(q.into()));
        }
        self
    }

    /// Add qubit allocations.
    #[must_use]
    pub fn qalloc(mut self, qubits: &[impl Into<QubitId> + Copy]) -> Self {
        for &q in qubits {
            self.queue.push(GateCommand::new(
                GateType::QAlloc,
                smallvec::smallvec![q.into()],
            ));
        }
        self
    }

    /// Add qubit deallocations.
    #[must_use]
    pub fn qfree(mut self, qubits: &[impl Into<QubitId> + Copy]) -> Self {
        for &q in qubits {
            self.queue.push(GateCommand::new(
                GateType::QFree,
                smallvec::smallvec![q.into()],
            ));
        }
        self
    }

    /// Add Z-basis measurements.
    #[must_use]
    pub fn mz(mut self, qubits: &[impl Into<QubitId> + Copy]) -> Self {
        for &q in qubits {
            self.queue.push(GateCommand::mz(q.into()));
        }
        self
    }

    /// Add measurements that also free the qubits.
    #[must_use]
    pub fn mz_free(mut self, qubits: &[impl Into<QubitId> + Copy]) -> Self {
        for &q in qubits {
            self.queue.push(GateCommand::new(
                GateType::MeasureFree,
                smallvec::smallvec![q.into()],
            ));
        }
        self
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

    /// Add idle periods for qubits with the same duration.
    ///
    /// The duration is in abstract time units. The interpretation
    /// (nanoseconds, clock cycles, etc.) is defined by the noise model.
    #[must_use]
    pub fn idle(
        mut self,
        qubits: &[impl Into<QubitId> + Copy],
        duration: impl Into<TimeUnits> + Copy,
    ) -> Self {
        for &q in qubits {
            self.queue
                .push(GateCommand::idle(q.into(), duration.into()));
        }
        self
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
            .pz(&[0, 1])
            .h(&[0])
            .cx(&[(0, 1)])
            .mz(&[0, 1])
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
            .rz(&[0], Angle64::QUARTER_TURN / 2u64) // pi/4
            .rzz(&[(0, 1)], Angle64::QUARTER_TURN) // pi/2
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
        let commands = CommandBuilder::new()
            .pz_all(0..4)
            .h(&[0])
            .mz_all(0..4)
            .build();

        assert_eq!(commands.len(), 9); // 4 preps + 1 H + 4 measures
    }
}
