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

//! Pauli propagation through quantum circuits.
//!
//! This module provides functions for propagating Pauli operators forward and backward
//! through quantum circuits. This is the foundation of fault tolerance analysis.

use super::{PauliFault, is_supported_noop_or_metadata_gate, is_supported_prep_gate};
use pecos_core::gate_type::GateType;
use pecos_core::{half_turn_decomposition, try_simplify_rotation, try_simplify_rxy1q};
use pecos_quantum::TickCircuit;
use pecos_simulators::{CliffordGateable, PauliProp};
use smallvec::SmallVec;

/// Whether a gate was faithfully handled by Pauli propagation.
///
/// `Propagated` includes gates that are intentionally transparent to Pauli
/// propagation. `Unsupported` means the gate changes quantum state in a way
/// this Pauli-only representation cannot express.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub enum PauliPropagationOutcome {
    Propagated,
    Unsupported,
}

// ============================================================================
// Direction and Unified Propagation
// ============================================================================

/// Direction of Pauli propagation through a circuit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// Forward propagation: P → G P G†
    /// Propagate from earlier ticks to later ticks.
    Forward,
    /// Backward propagation: P → G† P G
    /// Propagate from later ticks to earlier ticks.
    Backward,
}

/// Split a gate node's qubit list into its consecutive two-qubit pairs.
///
/// A `DagCircuit` node may carry several gate instances -- `DagCircuit::gate_count`
/// counts them individually -- so a two-qubit gate must act on every pair, not
/// just the first. `SXX`/`SYY`/`SZZ` and their adjoints previously took only
/// `(qubits[0], qubits[1])` while `CX`/`CY`/`CZ`/`SWAP` took all pairs, which
/// made backward propagation disagree with forward symbolic simulation on
/// batched nodes.
///
/// Returns a `SmallVec` sized for the overwhelmingly common single-pair node,
/// so routing the six non-self-adjoint arms through this helper does not put a
/// heap allocation on the analyzer's traversal path.
fn consecutive_pairs(
    qubits: &[pecos_core::QubitId],
) -> SmallVec<[(pecos_core::QubitId, pecos_core::QubitId); 1]> {
    qubits
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| (pair[0], pair[1]))
        .collect()
}

/// Applies a gate to a `PauliProp` in the specified direction.
///
/// For forward propagation (P → G P G†), we apply the gate's transformation.
/// For backward propagation (P → G† P G), we apply the adjoint transformation.
///
/// Most Clifford gates are self-adjoint (H, CX, CZ, X, Y, Z), so the transformation
/// is the same in both directions. For non-self-adjoint gates (SZ, SX, SY and their
/// daggers), we swap the gate with its adjoint for backward propagation.
///
/// Special handling:
/// - **Prep gates**: No transformation in either direction (transparent to propagation;
///   walkers that model resets clear at their own call sites)
/// - **Measure gates**: collapse via [`cross_measurement`] -- the component that
///   commutes with the measurement in the direction of travel is absorbed
///
/// Rotation eligibility follows the core lowering policy. Axis rotations such
/// as `RZ` require exact Clifford-angle equality, while `RXY1Q` snaps angles
/// within `1e-9` turns of its Clifford grid. [`InfluenceBuilder`](crate::fault_tolerance::InfluenceBuilder)
/// is intentionally more conservative: its symbolic replay accepts `RX`,
/// `RY`, `RZ`, `RXX`, `RYY`, `RZZ`, and `CRZ` only with one exactly-zero
/// angle, and rejects every `RXY1Q`, including `RXY1Q(0, phi)`.
///
/// Returns [`PauliPropagationOutcome::Unsupported`] when the gate changes the
/// state in a way this Pauli-only representation cannot faithfully express, or
/// when its payload fails [`pecos_core::Gate::validate`].
#[inline]
pub fn apply_gate(
    prop: &mut PauliProp,
    gate: &pecos_core::Gate,
    direction: Direction,
) -> PauliPropagationOutcome {
    if gate.validate().is_err() {
        return PauliPropagationOutcome::Unsupported;
    }

    apply_gate_unchecked(prop, gate, direction)
}

/// Apply a gate after a circuit-level preflight has validated its payload.
///
/// Unlike [`apply_gate`], this dispatcher deliberately avoids
/// [`pecos_core::Gate::validate`] so repeated propagation walks do not allocate
/// and sort the gate's qubits again. Its angle access remains bounds-safe so
/// permissive non-DEM callers cannot panic if they bypass a preflight.
#[inline]
pub(crate) fn apply_gate_unchecked(
    prop: &mut PauliProp,
    gate: &pecos_core::Gate,
    direction: Direction,
) -> PauliPropagationOutcome {
    if is_supported_prep_gate(gate.gate_type) || is_supported_noop_or_metadata_gate(gate.gate_type)
    {
        return PauliPropagationOutcome::Propagated;
    }

    if apply_named_gate(prop, gate.gate_type, &gate.qubits, direction) {
        return PauliPropagationOutcome::Propagated;
    }

    match gate.gate_type {
        GateType::RZ
        | GateType::RX
        | GateType::RY
        | GateType::RZZ
        | GateType::RXX
        | GateType::RYY => {
            let Some(&angle) = gate.angles.first() else {
                return PauliPropagationOutcome::Unsupported;
            };
            if let Some(clifford) = try_simplify_rotation(gate.gate_type, angle) {
                return if apply_named_gate(prop, clifford, &gate.qubits, direction) {
                    PauliPropagationOutcome::Propagated
                } else {
                    PauliPropagationOutcome::Unsupported
                };
            }

            if let Some(pauli) = half_turn_decomposition(gate.gate_type, angle) {
                for &qubit in &gate.qubits {
                    if !apply_named_gate(prop, pauli, &[qubit], direction) {
                        return PauliPropagationOutcome::Unsupported;
                    }
                }
                return PauliPropagationOutcome::Propagated;
            }
            PauliPropagationOutcome::Unsupported
        }
        GateType::RXY1Q if gate.angles.len() >= 2 => {
            let theta = gate.angles[0];
            let phi = gate.angles[1];
            if let Some(clifford) = try_simplify_rxy1q(theta, phi) {
                if apply_named_gate(prop, clifford, &gate.qubits, direction) {
                    PauliPropagationOutcome::Propagated
                } else {
                    PauliPropagationOutcome::Unsupported
                }
            } else {
                PauliPropagationOutcome::Unsupported
            }
        }
        _ => PauliPropagationOutcome::Unsupported,
    }
}

/// The component operations `cross_measurement` needs, implemented by both
/// Pauli-propagation representations so the collapse rule has exactly one
/// implementation.
pub trait PauliComponents {
    fn contains_x(&self, qubit: usize) -> bool;
    fn contains_z(&self, qubit: usize) -> bool;
    fn toggle_x(&mut self, qubit: usize);
    fn toggle_z(&mut self, qubit: usize);
    fn clear_qubit(&mut self, qubit: usize);
}

impl PauliComponents for PauliProp {
    fn contains_x(&self, qubit: usize) -> bool {
        Self::contains_x(self, qubit)
    }
    fn contains_z(&self, qubit: usize) -> bool {
        Self::contains_z(self, qubit)
    }
    fn toggle_x(&mut self, qubit: usize) {
        self.track_x(&[qubit]);
    }
    fn toggle_z(&mut self, qubit: usize) {
        self.track_z(&[qubit]);
    }
    fn clear_qubit(&mut self, qubit: usize) {
        Self::clear_qubit(self, qubit);
    }
}

impl PauliComponents for pecos_simulators::BitmaskPauliProp {
    fn contains_x(&self, qubit: usize) -> bool {
        Self::contains_x(self, qubit)
    }
    fn contains_z(&self, qubit: usize) -> bool {
        Self::contains_z(self, qubit)
    }
    fn toggle_x(&mut self, qubit: usize) {
        self.track_x(&[qubit]);
    }
    fn toggle_z(&mut self, qubit: usize) {
        self.track_z(&[qubit]);
    }
    fn clear_qubit(&mut self, qubit: usize) {
        Self::clear_qubit(self, qubit);
    }
}

/// Cross a measurement site with a propagating Pauli.
///
/// Clearing follows whether the gate *discards* the qubit, not whether it
/// measures. A non-destructive Z-collapse (`MZ`, `MeasureLeaked` -- executed
/// as a plain `MZ` with no reset):
///
/// - Forward: `(x, z) -> (x, 0)`. The X component survives -- the faulty and
///   ideal runs still differ by X after collapse, so later measurements keep
///   flipping (Stim's `M`-versus-`MR` distinction) -- while the Z component
///   is absorbed. This is Stim's frame algebra with the measurement gauge
///   fixed to identity instead of randomized; the two choices differ only on
///   individually non-deterministic measurements, where any detector is
///   invalid input, and the difference cancels in every deterministic
///   detector XOR.
/// - Backward: the symplectic adjoint, `(x, z) -> (0, z)`, so the two
///   directions agree by construction. A Z-type observable passes through; an
///   X-type observable is dropped -- relative to the gauge, nothing before
///   the collapse deterministically flips it.
///
/// `MeasureFree` discards the qubit: nothing crosses in either direction.
pub fn cross_measurement<P: PauliComponents>(
    prop: &mut P,
    qubit: usize,
    gate_type: GateType,
    direction: Direction,
) {
    match gate_type {
        GateType::MZ | GateType::MeasureLeaked => match direction {
            Direction::Forward => {
                if prop.contains_z(qubit) {
                    prop.toggle_z(qubit);
                }
            }
            Direction::Backward => {
                if prop.contains_x(qubit) {
                    prop.toggle_x(qubit);
                }
            }
        },
        GateType::MX => match direction {
            Direction::Forward => {
                if prop.contains_x(qubit) {
                    prop.toggle_x(qubit);
                }
            }
            Direction::Backward => {
                if prop.contains_z(qubit) {
                    prop.toggle_z(qubit);
                }
            }
        },
        // `MeasureFree` discards the qubit; `MPZ` resets it. Either way the
        // record flip is taken by the walker before the crossing and nothing
        // propagates across.
        GateType::MeasureFree | GateType::MPZ => prop.clear_qubit(qubit),
        _ => debug_assert!(false, "cross_measurement called on {gate_type:?}"),
    }
}

#[inline]
fn apply_named_gate(
    prop: &mut PauliProp,
    gate_type: GateType,
    qubits: &[pecos_core::QubitId],
    direction: Direction,
) -> bool {
    match gate_type {
        GateType::MX
        | GateType::MZ
        | GateType::MeasureFree
        | GateType::MeasureLeaked
        | GateType::MPZ => {
            for qid in qubits {
                cross_measurement(prop, qid.index(), gate_type, direction);
            }
        }
        // Self-adjoint single-qubit gates - same in both directions
        GateType::I => {
            prop.identity(qubits);
        }
        GateType::X => {
            prop.x(qubits);
        }
        GateType::Y => {
            prop.y(qubits);
        }
        GateType::Z => {
            prop.z(qubits);
        }
        GateType::H => {
            prop.h(qubits);
        }
        GateType::F => {
            match direction {
                Direction::Forward => prop.f(qubits),
                Direction::Backward => prop.fdg(qubits),
            };
        }
        GateType::Fdg => {
            match direction {
                Direction::Forward => prop.fdg(qubits),
                Direction::Backward => prop.f(qubits),
            };
        }

        // Non-self-adjoint single-qubit gates - swap with adjoint for backward
        GateType::SX => {
            match direction {
                Direction::Forward => prop.sx(qubits),
                Direction::Backward => prop.sxdg(qubits),
            };
        }
        GateType::SXdg => {
            match direction {
                Direction::Forward => prop.sxdg(qubits),
                Direction::Backward => prop.sx(qubits),
            };
        }
        GateType::SY => {
            match direction {
                Direction::Forward => prop.sy(qubits),
                Direction::Backward => prop.sydg(qubits),
            };
        }
        GateType::SYdg => {
            match direction {
                Direction::Forward => prop.sydg(qubits),
                Direction::Backward => prop.sy(qubits),
            };
        }
        GateType::SZ => {
            match direction {
                Direction::Forward => prop.sz(qubits),
                Direction::Backward => prop.szdg(qubits),
            };
        }
        GateType::SZdg => {
            match direction {
                Direction::Forward => prop.szdg(qubits),
                Direction::Backward => prop.sz(qubits),
            };
        }

        // Self-adjoint two-qubit gates - same in both directions
        GateType::CX => {
            prop.cx(&consecutive_pairs(qubits));
        }
        GateType::CY => {
            prop.cy(&consecutive_pairs(qubits));
        }
        GateType::CZ => {
            prop.cz(&consecutive_pairs(qubits));
        }
        GateType::SWAP => {
            prop.swap(&consecutive_pairs(qubits));
        }

        // Non-self-adjoint two-qubit Clifford gates - swap with adjoint for backward
        GateType::SXX => {
            let pairs = consecutive_pairs(qubits);
            match direction {
                Direction::Forward => prop.sxx(&pairs),
                Direction::Backward => prop.sxxdg(&pairs),
            };
        }
        GateType::SXXdg => {
            let pairs = consecutive_pairs(qubits);
            match direction {
                Direction::Forward => prop.sxxdg(&pairs),
                Direction::Backward => prop.sxx(&pairs),
            };
        }
        GateType::SYY => {
            let pairs = consecutive_pairs(qubits);
            match direction {
                Direction::Forward => prop.syy(&pairs),
                Direction::Backward => prop.syydg(&pairs),
            };
        }
        GateType::SYYdg => {
            let pairs = consecutive_pairs(qubits);
            match direction {
                Direction::Forward => prop.syydg(&pairs),
                Direction::Backward => prop.syy(&pairs),
            };
        }
        GateType::SZZ => {
            let pairs = consecutive_pairs(qubits);
            match direction {
                Direction::Forward => prop.szz(&pairs),
                Direction::Backward => prop.szzdg(&pairs),
            };
        }
        GateType::SZZdg => {
            let pairs = consecutive_pairs(qubits);
            match direction {
                Direction::Forward => prop.szzdg(&pairs),
                Direction::Backward => prop.szz(&pairs),
            };
        }

        _ => return false,
    }

    true
}

/// Propagates a `PauliProp` through a circuit in the specified direction.
///
/// # Arguments
/// * `circuit` - The circuit to propagate through
/// * `prop` - The `PauliProp` to propagate (modified in place)
/// * `direction` - Forward or Backward propagation
pub fn propagate_through_circuit(
    circuit: &TickCircuit,
    prop: &mut PauliProp,
    direction: Direction,
) {
    match direction {
        Direction::Forward => {
            for tick in circuit.ticks() {
                for gate in tick.iter_gate_batches() {
                    let _outcome = apply_gate(prop, gate.as_gate(), direction);
                }
            }
        }
        Direction::Backward => {
            for tick in circuit.ticks().iter().rev() {
                for gate in tick.iter_gate_batches() {
                    let _outcome = apply_gate(prop, gate.as_gate(), direction);
                }
            }
        }
    }
}

/// Propagates a `PauliProp` through a range of ticks in the specified direction.
///
/// # Arguments
/// * `circuit` - The circuit to propagate through
/// * `prop` - The `PauliProp` to propagate (modified in place)
/// * `start_tick` - The tick to start from (inclusive)
/// * `end_tick` - The tick to end at (inclusive)
/// * `direction` - Forward or Backward propagation
///
/// For Forward: propagates from `start_tick` to `end_tick`
/// For Backward: propagates from `end_tick` to `start_tick`
pub fn propagate_tick_range(
    circuit: &TickCircuit,
    prop: &mut PauliProp,
    start_tick: usize,
    end_tick: usize,
    direction: Direction,
) {
    let num_ticks = circuit.ticks().len();
    let start = start_tick.min(num_ticks.saturating_sub(1));
    let end = end_tick.min(num_ticks.saturating_sub(1));

    match direction {
        Direction::Forward => {
            for tick_idx in start..=end {
                let tick = &circuit.ticks()[tick_idx];
                for gate in tick.iter_gate_batches() {
                    let _outcome = apply_gate(prop, gate.as_gate(), direction);
                }
            }
        }
        Direction::Backward => {
            for tick_idx in (start..=end).rev() {
                let tick = &circuit.ticks()[tick_idx];
                for gate in tick.iter_gate_batches() {
                    let _outcome = apply_gate(prop, gate.as_gate(), direction);
                }
            }
        }
    }
}

// ============================================================================
// Standalone Backward Propagation Functions
// ============================================================================

/// Propagates a Pauli backward through a circuit from a given starting tick.
///
/// This is the backward analog of forward Pauli propagation. Starting with a Pauli
/// at `start_tick`, it propagates backward through all preceding ticks.
///
/// # Arguments
/// * `circuit` - The circuit to propagate through
/// * `prop` - The Pauli to propagate (will be modified in place)
/// * `start_tick` - The tick to start from (propagates backward from here to tick 0)
///
/// # Example
/// ```
/// use pecos_qec::fault_tolerance::propagator::propagate_backward_from_tick;
/// use pecos_quantum::TickCircuit;
/// use pecos_simulators::PauliProp;
///
/// let mut circuit = TickCircuit::new();
/// circuit.tick().pz(&[0]);
/// circuit.tick().h(&[0]);
/// circuit.tick().mz(&[0]);
///
/// // Start with Z at the measurement (tick 2) and propagate backward
/// let mut prop = PauliProp::new();
/// prop.track_z(&[0]);
/// propagate_backward_from_tick(&circuit, &mut prop, 2);
///
/// // After H gate backward propagation, Z becomes X
/// assert!(prop.contains_x(0));
/// assert!(!prop.contains_z(0));
/// ```
pub fn propagate_backward_from_tick(
    circuit: &TickCircuit,
    prop: &mut PauliProp,
    start_tick: usize,
) {
    propagate_tick_range(circuit, prop, 0, start_tick, Direction::Backward);
}

/// Propagates a fault backward through a circuit.
///
/// This is the backward analog of `propagate_fault`. Given a fault at a specific
/// location, it initializes a Pauli with that fault and propagates backward
/// through all preceding gates.
///
/// # Arguments
/// * `circuit` - The circuit to propagate through
/// * `fault` - The fault to propagate backward
///
/// # Returns
/// A `PauliProp` representing the backward-propagated fault (what it would have
/// looked like at the beginning of the circuit).
///
/// # Example
/// ```
/// use pecos_qec::fault_tolerance::{PauliFault, SpacetimeLocation};
/// use pecos_qec::fault_tolerance::propagator::propagate_fault_backward;
/// use pecos_quantum::TickCircuit;
/// use pecos_core::gate_type::GateType;
/// use pecos_core::QubitId;
///
/// let mut circuit = TickCircuit::new();
/// circuit.tick().pz(&[0]);
/// circuit.tick().h(&[0]);
/// circuit.tick().mz(&[0]);
///
/// // Create a Z fault at the measurement location
/// let loc = SpacetimeLocation {
///     tick: 2,
///     qubits: vec![QubitId(0)],
///     before: true,
///     gate_type: GateType::MZ,
///     gate_index: 0,
/// };
/// let fault = PauliFault::new(loc, vec![3]); // Z fault
///
/// let prop = propagate_fault_backward(&circuit, &fault);
/// // Z propagated backward through H becomes X
/// assert!(prop.contains_x(0));
/// ```
#[must_use]
pub fn propagate_fault_backward(circuit: &TickCircuit, fault: &PauliFault) -> PauliProp {
    let mut prop = init_pauli_prop_with_fault(fault);
    let fault_tick = fault.location.tick;

    // Determine which tick to start propagating from
    let end_tick = if fault.location.before {
        // Fault is before gates at fault_tick, so the fault exists at the START of fault_tick
        // Backward propagation goes through ticks [0, fault_tick-1]
        fault_tick.saturating_sub(1)
    } else {
        // Fault is after gates at fault_tick, so the fault exists at the END of fault_tick
        // Backward propagation goes through ticks [0, fault_tick]
        fault_tick
    };

    propagate_tick_range(circuit, &mut prop, 0, end_tick, Direction::Backward);
    prop
}

/// Propagates an observable backward through the circuit.
///
/// This is useful for understanding what an observable (like a Z-measurement or
/// a logical operator) looks like at earlier points in the circuit.
///
/// # Arguments
/// * `circuit` - The circuit to propagate through
/// * `x_positions` - Qubits with X in the observable
/// * `z_positions` - Qubits with Z in the observable
/// * `start_tick` - The tick where the observable is defined (e.g., measurement tick)
///
/// # Returns
/// A `PauliProp` representing the backward-propagated observable.
#[must_use]
pub fn propagate_observable_backward(
    circuit: &TickCircuit,
    x_positions: &[usize],
    z_positions: &[usize],
    start_tick: usize,
) -> PauliProp {
    let mut prop = PauliProp::new();

    for &q in x_positions {
        prop.track_x(&[q]);
    }
    for &q in z_positions {
        prop.track_z(&[q]);
    }

    propagate_backward_from_tick(circuit, &mut prop, start_tick);
    prop
}

/// Initialize a `PauliProp` with the given fault.
#[must_use]
pub fn init_pauli_prop_with_fault(fault: &PauliFault) -> PauliProp {
    let mut prop = PauliProp::new();
    for (qubit, &pauli) in fault.location.qubits.iter().zip(fault.paulis.iter()) {
        let q = qubit.index();
        match pauli {
            1 => prop.track_x(&[q]),
            2 => {
                prop.track_x(&[q]);
                prop.track_z(&[q]);
            }
            3 => prop.track_z(&[q]),
            _ => {}
        }
    }
    prop
}

#[cfg(test)]
mod batched_pair_tests {
    use super::{Direction, apply_gate};
    use pecos_core::gates::Gate;
    use pecos_quantum::GateType;
    use pecos_simulators::PauliProp;

    /// Seed the tracked Pauli on qubits 0 and 2, one per pair.
    ///
    /// The seed must ANTICOMMUTE with the gate on at least one pair, or the
    /// comparison is vacuous: `Z` alone is fixed by `CX`/`CY`/`CZ`/`SZZ`/
    /// `SZZdg`, and `X` alone is fixed by `SXX`. Sweeping X, Z and Y gives
    /// every gate below at least one seed it actually moves.
    fn seeded(seed: (bool, bool)) -> PauliProp {
        let mut prop = PauliProp::new();
        let (track_x, track_z) = seed;
        if track_x {
            prop.track_x(&[0, 2]);
        }
        if track_z {
            prop.track_z(&[0, 2]);
        }
        prop
    }

    fn signature(prop: &PauliProp) -> Vec<(bool, bool)> {
        (0..4)
            .map(|q| (prop.contains_x(q), prop.contains_z(q)))
            .collect()
    }

    /// A batched two-qubit node must propagate through every pair. `SXX`/`SYY`/
    /// `SZZ` and their adjoints once took only `(qubits[0], qubits[1])` while
    /// `CX`/`CY`/`CZ`/`SWAP` took all pairs, so backward propagation silently
    /// disagreed with forward symbolic simulation on batched nodes.
    #[test]
    fn batched_two_qubit_gates_propagate_through_every_pair() {
        const SEEDS: [(bool, bool); 3] = [(true, false), (false, true), (true, true)];
        for gate_type in [
            GateType::SXX,
            GateType::SXXdg,
            GateType::SYY,
            GateType::SYYdg,
            GateType::SZZ,
            GateType::SZZdg,
            GateType::CX,
            GateType::CY,
            GateType::CZ,
            GateType::SWAP,
        ] {
            for direction in [Direction::Forward, Direction::Backward] {
                for seed in SEEDS {
                    let batched = {
                        let mut prop = seeded(seed);
                        let gate =
                            Gate::simple(gate_type, vec![0.into(), 1.into(), 2.into(), 3.into()]);
                        let _outcome = apply_gate(&mut prop, &gate, direction);
                        signature(&prop)
                    };
                    let split = {
                        let mut prop = seeded(seed);
                        for pair in [[0usize, 1], [2, 3]] {
                            let gate =
                                Gate::simple(gate_type, vec![pair[0].into(), pair[1].into()]);
                            let _outcome = apply_gate(&mut prop, &gate, direction);
                        }
                        signature(&prop)
                    };
                    assert_eq!(
                        batched, split,
                        "{gate_type:?} ({direction:?}, seed {seed:?}): a batched node must \
                         propagate like the same pairs split across nodes"
                    );
                }
            }
        }
    }

    /// Guards the guard: every gate above must be moved by at least one seed,
    /// otherwise the comparison could pass while an arm drops the second pair.
    #[test]
    fn every_seed_sweep_actually_moves_each_gate() {
        const SEEDS: [(bool, bool); 3] = [(true, false), (false, true), (true, true)];
        for gate_type in [
            GateType::SXX,
            GateType::SXXdg,
            GateType::SYY,
            GateType::SYYdg,
            GateType::SZZ,
            GateType::SZZdg,
            GateType::CX,
            GateType::CY,
            GateType::CZ,
            GateType::SWAP,
        ] {
            let moves_second_pair = SEEDS.iter().any(|&seed| {
                let mut prop = seeded(seed);
                let before = signature(&prop);
                let gate = Gate::simple(gate_type, vec![2.into(), 3.into()]);
                let _outcome = apply_gate(&mut prop, &gate, Direction::Forward);
                signature(&prop) != before
            });
            assert!(
                moves_second_pair,
                "{gate_type:?}: no seed changes the second pair, so the batched \
                 comparison would be vacuous for this gate"
            );
        }
    }
}

#[cfg(test)]
mod collapse_tests {
    use super::{Direction, cross_measurement};
    use pecos_quantum::GateType;
    use pecos_simulators::PauliProp;

    /// Forward across a non-destructive measurement: X survives, Z is
    /// absorbed. This is the StateVec-verified behavior (X; MZ -> 1; MZ -> 1)
    /// and Stim's `M`.
    #[test]
    fn forward_nondestructive_keeps_x_drops_z() {
        for gate_type in [GateType::MZ, GateType::MeasureLeaked] {
            let mut prop = PauliProp::new();
            prop.track_y(&[0]);
            cross_measurement(&mut prop, 0, gate_type, Direction::Forward);
            assert!(prop.contains_x(0), "{gate_type:?}: X must survive");
            assert!(!prop.contains_z(0), "{gate_type:?}: Z must be absorbed");
        }
    }

    /// Backward is the mirror: Z passes (errors before the measurement flip
    /// both it and later measurements), X is dropped (an X-type observable
    /// gains no deterministic dependence on anything before the collapse).
    #[test]
    fn backward_nondestructive_keeps_z_drops_x() {
        for gate_type in [GateType::MZ, GateType::MeasureLeaked] {
            let mut prop = PauliProp::new();
            prop.track_y(&[0]);
            cross_measurement(&mut prop, 0, gate_type, Direction::Backward);
            assert!(!prop.contains_x(0), "{gate_type:?}: X must be dropped");
            assert!(prop.contains_z(0), "{gate_type:?}: Z must pass");
        }
    }

    /// A discarded (`MeasureFree`) or reset (`MPZ`) qubit carries nothing
    /// across, in either direction.
    #[test]
    fn measure_free_and_mpz_clear_both_directions() {
        for gate_type in [GateType::MeasureFree, GateType::MPZ] {
            for direction in [Direction::Forward, Direction::Backward] {
                let mut prop = PauliProp::new();
                prop.track_y(&[0]);
                cross_measurement(&mut prop, 0, gate_type, direction);
                assert!(
                    !prop.contains_x(0) && !prop.contains_z(0),
                    "{gate_type:?} {direction:?}"
                );
            }
        }
    }

    /// Untouched qubits are untouched.
    #[test]
    fn collapse_is_local_to_the_measured_qubit() {
        let mut prop = PauliProp::new();
        prop.track_y(&[0]);
        prop.track_y(&[1]);
        cross_measurement(&mut prop, 0, GateType::MZ, Direction::Forward);
        assert!(prop.contains_x(1) && prop.contains_z(1));
    }
}
