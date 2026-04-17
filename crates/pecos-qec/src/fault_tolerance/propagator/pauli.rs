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

use super::PauliFault;
use pecos_core::gate_type::GateType;
use pecos_core::{half_turn_decomposition, try_simplify_r1xy, try_simplify_rotation};
use pecos_quantum::TickCircuit;
use pecos_simulators::{CliffordGateable, PauliProp};

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
/// - **Prep gates**: No transformation in either direction (transparent to propagation)
/// - **Measure gates**: No transformation in either direction (for propagation purposes)
#[inline]
pub fn apply_gate(prop: &mut PauliProp, gate: &pecos_core::Gate, direction: Direction) {
    if apply_named_gate(prop, gate.gate_type, &gate.qubits, direction) {
        return;
    }

    match gate.gate_type {
        GateType::RZ
        | GateType::RX
        | GateType::RY
        | GateType::RZZ
        | GateType::RXX
        | GateType::RYY => {
            if let Some(&angle) = gate.angles.first() {
                if let Some(clifford) = try_simplify_rotation(gate.gate_type, angle) {
                    let _ = apply_named_gate(prop, clifford, &gate.qubits, direction);
                    return;
                }

                if let Some(pauli) = half_turn_decomposition(gate.gate_type, angle) {
                    for &qubit in &gate.qubits {
                        let _ = apply_named_gate(prop, pauli, &[qubit], direction);
                    }
                }
            }
        }
        GateType::R1XY if gate.angles.len() >= 2 => {
            let theta = gate.angles[0];
            let phi = gate.angles[1];
            if let Some(clifford) = try_simplify_r1xy(theta, phi) {
                let _ = apply_named_gate(prop, clifford, &gate.qubits, direction);
            }
        }
        _ => {}
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
            if qubits.len() >= 2 {
                prop.cx(&[(qubits[0], qubits[1])]);
            }
        }
        GateType::CY => {
            if qubits.len() >= 2 {
                prop.cy(&[(qubits[0], qubits[1])]);
            }
        }
        GateType::CZ => {
            if qubits.len() >= 2 {
                prop.cz(&[(qubits[0], qubits[1])]);
            }
        }
        GateType::SWAP => {
            if qubits.len() >= 2 {
                prop.swap(&[(qubits[0], qubits[1])]);
            }
        }

        // Non-self-adjoint two-qubit Clifford gates - swap with adjoint for backward
        GateType::SXX => {
            if qubits.len() >= 2 {
                match direction {
                    Direction::Forward => prop.sxx(&[(qubits[0], qubits[1])]),
                    Direction::Backward => prop.sxxdg(&[(qubits[0], qubits[1])]),
                };
            }
        }
        GateType::SXXdg => {
            if qubits.len() >= 2 {
                match direction {
                    Direction::Forward => prop.sxxdg(&[(qubits[0], qubits[1])]),
                    Direction::Backward => prop.sxx(&[(qubits[0], qubits[1])]),
                };
            }
        }
        GateType::SYY => {
            if qubits.len() >= 2 {
                match direction {
                    Direction::Forward => prop.syy(&[(qubits[0], qubits[1])]),
                    Direction::Backward => prop.syydg(&[(qubits[0], qubits[1])]),
                };
            }
        }
        GateType::SYYdg => {
            if qubits.len() >= 2 {
                match direction {
                    Direction::Forward => prop.syydg(&[(qubits[0], qubits[1])]),
                    Direction::Backward => prop.syy(&[(qubits[0], qubits[1])]),
                };
            }
        }
        GateType::SZZ => {
            if qubits.len() >= 2 {
                match direction {
                    Direction::Forward => prop.szz(&[(qubits[0], qubits[1])]),
                    Direction::Backward => prop.szzdg(&[(qubits[0], qubits[1])]),
                };
            }
        }
        GateType::SZZdg => {
            if qubits.len() >= 2 {
                match direction {
                    Direction::Forward => prop.szzdg(&[(qubits[0], qubits[1])]),
                    Direction::Backward => prop.szz(&[(qubits[0], qubits[1])]),
                };
            }
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
                for gate in tick.gates() {
                    apply_gate(prop, gate, direction);
                }
            }
        }
        Direction::Backward => {
            for tick in circuit.ticks().iter().rev() {
                for gate in tick.gates() {
                    apply_gate(prop, gate, direction);
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
                for gate in tick.gates() {
                    apply_gate(prop, gate, direction);
                }
            }
        }
        Direction::Backward => {
            for tick_idx in (start..=end).rev() {
                let tick = &circuit.ticks()[tick_idx];
                for gate in tick.gates() {
                    apply_gate(prop, gate, direction);
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
