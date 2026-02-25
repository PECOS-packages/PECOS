// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Teleportation circuit primitives.
//!
//! Provides builder functions that construct standard teleportation patterns
//! on a [`DagCircuit`], including Bell pairs, Bell measurements, Pauli
//! corrections, full teleportation, and quantum error-correcting teleportation
//! (QECT) gadgets.

use pecos_core::{ClassicalBitId, QubitId};

use crate::DagCircuit;

/// Create a Bell pair (EPR pair) between two qubits.
///
/// Adds: H(a), CX(a, b).
pub fn bell_pair(dag: &mut DagCircuit, a: impl Into<QubitId>, b: impl Into<QubitId>) {
    let a = a.into();
    let b = b.into();
    dag.h(a);
    dag.cx(a, b);
}

/// Perform a Bell basis measurement on two qubits.
///
/// Adds: CX(a, b), H(a), Measure(a) -> `cbit_a`, Measure(b) -> `cbit_b`.
pub fn bell_measure(
    dag: &mut DagCircuit,
    a: impl Into<QubitId>,
    b: impl Into<QubitId>,
    cbit_a: impl Into<ClassicalBitId>,
    cbit_b: impl Into<ClassicalBitId>,
) {
    let a = a.into();
    let b = b.into();
    dag.cx(a, b);
    dag.h(a);
    dag.mz_to(a, cbit_a);
    dag.mz_to(b, cbit_b);
}

/// Apply Pauli corrections conditioned on Bell measurement outcomes.
///
/// - if `cbit_x` == 1: apply X to target
/// - if `cbit_z` == 1: apply Z to target
pub fn teleport_corrections(
    dag: &mut DagCircuit,
    target: impl Into<QubitId>,
    cbit_x: impl Into<ClassicalBitId>,
    cbit_z: impl Into<ClassicalBitId>,
) {
    let target = target.into();
    dag.if_bit(cbit_x, true).x(target);
    dag.if_bit(cbit_z, true).z(target);
}

/// Full quantum teleportation protocol.
///
/// Teleports the state of `data` to `target` using `ancilla` as the
/// entangled helper qubit.
///
/// The protocol:
/// 1. Create Bell pair between `ancilla` and `target`
/// 2. Bell measurement of `data` and `ancilla`
/// 3. Pauli corrections on `target`
///
/// Requires at least 2 classical bits (for `cbit_x` and `cbit_z`).
pub fn teleportation(
    dag: &mut DagCircuit,
    data: impl Into<QubitId>,
    ancilla: impl Into<QubitId>,
    target: impl Into<QubitId>,
    cbit_x: impl Into<ClassicalBitId>,
    cbit_z: impl Into<ClassicalBitId>,
) {
    let data = data.into();
    let ancilla = ancilla.into();
    let target = target.into();
    let cbit_x = cbit_x.into();
    let cbit_z = cbit_z.into();

    // Step 1: Create Bell pair (ancilla, target)
    bell_pair(dag, ancilla, target);

    // Step 2: Bell measurement (data, ancilla) -> (cbit_x, cbit_z)
    bell_measure(dag, data, ancilla, cbit_x, cbit_z);

    // Step 3: Pauli corrections on target
    teleport_corrections(dag, target, cbit_x, cbit_z);
}

/// Quantum error-correcting teleportation (QECT) gadget.
///
/// Implements the Knill-style QECT round: Bell measurement of `data` with
/// one half of a resource Bell pair (`resource_a`), followed by Pauli
/// corrections on the other half (`resource_b`).
///
/// Assumes the Bell pair (`resource_a`, `resource_b`) is already prepared.
///
/// 1. Bell measurement of `data` and `resource_a` -> (`cbit_x`, `cbit_z`)
/// 2. Pauli corrections on `resource_b`
pub fn qect_gadget(
    dag: &mut DagCircuit,
    data: impl Into<QubitId>,
    resource_a: impl Into<QubitId>,
    resource_b: impl Into<QubitId>,
    cbit_x: impl Into<ClassicalBitId>,
    cbit_z: impl Into<ClassicalBitId>,
) {
    let data = data.into();
    let resource_a = resource_a.into();
    let resource_b = resource_b.into();
    let cbit_x = cbit_x.into();
    let cbit_z = cbit_z.into();

    // Bell measurement of data with resource_a
    bell_measure(dag, data, resource_a, cbit_x, cbit_z);

    // Pauli corrections on resource_b
    teleport_corrections(dag, resource_b, cbit_x, cbit_z);
}

#[cfg(test)]
mod tests {
    use super::*;
    use pecos_core::gate_type::GateType;

    #[test]
    fn test_bell_pair() {
        let mut dag = DagCircuit::new();
        bell_pair(&mut dag, 0usize, 1usize);

        assert_eq!(dag.gate_count(), 2);
        let gates: Vec<_> = dag
            .topological_order()
            .into_iter()
            .filter_map(|n| dag.gate(n).map(|g| g.gate_type))
            .collect();
        assert_eq!(gates, vec![GateType::H, GateType::CX]);
    }

    #[test]
    fn test_bell_measure() {
        let mut dag = DagCircuit::new();
        dag.set_num_cbits(2);
        bell_measure(
            &mut dag,
            0usize,
            1usize,
            ClassicalBitId::new(0),
            ClassicalBitId::new(1),
        );

        assert_eq!(dag.gate_count(), 4); // CX, H, MZ, MZ
        let gates: Vec<_> = dag
            .topological_order()
            .into_iter()
            .filter_map(|n| dag.gate(n).map(|g| g.gate_type))
            .collect();
        // Topological order may vary for independent qubits; check counts
        assert_eq!(gates.iter().filter(|&&g| g == GateType::CX).count(), 1);
        assert_eq!(gates.iter().filter(|&&g| g == GateType::H).count(), 1);
        assert_eq!(gates.iter().filter(|&&g| g == GateType::Measure).count(), 2);

        // Check measurement targets exist
        let meas_nodes: Vec<_> = dag
            .topological_order()
            .into_iter()
            .filter(|&n| {
                dag.gate(n)
                    .is_some_and(|g| g.gate_type == GateType::Measure)
            })
            .collect();
        assert_eq!(meas_nodes.len(), 2);
        let meas_cbits: Vec<_> = meas_nodes
            .iter()
            .filter_map(|&n| dag.measurement_target(n))
            .collect();
        assert!(meas_cbits.contains(&ClassicalBitId::new(0)));
        assert!(meas_cbits.contains(&ClassicalBitId::new(1)));
    }

    #[test]
    fn test_teleport_corrections() {
        let mut dag = DagCircuit::new();
        dag.set_num_cbits(2);
        teleport_corrections(
            &mut dag,
            2usize,
            ClassicalBitId::new(0),
            ClassicalBitId::new(1),
        );

        assert_eq!(dag.gate_count(), 2); // X, Z
        let gates: Vec<_> = dag
            .topological_order()
            .into_iter()
            .filter_map(|n| dag.gate(n).map(|g| g.gate_type))
            .collect();
        assert_eq!(gates, vec![GateType::X, GateType::Z]);

        // Both should be conditional
        let nodes = dag.topological_order();
        assert_eq!(
            dag.condition(nodes[0]),
            Some((ClassicalBitId::new(0), true))
        );
        assert_eq!(
            dag.condition(nodes[1]),
            Some((ClassicalBitId::new(1), true))
        );
    }

    #[test]
    fn test_full_teleportation() {
        let mut dag = DagCircuit::new();
        dag.set_num_cbits(2);
        teleportation(
            &mut dag,
            0usize,
            1usize,
            2usize,
            ClassicalBitId::new(0),
            ClassicalBitId::new(1),
        );

        // Bell pair: H, CX
        // Bell measure: CX, H, MZ, MZ
        // Corrections: X, Z
        assert_eq!(dag.gate_count(), 8);

        let gates: Vec<_> = dag
            .topological_order()
            .into_iter()
            .filter_map(|n| dag.gate(n).map(|g| g.gate_type))
            .collect();

        // Bell pair on (1,2): H(1), CX(1,2)
        // Bell measure on (0,1): CX(0,1), H(0), MZ(0), MZ(1)
        // Corrections on 2: X(2), Z(2)
        assert!(gates.contains(&GateType::H));
        assert!(gates.contains(&GateType::CX));
        assert!(gates.contains(&GateType::Measure));
        assert!(gates.contains(&GateType::X));
        assert!(gates.contains(&GateType::Z));
    }

    #[test]
    fn test_qect_gadget() {
        let mut dag = DagCircuit::new();
        dag.set_num_cbits(2);
        qect_gadget(
            &mut dag,
            0usize,
            1usize,
            2usize,
            ClassicalBitId::new(0),
            ClassicalBitId::new(1),
        );

        // Bell measure: CX, H, MZ, MZ
        // Corrections: X, Z
        assert_eq!(dag.gate_count(), 6);

        let gates: Vec<_> = dag
            .topological_order()
            .into_iter()
            .filter_map(|n| dag.gate(n).map(|g| g.gate_type))
            .collect();

        // Check gate type counts (topological order may vary)
        assert_eq!(gates.iter().filter(|&&g| g == GateType::CX).count(), 1);
        assert_eq!(gates.iter().filter(|&&g| g == GateType::H).count(), 1);
        assert_eq!(gates.iter().filter(|&&g| g == GateType::Measure).count(), 2);
        assert_eq!(gates.iter().filter(|&&g| g == GateType::X).count(), 1);
        assert_eq!(gates.iter().filter(|&&g| g == GateType::Z).count(), 1);
    }

    #[test]
    fn test_teleportation_qubit_coverage() {
        let mut dag = DagCircuit::new();
        dag.set_num_cbits(2);
        teleportation(
            &mut dag,
            0usize,
            1usize,
            2usize,
            ClassicalBitId::new(0),
            ClassicalBitId::new(1),
        );

        // Should use qubits 0, 1, 2
        assert_eq!(dag.max_qubit() + 1, 3);
    }

    #[test]
    fn test_teleportation_to_tick_circuit() {
        use crate::TickCircuit;

        let mut dag = DagCircuit::new();
        dag.set_num_cbits(2);
        teleportation(
            &mut dag,
            0usize,
            1usize,
            2usize,
            ClassicalBitId::new(0),
            ClassicalBitId::new(1),
        );

        let tc = TickCircuit::from(&dag);
        assert!(tc.num_ticks() > 0);
        assert_eq!(tc.num_cbits(), 2);
    }
}
