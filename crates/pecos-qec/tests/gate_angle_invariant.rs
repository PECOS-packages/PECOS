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

use pecos_core::{Angle64, Gate, GateAngles, PauliString, QubitId};
use pecos_qec::fault_tolerance::propagator::{CountingRecorder, DagFaultAnalyzer};
use pecos_qec::fault_tolerance::{
    Direction, PauliFrameLookup, propagate_backward_from_node, propagate_sparse_dag,
    propagate_through_dag,
};
use pecos_quantum::{DagCircuit, DagGateError, GateType};
use pecos_simulators::PauliProp;

fn rotation_dag() -> (DagCircuit, usize) {
    let mut dag = DagCircuit::new();
    dag.pz(&[0]);
    let rotation = dag.add_gate_auto_wire(Gate::rz(Angle64::QUARTER_TURN, &[0]));
    dag.mz(&[0]);
    (dag, rotation)
}

fn refused_missing_angle_update(dag: &mut DagCircuit, node: usize) {
    let before = dag.gate(node).cloned().expect("rotation exists");
    let error = dag
        .update_gate(node, |gate| gate.angles.clear())
        .expect_err("a stored RZ cannot lose its required angle");
    assert!(matches!(
        error,
        DagGateError::InvalidGate {
            node: Some(error_node),
            ref message,
        } if error_node == node
            && message == "Gate RZ expected 1 angle parameters, got 0"
    ));
    assert_eq!(dag.gate(node), Some(&before));
}

#[test]
fn infallible_bad_literal_cannot_reach_dag_propagation() {
    let mut constructed = false;
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let gate = Gate::with_angles(GateType::RZ, GateAngles::new(), vec![QubitId::from(0)]);
        constructed = true;
        let mut dag = DagCircuit::new();
        dag.add_gate_auto_wire(gate);
        propagate_through_dag(&dag, &mut PauliProp::new(), Direction::Forward);
    }))
    .expect_err("the malformed literal must fail at construction");
    assert!(!constructed, "the malformed Gate must never be constructed");
    let message = panic
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| panic.downcast_ref::<&str>().copied())
        .expect("assertion panic should carry text");
    assert!(message.contains("Gate RZ expected 1 angle parameters, got 0"));
}

#[test]
fn rejected_update_leaves_all_dag_propagation_entries_safe() {
    let (mut dag, rotation) = rotation_dag();
    refused_missing_angle_update(&mut dag, rotation);

    let mut dense = PauliProp::new();
    dense.track_x(&[0]);
    propagate_through_dag(&dag, &mut dense, Direction::Forward);

    let mut sparse = PauliProp::new();
    sparse.track_x(&[0]);
    propagate_sparse_dag(&dag, &mut sparse, Direction::Forward);

    let mut backward = PauliProp::new();
    backward.track_x(&[0]);
    propagate_backward_from_node(&dag, &mut backward, rotation);
}

#[test]
fn rejected_update_leaves_pauli_frame_lookup_safe() {
    let mut dag = DagCircuit::new();
    dag.pz(&[0]);
    for pauli in [
        PauliString::xs(&[0]),
        PauliString::ys(&[0]),
        PauliString::zs(&[0]),
    ] {
        dag.tracked_pauli(pauli);
    }
    let rotation = dag.add_gate_auto_wire(Gate::rz(Angle64::QUARTER_TURN, &[0]));
    dag.mz(&[0]);
    refused_missing_angle_update(&mut dag, rotation);

    PauliFrameLookup::from_circuit(&dag, &[vec![-1]], &[])
        .expect("the refused update must leave lookup input valid");
}

#[test]
fn rejected_update_leaves_all_dag_fault_analyzer_entries_safe() {
    let (mut dag, rotation) = rotation_dag();
    refused_missing_angle_update(&mut dag, rotation);

    let analyzer = DagFaultAnalyzer::new(&dag);
    let mut recorder = CountingRecorder::default();
    analyzer.propagate_all(&mut recorder);
    let _parallel = analyzer.propagate_all_parallel();
    let _forest = analyzer.propagate_all_forest();
}
