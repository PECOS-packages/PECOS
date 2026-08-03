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

//! Collapse-semantics discriminators at the influence-map level.
//!
//! Collapse projects; it does not reset. The physical simulators keep the
//! post-measurement eigenstate, so an X error before a non-destructive `MZ`
//! flips it AND every later measurement on the qubit until a reset or free.
//! These tests pin that end to end, where a repeated measurement actually
//! exists -- the surface-code circuits re-prepare every ancilla, so they
//! cannot distinguish collapse from reset.

use pecos_qec::fault_tolerance::propagator::DagFaultAnalyzer;
use pecos_quantum::{DagCircuit, GateType};

/// Influence indices of a fault at (`node`, `before`) for the given fault
/// Pauli, resolved by physical location rather than index.
fn detector_hits(
    map: &pecos_qec::fault_tolerance::propagator::DagFaultInfluenceMap,
    node: usize,
    before: bool,
    pauli: pecos_qec::fault_tolerance::propagator::types::Pauli,
) -> Vec<u32> {
    let loc_idx = map
        .locations
        .iter()
        .position(|loc| loc.node == node && loc.before == before)
        .expect("the fault location exists");
    let mut hits = map.get_detector_indices(loc_idx, pauli.as_u8()).to_vec();
    hits.sort_unstable();
    hits
}

/// X before the first of two measurements flips BOTH; between them, only the
/// second. The old clearing predicted the first case flips one measurement.
#[test]
fn an_x_before_a_repeated_mz_flips_both_measurements() {
    use pecos_qec::fault_tolerance::propagator::types::Pauli;

    let mut dag = DagCircuit::new();
    dag.pz(&[0]);
    let m1 = dag.mz(&[0]);
    let m2 = dag.mz(&[0]);
    let map = DagFaultAnalyzer::new(&dag).build_influence_map();

    let i1 = u32::try_from(
        map.meas_index_of(m1[0].meas_id)
            .expect("measurement 1 is in the map"),
    )
    .expect("fits");
    let i2 = u32::try_from(
        map.meas_index_of(m2[0].meas_id)
            .expect("measurement 2 is in the map"),
    )
    .expect("fits");

    // X after the prep (before both measurements): flips both.
    assert_eq!(
        detector_hits(&map, 0, false, Pauli::X),
        {
            let mut want = vec![i1, i2];
            want.sort_unstable();
            want
        },
        "collapse projects; it does not reset -- the X keeps flipping"
    );
    // X between the measurements (before the second): flips the second only.
    assert_eq!(detector_hits(&map, m2[0].node, true, Pauli::X), vec![i2]);
    // Z anywhere: flips nothing.
    assert_eq!(detector_hits(&map, 0, false, Pauli::Z), Vec::<u32>::new());
}

/// The backward mirror kills a phantom: with an H between two measurements,
/// the second measurement's observable arrives at the first collapse as an
/// X-type operator, which gains no deterministic dependence on anything
/// earlier. Passing it through (the old identity behavior) manufactured a
/// "Z fault before the first measurement flips the second" influence that no
/// simulator reproduces.
#[test]
fn a_z_before_a_collapse_does_not_haunt_a_later_rotated_measurement() {
    use pecos_qec::fault_tolerance::propagator::types::Pauli;

    let mut dag = DagCircuit::new();
    dag.pz(&[0]);
    let m1 = dag.mz(&[0]);
    dag.h(&[0]);
    let m2 = dag.mz(&[0]);
    let map = DagFaultAnalyzer::new(&dag).build_influence_map();

    let i1 = u32::try_from(
        map.meas_index_of(m1[0].meas_id)
            .expect("measurement 1 is in the map"),
    )
    .expect("fits");
    let i2 = u32::try_from(
        map.meas_index_of(m2[0].meas_id)
            .expect("measurement 2 is in the map"),
    )
    .expect("fits");

    // A fault after the prep can flip the first measurement (X component)
    // but must NOT influence the rotated second measurement at all: its
    // observable does not deterministically depend on anything before the
    // collapse.
    assert_eq!(
        detector_hits(&map, 0, false, Pauli::X),
        vec![i1],
        "X flips the first measurement only"
    );
    assert_eq!(
        detector_hits(&map, 0, false, Pauli::Z),
        Vec::<u32>::new(),
        "the old identity-crossing manufactured a phantom influence here"
    );
    let _ = i2;
}

/// `MeasureFree` genuinely discards: nothing before it reaches anything
/// after, even on the same qubit id.
#[test]
fn a_measure_free_stops_propagation_where_a_plain_mz_does_not() {
    use pecos_core::Gate;
    use pecos_qec::fault_tolerance::propagator::types::Pauli;

    let build = |leading: GateType| {
        let mut dag = DagCircuit::new();
        dag.pz(&[0]);
        let leading_gate = match leading {
            GateType::MeasureFree => Gate::mz_free(&[0usize]),
            _ => Gate::mz(&[0usize]),
        };
        dag.try_add_gate_auto_wire(leading_gate)
            .expect("gate is valid");
        let m2 = dag.mz(&[0]);
        let map = DagFaultAnalyzer::new(&dag).build_influence_map();
        let i2 = u32::try_from(
            map.meas_index_of(m2[0].meas_id)
                .expect("measurement 2 is in the map"),
        )
        .expect("fits");
        let hits = detector_hits(&map, 0, false, Pauli::X);
        (hits, i2)
    };

    let (mz_hits, mz_i2) = build(GateType::MZ);
    assert!(
        mz_hits.contains(&mz_i2),
        "through a plain MZ the X reaches the second measurement"
    );
    let (free_hits, free_i2) = build(GateType::MeasureFree);
    assert!(
        !free_hits.contains(&free_i2),
        "through a MeasureFree it does not -- the qubit was discarded"
    );
}
