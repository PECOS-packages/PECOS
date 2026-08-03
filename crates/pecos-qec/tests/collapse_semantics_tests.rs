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

/// A detector over two same-qubit measurements must keep the influences that
/// lie BETWEEN its members. Seeding the combined observable at the circuit
/// end cancels the two Z seeds before the walk starts, deleting the exact
/// mechanism the detector exists to catch (Stim: `error(p) D0`).
#[test]
fn a_detector_over_repeated_measurements_keeps_between_faults() {
    use pecos_qec::fault_tolerance::InfluenceBuilder;
    use pecos_qec::fault_tolerance::propagator::types::Pauli;

    let mut dag = DagCircuit::new();
    dag.pz(&[0]);
    dag.h(&[0]);
    let m0 = dag.mz(&[0]);
    dag.z(&[0]);
    let m1 = dag.mz(&[0]);
    dag.detector(&[m0[0], m1[0]])
        .expect("refs are from this circuit");
    let map = InfluenceBuilder::new(&dag)
        .with_circuit_annotations()
        .expect("annotations resolve")
        .build()
        .expect("circuit is replayable");

    let z_gate_loc = map
        .locations
        .iter()
        .position(|loc| loc.gate_type == GateType::Z && !loc.before)
        .expect("the Z gate has an after location");
    assert_eq!(
        map.get_detector_indices(z_gate_loc, Pauli::X.as_u8()),
        &[0],
        "an X between the two measurements flips exactly the second, so it \
         flips their XOR"
    );
    // Faults before the first measurement flip both members and cancel.
    let prep_loc = map
        .locations
        .iter()
        .position(|loc| loc.node == 0 && !loc.before)
        .expect("the prep has an after location");
    for pauli in [Pauli::X, Pauli::Y, Pauli::Z] {
        assert_eq!(
            map.get_detector_indices(prep_loc, pauli.as_u8()),
            Vec::<u32>::new(),
            "{pauli:?} before both members flips both, cancelling in the XOR"
        );
    }
}

/// An SX between two measurements delivers the backward observable to the
/// first collapse as a Y, exercising the keep-Z branch at integration level:
/// the Z part passes (faults before the first measurement reach the second,
/// relative to the measurement gauge), the X part is dropped.
#[test]
fn an_sx_rotated_repeated_measurement_keeps_the_z_component() {
    use pecos_qec::fault_tolerance::propagator::types::Pauli;

    let mut dag = DagCircuit::new();
    dag.pz(&[0]);
    let m1 = dag.mz(&[0]);
    dag.sx(&[0]);
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

    // X after the prep: flips m1 directly; the passed Z component of m2's
    // backward observable anticommutes with it too, so it reaches m2.
    let mut hits = detector_hits(&map, 0, false, Pauli::X);
    hits.sort_unstable();
    let mut want = vec![i1, i2];
    want.sort_unstable();
    assert_eq!(hits, want);
    // Z after the prep: commutes with everything that survives to cross.
    assert_eq!(detector_hits(&map, 0, false, Pauli::Z), Vec::<u32>::new());
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

/// `MPZ` stops propagation exactly where `MeasureFree` does -- its built-in
/// reset absorbs the error -- but records a real measurement whose flip the
/// fault still causes, and leaves the wire usable.
#[test]
fn an_mpz_records_its_flip_and_stops_propagation() {
    use pecos_qec::fault_tolerance::propagator::types::Pauli;

    let mut dag = DagCircuit::new();
    dag.pz(&[0]);
    let m1 = dag.mpz(&[0]);
    let m2 = dag.mz(&[0]);
    let map = DagFaultAnalyzer::new(&dag).build_influence_map();

    let i1 = u32::try_from(
        map.meas_index_of(m1[0].meas_id)
            .expect("the MPZ measurement is in the map"),
    )
    .expect("fits");
    let i2 = u32::try_from(
        map.meas_index_of(m2[0].meas_id)
            .expect("the second measurement is in the map"),
    )
    .expect("fits");

    // X after the prep flips the MPZ readout and nothing after the reset.
    assert_eq!(
        detector_hits(&map, 0, false, Pauli::X),
        vec![i1],
        "the flip is recorded; the reset stops it from reaching {i2}"
    );
    // Between the MPZ and the final MZ: only the final measurement flips.
    assert_eq!(detector_hits(&map, m2[0].node, true, Pauli::X), vec![i2]);
}

/// The tick-based analyzer extracts MPZ measurements like its DAG sibling --
/// the sweep that added MPZ to one previously missed the other, silently
/// thinning the influence map.
#[test]
fn the_tick_analyzer_extracts_mpz_measurements() {
    use pecos_qec::fault_tolerance::TickFaultAnalyzer;
    use pecos_quantum::TickCircuit;

    let mut tc = TickCircuit::new();
    tc.tick().pz(&[0]);
    tc.tick().mpz(&[0]);
    tc.tick().mz(&[0]);
    let map = TickFaultAnalyzer::new(&tc).build_influence_map();
    assert_eq!(
        map.measurements.len(),
        2,
        "the MPZ record must not vanish from the tick analyzer"
    );
}

/// An MPZ record takes measurement noise in the sampler lane: with p_meas
/// alone, its detector must carry a flip mechanism. The lane previously fell
/// through to the 1q-depolarizing bucket (or nothing), leaving MPZ records
/// silently noiseless.
#[test]
fn an_mpz_record_takes_measurement_noise_in_the_sampler_lane() {
    use pecos_qec::fault_tolerance::InfluenceBuilder;
    use pecos_qec::fault_tolerance::dem_builder::DemSamplerBuilder;

    let mut dag = DagCircuit::new();
    dag.pz(&[0]);
    let m = dag.mpz(&[0]);
    dag.detector(&[m[0]]).expect("refs are from this circuit");
    let map = InfluenceBuilder::new(&dag)
        .with_circuit_annotations()
        .expect("annotations resolve")
        .build()
        .expect("circuit is replayable");
    let sampler = DemSamplerBuilder::new(&map)
        .with_noise(0.0, 0.0, 0.2, 0.0)
        .raw_measurements()
        .build()
        .expect("sampler builds");
    assert!(
        sampler.average_error_probability() > 0.0,
        "p_meas must produce a mechanism on the MPZ record"
    );
}
