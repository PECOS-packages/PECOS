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

//! De-aliased measurement-identity tests.
//!
//! In most circuits `meas_id == record index == rank`, so three different
//! number spaces collapse into one and conflation bugs are invisible. Every
//! test here uses non-positional, non-contiguous ids (the `mz_with_ids`
//! pattern, e.g. 9, 1, 5) so that using the wrong space produces a wrong
//! answer instead of a coincidentally right one.

use pecos_core::{Gate, MeasId, QubitId};
use pecos_qec::fault_tolerance::InfluenceBuilder;
use pecos_qec::fault_tolerance::dem_builder::DemSamplerBuilder;
use pecos_qec::fault_tolerance::propagator::types::Pauli;
use pecos_quantum::{AnnotationKind, DagCircuit, TickCircuit};

/// `Gate::mz` on one qubit carrying a chosen id.
fn mz_with_id(qubit: usize, id: usize) -> Gate {
    let mut gate = Gate::mz(&[qubit]);
    gate.meas_ids = smallvec::smallvec![MeasId::from_raw(id)];
    gate
}

/// The same physical parity-check circuit, ids chosen by the caller.
///
/// Prep 0..=2, entangle data 0,1 onto ancilla 2, measure the ancilla and both
/// data qubits. `ids[0]` names the ancilla measurement, `ids[1]`/`ids[2]` the
/// data measurements.
fn parity_check_with_ids(ids: [usize; 3]) -> DagCircuit {
    let mut dag = DagCircuit::new();
    dag.pz(&[0, 1, 2]);
    dag.cx(&[(0, 2)]);
    dag.cx(&[(1, 2)]);
    for (qubit, id) in [(2, ids[0]), (0, ids[1]), (1, ids[2])] {
        dag.try_add_gate_auto_wire(mz_with_id(qubit, id))
            .expect("gate is valid and id is fresh");
    }
    let refs: Vec<_> = ids
        .iter()
        .map(|&id| {
            dag.find_measurement(MeasId::from_raw(id))
                .expect("the id was just supplied")
        })
        .collect();
    // Detector: ancilla XOR both data readouts is deterministic.
    dag.detector(&refs).expect("refs are from this circuit");
    // Observable: logical Z read from data qubit 0's measurement.
    dag.observable(&[refs[1]])
        .expect("refs are from this circuit");
    dag
}

/// Scrambled ids must produce the same fault-to-output relationships as
/// positional ids: identity is a name, not a coordinate.
#[test]
fn scrambled_ids_build_the_same_influence_relations_as_positional_ids() {
    let positional = parity_check_with_ids([0, 1, 2]);
    let scrambled = parity_check_with_ids([9, 4, 7]);

    let map_pos = InfluenceBuilder::new(&positional)
        .unwrap()
        .with_circuit_annotations()
        .expect("annotations resolve")
        .build()
        .expect("circuit is replayable");
    let map_scr = InfluenceBuilder::new(&scrambled)
        .unwrap()
        .with_circuit_annotations()
        .expect("annotations resolve")
        .build()
        .expect("circuit is replayable");

    assert_eq!(map_pos.locations.len(), map_scr.locations.len());
    assert_eq!(map_pos.num_observables(), map_scr.num_observables());
    // Location order is itself a per-build ordinal, so locations are matched
    // by their physical coordinates, never by index.
    let scr_by_key = location_index_by_key(&map_scr);
    let mut influence_seen = false;
    for (pos_idx, loc) in map_pos.locations.iter().enumerate() {
        let scr_idx = scr_by_key[&location_key(loc)];
        for pauli in [Pauli::X, Pauli::Y, Pauli::Z] {
            let pos = map_pos.get_observable_indices(pos_idx, pauli.as_u8());
            let scr = map_scr.get_observable_indices(scr_idx, pauli.as_u8());
            assert_eq!(
                pos, scr,
                "fault at {loc:?} pauli {pauli:?} must flip the same \
                 observables regardless of id values"
            );
            influence_seen |= !pos.is_empty();
        }
    }
    assert!(
        influence_seen,
        "the comparison is vacuous unless some fault influences some observable"
    );
}

type LocationKey = (usize, Vec<usize>, bool, pecos_core::gate_type::GateType);

fn location_key(
    loc: &pecos_qec::fault_tolerance::propagator::dag::DagSpacetimeLocation,
) -> LocationKey {
    (
        loc.node,
        loc.qubits.iter().map(pecos_core::QubitId::index).collect(),
        loc.before,
        loc.gate_type,
    )
}

fn location_index_by_key(
    map: &pecos_qec::fault_tolerance::propagator::DagFaultInfluenceMap,
) -> std::collections::BTreeMap<LocationKey, usize> {
    let index: std::collections::BTreeMap<LocationKey, usize> = map
        .locations
        .iter()
        .enumerate()
        .map(|(idx, loc)| (location_key(loc), idx))
        .collect();
    assert_eq!(
        index.len(),
        map.locations.len(),
        "location keys must be unique for by-key matching to be sound"
    );
    index
}

/// The per-measurement fault influence must be the same physical relation
/// under both id sets: each fault flips the same *named* measurements, with
/// indices compared through each map's own `meas_index_of` ordinal.
///
/// Shot-by-shot stream comparison at a shared seed is NOT a valid probe here:
/// mechanism tables are keyed by detector indices, so re-labeling ids reorders
/// the table and the same seed draws different faults. The structure is the
/// invariant; the streams are only equal in distribution.
#[test]
fn scrambled_ids_influence_the_same_named_measurements() {
    let positional = parity_check_with_ids([0, 1, 2]);
    let scrambled = parity_check_with_ids([9, 4, 7]);

    let build = |dag: &DagCircuit| {
        InfluenceBuilder::new(dag)
            .unwrap()
            .with_circuit_annotations()
            .expect("annotations resolve")
            .build()
            .expect("circuit is replayable")
    };
    let map_pos = build(&positional);
    let map_scr = build(&scrambled);

    // scr-index -> pos-index for the same physical measurement: ids[i] of one
    // circuit names the same physical measurement as ids[i] of the other.
    let mut scr_to_pos = [usize::MAX; 3];
    for (pos_id, scr_id) in [(0usize, 9usize), (1, 4), (2, 7)] {
        let pi = map_pos
            .meas_index_of(MeasId::from_raw(pos_id))
            .expect("id is in the positional map");
        let si = map_scr
            .meas_index_of(MeasId::from_raw(scr_id))
            .expect("id is in the scrambled map");
        scr_to_pos[si] = pi;
    }

    assert_eq!(map_pos.locations.len(), map_scr.locations.len());
    let scr_by_key = location_index_by_key(&map_scr);
    let mut influence_seen = false;
    for (pos_idx, loc) in map_pos.locations.iter().enumerate() {
        let scr_idx = scr_by_key[&location_key(loc)];
        for pauli in [Pauli::X, Pauli::Y, Pauli::Z] {
            let mut pos: Vec<u32> = map_pos
                .get_detector_indices(pos_idx, pauli.as_u8())
                .to_vec();
            let mut scr_mapped: Vec<u32> = map_scr
                .get_detector_indices(scr_idx, pauli.as_u8())
                .iter()
                .map(|&si| u32::try_from(scr_to_pos[si as usize]).expect("index fits"))
                .collect();
            pos.sort_unstable();
            scr_mapped.sort_unstable();
            assert_eq!(
                pos, scr_mapped,
                "fault at {loc:?} pauli {pauli:?} must flip the same \
                 named measurements regardless of id values"
            );
            influence_seen |= !pos.is_empty();
        }
    }
    assert!(
        influence_seen,
        "the comparison is vacuous unless some fault flips some measurement"
    );
}

/// Under scrambled ids, `sample_dual` detector events must XOR exactly the
/// raw channels that `meas_index_of` names. Both sides of the assertion
/// resolve through the same `meas_index_of`, so this pins the sampler's
/// internal consistency; the one-order-per-map invariant itself is killed by
/// `scrambled_ids_influence_the_same_named_measurements`, whose id-to-ordinal
/// map is checked against the independent influence data.
#[test]
fn scrambled_ids_dual_output_xors_the_named_raw_channels() {
    use pecos_random::PecosRng;

    // q1's measurement gets the LOW id, so its id rank disagrees with both
    // circuit order and qubit order.
    let mut dag = DagCircuit::new();
    dag.pz(&[0, 1]);
    for (qubit, id) in [(0usize, 7usize), (1, 3)] {
        dag.try_add_gate_auto_wire(mz_with_id(qubit, id))
            .expect("gate is valid");
    }
    let target = dag
        .find_measurement(MeasId::from_raw(3))
        .expect("the id was just supplied");
    dag.detector(&[target]).expect("refs are from this circuit");

    let map = InfluenceBuilder::new(&dag)
        .unwrap()
        .with_circuit_annotations()
        .expect("annotations resolve")
        .build()
        .expect("circuit is replayable");
    let raw_idx = map
        .meas_index_of(MeasId::from_raw(3))
        .expect("the id is in the map");
    let other_idx = map
        .meas_index_of(MeasId::from_raw(7))
        .expect("the id is in the map");
    let sampler = DemSamplerBuilder::new(&map)
        .with_uniform_noise(0.3)
        .raw_measurements()
        .with_circuit_annotations(&dag)
        .expect("annotations resolve against the influence map")
        .build()
        .expect("sampler builds");

    let mut rng = PecosRng::seed_from_u64(11);
    let (mut saw_flip, mut saw_clear, mut saw_disagreement) = (false, false, false);
    for _ in 0..500 {
        let shot = sampler
            .sample_dual(&mut rng)
            .expect("detectors are configured");
        assert_eq!(
            shot.detector_events[0], shot.raw_measurements[raw_idx],
            "the detector must XOR exactly its named raw channel"
        );
        saw_flip |= shot.raw_measurements[raw_idx];
        saw_clear |= !shot.raw_measurements[raw_idx];
        saw_disagreement |= shot.raw_measurements[raw_idx] != shot.raw_measurements[other_idx];
    }
    assert!(
        saw_flip && saw_clear && saw_disagreement,
        "the witness is vacuous unless the named channel varies and disagrees \
         with the other channel at least once"
    );
}

/// DAG mutation refuses to leave two measurements holding one id, so sampler
/// annotation resolution continues to use the original distinct identities.
#[test]
fn dag_update_refuses_duplicate_measurement_ids() {
    use pecos_qec::fault_tolerance::propagator::DagFaultAnalyzer;

    let mut dag = DagCircuit::new();
    dag.pz(&[0, 1]);
    let a = dag.mz(&[0]);
    let b = dag.mz(&[1]);
    let error = dag
        .update_gate(b[0].node, |gate| gate.meas_ids[0] = a[0].meas_id)
        .expect_err("measurement IDs are DAG-owned identity data");
    assert!(error.to_string().contains("cannot change measurement IDs"));

    let map = DagFaultAnalyzer::new(&dag).build_influence_map().unwrap();
    DemSamplerBuilder::new(&map)
        .with_uniform_noise(0.01)
        .raw_measurements()
        .with_circuit_annotations(&dag)
        .expect("the rejected edit must leave distinct measurement IDs");
}

/// A detector can reference exactly one measurement of a batched gate --
/// previously inexpressible, because a node id covered the whole batch. The
/// resolved measured Pauli must touch only that measurement's qubit.
#[test]
fn a_detector_on_one_measurement_of_a_batched_gate_touches_only_its_qubit() {
    let mut dag = DagCircuit::new();
    dag.pz(&[0, 1]);
    let node = dag
        .try_add_gate_auto_wire(Gate::mz(&[0, 1]))
        .expect("gate is valid");
    let refs = dag.meas_refs(node).expect("MZ holds measurements");
    assert_eq!(refs.len(), 2);
    dag.detector(&[refs[1]])
        .expect("refs are from this circuit");

    let ann = &dag.annotations()[0];
    let AnnotationKind::Detector {
        measurement_ids, ..
    } = &ann.kind
    else {
        panic!("expected detector annotation");
    };
    assert_eq!(measurement_ids.as_slice(), &[refs[1].meas_id]);
    assert_eq!(
        dag.measured_pauli(measurement_ids[0]).unwrap(),
        (pecos_core::Pauli::Z, QubitId(1)),
        "Z on qubit 1 only -- not sprayed over the whole batch"
    );
}

/// A reference to a removed measurement is its own loud failure at the
/// consumer, distinct from an unknown id. Construction validation cannot see
/// it: the annotation arrives pre-built, as it would from a conversion.
#[test]
fn a_reference_to_a_removed_measurement_is_a_removed_error() {
    let mut dag = DagCircuit::new();
    dag.pz(&[0, 1]);
    let ms = dag.mz(&[0]);
    dag.mz(&[1]);
    dag.remove_gate(ms[0].node);
    dag.add_annotation(pecos_quantum::PauliAnnotation {
        kind: AnnotationKind::Observable {
            measurement_ids: vec![ms[0].meas_id],
        },
        label: None,
    });

    let err = InfluenceBuilder::new(&dag)
        .unwrap()
        .with_circuit_annotations()
        .map(|_| ())
        .expect_err("the measurement was removed");
    assert!(matches!(
        err,
        pecos_qec::fault_tolerance::influence_builder::AnnotationIngestError::ObservableRefUnresolved {
            source: pecos_quantum::MeasResolveError::Removed(_),
            ..
        }
    ));
}

/// Tick -> Dag -> Tick round-trip preserves scrambled annotation ids verbatim.
#[test]
fn scrambled_annotation_ids_survive_tick_dag_round_trip() {
    let mut tc = TickCircuit::new();
    tc.tick().pz(&[0, 1]);
    for (qubit, id) in [(0usize, 9usize), (1, 5)] {
        tc.tick()
            .try_add_gate(mz_with_id(qubit, id))
            .expect("gate is valid");
    }
    let refs: Vec<_> = (0..2)
        .map(|tick| {
            tc.meas_ref(tick + 1, 0, QubitId::from(tick))
                .expect("the measurement is there")
        })
        .collect();
    tc.detector(&refs).expect("refs are from this circuit");
    tc.observable(&[refs[1]])
        .expect("refs are from this circuit");

    let dag = DagCircuit::try_from(&tc).expect("valid circuit");
    let back = TickCircuit::from(&dag);

    let expect_ids = |kind: &AnnotationKind, want: &[usize]| {
        let ids = match kind {
            AnnotationKind::Detector {
                measurement_ids, ..
            }
            | AnnotationKind::Observable { measurement_ids } => measurement_ids,
            AnnotationKind::TrackedPauli { .. } => panic!("unexpected tracked Pauli"),
        };
        let want: Vec<MeasId> = want.iter().map(|&id| MeasId::from_raw(id)).collect();
        assert_eq!(ids, &want);
    };
    for circuit_annotations in [dag.annotations(), back.annotations()] {
        assert_eq!(circuit_annotations.len(), 2);
        expect_ids(&circuit_annotations[0].kind, &[9, 5]);
        expect_ids(&circuit_annotations[1].kind, &[5]);
    }
}

/// A sparse id must not cause an id-sized allocation anywhere in the pipeline.
/// If any consumer indexes a dense structure by raw id value, this either
/// fails or takes far too long to pass.
#[test]
fn a_huge_sparse_id_causes_no_id_sized_allocation() {
    let huge = 1usize << 44;
    let mut dag = DagCircuit::new();
    dag.pz(&[0, 1, 2]);
    dag.cx(&[(0, 2)]);
    dag.cx(&[(1, 2)]);
    dag.try_add_gate_auto_wire(mz_with_id(2, huge))
        .expect("gate is valid");
    let mref = dag
        .find_measurement(MeasId::from_raw(huge))
        .expect("the id was just supplied");
    dag.detector(&[mref]).expect("refs are from this circuit");

    let map = InfluenceBuilder::new(&dag)
        .unwrap()
        .with_circuit_annotations()
        .expect("annotations resolve")
        .build()
        .expect("circuit is replayable");
    assert_eq!(
        map.meas_index_of(MeasId::from_raw(huge)),
        Some(0),
        "one measurement exists and the huge id ranks first"
    );
}

/// A public influence-map payload can contain duplicate stamped identities even
/// though DAG admission refuses duplicate Gate measurement IDs.
#[test]
fn sampler_defends_against_duplicate_ids_in_literal_map() {
    use pecos_qec::fault_tolerance::propagator::DagFaultInfluenceMap;
    let gates = [0, 1].map(|q| Gate {
        meas_ids: vec![MeasId::from_raw(5)].into(),
        ..Gate::mz(&[q])
    });
    let mut map = DagFaultInfluenceMap::with_capacity(0);
    map.meas_ids = gates
        .iter()
        .flat_map(|gate| gate.meas_ids.iter().copied())
        .collect();
    let result = DemSamplerBuilder::new(&map).with_circuit_annotations(&DagCircuit::new());
    let Err(error) = result else {
        panic!("duplicate IDs must not bind to the first record");
    };
    assert!(error.to_string().contains("duplicate stable MeasId 5"));
}

#[test]
fn repeated_measurement_qubits_preserve_spacetime_bases() {
    let mut dag = DagCircuit::new();
    let first = dag.mz(&[0]);
    let node = dag.try_add_gate_auto_wire(Gate::mx(&[0usize])).unwrap();
    let second = dag.meas_refs(node).unwrap();
    let refs = [second[0], first[0], second[0]];
    dag.detector(&refs).unwrap();
    dag.observable(&refs).unwrap();
    for annotation in dag.annotations() {
        let ids = match &annotation.kind {
            AnnotationKind::Detector {
                measurement_ids, ..
            }
            | AnnotationKind::Observable { measurement_ids } => measurement_ids,
            AnnotationKind::TrackedPauli { .. } => panic!("unexpected tracked Pauli"),
        };
        assert_eq!(
            ids,
            &[second[0].meas_id, first[0].meas_id, second[0].meas_id]
        );
    }
    assert_eq!(
        dag.measured_pauli(first[0].meas_id).unwrap(),
        (pecos_core::Pauli::Z, QubitId(0))
    );
    assert_eq!(
        dag.measured_pauli(second[0].meas_id).unwrap(),
        (pecos_core::Pauli::X, QubitId(0))
    );

    let mut ticks = TickCircuit::new();
    let first = ticks.tick().mz(&[0]);
    let second = ticks.tick().mx(&[0]);
    let refs = [second[0], first[0], second[0]];
    ticks.detector(&refs).unwrap();
    ticks.observable(&refs).unwrap();
    for annotation in ticks.annotations() {
        let ids = match &annotation.kind {
            AnnotationKind::Detector {
                measurement_ids, ..
            }
            | AnnotationKind::Observable { measurement_ids } => measurement_ids,
            AnnotationKind::TrackedPauli { .. } => panic!("unexpected tracked Pauli"),
        };
        assert_eq!(
            ids,
            &[second[0].meas_id, first[0].meas_id, second[0].meas_id]
        );
    }
    assert_eq!(
        ticks.measured_pauli(first[0].meas_id).unwrap(),
        (pecos_core::Pauli::Z, QubitId(0))
    );
    assert_eq!(
        ticks.measured_pauli(second[0].meas_id).unwrap(),
        (pecos_core::Pauli::X, QubitId(0))
    );
    let converted = DagCircuit::try_from(&ticks).unwrap();
    let round_trip = TickCircuit::from(&converted);
    for id in [first[0].meas_id, second[0].meas_id] {
        assert_eq!(
            converted.measured_pauli(id).unwrap(),
            ticks.measured_pauli(id).unwrap()
        );
        assert_eq!(
            round_trip.measured_pauli(id).unwrap(),
            ticks.measured_pauli(id).unwrap()
        );
    }
}

#[test]
fn measured_pauli_queries_read_current_gates_and_reject_unresolved_ids() {
    use pecos_core::{Pauli, gate_type::GateType};
    use pecos_quantum::{MeasResolveError, TickMeasResolveError};

    let mut dag = DagCircuit::new();
    let reference = dag.mz(&[2])[0];
    dag.update_gate(reference.node, |gate| gate.gate_type = GateType::MX)
        .unwrap();
    assert_eq!(
        dag.measured_pauli(reference.meas_id).unwrap(),
        (Pauli::X, QubitId(2))
    );
    dag.remove_gate(reference.node);
    assert_eq!(
        dag.measured_pauli(reference.meas_id),
        Err(MeasResolveError::Removed(reference.meas_id))
    );
    assert_eq!(
        dag.measured_pauli(MeasId::from_raw(99)),
        Err(MeasResolveError::Unknown(MeasId::from_raw(99)))
    );

    let mut ticks = TickCircuit::new();
    let reference = ticks.tick().mz(&[2])[0];
    ticks
        .get_tick_mut(reference.tick)
        .unwrap()
        .update_gate_batch(reference.gate_idx, |gate| gate.gate_type = GateType::MX)
        .unwrap();
    assert_eq!(
        ticks.measured_pauli(reference.meas_id).unwrap(),
        (Pauli::X, QubitId(2))
    );
    assert_eq!(
        ticks.measured_pauli(MeasId::from_raw(99)),
        Err(TickMeasResolveError::Unknown(MeasId::from_raw(99)))
    );
    let duplicate = mz_with_id(3, reference.meas_id.index());
    ticks.tick().try_add_gate(duplicate).unwrap();
    assert_eq!(
        ticks.measured_pauli(reference.meas_id),
        Err(TickMeasResolveError::Inconsistent(reference.meas_id))
    );

    let mut leaked = Gate::measure_leaked(&[4usize]);
    let id = MeasId::from_raw(7);
    leaked.meas_ids = smallvec::smallvec![id];
    let node = dag.try_add_gate_auto_wire(leaked.clone()).unwrap();
    assert_eq!(
        dag.measured_pauli(id),
        Err(MeasResolveError::RecordLess { id, node })
    );
    let mut ticks = TickCircuit::new();
    ticks.tick().try_add_gate(leaked).unwrap();
    assert_eq!(
        ticks.measured_pauli(id),
        Err(TickMeasResolveError::RecordLess {
            id,
            tick: 0,
            gate_idx: 0
        })
    );
}

#[test]
fn tick_batch_measurement_resolution_matches_individual_queries() {
    use pecos_quantum::TickMeasResolveError;

    let mut circuit = TickCircuit::new();
    let z = circuit.tick().mz(&[2])[0].meas_id;
    let x = circuit.tick().mx(&[2])[0].meas_id;
    let unknown = MeasId::from_raw(99);
    let ids = [x, z, x];
    let expected: Result<Vec<_>, _> = ids.iter().map(|&id| circuit.measured_pauli(id)).collect();
    assert_eq!(circuit.measured_paulis(&ids), expected);
    assert_eq!(circuit.measured_paulis(&[]).unwrap(), []);

    circuit
        .tick()
        .try_add_gate(mz_with_id(3, x.index()))
        .unwrap();
    // A duplicate holder for an unrequested identity must not affect valid results.
    assert_eq!(circuit.measured_paulis(&[z, z]).unwrap().len(), 2);
    // Report errors by input order, not by the order their holders are scanned.
    for ids in [[z, unknown, x], [z, x, unknown]] {
        let expected: Result<Vec<_>, _> =
            ids.iter().map(|&id| circuit.measured_pauli(id)).collect();
        assert_eq!(circuit.measured_paulis(&ids), expected);
    }
    assert_eq!(
        circuit.measured_paulis(&[unknown, x]),
        Err(TickMeasResolveError::Unknown(unknown))
    );
    assert_eq!(
        circuit.measured_paulis(&[x, unknown]),
        Err(TickMeasResolveError::Inconsistent(x))
    );

    let leaked_id = MeasId::from_raw(7);
    let mut leaked = Gate::measure_leaked(&[4usize]);
    leaked.meas_ids = smallvec::smallvec![leaked_id];
    circuit.tick().try_add_gate(leaked).unwrap();
    assert_eq!(
        circuit.measured_paulis(&[leaked_id]),
        circuit.measured_pauli(leaked_id).map(|value| vec![value])
    );
    circuit
        .tick()
        .try_add_gate(mz_with_id(4, leaked_id.index()))
        .unwrap();
    assert_eq!(
        circuit.measured_paulis(&[leaked_id]),
        Err(TickMeasResolveError::Inconsistent(leaked_id))
    );
}

#[test]
fn dag_batch_resolution_preserves_order_duplicates_and_errors() {
    use pecos_quantum::MeasResolveError;
    let mut circuit = DagCircuit::new();
    let z = circuit.mz(&[0])[0];
    let node = circuit.try_add_gate_auto_wire(Gate::mx(&[1usize])).unwrap();
    let x = circuit.meas_refs(node).unwrap()[0];
    let ids = [x.meas_id, z.meas_id, x.meas_id];
    let expected: Result<Vec<_>, _> = ids.iter().map(|&id| circuit.measured_pauli(id)).collect();
    assert_eq!(circuit.measured_paulis(&ids), expected);
    assert_eq!(circuit.measured_paulis(&[]).unwrap(), []);
    circuit.detector(&[z]).unwrap();
    let annotations = circuit.annotations().to_vec();
    circuit.remove_gate(z.node);
    assert_eq!(circuit.annotations(), annotations);
    let unknown = MeasId::from_raw(99);
    for ids in [[unknown, z.meas_id], [z.meas_id, unknown]] {
        let expected: Result<Vec<_>, _> =
            ids.iter().map(|&id| circuit.measured_pauli(id)).collect();
        assert_eq!(circuit.measured_paulis(&ids), expected);
    }
    assert_eq!(
        circuit.measured_paulis(&[z.meas_id]),
        Err(MeasResolveError::Removed(z.meas_id))
    );
}
