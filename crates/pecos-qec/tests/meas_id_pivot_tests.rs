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
        .with_circuit_annotations()
        .expect("annotations resolve")
        .build()
        .expect("circuit is replayable");
    let map_scr = InfluenceBuilder::new(&scrambled)
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

/// A `gate_mut` edit can leave two measurements holding one id. The sampler's
/// annotation resolution refuses the whole map instead of silently binding to
/// the first holder.
#[test]
fn sampler_annotations_refuse_a_map_with_duplicate_ids() {
    use pecos_qec::fault_tolerance::propagator::DagFaultAnalyzer;

    let mut dag = DagCircuit::new();
    dag.pz(&[0, 1]);
    let a = dag.mz(&[0]);
    let b = dag.mz(&[1]);
    dag.gate_mut(b[0].node).expect("gate exists").meas_ids[0] = a[0].meas_id;

    let map = DagFaultAnalyzer::new(&dag).build_influence_map();
    let err = DemSamplerBuilder::new(&map)
        .with_uniform_noise(0.01)
        .raw_measurements()
        .with_circuit_annotations(&dag)
        .map(|_| ())
        .expect_err("two measurements hold one id");
    assert!(matches!(
        err,
        pecos_qec::fault_tolerance::dem_builder::DetectorValidationError::InvalidMetadata { .. }
    ));
}

/// `measurement_order` is a legacy escape hatch for id-less circuits; on a
/// stamped-id circuit its qubit-occurrence heuristic silently mis-binds, so
/// the combination is refused outright.
#[test]
fn measurement_order_is_refused_on_a_stamped_id_circuit() {
    let dag = parity_check_with_ids([9, 4, 7]);
    let map = InfluenceBuilder::new(&dag)
        .with_circuit_annotations()
        .expect("annotations resolve")
        .build()
        .expect("circuit is replayable");

    let err = DemSamplerBuilder::new(&map)
        .with_uniform_noise(0.01)
        .raw_measurements()
        .with_measurement_order(vec![0, 1, 2])
        .build()
        .map(|_| ())
        .expect_err("stamped ids already define the mapping");
    assert!(matches!(
        err,
        pecos_qec::fault_tolerance::dem_builder::DetectorValidationError::InvalidMetadata { .. }
    ));
}

/// A detector can reference exactly one measurement of a batched gate --
/// previously inexpressible, because a node id covered the whole batch. The
/// annotation Pauli must touch only that measurement's qubit.
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
        ann.pauli,
        pecos_core::PauliString::zs(&[1usize]),
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
        pauli: pecos_core::PauliString::zs(&[0usize]),
        kind: AnnotationKind::Observable {
            measurement_ids: vec![ms[0].meas_id],
        },
        label: None,
    });

    let err = InfluenceBuilder::new(&dag)
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
            AnnotationKind::TrackedPauli => panic!("unexpected tracked Pauli"),
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
