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

//! Order-free guards captured before removing the measurement-order setters.

use pecos_core::{Gate, MeasId, QubitId};
use pecos_qec::fault_tolerance::dem_builder::{
    DemBuilder, DemSamplerBuilder, MemBuilder, NoiseConfig,
};
use pecos_qec::fault_tolerance::propagator::dag::DagSpacetimeLocation;
use pecos_qec::fault_tolerance::propagator::types::{DetectorId, MeasurementId};
use pecos_qec::fault_tolerance::propagator::{DagFaultAnalyzer, DagFaultInfluenceMap};
use pecos_quantum::DagCircuit;
use pecos_random::PecosRng;

fn noise() -> NoiseConfig {
    NoiseConfig::new(0.19, 0.07, 0.13, 0.03)
}

fn stamped_parity_check() -> DagCircuit {
    let mut dag = DagCircuit::new();
    dag.pz(&[0, 1, 2]);
    dag.cx(&[(0, 2)]);
    dag.cx(&[(1, 2)]);
    // Non-monotone qubits, repeated ancilla, and IDs unrelated to positions.
    // Only q0 has a one-qubit gate: faults distinguish data readout channels.
    for (qubit, id) in [(2, 9), (2, 4), (0, 7), (1, 2)] {
        if id == 4 {
            dag.pz(&[2]);
            dag.cx(&[(0, 2)]);
            dag.cx(&[(1, 2)]);
        }
        if qubit == 0 {
            dag.z(&[0]);
        }
        let mut gate = Gate::mz(&[qubit]);
        gate.meas_ids = vec![MeasId::from_raw(id)].into();
        dag.try_add_gate_auto_wire(gate).unwrap();
    }
    let refs: Vec<_> = [9, 4, 7, 2]
        .map(|id| dag.find_measurement(MeasId::from_raw(id)).unwrap())
        .into();
    dag.detector(&[refs[0], refs[1]]).unwrap();
    dag.detector(&[refs[1], refs[2], refs[3]]).unwrap();
    dag.observable(&[refs[2]]).unwrap();
    dag
}

const DETECTORS: &str = r#"[{"id":0,"records":[0,1]},{"id":1,"records":[1,2,3]}]"#;
const OBSERVABLES: &str = r#"[{"id":0,"records":[2]}]"#;
const STAMPED_DETECTORS: &str = r#"[{"id":0,"meas_ids":[9,4]},{"id":1,"meas_ids":[4,7,2]}]"#;
const STAMPED_OBSERVABLES: &str = r#"[{"id":0,"meas_ids":[7]}]"#;

#[test]
fn emission_records_stamps_and_circuit_builders_have_identical_dems() {
    let dag = stamped_parity_check();
    let map = DagFaultAnalyzer::new(&dag).build_influence_map().unwrap();
    assert_eq!(
        map.measurements.iter().map(|m| m.1).collect::<Vec<_>>(),
        [2, 2, 0, 1]
    );
    assert_eq!(map.meas_ids, [9, 4, 7, 2].map(MeasId::from_raw));
    let expected = DemBuilder::try_from_circuit_with_noise_config(&dag, noise())
        .unwrap()
        .to_string();
    assert!(expected.contains("error("));
    for (detectors, observables) in [
        (DETECTORS, OBSERVABLES),
        (STAMPED_DETECTORS, STAMPED_OBSERVABLES),
    ] {
        let dem = DemBuilder::new(&map)
            .with_noise_config(noise())
            .with_detectors_json(detectors)
            .unwrap()
            .with_observables_json(observables)
            .unwrap()
            .try_build()
            .unwrap();
        let sampler = DemSamplerBuilder::new(&map)
            .with_noise_config(noise())
            .with_detectors_json(detectors)
            .unwrap()
            .with_observables_json(observables)
            .unwrap()
            .build()
            .unwrap();
        assert_eq!(dem.to_string(), expected);
        // The engine accessor exports mechanisms only. Its documented contract
        // asks callers to supply declarations when serializing a complete DEM.
        let mut sampled_dem = sampler.to_detector_error_model();
        sampled_dem.detectors.clone_from(&dem.detectors);
        sampled_dem.observables.clone_from(&dem.observables);
        assert_eq!(sampled_dem.to_string(), expected);
    }
    // A data-channel permutation must be observable despite the parity detector.
    let swapped = DemBuilder::new(&map)
        .with_noise_config(noise())
        .with_detectors_json(DETECTORS)
        .unwrap()
        .with_observables_json(r#"[{"id":0,"records":[3]}]"#)
        .unwrap()
        .try_build()
        .unwrap();
    assert_ne!(swapped.to_string(), expected);
}

fn idless_map() -> DagFaultInfluenceMap {
    let mut map = DagFaultInfluenceMap::with_capacity(2);
    for (node, qubit, x, y, z) in [
        (0, 2, vec![0], vec![0, 1], vec![1]),
        (1, 0, vec![1, 2], vec![2], vec![1]),
    ] {
        map.locations.push(DagSpacetimeLocation::new(
            node,
            vec![QubitId::from(qubit)],
            false,
            &Gate::h(&[qubit]),
        ));
        map.influences.detectors_x.extend(x);
        map.influences.detectors_y.extend(y);
        map.influences.detectors_z.extend(z);
        map.influences.finish_location();
    }
    map.measurements = vec![(2, 2, 0), (3, 0, 0), (4, 2, 0)];
    map.detectors = map
        .measurements
        .iter()
        .map(|&(tick, qubit, basis)| DetectorId::single(MeasurementId { tick, qubit, basis }))
        .collect();
    assert_eq!(map.meas_ids, []);
    map
}

fn surface_like_circuit() -> DagCircuit {
    // Four data qubits on a plaquette, commuting XXXX and ZZZZ checks,
    // two syndrome rounds and final data-Z readout.
    let mut dag = DagCircuit::new();
    dag.pz(&[0, 1, 2, 3]);
    for _ in 0..2 {
        dag.pz(&[4, 5]);
        dag.h(&[5]);
        for q in 0..4 {
            dag.cx(&[(5, q)]);
        }
        for q in 0..4 {
            dag.cx(&[(q, 4)]);
        }
        dag.h(&[5]);
        dag.mz(&[4, 5]);
    }
    dag.mz(&[0, 1, 2, 3]);
    dag
}

fn bits(values: &[bool]) -> String {
    values
        .iter()
        .map(|&value| if value { '1' } else { '0' })
        .collect()
}

// Detector-mode DemSamplerBuilder uses the crate-private SamplingEngineBuilder;
// sample() and DEM export delegate to that engine without another sampling path.
fn capture_outputs(
    map: &DagFaultInfluenceMap,
    detectors: &[Vec<i32>],
    observables: Vec<Vec<i32>>,
) -> [String; 2] {
    let sampler = DemSamplerBuilder::new(map)
        .with_noise_config(noise())
        .with_detectors(detectors.to_vec(), observables)
        .build()
        .unwrap();
    let mem = MemBuilder::new(map)
        .with_noise_config(noise())
        .build()
        .unwrap();
    let mut sampler_rng = PecosRng::seed_from_u64(42);
    let mut mem_rng = PecosRng::seed_from_u64(42);
    let mut rows: [Vec<String>; 2] = std::array::from_fn(|_| Vec::new());
    for _ in 0..16 {
        let (d, o) = sampler.sample(&mut sampler_rng);
        rows[0].push(format!("{}|{}", bits(&d), bits(&o)));
        let (_, d) = mem.sample_with_detectors(detectors, &mut mem_rng);
        rows[1].push(bits(&d));
    }
    rows.map(|rows| rows.join(" "))
}

#[test]
fn idless_map_seeded_pre_removal_golden() {
    let actual = capture_outputs(&idless_map(), &[vec![0], vec![1, 2]], vec![vec![1]]);
    // Pre-removal goldens: 16 shots, seed 42, no measurement order.
    // Rows are detector bits|observable bits, then MEM detector bits.
    // Both detector streams and the observable stream contain nonzero events.
    assert_eq!(
        actual,
        [
            "00|0 00|0 00|0 01|0 00|0 00|0 00|0 01|1 01|1 00|0 10|0 00|0 00|0 00|0 01|0 00|0",
            "00 00 00 00 00 00 01 00 00 00 01 00 00 00 11 00",
        ]
    );
}

#[test]
fn stamped_circuit_seeded_pre_removal_golden() {
    let map = DagFaultAnalyzer::new(&stamped_parity_check())
        .build_influence_map()
        .unwrap();
    let actual = capture_outputs(&map, &[vec![0, 1], vec![1, 2, 3]], vec![vec![2]]);
    // Pre-removal goldens: 16 shots, seed 42, no measurement order.
    // Rows are detector bits|observable bits, then MEM detector bits.
    // Both detector streams and the observable stream contain nonzero events.
    assert_eq!(
        actual,
        [
            "00|0 10|0 00|0 00|1 00|0 00|0 10|0 10|0 01|0 01|0 01|1 01|0 11|0 00|0 10|0 11|1",
            "00 00 00 00 11 01 01 00 01 00 00 11 11 10 10 11",
        ]
    );
}

#[test]
fn surface_like_seeded_pre_removal_golden() {
    let map = DagFaultAnalyzer::new(&surface_like_circuit())
        .build_influence_map()
        .unwrap();
    let actual = capture_outputs(
        &map,
        &[vec![0], vec![0, 2], vec![1, 3], vec![2, 4, 5, 6, 7]],
        vec![vec![4, 5]],
    );
    // Pre-removal goldens: 16 shots, seed 42, no measurement order.
    // Rows are detector bits|observable bits, then MEM detector bits.
    // Both detector streams and the observable stream contain nonzero events.
    assert_eq!(
        actual,
        [
            "1100|0 1010|0 0011|0 0000|1 1111|0 0000|0 0010|0 0110|0 0111|1 1001|0 0100|0 0000|0 1011|1 0010|0 1110|1 0011|0",
            "0011 0011 0001 0110 1011 0000 0000 1101 1111 0101 1100 0010 1011 0001 0001 0001",
        ]
    );
}
