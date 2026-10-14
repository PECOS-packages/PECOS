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

use super::builder::DemBuilder;
use super::dem_sampler::SamplingEngineBuilder;
use super::sampler::DemSamplerBuilder;
use super::types::{DemOutput, DetectorDef};
use crate::fault_tolerance::propagator::{DagFaultAnalyzer, DagFaultInfluenceMap};
use pecos_quantum::DagCircuit;

fn two_measurements() -> DagCircuit {
    let mut circuit = DagCircuit::new();
    circuit.pz(&[0, 1]);
    // Only q0 has a one-qubit gate, so p1 noise flips only measurement 0.
    circuit.x(&[0]);
    circuit.mz(&[0, 1]);
    circuit
}

fn error_text<T, E: std::fmt::Display>(result: Result<T, E>) -> String {
    match result {
        Ok(_) => panic!("expected an error"),
        Err(error) => error.to_string(),
    }
}

#[test]
fn sampler_json_ids_select_output_channels() {
    let circuit = two_measurements();
    let map = DagFaultAnalyzer::new(&circuit)
        .build_influence_map()
        .expect("supported circuit");
    let json = r#"[{"id":1,"records":[0]},{"id":0,"records":[1]}]"#;
    let sampler = DemSamplerBuilder::new(&map)
        .with_noise(0.3, 0.0, 0.0, 0.0)
        .with_detectors_json(json)
        .unwrap()
        .with_observables_json(json)
        .unwrap()
        .build()
        .unwrap();
    let engine = SamplingEngineBuilder::new(&map)
        .with_noise(0.3, 0.0, 0.0, 0.0)
        .with_detectors_json(json)
        .unwrap()
        .with_observables_json(json)
        .unwrap()
        .build()
        .unwrap();
    for dem in [
        sampler.to_detector_error_model(),
        engine.to_detector_error_model(),
    ] {
        let text = dem.to_string();
        assert!(text.contains("D1 L1"), "{text}");
        assert!(!text.contains("D0") && !text.contains("L0"), "{text}");
    }
}

#[test]
fn sampler_json_ids_must_be_unique_and_dense() {
    let circuit = two_measurements();
    let map = DagFaultAnalyzer::new(&circuit)
        .build_influence_map()
        .expect("supported circuit");
    for (json, expected) in [
        (r#"[{"id":0,"records":[0]},{"id":0,"records":[1]}]"#, "id 0"),
        (
            r#"[{"id":2,"records":[0]},{"id":0,"records":[1]}]"#,
            "ids [0, 2]",
        ),
        (r#"[{"id":1,"records":[0]}]"#, "ids [1]"),
    ] {
        for error in [
            error_text(DemSamplerBuilder::new(&map).with_detectors_json(json)),
            error_text(DemSamplerBuilder::new(&map).with_observables_json(json)),
            error_text(SamplingEngineBuilder::new(&map).with_detectors_json(json)),
            error_text(SamplingEngineBuilder::new(&map).with_observables_json(json)),
        ] {
            assert!(error.contains(expected), "{error}");
        }
    }
    for json in ["[]", ""] {
        assert!(
            DemSamplerBuilder::new(&map)
                .with_detectors_json(json)
                .unwrap()
                .with_observables_json(json)
                .unwrap()
                .build()
                .is_ok()
        );
        assert!(
            SamplingEngineBuilder::new(&map)
                .with_detectors_json(json)
                .unwrap()
                .with_observables_json(json)
                .unwrap()
                .build()
                .is_ok()
        );
    }
}

#[test]
fn sampler_stamped_ids_resolve_to_map_positions() {
    let circuit = two_measurements();
    let mut map = DagFaultAnalyzer::new(&circuit)
        .build_influence_map()
        .expect("supported circuit");
    // Non-positional stamps separate identity from record position.
    map.meas_ids = vec![
        pecos_core::MeasId::from_raw(9),
        pecos_core::MeasId::from_raw(4),
    ];
    for json in [
        r#"[{"id":0,"meas_ids":[9]}]"#,
        r#"[{"id":0,"records":[0],"meas_ids":[9]}]"#,
        r#"[{"id":0,"records":[-2],"meas_ids":[9]}]"#,
    ] {
        let mut sampler = DemSamplerBuilder::new(&map).with_noise(0.3, 0.0, 0.0, 0.0);
        let mut engine = SamplingEngineBuilder::new(&map).with_noise(0.3, 0.0, 0.0, 0.0);
        sampler = sampler
            .with_detectors_json(json)
            .unwrap()
            .with_observables_json(json)
            .unwrap();
        engine = engine
            .with_detectors_json(json)
            .unwrap()
            .with_observables_json(json)
            .unwrap();
        for dem in [
            sampler.build().unwrap().to_detector_error_model(),
            engine.build().unwrap().to_detector_error_model(),
        ] {
            assert!(dem.to_string().contains("D0 L0"), "{}", dem.to_string());
        }
    }
    for json in [
        r#"[{"id":0,"records":[1],"meas_ids":[9]}]"#,
        r#"[{"id":0,"records":[0,1,1],"meas_ids":[9,9,4]}]"#,
    ] {
        for error in [
            error_text(
                DemSamplerBuilder::new(&map)
                    .with_detectors_json(json)
                    .unwrap()
                    .build(),
            ),
            error_text(
                DemSamplerBuilder::new(&map)
                    .with_observables_json(json)
                    .unwrap()
                    .build(),
            ),
            error_text(
                SamplingEngineBuilder::new(&map)
                    .with_detectors_json(json)
                    .unwrap()
                    .build(),
            ),
            error_text(
                SamplingEngineBuilder::new(&map)
                    .with_observables_json(json)
                    .unwrap()
                    .build(),
            ),
        ] {
            assert!(
                error.contains(
                    "has both 'records' and 'meas_ids' but they reference different measurements"
                ),
                "{error}"
            );
        }
    }
    // Legacy ids name map positions directly.
    let mut legacy = map;
    legacy.meas_ids.clear();
    let json = r#"[{"id":0,"records":[0],"meas_ids":[0]}]"#;
    let engine = SamplingEngineBuilder::new(&legacy)
        .with_noise(0.3, 0.0, 0.0, 0.0)
        .with_detectors_json(json)
        .unwrap()
        .build()
        .unwrap();
    assert!(engine.to_detector_error_model().to_string().contains("D0"));
}

#[test]
fn raw_sampler_records_are_validated_before_use() {
    let circuit = two_measurements();
    let map = DagFaultAnalyzer::new(&circuit)
        .build_influence_map()
        .expect("supported circuit");
    for record in [-3, 2, 5] {
        let refs = vec![vec![record]];
        for error in [
            error_text(
                DemSamplerBuilder::new(&map)
                    .with_detector_records(refs.clone())
                    .build(),
            ),
            error_text(
                DemSamplerBuilder::new(&map)
                    .with_observable_records(refs.clone())
                    .build(),
            ),
            error_text(
                DemSamplerBuilder::new(&map)
                    .with_detectors(refs.clone(), vec![])
                    .build(),
            ),
            error_text(
                DemSamplerBuilder::new(&map)
                    .with_detectors(vec![], refs.clone())
                    .build(),
            ),
            error_text(
                SamplingEngineBuilder::new(&map)
                    .with_detector_records(refs.clone())
                    .build(),
            ),
            error_text(
                SamplingEngineBuilder::new(&map)
                    .with_observable_records(refs)
                    .build(),
            ),
        ] {
            assert!(
                error.contains(&format!("record offset {record}"))
                    && error.contains("out of range"),
                "{error}"
            );
        }
    }
    for record in [-2, -1, 0, 1] {
        assert!(
            DemSamplerBuilder::new(&map)
                .with_detectors(vec![vec![record]], vec![vec![record]])
                .build()
                .is_ok()
        );
        assert!(
            SamplingEngineBuilder::new(&map)
                .with_detector_records(vec![vec![record]])
                .with_observable_records(vec![vec![record]])
                .build()
                .is_ok()
        );
    }
}

#[test]
fn dual_output_records_are_bounded_in_map_order() {
    let circuit = two_measurements();
    let map = DagFaultAnalyzer::new(&circuit)
        .build_influence_map()
        .expect("supported circuit");
    for index in [2, 5, usize::MAX] {
        let error = error_text(
            DemSamplerBuilder::new(&map)
                .with_dual_output(vec![vec![index]])
                .build(),
        );
        assert!(
            error.contains(&format!("record offset {index}")) && error.contains("out of range"),
            "{error}"
        );
    }
    assert!(
        DemSamplerBuilder::new(&map)
            .with_dual_output(vec![vec![0, 1]])
            .build()
            .is_ok()
    );
    let empty = DagFaultInfluenceMap::with_capacity(0);
    assert!(
        DemSamplerBuilder::new(&empty)
            .with_observable_records(vec![vec![-99, 5]])
            .with_dual_output(vec![vec![usize::MAX]])
            .build()
            .is_ok()
    );
    assert!(
        SamplingEngineBuilder::new(&empty)
            .with_detector_records(vec![vec![-99, 5]])
            .with_observable_records(vec![vec![-99, 5]])
            .build()
            .is_ok()
    );
}

#[test]
fn dem_stab_rejects_out_of_range_user_records() {
    for record in [-3, 2, 5] {
        for error in [
            error_text(
                crate::dem_stab::DemStabSim::builder()
                    .circuit(two_measurements())
                    .detectors(vec![DetectorDef::new(0).with_records([record])])
                    .build(),
            ),
            error_text(
                crate::dem_stab::DemStabSim::builder()
                    .circuit(two_measurements())
                    .observables(vec![DemOutput::new(0).with_records([record])])
                    .build(),
            ),
        ] {
            assert!(
                error.contains("record offset") && error.contains("out of range"),
                "{error}"
            );
        }
    }
}

#[test]
fn sampler_determinism_and_annotations_follow_map_positions() {
    // Annotation observables and explicit detector records name the same map position.
    let mut circuit = DagCircuit::new();
    circuit.pz(&[0, 1]);
    circuit.x(&[0]);
    let named = circuit.mz(&[0]);
    circuit.mz(&[1]);
    let map = DagFaultAnalyzer::new(&circuit)
        .build_influence_map()
        .expect("supported circuit");
    circuit.observable(&named).unwrap();
    let sampler = DemSamplerBuilder::new(&map)
        .with_noise(0.3, 0.0, 0.0, 0.0)
        .with_circuit_annotations(&circuit)
        .unwrap()
        .with_detector_records(vec![vec![0]])
        .build()
        .unwrap();
    let text = sampler.to_detector_error_model().to_string();
    assert!(text.contains("D0 L0"), "{text}");

    // q0 (position 0) is deterministic; q1 (position 1) is random.
    let mut circuit = DagCircuit::new();
    circuit.pz(&[0, 1]);
    circuit.h(&[1]);
    circuit.mz(&[0, 1]);
    let map = crate::fault_tolerance::influence_builder::InfluenceBuilder::new(&circuit)
        .expect("supported circuit")
        .build()
        .unwrap();
    assert!(
        DemSamplerBuilder::new(&map)
            .with_detector_records(vec![vec![0]])
            .build()
            .is_ok()
    );
    let error = error_text(
        DemSamplerBuilder::new(&map)
            .with_detector_records(vec![vec![1]])
            .build(),
    );
    assert!(error.contains("non-deterministic"), "{error}");
}

#[test]
fn sampler_record_setters_replace_pending_json() {
    let circuit = two_measurements();
    let map = DagFaultAnalyzer::new(&circuit)
        .build_influence_map()
        .expect("supported circuit");
    let json = r#"[{"id":0,"records":[1]}]"#;
    let sampler = DemSamplerBuilder::new(&map)
        .with_noise(0.3, 0.0, 0.0, 0.0)
        .with_detectors_json(json)
        .unwrap()
        .with_observables_json(json)
        .unwrap()
        .with_detectors(vec![vec![0]], vec![vec![0]])
        .build()
        .unwrap();
    let engine = SamplingEngineBuilder::new(&map)
        .with_noise(0.3, 0.0, 0.0, 0.0)
        .with_detectors_json(json)
        .unwrap()
        .with_observables_json(json)
        .unwrap()
        .with_detector_records(vec![vec![0]])
        .with_observable_records(vec![vec![0]])
        .build()
        .unwrap();
    for dem in [
        sampler.to_detector_error_model(),
        engine.to_detector_error_model(),
    ] {
        assert!(dem.to_string().contains("D0 L0"));
    }
}

#[test]
fn duplicate_stamped_ids_fail_at_all_build_boundaries() {
    use super::mem_builder::MemBuilder;
    use pecos_core::MeasId;

    // Exercise build-time validation without going through JSON or annotations.
    // Empty maps keep opaque record coordinates, but never ambiguous stamps.
    for measurements in [vec![], vec![(0, 0, 0), (1, 1, 0)]] {
        let mut map = DagFaultInfluenceMap::with_capacity(0);
        map.measurements = measurements;
        map.meas_ids = vec![MeasId::from_raw(9), MeasId::from_raw(9)];
        for error in [
            error_text(DemBuilder::new(&map).try_build()),
            error_text(DemSamplerBuilder::new(&map).build()),
            error_text(SamplingEngineBuilder::new(&map).build()),
            error_text(MemBuilder::new(&map).build()),
        ] {
            assert!(error.contains("duplicate stable MeasId 9"), "{error}");
        }
    }
}
