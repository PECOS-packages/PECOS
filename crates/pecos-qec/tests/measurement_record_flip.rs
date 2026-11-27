//! Record-only measurement noise and explicit boundaries of supporting consumers.

use pecos_core::{Gate, PauliString, QubitId, gate_type::GateType};
use pecos_qec::fault_tolerance::InfluenceBuilder;
use pecos_qec::fault_tolerance::dem_builder::{
    DemBuilder, DemSampler, DemSamplerBuilder, DetectorErrorModel, DirectSourceFamily,
    MeasurementNoiseKind, MemBuilder, NoiseConfig, PerGateTypeNoise, SamplingEngine,
};
use pecos_qec::fault_tolerance::fault_sampler::{StochasticNoiseParams, build_fault_table};
use pecos_qec::fault_tolerance::lookup_decoder::LookupDecoder;
use pecos_qec::fault_tolerance::propagator::{DagFaultAnalyzer, DagFaultInfluenceMap};
use pecos_quantum::{DagCircuit, TickCircuit};

fn noise(p: f64) -> NoiseConfig {
    NoiseConfig::new(0.0, 0.0, p, 0.0).set_measurement_noise_kind(MeasurementNoiseKind::RecordFlip)
}

fn repeated(kind: GateType, observable: bool) -> TickCircuit {
    let mut circuit = TickCircuit::new();
    if kind == GateType::MX {
        circuit.tick().px(&[0]);
    } else {
        circuit.tick().pz(&[0]);
    }
    let first = match kind {
        GateType::MX => circuit.tick().mx(&[0]),
        GateType::MPZ => circuit.tick().mpz(&[0]),
        _ => circuit.tick().mz(&[0]),
    };
    let second = if kind == GateType::MX {
        circuit.tick().mx(&[0])
    } else {
        circuit.tick().mz(&[0])
    };
    circuit.detector(&first).unwrap();
    circuit.detector(&second).unwrap();
    if observable {
        circuit.observable(&second).unwrap();
    }
    circuit
}

fn cross_qubit() -> TickCircuit {
    let mut circuit = TickCircuit::new();
    circuit.tick().pz(&[0, 1]);
    let first = circuit.tick().mz(&[0]);
    circuit.tick().cx(&[(0, 1)]);
    let second = circuit.tick().mz(&[1]);
    circuit.detector(&first).unwrap();
    circuit.detector(&second).unwrap();
    circuit
}

fn map(circuit: &TickCircuit) -> DagFaultInfluenceMap {
    DagFaultAnalyzer::new(&DagCircuit::try_from(circuit).unwrap())
        .build_influence_map()
        .unwrap()
}

fn assert_mechanisms(dem: &DetectorErrorModel, expected: &[(f64, Vec<u32>, Vec<u32>)]) {
    let mut actual = dem.to_mechanisms().0;
    actual.sort_by(|a, b| (&a.1, &a.2).cmp(&(&b.1, &b.2)));
    let mut expected = expected.to_vec();
    expected.sort_by(|a, b| (&a.1, &a.2).cmp(&(&b.1, &b.2)));
    assert_eq!(actual.len(), expected.len(), "{actual:?}");
    for (actual, expected) in actual.iter().zip(&expected) {
        assert_eq!((&actual.1, &actual.2), (&expected.1, &expected.2));
        assert!(
            (actual.0 - expected.0).abs() < 1e-12,
            "{actual:?} != {expected:?}"
        );
    }
}

fn check_both(circuit: &TickCircuit, state_first: &[u32]) {
    for kind in [
        MeasurementNoiseKind::RecordFlip,
        MeasurementNoiseKind::StateFlip,
    ] {
        let dem = DemBuilder::try_from_tick_circuit_with_noise_config(
            circuit,
            noise(0.1).set_measurement_noise_kind(kind),
        )
        .unwrap();
        let first = if kind == MeasurementNoiseKind::RecordFlip {
            vec![0]
        } else {
            state_first.to_vec()
        };
        assert_mechanisms(&dem, &[(0.1, first, vec![]), (0.1, vec![1], vec![])]);
    }
}

#[test]
fn t1_repeated_mz() {
    check_both(&repeated(GateType::MZ, false), &[0, 1]);
}
#[test]
fn t2_cross_qubit() {
    check_both(&cross_qubit(), &[0, 1]);
}
#[test]
fn t3_repeated_mx() {
    check_both(&repeated(GateType::MX, false), &[0, 1]);
}
#[test]
fn t4_mpz() {
    check_both(&repeated(GateType::MPZ, false), &[0]);
}

#[test]
fn t5a_annotation_observable() {
    let dem = DemBuilder::try_from_tick_circuit_with_noise_config(
        &repeated(GateType::MZ, true),
        noise(0.1),
    )
    .unwrap();
    assert_mechanisms(&dem, &[(0.1, vec![0], vec![]), (0.1, vec![1], vec![0])]);
}

#[test]
fn t5b_json_observable() {
    let influence = map(&repeated(GateType::MZ, false));
    let dem = DemBuilder::new(&influence)
        .with_noise_config(noise(0.1))
        .with_detectors_json(r#"[{"id":0,"records":[0]},{"id":1,"records":[1]}]"#)
        .unwrap()
        .with_observables_json(r#"[{"id":0,"records":[1]}]"#)
        .unwrap()
        .try_build()
        .unwrap();
    assert_mechanisms(&dem, &[(0.1, vec![0], vec![]), (0.1, vec![1], vec![0])]);
}

#[test]
fn t6_per_qubit_kind_from_base() {
    let mut circuit = TickCircuit::new();
    circuit.tick().pz(&[0, 1]);
    circuit.tick().mz(&[0, 1]);
    circuit.tick().mz(&[0, 1]);
    let influence = map(&circuit);
    let config = PerGateTypeNoise::from_base_noise(noise(0.0))
        .with_measurement_rate(QubitId(0), 0.1)
        .with_measurement_rate(QubitId(1), 0.2);
    let dem = DemBuilder::new(&influence).with_per_gate_noise(config)
        .with_detectors_json(r#"[{"id":0,"records":[0]},{"id":1,"records":[1]},{"id":2,"records":[2]},{"id":3,"records":[3]}]"#)
        .unwrap().try_build().unwrap();
    assert_mechanisms(
        &dem,
        &[
            (0.1, vec![0], vec![]),
            (0.2, vec![1], vec![]),
            (0.1, vec![2], vec![]),
            (0.2, vec![3], vec![]),
        ],
    );
}

#[test]
fn t7_missing_record() {
    let mut influence = map(&repeated(GateType::MZ, false));
    let node = influence.measurements[0].0;
    influence.measurements[0].0 = influence
        .locations
        .iter()
        .find(|loc| loc.gate_type == GateType::PZ)
        .unwrap()
        .node;
    let error = DemBuilder::new(&influence)
        .with_noise_config(noise(0.1))
        .try_build()
        .unwrap_err()
        .to_string();
    assert!(
        error.contains(&format!(
            "no measurement record for node {node} qubit 0: found 0"
        )),
        "{error}"
    );
    assert!(
        error.contains("use StateFlip or a circuit-built map"),
        "{error}"
    );
}

#[test]
fn empty_and_duplicate_records_error() {
    for duplicate in [false, true] {
        let mut influence = map(&repeated(GateType::MZ, false));
        if duplicate {
            influence.measurements[1] = influence.measurements[0];
        } else {
            influence.measurements.clear();
            influence.meas_ids.clear();
        }
        let error = DemBuilder::new(&influence)
            .with_noise_config(noise(0.1))
            .try_build()
            .unwrap_err()
            .to_string();
        assert!(
            error.contains(if duplicate { "found 2" } else { "found 0" }),
            "{error}"
        );
        assert!(
            error.contains("use StateFlip or a circuit-built map"),
            "{error}"
        );
    }
}

#[test]
fn observable_definition_required() {
    let circuit = DagCircuit::try_from(&repeated(GateType::MZ, true)).unwrap();
    let influence = InfluenceBuilder::new(&circuit)
        .unwrap()
        .with_circuit_annotations()
        .unwrap()
        .build()
        .unwrap();
    let error = DemBuilder::new(&influence)
        .with_noise_config(noise(0.1))
        .try_build()
        .unwrap_err()
        .to_string();
    assert!(error.contains("observable 0"), "{error}");
    assert!(
        error.contains("supply observable definitions or use StateFlip"),
        "{error}"
    );
    DemBuilder::new(&influence)
        .with_noise_config(noise(0.0))
        .try_build()
        .unwrap();
}

#[test]
fn record_flip_provenance_and_no_tracked_pauli_effect() {
    let mut circuit = repeated(GateType::MZ, false);
    circuit.tracked_pauli(PauliString::zs(&[0]));
    let dem = DemBuilder::try_from_tick_circuit_with_noise_config(&circuit, noise(0.1)).unwrap();
    assert_mechanisms(&dem, &[(0.1, vec![0], vec![]), (0.1, vec![1], vec![])]);
    for contribution in dem.contributions() {
        assert_eq!(
            contribution.direct_source_family,
            Some(DirectSourceFamily::MeasurementRecordFlip)
        );
        assert!(contribution.paulis.is_empty());
        assert!(contribution.effect.tracked_paulis.is_empty());
        assert_eq!(contribution.location_indices.len(), 1);
        assert_eq!(contribution.source_gate_types.as_slice(), &[GateType::MZ]);
        assert_eq!(contribution.source_before_flags.as_slice(), &[true]);
    }
    let state = DemBuilder::try_from_tick_circuit_with_noise_config(
        &circuit,
        noise(0.1).set_measurement_noise_kind(MeasurementNoiseKind::StateFlip),
    )
    .unwrap();
    assert_eq!(state.contributions().len(), 2);
    for contribution in state.contributions() {
        assert_eq!(contribution.effect.tracked_paulis.as_slice(), &[0]);
    }
}

#[test]
fn repeated_incidence_cancels_and_unused_records_are_skipped() {
    let influence = map(&repeated(GateType::MZ, false));
    let dem = DemBuilder::new(&influence)
        .with_noise_config(noise(0.1))
        .with_detectors_json(r#"[{"id":0,"records":[0,0]},{"id":1,"records":[1]}]"#)
        .unwrap()
        .with_observables_json(r#"[{"id":0,"records":[0,0,1]}]"#)
        .unwrap()
        .try_build()
        .unwrap();
    assert_mechanisms(&dem, &[(0.1, vec![1], vec![0])]);
}

#[test]
fn measure_free_record_flip() {
    let mut circuit = TickCircuit::new();
    circuit.tick().pz(&[0]);
    circuit
        .tick()
        .try_add_gate(Gate::simple(GateType::MeasureFree, vec![QubitId(0)]))
        .unwrap();
    let influence = map(&circuit);
    let dem = DemBuilder::new(&influence)
        .with_noise_config(noise(0.1))
        .with_detectors_json(r#"[{"id":0,"records":[0]}]"#)
        .unwrap()
        .try_build()
        .unwrap();
    assert_mechanisms(&dem, &[(0.1, vec![0], vec![])]);
}

#[test]
fn t8_fault_table_cross_implementation() {
    for circuit in [
        repeated(GateType::MZ, false),
        cross_qubit(),
        repeated(GateType::MX, false),
    ] {
        let dem =
            DemBuilder::try_from_tick_circuit_with_noise_config(&circuit, noise(0.1)).unwrap();
        let table = build_fault_table(
            &circuit,
            &StochasticNoiseParams {
                p1: 0.0,
                p2: 0.0,
                p_meas: 0.1,
                p_prep: 0.0,
            },
        )
        .unwrap();
        let expected: Vec<_> = table
            .iter()
            .filter(|fault| fault.probability > 0.0)
            .map(|fault| {
                assert_eq!(fault.alternatives.len(), 1);
                (
                    fault.probability,
                    fault.alternatives[0]
                        .iter()
                        .map(|&d| u32::try_from(d).unwrap())
                        .collect(),
                    vec![],
                )
            })
            .collect();
        assert_mechanisms(&dem, &expected);
    }
}

fn assert_rejected<T, E: std::fmt::Display>(result: Result<T, E>, consumer: &str) {
    let error = result
        .err()
        .expect("positive record-flip noise must be rejected")
        .to_string();
    assert!(error.contains(&format!("measurement_noise_kind = RecordFlip is not supported by {consumer}; use DemBuilder or StateFlip")), "{error}");
}

#[test]
fn rejects_mem_builder() {
    let influence = map(&repeated(GateType::MZ, false));
    assert_rejected(
        MemBuilder::new(&influence)
            .with_noise_config(noise(0.1))
            .build(),
        "MemBuilder",
    );
    MemBuilder::new(&influence)
        .with_noise_config(noise(0.0))
        .build()
        .unwrap();
}

#[test]
fn rejects_sampling_engine_from_map() {
    let influence = map(&repeated(GateType::MZ, false));
    let probabilities = vec![0.0; influence.locations.len()];
    assert_rejected(
        SamplingEngine::from_influence_map(&influence, &probabilities, &noise(0.1)),
        "SamplingEngine::from_influence_map",
    );
    SamplingEngine::from_influence_map(&influence, &probabilities, &noise(0.0)).unwrap();
}

#[test]
fn rejects_dem_sampler_builder() {
    let influence = map(&repeated(GateType::MZ, false));
    assert_rejected(
        DemSamplerBuilder::new(&influence)
            .with_noise_config(noise(0.1))
            .raw_measurements()
            .build(),
        "DemSamplerBuilder",
    );
    assert_rejected(
        DemSamplerBuilder::new(&influence)
            .with_noise_config(noise(0.1))
            .with_detector_records(vec![vec![0], vec![1]])
            .build(),
        "DemSamplerBuilder",
    );
    let config =
        PerGateTypeNoise::from_base_noise(noise(0.0)).with_measurement_rate(QubitId(99), 0.2);
    assert_rejected(
        DemSamplerBuilder::new(&influence)
            .with_per_gate_noise(config)
            .with_detector_records(vec![vec![0]])
            .build(),
        "DemSamplerBuilder",
    );
    DemSamplerBuilder::new(&influence)
        .with_noise_config(noise(0.0))
        .build()
        .unwrap();
}

#[test]
fn rejects_dem_sampler_from_tick_circuit() {
    let circuit = repeated(GateType::MZ, false);
    assert_rejected(
        DemSampler::from_tick_circuit(&circuit, &noise(0.1)),
        "DemSampler::from_tick_circuit",
    );
    DemSampler::from_tick_circuit(&circuit, &noise(0.0)).unwrap();
}

#[test]
fn rejects_dem_sampler_from_circuit() {
    let circuit = DagCircuit::try_from(&repeated(GateType::MZ, false)).unwrap();
    assert_rejected(
        DemSampler::from_circuit(&circuit, &noise(0.1)),
        "DemSampler::from_circuit",
    );
    DemSampler::from_circuit(&circuit, &noise(0.0)).unwrap();
}

#[test]
fn rejects_lookup_decoder() {
    let influence = map(&repeated(GateType::MZ, false));
    assert_rejected(
        LookupDecoder::build(&influence, &noise(0.1), 1),
        "LookupDecoder",
    );
    LookupDecoder::build(&influence, &noise(0.0), 1).unwrap();
}

#[test]
fn rejects_dem_build_facade() {
    let circuit = repeated(GateType::MZ, false);
    assert_rejected(
        DetectorErrorModel::builder()
            .with_circuit(circuit.clone())
            .with_noise(noise(0.1))
            .build(),
        "DemBuildBuilder",
    );
    DetectorErrorModel::builder()
        .with_circuit(circuit)
        .with_noise(noise(0.0))
        .build()
        .unwrap();
}

#[test]
fn facade_delegation_rejects() {
    let circuit = DagCircuit::try_from(&repeated(GateType::MZ, false)).unwrap();
    assert_rejected(
        pecos_qec::mem_stab::MemStabSim::builder()
            .circuit(circuit.clone())
            .noise(noise(0.1))
            .build(),
        "MemBuilder",
    );
    assert_rejected(
        pecos_qec::dem_stab::DemStabSim::builder()
            .circuit(circuit)
            .noise(noise(0.1))
            .build(),
        "DemSamplerBuilder",
    );
}

#[test]
#[cfg(feature = "neo")]
fn rejects_neo_channel() {
    for config in [
        PerGateTypeNoise::from_base_noise(noise(0.1)),
        PerGateTypeNoise::from_base_noise(noise(0.0)).with_measurement_rate(QubitId(99), 0.2),
    ] {
        assert_rejected(config.to_neo_channel(), "PerGateTypeNoise::to_neo_channel");
    }
    PerGateTypeNoise::from_base_noise(noise(0.0))
        .to_neo_channel()
        .unwrap();
}

#[test]
fn raw_location_measurement_rates_are_rejected() {
    let influence = map(&repeated(GateType::MZ, false));
    let probabilities: Vec<_> = influence
        .locations
        .iter()
        .map(|loc| {
            if loc.gate_type.consumes_measurement_record() && loc.before {
                0.1
            } else {
                0.0
            }
        })
        .collect();
    assert_rejected(
        SamplingEngine::from_influence_map(&influence, &probabilities, &noise(0.0)),
        "SamplingEngine::from_influence_map",
    );
}

#[test]
fn conflicting_builder_kinds_rejected_in_both_orders() {
    let influence = map(&repeated(GateType::MZ, false));
    for scalar_first in [false, true] {
        for (scalar, per_gate) in [
            (
                noise(0.0),
                PerGateTypeNoise::from_base_noise(NoiseConfig::uniform(0.0)),
            ),
            (
                NoiseConfig::uniform(0.0),
                PerGateTypeNoise::from_base_noise(noise(0.0)),
            ),
        ] {
            let builder = DemBuilder::new(&influence);
            let builder = if scalar_first {
                builder
                    .with_noise_config(scalar.clone())
                    .with_per_gate_noise(per_gate.clone())
            } else {
                builder
                    .with_per_gate_noise(per_gate.clone())
                    .with_noise_config(scalar.clone())
            };
            for error in [
                builder.build().unwrap_err(),
                builder.try_build().unwrap_err(),
            ] {
                assert!(
                    error
                        .to_string()
                        .contains("conflicting measurement_noise_kind"),
                    "{error}"
                );
                assert!(
                    error
                        .to_string()
                        .contains("set the same measurement_noise_kind"),
                    "{error}"
                );
            }
            let sampler = DemSamplerBuilder::new(&influence);
            let sampler = if scalar_first {
                sampler
                    .with_noise_config(scalar)
                    .with_per_gate_noise(per_gate)
            } else {
                sampler
                    .with_per_gate_noise(per_gate)
                    .with_noise_config(scalar)
            };
            let error = sampler
                .with_detector_records(vec![vec![0], vec![1]])
                .build()
                .unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("conflicting measurement_noise_kind"),
                "{error}"
            );
        }
        let builder = DemBuilder::new(&influence);
        let cfg = PerGateTypeNoise::from_base_noise(noise(0.1));
        let builder = if scalar_first {
            builder
                .with_noise_config(noise(0.1))
                .with_per_gate_noise(cfg)
        } else {
            builder
                .with_per_gate_noise(cfg)
                .with_noise_config(noise(0.1))
        };
        let dem = builder
            .with_detectors_json(r#"[{"id":0,"records":[0]},{"id":1,"records":[1]}]"#)
            .unwrap()
            .try_build()
            .unwrap();
        assert_mechanisms(&dem, &[(0.1, vec![0], vec![]), (0.1, vec![1], vec![])]);
    }
}

#[test]
fn discovered_detector_index_space_is_rejected() {
    let mut circuit = DagCircuit::new();
    circuit.pz(&[0]);
    circuit.h(&[0]);
    circuit.mz(&[0]);
    circuit.mz(&[0]);
    let influence = InfluenceBuilder::new(&circuit).unwrap().build().unwrap();
    assert_eq!(influence.measurements.len(), 2);
    assert_eq!(influence.detectors.len(), 1);
    assert_eq!(influence.detectors[0].measurements.len(), 2);
    for error in [
        DemBuilder::new(&influence)
            .with_noise_config(noise(0.1))
            .build()
            .unwrap_err(),
        DemBuilder::new(&influence)
            .with_noise_config(noise(0.1))
            .try_build()
            .unwrap_err(),
    ] {
        assert!(
            error
                .to_string()
                .contains("detector basis differs from its measurement records"),
            "{error}"
        );
        assert!(
            error
                .to_string()
                .contains("use DagFaultAnalyzer or a circuit-built DemBuilder"),
            "{error}"
        );
    }
    // A discovered map whose checks ARE individual records needs no rejection.
    let deterministic = DagCircuit::try_from(&repeated(GateType::MZ, false)).unwrap();
    let influence = InfluenceBuilder::new(&deterministic)
        .unwrap()
        .build()
        .unwrap();
    let dem = DemBuilder::new(&influence)
        .with_noise_config(noise(0.1))
        .with_detectors_json(r#"[{"id":0,"records":[0]},{"id":1,"records":[1]}]"#)
        .unwrap()
        .try_build()
        .unwrap();
    assert_mechanisms(&dem, &[(0.1, vec![0], vec![]), (0.1, vec![1], vec![])]);
}

#[test]
fn record_flip_build_validates_detector_and_observable_refs() {
    let influence = map(&repeated(GateType::MZ, false));
    for observable in [false, true] {
        for references in [r#""records":[2]"#, r#""meas_ids":[999]"#] {
            let json = format!(r#"[{{"id":0,{references}}}]"#);
            let builder = DemBuilder::new(&influence).with_noise_config(noise(0.1));
            let builder = if observable {
                builder.with_observables_json(&json)
            } else {
                builder.with_detectors_json(&json)
            }
            .unwrap();
            let error = builder.build().unwrap_err().to_string();
            assert_eq!(error, builder.try_build().unwrap_err().to_string());
            assert!(
                error.contains(if references.contains("records") {
                    "out of range"
                } else {
                    "not present"
                }),
                "{error}"
            );
        }
    }
}

#[test]
fn unmeasured_qubit_rate_does_not_require_observable_definition() {
    let circuit = DagCircuit::try_from(&repeated(GateType::MZ, true)).unwrap();
    let influence = InfluenceBuilder::new(&circuit)
        .unwrap()
        .with_circuit_annotations()
        .unwrap()
        .build()
        .unwrap();
    let cfg = PerGateTypeNoise::from_base_noise(noise(0.0)).with_measurement_rate(QubitId(99), 0.2);
    let builder = DemBuilder::new(&influence).with_per_gate_noise(cfg);
    assert_mechanisms(&builder.build().unwrap(), &[]);
    assert_mechanisms(&builder.try_build().unwrap(), &[]);
}

#[test]
fn batched_measurements_discriminate_qubits_at_one_node() {
    let mut tick = TickCircuit::new();
    tick.tick().pz(&[0, 1]);
    let records = tick.tick().mz(&[0, 1]);
    for record in records {
        tick.detector(&[record]).unwrap();
    }
    let split = map(&tick);
    assert_ne!(split.measurements[0].0, split.measurements[1].0);
    let dem = DemBuilder::try_from_tick_circuit_with_noise_config(&tick, noise(0.1)).unwrap();
    assert_mechanisms(&dem, &[(0.1, vec![0], vec![]), (0.1, vec![1], vec![])]);
    // Tick-to-DAG conversion splits instances. The direct DAG analyzer also
    // accepts a genuine multi-operand node, where the qubit key is essential.
    let mut dag = DagCircuit::new();
    dag.add_gate_auto_wire(Gate::pz(&[0, 1]));
    dag.add_gate_auto_wire(Gate::mz(&[0, 1]));
    let influence = DagFaultAnalyzer::new(&dag).build_influence_map().unwrap();
    assert_eq!(influence.measurements[0].0, influence.measurements[1].0);
    assert_ne!(influence.measurements[0].1, influence.measurements[1].1);
    let cfg = PerGateTypeNoise::from_base_noise(noise(0.0))
        .with_measurement_rate(QubitId(0), 0.1)
        .with_measurement_rate(QubitId(1), 0.2);
    let dem = DemBuilder::new(&influence)
        .with_per_gate_noise(cfg)
        .with_detectors_json(r#"[{"id":0,"records":[0]},{"id":1,"records":[1]}]"#)
        .unwrap()
        .try_build()
        .unwrap();
    assert_mechanisms(&dem, &[(0.1, vec![0], vec![]), (0.2, vec![1], vec![])]);
}

#[test]
fn stamped_ids_control_detector_and_observable_incidence() {
    let mut dag = DagCircuit::new();
    dag.pz(&[0]);
    for id in [91, 17] {
        let mut gate = Gate::mz(&[0]);
        gate.meas_ids = smallvec::smallvec![pecos_core::MeasId::from_raw(id)];
        dag.add_gate_auto_wire(gate);
    }
    let influence = DagFaultAnalyzer::new(&dag).build_influence_map().unwrap();
    let dem = DemBuilder::new(&influence)
        .with_noise_config(noise(0.1))
        .with_detectors_json(r#"[{"id":0,"meas_ids":[91]},{"id":1,"meas_ids":[17]}]"#)
        .unwrap()
        .with_observables_json(r#"[{"id":0,"meas_ids":[17]}]"#)
        .unwrap()
        .try_build()
        .unwrap();
    assert_mechanisms(&dem, &[(0.1, vec![0], vec![]), (0.1, vec![1], vec![0])]);
}

#[test]
fn one_record_can_flip_three_detectors() {
    let influence = map(&repeated(GateType::MZ, false));
    let dem = DemBuilder::new(&influence)
        .with_noise_config(noise(0.1))
        .with_detectors_json(
            r#"[{"id":0,"records":[0]},{"id":1,"records":[0]},{"id":2,"records":[0]}]"#,
        )
        .unwrap()
        .try_build()
        .unwrap();
    assert_mechanisms(&dem, &[(0.1, vec![0, 1, 2], vec![])]);
}

#[test]
fn state_flip_t1_t2_golden_text() {
    // In both circuits, the first physical X reaches D0 and D1; the second
    // reaches D1 alone. Each independent source has p=0.1. Two detector
    // declarations precede the lexicographically ordered mechanisms, with no
    // observable or trailing newline.
    let expected = "detector D0\ndetector D1\nerror(0.1) D0 D1\nerror(0.1) D1";
    for circuit in [repeated(GateType::MZ, false), cross_qubit()] {
        let dem = DemBuilder::try_from_tick_circuit_with_noise_config(
            &circuit,
            NoiseConfig::new(0.0, 0.0, 0.1, 0.0),
        )
        .unwrap();
        assert_eq!(dem.to_string(), expected);
    }
}

#[test]
fn record_flip_coexists_with_physical_measurement_crosstalk() {
    use pecos_qec::fault_tolerance::dem_builder::{
        MeasurementCrosstalkDemMode, MeasurementCrosstalkTransitionModel,
    };
    let mut circuit = TickCircuit::new();
    circuit.tick().pz(&[0]);
    circuit
        .tick()
        .try_add_gate(Gate::meas_crosstalk_local_payload(&[QubitId(0)]))
        .unwrap();
    let first = circuit.tick().mz(&[0]);
    let second = circuit.tick().mz(&[0]);
    circuit.detector(&first).unwrap();
    circuit.detector(&second).unwrap();
    let cfg = noise(0.1)
        .set_measurement_crosstalk_local_rate(0.25)
        .set_measurement_crosstalk_transition_model(MeasurementCrosstalkTransitionModel::bit_flip(
            1.0, 1.0,
        ))
        .set_measurement_crosstalk_dem_mode(MeasurementCrosstalkDemMode::ExactDeterministic);
    let dem = DemBuilder::try_from_tick_circuit_with_noise_config(&circuit, cfg).unwrap();
    assert_mechanisms(
        &dem,
        &[
            (0.1, vec![0], vec![]),
            (0.1, vec![1], vec![]),
            (0.25, vec![0, 1], vec![]),
        ],
    );
    let crosstalk = dem
        .contributions()
        .iter()
        .find(|contribution| {
            contribution.direct_source_family == Some(DirectSourceFamily::MeasurementCrosstalk)
        })
        .unwrap();
    assert_eq!(
        crosstalk.paulis.as_slice(),
        &[pecos_qec::fault_tolerance::propagator::Pauli::X]
    );
}

#[test]
fn raw_probability_length_error_precedes_kind_rejection() {
    let influence = map(&repeated(GateType::MZ, false));
    let error = SamplingEngine::from_influence_map(&influence, &[], &noise(0.1)).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("per-location probability length mismatch"),
        "{error}"
    );
}

#[test]
fn scalar_noise_uses_per_gate_kind_in_both_orders() {
    let influence = map(&repeated(GateType::MZ, false));
    for previous in [
        None,
        Some(MeasurementNoiseKind::StateFlip),
        Some(MeasurementNoiseKind::RecordFlip),
    ] {
        for kind in [
            MeasurementNoiseKind::StateFlip,
            MeasurementNoiseKind::RecordFlip,
        ] {
            for scalar_first in [false, true] {
                let config =
                    PerGateTypeNoise::from_base_noise(noise(0.1).set_measurement_noise_kind(kind));
                let mut builder = DemBuilder::new(&influence);
                if let Some(previous) = previous {
                    builder = builder.with_noise_config(
                        NoiseConfig::uniform(0.5).set_measurement_noise_kind(previous),
                    );
                }
                let builder = if scalar_first {
                    builder
                        .with_noise(0.0, 0.0, 0.1, 0.0)
                        .with_per_gate_noise(config)
                } else {
                    builder
                        .with_per_gate_noise(config)
                        .with_noise(0.0, 0.0, 0.1, 0.0)
                }
                .with_detectors_json(r#"[{"id":0,"records":[0]},{"id":1,"records":[1]}]"#)
                .unwrap();
                let first = if kind == MeasurementNoiseKind::StateFlip {
                    vec![0, 1]
                } else {
                    vec![0]
                };
                for dem in [builder.build().unwrap(), builder.try_build().unwrap()] {
                    assert_mechanisms(
                        &dem,
                        &[(0.1, first.clone(), vec![]), (0.1, vec![1], vec![])],
                    );
                }
                for p in [0.0, 0.1] {
                    let config = PerGateTypeNoise::from_base_noise(
                        noise(p).set_measurement_noise_kind(kind),
                    );
                    let mut sampler = DemSamplerBuilder::new(&influence);
                    if let Some(previous) = previous {
                        sampler = sampler.with_noise_config(
                            NoiseConfig::uniform(0.5).set_measurement_noise_kind(previous),
                        );
                    }
                    let sampler = if scalar_first {
                        sampler
                            .with_noise(0.0, 0.0, p, 0.0)
                            .with_per_gate_noise(config)
                    } else {
                        sampler
                            .with_per_gate_noise(config)
                            .with_noise(0.0, 0.0, p, 0.0)
                    };
                    let result = sampler
                        .with_detector_records(vec![vec![0], vec![1]])
                        .build();
                    if p > 0.0 && kind == MeasurementNoiseKind::RecordFlip {
                        assert_rejected(result, "DemSamplerBuilder");
                    } else {
                        result.unwrap();
                    }
                }
            }
        }
    }
}

#[test]
fn scalar_noise_replaces_explicit_kind() {
    let influence = map(&repeated(GateType::MZ, false));
    for scalar_first in [false, true] {
        let builder = DemBuilder::new(&influence);
        let builder = if scalar_first {
            builder
                .with_noise(0.0, 0.0, 0.1, 0.0)
                .with_noise_config(noise(0.1))
        } else {
            builder
                .with_noise_config(noise(0.1))
                .with_noise(0.0, 0.0, 0.1, 0.0)
        };
        let dem = builder
            .with_detectors_json(r#"[{"id":0,"records":[0]},{"id":1,"records":[1]}]"#)
            .unwrap()
            .try_build()
            .unwrap();
        let first = if scalar_first { vec![0] } else { vec![0, 1] };
        assert_mechanisms(&dem, &[(0.1, first, vec![]), (0.1, vec![1], vec![])]);
        let sampler = DemSamplerBuilder::new(&influence);
        let sampler = if scalar_first {
            sampler
                .with_noise(0.0, 0.0, 0.1, 0.0)
                .with_noise_config(noise(0.1))
        } else {
            sampler
                .with_noise_config(noise(0.1))
                .with_noise(0.0, 0.0, 0.1, 0.0)
        };
        let result = sampler
            .with_detector_records(vec![vec![0], vec![1]])
            .build();
        if scalar_first {
            assert_rejected(result, "DemSamplerBuilder");
        } else {
            result.unwrap();
        }
    }
}

#[test]
fn record_flip_rejects_unknown_output_kinds() {
    for tracked_pauli in [false, true] {
        let mut circuit = repeated(GateType::MZ, !tracked_pauli);
        if tracked_pauli {
            circuit.tracked_pauli(PauliString::zs(&[0]));
        }
        let dag = DagCircuit::try_from(&circuit).unwrap();
        let outputs = InfluenceBuilder::new(&dag)
            .unwrap()
            .with_circuit_annotations()
            .unwrap()
            .build()
            .unwrap();
        let mut influence = map(&circuit);
        influence.merge_dem_outputs_from(&outputs);
        assert!(influence.influences.max_dem_output_index().is_some());
        influence.dem_output_metadata.clear();
        // Even a same-id JSON definition cannot establish the untyped output's kind.
        let builder = DemBuilder::new(&influence)
            .with_noise_config(noise(0.1))
            .with_observables_json(r#"[{"id":0,"records":[1]}]"#)
            .unwrap();
        for result in [builder.build(), builder.try_build()] {
            let error = result.unwrap_err().to_string();
            assert!(error.contains("outputs with unknown kinds"), "{error}");
            assert!(error.contains("supply dem_output_metadata"), "{error}");
            assert!(error.contains("or use StateFlip"), "{error}");
        }
        // No measurement noise needs no interpretation of the untyped outputs.
        DemBuilder::new(&influence)
            .with_noise_config(noise(0.0))
            .try_build()
            .unwrap();
    }
}

#[test]
fn record_flip_rejects_short_output_metadata() {
    let mut circuit = repeated(GateType::MZ, true);
    circuit.tracked_pauli(PauliString::zs(&[0]));
    let dag = DagCircuit::try_from(&circuit).unwrap();
    let outputs = InfluenceBuilder::new(&dag)
        .unwrap()
        .with_circuit_annotations()
        .unwrap()
        .build()
        .unwrap();
    let mut influence = map(&circuit);
    influence.merge_dem_outputs_from(&outputs);
    assert_eq!(influence.dem_output_metadata.len(), 2);
    assert_eq!(influence.influences.max_dem_output_index(), Some(1));
    influence.dem_output_metadata.truncate(1);
    let builder = DemBuilder::new(&influence)
        .with_noise_config(noise(0.1))
        .with_observables_json(r#"[{"id":0,"records":[1]}]"#)
        .unwrap();
    for result in [builder.build(), builder.try_build()] {
        let error = result.unwrap_err().to_string();
        assert!(error.contains("outputs with unknown kinds"), "{error}");
        assert!(error.contains("supply dem_output_metadata"), "{error}");
    }
    DemBuilder::new(&influence)
        .with_noise_config(noise(0.0))
        .try_build()
        .unwrap();
}

#[test]
fn state_flip_build_still_drops_invalid_references() {
    // Without measurement classes, StateFlip build() skips reference validation.
    // Mechanisms use resolved IDs / in-range offsets; explicit record definitions
    // are retained verbatim, while unknown meas_ids have no resolvable offset.
    let influence = map(&repeated(GateType::MZ, false));
    for per_gate in [false, true] {
        for observable in [false, true] {
            for (references, records) in [
                (r#""records":[0,2,-3]"#, vec![0, 2, -3]),
                (r#""meas_ids":[0,999]"#, vec![-2]),
            ] {
                let json = format!(r#"[{{"id":0,{references}}}]"#);
                let config = NoiseConfig::new(0.0, 0.0, 0.1, 0.0);
                let builder = DemBuilder::new(&influence);
                let builder = if per_gate {
                    builder.with_per_gate_noise(PerGateTypeNoise::from_base_noise(config))
                } else {
                    builder.with_noise_config(config)
                };
                let builder = if observable {
                    builder.with_observables_json(&json)
                } else {
                    builder.with_detectors_json(&json)
                }
                .unwrap();
                let dem = builder.build().unwrap();
                let expected = if observable {
                    (0.1, vec![], vec![0])
                } else {
                    (0.1, vec![0], vec![])
                };
                assert_mechanisms(&dem, &[expected]);
                // Standard DEM text declares IDs but has no record-offset syntax.
                if observable {
                    assert_eq!(dem.to_string(), "logical_observable L0\nerror(0.1) L0");
                    assert_eq!(
                        dem.observables().next().unwrap().records.as_slice(),
                        records.as_slice()
                    );
                } else {
                    assert_eq!(dem.to_string(), "detector D0\nerror(0.1) D0");
                    assert_eq!(dem.detectors[0].records.as_slice(), records.as_slice());
                }
                assert!(builder.try_build().is_err());
            }
        }
    }
}
