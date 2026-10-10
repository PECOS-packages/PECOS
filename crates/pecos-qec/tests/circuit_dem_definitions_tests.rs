use pecos_core::{Gate, MeasId, PauliString};
use pecos_qec::fault_tolerance::circuit_definitions::{DefinitionError, dag_circuit_emission};
use pecos_qec::fault_tolerance::dem_builder::{
    DemBuilder, DemBuilderError, DemSampler, DetectorErrorModel, DetectorValidationError,
    NoiseConfig, PauliWeights,
};
use pecos_qec::fault_tolerance::propagator::{BucketRecorder, DagFaultAnalyzer};
use pecos_quantum::{Attribute, DagCircuit, TickCircuit};

fn errors(dem: &DetectorErrorModel) -> Vec<String> {
    dem.to_string()
        .lines()
        .filter(|line| line.starts_with("error("))
        .map(str::to_owned)
        .collect()
}

#[test]
fn analyzer_positions_equal_emission_for_scrambled_batched_measurements() {
    let mut dag = DagCircuit::new();
    dag.pz(&[0, 1, 2]);
    let mut gate = Gate::mz(&[2usize, 0]);
    gate.meas_ids = smallvec::smallvec![MeasId::from_raw(17), MeasId::from_raw(9)];
    let batch = dag.add_gate_auto_wire(gate);
    let last = dag.mz(&[1])[0];
    dag.update_gate(last.node, |gate| {
        gate.gate_type = pecos_quantum::GateType::MX;
    })
    .unwrap();
    let analyzer = DagFaultAnalyzer::new(&dag);
    let map = analyzer.build_influence_map().expect("supported circuit");
    // The first node emits q2 then q0 in its qubit-list order, then MX emits
    // q1. The supplied ids 17,9 reserve the next minted id, 18.
    assert_eq!(
        map.measurements,
        [(batch, 2, 0), (batch, 0, 0), (last.node, 1, 1)]
    );
    assert_eq!(
        map.meas_ids.iter().map(|id| id.index()).collect::<Vec<_>>(),
        [17, 9, 18]
    );
    assert_eq!(
        map.meas_ids.iter().copied().map(Some).collect::<Vec<_>>(),
        dag_circuit_emission(&dag)
    );
    for (index, measurement) in map.detectors.iter().enumerate() {
        let member = &measurement.measurements[0];
        assert_eq!(
            (member.tick, member.qubit, member.basis),
            map.measurements[index]
        );
    }
    // All three propagation entry points must attach effects to the same
    // ordinal, including the forest's separate per-wire time ordering.
    let mut serial = BucketRecorder::new(map.locations.len());
    analyzer
        .propagate_all(&mut serial)
        .expect("supported circuit");
    let serial = serial.into_soa();
    for other in [
        analyzer
            .propagate_all_parallel()
            .expect("supported circuit")
            .into_soa(),
        analyzer
            .propagate_all_forest()
            .expect("supported circuit")
            .into_soa(),
    ] {
        assert_eq!(serial.detectors_x.data, other.detectors_x.data);
        assert_eq!(serial.detectors_x.offsets, other.detectors_x.offsets);
        assert_eq!(serial.detectors_z.data, other.detectors_z.data);
        assert_eq!(serial.detectors_z.offsets, other.detectors_z.offsets);
    }
}

#[test]
fn analyzer_emission_handles_reused_nodes() {
    let mut dag = DagCircuit::new();
    let removed = dag.mz(&[0])[0];
    let earlier = dag.mz(&[1])[0];
    dag.remove_gate(removed.node).unwrap();
    let later = dag.mz(&[2])[0];
    assert_eq!(later.node, removed.node);
    // Reusing the smaller free index puts q2 before independent q1 in the
    // keyed emission walk, although q2's minted id is larger.
    let map = DagFaultAnalyzer::new(&dag)
        .build_influence_map()
        .expect("supported circuit");
    assert_eq!(map.measurements, [(later.node, 2, 0), (earlier.node, 1, 0)]);
    assert_eq!(map.meas_ids, [later.meas_id, earlier.meas_id]);
    assert_eq!(
        dag_circuit_emission(&dag),
        [Some(later.meas_id), Some(earlier.meas_id)]
    );
}

fn double_h_tick(annotation: bool) -> TickCircuit {
    let mut circuit = TickCircuit::new();
    circuit.tick().pz(&[0]);
    circuit.tick().h(&[0]);
    circuit.tick().h(&[0]);
    let measurements = circuit.tick().mz(&[0]);
    if annotation {
        circuit.detector(&measurements).unwrap();
    } else {
        circuit
            .add_detector_metadata(&[-1], None, None, None)
            .unwrap();
    }
    circuit
}

#[test]
fn detector_annotations_match_record_metadata_through_tick_builders() {
    let annotated = double_h_tick(true);
    let metadata = double_h_tick(false);
    let noise = NoiseConfig::new(0.1, 0.0, 0.0, 0.0);
    let expected =
        DemBuilder::try_from_tick_circuit_with_noise_config(&metadata, noise.clone()).unwrap();
    // Each H contributes a flip with probability 2p/3. Their odd parity is
    // 2*(1/15)*(14/15) = 28/225, printed as 0.124444.
    assert_eq!(errors(&expected), ["error(0.124444) D0"]);
    for dem in [
        DemBuilder::try_from_tick_circuit(&annotated, 0.1, 0.0, 0.0, 0.0).unwrap(),
        DemBuilder::try_from_tick_circuit_with_noise_config(&annotated, noise.clone()).unwrap(),
    ] {
        assert_eq!(dem.to_string(), expected.to_string());
    }
    let reference = DemSampler::from_tick_circuit(&metadata, &noise).unwrap();
    let actual = DemSampler::from_tick_circuit(&annotated, &noise).unwrap();
    assert_eq!(actual.num_detectors(), 1);
    assert_eq!(actual.num_mechanisms(), reference.num_mechanisms());
    assert!((actual.max_error_probability() - 28.0 / 225.0).abs() < 1e-12);
}

#[test]
fn circuit_reader_preserves_metadata_observable_label_and_normalizes_records() {
    let mut dag = DagCircuit::new();
    dag.pz(&[0]);
    dag.mz(&[0]);
    dag.set_attr(
        "observables",
        Attribute::String(r#"[{"id":3,"records":[0],"label":"logical readout"}]"#.into()),
    );
    for dem in [
        DemBuilder::try_from_circuit(&dag, 0.0, 0.0, 0.125, 0.0).unwrap(),
        DemBuilder::try_from_circuit_with_noise_config(
            &dag,
            NoiseConfig::new(0.0, 0.0, 0.125, 0.0),
        )
        .unwrap(),
    ] {
        let observable = dem.observables().next().unwrap();
        assert_eq!(observable.id, 3);
        assert_eq!(observable.label.as_deref(), Some("logical readout"));
        assert_eq!(observable.records.as_slice(), [-1]);
        assert_eq!(errors(&dem), ["error(0.125) L3"]);
    }
}

#[test]
fn circuit_reader_enforces_source_agreement_for_both_builders() {
    for kind in ["detectors", "observables"] {
        for metadata in [
            "",
            "[]",
            r#"[{"id":0,"records":[0]}]"#,
            r#"[{"id":0,"records":[1]}]"#,
        ] {
            let mut dag = DagCircuit::new();
            dag.pz(&[0, 1]);
            let measurements = dag.mz(&[0, 1]);
            if kind == "detectors" {
                dag.detector(&measurements[..1]).unwrap();
            } else {
                dag.observable(&measurements[..1]).unwrap();
            }
            dag.set_attr(kind, Attribute::String(metadata.into()));
            let noise = NoiseConfig::new(0.0, 0.0, 0.125, 0.0);
            let dem = DemBuilder::try_from_circuit_with_noise_config(&dag, noise.clone());
            let sampler = DemSampler::from_circuit(&dag, &noise);
            if metadata.contains("[1]") {
                assert!(
                    matches!(dem, Err(DemBuilderError::Definition(error)) if matches!(*error, DefinitionError::SourceMismatch { .. }))
                );
                assert!(
                    matches!(sampler, Err(DetectorValidationError::Definition(error)) if matches!(*error, DefinitionError::SourceMismatch { .. }))
                );
            } else {
                let dem = dem.unwrap();
                assert_eq!(
                    errors(&dem),
                    [if kind == "detectors" {
                        "error(0.125) D0"
                    } else {
                        "error(0.125) L0"
                    }]
                );
                assert_eq!(sampler.unwrap().num_mechanisms(), 1);
            }
        }
    }
}

#[test]
fn circuit_builders_reject_non_string_attributes_and_wrong_counts() {
    for (key, value) in [
        ("detectors", Attribute::Int(1)),
        ("observables", Attribute::Int(1)),
        ("num_measurements", Attribute::Int(1)),
        ("num_measurements", Attribute::String("2".into())),
        ("num_measurements", Attribute::String("invalid".into())),
    ] {
        let mut dag = DagCircuit::new();
        dag.pz(&[0]);
        dag.mz(&[0]);
        dag.set_attr(key, value);
        let noise = NoiseConfig::new(0.0, 0.0, 0.0, 0.0);
        let error =
            DemBuilder::try_from_circuit_with_noise_config(&dag, noise.clone()).unwrap_err();
        assert!(matches!(error, DemBuilderError::Definition(_)));
        let error = DemSampler::from_circuit(&dag, &noise).unwrap_err();
        assert!(matches!(error, DetectorValidationError::Definition(_)));
    }
}

#[test]
fn detector_annotation_coordinates_are_absent_three_or_error() {
    for coords in [
        vec![],
        vec![1.0, 2.0, 3.0],
        vec![1.0],
        vec![1.0, 2.0, 3.0, 4.0],
    ] {
        let mut dag = DagCircuit::new();
        dag.pz(&[0]);
        let measurements = dag.mz(&[0]);
        dag.detector_with_coords(&measurements, &coords).unwrap();
        let noise = NoiseConfig::new(0.0, 0.0, 0.125, 0.0);
        let dem = DemBuilder::try_from_circuit_with_noise_config(&dag, noise.clone());
        let sampler = DemSampler::from_circuit(&dag, &noise);
        if coords.is_empty() || coords.len() == 3 {
            assert_eq!(
                dem.unwrap().detectors[0].coords,
                if coords.is_empty() {
                    None
                } else {
                    Some([1.0, 2.0, 3.0])
                }
            );
            assert!(sampler.is_ok());
        } else {
            let error = dem.unwrap_err();
            assert!(error.to_string().contains(&format!(
                "detector id 0: coordinate length {}",
                coords.len()
            )));
            assert!(
                matches!(error, DemBuilderError::Definition(error) if matches!(*error, DefinitionError::InvalidDetectorCoordinates { id: 0, length } if length == coords.len()))
            );
            assert!(matches!(
                sampler,
                Err(DetectorValidationError::Definition(_))
            ));
        }
    }
}

#[test]
fn mx_annotation_and_metadata_observables_follow_measurement_basis() {
    for (pauli, expected) in [
        (PauliString::zs(&[0]), vec!["error(0.125) L0".to_string()]),
        (PauliString::xs(&[0]), vec![]),
    ] {
        let noise = NoiseConfig::new(0.125, 0.0, 0.0, 0.0)
            .set_p1_weights(PauliWeights::from([(pauli, 1.0)]));
        let mut dems = Vec::new();
        for annotation in [false, true] {
            let mut dag = DagCircuit::new();
            dag.pz(&[0]);
            dag.h(&[0]);
            let measurements = dag.mz(&[0]);
            dag.update_gate(measurements[0].node, |gate| {
                gate.gate_type = pecos_quantum::GateType::MX;
            })
            .unwrap();
            if annotation {
                dag.observable(&measurements).unwrap();
            } else {
                dag.set_attr(
                    "observables",
                    Attribute::String(r#"[{"id":0,"records":[-1]}]"#.into()),
                );
            }
            let dem = DemBuilder::try_from_circuit_with_noise_config(&dag, noise.clone()).unwrap();
            // H prepares |+>. Z after H anticommutes with MX and flips L0;
            // X commutes with MX, leaving L0 unchanged.
            assert_eq!(errors(&dem), expected);
            let sampler = DemSampler::from_circuit(&dag, &noise).unwrap();
            assert_eq!(sampler.num_mechanisms(), expected.len());
            dems.push(dem);
        }
        assert_eq!(dems[0].to_mechanisms(), dems[1].to_mechanisms());
    }
}

#[test]
fn stamped_circuit_records_name_emission_positions() {
    let mut results = Vec::new();
    for references in [
        r#""records":[0]"#,
        r#""records":[-2]"#,
        r#""meas_ids":[9]"#,
        r#""records":[0],"meas_ids":[9]"#,
    ] {
        let mut dag = DagCircuit::new();
        dag.pz(&[0, 1]);
        dag.h(&[1]);
        dag.h(&[1]);
        for (qubit, id) in [(1, 9), (0, 4)] {
            let mut gate = Gate::mz(&[qubit]);
            gate.meas_ids = smallvec::smallvec![MeasId::from_raw(id)];
            dag.add_gate_auto_wire(gate);
        }
        dag.set_attr(
            "detectors",
            Attribute::String(format!(r#"[{{"id":0,{references}}}]"#)),
        );
        let noise = NoiseConfig::new(0.1, 0.0, 0.0, 0.0);
        let dem = DemBuilder::try_from_circuit_with_noise_config(&dag, noise.clone()).unwrap();
        // First emission is q1 (id 9); only it encounters the two H faults.
        // Each flips with 1/15, giving odd parity 28/225. q0 has no noise.
        assert_eq!(errors(&dem), ["error(0.124444) D0"]);
        assert_eq!(dem.detectors[0].records.as_slice(), [-2]);
        let sampler = DemSampler::from_circuit(&dag, &noise).unwrap();
        assert_eq!(sampler.num_mechanisms(), 1);
        assert!((sampler.max_error_probability() - 28.0 / 225.0).abs() < 1e-12);
        results.push(dem.to_string());
    }
    assert!(results.windows(2).all(|pair| pair[0] == pair[1]));
}
