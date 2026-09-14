//! Regression coverage for PECOS issue #594.

use pecos_core::gate_type::GateType;
use pecos_core::{Angle64, Gate};
use pecos_qec::fault_tolerance::InfluenceBuilder;
use pecos_qec::fault_tolerance::dem_builder::{
    DemBuilder, DemBuilderError, DemSampler, DemSamplerBuilder, DetectorValidationError,
    MemBuilder, NoiseConfig, SamplingEngine,
};
use pecos_qec::fault_tolerance::influence_builder::InfluenceBuildError;
use pecos_qec::fault_tolerance::propagator::{
    DagFaultAnalyzer, Direction, PauliPropagationOutcome, UnsupportedGateLocation, apply_gate,
    is_supported_noop_or_metadata_gate, is_supported_prep_gate,
};
use pecos_qec::{DemStabError, DemStabSim, MemStabError, MemStabSim};
use pecos_quantum::{DagCircuit, TickCircuit};
use pecos_simulators::PauliProp;

fn t_dag() -> DagCircuit {
    let mut circuit = DagCircuit::new();
    circuit.pz(&[0]);
    circuit.t(&[0]);
    circuit.mz(&[0]);
    circuit
}

fn qalloc_dag() -> DagCircuit {
    let mut circuit = DagCircuit::new();
    circuit.qalloc(&[0]);
    circuit.h(&[0]);
    circuit.h(&[0]);
    circuit.mz(&[0]);
    circuit
}

fn qfree_dag() -> DagCircuit {
    let mut circuit = DagCircuit::new();
    circuit.pz(&[0]);
    circuit.h(&[0]);
    circuit.h(&[0]);
    circuit.mz(&[0]);
    circuit.qfree(&[0]);
    circuit
}

fn assert_dem_error(
    error: DemBuilderError,
    gate_type: GateType,
    location: UnsupportedGateLocation,
) {
    let DemBuilderError::UnsupportedGate(error) = error else {
        panic!("expected structured unsupported-gate error, got {error:?}");
    };
    assert_eq!(error.gate_type, gate_type);
    assert_eq!(error.location, location);
    assert_eq!(error.qubits, [0]);
}

fn assert_sampler_error(
    error: DetectorValidationError,
    gate_type: GateType,
    location: UnsupportedGateLocation,
) {
    let DetectorValidationError::UnsupportedGate(error) = error else {
        panic!("expected structured unsupported-gate error, got {error:?}");
    };
    assert_eq!(error.gate_type, gate_type);
    assert_eq!(error.location, location);
    assert_eq!(error.qubits, [0]);
}

/// Every DEM entry point must reject `gate_type`, placed as the single gate
/// between a preparation and a measurement, reporting that gate and its exact
/// source location. `add_dag` / `add_tick` insert the gate under test.
fn assert_rotation_rejected_by_every_dem_family(
    gate_type: GateType,
    add_dag: &dyn Fn(&mut DagCircuit),
    add_tick: &dyn Fn(&mut TickCircuit),
) {
    let mut circuit = DagCircuit::new();
    circuit.pz(&[0]);
    add_dag(&mut circuit);
    circuit.mz(&[0]);
    // Non-Clifford rejection tests pin the preflight, not the symbolic replay error mapping.
    let dag_location = UnsupportedGateLocation::DagNode { node: 1 };

    assert!(matches!(
        InfluenceBuilder::new(&circuit).build().unwrap_err(),
        InfluenceBuildError::UnsupportedPauliPropagation(_)
    ));
    let map = DagFaultAnalyzer::new(&circuit).build_influence_map();
    let probabilities = vec![0.0; map.locations.len()];
    assert_dem_error(
        DemBuilder::new(&map).build().unwrap_err(),
        gate_type,
        dag_location,
    );
    assert_dem_error(
        MemBuilder::new(&map).build().unwrap_err(),
        gate_type,
        dag_location,
    );
    assert_sampler_error(
        DemSamplerBuilder::new(&map).build().unwrap_err(),
        gate_type,
        dag_location,
    );
    assert_sampler_error(
        DemSampler::from_influence_map(&map, &probabilities).unwrap_err(),
        gate_type,
        dag_location,
    );
    let engine_error =
        SamplingEngine::from_influence_map(&map, &probabilities, &NoiseConfig::default())
            .unwrap_err();
    assert_dem_error(engine_error, gate_type, dag_location);

    assert_dem_error(
        DemBuilder::from_circuit(&circuit, 0.0, 0.0, 0.0, 0.0).unwrap_err(),
        gate_type,
        dag_location,
    );
    assert_sampler_error(
        DemSampler::from_circuit(&circuit, &NoiseConfig::default()).unwrap_err(),
        gate_type,
        dag_location,
    );
    assert!(matches!(
        DemStabSim::builder()
            .circuit(circuit.clone())
            .build()
            .unwrap_err(),
        DemStabError::DetectorValidation(DetectorValidationError::UnsupportedGate(_))
    ));
    assert!(matches!(
        MemStabSim::builder().circuit(circuit).build().unwrap_err(),
        MemStabError::DemBuilder(DemBuilderError::UnsupportedGate(_))
    ));

    let mut tick = TickCircuit::new();
    tick.tick().pz(&[0]);
    add_tick(&mut tick);
    tick.tick().mz(&[0]);
    let tick_location = UnsupportedGateLocation::Tick {
        tick: 1,
        gate_in_tick: 0,
    };
    assert_dem_error(
        DemBuilder::from_tick_circuit(&tick, 0.0, 0.0, 0.0, 0.0).unwrap_err(),
        gate_type,
        tick_location,
    );
    assert_sampler_error(
        DemSampler::from_tick_circuit(&tick, &NoiseConfig::default()).unwrap_err(),
        gate_type,
        tick_location,
    );
}

fn assert_all_dem_entry_points_build(circuit: &DagCircuit) {
    InfluenceBuilder::new(circuit).build().unwrap();

    let map = DagFaultAnalyzer::new(circuit).build_influence_map();
    assert!(map.unsupported_gate().is_none());
    let probabilities = vec![0.0; map.locations.len()];

    DemBuilder::new(&map).build().unwrap();
    DemBuilder::new(&map).try_build().unwrap();
    MemBuilder::new(&map).build().unwrap();
    DemSamplerBuilder::new(&map).build().unwrap();
    DemSamplerBuilder::new(&map)
        .with_detector_records(Vec::new())
        .build()
        .unwrap();
    DemSampler::from_influence_map(&map, &probabilities).unwrap();
    SamplingEngine::from_influence_map(&map, &probabilities, &NoiseConfig::default()).unwrap();

    DemBuilder::from_circuit(circuit, 0.0, 0.0, 0.0, 0.0).unwrap();
    DemBuilder::try_from_circuit(circuit, 0.0, 0.0, 0.0, 0.0).unwrap();
    DemBuilder::try_from_circuit_with_noise_config(circuit, NoiseConfig::default()).unwrap();
    DemSampler::from_circuit(circuit, &NoiseConfig::default()).unwrap();
    DemStabSim::builder()
        .circuit(circuit.clone())
        .build()
        .unwrap();
    MemStabSim::builder()
        .circuit(circuit.clone())
        .build()
        .unwrap();

    let tick = TickCircuit::from(circuit);
    DemBuilder::from_tick_circuit(&tick, 0.0, 0.0, 0.0, 0.0).unwrap();
    DemBuilder::try_from_tick_circuit(&tick, 0.0, 0.0, 0.0, 0.0).unwrap();
    DemBuilder::try_from_tick_circuit_with_noise_config(&tick, NoiseConfig::default()).unwrap();
    DemSampler::from_tick_circuit(&tick, &NoiseConfig::default()).unwrap();
}

#[test]
fn apply_gate_classifies_supported_transparent_and_unsupported_gates() {
    let mut prop = PauliProp::new();
    prop.track_x(&[0]);

    assert_eq!(
        apply_gate(&mut prop, &Gate::h(&[0]), Direction::Forward),
        PauliPropagationOutcome::Propagated
    );
    for gate in [
        Gate::rz(Angle64::QUARTER_TURN, &[0]),
        Gate::rz(Angle64::HALF_TURN, &[0]),
        // RXY1Q(pi/2, 0) lowers to the named SX and must propagate.
        Gate::rxy1q(Angle64::QUARTER_TURN, Angle64::ZERO, &[0]),
        Gate::u(Angle64::ZERO, Angle64::ZERO, Angle64::ZERO, &[0]),
        Gate::u(Angle64::ZERO, Angle64::ZERO, Angle64::HALF_TURN, &[0]),
    ] {
        assert_eq!(
            apply_gate(&mut prop, &gate, Direction::Forward),
            PauliPropagationOutcome::Propagated
        );
    }
    let transparent_gates = [
        Gate::px(&[0]),
        Gate::pz(&[0]),
        Gate::qalloc(&[0]),
        Gate::qfree(&[0]),
        Gate::simple(GateType::I, vec![0.into()]),
        Gate::idle(1.0, vec![0.into()]),
        Gate::meas_crosstalk_global_payload(&[0]),
        Gate::meas_crosstalk_local_payload(&[0]),
        Gate::simple(GateType::TrackedPauliMeta, vec![0.into()]),
    ];
    for gate in transparent_gates {
        let gate_type = gate.gate_type;
        assert!(is_supported_prep_gate(gate_type) || is_supported_noop_or_metadata_gate(gate_type));
        assert_eq!(
            apply_gate(&mut prop, &gate, Direction::Forward),
            PauliPropagationOutcome::Propagated,
            "{gate_type:?} must remain deliberately transparent"
        );
    }

    for gate in [
        Gate::t(&[0]),
        Gate::tdg(&[0]),
        Gate::rz(Angle64::from_turns(0.125), &[0]),
        Gate::rz(Angle64::from_turns(-0.125), &[0]),
        Gate::rz(Angle64::from_turns(0.1), &[0]),
        Gate::u(Angle64::HALF_TURN, Angle64::ZERO, Angle64::ZERO, &[0]),
        Gate::u(
            Angle64::ZERO,
            Angle64::ZERO,
            Angle64::from_turns(0.125),
            &[0],
        ),
    ] {
        assert_eq!(
            apply_gate(&mut prop, &gate, Direction::Forward),
            PauliPropagationOutcome::Unsupported,
            "{:?} must not be silently treated as identity",
            gate.gate_type
        );
    }
    // A non-Clifford RXY1Q has no Pauli conjugation rule and must not be
    // reported as propagated. This is the arm that took a merge conflict when
    // R1XY was renamed to RXY1Q.
    assert_eq!(
        apply_gate(
            &mut prop,
            &Gate::rxy1q(Angle64::from_turns(0.1), Angle64::ZERO, &[0]),
            Direction::Forward
        ),
        PauliPropagationOutcome::Unsupported
    );
}

#[test]
fn malformed_gate_payloads_are_unsupported() {
    let mut missing_rz_angle = Gate::rz(Angle64::ZERO, &[0]);
    missing_rz_angle.angles.clear();
    let mut extra_rz_angle = Gate::rz(Angle64::ZERO, &[0]);
    extra_rz_angle.angles.push(Angle64::ZERO);
    let mut missing_rxy1q_angle = Gate::rxy1q(Angle64::ZERO, Angle64::ZERO, &[0]);
    missing_rxy1q_angle.angles.pop();
    let mut extra_rxy1q_angle = Gate::rxy1q(Angle64::ZERO, Angle64::ZERO, &[0]);
    extra_rxy1q_angle.angles.push(Angle64::ZERO);
    let mut malformed_named = Gate::h(&[0]);
    malformed_named.angles.push(Angle64::ZERO);
    let mut malformed_transparent = Gate::pz(&[0]);
    malformed_transparent.angles.push(Angle64::ZERO);

    let mut prop = PauliProp::new();
    for gate in [
        missing_rz_angle,
        extra_rz_angle,
        missing_rxy1q_angle,
        extra_rxy1q_angle,
        malformed_named,
        malformed_transparent,
    ] {
        assert_eq!(
            apply_gate(&mut prop, &gate, Direction::Forward),
            PauliPropagationOutcome::Unsupported,
            "malformed {:?} payload must be rejected",
            gate.gate_type
        );
    }
}

#[test]
fn bare_t_is_rejected_by_every_influence_map_dem_entry_point() {
    let circuit = t_dag();
    let map = DagFaultAnalyzer::new(&circuit).build_influence_map();
    let probabilities = vec![0.0; map.locations.len()];
    let location = UnsupportedGateLocation::DagNode { node: 1 };

    assert_dem_error(
        DemBuilder::new(&map).build().unwrap_err(),
        GateType::T,
        location,
    );
    assert_dem_error(
        DemBuilder::new(&map).try_build().unwrap_err(),
        GateType::T,
        location,
    );

    let mem_error = MemBuilder::new(&map).build().unwrap_err();
    assert_dem_error(mem_error, GateType::T, location);

    assert_sampler_error(
        DemSamplerBuilder::new(&map).build().unwrap_err(),
        GateType::T,
        location,
    );
    assert_sampler_error(
        DemSamplerBuilder::new(&map)
            .with_detector_records(Vec::new())
            .build()
            .unwrap_err(),
        GateType::T,
        location,
    );
    assert_sampler_error(
        DemSampler::from_influence_map(&map, &probabilities).unwrap_err(),
        GateType::T,
        location,
    );

    let engine_error =
        SamplingEngine::from_influence_map(&map, &probabilities, &NoiseConfig::default())
            .unwrap_err();
    assert_dem_error(engine_error, GateType::T, location);
}

#[test]
fn bare_t_is_rejected_by_every_circuit_dem_entry_point() {
    let circuit = t_dag();
    let dag_location = UnsupportedGateLocation::DagNode { node: 1 };

    assert!(matches!(
        InfluenceBuilder::new(&circuit).build().unwrap_err(),
        InfluenceBuildError::UnsupportedPauliPropagation(_)
    ));
    assert_dem_error(
        DemBuilder::from_circuit(&circuit, 0.0, 0.0, 0.0, 0.0).unwrap_err(),
        GateType::T,
        dag_location,
    );
    assert_dem_error(
        DemBuilder::try_from_circuit(&circuit, 0.0, 0.0, 0.0, 0.0).unwrap_err(),
        GateType::T,
        dag_location,
    );
    assert_dem_error(
        DemBuilder::try_from_circuit_with_noise_config(&circuit, NoiseConfig::default())
            .unwrap_err(),
        GateType::T,
        dag_location,
    );
    assert_sampler_error(
        DemSampler::from_circuit(&circuit, &NoiseConfig::default()).unwrap_err(),
        GateType::T,
        dag_location,
    );
    let dem_stab_error = DemStabSim::builder()
        .circuit(circuit.clone())
        .build()
        .unwrap_err();
    assert!(matches!(
        dem_stab_error,
        DemStabError::DetectorValidation(DetectorValidationError::UnsupportedGate(_))
    ));
    let mem_stab_error = MemStabSim::builder()
        .circuit(circuit.clone())
        .build()
        .unwrap_err();
    assert!(matches!(
        mem_stab_error,
        MemStabError::DemBuilder(DemBuilderError::UnsupportedGate(_))
    ));

    let mut tick = TickCircuit::new();
    tick.tick().pz(&[0]);
    tick.tick().t(&[0]);
    tick.tick().mz(&[0]);
    let tick_location = UnsupportedGateLocation::Tick {
        tick: 1,
        gate_in_tick: 0,
    };
    assert_dem_error(
        DemBuilder::from_tick_circuit(&tick, 0.0, 0.0, 0.0, 0.0).unwrap_err(),
        GateType::T,
        tick_location,
    );
    assert_dem_error(
        DemBuilder::try_from_tick_circuit(&tick, 0.0, 0.0, 0.0, 0.0).unwrap_err(),
        GateType::T,
        tick_location,
    );
    assert_dem_error(
        DemBuilder::try_from_tick_circuit_with_noise_config(&tick, NoiseConfig::default())
            .unwrap_err(),
        GateType::T,
        tick_location,
    );
    assert_sampler_error(
        DemSampler::from_tick_circuit(&tick, &NoiseConfig::default()).unwrap_err(),
        GateType::T,
        tick_location,
    );
}

#[test]
fn non_clifford_rz_is_rejected_with_its_gate_and_location() {
    let mut circuit = DagCircuit::new();
    circuit.pz(&[0]);
    circuit.rz(Angle64::from_turns(0.1), &[0]);
    circuit.mz(&[0]);

    let error = DemBuilder::from_circuit(&circuit, 0.0, 0.0, 0.0, 0.0).unwrap_err();
    let DemBuilderError::UnsupportedGate(error) = error else {
        panic!("expected structured unsupported-gate error, got {error:?}");
    };
    assert_eq!(error.gate_type, GateType::RZ);
    assert_eq!(error.location, UnsupportedGateLocation::DagNode { node: 1 });
    assert_eq!(error.qubits, [0]);
}

#[test]
fn positive_eighth_turn_rz_is_rejected_by_every_dem_family() {
    let angle = Angle64::from_turns(0.125);
    assert_rotation_rejected_by_every_dem_family(
        GateType::RZ,
        &|c| {
            c.rz(angle, &[0]);
        },
        &|t| {
            t.tick().rz(angle, &[0]);
        },
    );
}

#[test]
fn negative_eighth_turn_rz_is_rejected_by_every_dem_family() {
    let angle = Angle64::from_turns(-0.125);
    assert_rotation_rejected_by_every_dem_family(
        GateType::RZ,
        &|c| {
            c.rz(angle, &[0]);
        },
        &|t| {
            t.tick().rz(angle, &[0]);
        },
    );
}

/// `RXY1Q` at a non-Clifford angle has no Pauli conjugation rule. This is the
/// arm that took a merge conflict when the gate was renamed from `R1XY`, so it
/// gets its own end-to-end guard rather than relying on the `RZ` cases.
#[test]
fn non_clifford_rxy1q_is_rejected_by_every_dem_family() {
    let theta = Angle64::from_turns(0.1);
    let phi = Angle64::ZERO;
    assert_rotation_rejected_by_every_dem_family(
        GateType::RXY1Q,
        &|c| {
            c.rxy1q(theta, phi, &[0]);
        },
        &|t| {
            t.tick().rxy1q(theta, phi, &[0]);
        },
    );
}

#[test]
fn independent_exact_replay_context_is_validated() {
    let mut base = DagCircuit::new();
    base.pz(&[0]);
    base.mz(&[0]);
    let map = DagFaultAnalyzer::new(&base).build_influence_map();

    let replay = t_dag();
    assert_dem_error(
        DemBuilder::new(&map)
            .with_exact_branch_replay_context(&replay)
            .build()
            .unwrap_err(),
        GateType::T,
        UnsupportedGateLocation::DagNode { node: 1 },
    );
    assert_dem_error(
        DemBuilder::new(&map)
            .with_exact_branch_replay_context(&replay)
            .try_build()
            .unwrap_err(),
        GateType::T,
        UnsupportedGateLocation::DagNode { node: 1 },
    );
}

#[test]
fn transparent_gates_still_build_a_dem() {
    let mut circuit = DagCircuit::new();
    circuit.pz(&[0]);
    circuit.add_gate_auto_wire(Gate::px(&[1]));
    circuit.idle(1u64, &[0, 1]);
    circuit.add_gate_auto_wire(Gate::simple(GateType::TrackedPauliMeta, vec![0.into()]));
    circuit.add_gate_auto_wire(Gate::meas_crosstalk_global_payload(&[0]));
    circuit.add_gate_auto_wire(Gate::meas_crosstalk_local_payload(&[1]));
    circuit.mz(&[0, 1]);

    DemBuilder::from_circuit(&circuit, 0.0, 0.0, 0.0, 0.0)
        .expect("transparent gates must not be rejected by the Pauli-propagation guard");
}

#[test]
fn qalloc_circuit_builds_through_every_dem_entry_point() {
    assert_all_dem_entry_points_build(&qalloc_dag());
}

#[test]
fn qfree_circuit_builds_through_every_dem_entry_point() {
    assert_all_dem_entry_points_build(&qfree_dag());
}

#[test]
fn leaked_measurement_replay_error_is_an_unsupported_gate() {
    let mut circuit = DagCircuit::new();
    circuit.pz(&[0, 1]);
    circuit.h(&[0]);
    let node = circuit.add_gate_auto_wire(Gate::measure_leaked(&[1]));
    circuit.mz(&[0]);
    assert!(
        DagFaultAnalyzer::new(&circuit)
            .build_influence_map()
            .unsupported_gate()
            .is_none()
    );
    let error = DemSampler::from_circuit(&circuit, &NoiseConfig::default()).unwrap_err();
    if let DetectorValidationError::UnsupportedGate(error) = &error {
        assert_eq!(error.qubits, vec![1]);
        assert_eq!(error.gate_type, GateType::MeasureLeaked);
        assert_eq!(error.location, UnsupportedGateLocation::DagNode { node });
    } else {
        panic!("expected UnsupportedGate, got {error:?}");
    }
}

fn detector_records(
    map: &pecos_qec::fault_tolerance::propagator::DagFaultInfluenceMap,
) -> Vec<Vec<usize>> {
    let mut records: Vec<Vec<usize>> = map
        .detectors
        .iter()
        .map(|detector| {
            let mut records: Vec<usize> = detector
                .measurements
                .iter()
                .map(|measurement| {
                    map.measurements
                        .iter()
                        .position(|&(node, qubit, basis)| {
                            (node, qubit, basis)
                                == (measurement.tick, measurement.qubit, measurement.basis)
                        })
                        .expect("detectors refer to measurements in their influence map")
                })
                .collect();
            records.sort_unstable();
            records
        })
        .collect();
    records.sort_unstable();
    records
}

fn clifford_rotation_replay_matches_named_gates(
    rotation: &Gate,
    named: Gate,
    preparation: &[Gate],
    readout: &[Gate],
    distinguishes_identity: bool,
    noise: &NoiseConfig,
) {
    const DETECTORS: &str = r#"[{"id":0,"records":[-2]},{"id":1,"records":[-1]}]"#;
    let make_circuit = |gate: Option<Gate>| {
        let mut circuit = DagCircuit::new();
        circuit.pz(&[0, 1]);
        for gate in preparation {
            circuit.add_gate_auto_wire(gate.clone());
        }
        if let Some(gate) = gate {
            circuit.add_gate_auto_wire(gate);
        }
        for gate in readout {
            circuit.add_gate_auto_wire(gate.clone());
        }
        circuit.mz(&[0]);
        circuit.mz(&[1]);
        circuit.set_attr(
            "detectors",
            pecos_quantum::Attribute::String(DETECTORS.to_string()),
        );
        circuit
    };
    let raw = make_circuit(Some(rotation.clone()));
    let lowered = make_circuit(Some(named));
    let raw_map = InfluenceBuilder::new(&raw).with_z(&[0, 1]).build().unwrap();
    let lowered_map = InfluenceBuilder::new(&lowered)
        .with_z(&[0, 1])
        .build()
        .unwrap();
    if distinguishes_identity {
        let removed = make_circuit(None);
        let removed_map = InfluenceBuilder::new(&removed)
            .with_z(&[0, 1])
            .build()
            .unwrap();
        // Measurement ordinals keep this comparison independent of shifted DAG node IDs.
        assert_ne!(
            detector_records(&removed_map),
            detector_records(&lowered_map),
            "{rotation:?}"
        );
    }
    // Paulis and identity preserve detector structure; those rows check acceptance and equality.
    assert_eq!(raw_map.detectors, lowered_map.detectors, "{rotation:?}");
    assert_eq!(raw_map.measurements, lowered_map.measurements);
    assert_eq!(raw_map.dem_output_metadata, lowered_map.dem_output_metadata);
    assert_eq!(raw_map.dem_output_labels, lowered_map.dem_output_labels);
    let builder_dem = |map| {
        let dem = DemBuilder::new(map)
            .with_noise_config(noise.clone())
            .with_detectors_json(DETECTORS)
            .unwrap()
            .build()
            .unwrap();
        let text = dem.to_string();
        assert!(text.contains("error("), "{rotation:?}: {text}");
        (text, dem.to_string_decomposed())
    };
    let sampler_dem = |circuit| {
        let dem = DemSampler::from_circuit(circuit, noise)
            .unwrap()
            .to_detector_error_model();
        let text = dem.to_string();
        assert!(
            text.split_whitespace().any(|token| token == "D0"),
            "{rotation:?}: {text}"
        );
        text
    };
    let raw_dems = (builder_dem(&raw_map), sampler_dem(&raw));
    let lowered_dems = (builder_dem(&lowered_map), sampler_dem(&lowered));
    assert_eq!(raw_dems, lowered_dems, "{rotation:?}");
}

#[test]
fn clifford_rotation_replay_rz_quarter_turn() {
    clifford_rotation_replay_matches_named_gates(
        &Gate::rz(Angle64::QUARTER_TURN, &[0]),
        Gate::sz(&[0]),
        &[Gate::h(&[0])],
        &[Gate::szdg(&[0]), Gate::h(&[0])],
        true,
        &NoiseConfig::new(0.01, 0.02, 0.01, 0.02),
    );
}

#[test]
fn clifford_rotation_replay_rz_half_turn() {
    clifford_rotation_replay_matches_named_gates(
        &Gate::rz(Angle64::HALF_TURN, &[0]),
        Gate::z(&[0]),
        &[],
        &[],
        false,
        &NoiseConfig::new(0.01, 0.02, 0.01, 0.02),
    );
}

#[test]
fn clifford_rotation_replay_rxy1q_sx() {
    clifford_rotation_replay_matches_named_gates(
        &Gate::rxy1q(Angle64::QUARTER_TURN, Angle64::ZERO, &[0]),
        Gate::sx(&[0]),
        &[],
        &[Gate::sxdg(&[0])],
        true,
        &NoiseConfig::new(0.01, 0.02, 0.01, 0.02),
    );
}

#[test]
fn clifford_rotation_replay_rxy1q_sydg() {
    clifford_rotation_replay_matches_named_gates(
        &Gate::rxy1q(Angle64::QUARTER_TURN, Angle64::THREE_QUARTERS_TURN, &[0]),
        Gate::sydg(&[0]),
        &[],
        &[Gate::sy(&[0])],
        true,
        &NoiseConfig::new(0.01, 0.02, 0.01, 0.02),
    );
}

#[test]
fn clifford_rotation_replay_rxy1q_identity() {
    // Gate noise is excluded because a zero rotation is a noise location while named identity is not.
    clifford_rotation_replay_matches_named_gates(
        &Gate::rxy1q(Angle64::ZERO, Angle64::from_radians(0.3), &[0]),
        Gate::simple(GateType::I, vec![0.into()]),
        &[],
        &[],
        false,
        &NoiseConfig::new(0.0, 0.0, 0.01, 0.02),
    );
}

#[test]
fn clifford_rotation_replay_rzz_quarter_turn() {
    clifford_rotation_replay_matches_named_gates(
        &Gate::rzz(Angle64::QUARTER_TURN, &[(0, 1)]),
        Gate::szz(&[(0, 1)]),
        &[Gate::h(&[0, 1])],
        &[Gate::szzdg(&[(0, 1)]), Gate::h(&[0, 1])],
        true,
        &NoiseConfig::new(0.01, 0.02, 0.01, 0.02),
    );
}

#[test]
fn clifford_rotation_replay_rzz_half_turn() {
    // Gate noise is excluded because a two-qubit rotation and two named Paulis have different noise locations.
    clifford_rotation_replay_matches_named_gates(
        &Gate::rzz(Angle64::HALF_TURN, &[(0, 1)]),
        Gate::z(&[0, 1]),
        &[],
        &[],
        false,
        &NoiseConfig::new(0.0, 0.0, 0.01, 0.02),
    );
}

#[test]
fn batched_measurement_sampler_returns_invalid_metadata() {
    let mut circuit = DagCircuit::new();
    circuit.pz(&[0, 1]);
    circuit.add_gate_auto_wire(Gate::mz(&[0, 1]));
    let error = DemSampler::from_circuit(&circuit, &NoiseConfig::default()).unwrap_err();
    assert!(
        matches!(error, DetectorValidationError::InvalidMetadata { message }
        if message.contains("2 measurements"))
    );
}

fn replacement_noise() -> NoiseConfig {
    use pecos_core::pauli::X;
    use pecos_qec::fault_tolerance::dem_builder::PauliWeights;
    let mut noise = NoiseConfig::new(0.0, 0.01, 0.0, 0.0);
    noise.p2_weights = Some(PauliWeights::with_replacement(
        [(X(0) & X(1), 0.5)],
        [(X(0) & X(1), 0.5)],
    ));
    noise
}

fn replacement_circuit(gate: Gate) -> DagCircuit {
    let mut circuit = DagCircuit::new();
    circuit.pz(&[0, 1]);
    circuit.h(&[0]);
    circuit.add_gate_auto_wire(gate);
    circuit.szzdg(&[(0, 1)]);
    circuit.h(&[0]);
    circuit.mz(&[0]);
    circuit.mz(&[1]);
    circuit.set_attr(
        "detectors",
        pecos_quantum::Attribute::String(
            r#"[{"id":0,"records":[-2]},{"id":1,"records":[-1]}]"#.to_string(),
        ),
    );
    circuit
}

#[test]
fn replacement_branches_match_lowered_clifford_rotations() {
    use pecos_core::pauli::X;
    use pecos_qec::fault_tolerance::dem_builder::{PauliWeights, ReplacementBranchApproximation};
    let dems = |gate, approximation| {
        let mut noise = replacement_noise();
        noise.p2_replacement_approximation = approximation;
        let circuit = replacement_circuit(gate);
        let map = DagFaultAnalyzer::new(&circuit).build_influence_map();
        let dem = DemBuilder::new(&map)
            .with_noise_config(noise.clone())
            .with_detectors_json(r#"[{"id":0,"records":[-2]},{"id":1,"records":[-1]}]"#)
            .unwrap()
            .try_build()
            .unwrap();
        let mut direct_noise = noise.clone();
        direct_noise.p2 *= 0.5;
        direct_noise.p2_weights = Some(PauliWeights::new([(X(0) & X(1), 1.0)]));
        let direct = DemBuilder::new(&map)
            .with_noise_config(direct_noise)
            .with_detectors_json(r#"[{"id":0,"records":[-2]},{"id":1,"records":[-1]}]"#)
            .unwrap()
            .try_build()
            .unwrap()
            .to_string();
        assert!(
            dem.to_string()
                .lines()
                .any(|line| line.starts_with("error(")
                    && !direct.lines().any(|direct_line| direct_line == line)),
            "replacement branches must contribute an error mechanism"
        );
        if approximation == ReplacementBranchApproximation::BranchImpact {
            assert!([&[0, 1][..], &[1][..], &[0][..]].iter().any(|effect| {
                dem.contributions_for_effect(effect, &[])
                    .iter()
                    .any(|c| c.replacement_branch)
            }));
        }
        let sampler = DemSampler::from_circuit(&circuit, &noise)
            .unwrap()
            .to_detector_error_model()
            .to_string();
        assert!(sampler.contains("error("));
        (dem.to_string(), sampler)
    };
    for (angle, named) in [
        (Angle64::QUARTER_TURN, Gate::szz(&[(0, 1)])),
        (Angle64::THREE_QUARTERS_TURN, Gate::szzdg(&[(0, 1)])),
    ] {
        for approximation in [
            ReplacementBranchApproximation::default(),
            ReplacementBranchApproximation::BranchImpact,
        ] {
            assert_eq!(
                dems(Gate::rzz(angle, &[(0, 1)]), approximation),
                dems(named.clone(), approximation)
            );
        }
    }
}

#[test]
fn zero_rotation_replacement_entries_have_identity_twirl() {
    use pecos_qec::fault_tolerance::dem_builder::ReplacementBranchApproximation;
    let circuit = replacement_circuit(Gate::rzz(Angle64::ZERO, &[(0, 1)]));
    let map = DagFaultAnalyzer::new(&circuit).build_influence_map();
    for approximation in [
        ReplacementBranchApproximation::PauliTwirlOmittedGate,
        ReplacementBranchApproximation::BranchImpact,
        ReplacementBranchApproximation::IgnoreGateRemoval,
    ] {
        let mut noise = replacement_noise();
        noise.p2_replacement_approximation = approximation;
        DemBuilder::new(&map)
            .with_noise_config(noise.clone())
            .try_build()
            .unwrap();
        DemSampler::from_circuit(&circuit, &noise).unwrap();
        SamplingEngine::from_influence_map(&map, &vec![0.01; map.locations.len()], &noise).unwrap();
        DemSamplerBuilder::new(&map)
            .with_noise_config(noise)
            .build()
            .unwrap();
    }
}

fn replacement_map_without_twirl() -> pecos_qec::fault_tolerance::propagator::DagFaultInfluenceMap {
    use pecos_core::CliffordLowering;
    let circuit = replacement_circuit(Gate::rzz(Angle64::ZERO, &[(0, 1)]));
    let mut map = DagFaultAnalyzer::new(&circuit).build_influence_map();
    // A malformed external map must not silently discard replacement branches.
    for loc in &mut map.locations {
        if loc.gate_type == GateType::RZZ {
            loc.clifford = CliffordLowering::Named(GateType::T);
        }
    }
    map
}

#[test]
fn replacement_entries_without_twirl_fail_loudly() {
    use pecos_qec::fault_tolerance::dem_builder::ReplacementBranchApproximation;
    let map = replacement_map_without_twirl();
    for approximation in [
        ReplacementBranchApproximation::PauliTwirlOmittedGate,
        ReplacementBranchApproximation::BranchImpact,
        ReplacementBranchApproximation::ExactBranchReplay,
    ] {
        let mut noise = replacement_noise();
        noise.p2_replacement_approximation = approximation;
        let check = |error| {
            let DemBuilderError::ConfigurationError(message) = error else {
                panic!("expected configuration error, got {error:?}")
            };
            assert!(message.contains("node 3"), "{message}");
            assert!(
                message.contains("RZZ") && message.contains("Named(T)"),
                "{message}"
            );
        };
        check(
            DemBuilder::new(&map)
                .with_noise_config(noise.clone())
                .try_build()
                .unwrap_err(),
        );
        check(
            SamplingEngine::from_influence_map(&map, &vec![0.01; map.locations.len()], &noise)
                .unwrap_err(),
        );
        for builder in [
            DemSamplerBuilder::new(&map).with_noise_config(noise.clone()),
            DemSamplerBuilder::new(&map)
                .with_noise_config(noise.clone())
                .with_detector_records(vec![vec![-2], vec![-1]]),
        ] {
            let error = builder.build().unwrap_err();
            assert!(
                matches!(error, DetectorValidationError::InvalidConfiguration { message } if message.contains("have no omitted-gate Pauli twirl"))
            );
        }
    }
}

#[test]
fn replacement_mode_without_twirl_does_not_require_it() {
    use pecos_qec::fault_tolerance::dem_builder::ReplacementBranchApproximation;
    let map = replacement_map_without_twirl();
    let mut noise = replacement_noise();
    noise.p2_replacement_approximation = ReplacementBranchApproximation::IgnoreGateRemoval;
    DemBuilder::new(&map)
        .with_noise_config(noise.clone())
        .try_build()
        .unwrap();
    SamplingEngine::from_influence_map(&map, &vec![0.01; map.locations.len()], &noise).unwrap();
    DemSamplerBuilder::new(&map)
        .with_noise_config(noise.clone())
        .build()
        .unwrap();
}

#[test]
fn replacement_sampling_engines_match_lowered_rotations() {
    use pecos_qec::fault_tolerance::dem_builder::ReplacementBranchApproximation;
    for approximation in [
        ReplacementBranchApproximation::PauliTwirlOmittedGate,
        ReplacementBranchApproximation::BranchImpact,
    ] {
        let dems = |gate| {
            let circuit = replacement_circuit(gate);
            let map = DagFaultAnalyzer::new(&circuit).build_influence_map();
            let mut noise = replacement_noise();
            noise.p2_replacement_approximation = approximation;
            let raw =
                SamplingEngine::from_influence_map(&map, &vec![0.01; map.locations.len()], &noise)
                    .unwrap()
                    .to_detector_error_model()
                    .to_string();
            let detector = DemSamplerBuilder::new(&map)
                .with_noise_config(noise)
                .with_detector_records(vec![vec![-2], vec![-1]])
                .build()
                .unwrap()
                .to_detector_error_model()
                .to_string();
            assert!(raw.contains("error(") && detector.contains("error("));
            (raw, detector)
        };
        assert_eq!(
            dems(Gate::rzz(Angle64::QUARTER_TURN, &[(0, 1)])),
            dems(Gate::szz(&[(0, 1)]))
        );
    }
}

#[test]
fn per_qubit_replacement_twirl_matches_exact_pauli_channel() {
    use pecos_core::pauli::{X, Y, Z};
    use pecos_qec::fault_tolerance::dem_builder::{PauliWeights, ReplacementBranchApproximation};
    let mut circuit = DagCircuit::new();
    circuit.pz(&[0, 1]);
    circuit.h(&[0]);
    circuit.add_gate_auto_wire(Gate::rzz(Angle64::HALF_TURN, &[(0, 1)]));
    circuit.h(&[0]);
    circuit.mz(&[0]);
    circuit.mz(&[1]);
    let detectors = r#"[{"id":0,"records":[-2]},{"id":1,"records":[-1]}]"#;
    circuit.set_attr(
        "detectors",
        pecos_quantum::Attribute::String(detectors.to_string()),
    );
    let map = DagFaultAnalyzer::new(&circuit).build_influence_map();
    for approximation in [
        ReplacementBranchApproximation::PauliTwirlOmittedGate,
        ReplacementBranchApproximation::BranchImpact,
    ] {
        let dems = |weights| {
            let mut noise = NoiseConfig::new(0.0, 0.01, 0.0, 0.0);
            noise.p2_weights = Some(weights);
            noise.p2_replacement_approximation = approximation;
            let dem = DemBuilder::new(&map)
                .with_noise_config(noise.clone())
                .with_detectors_json(detectors)
                .unwrap()
                .try_build()
                .unwrap()
                .to_string();
            let sampler = DemSampler::from_circuit(&circuit, &noise)
                .unwrap()
                .to_detector_error_model()
                .to_string();
            (dem, sampler)
        };
        for (replacement, direct) in [(X(0) & X(1), Y(0) & Y(1)), (Y(0) & Y(1), X(0) & X(1))] {
            let result = dems(PauliWeights::with_replacement([], [(replacement, 1.0)]));
            assert!(result.0.contains("error(") && result.1.contains("error("));
            assert_eq!(result, dems(PauliWeights::new([(direct, 1.0)])));
        }
        let identity = dems(PauliWeights::with_replacement([], [(Z(0) & Z(1), 1.0)]));
        assert!(!identity.0.contains("error(") && !identity.1.contains("error("));
        assert_eq!(
            identity,
            dems(PauliWeights::new([(
                pecos_core::PauliString::identity(),
                1.0
            )]))
        );
    }
}

#[test]
fn sampler_configuration_errors_retain_their_category() {
    use pecos_qec::fault_tolerance::dem_builder::ReplacementBranchApproximation;
    let circuit = replacement_circuit(Gate::rzz(Angle64::QUARTER_TURN, &[(0, 1)]));
    let mut noise = replacement_noise();
    noise.p2_replacement_approximation = ReplacementBranchApproximation::ExactBranchReplay;
    let error = DemSampler::from_circuit(&circuit, &noise).unwrap_err();
    assert!(
        matches!(error, DetectorValidationError::InvalidConfiguration { message } if message.contains("circuit-aware"))
    );
}

#[test]
fn unsupported_rotation_diagnostic_includes_angles() {
    let mut circuit = DagCircuit::new();
    circuit.add_gate_auto_wire(Gate::rzz(Angle64::from_turns(0.125), &[(0, 1)]));
    let map = DagFaultAnalyzer::new(&circuit).build_influence_map();
    let error = map.unsupported_gate().unwrap();
    assert_eq!(error.angles, vec![Angle64::from_turns(0.125)]);
    assert_eq!(
        error.to_string(),
        "unsupported gate RZZ(0.125000 turns) at DAG node 0 on qubits [0, 1]"
    );
}

#[test]
fn invalid_sampling_channel_probabilities_return_configuration_errors() {
    let circuit = replacement_circuit(Gate::szz(&[(0, 1)]));
    let map = DagFaultAnalyzer::new(&circuit).build_influence_map();
    let noise = NoiseConfig::new(0.0, 2.0, 0.0, 0.0);
    assert!(matches!(
        SamplingEngine::from_influence_map(&map, &vec![2.0; map.locations.len()], &noise),
        Err(DemBuilderError::ConfigurationError(_))
    ));
    assert!(matches!(
        DemSamplerBuilder::new(&map)
            .with_noise_config(noise)
            .with_detector_records(vec![vec![-2], vec![-1]])
            .build(),
        Err(DetectorValidationError::InvalidConfiguration { .. })
    ));
}

#[test]
fn mem_builder_invalid_probabilities_return_configuration_error() {
    let circuit = replacement_circuit(Gate::szz(&[(0, 1)]));
    let map = DagFaultAnalyzer::new(&circuit).build_influence_map();
    assert!(matches!(
        MemBuilder::new(&map)
            .with_noise_config(NoiseConfig::new(0.0, 4.0, 0.0, 0.0))
            .build(),
        Err(DemBuilderError::ConfigurationError(_))
    ));
}

#[test]
fn tick_dem_unsupported_rotation_diagnostic_preserves_angles() {
    let mut circuit = TickCircuit::new();
    circuit.tick().rz(Angle64::from_turns(0.125), &[0]);
    let error = DemSampler::from_tick_circuit(&circuit, &NoiseConfig::default()).unwrap_err();
    assert!(error.to_string().contains("RZ(0.125000 turns)"));
}

#[test]
fn fault_catalog_unsupported_rotation_diagnostic_preserves_angles() {
    use pecos_qec::fault_tolerance::fault_sampler::FaultCatalog;
    let mut circuit = TickCircuit::new();
    circuit.tick().rz(Angle64::from_turns(0.125), &[0]);
    let error = FaultCatalog::from_circuit(&circuit).unwrap_err();
    assert!(error.to_string().contains("RZ(0.125000 turns)"));
}

#[test]
fn symbolic_history_arity_diagnostic_preserves_angles() {
    use pecos_qec::fault_tolerance::fault_sampler::symbolic_measurement_history;
    let mut circuit = TickCircuit::new();
    circuit.tick().rz(Angle64::QUARTER_TURN, &[] as &[usize]);
    let error = symbolic_measurement_history(&circuit).unwrap_err();
    assert!(error.to_string().contains("RZ(0.250000 turns)"));
}

#[test]
fn symbolic_history_unsupported_rotation_diagnostic_preserves_angles() {
    use pecos_qec::fault_tolerance::fault_sampler::symbolic_measurement_history;
    let mut circuit = TickCircuit::new();
    circuit.tick().rz(Angle64::from_turns(0.125), &[0]);
    let error = symbolic_measurement_history(&circuit).unwrap_err();
    assert!(error.to_string().contains("RZ(0.125000 turns)"));
}
