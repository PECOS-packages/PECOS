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
    assert_eq!(engine_error.gate_type, gate_type);
    assert_eq!(engine_error.location, dag_location);

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
        Gate::u(Angle64::ZERO, Angle64::ZERO, Angle64::ZERO, &[0]),
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
    assert_eq!(engine_error.gate_type, GateType::T);
    assert_eq!(engine_error.location, location);
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
