use pecos_core::{Angle64, gate_type::GateType};
use pecos_qec::fault_tolerance::dem_builder::{
    DemBuilder, MemBuilder, NoiseConfig, PerGateTypeNoise,
};
use pecos_qec::fault_tolerance::propagator::DagFaultAnalyzer;
use pecos_quantum::{DagCircuit, TickCircuit};

fn phase_circuit() -> DagCircuit {
    let mut dag = DagCircuit::new();
    dag.pz(&[0]);
    dag.u(Angle64::ZERO, Angle64::ZERO, Angle64::QUARTER_TURN, &[0]);
    dag.mz(&[0]);
    dag
}

#[test]
fn phase_u_measurement_noise_resolves_operation_rate() {
    let dag = phase_circuit();
    let influence = DagFaultAnalyzer::new(&dag).build_influence_map();
    for (base, rz, expected) in [(0.3, 0.0, 0.0), (0.0, 0.3, 0.2)] {
        let noise = NoiseConfig::new(base, 0.0, 0.0, 0.0).set_p1_gate_rate(GateType::RZ, rz);
        let mem = MemBuilder::new(&influence)
            .with_noise_config(noise)
            .build()
            .unwrap();
        let probability: f64 = mem.mechanisms.values().sum();
        assert!(
            (probability - expected).abs() < 1e-14,
            "measurement flip: {probability} != {expected}"
        );
    }
}

#[test]
fn phase_u_explicit_scheduled_override_takes_precedence() {
    use pecos_qec::fault_tolerance::dem_builder::DemSamplerBuilder;
    let dag = phase_circuit();
    let influence = DagFaultAnalyzer::new(&dag).build_influence_map();
    let base = NoiseConfig::new(0.0, 0.0, 0.0, 0.0).set_p1_gate_rate(GateType::RZ, 0.0);
    for noise in [
        PerGateTypeNoise::from_base_noise(base.clone()).with_1q_rates_for_qubit(
            GateType::U,
            0.into(),
            [0.3, 0.0, 0.0],
        ),
        PerGateTypeNoise::from_base_noise(base.clone())
            .with_1q_rates(GateType::U, [0.3, 0.0, 0.0])
            .with_1q_rates_for_qubit(GateType::RZ, 0.into(), [0.0; 3]),
        PerGateTypeNoise::from_base_noise(base.clone().set_p1_gate_rate(GateType::U, 0.45))
            .with_1q_rates(GateType::RZ, [0.0; 3]),
    ] {
        let dem = DemBuilder::new(&influence)
            .with_per_gate_noise(noise.clone())
            .with_detectors_json(r#"[{"id":0,"records":[-1]}]"#)
            .unwrap()
            .build()
            .unwrap();
        assert!(
            dem.to_string().contains("error(0.3) D0"),
            "{}",
            dem.to_string()
        );
        let sampler = DemSamplerBuilder::new(&influence)
            .with_per_gate_noise(noise.clone())
            .with_detector_records(vec![vec![-1]])
            .build()
            .unwrap();
        assert!(sampler.sample_statistics(128, 42).per_detector[0] > 0);
        #[cfg(feature = "neo")]
        assert!(neo_ones(&noise, Angle64::ZERO, true) > 0);
    }
    let noise = base.set_p1_gate_rate(GateType::U, 0.45);
    let mem = MemBuilder::new(&influence)
        .with_noise_config(noise.clone())
        .build()
        .unwrap();
    assert!((mem.mechanisms.values().sum::<f64>() - 0.3).abs() < 1e-14);
    let dem = DemBuilder::new(&influence)
        .with_noise_config(noise.clone())
        .with_detectors_json(r#"[{"id":0,"records":[-1]}]"#)
        .unwrap()
        .build()
        .unwrap();
    assert!(dem.to_string().contains("error(0.3) D0"));
    let sampler = DemSamplerBuilder::new(&influence)
        .with_noise_config(noise)
        .with_detector_records(vec![vec![-1]])
        .build()
        .unwrap();
    assert!(sampler.sample_statistics(128, 42).per_detector[0] > 0);
}

#[test]
fn phase_u_fault_catalog_carries_action_and_provenance() {
    use pecos_qec::fault_tolerance::fault_sampler::FaultCatalog;
    let mut circuit = TickCircuit::new();
    circuit.tick().pz(&[0]);
    circuit
        .tick()
        .u(Angle64::ZERO, Angle64::ZERO, Angle64::QUARTER_TURN, &[0]);
    circuit.tick().mz(&[0]);
    let catalog = FaultCatalog::from_circuit(&circuit).unwrap();
    assert!(
        catalog
            .locations
            .iter()
            .any(|loc| loc.gate_type == GateType::U)
    );
    // X before S becomes Y; H then makes it observable in Z measurement.
    let mut phase = TickCircuit::new();
    phase.tick().pz(&[0]);
    phase
        .tick()
        .u(Angle64::ZERO, Angle64::ZERO, Angle64::QUARTER_TURN, &[0]);
    phase.tick().h(&[0]);
    phase.tick().mz(&[0]);
    let catalog = FaultCatalog::from_circuit(&phase).unwrap();
    assert_eq!(
        catalog.locations[0].faults[0].affected_measurements,
        vec![0]
    );
}

#[cfg(feature = "neo")]
fn neo_ones(noise: &PerGateTypeNoise, theta: Angle64, noiseless_set: bool) -> usize {
    use pecos_neo::prelude::*;
    use pecos_simulators::{QuantumSimulator, StateVec};
    let mut model = ComposableNoiseModel::new().add_channel(noise.to_neo_channel());
    if noiseless_set {
        model = model.with_noiseless_gate(pecos_neo::command::GateType::RZ);
    }
    let commands: CommandQueue = [
        GateCommand::new(
            pecos_neo::command::GateType::PZ,
            smallvec::smallvec![0.into()],
        ),
        GateCommand::with_angles(
            pecos_neo::command::GateType::U,
            smallvec::smallvec![0.into()],
            smallvec::smallvec![theta, Angle64::ZERO, Angle64::QUARTER_TURN],
        ),
        GateCommand::new(
            pecos_neo::command::GateType::MZ,
            smallvec::smallvec![0.into()],
        ),
    ]
    .into_iter()
    .collect();
    let mut state = StateVec::with_seed(1, 42);
    let mut runner = CircuitRunner::<StateVec>::rotations()
        .with_noise(model)
        .with_seed(42);
    let mut ones = 0;
    for _ in 0..128 {
        state.reset();
        let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();
        ones += usize::from(outcomes.bitstring(&[0.into()]).unwrap()[0]);
    }
    ones
}

#[cfg(feature = "neo")]
#[test]
fn phase_u_neo_and_dem_agree_on_noise() {
    let dag = phase_circuit();
    let influence = DagFaultAnalyzer::new(&dag).build_influence_map();
    let base = PerGateTypeNoise::from_base_noise(NoiseConfig::new(1.0, 0.0, 0.0, 0.0));
    let exempt = base.clone().with_1q_rates(GateType::RZ, [0.0; 3]);
    let dem = DemBuilder::new(&influence)
        .with_per_gate_noise(exempt.clone())
        .with_detectors_json(r#"[{"id":0,"records":[-1]}]"#)
        .unwrap()
        .build()
        .unwrap();
    assert_eq!(dem.num_contributions(), 0);
    for (noise, noiseless_set) in [(&exempt, false), (&base, true), (&exempt, true)] {
        assert_eq!(
            neo_ones(noise, Angle64::ZERO, noiseless_set),
            0,
            "phase U must agree with DEM"
        );
        let ones = neo_ones(noise, Angle64::HALF_TURN, noiseless_set);
        assert!(
            ones > 0 && ones < 128,
            "general U must remain noisy: {ones}"
        );
    }
}

#[test]
fn measurement_noise_shares_two_qubit_rates_and_idle_policy() {
    use pecos_core::pauli::X;
    use pecos_qec::fault_tolerance::dem_builder::PauliWeights;
    let mut dag = DagCircuit::new();
    dag.pz(&[0, 1]);
    dag.cx(&[(0, 1)]);
    dag.mz(&[0, 1]);
    let influence = DagFaultAnalyzer::new(&dag).build_influence_map();
    let noise = NoiseConfig::new(0.0, 0.0, 0.0, 0.0)
        .set_p2_gate_rate(GateType::CX, 0.3)
        .set_p2_weights(PauliWeights::from([(X(1), 1.0)]));
    let mem = MemBuilder::new(&influence)
        .with_noise_config(noise)
        .build()
        .unwrap();
    assert_eq!(mem.num_mechanisms(), 1);
    assert!((mem.mechanisms.values().sum::<f64>() - 0.3).abs() < 1e-14);
    let mut idle = DagCircuit::new();
    idle.pz(&[0]);
    idle.idle(1_u64, &[0]);
    idle.mz(&[0]);
    let influence = DagFaultAnalyzer::new(&idle).build_influence_map();
    let mem = MemBuilder::new(&influence)
        .with_noise(0.3, 0.0, 0.0, 0.0)
        .build()
        .unwrap();
    assert_eq!(mem.num_mechanisms(), 0);
}

#[test]
fn measurement_noise_rejects_unrepresented_replacement_branches() {
    use pecos_core::pauli::X;
    use pecos_qec::fault_tolerance::dem_builder::{PauliWeights, ReplacementBranchApproximation};
    let mut dag = DagCircuit::new();
    dag.pz(&[0, 1]);
    dag.cx(&[(0, 1)]);
    dag.mz(&[0, 1]);
    let influence = DagFaultAnalyzer::new(&dag).build_influence_map();
    let mut omitted_modes = Vec::new();
    for mode in [
        ReplacementBranchApproximation::BranchImpact,
        ReplacementBranchApproximation::ExactBranchReplay,
    ] {
        let noise = NoiseConfig::new(0.0, 0.12, 0.0, 0.0)
            .set_p2_weights(PauliWeights::with_replacement([], [(X(1), 1.0)]))
            .set_p2_replacement_approximation(mode);
        let dem = DemBuilder::new(&influence)
            .with_noise_config(noise.clone())
            .with_detectors_json(r#"[{"id":0,"records":[-2]},{"id":1,"records":[-1]}]"#)
            .unwrap()
            .build();
        match mode {
            ReplacementBranchApproximation::BranchImpact => {
                assert!(dem.unwrap().to_string().contains("error(0.0582) D1"));
            }
            ReplacementBranchApproximation::ExactBranchReplay => assert!(
                dem.unwrap_err()
                    .to_string()
                    .contains("requires a circuit-aware exact branch provider")
            ),
            _ => unreachable!(),
        }
        let result = MemBuilder::new(&influence)
            .with_noise_config(noise.clone())
            .build();
        match result {
            Ok(_) => omitted_modes.push(mode),
            Err(error) => assert!(error.to_string().contains("replacement")),
        }
        // Replacement branches have their own rate path in the DEM, so a zero
        // ordinary gate override is not proof that they can be omitted.
        let quiet = noise.clone().set_p2_gate_rate(GateType::CX, 0.0);
        assert!(
            MemBuilder::new(&influence)
                .with_noise_config(quiet)
                .build()
                .is_err()
        );
        let post_gate_only = noise.set_p2_weights(PauliWeights::from([(X(1), 1.0)]));
        assert!(
            MemBuilder::new(&influence)
                .with_noise_config(post_gate_only)
                .build()
                .is_ok()
        );
    }
    assert!(
        omitted_modes.is_empty(),
        "MEM silently omitted replacement modes: {omitted_modes:?}"
    );
}
