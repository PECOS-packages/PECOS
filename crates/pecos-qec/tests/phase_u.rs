use pecos_core::gate_type::GateType;
use pecos_core::{Angle64, Gate};
use pecos_engines::ByteMessage;
use pecos_phir::execution::processor::PhirProcessor;
use pecos_phir::ops::{Operation, QuantumOp, SSAValue};
use pecos_phir::phir::Instruction;
use pecos_qec::fault_tolerance::dem_builder::{DemBuilder, NoiseConfig};
use pecos_quantum::DagCircuit;

#[test]
fn phase_u_dem_including_phir_cphase() {
    let instruction = Instruction::new(
        Operation::Quantum(QuantumOp::CPhase(Angle64::HALF_TURN)),
        vec![SSAValue::new(0), SSAValue::new(1)],
        vec![],
        vec![],
    );
    let mut builder = ByteMessage::quantum_operations_builder();
    PhirProcessor::new()
        .process_instruction(&instruction, &mut builder)
        .unwrap();
    let gates = builder.build().quantum_ops().unwrap();
    assert!(gates.iter().any(|gate| gate.gate_type == GateType::U));
    for body in [
        vec![Gate::u(
            Angle64::ZERO,
            Angle64::ZERO,
            Angle64::HALF_TURN,
            &[0],
        )],
        gates,
    ] {
        let mut dag = DagCircuit::new();
        dag.pz(&[0, 1]);
        for gate in body {
            dag.add_gate_auto_wire(gate);
        }
        dag.mz(&[0, 1]);
        DemBuilder::try_from_circuit_with_noise_config(
            &dag,
            NoiseConfig::uniform(0.01).set_p1_gate_rate(GateType::RZ, 0.0),
        )
        .unwrap();
    }
}

#[test]
fn phase_u_dem_inherits_rz_rate_overrides() {
    use pecos_qec::fault_tolerance::dem_builder::{DemSamplerBuilder, PerGateTypeNoise};
    use pecos_qec::fault_tolerance::propagator::DagFaultAnalyzer;
    let mut dag = DagCircuit::new();
    dag.pz(&[0]);
    dag.u(Angle64::ZERO, Angle64::ZERO, Angle64::QUARTER_TURN, &[0]);
    dag.mz(&[0]);
    let influence = DagFaultAnalyzer::new(&dag).build_influence_map();
    for rate in [0.0, 0.3] {
        let noise = NoiseConfig::new(0.6, 0.0, 0.0, 0.0).set_p1_gate_rate(GateType::RZ, rate);
        let dem = DemBuilder::new(&influence)
            .with_noise_config(noise.clone())
            .with_detectors_json(r#"[{"id":0,"records":[-1]}]"#)
            .unwrap()
            .build()
            .unwrap();
        assert_eq!(dem.num_contributions() == 0, rate == 0.0);
        let per_gate = PerGateTypeNoise::from_base_noise(noise.clone());
        let other = DemBuilder::new(&influence)
            .with_per_gate_noise(per_gate.clone())
            .with_detectors_json(r#"[{"id":0,"records":[-1]}]"#)
            .unwrap()
            .build()
            .unwrap();
        assert_eq!(dem.to_string(), other.to_string());
        let sampler = DemSamplerBuilder::new(&influence)
            .with_noise_config(noise)
            .with_detector_records(vec![vec![-1]])
            .build()
            .unwrap();
        let per_gate_sampler = DemSamplerBuilder::new(&influence)
            .with_per_gate_noise(per_gate)
            .with_detector_records(vec![vec![-1]])
            .build()
            .unwrap();
        assert_eq!(
            sampler
                .sample_statistics(128, 42)
                .per_detector
                .iter()
                .sum::<usize>()
                == 0,
            rate == 0.0
        );
        assert_eq!(
            per_gate_sampler
                .sample_statistics(128, 42)
                .per_detector
                .iter()
                .sum::<usize>()
                == 0,
            rate == 0.0
        );
    }
    for loc in influence
        .locations
        .iter()
        .filter(|loc| loc.gate_type == GateType::U)
    {
        assert_eq!(loc.noise_gate_type, GateType::RZ);
    }
}
