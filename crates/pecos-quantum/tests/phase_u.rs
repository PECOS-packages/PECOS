use pecos_core::gate_type::GateType;
use pecos_core::{Angle64, CliffordLowering, Gate};
use pecos_engines::noise::GeneralNoiseModel;
use pecos_engines::quantum::StateVecEngine;
use pecos_engines::{ByteMessage, Engine};
use pecos_quantum::DagCircuit;

#[test]
fn phase_u_noise_exemption_and_general_u_faults() {
    let mut model = GeneralNoiseModel::builder()
        .with_seed(42)
        .with_p1(0.5)
        .with_noiseless_gate(GateType::RZ)
        .build();
    let noise = &mut model;
    for theta in [Angle64::ZERO, Angle64::HALF_TURN] {
        let mut ones = 0;
        for _ in 0..128 {
            let mut builder = ByteMessage::quantum_operations_builder();
            builder.u(theta, Angle64::ZERO, Angle64::QUARTER_TURN, &[0]);
            let noisy = noise.apply_noise_on_start(&builder.build()).unwrap();
            let mut sim = StateVecEngine::new(1);
            sim.process(noisy).unwrap();
            let mut measure = ByteMessage::quantum_operations_builder();
            measure.mz(&[0]);
            ones += sim.process(measure.build()).unwrap().outcomes().unwrap()[0];
        }
        if theta == Angle64::ZERO {
            assert_eq!(ones, 0, "phase U must inherit noiseless RZ");
        } else {
            assert!(
                ones > 0 && ones < 128,
                "general U must remain noisy: {ones}"
            );
        }
    }
}

#[test]
fn phase_u_exact_cliffords() {
    for (angle, named) in [
        (Angle64::ZERO, GateType::I),
        (Angle64::QUARTER_TURN, GateType::SZ),
        (Angle64::HALF_TURN, GateType::Z),
        (Angle64::THREE_QUARTERS_TURN, GateType::SZdg),
    ] {
        let phase = Gate::u(Angle64::ZERO, Angle64::ZERO, angle, &[0]);
        assert_eq!(
            pecos_core::try_lower_rotation_to_clifford(&phase),
            Some(CliffordLowering::Named(named))
        );
        for basis in 0..2 {
            let mut states = Vec::new();
            for gate in [phase.clone(), Gate::simple(named, vec![0.into()])] {
                let mut sim = StateVecEngine::new(1);
                let mut builder = ByteMessage::quantum_operations_builder();
                if basis == 1 {
                    builder.x(&[0]);
                }
                builder.add_gate_command(&gate);
                sim.process(builder.build()).unwrap();
                states.push(sim.simulator_mut().state().clone());
            }
            for (a, b) in states[0].iter().zip(&states[1]) {
                assert!((*a - *b).norm() < 1e-14, "matrix entry differs: {a} != {b}");
            }
        }
    }
}

#[cfg(feature = "hugr")]
#[test]
fn phase_u_hugr_export() {
    use pecos_quantum::hugr_convert::dag_circuit_to_hugr;
    use tket::hugr::HugrView;
    for angle in [Angle64::HALF_TURN, Angle64::from_radians(-0.37)] {
        let mut dag = DagCircuit::new();
        dag.u(Angle64::ZERO, Angle64::ZERO, angle, &[0, 1]);
        let hugr = dag_circuit_to_hugr(&dag).unwrap();
        hugr.validate().unwrap();
        let mut rotations = Vec::new();
        let mut global_phase = 0.0;
        let mut phase_count = 0;
        for node in hugr.nodes() {
            let Some(op) = hugr.get_optype(node).as_extension_op() else {
                continue;
            };
            let (port, is_phase) = match op.unqualified_id() {
                "Rz" => (1, false),
                "global_phase" => (0, true),
                other => panic!("unexpected exported operation {other}"),
            };
            let (load, _) = hugr.single_linked_output(node, port).unwrap();
            let (constant, _) = hugr.single_linked_output(load, 0).unwrap();
            let tket::hugr::ops::OpType::Const(value) = hugr.get_optype(constant) else {
                panic!("expected rotation constant")
            };
            let radians = value
                .get_custom_value::<tket::extension::rotation::ConstRotation>()
                .unwrap()
                .half_turns()
                * std::f64::consts::PI;
            if is_phase {
                global_phase += radians;
                phase_count += 1;
            } else {
                rotations.push(radians);
            }
        }
        assert_eq!(rotations.len(), 2);
        assert_eq!(phase_count, 2);
        for basis in 0_usize..4 {
            let actual = global_phase
                + rotations
                    .iter()
                    .enumerate()
                    .map(|(qubit, theta)| {
                        if basis & (1 << qubit) == 0 {
                            -theta / 2.0
                        } else {
                            theta / 2.0
                        }
                    })
                    .sum::<f64>();
            let expected = angle.to_radians_signed() * f64::from(basis.count_ones());
            assert!((actual.cos() - expected.cos()).abs() < 1e-14);
            assert!((actual.sin() - expected.sin()).abs() < 1e-14);
        }
    }
}

#[test]
fn phase_u_passes() {
    use pecos_quantum::pass::{
        CancelInverses, CircuitPass, MergeAdjacentRotations, StripIdentities,
    };
    let mut dag = DagCircuit::new();
    dag.u(Angle64::ZERO, Angle64::ZERO, Angle64::ZERO, &[0]);
    StripIdentities.apply_dag(&mut dag);
    assert_eq!(dag.gate_count(), 0);
    dag.u(Angle64::ZERO, Angle64::ZERO, Angle64::QUARTER_TURN, &[0]);
    dag.u(Angle64::ZERO, Angle64::ZERO, -Angle64::QUARTER_TURN, &[0]);
    CancelInverses.apply_dag(&mut dag);
    assert_eq!(dag.gate_count(), 0);
    dag.u(Angle64::ZERO, Angle64::ZERO, Angle64::QUARTER_TURN, &[0]);
    dag.u(Angle64::ZERO, Angle64::ZERO, Angle64::QUARTER_TURN, &[0]);
    MergeAdjacentRotations.apply_dag(&mut dag);
    assert_eq!(dag.gate_count(), 1);
    assert_eq!(
        dag.iter_gates_topo().next().unwrap().1.angles[2],
        Angle64::HALF_TURN
    );
}

#[test]
fn phase_u_recognition_is_exact() {
    for angle in [
        Angle64::ZERO,
        Angle64::HALF_TURN,
        Angle64::from_radians(-0.37),
    ] {
        let gate = Gate::u(Angle64::ZERO, Angle64::ZERO, angle, &[0, 1]);
        assert_eq!(gate.phase_angle(), Some(angle));
        for (theta, phi) in [
            (Angle64::from_turns(1e-12), Angle64::ZERO),
            (Angle64::ZERO, Angle64::from_turns(1e-12)),
        ] {
            assert_eq!(Gate::u(theta, phi, angle, &[0]).phase_angle(), None);
        }
    }
    assert_eq!(Gate::rz(Angle64::ZERO, &[0]).phase_angle(), None);
    for arity in [0, 1, 2, 4] {
        let mut gate = Gate::u(Angle64::ZERO, Angle64::ZERO, Angle64::ZERO, &[0]);
        gate.angles = vec![Angle64::ZERO; arity].into();
        assert_eq!(gate.phase_angle(), None);
    }
    for angle in [
        Angle64::from_turns(0.25 + 1e-12),
        Angle64::from_turn_ratio(1, 8),
    ] {
        assert_eq!(
            pecos_core::try_lower_rotation_to_clifford(&Gate::u(
                Angle64::ZERO,
                Angle64::ZERO,
                angle,
                &[0]
            )),
            None
        );
    }
}

#[test]
fn phase_u_tick_passes_and_general_u_barriers() {
    use pecos_quantum::TickCircuit;
    use pecos_quantum::pass::{
        AbsorbBasisGates, CancelInverses, CircuitPass, MergeAdjacentRotations, SimplifyRotations,
        StripIdentities,
    };
    let mut ticks = TickCircuit::new();
    ticks
        .tick()
        .u(Angle64::ZERO, Angle64::ZERO, Angle64::ZERO, &[0, 1]);
    StripIdentities.apply_tick(&mut ticks);
    assert_eq!(ticks.gate_count(), 0);
    ticks
        .tick()
        .u(Angle64::ZERO, Angle64::ZERO, Angle64::QUARTER_TURN, &[0, 1]);
    ticks.tick().u(
        Angle64::ZERO,
        Angle64::ZERO,
        -Angle64::QUARTER_TURN,
        &[0, 1],
    );
    CancelInverses.apply_tick(&mut ticks);
    assert_eq!(ticks.gate_count(), 0);
    for _ in 0..3 {
        ticks
            .tick()
            .u(Angle64::ZERO, Angle64::ZERO, Angle64::QUARTER_TURN, &[0, 1]);
    }
    MergeAdjacentRotations.apply_tick(&mut ticks);
    SimplifyRotations.apply_tick(&mut ticks);
    for (_, tick) in ticks.iter_ticks() {
        for gate in tick.iter_gate_batches() {
            assert_eq!(gate.gate_type, GateType::SZdg);
        }
    }
    let mut dag = DagCircuit::new();
    dag.pz(&[0]);
    dag.u(
        Angle64::ZERO,
        Angle64::ZERO,
        Angle64::from_radians(0.37),
        &[0],
    );
    AbsorbBasisGates.apply_dag(&mut dag);
    assert_eq!(dag.gate_count(), 1);
    for general_first in [true, false] {
        let mut dag = DagCircuit::new();
        for theta in if general_first {
            [Angle64::HALF_TURN, Angle64::ZERO]
        } else {
            [Angle64::ZERO, Angle64::HALF_TURN]
        } {
            dag.u(theta, Angle64::ZERO, Angle64::ZERO, &[0]);
        }
        MergeAdjacentRotations.apply_dag(&mut dag);
        CancelInverses.apply_dag(&mut dag);
        assert_eq!(dag.gate_count(), 2);
        StripIdentities.apply_dag(&mut dag);
        assert_eq!(dag.gate_count(), 1);
        assert_eq!(
            dag.iter_gates_topo().next().unwrap().1.angles[0],
            Angle64::HALF_TURN
        );
    }
}
