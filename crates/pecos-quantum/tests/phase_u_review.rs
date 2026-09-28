use pecos_core::Angle64;
use pecos_core::gate_type::GateType;
use pecos_quantum::TickCircuit;
use pecos_quantum::pass::{CircuitPass, SimplifyRotations, SimplifySingleQubitCliffordChains};
use pecos_synth::{GateToken, Matrix, OmegaExponent};

fn chain_matrix(circuit: &TickCircuit) -> Matrix {
    circuit
        .iter_gate_batches()
        .fold(Matrix::identity(), |product, gate| {
            let token = match gate.gate_type {
                GateType::X => GateToken::X,
                GateType::Z => GateToken::Z,
                GateType::I => GateToken::I,
                GateType::SZ => GateToken::SZ,
                GateType::SZdg => GateToken::SZdg,
                GateType::SY | GateType::SYdg => {
                    let phase = if gate.gate_type == GateType::SY {
                        GateToken::SZ
                    } else {
                        GateToken::SZdg
                    };
                    return Matrix::from_word(&[
                        GateToken::SZdg,
                        GateToken::H,
                        phase,
                        GateToken::H,
                        GateToken::SZ,
                    ]) * product;
                }
                GateType::F | GateType::Fdg => {
                    let f = Matrix::from_word(&[
                        GateToken::H,
                        GateToken::SZ,
                        GateToken::H,
                        GateToken::SZ,
                    ])
                    .with_global_phase(OmegaExponent::new(2));
                    return if gate.gate_type == GateType::F {
                        f
                    } else {
                        f.adjoint()
                    } * product;
                }
                GateType::Y => GateToken::Y,
                GateType::SX | GateType::SXdg => {
                    let phase = if gate.gate_type == GateType::SX {
                        GateToken::SZ
                    } else {
                        GateToken::SZdg
                    };
                    return Matrix::from_word(&[GateToken::H, phase, GateToken::H]) * product;
                }
                GateType::H => GateToken::H,
                GateType::U => {
                    assert_eq!(gate.as_gate().phase_angle(), Some(Angle64::HALF_TURN));
                    // The specified operator diag(1, exp(i*pi)) is exactly Z.
                    GateToken::Z
                }
                other => panic!("unexpected gate in test chain: {other:?}"),
            };
            Matrix::from_gate(token) * product
        })
}

#[test]
fn clifford_chain_preserves_operator_including_scalar() {
    for phase_u in [true, false] {
        let mut circuit = TickCircuit::new();
        for _ in 0..2 {
            circuit.tick().x(&[0]);
            if phase_u {
                circuit
                    .tick()
                    .u(Angle64::ZERO, Angle64::ZERO, Angle64::HALF_TURN, &[0]);
            } else {
                circuit.tick().z(&[0]);
            }
        }
        let original = circuit.clone();
        SimplifyRotations.apply_tick(&mut circuit);
        SimplifySingleQubitCliffordChains.apply_tick(&mut circuit);
        let before = chain_matrix(&original);
        assert_eq!(
            before,
            Matrix::identity().with_global_phase(OmegaExponent::new(4))
        );
        assert_eq!(before, chain_matrix(&circuit), "phase_u={phase_u}");
        assert_eq!(
            circuit.gate_count(),
            4,
            "scalar-changing substitution must be refused"
        );
    }
    let mut safe = TickCircuit::new();
    safe.tick().h(&[0]);
    safe.tick().h(&[0]);
    let before = chain_matrix(&safe);
    SimplifySingleQubitCliffordChains.apply_tick(&mut safe);
    assert_eq!(before, chain_matrix(&safe));
    assert_eq!(
        safe.gate_count(),
        0,
        "operator-preserving substitution should proceed"
    );
}

#[cfg(feature = "hugr")]
#[test]
fn phase_u_hugr_import_rejects_unrepresentable_scalar() {
    use pecos_core::Gate;
    use pecos_quantum::{
        DagCircuit,
        hugr_convert::{SimpleHugr, dag_circuit_to_hugr, hugr_to_dag_circuit},
    };
    let mut dag = DagCircuit::new();
    dag.add_gate_auto_wire(Gate::u(
        Angle64::ZERO,
        Angle64::ZERO,
        Angle64::QUARTER_TURN,
        &[0],
    ));
    let hugr = dag_circuit_to_hugr(&dag).unwrap();
    let error = hugr_to_dag_circuit(&hugr).expect_err("must not discard global phase");
    assert!(error.to_string().contains("global_phase"));
    assert!(SimpleHugr::new_relaxed(hugr).is_err());
}

#[test]
fn clifford_chain_selects_exact_shorter_candidate() {
    let mut circuit = TickCircuit::new();
    circuit.tick().x(&[0]);
    circuit.tick().y(&[0]);
    circuit.tick().sx(&[0]);
    let before = chain_matrix(&circuit);
    SimplifySingleQubitCliffordChains.apply_tick(&mut circuit);
    assert_eq!(before, chain_matrix(&circuit));
    assert_eq!(circuit.gate_count(), 2);
    assert_eq!(
        circuit
            .iter_gate_batches()
            .map(|g| g.gate_type)
            .collect::<Vec<_>>(),
        vec![GateType::SXdg, GateType::Z]
    );
}

#[cfg(feature = "hugr")]
#[test]
fn phase_u_zero_hugr_round_trip_preserves_operator() {
    use pecos_core::Gate;
    use pecos_quantum::unitary_matrix::{ToMatrix, UnitaryMatrix};
    use pecos_quantum::{
        DagCircuit,
        hugr_convert::{SimpleHugr, dag_circuit_to_hugr, hugr_to_dag_circuit},
    };
    let mut dag = DagCircuit::new();
    dag.add_gate_auto_wire(Gate::u(Angle64::ZERO, Angle64::ZERO, Angle64::ZERO, &[0]));
    let hugr = dag_circuit_to_hugr(&dag).unwrap();
    let round_trip = hugr_to_dag_circuit(&hugr).unwrap();
    assert_eq!(round_trip.iter_gates().count(), 1);
    assert!(SimpleHugr::new_relaxed(hugr).is_ok());
    // Both specified matrices are exactly the identity; no numerical phase quotient.
    for (_, gate) in round_trip.iter_gates() {
        assert_eq!(gate.gate_type, GateType::RZ);
        assert_eq!(gate.angles.as_slice(), &[Angle64::ZERO]);
        assert_eq!(
            pecos_core::unitary_rep::RZ(gate.angles[0], 0)
                .to_matrix()
                .inner(),
            UnitaryMatrix::identity(2).inner()
        );
    }
}

#[cfg(feature = "hugr")]
#[test]
fn hugr_scalar_requires_provably_even_half_turns() {
    use pecos_quantum::hugr_convert::hugr_to_dag_circuit;
    use tket::extension::{
        global_phase::GlobalPhase,
        rotation::{ConstRotation, RotationOp, rotation_type},
    };
    use tket::hugr::{
        builder::{DFGBuilder, Dataflow, DataflowHugr},
        extension::prelude::qb_t,
        types::Signature,
    };
    // Include the smallest subnormal: dividing by two before checking would
    // wrongly erase it. Integral full turns, including negative ones, are safe.
    for (half_turns, trivial) in [
        (0.0, true),
        (2.0, true),
        (-4.0, true),
        (1.0, false),
        (0.25, false),
        (f64::from_bits(1), false),
    ] {
        for computed in [false, true] {
            let mut builder = DFGBuilder::new(Signature::new(vec![qb_t()], vec![qb_t()])).unwrap();
            let [qubit] = builder.input_wires_arr();
            let mut phase = builder.add_load_value(ConstRotation::new(half_turns).unwrap());
            if computed {
                let zero = builder.add_load_value(ConstRotation::new(0.0).unwrap());
                phase = builder
                    .add_dataflow_op(RotationOp::radd, [phase, zero])
                    .unwrap()
                    .out_wire(0);
            }
            builder
                .add_dataflow_op(GlobalPhase.into_extension_op(), [phase])
                .unwrap();
            let hugr = builder.finish_hugr_with_outputs([qubit]).unwrap();
            assert_eq!(
                hugr_to_dag_circuit(&hugr).is_ok(),
                trivial && !computed,
                "half_turns={half_turns}, computed={computed}"
            );
        }
    }
    let mut builder =
        DFGBuilder::new(Signature::new(vec![qb_t(), rotation_type()], vec![qb_t()])).unwrap();
    let [qubit, phase] = builder.input_wires_arr();
    builder
        .add_dataflow_op(GlobalPhase.into_extension_op(), [phase])
        .unwrap();
    let hugr = builder.finish_hugr_with_outputs([qubit]).unwrap();
    assert!(
        hugr_to_dag_circuit(&hugr)
            .unwrap_err()
            .to_string()
            .contains("not provably constant")
    );
}

#[test]
fn clifford_chain_covers_longer_exact_representatives() {
    use pecos_core::Gate;
    for (gates, expected_score) in [
        (
            vec![
                GateType::H,
                GateType::SYdg,
                GateType::F,
                GateType::H,
                GateType::H,
            ],
            None,
        ),
        (
            vec![GateType::SYdg, GateType::SXdg, GateType::H, GateType::H],
            Some((1, 3)),
        ),
        (vec![GateType::SYdg, GateType::SXdg], Some((2, 2))),
    ] {
        let mut circuit = TickCircuit::new();
        for &gate in &gates {
            circuit
                .tick()
                .try_add_gate(Gate::simple(gate, vec![0.into()]))
                .unwrap();
        }
        let before = chain_matrix(&circuit);
        SimplifySingleQubitCliffordChains.apply_tick(&mut circuit);
        assert_eq!(chain_matrix(&circuit), before);
        let result = circuit
            .iter_gate_batches()
            .map(|g| g.gate_type)
            .collect::<Vec<_>>();
        let score = (
            result
                .iter()
                .filter(|g| !matches!(g, GateType::I | GateType::Z | GateType::SZ | GateType::SZdg))
                .count(),
            result.len(),
        );
        if let Some(expected) = expected_score {
            assert_eq!(score, expected, "{result:?}");
        } else {
            assert!(result.len() < gates.len(), "{result:?}");
        }
        assert!(
            result.len() <= gates.len(),
            "replacement must fit available positions"
        );
    }
}

#[cfg(feature = "hugr")]
#[test]
fn hugr_ordering_edge_cannot_prove_runtime_phase_trivial() {
    use pecos_quantum::hugr_convert::{SimpleHugr, hugr_to_dag_circuit};
    use tket::extension::{global_phase::GlobalPhase, rotation::RotationOp};
    use tket::hugr::{
        HugrView,
        builder::{DFGBuilder, Dataflow, DataflowHugr},
        extension::prelude::qb_t,
        hugr::hugrmut::HugrMut,
        std_extensions::arithmetic::float_types::{ConstF64, float64_type},
        types::Signature,
    };
    let mut builder = DFGBuilder::new(Signature::new(
        vec![qb_t(), float64_type()],
        vec![qb_t(), float64_type()],
    ))
    .unwrap();
    let [qubit, runtime] = builder.input_wires_arr();
    let unrelated_zero = builder.add_load_value(ConstF64::new(0.0));
    let phase = builder
        .add_dataflow_op(RotationOp::from_halfturns_unchecked, [runtime])
        .unwrap()
        .out_wire(0);
    builder
        .add_dataflow_op(GlobalPhase.into_extension_op(), [phase])
        .unwrap();
    let mut hugr = builder
        .finish_hugr_with_outputs([qubit, unrelated_zero])
        .unwrap();
    let ordering_output = hugr
        .get_optype(unrelated_zero.node())
        .other_output_port()
        .unwrap();
    let ordering_input = hugr.get_optype(phase.node()).other_input_port().unwrap();
    hugr.connect(
        unrelated_zero.node(),
        ordering_output,
        phase.node(),
        ordering_input,
    );
    hugr.validate().unwrap();
    assert!(
        hugr_to_dag_circuit(&hugr).is_err(),
        "runtime half-turn 1 contributes -1, not a trivial scalar"
    );
    assert!(SimpleHugr::new_relaxed(hugr).is_err());
}

#[cfg(feature = "hugr")]
#[test]
fn hugr_deep_computed_phase_is_rejected_cleanly() {
    use pecos_quantum::hugr_convert::{SimpleHugr, hugr_to_dag_circuit};
    use tket::extension::{
        global_phase::GlobalPhase,
        rotation::{ConstRotation, RotationOp},
    };
    use tket::hugr::{
        HugrView,
        builder::{DFGBuilder, Dataflow, DataflowHugr},
        extension::prelude::qb_t,
        types::Signature,
    };
    let mut builder = DFGBuilder::new(Signature::new(vec![qb_t()], vec![qb_t()])).unwrap();
    let [qubit] = builder.input_wires_arr();
    let zero = builder.add_load_value(ConstRotation::new(0.0).unwrap());
    let mut phase = zero;
    for _ in 0..10_000 {
        phase = builder
            .add_dataflow_op(RotationOp::radd, [phase, zero])
            .unwrap()
            .out_wire(0);
    }
    builder
        .add_dataflow_op(GlobalPhase.into_extension_op(), [phase])
        .unwrap();
    let hugr = builder.finish_hugr_with_outputs([qubit]).unwrap();
    hugr.validate().unwrap();
    assert!(hugr_to_dag_circuit(&hugr).is_err());
    assert!(SimpleHugr::new_relaxed(hugr).is_err());
}
