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
