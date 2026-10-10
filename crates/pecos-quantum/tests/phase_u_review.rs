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
