use pecos_core::Angle64;
use pecos_quantum::DagCircuit;
use pecos_zx::convert::{dag_to_zx, dag_to_zx_circuit};
use quizx::tensor::ToTensor;

#[test]
fn phase_u_zx_export() {
    // -0.37 radians has odd fraction 17360464587495211133, which Rational64
    // cannot represent exactly in half turns. Check rejection, then exercise
    // tensor conversion at the adjacent representable fraction (one tick less).
    let mut unrepresentable = DagCircuit::new();
    unrepresentable.u(
        Angle64::ZERO,
        Angle64::ZERO,
        pecos_core::radians!(-0.37),
        &[0],
    );
    assert!(
        dag_to_zx_circuit(&unrepresentable)
            .unwrap_err()
            .to_string()
            .contains("17360464587495211133")
    );
    for angle in [Angle64::HALF_TURN, Angle64::new(17_360_464_587_495_211_132)] {
        let mut dag = DagCircuit::new();
        dag.u(Angle64::ZERO, Angle64::ZERO, angle, &[0, 1]);
        let circuit = dag_to_zx_circuit(&dag).unwrap();
        assert_eq!(circuit.gates.len(), 2);
        assert!(
            circuit
                .gates
                .iter()
                .all(|gate| gate.t == quizx::gate::GType::ZPhase)
        );
        let graph = dag_to_zx(&dag).unwrap();
        for tensor in [circuit.to_tensor64(), graph.to_tensor64()] {
            for (index, actual) in tensor.iter().enumerate() {
                let row = index / 4;
                let col = index % 4;
                let expected = if row == col {
                    let radians = angle.to_radians_signed() * f64::from(row.count_ones());
                    (radians.cos(), radians.sin())
                } else {
                    (0.0, 0.0)
                };
                assert!((actual.re - expected.0).abs() < 1e-9);
                assert!((actual.im - expected.1).abs() < 1e-9);
            }
        }
    }
}
