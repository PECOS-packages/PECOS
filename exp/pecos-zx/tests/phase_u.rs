use pecos_core::Angle64;
use pecos_quantum::DagCircuit;
use pecos_zx::convert::{dag_to_zx, dag_to_zx_circuit};
use quizx::tensor::ToTensor;

#[test]
fn phase_u_zx_export() {
    let negative = pecos_core::radians!(-0.37);
    assert_eq!(negative.fraction(), 17_360_464_587_495_211_133);
    // The negative magnitude is 1_086_279_486_214_340_483 ticks. Rounding
    // away from zero gives 1_086_279_486_214_340_484; dividing by two and
    // reducing yields the pinned half-turn rational below.
    for (angle, numerator, denominator) in [
        (Angle64::HALF_TURN, 1, 1),
        (negative, -271_569_871_553_585_121, 1_i64 << 61),
    ] {
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
        for gate in &circuit.gates {
            let rational = gate.phase.to_rational();
            assert_eq!(
                (*rational.numer(), *rational.denom()),
                (numerator, denominator)
            );
        }
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
