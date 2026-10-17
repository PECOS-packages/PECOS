use pecos_qec::fault_tolerance::InfluenceBuilder;
use pecos_qec::geometry::{PauliOp, StabilizerCheck};
use pecos_quantum::DagCircuit;

#[test]
fn clifford_expanded_support_preserves_logical_width_errors() {
    let code = pecos_qec::StabilizerCode::repetition(2)
        .apply_clifford(&pecos_core::clifford_rep::CliffordRep::cx(2, 1));
    assert_eq!(code.num_qubits(), 2);
    let expected = pecos_quantum::SymplecticMatrixError::OperatorExceedsWidth {
        row: 0,
        qubit: 2,
        num_qubits: 2,
    };
    assert_eq!(code.logical_operators().unwrap_err(), expected);
    assert_eq!(code.distance().unwrap_err(), expected);
}

#[test]
fn influence_x_duplicates_return_errors() {
    let dag = DagCircuit::new();
    let error = InfluenceBuilder::new(&dag)
        .unwrap()
        .with_x(&[7, 7])
        .err()
        .unwrap();
    assert!(error.to_string().contains("qubit 7"));
    assert!(error.to_string().contains("merge"));
}

#[test]
fn influence_y_duplicates_return_errors() {
    let dag = DagCircuit::new();
    let error = InfluenceBuilder::new(&dag)
        .unwrap()
        .with_y(&[7, 7])
        .err()
        .unwrap();
    assert!(error.to_string().contains("qubit 7"));
    assert!(error.to_string().contains("merge"));
}

#[test]
fn influence_z_duplicates_return_errors() {
    let dag = DagCircuit::new();
    let error = InfluenceBuilder::new(&dag)
        .unwrap()
        .with_z(&[7, 7])
        .err()
        .unwrap();
    assert!(error.to_string().contains("qubit 7"));
    assert!(error.to_string().contains("merge"));
}

#[test]
fn stabilizer_check_duplicates_return_errors() {
    let error = StabilizerCheck::new(0, vec![PauliOp::x(7), PauliOp::z(7)]).unwrap_err();
    assert!(error.contains("qubit 7"));
    assert!(error.contains("merge"));
}
