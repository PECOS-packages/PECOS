use pecos_qec::fault_tolerance::InfluenceBuilder;
use pecos_qec::geometry::{PauliOp, StabilizerCheck};
use pecos_quantum::DagCircuit;

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
