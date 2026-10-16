use pecos_simulators::PauliProp;

#[test]
fn tracker_dense_string_width_returns_error() {
    let mut prop = PauliProp::new();
    prop.track_z(&[5]);
    let error = prop.dense_string_width(3).unwrap_err();
    assert_eq!(error.qubit, 5);
    assert_eq!(error.num_qubits, 3);
    assert_eq!(prop.dense_string(), "IIIIIZ");
}

#[test]
fn tracker_to_dense_string_width_returns_error() {
    let mut prop = PauliProp::new();
    prop.track_z(&[5]);
    let error = prop.to_dense_string_width(3).unwrap_err();
    assert_eq!(error.qubit, 5);
    assert_eq!(error.num_qubits, 3);
    assert_eq!(prop.dense_string(), "IIIIIZ");
}
