use pecos_simulators::{PauliProp, QuantumSimulator, StateVec, StateVecAoS, StateVecSoA};

#[test]
fn tracked_extent_overflow_returns_error() {
    let mut prop = PauliProp::new();
    prop.track_x(&[usize::MAX]);
    assert!(
        prop.num_qubits()
            .unwrap_err()
            .to_string()
            .contains(&usize::MAX.to_string())
    );
}

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

#[test]
fn aos_growth_returns_error_without_mutation() {
    let mut sim = StateVecAoS::new(1);
    let requested = usize::MAX;
    let message = sim.ensure_qubit_count(requested).unwrap_err().to_string();
    assert!(message.contains(&requested.to_string()));
    assert!(message.contains("maximum"));
    assert_eq!(sim.num_qubits(), 1);
}

#[test]
fn soa_growth_returns_error_without_mutation() {
    let mut sim = StateVecSoA::new(1);
    let requested = usize::BITS as usize;
    let message = sim.ensure_qubit_count(requested).unwrap_err().to_string();
    assert!(message.contains(&requested.to_string()));
    assert!(message.contains("maximum"));
    assert_eq!(sim.num_qubits(), 1);
}

#[test]
fn soa_capacity_returns_error_without_mutation() {
    let mut sim = StateVecSoA::new(1);
    let requested = usize::BITS as usize - 2;
    let message = sim.ensure_qubit_count(requested).unwrap_err().to_string();
    assert!(message.contains(&requested.to_string()));
    assert!(message.contains("maximum"));
    assert_eq!(sim.num_qubits(), 1);
}

#[test]
fn soa_pending_capacity_returns_error_without_mutation() {
    let mut sim = StateVecSoA::new(1);
    let requested = usize::MAX;
    let message = sim.ensure_qubit_count(requested).unwrap_err().to_string();
    assert!(message.contains(&requested.to_string()));
    assert!(message.contains("maximum"));
    assert_eq!(sim.num_qubits(), 1);
}

#[test]
fn sparse_frames_capacity_returns_error_without_mutation() {
    let mut sim = StateVec::new(1);
    let requested = usize::MAX;
    let message = sim.ensure_qubit_count(requested).unwrap_err().to_string();
    assert!(message.contains(&requested.to_string()));
    assert!(message.contains("maximum"));
    assert_eq!(sim.num_qubits(), 1);
}
