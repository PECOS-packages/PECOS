use pecos_engines::byte_message::ByteMessageBuilder;
use pecos_phir_json::v0_1::operations::OperationProcessor;

fn check_counts(gate: &str, qubits: usize, angles: usize) {
    let mut builder = ByteMessageBuilder::new();
    let error = OperationProcessor::new()
        .add_quantum_operation_to_builder(
            &mut builder,
            gate,
            &(0..qubits).collect::<Vec<_>>(),
            &vec![0.0; angles],
        )
        .unwrap_err();
    let message = error.to_string();
    assert!(message.contains(gate));
    assert!(message.contains(&format!("{angles} angles")));
    assert!(message.contains(&format!("{qubits} qubits")));
    assert!(message.contains("expected"));
}

#[test]
fn rz_missing_angle_0() {
    check_counts("RZ", 1, 0);
}

#[test]
fn rz_missing_qubits() {
    check_counts("RZ", 0, 1);
}

#[test]
fn r1xy_missing_angle_0() {
    check_counts("R1XY", 1, 0);
}

#[test]
fn r1xy_missing_angle_1() {
    check_counts("R1XY", 1, 1);
}

#[test]
fn r1xy_missing_qubits() {
    check_counts("R1XY", 0, 2);
}

#[test]
fn rxxryyrzz_missing_angle_0() {
    check_counts("RXXRYYRZZ", 2, 0);
}

#[test]
fn rxxryyrzz_missing_angle_1() {
    check_counts("RXXRYYRZZ", 2, 1);
}

#[test]
fn rxxryyrzz_missing_angle_2() {
    check_counts("RXXRYYRZZ", 2, 2);
}

#[test]
fn rxxryyrzz_missing_qubits() {
    check_counts("RXXRYYRZZ", 0, 3);
}
