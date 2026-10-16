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

fn assert_broadcast(name: &str, angles: &[f64], expected: &pecos_core::Gate) {
    let mut builder = ByteMessageBuilder::new();
    OperationProcessor::new()
        .add_quantum_operation_to_builder(&mut builder, name, &[1, 3], angles)
        .unwrap();
    let actual = builder.build().quantum_ops().unwrap();
    let mut expected_builder = ByteMessageBuilder::new();
    expected_builder.add_gate_command(expected);
    assert_eq!(actual, expected_builder.build().quantum_ops().unwrap());
}

#[test]
fn x_broadcasts_to_every_argument() {
    assert_broadcast("X", &[], &pecos_core::Gate::x(&[1, 3]));
}

#[test]
fn rz_broadcasts_to_every_argument() {
    assert_broadcast(
        "RZ",
        &[0.5],
        &pecos_core::Gate::rz(pecos_core::radians!(0.5), &[1, 3]),
    );
}

#[test]
fn phir_engine_broadcasts_through_quantum_execution() {
    use pecos_engines::{ClassicalEngine, Engine, StateVecEngine};
    use pecos_phir_json::PhirJsonEngine;
    let program = serde_json::json!({
        "format":"PHIR/JSON", "version":"0.1.0", "ops":[
            {"data":"qvar_define","data_type":"qubits","variable":"q","size":4},
            {"data":"cvar_define","data_type":"u32","variable":"m","size":4},
            {"qop":"Init","args":[["q",0],["q",1],["q",2],["q",3]]},
            {"block":"qparallel", "ops":[
                {"qop":"X","args":[["q",1],["q",3]]},
                {"qop":"R1XY","angles":[[1.0,0.0],"pi"],"args":[["q",0],["q",2]]}
            ]},
            {"qop":"H","args":[["q",0],["q",2]]},
            {"qop":"RZ","angles":[[1.0],"pi"],"args":[["q",0],["q",2]]},
            {"qop":"H","args":[["q",0],["q",2]]},
            {"qop":"Measure","args":[["q",0],["q",1],["q",2],["q",3]],
             "returns":[["m",0],["m",1],["m",2],["m",3]]}
        ]
    });
    let mut classical = PhirJsonEngine::from_json(&program.to_string()).unwrap();
    let mut quantum = StateVecEngine::new(4);
    let results = quantum
        .process(classical.generate_commands().unwrap())
        .unwrap();
    assert_eq!(results.outcomes().unwrap(), vec![0, 1, 0, 1]);
    classical.handle_measurements(results).unwrap();
}
