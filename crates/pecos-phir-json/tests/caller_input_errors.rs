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
    // Each pair finishes in |11>, up to global phases:
    // q0,q1: X|0> = |1>.
    // q2,q3: H Z H|0> = X|0> = |1>; Z is applied individually.
    // q4,q5: H RZ(pi) H|0> = -i X|0>; the H gates are individual.
    // q6,q7: R1XY(pi, 0)|0> = RX(pi)|0> = -i|1>.
    // If any broadcast gate KIND affects only its first argument, that pair's
    // second qubit instead stays |0>. For H both broadcast Hs disappear, leaving
    // Z|0>; for RZ the individual Hs cancel. No different gate kinds can cancel
    // a missing broadcast, and the same reasoning applies to either pair member.
    let program = serde_json::json!({
        "format":"PHIR/JSON", "version":"0.1.0", "ops":[
            {"data":"qvar_define","data_type":"qubits","variable":"q","size":8},
            {"data":"cvar_define","data_type":"u32","variable":"m","size":8},
            {"qop":"Init","args":[["q",0],["q",1],["q",2],["q",3],["q",4],["q",5],["q",6],["q",7]]},
            {"block":"qparallel", "ops":[
                {"qop":"X","args":[["q",0],["q",1]]},
                {"qop":"R1XY","angles":[[1.0,0.0],"pi"],"args":[["q",6],["q",7]]}
            ]},
            {"qop":"H","args":[["q",2],["q",3]]},
            {"qop":"Z","args":[["q",2]]},
            {"qop":"Z","args":[["q",3]]},
            {"qop":"H","args":[["q",2],["q",3]]},
            {"qop":"H","args":[["q",4]]},
            {"qop":"H","args":[["q",5]]},
            {"qop":"RZ","angles":[[1.0],"pi"],"args":[["q",4],["q",5]]},
            {"qop":"H","args":[["q",4]]},
            {"qop":"H","args":[["q",5]]},
            {"qop":"Measure","args":[["q",0],["q",1],["q",2],["q",3],["q",4],["q",5],["q",6],["q",7]],
             "returns":[["m",0],["m",1],["m",2],["m",3],["m",4],["m",5],["m",6],["m",7]]}
        ]
    });
    let mut classical = PhirJsonEngine::from_json(&program.to_string()).unwrap();
    let mut quantum = StateVecEngine::new(8);
    let results = quantum
        .process(classical.generate_commands().unwrap())
        .unwrap();
    assert_eq!(results.outcomes().unwrap(), vec![1; 8]);
    classical.handle_measurements(results).unwrap();
}
