use pecos_qasm::QASMParser;

#[test]
#[allow(clippy::too_many_lines)]
fn test_ten_qubit_quantum_algorithm() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[10];
        creg c[10];
        rz(0.5*pi) q[0];
        rz(0.5*pi) q[1];
        rz(0.5*pi) q[2];
        rz(0.5*pi) q[3];
        rz(0.5*pi) q[4];
        rz(0.5*pi) q[5];
        rz(0.5*pi) q[6];
        rz(0.5*pi) q[7];
        rz(0.5*pi) q[8];
        rz(0.5*pi) q[9];
        rx(0.5*pi) q[0];
        rx(0.5*pi) q[1];
        rx(0.5*pi) q[2];
        rx(0.5*pi) q[3];
        rx(0.5*pi) q[4];
        rx(0.5*pi) q[5];
        rx(0.5*pi) q[6];
        rx(0.5*pi) q[7];
        rx(0.5*pi) q[8];
        rx(0.5*pi) q[9];
        rz(0.5*pi) q[0];
        rz(0.5*pi) q[1];
        rz(0.5*pi) q[2];
        rz(0.5*pi) q[3];
        rz(0.5*pi) q[4];
        rz(0.5*pi) q[5];
        rz(0.5*pi) q[6];
        rz(0.5*pi) q[7];
        rz(0.5*pi) q[8];
        rz(0.5*pi) q[9];
        rx(1.7830369077719694*pi) q[0];
        rx(1.7830369077719694*pi) q[1];
        rx(1.7830369077719694*pi) q[2];
        rx(1.7830369077719694*pi) q[3];
        rx(1.7830369077719694*pi) q[4];
        rx(1.7830369077719694*pi) q[5];
        rx(1.7830369077719694*pi) q[6];
        rx(1.7830369077719694*pi) q[7];
        rx(1.7830369077719694*pi) q[8];
        rx(1.7830369077719694*pi) q[9];
        rz(0.5*pi) q[0];
        rx(0.5*pi) q[0];
        rz(0.5*pi) q[0];
        cz q[1],q[0];
        rz(0.5*pi) q[0];
        rx(0.5*pi) q[0];
        rz(0.5*pi) q[0];
        rz(1.8683763286244195*pi) q[0];
        rz(0.5*pi) q[0];
        rx(0.5*pi) q[0];
        rz(0.5*pi) q[0];
        cz q[1],q[0];
        rz(0.5*pi) q[0];
        rz(0.5*pi) q[1];
        rx(0.5*pi) q[0];
        rx(0.5*pi) q[1];
        rz(0.5*pi) q[0];
        rz(0.5*pi) q[1];
        rz(0.5*pi) q[0];
        cz q[5],q[1];
        rx(0.5*pi) q[0];
        rz(0.5*pi) q[1];
        rz(0.5*pi) q[0];
        rx(0.5*pi) q[1];
        cz q[7],q[0];
        rz(0.5*pi) q[1];
        rz(0.5*pi) q[0];
        rz(1.8683763286244195*pi) q[1];
        rx(0.5*pi) q[0];
        rz(0.5*pi) q[1];
        rz(0.5*pi) q[0];
        rx(0.5*pi) q[1];
        rz(1.8683763286244195*pi) q[0];
        rz(0.5*pi) q[1];
        rz(0.5*pi) q[0];
        cz q[5],q[1];
        rx(0.5*pi) q[0];
        rz(0.5*pi) q[1];
        rz(0.5*pi) q[0];
        rx(0.5*pi) q[1];
        cz q[7],q[0];
        rz(0.5*pi) q[1];
        rz(0.5*pi) q[0];
        rz(0.5*pi) q[1];
        rz(0.5*pi) q[7];
        rx(0.5*pi) q[0];
        rx(0.5*pi) q[1];
        rx(0.5*pi) q[7];
        rz(0.5*pi) q[0];
        rz(0.5*pi) q[1];
        rz(0.5*pi) q[7];
        rz(0.5*pi) q[0];
        cz q[8],q[1];
        cz q[2],q[7];
        rx(0.5*pi) q[0];
        rz(0.5*pi) q[1];
        rz(0.5*pi) q[7];
        rz(0.5*pi) q[0];
        rx(0.5*pi) q[1];
        rx(0.5*pi) q[7];
        cz q[6],q[0];
        rz(0.5*pi) q[1];
        rz(0.5*pi) q[7];
        rz(0.5*pi) q[0];
        rz(1.8683763286244195*pi) q[1];
        rz(1.8683763286244195*pi) q[7];
        rx(0.5*pi) q[0];
        rz(0.5*pi) q[1];
        rz(0.5*pi) q[7];
        rz(0.5*pi) q[0];
        rx(0.5*pi) q[1];
        rx(0.5*pi) q[7];
        rz(1.8683763286244195*pi) q[0];
        rz(0.5*pi) q[1];
        rz(0.5*pi) q[7];
        rz(0.5*pi) q[0];
        cz q[8],q[1];
        cz q[2],q[7];
        rx(0.5*pi) q[0];
        rz(0.5*pi) q[1];
        rz(0.5*pi) q[2];
        rz(0.5*pi) q[7];
        rz(0.5*pi) q[0];
        rx(0.5*pi) q[1];
        rx(0.5*pi) q[2];
        rx(0.5*pi) q[7];
        cz q[6],q[0];
        rz(0.5*pi) q[1];
        rz(0.5*pi) q[2];
        rz(0.5*pi) q[7];
        rz(0.5*pi) q[0];
        rx(1.85548120216805*pi) q[1];
        cz q[4],q[2];
        rz(0.5*pi) q[7];
        rx(0.5*pi) q[0];
        rz(0.5*pi) q[2];
        rx(0.5*pi) q[7];
        rz(0.5*pi) q[0];
        rx(0.5*pi) q[2];
        rz(0.5*pi) q[7];
        rx(1.85548120216805*pi) q[0];
        rz(0.5*pi) q[2];
        cz q[9],q[7];
        rz(0.5*pi) q[0];
        rz(1.8683763286244195*pi) q[2];
        rz(0.5*pi) q[7];
        rx(0.5*pi) q[0];
        rz(0.5*pi) q[2];
        rx(0.5*pi) q[7];
        rz(0.5*pi) q[0];
        rx(0.5*pi) q[2];
        rz(0.5*pi) q[7];
        cz q[1],q[0];
        rz(0.5*pi) q[2];
        rz(1.8683763286244195*pi) q[7];
        rz(0.5*pi) q[0];
        cz q[4],q[2];
        rz(0.5*pi) q[7];
        rx(0.5*pi) q[0];
        rz(0.5*pi) q[2];
        rz(0.5*pi) q[4];
        rx(0.5*pi) q[7];
        rz(0.5*pi) q[0];
        rx(0.5*pi) q[2];
        rx(0.5*pi) q[4];
        rz(0.5*pi) q[7];
        rz(1.7942353647778524*pi) q[0];
        rz(0.5*pi) q[2];
        rz(0.5*pi) q[4];
        cz q[9],q[7];
        rz(0.5*pi) q[0];
        rz(0.5*pi) q[2];
        cz q[3],q[4];
        rz(0.5*pi) q[7];
        rx(0.5*pi) q[0];
        rx(0.5*pi) q[2];
        rz(0.5*pi) q[4];
        rx(0.5*pi) q[7];
        rz(0.5*pi) q[0];
        rz(0.5*pi) q[2];
        rx(0.5*pi) q[4];
        rz(0.5*pi) q[7];
        cz q[1],q[0];
        cz q[9],q[2];
        rz(0.5*pi) q[4];
        rx(1.85548120216805*pi) q[7];
        rz(0.5*pi) q[0];
        rz(0.5*pi) q[1];
        rz(0.5*pi) q[2];
        rz(1.8683763286244195*pi) q[4];
        rx(0.5*pi) q[0];
        rx(0.5*pi) q[1];
        rx(0.5*pi) q[2];
        rz(0.5*pi) q[4];
        rz(0.5*pi) q[0];
        rz(0.5*pi) q[1];
        rz(0.5*pi) q[2];
        rx(0.5*pi) q[4];
        rz(0.5*pi) q[0];
        rz(1.8683763286244195*pi) q[2];
        rz(0.5*pi) q[4];
        rx(0.5*pi) q[0];
        rz(0.5*pi) q[2];
        cz q[3],q[4];
        rz(0.5*pi) q[0];
        rx(0.5*pi) q[2];
        rz(0.5*pi) q[3];
        rz(0.5*pi) q[4];
        cz q[7],q[0];
        rz(0.5*pi) q[2];
        rx(0.5*pi) q[3];
        rx(0.5*pi) q[4];
        rz(0.5*pi) q[0];
        cz q[9],q[2];
        rz(0.5*pi) q[3];
        rz(0.5*pi) q[4];
        rx(0.5*pi) q[0];
        rz(0.5*pi) q[2];
        cz q[8],q[3];
        rz(0.5*pi) q[4];
        rz(0.5*pi) q[0];
        rx(0.5*pi) q[2];
        rz(0.5*pi) q[3];
        rx(0.5*pi) q[4];
        rz(1.7942353647778524*pi) q[0];
        rz(0.5*pi) q[2];
        rx(0.5*pi) q[3];
        rz(0.5*pi) q[4];
        rz(0.5*pi) q[0];
        rx(1.85548120216805*pi) q[2];
        rz(0.5*pi) q[3];
        cz q[6],q[4];
        rx(0.5*pi) q[0];
        rz(1.8683763286244195*pi) q[3];
        rz(0.5*pi) q[4];
        rz(0.5*pi) q[0];
        rz(0.5*pi) q[3];
        rx(0.5*pi) q[4];
        cz q[7],q[0];
        rx(0.5*pi) q[3];
        rz(0.5*pi) q[4];
        rz(0.5*pi) q[0];
        rz(0.5*pi) q[3];
        rz(1.8683763286244195*pi) q[4];
        rz(0.5*pi) q[7];
        rx(0.5*pi) q[0];
        cz q[8],q[3];
        rz(0.5*pi) q[4];
        rx(0.5*pi) q[7];
        rz(0.5*pi) q[0];
        rz(0.5*pi) q[3];
        rx(0.5*pi) q[4];
        rz(0.5*pi) q[7];
        rz(0.5*pi) q[8];
        rz(0.5*pi) q[0];
        cz q[2],q[7];
        rx(0.5*pi) q[3];
        rz(0.5*pi) q[4];
        rx(0.5*pi) q[8];
        rx(0.5*pi) q[0];
        rz(0.5*pi) q[3];
        cz q[6],q[4];
        rz(0.5*pi) q[7];
        rz(0.5*pi) q[8];
        rz(0.5*pi) q[0];
        rz(0.5*pi) q[3];
        rz(0.5*pi) q[4];
        rz(0.5*pi) q[6];
        rx(0.5*pi) q[7];
        rx(0.5*pi) q[3];
        rx(0.5*pi) q[4];
        rx(0.5*pi) q[6];
        rz(0.5*pi) q[7];
        rz(0.5*pi) q[3];
        rz(0.5*pi) q[4];
        rz(0.5*pi) q[6];
        rz(1.7942353647778524*pi) q[7];
        cz q[5],q[3];
        rx(1.85548120216805*pi) q[4];
        cz q[9],q[6];
        rz(0.5*pi) q[7];
        rz(0.5*pi) q[3];
        rz(0.5*pi) q[6];
        rx(0.5*pi) q[7];
        rx(0.5*pi) q[3];
        rx(0.5*pi) q[6];
        rz(0.5*pi) q[7];
        cz q[2],q[7];
        rz(0.5*pi) q[3];
        rz(0.5*pi) q[6];
        rz(0.5*pi) q[2];
        rz(1.8683763286244195*pi) q[3];
        rz(1.8683763286244195*pi) q[6];
        rz(0.5*pi) q[7];
        rx(0.5*pi) q[2];
        rz(0.5*pi) q[3];
        rz(0.5*pi) q[6];
        rx(0.5*pi) q[7];
        rz(0.5*pi) q[2];
        rx(0.5*pi) q[3];
        rx(0.5*pi) q[6];
        rz(0.5*pi) q[7];
        cz q[4],q[2];
        rz(0.5*pi) q[3];
        rz(0.5*pi) q[6];
        rz(0.5*pi) q[7];
        rz(0.5*pi) q[2];
        cz q[5],q[3];
        cz q[9],q[6];
        rx(0.5*pi) q[7];
        rx(0.5*pi) q[2];
        rz(0.5*pi) q[3];
        cz q[5],q[8];
        rz(0.5*pi) q[6];
        rz(0.5*pi) q[7];
        rx(1.85548120216805*pi) q[9];
        rz(0.5*pi) q[2];
        rx(0.5*pi) q[3];
        rx(0.5*pi) q[6];
        cz q[9],q[7];
        rz(0.5*pi) q[8];
        rz(1.7942353647778524*pi) q[2];
        rz(0.5*pi) q[3];
        rz(0.5*pi) q[6];
        rz(0.5*pi) q[7];
        rx(0.5*pi) q[8];
        rz(0.5*pi) q[2];
        rx(1.85548120216805*pi) q[3];
        rx(1.85548120216805*pi) q[6];
        rx(0.5*pi) q[7];
        rz(0.5*pi) q[8];
        cz q[6],q[0];
        rx(0.5*pi) q[2];
        rz(0.5*pi) q[7];
        rz(1.8683763286244195*pi) q[8];
        rz(0.5*pi) q[0];
        rz(0.5*pi) q[2];
        rz(1.7942353647778524*pi) q[7];
        rz(0.5*pi) q[8];
        rx(0.5*pi) q[0];
        cz q[4],q[2];
        rz(0.5*pi) q[7];
        rx(0.5*pi) q[8];
        rz(0.5*pi) q[0];
        rz(0.5*pi) q[2];
        rz(0.5*pi) q[4];
        rx(0.5*pi) q[7];
        rz(0.5*pi) q[8];
        rz(1.7942353647778524*pi) q[0];
        rx(0.5*pi) q[2];
        rx(0.5*pi) q[4];
        cz q[5],q[8];
        rz(0.5*pi) q[7];
        rz(0.5*pi) q[0];
        rz(0.5*pi) q[2];
        rz(0.5*pi) q[4];
        rx(1.85548120216805*pi) q[5];
        cz q[9],q[7];
        rz(0.5*pi) q[8];
        rx(0.5*pi) q[0];
        cz q[5],q[1];
        rz(0.5*pi) q[2];
        cz q[3],q[4];
        rz(0.5*pi) q[7];
        rx(0.5*pi) q[8];
        rz(0.5*pi) q[0];
        rz(0.5*pi) q[1];
        rx(0.5*pi) q[2];
        rz(0.5*pi) q[4];
        rx(0.5*pi) q[7];
        rz(0.5*pi) q[8];
        cz q[6],q[0];
        rx(0.5*pi) q[1];
        rz(0.5*pi) q[2];
        rx(0.5*pi) q[4];
        rz(0.5*pi) q[7];
        rx(1.85548120216805*pi) q[8];
        rz(0.5*pi) q[0];
        rz(0.5*pi) q[1];
        cz q[9],q[2];
        rz(0.5*pi) q[4];
        rx(0.5*pi) q[0];
        rz(1.7942353647778524*pi) q[1];
        rz(0.5*pi) q[2];
        rz(1.7942353647778524*pi) q[4];
        rz(0.5*pi) q[0];
        rz(0.5*pi) q[1];
        rx(0.5*pi) q[2];
        rz(0.5*pi) q[4];
        rx(0.5*pi) q[1];
        rz(0.5*pi) q[2];
        rx(0.5*pi) q[4];
        rz(0.5*pi) q[1];
        rz(1.7942353647778524*pi) q[2];
        rz(0.5*pi) q[4];
        cz q[5],q[1];
        rz(0.5*pi) q[2];
        cz q[3],q[4];
        rz(0.5*pi) q[1];
        rx(0.5*pi) q[2];
        rz(0.5*pi) q[3];
        rz(0.5*pi) q[4];
        rx(0.5*pi) q[1];
        rz(0.5*pi) q[2];
        rx(0.5*pi) q[3];
        rx(0.5*pi) q[4];
        rz(0.5*pi) q[1];
        cz q[9],q[2];
        rz(0.5*pi) q[3];
        rz(0.5*pi) q[4];
        rz(0.5*pi) q[1];
        rz(0.5*pi) q[2];
        rz(0.5*pi) q[4];
        rx(0.5*pi) q[1];
        rx(0.5*pi) q[2];
        rx(0.5*pi) q[4];
        rz(0.5*pi) q[1];
        rz(0.5*pi) q[2];
        rz(0.5*pi) q[4];
        cz q[8],q[1];
        cz q[6],q[4];
        rz(0.5*pi) q[1];
        rz(0.5*pi) q[4];
        rx(0.5*pi) q[1];
        rx(0.5*pi) q[4];
        rz(0.5*pi) q[1];
        rz(0.5*pi) q[4];
        rz(1.7942353647778524*pi) q[1];
        rz(1.7942353647778524*pi) q[4];
        rz(0.5*pi) q[1];
        rz(0.5*pi) q[4];
        rx(0.5*pi) q[1];
        rx(0.5*pi) q[4];
        rz(0.5*pi) q[1];
        rz(0.5*pi) q[4];
        cz q[8],q[1];
        cz q[6],q[4];
        rz(0.5*pi) q[1];
        cz q[8],q[3];
        rz(0.5*pi) q[4];
        rz(0.5*pi) q[6];
        rx(0.5*pi) q[1];
        rz(0.5*pi) q[3];
        rx(0.5*pi) q[4];
        rx(0.5*pi) q[6];
        rz(0.5*pi) q[1];
        rx(0.5*pi) q[3];
        rz(0.5*pi) q[4];
        rz(0.5*pi) q[6];
        rz(0.5*pi) q[3];
        cz q[9],q[6];
        rz(1.7942353647778524*pi) q[3];
        rz(0.5*pi) q[6];
        rz(0.5*pi) q[3];
        rx(0.5*pi) q[6];
        rx(0.5*pi) q[3];
        rz(0.5*pi) q[6];
        rz(0.5*pi) q[3];
        rz(1.7942353647778524*pi) q[6];
        cz q[8],q[3];
        rz(0.5*pi) q[6];
        rz(0.5*pi) q[3];
        rx(0.5*pi) q[6];
        rz(0.5*pi) q[8];
        rx(0.5*pi) q[3];
        rz(0.5*pi) q[6];
        rx(0.5*pi) q[8];
        rz(0.5*pi) q[3];
        cz q[9],q[6];
        rz(0.5*pi) q[8];
        rz(0.5*pi) q[3];
        rz(0.5*pi) q[6];
        rx(0.5*pi) q[3];
        rx(0.5*pi) q[6];
        rz(0.5*pi) q[3];
        rz(0.5*pi) q[6];
        cz q[5],q[3];
        rz(0.5*pi) q[3];
        rx(0.5*pi) q[3];
        rz(0.5*pi) q[3];
        rz(1.7942353647778524*pi) q[3];
        rz(0.5*pi) q[3];
        rx(0.5*pi) q[3];
        rz(0.5*pi) q[3];
        cz q[5],q[3];
        rz(0.5*pi) q[3];
        cz q[5],q[8];
        rx(0.5*pi) q[3];
        rz(0.5*pi) q[8];
        rz(0.5*pi) q[3];
        rx(0.5*pi) q[8];
        rz(0.5*pi) q[8];
        rz(1.7942353647778524*pi) q[8];
        rz(0.5*pi) q[8];
        rx(0.5*pi) q[8];
        rz(0.5*pi) q[8];
        cz q[5],q[8];
        rz(0.5*pi) q[8];
        rx(0.5*pi) q[8];
        rz(0.5*pi) q[8];
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse 10-qubit algorithm");

    // Verify the circuit structure
    assert!(
        !program.operations.is_empty(),
        "Should have many operations"
    );
    assert_eq!(
        program.quantum_registers.len(),
        1,
        "Should have one quantum register"
    );
    assert_eq!(
        program.classical_registers.len(),
        1,
        "Should have one classical register"
    );
    assert_eq!(
        program.quantum_registers["q"].len(),
        10,
        "Should have 10 qubits"
    );
    assert_eq!(
        program.classical_registers["c"], 10,
        "Should have 10 classical bits"
    );
}

#[test]
fn test_cz_gate_dense_connectivity() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[10];

        // Test dense CZ connectivity pattern
        cz q[1],q[0];
        cz q[5],q[1];
        cz q[7],q[0];
        cz q[5],q[1];
        cz q[7],q[0];
        cz q[8],q[1];
        cz q[2],q[7];
        cz q[8],q[1];
        cz q[2],q[7];
        cz q[4],q[2];
        cz q[4],q[2];
        cz q[3],q[4];
        cz q[3],q[4];
        cz q[9],q[2];
        cz q[9],q[2];
        cz q[6],q[0];
        cz q[6],q[0];
        cz q[9],q[7];
        cz q[9],q[7];
        cz q[6],q[4];
        cz q[6],q[4];
        cz q[5],q[3];
        cz q[5],q[3];
        cz q[5],q[8];
        cz q[8],q[1];
        cz q[8],q[1];
        cz q[8],q[3];
        cz q[8],q[3];
        cz q[5],q[3];
        cz q[5],q[8];
        cz q[9],q[6];
        cz q[9],q[6];
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse CZ connectivity test");

    // Count CZ operations
    let cz_count = program
        .operations
        .iter()
        .filter(|op| matches!(op, pecos_qasm::Operation::Gate { name, .. } if name == "cz"))
        .count();

    assert_eq!(
        cz_count, 0,
        "CZ gates should be expanded and not appear directly"
    );

    // But we should have many operations from the expansions
    assert!(
        !program.operations.is_empty(),
        "Should have operations from CZ expansions"
    );
}

#[test]
fn test_precision_angle_values() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[4];

        // Test various high-precision angle values
        rx(1.7830369077719694*pi) q[0];
        rx(1.85548120216805*pi) q[1];
        rz(1.8683763286244195*pi) q[2];
        rz(1.7942353647778524*pi) q[3];

        // These precise values might be from circuit optimization
        // or error mitigation calibration
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse precision angle test");

    // Just verify it parses correctly with high precision values
    assert!(
        !program.operations.is_empty(),
        "Should have operations with precise angles"
    );
}

#[test]
fn test_phase_pattern_structure() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[3];

        // Common pattern: RZ-RX-RZ sequence (ZXZ decomposition)
        rz(0.5*pi) q[0];
        rx(0.5*pi) q[0];
        rz(0.5*pi) q[0];

        // Same pattern on another qubit
        rz(0.5*pi) q[1];
        rx(0.5*pi) q[1];
        rz(0.5*pi) q[1];

        // Entangling gate
        cz q[1],q[0];

        // Another phase pattern
        rz(0.5*pi) q[2];
        rx(0.5*pi) q[2];
        rz(0.5*pi) q[2];
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse phase pattern test");

    // Verify the structure parses correctly
    assert!(
        !program.operations.is_empty(),
        "Should have phase pattern operations"
    );
}
