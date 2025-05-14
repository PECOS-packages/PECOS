
// Test extended gate support in PECOS QASM
use pecos_qasm::QASMEngine;
use pecos_engines::engines::classical::ClassicalEngine;

#[test]
fn test_basic_rotation_gates() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        
        // Test RZ gate
        rz(pi/2) q[0];
        
        // Test S and T gates 
        s q[0];
        sdg q[0];
        t q[0];
        tdg q[0];
    "#;

    let mut engine = QASMEngine::new().unwrap();
    let result = engine.from_str(qasm);
    
    assert!(result.is_ok(), "Should successfully parse rotation gates");
}

#[test]
fn test_two_qubit_rotations() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        
        // Test RZZ gate with parameter
        rzz(pi/4) q[0], q[1];
        
        // Test SZZ gate
        szz q[0], q[1];
    "#;

    let mut engine = QASMEngine::new().unwrap();
    let result = engine.from_str(qasm);
    
    assert!(result.is_ok(), "Should successfully parse two-qubit rotation gates");
}

#[test]
fn test_decomposed_gates() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        
        // Test gates that are decomposed from the qelib1 library
        cz q[0], q[1];
        cy q[0], q[1];
        swap q[0], q[1];
    "#;

    let mut engine = QASMEngine::new().unwrap();
    let result = engine.from_str(qasm);
    
    assert!(result.is_ok(), "Should successfully parse decomposed gates");
}

#[test]
fn test_parameterized_gates() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        
        // Test parameterized gates
        rz(pi) q[0];
        rz(pi/2) q[0];
        rz(0.7854) q[0];  // pi/4 in decimal
    "#;

    let mut engine = QASMEngine::new().unwrap();
    let result = engine.from_str(qasm);
    
    assert!(result.is_ok(), "Should successfully parse parameterized gates");
}

#[test]
fn test_unsupported_gate_error() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        
        // This should fail - Toffoli is not supported
        ccx q[0], q[1], q[2];
    "#;

    let mut engine = QASMEngine::new().unwrap();
    let result = engine.from_str(qasm);
    
    // The gate should be parsed but fail during execution
    assert!(result.is_ok(), "Should parse unsupported gates");
    
    // But execution should fail
    match engine.generate_commands() {
        Ok(_) => panic!("Should fail on unsupported gate"),
        Err(e) => {
            let error_msg = format!("{:?}", e);
            assert!(error_msg.contains("Unsupported") || error_msg.contains("ccx"),
                    "Error should mention unsupported gate: {}", error_msg);
        }
    }
}