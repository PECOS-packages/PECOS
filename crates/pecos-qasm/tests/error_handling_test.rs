// Test cases for error handling in QASM parsing and execution
use pecos_engines::engines::classical::ClassicalEngine;
use pecos_qasm::QASMEngine;

#[test]
fn test_qubit_index_out_of_bounds() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        x q[4];
    "#;

    let mut engine = QASMEngine::new().unwrap();

    // First check if parsing succeeds
    let parse_result = engine.from_str(qasm);

    if parse_result.is_ok() {
        // If parsing succeeds, the error might be caught during execution
        // Let's try to execute the program
        match engine.generate_commands() {
            Ok(_) => {
                panic!("Expected error for out-of-bounds qubit index during execution");
            }
            Err(e) => {
                let error_msg = format!("{:?}", e);
                println!("Execution error: {}", error_msg);
                // Verify it's the right kind of error
                assert!(
                    error_msg.contains("out of bounds") ||
                    error_msg.contains("index") ||
                    error_msg.contains("4"),
                    "Error should mention out-of-bounds index: {}",
                    error_msg
                );
            }
        }
    } else if let Err(e) = parse_result {
        // Check that the parsing error mentions the issue
        let error_msg = format!("{:?}", e);
        println!("Parse error: {}", error_msg);
        assert!(
            error_msg.contains("out of bounds") ||
            error_msg.contains("index") ||
            error_msg.contains("4"),
            "Error should mention out-of-bounds index: {}",
            error_msg
        );
    }
}

#[test]
fn test_valid_qubit_indices() {
    // This should work fine - using valid indices
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        rz(1.5*pi) q[0];
        rz(1.5*pi) q[1];
        rz(1.5*pi) q[2];
    "#;

    let mut engine = QASMEngine::new().unwrap();
    let result = engine.from_str(qasm);
    
    assert!(result.is_ok(), "Should succeed with valid qubit indices");
}

#[test]
fn test_classical_register_out_of_bounds() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];

        // This should fail - c only has indices 0 and 1
        c[2] = 1;
    "#;

    let mut engine = QASMEngine::new().unwrap();
    let parse_result = engine.from_str(qasm);

    if parse_result.is_ok() {
        // If parsing succeeds, the error might be caught during execution
        match engine.generate_commands() {
            Ok(_) => {
                panic!("Expected error for out-of-bounds classical register during execution");
            }
            Err(e) => {
                let error_msg = format!("{:?}", e);
                println!("Execution error: {}", error_msg);
                // Verify it's the right kind of error
                assert!(
                    error_msg.contains("out of bounds") ||
                    error_msg.contains("index") ||
                    error_msg.contains("2"),
                    "Error should mention out-of-bounds index: {}",
                    error_msg
                );
            }
        }
    } else if let Err(e) = parse_result {
        let error_msg = format!("{:?}", e);
        println!("Parse error: {}", error_msg);
        assert!(
            error_msg.contains("out of bounds") ||
            error_msg.contains("index") ||
            error_msg.contains("2"),
            "Error should mention out-of-bounds index: {}",
            error_msg
        );
    }
}

#[test]
fn test_measure_to_out_of_bounds_classical() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];

        // This should fail - c only has indices 0 and 1
        measure q[0] -> c[2];
    "#;

    let mut engine = QASMEngine::new().unwrap();
    let parse_result = engine.from_str(qasm);

    if parse_result.is_ok() {
        // If parsing succeeds, the error might be caught during execution
        match engine.generate_commands() {
            Ok(_) => {
                panic!("Expected error for out-of-bounds classical register in measurement");
            }
            Err(e) => {
                let error_msg = format!("{:?}", e);
                println!("Execution error: {}", error_msg);
                // Verify it's the right kind of error
                assert!(
                    error_msg.contains("out of bounds") ||
                    error_msg.contains("index") ||
                    error_msg.contains("2"),
                    "Error should mention out-of-bounds index: {}",
                    error_msg
                );
            }
        }
    } else if let Err(e) = parse_result {
        let error_msg = format!("{:?}", e);
        println!("Parse error: {}", error_msg);
        assert!(
            error_msg.contains("out of bounds") ||
            error_msg.contains("index") ||
            error_msg.contains("2"),
            "Error should mention out-of-bounds index: {}",
            error_msg
        );
    }
}

#[test]
fn test_negative_register_size() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[-1];
    "#;

    let mut engine = QASMEngine::new().unwrap();
    let result = engine.from_str(qasm);
    
    assert!(result.is_err(), "Expected error for negative register size");
}

#[test]
fn test_gate_on_nonexistent_register() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];

        // This should fail - register 'p' doesn't exist
        x p[0];
    "#;

    let mut engine = QASMEngine::new().unwrap();
    let parse_result = engine.from_str(qasm);

    if parse_result.is_ok() {
        // If parsing succeeds, the error might be caught during execution
        match engine.generate_commands() {
            Ok(_) => {
                panic!("Expected error for gate on non-existent register");
            }
            Err(e) => {
                let error_msg = format!("{:?}", e);
                println!("Execution error: {}", error_msg);
                // Verify it's the right kind of error
                assert!(
                    error_msg.contains("not found") ||
                    error_msg.contains("register") ||
                    error_msg.contains("p"),
                    "Error should mention non-existent register: {}",
                    error_msg
                );
            }
        }
    } else if let Err(e) = parse_result {
        let error_msg = format!("{:?}", e);
        println!("Parse error: {}", error_msg);
        assert!(
            error_msg.contains("not found") ||
            error_msg.contains("register") ||
            error_msg.contains("p"),
            "Error should mention non-existent register: {}",
            error_msg
        );
    }
}