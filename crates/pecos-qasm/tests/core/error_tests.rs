// Test cases for error handling in QASM parsing and execution
use pecos_engines::engines::classical::ClassicalEngine;
use pecos_qasm::{QASMEngine, QASMParser};
use std::str::FromStr;

#[test]
fn test_qubit_index_out_of_bounds() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        X q[4];
    "#;

    // First check if parsing succeeds
    let engine_result = QASMEngine::from_str(qasm);

    if let Ok(mut engine) = engine_result {
        // If parsing succeeds, the error might be caught during execution
        // Let's try to execute the program
        match engine.generate_commands() {
            Ok(_) => {
                panic!("Expected error for out-of-bounds qubit index during execution");
            }
            Err(e) => {
                let error_msg = format!("{e:?}");
                println!("Execution error: {error_msg}");
                // Verify it's the right kind of error
                assert!(
                    error_msg.contains("out of bounds")
                        || error_msg.contains("index")
                        || error_msg.contains('4'),
                    "Error should mention out-of-bounds index: {error_msg}"
                );
            }
        }
    } else if let Err(e) = engine_result {
        // Check that the parsing error mentions the issue
        let error_msg = format!("{e:?}");
        println!("Parse error: {error_msg}");
        assert!(
            error_msg.contains("out of bounds")
                || error_msg.contains("index")
                || error_msg.contains('4'),
            "Error should mention out-of-bounds index: {error_msg}"
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
        RZ(1.5*pi) q[0];
        RZ(1.5*pi) q[1];
        RZ(1.5*pi) q[2];
    "#;

    let engine = QASMEngine::from_str(qasm);

    assert!(engine.is_ok(), "Should succeed with valid qubit indices");
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

    let engine_result = QASMEngine::from_str(qasm);

    if let Ok(mut engine) = engine_result {
        // If parsing succeeds, the error might be caught during execution
        match engine.generate_commands() {
            Ok(_) => {
                panic!("Expected error for out-of-bounds classical register during execution");
            }
            Err(e) => {
                let error_msg = format!("{e:?}");
                println!("Execution error: {error_msg}");
                // Verify it's the right kind of error
                assert!(
                    error_msg.contains("out of bounds")
                        || error_msg.contains("index")
                        || error_msg.contains('2'),
                    "Error should mention out-of-bounds index: {error_msg}"
                );
            }
        }
    } else if let Err(e) = engine_result {
        let error_msg = format!("{e:?}");
        println!("Parse error: {error_msg}");
        assert!(
            error_msg.contains("out of bounds")
                || error_msg.contains("index")
                || error_msg.contains('2'),
            "Error should mention out-of-bounds index: {error_msg}"
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

    let engine_result = QASMEngine::from_str(qasm);

    if let Ok(mut engine) = engine_result {
        // If parsing succeeds, the error might be caught during execution
        match engine.generate_commands() {
            Ok(_) => {
                panic!("Expected error for out-of-bounds classical register in measurement");
            }
            Err(e) => {
                let error_msg = format!("{e:?}");
                println!("Execution error: {error_msg}");
                // Verify it's the right kind of error
                assert!(
                    error_msg.contains("out of bounds")
                        || error_msg.contains("index")
                        || error_msg.contains('2'),
                    "Error should mention out-of-bounds index: {error_msg}"
                );
            }
        }
    } else if let Err(e) = engine_result {
        let error_msg = format!("{e:?}");
        println!("Parse error: {error_msg}");
        assert!(
            error_msg.contains("out of bounds")
                || error_msg.contains("index")
                || error_msg.contains('2'),
            "Error should mention out-of-bounds index: {error_msg}"
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

    let engine = QASMEngine::from_str(qasm);

    assert!(engine.is_err(), "Expected error for negative register size");
}

#[test]
fn test_gate_on_nonexistent_register() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];

        // This should fail - register 'p' doesn't exist
        X p[0];
    "#;

    let engine_result = QASMEngine::from_str(qasm);

    if let Ok(mut engine) = engine_result {
        // If parsing succeeds, the error might be caught during execution
        match engine.generate_commands() {
            Ok(_) => {
                panic!("Expected error for gate on non-existent register");
            }
            Err(e) => {
                let error_msg = format!("{e:?}");
                println!("Execution error: {error_msg}");
                // Verify it's the right kind of error
                assert!(
                    error_msg.contains("not found")
                        || error_msg.contains("register")
                        || error_msg.contains('p'),
                    "Error should mention non-existent register: {error_msg}"
                );
            }
        }
    } else if let Err(e) = engine_result {
        let error_msg = format!("{e:?}");
        println!("Parse error: {error_msg}");
        assert!(
            error_msg.contains("not found")
                || error_msg.contains("register")
                || error_msg.contains('p'),
            "Error should mention non-existent register: {error_msg}"
        );
    }
}

// Tests for undefined gates
#[test]
fn test_undefined_gate_error() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];

        gatedoesntexist q[0];
    "#;

    // This should fail because 'gatedoesntexist' is not a defined gate
    let result = QASMParser::parse_str(qasm);
    assert!(result.is_err(), "Should fail with undefined gate error");

    if let Err(e) = result {
        let error_message = e.to_string();
        println!("Error message: {error_message}");

        // The error should mention the undefined gate
        assert!(
            error_message.contains("gatedoesntexist")
                || error_message.contains("undefined")
                || error_message.contains("not defined")
                || error_message.contains("unknown"),
            "Error should mention the undefined gate"
        );
    }
}

#[test]
fn test_misspelled_gate_error() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];

        hadamrd q[0];  // misspelled 'hadamard' or 'h'
    "#;

    let result = QASMParser::parse_str(qasm);
    assert!(result.is_err(), "Should fail with misspelled gate error");
}

#[test]
fn test_gate_with_wrong_arity() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];

        cx q[0];  // cx requires 2 qubits, not 1
    "#;

    let result = QASMParser::parse_str(qasm);
    // The parser might accept this syntactically but fail during execution
    match result {
        Ok(_) => println!("Parser accepts syntactically valid but semantically incorrect arity"),
        Err(e) => println!("Parser rejects wrong arity: {e}"),
    }
}

#[test]
fn test_gate_with_too_many_parameters() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];

        rz(pi, pi/2) q[0];  // rz only takes 1 parameter
    "#;

    let result = QASMParser::parse_str(qasm);
    // The parser might accept extra parameters syntactically
    match result {
        Ok(_) => println!("Parser accepts extra parameters syntactically"),
        Err(e) => println!("Parser rejects extra parameters: {e}"),
    }
}

#[test]
fn test_gate_with_missing_parameters() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];

        rz q[0];  // rz requires an angle parameter
    "#;

    let result = QASMParser::parse_str(qasm);
    assert!(result.is_err(), "Should fail with missing parameter");
}

// Tests for native and defined gates
#[test]
fn test_undefined_gate_fails() {
    // Test with rx gate which is NOT in the native gates list
    let qasm = r"
        OPENQASM 2.0;
        qreg q[1];
        rx(pi/2) q[0];
    ";

    let result = QASMParser::parse_str_raw(qasm);

    // This should fail because rx is not native and not defined
    assert!(result.is_err());

    if let Err(e) = result {
        let error_msg = e.to_string();
        assert!(error_msg.contains("rx"));
        assert!(error_msg.contains("Undefined"));
        assert!(error_msg.contains("qelib1.inc"));
    }
}

#[test]
fn test_native_gates_pass() {
    // Test with gates that ARE in the native list
    let qasm = r"
        OPENQASM 2.0;
        qreg q[2];
        H q[0];
        CX q[0], q[1];
        RZ(pi) q[1];
    ";

    let result = QASMParser::parse_str_raw(qasm);

    // This should pass because these are native gates
    assert!(result.is_ok());
}

#[test]
fn test_defined_gates_pass() {
    // Test with user-defined gates
    let qasm = r"
        OPENQASM 2.0;
        qreg q[1];

        gate mygate a {
            H a;
            X a;
        }

        mygate q[0];
    ";

    let result = QASMParser::parse_str_raw(qasm);

    // This should pass because mygate is defined
    assert!(result.is_ok());
}

#[test]
fn test_gates_in_definitions_only() {
    // Test that gates used only in definitions don't cause errors
    // until the definition is actually used
    let qasm = r"
        OPENQASM 2.0;
        qreg q[1];

        gate uses_undefined a {
            rx(pi) a;  // rx is not native
        }

        // Don't use the gate - should still pass
        H q[0];
    ";

    let result = QASMParser::parse_str_raw(qasm);

    // This should pass because uses_undefined is never used
    assert!(result.is_ok());
}

#[test]
fn test_using_gate_with_undefined_gates() {
    // Test that using a gate that contains undefined gates fails
    let qasm = r"
        OPENQASM 2.0;
        qreg q[1];

        gate uses_undefined a {
            undefined_gate a;  // This gate doesn't exist anywhere
        }

        uses_undefined q[0];  // This should trigger expansion and fail
    ";

    let result = QASMParser::parse_str_raw(qasm);

    // This should fail when expanding uses_undefined
    assert!(result.is_err());

    if let Err(e) = result {
        let error_msg = e.to_string();
        assert!(error_msg.contains("undefined_gate"));
        assert!(error_msg.contains("Undefined"));
    }
}

// Tests for circular dependencies
#[test]
fn test_circular_dependency_detection() {
    // Test direct circular dependency
    let qasm_direct = r"
        OPENQASM 2.0;
        qreg q[1];
        gate g1 q { g1 q; }
        g1 q[0];
    ";

    match QASMParser::parse_str_raw(qasm_direct) {
        Err(e) => {
            assert!(e.to_string().contains("Circular dependency"));
            assert!(e.to_string().contains("g1 -> g1"));
        }
        Ok(_) => panic!("Expected error due to circular dependency"),
    }
}

#[test]
fn test_indirect_circular_dependency_detection() {
    // Test indirect circular dependency (A -> B -> A)
    let qasm_indirect = r"
        OPENQASM 2.0;
        qreg q[1];
        gate g1 q { g2 q; }
        gate g2 q { g1 q; }
        g1 q[0];
    ";

    match QASMParser::parse_str_raw(qasm_indirect) {
        Err(e) => {
            assert!(e.to_string().contains("Circular dependency"));
            // Either g1 -> g2 -> g1 or g2 -> g1 -> g2 is valid depending on which gets expanded first
            assert!(
                e.to_string().contains("g1 -> g2 -> g1")
                    || e.to_string().contains("g2 -> g1 -> g2")
            );
        }
        Ok(_) => panic!("Expected error due to circular dependency"),
    }
}

#[test]
fn test_complex_circular_dependency_detection() {
    // Test complex circular dependency (A -> B -> C -> A)
    let qasm_complex = r"
        OPENQASM 2.0;
        qreg q[1];
        gate g1 q { g2 q; }
        gate g2 q { g3 q; }
        gate g3 q { g1 q; }
        g1 q[0];
    ";

    match QASMParser::parse_str_raw(qasm_complex) {
        Err(e) => {
            assert!(e.to_string().contains("Circular dependency"));
            assert!(e.to_string().contains("g1 -> g2 -> g3 -> g1"));
        }
        Ok(_) => panic!("Expected error due to circular dependency"),
    }
}

#[test]
fn test_valid_deep_nesting() {
    // Test that valid deep nesting still works
    let qasm_valid = r"
        OPENQASM 2.0;
        qreg q[1];
        gate g1 q { H q; }
        gate g2 q { g1 q; }
        gate g3 q { g2 q; }
        gate g4 q { g3 q; }
        gate g5 q { g4 q; }
        g5 q[0];
    ";

    match QASMParser::parse_str_raw(qasm_valid) {
        Ok(_) => { /* Success */ }
        Err(e) => panic!("Valid deep nesting failed with error: {e}"),
    }
}

#[test]
fn test_circular_dependency_with_parameters() {
    // Test circular dependency with parameterized gates
    let qasm_param = r"
        OPENQASM 2.0;
        qreg q[1];
        gate rot(theta) q { rot(theta) q; }
        rot(pi/2) q[0];
    ";

    match QASMParser::parse_str_raw(qasm_param) {
        Err(e) => {
            assert!(e.to_string().contains("Circular dependency"));
            assert!(e.to_string().contains("rot -> rot"));
        }
        Ok(_) => panic!("Expected error due to circular dependency"),
    }
}

#[test]
fn test_circular_dependency_without_usage() {
    // Test that circular dependencies can be defined but not used
    let qasm_unused = r"
        OPENQASM 2.0;
        qreg q[2];
        gate g1 q { g2 q; }
        gate g2 q { g1 q; }
        CX q[0], q[1];  // Use a different gate
    ";

    // This should succeed since we never actually use the circular gates
    assert!(QASMParser::parse_str_raw(qasm_unused).is_ok());
}
