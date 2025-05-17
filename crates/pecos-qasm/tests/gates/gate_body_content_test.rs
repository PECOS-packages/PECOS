use pecos_qasm::QASMParser;

#[test]
fn test_gate_with_barrier_attempt() {
    // Test if barriers can be included in gate definitions
    let qasm = r"
        OPENQASM 2.0;
        qreg q[2];

        gate bell_with_barrier a, b {
            H a;
            barrier a, b;  // Can we include barriers?
            CX a, b;
        }

        bell_with_barrier q[0], q[1];
    ";

    let result = QASMParser::parse_str_raw(qasm);
    println!("Gate with barrier result: {:?}", result.is_ok());

    // This will likely fail with current grammar
    if let Err(e) = result {
        println!("Expected error: {e}");
    }
}

#[test]
fn test_gate_with_measurement_attempt() {
    // Test if measurements can be included in gate definitions
    let qasm = r"
        OPENQASM 2.0;
        qreg q[2];
        creg c[2];

        gate measure_gate a {
            H a;
            measure a -> c[0];  // This shouldn't be allowed
        }

        measure_gate q[0];
    ";

    let result = QASMParser::parse_str_raw(qasm);
    println!("Gate with measurement result: {:?}", result.is_ok());

    // This should definitely fail
    if let Err(e) = result {
        println!("Expected error: {e}");
    }
}

#[test]
fn test_gate_with_reset_attempt() {
    // Test if reset can be included in gate definitions
    let qasm = r"
        OPENQASM 2.0;
        qreg q[1];

        gate reset_gate a {
            reset a;  // Reset is also non-unitary
            H a;
        }

        reset_gate q[0];
    ";

    let result = QASMParser::parse_str_raw(qasm);
    println!("Gate with reset result: {:?}", result.is_ok());

    if let Err(e) = result {
        println!("Expected error: {e}");
    }
}

#[test]
fn test_gate_with_if_statement() {
    // Test if conditional statements can be included
    let qasm = r"
        OPENQASM 2.0;
        qreg q[1];
        creg c[1];

        gate conditional_gate a {
            if (c == 1) H a;  // Conditionals don't make sense in gates
        }

        conditional_gate q[0];
    ";

    let result = QASMParser::parse_str_raw(qasm);
    println!("Gate with if statement result: {:?}", result.is_ok());

    if let Err(e) = result {
        println!("Expected error: {e}");
    }
}

#[test]
fn test_proper_gate_content() {
    // Test what should work - only unitary operations
    let qasm = r"
        OPENQASM 2.0;
        qreg q[3];

        gate good_gate a, b, c {
            H a;
            CX a, b;
            ccx a, b, c;
            rx(pi/4) c;
            barrier a;  // Maybe this could work?
        }

        good_gate q[0], q[1], q[2];
    ";

    let result = QASMParser::parse_str_raw(qasm);
    println!("Proper gate content result: {:?}", result.is_ok());
}
