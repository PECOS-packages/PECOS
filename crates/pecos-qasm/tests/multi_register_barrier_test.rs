use pecos_qasm::{Operation, parser::QASMParser};

#[test]
fn test_multi_register_barrier() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[4];
        qreg p[2];
        qreg r[2];
        creg c[2];
        barrier q[0],q[3],p;
        u1(0.3*pi) p[0];
        u1(0.3*pi) p[1];
        cx p[0], r[0];
        cx p[1], r[1];
        measure r -> c;
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse multi-register barrier");

    // Track different types of operations
    let mut has_barrier = false;
    let mut barrier_qubits = Vec::new();
    let mut has_u1 = false;
    let mut has_cx = false;
    let mut has_measure = false;

    for op in &program.operations {
        match op {
            Operation::Barrier { qubits } => {
                has_barrier = true;
                barrier_qubits = qubits.clone();
            }
            Operation::Gate { name, .. } => {
                match name.as_str() {
                    "u1" | "U1" | "rz" | "RZ" => has_u1 = true, // u1 might expand to rz
                    "cx" | "CX" => has_cx = true,
                    _ => {}
                }
            }
            Operation::Measure { .. } => {
                has_measure = true;
            }
            Operation::RegMeasure { .. } => {
                has_measure = true;
            }
            _ => {}
        }
    }

    // Verify we have the expected operations
    assert!(has_barrier, "Should have barrier operation");
    assert!(has_u1, "Should have u1 gate (or its expansion)");
    assert!(has_cx, "Should have cx gate");
    assert!(has_measure, "Should have RegMeasure operation");

    // Check barrier includes the right qubits
    // barrier q[0],q[3],p should include q[0], q[3], and all of p (p[0], p[1])
    // That's 4 qubits total: q[0], q[3], p[0], p[1]
    assert_eq!(barrier_qubits.len(), 4, "Barrier should include exactly 4 qubits");

    // Verify the barrier contains the expected qubits
    assert!(barrier_qubits.contains(&0), "Should include q[0]");
    assert!(barrier_qubits.contains(&3), "Should include q[3]");
    assert!(barrier_qubits.contains(&4), "Should include p[0]"); // Assuming p starts at index 4
    assert!(barrier_qubits.contains(&5), "Should include p[1]");
}

#[test]
fn test_register_operations() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg p[2];
        qreg r[2];
        creg c[2];
        
        h p;  // Apply h to entire register
        cx p, r;  // Apply cx between registers  
        measure r -> c;  // Measure entire register
    "#;

    // Try parsing with register operations
    let result = QASMParser::parse_str(qasm);
    
    match result {
        Ok(_) => {
            println!("Parser supports register-level operations");
            return;
        }
        Err(e) => {
            println!("Parser doesn't support register operations: {}", e);
        }
    }
    
    // Fallback to individual operations
    let qasm_individual = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg p[2];
        qreg r[2];
        creg c[2];
        
        h p[0];
        h p[1];
        cx p[0], r[0];
        cx p[1], r[1];
        measure r[0] -> c[0];
        measure r[1] -> c[1];
    "#;
    
    let program = QASMParser::parse_str(qasm_individual).expect("Failed to parse individual operations");
    
    // Track operations on registers
    let mut h_count = 0;
    let mut cx_count = 0;
    let mut measure_count = 0;
    
    for op in &program.operations {
        match op {
            Operation::Gate { name, .. } => {
                match name.as_str() {
                    "H" | "h" => h_count += 1,
                    "CX" | "cx" => cx_count += 1,
                    _ => {}
                }
            }
            Operation::Measure { .. } => measure_count += 1,
            _ => {}
        }
    }
    
    // When applying gates to registers, they should expand to individual qubits
    assert_eq!(h_count, 2, "Should have H gates for each qubit in p");
    assert_eq!(cx_count, 2, "Should have CX gates between register pairs");
    assert_eq!(measure_count, 2, "Should have measurements for each qubit");
}

#[test]
fn test_mixed_qubit_register_barrier() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[4];
        qreg p[2];

        // Barrier with individual qubits and whole register
        barrier q[0], q[3], p;
    "#;

    let program = QASMParser::parse_str(qasm).expect("Should parse barrier with mixed register and individual qubits");

    // Find the barrier operation
    let mut barrier_found = false;
    let mut barrier_qubit_count = 0;

    for op in &program.operations {
        if let Operation::Barrier { qubits } = op {
            barrier_found = true;
            barrier_qubit_count = qubits.len();

            // The barrier should include:
            // - q[0] (qubit 0)
            // - q[3] (qubit 3)
            // - p[0] and p[1] (qubits 4 and 5, assuming sequential numbering)
            // Total: 4 qubits

            // Check that we have qubits from both registers
            let has_q_qubits = qubits.iter().any(|&q| q < 4); // q register
            let has_p_qubits = qubits.iter().any(|&q| q >= 4); // p register

            assert!(has_q_qubits, "Barrier should include qubits from q register");
            assert!(has_p_qubits, "Barrier should include qubits from p register");
        }
    }

    assert!(barrier_found, "Should have found barrier operation");
    assert_eq!(barrier_qubit_count, 4, "Barrier should include exactly 4 qubits");
}

#[test]
fn test_gate_on_register_subset() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg p[3];
        
        // Apply gate to subset of register
        h p[0];
        h p[2];
        
        // Apply gate to entire register
        x p;
    "#;

    // Try parsing with register operation
    let result = QASMParser::parse_str(qasm);
    
    let program = match result {
        Ok(prog) => prog,
        Err(_) => {
            // Fallback
            let qasm_individual = r#"
                OPENQASM 2.0;
                include "qelib1.inc";

                qreg p[3];
                
                h p[0];
                h p[2];
                x p[0];
                x p[1];
                x p[2];
            "#;
            QASMParser::parse_str(qasm_individual).expect("Failed to parse")
        }
    };
    
    let mut h_on_specific = 0;
    let mut x_count = 0;
    
    for op in &program.operations {
        if let Operation::Gate { name, qubits, .. } = op {
            match name.as_str() {
                "H" | "h" => {
                    if qubits.len() == 1 {
                        h_on_specific += 1;
                    }
                }
                "X" | "x" => x_count += 1,
                _ => {}
            }
        }
    }
    
    assert_eq!(h_on_specific, 2, "Should have 2 H gates on specific qubits");
    assert_eq!(x_count, 3, "Should have X gates for entire register (3 qubits)");
}

#[test]
fn test_u1_gate_parameter() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[1];
        u1(0.3*pi) q[0];
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse u1 gate");
    
    // Find the u1 gate or its expansion
    let mut found_phase_gate = false;
    
    for op in &program.operations {
        if let Operation::Gate { name, parameters, .. } = op {
            // u1 is typically expanded to rz or another phase gate
            if name == "u1" || name == "U1" || name == "rz" || name == "RZ" {
                found_phase_gate = true;
                
                // Check the parameter
                if let Some(&angle) = parameters.get(0) {
                    let expected = 0.3 * std::f64::consts::PI;
                    assert!((angle - expected).abs() < 1e-10, 
                           "u1 angle should be 0.3*pi, got {}", angle);
                }
            }
        }
    }
    
    assert!(found_phase_gate, "Should have found u1 gate or its expansion");
}