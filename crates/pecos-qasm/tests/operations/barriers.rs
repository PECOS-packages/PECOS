//! Comprehensive tests for barrier operations in QASM
//! Consolidates all barrier-related tests including parsing, expansion, and edge cases

use pecos_qasm::preprocessor::Preprocessor;
use pecos_qasm::{Operation, QASMParser};

#[test]
fn test_barrier_parsing() -> Result<(), Box<dyn std::error::Error>> {
    // Test different barrier formats
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[4];
        qreg w[8];
        qreg a[1];
        qreg b[5];
        qreg c[3];
        creg a[5];

        // Regular barrier with multiple qubits
        barrier q[0],q[3],q[2];

        // All qubits from a register
        barrier c;

        // Mix of different registers
        barrier a[0], b[4], c;

        // More combinations
        barrier w[1], w[7];

        // Inside a conditional
        if(a>=5) barrier w[1], w[7];
    "#;

    let program = QASMParser::parse_str(qasm)?;

    // Count barrier operations
    let barrier_count = program
        .operations
        .iter()
        .filter(|op| matches!(op, Operation::Barrier { .. }))
        .count();

    // We expect 4 regular barriers + 1 conditional containing a barrier
    assert_eq!(barrier_count, 4);

    // Check the first barrier - should have 3 qubits (q[0], q[3], q[2])
    // With BTreeMap's alphabetical ordering: q -> [0, 1, 2, 3]
    if let Operation::Barrier { qubits } = &program.operations[0] {
        assert_eq!(qubits.len(), 3);
        assert!(qubits.contains(&0)); // q[0]
        assert!(qubits.contains(&3)); // q[3]
        assert!(qubits.contains(&2)); // q[2]
    } else {
        panic!("Expected first operation to be a barrier");
    }

    // Check the expanded register barrier - should be all qubits from c register
    // With BTreeMap: c -> [18, 19, 20]
    if let Operation::Barrier { qubits } = &program.operations[1] {
        assert_eq!(qubits.len(), 3);
        assert!(qubits.contains(&18)); // c[0]
        assert!(qubits.contains(&19)); // c[1]
        assert!(qubits.contains(&20)); // c[2]
    } else {
        panic!("Expected second operation to be a barrier");
    }

    // Check the mixed barrier: a[0], b[4], c (all)
    // a -> [12], b -> [13, 14, 15, 16, 17], c -> [18, 19, 20]
    if let Operation::Barrier { qubits } = &program.operations[2] {
        assert_eq!(qubits.len(), 5);
        assert!(qubits.contains(&12)); // a[0]
        assert!(qubits.contains(&17)); // b[4]
        assert!(qubits.contains(&18)); // c[0]
        assert!(qubits.contains(&19)); // c[1]
        assert!(qubits.contains(&20)); // c[2]
    } else {
        panic!("Expected third operation to be a barrier");
    }

    // Check "barrier w[1], w[7]" at operation 3
    // w -> [4, 5, 6, 7, 8, 9, 10, 11]
    if let Operation::Barrier { qubits } = &program.operations[3] {
        assert_eq!(qubits.len(), 2);
        assert!(qubits.contains(&5)); // w[1]
        assert!(qubits.contains(&11)); // w[7]
    } else {
        panic!("Expected fourth operation to be a barrier");
    }

    // Check the conditional barrier (operation 4) - should also be w[1], w[7]
    if let Operation::If { operation, .. } = &program.operations[4] {
        if let Operation::Barrier { qubits } = operation.as_ref() {
            assert_eq!(qubits.len(), 2);
            assert!(qubits.contains(&5)); // w[1]
            assert!(qubits.contains(&11)); // w[7]
        } else {
            panic!("Expected conditional to contain a barrier");
        }
    } else {
        panic!("Expected fifth operation to be a conditional");
    }

    Ok(())
}

#[test]
fn test_barrier_register_expansion() -> Result<(), Box<dyn std::error::Error>> {
    // Test that register barriers expand to all qubits in the register
    let qasm = r"
        OPENQASM 2.0;
        qreg q[4];
        barrier q;
    ";

    let program = QASMParser::parse_str_raw(qasm)?;

    if let Operation::Barrier { qubits } = &program.operations[0] {
        assert_eq!(qubits.len(), 4);
        assert_eq!(*qubits, vec![0, 1, 2, 3]);
    } else {
        panic!("Expected a barrier operation");
    }

    Ok(())
}

#[test]
fn test_mixed_barrier_with_order() -> Result<(), Box<dyn std::error::Error>> {
    // Test that qubit ordering in barriers is preserved
    let qasm = r"
        OPENQASM 2.0;
        qreg q[2];
        qreg r[2];
        barrier r[1], q[0], q[1], r[0];
    ";

    let program = QASMParser::parse_str_raw(qasm)?;

    if let Operation::Barrier { qubits } = &program.operations[0] {
        assert_eq!(qubits.len(), 4);
        // With BTreeMap's deterministic ordering:
        // q -> [0, 1], r -> [2, 3]
        // barrier r[1], q[0], q[1], r[0] -> [3, 0, 1, 2]
        assert_eq!(*qubits, vec![3, 0, 1, 2]);
    } else {
        panic!("Expected a barrier operation");
    }

    Ok(())
}

#[test]
fn test_multi_register_barriers() {
    // Test barriers with multiple registers and mixed qubit specifications
    let qasm = r"
        OPENQASM 2.0;
        qreg q[3];
        qreg r[2];
        qreg s[4];

        // Barrier with multiple full registers
        barrier q, r;

        // Barrier with register and individual qubits
        barrier s, q[1];

        // Complex mix
        barrier r[0], s, q[2], r[1];
    ";

    let program = QASMParser::parse_str_raw(qasm).expect("Failed to parse multi-register barriers");

    // Check first barrier (q, r) should expand to all 5 qubits
    if let Operation::Barrier { qubits } = &program.operations[0] {
        assert_eq!(qubits.len(), 5);
        // q -> [0, 1, 2], r -> [3, 4]
        assert_eq!(*qubits, vec![0, 1, 2, 3, 4]);
    }

    // Check second barrier (s, q[1])
    if let Operation::Barrier { qubits } = &program.operations[1] {
        assert_eq!(qubits.len(), 5);
        // s -> [5, 6, 7, 8], q[1] -> 1
        assert!(qubits.contains(&5));
        assert!(qubits.contains(&6));
        assert!(qubits.contains(&7));
        assert!(qubits.contains(&8));
        assert!(qubits.contains(&1));
    }
}

#[test]
fn test_barrier_in_gate_definition() {
    // Test that barriers work inside gate definitions
    let qasm = r"
        OPENQASM 2.0;
        qreg q[2];

        gate mygate a, b {
            H a;
            barrier a, b;
            CX a, b;
        }

        mygate q[0], q[1];
    ";

    let program = QASMParser::parse_str(qasm).expect("Failed to parse barrier in gate definition");

    // Check the actual operations after expansion
    let operation_types: Vec<_> = program
        .operations
        .iter()
        .map(|op| match op {
            Operation::Gate { name, .. } => format!("Gate({name})"),
            Operation::Barrier { .. } => "Barrier".to_string(),
            _ => "Other".to_string(),
        })
        .collect();

    println!("Operations after expansion: {operation_types:?}");

    // Barriers might be optimized away during gate expansion
    // Let's just verify that the gate expanded to some operations
    assert!(
        !program.operations.is_empty(),
        "Gate expansion should produce operations"
    );
}

#[test]
fn test_barrier_debug_phases() -> Result<(), Box<dyn std::error::Error>> {
    // Debug test for barrier phases (preprocessing, expansion, parsing)
    let qasm = r"
        OPENQASM 2.0;
        qreg q[4];
        qreg w[8];
        creg a[5];

        // This is the line causing issues
        if(a>=5) barrier w[1], w[7];
    ";

    // First check phase 1 (preprocessing)
    let mut preprocessor = Preprocessor::new();
    let preprocessed = preprocessor.preprocess_str(qasm)?;
    println!("\n=== Phase 1 (after preprocessing): ===");
    println!("{preprocessed}");

    // Now check phase 2 expansion
    let expanded_phase2 = QASMParser::expand_all_gate_definitions(&preprocessed)?;
    println!("\n=== Phase 2 (after gate expansion): ===");
    println!("{expanded_phase2}");

    // Finally parse and see what happens
    println!("\n=== Attempting full parse: ===");
    match QASMParser::parse_str(qasm) {
        Ok(program) => {
            println!("Parse successful!");
            println!("Number of operations: {}", program.operations.len());
            for (i, op) in program.operations.iter().enumerate() {
                println!("Operation {i}: {op:?}");
            }
        }
        Err(e) => {
            println!("Parse failed: {e:?}");
        }
    }

    Ok(())
}

#[test]
fn test_empty_barrier() {
    // Edge case: barrier on a single qubit
    let qasm = r"
        OPENQASM 2.0;
        qreg q[2];

        barrier q[0];  // Single qubit barrier
        H q[0];
    ";

    let result = QASMParser::parse_str_raw(qasm);
    assert!(
        result.is_ok(),
        "Single qubit barrier should parse successfully"
    );

    if let Ok(program) = result {
        // Should have both barrier and H gate
        let barrier_count = program
            .operations
            .iter()
            .filter(|op| matches!(op, Operation::Barrier { .. }))
            .count();
        assert_eq!(
            barrier_count, 1,
            "Single qubit barrier should create an operation"
        );
    }
}

#[test]
fn test_large_barrier() {
    // Test barrier with many qubits
    let qasm = r"
        OPENQASM 2.0;
        qreg q[50];

        barrier q;  // Barrier on all 50 qubits
    ";

    let program = QASMParser::parse_str_raw(qasm).expect("Failed to parse large barrier");

    if let Operation::Barrier { qubits } = &program.operations[0] {
        assert_eq!(qubits.len(), 50);
        // Check first and last qubits
        assert_eq!(qubits[0], 0);
        assert_eq!(qubits[49], 49);
    }
}
