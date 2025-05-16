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
