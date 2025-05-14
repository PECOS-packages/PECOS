use pecos_qasm::parser::{QASMParser, Operation};

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
    let barrier_count = program.operations.iter().filter(|op| {
        matches!(op, Operation::Barrier { .. })
    }).count();

    // We expect 4 regular barriers + 1 conditional containing a barrier
    println!("Found {} barrier operations", barrier_count);
    
    // Check the first barrier
    if let Operation::Barrier { qubits } = &program.operations[0] {
        println!("First barrier qubits: {:?}", qubits);
        assert_eq!(qubits.len(), 3);
        assert!(qubits.contains(&0)); // q[0]
        assert!(qubits.contains(&3)); // q[3]
        assert!(qubits.contains(&2)); // q[2]
    } else {
        panic!("Expected first operation to be a barrier");
    }
    
    // Check the expanded register barrier
    if let Operation::Barrier { qubits } = &program.operations[1] {
        println!("Register barrier qubits: {:?}", qubits);
        // c[0], c[1], c[2]
        assert_eq!(qubits.len(), 3);
        // c register starts at global ID 18 (after q[4], w[8], a[1], b[5])
        let c_start = 4 + 8 + 1 + 5;
        assert!(qubits.contains(&(c_start + 0))); // c[0]
        assert!(qubits.contains(&(c_start + 1))); // c[1]
        assert!(qubits.contains(&(c_start + 2))); // c[2]
    } else {
        panic!("Expected second operation to be a barrier");
    }
    
    // Check the mixed barrier
    if let Operation::Barrier { qubits } = &program.operations[2] {
        println!("Mixed barrier qubits: {:?}", qubits);
        // a[0] + b[4] + c[0], c[1], c[2]
        assert_eq!(qubits.len(), 5); // 1 + 1 + 3
        // Verify we have the right qubits
        let a_start = 4 + 8; // after q[4], w[8]
        let b_start = 4 + 8 + 1; // after q[4], w[8], a[1]
        let c_start = 4 + 8 + 1 + 5; // after q[4], w[8], a[1], b[5]

        assert!(qubits.contains(&(a_start + 0))); // a[0]
        assert!(qubits.contains(&(b_start + 4))); // b[4]
        assert!(qubits.contains(&(c_start + 0))); // c[0]
        assert!(qubits.contains(&(c_start + 1))); // c[1]
        assert!(qubits.contains(&(c_start + 2))); // c[2]
    } else {
        panic!("Expected third operation to be a barrier");
    }
    
    // Check the conditional barrier
    let has_conditional_barrier = program.operations.iter().any(|op| {
        if let Operation::If { operation, .. } = op {
            matches!(operation.as_ref(), Operation::Barrier { .. })
        } else {
            false
        }
    });
    
    assert!(has_conditional_barrier, "Should have a conditional barrier");
    
    Ok(())
}

#[test]
fn test_barrier_register_expansion() -> Result<(), Box<dyn std::error::Error>> {
    // Test that register barriers expand to all qubits in the register
    let qasm = r#"
        OPENQASM 2.0;
        qreg q[4];
        barrier q;
    "#;

    let program = QASMParser::parse_str(qasm)?;

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
    let qasm = r#"
        OPENQASM 2.0;
        qreg q[2];
        qreg r[2];
        barrier r[1], q[0], q[1], r[0];
    "#;

    let program = QASMParser::parse_str(qasm)?;
    
    if let Operation::Barrier { qubits } = &program.operations[0] {
        assert_eq!(qubits.len(), 4);
        // r[1] -> global ID 3, q[0] -> 0, q[1] -> 1, r[0] -> 2
        assert_eq!(*qubits, vec![3, 0, 1, 2]);
    } else {
        panic!("Expected a barrier operation");
    }
    
    Ok(())
}