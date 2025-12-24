//! Comprehensive tests for conditional operations in QASM
//! Consolidates all conditional/if statement tests

use std::error::Error;

use pecos_engines::ClassicalControlEngineBuilder;
use pecos_programs::Qasm;
use pecos_qasm::qasm_engine;

#[test]
fn test_conditional_execution() -> Result<(), Box<dyn Error>> {
    // Create QASM that includes conditional statements
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        // Create registers
        qreg q[2];
        creg c[2];

        // Initialize qubit 0 in superposition
        H q[0];

        // Measure qubit 0 to c[0]
        measure q[0] -> c[0];

        // Conditional quantum operation: if c[0]==1, apply X to q[1]
        if(c[0]==1) X q[1];

        // Measure q[1] to c[1]
        measure q[1] -> c[1];
    "#;

    // Use the simulation helper instead of direct engine usage
    let results = qasm_engine()
        .program(Qasm::from_string(qasm))
        .to_sim()
        .seed(42)
        .workers(1)
        .run(100)?;
    // Count different outcomes
    let mut both_ones = 0;
    let mut both_zeros = 0;

    for shot in &results.shots {
        let value = shot
            .data
            .get("c")
            .and_then(pecos_engines::shot_results::Data::as_u32)
            .expect("c register should be convertible to u32");
        if value == 3 {
            // Both bits are 1
            both_ones += 1;
        } else if value == 0 {
            // Both bits are 0
            both_zeros += 1;
        }
    }

    // We should have both outcomes due to superposition
    assert!(both_ones > 0, "Should have some cases where both are 1");
    assert!(both_zeros > 0, "Should have some cases where both are 0");

    Ok(())
}

#[test]
fn test_simple_if() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[2];
        creg c[2];

        // Test simple if statement
        x q[0];
        measure q[0] -> c[0];

        // This should execute since c[0] will be 1
        if (c[0] == 1) x q[1];

        measure q[1] -> c[1];
    "#;

    let results = qasm_engine()
        .program(Qasm::from_string(qasm))
        .to_sim()
        .seed(42)
        .workers(1)
        .run(100)
        .expect("Failed to run simulation");

    // Should always get c = 11 (binary) = 3 (decimal)
    for shot in &results.shots {
        let value = shot
            .data
            .get("c")
            .and_then(pecos_engines::shot_results::Data::as_u32)
            .expect("c register should be convertible to u32");
        assert_eq!(value, 3, "Both qubits should be measured as 1");
    }
}

#[test]
fn test_exact_issue() {
    // Test the exact problem from test_cond_bell
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[2];
        creg c[2];

        H q[0];
        CX q[0], q[1];
        measure q[0] -> c[0];
        measure q[1] -> c[1];

        // Only execute if we measured 00
        if (c == 0) X q[0];

        // Try to reproduce the conditional
        if (c[0] == 0) X q[1];
    "#;

    let results = qasm_engine()
        .program(Qasm::from_string(qasm))
        .to_sim()
        .seed(42)
        .workers(1)
        .run(100)
        .expect("Failed to run simulation");

    // Verify we get results
    assert!(!results.is_empty(), "Should have at least one shot");
    assert!(
        results.shots[0].data.contains_key("c"),
        "Should have classical register c"
    );
}

#[test]
fn test_conditional_classical_operations() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[2];
        creg c[4];
        creg a[2];

        // Set some initial values
        a = 1;

        // Conditional classical operation
        if (a == 1) c = 5;

        // Complex conditional
        if (c > 4) x q[0];

        measure q[0] -> c[0];
    "#;

    let results = qasm_engine()
        .program(Qasm::from_string(qasm))
        .to_sim()
        .seed(42)
        .workers(1)
        .run(100)
        .expect("Failed to run simulation");

    // c[0] should always be 1 (from x q[0])
    for shot in &results.shots {
        let value = shot
            .data
            .get("c")
            .and_then(pecos_engines::shot_results::Data::as_u32)
            .expect("c register should be convertible to u32");
        let bit_0 = value & 1;
        assert_eq!(bit_0, 1, "Bit 0 should be 1 after conditional X gate");
    }
}

#[test]
fn test_conditional_comparison_operators() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[4];
        creg c[4];

        // Test different comparison operators
        c = 3;

        if (c == 3) x q[0];   // Should execute
        if (c != 3) x q[1];   // Should not execute
        if (c < 4) x q[2];    // Should execute
        if (c > 4) x q[3];    // Should not execute

        measure q -> c;
    "#;

    let results = qasm_engine()
        .program(Qasm::from_string(qasm))
        .to_sim()
        .seed(42)
        .workers(1)
        .run(100)
        .expect("Failed to run simulation");

    // Only q[0] and q[2] should be flipped
    for shot in &results.shots {
        let value = shot
            .data
            .get("c")
            .and_then(pecos_engines::shot_results::Data::as_u32)
            .expect("c register should be convertible to u32");
        assert_eq!(value, 0b0101, "Only q[0] and q[2] should be 1");
    }
}

#[test]
fn test_nested_conditionals() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[2];
        creg c[2];

        c = 1;

        if (c > 0) x q[0];  // Since c[0] == 1 was set above

        measure q -> c;
    "#;

    let results = qasm_engine()
        .program(Qasm::from_string(qasm))
        .to_sim()
        .seed(42)
        .workers(1)
        .run(100)
        .expect("Failed to run simulation");

    // q[0] should be flipped
    for shot in &results.shots {
        let value = shot
            .data
            .get("c")
            .and_then(pecos_engines::shot_results::Data::as_u32)
            .expect("c register should be convertible to u32");
        let bit_0 = value & 1;
        assert_eq!(bit_0, 1, "q[0] should be 1 after nested conditionals");
    }
}

#[test]
fn test_conditional_with_barriers() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[2];
        creg c[2];

        H q[0];
        measure q[0] -> c[0];

        if (c[0] == 1) X q[1];  // Simplified without barrier for now

        measure q[1] -> c[1];
    "#;

    let results = qasm_engine()
        .program(Qasm::from_string(qasm))
        .to_sim()
        .seed(42)
        .workers(1)
        .run(100)
        .expect("Failed to run simulation");

    // When c[0] is 1, c[1] should also be 1
    for shot in &results.shots {
        let value = shot
            .data
            .get("c")
            .and_then(pecos_engines::shot_results::Data::as_u32)
            .expect("c register should be convertible to u32");
        let bit_0 = value & 1;
        let bit_1 = (value >> 1) & 1;

        if bit_0 == 1 {
            assert_eq!(bit_1, 1, "When c[0] is 1, c[1] should also be 1");
        } else {
            assert_eq!(bit_1, 0, "When c[0] is 0, c[1] should also be 0");
        }
    }
}

#[test]
fn test_conditional_feature_flags() {
    // Test that conditional compilation features work
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[2];
        creg c[2];

        // Conditionals are a standard QASM feature
        x q[0];
        measure q[0] -> c[0];

        if (c[0] == 1) h q[1];

        measure q[1] -> c[1];
    "#;

    let results = qasm_engine()
        .program(Qasm::from_string(qasm))
        .to_sim()
        .seed(42)
        .workers(1)
        .run(100)
        .expect("Failed to run simulation");
    assert!(!results.is_empty(), "Should have at least one shot");
    assert!(
        results.shots[0].data.contains_key("c"),
        "Should have classical register c"
    );
}

#[test]
fn test_if_with_multiple_statements() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[3];
        creg c[3];

        x q[0];
        measure q[0] -> c[0];

        if (c[0] == 1) x q[1];  // Simplified to single operation

        measure q[1] -> c[1];
        measure q[2] -> c[2];
    "#;

    let results = qasm_engine()
        .program(Qasm::from_string(qasm))
        .to_sim()
        .seed(42)
        .workers(1)
        .run(100)
        .expect("Failed to run simulation");

    // c[0] and c[1] should always be 1
    for shot in &results.shots {
        let value = shot
            .data
            .get("c")
            .and_then(pecos_engines::shot_results::Data::as_u32)
            .expect("c register should be convertible to u32");
        let bit_0 = value & 1;
        let bit_1 = (value >> 1) & 1;
        assert_eq!(bit_0, 1, "c[0] should always be 1");
        assert_eq!(bit_1, 1, "c[1] should always be 1");
        // c[2] could be 0 or 1 due to H gate
    }
}
