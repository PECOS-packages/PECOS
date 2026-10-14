// Test to verify that measurements are tracked by order, not by explicit IDs

use pecos_core::prelude::*;
use pecos_engines::{ByteMessage, ClassicalEngine};
use pecos_qasm::QASMEngine;

#[test]
fn test_measurement_order_tracking() -> Result<(), PecosError> {
    // Create a simple QASM program with multiple measurements
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[3];
        creg c[3];

        // Apply X gates to create a known state
        x q[0];
        x q[2];

        // Measure all qubits
        measure q[0] -> c[0];  // Should be 1
        measure q[1] -> c[1];  // Should be 0
        measure q[2] -> c[2];  // Should be 1
    "#;

    // Create engine from string
    let mut engine = qasm.parse::<QASMEngine>()?;

    // Process all measurements - the engine breaks after each measurement
    // First batch: X gates + first measurement
    let commands1 = engine.generate_commands()?;
    let operations1 = commands1.quantum_ops()?;
    println!("First batch operations: {operations1:?}");

    // Handle first measurement
    let mut results_builder = ByteMessage::outcomes_builder();
    results_builder.add_outcomes(&[1]); // q0=1
    engine.handle_measurements(results_builder.build())?;

    // Second batch: second measurement
    let commands2 = engine.generate_commands()?;
    let operations2 = commands2.quantum_ops()?;
    println!("Second batch operations: {operations2:?}");

    // Handle second measurement
    let mut results_builder = ByteMessage::outcomes_builder();
    results_builder.add_outcomes(&[0]); // q1=0
    engine.handle_measurements(results_builder.build())?;

    // Third batch: third measurement
    let commands3 = engine.generate_commands()?;
    let operations3 = commands3.quantum_ops()?;
    println!("Third batch operations: {operations3:?}");

    // Handle third measurement
    let mut results_builder = ByteMessage::outcomes_builder();
    results_builder.add_outcomes(&[1]); // q2=1
    engine.handle_measurements(results_builder.build())?;

    // Get final results
    let shot_result = engine.get_results()?;

    // Verify results
    let c_value = shot_result
        .data
        .get("c")
        .and_then(pecos_engines::prelude::Data::as_u32)
        .unwrap_or(0);
    let bit0 = c_value & 1;
    let bit1 = (c_value >> 1) & 1;
    let bit2 = (c_value >> 2) & 1;

    assert_eq!(bit0, 1, "c[0] should be 1");
    assert_eq!(bit1, 0, "c[1] should be 0");
    assert_eq!(bit2, 1, "c[2] should be 1");

    Ok(())
}

#[test]
fn test_measurement_order_with_batches() -> Result<(), PecosError> {
    // Test that measurement order is preserved across multiple batches
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[2];
        creg c1[1];
        creg c2[1];

        // First measurement
        x q[0];
        measure q[0] -> c1[0];  // Should be 1

        // Second measurement
        measure q[1] -> c2[0];  // Should be 0
    "#;

    let mut engine = qasm.parse::<QASMEngine>()?;

    // First batch
    let _commands1 = engine.generate_commands()?;

    // Handle first measurement
    let mut results_builder = ByteMessage::outcomes_builder();
    results_builder.add_outcomes(&[1]); // First measurement result
    engine.handle_measurements(results_builder.build())?;

    // Second batch
    let _commands2 = engine.generate_commands()?;

    // Handle second measurement
    let mut results_builder = ByteMessage::outcomes_builder();
    results_builder.add_outcomes(&[0]); // Second measurement result
    engine.handle_measurements(results_builder.build())?;

    // Get final results
    let shot_result = engine.get_results()?;

    // Verify results
    let c1_value = shot_result
        .data
        .get("c1")
        .and_then(pecos_engines::prelude::Data::as_u32)
        .unwrap_or(0);
    let c2_value = shot_result
        .data
        .get("c2")
        .and_then(pecos_engines::prelude::Data::as_u32)
        .unwrap_or(0);

    assert_eq!(c1_value & 1, 1, "c1[0] should be 1");
    assert_eq!(c2_value & 1, 0, "c2[0] should be 0");

    Ok(())
}

#[test]
fn surplus_outcomes_are_rejected_instead_of_desyncing() -> Result<(), PecosError> {
    // The first batch measures one qubit. Handing back two outcomes used to
    // drop the second at debug level while still advancing the measurement
    // counter, so every later measurement landed at the wrong index.
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        x q[0];
        measure q[0] -> c[0];
        measure q[1] -> c[1];
    "#;
    let mut engine = qasm.parse::<QASMEngine>()?;
    let _first_batch = engine.generate_commands()?;

    let mut results_builder = ByteMessage::outcomes_builder();
    results_builder.add_outcomes(&[1, 0]);
    let err = engine
        .handle_measurements(results_builder.build())
        .expect_err("two outcomes for a one-measurement batch must be rejected");
    assert!(
        err.to_string().contains("only 1 measurements are pending"),
        "unexpected error: {err}"
    );
    Ok(())
}
