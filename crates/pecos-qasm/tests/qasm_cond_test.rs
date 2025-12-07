use pecos_engines::sim_builder;
use pecos_programs::Qasm;
use pecos_qasm::qasm_engine;

#[test]
fn test_uncond_reset_register() {
    // Test unconditional reset on entire register
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        creg c[3];

        // Prepare all qubits in |1⟩
        x q;

        // Reset entire register
        reset q;

        // Measure
        measure q -> c;
    "#;

    let results = sim_builder()
        .classical(qasm_engine().program(Qasm::from_string(qasm)))
        .run(100)
        .unwrap();
    let shot_map = results.try_as_shot_map().unwrap();
    let values = shot_map.try_bits_as_u64("c").unwrap();

    for val in values {
        assert_eq!(val, 0, "Expected all qubits to be reset to |0⟩");
    }
}

#[test]
fn test_cond_reset_v1() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];

        if(c[0] == 0) reset q;
    "#;

    let results = sim_builder()
        .classical(qasm_engine().program(Qasm::from_string(qasm)))
        .run(100)
        .unwrap();
    assert_eq!(results.len(), 100);
}

#[test]
fn test_cond_reset_v2() {
    let qasm = r#"
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    creg c[2];

    if(c[0] == 0) reset q[0];
    "#;

    let results = sim_builder()
        .classical(qasm_engine().program(Qasm::from_string(qasm)))
        .run(100)
        .unwrap();
    assert_eq!(results.len(), 100);
}

#[test]
fn test_cond_reset_single_qubit() {
    // Test conditional reset on a single qubit
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[4];
        creg c[4];

        // Prepare some qubits in |1⟩
        x q[0];
        x q[1];
        x q[3];

        // Reset only q[1] conditionally
        if(c[0] == 0) reset q[1];

        // Measure all
        measure q -> c;
    "#;

    let results = sim_builder()
        .classical(qasm_engine().program(Qasm::from_string(qasm)))
        .run(100)
        .unwrap();
    let shot_map = results.try_as_shot_map().unwrap();
    let values = shot_map.try_bits_as_u64("c").unwrap();

    for val in values {
        // q[0] and q[3] should be 1, q[1] should be 0, q[2] was never set
        assert_eq!(val & 0b0001, 0b0001, "Expected q[0] to be |1⟩");
        assert_eq!(val & 0b0010, 0b0000, "Expected q[1] to be reset to |0⟩");
        assert_eq!(val & 0b0100, 0b0000, "Expected q[2] to be |0⟩");
        assert_eq!(val & 0b1000, 0b1000, "Expected q[3] to be |1⟩");
    }
}

#[test]
fn test_cond_reset_with_state_preparation() {
    // Test that reset actually resets qubits to |0⟩
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];

        // Prepare qubits in |1⟩ state
        x q[0];
        x q[1];

        // Conditionally reset them
        if(c[0] == 0) reset q;

        // Measure to verify they're in |0⟩
        measure q -> c;
    "#;

    let results = sim_builder()
        .classical(qasm_engine().program(Qasm::from_string(qasm)))
        .run(100)
        .unwrap();
    // All results should be "00" since c[0] starts as 0 and reset happens
    let shot_map = results.try_as_shot_map().unwrap();
    let values = shot_map.try_bits_as_u64("c").unwrap();

    for val in values {
        assert_eq!(val, 0, "Expected all qubits to be reset to |0⟩");
    }
}

#[test]
fn test_cond_reset_false_condition() {
    // Test that reset doesn't happen when condition is false
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];

        // Set c[0] to 1
        x q[0];
        measure q[0] -> c[0];

        // Prepare q[1] in |1⟩ state
        x q[1];

        // This reset should NOT happen since c[0] == 1
        if(c[0] == 0) reset q[1];

        // Measure q[1]
        measure q[1] -> c[1];
    "#;

    let results = sim_builder()
        .classical(qasm_engine().program(Qasm::from_string(qasm)))
        .run(100)
        .unwrap();
    // All results should have c[1] = 1 since reset didn't happen
    let shot_map = results.try_as_shot_map().unwrap();
    let values = shot_map.try_bits_as_u64("c").unwrap();

    for val in values {
        // Check that bit 1 is set (c[1] = 1)
        assert_eq!(val & 0b10, 0b10, "Expected q[1] to remain in |1⟩");
    }
}

#[test]
fn test_cond_reset_full_register_then_single_qubit() {
    // Test resetting a full register followed by resetting a single qubit
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        qreg r[2];
        creg c[5];

        // Prepare some qubits in |1⟩
        x q;
        x r[0];

        // Reset entire q register conditionally
        if(c[0] == 0) reset q;

        // Also reset r[0] conditionally
        if(c[1] == 0) reset r[0];

        // Measure all
        measure q[0] -> c[0];
        measure q[1] -> c[1];
        measure q[2] -> c[2];
        measure r[0] -> c[3];
        measure r[1] -> c[4];
    "#;

    let results = sim_builder()
        .classical(qasm_engine().program(Qasm::from_string(qasm)))
        .run(100)
        .unwrap();
    let shot_map = results.try_as_shot_map().unwrap();
    let values = shot_map.try_bits_as_u64("c").unwrap();

    for val in values {
        // All bits should be 0 (q[0-2] and r[0] were reset, r[1] was never set)
        assert_eq!(val, 0, "Expected all measured qubits to be |0⟩");
    }
}

#[test]
fn test_multiple_cond_resets() {
    // Test multiple conditional resets with different conditions
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        creg c[3];

        // Prepare all qubits in |1⟩
        x q;

        // Multiple conditional resets
        if(c[0] == 0) reset q[0];
        if(c[1] == 0) reset q[1];
        if(c[2] == 0) reset q[2];

        // Measure
        measure q -> c;
    "#;

    let results = sim_builder()
        .classical(qasm_engine().program(Qasm::from_string(qasm)))
        .run(100)
        .unwrap();
    // All should be reset to |0⟩
    let shot_map = results.try_as_shot_map().unwrap();
    let values = shot_map.try_bits_as_u64("c").unwrap();

    for val in values {
        assert_eq!(val, 0, "Expected all qubits to be reset");
    }
}

#[test]
fn test_cond_reset_with_register_comparison() {
    // Test reset with register-wide comparison
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];

        // Prepare qubits in |1⟩
        x q;

        // This should NOT reset since c == 0
        if(c == 2) reset q;

        // Measure - should still be |11⟩
        measure q -> c;
    "#;

    let results = sim_builder()
        .classical(qasm_engine().program(Qasm::from_string(qasm)))
        .run(100)
        .unwrap();
    let shot_map = results.try_as_shot_map().unwrap();
    let values = shot_map.try_bits_as_u64("c").unwrap();

    for val in values {
        assert_eq!(val, 3, "Expected qubits to remain |11⟩ since c != 2");
    }
}
