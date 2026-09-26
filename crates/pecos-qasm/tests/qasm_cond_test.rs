use pecos_engines::ClassicalEngine;
use pecos_engines::sim_builder;
use pecos_programs::Qasm;
use pecos_qasm::{QASMEngine, qasm_engine};

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

/// Run `qasm` noiselessly and return register `d` from every shot.
fn register_d_values(qasm: &str, shots: usize) -> Vec<u64> {
    let results = sim_builder()
        .classical(qasm_engine().program(Qasm::from_string(qasm.to_string())))
        .run(shots)
        .unwrap();
    assert_eq!(results.len(), shots);
    results
        .try_as_shot_map()
        .unwrap()
        .try_bits_as_u64("d")
        .unwrap()
}

#[test]
fn cond_measure_single_qubit_runs_when_condition_holds() {
    // q[0] is |1>, so c[0] = 1 and the conditional measurement must fire and
    // write d[0] = 1. Before the engine handled measurements inside `if`,
    // the statement was silently dropped and d stayed 0.
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];
        creg d[1];

        x q[0];
        measure q[0] -> c[0];
        if (c == 1) measure q[0] -> d[0];
    "#;
    let d = register_d_values(qasm, 20);
    assert!(
        d.iter().all(|&v| v == 1),
        "d[0] should be 1 in every shot, got {d:?}"
    );
}

#[test]
fn cond_measure_single_qubit_skipped_when_condition_fails() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];
        creg d[1];

        x q[0];
        measure q[0] -> c[0];
        if (c == 0) measure q[0] -> d[0];
    "#;
    let d = register_d_values(qasm, 20);
    assert!(
        d.iter().all(|&v| v == 0),
        "d should stay 0 in every shot, got {d:?}"
    );
}

#[test]
fn cond_measure_register_runs_when_condition_holds() {
    // Register-form measurement inside `if` reaches the engine unexpanded, so
    // the whole register must be measured under one condition evaluation:
    // q[0] = |1>, q[1] = |0> gives d = 0b01.
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[1];
        creg d[2];

        x q[0];
        measure q[0] -> c[0];
        if (c == 1) measure q -> d;
    "#;
    let d = register_d_values(qasm, 20);
    assert!(
        d.iter().all(|&v| v == 0b01),
        "d should be 0b01 in every shot, got {d:?}"
    );
}

#[test]
fn cond_measure_register_skipped_when_condition_fails() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[1];
        creg d[2];

        x q[0];
        measure q[0] -> c[0];
        if (c == 0) measure q -> d;
    "#;
    let d = register_d_values(qasm, 20);
    assert!(
        d.iter().all(|&v| v == 0),
        "d should stay 0 in every shot, got {d:?}"
    );
}

#[test]
fn cond_barrier_is_accepted() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[1];
        creg d[1];

        x q[0];
        measure q[0] -> c[0];
        if (c == 1) barrier q;
        measure q[0] -> d[0];
    "#;
    let d = register_d_values(qasm, 20);
    assert!(
        d.iter().all(|&v| v == 1),
        "barrier must not disturb the run, got {d:?}"
    );
}

#[test]
fn cond_measure_register_evaluates_condition_once() {
    // Both qubits are |1> and the condition reads the register being written.
    // Expanding the statement per qubit would re-evaluate `d == 0` after the
    // first bit lands and skip the second; one evaluation gives d = 0b11.
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg d[2];

        x q[0];
        x q[1];
        if (d == 0) measure q -> d;
    "#;
    let d = register_d_values(qasm, 20);
    assert!(
        d.iter().all(|&v| v == 0b11),
        "d should be 0b11 in every shot, got {d:?}"
    );
}

#[test]
fn cond_measure_result_is_visible_to_the_next_conditional() {
    // The conditional measurement must end its batch so that the following
    // `if` reads the freshly written bit: d[0] = 1 drives x on q[1].
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[1];
        creg d[1];
        creg e[1];

        x q[0];
        measure q[0] -> c[0];
        if (c == 1) measure q[0] -> d[0];
        if (d == 1) x q[1];
        measure q[1] -> e[0];
    "#;
    let results = sim_builder()
        .classical(qasm_engine().program(Qasm::from_string(qasm.to_string())))
        .run(20)
        .unwrap();
    let e = results
        .try_as_shot_map()
        .unwrap()
        .try_bits_as_u64("e")
        .unwrap();
    assert!(
        e.iter().all(|&v| v == 1),
        "e should be 1 in every shot, got {e:?}"
    );
}

#[test]
fn cond_measure_register_result_is_visible_to_the_next_conditional() {
    // The condition tests a single bit: comparing a multi-bit register with a
    // literal (`d == 3`) is currently evaluated wrongly, see the issue linked
    // from the PR that added this test.
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        qreg r[1];
        creg c[1];
        creg d[2];
        creg e[1];

        x q[0];
        x q[1];
        measure q[0] -> c[0];
        if (c == 1) measure q -> d;
        if (d[1] == 1) x r[0];
        measure r[0] -> e[0];
    "#;
    let results = sim_builder()
        .classical(qasm_engine().program(Qasm::from_string(qasm.to_string())))
        .run(20)
        .unwrap();
    let e = results
        .try_as_shot_map()
        .unwrap()
        .try_bits_as_u64("e")
        .unwrap();
    assert!(
        e.iter().all(|&v| v == 1),
        "e should be 1 in every shot, got {e:?}"
    );
}

#[test]
fn cond_measure_register_size_mismatch_is_an_error() {
    // Top-level register measurements are size-checked by the parser; the
    // conditional form reaches the engine unexpanded and must be checked
    // there instead of silently measuring the shorter register's worth.
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg d[1];

        if (d == 0) measure q -> d;
    "#;
    let mut engine = qasm.parse::<QASMEngine>().unwrap();
    let Err(err) = engine.generate_commands() else {
        panic!("mismatched register sizes must not be measured");
    };
    assert!(
        err.to_string().contains("size mismatch"),
        "unexpected error: {err}"
    );
}
