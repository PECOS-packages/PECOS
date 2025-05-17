//! Integration tests that validate quantum simulation results
//! These tests go beyond parsing and actually verify quantum circuit behavior

#[allow(clippy::duplicate_mod)]
#[path = "../helper.rs"]
mod helper;

use helper::run_qasm_sim;

#[test]
fn test_bell_state_simulation() {
    // Test creating and measuring a Bell state
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];

        h q[0];
        cx q[0], q[1];
        measure q -> c;
    "#;

    let results = run_qasm_sim(qasm, 1000, Some(42)).unwrap();
    let c_values = results.get("c").unwrap();

    // Count occurrences of |00⟩ and |11⟩
    let mut count_00 = 0;
    let mut count_11 = 0;

    for &value in c_values {
        match value {
            0b00 => count_00 += 1,
            0b11 => count_11 += 1,
            _ => panic!("Bell state should only produce |00⟩ or |11⟩"),
        }
    }

    // Bell state should produce roughly 50/50 split
    assert!(
        count_00 > 400 && count_00 < 600,
        "Expected ~500 |00⟩ states, got {count_00}"
    );
    assert!(
        count_11 > 400 && count_11 < 600,
        "Expected ~500 |11⟩ states, got {count_11}"
    );
}

#[test]
fn test_ghz_state_simulation() {
    // Test creating and measuring a 3-qubit GHZ state
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        creg c[3];

        h q[0];
        cx q[0], q[1];
        cx q[1], q[2];
        measure q -> c;
    "#;

    let results = run_qasm_sim(qasm, 1000, Some(42)).unwrap();
    let c_values = results.get("c").unwrap();

    // Count occurrences of |000⟩ and |111⟩
    let mut count_000 = 0;
    let mut count_111 = 0;

    for &value in c_values {
        match value {
            0b000 => count_000 += 1,
            0b111 => count_111 += 1,
            _ => panic!("GHZ state should only produce |000⟩ or |111⟩"),
        }
    }

    // GHZ state should produce roughly 50/50 split
    assert!(
        count_000 > 400 && count_000 < 600,
        "Expected ~500 |000⟩ states, got {count_000}"
    );
    assert!(
        count_111 > 400 && count_111 < 600,
        "Expected ~500 |111⟩ states, got {count_111}"
    );
}

#[test]
fn test_phase_kickback() {
    // Test phase kickback with controlled gates
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];

        // Prepare control in superposition
        h q[0];

        // Prepare target in |1⟩ state
        x q[1];

        // Apply controlled-Z
        cz q[0], q[1];

        // Measure in computational basis
        h q[0];
        measure q -> c;
    "#;

    let results = run_qasm_sim(qasm, 1000, Some(42)).unwrap();
    let c_values = results.get("c").unwrap();

    // After phase kickback, control qubit should be |1⟩
    for &value in c_values {
        let control_bit = value & 1;
        assert_eq!(
            control_bit, 1,
            "Control qubit should always be |1⟩ after phase kickback"
        );

        let target_bit = (value >> 1) & 1;
        assert_eq!(target_bit, 1, "Target qubit should remain |1⟩");
    }
}
