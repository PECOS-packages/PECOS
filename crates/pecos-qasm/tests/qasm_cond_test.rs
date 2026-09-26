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
    register_values(qasm, shots, "d", false)
}

/// Run a program and require one named register result for every requested shot.
fn register_values(qasm: &str, shots: usize, register: &str, complex: bool) -> Vec<u64> {
    let mut builder = qasm_engine().program(Qasm::from_string(qasm));
    if complex {
        builder = builder.allow_complex_conditionals(true);
    }
    let results = sim_builder().classical(builder).run(shots).unwrap();
    assert_eq!(results.len(), shots);
    let values = results
        .try_as_shot_map()
        .unwrap()
        .try_bits_as_u64(register)
        .unwrap();
    assert_eq!(values.len(), shots);
    values
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
    // A conditional register measurement must be visible to a subsequent
    // comparison of the complete register, including its top bit.
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
        if (d == 3) x r[0];
        measure r[0] -> e[0];
    "#;
    let e = register_values(qasm, 20, "e", false);
    assert!(
        e.iter().all(|&v| v == 1),
        "e should be 1 in every shot, got {e:?}"
    );
}

#[test]
fn cond_measure_register_size_mismatch_is_a_parse_error() {
    // A conditional register measurement is not expanded per qubit, so the
    // parser must still size-check it there: the error has to be static, not
    // one that only appears in the shots where the condition holds.
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg d[1];

        if (d == 0) measure q -> d;
    "#;
    let err = qasm
        .parse::<QASMEngine>()
        .expect_err("mismatched register sizes must not parse");
    assert!(
        err.to_string()
            .contains("Register size mismatch in measure q -> d"),
        "unexpected error: {err}"
    );
}

// Each row is a separate program and test so failures identify the comparison.
macro_rules! unsigned_condition_cases {
    ($($name:ident: ($width:expr, $register:literal, $prepare:literal, $condition:literal, $expected:expr)),+ $(,)?) => {
        $(
            #[test]
            fn $name() {
                let qasm = format!(
                    r#"OPENQASM 2.0;
                    include "qelib1.inc";
                    qreg q[{}]; qreg r[1];
                    creg {}[{}]; creg e[1];
                    {}
                    measure q -> {};
                    if ({}) x r[0];
                    measure r[0] -> e[0];"#,
                    $width, $register, $width, $prepare, $register, $condition,
                );
                assert_eq!(register_values(&qasm, 20, "e", false), vec![$expected; 20]);
            }
        )+
    };
}

unsigned_condition_cases! {
    unsigned_d3_eq3: (2, "d", "x q[0]; x q[1];", "d == 3", 1),
    unsigned_d3_ne0: (2, "d", "x q[0]; x q[1];", "d != 0", 1),
    unsigned_d3_gt2: (2, "d", "x q[0]; x q[1];", "d > 2", 1),
    unsigned_d3_ge3: (2, "d", "x q[0]; x q[1];", "d >= 3", 1),
    unsigned_d3_le3: (2, "d", "x q[0]; x q[1];", "d <= 3", 1),
    unsigned_d3_eq1: (2, "d", "x q[0]; x q[1];", "d == 1", 0),
    unsigned_d3_lt3: (2, "d", "x q[0]; x q[1];", "d < 3", 0),
    unsigned_d3_eq2: (2, "d", "x q[0]; x q[1];", "d == 2", 0),
    unsigned_d3_eq4: (2, "d", "x q[0]; x q[1];", "d == 4", 0),
    unsigned_d2_eq2: (2, "d", "x q[1];", "d == 2", 1),
    unsigned_d2_eq3: (2, "d", "x q[1];", "d == 3", 0),
    unsigned_c1_gt0: (1, "c", "x q[0];", "c > 0", 1),
    unsigned_c1_eq1: (1, "c", "x q[0];", "c == 1", 1),
    unsigned_c1_eq0: (1, "c", "x q[0];", "c == 0", 0),
    unsigned_c0_eq0: (1, "c", "", "c == 0", 1),
    unsigned_c0_gt0: (1, "c", "", "c > 0", 0),
    unsigned_d3_ne3: (2, "d", "x q[0]; x q[1];", "d != 3", 0),
    unsigned_d3_gt3: (2, "d", "x q[0]; x q[1];", "d > 3", 0),
    unsigned_d3_ge4: (2, "d", "x q[0]; x q[1];", "d >= 4", 0),
    unsigned_d3_le2: (2, "d", "x q[0]; x q[1];", "d <= 2", 0),
    unsigned_d2_lt3: (2, "d", "x q[1];", "d < 3", 1),
}

macro_rules! unsigned_assignment_cases {
    (@width) => { 4 };
    (@width $width:literal) => { $width };
    ($($name:ident: ($assignment:literal, $register:literal, $expected:expr $(, $width:literal)?)),+ $(,)?) => {
        $(
            #[test]
            fn $name() {
                let qasm = format!(concat!(
                    "OPENQASM 2.0; include \"qelib1.inc\"; ",
                    "qreg r[1]; creg a[2]; creg b[{}]; creg e[1]; ",
                    $assignment,
                    " measure r[0] -> e[0];",
                ), unsigned_assignment_cases!(@width $($width)?));
                assert_eq!(register_values(&qasm, 20, $register, false), vec![$expected; 20]);
            }
        )+
    };
}

unsigned_assignment_cases! {
    unsigned_assignment_copy: ("a = 3; b = a;", "b", 3),
    unsigned_assignment_negative: ("b = -1;", "b", 15),
    unsigned_assignment_copy_condition: ("a = 3; b = a; if (b == 3) x r[0];", "e", 1),
    unsigned_assignment_copy_false_condition: ("a = 3; b = a; if (b == 15) x r[0];", "e", 0),
    unsigned_assignment_conditional_copy: ("a = 3; if (e == 0) b = a;", "b", 3),
    unsigned_assignment_conditional_negative: ("if (e == 0) b = -1;", "b", 15),
    unsigned_assignment_conditional_skipped: ("a = 3; if (e == 1) b = a;", "b", 0),
    unsigned_assignment_truncation: ("b = 15; a = b;", "a", 3),
    unsigned_literal_c8_eq8: ("creg c[8]; c = 8; if (c == 8) x r[0];", "e", 1),
    unsigned_literal_c64_eq64: ("creg c[8]; c = 64; if (c == 64) x r[0];", "e", 1),
    unsigned_literal_c8_eq9: ("creg c[8]; c = 8; if (c == 9) x r[0];", "e", 0),
    unsigned_assignment_negate_a1: ("creg w[8]; a = 1; w = -a;", "w", 255),
    unsigned_assignment_negate_a3: ("creg w[8]; a = 3; w = -a;", "w", 253),
    unsigned_assignment_conditional_negate_a1: ("creg w[8]; a = 1; if (e == 0) w = -a;", "w", 255),
    unsigned_assignment_conditional_negate_a3: ("creg w[8]; a = 3; if (e == 0) w = -a;", "w", 253),
    unsigned_assignment_bit: ("a[1] = 1; b = a[1];", "b", 1),
    unsigned_assignment_bool: ("a = 3; b = (a == 3);", "b", 1),
    unsigned_assignment_add: ("a = 3; b = a + 1;", "b", 4),
    unsigned_assignment_subtract: ("b = 15; a = b - 12;", "a", 3),
    unsigned_assignment_not_literal_or: ("b = ~1 | 0;", "b", 254, 8),
    unsigned_assignment_not_literal: ("b = ~1;", "b", 254, 8),
    // The register evaluates at width 2, so NOT turns 11 into 00 before assignment zero-extends it.
    unsigned_assignment_not_register: ("a = 3; b = ~a;", "b", 0),
    unsigned_assignment_bitwise_or: ("a = 3; b = a | 0;", "b", 3),
    unsigned_shift_not_right: ("b = ~1 >> 8;", "b", 0),
    unsigned_shift_not_left: ("b = ~1 << 8;", "b", 0),
    unsigned_shift_literal_left: ("b = 1 << 2;", "b", 4),
    unsigned_shift_register_count: ("a = 3; b = 1 << a;", "b", 8),
    unsigned_shift_literal_right: ("b = 8 >> 1;", "b", 4),
}

macro_rules! unsigned_complex_condition_cases {
    ($($name:ident: ($condition:literal, $expected:expr $(, $setup:literal)?)),+ $(,)?) => {
        $(
            #[test]
            fn $name() {
                let qasm = concat!(
                    "OPENQASM 2.0; include \"qelib1.inc\"; ",
                    "qreg r[1]; creg b[4]; creg e[1]; ",
                    $($setup,)?
                    "b = 15; if (",
                    $condition,
                    ") x r[0]; measure r[0] -> e[0];",
                );
                assert_eq!(register_values(qasm, 20, "e", true), vec![$expected; 20]);
            }
        )+
    };
}

unsigned_complex_condition_cases! {
    unsigned_b15_eq_negative1: ("b == -1", 0),
    unsigned_b15_gt_negative1: ("b > -1", 1),
    unsigned_b15_lt0: ("b < 0", 0),
    unsigned_b15_eq15: ("b == 15", 1),
    unsigned_b15_gt15: ("b > 15", 0),
    unsigned_b15_lt16: ("b < 16", 1),
    unsigned_arithmetic_condition: ("a + 1 == 4", 1, "creg a[2]; a = 3; "),
    unsigned_negated_register_gt_negative1: ("-a > -1", 0, "creg a[2]; a = 3; "),
    unsigned_negated_register_lt0: ("-a < 0", 1, "creg a[2]; a = 3; "),
    unsigned_negated_register_eq_negative3: ("-a == -3", 1, "creg a[2]; a = 3; "),
    unsigned_not_condition_width: ("~1 == 14", 1),
    unsigned_not_condition_not_assignment_width: ("~1 == 254", 0),
}

#[test]
fn multi_bit_conditional_fixture_flips_the_qubit_back() {
    // The hardware-validation fixture documents `m = 3, r = 0`: both bits of a
    // 2-bit register are measured as 1 and `if (m == 3)` must fire. Until
    // registers compared as unsigned values that condition was false and the
    // fixture was only ever run through the PHIR converter, never simulated.
    let qasm = include_str!("fixtures/qasm_validation/multi_bit_conditional.qasm");
    assert_eq!(register_values(qasm, 20, "m", false), vec![3; 20]);
    assert_eq!(register_values(qasm, 20, "r", false), vec![0; 20]);
}
