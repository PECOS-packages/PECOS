use pecos_engines::{shot_results::Data, sim_builder, state_vector};
use pecos_programs::Qasm;
use pecos_qasm::qasm_engine;

#[test]
fn test_float_in_classical_expression_error() {
    // Test that float literals in classical expressions produce an error
    let qasm = r"
        OPENQASM 2.0;

        creg c[8];
        c = 3.14;  // This should error
    ";

    let result = sim_builder()
        .classical(qasm_engine().program(Qasm::from_string(qasm)))
        .run(1);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("Float literals are not allowed"));
}

#[test]
fn test_pi_in_classical_expression_error() {
    // Test that pi in classical expressions produces an error
    let qasm = r"
        OPENQASM 2.0;

        creg c[8];
        c = pi;  // This should error
    ";

    let result = sim_builder()
        .classical(qasm_engine().program(Qasm::from_string(qasm)))
        .run(1);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("Pi constant is not allowed"));
}

#[test]
fn test_bitwise_in_gate_parameter_error() {
    // Test that bitwise operations in gate parameters produce an error
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[1];
        rx(1 & 2) q[0];  // This should error
    "#;

    let result = sim_builder()
        .classical(qasm_engine().program(Qasm::from_string(qasm)))
        .run(1);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("not supported in gate parameter"));
}

#[test]
fn test_float_expressions_in_gates_work() {
    // Test that float expressions work correctly in gate parameters
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[2];
        creg c[2];

        // These should all work fine
        rx(pi/2) q[0];
        ry(2.5 * pi) q[1];
        rz(sin(pi/4)) q[0];
        u3(pi, pi/2, -pi/4) q[1];

        measure q -> c;
    "#;

    let result = sim_builder()
        .classical(qasm_engine().program(Qasm::from_string(qasm)))
        .quantum(state_vector())
        .run(1);
    match result {
        Ok(_) => {}
        Err(e) => {
            panic!("Float expressions in gates should work, but got error: {e}");
        }
    }
}

#[test]
fn test_integer_expressions_in_classical_work() {
    // Test that integer expressions work correctly in classical registers
    let qasm = r"
        OPENQASM 2.0;

        creg a[8];
        creg b[8];
        creg c[8];

        a = 5;
        b = 3;
        c = a + b;        // Should be 8
        c = c << 1;       // Should be 16
        c = c | 1;        // Should be 17
        c = c & 255;      // Should be 17
    ";

    let shot_vec = sim_builder()
        .classical(qasm_engine().program(Qasm::from_string(qasm)))
        .run(1)
        .unwrap();
    let shot = &shot_vec.shots[0];

    if let Data::BitVec(c_bits) = &shot.data["c"] {
        // 17 = 10001 in binary
        assert!(c_bits[0], "bit 0 should be 1");
        assert!(!c_bits[1], "bit 1 should be 0");
        assert!(!c_bits[2], "bit 2 should be 0");
        assert!(!c_bits[3], "bit 3 should be 0");
        assert!(c_bits[4], "bit 4 should be 1");
    } else {
        panic!("Expected BitVec for register c");
    }
}
