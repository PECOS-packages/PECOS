//! Integration tests verifying that QASM programs with Clifford-angle rotation gates
//! execute correctly through the full QASM parse -> engine -> simulation pipeline.

use pecos_engines::ClassicalControlEngineBuilder;
use pecos_programs::Qasm;
use pecos_qasm::qasm_engine;

#[test]
fn qasm_rz_pi_acts_as_z_gate() {
    // H -> rz(pi) -> H = X, so |0> -> |1> deterministically.
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];

        h q[0];
        rz(pi) q[0];
        h q[0];
        measure q -> c;
    "#;

    let results = qasm_engine()
        .program(Qasm::from_string(qasm))
        .to_sim()
        .seed(42)
        .workers(1)
        .run(10)
        .unwrap();

    for shot in &results.shots {
        let value = shot.data.get("c").unwrap().as_u32().unwrap();
        assert_eq!(value, 1, "H*Z*H|0> = X|0> = |1>, should always measure 1");
    }
}

#[test]
fn qasm_rz_pi_over_2_acts_as_s_gate() {
    // Two S gates compose to Z: H -> S -> S -> H = H*Z*H = X.
    // |0> -> |1> deterministically.
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];

        h q[0];
        rz(pi/2) q[0];
        rz(pi/2) q[0];
        h q[0];
        measure q -> c;
    "#;

    let results = qasm_engine()
        .program(Qasm::from_string(qasm))
        .to_sim()
        .seed(42)
        .workers(1)
        .run(10)
        .unwrap();

    for shot in &results.shots {
        let value = shot.data.get("c").unwrap().as_u32().unwrap();
        assert_eq!(value, 1, "H*S*S*H|0> = H*Z*H|0> = X|0> = |1>");
    }
}

#[test]
fn qasm_rz_zero_is_identity() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];

        rz(0) q[0];
        measure q -> c;
    "#;

    let results = qasm_engine()
        .program(Qasm::from_string(qasm))
        .to_sim()
        .seed(42)
        .workers(1)
        .run(10)
        .unwrap();

    for shot in &results.shots {
        let value = shot.data.get("c").unwrap().as_u32().unwrap();
        assert_eq!(value, 0, "RZ(0)|0> = |0>");
    }
}

#[test]
fn qasm_four_rz_quarter_turns_compose_to_identity() {
    // 4 * RZ(pi/2) = RZ(2pi) = I. Sandwiched with H: H*I*H = I.
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];

        h q[0];
        rz(pi/2) q[0];
        rz(pi/2) q[0];
        rz(pi/2) q[0];
        rz(pi/2) q[0];
        h q[0];
        measure q -> c;
    "#;

    let results = qasm_engine()
        .program(Qasm::from_string(qasm))
        .to_sim()
        .seed(42)
        .workers(1)
        .run(10)
        .unwrap();

    for shot in &results.shots {
        let value = shot.data.get("c").unwrap().as_u32().unwrap();
        assert_eq!(value, 0, "H * RZ(2pi) * H = I, so |0> -> |0>");
    }
}

#[test]
fn qasm_rzz_pi_over_2_entangles_qubits() {
    // |+,+> -> SZZ -> phase entanglement.
    // (H x H) * SZZ * SZZ * (H x H) = (H x H) * ZZ * (H x H) = XX
    // XX|00> = |11>
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];

        h q[0];
        h q[1];
        rzz(pi/2) q[0], q[1];
        rzz(pi/2) q[0], q[1];
        h q[0];
        h q[1];
        measure q -> c;
    "#;

    let results = qasm_engine()
        .program(Qasm::from_string(qasm))
        .to_sim()
        .seed(42)
        .workers(1)
        .run(10)
        .unwrap();

    for shot in &results.shots {
        let value = shot.data.get("c").unwrap().as_u32().unwrap();
        assert_eq!(value, 0b11, "XX|00> = |11>");
    }
}

#[test]
fn qasm_rx_pi_via_decomposition_acts_as_x() {
    // In qelib1.inc, rx(theta) decomposes to h; rz(theta); h.
    // rx(pi) = H*Z*H = X. So |0> -> |1>.
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];

        rx(pi) q[0];
        measure q -> c;
    "#;

    let results = qasm_engine()
        .program(Qasm::from_string(qasm))
        .to_sim()
        .seed(42)
        .workers(1)
        .run(10)
        .unwrap();

    for shot in &results.shots {
        let value = shot.data.get("c").unwrap().as_u32().unwrap();
        assert_eq!(value, 1, "rx(pi) = X, so |0> -> |1>");
    }
}

#[test]
fn qasm_hqslib1_rz_pi_acts_as_z() {
    // Same test but using hqslib1.inc where rz maps to native RZ.
    let qasm = r#"
        OPENQASM 2.0;
        include "hqslib1.inc";
        qreg q[1];
        creg c[1];

        h q[0];
        rz(pi) q[0];
        h q[0];
        measure q -> c;
    "#;

    let results = qasm_engine()
        .program(Qasm::from_string(qasm))
        .to_sim()
        .seed(42)
        .workers(1)
        .run(10)
        .unwrap();

    for shot in &results.shots {
        let value = shot.data.get("c").unwrap().as_u32().unwrap();
        assert_eq!(value, 1, "H*Z*H|0> = |1>");
    }
}

#[test]
fn qasm_hqslib1_rx_pi_via_r1xy_acts_as_x() {
    // In hqslib1.inc, rx(theta) maps to R1XY(theta, 0).
    // R1XY(pi, 0) = X. So |0> -> |1>.
    let qasm = r#"
        OPENQASM 2.0;
        include "hqslib1.inc";
        qreg q[1];
        creg c[1];

        rx(pi) q[0];
        measure q -> c;
    "#;

    let results = qasm_engine()
        .program(Qasm::from_string(qasm))
        .to_sim()
        .seed(42)
        .workers(1)
        .run(10)
        .unwrap();

    for shot in &results.shots {
        let value = shot.data.get("c").unwrap().as_u32().unwrap();
        assert_eq!(value, 1, "R1XY(pi, 0) = X, so |0> -> |1>");
    }
}

#[test]
fn qasm_hqslib1_ry_pi_via_r1xy_acts_as_y() {
    // In hqslib1.inc, ry(theta) maps to R1XY(theta, pi/2).
    // R1XY(pi, pi/2) = Y. Y|0> = i|1>, outcome 1.
    let qasm = r#"
        OPENQASM 2.0;
        include "hqslib1.inc";
        qreg q[1];
        creg c[1];

        ry(pi) q[0];
        measure q -> c;
    "#;

    let results = qasm_engine()
        .program(Qasm::from_string(qasm))
        .to_sim()
        .seed(42)
        .workers(1)
        .run(10)
        .unwrap();

    for shot in &results.shots {
        let value = shot.data.get("c").unwrap().as_u32().unwrap();
        assert_eq!(value, 1, "R1XY(pi, pi/2) = Y, so |0> -> |1>");
    }
}

// --- Negative angle tests ---

#[test]
fn qasm_rz_negative_pi_over_2_acts_as_sdg() {
    // RZ(-pi/2) = Sdg. Sdg * S = I.
    // H -> rz(-pi/2) -> rz(pi/2) -> H = H*I*H = I. |0> -> |0>
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];

        h q[0];
        rz(-pi/2) q[0];
        rz(pi/2) q[0];
        h q[0];
        measure q -> c;
    "#;

    let results = qasm_engine()
        .program(Qasm::from_string(qasm))
        .to_sim()
        .seed(42)
        .workers(1)
        .run(10)
        .unwrap();

    for shot in &results.shots {
        let value = shot.data.get("c").unwrap().as_u32().unwrap();
        assert_eq!(value, 0, "H * Sdg * S * H = I, so |0> -> |0>");
    }
}

#[test]
fn qasm_rz_negative_pi_acts_as_z() {
    // RZ(-pi) = Z (same as RZ(pi) up to global phase).
    // H -> Z -> H = X. |0> -> |1>
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];

        h q[0];
        rz(-pi) q[0];
        h q[0];
        measure q -> c;
    "#;

    let results = qasm_engine()
        .program(Qasm::from_string(qasm))
        .to_sim()
        .seed(42)
        .workers(1)
        .run(10)
        .unwrap();

    for shot in &results.shots {
        let value = shot.data.get("c").unwrap().as_u32().unwrap();
        assert_eq!(value, 1, "H * Z * H = X, so |0> -> |1>");
    }
}

#[test]
fn qasm_sdg_s_cancel_via_rotation() {
    // sdg then s = identity. Both are rz(-pi/2) and rz(pi/2) under the hood.
    // Sandwich with H to detect: H * I * H = I. |0> -> |0>
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];

        h q[0];
        sdg q[0];
        s q[0];
        h q[0];
        measure q -> c;
    "#;

    let results = qasm_engine()
        .program(Qasm::from_string(qasm))
        .to_sim()
        .seed(42)
        .workers(1)
        .run(10)
        .unwrap();

    for shot in &results.shots {
        let value = shot.data.get("c").unwrap().as_u32().unwrap();
        assert_eq!(value, 0, "Sdg * S = I");
    }
}

// --- U gate tests ---

#[test]
fn qasm_u_z_gate() {
    // u(0, 0, pi) = RZ(pi) = Z. Sandwich with H: H*Z*H = X.
    // |0> -> |1>
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];

        h q[0];
        u(0, 0, pi) q[0];
        h q[0];
        measure q -> c;
    "#;

    let results = qasm_engine()
        .program(Qasm::from_string(qasm))
        .to_sim()
        .seed(42)
        .workers(1)
        .run(10)
        .unwrap();

    for shot in &results.shots {
        let value = shot.data.get("c").unwrap().as_u32().unwrap();
        assert_eq!(value, 1, "u(0,0,pi) = Z, so H*Z*H = X, |0> -> |1>");
    }
}

#[test]
fn qasm_u_x_gate() {
    // u(pi, 0, pi) decomposes to RZ(0)*RY(pi)*RZ(pi) = Y*Z = iX.
    // iX|0> = i|1>, outcome 1.
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];

        u(pi, 0, pi) q[0];
        measure q -> c;
    "#;

    let results = qasm_engine()
        .program(Qasm::from_string(qasm))
        .to_sim()
        .seed(42)
        .workers(1)
        .run(10)
        .unwrap();

    for shot in &results.shots {
        let value = shot.data.get("c").unwrap().as_u32().unwrap();
        assert_eq!(value, 1, "u(pi, 0, pi) acts as X, |0> -> |1>");
    }
}

#[test]
fn qasm_u_s_gate() {
    // u(0, 0, pi/2) = RZ(pi/2) = S. Two S = Z.
    // H -> S -> S -> H = H*Z*H = X.
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];

        h q[0];
        u(0, 0, pi/2) q[0];
        u(0, 0, pi/2) q[0];
        h q[0];
        measure q -> c;
    "#;

    let results = qasm_engine()
        .program(Qasm::from_string(qasm))
        .to_sim()
        .seed(42)
        .workers(1)
        .run(10)
        .unwrap();

    for shot in &results.shots {
        let value = shot.data.get("c").unwrap().as_u32().unwrap();
        assert_eq!(value, 1, "Two u(0,0,pi/2) = Z, H*Z*H = X");
    }
}
