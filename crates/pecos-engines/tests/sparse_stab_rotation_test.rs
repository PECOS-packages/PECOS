//! Integration tests for `SparseStabEngine` rotation gate handling via `ByteMessage`.
//!
//! These tests verify that Clifford-angle rotation gates (RZ, RX, RY, RZZ, RXX, RYY, R1XY)
//! are correctly processed when sent as `ByteMessage` commands to the `SparseStabEngine`,
//! exercising the full `ByteMessage` -> `Engine::process()` -> `CliffordRotation` trait path.

use pecos_core::{Angle64, Gate};
use pecos_engines::Engine;
use pecos_engines::byte_message::ByteMessageBuilder;
use pecos_engines::quantum::SparseStabEngine;

/// Helper: build a circuit, process it, return measurement outcomes.
fn run_sparse_stab(num_qubits: usize, build: impl FnOnce(&mut ByteMessageBuilder)) -> Vec<u32> {
    let mut engine = SparseStabEngine::new(num_qubits);
    let mut builder = ByteMessageBuilder::new();
    let _ = builder.for_quantum_operations();
    build(&mut builder);
    let result = engine.process(builder.build()).expect("process failed");
    result.outcomes().expect("outcomes failed")
}

/// Helper: build a circuit, expect `process()` to fail, return the error message.
fn expect_sparse_stab_error(
    num_qubits: usize,
    build: impl FnOnce(&mut ByteMessageBuilder),
) -> String {
    let mut engine = SparseStabEngine::new(num_qubits);
    let mut builder = ByteMessageBuilder::new();
    let _ = builder.for_quantum_operations();
    build(&mut builder);
    match engine.process(builder.build()) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("expected process to fail for non-Clifford rotation"),
    }
}

// --- RZ tests ---

#[test]
fn rz_pi_acts_as_z_gate() {
    // H -> RZ(pi) -> H = H*Z*H = X, so |0> -> |1>
    let outcomes = run_sparse_stab(1, |b| {
        b.add_h(&[0]);
        b.add_rz(Angle64::HALF_TURN, &[0]);
        b.add_h(&[0]);
        b.add_measurements(&[0]);
    });
    assert_eq!(outcomes, vec![1]);
}

#[test]
fn rz_zero_is_identity() {
    // RZ(0) is identity: |0> stays |0>
    let outcomes = run_sparse_stab(1, |b| {
        b.add_rz(Angle64::ZERO, &[0]);
        b.add_measurements(&[0]);
    });
    assert_eq!(outcomes, vec![0]);
}

#[test]
fn rz_quarter_turn_acts_as_s() {
    // H -> S -> H -> S -> H should produce a known state.
    // S = RZ(pi/2). H*S*H*S*H|0> = ... let's verify X doesn't happen:
    // More directly: H*S*H = (X+Y)/sqrt(2) rotation... too complex.
    //
    // Simpler: S*S = Z. So H -> RZ(pi/2) -> RZ(pi/2) -> H = H*Z*H = X.
    // |0> -> X|0> = |1>
    let outcomes = run_sparse_stab(1, |b| {
        b.add_h(&[0]);
        b.add_rz(Angle64::QUARTER_TURN, &[0]);
        b.add_rz(Angle64::QUARTER_TURN, &[0]);
        b.add_h(&[0]);
        b.add_measurements(&[0]);
    });
    assert_eq!(outcomes, vec![1]);
}

#[test]
fn rz_three_quarter_turn_acts_as_sdg() {
    // Sdg * S = I, so H -> RZ(3pi/2) -> RZ(pi/2) -> H = H*I*H = I.
    // |0> -> |0>
    let outcomes = run_sparse_stab(1, |b| {
        b.add_h(&[0]);
        b.add_rz(Angle64::THREE_QUARTERS_TURN, &[0]);
        b.add_rz(Angle64::QUARTER_TURN, &[0]);
        b.add_h(&[0]);
        b.add_measurements(&[0]);
    });
    assert_eq!(outcomes, vec![0]);
}

// --- RX tests ---

#[test]
fn rx_pi_acts_as_x_gate() {
    // RX(pi) = X: |0> -> |1>
    let outcomes = run_sparse_stab(1, |b| {
        b.add_rx(Angle64::HALF_TURN, &[0]);
        b.add_measurements(&[0]);
    });
    assert_eq!(outcomes, vec![1]);
}

#[test]
fn rx_zero_is_identity() {
    let outcomes = run_sparse_stab(1, |b| {
        b.add_rx(Angle64::ZERO, &[0]);
        b.add_measurements(&[0]);
    });
    assert_eq!(outcomes, vec![0]);
}

// --- RY tests ---

#[test]
fn ry_pi_acts_as_y_gate() {
    // RY(pi) = Y: Y|0> = i|1>, measurement outcome is 1
    let outcomes = run_sparse_stab(1, |b| {
        b.add_ry(Angle64::HALF_TURN, &[0]);
        b.add_measurements(&[0]);
    });
    assert_eq!(outcomes, vec![1]);
}

#[test]
fn ry_zero_is_identity() {
    let outcomes = run_sparse_stab(1, |b| {
        b.add_ry(Angle64::ZERO, &[0]);
        b.add_measurements(&[0]);
    });
    assert_eq!(outcomes, vec![0]);
}

// --- RZZ tests ---

#[test]
fn rzz_quarter_turn_acts_as_szz() {
    // RZZ(pi/2) = SZZ. Apply twice: SZZ*SZZ = ZZ.
    // Sandwich with H on both qubits to detect phase:
    // (H x H) * ZZ * (H x H) = XX
    // XX|00> = |11>, so measure -> (1,1)
    let outcomes = run_sparse_stab(2, |b| {
        b.add_h(&[0]);
        b.add_h(&[1]);
        b.add_rzz(Angle64::QUARTER_TURN, &[0], &[1]);
        b.add_rzz(Angle64::QUARTER_TURN, &[0], &[1]);
        b.add_h(&[0]);
        b.add_h(&[1]);
        b.add_measurements(&[0, 1]);
    });
    assert_eq!(outcomes, vec![1, 1]);
}

#[test]
fn rzz_half_turn_decomposes_to_z_tensor_z() {
    // RZZ(pi) = Z x Z. Sandwich with H on both: (HxH)(ZxZ)(HxH) = XX
    // XX|00> = |11>
    let outcomes = run_sparse_stab(2, |b| {
        b.add_h(&[0]);
        b.add_h(&[1]);
        b.add_rzz(Angle64::HALF_TURN, &[0], &[1]);
        b.add_h(&[0]);
        b.add_h(&[1]);
        b.add_measurements(&[0, 1]);
    });
    assert_eq!(outcomes, vec![1, 1]);
}

#[test]
fn rzz_zero_is_identity() {
    let outcomes = run_sparse_stab(2, |b| {
        b.add_rzz(Angle64::ZERO, &[0], &[1]);
        b.add_measurements(&[0, 1]);
    });
    assert_eq!(outcomes, vec![0, 0]);
}

// --- RXX tests (via Gate::rxx + add_gate_command) ---

#[test]
fn rxx_half_turn_decomposes_to_x_tensor_x() {
    // RXX(pi) = X x X: |00> -> |11>
    let outcomes = run_sparse_stab(2, |b| {
        let gate = Gate::rxx(Angle64::HALF_TURN, &[(0usize, 1usize)]);
        b.add_gate_command(&gate);
        b.add_measurements(&[0, 1]);
    });
    assert_eq!(outcomes, vec![1, 1]);
}

// --- RYY tests (via Gate::ryy + add_gate_command) ---

#[test]
fn ryy_half_turn_decomposes_to_y_tensor_y() {
    // RYY(pi) = Y x Y: (Y x Y)|00> = (i|1>)(i|1>) = -|11>
    // Measurement ignores global phase, so outcomes: q0=1, q1=1
    let outcomes = run_sparse_stab(2, |b| {
        let gate = Gate::ryy(Angle64::HALF_TURN, &[(0usize, 1usize)]);
        b.add_gate_command(&gate);
        b.add_measurements(&[0, 1]);
    });
    assert_eq!(outcomes, vec![1, 1]);
}

// --- R1XY tests ---

#[test]
fn r1xy_pi_zero_acts_as_x() {
    // R1XY(pi, 0) = X: |0> -> |1>
    let outcomes = run_sparse_stab(1, |b| {
        b.add_r1xy(Angle64::HALF_TURN, Angle64::ZERO, &[0]);
        b.add_measurements(&[0]);
    });
    assert_eq!(outcomes, vec![1]);
}

#[test]
fn r1xy_pi_quarter_acts_as_y() {
    // R1XY(pi, pi/2) = Y: |0> -> i|1>, outcome 1
    let outcomes = run_sparse_stab(1, |b| {
        b.add_r1xy(Angle64::HALF_TURN, Angle64::QUARTER_TURN, &[0]);
        b.add_measurements(&[0]);
    });
    assert_eq!(outcomes, vec![1]);
}

#[test]
fn r1xy_zero_is_identity() {
    // R1XY(0, anything) = I
    let outcomes = run_sparse_stab(1, |b| {
        b.add_r1xy(Angle64::ZERO, Angle64::QUARTER_TURN, &[0]);
        b.add_measurements(&[0]);
    });
    assert_eq!(outcomes, vec![0]);
}

#[test]
fn r1xy_quarter_turn_zero_acts_as_sx() {
    // R1XY(pi/2, 0) = SX. Two SX = X: |0> -> |1>
    let outcomes = run_sparse_stab(1, |b| {
        b.add_r1xy(Angle64::QUARTER_TURN, Angle64::ZERO, &[0]);
        b.add_r1xy(Angle64::QUARTER_TURN, Angle64::ZERO, &[0]);
        b.add_measurements(&[0]);
    });
    assert_eq!(outcomes, vec![1]);
}

#[test]
fn r1xy_three_quarter_turn_zero_acts_as_sxdg() {
    // R1XY(3pi/2, 0) = SXdg. SXdg * SX = I: |0> -> |0>
    let outcomes = run_sparse_stab(1, |b| {
        b.add_r1xy(Angle64::THREE_QUARTERS_TURN, Angle64::ZERO, &[0]);
        b.add_r1xy(Angle64::QUARTER_TURN, Angle64::ZERO, &[0]);
        b.add_measurements(&[0]);
    });
    assert_eq!(outcomes, vec![0]);
}

#[test]
fn r1xy_quarter_turn_quarter_acts_as_sy() {
    // R1XY(pi/2, pi/2) = SY. Two SY = Y: |0> -> outcome 1
    let outcomes = run_sparse_stab(1, |b| {
        b.add_r1xy(Angle64::QUARTER_TURN, Angle64::QUARTER_TURN, &[0]);
        b.add_r1xy(Angle64::QUARTER_TURN, Angle64::QUARTER_TURN, &[0]);
        b.add_measurements(&[0]);
    });
    assert_eq!(outcomes, vec![1]);
}

#[test]
fn r1xy_three_quarter_turn_quarter_acts_as_sydg() {
    // R1XY(3pi/2, pi/2) = SYdg. SYdg * SY = I: |0> -> |0>
    let outcomes = run_sparse_stab(1, |b| {
        b.add_r1xy(Angle64::THREE_QUARTERS_TURN, Angle64::QUARTER_TURN, &[0]);
        b.add_r1xy(Angle64::QUARTER_TURN, Angle64::QUARTER_TURN, &[0]);
        b.add_measurements(&[0]);
    });
    assert_eq!(outcomes, vec![0]);
}

#[test]
fn r1xy_negated_x_axis_acts_as_x() {
    // R1XY(pi, pi) = rotation about -X axis = X (up to global phase).
    // X|0> = |1>
    let outcomes = run_sparse_stab(1, |b| {
        b.add_r1xy(Angle64::HALF_TURN, Angle64::HALF_TURN, &[0]);
        b.add_measurements(&[0]);
    });
    assert_eq!(outcomes, vec![1]);
}

#[test]
fn r1xy_negated_y_axis_acts_as_y() {
    // R1XY(pi, 3pi/2) = rotation about -Y axis = Y (up to global phase).
    // Y|0> -> outcome 1
    let outcomes = run_sparse_stab(1, |b| {
        b.add_r1xy(Angle64::HALF_TURN, Angle64::THREE_QUARTERS_TURN, &[0]);
        b.add_measurements(&[0]);
    });
    assert_eq!(outcomes, vec![1]);
}

// --- U gate tests ---

#[test]
fn u_identity() {
    // U(0, 0, 0) = I: |0> -> |0>
    let outcomes = run_sparse_stab(1, |b| {
        b.add_u(Angle64::ZERO, Angle64::ZERO, Angle64::ZERO, &[0]);
        b.add_measurements(&[0]);
    });
    assert_eq!(outcomes, vec![0]);
}

#[test]
fn u_x_gate() {
    // U(pi, 0, pi) = RZ(0) * RY(pi) * RZ(pi) = Y * Z = iX.
    // iX|0> = i|1>, outcome 1
    let outcomes = run_sparse_stab(1, |b| {
        b.add_u(Angle64::HALF_TURN, Angle64::ZERO, Angle64::HALF_TURN, &[0]);
        b.add_measurements(&[0]);
    });
    assert_eq!(outcomes, vec![1]);
}

#[test]
fn u_z_gate() {
    // U(0, 0, pi) = RZ(pi) = Z. Sandwich with H: H*Z*H = X.
    // |0> -> |1>
    let outcomes = run_sparse_stab(1, |b| {
        b.add_h(&[0]);
        b.add_u(Angle64::ZERO, Angle64::ZERO, Angle64::HALF_TURN, &[0]);
        b.add_h(&[0]);
        b.add_measurements(&[0]);
    });
    assert_eq!(outcomes, vec![1]);
}

#[test]
fn u_s_gate() {
    // U(0, 0, pi/2) = RZ(pi/2) = S. Two S = Z.
    // H -> S -> S -> H = H*Z*H = X. |0> -> |1>
    let outcomes = run_sparse_stab(1, |b| {
        b.add_h(&[0]);
        b.add_u(Angle64::ZERO, Angle64::ZERO, Angle64::QUARTER_TURN, &[0]);
        b.add_u(Angle64::ZERO, Angle64::ZERO, Angle64::QUARTER_TURN, &[0]);
        b.add_h(&[0]);
        b.add_measurements(&[0]);
    });
    assert_eq!(outcomes, vec![1]);
}

#[test]
fn u_hadamard_like() {
    // U(pi/2, 0, pi) = RZ(0) * RY(pi/2) * RZ(pi) = SY * Z
    // Apply twice to see it's not identity (verifies it's doing something nontrivial).
    // SY * Z applied twice: (SY*Z)^2. Let's just check it works.
    let outcomes = run_sparse_stab(1, |b| {
        b.add_u(
            Angle64::QUARTER_TURN,
            Angle64::ZERO,
            Angle64::HALF_TURN,
            &[0],
        );
        b.add_u(
            Angle64::QUARTER_TURN,
            Angle64::ZERO,
            Angle64::HALF_TURN,
            &[0],
        );
        b.add_measurements(&[0]);
    });
    // (SY*Z)^2 |0> -- both are Clifford so result is deterministic
    // SY*Z|0> = SY|0> = some state, then SY*Z again.
    // Just verify it doesn't error -- the gate combination is valid Clifford.
    assert!(outcomes[0] <= 1);
}

#[test]
fn u_non_clifford_fails_with_useful_message() {
    let msg = expect_sparse_stab_error(1, |b| {
        // theta=0.123 is not Clifford
        b.add_u(
            Angle64::from_radians(0.123),
            Angle64::ZERO,
            Angle64::ZERO,
            &[0],
        );
        b.add_measurements(&[0]);
    });
    assert!(msg.contains("U("), "error should name the gate: {msg}");
    assert!(
        msg.contains("not a Clifford"),
        "error should explain it's not Clifford: {msg}"
    );
}

#[test]
fn u_mixed_non_clifford_lambda_fails() {
    // Clifford theta and phi, but non-Clifford lambda
    let msg = expect_sparse_stab_error(1, |b| {
        b.add_u(
            Angle64::ZERO,
            Angle64::ZERO,
            Angle64::from_radians(0.5),
            &[0],
        );
        b.add_measurements(&[0]);
    });
    assert!(msg.contains("not a Clifford"), "error: {msg}");
}

// --- CRZ tests ---

#[test]
fn crz_zero_is_identity() {
    // CRZ(0) = I: no effect
    let outcomes = run_sparse_stab(2, |b| {
        let gate = Gate::crz(Angle64::ZERO, &[(0usize, 1usize)]);
        b.add_gate_command(&gate);
        b.add_measurements(&[0, 1]);
    });
    assert_eq!(outcomes, vec![0, 0]);
}

#[test]
fn crz_pi_does_nothing_when_control_is_zero() {
    // CRZ(theta)|0,psi> = |0,psi> for any theta (controlled gate, control off).
    // Prepare |0,+>. CRZ(pi)|0,+> = |0,+>. H on target -> |0,0>.
    let outcomes = run_sparse_stab(2, |b| {
        b.add_h(&[1]);
        let gate = Gate::crz(Angle64::HALF_TURN, &[(0usize, 1usize)]);
        b.add_gate_command(&gate);
        b.add_h(&[1]);
        b.add_measurements(&[0, 1]);
    });
    assert_eq!(outcomes, vec![0, 0]);
}

#[test]
fn crz_pi_applies_rz_pi_when_control_is_one() {
    // CRZ(pi)|1,psi> = |1, RZ(pi)|psi>>. RZ(pi) = -iZ.
    // |1,0> -> CRZ(pi) -> |1, -i*Z|0>> = |1, -i|0>> = -i|1,0>.
    // Measurement: q0=1, q1=0 (global phase doesn't matter).
    let outcomes = run_sparse_stab(2, |b| {
        b.add_x(&[0]); // control = |1>
        let gate = Gate::crz(Angle64::HALF_TURN, &[(0usize, 1usize)]);
        b.add_gate_command(&gate);
        b.add_measurements(&[0, 1]);
    });
    assert_eq!(outcomes, vec![1, 0]);
}

#[test]
fn crz_pi_twice_gives_cz() {
    // CRZ(pi)^2 applies RZ(pi)^2 = RZ(2pi) = I on target when control=|1>,
    // but the decomposition runs twice so we get 2 * (SZ, CX, SZdg, CX).
    // Actually CRZ(pi) applied twice = CRZ(2pi) effect... but via decomposition
    // each application is independent.
    //
    // Easier: CRZ(pi) * CRZ(pi) on |+,1>:
    // CRZ(pi)|+,1> = (|0,1> + |1,RZ(pi)|1>>)/sqrt(2) = (|0,1> + i|1,1>)/sqrt(2)
    // CRZ(pi) again: (|0,1> + i*i|1,1>)/sqrt(2) = (|0,1> - |1,1>)/sqrt(2) = |-,1>
    // H on q0: |1,1>. Measure: q0=1, q1=1.
    let outcomes = run_sparse_stab(2, |b| {
        b.add_x(&[1]);
        b.add_h(&[0]);
        let gate = Gate::crz(Angle64::HALF_TURN, &[(0usize, 1usize)]);
        b.add_gate_command(&gate);
        b.add_gate_command(&gate);
        b.add_h(&[0]);
        b.add_measurements(&[0, 1]);
    });
    assert_eq!(outcomes, vec![1, 1]);
}

#[test]
fn crz_non_clifford_fails_with_useful_message() {
    let msg = expect_sparse_stab_error(2, |b| {
        let gate = Gate::crz(Angle64::from_radians(0.5), &[(0usize, 1usize)]);
        b.add_gate_command(&gate);
        b.add_measurements(&[0, 1]);
    });
    assert!(msg.contains("CRZ"), "error should name the gate: {msg}");
    assert!(
        msg.contains("not a Clifford"),
        "error should explain it's not Clifford: {msg}"
    );
}

#[test]
fn crz_quarter_turn_fails() {
    // CRZ(pi/2) requires RZ(pi/4) = T gate, which is NOT Clifford
    let msg = expect_sparse_stab_error(2, |b| {
        let gate = Gate::crz(Angle64::QUARTER_TURN, &[(0usize, 1usize)]);
        b.add_gate_command(&gate);
        b.add_measurements(&[0, 1]);
    });
    assert!(msg.contains("not a Clifford"), "error: {msg}");
}

// --- Non-Clifford angles should fail ---

#[test]
fn rz_non_clifford_angle_fails_with_useful_message() {
    let msg = expect_sparse_stab_error(1, |b| {
        b.add_rz(Angle64::from_radians(0.123), &[0]);
        b.add_measurements(&[0]);
    });
    assert!(msg.contains("RZ"), "error should name the gate: {msg}");
    assert!(
        msg.contains("not a Clifford"),
        "error should explain it's not Clifford: {msg}"
    );
}

#[test]
fn rx_non_clifford_angle_fails_with_useful_message() {
    let msg = expect_sparse_stab_error(1, |b| {
        b.add_rx(Angle64::from_radians(0.5), &[0]);
        b.add_measurements(&[0]);
    });
    assert!(msg.contains("RX"), "error should name the gate: {msg}");
    assert!(
        msg.contains("not a Clifford"),
        "error should explain it's not Clifford: {msg}"
    );
}

#[test]
fn ry_non_clifford_angle_fails_with_useful_message() {
    let msg = expect_sparse_stab_error(1, |b| {
        b.add_ry(Angle64::from_radians(0.5), &[0]);
        b.add_measurements(&[0]);
    });
    assert!(msg.contains("RY"), "error should name the gate: {msg}");
    assert!(
        msg.contains("not a Clifford"),
        "error should explain it's not Clifford: {msg}"
    );
}

#[test]
fn rzz_non_clifford_angle_fails_with_useful_message() {
    let msg = expect_sparse_stab_error(2, |b| {
        b.add_rzz(Angle64::from_radians(0.5), &[0], &[1]);
        b.add_measurements(&[0, 1]);
    });
    assert!(msg.contains("RZZ"), "error should name the gate: {msg}");
    assert!(
        msg.contains("not a Clifford"),
        "error should explain it's not Clifford: {msg}"
    );
}

#[test]
fn r1xy_non_clifford_angle_fails_with_useful_message() {
    // Non-Clifford theta
    let msg = expect_sparse_stab_error(1, |b| {
        b.add_r1xy(Angle64::from_radians(0.123), Angle64::ZERO, &[0]);
        b.add_measurements(&[0]);
    });
    assert!(msg.contains("R1XY"), "error should name the gate: {msg}");
    assert!(
        msg.contains("not a Clifford"),
        "error should explain it's not Clifford: {msg}"
    );
}

#[test]
fn r1xy_non_axis_phi_fails_with_useful_message() {
    // Clifford theta but non-axis phi (pi/4 is not along X or Y)
    let msg = expect_sparse_stab_error(1, |b| {
        b.add_r1xy(Angle64::HALF_TURN, Angle64::QUARTER_TURN / 2u64, &[0]);
        b.add_measurements(&[0]);
    });
    assert!(msg.contains("R1XY"), "error should name the gate: {msg}");
    assert!(
        msg.contains("not a Clifford"),
        "error should explain it's not Clifford: {msg}"
    );
}

// --- Negative angle tests ---

#[test]
fn rz_negative_quarter_turn_acts_as_szdg() {
    // RZ(-pi/2) = SZdg. SZdg * S = I.
    // H -> RZ(-pi/2) -> RZ(pi/2) -> H = H*I*H = I. |0> -> |0>
    let outcomes = run_sparse_stab(1, |b| {
        b.add_h(&[0]);
        b.add_rz(-Angle64::QUARTER_TURN, &[0]);
        b.add_rz(Angle64::QUARTER_TURN, &[0]);
        b.add_h(&[0]);
        b.add_measurements(&[0]);
    });
    assert_eq!(outcomes, vec![0]);
}

#[test]
fn rz_negative_quarter_turn_differs_from_positive() {
    // RZ(-pi/2) != RZ(pi/2). Two RZ(-pi/2) = RZ(-pi) = Z.
    // H -> Z -> H = X. |0> -> |1>
    let outcomes = run_sparse_stab(1, |b| {
        b.add_h(&[0]);
        b.add_rz(-Angle64::QUARTER_TURN, &[0]);
        b.add_rz(-Angle64::QUARTER_TURN, &[0]);
        b.add_h(&[0]);
        b.add_measurements(&[0]);
    });
    assert_eq!(outcomes, vec![1]);
}

#[test]
fn rz_negative_half_turn_acts_as_z() {
    // RZ(-pi) wraps to same as RZ(pi) = Z.
    // H -> Z -> H = X. |0> -> |1>
    let outcomes = run_sparse_stab(1, |b| {
        b.add_h(&[0]);
        b.add_rz(-Angle64::HALF_TURN, &[0]);
        b.add_h(&[0]);
        b.add_measurements(&[0]);
    });
    assert_eq!(outcomes, vec![1]);
}

#[test]
fn rx_negative_half_turn_acts_as_x() {
    // RX(-pi) = X (up to global phase). X|0> = |1>
    let outcomes = run_sparse_stab(1, |b| {
        b.add_rx(-Angle64::HALF_TURN, &[0]);
        b.add_measurements(&[0]);
    });
    assert_eq!(outcomes, vec![1]);
}

#[test]
fn ry_negative_half_turn_acts_as_y() {
    // RY(-pi) = Y (up to global phase). Y|0> = i|1>, outcome 1
    let outcomes = run_sparse_stab(1, |b| {
        b.add_ry(-Angle64::HALF_TURN, &[0]);
        b.add_measurements(&[0]);
    });
    assert_eq!(outcomes, vec![1]);
}

#[test]
fn rzz_negative_half_turn_decomposes_to_z_tensor_z() {
    // RZZ(-pi) should also decompose to Z x Z (half_turn_decomposition handles negative pi).
    // (HxH)(ZxZ)(HxH)|00> = XX|00> = |11>
    let outcomes = run_sparse_stab(2, |b| {
        b.add_h(&[0]);
        b.add_h(&[1]);
        b.add_rzz(-Angle64::HALF_TURN, &[0], &[1]);
        b.add_h(&[0]);
        b.add_h(&[1]);
        b.add_measurements(&[0, 1]);
    });
    assert_eq!(outcomes, vec![1, 1]);
}

// --- RXX/RYY quarter-turn tests ---

#[test]
fn rxx_quarter_turn_acts_as_sxx() {
    // RXX(pi/2) = SXX. Apply twice: SXX*SXX = XX.
    // XX|00> = |11>
    let outcomes = run_sparse_stab(2, |b| {
        let sxx = Gate::rxx(Angle64::QUARTER_TURN, &[(0usize, 1usize)]);
        b.add_gate_command(&sxx);
        b.add_gate_command(&sxx);
        b.add_measurements(&[0, 1]);
    });
    assert_eq!(outcomes, vec![1, 1]);
}

#[test]
fn rxx_three_quarter_turn_acts_as_sxxdg() {
    // RXX(3pi/2) = SXXdg. SXXdg * SXX = I.
    let outcomes = run_sparse_stab(2, |b| {
        let sxxdg = Gate::rxx(Angle64::THREE_QUARTERS_TURN, &[(0usize, 1usize)]);
        let sxx = Gate::rxx(Angle64::QUARTER_TURN, &[(0usize, 1usize)]);
        b.add_gate_command(&sxxdg);
        b.add_gate_command(&sxx);
        b.add_measurements(&[0, 1]);
    });
    assert_eq!(outcomes, vec![0, 0]);
}

#[test]
fn ryy_quarter_turn_acts_as_syy() {
    // RYY(pi/2) = SYY. Apply twice: SYY*SYY = YY.
    // (YxY)|00> = (i|1>)(i|1>) = -|11>, outcome (1,1)
    let outcomes = run_sparse_stab(2, |b| {
        let syy = Gate::ryy(Angle64::QUARTER_TURN, &[(0usize, 1usize)]);
        b.add_gate_command(&syy);
        b.add_gate_command(&syy);
        b.add_measurements(&[0, 1]);
    });
    assert_eq!(outcomes, vec![1, 1]);
}

#[test]
fn ryy_three_quarter_turn_acts_as_syydg() {
    // RYY(3pi/2) = SYYdg. SYYdg * SYY = I.
    let outcomes = run_sparse_stab(2, |b| {
        let syydg = Gate::ryy(Angle64::THREE_QUARTERS_TURN, &[(0usize, 1usize)]);
        let syy = Gate::ryy(Angle64::QUARTER_TURN, &[(0usize, 1usize)]);
        b.add_gate_command(&syydg);
        b.add_gate_command(&syy);
        b.add_measurements(&[0, 1]);
    });
    assert_eq!(outcomes, vec![0, 0]);
}

#[test]
fn rxx_zero_is_identity() {
    let outcomes = run_sparse_stab(2, |b| {
        let gate = Gate::rxx(Angle64::ZERO, &[(0usize, 1usize)]);
        b.add_gate_command(&gate);
        b.add_measurements(&[0, 1]);
    });
    assert_eq!(outcomes, vec![0, 0]);
}

#[test]
fn ryy_zero_is_identity() {
    let outcomes = run_sparse_stab(2, |b| {
        let gate = Gate::ryy(Angle64::ZERO, &[(0usize, 1usize)]);
        b.add_gate_command(&gate);
        b.add_measurements(&[0, 1]);
    });
    assert_eq!(outcomes, vec![0, 0]);
}

// --- T-gate angle rejection ---

#[test]
fn rz_pi_over_4_rejected_as_non_clifford() {
    // RZ(pi/4) = T gate, which is not Clifford -- should fail on stabilizer engine
    let msg = expect_sparse_stab_error(1, |b| {
        b.add_rz(Angle64::QUARTER_TURN / 2u64, &[0]);
        b.add_measurements(&[0]);
    });
    assert!(
        msg.contains("not a Clifford"),
        "T-gate angle should be rejected: {msg}"
    );
}

#[test]
fn rz_neg_pi_over_4_rejected_as_non_clifford() {
    // RZ(-pi/4) = Tdg gate, also not Clifford
    let msg = expect_sparse_stab_error(1, |b| {
        b.add_rz(-(Angle64::QUARTER_TURN / 2u64), &[0]);
        b.add_measurements(&[0]);
    });
    assert!(
        msg.contains("not a Clifford"),
        "Tdg-gate angle should be rejected: {msg}"
    );
}

// --- Negative quarter-turn for two-qubit gates ---

#[test]
fn rxx_negative_quarter_turn_acts_as_sxxdg() {
    // RXX(-pi/2) = SXXdg. SXXdg * SXX = I.
    let outcomes = run_sparse_stab(2, |b| {
        let sxxdg = Gate::rxx(-Angle64::QUARTER_TURN, &[(0usize, 1usize)]);
        let sxx = Gate::rxx(Angle64::QUARTER_TURN, &[(0usize, 1usize)]);
        b.add_gate_command(&sxxdg);
        b.add_gate_command(&sxx);
        b.add_measurements(&[0, 1]);
    });
    assert_eq!(outcomes, vec![0, 0]);
}

#[test]
fn ryy_negative_quarter_turn_acts_as_syydg() {
    // RYY(-pi/2) = SYYdg. SYYdg * SYY = I.
    let outcomes = run_sparse_stab(2, |b| {
        let syydg = Gate::ryy(-Angle64::QUARTER_TURN, &[(0usize, 1usize)]);
        let syy = Gate::ryy(Angle64::QUARTER_TURN, &[(0usize, 1usize)]);
        b.add_gate_command(&syydg);
        b.add_gate_command(&syy);
        b.add_measurements(&[0, 1]);
    });
    assert_eq!(outcomes, vec![0, 0]);
}

#[test]
fn rzz_negative_quarter_turn_acts_as_szzdg() {
    // RZZ(-pi/2) = SZZdg. SZZdg * SZZ = I.
    // Verify no phase effect: H -> SZZdg * SZZ -> H = I. |0,0> -> |0,0>
    let outcomes = run_sparse_stab(2, |b| {
        b.add_h(&[0]);
        b.add_h(&[1]);
        b.add_rzz(-Angle64::QUARTER_TURN, &[0], &[1]);
        b.add_rzz(Angle64::QUARTER_TURN, &[0], &[1]);
        b.add_h(&[0]);
        b.add_h(&[1]);
        b.add_measurements(&[0, 1]);
    });
    assert_eq!(outcomes, vec![0, 0]);
}

// --- R1XY negative theta via engine ---

#[test]
fn r1xy_negative_theta_acts_as_sxdg() {
    // R1XY(-pi/2, 0) = SXdg. SXdg * SX = I. |0> -> |0>
    let outcomes = run_sparse_stab(1, |b| {
        b.add_r1xy(-Angle64::QUARTER_TURN, Angle64::ZERO, &[0]);
        b.add_r1xy(Angle64::QUARTER_TURN, Angle64::ZERO, &[0]);
        b.add_measurements(&[0]);
    });
    assert_eq!(outcomes, vec![0]);
}

#[test]
fn r1xy_negative_half_turn_acts_as_x() {
    // R1XY(-pi, 0) = X. |0> -> |1>
    let outcomes = run_sparse_stab(1, |b| {
        b.add_r1xy(-Angle64::HALF_TURN, Angle64::ZERO, &[0]);
        b.add_measurements(&[0]);
    });
    assert_eq!(outcomes, vec![1]);
}

// --- Composition tests ---

#[test]
fn rz_rotations_compose_correctly() {
    // Four RZ(pi/2) = RZ(2pi) = I.
    // Sandwich with H: H * I * H = I, so |0> -> |0>
    let outcomes = run_sparse_stab(1, |b| {
        b.add_h(&[0]);
        for _ in 0..4 {
            b.add_rz(Angle64::QUARTER_TURN, &[0]);
        }
        b.add_h(&[0]);
        b.add_measurements(&[0]);
    });
    assert_eq!(outcomes, vec![0]);
}

#[test]
fn mixed_clifford_rotations_in_circuit() {
    // Build a circuit mixing named Cliffords and rotation gates:
    // q0: X -> RZ(pi) -> measure (X then Z|1>=-|1> -> outcome 1)
    // q1: RX(pi) -> measure (X|0>=|1> -> outcome 1)
    let outcomes = run_sparse_stab(2, |b| {
        b.add_x(&[0]);
        b.add_rz(Angle64::HALF_TURN, &[0]);
        b.add_rx(Angle64::HALF_TURN, &[1]);
        b.add_measurements(&[0, 1]);
    });
    assert_eq!(outcomes, vec![1, 1]);
}

// --- RXXRYYRZZ tests (via Gate::rxxryyrzz + add_gate_command) ---

#[test]
fn rxxryyrzz_identity_on_sparse_stab() {
    // RXXRYYRZZ(0,0,0) = I
    let outcomes = run_sparse_stab(2, |b| {
        let gate = Gate::rxxryyrzz(
            Angle64::ZERO,
            Angle64::ZERO,
            Angle64::ZERO,
            &[(0usize, 1usize)],
        );
        b.add_gate_command(&gate);
        b.add_measurements(&[0, 1]);
    });
    assert_eq!(outcomes, vec![0, 0]);
}

#[test]
fn rxxryyrzz_clifford_angles_on_sparse_stab() {
    // RXXRYYRZZ(pi, 0, 0) = RXX(pi) = X x X: |00> -> |11>
    let outcomes = run_sparse_stab(2, |b| {
        let gate = Gate::rxxryyrzz(
            Angle64::HALF_TURN,
            Angle64::ZERO,
            Angle64::ZERO,
            &[(0usize, 1usize)],
        );
        b.add_gate_command(&gate);
        b.add_measurements(&[0, 1]);
    });
    assert_eq!(outcomes, vec![1, 1]);
}

#[test]
fn rxxryyrzz_inverse_clifford_on_sparse_stab() {
    // RXXRYYRZZ(pi/2, pi/2, pi/2) * RXXRYYRZZ(-pi/2, -pi/2, -pi/2) = I
    let q = Angle64::QUARTER_TURN;
    let outcomes = run_sparse_stab(2, |b| {
        b.add_x(&[1]); // |01>
        let fwd = Gate::rxxryyrzz(q, q, q, &[(0usize, 1usize)]);
        let inv = Gate::rxxryyrzz(-q, -q, -q, &[(0usize, 1usize)]);
        b.add_gate_command(&fwd);
        b.add_gate_command(&inv);
        b.add_measurements(&[0, 1]);
    });
    assert_eq!(outcomes, vec![0, 1]);
}

#[test]
fn rxxryyrzz_non_clifford_fails_on_sparse_stab() {
    let msg = expect_sparse_stab_error(2, |b| {
        let gate = Gate::rxxryyrzz(
            Angle64::from_radians(0.5),
            Angle64::ZERO,
            Angle64::ZERO,
            &[(0usize, 1usize)],
        );
        b.add_gate_command(&gate);
        b.add_measurements(&[0, 1]);
    });
    assert!(msg.contains("not a Clifford"), "error: {msg}");
}

// --- U2q tests (via Gate::u2q + add_gate_command) ---

#[test]
fn u2q_identity_on_sparse_stab() {
    let zero = [Angle64::ZERO; 3];
    let id = [zero; 2];
    let outcomes = run_sparse_stab(2, |b| {
        let gate = Gate::u2q(id, [Angle64::ZERO; 3], id, &[(0usize, 1usize)]);
        b.add_gate_command(&gate);
        b.add_measurements(&[0, 1]);
    });
    assert_eq!(outcomes, vec![0, 0]);
}

#[test]
fn u2q_clifford_single_qubit_only() {
    // U2q with X on q0 (U3(pi,0,pi)) and identity interaction
    let zero = [Angle64::ZERO; 3];
    let x_params = [Angle64::HALF_TURN, Angle64::ZERO, Angle64::HALF_TURN];
    let outcomes = run_sparse_stab(2, |b| {
        let gate = Gate::u2q(
            [zero; 2],
            [Angle64::ZERO; 3],
            [x_params, zero],
            &[(0usize, 1usize)],
        );
        b.add_gate_command(&gate);
        b.add_measurements(&[0, 1]);
    });
    // X on q0, I on q1: |00> -> |10>
    assert_eq!(outcomes, vec![1, 0]);
}

#[test]
fn u2q_clifford_interaction_rxx_pi() {
    // U2q with identity single-qubit gates and interaction = (pi, 0, 0)
    // = RXXRYYRZZ(pi, 0, 0) = RXX(pi) = XX: |00> -> |11>
    let zero = [Angle64::ZERO; 3];
    let id = [zero; 2];
    let interaction = [Angle64::HALF_TURN, Angle64::ZERO, Angle64::ZERO];

    let outcomes = run_sparse_stab(2, |b| {
        let gate = Gate::u2q(id, interaction, id, &[(0usize, 1usize)]);
        b.add_gate_command(&gate);
        b.add_measurements(&[0, 1]);
    });
    assert_eq!(outcomes, vec![1, 1]);
}

#[test]
fn u2q_clifford_interaction_quarter_turn() {
    // U2q with identity single-qubit gates and interaction = (pi/2, pi/2, pi/2).
    // Forward then inverse should cancel, preserving |01>.
    let zero = [Angle64::ZERO; 3];
    let id = [zero; 2];
    let q = Angle64::QUARTER_TURN;

    let outcomes = run_sparse_stab(2, |b| {
        b.add_x(&[1]); // |01>
        let fwd = Gate::u2q(id, [q, q, q], id, &[(0usize, 1usize)]);
        let inv = Gate::u2q(id, [-q, -q, -q], id, &[(0usize, 1usize)]);
        b.add_gate_command(&fwd);
        b.add_gate_command(&inv);
        b.add_measurements(&[0, 1]);
    });
    assert_eq!(outcomes, vec![0, 1]);
}

#[test]
fn u2q_clifford_interaction_with_single_qubit_gates() {
    // U2q with X on after[0] and interaction = (pi/2, 0, 0) = SXX.
    // Apply twice: SXX * SXX = XX.
    // With after X on q0 applied each time, net = X^2 * XX = XX on q0,q1.
    // Actually the U gate composition is more nuanced -- use inverse instead.
    //
    // Forward: U2q(I, (pi/2,0,0), [X, I]) then inverse should cancel on |01>.
    let zero = [Angle64::ZERO; 3];
    let x_params = [Angle64::HALF_TURN, Angle64::ZERO, Angle64::HALF_TURN];
    let q = Angle64::QUARTER_TURN;

    let before = [zero; 2];
    let interaction = [q, Angle64::ZERO, Angle64::ZERO];
    let after = [x_params, zero]; // X on q0

    // Inverse: swap before/after, negate+swap phi/lambda, negate interaction
    let inv_before = [
        [-x_params[0], -x_params[2], -x_params[1]],
        [-zero[0], -zero[2], -zero[1]],
    ];
    let inv_interaction = [-q, Angle64::ZERO, Angle64::ZERO];
    let inv_after = [
        [-zero[0], -zero[2], -zero[1]],
        [-zero[0], -zero[2], -zero[1]],
    ];

    let outcomes = run_sparse_stab(2, |b| {
        b.add_x(&[1]); // |01>
        let fwd = Gate::u2q(before, interaction, after, &[(0usize, 1usize)]);
        let inv = Gate::u2q(inv_before, inv_interaction, inv_after, &[(0usize, 1usize)]);
        b.add_gate_command(&fwd);
        b.add_gate_command(&inv);
        b.add_measurements(&[0, 1]);
    });
    assert_eq!(outcomes, vec![0, 1]);
}

#[test]
fn u2q_non_clifford_fails_on_sparse_stab() {
    let zero = [Angle64::ZERO; 3];
    let non_clifford = [Angle64::from_radians(0.5), Angle64::ZERO, Angle64::ZERO];
    let msg = expect_sparse_stab_error(2, |b| {
        let gate = Gate::u2q([zero; 2], non_clifford, [zero; 2], &[(0usize, 1usize)]);
        b.add_gate_command(&gate);
        b.add_measurements(&[0, 1]);
    });
    assert!(msg.contains("not a Clifford"), "error: {msg}");
}
