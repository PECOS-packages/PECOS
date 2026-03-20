//! Integration tests for `StateVecEngine` rotation gate handling via `ByteMessage`.
//!
//! These tests verify that RXXRYYRZZ and U2q gates are correctly processed when
//! sent as `ByteMessage` commands to the `StateVecEngine`, exercising the full
//! `ByteMessage` -> `Engine::process()` -> `ArbitraryRotationGateable` trait path.

use pecos_core::{Angle64, Gate, Unitary};
use pecos_engines::Engine;
use pecos_engines::byte_message::ByteMessageBuilder;
use pecos_engines::quantum::DenseStateVecEngine;

/// Helper: build a circuit, process it, return measurement outcomes.
/// Uses a fixed seed for deterministic results.
fn run_state_vec(num_qubits: usize, build: impl FnOnce(&mut ByteMessageBuilder)) -> Vec<u32> {
    let mut engine = DenseStateVecEngine::with_seed(num_qubits, 42);
    let mut builder = ByteMessageBuilder::new();
    let _ = builder.for_quantum_operations();
    build(&mut builder);
    let result = engine.process(builder.build()).expect("process failed");
    result.outcomes().expect("outcomes failed")
}

// --- RXXRYYRZZ tests ---

#[test]
fn rxxryyrzz_identity_is_noop() {
    // RXXRYYRZZ(0,0,0) = I
    let outcomes = run_state_vec(2, |b| {
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
fn rxxryyrzz_inverse_cancels() {
    // RXXRYYRZZ(a,b,c) * RXXRYYRZZ(-a,-b,-c) = I
    let a = Angle64::from_radians(0.5);
    let b = Angle64::from_radians(0.3);
    let c = Angle64::from_radians(0.7);

    let outcomes = run_state_vec(2, |builder| {
        builder.add_x(&[1]); // |01>
        let fwd = Gate::rxxryyrzz(a, b, c, &[(0usize, 1usize)]);
        let inv = Gate::rxxryyrzz(-a, -b, -c, &[(0usize, 1usize)]);
        builder.add_gate_command(&fwd);
        builder.add_gate_command(&inv);
        builder.add_measurements(&[0, 1]);
    });
    assert_eq!(outcomes, vec![0, 1]);
}

#[test]
fn rxxryyrzz_pure_xx_matches_rxx() {
    // RXXRYYRZZ(pi, 0, 0) should have same effect as RXX(pi): |00> -> -|11>
    let outcomes = run_state_vec(2, |b| {
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

// --- U2q tests ---

#[test]
fn u2q_identity_is_noop() {
    let zero = [Angle64::ZERO; 3];
    let id = [zero; 2];

    let outcomes = run_state_vec(2, |b| {
        let gate = Gate::u2q(id, [Angle64::ZERO; 3], id, &[(0usize, 1usize)]);
        b.add_gate_command(&gate);
        b.add_measurements(&[0, 1]);
    });
    assert_eq!(outcomes, vec![0, 0]);
}

#[test]
fn u2q_identity_preserves_input_state() {
    let zero = [Angle64::ZERO; 3];
    let id = [zero; 2];

    // On |10>
    let outcomes = run_state_vec(2, |b| {
        b.add_x(&[0]);
        let gate = Gate::u2q(id, [Angle64::ZERO; 3], id, &[(0usize, 1usize)]);
        b.add_gate_command(&gate);
        b.add_measurements(&[0, 1]);
    });
    assert_eq!(outcomes, vec![1, 0]);
}

#[test]
fn u2q_inverse_cancels() {
    let before = [
        [
            Angle64::from_radians(0.5),
            Angle64::from_radians(0.3),
            Angle64::from_radians(0.7),
        ],
        [
            Angle64::from_radians(1.0),
            Angle64::from_radians(0.2),
            Angle64::from_radians(0.4),
        ],
    ];
    let interaction = [
        Angle64::from_radians(0.6),
        Angle64::from_radians(0.3),
        Angle64::from_radians(0.8),
    ];
    let after = [
        [
            Angle64::from_radians(0.9),
            Angle64::from_radians(0.1),
            Angle64::from_radians(0.5),
        ],
        [
            Angle64::from_radians(0.4),
            Angle64::from_radians(0.7),
            Angle64::from_radians(0.2),
        ],
    ];

    // Inverse: swap before/after and negate+swap phi/lambda, negate interaction
    let inv_before = [
        [-after[0][0], -after[0][2], -after[0][1]],
        [-after[1][0], -after[1][2], -after[1][1]],
    ];
    let inv_interaction = [-interaction[0], -interaction[1], -interaction[2]];
    let inv_after = [
        [-before[0][0], -before[0][2], -before[0][1]],
        [-before[1][0], -before[1][2], -before[1][1]],
    ];

    let outcomes = run_state_vec(2, |b| {
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
fn u2q_interaction_only_matches_rxxryyrzz() {
    // U2q with identity single-qubit gates should act like RXXRYYRZZ.
    // Apply U2q(I, angles, I) then RXXRYYRZZ(-angles) -- should cancel.
    let zero = [Angle64::ZERO; 3];
    let id = [zero; 2];
    let a = Angle64::from_radians(0.5);
    let b = Angle64::from_radians(0.3);
    let c = Angle64::from_radians(0.7);

    let outcomes = run_state_vec(2, |builder| {
        builder.add_x(&[1]); // |01>
        let u2q_gate = Gate::u2q(id, [a, b, c], id, &[(0usize, 1usize)]);
        let inv = Gate::rxxryyrzz(-a, -b, -c, &[(0usize, 1usize)]);
        builder.add_gate_command(&u2q_gate);
        builder.add_gate_command(&inv);
        builder.add_measurements(&[0, 1]);
    });
    assert_eq!(outcomes, vec![0, 1]);
}

#[test]
fn u2q_with_single_qubit_gates_affects_state() {
    // U2q with a non-trivial "after" single-qubit gate (X on qubit 0)
    // should flip qubit 0. U3(pi, 0, pi) = X gate.
    let zero = [Angle64::ZERO; 3];
    let x_gate_params = [Angle64::HALF_TURN, Angle64::ZERO, Angle64::HALF_TURN];
    let after = [x_gate_params, zero]; // X on q0, I on q1
    let before = [zero; 2]; // identity
    let interaction = [Angle64::ZERO; 3]; // no interaction

    let outcomes = run_state_vec(2, |b| {
        // Start in |00>, after gate should give X|0> x I|0> = |10>
        let gate = Gate::u2q(before, interaction, after, &[(0usize, 1usize)]);
        b.add_gate_command(&gate);
        b.add_measurements(&[0, 1]);
    });
    assert_eq!(outcomes, vec![1, 0]);
}

// --- Roundtrip: Unitary -> to_gates() -> Engine matches Gate::u2q() -> Engine ---

#[test]
fn u2q_unitary_rep_to_gates_roundtrip() {
    // Build a Unitary::U2q, convert via UnitaryRep::to_gates(), and verify
    // the resulting Gate produces the same state as Gate::u2q() directly.
    let before = [
        [
            Angle64::from_radians(0.5),
            Angle64::from_radians(0.3),
            Angle64::from_radians(0.7),
        ],
        [
            Angle64::from_radians(1.0),
            Angle64::from_radians(0.2),
            Angle64::from_radians(0.4),
        ],
    ];
    let interaction = [
        Angle64::from_radians(0.6),
        Angle64::from_radians(0.3),
        Angle64::from_radians(0.8),
    ];
    let after = [
        [
            Angle64::from_radians(0.9),
            Angle64::from_radians(0.1),
            Angle64::from_radians(0.5),
        ],
        [
            Angle64::from_radians(0.4),
            Angle64::from_radians(0.7),
            Angle64::from_radians(0.2),
        ],
    ];

    // Path 1: UnitaryRep::to_gates() -> Engine
    let unitary = Unitary::U2q {
        before,
        interaction,
        after,
    };
    let rep = unitary.on_qubits(0, 1);
    let gates = rep.decompose();
    assert_eq!(gates.len(), 1, "U2q should produce exactly one gate");
    let outcomes_via_rep = run_state_vec(2, |b| {
        b.add_x(&[1]); // |01>
        b.add_gate_command(&gates[0]);
        b.add_measurements(&[0, 1]);
    });

    // Path 2: Gate::u2q() directly -> Engine
    let outcomes_direct = run_state_vec(2, |b| {
        b.add_x(&[1]); // |01>
        let gate = Gate::u2q(before, interaction, after, &[(0usize, 1usize)]);
        b.add_gate_command(&gate);
        b.add_measurements(&[0, 1]);
    });

    assert_eq!(
        outcomes_via_rep, outcomes_direct,
        "UnitaryRep::to_gates() and Gate::u2q() should produce identical results"
    );
}

// --- Multi-pair tests ---

#[test]
fn rxxryyrzz_multi_pair_identity() {
    // RXXRYYRZZ(0,0,0) on two pairs should be identity on all 4 qubits.
    let outcomes = run_state_vec(4, |b| {
        let gate = Gate::rxxryyrzz(
            Angle64::ZERO,
            Angle64::ZERO,
            Angle64::ZERO,
            &[(0usize, 1usize), (2usize, 3usize)],
        );
        b.add_gate_command(&gate);
        b.add_measurements(&[0, 1, 2, 3]);
    });
    assert_eq!(outcomes, vec![0, 0, 0, 0]);
}

#[test]
fn rxxryyrzz_multi_pair_pure_xx() {
    // RXXRYYRZZ(pi,0,0) = RXX(pi) on both pairs: |0000> -> |1111>
    let outcomes = run_state_vec(4, |b| {
        let gate = Gate::rxxryyrzz(
            Angle64::HALF_TURN,
            Angle64::ZERO,
            Angle64::ZERO,
            &[(0usize, 1usize), (2usize, 3usize)],
        );
        b.add_gate_command(&gate);
        b.add_measurements(&[0, 1, 2, 3]);
    });
    assert_eq!(outcomes, vec![1, 1, 1, 1]);
}

#[test]
fn rxxryyrzz_multi_pair_inverse_cancels() {
    // Forward then inverse on two pairs should preserve |0110>
    let a = Angle64::from_radians(0.5);
    let b = Angle64::from_radians(0.3);
    let c = Angle64::from_radians(0.7);

    let outcomes = run_state_vec(4, |builder| {
        builder.add_x(&[1, 2]); // |0110>
        let fwd = Gate::rxxryyrzz(a, b, c, &[(0usize, 1usize), (2usize, 3usize)]);
        let inv = Gate::rxxryyrzz(-a, -b, -c, &[(0usize, 1usize), (2usize, 3usize)]);
        builder.add_gate_command(&fwd);
        builder.add_gate_command(&inv);
        builder.add_measurements(&[0, 1, 2, 3]);
    });
    assert_eq!(outcomes, vec![0, 1, 1, 0]);
}

#[test]
fn u2q_multi_pair_identity() {
    let zero = [Angle64::ZERO; 3];
    let id = [zero; 2];

    let outcomes = run_state_vec(4, |b| {
        let gate = Gate::u2q(
            id,
            [Angle64::ZERO; 3],
            id,
            &[(0usize, 1usize), (2usize, 3usize)],
        );
        b.add_gate_command(&gate);
        b.add_measurements(&[0, 1, 2, 3]);
    });
    assert_eq!(outcomes, vec![0, 0, 0, 0]);
}

#[test]
fn u2q_multi_pair_single_qubit_x_on_first() {
    // U2q with X on after[0] (first qubit of each pair) applied to pairs (0,1) and (2,3).
    // Should flip q0 and q2: |0000> -> |1010>
    let zero = [Angle64::ZERO; 3];
    let x_params = [Angle64::HALF_TURN, Angle64::ZERO, Angle64::HALF_TURN];
    let after = [x_params, zero]; // X on first qubit, I on second
    let before = [zero; 2];
    let interaction = [Angle64::ZERO; 3];

    let outcomes = run_state_vec(4, |b| {
        let gate = Gate::u2q(
            before,
            interaction,
            after,
            &[(0usize, 1usize), (2usize, 3usize)],
        );
        b.add_gate_command(&gate);
        b.add_measurements(&[0, 1, 2, 3]);
    });
    assert_eq!(outcomes, vec![1, 0, 1, 0]);
}

#[test]
fn u2q_multi_pair_inverse_cancels() {
    let before = [
        [
            Angle64::from_radians(0.5),
            Angle64::from_radians(0.3),
            Angle64::from_radians(0.7),
        ],
        [
            Angle64::from_radians(1.0),
            Angle64::from_radians(0.2),
            Angle64::from_radians(0.4),
        ],
    ];
    let interaction = [
        Angle64::from_radians(0.6),
        Angle64::from_radians(0.3),
        Angle64::from_radians(0.8),
    ];
    let after = [
        [
            Angle64::from_radians(0.9),
            Angle64::from_radians(0.1),
            Angle64::from_radians(0.5),
        ],
        [
            Angle64::from_radians(0.4),
            Angle64::from_radians(0.7),
            Angle64::from_radians(0.2),
        ],
    ];

    let inv_before = [
        [-after[0][0], -after[0][2], -after[0][1]],
        [-after[1][0], -after[1][2], -after[1][1]],
    ];
    let inv_interaction = [-interaction[0], -interaction[1], -interaction[2]];
    let inv_after = [
        [-before[0][0], -before[0][2], -before[0][1]],
        [-before[1][0], -before[1][2], -before[1][1]],
    ];

    let outcomes = run_state_vec(4, |b| {
        b.add_x(&[1, 2]); // |0110>
        let fwd = Gate::u2q(
            before,
            interaction,
            after,
            &[(0usize, 1usize), (2usize, 3usize)],
        );
        let inv = Gate::u2q(
            inv_before,
            inv_interaction,
            inv_after,
            &[(0usize, 1usize), (2usize, 3usize)],
        );
        b.add_gate_command(&fwd);
        b.add_gate_command(&inv);
        b.add_measurements(&[0, 1, 2, 3]);
    });
    assert_eq!(outcomes, vec![0, 1, 1, 0]);
}

// --- Roundtrip: Unitary -> to_gates() -> Engine matches Gate::u2q() -> Engine ---

#[test]
fn rxxryyrzz_unitary_rep_to_gates_roundtrip() {
    let alpha = Angle64::from_radians(0.5);
    let beta = Angle64::from_radians(0.3);
    let gamma = Angle64::from_radians(0.7);

    let unitary = Unitary::RXXRYYRZZ { alpha, beta, gamma };
    let rep = unitary.on_qubits(0, 1);
    let gates = rep.decompose();
    assert_eq!(gates.len(), 1);

    let outcomes_via_rep = run_state_vec(2, |b| {
        b.add_x(&[1]); // |01>
        b.add_gate_command(&gates[0]);
        b.add_measurements(&[0, 1]);
    });

    let outcomes_direct = run_state_vec(2, |b| {
        b.add_x(&[1]);
        let gate = Gate::rxxryyrzz(alpha, beta, gamma, &[(0usize, 1usize)]);
        b.add_gate_command(&gate);
        b.add_measurements(&[0, 1]);
    });

    assert_eq!(outcomes_via_rep, outcomes_direct);
}
