// Copyright 2025 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License.You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! Test utilities for verifying state vector simulator implementations.
//!
//! This module provides generic test functions that can be used to verify any
//! state vector simulator that implements [`CliffordGateable`], [`ArbitraryRotationGateable`],
//! and [`QuantumSimulator`].
//!
//! # Example
//!
//! ```
//! use pecos_simulators::state_vector_test_utils::*;
//! use pecos_simulators::StateVecAoS;
//!
//! let mut sim = StateVecAoS::with_seed(4, 42);
//! verify_h_gate(&mut sim);
//! verify_bell_state_preparation(&mut sim);
//! ```

#![allow(clippy::missing_panics_doc)]

use crate::{ArbitraryRotationGateable, CliffordGateable, QuantumSimulator};
use num_complex::Complex64;
use pecos_core::{Angle64, QubitId};
use std::f64::consts::{FRAC_1_SQRT_2, FRAC_PI_2, FRAC_PI_4, PI};

// --- State Vector Simulator Marker Trait ---

/// Marker trait for state vector simulators that support quantum simulation.
///
/// Implementing this trait indicates that a simulator:
/// - Implements all Clifford gates via [`CliffordGateable`]
/// - Supports basic simulator operations via [`QuantumSimulator`]
/// - Can retrieve amplitudes for verification
/// - Can be constructed with a seed for reproducible tests
///
/// Simulators implementing this trait can use the [`state_vector_test_suite!`] macro
/// to automatically generate a comprehensive test suite. For simulators that also
/// implement [`ArbitraryRotationGateable`], use [`full_state_vector_test_suite!`]
/// to include rotation gate tests.
pub trait StateVectorSimulator: CliffordGateable + QuantumSimulator + Sized {
    /// Create a new simulator with the given number of qubits and RNG seed.
    fn with_seed(num_qubits: usize, seed: u64) -> Self;

    /// Get the amplitude for a specific basis state.
    ///
    /// For sparse implementations, returns `Complex64::new(0.0, 0.0)` for
    /// basis states not present in the state vector.
    fn get_amplitude(&mut self, basis_state: usize) -> Complex64;

    /// Get the number of qubits in the simulator.
    fn num_qubits(&self) -> usize;
}

/// Generates a Clifford-only test suite for a state vector simulator.
///
/// This macro creates test functions that verify correct implementation of
/// Clifford gates, measurement behavior, and basic properties. Use this for
/// simulators that only implement [`CliffordGateable`] (e.g., sparse simulators).
///
/// # Arguments
///
/// * `$sim_type` - The type implementing [`StateVectorSimulator`]
/// * `$num_qubits` - Number of qubits to use for testing (default: 4)
///
/// # Example
///
/// ```no_run
/// use pecos_simulators::state_vector_test_suite;
/// use pecos_simulators::SparseStateVecAoS;
///
/// state_vector_test_suite!(SparseStateVecAoS);
/// ```
#[macro_export]
macro_rules! state_vector_test_suite {
    ($sim_type:ty) => {
        $crate::state_vector_test_suite!($sim_type, 4);
    };
    ($sim_type:ty, $num_qubits:expr) => {
        paste::paste! {
            #[test]
            fn [<test_ $sim_type:snake _basic_suite>]() {
                use $crate::state_vector_test_utils::run_basic_state_vector_test_suite;
                let mut sim = <$sim_type>::with_seed($num_qubits, 42);
                run_basic_state_vector_test_suite(&mut sim);
            }

            #[test]
            fn [<test_ $sim_type:snake _clifford_suite>]() {
                use $crate::state_vector_test_utils::run_clifford_test_suite;
                let mut sim = <$sim_type>::with_seed($num_qubits, 42);
                run_clifford_test_suite(&mut sim);
            }

            #[test]
            fn [<test_ $sim_type:snake _measurement_suite>]() {
                use $crate::state_vector_test_utils::run_measurement_test_suite;
                let mut sim = <$sim_type>::with_seed($num_qubits, 42);
                run_measurement_test_suite(&mut sim);
            }
        }
    };
}

/// Generates a full test suite for a state vector simulator with rotation support.
///
/// This macro creates test functions that verify correct implementation of
/// all gates including rotation gates. Use this for simulators that implement
/// both [`CliffordGateable`] and [`ArbitraryRotationGateable`].
///
/// # Arguments
///
/// * `$sim_type` - The type implementing [`StateVectorSimulator`] + [`ArbitraryRotationGateable`]
/// * `$num_qubits` - Number of qubits to use for testing (default: 4)
///
/// # Example
///
/// ```no_run
/// use pecos_simulators::full_state_vector_test_suite;
/// use pecos_simulators::StateVecAoS;
///
/// full_state_vector_test_suite!(StateVecAoS);
/// ```
#[macro_export]
macro_rules! full_state_vector_test_suite {
    ($sim_type:ty) => {
        $crate::full_state_vector_test_suite!($sim_type, 4);
    };
    ($sim_type:ty, $num_qubits:expr) => {
        // Include all Clifford tests
        $crate::state_vector_test_suite!($sim_type, $num_qubits);

        paste::paste! {
            #[test]
            fn [<test_ $sim_type:snake _rotation_suite>]() {
                use $crate::state_vector_test_utils::run_rotation_test_suite;

                let mut sim = <$sim_type>::with_seed($num_qubits, 42);
                run_rotation_test_suite(&mut sim);
            }

            #[test]
            fn [<test_ $sim_type:snake _full_suite>]() {
                use $crate::state_vector_test_utils::run_full_state_vector_test_suite;

                let mut sim = <$sim_type>::with_seed($num_qubits, 42);
                run_full_state_vector_test_suite(&mut sim);
            }
        }
    };
}

// --- Helper Functions ---

const TOLERANCE: f64 = 1e-9;

fn qid(n: usize) -> [QubitId; 1] {
    [QubitId(n)]
}

fn qid2(a: usize, b: usize) -> [(QubitId, QubitId); 1] {
    [(QubitId(a), QubitId(b))]
}

fn assert_amplitude_eq(actual: Complex64, expected: Complex64, msg: &str) {
    assert!(
        (actual - expected).norm() < TOLERANCE,
        "{msg}: expected {expected:?}, got {actual:?}, diff = {}",
        (actual - expected).norm()
    );
}

fn assert_amplitude_near_zero(actual: Complex64, msg: &str) {
    assert!(
        actual.norm() < TOLERANCE,
        "{msg}: expected ~0, got {actual:?}"
    );
}

fn assert_probability_eq(actual: f64, expected: f64, msg: &str) {
    assert!(
        (actual - expected).abs() < TOLERANCE,
        "{msg}: expected {expected}, got {actual}"
    );
}

// --- Basic Tests ---

/// Verify the initial state is |0...0⟩.
pub fn verify_initial_state<S: StateVectorSimulator>(sim: &mut S) {
    sim.reset();
    let n = sim.num_qubits();
    let dim = 1 << n;

    assert_amplitude_eq(
        sim.get_amplitude(0),
        Complex64::new(1.0, 0.0),
        "Initial state amplitude at |0...0⟩",
    );

    for i in 1..dim {
        assert_amplitude_near_zero(
            sim.get_amplitude(i),
            &format!("Initial state amplitude at |{i}⟩"),
        );
    }
}

/// Verify reset returns to |0...0⟩.
pub fn verify_reset<S: StateVectorSimulator>(sim: &mut S) {
    sim.reset();
    sim.h(&qid(0));
    if sim.num_qubits() >= 2 {
        sim.cx(&qid2(0, 1));
    }
    sim.reset();

    assert_amplitude_eq(
        sim.get_amplitude(0),
        Complex64::new(1.0, 0.0),
        "Reset should return to |0...0⟩",
    );
}

/// Verify normalization is preserved after gates.
pub fn verify_normalization<S: StateVectorSimulator>(sim: &mut S) {
    sim.reset();
    sim.h(&qid(0));
    if sim.num_qubits() >= 2 {
        sim.cx(&qid2(0, 1));
    }

    let dim = 1 << sim.num_qubits();
    let mut norm_sq = 0.0;
    for i in 0..dim {
        norm_sq += sim.get_amplitude(i).norm_sqr();
    }

    assert_probability_eq(norm_sq, 1.0, "State should be normalized");
}

/// Verify probability calculation matches amplitude squared.
pub fn verify_probability<S: StateVectorSimulator>(sim: &mut S) {
    sim.reset();

    // |0⟩ state: probability of 0 should be 1
    let amp = sim.get_amplitude(0);
    assert_probability_eq(amp.norm_sqr(), 1.0, "P(|0⟩) in |0⟩ state");

    // |+⟩ state: probability of 0 and 1 should each be 0.5
    sim.h(&qid(0));
    let amp0 = sim.get_amplitude(0);
    let amp1 = sim.get_amplitude(1);
    assert_probability_eq(amp0.norm_sqr(), 0.5, "P(|0⟩) in |+⟩ state");
    assert_probability_eq(amp1.norm_sqr(), 0.5, "P(|1⟩) in |+⟩ state");

    // Bell state: P(|00⟩) = P(|11⟩) = 0.5, P(|01⟩) = P(|10⟩) = 0
    if sim.num_qubits() >= 2 {
        sim.reset();
        sim.h(&qid(0));
        sim.cx(&qid2(0, 1));

        assert_probability_eq(
            sim.get_amplitude(0b00).norm_sqr(),
            0.5,
            "P(|00⟩) in Bell state",
        );
        assert_probability_eq(
            sim.get_amplitude(0b01).norm_sqr(),
            0.0,
            "P(|01⟩) in Bell state",
        );
        assert_probability_eq(
            sim.get_amplitude(0b10).norm_sqr(),
            0.0,
            "P(|10⟩) in Bell state",
        );
        assert_probability_eq(
            sim.get_amplitude(0b11).norm_sqr(),
            0.5,
            "P(|11⟩) in Bell state",
        );
    }
}

/// Verify preparation of all computational basis states.
pub fn verify_prepare_all_basis_states<S: StateVectorSimulator>(sim: &mut S) {
    let n = sim.num_qubits().min(4); // Limit to 4 qubits to keep test fast
    let dim = 1 << n;

    for target_state in 0..dim {
        sim.reset();

        // Prepare |target_state⟩ by applying X to qubits that should be |1⟩
        for q in 0..n {
            if (target_state >> q) & 1 == 1 {
                sim.x(&qid(q));
            }
        }

        // Verify we're in the correct state
        for i in 0..dim {
            let expected = if i == target_state { 1.0 } else { 0.0 };
            assert_probability_eq(
                sim.get_amplitude(i).norm_sqr(),
                expected,
                &format!("Preparing |{target_state}⟩: P(|{i}⟩)"),
            );
        }
    }
}

/// Verify unitarity: operations preserve total probability.
pub fn verify_unitarity<S: StateVectorSimulator>(sim: &mut S) {
    let dim = 1 << sim.num_qubits();

    // Test various gate sequences
    #[allow(clippy::type_complexity)]
    let test_sequences: Vec<Box<dyn Fn(&mut S)>> = vec![
        Box::new(|s: &mut S| {
            s.h(&qid(0));
        }),
        Box::new(|s: &mut S| {
            s.x(&qid(0));
            s.y(&qid(0));
            s.z(&qid(0));
        }),
        Box::new(|s: &mut S| {
            s.h(&qid(0));
            s.sz(&qid(0));
            s.h(&qid(0));
        }),
    ];

    for (i, sequence) in test_sequences.iter().enumerate() {
        sim.reset();
        sequence(sim);

        let mut norm_sq = 0.0;
        for j in 0..dim {
            norm_sq += sim.get_amplitude(j).norm_sqr();
        }

        assert!(
            (norm_sq - 1.0).abs() < TOLERANCE,
            "Unitarity test {i}: norm^2 = {norm_sq}, expected 1.0"
        );
    }
}

/// Verify unitarity for two-qubit operations.
pub fn verify_unitarity_two_qubit<S: StateVectorSimulator>(sim: &mut S) {
    if sim.num_qubits() < 2 {
        return;
    }

    let dim = 1 << sim.num_qubits();

    // Test two-qubit gate sequences
    sim.reset();
    sim.h(&qid(0));
    sim.cx(&qid2(0, 1));
    sim.cz(&qid2(0, 1));
    sim.swap(&qid2(0, 1));

    let mut norm_sq = 0.0;
    for i in 0..dim {
        norm_sq += sim.get_amplitude(i).norm_sqr();
    }

    assert!(
        (norm_sq - 1.0).abs() < TOLERANCE,
        "Two-qubit unitarity: norm^2 = {norm_sq}, expected 1.0"
    );
}

// --- Single-Qubit Clifford Gate Tests ---

/// Verify X gate: X|0⟩ = |1⟩, X|1⟩ = |0⟩.
pub fn verify_x_gate<S: StateVectorSimulator>(sim: &mut S) {
    sim.reset();
    sim.x(&qid(0));

    assert_amplitude_near_zero(sim.get_amplitude(0), "X|0⟩: amplitude at |0⟩");
    assert_amplitude_eq(
        sim.get_amplitude(1),
        Complex64::new(1.0, 0.0),
        "X|0⟩: amplitude at |1⟩",
    );

    // X^2 = I
    sim.x(&qid(0));
    assert_amplitude_eq(
        sim.get_amplitude(0),
        Complex64::new(1.0, 0.0),
        "X^2|0⟩: amplitude at |0⟩",
    );
}

/// Verify Y gate: Y|0⟩ = i|1⟩, Y|1⟩ = -i|0⟩.
pub fn verify_y_gate<S: StateVectorSimulator>(sim: &mut S) {
    sim.reset();
    sim.y(&qid(0));

    assert_amplitude_near_zero(sim.get_amplitude(0), "Y|0⟩: amplitude at |0⟩");
    assert_amplitude_eq(
        sim.get_amplitude(1),
        Complex64::i(),
        "Y|0⟩: amplitude at |1⟩",
    );

    // Y^2 = I
    sim.y(&qid(0));
    // After Y^2, we should have |0⟩
    assert_amplitude_eq(
        sim.get_amplitude(0),
        Complex64::new(1.0, 0.0),
        "Y^2|0⟩: amplitude at |0⟩",
    );
}

/// Verify Z gate: Z|0⟩ = |0⟩, Z|1⟩ = -|1⟩.
pub fn verify_z_gate<S: StateVectorSimulator>(sim: &mut S) {
    sim.reset();
    sim.z(&qid(0));
    assert_amplitude_eq(
        sim.get_amplitude(0),
        Complex64::new(1.0, 0.0),
        "Z|0⟩ should equal |0⟩",
    );

    sim.reset();
    sim.x(&qid(0));
    sim.z(&qid(0));
    assert_amplitude_eq(
        sim.get_amplitude(1),
        Complex64::new(-1.0, 0.0),
        "Z|1⟩ should equal -|1⟩",
    );
}

/// Verify H gate: H|0⟩ = |+⟩ = (|0⟩ + |1⟩)/√2.
pub fn verify_h_gate<S: StateVectorSimulator>(sim: &mut S) {
    sim.reset();
    sim.h(&qid(0));

    let expected = Complex64::new(FRAC_1_SQRT_2, 0.0);
    assert_amplitude_eq(sim.get_amplitude(0), expected, "H|0⟩: amplitude at |0⟩");
    assert_amplitude_eq(sim.get_amplitude(1), expected, "H|0⟩: amplitude at |1⟩");

    // H^2 = I
    sim.h(&qid(0));
    assert_amplitude_eq(
        sim.get_amplitude(0),
        Complex64::new(1.0, 0.0),
        "H^2|0⟩: amplitude at |0⟩",
    );
    assert_amplitude_near_zero(sim.get_amplitude(1), "H^2|0⟩: amplitude at |1⟩");
}

/// Verify S gate (√Z): S|0⟩ = |0⟩, S|1⟩ = i|1⟩.
pub fn verify_sz_gate<S: StateVectorSimulator>(sim: &mut S) {
    sim.reset();
    sim.sz(&qid(0));
    assert_amplitude_eq(
        sim.get_amplitude(0),
        Complex64::new(1.0, 0.0),
        "S|0⟩ should equal |0⟩",
    );

    sim.reset();
    sim.x(&qid(0));
    sim.sz(&qid(0));
    assert_amplitude_eq(
        sim.get_amplitude(1),
        Complex64::i(),
        "S|1⟩ should equal i|1⟩",
    );

    // S^2 = Z
    sim.reset();
    sim.x(&qid(0));
    sim.sz(&qid(0));
    sim.sz(&qid(0));
    assert_amplitude_eq(
        sim.get_amplitude(1),
        Complex64::new(-1.0, 0.0),
        "S^2|1⟩ should equal -|1⟩",
    );
}

/// Verify SX gate (√X).
pub fn verify_sx_gate<S: StateVectorSimulator>(sim: &mut S) {
    sim.reset();
    sim.sx(&qid(0));
    sim.sx(&qid(0));

    // SX^2 = X
    assert_amplitude_near_zero(sim.get_amplitude(0), "SX^2|0⟩: amplitude at |0⟩");
    assert_amplitude_eq(
        sim.get_amplitude(1),
        Complex64::new(1.0, 0.0),
        "SX^2|0⟩: amplitude at |1⟩",
    );
}

/// Verify SY gate.
///
/// Note: Different simulators may have different phase conventions for SY.
/// We verify the magnitude is correct (|0⟩ → fully to |1⟩ after SY^2).
pub fn verify_sy_gate<S: StateVectorSimulator>(sim: &mut S) {
    sim.reset();
    sim.sy(&qid(0));
    sim.sy(&qid(0));

    // SY^2|0⟩ should flip to |1⟩ (up to phase)
    assert_amplitude_near_zero(sim.get_amplitude(0), "SY^2|0⟩: amplitude at |0⟩");
    assert!(
        (sim.get_amplitude(1).norm() - 1.0).abs() < TOLERANCE,
        "SY^2|0⟩: |1⟩ should have unit magnitude"
    );
}

// --- Two-Qubit Clifford Gate Tests ---

/// Verify CX (CNOT) gate.
pub fn verify_cx_gate<S: StateVectorSimulator>(sim: &mut S) {
    if sim.num_qubits() < 2 {
        return;
    }

    // CX|00⟩ = |00⟩
    sim.reset();
    sim.cx(&qid2(0, 1));
    assert_amplitude_eq(
        sim.get_amplitude(0),
        Complex64::new(1.0, 0.0),
        "CX|00⟩ = |00⟩",
    );

    // CX|10⟩ = |11⟩ (control=1, target flips)
    sim.reset();
    sim.x(&qid(0));
    sim.cx(&qid2(0, 1));
    assert_amplitude_eq(
        sim.get_amplitude(0b11),
        Complex64::new(1.0, 0.0),
        "CX|10⟩ = |11⟩",
    );

    // Bell state: CX(H⊗I)|00⟩ = (|00⟩ + |11⟩)/√2
    sim.reset();
    sim.h(&qid(0));
    sim.cx(&qid2(0, 1));
    let expected = Complex64::new(FRAC_1_SQRT_2, 0.0);
    assert_amplitude_eq(sim.get_amplitude(0b00), expected, "Bell state |00⟩");
    assert_amplitude_near_zero(sim.get_amplitude(0b01), "Bell state |01⟩");
    assert_amplitude_near_zero(sim.get_amplitude(0b10), "Bell state |10⟩");
    assert_amplitude_eq(sim.get_amplitude(0b11), expected, "Bell state |11⟩");
}

/// Verify CY gate.
///
/// Note: Different simulators may have different phase conventions for CY.
/// We verify that CY flips target when control is |1⟩ (checking magnitude).
pub fn verify_cy_gate<S: StateVectorSimulator>(sim: &mut S) {
    if sim.num_qubits() < 2 {
        return;
    }

    // CY|10⟩ should give |11⟩ (up to phase)
    sim.reset();
    sim.x(&qid(0));
    sim.cy(&qid2(0, 1));
    assert!(
        (sim.get_amplitude(0b11).norm() - 1.0).abs() < TOLERANCE,
        "CY|10⟩: |11⟩ should have unit magnitude",
    );
}

/// Verify CZ gate.
pub fn verify_cz_gate<S: StateVectorSimulator>(sim: &mut S) {
    if sim.num_qubits() < 2 {
        return;
    }

    // CZ|11⟩ = -|11⟩
    sim.reset();
    sim.x(&qid(0));
    sim.x(&qid(1));
    sim.cz(&qid2(0, 1));
    assert_amplitude_eq(
        sim.get_amplitude(0b11),
        Complex64::new(-1.0, 0.0),
        "CZ|11⟩ = -|11⟩",
    );

    // CZ on |++⟩ gives (|00⟩ + |01⟩ + |10⟩ - |11⟩)/2
    sim.reset();
    sim.h(&qid(0));
    sim.h(&qid(1));
    sim.cz(&qid2(0, 1));
    let half = Complex64::new(0.5, 0.0);
    assert_amplitude_eq(sim.get_amplitude(0b00), half, "CZ|++⟩: |00⟩");
    assert_amplitude_eq(sim.get_amplitude(0b01), half, "CZ|++⟩: |01⟩");
    assert_amplitude_eq(sim.get_amplitude(0b10), half, "CZ|++⟩: |10⟩");
    assert_amplitude_eq(sim.get_amplitude(0b11), -half, "CZ|++⟩: |11⟩");
}

/// Verify SWAP gate.
pub fn verify_swap_gate<S: StateVectorSimulator>(sim: &mut S) {
    if sim.num_qubits() < 2 {
        return;
    }

    // SWAP|10⟩ = |01⟩
    sim.reset();
    sim.x(&qid(0));
    sim.swap(&qid2(0, 1));
    assert_amplitude_eq(
        sim.get_amplitude(0b10),
        Complex64::new(1.0, 0.0),
        "SWAP|10⟩ = |01⟩ (bit ordering: |01⟩ is index 2)",
    );
}

/// Verify iSWAP gate.
pub fn verify_iswap_gate<S: StateVectorSimulator>(sim: &mut S) {
    if sim.num_qubits() < 2 {
        return;
    }

    // iSWAP|10⟩ = i|01⟩
    sim.reset();
    sim.x(&qid(0));
    sim.iswap(&qid2(0, 1));
    assert_amplitude_eq(sim.get_amplitude(0b10), Complex64::i(), "iSWAP|10⟩ = i|01⟩");
}

// --- Rotation Gate Tests ---

/// Verify RX gate.
pub fn verify_rx_gate<S: StateVectorSimulator + ArbitraryRotationGateable>(sim: &mut S) {
    // RX(π)|0⟩ = -i|1⟩
    sim.reset();
    sim.rx(Angle64::from_radians(PI), &qid(0));
    assert_amplitude_near_zero(sim.get_amplitude(0), "RX(π)|0⟩: |0⟩");
    assert_amplitude_eq(
        sim.get_amplitude(1),
        Complex64::new(0.0, -1.0),
        "RX(π)|0⟩: |1⟩",
    );

    // RX(π/2)|0⟩ = (|0⟩ - i|1⟩)/√2
    sim.reset();
    sim.rx(Angle64::from_radians(FRAC_PI_2), &qid(0));
    let expected_0 = Complex64::new(FRAC_1_SQRT_2, 0.0);
    let expected_1 = Complex64::new(0.0, -FRAC_1_SQRT_2);
    assert_amplitude_eq(sim.get_amplitude(0), expected_0, "RX(π/2)|0⟩: |0⟩");
    assert_amplitude_eq(sim.get_amplitude(1), expected_1, "RX(π/2)|0⟩: |1⟩");
}

/// Verify RY gate.
pub fn verify_ry_gate<S: StateVectorSimulator + ArbitraryRotationGateable>(sim: &mut S) {
    // RY(π)|0⟩ = |1⟩
    sim.reset();
    sim.ry(Angle64::from_radians(PI), &qid(0));
    assert_amplitude_near_zero(sim.get_amplitude(0), "RY(π)|0⟩: |0⟩");
    assert_amplitude_eq(
        sim.get_amplitude(1),
        Complex64::new(1.0, 0.0),
        "RY(π)|0⟩: |1⟩",
    );

    // RY(π/2)|0⟩ = (|0⟩ + |1⟩)/√2
    sim.reset();
    sim.ry(Angle64::from_radians(FRAC_PI_2), &qid(0));
    let expected = Complex64::new(FRAC_1_SQRT_2, 0.0);
    assert_amplitude_eq(sim.get_amplitude(0), expected, "RY(π/2)|0⟩: |0⟩");
    assert_amplitude_eq(sim.get_amplitude(1), expected, "RY(π/2)|0⟩: |1⟩");
}

/// Verify RZ gate.
pub fn verify_rz_gate<S: StateVectorSimulator + ArbitraryRotationGateable>(sim: &mut S) {
    // RZ on |0⟩ only adds global phase
    sim.reset();
    sim.rz(Angle64::from_radians(PI), &qid(0));
    // |amplitude|^2 should still be 1 at |0⟩
    let amp = sim.get_amplitude(0);
    assert_probability_eq(amp.norm_sqr(), 1.0, "RZ(π)|0⟩: probability at |0⟩");

    // RZ on |+⟩ gives rotation
    sim.reset();
    sim.h(&qid(0));
    sim.rz(Angle64::from_radians(PI), &qid(0));
    // Should give |-⟩ up to global phase
    let amp0 = sim.get_amplitude(0);
    let amp1 = sim.get_amplitude(1);
    // Check relative phase: amp0/amp1 should be -1 (for |−⟩)
    let ratio = amp0 / amp1;
    assert_amplitude_eq(ratio, Complex64::new(-1.0, 0.0), "RZ(π)|+⟩ gives |-⟩");
}

/// Verify RXX gate.
pub fn verify_rxx_gate<S: StateVectorSimulator + ArbitraryRotationGateable>(sim: &mut S) {
    if sim.num_qubits() < 2 {
        return;
    }

    // RXX(π/2) on |00⟩
    sim.reset();
    sim.rxx(Angle64::from_radians(FRAC_PI_2), &qid2(0, 1));

    // Should create entanglement
    let amp00 = sim.get_amplitude(0b00);
    let amp11 = sim.get_amplitude(0b11);

    // Both should have equal magnitude
    assert!(
        (amp00.norm() - amp11.norm()).abs() < TOLERANCE,
        "RXX(π/2)|00⟩: |00⟩ and |11⟩ should have equal magnitude"
    );
}

/// Verify RYY gate.
pub fn verify_ryy_gate<S: StateVectorSimulator + ArbitraryRotationGateable>(sim: &mut S) {
    if sim.num_qubits() < 2 {
        return;
    }

    sim.reset();
    sim.ryy(Angle64::from_radians(FRAC_PI_2), &qid2(0, 1));

    // Should create entanglement
    let amp00 = sim.get_amplitude(0b00);
    let amp11 = sim.get_amplitude(0b11);

    assert!(
        (amp00.norm() - amp11.norm()).abs() < TOLERANCE,
        "RYY(π/2)|00⟩: |00⟩ and |11⟩ should have equal magnitude"
    );
}

/// Verify RZZ gate.
pub fn verify_rzz_gate<S: StateVectorSimulator + ArbitraryRotationGateable>(sim: &mut S) {
    if sim.num_qubits() < 2 {
        return;
    }

    // RZZ on |00⟩ only adds global phase
    sim.reset();
    sim.rzz(Angle64::from_radians(PI), &qid2(0, 1));
    let amp = sim.get_amplitude(0);
    assert_probability_eq(amp.norm_sqr(), 1.0, "RZZ(π)|00⟩: probability at |00⟩");

    // RZZ on Bell state should add relative phases
    sim.reset();
    sim.h(&qid(0));
    sim.cx(&qid2(0, 1));
    sim.rzz(Angle64::from_radians(PI), &qid2(0, 1));

    // Bell state (|00⟩ + |11⟩)/√2 after RZZ(π) should have relative phase
    let amp00 = sim.get_amplitude(0b00);
    let amp11 = sim.get_amplitude(0b11);
    assert!(
        (amp00.norm() - FRAC_1_SQRT_2).abs() < TOLERANCE,
        "RZZ on Bell: |00⟩ magnitude"
    );
    assert!(
        (amp11.norm() - FRAC_1_SQRT_2).abs() < TOLERANCE,
        "RZZ on Bell: |11⟩ magnitude"
    );
}

// --- Measurement Tests ---

/// Verify measurement of deterministic states.
///
/// Note: We don't check `is_deterministic` flag as not all simulators implement it.
pub fn verify_deterministic_measurement<S: StateVectorSimulator>(sim: &mut S) {
    // Measure |0⟩
    sim.reset();
    let result = sim.mz(&qid(0));
    assert!(!result[0].outcome, "Measuring |0⟩ should give 0");

    // Measure |1⟩
    sim.reset();
    sim.x(&qid(0));
    let result = sim.mz(&qid(0));
    assert!(result[0].outcome, "Measuring |1⟩ should give 1");
}

/// Verify measurement collapses the state.
pub fn verify_measurement_collapse<S: StateVectorSimulator>(sim: &mut S) {
    sim.reset();
    sim.h(&qid(0));

    let result = sim.mz(&qid(0));

    // After measurement, state should be collapsed
    if result[0].outcome {
        assert_amplitude_near_zero(sim.get_amplitude(0), "After measuring 1: |0⟩");
        assert!(
            (sim.get_amplitude(1).norm() - 1.0).abs() < TOLERANCE,
            "After measuring 1: |1⟩ should have amplitude 1"
        );
    } else {
        assert!(
            (sim.get_amplitude(0).norm() - 1.0).abs() < TOLERANCE,
            "After measuring 0: |0⟩ should have amplitude 1"
        );
        assert_amplitude_near_zero(sim.get_amplitude(1), "After measuring 0: |1⟩");
    }
}

// --- State Preparation Tests ---

/// Verify pz (prepare |0⟩) operation.
pub fn verify_pz<S: StateVectorSimulator>(sim: &mut S) {
    sim.reset();
    sim.h(&qid(0)); // Create superposition
    sim.pz(&qid(0)); // Prepare |0⟩

    // After pz, qubit should be in |0⟩
    let result = sim.mz(&qid(0));
    assert!(!result[0].outcome, "pz should prepare |0⟩");
}

/// Verify pnz (prepare |1⟩) operation.
pub fn verify_pnz<S: StateVectorSimulator>(sim: &mut S) {
    sim.reset();
    sim.h(&qid(0));
    sim.pnz(&qid(0));

    let result = sim.mz(&qid(0));
    assert!(result[0].outcome, "pnz should prepare |1⟩");
}

/// Verify pz on multiple qubits.
pub fn verify_pz_multiple_qubits<S: StateVectorSimulator>(sim: &mut S) {
    if sim.num_qubits() < 2 {
        return;
    }

    // Test pz on multiple independent qubits in superposition
    sim.reset();
    sim.h(&qid(0));
    sim.h(&qid(1));

    sim.pz(&qid(0));
    sim.pz(&qid(1));

    let result0 = sim.mz(&qid(0));
    let result1 = sim.mz(&qid(1));

    assert!(!result0[0].outcome, "pz should prepare qubit 0 to |0⟩");
    assert!(!result1[0].outcome, "pz should prepare qubit 1 to |0⟩");

    // Test pz on one qubit while another is in |1⟩
    sim.reset();
    sim.h(&qid(0));
    sim.x(&qid(1)); // qubit 1 in |1⟩

    sim.pz(&qid(0)); // Only pz on qubit 0

    let result0 = sim.mz(&qid(0));
    let result1 = sim.mz(&qid(1));

    assert!(!result0[0].outcome, "pz should prepare qubit 0 to |0⟩");
    assert!(result1[0].outcome, "qubit 1 should still be |1⟩");
}

/// Verify measurement consistency - measuring the same qubit multiple times.
pub fn verify_measurement_consistency<S: StateVectorSimulator>(sim: &mut S) {
    // After a measurement, repeated measurements should give the same result
    for _ in 0..10 {
        sim.reset();
        sim.h(&qid(0));

        // First measurement collapses the state
        let result1 = sim.mz(&qid(0));

        // Subsequent measurements should give the same result
        for _ in 0..5 {
            let result_n = sim.mz(&qid(0));
            assert_eq!(
                result1[0].outcome, result_n[0].outcome,
                "Repeated measurements should be consistent"
            );
        }
    }
}

/// Verify detailed mz behavior with multiple qubits.
pub fn verify_mz_detailed<S: StateVectorSimulator>(sim: &mut S) {
    // Measure computational basis state
    sim.reset();
    sim.x(&qid(0)); // |1⟩

    let result = sim.mz(&qid(0));
    assert!(result[0].outcome, "mz on |1⟩ should give 1");

    // State should still be |1⟩ after measurement
    assert!(
        (sim.get_amplitude(1).norm() - 1.0).abs() < TOLERANCE,
        "State should remain |1⟩ after measuring |1⟩"
    );

    // Test multi-qubit measurement
    if sim.num_qubits() >= 2 {
        sim.reset();
        sim.x(&qid(0));
        sim.x(&qid(1)); // |11⟩

        let result0 = sim.mz(&qid(0));
        let result1 = sim.mz(&qid(1));

        assert!(result0[0].outcome, "mz on qubit 0 of |11⟩ should give 1");
        assert!(result1[0].outcome, "mz on qubit 1 of |11⟩ should give 1");
    }
}

// --- Gate Identity Tests ---

/// Verify various gate identities.
pub fn verify_gate_identities<S: StateVectorSimulator>(sim: &mut S) {
    // X^2 = I
    sim.reset();
    sim.x(&qid(0));
    sim.x(&qid(0));
    assert_amplitude_eq(sim.get_amplitude(0), Complex64::new(1.0, 0.0), "X^2 = I");

    // H^2 = I
    sim.reset();
    sim.h(&qid(0));
    sim.h(&qid(0));
    assert_amplitude_eq(sim.get_amplitude(0), Complex64::new(1.0, 0.0), "H^2 = I");

    // S^4 = I
    sim.reset();
    for _ in 0..4 {
        sim.sz(&qid(0));
    }
    assert_amplitude_eq(sim.get_amplitude(0), Complex64::new(1.0, 0.0), "S^4 = I");

    // HZH = X
    sim.reset();
    sim.h(&qid(0));
    sim.z(&qid(0));
    sim.h(&qid(0));
    assert_amplitude_near_zero(sim.get_amplitude(0), "HZH = X: |0⟩");
    assert_amplitude_eq(
        sim.get_amplitude(1),
        Complex64::new(1.0, 0.0),
        "HZH = X: |1⟩",
    );

    // HXH = Z (on |1⟩ state)
    sim.reset();
    sim.x(&qid(0)); // Prepare |1⟩
    sim.h(&qid(0));
    sim.x(&qid(0));
    sim.h(&qid(0));
    // Should give -|1⟩
    assert_amplitude_eq(
        sim.get_amplitude(1),
        Complex64::new(-1.0, 0.0),
        "HXH = Z on |1⟩",
    );
}

// --- Batch Operation Tests ---

/// Verify batch single-qubit gates produce same result as sequential application.
pub fn verify_batch_single_qubit_gates<S: StateVectorSimulator>(sim: &mut S) {
    if sim.num_qubits() < 3 {
        return;
    }

    let qubits_batch = [QubitId(0), QubitId(1), QubitId(2)];

    // Test H gate: batch vs sequential
    sim.reset();
    sim.h(&qubits_batch);
    let batch_amps: Vec<_> = (0..(1 << sim.num_qubits()))
        .map(|i| sim.get_amplitude(i))
        .collect();

    sim.reset();
    sim.h(&qid(0));
    sim.h(&qid(1));
    sim.h(&qid(2));
    let seq_amps: Vec<_> = (0..(1 << sim.num_qubits()))
        .map(|i| sim.get_amplitude(i))
        .collect();

    for (i, (b, s)) in batch_amps.iter().zip(seq_amps.iter()).enumerate() {
        assert!(
            (b - s).norm() < TOLERANCE,
            "H batch vs sequential mismatch at index {i}: batch={b:?}, seq={s:?}"
        );
    }

    // Test X gate: batch vs sequential
    sim.reset();
    sim.x(&qubits_batch);
    let batch_amps: Vec<_> = (0..(1 << sim.num_qubits()))
        .map(|i| sim.get_amplitude(i))
        .collect();

    sim.reset();
    sim.x(&qid(0));
    sim.x(&qid(1));
    sim.x(&qid(2));
    let seq_amps: Vec<_> = (0..(1 << sim.num_qubits()))
        .map(|i| sim.get_amplitude(i))
        .collect();

    for (i, (b, s)) in batch_amps.iter().zip(seq_amps.iter()).enumerate() {
        assert!(
            (b - s).norm() < TOLERANCE,
            "X batch vs sequential mismatch at index {i}: batch={b:?}, seq={s:?}"
        );
    }

    // Test Z gate: batch vs sequential (on superposition state)
    sim.reset();
    sim.h(&qubits_batch); // Create superposition first
    sim.z(&qubits_batch);
    let batch_amps: Vec<_> = (0..(1 << sim.num_qubits()))
        .map(|i| sim.get_amplitude(i))
        .collect();

    sim.reset();
    sim.h(&qid(0));
    sim.h(&qid(1));
    sim.h(&qid(2));
    sim.z(&qid(0));
    sim.z(&qid(1));
    sim.z(&qid(2));
    let seq_amps: Vec<_> = (0..(1 << sim.num_qubits()))
        .map(|i| sim.get_amplitude(i))
        .collect();

    for (i, (b, s)) in batch_amps.iter().zip(seq_amps.iter()).enumerate() {
        assert!(
            (b - s).norm() < TOLERANCE,
            "Z batch vs sequential mismatch at index {i}: batch={b:?}, seq={s:?}"
        );
    }

    // Test SZ gate: batch vs sequential
    sim.reset();
    sim.h(&qubits_batch);
    sim.sz(&qubits_batch);
    let batch_amps: Vec<_> = (0..(1 << sim.num_qubits()))
        .map(|i| sim.get_amplitude(i))
        .collect();

    sim.reset();
    sim.h(&qid(0));
    sim.h(&qid(1));
    sim.h(&qid(2));
    sim.sz(&qid(0));
    sim.sz(&qid(1));
    sim.sz(&qid(2));
    let seq_amps: Vec<_> = (0..(1 << sim.num_qubits()))
        .map(|i| sim.get_amplitude(i))
        .collect();

    for (i, (b, s)) in batch_amps.iter().zip(seq_amps.iter()).enumerate() {
        assert!(
            (b - s).norm() < TOLERANCE,
            "SZ batch vs sequential mismatch at index {i}: batch={b:?}, seq={s:?}"
        );
    }
}

/// Verify batch two-qubit gates produce same result as sequential application.
pub fn verify_batch_two_qubit_gates<S: StateVectorSimulator>(sim: &mut S) {
    if sim.num_qubits() < 4 {
        return;
    }

    // Test CX gate: batch of two pairs vs sequential
    // Batch: CX on (0,1) and (2,3) simultaneously
    let pairs_batch = [(QubitId(0), QubitId(1)), (QubitId(2), QubitId(3))];

    sim.reset();
    sim.x(&qid(0)); // Set control qubits to |1⟩
    sim.x(&qid(2));
    sim.cx(&pairs_batch);
    let batch_amps: Vec<_> = (0..(1 << sim.num_qubits()))
        .map(|i| sim.get_amplitude(i))
        .collect();

    sim.reset();
    sim.x(&qid(0));
    sim.x(&qid(2));
    sim.cx(&qid2(0, 1));
    sim.cx(&qid2(2, 3));
    let seq_amps: Vec<_> = (0..(1 << sim.num_qubits()))
        .map(|i| sim.get_amplitude(i))
        .collect();

    for (i, (b, s)) in batch_amps.iter().zip(seq_amps.iter()).enumerate() {
        assert!(
            (b - s).norm() < TOLERANCE,
            "CX batch vs sequential mismatch at index {i}: batch={b:?}, seq={s:?}"
        );
    }

    // Test CZ gate: batch vs sequential
    sim.reset();
    sim.h(&[QubitId(0), QubitId(1), QubitId(2), QubitId(3)]); // Superposition
    sim.cz(&pairs_batch);
    let batch_amps: Vec<_> = (0..(1 << sim.num_qubits()))
        .map(|i| sim.get_amplitude(i))
        .collect();

    sim.reset();
    sim.h(&[QubitId(0), QubitId(1), QubitId(2), QubitId(3)]);
    sim.cz(&qid2(0, 1));
    sim.cz(&qid2(2, 3));
    let seq_amps: Vec<_> = (0..(1 << sim.num_qubits()))
        .map(|i| sim.get_amplitude(i))
        .collect();

    for (i, (b, s)) in batch_amps.iter().zip(seq_amps.iter()).enumerate() {
        assert!(
            (b - s).norm() < TOLERANCE,
            "CZ batch vs sequential mismatch at index {i}: batch={b:?}, seq={s:?}"
        );
    }

    // Test SWAP gate: batch vs sequential
    sim.reset();
    sim.x(&qid(0)); // |1000⟩
    sim.swap(&pairs_batch);
    let batch_amps: Vec<_> = (0..(1 << sim.num_qubits()))
        .map(|i| sim.get_amplitude(i))
        .collect();

    sim.reset();
    sim.x(&qid(0));
    sim.swap(&qid2(0, 1));
    sim.swap(&qid2(2, 3));
    let seq_amps: Vec<_> = (0..(1 << sim.num_qubits()))
        .map(|i| sim.get_amplitude(i))
        .collect();

    for (i, (b, s)) in batch_amps.iter().zip(seq_amps.iter()).enumerate() {
        assert!(
            (b - s).norm() < TOLERANCE,
            "SWAP batch vs sequential mismatch at index {i}: batch={b:?}, seq={s:?}"
        );
    }
}

/// Verify batch measurements work correctly.
pub fn verify_batch_measurements<S: StateVectorSimulator>(sim: &mut S) {
    if sim.num_qubits() < 3 {
        return;
    }

    // Test measuring deterministic state in batch
    sim.reset();
    sim.x(&qid(0));
    sim.x(&qid(2)); // State: |101⟩ (in qubit ordering)

    let qubits = [QubitId(0), QubitId(1), QubitId(2)];
    let results = sim.mz(&qubits);

    assert_eq!(
        results.len(),
        3,
        "Batch measurement should return 3 results"
    );
    assert!(results[0].outcome, "Qubit 0 should measure |1⟩");
    assert!(!results[1].outcome, "Qubit 1 should measure |0⟩");
    assert!(results[2].outcome, "Qubit 2 should measure |1⟩");

    // Verify state after batch measurement matches expectations
    // After measuring |101⟩, state should still be |101⟩
    let expected_idx = 0b101; // qubit 0 and 2 are |1⟩
    assert!(
        (sim.get_amplitude(expected_idx).norm() - 1.0).abs() < TOLERANCE,
        "State should be |101⟩ after measurement"
    );
}

/// Verify batch rotation gates produce same result as sequential application.
pub fn verify_batch_rotation_gates<S: StateVectorSimulator + ArbitraryRotationGateable>(
    sim: &mut S,
) {
    if sim.num_qubits() < 3 {
        return;
    }

    let qubits_batch = [QubitId(0), QubitId(1), QubitId(2)];
    let theta = Angle64::from_radians(FRAC_PI_4);

    // Test RX gate: batch vs sequential
    sim.reset();
    sim.rx(theta, &qubits_batch);
    let batch_amps: Vec<_> = (0..(1 << sim.num_qubits()))
        .map(|i| sim.get_amplitude(i))
        .collect();

    sim.reset();
    sim.rx(theta, &qid(0));
    sim.rx(theta, &qid(1));
    sim.rx(theta, &qid(2));
    let seq_amps: Vec<_> = (0..(1 << sim.num_qubits()))
        .map(|i| sim.get_amplitude(i))
        .collect();

    for (i, (b, s)) in batch_amps.iter().zip(seq_amps.iter()).enumerate() {
        assert!(
            (b - s).norm() < TOLERANCE,
            "RX batch vs sequential mismatch at index {i}: batch={b:?}, seq={s:?}"
        );
    }

    // Test RY gate: batch vs sequential
    sim.reset();
    sim.ry(theta, &qubits_batch);
    let batch_amps: Vec<_> = (0..(1 << sim.num_qubits()))
        .map(|i| sim.get_amplitude(i))
        .collect();

    sim.reset();
    sim.ry(theta, &qid(0));
    sim.ry(theta, &qid(1));
    sim.ry(theta, &qid(2));
    let seq_amps: Vec<_> = (0..(1 << sim.num_qubits()))
        .map(|i| sim.get_amplitude(i))
        .collect();

    for (i, (b, s)) in batch_amps.iter().zip(seq_amps.iter()).enumerate() {
        assert!(
            (b - s).norm() < TOLERANCE,
            "RY batch vs sequential mismatch at index {i}: batch={b:?}, seq={s:?}"
        );
    }

    // Test RZ gate: batch vs sequential
    sim.reset();
    sim.h(&qubits_batch); // Create superposition to see RZ effects
    sim.rz(theta, &qubits_batch);
    let batch_amps: Vec<_> = (0..(1 << sim.num_qubits()))
        .map(|i| sim.get_amplitude(i))
        .collect();

    sim.reset();
    sim.h(&qid(0));
    sim.h(&qid(1));
    sim.h(&qid(2));
    sim.rz(theta, &qid(0));
    sim.rz(theta, &qid(1));
    sim.rz(theta, &qid(2));
    let seq_amps: Vec<_> = (0..(1 << sim.num_qubits()))
        .map(|i| sim.get_amplitude(i))
        .collect();

    for (i, (b, s)) in batch_amps.iter().zip(seq_amps.iter()).enumerate() {
        assert!(
            (b - s).norm() < TOLERANCE,
            "RZ batch vs sequential mismatch at index {i}: batch={b:?}, seq={s:?}"
        );
    }
}

/// Verify batch two-qubit rotation gates produce same result as sequential.
pub fn verify_batch_two_qubit_rotation_gates<
    S: StateVectorSimulator + ArbitraryRotationGateable,
>(
    sim: &mut S,
) {
    if sim.num_qubits() < 4 {
        return;
    }

    let pairs_batch = [(QubitId(0), QubitId(1)), (QubitId(2), QubitId(3))];
    let theta = Angle64::from_radians(FRAC_PI_4);

    // Test RZZ gate: batch vs sequential
    sim.reset();
    sim.h(&[QubitId(0), QubitId(1), QubitId(2), QubitId(3)]); // Superposition
    sim.rzz(theta, &pairs_batch);
    let batch_amps: Vec<_> = (0..(1 << sim.num_qubits()))
        .map(|i| sim.get_amplitude(i))
        .collect();

    sim.reset();
    sim.h(&[QubitId(0), QubitId(1), QubitId(2), QubitId(3)]);
    sim.rzz(theta, &qid2(0, 1));
    sim.rzz(theta, &qid2(2, 3));
    let seq_amps: Vec<_> = (0..(1 << sim.num_qubits()))
        .map(|i| sim.get_amplitude(i))
        .collect();

    for (i, (b, s)) in batch_amps.iter().zip(seq_amps.iter()).enumerate() {
        assert!(
            (b - s).norm() < TOLERANCE,
            "RZZ batch vs sequential mismatch at index {i}: batch={b:?}, seq={s:?}"
        );
    }

    // Test RXX gate: batch vs sequential
    sim.reset();
    sim.rxx(theta, &pairs_batch);
    let batch_amps: Vec<_> = (0..(1 << sim.num_qubits()))
        .map(|i| sim.get_amplitude(i))
        .collect();

    sim.reset();
    sim.rxx(theta, &qid2(0, 1));
    sim.rxx(theta, &qid2(2, 3));
    let seq_amps: Vec<_> = (0..(1 << sim.num_qubits()))
        .map(|i| sim.get_amplitude(i))
        .collect();

    for (i, (b, s)) in batch_amps.iter().zip(seq_amps.iter()).enumerate() {
        assert!(
            (b - s).norm() < TOLERANCE,
            "RXX batch vs sequential mismatch at index {i}: batch={b:?}, seq={s:?}"
        );
    }
}

/// Verify rotation gate identities.
pub fn verify_rotation_identities<S: StateVectorSimulator + ArbitraryRotationGateable>(
    sim: &mut S,
) {
    // RX(π) ≈ -iX (up to global phase)
    sim.reset();
    sim.rx(Angle64::from_radians(PI), &qid(0));
    let amp1_rx = sim.get_amplitude(1);

    sim.reset();
    sim.x(&qid(0));
    let amp1_x = sim.get_amplitude(1);

    // |amp1_rx| should equal |amp1_x|
    assert!(
        (amp1_rx.norm() - amp1_x.norm()).abs() < TOLERANCE,
        "RX(π) should flip like X"
    );

    // RY(π) ≈ iY (up to global phase), which means RY(π)|0⟩ = |1⟩
    sim.reset();
    sim.ry(Angle64::from_radians(PI), &qid(0));
    assert_amplitude_near_zero(sim.get_amplitude(0), "RY(π)|0⟩: |0⟩");
    assert!(
        (sim.get_amplitude(1).norm() - 1.0).abs() < TOLERANCE,
        "RY(π)|0⟩: |1⟩ magnitude"
    );

    // RZ(2π) = I (up to global phase)
    sim.reset();
    sim.h(&qid(0)); // Create superposition to detect phase
    let amp0_before = sim.get_amplitude(0);
    let amp1_before = sim.get_amplitude(1);
    sim.rz(Angle64::from_radians(2.0 * PI), &qid(0));
    let amp0_after = sim.get_amplitude(0);
    let amp1_after = sim.get_amplitude(1);

    // Relative phase should be preserved
    let ratio_before = amp0_before / amp1_before;
    let ratio_after = amp0_after / amp1_after;
    assert_amplitude_eq(ratio_before, ratio_after, "RZ(2π) preserves relative phase");
}

/// Verify U gate (general single-qubit unitary).
///
/// U(θ, φ, λ) = [[cos(θ/2), -e^(iλ)sin(θ/2)],
///               [e^(iφ)sin(θ/2), e^(i(φ+λ))cos(θ/2)]]
pub fn verify_u_gate<S: StateVectorSimulator + ArbitraryRotationGateable>(sim: &mut S) {
    // U(π, 0, π) = X (Pauli X gate)
    sim.reset();
    sim.u(
        Angle64::from_radians(PI),
        Angle64::from_radians(0.0),
        Angle64::from_radians(PI),
        &qid(0),
    );

    assert_amplitude_near_zero(sim.get_amplitude(0), "U(π,0,π)|0⟩: |0⟩");
    assert!(
        (sim.get_amplitude(1).norm() - 1.0).abs() < TOLERANCE,
        "U(π,0,π)|0⟩: |1⟩ magnitude should be 1"
    );

    // U(π, π/2, π/2) = Y (Pauli Y gate, up to global phase)
    sim.reset();
    sim.u(
        Angle64::from_radians(PI),
        Angle64::from_radians(FRAC_PI_2),
        Angle64::from_radians(FRAC_PI_2),
        &qid(0),
    );

    assert_amplitude_near_zero(sim.get_amplitude(0), "U(π,π/2,π/2)|0⟩: |0⟩");
    assert!(
        (sim.get_amplitude(1).norm() - 1.0).abs() < TOLERANCE,
        "U(π,π/2,π/2)|0⟩: |1⟩ magnitude should be 1"
    );

    // U(0, 0, π) = Z (Pauli Z gate) - only adds phase to |1⟩
    sim.reset();
    sim.x(&qid(0)); // Prepare |1⟩
    let amp_before = sim.get_amplitude(1);
    sim.u(
        Angle64::from_radians(0.0),
        Angle64::from_radians(0.0),
        Angle64::from_radians(PI),
        &qid(0),
    );
    let amp_after = sim.get_amplitude(1);

    // Magnitude should be preserved
    assert!(
        (amp_before.norm() - amp_after.norm()).abs() < TOLERANCE,
        "U(0,0,π) preserves magnitude"
    );

    // Unitarity check: U preserves normalization
    sim.reset();
    sim.h(&qid(0)); // Start with superposition
    sim.u(
        Angle64::from_radians(1.23),
        Angle64::from_radians(0.45),
        Angle64::from_radians(0.67),
        &qid(0),
    ); // Apply U with arbitrary angles

    let amp0 = sim.get_amplitude(0);
    let amp1 = sim.get_amplitude(1);
    let norm_sq = amp0.norm_sqr() + amp1.norm_sqr();

    assert!(
        (norm_sq - 1.0).abs() < TOLERANCE,
        "U gate preserves normalization"
    );
}

/// Verify R1XY gate.
///
/// R1XY(θ, φ) applies a rotation in the XY plane of the Bloch sphere.
pub fn verify_r1xy_gate<S: StateVectorSimulator + ArbitraryRotationGateable>(sim: &mut S) {
    // R1XY(π, 0) should act like X (flip |0⟩ to |1⟩)
    sim.reset();
    sim.r1xy(
        Angle64::from_radians(PI),
        Angle64::from_radians(0.0),
        &qid(0),
    );

    assert_amplitude_near_zero(sim.get_amplitude(0), "R1XY(π,0)|0⟩: |0⟩");
    assert!(
        (sim.get_amplitude(1).norm() - 1.0).abs() < TOLERANCE,
        "R1XY(π,0)|0⟩: |1⟩ magnitude should be 1"
    );

    // R1XY(π, π/2) should act like Y (up to global phase)
    sim.reset();
    sim.r1xy(
        Angle64::from_radians(PI),
        Angle64::from_radians(FRAC_PI_2),
        &qid(0),
    );

    assert_amplitude_near_zero(sim.get_amplitude(0), "R1XY(π,π/2)|0⟩: |0⟩");
    assert!(
        (sim.get_amplitude(1).norm() - 1.0).abs() < TOLERANCE,
        "R1XY(π,π/2)|0⟩: |1⟩ magnitude should be 1"
    );

    // R1XY(π/2, 0) should create superposition like a Hadamard-like rotation
    sim.reset();
    sim.r1xy(
        Angle64::from_radians(FRAC_PI_2),
        Angle64::from_radians(0.0),
        &qid(0),
    );

    // Both amplitudes should have equal magnitude
    let amp0 = sim.get_amplitude(0);
    let amp1 = sim.get_amplitude(1);
    assert!(
        (amp0.norm() - amp1.norm()).abs() < TOLERANCE,
        "R1XY(π/2,0)|0⟩: equal superposition magnitudes"
    );

    // Verify unitarity
    let norm_sq = amp0.norm_sqr() + amp1.norm_sqr();
    assert!(
        (norm_sq - 1.0).abs() < TOLERANCE,
        "R1XY preserves normalization"
    );
}

/// Verify single-qubit rotation with various angles.
pub fn verify_single_qubit_rotation<S: StateVectorSimulator + ArbitraryRotationGateable>(
    sim: &mut S,
) {
    // Test rotation composition: RZ(α)·RY(β)·RZ(γ) decomposition
    let angles = [0.0, FRAC_PI_4, FRAC_PI_2, PI, 3.0 * FRAC_PI_2];

    for &theta in &angles {
        sim.reset();
        sim.ry(Angle64::from_radians(theta), &qid(0));

        // Verify normalization after rotation
        let amp0 = sim.get_amplitude(0);
        let amp1 = sim.get_amplitude(1);
        let norm_sq = amp0.norm_sqr() + amp1.norm_sqr();

        assert!(
            (norm_sq - 1.0).abs() < TOLERANCE,
            "RY({theta}) preserves normalization"
        );

        // Verify expected probabilities for RY
        // RY(θ)|0⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩
        let expected_p0 = (theta / 2.0).cos().powi(2);
        let expected_p1 = (theta / 2.0).sin().powi(2);

        assert!(
            (amp0.norm_sqr() - expected_p0).abs() < TOLERANCE,
            "RY({theta})|0⟩: P(|0⟩) = cos²(θ/2)"
        );
        assert!(
            (amp1.norm_sqr() - expected_p1).abs() < TOLERANCE,
            "RY({theta})|0⟩: P(|1⟩) = sin²(θ/2)"
        );
    }
}

// --- Locality Tests ---

/// Verify single-qubit gates only affect the target qubit.
pub fn verify_single_qubit_locality<S: StateVectorSimulator>(sim: &mut S) {
    if sim.num_qubits() < 3 {
        return;
    }

    let n = 1 << sim.num_qubits();

    // Prepare qubit 1 as |1⟩, then apply H to qubit 0 only.
    // Qubit 1 and 2 should be unaffected.
    sim.reset();
    sim.x(&qid(1));
    sim.h(&qid(0));

    // Basis index: bit i = value of qubit i
    // q1=1 means bit 1 set. q0 in superposition: bit 0 = 0 or 1.
    // Expected nonzero: 0b010 (q0=0,q1=1) and 0b011 (q0=1,q1=1)
    for i in 0..n {
        let amp = sim.get_amplitude(i);
        if i == 0b010 || i == 0b011 {
            assert!(
                (amp.norm() - FRAC_1_SQRT_2).abs() < TOLERANCE,
                "H on q0: state {i:#b} should have amplitude 1/sqrt2, got {amp:?}"
            );
        } else {
            assert_amplitude_near_zero(amp, &format!("H on q0: state {i:#b} should be zero"));
        }
    }

    // Apply X to qubit 2 only -- qubits 0 and 1 should be unaffected
    sim.reset();
    sim.h(&qid(0)); // q0 in superposition
    sim.x(&qid(2)); // flip q2

    // q0 in superposition (bit 0 = 0 or 1), q2=1 (bit 2 set)
    // Expected nonzero: 0b100 (q0=0,q2=1) and 0b101 (q0=1,q2=1)
    for i in 0..n {
        let amp = sim.get_amplitude(i);
        if i == 0b100 || i == 0b101 {
            assert!(
                (amp.norm() - FRAC_1_SQRT_2).abs() < TOLERANCE,
                "X on q2 after H on q0: state {i:#b} should have amplitude 1/sqrt2, got {amp:?}"
            );
        } else {
            assert_amplitude_near_zero(
                amp,
                &format!("X on q2 after H on q0: state {i:#b} should be zero"),
            );
        }
    }
}

/// Verify two-qubit gates only affect the target qubits.
pub fn verify_two_qubit_locality<S: StateVectorSimulator>(sim: &mut S) {
    if sim.num_qubits() < 4 {
        return;
    }

    // Prepare |1000⟩ (qubit 0 is |1⟩) then apply CX(0,1)
    // Only qubits 0 and 1 should change; qubits 2 and 3 remain |0⟩
    sim.reset();
    sim.x(&qid(0)); // |1000⟩
    sim.cx(&qid2(0, 1)); // should flip qubit 1: |1100⟩

    let n = 1 << sim.num_qubits();
    let expected_idx = 0b0011; // qubits 0 and 1 are |1⟩
    for i in 0..n {
        let amp = sim.get_amplitude(i);
        if i == expected_idx {
            assert!(
                (amp.norm() - 1.0).abs() < TOLERANCE,
                "CX(0,1) on |1000⟩: state {expected_idx:#b} should have amplitude 1"
            );
        } else {
            assert_amplitude_near_zero(
                amp,
                &format!("CX(0,1) on |1000⟩: state {i:#b} should be zero"),
            );
        }
    }

    // Apply SWAP(2,3) — qubits 0 and 1 should be unaffected
    sim.reset();
    sim.x(&qid(0)); // |1000⟩
    sim.x(&qid(2)); // |1010⟩
    sim.swap(&qid2(2, 3)); // swap q2 and q3: |1001⟩

    let expected_idx = 0b1001; // q0=1, q3=1
    for i in 0..n {
        let amp = sim.get_amplitude(i);
        if i == expected_idx {
            assert!(
                (amp.norm() - 1.0).abs() < TOLERANCE,
                "SWAP(2,3): state {expected_idx:#b} should have amplitude 1"
            );
        } else {
            assert_amplitude_near_zero(amp, &format!("SWAP(2,3): state {i:#b} should be zero"));
        }
    }
}

// --- Adjoint Gate Tests ---

/// Verify that adjoint (dagger) gates are inverses of the corresponding gates.
pub fn verify_adjoint_gates<S: StateVectorSimulator>(sim: &mut S) {
    // Helper: apply gate then adjoint, should return to |0⟩
    // SX · SXdg = I
    sim.reset();
    sim.h(&qid(0)); // start in superposition to detect phase issues
    let amp0_before = sim.get_amplitude(0);
    let amp1_before = sim.get_amplitude(1);
    sim.sx(&qid(0));
    sim.sxdg(&qid(0));
    assert_amplitude_eq(
        sim.get_amplitude(0),
        amp0_before,
        "SX·SXdg = I: |0⟩ component",
    );
    assert_amplitude_eq(
        sim.get_amplitude(1),
        amp1_before,
        "SX·SXdg = I: |1⟩ component",
    );

    // SXdg · SX = I
    sim.reset();
    sim.h(&qid(0));
    let amp0_before = sim.get_amplitude(0);
    let amp1_before = sim.get_amplitude(1);
    sim.sxdg(&qid(0));
    sim.sx(&qid(0));
    assert_amplitude_eq(
        sim.get_amplitude(0),
        amp0_before,
        "SXdg·SX = I: |0⟩ component",
    );
    assert_amplitude_eq(
        sim.get_amplitude(1),
        amp1_before,
        "SXdg·SX = I: |1⟩ component",
    );

    // SY · SYdg = I
    sim.reset();
    sim.h(&qid(0));
    let amp0_before = sim.get_amplitude(0);
    let amp1_before = sim.get_amplitude(1);
    sim.sy(&qid(0));
    sim.sydg(&qid(0));
    assert_amplitude_eq(
        sim.get_amplitude(0),
        amp0_before,
        "SY·SYdg = I: |0⟩ component",
    );
    assert_amplitude_eq(
        sim.get_amplitude(1),
        amp1_before,
        "SY·SYdg = I: |1⟩ component",
    );

    // SYdg · SY = I
    sim.reset();
    sim.h(&qid(0));
    let amp0_before = sim.get_amplitude(0);
    let amp1_before = sim.get_amplitude(1);
    sim.sydg(&qid(0));
    sim.sy(&qid(0));
    assert_amplitude_eq(
        sim.get_amplitude(0),
        amp0_before,
        "SYdg·SY = I: |0⟩ component",
    );
    assert_amplitude_eq(
        sim.get_amplitude(1),
        amp1_before,
        "SYdg·SY = I: |1⟩ component",
    );

    // SZ · SZdg = I
    sim.reset();
    sim.h(&qid(0));
    let amp0_before = sim.get_amplitude(0);
    let amp1_before = sim.get_amplitude(1);
    sim.sz(&qid(0));
    sim.szdg(&qid(0));
    assert_amplitude_eq(
        sim.get_amplitude(0),
        amp0_before,
        "SZ·SZdg = I: |0⟩ component",
    );
    assert_amplitude_eq(
        sim.get_amplitude(1),
        amp1_before,
        "SZ·SZdg = I: |1⟩ component",
    );

    // SZdg · SZ = I
    sim.reset();
    sim.h(&qid(0));
    let amp0_before = sim.get_amplitude(0);
    let amp1_before = sim.get_amplitude(1);
    sim.szdg(&qid(0));
    sim.sz(&qid(0));
    assert_amplitude_eq(
        sim.get_amplitude(0),
        amp0_before,
        "SZdg·SZ = I: |0⟩ component",
    );
    assert_amplitude_eq(
        sim.get_amplitude(1),
        amp1_before,
        "SZdg·SZ = I: |1⟩ component",
    );
}

/// Verify that two-qubit adjoint gates are proper inverses.
pub fn verify_adjoint_two_qubit_gates<S: StateVectorSimulator>(sim: &mut S) {
    if sim.num_qubits() < 2 {
        return;
    }

    let n = 1 << sim.num_qubits();

    // SXX · SXXdg = I
    sim.reset();
    sim.h(&qid(0));
    sim.cx(&qid2(0, 1));
    let before: Vec<_> = (0..n).map(|i| sim.get_amplitude(i)).collect();
    sim.sxx(&qid2(0, 1));
    sim.sxxdg(&qid2(0, 1));
    for (i, amp_before) in before.iter().enumerate() {
        assert_amplitude_eq(
            sim.get_amplitude(i),
            *amp_before,
            &format!("SXX·SXXdg = I at {i}"),
        );
    }

    // SYY · SYYdg = I
    sim.reset();
    sim.h(&qid(0));
    sim.cx(&qid2(0, 1));
    let before: Vec<_> = (0..n).map(|i| sim.get_amplitude(i)).collect();
    sim.syy(&qid2(0, 1));
    sim.syydg(&qid2(0, 1));
    for (i, amp_before) in before.iter().enumerate() {
        assert_amplitude_eq(
            sim.get_amplitude(i),
            *amp_before,
            &format!("SYY·SYYdg = I at {i}"),
        );
    }

    // SZZ · SZZdg = I
    sim.reset();
    sim.h(&qid(0));
    sim.cx(&qid2(0, 1));
    let before: Vec<_> = (0..n).map(|i| sim.get_amplitude(i)).collect();
    sim.szz(&qid2(0, 1));
    sim.szzdg(&qid2(0, 1));
    for (i, amp_before) in before.iter().enumerate() {
        assert_amplitude_eq(
            sim.get_amplitude(i),
            *amp_before,
            &format!("SZZ·SZZdg = I at {i}"),
        );
    }
}

// --- State Preparation Tests ---

/// Verify Bell state preparation via standard circuit.
pub fn verify_bell_state_preparation<S: StateVectorSimulator>(sim: &mut S) {
    if sim.num_qubits() < 2 {
        return;
    }

    // Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2 via H·CX
    sim.reset();
    sim.h(&qid(0));
    sim.cx(&qid2(0, 1));

    assert!(
        (sim.get_amplitude(0).norm() - FRAC_1_SQRT_2).abs() < TOLERANCE,
        "Bell |Φ+⟩: |00⟩ amplitude should be 1/√2"
    );
    assert_amplitude_near_zero(sim.get_amplitude(1), "Bell |Φ+⟩: |01⟩ should be 0");
    assert_amplitude_near_zero(sim.get_amplitude(2), "Bell |Φ+⟩: |10⟩ should be 0");
    assert!(
        (sim.get_amplitude(3).norm() - FRAC_1_SQRT_2).abs() < TOLERANCE,
        "Bell |Φ+⟩: |11⟩ amplitude should be 1/√2"
    );

    // Bell state |Ψ+⟩ = (|01⟩ + |10⟩)/√2 via H(q0)·CX(0,1)·X(q1)
    // H·CX gives |Φ+⟩ = (|00⟩ + |11⟩)/√2, then X(q1) flips q1 to get |Ψ+⟩
    sim.reset();
    sim.h(&qid(0));
    sim.cx(&qid2(0, 1));
    sim.x(&qid(1));

    assert_amplitude_near_zero(sim.get_amplitude(0), "Bell |Ψ+⟩: |00⟩ should be 0");
    assert!(
        (sim.get_amplitude(1).norm() - FRAC_1_SQRT_2).abs() < TOLERANCE,
        "Bell |Ψ+⟩: |01⟩ amplitude should be 1/√2"
    );
    assert!(
        (sim.get_amplitude(2).norm() - FRAC_1_SQRT_2).abs() < TOLERANCE,
        "Bell |Ψ+⟩: |10⟩ amplitude should be 1/√2"
    );
    assert_amplitude_near_zero(sim.get_amplitude(3), "Bell |Ψ+⟩: |11⟩ should be 0");
}

/// Verify GHZ state preparation.
pub fn verify_ghz_state<S: StateVectorSimulator>(sim: &mut S) {
    if sim.num_qubits() < 3 {
        return;
    }

    // GHZ = (|000⟩ + |111⟩)/√2 via H(0)·CX(0,1)·CX(0,2)
    sim.reset();
    sim.h(&qid(0));
    sim.cx(&qid2(0, 1));
    sim.cx(&qid2(0, 2));

    let n = 1 << sim.num_qubits();
    let idx_all_zero = 0;
    let idx_first_three_one = 0b111; // qubits 0, 1, 2 are |1⟩

    for i in 0..n {
        let amp = sim.get_amplitude(i);
        if i == idx_all_zero || i == idx_first_three_one {
            assert!(
                (amp.norm() - FRAC_1_SQRT_2).abs() < TOLERANCE,
                "GHZ: state {i:#b} should have amplitude 1/√2, got {amp:?}"
            );
        } else {
            assert_amplitude_near_zero(amp, &format!("GHZ: state {i:#b} should be zero"));
        }
    }
}

/// Verify that H on multiple qubits creates equal superposition.
pub fn verify_equal_superposition<S: StateVectorSimulator + ArbitraryRotationGateable>(
    sim: &mut S,
) {
    if sim.num_qubits() < 3 {
        return;
    }

    sim.reset();
    for q in 0..sim.num_qubits().min(3) {
        sim.h(&[QubitId(q)]);
    }

    // Should be equal superposition of first 8 basis states
    let n_super = 1 << 3;
    #[allow(clippy::cast_precision_loss)]
    let expected_amp = 1.0 / (n_super as f64).sqrt();
    for i in 0..n_super {
        assert!(
            (sim.get_amplitude(i).norm() - expected_amp).abs() < TOLERANCE,
            "Equal superposition: state {i} should have amplitude {expected_amp}"
        );
    }
}

// --- Gate Decomposition Tests ---

/// Verify standard gate decompositions hold.
pub fn verify_gate_decompositions<S: StateVectorSimulator>(sim: &mut S) {
    if sim.num_qubits() < 2 {
        return;
    }

    // CZ = H(target)·CX·H(target)
    sim.reset();
    sim.h(&qid(0));
    sim.h(&qid(1)); // superposition to detect phase
    sim.cz(&qid2(0, 1));
    let cz_amps: Vec<_> = (0..(1 << sim.num_qubits()))
        .map(|i| sim.get_amplitude(i))
        .collect();

    sim.reset();
    sim.h(&qid(0));
    sim.h(&qid(1));
    sim.h(&qid(1));
    sim.cx(&qid2(0, 1));
    sim.h(&qid(1));
    let decomp_amps: Vec<_> = (0..(1 << sim.num_qubits()))
        .map(|i| sim.get_amplitude(i))
        .collect();

    for (i, (cz, dec)) in cz_amps.iter().zip(decomp_amps.iter()).enumerate() {
        assert!(
            (cz - dec).norm() < TOLERANCE,
            "CZ = H·CX·H decomposition mismatch at index {i}: cz={cz:?}, decomp={dec:?}"
        );
    }
}

/// Verify SWAP = CX(a,b)·CX(b,a)·CX(a,b).
pub fn verify_swap_decomposition<S: StateVectorSimulator>(sim: &mut S) {
    if sim.num_qubits() < 2 {
        return;
    }

    // Test on a non-trivial state: q0 in superposition, q1 = |1⟩
    sim.reset();
    sim.h(&qid(0));
    sim.x(&qid(1));
    sim.swap(&qid2(0, 1));
    let swap_amps: Vec<_> = (0..(1 << sim.num_qubits()))
        .map(|i| sim.get_amplitude(i))
        .collect();

    sim.reset();
    sim.h(&qid(0));
    sim.x(&qid(1));
    sim.cx(&qid2(0, 1));
    sim.cx(&qid2(1, 0));
    sim.cx(&qid2(0, 1));
    let decomp_amps: Vec<_> = (0..(1 << sim.num_qubits()))
        .map(|i| sim.get_amplitude(i))
        .collect();

    for (i, (sw, dec)) in swap_amps.iter().zip(decomp_amps.iter()).enumerate() {
        assert!(
            (sw - dec).norm() < TOLERANCE,
            "SWAP = CX·CX·CX decomposition mismatch at index {i}: swap={sw:?}, decomp={dec:?}"
        );
    }
}

/// Verify X = HZH identity.
pub fn verify_x_hzh_decomposition<S: StateVectorSimulator>(sim: &mut S) {
    // Apply X to |0⟩
    sim.reset();
    sim.x(&qid(0));
    let x_amps: Vec<_> = (0..(1 << sim.num_qubits()))
        .map(|i| sim.get_amplitude(i))
        .collect();

    // Apply HZH to |0⟩
    sim.reset();
    sim.h(&qid(0));
    sim.z(&qid(0));
    sim.h(&qid(0));
    let hzh_amps: Vec<_> = (0..(1 << sim.num_qubits()))
        .map(|i| sim.get_amplitude(i))
        .collect();

    for (i, (x, hzh)) in x_amps.iter().zip(hzh_amps.iter()).enumerate() {
        assert!(
            (x - hzh).norm() < TOLERANCE,
            "X = HZH decomposition mismatch at index {i}: x={x:?}, hzh={hzh:?}"
        );
    }
}

/// Verify Z = SZ·SZ (S squared equals Z).
pub fn verify_z_from_sz<S: StateVectorSimulator>(sim: &mut S) {
    // On superposition state to detect phase
    sim.reset();
    sim.h(&qid(0));
    sim.z(&qid(0));
    let z_amps: Vec<_> = (0..(1 << sim.num_qubits()))
        .map(|i| sim.get_amplitude(i))
        .collect();

    sim.reset();
    sim.h(&qid(0));
    sim.sz(&qid(0));
    sim.sz(&qid(0));
    let ss_amps: Vec<_> = (0..(1 << sim.num_qubits()))
        .map(|i| sim.get_amplitude(i))
        .collect();

    for (i, (z, ss)) in z_amps.iter().zip(ss_amps.iter()).enumerate() {
        assert!(
            (z - ss).norm() < TOLERANCE,
            "Z = SZ·SZ decomposition mismatch at index {i}: z={z:?}, sz²={ss:?}"
        );
    }
}

/// Verify Y = iXZ (up to global phase).
pub fn verify_y_xz_decomposition<S: StateVectorSimulator>(sim: &mut S) {
    // Apply Y to a superposition state
    sim.reset();
    sim.h(&qid(0));
    sim.y(&qid(0));
    let y_amps: Vec<_> = (0..(1 << sim.num_qubits()))
        .map(|i| sim.get_amplitude(i))
        .collect();

    // Y = iXZ, so Y|psi> and XZ|psi> differ by a global phase of i.
    sim.reset();
    sim.h(&qid(0));
    sim.z(&qid(0));
    sim.x(&qid(0));
    let xz_amps: Vec<_> = (0..(1 << sim.num_qubits()))
        .map(|i| sim.get_amplitude(i))
        .collect();

    // Verify Y and XZ differ by a consistent global phase.
    // Find the first non-zero amplitude pair and compute the phase ratio.
    let mut phase_ratio = None;
    for (i, (y, xz)) in y_amps.iter().zip(xz_amps.iter()).enumerate() {
        if y.norm() < TOLERANCE && xz.norm() < TOLERANCE {
            continue;
        }
        assert!(
            y.norm() > TOLERANCE && xz.norm() > TOLERANCE,
            "Y vs XZ: index {i} has one zero and one non-zero"
        );
        let ratio = y / xz;
        match phase_ratio {
            None => phase_ratio = Some(ratio),
            Some(expected) => {
                assert!(
                    (ratio - expected).norm() < TOLERANCE,
                    "Y vs XZ: inconsistent phase ratio at index {i}: {ratio:?} vs {expected:?}"
                );
            }
        }
    }
    assert!(
        phase_ratio.is_some(),
        "Y and XZ should have non-zero amplitudes"
    );
}

// --- Commutativity Tests ---

/// Verify that non-commuting gates produce different results when reordered.
pub fn verify_non_commutativity<S: StateVectorSimulator>(sim: &mut S) {
    // H and X do not commute: HX|0⟩ ≠ XH|0⟩
    sim.reset();
    sim.h(&qid(0));
    sim.x(&qid(0));
    let hx_amps: Vec<_> = (0..(1 << sim.num_qubits()))
        .map(|i| sim.get_amplitude(i))
        .collect();

    sim.reset();
    sim.x(&qid(0));
    sim.h(&qid(0));
    let xh_amps: Vec<_> = (0..(1 << sim.num_qubits()))
        .map(|i| sim.get_amplitude(i))
        .collect();

    let differs = hx_amps
        .iter()
        .zip(xh_amps.iter())
        .any(|(a, b)| (a - b).norm() > TOLERANCE);
    assert!(differs, "HX and XH should produce different states");

    // H and Z do not commute: HZ|0⟩ ≠ ZH|0⟩
    // HZ|0⟩ = H(Z|0⟩) = H|0⟩ = |+⟩
    // ZH|0⟩ = Z(H|0⟩) = Z|+⟩ = |−⟩
    sim.reset();
    sim.z(&qid(0));
    sim.h(&qid(0));
    let hz_amps: Vec<_> = (0..(1 << sim.num_qubits()))
        .map(|i| sim.get_amplitude(i))
        .collect();

    sim.reset();
    sim.h(&qid(0));
    sim.z(&qid(0));
    let zh_amps: Vec<_> = (0..(1 << sim.num_qubits()))
        .map(|i| sim.get_amplitude(i))
        .collect();

    let differs = hz_amps
        .iter()
        .zip(zh_amps.iter())
        .any(|(a, b)| (a - b).norm() > TOLERANCE);
    assert!(differs, "HZ and ZH should produce different states");
}

/// Verify that commuting gates produce the same result regardless of order.
pub fn verify_commutativity<S: StateVectorSimulator>(sim: &mut S) {
    if sim.num_qubits() < 2 {
        return;
    }

    // Gates on different qubits commute: H(0)·X(1) = X(1)·H(0)
    sim.reset();
    sim.h(&qid(0));
    sim.x(&qid(1));
    let hx_amps: Vec<_> = (0..(1 << sim.num_qubits()))
        .map(|i| sim.get_amplitude(i))
        .collect();

    sim.reset();
    sim.x(&qid(1));
    sim.h(&qid(0));
    let xh_amps: Vec<_> = (0..(1 << sim.num_qubits()))
        .map(|i| sim.get_amplitude(i))
        .collect();

    for (i, (a, b)) in hx_amps.iter().zip(xh_amps.iter()).enumerate() {
        assert!(
            (a - b).norm() < TOLERANCE,
            "Gates on different qubits should commute: mismatch at index {i}"
        );
    }

    // X and Z on same qubit do not commute: XZ = -ZX
    // On |+⟩ state: XZ|+⟩ ≠ ZX|+⟩
    sim.reset();
    sim.h(&qid(0)); // |+⟩
    sim.x(&qid(0));
    sim.z(&qid(0));
    let xz_amps: Vec<_> = (0..(1 << sim.num_qubits()))
        .map(|i| sim.get_amplitude(i))
        .collect();

    sim.reset();
    sim.h(&qid(0));
    sim.z(&qid(0));
    sim.x(&qid(0));
    let zx_amps: Vec<_> = (0..(1 << sim.num_qubits()))
        .map(|i| sim.get_amplitude(i))
        .collect();

    let differs = xz_amps
        .iter()
        .zip(zx_amps.iter())
        .any(|(a, b)| (a - b).norm() > TOLERANCE);
    assert!(differs, "XZ and ZX should produce different states on |+⟩");
}

// --- Face Gate Tests ---

/// Verify face gates: F·Fdg = I and F^3 = I (cyclic permutation of Paulis).
pub fn verify_face_gates<S: StateVectorSimulator>(sim: &mut S) {
    let n = 1 << sim.num_qubits();

    // F · Fdg = I
    sim.reset();
    sim.h(&qid(0)); // superposition to detect phase
    let before: Vec<_> = (0..n).map(|i| sim.get_amplitude(i)).collect();
    sim.f(&qid(0));
    sim.fdg(&qid(0));
    for (i, amp_before) in before.iter().enumerate() {
        assert_amplitude_eq(
            sim.get_amplitude(i),
            *amp_before,
            &format!("F·Fdg = I at index {i}"),
        );
    }

    // F^3 = I (up to global phase): F cycles X->Y->Z->X, so 3 applications = identity
    sim.reset();
    sim.h(&qid(0));
    let before: Vec<_> = (0..n).map(|i| sim.get_amplitude(i)).collect();
    sim.f(&qid(0));
    sim.f(&qid(0));
    sim.f(&qid(0));
    // Check global phase equivalence
    let mut phase_ratio = None;
    for (i, &bef) in before.iter().enumerate() {
        let after = sim.get_amplitude(i);
        if bef.norm() < TOLERANCE && after.norm() < TOLERANCE {
            continue;
        }
        assert!(
            bef.norm() > TOLERANCE && after.norm() > TOLERANCE,
            "F^3: index {i} zero mismatch"
        );
        let ratio = after / bef;
        match phase_ratio {
            None => phase_ratio = Some(ratio),
            Some(expected) => {
                assert!(
                    (ratio - expected).norm() < TOLERANCE,
                    "F^3: inconsistent phase at index {i}: {ratio:?} vs {expected:?}"
                );
            }
        }
    }
}

/// Verify face gate variants: F2·F2dg = I, F3·F3dg = I, F4·F4dg = I.
pub fn verify_face_gate_adjoints<S: StateVectorSimulator>(sim: &mut S) {
    let n = 1 << sim.num_qubits();
    let q = &qid(0);

    // F2 · F2dg = I, F3 · F3dg = I, F4 · F4dg = I (exact amplitude check)
    let variant_names = ["F2", "F3", "F4"];
    for name in &variant_names {
        sim.reset();
        sim.h(q);
        let before: Vec<_> = (0..n).map(|i| sim.get_amplitude(i)).collect();
        match *name {
            "F2" => {
                sim.f2(q);
                sim.f2dg(q);
            }
            "F3" => {
                sim.f3(q);
                sim.f3dg(q);
            }
            "F4" => {
                sim.f4(q);
                sim.f4dg(q);
            }
            _ => unreachable!(),
        }
        for (i, amp_before) in before.iter().enumerate() {
            assert_amplitude_eq(
                sim.get_amplitude(i),
                *amp_before,
                &format!("{name}·{name}dg = I at index {i}"),
            );
        }
    }
}

// --- Hadamard Variant Tests ---

/// Verify Hadamard variants are involutions: Hi^2 = I (up to global phase).
pub fn verify_hadamard_variants<S: StateVectorSimulator>(sim: &mut S) {
    let n = 1 << sim.num_qubits();
    let q = &qid(0);

    // Each Hadamard variant should be an involution (Hi^2 = I up to global phase).
    // Check that Hi^2 differs from the original state by at most a global phase.
    let variant_names = ["H2", "H3", "H4", "H5", "H6"];
    for name in &variant_names {
        sim.reset();
        sim.h(q); // superposition to detect changes
        let before: Vec<_> = (0..n).map(|i| sim.get_amplitude(i)).collect();
        // Apply twice
        for _ in 0..2 {
            match *name {
                "H2" => {
                    sim.h2(q);
                }
                "H3" => {
                    sim.h3(q);
                }
                "H4" => {
                    sim.h4(q);
                }
                "H5" => {
                    sim.h5(q);
                }
                "H6" => {
                    sim.h6(q);
                }
                _ => unreachable!(),
            }
        }
        // Verify states differ by at most a global phase
        let mut phase_ratio = None;
        for (i, &bef) in before.iter().enumerate() {
            let after = sim.get_amplitude(i);
            if bef.norm() < TOLERANCE && after.norm() < TOLERANCE {
                continue;
            }
            assert!(
                bef.norm() > TOLERANCE && after.norm() > TOLERANCE,
                "{name}^2: index {i} zero mismatch"
            );
            let ratio = after / bef;
            match phase_ratio {
                None => phase_ratio = Some(ratio),
                Some(expected) => {
                    assert!(
                        (ratio - expected).norm() < TOLERANCE,
                        "{name}^2: inconsistent phase at index {i}: {ratio:?} vs {expected:?}"
                    );
                }
            }
        }
        // Verify the phase is unit magnitude (it's a global phase, not scaling)
        if let Some(ratio) = phase_ratio {
            assert!(
                (ratio.norm() - 1.0).abs() < TOLERANCE,
                "{name}^2: phase ratio should have unit magnitude, got {}",
                ratio.norm()
            );
        }
    }
}

/// Verify Hadamard variants produce different states from each other.
pub fn verify_hadamard_variants_distinct<S: StateVectorSimulator>(sim: &mut S) {
    // Apply each variant to |0⟩ and verify they're not all the same
    let mut results = Vec::new();
    let q = &qid(0);
    let variant_names = ["H", "H2", "H3", "H4", "H5", "H6"];

    for name in &variant_names {
        sim.reset();
        match *name {
            "H" => {
                sim.h(q);
            }
            "H2" => {
                sim.h2(q);
            }
            "H3" => {
                sim.h3(q);
            }
            "H4" => {
                sim.h4(q);
            }
            "H5" => {
                sim.h5(q);
            }
            "H6" => {
                sim.h6(q);
            }
            _ => unreachable!(),
        }
        results.push((sim.get_amplitude(0), sim.get_amplitude(1)));
    }

    // At least some should differ from standard H
    let h_result = results[0];
    let some_differ = results[1..].iter().any(|(a0, a1)| {
        (a0 - h_result.0).norm() > TOLERANCE || (a1 - h_result.1).norm() > TOLERANCE
    });
    assert!(
        some_differ,
        "Hadamard variants should differ from standard H"
    );
}

// --- Multi-Gate Sequence / Random Circuit Tests ---

/// Verify normalization is preserved through long gate sequences.
/// Verify that a long gate sequence and its inverse returns to the initial state.
pub fn verify_long_circuit_inverse<S: StateVectorSimulator>(sim: &mut S) {
    if sim.num_qubits() < 3 {
        return;
    }

    let n = 1 << sim.num_qubits();

    sim.reset();
    let before: Vec<_> = (0..n).map(|i| sim.get_amplitude(i)).collect();

    // Forward: 12-gate sequence using varied gate types
    sim.h(&qid(0));
    sim.cx(&qid2(0, 1));
    sim.sz(&qid(2));
    sim.h(&qid(1));
    sim.cz(&qid2(1, 2));
    sim.x(&qid(0));
    sim.swap(&qid2(0, 2));
    sim.h(&qid(1));
    sim.cx(&qid2(2, 0));
    sim.z(&qid(1));
    sim.sx(&qid(0));
    sim.sy(&qid(2));

    // Reverse: each gate's inverse in reverse order
    // SY^-1 = SYdg, SX^-1 = SXdg, Z^-1 = Z, CX^-1 = CX, H^-1 = H,
    // SWAP^-1 = SWAP, X^-1 = X, CZ^-1 = CZ, SZ^-1 = SZdg
    sim.sydg(&qid(2));
    sim.sxdg(&qid(0));
    sim.z(&qid(1));
    sim.cx(&qid2(2, 0));
    sim.h(&qid(1));
    sim.swap(&qid2(0, 2));
    sim.x(&qid(0));
    sim.cz(&qid2(1, 2));
    sim.h(&qid(1));
    sim.szdg(&qid(2));
    sim.cx(&qid2(0, 1));
    sim.h(&qid(0));

    for (i, amp_before) in before.iter().enumerate() {
        assert_amplitude_eq(
            sim.get_amplitude(i),
            *amp_before,
            &format!("Long circuit inverse at index {i}"),
        );
    }
}

/// Verify that applying a circuit and its reverse returns to the initial state.
pub fn verify_circuit_inverse<S: StateVectorSimulator>(sim: &mut S) {
    if sim.num_qubits() < 3 {
        return;
    }

    let n = 1 << sim.num_qubits();

    sim.reset();
    let before: Vec<_> = (0..n).map(|i| sim.get_amplitude(i)).collect();

    // Forward circuit
    sim.h(&qid(0));
    sim.cx(&qid2(0, 1));
    sim.sz(&qid(2));
    sim.h(&qid(1));

    // Reverse circuit (each gate is its own inverse or use adjoint)
    // H^-1 = H, CX^-1 = CX, SZ^-1 = SZdg
    sim.h(&qid(1));
    sim.szdg(&qid(2));
    sim.cx(&qid2(0, 1));
    sim.h(&qid(0));

    for (i, amp_before) in before.iter().enumerate() {
        assert_amplitude_eq(
            sim.get_amplitude(i),
            *amp_before,
            &format!("Circuit inverse returns to initial state at index {i}"),
        );
    }
}

/// Verify that a rotation sequence and its inverse returns to the initial state.
pub fn verify_rotation_circuit_inverse<S: StateVectorSimulator + ArbitraryRotationGateable>(
    sim: &mut S,
) {
    if sim.num_qubits() < 3 {
        return;
    }

    let n = 1 << sim.num_qubits();

    sim.reset();
    let before: Vec<_> = (0..n).map(|i| sim.get_amplitude(i)).collect();

    // Forward rotation sequence
    sim.rx(Angle64::from_radians(0.7), &qid(0));
    sim.ry(Angle64::from_radians(1.3), &qid(1));
    sim.rz(Angle64::from_radians(2.1), &qid(2));
    sim.rzz(Angle64::from_radians(0.5), &qid2(0, 1));
    sim.rxx(Angle64::from_radians(1.1), &qid2(1, 2));
    sim.ry(Angle64::from_radians(0.9), &qid(1));
    sim.rzz(Angle64::from_radians(2.3), &qid2(0, 2));

    // Reverse: R(theta)^-1 = R(-theta)
    sim.rzz(Angle64::from_radians(-2.3), &qid2(0, 2));
    sim.ry(Angle64::from_radians(-0.9), &qid(1));
    sim.rxx(Angle64::from_radians(-1.1), &qid2(1, 2));
    sim.rzz(Angle64::from_radians(-0.5), &qid2(0, 1));
    sim.rz(Angle64::from_radians(-2.1), &qid(2));
    sim.ry(Angle64::from_radians(-1.3), &qid(1));
    sim.rx(Angle64::from_radians(-0.7), &qid(0));

    for (i, amp_before) in before.iter().enumerate() {
        assert_amplitude_eq(
            sim.get_amplitude(i),
            *amp_before,
            &format!("Rotation circuit inverse at index {i}"),
        );
    }
}

// --- Suite Runner Functions ---

/// Run basic state vector tests.
pub fn run_basic_state_vector_test_suite<S: StateVectorSimulator>(sim: &mut S) {
    verify_initial_state(sim);
    verify_reset(sim);
    verify_normalization(sim);
    verify_probability(sim);
    verify_prepare_all_basis_states(sim);
    verify_unitarity(sim);
    verify_unitarity_two_qubit(sim);
}

/// Run Clifford gate tests.
pub fn run_clifford_test_suite<S: StateVectorSimulator>(sim: &mut S) {
    // Shared measurement-based Clifford tests (gate identities, adjoint pairs,
    // face gates, Hadamard variants, entanglement, decompositions, and more).
    let num_qubits = sim.num_qubits();
    crate::clifford_test_utils::run_clifford_gate_tests(sim, num_qubits);

    // Amplitude-specific tests below test exact gate matrix values.

    // Single-qubit gates
    verify_x_gate(sim);
    verify_y_gate(sim);
    verify_z_gate(sim);
    verify_h_gate(sim);
    verify_sz_gate(sim);
    verify_sx_gate(sim);
    verify_sy_gate(sim);

    // Two-qubit gates
    verify_cx_gate(sim);
    verify_cy_gate(sim);
    verify_cz_gate(sim);
    verify_swap_gate(sim);
    verify_iswap_gate(sim);

    // Gate identities and decompositions
    verify_gate_identities(sim);
    verify_gate_decompositions(sim);
    verify_swap_decomposition(sim);
    verify_x_hzh_decomposition(sim);
    verify_z_from_sz(sim);
    verify_y_xz_decomposition(sim);

    // Adjoint gates
    verify_adjoint_gates(sim);
    verify_adjoint_two_qubit_gates(sim);

    // Locality
    verify_single_qubit_locality(sim);
    verify_two_qubit_locality(sim);

    // State preparation
    verify_bell_state_preparation(sim);
    verify_ghz_state(sim);

    // Commutativity
    verify_non_commutativity(sim);
    verify_commutativity(sim);

    // Face gates
    verify_face_gates(sim);
    verify_face_gate_adjoints(sim);

    // Hadamard variants
    verify_hadamard_variants(sim);
    verify_hadamard_variants_distinct(sim);

    // Multi-gate sequences
    verify_long_circuit_inverse(sim);
    verify_circuit_inverse(sim);

    // Batch operations
    verify_batch_single_qubit_gates(sim);
    verify_batch_two_qubit_gates(sim);
    verify_batch_measurements(sim);
}

/// Run rotation gate tests.
pub fn run_rotation_test_suite<S: StateVectorSimulator + ArbitraryRotationGateable>(sim: &mut S) {
    // Shared measurement-based rotation tests (Clifford-angle equivalences, inverse
    // rotations, T gate, composition, two-qubit rotations, and more).
    let num_qubits = sim.num_qubits();
    crate::rotation_test_utils::run_rotation_gate_tests(sim, num_qubits);

    // Amplitude-specific rotation tests below.
    verify_rx_gate(sim);
    verify_ry_gate(sim);
    verify_rz_gate(sim);
    verify_rxx_gate(sim);
    verify_ryy_gate(sim);
    verify_rzz_gate(sim);
    verify_rotation_identities(sim);
    verify_u_gate(sim);
    verify_r1xy_gate(sim);
    verify_single_qubit_rotation(sim);

    // State preparation requiring rotations
    verify_equal_superposition(sim);

    // Normalization after rotation sequences
    verify_rotation_circuit_inverse(sim);

    // Batch rotation operations
    verify_batch_rotation_gates(sim);
    verify_batch_two_qubit_rotation_gates(sim);
}

/// Run measurement tests.
pub fn run_measurement_test_suite<S: StateVectorSimulator>(sim: &mut S) {
    verify_deterministic_measurement(sim);
    verify_measurement_collapse(sim);
    // verify_measurement_idempotence, verify_bell_state_correlation, and
    // verify_measurement_statistics are covered by the shared Clifford suite
    // (called from run_clifford_test_suite).
    verify_pz(sim);
    verify_pnz(sim);
    verify_pz_multiple_qubits(sim);
    verify_measurement_consistency(sim);
    verify_mz_detailed(sim);
}

/// Run the full state vector test suite.
pub fn run_full_state_vector_test_suite<S: StateVectorSimulator + ArbitraryRotationGateable>(
    sim: &mut S,
) {
    run_basic_state_vector_test_suite(sim);
    run_clifford_test_suite(sim);
    run_rotation_test_suite(sim);
    run_measurement_test_suite(sim);
}

// --- Trait Implementations ---

use crate::{SparseStateVecAoS, SparseStateVecSoA, StateVecAoS, StateVecSoA};

impl StateVectorSimulator for StateVecAoS {
    fn with_seed(num_qubits: usize, seed: u64) -> Self {
        StateVecAoS::with_seed(num_qubits, seed)
    }

    fn get_amplitude(&mut self, basis_state: usize) -> Complex64 {
        self.state()[basis_state]
    }

    fn num_qubits(&self) -> usize {
        StateVecAoS::num_qubits(self)
    }
}

impl StateVectorSimulator for StateVecSoA {
    fn with_seed(num_qubits: usize, seed: u64) -> Self {
        let mut sim = StateVecSoA::new(num_qubits);
        sim.set_seed(seed);
        sim
    }

    fn get_amplitude(&mut self, basis_state: usize) -> Complex64 {
        StateVecSoA::get_amplitude(self, basis_state)
    }

    fn num_qubits(&self) -> usize {
        StateVecSoA::num_qubits(self)
    }
}

impl StateVectorSimulator for SparseStateVecAoS {
    fn with_seed(num_qubits: usize, seed: u64) -> Self {
        SparseStateVecAoS::with_seed(num_qubits, seed)
    }

    fn get_amplitude(&mut self, basis_state: usize) -> Complex64 {
        SparseStateVecAoS::get_amplitude(self, basis_state)
    }

    fn num_qubits(&self) -> usize {
        SparseStateVecAoS::num_qubits(self)
    }
}

impl StateVectorSimulator for SparseStateVecSoA {
    fn with_seed(num_qubits: usize, seed: u64) -> Self {
        SparseStateVecSoA::with_seed(num_qubits, seed)
    }

    fn get_amplitude(&mut self, basis_state: usize) -> Complex64 {
        SparseStateVecSoA::get_amplitude(self, basis_state)
    }

    fn num_qubits(&self) -> usize {
        SparseStateVecSoA::num_qubits(self)
    }
}

// --- Module Tests ---

#[cfg(test)]
mod tests {
    use crate::{SparseStateVecAoS, SparseStateVecSoA, StateVecAoS, StateVecSoA};

    // Dense simulators with rotation gate support
    full_state_vector_test_suite!(StateVecAoS, 4);
    full_state_vector_test_suite!(StateVecSoA, 4);

    // Sparse simulators (now with rotation gate support)
    full_state_vector_test_suite!(SparseStateVecAoS, 4);
    full_state_vector_test_suite!(SparseStateVecSoA, 4);
}
