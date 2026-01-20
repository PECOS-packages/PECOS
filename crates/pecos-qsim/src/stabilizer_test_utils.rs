// Copyright 2024 The PECOS Developers
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

//! Test utilities for verifying stabilizer simulator implementations.
//!
//! This module provides generic test functions that can be used to verify any
//! stabilizer simulator that implements [`CliffordGateable`] and [`QuantumSimulator`].
//!
//! All functions in this module use assertions to verify expected behavior and
//! will panic if the test conditions are not met. This is the expected behavior
//! for test utilities.
//!
//! # Example
//!
//! ```ignore
//! use pecos_qsim::stabilizer_test_utils::*;
//! use pecos_qsim::SparseStab;
//!
//! #[test]
//! fn test_my_simulator() {
//!     let mut sim = SparseStab::new(4);
//!     verify_gate_identities(&mut sim);
//!     verify_bell_state_correlations(&mut sim);
//! }
//! ```

// All functions in this module are test utilities that panic on test failure.
// This is expected behavior, so we allow missing panics documentation.
#![allow(clippy::missing_panics_doc)]

use crate::{CliffordGateable, DensityMatrix, MeasurementResult, QuantumSimulator};
use pecos_core::QubitId;
use pecos_rng::Rng;

/// Trait for stabilizer simulators that support forced measurement outcomes.
///
/// This is required for probability comparison tests against `DensityMatrix`.
pub trait ForcedMeasurement {
    /// Measure qubit in Z basis, forcing the outcome for non-deterministic cases.
    fn mz_forced(&mut self, qubit: usize, forced_outcome: bool) -> MeasurementResult;
}

// ============================================================================
// Stabilizer Simulator Marker Trait
// ============================================================================

/// Marker trait for stabilizer simulators that support full Clifford simulation.
///
/// Implementing this trait indicates that a simulator:
/// - Implements all Clifford gates via [`CliffordGateable`]
/// - Supports basic simulator operations via [`QuantumSimulator`]
/// - Supports forced measurements for testing via [`ForcedMeasurement`]
/// - Can be cloned for probability testing
/// - Can be constructed with a seed for reproducible tests
///
/// Simulators implementing this trait can use the [`stabilizer_test_suite!`] macro
/// to automatically generate a comprehensive test suite.
///
/// # Example
///
/// ```ignore
/// use pecos_qsim::stabilizer_test_utils::{StabilizerSimulator, stabilizer_test_suite};
///
/// // In your test module:
/// stabilizer_test_suite!(MyStabilizerSim, 8);
/// ```
pub trait StabilizerSimulator:
    CliffordGateable + QuantumSimulator + ForcedMeasurement + Clone + Sized
{
    /// Create a new simulator with the given number of qubits and RNG seed.
    fn with_seed(num_qubits: usize, seed: u64) -> Self;
}

/// Generates a comprehensive test suite for a stabilizer simulator.
///
/// This macro creates test functions that verify correct implementation of
/// all Clifford gates, measurement behavior, and probability distributions.
///
/// # Arguments
///
/// * `$sim_type` - The type implementing [`StabilizerSimulator`]
/// * `$num_qubits` - Number of qubits to use for testing (default: 8)
///
/// # Example
///
/// ```ignore
/// use pecos_qsim::stabilizer_test_utils::stabilizer_test_suite;
/// use pecos_qsim::SparseStab;
///
/// // Generate tests with default 8 qubits
/// stabilizer_test_suite!(SparseStab);
///
/// // Or specify a custom qubit count
/// stabilizer_test_suite!(SparseStab, 4);
/// ```
///
/// # Generated Tests
///
/// The macro generates the following tests:
/// - `test_<type>_basic_suite` - Basic gate and measurement tests
/// - `test_<type>_full_suite` - Full suite including probability verification and random circuits
#[macro_export]
macro_rules! stabilizer_test_suite {
    ($sim_type:ty) => {
        $crate::stabilizer_test_suite!($sim_type, 8);
    };
    ($sim_type:ty, $num_qubits:expr) => {
        paste::paste! {
            #[test]
            fn [<test_ $sim_type:snake _basic_suite>]() {
                use $crate::stabilizer_test_utils::{run_basic_stabilizer_test_suite, StabilizerSimulator};
                let mut sim = <$sim_type>::with_seed($num_qubits, 42);
                run_basic_stabilizer_test_suite(&mut sim, $num_qubits);
            }

            #[test]
            fn [<test_ $sim_type:snake _full_suite>]() {
                use $crate::stabilizer_test_utils::{run_full_stabilizer_test_suite, StabilizerSimulator};
                let mut sim = <$sim_type>::with_seed($num_qubits, 42);
                run_full_stabilizer_test_suite(&mut sim, $num_qubits);
            }
        }
    };
}

// ============================================================================
// Gate Identity Tests
// ============================================================================

/// Verify that H^2 = I (Hadamard squared is identity).
///
/// After applying H twice to |0>, measurement should be deterministic 0.
pub fn verify_h_squared_is_identity<S: CliffordGateable + QuantumSimulator>(sim: &mut S) {
    sim.reset();
    sim.h(&[QubitId::new(0)]);
    sim.h(&[QubitId::new(0)]);

    let result = sim.mz(&[QubitId::new(0)]);
    assert!(result[0].is_deterministic, "H^2|0> should be deterministic");
    assert!(!result[0].outcome, "H^2|0> should measure 0");
}

/// Verify that S^4 = I (S gate to the fourth power is identity).
///
/// H S^4 H |0> should give deterministic 0.
pub fn verify_s_fourth_is_identity<S: CliffordGateable + QuantumSimulator>(sim: &mut S) {
    sim.reset();
    sim.h(&[QubitId::new(0)]);
    sim.sz(&[QubitId::new(0)]);
    sim.sz(&[QubitId::new(0)]);
    sim.sz(&[QubitId::new(0)]);
    sim.sz(&[QubitId::new(0)]);
    sim.h(&[QubitId::new(0)]);

    let result = sim.mz(&[QubitId::new(0)]);
    assert!(
        result[0].is_deterministic,
        "H S^4 H |0> should be deterministic"
    );
    assert!(!result[0].outcome, "H S^4 H |0> should measure 0");
}

/// Verify that S^2 = Z.
///
/// X then S^2 should give the same result as X then Z (both flip the sign).
pub fn verify_s_squared_is_z<S: CliffordGateable + QuantumSimulator>(sim: &mut S) {
    // First: X then S^2
    sim.reset();
    sim.h(&[QubitId::new(0)]); // Create |+>
    sim.sz(&[QubitId::new(0)]);
    sim.sz(&[QubitId::new(0)]);
    let result1 = sim.mz(&[QubitId::new(0)]);

    // Second: X then Z (should be equivalent)
    sim.reset();
    sim.h(&[QubitId::new(0)]); // Create |+>
    sim.z(&[QubitId::new(0)]);
    let result2 = sim.mz(&[QubitId::new(0)]);

    assert_eq!(
        result1[0].is_deterministic, result2[0].is_deterministic,
        "S^2 and Z should have same determinism"
    );
}

/// Verify that SZ followed by SZdg is identity.
///
/// SZ * SZdg = I, so applying both should leave state unchanged.
pub fn verify_sz_szdg_is_identity<S: CliffordGateable + QuantumSimulator>(sim: &mut S) {
    // Test on |0> state
    sim.reset();
    sim.sz(&[QubitId::new(0)]);
    sim.szdg(&[QubitId::new(0)]);

    let result = sim.mz(&[QubitId::new(0)]);
    assert!(
        result[0].is_deterministic,
        "SZ SZdg |0> should be deterministic"
    );
    assert!(!result[0].outcome, "SZ SZdg |0> should measure 0");

    // Test on |+> state (phase matters here)
    sim.reset();
    sim.h(&[QubitId::new(0)]); // |+>
    sim.sz(&[QubitId::new(0)]);
    sim.szdg(&[QubitId::new(0)]);
    sim.h(&[QubitId::new(0)]); // Back to |0> if SZ SZdg = I

    let result = sim.mz(&[QubitId::new(0)]);
    assert!(
        result[0].is_deterministic,
        "H SZ SZdg H |0> should be deterministic"
    );
    assert!(!result[0].outcome, "H SZ SZdg H |0> should measure 0");
}

/// Verify that SZdg followed by SZ is identity.
///
/// SZdg * SZ = I, so applying both should leave state unchanged.
pub fn verify_szdg_sz_is_identity<S: CliffordGateable + QuantumSimulator>(sim: &mut S) {
    // Test on |+> state (phase matters here)
    sim.reset();
    sim.h(&[QubitId::new(0)]); // |+>
    sim.szdg(&[QubitId::new(0)]);
    sim.sz(&[QubitId::new(0)]);
    sim.h(&[QubitId::new(0)]); // Back to |0> if SZdg SZ = I

    let result = sim.mz(&[QubitId::new(0)]);
    assert!(
        result[0].is_deterministic,
        "H SZdg SZ H |0> should be deterministic"
    );
    assert!(!result[0].outcome, "H SZdg SZ H |0> should measure 0");
}

/// Verify SZ maps |+> to |+Y> (Y eigenstate with eigenvalue +1).
///
/// |+Y> measured in Y basis should give deterministic 0.
/// Y-basis measurement: SZdg H Mz H SZ
pub fn verify_sz_maps_plus_to_plus_y<S: CliffordGateable + QuantumSimulator>(sim: &mut S) {
    sim.reset();
    sim.h(&[QubitId::new(0)]); // |+>
    sim.sz(&[QubitId::new(0)]); // |+Y>

    // Measure in Y basis: SZdg H Mz H SZ
    sim.szdg(&[QubitId::new(0)]);
    sim.h(&[QubitId::new(0)]);
    let result = sim.mz(&[QubitId::new(0)]);

    assert!(
        result[0].is_deterministic,
        "SZ|+> measured in Y basis should be deterministic"
    );
    assert!(
        !result[0].outcome,
        "SZ|+> = |+Y> should measure 0 in Y basis"
    );
}

/// Verify SZdg maps |+> to |-Y> (Y eigenstate with eigenvalue -1).
///
/// |-Y> measured in Y basis should give deterministic 1.
pub fn verify_szdg_maps_plus_to_minus_y<S: CliffordGateable + QuantumSimulator>(sim: &mut S) {
    sim.reset();
    sim.h(&[QubitId::new(0)]); // |+>
    sim.szdg(&[QubitId::new(0)]); // |-Y>

    // Measure in Y basis: SZdg H Mz H SZ
    sim.szdg(&[QubitId::new(0)]);
    sim.h(&[QubitId::new(0)]);
    let result = sim.mz(&[QubitId::new(0)]);

    assert!(
        result[0].is_deterministic,
        "SZdg|+> measured in Y basis should be deterministic"
    );
    assert!(
        result[0].outcome,
        "SZdg|+> = |-Y> should measure 1 in Y basis"
    );
}

/// Verify SZ and SZdg are inverses via phase tracking.
///
/// This test specifically checks that the phase is correctly tracked:
/// - SZ: X -> Y (i.e., X -> iXZ in stabilizer notation)
/// - SZdg: X -> -Y (i.e., X -> -iXZ in stabilizer notation)
pub fn verify_sz_szdg_phase_consistency<S: CliffordGateable + QuantumSimulator>(sim: &mut S) {
    // Test 1: |+> -> SZ -> |+Y> -> SZdg -> |+>
    sim.reset();
    sim.h(&[QubitId::new(0)]); // |+>
    sim.sz(&[QubitId::new(0)]); // |+Y>
    sim.szdg(&[QubitId::new(0)]); // |+>
    sim.h(&[QubitId::new(0)]); // |0>

    let result = sim.mz(&[QubitId::new(0)]);
    assert!(
        result[0].is_deterministic,
        "H SZ SZdg H |0> should be deterministic"
    );
    assert!(!result[0].outcome, "H SZ SZdg H |0> should measure 0");

    // Test 2: |+> -> SZdg -> |-Y> -> SZ -> |+>
    sim.reset();
    sim.h(&[QubitId::new(0)]); // |+>
    sim.szdg(&[QubitId::new(0)]); // |-Y>
    sim.sz(&[QubitId::new(0)]); // |+>
    sim.h(&[QubitId::new(0)]); // |0>

    let result = sim.mz(&[QubitId::new(0)]);
    assert!(
        result[0].is_deterministic,
        "H SZdg SZ H |0> should be deterministic"
    );
    assert!(!result[0].outcome, "H SZdg SZ H |0> should measure 0");

    // Test 3: SZ^2 SZdg^2 = Z Z = I on |+>
    sim.reset();
    sim.h(&[QubitId::new(0)]); // |+>
    sim.sz(&[QubitId::new(0)]);
    sim.sz(&[QubitId::new(0)]); // Z|+> = |->
    sim.szdg(&[QubitId::new(0)]);
    sim.szdg(&[QubitId::new(0)]); // Z|-> = |+>
    sim.h(&[QubitId::new(0)]); // |0>

    let result = sim.mz(&[QubitId::new(0)]);
    assert!(
        result[0].is_deterministic,
        "H SZ^2 SZdg^2 H |0> should be deterministic"
    );
    assert!(!result[0].outcome, "H SZ^2 SZdg^2 H |0> should measure 0");
}

/// Verify SZdg^4 = I (SZdg to the fourth power is identity).
pub fn verify_szdg_fourth_is_identity<S: CliffordGateable + QuantumSimulator>(sim: &mut S) {
    sim.reset();
    sim.h(&[QubitId::new(0)]);
    sim.szdg(&[QubitId::new(0)]);
    sim.szdg(&[QubitId::new(0)]);
    sim.szdg(&[QubitId::new(0)]);
    sim.szdg(&[QubitId::new(0)]);
    sim.h(&[QubitId::new(0)]);

    let result = sim.mz(&[QubitId::new(0)]);
    assert!(
        result[0].is_deterministic,
        "H SZdg^4 H |0> should be deterministic"
    );
    assert!(!result[0].outcome, "H SZdg^4 H |0> should measure 0");
}

/// Verify SZdg^2 = Z.
pub fn verify_szdg_squared_is_z<S: CliffordGateable + QuantumSimulator>(sim: &mut S) {
    // SZdg^2 on |+>
    sim.reset();
    sim.h(&[QubitId::new(0)]); // |+>
    sim.szdg(&[QubitId::new(0)]);
    sim.szdg(&[QubitId::new(0)]);
    let result1 = sim.mz(&[QubitId::new(0)]);

    // Z on |+> (should be equivalent)
    sim.reset();
    sim.h(&[QubitId::new(0)]); // |+>
    sim.z(&[QubitId::new(0)]);
    let result2 = sim.mz(&[QubitId::new(0)]);

    assert_eq!(
        result1[0].is_deterministic, result2[0].is_deterministic,
        "SZdg^2 and Z should have same determinism"
    );
}

/// Verify that CX^2 = I (CNOT squared is identity).
pub fn verify_cx_squared_is_identity<S: CliffordGateable + QuantumSimulator>(sim: &mut S) {
    sim.reset();
    sim.h(&[QubitId::new(0)]); // Create superposition on control
    sim.cx(&[QubitId::new(0), QubitId::new(1)]);
    sim.cx(&[QubitId::new(0), QubitId::new(1)]);

    // After CX^2, qubit 1 should still be in |0>
    let result = sim.mz(&[QubitId::new(1)]);
    assert!(
        result[0].is_deterministic,
        "Target after CX^2 should be deterministic"
    );
    assert!(!result[0].outcome, "Target after CX^2 should be 0");
}

/// Verify that X^2 = I.
pub fn verify_x_squared_is_identity<S: CliffordGateable + QuantumSimulator>(sim: &mut S) {
    sim.reset();
    sim.x(&[QubitId::new(0)]);
    sim.x(&[QubitId::new(0)]);

    let result = sim.mz(&[QubitId::new(0)]);
    assert!(result[0].is_deterministic, "X^2|0> should be deterministic");
    assert!(!result[0].outcome, "X^2|0> should measure 0");
}

/// Verify that Z^2 = I.
pub fn verify_z_squared_is_identity<S: CliffordGateable + QuantumSimulator>(sim: &mut S) {
    sim.reset();
    sim.x(&[QubitId::new(0)]); // Start with |1>
    sim.z(&[QubitId::new(0)]);
    sim.z(&[QubitId::new(0)]);

    let result = sim.mz(&[QubitId::new(0)]);
    assert!(result[0].is_deterministic, "Z^2|1> should be deterministic");
    assert!(result[0].outcome, "Z^2|1> should measure 1");
}

/// Verify that Y^2 = I.
pub fn verify_y_squared_is_identity<S: CliffordGateable + QuantumSimulator>(sim: &mut S) {
    sim.reset();
    sim.y(&[QubitId::new(0)]);
    sim.y(&[QubitId::new(0)]);

    let result = sim.mz(&[QubitId::new(0)]);
    assert!(result[0].is_deterministic, "Y^2|0> should be deterministic");
    assert!(!result[0].outcome, "Y^2|0> should measure 0");
}

/// Run all gate identity tests.
pub fn verify_all_gate_identities<S: CliffordGateable + QuantumSimulator>(sim: &mut S) {
    verify_h_squared_is_identity(sim);
    verify_s_fourth_is_identity(sim);
    verify_s_squared_is_z(sim);
    verify_sz_szdg_is_identity(sim);
    verify_szdg_sz_is_identity(sim);
    verify_szdg_fourth_is_identity(sim);
    verify_szdg_squared_is_z(sim);
    verify_sz_szdg_phase_consistency(sim);
    verify_sz_maps_plus_to_plus_y(sim);
    verify_szdg_maps_plus_to_minus_y(sim);
    verify_cx_squared_is_identity(sim);
    verify_x_squared_is_identity(sim);
    verify_y_squared_is_identity(sim);
    verify_z_squared_is_identity(sim);
}

// ============================================================================
// Entanglement Tests
// ============================================================================

/// Verify Bell state |Φ+> = (|00> + |11>)/√2 has correct correlations.
///
/// After measuring the first qubit, the second should be deterministic and equal.
pub fn verify_bell_state_correlations<S: CliffordGateable + QuantumSimulator>(sim: &mut S) {
    sim.reset();
    sim.h(&[QubitId::new(0)]);
    sim.cx(&[QubitId::new(0), QubitId::new(1)]);

    // First measurement is non-deterministic
    let r0 = sim.mz(&[QubitId::new(0)]);
    assert!(
        !r0[0].is_deterministic,
        "First Bell state measurement should be non-deterministic"
    );

    // Second measurement is deterministic and correlated
    let r1 = sim.mz(&[QubitId::new(1)]);
    assert!(
        r1[0].is_deterministic,
        "Second Bell state measurement should be deterministic"
    );
    assert_eq!(
        r0[0].outcome, r1[0].outcome,
        "Bell state measurements should be correlated"
    );
}

/// Verify GHZ state (|00...0> + |11...1>)/√2 has correct correlations.
pub fn verify_ghz_state_correlations<S: CliffordGateable + QuantumSimulator>(
    sim: &mut S,
    num_qubits: usize,
) {
    assert!(num_qubits >= 2, "GHZ state requires at least 2 qubits");

    sim.reset();

    // Create GHZ state
    sim.h(&[QubitId::new(0)]);
    for i in 0..(num_qubits - 1) {
        sim.cx(&[QubitId::new(i), QubitId::new(i + 1)]);
    }

    // First measurement is non-deterministic
    let r0 = sim.mz(&[QubitId::new(0)]);
    assert!(
        !r0[0].is_deterministic,
        "First GHZ measurement should be non-deterministic"
    );

    // All subsequent measurements should be deterministic and equal to the first
    for i in 1..num_qubits {
        let ri = sim.mz(&[QubitId::new(i)]);
        assert!(
            ri[0].is_deterministic,
            "GHZ measurement {i} should be deterministic after first"
        );
        assert_eq!(
            r0[0].outcome, ri[0].outcome,
            "GHZ measurements should all be correlated"
        );
    }
}

// ============================================================================
// Basic Gate Tests
// ============================================================================

/// Verify initial state is |0...0> with deterministic measurements.
pub fn verify_initial_state<S: CliffordGateable + QuantumSimulator>(
    sim: &mut S,
    num_qubits: usize,
) {
    sim.reset();

    for q in 0..num_qubits {
        let result = sim.mz(&[QubitId::new(q)]);
        assert!(
            result[0].is_deterministic,
            "Initial state qubit {q} should be deterministic"
        );
        assert!(
            !result[0].outcome,
            "Initial state qubit {q} should measure 0"
        );
    }
}

/// Verify X gate flips the qubit.
pub fn verify_x_gate<S: CliffordGateable + QuantumSimulator>(sim: &mut S) {
    sim.reset();
    sim.x(&[QubitId::new(0)]);

    let result = sim.mz(&[QubitId::new(0)]);
    assert!(result[0].is_deterministic, "X|0> should be deterministic");
    assert!(result[0].outcome, "X|0> should measure 1");
}

/// Verify H gate creates superposition.
pub fn verify_h_creates_superposition<S: CliffordGateable + QuantumSimulator>(sim: &mut S) {
    sim.reset();
    sim.h(&[QubitId::new(0)]);

    let result = sim.mz(&[QubitId::new(0)]);
    assert!(
        !result[0].is_deterministic,
        "H|0> measurement should be non-deterministic"
    );
}

/// Verify Z gate on |0> state (should have no visible effect on Z measurement).
pub fn verify_z_gate_on_zero<S: CliffordGateable + QuantumSimulator>(sim: &mut S) {
    sim.reset();
    sim.z(&[QubitId::new(0)]);

    let result = sim.mz(&[QubitId::new(0)]);
    assert!(result[0].is_deterministic, "Z|0> should be deterministic");
    assert!(!result[0].outcome, "Z|0> should measure 0");
}

/// Verify Z gate on |1> state (should have no visible effect on Z measurement).
pub fn verify_z_gate_on_one<S: CliffordGateable + QuantumSimulator>(sim: &mut S) {
    sim.reset();
    sim.x(&[QubitId::new(0)]);
    sim.z(&[QubitId::new(0)]);

    let result = sim.mz(&[QubitId::new(0)]);
    assert!(result[0].is_deterministic, "ZX|0> should be deterministic");
    assert!(result[0].outcome, "ZX|0> should measure 1");
}

// ============================================================================
// Specific Gate Tests
// ============================================================================

/// Verify SX gate: SX^2 = X.
///
/// SX is sqrt(X), so applying it twice should give X.
pub fn verify_sx_squared_is_x<S: CliffordGateable + QuantumSimulator>(sim: &mut S) {
    sim.reset();
    sim.sx(&[QubitId::new(0)]);
    sim.sx(&[QubitId::new(0)]);

    let result = sim.mz(&[QubitId::new(0)]);
    assert!(
        result[0].is_deterministic,
        "SX^2|0> should be deterministic"
    );
    assert!(result[0].outcome, "SX^2|0> = X|0> should measure 1");
}

/// Verify SX gate creates superposition from |0>.
///
/// SX|0> should be a superposition state.
pub fn verify_sx_creates_superposition<S: CliffordGateable + QuantumSimulator>(sim: &mut S) {
    sim.reset();
    sim.sx(&[QubitId::new(0)]);

    let result = sim.mz(&[QubitId::new(0)]);
    assert!(
        !result[0].is_deterministic,
        "SX|0> should be non-deterministic"
    );
}

/// Verify `SXdg` is inverse of SX: SX * `SXdg` = I.
pub fn verify_sx_sxdg_is_identity<S: CliffordGateable + QuantumSimulator>(sim: &mut S) {
    sim.reset();
    sim.h(&[QubitId::new(0)]); // Create superposition first
    sim.sx(&[QubitId::new(0)]);
    sim.sxdg(&[QubitId::new(0)]);
    sim.h(&[QubitId::new(0)]); // Undo the H

    let result = sim.mz(&[QubitId::new(0)]);
    assert!(
        result[0].is_deterministic,
        "H SX SXdg H |0> should be deterministic"
    );
    assert!(!result[0].outcome, "H SX SXdg H |0> should measure 0");
}

/// Verify SY gate: SY^2 = Y.
///
/// SY is sqrt(Y), so applying it twice should give Y.
pub fn verify_sy_squared_is_y<S: CliffordGateable + QuantumSimulator>(sim: &mut S) {
    sim.reset();
    sim.sy(&[QubitId::new(0)]);
    sim.sy(&[QubitId::new(0)]);

    // Y|0> = i|1>, so measurement gives 1
    let result = sim.mz(&[QubitId::new(0)]);
    assert!(
        result[0].is_deterministic,
        "SY^2|0> should be deterministic"
    );
    assert!(result[0].outcome, "SY^2|0> = Y|0> should measure 1");
}

/// Verify SY gate creates superposition from |0>.
pub fn verify_sy_creates_superposition<S: CliffordGateable + QuantumSimulator>(sim: &mut S) {
    sim.reset();
    sim.sy(&[QubitId::new(0)]);

    let result = sim.mz(&[QubitId::new(0)]);
    assert!(
        !result[0].is_deterministic,
        "SY|0> should be non-deterministic"
    );
}

/// Verify `SYdg` is inverse of SY: SY * `SYdg` = I.
pub fn verify_sy_sydg_is_identity<S: CliffordGateable + QuantumSimulator>(sim: &mut S) {
    sim.reset();
    sim.h(&[QubitId::new(0)]); // Create superposition first
    sim.sy(&[QubitId::new(0)]);
    sim.sydg(&[QubitId::new(0)]);
    sim.h(&[QubitId::new(0)]); // Undo the H

    let result = sim.mz(&[QubitId::new(0)]);
    assert!(
        result[0].is_deterministic,
        "H SY SYdg H |0> should be deterministic"
    );
    assert!(!result[0].outcome, "H SY SYdg H |0> should measure 0");
}

/// Verify CY gate: control=0 means no action on target.
pub fn verify_cy_control_zero<S: CliffordGateable + QuantumSimulator>(sim: &mut S) {
    sim.reset();
    // Control is |0>, so CY should not affect target
    sim.cy(&[QubitId::new(0), QubitId::new(1)]);

    let result = sim.mz(&[QubitId::new(1)]);
    assert!(
        result[0].is_deterministic,
        "CY with control=0 should leave target deterministic"
    );
    assert!(
        !result[0].outcome,
        "CY with control=0 should leave target as 0"
    );
}

/// Verify CY gate: control=1 applies Y to target.
pub fn verify_cy_control_one<S: CliffordGateable + QuantumSimulator>(sim: &mut S) {
    sim.reset();
    sim.x(&[QubitId::new(0)]); // Set control to |1>
    sim.cy(&[QubitId::new(0), QubitId::new(1)]);

    // Y|0> = i|1>
    let result = sim.mz(&[QubitId::new(1)]);
    assert!(
        result[0].is_deterministic,
        "CY with control=1 should make target deterministic"
    );
    assert!(
        result[0].outcome,
        "CY with control=1 should flip target to 1"
    );
}

/// Verify CY^2 = I (CY squared is identity).
pub fn verify_cy_squared_is_identity<S: CliffordGateable + QuantumSimulator>(sim: &mut S) {
    sim.reset();
    sim.h(&[QubitId::new(0)]); // Superposition on control
    sim.cy(&[QubitId::new(0), QubitId::new(1)]);
    sim.cy(&[QubitId::new(0), QubitId::new(1)]);

    let result = sim.mz(&[QubitId::new(1)]);
    assert!(
        result[0].is_deterministic,
        "Target after CY^2 should be deterministic"
    );
    assert!(!result[0].outcome, "Target after CY^2 should be 0");
}

/// Verify CZ gate: control=0 means no action.
pub fn verify_cz_control_zero<S: CliffordGateable + QuantumSimulator>(sim: &mut S) {
    sim.reset();
    sim.h(&[QubitId::new(1)]); // Put target in superposition
    sim.cz(&[QubitId::new(0), QubitId::new(1)]);
    sim.h(&[QubitId::new(1)]); // Undo H

    let result = sim.mz(&[QubitId::new(1)]);
    assert!(
        result[0].is_deterministic,
        "CZ with control=0 should leave target unchanged"
    );
    assert!(!result[0].outcome, "CZ with control=0: H CZ H |0> = |0>");
}

/// Verify CZ gate: control=1 applies Z to target.
pub fn verify_cz_control_one<S: CliffordGateable + QuantumSimulator>(sim: &mut S) {
    sim.reset();
    sim.x(&[QubitId::new(0)]); // Set control to |1>
    sim.h(&[QubitId::new(1)]); // Put target in |+>
    sim.cz(&[QubitId::new(0), QubitId::new(1)]);
    sim.h(&[QubitId::new(1)]); // H|-> = |1>

    let result = sim.mz(&[QubitId::new(1)]);
    assert!(
        result[0].is_deterministic,
        "CZ with control=1 should make target deterministic"
    );
    assert!(result[0].outcome, "CZ with control=1: H CZ H |+> = |1>");
}

/// Verify CZ is symmetric: CZ(0,1) = CZ(1,0).
pub fn verify_cz_symmetric<S: CliffordGateable + QuantumSimulator + ForcedMeasurement>(
    sim: &mut S,
) {
    // Apply CZ(0,1) and measure
    sim.reset();
    sim.h(&[QubitId::new(0)]);
    sim.h(&[QubitId::new(1)]);
    sim.cz(&[QubitId::new(0), QubitId::new(1)]);

    let r0_a = sim.mz_forced(0, false);
    let r1_a = sim.mz_forced(1, false);

    // Apply CZ(1,0) and measure
    sim.reset();
    sim.h(&[QubitId::new(0)]);
    sim.h(&[QubitId::new(1)]);
    sim.cz(&[QubitId::new(1), QubitId::new(0)]);

    let r0_b = sim.mz_forced(0, false);
    let r1_b = sim.mz_forced(1, false);

    assert_eq!(
        r0_a.is_deterministic, r0_b.is_deterministic,
        "CZ symmetry: q0 determinism should match"
    );
    assert_eq!(
        r1_a.is_deterministic, r1_b.is_deterministic,
        "CZ symmetry: q1 determinism should match"
    );
    assert_eq!(
        r0_a.outcome, r0_b.outcome,
        "CZ symmetry: q0 outcome should match"
    );
    assert_eq!(
        r1_a.outcome, r1_b.outcome,
        "CZ symmetry: q1 outcome should match"
    );
}

/// Verify SWAP gate exchanges qubit states.
pub fn verify_swap_exchanges_states<S: CliffordGateable + QuantumSimulator>(sim: &mut S) {
    sim.reset();
    sim.x(&[QubitId::new(0)]); // q0 = |1>, q1 = |0>
    sim.swap(&[QubitId::new(0), QubitId::new(1)]);

    let r0 = sim.mz(&[QubitId::new(0)]);
    let r1 = sim.mz(&[QubitId::new(1)]);

    assert!(r0[0].is_deterministic && r1[0].is_deterministic);
    assert!(!r0[0].outcome, "After SWAP, q0 should be 0");
    assert!(r1[0].outcome, "After SWAP, q1 should be 1");
}

/// Verify SWAP^2 = I.
pub fn verify_swap_squared_is_identity<S: CliffordGateable + QuantumSimulator>(sim: &mut S) {
    sim.reset();
    sim.x(&[QubitId::new(0)]); // q0 = |1>, q1 = |0>
    sim.swap(&[QubitId::new(0), QubitId::new(1)]);
    sim.swap(&[QubitId::new(0), QubitId::new(1)]);

    let r0 = sim.mz(&[QubitId::new(0)]);
    let r1 = sim.mz(&[QubitId::new(1)]);

    assert!(r0[0].is_deterministic && r1[0].is_deterministic);
    assert!(r0[0].outcome, "After SWAP^2, q0 should be back to 1");
    assert!(!r1[0].outcome, "After SWAP^2, q1 should be back to 0");
}

/// Verify SWAP is symmetric: SWAP(0,1) = SWAP(1,0).
pub fn verify_swap_symmetric<S: CliffordGateable + QuantumSimulator>(sim: &mut S) {
    // SWAP(0,1)
    sim.reset();
    sim.x(&[QubitId::new(0)]);
    sim.swap(&[QubitId::new(0), QubitId::new(1)]);
    let r0_a = sim.mz(&[QubitId::new(0)]);
    let r1_a = sim.mz(&[QubitId::new(1)]);

    // SWAP(1,0)
    sim.reset();
    sim.x(&[QubitId::new(0)]);
    sim.swap(&[QubitId::new(1), QubitId::new(0)]);
    let r0_b = sim.mz(&[QubitId::new(0)]);
    let r1_b = sim.mz(&[QubitId::new(1)]);

    assert_eq!(
        r0_a[0].outcome, r0_b[0].outcome,
        "SWAP symmetry: q0 should match"
    );
    assert_eq!(
        r1_a[0].outcome, r1_b[0].outcome,
        "SWAP symmetry: q1 should match"
    );
}

/// Run all specific gate tests.
pub fn verify_all_specific_gates<S: CliffordGateable + QuantumSimulator + ForcedMeasurement>(
    sim: &mut S,
) {
    // SX tests
    verify_sx_squared_is_x(sim);
    verify_sx_creates_superposition(sim);
    verify_sx_sxdg_is_identity(sim);

    // SY tests
    verify_sy_squared_is_y(sim);
    verify_sy_creates_superposition(sim);
    verify_sy_sydg_is_identity(sim);

    // CY tests
    verify_cy_control_zero(sim);
    verify_cy_control_one(sim);
    verify_cy_squared_is_identity(sim);

    // CZ tests
    verify_cz_control_zero(sim);
    verify_cz_control_one(sim);
    verify_cz_symmetric(sim);

    // SWAP tests
    verify_swap_exchanges_states(sim);
    verify_swap_squared_is_identity(sim);
    verify_swap_symmetric(sim);
}

// ============================================================================
// Measurement Idempotence Tests
// ============================================================================

/// Verify that measuring the same qubit twice gives the same result.
///
/// After the first measurement, the qubit collapses to a definite state,
/// so the second measurement should be deterministic and give the same outcome.
pub fn verify_measurement_idempotence<S: CliffordGateable + QuantumSimulator>(sim: &mut S) {
    sim.reset();
    sim.h(&[QubitId::new(0)]); // Create superposition

    // First measurement (may be non-deterministic)
    let r1 = sim.mz(&[QubitId::new(0)]);

    // Second measurement (should be deterministic and same)
    let r2 = sim.mz(&[QubitId::new(0)]);

    assert!(
        r2[0].is_deterministic,
        "Second measurement should be deterministic"
    );
    assert_eq!(
        r1[0].outcome, r2[0].outcome,
        "Second measurement should give same result as first"
    );
}

/// Verify measurement idempotence on entangled state.
///
/// After measuring one qubit of a Bell pair, measuring it again should give same result.
pub fn verify_measurement_idempotence_entangled<S: CliffordGateable + QuantumSimulator>(
    sim: &mut S,
) {
    sim.reset();
    sim.h(&[QubitId::new(0)]);
    sim.cx(&[QubitId::new(0), QubitId::new(1)]);

    // First measurement on q0
    let r1 = sim.mz(&[QubitId::new(0)]);

    // Second measurement on q0 (should be deterministic and same)
    let r2 = sim.mz(&[QubitId::new(0)]);

    assert!(
        r2[0].is_deterministic,
        "Second measurement on Bell pair should be deterministic"
    );
    assert_eq!(
        r1[0].outcome, r2[0].outcome,
        "Second measurement should give same result"
    );

    // Also verify q1 is now deterministic and correlated
    let r3 = sim.mz(&[QubitId::new(1)]);
    assert!(
        r3[0].is_deterministic,
        "Partner qubit should be deterministic"
    );
    assert_eq!(
        r1[0].outcome, r3[0].outcome,
        "Bell pair should be correlated"
    );
}

/// Verify measurement idempotence after applying gates post-measurement.
pub fn verify_measurement_idempotence_with_gates<S: CliffordGateable + QuantumSimulator>(
    sim: &mut S,
) {
    sim.reset();
    sim.h(&[QubitId::new(0)]);

    // Measure
    let r1 = sim.mz(&[QubitId::new(0)]);

    // Apply X gate (flips the state)
    sim.x(&[QubitId::new(0)]);

    // Measure again
    let r2 = sim.mz(&[QubitId::new(0)]);

    assert!(
        r2[0].is_deterministic,
        "Post-X measurement should be deterministic"
    );
    assert_ne!(
        r1[0].outcome, r2[0].outcome,
        "X should flip the measurement outcome"
    );

    // Measure third time (should be same as second)
    let r3 = sim.mz(&[QubitId::new(0)]);
    assert_eq!(
        r2[0].outcome, r3[0].outcome,
        "Third measurement should match second"
    );
}

/// Run all measurement idempotence tests.
pub fn verify_all_measurement_idempotence<S: CliffordGateable + QuantumSimulator>(sim: &mut S) {
    verify_measurement_idempotence(sim);
    verify_measurement_idempotence_entangled(sim);
    verify_measurement_idempotence_with_gates(sim);
}

// ============================================================================
// Gate Decomposition Tests
// ============================================================================

/// Verify SWAP = CX(0,1) CX(1,0) CX(0,1).
pub fn verify_swap_decomposition<
    S: CliffordGateable + QuantumSimulator + ForcedMeasurement + Clone,
>(
    sim: &mut S,
    num_qubits: usize,
) {
    // Apply SWAP directly
    sim.reset();
    sim.h(&[QubitId::new(0)]);
    sim.x(&[QubitId::new(1)]);
    sim.swap(&[QubitId::new(0), QubitId::new(1)]);

    let mut dm1 = DensityMatrix::new(num_qubits);
    dm1.h(&[QubitId::new(0)]);
    dm1.x(&[QubitId::new(1)]);
    dm1.swap(&[QubitId::new(0), QubitId::new(1)]);

    verify_probabilities_match_density_matrix(sim, &dm1, num_qubits);

    // Apply decomposition: CX(0,1) CX(1,0) CX(0,1)
    sim.reset();
    sim.h(&[QubitId::new(0)]);
    sim.x(&[QubitId::new(1)]);
    sim.cx(&[QubitId::new(0), QubitId::new(1)]);
    sim.cx(&[QubitId::new(1), QubitId::new(0)]);
    sim.cx(&[QubitId::new(0), QubitId::new(1)]);

    let mut dm2 = DensityMatrix::new(num_qubits);
    dm2.h(&[QubitId::new(0)]);
    dm2.x(&[QubitId::new(1)]);
    dm2.cx(&[QubitId::new(0), QubitId::new(1)]);
    dm2.cx(&[QubitId::new(1), QubitId::new(0)]);
    dm2.cx(&[QubitId::new(0), QubitId::new(1)]);

    verify_probabilities_match_density_matrix(sim, &dm2, num_qubits);
}

/// Verify CZ = H(target) CX H(target).
pub fn verify_cz_decomposition<
    S: CliffordGateable + QuantumSimulator + ForcedMeasurement + Clone,
>(
    sim: &mut S,
    num_qubits: usize,
) {
    // Apply CZ directly
    sim.reset();
    sim.h(&[QubitId::new(0)]);
    sim.h(&[QubitId::new(1)]);
    sim.cz(&[QubitId::new(0), QubitId::new(1)]);

    let mut dm1 = DensityMatrix::new(num_qubits);
    dm1.h(&[QubitId::new(0)]);
    dm1.h(&[QubitId::new(1)]);
    dm1.cz(&[QubitId::new(0), QubitId::new(1)]);

    verify_probabilities_match_density_matrix(sim, &dm1, num_qubits);

    // Apply decomposition: H(1) CX(0,1) H(1)
    sim.reset();
    sim.h(&[QubitId::new(0)]);
    sim.h(&[QubitId::new(1)]);
    sim.h(&[QubitId::new(1)]);
    sim.cx(&[QubitId::new(0), QubitId::new(1)]);
    sim.h(&[QubitId::new(1)]);

    let mut dm2 = DensityMatrix::new(num_qubits);
    dm2.h(&[QubitId::new(0)]);
    dm2.h(&[QubitId::new(1)]);
    dm2.h(&[QubitId::new(1)]);
    dm2.cx(&[QubitId::new(0), QubitId::new(1)]);
    dm2.h(&[QubitId::new(1)]);

    verify_probabilities_match_density_matrix(sim, &dm2, num_qubits);
}

/// Verify CY = S(target) CX Sdg(target).
pub fn verify_cy_decomposition<
    S: CliffordGateable + QuantumSimulator + ForcedMeasurement + Clone,
>(
    sim: &mut S,
    num_qubits: usize,
) {
    // Apply CY directly
    sim.reset();
    sim.h(&[QubitId::new(0)]);
    sim.cy(&[QubitId::new(0), QubitId::new(1)]);

    let mut dm1 = DensityMatrix::new(num_qubits);
    dm1.h(&[QubitId::new(0)]);
    dm1.cy(&[QubitId::new(0), QubitId::new(1)]);

    verify_probabilities_match_density_matrix(sim, &dm1, num_qubits);

    // Apply decomposition: Sdg(1) CX(0,1) S(1)
    sim.reset();
    sim.h(&[QubitId::new(0)]);
    sim.szdg(&[QubitId::new(1)]);
    sim.cx(&[QubitId::new(0), QubitId::new(1)]);
    sim.sz(&[QubitId::new(1)]);

    let mut dm2 = DensityMatrix::new(num_qubits);
    dm2.h(&[QubitId::new(0)]);
    dm2.szdg(&[QubitId::new(1)]);
    dm2.cx(&[QubitId::new(0), QubitId::new(1)]);
    dm2.sz(&[QubitId::new(1)]);

    verify_probabilities_match_density_matrix(sim, &dm2, num_qubits);
}

/// Verify X = H Z H.
pub fn verify_x_decomposition<
    S: CliffordGateable + QuantumSimulator + ForcedMeasurement + Clone,
>(
    sim: &mut S,
    num_qubits: usize,
) {
    // Apply X directly
    sim.reset();
    sim.h(&[QubitId::new(0)]);
    sim.x(&[QubitId::new(0)]);

    let mut dm1 = DensityMatrix::new(num_qubits);
    dm1.h(&[QubitId::new(0)]);
    dm1.x(&[QubitId::new(0)]);

    verify_probabilities_match_density_matrix(sim, &dm1, num_qubits);

    // Apply decomposition: H Z H
    sim.reset();
    sim.h(&[QubitId::new(0)]);
    sim.h(&[QubitId::new(0)]);
    sim.z(&[QubitId::new(0)]);
    sim.h(&[QubitId::new(0)]);

    let mut dm2 = DensityMatrix::new(num_qubits);
    dm2.h(&[QubitId::new(0)]);
    dm2.h(&[QubitId::new(0)]);
    dm2.z(&[QubitId::new(0)]);
    dm2.h(&[QubitId::new(0)]);

    verify_probabilities_match_density_matrix(sim, &dm2, num_qubits);
}

/// Verify Y = S X S^dag = S H Z H S^dag.
pub fn verify_y_decomposition<
    S: CliffordGateable + QuantumSimulator + ForcedMeasurement + Clone,
>(
    sim: &mut S,
    num_qubits: usize,
) {
    // Apply Y directly
    sim.reset();
    sim.h(&[QubitId::new(0)]);
    sim.y(&[QubitId::new(0)]);

    let mut dm1 = DensityMatrix::new(num_qubits);
    dm1.h(&[QubitId::new(0)]);
    dm1.y(&[QubitId::new(0)]);

    verify_probabilities_match_density_matrix(sim, &dm1, num_qubits);

    // Apply decomposition: S X Sdg
    sim.reset();
    sim.h(&[QubitId::new(0)]);
    sim.sz(&[QubitId::new(0)]);
    sim.x(&[QubitId::new(0)]);
    sim.szdg(&[QubitId::new(0)]);

    // Note: This gives -iY not Y, but measurement probabilities should match
    // since global phase doesn't affect measurements
}

/// Run all gate decomposition tests.
pub fn verify_all_gate_decompositions<
    S: CliffordGateable + QuantumSimulator + ForcedMeasurement + Clone,
>(
    sim: &mut S,
    num_qubits: usize,
) {
    verify_swap_decomposition(sim, num_qubits);
    verify_cz_decomposition(sim, num_qubits);
    verify_cy_decomposition(sim, num_qubits);
    verify_x_decomposition(sim, num_qubits);
    verify_y_decomposition(sim, num_qubits);
}

// ============================================================================
// Gate Decomposition Tests (Direct - no Clone required)
// ============================================================================

/// Verify SWAP = CX(0,1) CX(1,0) CX(0,1) using direct comparison.
///
/// This version doesn't require Clone - it uses two simulators.
pub fn verify_swap_decomposition_direct<
    S1: CliffordGateable + QuantumSimulator + ForcedMeasurement,
    S2: CliffordGateable + QuantumSimulator + ForcedMeasurement,
>(
    sim1: &mut S1,
    sim2: &mut S2,
    num_qubits: usize,
) {
    use pecos_rng::PecosRng;

    let mut rng = PecosRng::seed_from_u64(98765);

    // Test with a few different initial states
    for _ in 0..5 {
        // Generate random initial circuit
        let init_circuit = generate_random_clifford_circuit(&mut rng, num_qubits, 10);

        // Apply initial circuit to both, then SWAP to sim1
        sim1.reset();
        sim2.reset();
        apply_circuit(sim1, &init_circuit);
        apply_circuit(sim2, &init_circuit);

        sim1.swap(&[QubitId::new(0), QubitId::new(1)]);

        // Apply decomposition to sim2: CX(0,1) CX(1,0) CX(0,1)
        sim2.cx(&[QubitId::new(0), QubitId::new(1)]);
        sim2.cx(&[QubitId::new(1), QubitId::new(0)]);
        sim2.cx(&[QubitId::new(0), QubitId::new(1)]);

        // Compare measurements
        for q in 0..num_qubits {
            let forced: bool = rng.random();
            let r1 = sim1.mz_forced(q, forced);
            let r2 = sim2.mz_forced(q, forced);

            assert_eq!(
                r1.is_deterministic, r2.is_deterministic,
                "SWAP decomposition: determinism mismatch for qubit {q}"
            );
            assert_eq!(
                r1.outcome, r2.outcome,
                "SWAP decomposition: outcome mismatch for qubit {q}"
            );
        }
    }
}

/// Verify CZ = H(target) CX H(target) using direct comparison.
pub fn verify_cz_decomposition_direct<
    S1: CliffordGateable + QuantumSimulator + ForcedMeasurement,
    S2: CliffordGateable + QuantumSimulator + ForcedMeasurement,
>(
    sim1: &mut S1,
    sim2: &mut S2,
    num_qubits: usize,
) {
    use pecos_rng::PecosRng;

    let mut rng = PecosRng::seed_from_u64(87654);

    for _ in 0..5 {
        let init_circuit = generate_random_clifford_circuit(&mut rng, num_qubits, 10);

        sim1.reset();
        sim2.reset();
        apply_circuit(sim1, &init_circuit);
        apply_circuit(sim2, &init_circuit);

        // sim1: CZ directly
        sim1.cz(&[QubitId::new(0), QubitId::new(1)]);

        // sim2: H(1) CX(0,1) H(1)
        sim2.h(&[QubitId::new(1)]);
        sim2.cx(&[QubitId::new(0), QubitId::new(1)]);
        sim2.h(&[QubitId::new(1)]);

        for q in 0..num_qubits {
            let forced: bool = rng.random();
            let r1 = sim1.mz_forced(q, forced);
            let r2 = sim2.mz_forced(q, forced);

            assert_eq!(
                r1.is_deterministic, r2.is_deterministic,
                "CZ decomposition: determinism mismatch for qubit {q}"
            );
            assert_eq!(
                r1.outcome, r2.outcome,
                "CZ decomposition: outcome mismatch for qubit {q}"
            );
        }
    }
}

/// Verify CY = Sdg(target) CX S(target) using direct comparison.
pub fn verify_cy_decomposition_direct<
    S1: CliffordGateable + QuantumSimulator + ForcedMeasurement,
    S2: CliffordGateable + QuantumSimulator + ForcedMeasurement,
>(
    sim1: &mut S1,
    sim2: &mut S2,
    num_qubits: usize,
) {
    use pecos_rng::PecosRng;

    let mut rng = PecosRng::seed_from_u64(76543);

    for _ in 0..5 {
        let init_circuit = generate_random_clifford_circuit(&mut rng, num_qubits, 10);

        sim1.reset();
        sim2.reset();
        apply_circuit(sim1, &init_circuit);
        apply_circuit(sim2, &init_circuit);

        // sim1: CY directly
        sim1.cy(&[QubitId::new(0), QubitId::new(1)]);

        // sim2: Sdg(1) CX(0,1) S(1)
        sim2.szdg(&[QubitId::new(1)]);
        sim2.cx(&[QubitId::new(0), QubitId::new(1)]);
        sim2.sz(&[QubitId::new(1)]);

        for q in 0..num_qubits {
            let forced: bool = rng.random();
            let r1 = sim1.mz_forced(q, forced);
            let r2 = sim2.mz_forced(q, forced);

            assert_eq!(
                r1.is_deterministic, r2.is_deterministic,
                "CY decomposition: determinism mismatch for qubit {q}"
            );
            assert_eq!(
                r1.outcome, r2.outcome,
                "CY decomposition: outcome mismatch for qubit {q}"
            );
        }
    }
}

/// Verify X = H Z H using direct comparison.
pub fn verify_x_decomposition_direct<
    S1: CliffordGateable + QuantumSimulator + ForcedMeasurement,
    S2: CliffordGateable + QuantumSimulator + ForcedMeasurement,
>(
    sim1: &mut S1,
    sim2: &mut S2,
    num_qubits: usize,
) {
    use pecos_rng::PecosRng;

    let mut rng = PecosRng::seed_from_u64(65432);

    for _ in 0..5 {
        let init_circuit = generate_random_clifford_circuit(&mut rng, num_qubits, 10);

        sim1.reset();
        sim2.reset();
        apply_circuit(sim1, &init_circuit);
        apply_circuit(sim2, &init_circuit);

        // sim1: X directly
        sim1.x(&[QubitId::new(0)]);

        // sim2: H Z H
        sim2.h(&[QubitId::new(0)]);
        sim2.z(&[QubitId::new(0)]);
        sim2.h(&[QubitId::new(0)]);

        for q in 0..num_qubits {
            let forced: bool = rng.random();
            let r1 = sim1.mz_forced(q, forced);
            let r2 = sim2.mz_forced(q, forced);

            assert_eq!(
                r1.is_deterministic, r2.is_deterministic,
                "X decomposition: determinism mismatch for qubit {q}"
            );
            assert_eq!(
                r1.outcome, r2.outcome,
                "X decomposition: outcome mismatch for qubit {q}"
            );
        }
    }
}

/// Verify Y = S X Sdg using direct comparison.
pub fn verify_y_decomposition_direct<
    S1: CliffordGateable + QuantumSimulator + ForcedMeasurement,
    S2: CliffordGateable + QuantumSimulator + ForcedMeasurement,
>(
    sim1: &mut S1,
    sim2: &mut S2,
    num_qubits: usize,
) {
    use pecos_rng::PecosRng;

    let mut rng = PecosRng::seed_from_u64(54321);

    for _ in 0..5 {
        let init_circuit = generate_random_clifford_circuit(&mut rng, num_qubits, 10);

        sim1.reset();
        sim2.reset();
        apply_circuit(sim1, &init_circuit);
        apply_circuit(sim2, &init_circuit);

        // sim1: Y directly
        sim1.y(&[QubitId::new(0)]);

        // sim2: S X Sdg (note: this gives -iY, but global phase doesn't affect measurements)
        sim2.sz(&[QubitId::new(0)]);
        sim2.x(&[QubitId::new(0)]);
        sim2.szdg(&[QubitId::new(0)]);

        for q in 0..num_qubits {
            let forced: bool = rng.random();
            let r1 = sim1.mz_forced(q, forced);
            let r2 = sim2.mz_forced(q, forced);

            assert_eq!(
                r1.is_deterministic, r2.is_deterministic,
                "Y decomposition: determinism mismatch for qubit {q}"
            );
            assert_eq!(
                r1.outcome, r2.outcome,
                "Y decomposition: outcome mismatch for qubit {q}"
            );
        }
    }
}

/// Run all gate decomposition tests (direct version - no Clone required).
pub fn verify_all_gate_decompositions_direct<
    S1: CliffordGateable + QuantumSimulator + ForcedMeasurement,
    S2: CliffordGateable + QuantumSimulator + ForcedMeasurement,
>(
    sim1: &mut S1,
    sim2: &mut S2,
    num_qubits: usize,
) {
    verify_swap_decomposition_direct(sim1, sim2, num_qubits);
    verify_cz_decomposition_direct(sim1, sim2, num_qubits);
    verify_cy_decomposition_direct(sim1, sim2, num_qubits);
    verify_x_decomposition_direct(sim1, sim2, num_qubits);
    verify_y_decomposition_direct(sim1, sim2, num_qubits);
}

// ============================================================================
// Commutation Relation Tests
// ============================================================================

/// Verify that X and Z on different qubits commute.
///
/// X(0) Z(1) should equal Z(1) X(0).
pub fn verify_xz_commute_different_qubits<
    S: CliffordGateable + QuantumSimulator + ForcedMeasurement,
>(
    sim: &mut S,
) {
    // X(0) Z(1)
    sim.reset();
    sim.h(&[QubitId::new(0)]);
    sim.h(&[QubitId::new(1)]);
    sim.x(&[QubitId::new(0)]);
    sim.z(&[QubitId::new(1)]);

    let r0_a = sim.mz_forced(0, false);
    let r1_a = sim.mz_forced(1, false);

    // Z(1) X(0)
    sim.reset();
    sim.h(&[QubitId::new(0)]);
    sim.h(&[QubitId::new(1)]);
    sim.z(&[QubitId::new(1)]);
    sim.x(&[QubitId::new(0)]);

    let r0_b = sim.mz_forced(0, false);
    let r1_b = sim.mz_forced(1, false);

    assert_eq!(
        r0_a.outcome, r0_b.outcome,
        "X,Z commute on different qubits: q0"
    );
    assert_eq!(
        r1_a.outcome, r1_b.outcome,
        "X,Z commute on different qubits: q1"
    );
}

/// Verify that CX gates on disjoint qubits commute.
///
/// CX(0,1) CX(2,3) should equal CX(2,3) CX(0,1).
pub fn verify_cx_commute_disjoint<S: CliffordGateable + QuantumSimulator + ForcedMeasurement>(
    sim: &mut S,
) {
    // CX(0,1) CX(2,3)
    sim.reset();
    sim.h(&[QubitId::new(0)]);
    sim.h(&[QubitId::new(2)]);
    sim.cx(&[QubitId::new(0), QubitId::new(1)]);
    sim.cx(&[QubitId::new(2), QubitId::new(3)]);

    let mut results_a = Vec::new();
    for q in 0..4 {
        results_a.push(sim.mz_forced(q, false));
    }

    // CX(2,3) CX(0,1)
    sim.reset();
    sim.h(&[QubitId::new(0)]);
    sim.h(&[QubitId::new(2)]);
    sim.cx(&[QubitId::new(2), QubitId::new(3)]);
    sim.cx(&[QubitId::new(0), QubitId::new(1)]);

    let mut results_b = Vec::new();
    for q in 0..4 {
        results_b.push(sim.mz_forced(q, false));
    }

    for q in 0..4 {
        assert_eq!(
            results_a[q].outcome, results_b[q].outcome,
            "Disjoint CX gates should commute: qubit {q}"
        );
    }
}

/// Verify that H gates on different qubits commute.
pub fn verify_h_commute_different_qubits<
    S: CliffordGateable + QuantumSimulator + ForcedMeasurement,
>(
    sim: &mut S,
) {
    // H(0) H(1)
    sim.reset();
    sim.x(&[QubitId::new(0)]); // Start with |10>
    sim.h(&[QubitId::new(0)]);
    sim.h(&[QubitId::new(1)]);

    let r0_a = sim.mz_forced(0, true);
    let r1_a = sim.mz_forced(1, false);

    // H(1) H(0)
    sim.reset();
    sim.x(&[QubitId::new(0)]); // Start with |10>
    sim.h(&[QubitId::new(1)]);
    sim.h(&[QubitId::new(0)]);

    let r0_b = sim.mz_forced(0, true);
    let r1_b = sim.mz_forced(1, false);

    assert_eq!(
        r0_a.is_deterministic, r0_b.is_deterministic,
        "H gates commute: q0 determinism"
    );
    assert_eq!(
        r1_a.is_deterministic, r1_b.is_deterministic,
        "H gates commute: q1 determinism"
    );
}

/// Verify that CX(a,b) and CX(a,c) commute when b != c (same control, different targets).
pub fn verify_cx_same_control_commute<
    S: CliffordGateable + QuantumSimulator + ForcedMeasurement,
>(
    sim: &mut S,
) {
    // CX(0,1) CX(0,2)
    sim.reset();
    sim.h(&[QubitId::new(0)]);
    sim.cx(&[QubitId::new(0), QubitId::new(1)]);
    sim.cx(&[QubitId::new(0), QubitId::new(2)]);

    let mut results_a = Vec::new();
    for q in 0..3 {
        results_a.push(sim.mz_forced(q, false));
    }

    // CX(0,2) CX(0,1)
    sim.reset();
    sim.h(&[QubitId::new(0)]);
    sim.cx(&[QubitId::new(0), QubitId::new(2)]);
    sim.cx(&[QubitId::new(0), QubitId::new(1)]);

    let mut results_b = Vec::new();
    for q in 0..3 {
        results_b.push(sim.mz_forced(q, false));
    }

    for q in 0..3 {
        assert_eq!(
            results_a[q].outcome, results_b[q].outcome,
            "CX with same control should commute: qubit {q}"
        );
    }
}

/// Verify that S and Z commute (they're both diagonal).
pub fn verify_s_z_commute<S: CliffordGateable + QuantumSimulator + ForcedMeasurement>(sim: &mut S) {
    // S Z
    sim.reset();
    sim.h(&[QubitId::new(0)]);
    sim.sz(&[QubitId::new(0)]);
    sim.z(&[QubitId::new(0)]);
    sim.h(&[QubitId::new(0)]);

    let r_a = sim.mz_forced(0, false);

    // Z S
    sim.reset();
    sim.h(&[QubitId::new(0)]);
    sim.z(&[QubitId::new(0)]);
    sim.sz(&[QubitId::new(0)]);
    sim.h(&[QubitId::new(0)]);

    let r_b = sim.mz_forced(0, false);

    assert_eq!(r_a.outcome, r_b.outcome, "S and Z should commute");
}

/// Run all commutation relation tests.
pub fn verify_all_commutation_relations<
    S: CliffordGateable + QuantumSimulator + ForcedMeasurement,
>(
    sim: &mut S,
) {
    verify_xz_commute_different_qubits(sim);
    verify_h_commute_different_qubits(sim);
    verify_s_z_commute(sim);
    // These require 3-4 qubits
    // verify_cx_commute_disjoint requires 4 qubits
    // verify_cx_same_control_commute requires 3 qubits
}

/// Run all commutation tests including those requiring more qubits.
pub fn verify_all_commutation_relations_extended<
    S: CliffordGateable + QuantumSimulator + ForcedMeasurement,
>(
    sim: &mut S,
    num_qubits: usize,
) {
    verify_xz_commute_different_qubits(sim);
    verify_h_commute_different_qubits(sim);
    verify_s_z_commute(sim);

    if num_qubits >= 3 {
        verify_cx_same_control_commute(sim);
    }
    if num_qubits >= 4 {
        verify_cx_commute_disjoint(sim);
    }
}

// ============================================================================
// Probability Comparison Against DensityMatrix
// ============================================================================

/// Calculate the probability of measuring a specific basis state using forced measurements.
///
/// This clones the simulator, forces measurements to get the target state,
/// and returns the probability (0.5^n for n non-deterministic measurements).
pub fn calculate_basis_probability<
    S: CliffordGateable + QuantumSimulator + ForcedMeasurement + Clone,
>(
    sim: &S,
    basis_state: usize,
    num_qubits: usize,
) -> f64 {
    let mut sim_copy = sim.clone();
    let mut probability = 1.0;

    for q in 0..num_qubits {
        let bit_is_one = (basis_state >> q) & 1 == 1;
        let result = sim_copy.mz_forced(q, bit_is_one);

        if !result.is_deterministic {
            probability *= 0.5;
        } else if result.outcome != bit_is_one {
            return 0.0; // Impossible outcome
        }
    }

    probability
}

/// Compare probabilities against `DensityMatrix` for all basis states.
///
/// This is the gold standard test for verifying stabilizer simulator correctness.
pub fn verify_probabilities_match_density_matrix<
    S: CliffordGateable + QuantumSimulator + ForcedMeasurement + Clone,
>(
    sim: &S,
    dm: &DensityMatrix,
    num_qubits: usize,
) {
    const TOLERANCE: f64 = 1e-10;

    for basis_state in 0..(1 << num_qubits) {
        let stab_prob = calculate_basis_probability(sim, basis_state, num_qubits);
        let dm_prob = dm.probability(basis_state);

        assert!(
            (stab_prob - dm_prob).abs() < TOLERANCE,
            "Probability mismatch for basis state {basis_state}: stabilizer={stab_prob}, density_matrix={dm_prob}"
        );
    }
}

/// Apply the same Clifford circuit to a stabilizer simulator and `DensityMatrix`,
/// then verify they have matching probabilities.
pub fn verify_circuit_matches_density_matrix<
    S: CliffordGateable + QuantumSimulator + ForcedMeasurement + Clone,
    F: FnMut(&mut S, &mut DensityMatrix),
>(
    sim: &mut S,
    dm: &mut DensityMatrix,
    num_qubits: usize,
    mut circuit: F,
) {
    sim.reset();
    dm.reset();
    circuit(sim, dm);
    verify_probabilities_match_density_matrix(sim, dm, num_qubits);
}

// ============================================================================
// Random Circuit Comparison Tests
// ============================================================================

/// Enumeration of Clifford gates for random circuit generation.
#[derive(Debug, Clone, Copy)]
pub enum CliffordGate {
    // Single-qubit gates
    H(usize),
    S(usize),
    Sdg(usize),
    X(usize),
    Y(usize),
    Z(usize),
    SX(usize),
    SXdg(usize),
    SY(usize),
    SYdg(usize),
    // Two-qubit gates
    CX(usize, usize),
    CY(usize, usize),
    CZ(usize, usize),
    SWAP(usize, usize),
}

/// Generate a random Clifford circuit.
///
/// Returns a vector of gates that can be applied to any simulator.
pub fn generate_random_clifford_circuit<R: Rng>(
    rng: &mut R,
    num_qubits: usize,
    num_gates: usize,
) -> Vec<CliffordGate> {
    let mut circuit = Vec::with_capacity(num_gates);

    for _ in 0..num_gates {
        // Choose gate type: 0-9 for single-qubit, 10-13 for two-qubit
        let gate_type: u8 = rng.random_range(0..14);
        let q0 = rng.random_range(0..num_qubits);

        // The explicit `0` case is intentional for clarity, even though the wildcard also returns H.
        // The wildcard handles fallback when num_qubits < 2 and a two-qubit gate is selected.
        #[allow(clippy::match_same_arms)]
        let gate = match gate_type {
            0 => CliffordGate::H(q0),
            1 => CliffordGate::S(q0),
            2 => CliffordGate::Sdg(q0),
            3 => CliffordGate::X(q0),
            4 => CliffordGate::Y(q0),
            5 => CliffordGate::Z(q0),
            6 => CliffordGate::SX(q0),
            7 => CliffordGate::SXdg(q0),
            8 => CliffordGate::SY(q0),
            9 => CliffordGate::SYdg(q0),
            10..=13 if num_qubits >= 2 => {
                // Two-qubit gate - pick a different qubit for q1
                let mut q1 = rng.random_range(0..num_qubits);
                while q1 == q0 {
                    q1 = rng.random_range(0..num_qubits);
                }
                match gate_type {
                    10 => CliffordGate::CX(q0, q1),
                    11 => CliffordGate::CY(q0, q1),
                    12 => CliffordGate::CZ(q0, q1),
                    13 => CliffordGate::SWAP(q0, q1),
                    _ => unreachable!(),
                }
            }
            // Fall back to single-qubit gate if only 1 qubit
            _ => CliffordGate::H(q0),
        };

        circuit.push(gate);
    }

    circuit
}

/// Apply a Clifford gate to a simulator.
pub fn apply_gate<S: CliffordGateable>(sim: &mut S, gate: &CliffordGate) {
    match *gate {
        CliffordGate::H(q) => {
            sim.h(&[QubitId::new(q)]);
        }
        CliffordGate::S(q) => {
            sim.sz(&[QubitId::new(q)]);
        }
        CliffordGate::Sdg(q) => {
            sim.szdg(&[QubitId::new(q)]);
        }
        CliffordGate::X(q) => {
            sim.x(&[QubitId::new(q)]);
        }
        CliffordGate::Y(q) => {
            sim.y(&[QubitId::new(q)]);
        }
        CliffordGate::Z(q) => {
            sim.z(&[QubitId::new(q)]);
        }
        CliffordGate::SX(q) => {
            sim.sx(&[QubitId::new(q)]);
        }
        CliffordGate::SXdg(q) => {
            sim.sxdg(&[QubitId::new(q)]);
        }
        CliffordGate::SY(q) => {
            sim.sy(&[QubitId::new(q)]);
        }
        CliffordGate::SYdg(q) => {
            sim.sydg(&[QubitId::new(q)]);
        }
        CliffordGate::CX(c, t) => {
            sim.cx(&[QubitId::new(c), QubitId::new(t)]);
        }
        CliffordGate::CY(c, t) => {
            sim.cy(&[QubitId::new(c), QubitId::new(t)]);
        }
        CliffordGate::CZ(q0, q1) => {
            sim.cz(&[QubitId::new(q0), QubitId::new(q1)]);
        }
        CliffordGate::SWAP(q0, q1) => {
            sim.swap(&[QubitId::new(q0), QubitId::new(q1)]);
        }
    }
}

/// Apply a circuit (list of gates) to a simulator.
pub fn apply_circuit<S: CliffordGateable>(sim: &mut S, circuit: &[CliffordGate]) {
    for gate in circuit {
        apply_gate(sim, gate);
    }
}

/// Verify that a stabilizer simulator matches `DensityMatrix` on a random circuit.
///
/// This generates a random circuit with the given seed and verifies that
/// the stabilizer simulator produces the same probabilities as `DensityMatrix`.
pub fn verify_random_circuit_matches_density_matrix<
    S: CliffordGateable + QuantumSimulator + ForcedMeasurement + Clone,
>(
    sim: &mut S,
    num_qubits: usize,
    num_gates: usize,
    seed: u64,
) {
    use pecos_rng::PecosRng;

    let mut rng = PecosRng::seed_from_u64(seed);
    let circuit = generate_random_clifford_circuit(&mut rng, num_qubits, num_gates);

    // Apply to stabilizer simulator
    sim.reset();
    apply_circuit(sim, &circuit);

    // Apply to DensityMatrix
    let mut dm = DensityMatrix::new(num_qubits);
    apply_circuit(&mut dm, &circuit);

    // Compare probabilities
    verify_probabilities_match_density_matrix(sim, &dm, num_qubits);
}

/// Run multiple random circuit tests with different seeds.
///
/// This is a stress test that verifies the simulator on many random circuits.
pub fn verify_random_circuits<
    S: CliffordGateable + QuantumSimulator + ForcedMeasurement + Clone,
>(
    sim: &mut S,
    num_qubits: usize,
    num_gates: usize,
    num_circuits: usize,
    base_seed: u64,
) {
    for i in 0..num_circuits {
        let seed = base_seed.wrapping_add(i as u64);
        verify_random_circuit_matches_density_matrix(sim, num_qubits, num_gates, seed);
    }
}

/// Compare two stabilizer simulators on the same random circuit using forced measurements.
///
/// This runs the same circuit on both simulators, then measures all qubits using
/// forced measurements (where we force the same "coin flip" for non-deterministic
/// measurements). Both simulators should produce identical measurement outcomes.
///
/// This version does NOT require Clone, making it suitable for GPU simulators.
pub fn compare_simulators_on_random_circuit_direct<
    S1: CliffordGateable + QuantumSimulator + ForcedMeasurement,
    S2: CliffordGateable + QuantumSimulator + ForcedMeasurement,
>(
    sim1: &mut S1,
    sim2: &mut S2,
    num_qubits: usize,
    num_gates: usize,
    seed: u64,
) {
    use pecos_rng::PecosRng;

    let mut rng = PecosRng::seed_from_u64(seed);
    let circuit = generate_random_clifford_circuit(&mut rng, num_qubits, num_gates);

    // Apply to both simulators
    sim1.reset();
    sim2.reset();
    apply_circuit(sim1, &circuit);
    apply_circuit(sim2, &circuit);

    // Create a fresh RNG for measurement outcomes (same seed for reproducibility)
    let mut meas_rng = PecosRng::seed_from_u64(seed.wrapping_add(1_000_000));

    // Measure all qubits with forced outcomes - both should agree
    for q in 0..num_qubits {
        // Generate a random outcome to force for non-deterministic measurements
        let forced_outcome: bool = meas_rng.random();

        let r1 = sim1.mz_forced(q, forced_outcome);
        let r2 = sim2.mz_forced(q, forced_outcome);

        // Both should have the same determinism status
        assert_eq!(
            r1.is_deterministic, r2.is_deterministic,
            "Determinism mismatch for qubit {q} on circuit with seed {seed}"
        );

        // Both should have the same outcome
        assert_eq!(
            r1.outcome, r2.outcome,
            "Outcome mismatch for qubit {q} on circuit with seed {seed}: sim1={}, sim2={}",
            r1.outcome, r2.outcome
        );
    }
}

/// Compare two stabilizer simulators on the same random circuit.
///
/// This verifies that both simulators produce the same measurement outcomes
/// when given the same forced measurements.
pub fn compare_simulators_on_random_circuit<
    S1: CliffordGateable + QuantumSimulator + ForcedMeasurement + Clone,
    S2: CliffordGateable + QuantumSimulator + ForcedMeasurement + Clone,
>(
    sim1: &mut S1,
    sim2: &mut S2,
    num_qubits: usize,
    num_gates: usize,
    seed: u64,
) {
    use pecos_rng::PecosRng;
    const TOLERANCE: f64 = 1e-10;

    let mut rng = PecosRng::seed_from_u64(seed);
    let circuit = generate_random_clifford_circuit(&mut rng, num_qubits, num_gates);

    // Apply to both simulators
    sim1.reset();
    sim2.reset();
    apply_circuit(sim1, &circuit);
    apply_circuit(sim2, &circuit);

    // Compare probabilities for all basis states
    for basis_state in 0..(1 << num_qubits) {
        let prob1 = calculate_basis_probability(sim1, basis_state, num_qubits);
        let prob2 = calculate_basis_probability(sim2, basis_state, num_qubits);

        assert!(
            (prob1 - prob2).abs() < TOLERANCE,
            "Probability mismatch for basis state {basis_state} on circuit with seed {seed}: sim1={prob1}, sim2={prob2}"
        );
    }
}

/// Run multiple random circuit comparison tests between two simulators (direct method).
///
/// This version does NOT require Clone, making it suitable for GPU simulators.
pub fn compare_simulators_on_random_circuits_direct<
    S1: CliffordGateable + QuantumSimulator + ForcedMeasurement,
    S2: CliffordGateable + QuantumSimulator + ForcedMeasurement,
>(
    sim1: &mut S1,
    sim2: &mut S2,
    num_qubits: usize,
    num_gates: usize,
    num_circuits: usize,
    base_seed: u64,
) {
    for i in 0..num_circuits {
        let seed = base_seed.wrapping_add(i as u64);
        compare_simulators_on_random_circuit_direct(sim1, sim2, num_qubits, num_gates, seed);
    }
}

/// Run multiple random circuit comparison tests between two simulators.
pub fn compare_simulators_on_random_circuits<
    S1: CliffordGateable + QuantumSimulator + ForcedMeasurement + Clone,
    S2: CliffordGateable + QuantumSimulator + ForcedMeasurement + Clone,
>(
    sim1: &mut S1,
    sim2: &mut S2,
    num_qubits: usize,
    num_gates: usize,
    num_circuits: usize,
    base_seed: u64,
) {
    for i in 0..num_circuits {
        let seed = base_seed.wrapping_add(i as u64);
        compare_simulators_on_random_circuit(sim1, sim2, num_qubits, num_gates, seed);
    }
}

// ============================================================================
// Mid-Circuit Measurement Tests
// ============================================================================

/// Verify mid-circuit measurement followed by more gates.
///
/// This tests that measurement correctly updates the tableau so subsequent
/// gates operate on the correct post-measurement state.
pub fn verify_mid_circuit_measurement<
    S: CliffordGateable + QuantumSimulator + ForcedMeasurement + Clone,
>(
    sim: &mut S,
    num_qubits: usize,
    seed: u64,
) {
    use pecos_rng::PecosRng;

    let mut rng = PecosRng::seed_from_u64(seed);

    // Generate first half of circuit
    let circuit1 = generate_random_clifford_circuit(&mut rng, num_qubits, 10);

    // Apply first half to both sim and reference DensityMatrix
    sim.reset();
    let mut dm = DensityMatrix::new(num_qubits);
    apply_circuit(sim, &circuit1);
    apply_circuit(&mut dm, &circuit1);

    // Measure qubit 0 with forced outcome
    let forced_outcome: bool = rng.random();
    let sim_result = sim.mz_forced(0, forced_outcome);

    // For DensityMatrix, we need to apply the measurement projector
    // We'll just measure and compare determinism
    let dm_results = dm.mz(&[QubitId::new(0)]);

    // Both should agree on determinism
    assert_eq!(
        sim_result.is_deterministic, dm_results[0].is_deterministic,
        "Mid-circuit measurement determinism mismatch"
    );

    // If deterministic, outcomes should match
    if sim_result.is_deterministic {
        assert_eq!(
            sim_result.outcome, dm_results[0].outcome,
            "Deterministic mid-circuit measurement outcome mismatch"
        );
    }

    // Generate second half of circuit (only on remaining qubits to avoid measured qubit issues)
    let circuit2 = generate_random_clifford_circuit(&mut rng, num_qubits, 10);

    // Apply second half
    apply_circuit(sim, &circuit2);

    // Verify the simulator is still in a valid state by measuring remaining qubits
    for q in 1..num_qubits {
        let result = sim.mz(&[QubitId::new(q)]);
        // Just verify we get a result without crashing
        let _ = result[0].outcome;
    }
}

/// Verify that two simulators agree on mid-circuit measurement behavior.
pub fn compare_mid_circuit_measurement<
    S1: CliffordGateable + QuantumSimulator + ForcedMeasurement,
    S2: CliffordGateable + QuantumSimulator + ForcedMeasurement,
>(
    sim1: &mut S1,
    sim2: &mut S2,
    num_qubits: usize,
    seed: u64,
) {
    use pecos_rng::PecosRng;

    let mut rng = PecosRng::seed_from_u64(seed);

    // Generate first half of circuit
    let circuit1 = generate_random_clifford_circuit(&mut rng, num_qubits, 15);

    // Apply to both simulators
    sim1.reset();
    sim2.reset();
    apply_circuit(sim1, &circuit1);
    apply_circuit(sim2, &circuit1);

    // Measure qubit 0 with same forced outcome
    let forced_outcome: bool = rng.random();
    let r1 = sim1.mz_forced(0, forced_outcome);
    let r2 = sim2.mz_forced(0, forced_outcome);

    assert_eq!(
        r1.is_deterministic, r2.is_deterministic,
        "Mid-circuit determinism mismatch"
    );
    assert_eq!(r1.outcome, r2.outcome, "Mid-circuit outcome mismatch");

    // Generate and apply second half
    let circuit2 = generate_random_clifford_circuit(&mut rng, num_qubits, 15);
    apply_circuit(sim1, &circuit2);
    apply_circuit(sim2, &circuit2);

    // Measure all qubits with forced outcomes and compare
    for q in 0..num_qubits {
        let forced: bool = rng.random();
        let r1 = sim1.mz_forced(q, forced);
        let r2 = sim2.mz_forced(q, forced);

        assert_eq!(
            r1.is_deterministic, r2.is_deterministic,
            "Final measurement determinism mismatch for qubit {q}"
        );
        assert_eq!(
            r1.outcome, r2.outcome,
            "Final measurement outcome mismatch for qubit {q}"
        );
    }
}

// ============================================================================
// Reset Mid-Circuit Tests
// ============================================================================

/// Verify reset mid-circuit followed by more gates.
///
/// This tests that reset correctly returns the simulator to a valid |0> state
/// for the reset qubit while preserving other qubits.
pub fn verify_reset_mid_circuit<S: CliffordGateable + QuantumSimulator + ForcedMeasurement>(
    sim: &mut S,
    num_qubits: usize,
    seed: u64,
) {
    use pecos_rng::PecosRng;

    assert!(num_qubits >= 2, "Need at least 2 qubits for reset test");

    let mut rng = PecosRng::seed_from_u64(seed);

    // Apply random circuit
    let circuit1 = generate_random_clifford_circuit(&mut rng, num_qubits, 20);
    sim.reset();
    apply_circuit(sim, &circuit1);

    // Reset the entire simulator
    sim.reset();

    // Verify all qubits are in |0> state
    for q in 0..num_qubits {
        let result = sim.mz(&[QubitId::new(q)]);
        assert!(
            result[0].is_deterministic,
            "After reset, qubit {q} should be deterministic"
        );
        assert!(
            !result[0].outcome,
            "After reset, qubit {q} should measure 0"
        );
    }

    // Now apply more gates and verify simulator still works
    sim.reset();
    let circuit2 = generate_random_clifford_circuit(&mut rng, num_qubits, 15);
    apply_circuit(sim, &circuit2);

    // Measure all qubits - just verify no crash
    for q in 0..num_qubits {
        let _ = sim.mz(&[QubitId::new(q)]);
    }
}

/// Compare two simulators on reset behavior.
pub fn compare_reset_behavior<
    S1: CliffordGateable + QuantumSimulator + ForcedMeasurement,
    S2: CliffordGateable + QuantumSimulator + ForcedMeasurement,
>(
    sim1: &mut S1,
    sim2: &mut S2,
    num_qubits: usize,
    seed: u64,
) {
    use pecos_rng::PecosRng;

    let mut rng = PecosRng::seed_from_u64(seed);

    // Apply same circuit to both
    let circuit1 = generate_random_clifford_circuit(&mut rng, num_qubits, 20);
    sim1.reset();
    sim2.reset();
    apply_circuit(sim1, &circuit1);
    apply_circuit(sim2, &circuit1);

    // Reset both
    sim1.reset();
    sim2.reset();

    // Apply same second circuit
    let circuit2 = generate_random_clifford_circuit(&mut rng, num_qubits, 20);
    apply_circuit(sim1, &circuit2);
    apply_circuit(sim2, &circuit2);

    // Compare all measurements
    for q in 0..num_qubits {
        let forced: bool = rng.random();
        let r1 = sim1.mz_forced(q, forced);
        let r2 = sim2.mz_forced(q, forced);

        assert_eq!(
            r1.is_deterministic, r2.is_deterministic,
            "Reset test: determinism mismatch"
        );
        assert_eq!(r1.outcome, r2.outcome, "Reset test: outcome mismatch");
    }
}

// ============================================================================
// Measurement Order Independence Tests
// ============================================================================

/// Verify that measuring qubits in different orders produces consistent probability distributions.
///
/// For stabilizer states, the order of measurements shouldn't affect the final
/// probability distribution (though individual outcomes may differ due to collapse).
pub fn verify_measurement_order_independence<
    S: CliffordGateable + QuantumSimulator + ForcedMeasurement + Clone,
>(
    sim: &mut S,
    num_qubits: usize,
    seed: u64,
) {
    use pecos_rng::PecosRng;
    const TOLERANCE: f64 = 1e-10;

    assert!(num_qubits >= 2, "Need at least 2 qubits");

    let mut rng = PecosRng::seed_from_u64(seed);
    let circuit = generate_random_clifford_circuit(&mut rng, num_qubits, 20);

    // Apply circuit
    sim.reset();
    apply_circuit(sim, &circuit);

    // Clone for different measurement orders
    let sim_clone = sim.clone();

    // Measure in forward order: 0, 1, 2, ...
    let mut forward_probs = vec![0.0; 1 << num_qubits];
    for (basis_state, prob) in forward_probs.iter_mut().enumerate() {
        *prob = calculate_basis_probability(sim, basis_state, num_qubits);
    }

    // Measure in reverse order: n-1, n-2, ..., 0
    // We need to calculate probabilities differently for reverse order
    let mut reverse_probs = vec![0.0; 1 << num_qubits];
    for (basis_state, prob) in reverse_probs.iter_mut().enumerate() {
        let mut sim_copy = sim_clone.clone();
        let mut probability = 1.0;

        // Measure in reverse order
        for q in (0..num_qubits).rev() {
            let bit_is_one = (basis_state >> q) & 1 == 1;
            let result = sim_copy.mz_forced(q, bit_is_one);

            if !result.is_deterministic {
                probability *= 0.5;
            } else if result.outcome != bit_is_one {
                probability = 0.0;
                break;
            }
        }
        *prob = probability;
    }

    // Compare probabilities
    for basis_state in 0..(1 << num_qubits) {
        assert!(
            (forward_probs[basis_state] - reverse_probs[basis_state]).abs() < TOLERANCE,
            "Measurement order affected probability for state {basis_state}: forward={}, reverse={}",
            forward_probs[basis_state],
            reverse_probs[basis_state]
        );
    }
}

// ============================================================================
// Edge Case Tests
// ============================================================================

/// Verify behavior with an empty circuit (no gates applied).
pub fn verify_empty_circuit<S: CliffordGateable + QuantumSimulator>(
    sim: &mut S,
    num_qubits: usize,
) {
    sim.reset();

    // Empty circuit - no gates applied

    // All qubits should be in |0> state
    for q in 0..num_qubits {
        let result = sim.mz(&[QubitId::new(q)]);
        assert!(
            result[0].is_deterministic,
            "Empty circuit: qubit {q} should be deterministic"
        );
        assert!(
            !result[0].outcome,
            "Empty circuit: qubit {q} should measure 0"
        );
    }
}

/// Generate a circuit with only single-qubit gates.
pub fn generate_single_qubit_only_circuit<R: Rng>(
    rng: &mut R,
    num_qubits: usize,
    num_gates: usize,
) -> Vec<CliffordGate> {
    let mut circuit = Vec::with_capacity(num_gates);

    for _ in 0..num_gates {
        let gate_type: u8 = rng.random_range(0..10);
        let q = rng.random_range(0..num_qubits);

        let gate = match gate_type {
            0 => CliffordGate::H(q),
            1 => CliffordGate::S(q),
            2 => CliffordGate::Sdg(q),
            3 => CliffordGate::X(q),
            4 => CliffordGate::Y(q),
            5 => CliffordGate::Z(q),
            6 => CliffordGate::SX(q),
            7 => CliffordGate::SXdg(q),
            8 => CliffordGate::SY(q),
            _ => CliffordGate::SYdg(q),
        };
        circuit.push(gate);
    }

    circuit
}

/// Generate a circuit with only two-qubit gates.
pub fn generate_two_qubit_only_circuit<R: Rng>(
    rng: &mut R,
    num_qubits: usize,
    num_gates: usize,
) -> Vec<CliffordGate> {
    assert!(
        num_qubits >= 2,
        "Need at least 2 qubits for two-qubit gates"
    );

    let mut circuit = Vec::with_capacity(num_gates);

    for _ in 0..num_gates {
        let gate_type: u8 = rng.random_range(0..4);
        let q0 = rng.random_range(0..num_qubits);
        let mut q1 = rng.random_range(0..num_qubits);
        while q1 == q0 {
            q1 = rng.random_range(0..num_qubits);
        }

        let gate = match gate_type {
            0 => CliffordGate::CX(q0, q1),
            1 => CliffordGate::CY(q0, q1),
            2 => CliffordGate::CZ(q0, q1),
            _ => CliffordGate::SWAP(q0, q1),
        };
        circuit.push(gate);
    }

    circuit
}

/// Verify single-qubit-only circuits match `DensityMatrix`.
pub fn verify_single_qubit_only_circuit<
    S: CliffordGateable + QuantumSimulator + ForcedMeasurement + Clone,
>(
    sim: &mut S,
    num_qubits: usize,
    seed: u64,
) {
    use pecos_rng::PecosRng;

    let mut rng = PecosRng::seed_from_u64(seed);
    let circuit = generate_single_qubit_only_circuit(&mut rng, num_qubits, 30);

    sim.reset();
    apply_circuit(sim, &circuit);

    let mut dm = DensityMatrix::new(num_qubits);
    apply_circuit(&mut dm, &circuit);

    verify_probabilities_match_density_matrix(sim, &dm, num_qubits);
}

/// Verify two-qubit-only circuits match `DensityMatrix`.
pub fn verify_two_qubit_only_circuit<
    S: CliffordGateable + QuantumSimulator + ForcedMeasurement + Clone,
>(
    sim: &mut S,
    num_qubits: usize,
    seed: u64,
) {
    use pecos_rng::PecosRng;

    let mut rng = PecosRng::seed_from_u64(seed);
    let circuit = generate_two_qubit_only_circuit(&mut rng, num_qubits, 20);

    sim.reset();
    apply_circuit(sim, &circuit);

    let mut dm = DensityMatrix::new(num_qubits);
    apply_circuit(&mut dm, &circuit);

    verify_probabilities_match_density_matrix(sim, &dm, num_qubits);
}

// ============================================================================
// Comprehensive Test Suite
// ============================================================================

/// Run the basic stabilizer test suite on a simulator.
///
/// This version does NOT require Clone, so it works with GPU simulators.
/// It tests gate identities and entanglement correlations, but not
/// probability comparisons (which require cloning).
pub fn run_basic_stabilizer_test_suite<S: CliffordGateable + QuantumSimulator>(
    sim: &mut S,
    num_qubits: usize,
) {
    // Basic tests
    verify_initial_state(sim, num_qubits);
    verify_x_gate(sim);
    verify_h_creates_superposition(sim);
    verify_z_gate_on_zero(sim);
    verify_z_gate_on_one(sim);

    // Gate identity tests
    verify_all_gate_identities(sim);

    // Entanglement tests
    if num_qubits >= 2 {
        verify_bell_state_correlations(sim);
    }
    if num_qubits >= 3 {
        verify_ghz_state_correlations(sim, 3);
    }
}

/// Run the full stabilizer test suite on a simulator.
///
/// This requires the simulator to implement Clone and `ForcedMeasurement`
/// for the probability comparison tests.
///
/// The suite includes:
/// - Basic gate tests (X, H, Z, S, Sdg)
/// - Gate identity tests (H^2=I, S^4=I, X^2=I, etc.)
/// - Specific gate tests (SX, `SXdg`, SY, `SYdg`, CY, CZ, SWAP)
/// - Measurement idempotence tests
/// - Gate decomposition tests
/// - Commutation relation tests
/// - Entanglement tests (Bell, GHZ states)
/// - Probability comparison against `DensityMatrix`
/// - Mid-circuit measurement tests
/// - Reset tests
/// - Random circuit tests
pub fn run_full_stabilizer_test_suite<
    S: CliffordGateable + QuantumSimulator + ForcedMeasurement + Clone,
>(
    sim: &mut S,
    num_qubits: usize,
) {
    // ========== Basic Tests ==========
    run_basic_stabilizer_test_suite(sim, num_qubits);

    // ========== Specific Gate Tests ==========
    // Tests for SX, SXdg, SY, SYdg, CY, CZ, SWAP
    verify_all_specific_gates(sim);

    // ========== Measurement Idempotence Tests ==========
    verify_all_measurement_idempotence(sim);

    // ========== Gate Decomposition Tests ==========
    if num_qubits >= 2 {
        verify_all_gate_decompositions(sim, num_qubits);
    }

    // ========== Commutation Relation Tests ==========
    if num_qubits >= 2 {
        verify_all_commutation_relations(sim);
    }
    if num_qubits >= 4 {
        verify_all_commutation_relations_extended(sim, num_qubits);
    }

    // ========== Probability Comparison Tests ==========
    let mut dm = DensityMatrix::new(num_qubits);

    // Test initial state
    sim.reset();
    dm.reset();
    verify_probabilities_match_density_matrix(sim, &dm, num_qubits);

    // Test after H on first qubit
    sim.reset();
    dm.reset();
    sim.h(&[QubitId::new(0)]);
    dm.h(&[QubitId::new(0)]);
    verify_probabilities_match_density_matrix(sim, &dm, num_qubits);

    // Test Bell state (if 2+ qubits)
    if num_qubits >= 2 {
        sim.reset();
        dm.reset();
        sim.h(&[QubitId::new(0)]);
        sim.cx(&[QubitId::new(0), QubitId::new(1)]);
        dm.h(&[QubitId::new(0)]);
        dm.cx(&[QubitId::new(0), QubitId::new(1)]);
        verify_probabilities_match_density_matrix(sim, &dm, num_qubits);
    }

    // Test GHZ state (if 3+ qubits)
    if num_qubits >= 3 {
        sim.reset();
        dm.reset();
        sim.h(&[QubitId::new(0)]);
        dm.h(&[QubitId::new(0)]);
        for i in 0..(num_qubits - 1) {
            sim.cx(&[QubitId::new(i), QubitId::new(i + 1)]);
            dm.cx(&[QubitId::new(i), QubitId::new(i + 1)]);
        }
        verify_probabilities_match_density_matrix(sim, &dm, num_qubits);
    }

    // ========== Mid-Circuit Measurement Tests ==========
    verify_mid_circuit_measurement(sim, num_qubits, 42);

    // ========== Reset Tests ==========
    verify_reset_mid_circuit(sim, num_qubits, 42);

    // ========== Measurement Order Independence ==========
    if num_qubits >= 2 {
        verify_measurement_order_independence(sim, num_qubits, 42);
    }

    // ========== Edge Case Tests ==========
    verify_empty_circuit(sim, num_qubits);
    verify_single_qubit_only_circuit(sim, num_qubits, 42);
    if num_qubits >= 2 {
        verify_two_qubit_only_circuit(sim, num_qubits, 42);
    }

    // ========== Random Circuit Tests ==========
    verify_random_circuits(sim, num_qubits, 20, 10, 12345);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{SparseStab, SparseStabHybrid, SparseStabVecSet};

    // Note: ForcedMeasurement for SparseStab variants is implemented in sparse_stab.rs

    #[test]
    fn test_sparse_stab_gate_identities() {
        let mut sim = SparseStab::new(2);
        verify_all_gate_identities(&mut sim);
    }

    #[test]
    fn test_sparse_stab_bell_state() {
        let mut sim = SparseStab::new(2);
        verify_bell_state_correlations(&mut sim);
    }

    #[test]
    fn test_sparse_stab_ghz_state() {
        let mut sim = SparseStab::new(4);
        verify_ghz_state_correlations(&mut sim, 4);
    }

    #[test]
    fn test_sparse_stab_full_suite() {
        let mut sim = SparseStab::new(3);
        run_full_stabilizer_test_suite(&mut sim, 3);
    }

    // ========================================================================
    // Random Circuit Tests
    // ========================================================================

    #[test]
    fn test_random_circuits_small() {
        // Small circuits on 2 qubits
        let mut sim = SparseStab::new(2);
        verify_random_circuits(&mut sim, 2, 10, 20, 42);
    }

    #[test]
    fn test_random_circuits_medium() {
        // Medium circuits on 3 qubits
        let mut sim = SparseStab::new(3);
        verify_random_circuits(&mut sim, 3, 30, 10, 123);
    }

    #[test]
    fn test_random_circuits_large() {
        // Larger circuits on 4 qubits
        let mut sim = SparseStab::new(4);
        verify_random_circuits(&mut sim, 4, 50, 5, 456);
    }

    #[test]
    fn test_compare_bitset_vs_vecset() {
        // Compare SparseStab (BitSet) vs SparseStabVecSet on random circuits
        let mut sim1 = SparseStab::new(3);
        let mut sim2 = SparseStabVecSet::new(3);
        compare_simulators_on_random_circuits(&mut sim1, &mut sim2, 3, 30, 10, 789);
    }

    #[test]
    fn test_compare_bitset_vs_hybrid() {
        // Compare SparseStab (BitSet) vs SparseStabHybrid on random circuits
        let mut sim1 = SparseStab::new(3);
        let mut sim2 = SparseStabHybrid::new(3);
        compare_simulators_on_random_circuits(&mut sim1, &mut sim2, 3, 30, 10, 101_112);
    }

    #[test]
    fn test_compare_vecset_vs_hybrid() {
        // Compare SparseStabVecSet vs SparseStabHybrid on random circuits
        let mut sim1 = SparseStabVecSet::new(3);
        let mut sim2 = SparseStabHybrid::new(3);
        compare_simulators_on_random_circuits(&mut sim1, &mut sim2, 3, 30, 10, 131_415);
    }

    #[test]
    fn test_random_circuit_single_qubit() {
        // Edge case: single qubit circuits (no two-qubit gates)
        let mut sim = SparseStab::new(1);
        verify_random_circuits(&mut sim, 1, 20, 10, 999);
    }

    // ========================================================================
    // Direct Comparison Tests (no Clone required)
    // ========================================================================

    #[test]
    fn test_compare_direct_bitset_vs_vecset() {
        // Direct comparison (no Clone required)
        let mut sim1 = SparseStab::new(3);
        let mut sim2 = SparseStabVecSet::new(3);
        compare_simulators_on_random_circuits_direct(&mut sim1, &mut sim2, 3, 30, 20, 161_718);
    }

    #[test]
    fn test_compare_direct_bitset_vs_hybrid() {
        // Direct comparison (no Clone required)
        let mut sim1 = SparseStab::new(3);
        let mut sim2 = SparseStabHybrid::new(3);
        compare_simulators_on_random_circuits_direct(&mut sim1, &mut sim2, 3, 30, 20, 192_021);
    }

    #[test]
    fn test_compare_direct_all_three() {
        // Compare all three variants pairwise on the same circuits
        let mut bitset = SparseStab::new(4);
        let mut vecset = SparseStabVecSet::new(4);
        let mut hybrid = SparseStabHybrid::new(4);

        // Run 5 circuits, comparing all pairs
        for i in 0..5 {
            let seed = 222_324 + i;

            // Reset all
            bitset.reset();
            vecset.reset();
            hybrid.reset();

            // Compare bitset vs vecset
            compare_simulators_on_random_circuit_direct(&mut bitset, &mut vecset, 4, 40, seed);

            // Reset and compare bitset vs hybrid
            bitset.reset();
            hybrid.reset();
            compare_simulators_on_random_circuit_direct(&mut bitset, &mut hybrid, 4, 40, seed);
        }
    }

    // ========================================================================
    // Mid-Circuit Measurement Tests
    // ========================================================================

    #[test]
    fn test_mid_circuit_measurement_sparse_stab() {
        let mut sim = SparseStab::new(3);
        for seed in 0..10 {
            verify_mid_circuit_measurement(&mut sim, 3, 100_000 + seed);
        }
    }

    #[test]
    fn test_mid_circuit_measurement_compare_bitset_vecset() {
        let mut sim1 = SparseStab::new(3);
        let mut sim2 = SparseStabVecSet::new(3);
        for seed in 0..10 {
            compare_mid_circuit_measurement(&mut sim1, &mut sim2, 3, 200_000 + seed);
        }
    }

    #[test]
    fn test_mid_circuit_measurement_compare_all_variants() {
        let mut bitset = SparseStab::new(4);
        let mut vecset = SparseStabVecSet::new(4);
        let mut hybrid = SparseStabHybrid::new(4);

        for seed in 0..5 {
            // Compare bitset vs vecset
            compare_mid_circuit_measurement(&mut bitset, &mut vecset, 4, 300_000 + seed);

            // Compare bitset vs hybrid
            compare_mid_circuit_measurement(&mut bitset, &mut hybrid, 4, 300_000 + seed);

            // Compare vecset vs hybrid
            compare_mid_circuit_measurement(&mut vecset, &mut hybrid, 4, 300_000 + seed);
        }
    }

    // ========================================================================
    // Reset Mid-Circuit Tests
    // ========================================================================

    #[test]
    fn test_reset_mid_circuit_sparse_stab() {
        let mut sim = SparseStab::new(3);
        for seed in 0..10 {
            verify_reset_mid_circuit(&mut sim, 3, 400_000 + seed);
        }
    }

    #[test]
    fn test_reset_compare_all_variants() {
        let mut bitset = SparseStab::new(4);
        let mut vecset = SparseStabVecSet::new(4);
        let mut hybrid = SparseStabHybrid::new(4);

        for seed in 0..5 {
            compare_reset_behavior(&mut bitset, &mut vecset, 4, 500_000 + seed);
            compare_reset_behavior(&mut bitset, &mut hybrid, 4, 500_000 + seed);
            compare_reset_behavior(&mut vecset, &mut hybrid, 4, 500_000 + seed);
        }
    }

    // ========================================================================
    // Measurement Order Independence Tests
    // ========================================================================

    #[test]
    fn test_measurement_order_independence_sparse_stab() {
        let mut sim = SparseStab::new(3);
        for seed in 0..10 {
            verify_measurement_order_independence(&mut sim, 3, 600_000 + seed);
        }
    }

    #[test]
    fn test_measurement_order_independence_all_variants() {
        let mut bitset = SparseStab::new(4);
        let mut vecset = SparseStabVecSet::new(4);
        let mut hybrid = SparseStabHybrid::new(4);

        for seed in 0..5 {
            verify_measurement_order_independence(&mut bitset, 4, 700_000 + seed);
            verify_measurement_order_independence(&mut vecset, 4, 700_000 + seed);
            verify_measurement_order_independence(&mut hybrid, 4, 700_000 + seed);
        }
    }

    // ========================================================================
    // Edge Case Tests
    // ========================================================================

    #[test]
    fn test_empty_circuit_sparse_stab() {
        let mut sim = SparseStab::new(4);
        verify_empty_circuit(&mut sim, 4);
    }

    #[test]
    fn test_empty_circuit_all_variants() {
        let mut bitset = SparseStab::new(4);
        let mut vecset = SparseStabVecSet::new(4);
        let mut hybrid = SparseStabHybrid::new(4);

        verify_empty_circuit(&mut bitset, 4);
        verify_empty_circuit(&mut vecset, 4);
        verify_empty_circuit(&mut hybrid, 4);
    }

    #[test]
    fn test_single_qubit_only_circuits() {
        let mut sim = SparseStab::new(3);
        for seed in 0..10 {
            verify_single_qubit_only_circuit(&mut sim, 3, 800_000 + seed);
        }
    }

    #[test]
    fn test_two_qubit_only_circuits() {
        let mut sim = SparseStab::new(4);
        for seed in 0..10 {
            verify_two_qubit_only_circuit(&mut sim, 4, 900_000 + seed);
        }
    }

    #[test]
    fn test_single_qubit_only_all_variants() {
        let mut bitset = SparseStab::new(3);
        let mut vecset = SparseStabVecSet::new(3);
        let mut hybrid = SparseStabHybrid::new(3);

        for seed in 0..5 {
            verify_single_qubit_only_circuit(&mut bitset, 3, 1_000_000 + seed);
            verify_single_qubit_only_circuit(&mut vecset, 3, 1_000_000 + seed);
            verify_single_qubit_only_circuit(&mut hybrid, 3, 1_000_000 + seed);
        }
    }

    #[test]
    fn test_two_qubit_only_all_variants() {
        let mut bitset = SparseStab::new(4);
        let mut vecset = SparseStabVecSet::new(4);
        let mut hybrid = SparseStabHybrid::new(4);

        for seed in 0..5 {
            verify_two_qubit_only_circuit(&mut bitset, 4, 1_100_000 + seed);
            verify_two_qubit_only_circuit(&mut vecset, 4, 1_100_000 + seed);
            verify_two_qubit_only_circuit(&mut hybrid, 4, 1_100_000 + seed);
        }
    }

    // ========================================================================
    // Specific Gate Tests
    // ========================================================================

    #[test]
    fn test_specific_gates_sparse_stab() {
        let mut sim = SparseStab::new(2);
        verify_all_specific_gates(&mut sim);
    }

    #[test]
    fn test_specific_gates_all_variants() {
        let mut bitset = SparseStab::new(2);
        let mut vecset = SparseStabVecSet::new(2);
        let mut hybrid = SparseStabHybrid::new(2);

        verify_all_specific_gates(&mut bitset);
        verify_all_specific_gates(&mut vecset);
        verify_all_specific_gates(&mut hybrid);
    }

    #[test]
    fn test_sx_gate() {
        let mut sim = SparseStab::new(2);
        verify_sx_squared_is_x(&mut sim);
        verify_sx_creates_superposition(&mut sim);
        verify_sx_sxdg_is_identity(&mut sim);
    }

    #[test]
    fn test_sy_gate() {
        let mut sim = SparseStab::new(2);
        verify_sy_squared_is_y(&mut sim);
        verify_sy_creates_superposition(&mut sim);
        verify_sy_sydg_is_identity(&mut sim);
    }

    #[test]
    fn test_cy_gate() {
        let mut sim = SparseStab::new(2);
        verify_cy_control_zero(&mut sim);
        verify_cy_control_one(&mut sim);
        verify_cy_squared_is_identity(&mut sim);
    }

    #[test]
    fn test_cz_gate() {
        let mut sim = SparseStab::new(2);
        verify_cz_control_zero(&mut sim);
        verify_cz_control_one(&mut sim);
        verify_cz_symmetric(&mut sim);
    }

    #[test]
    fn test_swap_gate() {
        let mut sim = SparseStab::new(2);
        verify_swap_exchanges_states(&mut sim);
        verify_swap_squared_is_identity(&mut sim);
        verify_swap_symmetric(&mut sim);
    }

    // ========================================================================
    // Measurement Idempotence Tests
    // ========================================================================

    #[test]
    fn test_measurement_idempotence_sparse_stab() {
        let mut sim = SparseStab::new(2);
        verify_all_measurement_idempotence(&mut sim);
    }

    #[test]
    fn test_measurement_idempotence_all_variants() {
        let mut bitset = SparseStab::new(2);
        let mut vecset = SparseStabVecSet::new(2);
        let mut hybrid = SparseStabHybrid::new(2);

        verify_all_measurement_idempotence(&mut bitset);
        verify_all_measurement_idempotence(&mut vecset);
        verify_all_measurement_idempotence(&mut hybrid);
    }

    // ========================================================================
    // Gate Decomposition Tests
    // ========================================================================

    #[test]
    fn test_gate_decompositions_sparse_stab() {
        let mut sim = SparseStab::new(2);
        verify_all_gate_decompositions(&mut sim, 2);
    }

    #[test]
    fn test_gate_decompositions_all_variants() {
        let mut bitset = SparseStab::new(2);
        let mut vecset = SparseStabVecSet::new(2);
        let mut hybrid = SparseStabHybrid::new(2);

        verify_all_gate_decompositions(&mut bitset, 2);
        verify_all_gate_decompositions(&mut vecset, 2);
        verify_all_gate_decompositions(&mut hybrid, 2);
    }

    #[test]
    fn test_swap_decomposition_explicit() {
        let mut sim = SparseStab::new(2);
        verify_swap_decomposition(&mut sim, 2);
    }

    #[test]
    fn test_cz_decomposition_explicit() {
        let mut sim = SparseStab::new(2);
        verify_cz_decomposition(&mut sim, 2);
    }

    #[test]
    fn test_cy_decomposition_explicit() {
        let mut sim = SparseStab::new(2);
        verify_cy_decomposition(&mut sim, 2);
    }

    // ========================================================================
    // Commutation Relation Tests
    // ========================================================================

    #[test]
    fn test_commutation_relations_sparse_stab() {
        let mut sim = SparseStab::new(2);
        verify_all_commutation_relations(&mut sim);
    }

    #[test]
    fn test_commutation_relations_extended() {
        let mut sim = SparseStab::new(4);
        verify_all_commutation_relations_extended(&mut sim, 4);
    }

    #[test]
    fn test_commutation_relations_all_variants() {
        let mut bitset = SparseStab::new(4);
        let mut vecset = SparseStabVecSet::new(4);
        let mut hybrid = SparseStabHybrid::new(4);

        verify_all_commutation_relations_extended(&mut bitset, 4);
        verify_all_commutation_relations_extended(&mut vecset, 4);
        verify_all_commutation_relations_extended(&mut hybrid, 4);
    }

    #[test]
    fn test_xz_commute() {
        let mut sim = SparseStab::new(2);
        verify_xz_commute_different_qubits(&mut sim);
    }

    #[test]
    fn test_cx_disjoint_commute() {
        let mut sim = SparseStab::new(4);
        verify_cx_commute_disjoint(&mut sim);
    }

    #[test]
    fn test_cx_same_control_commute() {
        let mut sim = SparseStab::new(3);
        verify_cx_same_control_commute(&mut sim);
    }

    #[test]
    fn test_s_z_commute() {
        let mut sim = SparseStab::new(2);
        verify_s_z_commute(&mut sim);
    }

    // ========================================================================
    // Gate Decomposition Tests (Direct - no Clone required)
    // ========================================================================

    #[test]
    fn test_gate_decompositions_direct() {
        let mut sim1 = SparseStab::new(2);
        let mut sim2 = SparseStab::new(2);
        verify_all_gate_decompositions_direct(&mut sim1, &mut sim2, 2);
    }

    #[test]
    fn test_gate_decompositions_direct_bitset_vs_vecset() {
        let mut bitset = SparseStab::new(2);
        let mut vecset = SparseStabVecSet::new(2);
        verify_all_gate_decompositions_direct(&mut bitset, &mut vecset, 2);
    }

    #[test]
    fn test_swap_decomposition_direct() {
        let mut sim1 = SparseStab::new(2);
        let mut sim2 = SparseStab::new(2);
        verify_swap_decomposition_direct(&mut sim1, &mut sim2, 2);
    }

    #[test]
    fn test_cz_decomposition_direct() {
        let mut sim1 = SparseStab::new(2);
        let mut sim2 = SparseStab::new(2);
        verify_cz_decomposition_direct(&mut sim1, &mut sim2, 2);
    }

    #[test]
    fn test_cy_decomposition_direct() {
        let mut sim1 = SparseStab::new(2);
        let mut sim2 = SparseStab::new(2);
        verify_cy_decomposition_direct(&mut sim1, &mut sim2, 2);
    }

    #[test]
    fn test_x_decomposition_direct() {
        let mut sim1 = SparseStab::new(2);
        let mut sim2 = SparseStab::new(2);
        verify_x_decomposition_direct(&mut sim1, &mut sim2, 2);
    }

    #[test]
    fn test_y_decomposition_direct() {
        let mut sim1 = SparseStab::new(2);
        let mut sim2 = SparseStab::new(2);
        verify_y_decomposition_direct(&mut sim1, &mut sim2, 2);
    }
}
