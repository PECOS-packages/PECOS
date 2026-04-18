// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! Measurement stress tests for any `ArbitraryRotationGateable` simulator.
//!
//! These tests exercise measurement-related edge cases discovered during
//! STN (Stabilizer Tensor Network) development. They use only measurement
//! outcomes (no state vector access), so any simulator can use them.
//!
//! Test categories:
//! - Re-measurement consistency: measure, then re-measure the same qubit
//! - Measure-gate-measure: measure, apply gates, measure again
//! - Clifford rotations after non-Clifford: RX(pi)/RZ(pi) after T gates
//! - Negative-angle rotations: Tdg, negative RZ/RX angles

#![allow(clippy::missing_panics_doc)]

use crate::ArbitraryRotationGateable;
use pecos_core::{Angle64, QubitId, qid};

// ============================================================================
// Re-measurement consistency
// ============================================================================

/// After measuring a qubit, re-measuring it should give the same outcome.
/// This verifies that measurement collapse is implemented correctly.
pub fn verify_remeasurement_consistency<S: ArbitraryRotationGateable>(sim: &mut S) {
    let t = Angle64::QUARTER_TURN / 2u64;

    // Case 1: T|+> then measure twice
    sim.reset();
    sim.h(&qid(0));
    sim.rz(t, &qid(0));
    let r1 = sim.mz(&qid(0))[0].outcome;
    let r2 = sim.mz(&qid(0))[0].outcome;
    assert_eq!(r1, r2, "T|+>: re-measurement should give same outcome");

    // Case 2: Bell+T then measure both, re-measure first
    {
        sim.reset();
        sim.h(&qid(0));
        sim.cx(&[(QubitId(0), QubitId(1))]);
        sim.rz(t, &qid(0));
        let r0 = sim.mz(&qid(0))[0].outcome;
        let _r1 = sim.mz(&qid(1))[0].outcome;
        let r0_again = sim.mz(&qid(0))[0].outcome;
        assert_eq!(
            r0, r0_again,
            "Bell+T: re-measurement of q0 should be stable"
        );
    }
}

// ============================================================================
// Measure-gate-measure
// ============================================================================

/// Measure a qubit, apply more gates (including non-Clifford), then measure
/// again. Verifies the post-measurement state is usable.
pub fn verify_measure_gate_measure<S: ArbitraryRotationGateable>(sim: &mut S) {
    let t = Angle64::QUARTER_TURN / 2u64;

    // Measure q0, then apply T+H on q1, then measure q1
    sim.reset();
    sim.h(&qid(0));
    sim.cx(&[(QubitId(0), QubitId(1))]);
    sim.rz(t, &qid(0));

    let _r0 = sim.mz(&qid(0))[0].outcome;

    // After measuring q0, apply gates on q1
    sim.h(&qid(1));
    sim.rz(t, &qid(1));
    let _r1 = sim.mz(&qid(1)); // Should not panic
}

/// Multiple rounds of measure-gate-measure on a 3-qubit system.
pub fn verify_measure_gate_measure_3qubit<S: ArbitraryRotationGateable>(sim: &mut S) {
    let t = Angle64::QUARTER_TURN / 2u64;

    sim.reset();
    sim.h(&qid(0));
    sim.cx(&[(QubitId(0), QubitId(1))]);
    sim.cx(&[(QubitId(1), QubitId(2))]);
    sim.rz(t, &qid(1));

    // Measure q0
    let _r0 = sim.mz(&qid(0))[0].outcome;

    // Apply more gates after measurement
    sim.h(&qid(1));
    sim.rz(t, &qid(1));

    // Measure q1 and q2
    let _r1 = sim.mz(&qid(1))[0].outcome;
    let _r2 = sim.mz(&qid(2))[0].outcome;
}

// ============================================================================
// Clifford rotations after non-Clifford gates
// ============================================================================

/// RX(pi) = -i*X after non-Clifford gates. The Clifford-angle detection
/// path in RZ must handle the case where the MPS already has non-Clifford
/// content. Tests the X/Y/Z gate destab sign tracking.
pub fn verify_rx_pi_after_nonclifford<S: ArbitraryRotationGateable>(sim: &mut S) {
    let t = Angle64::QUARTER_TURN / 2u64;
    let pi = Angle64::from_radians(std::f64::consts::PI);

    // H, T, RX(pi) on single qubit
    sim.reset();
    sim.h(&qid(0));
    sim.rz(t, &qid(0));
    sim.rx(pi, &qid(0));
    // RX(pi)*T|+> should be measurable without panic
    let _r = sim.mz(&qid(0));

    // H, T, then RZ(pi) (= Z up to phase)
    sim.reset();
    sim.h(&qid(0));
    sim.rz(t, &qid(0));
    sim.rz(Angle64::HALF_TURN, &qid(0));
    let _r = sim.mz(&qid(0));

    // Entangled case: Bell + T + RX(pi)
    {
        sim.reset();
        sim.h(&qid(0));
        sim.cx(&[(QubitId(0), QubitId(1))]);
        sim.rz(t, &qid(0));
        sim.rz(t, &qid(1));
        sim.rx(pi, &qid(0));

        let r0 = sim.mz(&qid(0))[0].outcome;
        let r1 = sim.mz(&qid(1))[0].outcome;
        // RX(pi)=X flips one qubit of the Bell pair: outcomes are anti-correlated
        assert_ne!(r0, r1, "Bell+T+RX(pi): outcomes should be anti-correlated");
    }
}

/// RZ at all Clifford angles after non-Clifford gates.
pub fn verify_rz_clifford_angles_after_nonclifford<S: ArbitraryRotationGateable>(sim: &mut S) {
    let t = Angle64::QUARTER_TURN / 2u64;

    let clifford_angles = [
        Angle64::ZERO,
        Angle64::QUARTER_TURN,
        Angle64::HALF_TURN,
        Angle64::THREE_QUARTERS_TURN,
    ];

    for &angle in &clifford_angles {
        sim.reset();
        sim.h(&qid(0));
        sim.rz(t, &qid(0)); // Non-Clifford first
        sim.rz(angle, &qid(0)); // Clifford angle after
        let _r = sim.mz(&qid(0)); // Should not panic
    }
}

// ============================================================================
// Negative-angle rotations
// ============================================================================

/// Tdg = RZ(-pi/4). Verify T * Tdg = I and Tdg alone works.
pub fn verify_tdg_basic<S: ArbitraryRotationGateable>(sim: &mut S) {
    let t = Angle64::QUARTER_TURN / 2u64;
    let tdg = -t;

    // T * Tdg = I on |+>
    sim.reset();
    sim.h(&qid(0));
    sim.rz(t, &qid(0));
    sim.rz(tdg, &qid(0));
    // Should be back in |+>, deterministic X measurement
    let r = sim.mx(&qid(0));
    assert!(
        r[0].is_deterministic,
        "T*Tdg|+> should be deterministic in X"
    );
    assert!(!r[0].outcome, "T*Tdg|+> should measure 0 in X");

    // Tdg alone on |+>: p(0) = p(1) = 0.5. Just verify no panic.
    sim.reset();
    sim.h(&qid(0));
    sim.rz(tdg, &qid(0));
    let _r = sim.mz(&qid(0));
}

/// Negative-angle rotations produce valid states.
pub fn verify_negative_angle_rotations<S: ArbitraryRotationGateable>(sim: &mut S) {
    let t = Angle64::QUARTER_TURN / 2u64;
    let tdg = -t;

    // Tdg on |0>: deterministic (Z is stabilizer), should always measure 0
    sim.reset();
    sim.rz(tdg, &qid(0));
    let r = sim.mz(&qid(0));
    assert!(!r[0].outcome, "Tdg|0> should measure 0");

    // Tdg on |+> produces valid state (no panic, non-deterministic in Z)
    sim.reset();
    sim.h(&qid(0));
    sim.rz(tdg, &qid(0));
    let _r = sim.mz(&qid(0));

    // Negative-angle RX on |0> (no panic)
    sim.reset();
    sim.rx(-t, &qid(0));
    let _r = sim.mz(&qid(0));
}

// ============================================================================
// Measurement probability distribution check
// ============================================================================

/// Statistical check: RX(pi/3)|0> should give p(0) = cos^2(pi/6) = 3/4.
/// Runs many trials and checks the distribution.
pub fn verify_rx_measurement_probabilities<S: ArbitraryRotationGateable>(
    sim: &mut S,
    num_trials: usize,
) {
    let theta = Angle64::from_radians(std::f64::consts::FRAC_PI_3);
    let expected_p0 = 0.75;
    let mut count_0 = 0u32;

    for _ in 0..num_trials {
        sim.reset();
        sim.rx(theta, &qid(0));
        if !sim.mz(&qid(0))[0].outcome {
            count_0 += 1;
        }
    }

    let p0 = f64::from(count_0)
        / f64::from(u32::try_from(num_trials).expect("num_trials must fit in u32"));
    assert!(
        (p0 - expected_p0).abs() < 0.1,
        "RX(pi/3)|0> p(0) = {p0:.3}, expected {expected_p0:.3}"
    );
}

// ============================================================================
// Main runner
// ============================================================================

/// Run the full measurement stress test suite.
pub fn run_measurement_stress_tests<S: ArbitraryRotationGateable>(sim: &mut S) {
    verify_remeasurement_consistency(sim);
    verify_measure_gate_measure(sim);
    verify_measure_gate_measure_3qubit(sim);
    verify_rx_pi_after_nonclifford(sim);
    verify_rz_clifford_angles_after_nonclifford(sim);
    verify_tdg_basic(sim);
    verify_negative_angle_rotations(sim);
    verify_rx_measurement_probabilities(sim, 200);
}

/// Generate a test that runs the measurement stress suite on a simulator type.
///
/// Usage:
/// ```ignore
/// use pecos_simulators::measurement_stress_test_suite;
/// measurement_stress_test_suite!(StabVec, 4);
/// ```
#[macro_export]
macro_rules! measurement_stress_test_suite {
    ($sim_type:ty) => {
        $crate::measurement_stress_test_suite!($sim_type, 4);
    };
    ($sim_type:ty, $num_qubits:expr) => {
        paste::paste! {
            #[test]
            fn [<test_ $sim_type:snake _measurement_stress>]() {
                use $crate::measurement_stress_test_utils::run_measurement_stress_tests;
                let mut sim = <$sim_type>::builder($num_qubits).seed(42).build();
                run_measurement_stress_tests(&mut sim);
            }
        }
    };
    ($sim_type:ty, $num_qubits:expr, $constructor:expr) => {
        paste::paste! {
            #[test]
            fn [<test_ $sim_type:snake _measurement_stress>]() {
                use $crate::measurement_stress_test_utils::run_measurement_stress_tests;
                #[allow(unused_variables)]
                let num_qubits: usize = $num_qubits;
                let mut sim = $constructor;
                run_measurement_stress_tests(&mut sim);
            }
        }
    };
}
