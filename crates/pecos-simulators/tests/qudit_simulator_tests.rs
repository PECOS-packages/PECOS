// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the
// License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
// either express or implied. See the License for the specific language governing permissions and
// limitations under the License.

use num_complex::Complex64;
use pecos_random::PecosRng;
use pecos_simulators::{
    QuditDensityMatrix, QuditError, QuditStateVec, QutritDensityMatrix, QutritStateVec, basis_swap,
    qutrit_leakage_channel, qutrit_seepage_channel,
};
use std::f64::consts::FRAC_1_SQRT_2;

const TOLERANCE: f64 = 1e-10;

fn c(value: f64) -> Complex64 {
    Complex64::new(value, 0.0)
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < TOLERANCE,
        "expected {expected}, received {actual}"
    );
}

fn pure_density(state: &[Complex64]) -> Vec<Complex64> {
    let mut density = Vec::with_capacity(state.len() * state.len());
    for row in state {
        for column in state {
            density.push(*row * column.conj());
        }
    }
    density
}

#[test]
fn constructors_validate_dimension_and_normalization() {
    assert_eq!(
        QuditStateVec::with_seed(1, 1, 7).unwrap_err(),
        QuditError::InvalidLocalDimension(1)
    );
    assert_eq!(
        QuditStateVec::from_state(
            1,
            3,
            vec![c(1.0), c(1.0), c(0.0)],
            PecosRng::seed_from_u64(7),
        )
        .unwrap_err(),
        QuditError::NotNormalized { norm: 2.0 }
    );

    let state = QutritStateVec::qutrit_with_seed(2, 7).unwrap();
    assert_eq!(state.num_sites(), 2);
    assert_eq!(state.local_dimension(), 3);
    assert_eq!(state.dimension(), 9);
    assert_close(state.probability(0).unwrap(), 1.0);
}

#[test]
fn target_and_matrix_order_follow_little_endian_radix_digits() {
    let mut state = QutritStateVec::qutrit_with_seed(2, 7).unwrap();
    state
        .apply_operator(&[1], &basis_swap(3, 0, 1).unwrap())
        .unwrap();
    assert_close(state.probability(3).unwrap(), 1.0);

    // targets[0] is the least-significant local digit. Local state 1 becomes 2,
    // which changes site 1 from |1> to |L> while site 0 remains |0>.
    state
        .apply_operator(&[1, 0], &basis_swap(9, 1, 2).unwrap())
        .unwrap();
    assert_close(state.probability(6).unwrap(), 1.0);
}

#[test]
fn embedded_qubit_gates_leave_the_leakage_level_unchanged() {
    let h = [
        c(FRAC_1_SQRT_2),
        c(FRAC_1_SQRT_2),
        c(FRAC_1_SQRT_2),
        c(-FRAC_1_SQRT_2),
    ];
    let mut state = QutritStateVec::qutrit_with_seed(1, 7).unwrap();
    state
        .apply_operator(&[0], &basis_swap(3, 0, 2).unwrap())
        .unwrap()
        .apply_embedded_qubit_unitary(0, &h)
        .unwrap();
    assert_close(state.probability(2).unwrap(), 1.0);
}

#[test]
fn state_vector_and_density_matrix_agree_for_coherent_qutrit_evolution() {
    let fourier_scale = 1.0 / 3.0_f64.sqrt();
    let omega = Complex64::from_polar(1.0, 2.0 * std::f64::consts::PI / 3.0);
    let fourier = vec![
        c(fourier_scale),
        c(fourier_scale),
        c(fourier_scale),
        c(fourier_scale),
        omega * fourier_scale,
        omega.powu(2) * fourier_scale,
        c(fourier_scale),
        omega.powu(2) * fourier_scale,
        omega * fourier_scale,
    ];
    let mut state = QutritStateVec::qutrit_with_seed(2, 11).unwrap();
    let mut density = QutritDensityMatrix::qutrit_with_seed(2, 11).unwrap();
    for target in [0, 1] {
        state.apply_operator(&[target], &fourier).unwrap();
        density.apply_operator(&[target], &fourier).unwrap();
    }
    let exchange = basis_swap(9, 2, 7).unwrap();
    state.apply_operator(&[0, 1], &exchange).unwrap();
    density.apply_operator(&[0, 1], &exchange).unwrap();

    let expected = pure_density(state.state());
    for (actual, expected) in density.density_matrix().iter().zip(expected) {
        assert!((*actual - expected).norm() < TOLERANCE);
    }
    assert!(density.diagnostics().is_physical(TOLERANCE));
}

#[test]
fn exact_leakage_and_seepage_channels_have_analytic_distributions() {
    let h = [
        c(FRAC_1_SQRT_2),
        c(FRAC_1_SQRT_2),
        c(FRAC_1_SQRT_2),
        c(-FRAC_1_SQRT_2),
    ];
    let mut density = QutritDensityMatrix::qutrit_with_seed(1, 13).unwrap();
    density
        .apply_embedded_qubit_unitary(0, &h)
        .unwrap()
        .apply_kraus(&[0], &qutrit_leakage_channel(0.3).unwrap())
        .unwrap();
    let probabilities = density.outcome_probabilities(0).unwrap();
    assert_close(probabilities[0], 0.35);
    assert_close(probabilities[1], 0.35);
    assert_close(probabilities[2], 0.3);

    let mut leaked = QutritDensityMatrix::qutrit_with_seed(1, 13).unwrap();
    leaked
        .apply_operator(&[0], &basis_swap(3, 0, 2).unwrap())
        .unwrap()
        .apply_kraus(&[0], &qutrit_seepage_channel(0.4, 0.25).unwrap())
        .unwrap();
    let probabilities = leaked.outcome_probabilities(0).unwrap();
    assert_close(probabilities[0], 0.1);
    assert_close(probabilities[1], 0.3);
    assert_close(probabilities[2], 0.6);
    assert!(leaked.diagnostics().is_physical(TOLERANCE));
}

#[test]
fn state_vector_trajectories_converge_to_exact_kraus_evolution() {
    let h = [
        c(FRAC_1_SQRT_2),
        c(FRAC_1_SQRT_2),
        c(FRAC_1_SQRT_2),
        c(-FRAC_1_SQRT_2),
    ];
    let channel = qutrit_leakage_channel(0.3).unwrap();
    let mut counts = [0_u32; 3];
    let shots = 12_000_u32;
    for seed in 0..shots {
        let mut state = QutritStateVec::qutrit_with_seed(1, u64::from(seed)).unwrap();
        state.apply_embedded_qubit_unitary(0, &h).unwrap();
        state.apply_kraus(&[0], &channel).unwrap();
        counts[state.measure(0).unwrap()] += 1;
    }
    let observed = counts.map(|count| f64::from(count) / f64::from(shots));
    for (actual, expected) in observed.into_iter().zip([0.35, 0.35, 0.3]) {
        assert!((actual - expected).abs() < 0.015, "{observed:?}");
    }
}

#[test]
fn two_site_kraus_channels_work_in_both_backends() {
    let mut identity = vec![c(0.0); 81];
    for index in 0..9 {
        identity[index * 9 + index] = c(0.7_f64.sqrt());
    }
    let mut exchange = basis_swap(9, 1, 3).unwrap();
    for value in &mut exchange {
        *value *= 0.3_f64.sqrt();
    }
    let channel = vec![identity, exchange];

    let mut density = QutritDensityMatrix::qutrit_with_seed(2, 17).unwrap();
    density.prepare_basis(0, 1).unwrap();
    density.apply_kraus(&[0, 1], &channel).unwrap();
    assert_close(density.probability(1).unwrap(), 0.7);
    assert_close(density.probability(3).unwrap(), 0.3);

    let mut exchanged = 0_u32;
    let shots = 8_000_u32;
    for seed in 0..shots {
        let mut state = QutritStateVec::qutrit_with_seed(2, u64::from(seed)).unwrap();
        state.prepare_basis(0, 1).unwrap();
        state.apply_kraus(&[0, 1], &channel).unwrap();
        if state.probability(3).unwrap() > 0.5 {
            exchanged += 1;
        }
    }
    let observed = f64::from(exchanged) / f64::from(shots);
    assert!((observed - 0.3).abs() < 0.02, "observed {observed}");
}

#[test]
fn seeded_trajectory_execution_is_reproducible() {
    let channel = qutrit_leakage_channel(0.37).unwrap();
    let run = || {
        let mut state = QutritStateVec::qutrit_with_seed(1, 18_273).unwrap();
        (0..64)
            .map(|_| {
                state.prepare_basis(0, 1).unwrap();
                let branch = state.apply_kraus(&[0], &channel).unwrap();
                let outcome = state.measure(0).unwrap();
                (branch, outcome)
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(run(), run());
}

#[test]
fn measurement_collapses_and_reset_returns_zero() {
    let mut state = QutritStateVec::qutrit_with_seed(1, 19).unwrap();
    let equal = 1.0 / 3.0_f64.sqrt();
    let fourier = vec![
        c(equal),
        c(equal),
        c(equal),
        c(equal),
        Complex64::from_polar(equal, 2.0 * std::f64::consts::PI / 3.0),
        Complex64::from_polar(equal, 4.0 * std::f64::consts::PI / 3.0),
        c(equal),
        Complex64::from_polar(equal, 4.0 * std::f64::consts::PI / 3.0),
        Complex64::from_polar(equal, 2.0 * std::f64::consts::PI / 3.0),
    ];
    state.apply_operator(&[0], &fourier).unwrap();
    let outcome = state.measure(0).unwrap();
    assert_close(state.probability(outcome).unwrap(), 1.0);
    state.reset(0).unwrap();
    assert_close(state.probability(0).unwrap(), 1.0);
}

#[test]
fn computational_measurement_is_strict_about_leakage() {
    let mut state = QutritStateVec::qutrit_with_seed(1, 20).unwrap();
    state.prepare_basis(0, 1).unwrap();
    assert!(state.measure_computational(0).unwrap());
    state.prepare_basis(0, 2).unwrap();
    assert!(matches!(
        state.measure_computational(0).unwrap_err(),
        QuditError::LeakagePopulation { probability } if (probability - 1.0).abs() < TOLERANCE
    ));
    assert_eq!(state.measure(0).unwrap(), 2);

    let mut density = QutritDensityMatrix::qutrit_with_seed(1, 20).unwrap();
    density.prepare_basis(0, 2).unwrap();
    assert!(matches!(
        density.measure_computational(0).unwrap_err(),
        QuditError::LeakagePopulation { probability } if (probability - 1.0).abs() < TOLERANCE
    ));
    assert_close(density.probability(2).unwrap(), 1.0);
}

#[test]
fn exact_reset_discards_entanglement_and_reduced_state_preserves_mixture() {
    let mut state = vec![c(0.0); 9];
    state[0] = c(FRAC_1_SQRT_2);
    state[4] = c(FRAC_1_SQRT_2);
    let mut density = QuditDensityMatrix::from_density_matrix(
        2,
        3,
        pure_density(&state),
        PecosRng::seed_from_u64(23),
    )
    .unwrap();
    density.reset(0).unwrap();
    let reset_probabilities = density.outcome_probabilities(0).unwrap();
    assert_close(reset_probabilities[0], 1.0);
    assert_close(reset_probabilities[1], 0.0);
    assert_close(reset_probabilities[2], 0.0);
    let reduced = density.reduced_density_matrix(&[1]).unwrap();
    assert_close(reduced[0].re, 0.5);
    assert_close(reduced[4].re, 0.5);
    assert_close(reduced[8].re, 0.0);
    assert!(density.diagnostics().is_physical(TOLERANCE));
}

#[test]
fn reduced_density_matrix_honors_requested_target_order() {
    // |site2=1, site1=L, site0=0> has global radix index 1*9 + 2*3 = 15.
    let mut state = vec![c(0.0); 27];
    state[15] = c(1.0);
    let density = QuditDensityMatrix::from_density_matrix(
        3,
        3,
        pure_density(&state),
        PecosRng::seed_from_u64(29),
    )
    .unwrap();
    let reduced = density.reduced_density_matrix(&[2, 1]).unwrap();
    // target 2 is local digit zero: 1 + 3*2 = 7.
    assert_close(reduced[7 * 9 + 7].re, 1.0);
}

#[test]
fn diagnostics_reject_non_hermitian_trace_one_input() {
    let density = QuditDensityMatrix::from_density_matrix(
        1,
        3,
        vec![
            c(1.0),
            c(0.5),
            c(0.0),
            c(0.0),
            c(0.0),
            c(0.0),
            c(0.0),
            c(0.0),
            c(0.0),
        ],
        PecosRng::seed_from_u64(31),
    )
    .unwrap();
    let diagnostics = density.diagnostics();
    assert!(diagnostics.hermiticity_error > 0.4);
    assert!(diagnostics.minimum_eigenvalue < -0.05);
    assert!(!diagnostics.is_physical(TOLERANCE));
}

#[test]
fn invalid_targets_and_channels_return_errors() {
    let mut state = QutritStateVec::qutrit_with_seed(2, 37).unwrap();
    let identity = basis_swap(9, 0, 0).unwrap();
    assert_eq!(
        state.apply_operator(&[0, 0], &identity).unwrap_err(),
        QuditError::DuplicateTarget(0)
    );
    assert_eq!(
        state.measure(2).unwrap_err(),
        QuditError::TargetOutOfRange {
            target: 2,
            num_sites: 2
        }
    );
    assert_eq!(
        state.apply_kraus(&[0], &[]).unwrap_err(),
        QuditError::EmptyKrausChannel
    );
    assert!(matches!(
        state.apply_operator(&[0], &[c(1.0); 9]).unwrap_err(),
        QuditError::NonUnitary { .. }
    ));
    assert!(matches!(
        state.apply_kraus(&[0], &[vec![c(0.5); 9]]).unwrap_err(),
        QuditError::NotTracePreserving { .. }
    ));

    let density = QuditDensityMatrix::<PecosRng>::with_seed(usize::MAX, 3, 1).unwrap_err();
    assert_eq!(density, QuditError::DimensionOverflow);
}
