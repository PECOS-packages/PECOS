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

use nalgebra::DMatrix;
use num_complex::Complex64;
use pecos_random::{PecosRng, RngExt, RngManageable};
use pecos_simulators::{
    QuditDensityMatrix, QuditError, QuditStateVec, QutritDensityMatrix, QutritStateVec, basis_swap,
    embedded_qubit_unitary, qutrit_leakage_channel, qutrit_seepage_channel,
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

fn assert_complex_close(actual: Complex64, expected: Complex64) {
    assert!(
        (actual - expected).norm() < TOLERANCE,
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

fn reseed<T>(simulator: &mut T)
where
    T: RngManageable<Rng = PecosRng>,
{
    simulator.set_seed(131);
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

    let density = QutritDensityMatrix::with_seed(2, 7).unwrap();
    assert_eq!(density.local_dimension(), 3);
    assert_eq!(density.dimension(), 9);
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
fn embedded_qubit_helper_rejects_nonunitary_input() {
    let nonunitary = [c(1.0), c(1.0), c(0.0), c(1.0)];
    assert!(matches!(
        embedded_qubit_unitary(3, &nonunitary).unwrap_err(),
        QuditError::NonUnitary { .. }
    ));
}

#[test]
fn accepted_kraus_tolerance_is_consistent_with_result_normalization() {
    let scale = (1.0 + 5e-11_f64).sqrt();
    let mut operator = basis_swap(3, 0, 0).unwrap();
    for value in &mut operator {
        *value *= scale;
    }
    let channel = vec![operator];

    let mut trajectory = QutritStateVec::with_seed(1, 7).unwrap();
    trajectory.apply_kraus(&[0], &channel).unwrap();
    assert_close(
        trajectory.state().iter().map(Complex64::norm_sqr).sum(),
        1.0,
    );

    let mut exact = QutritDensityMatrix::with_seed(1, 7).unwrap();
    exact.apply_kraus(&[0], &channel).unwrap();
    assert!((exact.trace().re - 1.0).abs() < 1e-10);
    assert_close(exact.trace().im, 0.0);
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
fn randomized_multilevel_unitaries_cross_validate_backends() {
    let mut rng = PecosRng::seed_from_u64(0xD1_7E_57);
    for local_dimension in 2..=4 {
        for num_sites in 1..=3 {
            for trial in 0..8_u64 {
                let seed = 10_000 * u64::try_from(local_dimension).unwrap()
                    + 100 * u64::try_from(num_sites).unwrap()
                    + trial;
                let mut state = QuditStateVec::with_seed(num_sites, local_dimension, seed).unwrap();
                let mut density =
                    QuditDensityMatrix::with_seed(num_sites, local_dimension, seed).unwrap();
                for _ in 0..8 {
                    let first = rng.random_range(0..num_sites);
                    let targets = if num_sites > 1 && rng.random::<bool>() {
                        let mut second = rng.random_range(0..num_sites);
                        while second == first {
                            second = rng.random_range(0..num_sites);
                        }
                        if rng.random::<bool>() {
                            vec![first, second]
                        } else {
                            vec![second, first]
                        }
                    } else {
                        vec![first]
                    };
                    let size = local_dimension.pow(u32::try_from(targets.len()).unwrap());
                    let random_matrix = DMatrix::from_fn(size, size, |_, _| {
                        Complex64::new(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0))
                    });
                    let unitary = random_matrix.qr().q();
                    let mut row_major = Vec::with_capacity(size * size);
                    for row in 0..size {
                        for column in 0..size {
                            row_major.push(unitary[(row, column)]);
                        }
                    }
                    state.apply_operator(&targets, &row_major).unwrap();
                    density.apply_operator(&targets, &row_major).unwrap();
                }

                for (actual, expected) in density
                    .density_matrix()
                    .iter()
                    .zip(pure_density(state.state()))
                {
                    assert!(
                        (*actual - expected).norm() < 1e-9,
                        "d={local_dimension}, sites={num_sites}, trial={trial}"
                    );
                }
                for (state_value, density_value) in state
                    .reduced_density_matrix(&[0])
                    .unwrap()
                    .into_iter()
                    .zip(density.reduced_density_matrix(&[0]).unwrap())
                {
                    assert!((state_value - density_value).norm() < 1e-9);
                }
                assert!(density.diagnostics().is_physical(1e-9));
            }
        }
    }
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
                let branch = state.apply_kraus(&[0], &channel).unwrap().operator_index;
                let outcome = state.measure(0).unwrap();
                (branch, outcome)
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(run(), run());
}

#[test]
fn kraus_samples_report_branch_probability() {
    let channel = qutrit_leakage_channel(0.3).unwrap();
    for seed in 0..64_u64 {
        let mut state = QutritStateVec::qutrit_with_seed(1, seed).unwrap();
        state.prepare_basis(0, 1).unwrap();
        let sample = state.apply_kraus(&[0], &channel).unwrap();
        match sample.operator_index {
            0 => {
                assert_close(sample.probability, 0.7);
                assert_close(state.probability(1).unwrap(), 1.0);
            }
            2 => {
                assert_close(sample.probability, 0.3);
                assert_close(state.probability(2).unwrap(), 1.0);
            }
            index => panic!("zero-probability Kraus branch {index} was selected"),
        }
    }
}

#[test]
fn joint_and_partitioned_measurements_agree_across_backends() {
    let mut state = vec![c(0.0); 9];
    let amplitude = 1.0 / 3.0_f64.sqrt();
    state[0] = c(amplitude);
    state[4] = c(amplitude);
    state[8] = c(amplitude);
    let mut state_vector =
        QuditStateVec::from_state(2, 3, state.clone(), PecosRng::seed_from_u64(71)).unwrap();
    let mut density = QuditDensityMatrix::from_density_matrix(
        2,
        3,
        pure_density(&state),
        PecosRng::seed_from_u64(71),
    )
    .unwrap();

    let expected = [
        1.0 / 3.0,
        0.0,
        0.0,
        0.0,
        1.0 / 3.0,
        0.0,
        0.0,
        0.0,
        1.0 / 3.0,
    ];
    for (actual, expected) in state_vector
        .joint_outcome_probabilities(&[0, 1])
        .unwrap()
        .into_iter()
        .zip(expected)
    {
        assert_close(actual, expected);
    }
    assert_eq!(
        state_vector.joint_outcome_probabilities(&[0, 1]).unwrap(),
        density.joint_outcome_probabilities(&[0, 1]).unwrap()
    );

    // First outcome: both sites remain in the computational subspace.
    // Second outcome: at least one site is leaked.
    let partition = vec![vec![0, 1, 3, 4], vec![2, 5, 6, 7, 8]];
    let state_sample = state_vector.measure_partition(&[0, 1], &partition).unwrap();
    let density_sample = density.measure_partition(&[0, 1], &partition).unwrap();
    assert_eq!(state_sample.outcome, density_sample.outcome);
    assert_close(state_sample.probability, density_sample.probability);
    let expected_probability = if state_sample.outcome == 0 {
        2.0 / 3.0
    } else {
        1.0 / 3.0
    };
    assert_close(state_sample.probability, expected_probability);
    for (actual, expected) in density
        .density_matrix()
        .iter()
        .zip(pure_density(state_vector.state()))
    {
        assert!((*actual - expected).norm() < TOLERANCE);
    }

    assert_eq!(
        state_vector
            .measure_partition(&[0], &[vec![0], vec![0, 1, 2]])
            .unwrap_err(),
        QuditError::InvalidMeasurementPartition
    );
}

#[test]
fn generalized_measurement_instruments_preserve_outcome_semantics() {
    let projector = |level: usize| {
        let mut operator = vec![c(0.0); 9];
        operator[level * 3 + level] = c(1.0);
        operator
    };
    let instrument = vec![vec![projector(0), projector(1)], vec![projector(2)]];
    let amplitude = 1.0 / 3.0_f64.sqrt();
    let initial = vec![c(amplitude); 3];

    for seed in 0..64_u64 {
        let mut state =
            QuditStateVec::from_state(1, 3, initial.clone(), PecosRng::seed_from_u64(seed))
                .unwrap();
        let mut density = QuditDensityMatrix::from_density_matrix(
            1,
            3,
            pure_density(&initial),
            PecosRng::seed_from_u64(seed),
        )
        .unwrap();
        let state_probabilities = state.instrument_probabilities(&[0], &instrument).unwrap();
        let density_probabilities = density.instrument_probabilities(&[0], &instrument).unwrap();
        assert_close(state_probabilities[0], 2.0 / 3.0);
        assert_close(state_probabilities[1], 1.0 / 3.0);
        assert_eq!(state_probabilities, density_probabilities);
        let trajectory = state.measure_instrument(&[0], &instrument).unwrap();
        let exact = density.measure_instrument(&[0], &instrument).unwrap();
        assert_eq!(trajectory.outcome, exact.outcome);
        assert_close(trajectory.branch_probability, 1.0 / 3.0);
        assert_close(trajectory.outcome_probability, exact.probability);
        if exact.outcome == 0 {
            assert_close(exact.probability, 2.0 / 3.0);
            assert_close(density.probability(0).unwrap(), 0.5);
            assert_close(density.probability(1).unwrap(), 0.5);
            assert_close(density.probability(2).unwrap(), 0.0);
            assert_close(density.purity(), 0.5);
            assert!(trajectory.operator_index < 2);
        } else {
            assert_close(exact.probability, 1.0 / 3.0);
            assert_close(density.probability(2).unwrap(), 1.0);
            assert_close(density.purity(), 1.0);
            assert_eq!(trajectory.operator_index, 0);
        }
        assert!(density.diagnostics().is_physical(TOLERANCE));
    }

    let mut state = QutritStateVec::qutrit_with_seed(1, 1).unwrap();
    assert_eq!(
        state.measure_instrument(&[0], &[]).unwrap_err(),
        QuditError::InvalidMeasurementInstrument
    );
}

#[test]
fn memory_estimates_cover_qudits_and_density_operators() {
    assert_eq!(QuditStateVec::required_memory_bytes(3, 4).unwrap(), 1_024);
    assert_eq!(
        QuditDensityMatrix::required_memory_bytes(3, 4).unwrap(),
        65_536
    );
    assert_eq!(QutritStateVec::required_memory_bytes(3).unwrap(), 432);
    assert_eq!(
        QutritDensityMatrix::required_memory_bytes(3).unwrap(),
        11_664
    );
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
    state.reset_site(0).unwrap();
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
    density.reset_site(0).unwrap();
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
    let entries = vec![
        c(1.0),
        c(0.5),
        c(0.0),
        c(0.0),
        c(0.0),
        c(0.0),
        c(0.0),
        c(0.0),
        c(0.0),
    ];
    assert!(matches!(
        QuditDensityMatrix::from_density_matrix(
            1,
            3,
            entries.clone(),
            PecosRng::seed_from_u64(31),
        )
        .unwrap_err(),
        QuditError::NonHermitian { .. }
    ));
    let density = QuditDensityMatrix::from_density_matrix_unchecked(
        1,
        3,
        entries,
        PecosRng::seed_from_u64(31),
    )
    .unwrap();
    let diagnostics = density.diagnostics();
    assert!(diagnostics.hermiticity_error > 0.4);
    assert!(diagnostics.minimum_eigenvalue < -0.05);
    assert!(!diagnostics.is_physical(TOLERANCE));

    assert!(matches!(
        QuditDensityMatrix::from_density_matrix(
            1,
            3,
            vec![
                c(1.1),
                c(0.0),
                c(0.0),
                c(0.0),
                c(-0.1),
                c(0.0),
                c(0.0),
                c(0.0),
                c(0.0)
            ],
            PecosRng::seed_from_u64(31),
        )
        .unwrap_err(),
        QuditError::NotPositiveSemidefinite { .. }
    ));
    assert_eq!(
        density.validate_physicality(-1.0).unwrap_err(),
        QuditError::InvalidTolerance(-1.0)
    );
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
    let mut density = QutritDensityMatrix::with_seed(2, 37).unwrap();
    assert_eq!(
        density.apply_kraus(&[0], &[]).unwrap_err(),
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

#[test]
fn reduced_density_matrices_match_literal_multisite_oracles() {
    // Four amplitudes occupy three different values of the traced-out middle site.
    // This pins both partial-trace matching and the requested target order without
    // treating either simulator backend as the oracle for the other.
    let mut amplitudes = vec![c(0.0); 27];
    amplitudes[0] = c(0.5);
    amplitudes[11] = Complex64::new(0.0, 0.5);
    amplitudes[3] = c(0.5);
    amplitudes[25] = c(-0.5);
    let state =
        QuditStateVec::from_state(3, 3, amplitudes.clone(), PecosRng::seed_from_u64(101)).unwrap();
    let density = QuditDensityMatrix::from_density_matrix(
        3,
        3,
        pure_density(&amplitudes),
        PecosRng::seed_from_u64(101),
    )
    .unwrap();

    let mut expected_20 = vec![c(0.0); 81];
    expected_20[0] = c(0.5);
    expected_20[7] = Complex64::new(0.0, -0.25);
    expected_20[7 * 9] = Complex64::new(0.0, 0.25);
    expected_20[7 * 9 + 7] = c(0.25);
    expected_20[5 * 9 + 5] = c(0.25);

    let mut expected_02 = vec![c(0.0); 81];
    expected_02[0] = c(0.5);
    expected_02[5] = Complex64::new(0.0, -0.25);
    expected_02[5 * 9] = Complex64::new(0.0, 0.25);
    expected_02[5 * 9 + 5] = c(0.25);
    expected_02[7 * 9 + 7] = c(0.25);

    for simulator_result in [
        state.reduced_density_matrix(&[2, 0]).unwrap(),
        density.reduced_density_matrix(&[2, 0]).unwrap(),
    ] {
        for (actual, expected) in simulator_result.into_iter().zip(&expected_20) {
            assert_complex_close(actual, *expected);
        }
    }
    for simulator_result in [
        state.reduced_density_matrix(&[0, 2]).unwrap(),
        density.reduced_density_matrix(&[0, 2]).unwrap(),
    ] {
        for (actual, expected) in simulator_result.into_iter().zip(&expected_02) {
            assert_complex_close(actual, *expected);
        }
    }
}

#[test]
fn embedded_unitary_and_purity_match_literal_complex_oracles() {
    let a = FRAC_1_SQRT_2;
    let unitary = [
        c(a),
        Complex64::new(0.5, 0.5),
        Complex64::new(-0.5, 0.5),
        c(a),
    ];
    let embedded = embedded_qubit_unitary(3, &unitary).unwrap();
    let expected = [
        c(a),
        Complex64::new(0.5, 0.5),
        c(0.0),
        Complex64::new(-0.5, 0.5),
        c(a),
        c(0.0),
        c(0.0),
        c(0.0),
        c(1.0),
    ];
    for (actual, expected) in embedded.iter().zip(expected) {
        assert_complex_close(*actual, expected);
    }

    let mut computational = QutritStateVec::qutrit_with_seed(1, 103).unwrap();
    computational
        .apply_embedded_qubit_unitary(0, &unitary)
        .unwrap();
    for (actual, expected) in
        computational
            .state()
            .iter()
            .zip([c(a), Complex64::new(-0.5, 0.5), c(0.0)])
    {
        assert_complex_close(*actual, expected);
    }

    let mut leaked = QutritStateVec::qutrit_with_seed(1, 103).unwrap();
    leaked.prepare_basis(0, 2).unwrap();
    leaked.apply_embedded_qubit_unitary(0, &unitary).unwrap();
    for (actual, expected) in leaked.state().iter().zip([c(0.0), c(0.0), c(1.0)]) {
        assert_complex_close(*actual, expected);
    }

    let mixed = vec![
        c(0.5),
        Complex64::new(0.0, 0.25),
        c(0.0),
        Complex64::new(0.0, -0.25),
        c(0.5),
        c(0.0),
        c(0.0),
        c(0.0),
        c(0.0),
    ];
    let density =
        QutritDensityMatrix::from_density_matrix(1, mixed, PecosRng::seed_from_u64(103)).unwrap();
    assert_close(density.purity(), 0.625);
}

#[test]
fn instrument_post_states_and_partition_completeness_have_literal_oracles() {
    let zero = vec![c(0.0); 9];
    let identity = basis_swap(3, 0, 0).unwrap();
    let mut trajectory = QutritStateVec::qutrit_with_seed(1, 107).unwrap();
    trajectory.prepare_basis(0, 1).unwrap();
    let sample = trajectory
        .measure_instrument(&[0], &[vec![zero, identity]])
        .unwrap();
    assert_eq!(sample.outcome, 0);
    assert_eq!(sample.operator_index, 1);
    for (actual, expected) in trajectory.state().iter().zip([c(0.0), c(1.0), c(0.0)]) {
        assert_complex_close(*actual, expected);
    }

    let projector = |level: usize| {
        let mut operator = vec![c(0.0); 9];
        operator[level * 3 + level] = c(1.0);
        operator
    };
    let plus = vec![c(FRAC_1_SQRT_2), c(FRAC_1_SQRT_2), c(0.0)];
    let mut exact = QutritDensityMatrix::from_density_matrix(
        1,
        pure_density(&plus),
        PecosRng::seed_from_u64(107),
    )
    .unwrap();
    let sample = exact
        .measure_instrument(&[0], &[vec![projector(0), projector(1), projector(2)]])
        .unwrap();
    assert_eq!(sample.outcome, 0);
    assert_close(sample.probability, 1.0);
    for (actual, expected) in exact.density_matrix().iter().zip([
        c(0.5),
        c(0.0),
        c(0.0),
        c(0.0),
        c(0.5),
        c(0.0),
        c(0.0),
        c(0.0),
        c(0.0),
    ]) {
        assert_complex_close(*actual, expected);
    }

    assert_eq!(
        trajectory
            .measure_partition(&[0], &[vec![0], vec![1]])
            .unwrap_err(),
        QuditError::InvalidMeasurementPartition
    );
}

#[test]
fn reversed_three_site_target_unitary_has_a_literal_outcome() {
    let mut amplitudes = vec![c(0.0); 27];
    amplitudes[15] = c(1.0);
    let mut state =
        QuditStateVec::from_state(3, 3, amplitudes, PecosRng::seed_from_u64(109)).unwrap();
    state
        .apply_operator(&[2, 0, 1], &basis_swap(27, 19, 5).unwrap())
        .unwrap();
    for (index, amplitude) in state.state().iter().enumerate() {
        assert_complex_close(*amplitude, if index == 19 { c(1.0) } else { c(0.0) });
    }
}

#[test]
fn exact_leakage_channel_preserves_the_expected_coherences() {
    let scale = 1.0 / 3.0_f64.sqrt();
    let amplitudes = vec![c(scale), Complex64::new(0.0, scale), c(scale)];
    let mut density = QutritDensityMatrix::from_density_matrix(
        1,
        pure_density(&amplitudes),
        PecosRng::seed_from_u64(113),
    )
    .unwrap();
    density
        .apply_kraus(&[0], &qutrit_leakage_channel(0.25).unwrap())
        .unwrap();
    let comp_leak_coherence = 0.75_f64.sqrt() / 3.0;
    let expected = [
        c(0.25),
        Complex64::new(0.0, -0.25),
        c(comp_leak_coherence),
        Complex64::new(0.0, 0.25),
        c(0.25),
        Complex64::new(0.0, comp_leak_coherence),
        c(comp_leak_coherence),
        Complex64::new(0.0, -comp_leak_coherence),
        c(0.5),
    ];
    for (actual, expected) in density.density_matrix().iter().zip(expected) {
        assert_complex_close(*actual, expected);
    }
}

#[test]
fn target_probability_tolerance_and_rng_contracts_are_explicit() {
    let mut state = QutritStateVec::qutrit_with_seed(1, 127).unwrap();
    assert_eq!(
        state.apply_operator(&[], &[]).unwrap_err(),
        QuditError::EmptyTargets
    );
    assert_eq!(
        state.measure_joint(&[]).unwrap_err(),
        QuditError::EmptyTargets
    );

    let scale = 1.0 / 3.0_f64.sqrt();
    let omega = Complex64::from_polar(1.0, 2.0 * std::f64::consts::PI / 3.0);
    let fourier = vec![
        c(scale),
        c(scale),
        c(scale),
        c(scale),
        omega * scale,
        omega.powu(2) * scale,
        c(scale),
        omega.powu(2) * scale,
        omega * scale,
    ];
    for _ in 0..10_000 {
        state.apply_operator(&[0], &fourier).unwrap();
    }
    QutritStateVec::from_state(1, state.state().to_vec(), PecosRng::seed_from_u64(127)).unwrap();

    let nearly_physical = vec![
        c(-5e-12),
        c(0.0),
        c(0.0),
        c(0.0),
        c(0.5),
        c(0.0),
        c(0.0),
        c(0.0),
        c(0.500_000_000_005),
    ];
    let density =
        QutritDensityMatrix::from_density_matrix(1, nearly_physical, PecosRng::seed_from_u64(127))
            .unwrap();
    let probabilities = density.outcome_probabilities(0).unwrap();
    assert_close(probabilities[0], 0.0);
    assert!(probabilities.iter().all(|probability| *probability >= 0.0));

    let mut general_state = QuditStateVec::with_seed(1, 4, 1).unwrap();
    let mut qutrit_state = QutritStateVec::with_seed(1, 1).unwrap();
    let mut general_density = QuditDensityMatrix::with_seed(1, 4, 1).unwrap();
    let mut qutrit_density = QutritDensityMatrix::with_seed(1, 1).unwrap();
    reseed(&mut general_state);
    reseed(&mut qutrit_state);
    reseed(&mut general_density);
    reseed(&mut qutrit_density);

    let mut expected_rng = PecosRng::seed_from_u64(131);
    let expected_first_draw = expected_rng.random::<u64>();
    assert_eq!(general_state.rng_mut().random::<u64>(), expected_first_draw);
    assert_eq!(qutrit_state.rng_mut().random::<u64>(), expected_first_draw);
    assert_eq!(
        general_density.rng_mut().random::<u64>(),
        expected_first_draw
    );
    assert_eq!(
        qutrit_density.rng_mut().random::<u64>(),
        expected_first_draw
    );
}

#[test]
fn materially_negative_unchecked_density_has_a_specific_diagnostic() {
    let entries = vec![
        c(-0.2),
        c(0.0),
        c(0.0),
        c(0.0),
        c(0.6),
        c(0.0),
        c(0.0),
        c(0.0),
        c(0.6),
    ];
    let mut density = QutritDensityMatrix::from_density_matrix_unchecked(
        1,
        entries,
        PecosRng::seed_from_u64(137),
    )
    .unwrap();
    assert_eq!(
        density.outcome_probabilities(0).unwrap_err(),
        QuditError::InvalidProbability(-0.2)
    );
    assert_eq!(
        density.measure(0).unwrap_err(),
        QuditError::InvalidProbability(-0.2)
    );
}

#[test]
fn qutrit_convenience_constructors_and_unwrapping_preserve_contracts() {
    let wrapped_state = QutritStateVec::qutrit(2).unwrap();
    assert_eq!(wrapped_state.local_dimension(), 3);
    assert_eq!(wrapped_state.dimension(), 9);
    assert_close(wrapped_state.probability(0).unwrap(), 1.0);

    let general_state = QuditStateVec::qutrit(2).unwrap();
    assert_eq!(general_state.local_dimension(), 3);
    assert_eq!(general_state.dimension(), 9);
    assert_close(general_state.probability(0).unwrap(), 1.0);

    let wrapped_density = QutritDensityMatrix::qutrit(2).unwrap();
    assert_eq!(wrapped_density.local_dimension(), 3);
    assert_eq!(wrapped_density.dimension(), 9);
    assert_close(wrapped_density.probability(0).unwrap(), 1.0);

    let general_density = QuditDensityMatrix::qutrit(2).unwrap();
    assert_eq!(general_density.local_dimension(), 3);
    assert_eq!(general_density.dimension(), 9);
    assert_close(general_density.probability(0).unwrap(), 1.0);

    let state_seed = 149;
    let mut wrapped_state =
        QutritStateVec::with_rng(2, PecosRng::seed_from_u64(state_seed)).unwrap();
    wrapped_state
        .apply_operator(&[1], &basis_swap(3, 0, 2).unwrap())
        .unwrap();
    let mut inner_state = wrapped_state.into_inner();
    assert_eq!(inner_state.local_dimension(), 3);
    assert_close(inner_state.probability(6).unwrap(), 1.0);
    let mut expected_state_rng = PecosRng::seed_from_u64(state_seed);
    assert_eq!(
        inner_state.rng_mut().random::<u64>(),
        expected_state_rng.random::<u64>()
    );

    let density_seed = 151;
    let mut wrapped_density =
        QutritDensityMatrix::with_rng(2, PecosRng::seed_from_u64(density_seed)).unwrap();
    wrapped_density
        .apply_operator(&[0], &basis_swap(3, 0, 1).unwrap())
        .unwrap();
    let mut inner_density = wrapped_density.into_inner();
    assert_eq!(inner_density.local_dimension(), 3);
    assert_close(inner_density.probability(1).unwrap(), 1.0);
    let mut expected_density_rng = PecosRng::seed_from_u64(density_seed);
    assert_eq!(
        inner_density.rng_mut().random::<u64>(),
        expected_density_rng.random::<u64>()
    );
}
