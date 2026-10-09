// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
// except in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the
// License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
// express or implied. See the License for the specific language governing permissions and
// limitations under the License.

use super::*;

pub(super) fn with_fast_path<T>(enabled: bool, run: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            DISABLE_TRIVIAL_EXACT_MEASUREMENT.set(self.0);
        }
    }
    let _restore = Restore(DISABLE_TRIVIAL_EXACT_MEASUREMENT.replace(!enabled));
    run()
}

fn tensor_bits(mps: &Mps) -> Vec<Vec<(u64, u64)>> {
    mps.tensors()
        .iter()
        .map(|tensor| {
            tensor
                .iter()
                .map(|value| (value.re.to_bits(), value.im.to_bits()))
                .collect()
        })
        .collect()
}

pub(super) fn assert_pair_bits(
    first_tableau: &SparseStabY,
    first_mps: &Mps,
    second_tableau: &SparseStabY,
    second_mps: &Mps,
) {
    assert_eq!(tensor_bits(first_mps), tensor_bits(second_mps));
    assert_eq!(first_mps.bond_dims(), second_mps.bond_dims());
    assert_eq!(
        first_mps.tracked_center_for_test(),
        second_mps.tracked_center_for_test()
    );
    assert_eq!(
        format!("{:?}", first_tableau.stabs()),
        format!("{:?}", second_tableau.stabs())
    );
    assert_eq!(
        format!("{:?}", first_tableau.destabs()),
        format!("{:?}", second_tableau.destabs())
    );
    assert_eq!(
        first_tableau.rng().clone().next_u64(),
        second_tableau.rng().clone().next_u64()
    );
}

fn assert_simulator_bits(fast: &StabMps, slow: &StabMps) {
    assert_pair_bits(&fast.tableau, &fast.mps, &slow.tableau, &slow.mps);
    assert_eq!(
        fast.rng.clone().next_u64(),
        slow.rng.clone().next_u64(),
        "measurement must consume the same RNG stream"
    );
    assert_eq!(fast.disent_flags, slow.disent_flags);
}

#[test]
fn trivial_exact_random_circuits_match_general_path_bitwise() {
    for seed in 0..64 {
        for truncating in [false, true] {
            for non_clifford in [false, true] {
                let mut fast = StabMps::builder(4)
                    .seed(seed)
                    .measurement(MeasurementMode::Exact)
                    .max_bond_dim(if truncating { 2 } else { 4 })
                    .svd_cutoff(if truncating { 1e-7 } else { 0.0 })
                    .max_truncation_error(0.0)
                    .merge_rz(false)
                    .build();
                let mut slow = fast.clone();
                let mut circuit_rng = PecosRng::seed_from_u64(seed ^ 0xa53d);
                for round in 0..8 {
                    // Each round starts trivial and can return from a non-trivial
                    // coefficient state by measuring the rotated qubit.
                    assert!(measure::is_mps_trivial(&fast.mps));
                    if non_clifford {
                        let a = with_fast_path(true, || fast.reset_qubit(QubitId(0)));
                        let b = with_fast_path(false, || slow.reset_qubit(QubitId(0)));
                        assert_eq!(a, b);
                        for simulator in [&mut fast, &mut slow] {
                            simulator.h(&[QubitId(0)]);
                            simulator.rz(Angle64::from_radians(0.37), &[QubitId(0)]);
                        }
                        assert!(!measure::is_mps_trivial(&fast.mps));
                        let a = with_fast_path(true, || fast.mz(&[QubitId(0)]));
                        let b = with_fast_path(false, || slow.mz(&[QubitId(0)]));
                        assert_eq!(a[0].outcome, b[0].outcome);
                        assert_eq!(a[0].is_deterministic, b[0].is_deterministic);
                        assert!(measure::is_mps_trivial(&fast.mps));
                        assert_simulator_bits(&fast, &slow);
                    }
                    for _ in 0..12 {
                        let q = (circuit_rng.next_u64() % 4) as usize;
                        let r = (q + 1 + (circuit_rng.next_u64() % 3) as usize) % 4;
                        let gate = circuit_rng.next_u64() % 5;
                        for simulator in [&mut fast, &mut slow] {
                            match gate {
                                0 => simulator.h(&[QubitId(q)]),
                                1 => simulator.sz(&[QubitId(q)]),
                                2 => simulator.x(&[QubitId(q)]),
                                3 => simulator.cx(&[(QubitId(q), QubitId(r))]),
                                _ => simulator.cz(&[(QubitId(q), QubitId(r))]),
                            };
                        }
                    }
                    let q = QubitId((circuit_rng.next_u64() % 4) as usize);
                    // Repeating a measurement covers deterministic RNG endpoints.
                    for _ in 0..2 {
                        let a = with_fast_path(true, || fast.mz(&[q]));
                        let b = with_fast_path(false, || slow.mz(&[q]));
                        assert_eq!(a[0].outcome, b[0].outcome);
                        assert_eq!(a[0].is_deterministic, b[0].is_deterministic);
                        assert_simulator_bits(&fast, &slow);
                    }
                    let a = with_fast_path(true, || fast.reset_qubit(q));
                    let b = with_fast_path(false, || slow.reset_qubit(q));
                    assert_eq!(a, b);
                    with_fast_path(true, || fast.pz(QubitId(round % 4)));
                    with_fast_path(false, || slow.pz(QubitId(round % 4)));
                    assert_simulator_bits(&fast, &slow);
                }
            }
        }
    }
}

#[test]
fn trivial_exact_retains_rz_normalization_bits() {
    for normalize in [false, true] {
        let mut fast = StabMps::builder(1)
            .seed(9)
            .measurement(MeasurementMode::Exact)
            .max_bond_dim(1)
            .svd_cutoff(0.0)
            .max_truncation_error(0.0)
            .merge_rz(false)
            .normalize_after_gate(normalize)
            .build();
        fast.rz(Angle64::from_radians(0.001), &[QubitId(0)]);
        let mut slow = fast.clone();
        let before = tensor_bits(&fast.mps);
        let a = with_fast_path(true, || fast.mz(&[QubitId(0)]));
        let b = with_fast_path(false, || slow.mz(&[QubitId(0)]));
        assert_eq!(a[0].outcome, b[0].outcome);
        assert_eq!(a[0].is_deterministic, b[0].is_deterministic);
        assert_ne!(before, tensor_bits(&slow.mps));
        assert_simulator_bits(&fast, &slow);
    }
}

#[test]
fn trivial_exact_basis_word_preserves_update_and_rng_bits() {
    for seed in 0..32 {
        for random in [false, true] {
            let mut fast = StabMps::builder(4).seed(seed).build();
            for site in [1, 3] {
                fast.mps.tensors_mut()[site][(0, 0)] = Complex64::new(0.0, 0.0);
                fast.mps.tensors_mut()[site][(0, 1)] = Complex64::new(1.0, 0.0);
            }
            if random {
                fast.h(&[QubitId(1)]);
                fast.cx(&[(QubitId(1), QubitId(2))]);
            }
            let mut slow = fast.clone();
            let a = with_fast_path(true, || {
                measure_qubit_exact_transactional(
                    &mut fast.tableau,
                    &mut fast.mps,
                    &mut fast.rng,
                    1,
                    "test",
                )
                .unwrap()
            });
            let b = with_fast_path(false, || {
                measure_qubit_exact_transactional(
                    &mut slow.tableau,
                    &mut slow.mps,
                    &mut slow.rng,
                    1,
                    "test",
                )
                .unwrap()
            });
            assert_eq!(a.measurement.outcome, b.measurement.outcome);
            assert_eq!(
                a.measurement.is_deterministic,
                b.measurement.is_deterministic
            );
            assert_eq!(a.update.collapsed_site, b.update.collapsed_site);
            assert_eq!(a.update.modified_sites, b.update.modified_sites);
            assert_eq!(a.update.modified_sites, vec![1, 3]);
            assert_simulator_bits(&fast, &slow);
        }
    }
}

#[test]
fn trivial_exact_skips_transactions_and_expectations() {
    for tracked_center in [false, true] {
        let mut stn = StabMps::builder(4)
            .seed(9)
            .max_bond_dim(1)
            .svd_cutoff(0.0)
            .max_truncation_error(0.0)
            .build();
        stn.h(&[QubitId(0)]);
        for q in 1..4 {
            stn.cx(&[(QubitId(0), QubitId(q))]);
        }
        if !tracked_center {
            stn.mps.set_tracked_center_for_test(None);
        }
        let before = EXACT_MEASUREMENT_TRANSACTIONS.get();
        measure::Z_EXPECTATION_EVALUATIONS.set(0);
        crate::mps::NORM_SQUARED_EVALUATIONS.set(0);
        measure::TRIVIAL_MPS_NORM_EVALUATIONS.set(0);
        with_fast_path(true, || stn.mz(&[QubitId(0)]));
        assert_eq!(EXACT_MEASUREMENT_TRANSACTIONS.get() - before, 0);
        assert_eq!(measure::Z_EXPECTATION_EVALUATIONS.get(), 0);
        // Routing performs one read-only scalar-product norm, counted separately
        // from full contractions by TRIVIAL_MPS_NORM_EVALUATIONS.
        assert_eq!(measure::TRIVIAL_MPS_NORM_EVALUATIONS.get(), 1);
        // normalize() contracts once only when no center is tracked; otherwise
        // it uses the center tensor's norm. Debug canonicalization adds one
        // full norm. No probability or survival contraction remains.
        assert_eq!(
            crate::mps::NORM_SQUARED_EVALUATIONS.get(),
            usize::from(!tracked_center) + usize::from(cfg!(debug_assertions))
        );
    }
}

#[cfg(not(debug_assertions))]
fn assert_unnormalized_trivial_measurement_matches(mut fast: StabMps) {
    assert!(!measure::is_mps_trivial(&fast.mps));
    assert!(fast.mps.norm_squared() < 1e-12);
    // Both routes receive the same decayed superposition and must establish
    // measurement normalization themselves, regardless of fast-path enablement.
    let mut slow = fast.clone();
    let a = with_fast_path(true, || fast.mz(&[QubitId(0)]));
    let b = with_fast_path(false, || slow.mz(&[QubitId(0)]));
    assert!(!b[0].outcome && b[0].is_deterministic);
    assert!(!a[0].outcome && a[0].is_deterministic);
    assert_simulator_bits(&fast, &slow);
    for _ in 0..3 {
        for simulator in [&mut fast, &mut slow] {
            simulator.h(&[QubitId(0)]);
            simulator.rz(Angle64::from_radians(0.37), &[QubitId(0)]);
            if simulator.num_qubits() > 1 {
                simulator.cx(&[(QubitId(0), QubitId(1))]);
                simulator.h(&[QubitId(1)]);
            }
        }
        assert_simulator_bits(&fast, &slow);
        for q in 0..fast.num_qubits() {
            let a = with_fast_path(true, || fast.mz(&[QubitId(q)]));
            let b = with_fast_path(false, || slow.mz(&[QubitId(q)]));
            assert_eq!(a[0].outcome, b[0].outcome);
            assert_eq!(a[0].is_deterministic, b[0].is_deterministic);
            assert_simulator_bits(&fast, &slow);
        }
    }
}

#[cfg(not(debug_assertions))]
#[test]
fn trivial_exact_unnormalized_public_circuit_matches_general_path() {
    for seed in 0..4 {
        let mut stn = StabMps::builder(2)
            .seed(seed)
            .measurement(MeasurementMode::Exact)
            .max_bond_dim(1)
            .svd_cutoff(1e-7)
            .normalize_after_gate(false)
            .merge_rz(false)
            .build();
        for _ in 0..271 {
            stn.h(&[QubitId(0)]);
            stn.rz(Angle64::from_radians(0.9), &[QubitId(0)]);
            stn.cx(&[(QubitId(0), QubitId(1))]);
            stn.h(&[QubitId(1)]);
        }
        assert_unnormalized_trivial_measurement_matches(stn);
    }
}

#[cfg(not(debug_assertions))]
#[test]
fn trivial_exact_unnormalized_small_blocks_match_general_path() {
    for seed in 0..4 {
        let mut stn = StabMps::builder(1)
            .seed(seed)
            .measurement(MeasurementMode::Exact)
            .merge_rz(false)
            .build();
        stn.h(&[QubitId(0)]);
        for amplitude in stn.mps.tensors_mut()[0].iter_mut() {
            *amplitude = Complex64::new(5e-7, 0.0);
        }
        assert_unnormalized_trivial_measurement_matches(stn);
    }
}

#[test]
fn trivial_exact_bond_one_norm_is_read_only() {
    for tracked_center in [false, true] {
        for scale in [0.0, 1e-6, 0.5, 1.0, 2.0] {
            let mut stn = StabMps::builder(4).seed(9).build();
            stn.mps.scale(Complex64::new(scale, 0.0));
            if !tracked_center {
                stn.mps.set_tracked_center_for_test(None);
            }
            let before = stn.clone();
            let expected = stn.mps.norm_squared();
            crate::mps::NORM_SQUARED_EVALUATIONS.set(0);
            measure::TRIVIAL_MPS_NORM_EVALUATIONS.set(0);
            let actual = measure::trivial_mps_norm_squared(&stn.mps);
            assert_eq!(actual.to_bits(), expected.to_bits());
            assert_eq!(measure::TRIVIAL_MPS_NORM_EVALUATIONS.get(), 1);
            assert_eq!(crate::mps::NORM_SQUARED_EVALUATIONS.get(), 0);
            assert_simulator_bits(&stn, &before);
        }
    }
}

#[test]
fn scaled_site_triviality_is_relative() {
    for scale in [1.0, 1e-7, 1e-100, 1e-200, f64::from_bits(1), 1e200] {
        for (amplitudes, expected) in [
            ([scale, scale], false),
            ([scale, 0.0], true),
            ([0.0, scale], true),
        ] {
            let mut mps = Mps::new(1, MpsConfig::default());
            for (entry, amplitude) in mps.tensors_mut()[0].iter_mut().zip(amplitudes) {
                *entry = Complex64::new(amplitude, 0.0);
            }
            assert_eq!(
                measure::is_mps_trivial(&mps),
                expected,
                "amplitudes={amplitudes:?}"
            );
        }
    }
    let mut zero = Mps::new(1, MpsConfig::default());
    zero.scale(Complex64::new(0.0, 0.0));
    assert!(!measure::is_mps_trivial(&zero));
}

fn decayed_circuit(seed: u64) -> StabMps {
    let mut stn = StabMps::builder(2)
        .seed(seed)
        .measurement(MeasurementMode::Exact)
        .max_bond_dim(1)
        .svd_cutoff(1e-7)
        .normalize_after_gate(false)
        .merge_rz(false)
        .build();
    for _ in 0..271 {
        stn.h(&[QubitId(0)]);
        stn.rz(Angle64::from_radians(0.9), &[QubitId(0)]);
        stn.cx(&[(QubitId(0), QubitId(1))]);
        stn.h(&[QubitId(1)]);
    }
    assert!(stn.mps.norm_squared() > 0.0 && stn.mps.norm_squared() < 1e-12);
    stn
}

pub(super) fn assert_pair_close(
    first_tableau: &SparseStabY,
    first_mps: &Mps,
    second_tableau: &SparseStabY,
    second_mps: &Mps,
) {
    assert_eq!(
        format!("{:?}", first_tableau.stabs()),
        format!("{:?}", second_tableau.stabs())
    );
    assert_eq!(
        format!("{:?}", first_tableau.destabs()),
        format!("{:?}", second_tableau.destabs())
    );
    assert_eq!(first_mps.bond_dims(), second_mps.bond_dims());
    for (first, second) in first_mps.tensors().iter().zip(second_mps.tensors()) {
        assert!(
            (first - second).norm() < 1e-10,
            "coefficient tensors differ: {first:?} vs {second:?}"
        );
    }
}

#[test]
fn decayed_exact_measurement_matches_normalized_state() {
    for seed in 0..8 {
        let mut decayed = decayed_circuit(seed);
        let mut normalized = decayed.clone();
        normalized.mps.normalize();
        let actual = decayed.mz(&[QubitId(0)]);
        let expected = normalized.mz(&[QubitId(0)]);
        assert_eq!(actual[0].outcome, expected[0].outcome, "seed={seed}");
        assert_eq!(
            actual[0].is_deterministic, expected[0].is_deterministic,
            "seed={seed}"
        );
        assert_pair_close(
            &decayed.tableau,
            &decayed.mps,
            &normalized.tableau,
            &normalized.mps,
        );
    }
}

#[test]
fn relative_triviality_preserves_normalized_threshold() {
    for probability_one in [0.5e-12_f64, 2e-12] {
        for scale in [1.0, 1e-7, 1e-200, 1e200] {
            let mut mps = Mps::new(1, MpsConfig::default());
            mps.tensors_mut()[0][(0, 0)] =
                Complex64::new((1.0 - probability_one).sqrt() * scale, 0.0);
            mps.tensors_mut()[0][(0, 1)] = Complex64::new(0.0, probability_one.sqrt() * scale);
            assert_eq!(measure::is_mps_trivial(&mps), probability_one < 1e-12);
        }
    }
}

#[test]
fn decayed_reset_matches_normalized_state() {
    for seed in 0..8 {
        let mut decayed = decayed_circuit(seed);
        let mut normalized = decayed.clone();
        normalized.mps.normalize();
        assert_eq!(
            decayed.reset_qubit(QubitId(0)),
            normalized.reset_qubit(QubitId(0))
        );
        assert_pair_close(
            &decayed.tableau,
            &decayed.mps,
            &normalized.tableau,
            &normalized.mps,
        );
        assert_eq!(decayed.disent_flags, normalized.disent_flags);
    }
}

#[test]
fn scaled_measurement_modes_match_normalized_state() {
    for mode in [
        MeasurementMode::Exact,
        MeasurementMode::Pragmatic,
        MeasurementMode::Lazy,
    ] {
        for probability_one in [0.0_f64, 0.37, 1.0] {
            for seed in 0..8 {
                let mut normalized = StabMps::builder(1).seed(seed).measurement(mode).build();
                normalized.mps.tensors_mut()[0][(0, 0)] =
                    Complex64::new((1.0 - probability_one).sqrt(), 0.0);
                normalized.mps.tensors_mut()[0][(0, 1)] =
                    Complex64::new(probability_one.sqrt(), 0.0);
                let mut scaled = normalized.clone();
                scaled.mps.scale(Complex64::new(1e-7, 0.0));
                let actual = scaled.mz(&[QubitId(0)]);
                let expected = normalized.mz(&[QubitId(0)]);
                assert_eq!(actual[0].outcome, expected[0].outcome);
                assert_eq!(actual[0].is_deterministic, expected[0].is_deterministic);
                assert_pair_close(
                    &scaled.tableau,
                    &scaled.mps,
                    &normalized.tableau,
                    &normalized.mps,
                );
                assert_eq!(scaled.deferred_ops.len(), normalized.deferred_ops.len());
            }
        }
    }
}

#[test]
fn zero_norm_measurement_is_rejected_in_every_mode() {
    for mode in [
        MeasurementMode::Exact,
        MeasurementMode::Pragmatic,
        MeasurementMode::Lazy,
    ] {
        let mut stn = StabMps::builder(1).measurement(mode).build();
        stn.mps.scale(Complex64::new(0.0, 0.0));
        let panic =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| stn.mz(&[QubitId(0)])))
                .err()
                .expect("zero norm must be rejected");
        let message = panic
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| panic.downcast_ref::<&str>().copied())
            .expect("panic message");
        assert!(message.contains("non-finite or zero norm"), "{message}");
    }
}

#[test]
fn scaled_forced_projection_matches_normalized_state() {
    for probability_one in [0.0_f64, 0.37, 1.0] {
        for entangled in [false, true] {
            let mut normalized = StabMps::builder(2).svd_cutoff(1e-7).build();
            normalized.mps.tensors_mut()[0][(0, 0)] =
                Complex64::new((1.0 - probability_one).sqrt(), 0.0);
            normalized.mps.tensors_mut()[0][(0, 1)] = Complex64::new(0.0, probability_one.sqrt());
            if entangled {
                measure::flush_deferred(&mut normalized.mps, &mut vec![(0, 1)]).unwrap();
                normalized.h(&[QubitId(0)]);
            }
            normalized.mps.normalize();
            for outcome in [false, true] {
                let mut expected = normalized.clone();
                let mut scaled = normalized.clone();
                scaled.mps.scale(Complex64::new(1e-7, 0.0));
                let actual_projection = measure::project_forced_z_with_update(
                    &mut scaled.tableau,
                    &mut scaled.mps,
                    0,
                    outcome,
                    None,
                )
                .unwrap();
                let expected_projection = measure::project_forced_z_with_update(
                    &mut expected.tableau,
                    &mut expected.mps,
                    0,
                    outcome,
                    None,
                )
                .unwrap();
                assert!(
                    (actual_projection.snapped_probability
                        - expected_projection.snapped_probability)
                        .abs()
                        < 1e-12
                );
                assert!(
                    (actual_projection.survival_ratio - expected_projection.survival_ratio).abs()
                        < 1e-12
                );
                assert_pair_close(
                    &scaled.tableau,
                    &scaled.mps,
                    &expected.tableau,
                    &expected.mps,
                );
            }
        }
    }
}

#[test]
fn triviality_and_measurement_are_gauge_invariant() {
    for probability_one in [5e-13_f64, 2e-12, 0.37] {
        for seed in 0..8 {
            let mut reference = StabMps::builder(2).seed(seed).build();
            reference.mps.tensors_mut()[0][(0, 0)] =
                Complex64::new((1.0 - probability_one).sqrt(), 0.0);
            reference.mps.tensors_mut()[0][(0, 1)] = Complex64::new(probability_one.sqrt(), 0.0);
            for gauge in [0.25, 0.5, 1.0, 2.0, 4.0, 1e-7, 1e7] {
                let mut gauged = reference.clone();
                gauged.mps.tensors_mut()[0] *= Complex64::new(gauge, 0.0);
                gauged.mps.tensors_mut()[1] /= Complex64::new(gauge, 0.0);
                assert!((gauged.mps.norm_squared() - 1.0).abs() < 1e-14);
                assert_eq!(
                    measure::is_mps_trivial(&gauged.mps),
                    probability_one < 1e-12
                );
                assert_eq!(
                    measure::is_mps_trivial(&gauged.mps),
                    measure::is_mps_trivial(&reference.mps)
                );
                let mut expected = reference.clone();
                let actual = gauged.mz(&[QubitId(0)]);
                let result = expected.mz(&[QubitId(0)]);
                assert_eq!(actual[0].outcome, result[0].outcome);
                assert_eq!(actual[0].is_deterministic, result[0].is_deterministic);
                assert_eq!(
                    gauged.rng.clone().next_u64(),
                    expected.rng.clone().next_u64()
                );
                assert_eq!(
                    format!("{:?}", gauged.tableau.stabs()),
                    format!("{:?}", expected.tableau.stabs())
                );
                assert_eq!(
                    format!("{:?}", gauged.tableau.destabs()),
                    format!("{:?}", expected.tableau.destabs())
                );
                for (actual, expected) in gauged
                    .mps
                    .state_vector()
                    .iter()
                    .zip(expected.mps.state_vector())
                {
                    assert!((*actual - expected).norm() < 1e-12);
                }
            }
        }
    }
}

// Independent normalizations can give SVD different tensor signs. Compare
// the full complex coefficient ket (without aligning away a global phase),
// together with the exact tableau generators, rather than its internal gauge.
fn assert_pair_state_close(actual: &StabMps, expected: &StabMps) {
    assert_eq!(
        format!("{:?}", actual.tableau.stabs()),
        format!("{:?}", expected.tableau.stabs())
    );
    assert_eq!(
        format!("{:?}", actual.tableau.destabs()),
        format!("{:?}", expected.tableau.destabs())
    );
    assert!((actual.mps.norm_squared() - 1.0).abs() < 1e-12);
    for (actual, expected) in actual
        .mps
        .state_vector()
        .iter()
        .zip(expected.mps.state_vector())
    {
        assert!(
            (*actual - expected).norm() < 1e-12,
            "coefficient amplitudes differ: {actual} vs {expected}"
        );
    }
}

fn subnormal_measurement_pair(
    scale: f64,
    sites: usize,
    tracked: bool,
    seed: u64,
) -> (StabMps, StabMps) {
    let mut normalized = StabMps::builder(sites).seed(seed).merge_rz(false).build();
    for tensor in normalized.mps.tensors_mut() {
        tensor[(0, 0)] = Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0);
        tensor[(0, 1)] = Complex64::new(0.0, std::f64::consts::FRAC_1_SQRT_2);
    }
    normalized.disent_flags.fill(None);
    let site = sites / 2;
    let mut scaled = normalized.clone();
    scaled.mps.tensors_mut()[site][(0, 0)] = Complex64::new(scale, 0.0);
    scaled.mps.tensors_mut()[site][(0, 1)] = Complex64::new(0.0, scale);
    if tracked {
        scaled.mps.set_tracked_center_for_test(Some(site));
        normalized.mps.set_tracked_center_for_test(Some(site));
    }
    assert!(scaled.mps.norm_squared().is_subnormal());
    (scaled, normalized)
}

#[test]
fn subnormal_exact_mz_matches_normalized_state() {
    for scale in [1e-158, 1e-160] {
        for sites in [1, 3] {
            for tracked in [false, true] {
                for seed in 0..8 {
                    let (mut scaled, mut normalized) =
                        subnormal_measurement_pair(scale, sites, tracked, seed);
                    let actual = scaled.mz(&[QubitId(sites / 2)]);
                    let expected = normalized.mz(&[QubitId(sites / 2)]);
                    assert_eq!(actual[0].outcome, expected[0].outcome);
                    assert_eq!(actual[0].is_deterministic, expected[0].is_deterministic);
                    assert_pair_state_close(&scaled, &normalized);
                    assert_eq!(
                        scaled.rng.clone().next_u64(),
                        normalized.rng.clone().next_u64()
                    );
                }
            }
        }
    }
}

#[test]
fn subnormal_reset_matches_normalized_state() {
    for scale in [1e-158, 1e-160] {
        for sites in [1, 3] {
            for tracked in [false, true] {
                for seed in 0..8 {
                    let (mut scaled, mut normalized) =
                        subnormal_measurement_pair(scale, sites, tracked, seed);
                    assert_eq!(
                        scaled.reset_qubit(QubitId(sites / 2)),
                        normalized.reset_qubit(QubitId(sites / 2))
                    );
                    assert_pair_state_close(&scaled, &normalized);
                    assert_eq!(scaled.disent_flags, normalized.disent_flags);
                }
            }
        }
    }
}

#[test]
fn subnormal_forced_projection_matches_normalized_state() {
    for scale in [1e-158, 1e-160] {
        for sites in [1, 3] {
            for tracked in [false, true] {
                for outcome in [false, true] {
                    let (mut scaled, mut normalized) =
                        subnormal_measurement_pair(scale, sites, tracked, 0);
                    let actual = measure::project_forced_z_with_update(
                        &mut scaled.tableau,
                        &mut scaled.mps,
                        sites / 2,
                        outcome,
                        None,
                    )
                    .unwrap();
                    let expected = measure::project_forced_z_with_update(
                        &mut normalized.tableau,
                        &mut normalized.mps,
                        sites / 2,
                        outcome,
                        None,
                    )
                    .unwrap();
                    assert!(
                        (actual.snapped_probability - expected.snapped_probability).abs() < 1e-12
                    );
                    assert!((actual.survival_ratio - expected.survival_ratio).abs() < 1e-12);
                    assert_pair_state_close(&scaled, &normalized);
                }
            }
        }
    }
}

#[test]
fn subnormal_normalize_after_gate_matches_normalized_state() {
    for scale in [1e-158, 1e-160] {
        for sites in [1, 3] {
            for tracked in [false, true] {
                let (mut scaled, mut normalized) =
                    subnormal_measurement_pair(scale, sites, tracked, 0);
                for simulator in [&mut scaled, &mut normalized] {
                    simulator.rz(Angle64::from_radians(0.37), &[QubitId(sites / 2)]);
                }
                assert_pair_state_close(&scaled, &normalized);
            }
        }
    }
}
