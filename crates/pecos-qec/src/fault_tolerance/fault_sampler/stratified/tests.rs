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

use super::*;
use crate::fault_tolerance::fault_sampler::{FaultChannel, StochasticNoiseParams};
use pecos_core::{PauliString, QubitId};
use pecos_quantum::{Attribute, TickCircuit};

fn catalog() -> FaultCatalog {
    let mut circuit = TickCircuit::new();
    circuit.tick().pz(&[QubitId(0), QubitId(1)]);
    circuit.tick().h(&[QubitId(0), QubitId(1)]);
    circuit.tick().cx(&[(QubitId(0), QubitId(1))]);
    circuit.tick().mz(&[QubitId(0), QubitId(1)]);
    circuit.set_meta("num_measurements", Attribute::String("2".into()));
    circuit
        .add_detector_metadata(&[-2, -1], None, Some("D"), Some(0))
        .unwrap();
    circuit
        .add_observable_metadata(&[-2], Some(0), Some("L"))
        .unwrap();
    circuit.tracked_pauli_labeled("Z0", PauliString::z(0));
    FaultCatalog::from_circuit(&circuit)
        .unwrap()
        .parameterized(&StochasticNoiseParams {
            p1: 0.09,
            p2: 0.17,
            p_meas: 0.12,
            p_prep: 0.0,
        })
}

fn distinct_catalog() -> FaultCatalog {
    let mut catalog = catalog();
    for (i, loc) in catalog.locations.iter_mut().enumerate() {
        let p = 0.04 + 0.025 * count_f64(i);
        loc.channel_probability = p;
        loc.no_fault_probability = 1.0 - p;
        // Nonuniform alternatives with a zero-probability alternative exercise
        // the actual stored weights, independently of conditional_probability.
        let n = loc.faults.len();
        let denominator: f64 = (1..=n).map(count_f64).sum();
        for (a, alt) in loc.faults.iter_mut().enumerate() {
            alt.absolute_probability = p * count_f64(a + 1) / denominator;
        }
        if n > 1 {
            let removed = loc.faults[0].absolute_probability;
            loc.faults[0].absolute_probability = 0.0;
            loc.faults[1].absolute_probability += removed;
        }
    }
    catalog
}

fn active_count(catalog: &FaultCatalog) -> usize {
    catalog
        .locations
        .iter()
        .filter(|l| l.faults.iter().any(|a| a.absolute_probability > 0.0))
        .count()
}

fn close(actual: f64, expected: f64, relative: f64) {
    assert!(
        (actual - expected).abs() <= relative * expected.abs().max(1e-300),
        "actual={actual:e}, expected={expected:e}"
    );
}

#[test]
fn pmf_matches_enumeration_and_fire_patterns() {
    for catalog in [catalog(), distinct_catalog()] {
        let n = active_count(&catalog);
        let pmf = catalog.fault_count_pmf(n).unwrap();
        let mut brute = vec![0.0; n + 1];
        let active: Vec<_> = catalog
            .locations
            .iter()
            .filter(|l| l.faults.iter().any(|a| a.absolute_probability > 0.0))
            .collect();
        for mask in 0_usize..(1 << n) {
            let probability: f64 = active
                .iter()
                .enumerate()
                .map(|(i, loc)| {
                    if mask & (1 << i) == 0 {
                        1.0 - loc.channel_probability
                    } else {
                        loc.channel_probability
                    }
                })
                .product();
            brute[usize::try_from(mask.count_ones()).unwrap()] += probability;
        }
        for (k, &mass) in brute.iter().enumerate() {
            let enumerated: f64 = catalog
                .fault_configurations(k)
                .map(|c| c.configuration_probability)
                .sum();
            close(pmf.masses()[k], enumerated, 1e-12);
            close(pmf.masses()[k], mass, 1e-12);
            close(pmf.log_masses()[k].exp(), mass, 1e-12);
        }
    }
}

fn log_binomial_mass(n: usize, k: usize, p: f64) -> f64 {
    pecos_num::ln_gamma(count_f64(n + 1)).unwrap()
        - pecos_num::ln_gamma(count_f64(k + 1)).unwrap()
        - pecos_num::ln_gamma(count_f64(n - k + 1)).unwrap()
        + count_f64(k) * p.ln()
        + count_f64(n - k) * (-p).ln_1p()
}

#[test]
fn large_pmf_log_masses_match_closed_forms() {
    let template = catalog()
        .locations
        .into_iter()
        .find(|l| l.channel == FaultChannel::PMeas)
        .unwrap();
    let mut catalog = FaultCatalog {
        locations: vec![template; 100_000],
    };
    catalog.with_noise(&StochasticNoiseParams {
        p1: 0.0,
        p2: 0.0,
        p_meas: 0.01,
        p_prep: 0.0,
    });
    let pmf = catalog.fault_count_pmf(1002).unwrap();
    assert!(pmf.log_masses().iter().all(|p| p.is_finite()));
    assert_eq!(pmf.masses()[0].to_bits(), 0.0_f64.to_bits());
    for k in 998..=1002 {
        assert!(pmf.masses()[k] > 0.0);
        close(
            pmf.masses()[k],
            log_binomial_mass(100_000, k, 0.01).exp(),
            1e-8,
        );
    }
    for config in catalog.sample_fault_configurations(1, 3, 321).unwrap() {
        assert_eq!(config.location_indices.len(), 1);
    }
    for (i, loc) in catalog.locations.iter_mut().enumerate() {
        let p = if i < 50_000 { 0.005 } else { 0.015 };
        loc.channel_probability = p;
        loc.no_fault_probability = 1.0 - p;
        loc.faults[0].absolute_probability = p;
    }
    let pmf = catalog.fault_count_pmf(1002).unwrap();
    assert!(pmf.log_masses().iter().all(|p| p.is_finite()));
    for k in 998..=1002 {
        let terms: Vec<_> = (0..=k)
            .map(|j| log_binomial_mass(50_000, j, 0.005) + log_binomial_mass(50_000, k - j, 0.015))
            .collect();
        let maximum = terms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let expected = maximum.exp() * terms.iter().map(|x| (x - maximum).exp()).sum::<f64>();
        close(pmf.masses()[k], expected, 1e-8);
    }
}

#[test]
fn tail_bounds_cover_exact_tails() {
    let mut catalog = catalog();
    catalog.with_noise(&StochasticNoiseParams {
        p1: 0.65,
        p2: 0.7,
        p_meas: 0.6,
        p_prep: 0.0,
    });
    let n = active_count(&catalog);
    let full: Vec<f64> = (0..=n)
        .map(|k| {
            catalog
                .fault_configurations(k)
                .map(|c| c.configuration_probability)
                .sum()
        })
        .collect();
    let mean: f64 = catalog
        .locations
        .iter()
        .map(|l| l.channel_probability)
        .sum();
    for max_k in 0..=n {
        let bound = catalog.fault_count_pmf(max_k).unwrap().tail_bound();
        let exact: f64 = full.iter().skip(max_k + 1).sum();
        assert!(bound >= exact, "{max_k}: {bound} < {exact}");
        if count_f64(max_k + 1) <= mean {
            assert!((bound - 1.0).abs() < f64::EPSILON);
        } else {
            let a = count_f64(max_k + 1);
            close(
                bound,
                (-mean).exp() * (std::f64::consts::E * mean / a).powf(a),
                1e-13,
            );
        }
    }
    catalog.with_noise(&StochasticNoiseParams {
        p1: 0.0,
        p2: 0.0,
        p_meas: 0.0,
        p_prep: 0.0,
    });
    let pmf = catalog.fault_count_pmf(0).unwrap();
    assert_eq!(pmf.tail_bound().to_bits(), 0.0_f64.to_bits());
    assert_eq!(pmf.masses()[0].to_bits(), 1.0_f64.to_bits());
}

fn key(config: &FaultConfiguration) -> (Vec<usize>, Vec<usize>) {
    (
        config.location_indices.clone(),
        config.alternative_indices.clone(),
    )
}

fn check_config(actual: &FaultConfiguration, expected: &FaultConfiguration) {
    assert_eq!(key(actual), key(expected));
    assert_eq!(actual.affected_measurements, expected.affected_measurements);
    assert_eq!(actual.affected_detectors, expected.affected_detectors);
    assert_eq!(actual.affected_observables, expected.affected_observables);
    assert_eq!(
        actual.affected_tracked_paulis,
        expected.affected_tracked_paulis
    );
    close(
        actual.selected_probability,
        expected.selected_probability,
        1e-15,
    );
    close(
        actual.configuration_probability,
        expected.configuration_probability,
        1e-15,
    );
}

// For even degrees of freedom 2m, the chi-square survival function is exactly
// exp(-x/2) sum_{j=0}^{m-1} (x/2)^j/j!. Merge cells to obtain even df.
fn chi_square_survival(statistic: f64, degrees: usize) -> f64 {
    assert_eq!(degrees % 2, 0);
    let x = statistic / 2.0;
    let logs: Vec<_> = (0..degrees / 2)
        .map(|j| -x + count_f64(j) * x.ln() - pecos_num::ln_gamma(count_f64(j + 1)).unwrap())
        .collect();
    let maximum = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    maximum.exp() * logs.iter().map(|v| (v - maximum).exp()).sum::<f64>()
}

#[test]
fn conditional_law_chi_square_and_effects() {
    for catalog in [catalog(), distinct_catalog()] {
        for k in 1..=3 {
            let oracle: BTreeMap<_, _> = catalog
                .fault_configurations(k)
                .map(|c| (key(&c), c))
                .collect();
            let mass: f64 = oracle.values().map(|c| c.configuration_probability).sum();
            let shots = 200_000;
            let mut observed = BTreeMap::<_, u32>::new();
            for config in catalog
                .sample_fault_configurations(k, shots, 987_654)
                .unwrap()
            {
                assert_eq!(config.location_indices.len(), k);
                assert!(config.location_indices.windows(2).all(|w| w[0] < w[1]));
                for (&i, &a) in config
                    .location_indices
                    .iter()
                    .zip(&config.alternative_indices)
                {
                    assert!(catalog.locations[i].faults[a].absolute_probability > 0.0);
                }
                check_config(&config, &oracle[&key(&config)]);
                *observed.entry(key(&config)).or_default() += 1;
            }
            let mut cells: Vec<(f64, f64)> = Vec::new();
            let mut small = (0.0, 0.0);
            for (key, config) in &oracle {
                let expected = count_f64(shots) * config.configuration_probability / mass;
                let actual = f64::from(observed.get(key).copied().unwrap_or(0));
                if expected < 5.0 {
                    small.0 += expected;
                    small.1 += actual;
                } else {
                    cells.push((expected, actual));
                }
            }
            if small.0 > 0.0 {
                cells[0].0 += small.0;
                cells[0].1 += small.1;
            }
            if cells.len().is_multiple_of(2) {
                let last = cells.pop().unwrap();
                cells[0].0 += last.0;
                cells[0].1 += last.1;
            }
            assert!(cells.iter().all(|(e, _)| *e >= 5.0));
            let chi: f64 = cells.iter().map(|(e, o)| (o - e).powi(2) / e).sum();
            let p_value = chi_square_survival(chi, cells.len() - 1);
            // Fixed-seed Pearson goodness-of-fit, significance alpha=1e-4.
            assert!(p_value > 1e-4, "k={k}, chi={chi}, p={p_value}");
        }
    }
}

#[test]
fn seeds_and_boundary_strata() {
    let catalog = catalog();
    let a = catalog.sample_fault_configurations(2, 100, 12).unwrap();
    let b = catalog.sample_fault_configurations(2, 100, 12).unwrap();
    for (a, b) in a.iter().zip(&b) {
        check_config(a, b);
    }
    let different = catalog.sample_fault_configurations(2, 100, 13).unwrap();
    assert_ne!(
        a.iter().map(key).collect::<Vec<_>>(),
        different.iter().map(key).collect::<Vec<_>>()
    );
    let zero = catalog.sample_fault_configurations(0, 3, 1).unwrap();
    for config in zero {
        check_config(&config, &catalog.fault_configurations(0).next().unwrap());
    }
    let n = active_count(&catalog);
    assert!(matches!(
        catalog.fault_count_pmf(n + 1),
        Err(FaultCountError::FaultCountTooLarge { .. })
    ));
    assert!(matches!(
        catalog.sample_fault_configurations(n + 1, 0, 0),
        Err(FaultCountError::FaultCountTooLarge { .. })
    ));
    for config in catalog.sample_fault_configurations(n, 100, 2).unwrap() {
        assert_eq!(config.location_indices.len(), n);
        assert!(config.location_indices.windows(2).all(|w| w[0] < w[1]));
    }
    assert!(
        catalog
            .sample_fault_configurations(0, 0, 1)
            .unwrap()
            .is_empty()
    );
    let empty = FaultCatalog { locations: vec![] };
    assert_eq!(
        empty.fault_count_pmf(0).unwrap().masses()[0].to_bits(),
        1.0_f64.to_bits()
    );
    assert_eq!(empty.sample_fault_configurations(0, 1, 0).unwrap().len(), 1);
}

fn rejects(catalog: &FaultCatalog, expected: &str) {
    for error in [
        catalog.fault_count_pmf(0).unwrap_err(),
        catalog.sample_fault_configurations(0, 0, 0).unwrap_err(),
    ] {
        assert!(error.to_string().contains(expected), "{error}");
    }
}

#[test]
fn inconsistent_catalogs_are_rejected() {
    let baseline = distinct_catalog();
    let mut c = baseline.clone();
    c.locations[0].faults[0].absolute_probability *= 0.5;
    rejects(&c, "must sum");
    let mut c = baseline.clone();
    c.locations[0].no_fault_probability *= 0.5;
    rejects(&c, "must equal");
    let mut c = baseline.clone();
    c.locations[0]
        .faults
        .iter_mut()
        .for_each(|a| a.absolute_probability = 0.0);
    rejects(&c, "inactive location");
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.01] {
        for field in 0..4 {
            let mut c = baseline.clone();
            match field {
                0 => c.locations[0].channel_probability = invalid,
                1 => c.locations[0].no_fault_probability = invalid,
                2 => c.locations[0].faults[0].absolute_probability = invalid,
                _ => c.locations[0].faults[0].conditional_probability = invalid,
            }
            rejects(&c, "finite and non-negative");
        }
    }
    let mut c = baseline.clone();
    c.locations[0].channel_probability = 0.0;
    rejects(&c, "must sum");
    // Sum overflow, while each individual field is finite.
    let mut c = baseline.clone();
    let loc = c.locations.iter_mut().find(|l| l.faults.len() > 1).unwrap();
    loc.faults
        .iter_mut()
        .for_each(|a| a.absolute_probability = f64::MAX);
    rejects(&c, "must sum");
    for p in [1.0, 1.2] {
        let mut c = baseline.clone();
        c.locations[0].channel_probability = p;
        rejects(&c, ">= 1");
    }
    let k = active_count(&baseline) + 1;
    assert!(matches!(
        baseline.fault_count_pmf(k),
        Err(FaultCountError::FaultCountTooLarge { .. })
    ));
    assert!(matches!(
        baseline.sample_fault_configurations(k, 0, 0),
        Err(FaultCountError::FaultCountTooLarge { .. })
    ));
}

#[test]
fn consistency_tolerance_and_inactive_channel() {
    let mut c = catalog();
    let inactive = c
        .locations
        .iter_mut()
        .find(|l| l.channel == FaultChannel::PPrep)
        .unwrap();
    // The cursor uses alternatives, not the channel field, to decide activity.
    inactive.channel_probability = 0.5;
    let n = active_count(&c);
    let pmf = c.fault_count_pmf(n).unwrap();
    for k in 0..=n {
        close(
            pmf.masses()[k],
            c.fault_configurations(k)
                .map(|v| v.configuration_probability)
                .sum(),
            1e-12,
        );
    }
    let loc = c
        .locations
        .iter_mut()
        .find(|l| l.channel == FaultChannel::PMeas)
        .unwrap();
    loc.faults[0].absolute_probability *= 1.0 + 5e-13;
    loc.no_fault_probability *= 1.0 + 5e-13;
    assert!(c.fault_count_pmf(n).is_ok());
}

#[test]
fn estimator_arithmetic_and_omissions() {
    let pmf = catalog().fault_count_pmf(3).unwrap();
    let counts = [
        FaultStratumCounts {
            k: 0,
            attempted: 100,
            survived: 80,
            failed: 10,
        },
        FaultStratumCounts {
            k: 2,
            attempted: 200,
            survived: 120,
            failed: 30,
        },
        FaultStratumCounts {
            k: 3,
            attempted: 50,
            survived: 25,
            failed: 0,
        },
    ];
    let estimate = StratifiedEstimate::from_counts(&pmf, &counts).unwrap();
    let p = pmf.masses();
    let a = p[0] * 0.1 + p[2] * 0.15;
    let b = p[0] * 0.8 + p[2] * 0.6 + p[3] * 0.5;
    let r = a / b;
    let va = p[0].powi(2) * 0.1 * 0.9 / 100.0 + p[2].powi(2) * 0.15 * 0.85 / 200.0;
    let vb = p[0].powi(2) * 0.8 * 0.2 / 100.0
        + p[2].powi(2) * 0.6 * 0.4 / 200.0
        + p[3].powi(2) * 0.5 * 0.5 / 50.0;
    let vr: f64 = [
        (0, 100.0, 80.0, 10.0),
        (2, 200.0, 120.0, 30.0),
        (3, 50.0, 25.0, 0.0),
    ]
    .iter()
    .map(|&(k, n, s, f)| {
        let v = (f * (1.0 - r).powi(2) + (s - f) * r.powi(2)) / n - (f / n - r * s / n).powi(2);
        p[k].powi(2) * v / n / b.powi(2)
    })
    .sum();
    close(estimate.failure_probability, a, 1e-14);
    close(estimate.survival_probability, b, 1e-14);
    close(estimate.failure_given_survival.unwrap(), r, 1e-14);
    close(estimate.failure_standard_error, va.sqrt(), 1e-14);
    close(estimate.survival_standard_error, vb.sqrt(), 1e-14);
    close(estimate.ratio_standard_error.unwrap(), vr.sqrt(), 1e-14);
    close(estimate.unsampled_mass, p[1], 1e-14);
    close(estimate.tail_bound, pmf.tail_bound(), 1e-14);
    let zero = StratifiedEstimate::from_counts(
        &pmf,
        &[FaultStratumCounts {
            k: 1,
            attempted: 100,
            survived: 0,
            failed: 0,
        }],
    )
    .unwrap();
    assert!(zero.failure_given_survival.is_none());
    assert!(zero.ratio_standard_error.is_none());
    close(zero.unsampled_mass, p[0] + p[2] + p[3], 1e-14);
    let empty = StratifiedEstimate::from_counts(&pmf, &[]).unwrap();
    close(empty.unsampled_mass, p.iter().sum(), 1e-14);
    assert!(empty.failure_given_survival.is_none());
}

#[test]
fn estimator_rejects_invalid_counts() {
    let pmf = catalog().fault_count_pmf(2).unwrap();
    for (attempted, survived, failed) in [(0, 0, 0), (10, 5, 6), (10, 11, 1)] {
        let counts = [FaultStratumCounts {
            k: 0,
            attempted,
            survived,
            failed,
        }];
        assert!(matches!(
            StratifiedEstimate::from_counts(&pmf, &counts),
            Err(FaultCountError::InvalidCounts { .. })
        ));
    }
    let outside = FaultStratumCounts {
        k: 3,
        attempted: 1,
        survived: 1,
        failed: 0,
    };
    assert!(matches!(
        StratifiedEstimate::from_counts(&pmf, &[outside]),
        Err(FaultCountError::StratumOutsidePmf { .. })
    ));
    let valid = FaultStratumCounts { k: 0, ..outside };
    assert!(matches!(
        StratifiedEstimate::from_counts(&pmf, &[valid, valid]),
        Err(FaultCountError::DuplicateStratum { .. })
    ));
}

#[test]
fn repetition_memory_matches_enumeration_and_plain_sampling() {
    let mut circuit = TickCircuit::new();
    circuit.tick().pz(&[QubitId(0), QubitId(1), QubitId(2)]);
    circuit.tick().cx(&[(QubitId(0), QubitId(1))]);
    circuit.tick().cx(&[(QubitId(0), QubitId(2))]);
    circuit.tick().mz(&[QubitId(0), QubitId(1), QubitId(2)]);
    circuit.set_meta("num_measurements", Attribute::String("3".into()));
    circuit
        .add_detector_metadata(&[-3, -2], None, Some("D0"), Some(0))
        .unwrap();
    circuit
        .add_detector_metadata(&[-2, -1], None, Some("D1"), Some(1))
        .unwrap();
    circuit
        .add_observable_metadata(&[-3], Some(0), Some("logical Z"))
        .unwrap();
    let catalog =
        FaultCatalog::from_circuit(&circuit)
            .unwrap()
            .parameterized(&StochasticNoiseParams {
                p1: 0.02,
                p2: 0.02,
                p_meas: 0.02,
                p_prep: 0.0,
            });
    let max_k = 4;
    let shots = 40_000;
    let pmf = catalog.fault_count_pmf(max_k).unwrap();
    let mut counts = Vec::new();
    let mut exact_failure = 0.0;
    for k in 0..=max_k {
        let failed = catalog
            .sample_fault_configurations(k, shots, 900 + u64::try_from(k).unwrap())
            .unwrap()
            .iter()
            .filter(|c| !c.affected_observables.is_empty())
            .count();
        if k > 0 {
            assert!(failed > 0);
        }
        counts.push(FaultStratumCounts {
            k,
            attempted: shots,
            survived: shots,
            failed,
        });
        exact_failure += catalog
            .fault_configurations(k)
            .filter(|c| !c.affected_observables.is_empty())
            .map(|c| c.configuration_probability)
            .sum::<f64>();
    }
    assert!(exact_failure > 0.0);
    let estimate = StratifiedEstimate::from_counts(&pmf, &counts).unwrap();
    assert!(estimate.failure_standard_error > 0.0);
    assert!(pmf.tail_bound() < 0.1 * estimate.failure_standard_error);
    assert!(
        (estimate.failure_probability - exact_failure).abs()
            < 4.0 * estimate.failure_standard_error
    );
    let plain_shots = 300_000;
    let mut rng = PecosRng::seed_from_u64(70);
    let mut failures = 0_u32;
    for _ in 0..plain_shots {
        let mut observable = BTreeSet::new();
        for loc in &catalog.locations {
            if rng.random::<f64>() < loc.channel_probability {
                let threshold = rng.random::<f64>() * loc.channel_probability;
                let mut cumulative = 0.0;
                let alt = loc
                    .faults
                    .iter()
                    .find(|alt| {
                        cumulative += alt.absolute_probability;
                        cumulative > threshold
                    })
                    .unwrap();
                for &o in &alt.affected_observables {
                    if !observable.remove(&o) {
                        observable.insert(o);
                    }
                }
            }
        }
        failures += u32::from(!observable.is_empty());
    }
    assert!(failures > 0);
    let plain = f64::from(failures) / f64::from(plain_shots);
    let plain_variance = plain * (1.0 - plain) / f64::from(plain_shots);
    assert!(
        (estimate.failure_probability - plain).abs()
            < 4.0 * (estimate.failure_standard_error.powi(2) + plain_variance).sqrt()
                + pmf.tail_bound()
    );
}

#[test]
fn positive_tail_bound_does_not_underflow_to_zero() {
    let mut c = catalog();
    c.with_noise(&StochasticNoiseParams {
        p1: 1e-200,
        p2: 1e-200,
        p_meas: 1e-200,
        p_prep: 0.0,
    });
    // There are more than two active locations, so this tail is positive even
    // though both the probability and its Chernoff expression underflow f64.
    assert!(c.fault_count_pmf(2).unwrap().tail_bound() > 0.0);
}

#[test]
fn log_sum_matches_probability_addition_with_zero_terms() {
    for (a, b) in [
        (0.0_f64, 0.5_f64),
        (0.5, 0.0),
        (0.25, 0.5),
        (1e-200, 1e-200),
    ] {
        close(log_add(a.ln(), b.ln()).exp(), a + b, 1e-12);
    }
    let zero = log_add(f64::NEG_INFINITY, f64::NEG_INFINITY);
    assert!(zero.is_infinite() && zero.is_sign_negative());
}

#[test]
fn oversized_batch_is_an_error() {
    assert!(matches!(
        catalog().sample_fault_configurations(1, usize::MAX, 0),
        Err(FaultCountError::ConfigurationAllocationFailed {
            requested: usize::MAX
        })
    ));
}

#[test]
fn standard_errors_scale_with_tiny_masses() {
    let counts = [FaultStratumCounts {
        k: 1,
        attempted: 100,
        survived: 50,
        failed: 10,
    }];
    let estimate = |mass: f64| {
        let pmf = FaultCountPmf {
            masses: vec![0.0, mass],
            log_masses: vec![f64::NEG_INFINITY, mass.ln()],
            tail_bound: 0.0,
        };
        StratifiedEstimate::from_counts(&pmf, &counts).unwrap()
    };
    let unit = estimate(1.0);
    let tiny = estimate(1e-170);
    // Squaring a 1e-170 mass underflows f64; the standard errors must not.
    close(
        tiny.failure_standard_error,
        1e-170 * unit.failure_standard_error,
        1e-12,
    );
    close(
        tiny.survival_standard_error,
        1e-170 * unit.survival_standard_error,
        1e-12,
    );
    close(
        tiny.ratio_standard_error.unwrap(),
        unit.ratio_standard_error.unwrap(),
        1e-12,
    );
}

#[test]
fn tail_bound_never_exceeds_one() {
    let mut c = catalog();
    let p = 1.0_f64.next_down();
    c.with_noise(&StochasticNoiseParams {
        p1: p,
        p2: p,
        p_meas: p,
        p_prep: 0.0,
    });
    let template = c.locations[c.locations.len() - 1].clone();
    c.locations.resize(1000, template);
    let active = active_count(&c);
    // With K nearly deterministic, max_k + 1 sits just above the mean, where the
    // unclamped Chernoff expression rounds above one.
    let bound = c.fault_count_pmf(active - 1).unwrap().tail_bound();
    assert!(bound <= 1.0, "bound={bound}");
}

fn measurement_catalog(locations: usize, probability: f64) -> FaultCatalog {
    let mut circuit = TickCircuit::new();
    circuit.tick().mz(&[QubitId(0)]);
    circuit.set_meta("num_measurements", Attribute::String("1".into()));
    circuit.set_meta("detectors", Attribute::String("[]".into()));
    circuit.set_meta("observables", Attribute::String("[]".into()));
    let mut catalog =
        FaultCatalog::from_circuit(&circuit)
            .unwrap()
            .parameterized(&StochasticNoiseParams {
                p1: 0.0,
                p2: 0.0,
                p_meas: probability,
                p_prep: 0.0,
            });
    assert_eq!(catalog.locations.len(), 1);
    catalog
        .locations
        .resize(locations, catalog.locations[0].clone());
    catalog
}

#[test]
fn log_estimator_preserves_tiny_variance_beside_unit_mass() {
    let p = 1e-170;
    let pmf = measurement_catalog(1, p).fault_count_pmf(1).unwrap();
    let counts = [
        FaultStratumCounts {
            k: 0,
            attempted: 100,
            survived: 0,
            failed: 0,
        },
        FaultStratumCounts {
            k: 1,
            attempted: 100,
            survived: 50,
            failed: 10,
        },
    ];
    let estimate = StratifiedEstimate::from_counts(&pmf, &counts).unwrap();
    close(estimate.failure_probability, p * 0.1, 1e-12);
    close(estimate.survival_probability, p * 0.5, 1e-12);
    close(
        estimate.failure_standard_error,
        p * (0.1_f64 * 0.9 / 100.0).sqrt(),
        1e-12,
    );
    close(
        estimate.survival_standard_error,
        p * (0.5_f64 * 0.5 / 100.0).sqrt(),
        1e-12,
    );
    close(estimate.failure_given_survival.unwrap(), 0.2, 1e-12);
    close(
        estimate.ratio_standard_error.unwrap(),
        (0.2_f64 * 0.8 / 50.0).sqrt(),
        1e-12,
    );
}

fn check_subnormal_ratio(attempted: usize) {
    let pmf = measurement_catalog(1, 1e-322).fault_count_pmf(1).unwrap();
    assert!(pmf.masses()[1].is_subnormal());
    let counts = [FaultStratumCounts {
        k: 1,
        attempted,
        survived: attempted / 2,
        failed: attempted / 10,
    }];
    let estimate = StratifiedEstimate::from_counts(&pmf, &counts).unwrap();
    // With a single sampled stratum its mass cancels: a binomial proportion
    // among the surviving shots has variance R(1-R)/S.
    close(estimate.failure_given_survival.unwrap(), 0.2, 1e-12);
    close(
        estimate.ratio_standard_error.unwrap(),
        (0.2 * 0.8 / count_f64(attempted / 2)).sqrt(),
        1e-12,
    );
}

#[test]
fn log_estimator_preserves_ratio_se_with_subnormal_mass_100_shots() {
    check_subnormal_ratio(100);
}

#[test]
fn log_estimator_preserves_ratio_se_with_subnormal_mass_10000_shots() {
    check_subnormal_ratio(10_000);
}

#[test]
fn log_estimator_defines_ratio_when_sampled_masses_underflow() {
    let pmf = measurement_catalog(2, 1e-200).fault_count_pmf(2).unwrap();
    let mut counts = [FaultStratumCounts {
        k: 2,
        attempted: 100,
        survived: 50,
        failed: 10,
    }];
    assert_eq!(pmf.masses()[2].to_bits(), 0.0_f64.to_bits());
    assert!(pmf.log_masses()[2].is_finite());
    let estimate = StratifiedEstimate::from_counts(&pmf, &counts).unwrap();
    assert_eq!(estimate.failure_probability.to_bits(), 0.0_f64.to_bits());
    assert_eq!(estimate.survival_probability.to_bits(), 0.0_f64.to_bits());
    close(estimate.failure_given_survival.unwrap(), 0.2, 1e-12);
    close(
        estimate.ratio_standard_error.unwrap(),
        (0.2_f64 * 0.8 / 50.0).sqrt(),
        1e-12,
    );
    // Observing no failures still gives a defined zero ratio and plug-in SE
    // when there are survivors, even if their total mass underflows.
    counts[0].failed = 0;
    let no_failures = StratifiedEstimate::from_counts(&pmf, &counts).unwrap();
    assert_eq!(
        no_failures.failure_given_survival.unwrap().to_bits(),
        0.0_f64.to_bits()
    );
    assert_eq!(
        no_failures.ratio_standard_error.unwrap().to_bits(),
        0.0_f64.to_bits()
    );
}

fn check_nearly_deterministic_tail(n: usize) {
    let p = 1.0_f64.next_down();
    let bound = measurement_catalog(n, p)
        .fault_count_pmf(n - 1)
        .unwrap()
        .tail_bound();
    // K > n-1 means every location fires, independently: P(K=n) = p^n.
    let exact = (count_f64(n) * p.ln()).exp();
    assert!(
        bound >= exact,
        "n={n}: bound={bound:.17e}, exact={exact:.17e}"
    );
    assert!(bound <= 1.0, "n={n}: bound={bound:.17e}");
}

#[test]
fn tail_bound_covers_nearly_deterministic_181_locations() {
    check_nearly_deterministic_tail(181);
}

#[test]
fn tail_bound_covers_nearly_deterministic_sweep() {
    for n in [10, 100, 181, 1000, 10_000] {
        check_nearly_deterministic_tail(n);
    }
}

#[test]
fn tail_bound_covers_far_regime_small_means() {
    for p in [1e-20, 1e-170] {
        let bound = measurement_catalog(1, p)
            .fault_count_pmf(0)
            .unwrap()
            .tail_bound();
        let chernoff = (-p).exp() * std::f64::consts::E * p;
        assert!(bound >= p, "bound={bound:e}, exact tail={p:e}");
        close(bound, chernoff, 1e-12);
    }
}

#[test]
fn tail_bound_is_continuous_across_regime_switch() {
    let mut p = 0.5_f64;
    for _ in 0..4 {
        p = p.next_down();
    }
    let mut previous: Option<f64> = None;
    for _ in 0..9 {
        // One location and a=1 put the switch exactly at p=1/2;
        // the exact tail is p and the Chernoff value is p * exp(1-p).
        let bound = measurement_catalog(1, p)
            .fault_count_pmf(0)
            .unwrap()
            .tail_bound();
        let chernoff = p * (1.0 - p).exp();
        assert!(bound >= p);
        assert!(bound <= 1.0);
        close(bound, chernoff, 4.0 * f64::EPSILON);
        if let Some(previous) = previous {
            close(bound, previous, 4.0 * f64::EPSILON);
        }
        previous = Some(bound);
        p = p.next_up();
    }
}
