// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0.

use super::*;
use crate::MeasurementCase;
use std::collections::BTreeMap;

// Limit joint noise/measurement leaves, independently of register width.
const MAX_BRANCHES: usize = 4096;
pub(super) type Distribution = BTreeMap<Vec<bool>, f64>;

#[derive(Debug, Default)]
pub(super) struct JointDistribution {
    pub symbols: Distribution,
    pub records: Distribution,
}

pub(super) fn conditional(
    state: &StabActive,
    pauli: &VirtualPauli,
    negative: bool,
) -> (f64, MeasurementCase) {
    let parts = crate::coordinate_tableau::decompose(
        &state.tableau,
        &state.active,
        pauli.factors(),
        negative,
    );
    (state.probability_one(&parts), parts.measurement_case())
}

fn check_mass(distribution: &Distribution) {
    assert!(distribution.values().all(|p| p.is_finite() && *p >= 0.0));
    let total: f64 = distribution.values().sum();
    assert!((total - 1.0).abs() < 1e-10, "total probability {total}");
}

pub(super) fn assert_distribution(actual: &Distribution, expected: &Distribution) {
    for key in actual.keys().chain(expected.keys()) {
        let a = actual.get(key).copied().unwrap_or(0.0);
        let b = expected.get(key).copied().unwrap_or(0.0);
        assert!((a - b).abs() < 1e-10, "outcomes {key:?}: {a} != {b}");
    }
}

pub(super) fn assert_joint(actual: &JointDistribution, expected: &JointDistribution) {
    assert_distribution(&actual.symbols, &expected.symbols);
    assert_distribution(&actual.records, &expected.records);
}

fn enumerate_measurements(
    program: &HeisenbergProgram,
    noise: &[bool],
    branches: &mut usize,
) -> JointDistribution {
    assert_eq!(noise.len(), program.num_noise_symbols);
    let mut distribution = JointDistribution::default();
    let mut pending = vec![Vec::new()];
    while let Some(prefix) = pending.pop() {
        *branches += 1;
        assert!(
            *branches <= MAX_BRANCHES,
            "joint oracle exceeds {MAX_BRANCHES} noise/measurement branches"
        );
        let mut choices = Vec::new();
        let mut probability = 1.0;
        let (shot, outcomes) = program.execute_with_outcomes(
            StabActive::with_seed(program.num_qubits, 0),
            noise,
            |state, pauli, negative, _, _| {
                let (p, _) = conditional(state, pauli, negative);
                assert!(p.is_finite() && (0.0..=1.0).contains(&p));
                // Forcing an impossible deterministic result is ignored by the
                // executor, so it must not create a second branch here.
                if p <= 0.0 || p >= 1.0 {
                    return Some(p >= 1.0);
                }
                let outcome = if let Some(&outcome) = prefix.get(choices.len()) {
                    outcome
                } else {
                    let mut alternative = choices.clone();
                    alternative.push(true);
                    pending.push(alternative);
                    false
                };
                choices.push(outcome);
                probability *= if outcome { p } else { 1.0 - p };
                Some(outcome)
            },
        );
        assert!(probability.is_finite() && probability >= 0.0);
        *distribution.symbols.entry(outcomes).or_default() += probability;
        *distribution.records.entry(shot.records).or_default() += probability;
    }
    check_mass(&distribution.symbols);
    check_mass(&distribution.records);
    distribution
}

pub(super) fn fixed_noise(program: &HeisenbergProgram, noise: &[bool]) -> JointDistribution {
    enumerate_measurements(program, noise, &mut 0)
}

pub(super) fn with_noise(program: &HeisenbergProgram) -> JointDistribution {
    let mut alternatives = vec![(1.0, vec![false; program.num_noise_symbols])];
    for channel in &program.noise_channels {
        let total: f64 = channel.alternatives.iter().map(|(p, _)| p).sum();
        assert!(total.is_finite() && total > 0.0);
        let mut next = Vec::new();
        for (weight, bits) in alternatives {
            for (p, values) in &channel.alternatives {
                assert!(p.is_finite() && *p >= 0.0);
                if *p == 0.0 {
                    continue;
                }
                assert!(
                    next.len() < MAX_BRANCHES,
                    "joint oracle exceeds {MAX_BRANCHES} noise/measurement branches"
                );
                let mut bits = bits.clone();
                bits[channel.first_symbol..channel.first_symbol + values.len()]
                    .copy_from_slice(values);
                next.push((weight * p / total, bits));
            }
        }
        alternatives = next;
    }
    let mut distribution = JointDistribution::default();
    let mut branches = 0;
    for (weight, bits) in alternatives {
        let fixed = enumerate_measurements(program, &bits, &mut branches);
        for (key, p) in fixed.symbols {
            let probability = weight * p;
            assert!(probability.is_finite() && probability >= 0.0);
            *distribution.symbols.entry(key).or_default() += probability;
        }
        for (key, p) in fixed.records {
            *distribution.records.entry(key).or_default() += weight * p;
        }
    }
    check_mass(&distribution.symbols);
    check_mass(&distribution.records);
    distribution
}
