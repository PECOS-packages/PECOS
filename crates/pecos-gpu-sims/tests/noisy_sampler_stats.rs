// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
// except in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0

//! Statistical audit of `GpuNoisySampler` + `DepolarizingNoiseSampler`.
//!
//! Existing tests check API surface and deterministic seed-replay. These
//! add distribution-shape checks that would catch a noise-rate off by a
//! factor (e.g. the sampler applying p^2 instead of p, or applying per-gate
//! when it should be per-shot).

use pecos_gpu_sims::{CircuitBuilder, DepolarizingNoiseSampler, GpuNoisySampler};

/// For the trivial circuit `mz(0)` on |0> with measurement-error probability
/// `p_meas`, the fraction of shots returning 1 should be close to `p_meas`.
#[test]
#[ignore = "Slow statistical GPU audit; run explicitly with: cargo test -p pecos-gpu-sims --test noisy_sampler_stats --release -- --ignored --test-threads=1"]
#[allow(clippy::cast_precision_loss)] // shots <= 4096, exact in f64
fn measurement_error_rate_matches_p() {
    let shots = 4096usize;
    for &p in &[0.0_f64, 0.05, 0.1, 0.3] {
        let sampler = DepolarizingNoiseSampler::with_seed(0.0, 0.0, p, 0x1234);
        let mut gpu = GpuNoisySampler::new(1, sampler);
        let mut circuit = CircuitBuilder::new();
        circuit.mz(&[0]);

        let results = gpu
            .sample(shots, |b| {
                b.mz(&[0]);
            })
            .expect("sample failed");

        let mut ones = 0usize;
        for shot in &results {
            if shot.outcomes.first().copied().unwrap_or(false) {
                ones += 1;
            }
        }
        let observed = ones as f64 / shots as f64;
        // 3-sigma window for Binomial(shots, p): sigma = sqrt(p(1-p)/shots).
        let sigma = (p * (1.0 - p) / shots as f64).sqrt();
        let delta = (observed - p).abs();
        assert!(
            delta < 4.0 * sigma + 0.01,
            "p_meas={p}: observed={observed:.4} expected~{p} (4 sigma = {:.4})",
            4.0 * sigma + 0.01
        );
    }
}

/// With no noise at all, `mz(0)` on the computational basis state should be
/// 0 on every shot.
#[test]
fn zero_noise_zero_ones() {
    let sampler = DepolarizingNoiseSampler::with_seed(0.0, 0.0, 0.0, 0x5678);
    let mut gpu = GpuNoisySampler::new(1, sampler);
    let results = gpu
        .sample(256, |b| {
            b.mz(&[0]);
        })
        .expect("sample failed");
    for shot in &results {
        assert!(!shot.outcomes[0], "zero noise should give 0 outcome");
    }
}

/// For a single-qubit depolarizing channel with probability p1, applying a
/// `noise_1q` after preparing |0> should flip the measurement with probability
/// ~2p/3 (X and Y flip Z-basis; Z doesn't). Check within 4 sigma.
#[test]
#[ignore = "Slow statistical GPU audit; run explicitly with: cargo test -p pecos-gpu-sims --test noisy_sampler_stats --release -- --ignored --test-threads=1"]
#[allow(clippy::cast_precision_loss)] // shots bounded, exact in f64
fn depol1_flip_rate() {
    let shots = 4096usize;
    for &p in &[0.1_f64, 0.3] {
        let sampler = DepolarizingNoiseSampler::with_seed(p, 0.0, 0.0, 0xbeef);
        let mut gpu = GpuNoisySampler::new(1, sampler);
        let results = gpu
            .sample(shots, |b| {
                b.noise_1q(&[0]);
                b.mz(&[0]);
            })
            .expect("sample failed");
        let ones: usize = results.iter().filter(|r| r.outcomes[0]).count();
        let observed = ones as f64 / shots as f64;
        let expected = 2.0 * p / 3.0;
        let sigma = (expected * (1.0 - expected) / shots as f64).sqrt();
        let delta = (observed - expected).abs();
        assert!(
            delta < 4.0 * sigma + 0.01,
            "p1={p}: flip rate observed={observed:.4} expected {expected:.4}"
        );
    }
}

/// Two-qubit depolarizing `noise_2q` uses p2. For a Bell pair + `noise_2q` +
/// measure both: the *correlation* between the two measurements should drop
/// by a quantity related to p2. This tests that p2 actually plumbs to the
/// 2q noise path (not re-using p1).
#[test]
#[ignore = "Slow statistical GPU audit; run explicitly with: cargo test -p pecos-gpu-sims --test noisy_sampler_stats --release -- --ignored --test-threads=1"]
#[allow(clippy::cast_precision_loss)] // shots bounded, exact in f64
fn depol2_reduces_bell_correlation() {
    let shots = 4096usize;
    let sampler_clean = DepolarizingNoiseSampler::with_seed(0.0, 0.0, 0.0, 0x100);
    let sampler_noisy = DepolarizingNoiseSampler::with_seed(0.0, 0.5, 0.0, 0x100);

    let circuit_fn = |b: &mut CircuitBuilder| {
        b.h(&[0]);
        b.cx(&[(0, 1)]);
        b.noise_2q(&[(0, 1)]);
        b.mz(&[0, 1]);
    };

    let mut gpu_clean = GpuNoisySampler::new(2, sampler_clean);
    let clean = gpu_clean.sample(shots, circuit_fn).expect("clean sample");
    let mut gpu_noisy = GpuNoisySampler::new(2, sampler_noisy);
    let noisy = gpu_noisy.sample(shots, circuit_fn).expect("noisy sample");

    let clean_correlated = clean
        .iter()
        .filter(|r| r.outcomes[0] == r.outcomes[1])
        .count();
    let noisy_correlated = noisy
        .iter()
        .filter(|r| r.outcomes[0] == r.outcomes[1])
        .count();

    // Perfectly clean Bell pair: correlation = 100%. With strong 2q noise,
    // correlation drops notably.
    let clean_rate = clean_correlated as f64 / shots as f64;
    let noisy_rate = noisy_correlated as f64 / shots as f64;
    assert!(
        clean_rate > 0.99,
        "clean Bell correlation {clean_rate} should be ~1"
    );
    assert!(
        noisy_rate < 0.95,
        "noisy (p2=0.5) Bell correlation {noisy_rate} should be visibly below clean"
    );
}

/// Determinism: same seed => same `ShotResult` stream.
#[test]
fn same_seed_same_results() {
    let p1 = 0.1;
    let s1 = DepolarizingNoiseSampler::with_seed(p1, 0.0, 0.0, 42);
    let s2 = DepolarizingNoiseSampler::with_seed(p1, 0.0, 0.0, 42);
    let mut g1 = GpuNoisySampler::with_seed(1, s1, 777);
    let mut g2 = GpuNoisySampler::with_seed(1, s2, 777);
    let r1 = g1
        .sample(128, |b| {
            b.noise_1q(&[0]);
            b.mz(&[0]);
        })
        .expect("g1 sample");
    let r2 = g2
        .sample(128, |b| {
            b.noise_1q(&[0]);
            b.mz(&[0]);
        })
        .expect("g2 sample");
    for (a, b) in r1.iter().zip(r2.iter()) {
        assert_eq!(a.outcomes, b.outcomes, "same seed should match");
    }
}
