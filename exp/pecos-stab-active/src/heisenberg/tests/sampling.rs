// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0.
use super::*;

// Pearson chi-square tests at alpha=0.001: critical values 16.266 (3 df)
// and 37.697 (15 df). Every expected bin count is at least 100.
#[test]
fn channel_sampling_preserves_joint_probabilities() {
    for expression in [
        channel::Depolarizing(0.3, 0),
        channel::Depolarizing2(0.6, 0, 1),
        channel::PauliChannel(0.1, 0.2, 0.3, 0),
    ] {
        let mut circuit = TickCircuit::new();
        circuit.tick().channel(expression);
        let program = HeisenbergProgram::compile(&circuit).unwrap();
        let channel = &program.noise_channels[0];
        let mut rng = PecosRng::seed_from_u64(8801);
        let mut bits = vec![false; program.num_noise_symbols];
        let mut counts = vec![0u32; channel.alternatives.len()];
        for _ in 0..20_000 {
            channel.sample(&mut rng, &mut bits);
            let index = channel
                .alternatives
                .iter()
                .position(|(_, b)| b == &bits)
                .unwrap();
            counts[index] += 1;
        }
        let statistic: f64 = counts
            .iter()
            .zip(&channel.alternatives)
            .map(|(&c, (p, _))| {
                let expected = 20_000.0 * p;
                assert!(expected >= 100.0);
                (f64::from(c) - expected).powi(2) / expected
            })
            .sum();
        let threshold = if counts.len() == 4 { 16.266 } else { 37.697 };
        assert!(statistic < threshold, "chi-square {statistic}");
    }
}

#[test]
fn independent_classical_quantum_sampling_and_seed_replay() {
    let mut circuit = TickCircuit::new();
    circuit.tick().channel(channel::BitFlip(0.5, 0));
    circuit.tick().h(&[1]);
    circuit.tick().mz(&[0, 1]);
    let program = HeisenbergProgram::compile(&circuit).unwrap();
    let mut counts = [0u32; 4];
    for seed in 0..12_000 {
        let shot = program.run(seed);
        if seed < 100 {
            assert_eq!(shot, program.run(seed));
        }
        counts[usize::from(shot.records[0]) + 2 * usize::from(shot.records[1])] += 1;
    }
    // Four equiprobable bins, 3 df, alpha=0.001.
    let statistic: f64 = counts
        .iter()
        .map(|&c| (f64::from(c) - 3000.0).powi(2) / 3000.0)
        .sum();
    assert!(
        statistic < 16.266,
        "chi-square {statistic}, bins {counts:?}"
    );
}

fn memory(p: f64) -> TickCircuit {
    let mut circuit = TickCircuit::new();
    circuit.tick().pz(&[0, 1, 2, 3, 4]);
    // Stable IDs deliberately have a gap, so they cannot be record ordinals.
    circuit.advance_meas_counter(17);
    for q in 0..3 {
        circuit.tick().channel(channel::BitFlip(p, q));
    }
    circuit.tick().cx(&[(0, 3), (2, 4)]);
    circuit.tick().cx(&[(1, 3)]);
    circuit.tick().cx(&[(1, 4)]);
    let checks = circuit.tick().mz(&[3, 4]);
    let data = circuit.tick().mz(&[0, 1, 2]);
    circuit.detector(&[checks[0]]).unwrap();
    circuit.detector(&[checks[1]]).unwrap();
    circuit.detector(&[checks[0], data[0], data[1]]).unwrap();
    circuit.observable(&[data[0], data[1], data[2]]).unwrap();
    circuit
}

#[test]
fn repetition_memory_annotations_analytic_and_physical() {
    let noiseless = HeisenbergProgram::compile(&memory(0.0)).unwrap();
    assert_eq!(noiseless.detectors(), &[vec![0], vec![1], vec![0, 2, 3]]);
    assert_eq!(noiseless.observables(), &[vec![2, 3, 4]]);
    for seed in 0..20 {
        let shot = noiseless.run(seed);
        assert_eq!(shot.detectors, [false; 3]);
        assert_eq!(shot.observables, [false]);
    }
    let p = 0.13;
    let program = HeisenbergProgram::compile(&memory(p)).unwrap();
    let mut counts = [0u32; 3];
    let mut physical_counts = [0u32; 3];
    for seed in 0..10_000 {
        let shot = program.run(seed);
        assert!(!shot.detectors[2]);
        let mut physical = StabActive::with_seed(5, seed);
        let mut noise = vec![false; program.num_noise_symbols];
        for channel in &program.noise_channels {
            channel.sample(&mut physical.rng, &mut noise);
        }
        for q in 0..3 {
            if noise[2 * q] {
                physical.x(&[QubitId(q)]);
            }
        }
        physical.cx(&[(QubitId(0), QubitId(3)), (QubitId(2), QubitId(4))]);
        physical.cx(&[(QubitId(1), QubitId(3)), (QubitId(1), QubitId(4))]);
        let checks = physical.mz(&[QubitId(3), QubitId(4)]);
        let data = physical.mz(&[QubitId(0), QubitId(1), QubitId(2)]);
        let expected = [
            checks[0].outcome,
            checks[1].outcome,
            data.iter().fold(false, |s, r| s ^ r.outcome),
        ];
        let observed = [shot.detectors[0], shot.detectors[1], shot.observables[0]];
        assert_eq!(observed, expected);
        for i in 0..3 {
            counts[i] += u32::from(observed[i]);
            physical_counts[i] += u32::from(expected[i]);
        }
    }
    // Each check is the XOR of two iid Bernoulli(p) faults: 2p(1-p).
    // The observable is the XOR of three: (1-(1-2p)^3)/2.
    let rates = [
        2.0 * p * (1.0 - p),
        2.0 * p * (1.0 - p),
        (1.0 - (1.0 - 2.0 * p).powi(3)) / 2.0,
    ];
    for (i, rate) in rates.into_iter().enumerate() {
        let mean = 10_000.0 * rate;
        assert!(mean > 100.0 && counts[i] > 0);
        let sigma = (10_000.0 * rate * (1.0 - rate)).sqrt();
        assert!((f64::from(counts[i]) - mean).abs() < 4.0 * sigma);
        assert!(
            (f64::from(counts[i]) - f64::from(physical_counts[i])).abs()
                < 4.0 * (2.0_f64).sqrt() * sigma
        );
    }
}
