// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0.

use super::*;
use crate::distillation::TriorthogonalMatrix;

fn matrices() -> [TriorthogonalMatrix; 3] {
    [
        TriorthogonalMatrix::rm15(),
        TriorthogonalMatrix::bravyi_haah(4).unwrap(),
        TriorthogonalMatrix::bravyi_haah(10).unwrap(),
    ]
}

#[test]
fn distillation_deterministic_patterns() {
    for matrix in matrices() {
        let n = matrix.n();
        let base = matrix.tick_circuit(0.0).unwrap();
        let locations: Vec<_> = base
            .iter_gate_batches_with_tick()
            .filter(|(_, gate)| gate.gate_type == GateType::Channel)
            .map(|(tick, gate)| (tick, gate.batch_index(), gate.qubits[0]))
            .collect();
        assert_eq!(locations.len(), n);
        let mut patterns = vec![vec![false; n]];
        for a in 0..n {
            let mut pattern = vec![false; n];
            pattern[a] = true;
            patterns.push(pattern);
            for b in a + 1..n {
                let mut pattern = vec![false; n];
                pattern[a] = true;
                pattern[b] = true;
                patterns.push(pattern);
            }
        }
        let mut rng = PecosRng::seed_from_u64(71);
        for _ in 0..32 {
            patterns.push((0..n).map(|_| rng.next_bool_fast()).collect());
        }
        let mut sim = PhasePoly::with_seed(n, 72);
        for pattern in &patterns {
            let mut circuit = base.clone();
            for &(tick, batch, q) in &locations {
                circuit
                    .get_tick_mut(tick)
                    .unwrap()
                    .replace_gate_batch(
                        batch,
                        Gate::channel(channel::Dephasing(f64::from(u8::from(pattern[q.0])), q)),
                    )
                    .unwrap();
            }
            let program = compile(&circuit).unwrap();
            assert_eq!(program.num_records(), matrix.m() - matrix.k());
            assert_eq!(program.observables(), Vec::<Vec<usize>>::new());
            let shot = program.run_shot(&mut sim, &mut rng).unwrap();
            let expected = matrix.pattern_oracle(pattern).unwrap();
            assert_eq!(shot.records, expected.syndrome);
            assert_eq!(shot.detectors, expected.syndrome);
            assert_eq!(shot.detectors.iter().all(|b| !b), expected.accepted);
        }
        println!(
            "TickCircuit patterns k={}: {} (all weight <= 2 plus 32 random)",
            matrix.k(),
            patterns.len()
        );
    }
}

#[test]
fn distillation_acceptance_matches_binomial_bound() {
    // Two-sided Bernstein bound for Bernoulli means, delta=1e-8 per code.
    // A union bound over all three comparisons gives failure <=3e-8 <1e-6.
    let log = (2.0_f64 / 1e-8).ln();
    for (matrix, p, seed) in matrices()
        .into_iter()
        .zip([0.05, 0.05, 0.02])
        .zip([2027, 2026, 2028])
        .map(|((m, p), s)| (m, p, s))
    {
        let expected = matrix.weight_enumerators().unwrap().oracles(p).unwrap().p_s;
        let program = compile(&matrix.tick_circuit(p).unwrap()).unwrap();
        let mut sim = PhasePoly::with_seed(matrix.n(), seed);
        let mut rng = PecosRng::seed_from_u64(seed);
        let shots = 20_000_u32;
        let mut accepted = 0_u32;
        for _ in 0..shots {
            let shot = program.run_shot(&mut sim, &mut rng).unwrap();
            accepted += u32::from(shot.detectors.iter().all(|b| !b));
        }
        let rate = f64::from(accepted) / f64::from(shots);
        let bound = (2.0 * expected * (1.0 - expected) * log / f64::from(shots)).sqrt()
            + 2.0 * log / (3.0 * f64::from(shots));
        assert!((rate - expected).abs() <= bound);
        println!(
            "TickCircuit acceptance k={} p={p}: {accepted}/{shots} = {rate:.8}, P_s={expected:.10}, bound={bound:.10}; joint failure <=3e-8",
            matrix.k()
        );
        let ideal = compile(&matrix.tick_circuit(0.0).unwrap()).unwrap();
        for _ in 0..100 {
            assert!(
                ideal
                    .run_shot(&mut sim, &mut rng)
                    .unwrap()
                    .detectors
                    .iter()
                    .all(|b| !b)
            );
        }
    }
}

#[test]
fn distillation_boundary_matrices_and_probabilities() {
    for matrix in [
        TriorthogonalMatrix::new(&[], 0).unwrap(),
        TriorthogonalMatrix::new(&[vec![true]], 1).unwrap(),
        TriorthogonalMatrix::new(&[vec![true, true]], 0).unwrap(),
    ] {
        let circuit = matrix.tick_circuit(0.0).unwrap();
        let program = compile(&circuit).unwrap();
        assert_eq!(program.num_qubits(), matrix.n());
        for shot in program.run_shots(10, 0).unwrap() {
            assert!(shot.detectors.iter().all(|b| !b));
        }
        for p in [-0.01, 1.01, f64::NAN, f64::INFINITY] {
            assert!(matrix.tick_circuit(p).is_err());
        }
    }
}
