// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0

use pecos_simulators::measurement_sampler::{MeasurementSampler, SequentialMeasurementSampler};
use pecos_simulators::symbolic_sparse_stab::SymbolicMeasurementResult;
use pecos_simulators::{SymbolicSparseStab, SymbolicSparseStabVecSet};

// `BitSet` equality depends on storage width, so compare the record indices.
fn records(result: &SymbolicMeasurementResult) -> Vec<usize> {
    result.outcome.iter().collect()
}

macro_rules! reset_cases {
    ($sim:ty) => {{
        let mut sim = <$sim>::new(4);
        sim.h(&[0, 2]).cx(&[(0, 1), (2, 3)]);
        sim.pz(0).pz(2);
        assert_eq!(sim.measurement_count(), 0);
        let results = sim.mz(&[0, 1, 2, 3, 1, 3]);
        for column in [0, 2] {
            assert!(results[column].outcome.is_empty());
            assert!(!results[column].flip);
        }
        assert!(!results[1].is_deterministic);
        assert!(!results[3].is_deterministic);
        assert!(results[4].is_deterministic);
        assert!(results[5].is_deterministic);
        assert_eq!(results[1].outcome.len(), 1);
        assert_eq!(results[3].outcome.len(), 1);
        assert_ne!(records(&results[1]), records(&results[3]));
        assert_eq!(records(&results[1]), records(&results[4]));
        assert_eq!(records(&results[3]), records(&results[5]));
        // Resetting a conditionally known eigenstate must clear its own sign.
        sim.pz(1).pz(3);
        for result in sim.mz(&[1, 3]) {
            assert!(result.outcome.is_empty());
            assert!(!result.flip);
        }
        sim.reset();
        sim.h(&[0]).cx(&[(0, 1), (0, 2)]);
        sim.mz_unrecorded(&[0]);
        assert_eq!(sim.measurement_count(), 0);
        let results = sim.mz(&[0, 1, 2]);
        assert_eq!(results[0].outcome.len(), 1);
        assert_eq!(records(&results[0]), records(&results[1]));
        assert_eq!(records(&results[0]), records(&results[2]));
        assert_eq!(results[0].index, 0);
        assert!(!results[0].is_deterministic);
        assert!(results[1].is_deterministic);
        assert!(results[2].is_deterministic);
    }};
}

#[test]
fn symbolic_bitset_reset_and_discard_preserve_hidden_correlations() {
    reset_cases!(SymbolicSparseStab);
}

#[test]
fn symbolic_vecset_reset_and_discard_preserve_hidden_correlations() {
    reset_cases!(SymbolicSparseStabVecSet);
}

#[test]
fn record_samplers_accept_hidden_sources_and_preserve_correlations() {
    for reset in [false, true] {
        let mut sim = SymbolicSparseStab::new(2);
        sim.h(&[0]).cx(&[(0, 1)]);
        if reset {
            sim.pz(0);
        } else {
            sim.mz_unrecorded(&[0]);
        }
        sim.mz(&[1, 1]);
        let history = sim.measurement_history();
        assert_eq!(history.deterministic().len(), 1);
        assert_eq!(history.nondeterministic().len(), 1);
        for sample in [
            MeasurementSampler::new(history).sample_with_seed(256, 42),
            SequentialMeasurementSampler::new(history).sample_with_seed(256, 42),
        ] {
            let mut seen = [false; 2];
            for shot in 0..sample.shots() {
                let bit = sample.get(shot, 0);
                seen[usize::from(bit.0)] = true;
                assert_eq!(bit, sample.get(shot, 1));
            }
            assert_eq!(seen, [true, true]);
        }
    }
}
