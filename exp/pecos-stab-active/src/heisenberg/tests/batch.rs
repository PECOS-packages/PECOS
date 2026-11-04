// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0.

use super::*;
use crate::heisenberg::plan::{
    Instruction,
    batch::{BatchSampler, chunk_shots},
};
use num_complex::Complex64;

#[derive(Debug, Default)]
struct Coverage {
    rotation_splits: usize,
    mixed_negative: usize,
    projection_splits: usize,
    shared_value: usize,
    unprojected_outcomes: usize,
}

impl Coverage {
    fn assert_complete(&self) {
        assert!(self.rotation_splits > 0, "{self:?}");
        assert!(self.mixed_negative > 0, "{self:?}");
        assert!(self.projection_splits > 0, "{self:?}");
        assert!(self.shared_value > 0, "{self:?}");
        assert!(self.unprojected_outcomes > 0, "{self:?}");
    }
}

fn observed(plan: &SamplingPlan, seeds: &[u64], coverage: &mut Coverage) -> Vec<ShotResult> {
    let mut previous = vec![(0..seeds.len()).collect::<Vec<_>>()];
    BatchSampler::new(plan, seeds).run_observed(|operation, sampler| {
        let op = &plan.operations[operation];
        let mut splits = 0;
        for ids in &previous {
            let mut negatives = [false; 2];
            let mut outcomes = [false; 2];
            let mut values = [false; 2];
            let mut outcomes_by_value = [[false; 2]; 2];
            for &id in ids {
                let shot = &sampler.shots[id];
                negatives[usize::from(op.sign.evaluate(&shot.noise, &shot.measurements))] = true;
                if let Instruction::Measurement {
                    symbol, projection, ..
                } = &op.instruction
                {
                    let outcome = usize::from(shot.measurements[*symbol]);
                    outcomes[outcome] = true;
                    if let Some(projection) = projection {
                        let value =
                            usize::from(projection.value.evaluate(&shot.noise, &shot.measurements));
                        values[value] = true;
                        outcomes_by_value[value][outcome] = true;
                    }
                }
            }
            match &op.instruction {
                Instruction::Rotation { .. } => {
                    let split = usize::from(negatives == [true; 2]);
                    coverage.rotation_splits += split;
                    splits += split;
                }
                Instruction::Measurement { projection, .. } => {
                    coverage.mixed_negative += usize::from(negatives == [true; 2]);
                    if projection.is_some() {
                        let split = usize::from(values == [true; 2]);
                        coverage.projection_splits += split;
                        splits += split;
                        coverage.shared_value +=
                            usize::from(outcomes_by_value.contains(&[true; 2]));
                    } else {
                        coverage.unprojected_outcomes += usize::from(outcomes == [true; 2]);
                    }
                }
                Instruction::Clifford => {}
            }
        }
        assert_eq!(
            sampler.lanes.len(),
            previous.len() + splits,
            "op={operation}"
        );
        let mut seen = vec![false; seeds.len()];
        previous = sampler
            .lanes
            .iter()
            .map(|lane| {
                assert_eq!(lane.amplitudes.capacity(), 1 << plan.peak);
                let ids = &sampler.indices[lane.shots.clone()];
                assert_ne!(ids, [] as [usize; 0]);
                assert!(ids.windows(2).all(|w| w[0] < w[1]));
                for &id in ids {
                    assert!(!seen[id]);
                    seen[id] = true;
                }
                ids.to_vec()
            })
            .collect();
        assert!(seen.into_iter().all(|present| present));
    })
}

fn compare(program: &HeisenbergProgram, seeds: &[u64], coverage: &mut Coverage) {
    for plan in [program.plan(26), plan_scheduled(program, 26)] {
        // These small fixtures must fit limit 26; do not silently skip a
        // planning regression that would otherwise shrink the batch corpus.
        let plan = plan.unwrap();
        let expected: Vec<_> = seeds.iter().map(|&seed| plan.run(seed)).collect();
        let lane = size_of::<Complex64>() << plan.peak;
        for budget in [0, lane - 1, lane, 3 * lane, usize::MAX] {
            assert_eq!(plan.run_batch(seeds, budget), expected, "budget={budget}");
        }
        assert_eq!(observed(&plan, seeds, coverage), expected);
    }
}

#[test]
fn fixture_bit_identity() {
    let seeds: Vec<_> = (0..64).collect();
    let mut coverage = Coverage::default();
    for program in plan::fixtures() {
        compare(&program, &seeds, &mut coverage);
    }
}

#[test]
fn annotations_and_repeated_seeds() {
    let mut circuit = TickCircuit::new();
    circuit.tick().h(&[0]);
    circuit.tick().rx(pecos_core::radians!(0.37), &[1]);
    let first = circuit.tick().mz(&[0, 1]);
    circuit.tick().pz(&[0]);
    circuit.tick().channel(channel::BitFlip(0.5, 0));
    let last = circuit.tick().mz(&[0, 1]);
    circuit.detector(&[first[0], first[1], last[0]]).unwrap();
    circuit.observable(&[last[1], first[0]]).unwrap();
    let program = HeisenbergProgram::compile(&circuit).unwrap();
    let seeds: Vec<_> = [7, 7, 0, u64::MAX].into_iter().chain(0..60).collect();
    compare(&program, &seeds, &mut Coverage::default());
    for plan in [
        program.plan(26).unwrap(),
        plan_scheduled(&program, 26).unwrap(),
    ] {
        let budget = 3 * (size_of::<Complex64>() << plan.peak);
        let shots = plan.run_batch(&seeds, budget);
        assert_eq!(shots, plan.par_run_batch(&seeds, budget));
        assert_eq!(shots[0], shots[1]);
        assert!(shots.iter().any(|shot| shot.detectors[0]));
        assert!(shots.iter().any(|shot| !shot.detectors[0]));
        assert!(shots.iter().any(|shot| shot.observables[0]));
        assert!(shots.iter().any(|shot| !shot.observables[0]));
    }
}

#[test]
fn random_bit_identity_and_coverage() {
    let mut rng = PecosRng::seed_from_u64(0xb47c);
    let seeds: Vec<_> = (0..64).map(|_| rng.next_u64()).collect();
    let mut coverage = Coverage::default();
    for generator in [passes::random_circuit, reordering::random_circuit] {
        for _ in 0..120 {
            let program = HeisenbergProgram::compile(&generator(&mut rng)).unwrap();
            compare(&program, &seeds, &mut coverage);
        }
    }
    println!("batch coverage: {coverage:?}");
    coverage.assert_complete();
}

#[test]
fn random_measurement_keeps_rotation_lane_shared() {
    let mut circuit = TickCircuit::new();
    circuit.tick().h(&[0, 1, 2, 3]);
    circuit.tick().h(&[4]);
    circuit.tick().mz(&[4]);
    circuit.tick().t(&[0, 1, 2]);
    // Overlapping ladder edges are sequential gate commands, not one batch.
    circuit.tick().cx(&[(0, 1)]);
    circuit.tick().cx(&[(1, 2)]);
    circuit.tick().cx(&[(2, 3)]);
    circuit.tick().mz(&[0, 1, 2, 3]);
    let plan = HeisenbergProgram::compile(&circuit)
        .unwrap()
        .plan(26)
        .unwrap();
    let seeds: Vec<_> = (0..256).collect();
    assert!(chunk_shots(plan.peak, usize::MAX) >= seeds.len());
    let mut rotations = 0;
    let actual = BatchSampler::new(&plan, &seeds).run_observed(|operation, sampler| {
        if matches!(
            plan.operations[operation].instruction,
            Instruction::Rotation { .. }
        ) {
            rotations += 1;
            assert_eq!(sampler.lanes.len(), 1);
        }
    });
    assert_eq!(rotations, 3);
    assert!(actual.iter().any(|shot| shot.records[0]));
    assert!(actual.iter().any(|shot| !shot.records[0]));
    assert_eq!(actual, plan.run_batch(&seeds, usize::MAX));
    assert_eq!(
        actual,
        seeds.iter().map(|&seed| plan.run(seed)).collect::<Vec<_>>()
    );
}

#[test]
fn active_measurement_shares_value_across_opposite_outcomes() {
    // Undo the rotation without fusion: the coordinate remains Active but its
    // projection is certain. A later X fault changes the raw outcome only.
    let mut circuit = TickCircuit::new();
    let angle = pecos_core::radians!(0.37);
    circuit.tick().ry(angle, &[0]);
    circuit.tick().ry(-angle, &[0]);
    circuit.tick().channel(channel::BitFlip(0.5, 0));
    circuit.tick().mz(&[0]);
    circuit.tick().rx(angle, &[1]);
    circuit.tick().mz(&[1]);
    let plan = HeisenbergProgram::compile(&circuit)
        .unwrap()
        .plan(26)
        .unwrap();
    let seeds: Vec<_> = (0..256).collect();
    let mut witnessed = false;
    let actual = BatchSampler::new(&plan, &seeds).run_observed(|operation, sampler| {
        if let Instruction::Measurement {
            symbol: 0,
            projection: Some(projection),
            ..
        } = &plan.operations[operation].instruction
        {
            let mut negatives = [false; 2];
            let mut outcomes = [false; 2];
            let mut values = [false; 2];
            for shot in &sampler.shots {
                negatives[usize::from(
                    plan.operations[operation]
                        .sign
                        .evaluate(&shot.noise, &shot.measurements),
                )] = true;
                outcomes[usize::from(shot.measurements[0])] = true;
                values[usize::from(projection.value.evaluate(&shot.noise, &shot.measurements))] =
                    true;
            }
            assert_eq!(negatives, [true; 2]);
            assert_eq!(outcomes, [true; 2]);
            assert_ne!(values, [true; 2]);
            assert_eq!(sampler.lanes.len(), 1);
            witnessed = true;
        }
    });
    assert!(witnessed);
    assert_eq!(actual, plan.run_batch(&seeds, usize::MAX));
    assert_eq!(
        actual,
        seeds.iter().map(|&seed| plan.run(seed)).collect::<Vec<_>>()
    );
}

#[test]
fn parallel_random_families() {
    // An explicit pool, so chunks run concurrently even on a one-core runner.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let mut rng = PecosRng::seed_from_u64(0x9a47);
    let seeds: Vec<_> = (0..1000).map(|_| rng.next_u64()).collect();
    for generator in [passes::random_circuit, reordering::random_circuit] {
        for _ in 0..120 {
            let program = HeisenbergProgram::compile(&generator(&mut rng)).unwrap();
            for plan in [
                program.plan(26).unwrap(),
                plan_scheduled(&program, 26).unwrap(),
            ] {
                let budget = 3 * (size_of::<Complex64>() << plan.peak);
                assert!(chunk_shots(plan.peak, budget) < seeds.len());
                let parallel = pool.install(|| plan.par_run_batch(&seeds, budget));
                assert_eq!(parallel, plan.run_batch(&seeds, budget));
                for (shot, &seed) in parallel.iter().zip(&seeds).step_by(97) {
                    assert_eq!(*shot, plan.run(seed));
                }
            }
        }
    }
}

#[test]
fn chunk_arithmetic_and_empty_input() {
    assert_eq!(chunk_shots(26, usize::MAX), usize::MAX >> 30);
    assert_eq!(chunk_shots(0, 16), 1);
    assert_eq!(chunk_shots(0, 15), 1);
    assert_eq!(chunk_shots(0, 0), 1);
    assert_eq!(chunk_shots(3, 3 * (16 << 3)), 3);
    let plan = HeisenbergProgram::compile(&TickCircuit::new())
        .unwrap()
        .plan(26)
        .unwrap();
    for budget in [0, 48, usize::MAX] {
        assert_eq!(plan.run_batch(&[], budget), []);
        assert_eq!(plan.par_run_batch(&[], budget), []);
        let seeds = [7, 7, 0, u64::MAX];
        let expected: Vec<_> = seeds.iter().map(|&seed| plan.run(seed)).collect();
        assert_eq!(plan.run_batch(&seeds, budget), expected);
        assert_eq!(plan.par_run_batch(&seeds, budget), expected);
    }
}
