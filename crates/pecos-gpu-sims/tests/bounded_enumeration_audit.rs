// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
// except in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0

use pecos_gpu_sims::{GpuBoundedEnumerationBackend, gpu_bounded_enumeration_code_distance};
use pecos_qec::{
    BoundedEnumerationDistance, CpuLevelEnumerationBackend, LevelEnumerationBackend,
    LevelEnumerationInput, PackedSystematicGenerator, ParityCheckMatrix,
    bounded_enumeration_code_distance, bounded_enumeration_code_distance_with_backend,
};
use pecos_quantum::F2Matrix;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

fn checks_for_generator(rows: Vec<Vec<u8>>) -> ParityCheckMatrix {
    let generator = F2Matrix::from_rows(rows);
    ParityCheckMatrix::from_dense(generator.kernel()).unwrap()
}

#[test]
fn seeded_small_pairs_are_bit_identical() {
    let Ok(mut gpu) = GpuBoundedEnumerationBackend::try_new() else {
        return;
    };
    let mut rng = StdRng::seed_from_u64(0xB0A0_DED0_5EED_0027);

    for case in 0..64 {
        let width = rng.random_range(1..=10);
        let h = ParityCheckMatrix::from_dense(
            (0..rng.random_range(1..=width))
                .map(|_| (0..width).map(|_| u8::from(rng.random_bool(0.5))).collect())
                .collect(),
        )
        .unwrap();
        let l = ParityCheckMatrix::from_dense(
            (0..rng.random_range(1..=3))
                .map(|_| (0..width).map(|_| u8::from(rng.random_bool(0.5))).collect())
                .collect(),
        )
        .unwrap();
        let max_level = rng.random_range(0..=width);
        let cpu = bounded_enumeration_code_distance(&h, &l, max_level);
        let accelerated =
            bounded_enumeration_code_distance_with_backend(&h, &l, max_level, &mut gpu).unwrap();

        assert_eq!(accelerated, cpu, "CPU/GPU mismatch in seeded case {case}");
    }
}

#[test]
fn kernel_handles_first_and_last_level_boundaries() {
    let Ok(mut gpu) = GpuBoundedEnumerationBackend::try_new() else {
        return;
    };
    let systematic_generators = vec![PackedSystematicGenerator {
        rows: vec![0b001, 0b010, 0b100],
    }];
    let active = vec![0];
    let logical_rows = vec![0b111];
    let make_input = |level| LevelEnumerationInput {
        level,
        dimension: 3,
        codeword_bits: 3,
        row_stride_words: 1,
        systematic_generators: &systematic_generators,
        active_systematic_indices: &active,
        logical_rows: &logical_rows,
        logical_count: 1,
    };
    let mut cpu = CpuLevelEnumerationBackend;

    for (level, expected_weight) in [(1, 1), (3, 3)] {
        let cpu_minimum = cpu.enumerate_level(make_input(level)).unwrap();
        let gpu_minimum = gpu.enumerate_level(make_input(level)).unwrap();
        assert_eq!(cpu_minimum.weight, Some(expected_weight));
        assert_eq!(gpu_minimum.weight, cpu_minimum.weight);
        assert!(gpu_minimum.witness.is_none());
    }
}

#[test]
fn hand_analyzed_six_two_four_terminates_identically() {
    let Ok(_) = GpuBoundedEnumerationBackend::try_new() else {
        return;
    };
    let h = checks_for_generator(vec![vec![1, 1, 1, 1, 0, 0], vec![0, 0, 1, 1, 1, 1]]);
    let l = ParityCheckMatrix::from_dense(vec![vec![1, 0, 0, 0, 0, 0]]).unwrap();
    let cpu = bounded_enumeration_code_distance(&h, &l, 2).unwrap();
    let gpu = gpu_bounded_enumeration_code_distance(&h, &l, 2)
        .unwrap()
        .unwrap();

    assert_eq!(gpu, cpu);
    assert!(matches!(
        gpu,
        BoundedEnumerationDistance::CertifiedByBounds {
            distance: 4,
            level: 1,
            ..
        }
    ));
}
