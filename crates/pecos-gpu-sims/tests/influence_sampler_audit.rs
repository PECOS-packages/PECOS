// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
// except in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0

//! Targeted audit of `GpuInfluenceSampler`.
//!
//! Semantics: for each shot, each location has probability `p_error` of a
//! fault. If a fault fires, a random Pauli (X/Y/Z, uniformly) is applied.
//! Each fault toggles a CSR-encoded set of detectors and logicals.
//!
//! We don't have a CPU reference implementation to cross-check against.
//! Instead, we use tight edge-case tests + distributional sanity checks.

use pecos_gpu_sims::{GpuInfluenceMapData, GpuInfluenceSampler};

/// Build an influence map with `n_loc` locations, `n_det` detectors, and
/// `n_log` logicals, where:
///   - X fault at location `k` toggles detector `k % n_det`
///   - Z fault at location `k` toggles logical `k % n_log`
///   - Y fault at location `k` toggles both
///
/// Written as three separate CSR tables (X, Y, Z each have a row per location).
#[allow(clippy::cast_possible_truncation)] // CSR row offsets, n_loc bounded by test inputs (<= u32::MAX trivially)
fn simple_diagonal_map(n_loc: u32, n_det: u32, n_log: u32) -> GpuInfluenceMapData {
    let mut det_off_x = vec![0u32; (n_loc + 1) as usize];
    let mut det_dat_x = Vec::<u32>::new();
    let mut det_off_y = vec![0u32; (n_loc + 1) as usize];
    let mut det_dat_y = Vec::<u32>::new();
    let mut det_off_z = vec![0u32; (n_loc + 1) as usize];
    let det_dat_z = Vec::<u32>::new();
    let mut log_off_x = vec![0u32; (n_loc + 1) as usize];
    let log_dat_x = Vec::<u32>::new();
    let mut log_off_y = vec![0u32; (n_loc + 1) as usize];
    let mut log_dat_y = Vec::<u32>::new();
    let mut log_off_z = vec![0u32; (n_loc + 1) as usize];
    let mut log_dat_z = Vec::<u32>::new();

    for k in 0..n_loc {
        // X at k -> detector (k % n_det)
        det_dat_x.push(k % n_det);
        det_off_x[(k + 1) as usize] = det_dat_x.len() as u32;

        // Z at k -> logical (k % n_log)
        log_dat_z.push(k % n_log);
        log_off_z[(k + 1) as usize] = log_dat_z.len() as u32;

        // Y at k -> both
        det_dat_y.push(k % n_det);
        det_off_y[(k + 1) as usize] = det_dat_y.len() as u32;
        log_dat_y.push(k % n_log);
        log_off_y[(k + 1) as usize] = log_dat_y.len() as u32;

        // X touches no logicals (empty row)
        log_off_x[(k + 1) as usize] = log_dat_x.len() as u32;
        // Z touches no detectors (empty row)
        det_off_z[(k + 1) as usize] = det_dat_z.len() as u32;
    }

    GpuInfluenceMapData::from_csr(
        n_loc, n_det, n_log, det_off_x, det_dat_x, det_off_y, det_dat_y, det_off_z, det_dat_z,
        log_off_x, log_dat_x, log_off_y, log_dat_y, log_off_z, log_dat_z,
    )
}

fn no_flips(flips: &[u32]) -> bool {
    flips.iter().all(|&w| w == 0)
}

#[test]
fn zero_prob_no_flips() {
    let map = simple_diagonal_map(32, 8, 4);
    let Ok(mut sampler) = GpuInfluenceSampler::new(&map, 42) else {
        return;
    };
    let result = sampler.sample_uniform(200, 0.0);

    assert_eq!(result.count_logical_errors(), 0);
    for shot in 0..200 {
        assert!(
            !result.has_logical_error(shot),
            "p=0 shot {shot} reports logical error"
        );
        let flips = result.detector_flips_for_shot(shot);
        assert!(
            no_flips(&flips),
            "p=0 shot {shot} has detector flips: {flips:?}"
        );
    }
}

#[test]
fn empty_map_no_flips() {
    // Empty influence map: even at p=1, nothing toggles.
    let map = GpuInfluenceMapData::empty();
    let Ok(mut sampler) = GpuInfluenceSampler::new(&map, 42) else {
        return;
    };
    let result = sampler.sample_uniform(64, 1.0);
    assert_eq!(result.count_logical_errors(), 0);
    for shot in 0..64 {
        assert!(!result.has_logical_error(shot));
        assert!(no_flips(&result.detector_flips_for_shot(shot)));
    }
}

#[test]
fn full_prob_saturates_parity() {
    // At p=1 every location fires every shot. For a map where every
    // location touches at most one detector and one logical, every shot is
    // an independent draw of X/Y/Z per location. The parity of the total
    // toggle count per detector is a deterministic function of the
    // per-location Pauli choices, but statistically the number of shots
    // that flip detector 0 should be non-zero.
    let map = simple_diagonal_map(16, 1, 1); // all locations -> detector 0, logical 0
    let Ok(mut sampler) = GpuInfluenceSampler::new(&map, 7) else {
        return;
    };
    let result = sampler.sample_uniform(256, 1.0);

    let mut any_detector_flip = 0usize;
    let mut any_logical_error = 0usize;
    for shot in 0..256 {
        if !no_flips(&result.detector_flips_for_shot(shot)) {
            any_detector_flip += 1;
        }
        if result.has_logical_error(shot) {
            any_logical_error += 1;
        }
    }
    // At p=1 with 16 locations each randomly {X, Y, Z} mapped to det 0 via
    // X and Y: the parity of detector 0 over 16 flips is ~50/50.
    // Not all 256 shots flip or all stay; expect a healthy mix.
    assert!(
        any_detector_flip > 16,
        "too few detector flips: {any_detector_flip}/256"
    );
    assert!(
        any_detector_flip < 240,
        "too many detector flips: {any_detector_flip}/256"
    );
    assert!(
        any_logical_error > 16,
        "too few logical errors: {any_logical_error}/256"
    );
    assert!(
        any_logical_error < 240,
        "too many logical errors: {any_logical_error}/256"
    );
}

#[test]
fn determinism_with_same_seed() {
    // Two samplers with the same seed should produce identical results.
    let map = simple_diagonal_map(32, 8, 4);
    let Ok(mut a) = GpuInfluenceSampler::new(&map, 99) else {
        return;
    };
    let Ok(mut b) = GpuInfluenceSampler::new(&map, 99) else {
        return;
    };
    let ra = a.sample_uniform(64, 0.1);
    let rb = b.sample_uniform(64, 0.1);
    assert_eq!(ra.count_logical_errors(), rb.count_logical_errors());
    for shot in 0..64 {
        assert_eq!(
            ra.has_logical_error(shot),
            rb.has_logical_error(shot),
            "shot {shot} logical mismatch"
        );
        assert_eq!(
            ra.detector_flips_for_shot(shot),
            rb.detector_flips_for_shot(shot),
            "shot {shot} detector mismatch"
        );
    }
}

#[test]
fn scaling_with_p_error() {
    // Logical error rate should monotonically increase with p.
    let map = simple_diagonal_map(32, 8, 4);
    let Ok(mut sampler) = GpuInfluenceSampler::new(&map, 42) else {
        return;
    };

    let r_low = sampler.sample_uniform(512, 0.01);
    let r_mid = sampler.sample_uniform(512, 0.1);
    let r_high = sampler.sample_uniform(512, 0.5);

    let cnt_low = r_low.count_logical_errors();
    let cnt_mid = r_mid.count_logical_errors();
    let cnt_high = r_high.count_logical_errors();

    assert!(
        cnt_low < cnt_mid,
        "low p={cnt_low} should have fewer errors than mid p={cnt_mid}"
    );
    assert!(
        cnt_mid < cnt_high,
        "mid p={cnt_mid} should have fewer errors than high p={cnt_high}"
    );
}
