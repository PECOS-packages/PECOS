// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0.

use pecos_frontier::windowed::{WindowConfig, WindowPlan, WindowedDecoder, ZeroSyndromePolicy};
use pecos_frontier::{
    FrontierCommittee, FrontierConfig, MetricMode, SparseDem, deadline_column_order,
};
use std::collections::BTreeMap;
use std::fmt::Write;

fn settings() -> FrontierConfig {
    FrontierConfig {
        k: 512,
        delta: 1e9,
        score_alpha: 0.8,
        column_order: None,
        merge_indistinguishable: false,
        bp_score_iterations: 0,
        metric_mode: MetricMode::LogSumExpFloat,
        int_metric_scale: 1024,
    }
}
fn window(blocks: Vec<Vec<u32>>, width: usize, stride: usize) -> WindowConfig {
    WindowConfig {
        detector_blocks: blocks,
        window_blocks: width,
        stride_blocks: stride,
        zero_policy: ZeroSyndromePolicy::SkipAndCommitZero,
        detector_free_error_owners: BTreeMap::new(),
    }
}

#[test]
fn hyperedges_permuted_empty_blocks_future_effects_and_final_tail() {
    let dem =
        SparseDem::from_dem_str("error(0.1) D3 D2 D1 L0 L2\nerror(0.2) D0 L1\nerror(0.05) D1 L2\n")
            .unwrap();
    let config = window(
        vec![vec![3], vec![], vec![2], vec![0], vec![], vec![1]],
        4,
        2,
    );
    let mut decoder = WindowedDecoder::from_sparse_dem(&dem, config, settings(), 32).unwrap();
    let plan = decoder.plan();
    assert_eq!(plan.windows().len(), 2);
    assert_eq!(plan.windows()[0].detector_ids, [3, 2, 0]);
    assert_eq!(plan.windows()[0].future_detector_ids, [1, 2]);
    assert_eq!(plan.windows()[0].mechanism_indices, [0, 1]);
    assert_eq!(plan.windows()[1].commit_stop, 6);
    let first = &plan.models()[plan.windows()[0].model_id];
    assert_eq!(first.mechanisms[0], (0.1, vec![0, 1], vec![0, 2, 3, 4]));
    assert_eq!(first.mechanisms[1], (0.2, vec![2], vec![]));
    let result = decoder.decode(&[1, 0, 1, 1]).unwrap();
    assert_eq!(result.predicted.to_u64(), Some(3));
    assert_eq!(result.windows[0].selected_joint_class.to_u64(), Some(29));
    assert_eq!(result.windows[1].observed, [0, 1, 1]);
    assert_eq!(result.windows[1].selected_joint_class.to_u64(), Some(6));
    assert_eq!(result, decoder.decode(&[1, 0, 1, 1]).unwrap());
}

#[test]
fn final_window_commits_entire_tail_and_long_edge_updates_residual() {
    let dem = SparseDem::from_dem_str("error(0.1) D0 D8 L0\n").unwrap();
    let mut decoder = WindowedDecoder::from_sparse_dem(
        &dem,
        window((0..9).map(|d| vec![d]).collect(), 8, 2),
        settings(),
        32,
    )
    .unwrap();
    assert_eq!(
        decoder
            .plan()
            .windows()
            .iter()
            .map(|w| (w.start, w.stop, w.commit_stop))
            .collect::<Vec<_>>(),
        [(0, 8, 2), (2, 9, 9)]
    );
    let mut syndrome = vec![0; 9];
    syndrome[0] = 1;
    syndrome[8] = 1;
    let result = decoder.decode(&syndrome).unwrap();
    assert_eq!(result.predicted.to_u64(), Some(1));
    assert_eq!(result.windows[0].selected_joint_class.to_u64(), Some(3));
    assert_eq!(result.windows[1].observed, vec![0; 7]);
    assert!(result.windows[1].committee.is_none());
}

#[test]
fn joint_map_is_not_replaced_by_marginal_map_or_gap() {
    let dem =
        SparseDem::from_dem_str("error(0.201) D0 L0\nerror(0.2) D0 L1\nerror(0.199) D0 L0 L1\n")
            .unwrap();
    let mut decoder =
        WindowedDecoder::from_sparse_dem(&dem, window(vec![vec![0]], 8, 2), settings(), 32)
            .unwrap();
    let result = decoder.decode(&[1]).unwrap();
    let selected = &result.windows[0].committee.as_ref().unwrap().selected;
    assert_eq!(selected.predicted.to_u64(), Some(1));
    assert_eq!(result.predicted.to_u64(), Some(1));
    let mut marginal = [0.0; 2];
    for class in &selected.logical_masses {
        marginal[usize::from(class.logical.get(1))] += class.log_mass.exp();
    }
    assert!(marginal[1] > marginal[0]);
    let expected = (marginal[1] / marginal[0]).ln().abs();
    assert!((result.min_abs_llrs[1] - expected).abs() < 1e-12);
    assert!((result.min_abs_llrs[1] - selected.runner_up_gap.unwrap()).abs() > 1e-3);
}

#[test]
fn detector_free_ownership_and_zero_policy_are_explicit_with_wide_classes() {
    let dem = SparseDem::from_dem_str("error(0.8) L130\n").unwrap();
    let mut config = window(vec![vec![], vec![]], 8, 2);
    assert!(WindowPlan::from_sparse_dem(&dem, config.clone()).is_err());
    config.detector_free_error_owners.insert(0, 0);
    let mut skip = WindowedDecoder::from_sparse_dem(&dem, config.clone(), settings(), 32).unwrap();
    let skipped = skip.decode(&[]).unwrap();
    assert!(skipped.predicted.is_zero());
    assert!(skipped.windows[0].committee.is_none());
    config.zero_policy = ZeroSyndromePolicy::Decode;
    let mut decode = WindowedDecoder::from_sparse_dem(&dem, config, settings(), 32).unwrap();
    let result = decode.decode(&[]).unwrap();
    assert!(result.predicted.get(130));
    assert!(result.windows[0].selected_joint_class.get(130));
    assert_eq!(result.predicted.count_ones(), 1);
    assert_ne!(result.trace_hash, skipped.trace_hash);
    assert!(result.min_abs_llrs[130].is_finite());
}

#[test]
fn zero_probability_order_and_xor_normalized_targets_are_preserved() {
    let dem = SparseDem {
        mechanisms: vec![(0.0, vec![0, 0, 1], vec![0, 0]), (0.2, vec![1, 0], vec![0])],
        detector_coords: BTreeMap::new(),
        num_detectors: 2,
        num_observables: 1,
    };
    let plan = WindowPlan::from_sparse_dem(&dem, window(vec![vec![0], vec![1]], 8, 2)).unwrap();
    assert_eq!(
        plan.models()[0].mechanisms,
        [(0.0, vec![1], vec![]), (0.2, vec![0, 1], vec![0])]
    );
    assert_eq!(
        plan.models()[0].mechanisms[0].0.to_bits(),
        0.0_f64.to_bits()
    );
}

#[test]
fn partitions_ownership_and_column_budgets_are_validated() {
    let dem = SparseDem::from_dem_str("error(0.1) D0 D1 L0\n").unwrap();
    for blocks in [
        vec![vec![0], vec![0]],
        vec![vec![0]],
        vec![vec![0], vec![2]],
    ] {
        assert!(WindowPlan::from_sparse_dem(&dem, window(blocks, 8, 2)).is_err());
    }
    for (width, stride) in [(0, 0), (1, 2), (8, 0)] {
        assert!(
            WindowPlan::from_sparse_dem(&dem, window(vec![vec![0], vec![1]], width, stride))
                .is_err()
        );
    }
    let good = window(vec![vec![0], vec![1]], 8, 2);
    let mut bad = good.clone();
    bad.detector_free_error_owners.insert(0, 0);
    assert!(WindowPlan::from_sparse_dem(&dem, bad).is_err());
    assert!(WindowedDecoder::from_sparse_dem(&dem, good.clone(), settings(), 0).is_err());
    let mut ordered = settings();
    ordered.column_order = Some(vec![0]);
    assert!(WindowedDecoder::from_sparse_dem(&dem, good, ordered, 32).is_err());
}

#[test]
fn failed_window_retains_error_then_next_nonzero_shot_starts_fresh() {
    let dem = SparseDem::from_dem_str("error(0.1) D0 L0\ndetector D1\n").unwrap();
    let mut decoder = WindowedDecoder::from_sparse_dem(
        &dem,
        window(vec![vec![0], vec![1]], 8, 2),
        settings(),
        32,
    )
    .unwrap();
    let error = decoder.decode(&[0, 1]).unwrap_err();
    assert_eq!(error.failed_window, Some(0));
    assert!(error.completed_windows.is_empty());
    let next = decoder.decode(&[1, 0]).unwrap();
    assert_eq!(next.predicted.to_u64(), Some(1));
    assert!(next.windows[0].committee.is_some());
    assert!(decoder.decode(&[2, 0]).is_err());
    assert!(decoder.decode(&[0]).is_err());
}

#[test]
fn mid_shot_failure_keeps_completed_windows_without_returning_a_partial_prediction() {
    let dem = SparseDem::from_dem_str("error(0.1) D0 L0\ndetector D8\n").unwrap();
    let config = window((0..9).map(|d| vec![d]).collect(), 8, 2);
    let mut decoder = WindowedDecoder::from_sparse_dem(&dem, config, settings(), 32).unwrap();
    let mut invalid = vec![0; 9];
    invalid[0] = 1;
    invalid[8] = 1;
    let failure = decoder.decode(&invalid).unwrap_err();
    assert_eq!(failure.failed_window, Some(1));
    assert_eq!(failure.completed_windows.len(), 1);
    assert_eq!(
        failure.completed_windows[0].selected_joint_class.to_u64(),
        Some(1)
    );
    invalid[8] = 0;
    let valid = decoder.decode(&invalid).unwrap();
    assert_eq!(valid.predicted.to_u64(), Some(1));
    assert_eq!(valid.windows.len(), 2);
}

#[test]
fn bounded_pruned_window_matches_direct_full_result_for_repeated_shots() {
    let mut text = String::new();
    for j in 0..40 {
        writeln!(
            &mut text,
            "error({}) L{}",
            0.05 + f64::from(j % 3) * 0.02,
            j % 12
        )
        .unwrap();
    }
    let dem = SparseDem::from_dem_str(&text).unwrap();
    let mut config = window(vec![vec![]], 8, 2);
    config.zero_policy = ZeroSyndromePolicy::Decode;
    config.detector_free_error_owners = (0..40).map(|j| (j, 0)).collect();
    let mut decoder = WindowedDecoder::from_sparse_dem(&dem, config, settings(), 32).unwrap();
    let mut direct_settings = settings();
    direct_settings.column_order = Some(deadline_column_order(&dem).unwrap());
    let mut direct = FrontierCommittee::from_sparse_dem(&dem, direct_settings).unwrap();
    for _ in 0..2 {
        let expected = direct.decode(&[]).unwrap();
        let actual = decoder.decode(&[]).unwrap();
        assert_eq!(actual.windows[0].committee.as_ref(), Some(&expected));
        assert!(expected.selected.dropped_states > 0);
        assert!(actual.windows[0].step_calls >= 6);
    }
}
