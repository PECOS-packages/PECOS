// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use
// this file except in compliance with the License. You may obtain a copy of the
// License at https://www.apache.org/licenses/LICENSE-2.0

//! Clone isolation for every public matrix mutator, including methods with no
//! CH-form callers. Keep these in the correctness gate so it covers those APIs too.

use pecos_core::BitSet;
use pecos_simulators::SparseBinaryMatrix;

type Mat = SparseBinaryMatrix<BitSet>;

fn snapshot(mat: &Mat) -> Vec<Vec<bool>> {
    (0..mat.dim())
        .map(|i| (0..mat.dim()).map(|j| mat.get(i, j)).collect())
        .collect()
}

fn assert_views_match(mat: &Mat, expected: &[Vec<bool>]) {
    for (i, row) in expected.iter().enumerate() {
        for (j, &bit) in row.iter().enumerate() {
            assert_eq!(mat.row(i).contains(j), bit, "row view [{i}][{j}]");
            assert_eq!(mat.col(j).contains(i), bit, "column view [{i}][{j}]");
        }
    }
}

fn assert_clone_isolated(mutate: impl FnOnce(&mut Mat)) {
    let mut original = Mat::identity(4);
    original.set(0, 2, true);
    original.set(2, 1, true);
    let before = snapshot(&original);
    let mut cloned = original.clone();
    mutate(&mut cloned);
    assert_views_match(&original, &before);
    let after = snapshot(&cloned);
    assert_ne!(after, before, "fixture must actually change the clone");
    assert_views_match(&cloned, &after);
}

#[test]
fn cow_set_true() {
    assert_clone_isolated(|m| m.set(3, 0, true));
}

#[test]
fn cow_set_false() {
    assert_clone_isolated(|m| m.set(0, 2, false));
}

#[test]
fn cow_toggle() {
    assert_clone_isolated(|m| m.toggle(3, 0));
}

#[test]
fn cow_row_xor_assign() {
    assert_clone_isolated(|m| m.row_xor_assign(3, 0));
}

#[test]
fn cow_row_xor_from() {
    assert_clone_isolated(|m| m.row_xor_from(3, &m.clone(), 0));
}

#[test]
fn cow_col_xor_assign() {
    assert_clone_isolated(|m| m.col_xor_assign(3, 2));
}

#[test]
fn cow_col_xor_from() {
    assert_clone_isolated(|m| m.col_xor_from(3, &m.clone(), 2));
}

#[test]
fn cow_swap_rows() {
    assert_clone_isolated(|m| m.swap_rows(0, 3));
}

#[test]
fn cow_swap_cols() {
    assert_clone_isolated(|m| m.swap_cols(0, 3));
}

#[test]
fn cow_row_xor_set() {
    assert_clone_isolated(|m| {
        let mut set = BitSet::new();
        set.insert(0);
        set.insert(3);
        m.row_xor_set(2, &set);
    });
}

#[test]
fn cow_reset_to_zero() {
    assert_clone_isolated(Mat::reset_to_zero);
}

#[test]
fn cow_reset_to_identity() {
    assert_clone_isolated(Mat::reset_to_identity);
}
