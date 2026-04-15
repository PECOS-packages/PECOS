// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
// except in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0

//! Correctness audit for `StateVecSoA::flush_blocked`.
//!
//! `flush_blocked` is the cache-blocked pending-gate flush path. It fires when
//! `num_qubits >= 21` AND `pending_count >= 3`. It does two things:
//!  1. Applies low-stride (q < `block_bits`, default 14) pending gates in a
//!     block-by-block loop so each block is loaded from DRAM once.
//!  2. Applies remaining high-stride pending gates individually, with an
//!     adjacent-pair optimisation (`flush_pair`).
//!
//! These tests cross-check the blocked path against a `set_fusion(false)`
//! reference that dispatches every gate immediately via `apply_fused_matrix`
//! (no batching, no blocking). Both should produce bit-identical output.
//!
//! N=21 → 32 MB per state vector. Each test creates a handful.

use pecos_core::{Angle64, QubitId};
use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable, StateVecSoA};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

const N: usize = 21;

/// A single-qubit-gate instruction (picks arbitrary 1q gates).
#[derive(Clone, Copy)]
enum Op1q {
    H(usize),
    X(usize),
    Y(usize),
    Z(usize),
    Sx(usize),
    Sxdg(usize),
    Sy(usize),
    Sydg(usize),
    Sz(usize),
    Szdg(usize),
    T(usize),
    Tdg(usize),
    Rx(usize, f64),
    Ry(usize, f64),
    Rz(usize, f64),
}

fn gen_1q(rng: &mut StdRng, n: usize) -> Op1q {
    let kind = rng.random_range(0u32..15);
    let q = rng.random_range(0..n);
    match kind {
        0 => Op1q::H(q),
        1 => Op1q::X(q),
        2 => Op1q::Y(q),
        3 => Op1q::Z(q),
        4 => Op1q::Sx(q),
        5 => Op1q::Sxdg(q),
        6 => Op1q::Sy(q),
        7 => Op1q::Sydg(q),
        8 => Op1q::Sz(q),
        9 => Op1q::Szdg(q),
        10 => Op1q::T(q),
        11 => Op1q::Tdg(q),
        12 => Op1q::Rx(q, rng.random_range(-3.1..3.1)),
        13 => Op1q::Ry(q, rng.random_range(-3.1..3.1)),
        _ => Op1q::Rz(q, rng.random_range(-3.1..3.1)),
    }
}

fn apply_1q(sim: &mut StateVecSoA, op: Op1q) {
    match op {
        Op1q::H(q) => {
            sim.h(&[QubitId(q)]);
        }
        Op1q::X(q) => {
            sim.x(&[QubitId(q)]);
        }
        Op1q::Y(q) => {
            sim.y(&[QubitId(q)]);
        }
        Op1q::Z(q) => {
            sim.z(&[QubitId(q)]);
        }
        Op1q::Sx(q) => {
            sim.sx(&[QubitId(q)]);
        }
        Op1q::Sxdg(q) => {
            sim.sxdg(&[QubitId(q)]);
        }
        Op1q::Sy(q) => {
            sim.sy(&[QubitId(q)]);
        }
        Op1q::Sydg(q) => {
            sim.sydg(&[QubitId(q)]);
        }
        Op1q::Sz(q) => {
            sim.sz(&[QubitId(q)]);
        }
        Op1q::Szdg(q) => {
            sim.szdg(&[QubitId(q)]);
        }
        Op1q::T(q) => {
            sim.t(&[QubitId(q)]);
        }
        Op1q::Tdg(q) => {
            sim.tdg(&[QubitId(q)]);
        }
        Op1q::Rx(q, t) => {
            sim.rx(Angle64::from_radians(t), &[QubitId(q)]);
        }
        Op1q::Ry(q, t) => {
            sim.ry(Angle64::from_radians(t), &[QubitId(q)]);
        }
        Op1q::Rz(q, t) => {
            sim.rz(Angle64::from_radians(t), &[QubitId(q)]);
        }
    }
}

fn max_amp_diff(a: &mut StateVecSoA, b: &mut StateVecSoA) -> f64 {
    let sa = a.state();
    let sb = b.state();
    assert_eq!(sa.len(), sb.len());
    sa.iter()
        .zip(sb.iter())
        .map(|(x, y)| {
            let dr = x.re - y.re;
            let di = x.im - y.im;
            (dr * dr + di * di).sqrt()
        })
        .fold(0.0, f64::max)
}

/// Prepare the reference and under-test simulators from the same seed.
/// `reference` uses `set_fusion(false)` so every gate goes through
/// `apply_fused_matrix` immediately -- no batching, no blocking.
/// `under_test` uses default (fusion on) so a large enough pending set
/// triggers `flush_blocked`.
fn two_sims() -> (StateVecSoA, StateVecSoA) {
    let mut reference = StateVecSoA::new(N);
    reference.set_fusion(false);
    let under_test = StateVecSoA::new(N);
    assert!(under_test.fusion_enabled(), "default should have fusion on");
    (reference, under_test)
}

/// Force a full flush on the batched simulator by reading the state.
/// `state()` is currently &self so we instead use a lightweight trick: apply
/// a CX on qubits that aren't present in pending to force `flush_two_qubit`
/// then undo it. Simpler: call `flush()` directly via the public API.
fn trigger_flush(sim: &mut StateVecSoA) {
    sim.flush();
}

#[test]
fn flush_blocked_single_qubit_fuzz() {
    // Many random 1q gates on different qubits across the whole register.
    // With fusion on + many pending gates + N=21, flush_blocked fires.
    let mut rng = StdRng::seed_from_u64(0xdead_beef);
    let ops: Vec<Op1q> = (0..60).map(|_| gen_1q(&mut rng, N)).collect();

    let (mut reference, mut under_test) = two_sims();
    for &op in &ops {
        apply_1q(&mut reference, op);
        apply_1q(&mut under_test, op);
    }
    // Reference was flushed per-gate (fusion off). under_test has pending gates.
    trigger_flush(&mut under_test);

    let d = max_amp_diff(&mut reference, &mut under_test);
    assert!(
        d < 1e-10,
        "flush_blocked diverged from per-gate ref: max_diff={d:.3e}"
    );
}

#[test]
fn flush_blocked_only_low_stride() {
    // Gates only on low-stride qubits (q < 14). Exercises the blocked path
    // exclusively -- no high-stride cleanup pass.
    let mut rng = StdRng::seed_from_u64(0x0_fedb_acca);
    let ops: Vec<Op1q> = (0..80)
        .map(|_| {
            let mut op = gen_1q(&mut rng, 14);
            // gen_1q uses n for qubit range; ensure q < 14 (already set by n=14).
            // Pass-through.
            op = match op {
                Op1q::H(q) => Op1q::H(q.min(13)),
                Op1q::X(q) => Op1q::X(q.min(13)),
                _ => op,
            };
            op
        })
        .collect();

    let (mut reference, mut under_test) = two_sims();
    for &op in &ops {
        apply_1q(&mut reference, op);
        apply_1q(&mut under_test, op);
    }
    trigger_flush(&mut under_test);

    let d = max_amp_diff(&mut reference, &mut under_test);
    assert!(d < 1e-10, "low-stride only: max_diff={d:.3e}");
}

#[test]
fn flush_blocked_only_high_stride() {
    // Gates only on high-stride qubits (q >= 14). flush_blocked's low-stride
    // loop produces no work; only the pairing cleanup runs.
    let mut rng = StdRng::seed_from_u64(0x00c0_ffee);
    let ops: Vec<Op1q> = (0..40)
        .map(|_| {
            let base = gen_1q(&mut rng, 7); // maps 0..7
            // Remap qubit to 14..21.
            match base {
                Op1q::H(q) => Op1q::H(q + 14),
                Op1q::X(q) => Op1q::X(q + 14),
                Op1q::Y(q) => Op1q::Y(q + 14),
                Op1q::Z(q) => Op1q::Z(q + 14),
                Op1q::Sx(q) => Op1q::Sx(q + 14),
                Op1q::Sxdg(q) => Op1q::Sxdg(q + 14),
                Op1q::Sy(q) => Op1q::Sy(q + 14),
                Op1q::Sydg(q) => Op1q::Sydg(q + 14),
                Op1q::Sz(q) => Op1q::Sz(q + 14),
                Op1q::Szdg(q) => Op1q::Szdg(q + 14),
                Op1q::T(q) => Op1q::T(q + 14),
                Op1q::Tdg(q) => Op1q::Tdg(q + 14),
                Op1q::Rx(q, t) => Op1q::Rx(q + 14, t),
                Op1q::Ry(q, t) => Op1q::Ry(q + 14, t),
                Op1q::Rz(q, t) => Op1q::Rz(q + 14, t),
            }
        })
        .collect();

    let (mut reference, mut under_test) = two_sims();
    for &op in &ops {
        apply_1q(&mut reference, op);
        apply_1q(&mut under_test, op);
    }
    trigger_flush(&mut under_test);

    let d = max_amp_diff(&mut reference, &mut under_test);
    assert!(d < 1e-10, "high-stride only: max_diff={d:.3e}");
}

#[test]
fn flush_blocked_boundary_qubits() {
    // Exactly the qubits right at the low/high boundary: 12, 13, 14, 15.
    // q=13 has step=8192 == block_size/2 (one outer iteration per block).
    // q=14 is the first high-stride qubit.
    let mut rng = StdRng::seed_from_u64(0x1337);
    let ops: Vec<Op1q> = (0..50)
        .map(|_| {
            let q = 12 + rng.random_range(0u32..4) as usize;
            match rng.random_range(0u32..5) {
                0 => Op1q::H(q),
                1 => Op1q::Sx(q),
                2 => Op1q::Sz(q),
                3 => Op1q::T(q),
                _ => Op1q::Rz(q, rng.random_range(-3.1..3.1)),
            }
        })
        .collect();

    let (mut reference, mut under_test) = two_sims();
    for &op in &ops {
        apply_1q(&mut reference, op);
        apply_1q(&mut under_test, op);
    }
    trigger_flush(&mut under_test);

    let d = max_amp_diff(&mut reference, &mut under_test);
    assert!(d < 1e-10, "boundary qubits: max_diff={d:.3e}");
}

#[test]
fn flush_blocked_interleaved_with_cx() {
    // Mixed: 1q gates (queued) interleaved with cx (which force
    // flush_two_qubit on those qubits). This exercises the partial-flush
    // paths and ensures flush_blocked still agrees when it fires for the
    // remaining pending set.
    let mut rng = StdRng::seed_from_u64(0xbeef);
    let (mut reference, mut under_test) = two_sims();

    for _ in 0..30 {
        // Burst of 1q gates
        for _ in 0..5 {
            let op = gen_1q(&mut rng, N);
            apply_1q(&mut reference, op);
            apply_1q(&mut under_test, op);
        }
        // Random cx pair
        let a = rng.random_range(0..N);
        let mut b = rng.random_range(0..N);
        while b == a {
            b = rng.random_range(0..N);
        }
        reference.cx(&[(QubitId(a), QubitId(b))]);
        under_test.cx(&[(QubitId(a), QubitId(b))]);
    }
    trigger_flush(&mut under_test);

    let d = max_amp_diff(&mut reference, &mut under_test);
    assert!(d < 1e-10, "interleaved: max_diff={d:.3e}");
}

#[test]
fn flush_blocked_minimum_pending_count() {
    // The threshold requires pending_count >= 3 AND num_qubits >= 21.
    // With exactly 3 pending gates at N=21, flush_blocked MUST fire and
    // produce identical output to the non-blocked path.
    let (mut reference, mut under_test) = two_sims();

    let ops = [Op1q::H(0), Op1q::Sz(10), Op1q::Rx(20, 0.7)];
    for &op in &ops {
        apply_1q(&mut reference, op);
        apply_1q(&mut under_test, op);
    }
    trigger_flush(&mut under_test);

    let d = max_amp_diff(&mut reference, &mut under_test);
    assert!(d < 1e-10, "min pending: max_diff={d:.3e}");
}

#[test]
fn flush_blocked_all_qubits_pending() {
    // All 21 qubits with a single pending gate each -- maximum pending set.
    let mut rng = StdRng::seed_from_u64(0x00ab_c123);
    let (mut reference, mut under_test) = two_sims();

    for q in 0..N {
        // Pick a non-trivial single-qubit matrix via random Rz + H (cheap but
        // not identity).
        let theta = rng.random_range(0.1..3.0);
        reference.h(&[QubitId(q)]);
        reference.rz(Angle64::from_radians(theta), &[QubitId(q)]);
        under_test.h(&[QubitId(q)]);
        under_test.rz(Angle64::from_radians(theta), &[QubitId(q)]);
    }
    trigger_flush(&mut under_test);

    let d = max_amp_diff(&mut reference, &mut under_test);
    assert!(d < 1e-10, "all qubits pending: max_diff={d:.3e}");
}
