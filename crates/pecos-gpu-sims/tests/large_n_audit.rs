// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
// except in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0

//! Large-N GPU state-vector correctness at N=18, 20, 22. Previous audits
//! only cover N<=14 (`gate_audit`) and N<=14 fuzz. This one exercises the
//! workgroup dispatch / indexing at state sizes 2 MB, 8 MB, and 32 MB where
//! any 2^N overflow, workgroup-stride, or buffer-layout bug would surface.
//!
//! Reference: CPU `StateVecSoA`. Tolerance: f64 1e-5 (f32 gate constants limit
//! precision regardless of backend), f32 5e-3.

use pecos_core::{Angle64, QubitId};
use pecos_gpu_sims::{GpuStateVec32, GpuStateVec64};
use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable, StateVecSoA};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

#[derive(Clone, Copy)]
enum Op {
    H(usize),
    Rz(usize, f64),
    Cx(usize, usize),
    Cz(usize, usize),
}

fn gen_op(rng: &mut StdRng, n: usize) -> Op {
    match rng.random_range(0u32..4) {
        0 => Op::H(rng.random_range(0..n)),
        1 => Op::Rz(rng.random_range(0..n), rng.random_range(-3.0..3.0)),
        2 => {
            let a = rng.random_range(0..n);
            let mut b = rng.random_range(0..n);
            while b == a {
                b = rng.random_range(0..n);
            }
            Op::Cx(a, b)
        }
        _ => {
            let a = rng.random_range(0..n);
            let mut b = rng.random_range(0..n);
            while b == a {
                b = rng.random_range(0..n);
            }
            Op::Cz(a, b)
        }
    }
}

fn apply_cpu(sim: &mut StateVecSoA, op: Op) {
    match op {
        Op::H(q) => {
            sim.h(&[QubitId(q)]);
        }
        Op::Rz(q, t) => {
            sim.rz(Angle64::from_radians(t), &[QubitId(q)]);
        }
        Op::Cx(a, b) => {
            sim.cx(&[(QubitId(a), QubitId(b))]);
        }
        Op::Cz(a, b) => {
            sim.cz(&[(QubitId(a), QubitId(b))]);
        }
    }
}

fn apply_gpu32(sim: &mut GpuStateVec32, op: Op) {
    match op {
        Op::H(q) => {
            sim.h(&[QubitId(q)]);
        }
        Op::Rz(q, t) => {
            sim.rz(Angle64::from_radians(t), &[QubitId(q)]);
        }
        Op::Cx(a, b) => {
            sim.cx(&[(QubitId(a), QubitId(b))]);
        }
        Op::Cz(a, b) => {
            sim.cz(&[(QubitId(a), QubitId(b))]);
        }
    }
}

fn apply_gpu64(sim: &mut GpuStateVec64, op: Op) {
    match op {
        Op::H(q) => {
            sim.h(&[QubitId(q)]);
        }
        Op::Rz(q, t) => {
            sim.rz(Angle64::from_radians(t), &[QubitId(q)]);
        }
        Op::Cx(a, b) => {
            sim.cx(&[(QubitId(a), QubitId(b))]);
        }
        Op::Cz(a, b) => {
            sim.cz(&[(QubitId(a), QubitId(b))]);
        }
    }
}

fn run_cross_check(n: usize, gates: usize, seed: u64) {
    let mut rng = StdRng::seed_from_u64(seed);
    let ops: Vec<Op> = (0..gates).map(|_| gen_op(&mut rng, n)).collect();

    // CPU reference
    let mut cpu = StateVecSoA::new(n);
    for &op in &ops {
        apply_cpu(&mut cpu, op);
    }
    let cpu_state: Vec<[f64; 2]> = cpu.state().into_iter().map(|c| [c.re, c.im]).collect();

    let n_u32 = u32::try_from(n).expect("test N fits in u32");
    // f32 GPU
    if let Ok(mut g32) = GpuStateVec32::new(n_u32) {
        for &op in &ops {
            apply_gpu32(&mut g32, op);
        }
        let s: Vec<[f64; 2]> = g32
            .state()
            .into_iter()
            .map(|[re, im]| [f64::from(re), f64::from(im)])
            .collect();
        let d = max_diff(&s, &cpu_state);
        assert!(d < 5e-3, "N={n} G={gates} seed={seed} f32 diff={d:.3e}");
    }

    // f64 GPU
    if let Ok(mut g64) = GpuStateVec64::new(n_u32) {
        for &op in &ops {
            apply_gpu64(&mut g64, op);
        }
        let s = g64.state();
        let d = max_diff(&s, &cpu_state);
        assert!(d < 1e-5, "N={n} G={gates} seed={seed} f64 diff={d:.3e}");
    }
}

fn max_diff(a: &[[f64; 2]], b: &[[f64; 2]]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|([x0, x1], [y0, y1])| {
            let dr = x0 - y0;
            let di = x1 - y1;
            (dr * dr + di * di).sqrt()
        })
        .fold(0.0, f64::max)
}

#[test]
fn n18_cross_check() {
    // 2 MB state
    run_cross_check(18, 40, 0x1818);
    run_cross_check(18, 40, 0x1819);
}

#[test]
fn n20_cross_check() {
    // 8 MB state
    run_cross_check(20, 30, 0x2020);
}

#[test]
fn n22_cross_check() {
    // 32 MB state -- heaviest test in the suite, single run
    run_cross_check(22, 20, 0x2222);
}
