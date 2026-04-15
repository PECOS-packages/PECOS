// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
// except in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0

//! Supplementary audits covering gaps surfaced during the review:
//!
//! - `GpuStabMulti` shot-consistency on deterministic circuits (all shots agree,
//!   match CPU single-shot).
//! - `GpuDensityMatrix` measurement distribution vs `StateVecSoA` expectation over
//!   many trials.
//! - Gate fusion replay: force per-gate dispatch vs normal fused batching, the
//!   two must produce bit-identical output (modulo f32 rounding) for any
//!   circuit.

use pecos_core::{Angle64, QubitId};
use pecos_gpu_sims::{GpuDensityMatrix32, GpuStabMulti, GpuStateVec32, GpuStateVec64};
use pecos_simulators::{
    ArbitraryRotationGateable, CliffordGateable, QuantumSimulator, SparseStab, StateVecSoA,
};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

// =============================================================================
// #2: GpuStabMulti shot-consistency on deterministic Clifford circuits
// =============================================================================

/// Build a circuit whose Z-basis measurements are deterministic (no
/// superposition on the measured qubits), then run it through `GpuStabMulti`
/// with many shots and CPU `SparseStab` once. Every GPU shot must match the
/// single CPU outcome.
fn check_deterministic_multi<F>(label: &str, n: usize, shots: usize, build: F)
where
    F: Fn(&mut dyn FnMut(usize, GateOp)),
{
    // Collect the gate sequence.
    let mut gates: Vec<(usize, GateOp)> = Vec::new();
    build(&mut |q, g| gates.push((q, g)));

    // CPU reference.
    let mut cpu = SparseStab::new(n);
    for (_idx, op) in &gates {
        apply_cpu_clifford(&mut cpu, *op);
    }
    let cpu_results: Vec<bool> = (0..n).map(|q| cpu.mz(&[QubitId(q)])[0].outcome).collect();

    // GPU multi-shot.
    let Ok(mut gpu) = GpuStabMulti::<pecos_random::PecosRng>::with_seed(n, shots, 42) else {
        return;
    };
    for (_idx, op) in &gates {
        apply_multi_clifford(&mut gpu, *op);
    }
    let gpu_results: Vec<Vec<bool>> = gpu.mz(&(0..n).map(QubitId).collect::<Vec<_>>());

    assert_eq!(gpu_results.len(), shots, "{label}: shot count");
    for (shot, row) in gpu_results.iter().enumerate() {
        assert_eq!(row, &cpu_results, "{label}: shot {shot} disagrees with CPU");
    }
}

#[derive(Clone, Copy, Debug)]
enum GateOp {
    X(usize),
    H(usize),
    Cx(usize, usize),
    Cz(usize, usize),
}

fn apply_cpu_clifford(sim: &mut SparseStab, op: GateOp) {
    match op {
        GateOp::X(q) => {
            sim.x(&[QubitId(q)]);
        }
        GateOp::H(q) => {
            sim.h(&[QubitId(q)]);
        }
        GateOp::Cx(a, b) => {
            sim.cx(&[(QubitId(a), QubitId(b))]);
        }
        GateOp::Cz(a, b) => {
            sim.cz(&[(QubitId(a), QubitId(b))]);
        }
    }
}

fn apply_multi_clifford(sim: &mut GpuStabMulti, op: GateOp) {
    match op {
        GateOp::X(q) => {
            sim.x(&[QubitId(q)]);
        }
        GateOp::H(q) => {
            sim.h(&[QubitId(q)]);
        }
        GateOp::Cx(a, b) => {
            sim.cx(&[(QubitId(a), QubitId(b))]);
        }
        GateOp::Cz(a, b) => {
            sim.cz(&[(QubitId(a), QubitId(b))]);
        }
    }
}

#[test]
fn stab_multi_all_zero_basis() {
    // No gates: all qubits in |0>, every shot must read 0s.
    check_deterministic_multi("all |0>", 5, 128, |_| {});
}

#[test]
fn stab_multi_x_prep() {
    // Apply X to qubits 1 and 3: expect 01010 reading.
    check_deterministic_multi("X prep", 5, 128, |emit| {
        emit(0, GateOp::X(1));
        emit(1, GateOp::X(3));
    });
}

#[test]
fn stab_multi_cx_chain() {
    // X(0) then CX(0,1), CX(1,2), ..., CX(n-2, n-1) => all bits flipped (GHZ-with-X).
    check_deterministic_multi("CX chain", 5, 128, |emit| {
        emit(0, GateOp::X(0));
        for q in 0..4 {
            emit(q, GateOp::Cx(q, q + 1));
        }
    });
}

#[test]
fn stab_multi_cz_noop_on_zeros() {
    // CZ on |00..0> is identity: every shot still reads 0s.
    check_deterministic_multi("CZ on zeros", 5, 128, |emit| {
        for q in 0..4 {
            emit(q, GateOp::Cz(q, q + 1));
        }
    });
}

#[test]
fn stab_multi_h_h_identity() {
    // H H is identity: |0> -> |+> -> |0>, deterministic.
    check_deterministic_multi("H H identity", 4, 64, |emit| {
        for q in 0..4 {
            emit(q, GateOp::H(q));
            emit(q, GateOp::H(q));
        }
    });
}

// =============================================================================
// #3: Density matrix mz distribution vs state-vector expectation
// =============================================================================

/// For |+> state: P(0) = P(1) = 0.5. Running GPU DM mz many times should
/// produce ~50/50.
#[test]
fn dm_mz_plus_state_distribution() {
    let Ok(mut gpu) = GpuDensityMatrix32::with_seed(1, 123) else {
        return;
    };
    gpu.h(&[QubitId(0)]);

    let trials = 400;
    let mut ones = 0usize;
    for _ in 0..trials {
        // Fresh prep each trial since mz collapses.
        gpu.reset();
        gpu.h(&[QubitId(0)]);
        let res = gpu.mz(&[QubitId(0)]);
        if res[0].outcome {
            ones += 1;
        }
    }

    // 3-sigma window on Binomial(400, 0.5): sigma = 10, so |ones - 200| < 30 typically.
    #[allow(clippy::cast_precision_loss)] // ones <= trials = 400, exact in f64
    let p = ones as f64 / f64::from(trials);
    assert!(
        (p - 0.5).abs() < 0.1,
        "|+> measurement: got P(1) = {p} from {ones}/{trials}"
    );
}

/// Bell state: measurements should be perfectly correlated across shots.
#[test]
fn dm_mz_bell_state_correlation() {
    let Ok(mut gpu) = GpuDensityMatrix32::with_seed(2, 321) else {
        return;
    };

    let trials = 200;
    let mut correlated = 0usize;
    for _ in 0..trials {
        gpu.reset();
        gpu.h(&[QubitId(0)]).cx(&[(QubitId(0), QubitId(1))]);
        let res = gpu.mz(&[QubitId(0), QubitId(1)]);
        if res[0].outcome == res[1].outcome {
            correlated += 1;
        }
    }

    assert_eq!(correlated, trials, "Bell state: correlation must be exact");
}

// =============================================================================
// #4: Gate fusion replay -- per-gate dispatch vs normal fused batching
// =============================================================================

/// Apply a long sequence of gates two ways: once normally (fusion groups them),
/// and once with a forced readback after every gate (prevents fusion across
/// operations). Both outputs must agree.
#[test]
fn fusion_replay_f32() {
    let n: u32 = 10;
    let n_us = usize::try_from(n).unwrap();
    let mut rng = StdRng::seed_from_u64(777);
    let seq: Vec<Instr> = (0..60).map(|_| gen_instr(&mut rng, n_us)).collect();

    let Ok(mut fused) = GpuStateVec32::new(n) else {
        return;
    };
    let qubits: Vec<QubitId> = (0..n_us).map(QubitId).collect();
    fused.h(&qubits);
    for instr in &seq {
        apply_instr_f32(&mut fused, *instr);
    }
    let fused_state: Vec<[f32; 2]> = fused.state();

    let Ok(mut unfused) = GpuStateVec32::new(n) else {
        return;
    };
    unfused.h(&qubits);
    let _ = unfused.state(); // force H flush
    for instr in &seq {
        apply_instr_f32(&mut unfused, *instr);
        let _ = unfused.state(); // force per-gate flush (no fusion across the boundary)
    }
    let unfused_state = unfused.state();

    assert_eq!(fused_state.len(), unfused_state.len());
    let tol: f32 = 5e-3;
    let mut max_diff = 0.0f32;
    for ([fr, fi], [ur, ui]) in fused_state.iter().zip(unfused_state.iter()) {
        let dr = fr - ur;
        let di = fi - ui;
        let d = (dr * dr + di * di).sqrt();
        if d > max_diff {
            max_diff = d;
        }
    }
    assert!(
        max_diff < tol,
        "f32 fused vs per-gate dispatch diverged: max_diff = {max_diff:.3e}"
    );
}

#[test]
fn fusion_replay_f64() {
    let n: u32 = 10;
    let n_us = usize::try_from(n).unwrap();
    let mut rng = StdRng::seed_from_u64(888);
    let seq: Vec<Instr> = (0..60).map(|_| gen_instr(&mut rng, n_us)).collect();

    let Ok(mut fused) = GpuStateVec64::new(n) else {
        return;
    };
    let qubits: Vec<QubitId> = (0..n_us).map(QubitId).collect();
    fused.h(&qubits);
    for instr in &seq {
        apply_instr_f64(&mut fused, *instr);
    }
    let fused_state = fused.state();

    let Ok(mut unfused) = GpuStateVec64::new(n) else {
        return;
    };
    unfused.h(&qubits);
    let _ = unfused.state();
    for instr in &seq {
        apply_instr_f64(&mut unfused, *instr);
        let _ = unfused.state();
    }
    let unfused_state = unfused.state();

    assert_eq!(fused_state.len(), unfused_state.len());
    let tol: f64 = 1e-5;
    let mut max_diff = 0.0f64;
    for ([fr, fi], [ur, ui]) in fused_state.iter().zip(unfused_state.iter()) {
        let dr = fr - ur;
        let di = fi - ui;
        let d = (dr * dr + di * di).sqrt();
        if d > max_diff {
            max_diff = d;
        }
    }
    assert!(
        max_diff < tol,
        "f64 fused vs per-gate dispatch diverged: max_diff = {max_diff:.3e}"
    );
}

#[derive(Clone, Copy)]
enum Instr {
    H(usize),
    Rz(usize, f64),
    Cx(usize, usize),
    Cz(usize, usize),
    Rzz(usize, usize, f64),
}

fn gen_instr(rng: &mut StdRng, n: usize) -> Instr {
    match rng.random_range(0u32..5) {
        0 => Instr::H(rng.random_range(0..n)),
        1 => Instr::Rz(rng.random_range(0..n), rng.random_range(-3.0..3.0)),
        2 => {
            let a = rng.random_range(0..n);
            let mut b = rng.random_range(0..n);
            while b == a {
                b = rng.random_range(0..n);
            }
            Instr::Cx(a, b)
        }
        3 => {
            let a = rng.random_range(0..n);
            let mut b = rng.random_range(0..n);
            while b == a {
                b = rng.random_range(0..n);
            }
            Instr::Cz(a, b)
        }
        _ => {
            let a = rng.random_range(0..n);
            let mut b = rng.random_range(0..n);
            while b == a {
                b = rng.random_range(0..n);
            }
            Instr::Rzz(a, b, rng.random_range(-3.0..3.0))
        }
    }
}

fn apply_instr_f32(sim: &mut GpuStateVec32, instr: Instr) {
    match instr {
        Instr::H(q) => {
            sim.h(&[QubitId(q)]);
        }
        Instr::Rz(q, t) => {
            sim.rz(Angle64::from_radians(t), &[QubitId(q)]);
        }
        Instr::Cx(a, b) => {
            sim.cx(&[(QubitId(a), QubitId(b))]);
        }
        Instr::Cz(a, b) => {
            sim.cz(&[(QubitId(a), QubitId(b))]);
        }
        Instr::Rzz(a, b, t) => {
            sim.rzz(Angle64::from_radians(t), &[(QubitId(a), QubitId(b))]);
        }
    }
}

fn apply_instr_f64(sim: &mut GpuStateVec64, instr: Instr) {
    match instr {
        Instr::H(q) => {
            sim.h(&[QubitId(q)]);
        }
        Instr::Rz(q, t) => {
            sim.rz(Angle64::from_radians(t), &[QubitId(q)]);
        }
        Instr::Cx(a, b) => {
            sim.cx(&[(QubitId(a), QubitId(b))]);
        }
        Instr::Cz(a, b) => {
            sim.cz(&[(QubitId(a), QubitId(b))]);
        }
        Instr::Rzz(a, b, t) => {
            sim.rzz(Angle64::from_radians(t), &[(QubitId(a), QubitId(b))]);
        }
    }
}

// =============================================================================
// Sanity: StateVecSoA reference for the fusion replay isn't needed -- the
// per-gate and fused paths are compared against each other. But verify once
// against CPU to make sure both match the ground truth for a small N.
// =============================================================================

#[test]
fn fusion_replay_matches_cpu() {
    let n: usize = 6;
    let mut rng = StdRng::seed_from_u64(999);
    let seq: Vec<Instr> = (0..40).map(|_| gen_instr(&mut rng, n)).collect();

    let mut cpu = StateVecSoA::new(n);
    let qubits: Vec<QubitId> = (0..n).map(QubitId).collect();
    cpu.h(&qubits);
    for instr in &seq {
        match *instr {
            Instr::H(q) => {
                cpu.h(&[QubitId(q)]);
            }
            Instr::Rz(q, t) => {
                cpu.rz(Angle64::from_radians(t), &[QubitId(q)]);
            }
            Instr::Cx(a, b) => {
                cpu.cx(&[(QubitId(a), QubitId(b))]);
            }
            Instr::Cz(a, b) => {
                cpu.cz(&[(QubitId(a), QubitId(b))]);
            }
            Instr::Rzz(a, b, t) => {
                cpu.rzz(Angle64::from_radians(t), &[(QubitId(a), QubitId(b))]);
            }
        }
    }
    let cpu_state: Vec<[f64; 2]> = cpu.state().into_iter().map(|c| [c.re, c.im]).collect();

    let Ok(mut gpu) = GpuStateVec64::new(u32::try_from(n).expect("test N fits in u32")) else {
        return;
    };
    gpu.h(&qubits);
    for instr in &seq {
        apply_instr_f64(&mut gpu, *instr);
    }
    let gpu_state = gpu.state();

    let mut max_diff = 0.0f64;
    for ([gr, gi], [cr, ci]) in gpu_state.iter().zip(cpu_state.iter()) {
        let dr = gr - cr;
        let di = gi - ci;
        let d = (dr * dr + di * di).sqrt();
        if d > max_diff {
            max_diff = d;
        }
    }
    assert!(
        max_diff < 1e-5,
        "GPU (fused) vs CPU ground truth diverged: max_diff = {max_diff:.3e}"
    );
}
