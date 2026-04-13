// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
// except in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0

//! Cross-check `GpuPauliProp` (shot 0) against the CPU `PauliProp` reference.
//!
//! Both simulators propagate a single Pauli fault through a Clifford circuit.
//! CPU `PauliProp` tracks one X-set and one Z-set; GPU `GpuPauliProp` tracks
//! per-shot X and Z fault bitmaps. With shots=1 and a single seed fault, they
//! must agree on which qubits end up with X-, Z-, or Y-components.
//!
//! The correspondence used:
//!   GPU `measure_z_flips`[0][q]  <=>  CPU `contains_x(q)` || `contains_y(q)`
//!   GPU `measure_x_flips`[0][q]  <=>  CPU `contains_z(q)` || `contains_y(q)`
//! (a Z-basis measurement of qubit q flips iff the tracked Pauli has X or Y on q;
//! an X-basis measurement flips iff the Pauli has Z or Y on q).

use pecos_core::QubitId;
use pecos_gpu_sims::GpuPauliProp;
use pecos_simulators::{CliffordGateable, PauliProp};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

#[derive(Clone, Copy, Debug)]
enum Op {
    H(usize),
    Sz(usize),
    Szdg(usize),
    X(usize),
    Y(usize),
    Z(usize),
    Cx(usize, usize),
    Cz(usize, usize),
    Swap(usize, usize),
}

fn pick_two(rng: &mut StdRng, n: usize) -> (usize, usize) {
    let a = rng.random_range(0..n);
    let mut b = rng.random_range(0..n);
    while b == a {
        b = rng.random_range(0..n);
    }
    (a, b)
}

fn gen_op(rng: &mut StdRng, n: usize) -> Op {
    match rng.random_range(0u32..9) {
        0 => Op::H(rng.random_range(0..n)),
        1 => Op::Sz(rng.random_range(0..n)),
        2 => Op::Szdg(rng.random_range(0..n)),
        3 => Op::X(rng.random_range(0..n)),
        4 => Op::Y(rng.random_range(0..n)),
        5 => Op::Z(rng.random_range(0..n)),
        6 => {
            let (a, b) = pick_two(rng, n);
            Op::Cx(a, b)
        }
        7 => {
            let (a, b) = pick_two(rng, n);
            Op::Cz(a, b)
        }
        _ => {
            let (a, b) = pick_two(rng, n);
            Op::Swap(a, b)
        }
    }
}

fn apply_cpu(sim: &mut PauliProp, op: Op) {
    match op {
        Op::H(q) => {
            sim.h(&[QubitId(q)]);
        }
        Op::Sz(q) => {
            sim.sz(&[QubitId(q)]);
        }
        Op::Szdg(q) => {
            sim.szdg(&[QubitId(q)]);
        }
        Op::X(q) => {
            sim.x(&[QubitId(q)]);
        }
        Op::Y(q) => {
            sim.y(&[QubitId(q)]);
        }
        Op::Z(q) => {
            sim.z(&[QubitId(q)]);
        }
        Op::Cx(a, b) => {
            sim.cx(&[(QubitId(a), QubitId(b))]);
        }
        Op::Cz(a, b) => {
            sim.cz(&[(QubitId(a), QubitId(b))]);
        }
        Op::Swap(a, b) => {
            sim.swap(&[(QubitId(a), QubitId(b))]);
        }
    }
}

fn apply_gpu(sim: &mut GpuPauliProp, op: Op) {
    match op {
        Op::H(q) => sim.h(&[q]),
        Op::Sz(q) => sim.sz(&[q]),
        Op::Szdg(q) => sim.szdg(&[q]),
        Op::X(q) => sim.x(&[q]),
        Op::Y(q) => sim.y(&[q]),
        Op::Z(q) => sim.z(&[q]),
        Op::Cx(a, b) => sim.cx(&[(a, b)]),
        Op::Cz(a, b) => sim.cz(&[(a, b)]),
        Op::Swap(a, b) => sim.swap(&[(a, b)]),
    }
}

fn run_cross_check(seed: u64, n: usize, gates: usize, fault_qubit: usize, fault_kind: &str) {
    let Ok(mut gpu) = GpuPauliProp::with_seed(n, 1, seed) else {
        return;
    };
    let mut cpu = PauliProp::new();

    // Inject same fault on both.
    match fault_kind {
        "x" => {
            gpu.inject_x_fault(fault_qubit);
            cpu.track_x(&[fault_qubit]);
        }
        "z" => {
            gpu.inject_z_fault(fault_qubit);
            cpu.track_z(&[fault_qubit]);
        }
        "y" => {
            gpu.inject_y_fault(fault_qubit);
            cpu.track_y(&[fault_qubit]);
        }
        _ => unreachable!(),
    }

    // Apply same random Clifford circuit to both.
    let mut rng = StdRng::seed_from_u64(seed);
    let ops: Vec<Op> = (0..gates).map(|_| gen_op(&mut rng, n)).collect();
    for &op in &ops {
        apply_gpu(&mut gpu, op);
        apply_cpu(&mut cpu, op);
    }

    // Read final Pauli frame.
    let qubits: Vec<usize> = (0..n).collect();
    let z_flips = gpu.measure_z_flips(&qubits);
    let x_flips = gpu.measure_x_flips(&qubits);

    for q in 0..n {
        // GPU Z-basis flip <=> CPU has X component (X or Y).
        let gpu_z = z_flips[0][q];
        let cpu_z = cpu.contains_x(q) || cpu.contains_y(q);
        assert_eq!(
            gpu_z, cpu_z,
            "fault={fault_kind}@{fault_qubit} seed={seed} N={n} G={gates} Z-flip mismatch at q={q}: gpu={gpu_z} cpu={cpu_z}"
        );

        // GPU X-basis flip <=> CPU has Z component (Z or Y).
        let gpu_x = x_flips[0][q];
        let cpu_x = cpu.contains_z(q) || cpu.contains_y(q);
        assert_eq!(
            gpu_x, cpu_x,
            "fault={fault_kind}@{fault_qubit} seed={seed} N={n} G={gates} X-flip mismatch at q={q}: gpu={gpu_x} cpu={cpu_x}"
        );
    }
}

#[test]
fn shrink_deterministic_check() {
    // Repeat the same 4-op sequence many times to check determinism.
    for trial in 0..10 {
        let Ok(mut gpu) = GpuPauliProp::with_seed(3, 1, 0) else {
            return;
        };
        gpu.inject_x_fault(0);
        gpu.cz(&[(2, 1)]);
        gpu.cx(&[(2, 0)]);
        gpu.cx(&[(1, 2)]);
        gpu.szdg(&[0]);
        let zf = gpu.measure_z_flips(&[0, 1, 2]);
        let xf = gpu.measure_x_flips(&[0, 1, 2]);
        eprintln!("trial {trial}: zf={:?} xf={:?}", zf[0], xf[0]);
    }
}

#[test]
fn trace_after_each_gate() {
    // Apply gates one by one, printing Pauli frame after each.
    let ops = [Op::Cz(2, 1), Op::Cx(2, 0), Op::Cx(1, 2), Op::Szdg(0)];
    let Ok(mut gpu) = GpuPauliProp::with_seed(3, 1, 0) else {
        return;
    };
    let mut cpu = PauliProp::new();
    gpu.inject_x_fault(0);
    cpu.track_x(&[0]);

    eprintln!(
        "START: CPU x={}  y={}  z={}",
        (0..3)
            .map(|q| if cpu.contains_x(q) { '1' } else { '0' })
            .collect::<String>(),
        (0..3)
            .map(|q| if cpu.contains_y(q) { '1' } else { '0' })
            .collect::<String>(),
        (0..3)
            .map(|q| if cpu.contains_z(q) { '1' } else { '0' })
            .collect::<String>(),
    );
    let qubits = vec![0usize, 1, 2];

    for (i, op) in ops.iter().enumerate() {
        apply_gpu(&mut gpu, *op);
        apply_cpu(&mut cpu, *op);
        let zf = gpu.measure_z_flips(&qubits);
        let xf = gpu.measure_x_flips(&qubits);
        eprintln!(
            "After [{i}] {op:?}: GPU z={}  x={}  | CPU x={}  y={}  z={}",
            zf[0]
                .iter()
                .map(|b| if *b { '1' } else { '0' })
                .collect::<String>(),
            xf[0]
                .iter()
                .map(|b| if *b { '1' } else { '0' })
                .collect::<String>(),
            (0..3)
                .map(|q| if cpu.contains_x(q) { '1' } else { '0' })
                .collect::<String>(),
            (0..3)
                .map(|q| if cpu.contains_y(q) { '1' } else { '0' })
                .collect::<String>(),
            (0..3)
                .map(|q| if cpu.contains_z(q) { '1' } else { '0' })
                .collect::<String>(),
        );
    }
}

/// Shrink a failing (seed, n, gates) down to the minimum prefix length that fails.
#[test]
fn shrink_failing_case() {
    // Known failing case: cross_check_x_faults seed=0 N=3 G=30 fault=x@0.
    for gates in 1..=30 {
        let Ok(mut gpu) = GpuPauliProp::with_seed(3, 1, 0) else {
            return;
        };
        let mut cpu = PauliProp::new();
        gpu.inject_x_fault(0);
        cpu.track_x(&[0]);

        let mut rng = StdRng::seed_from_u64(0);
        let ops: Vec<Op> = (0..30).map(|_| gen_op(&mut rng, 3)).collect();
        for &op in ops.iter().take(gates) {
            apply_gpu(&mut gpu, op);
            apply_cpu(&mut cpu, op);
        }

        let qubits: Vec<usize> = (0..3).collect();
        let zf = gpu.measure_z_flips(&qubits);
        let xf = gpu.measure_x_flips(&qubits);

        for q in 0..3 {
            let cpu_zf = cpu.contains_x(q) || cpu.contains_y(q);
            let cpu_xf = cpu.contains_z(q) || cpu.contains_y(q);
            if zf[0][q] != cpu_zf || xf[0][q] != cpu_xf {
                eprintln!("MISMATCH at gates={gates}, q={q}:");
                eprintln!("  ops so far:");
                for (i, &op) in ops.iter().take(gates).enumerate() {
                    eprintln!("    [{i}] {op:?}");
                }
                eprintln!(
                    "  gpu z_flip[{q}]={} x_flip[{q}]={} cpu: x={} y={} z={}",
                    zf[0][q],
                    xf[0][q],
                    cpu.contains_x(q),
                    cpu.contains_y(q),
                    cpu.contains_z(q)
                );
                panic!("divergence at gate {gates}");
            }
        }
    }
}

#[test]
fn simple_cx_check() {
    // X(0) then CX(0,1): X_0 -> X_0 X_1. So Pauli = XX.
    let Ok(mut gpu) = GpuPauliProp::with_seed(2, 1, 0) else {
        return;
    };
    let mut cpu = PauliProp::new();
    gpu.inject_x_fault(0);
    cpu.track_x(&[0]);
    gpu.cx(&[(0, 1)]);
    cpu.cx(&[(QubitId(0), QubitId(1))]);
    let zf = gpu.measure_z_flips(&[0, 1]);
    let xf = gpu.measure_x_flips(&[0, 1]);
    eprintln!(
        "After X(0) CX(0,1): gpu z_flip={:?} x_flip={:?} ; cpu x(0)={} x(1)={} z(0)={} z(1)={}",
        zf[0],
        xf[0],
        cpu.contains_x(0),
        cpu.contains_x(1),
        cpu.contains_z(0),
        cpu.contains_z(1)
    );
    // Expected XX: z_flip = [true, true], x_flip = [false, false]
    for q in 0..2 {
        assert_eq!(
            zf[0][q],
            cpu.contains_x(q) || cpu.contains_y(q),
            "z_flip q={q}"
        );
        assert_eq!(
            xf[0][q],
            cpu.contains_z(q) || cpu.contains_y(q),
            "x_flip q={q}"
        );
    }
}

#[test]
fn simple_cz_check() {
    // Z(0) then CZ(0,1): Z_0 commutes with CZ; Pauli stays Z_0.
    let Ok(mut gpu) = GpuPauliProp::with_seed(2, 1, 0) else {
        return;
    };
    let mut cpu = PauliProp::new();
    gpu.inject_z_fault(0);
    cpu.track_z(&[0]);
    gpu.cz(&[(0, 1)]);
    cpu.cz(&[(QubitId(0), QubitId(1))]);
    let zf = gpu.measure_z_flips(&[0, 1]);
    let xf = gpu.measure_x_flips(&[0, 1]);
    eprintln!(
        "After Z(0) CZ(0,1): gpu z_flip={:?} x_flip={:?} ; cpu x(0)={} x(1)={} z(0)={} z(1)={}",
        zf[0],
        xf[0],
        cpu.contains_x(0),
        cpu.contains_x(1),
        cpu.contains_z(0),
        cpu.contains_z(1)
    );
    for q in 0..2 {
        assert_eq!(
            zf[0][q],
            cpu.contains_x(q) || cpu.contains_y(q),
            "z_flip q={q}"
        );
        assert_eq!(
            xf[0][q],
            cpu.contains_z(q) || cpu.contains_y(q),
            "x_flip q={q}"
        );
    }
}

#[test]
fn simple_x_cz_check() {
    // X(0) then CZ(0,1): X_0 -> X_0 Z_1. So Pauli = X_0 Z_1.
    let Ok(mut gpu) = GpuPauliProp::with_seed(2, 1, 0) else {
        return;
    };
    let mut cpu = PauliProp::new();
    gpu.inject_x_fault(0);
    cpu.track_x(&[0]);
    gpu.cz(&[(0, 1)]);
    cpu.cz(&[(QubitId(0), QubitId(1))]);
    let zf = gpu.measure_z_flips(&[0, 1]);
    let xf = gpu.measure_x_flips(&[0, 1]);
    eprintln!(
        "After X(0) CZ(0,1): gpu z_flip={:?} x_flip={:?} ; cpu x(0)={} x(1)={} z(0)={} z(1)={}",
        zf[0],
        xf[0],
        cpu.contains_x(0),
        cpu.contains_x(1),
        cpu.contains_z(0),
        cpu.contains_z(1)
    );
    // Expected X_0 Z_1: z_flip = [true, false], x_flip = [false, true]
    for q in 0..2 {
        assert_eq!(
            zf[0][q],
            cpu.contains_x(q) || cpu.contains_y(q),
            "z_flip q={q}"
        );
        assert_eq!(
            xf[0][q],
            cpu.contains_z(q) || cpu.contains_y(q),
            "x_flip q={q}"
        );
    }
}

#[test]
fn minimal_single_gate_checks() {
    // X fault on q=0 followed by H(0): X -> Z, so Z-flip(0) should be FALSE and X-flip(0) TRUE.
    let Ok(mut gpu) = GpuPauliProp::with_seed(2, 1, 0) else {
        return;
    };
    let mut cpu = PauliProp::new();
    gpu.inject_x_fault(0);
    cpu.track_x(&[0]);
    gpu.h(&[0]);
    cpu.h(&[QubitId(0)]);
    let zf = gpu.measure_z_flips(&[0, 1]);
    let xf = gpu.measure_x_flips(&[0, 1]);
    eprintln!(
        "After X(0) H(0): gpu z_flip={:?} x_flip={:?} ; cpu contains_x(0)={} z(0)={}",
        zf[0],
        xf[0],
        cpu.contains_x(0),
        cpu.contains_z(0)
    );
    // Expected: Pauli is Z on qubit 0. GPU z_flip[0] = false, x_flip[0] = true. CPU contains_z(0) = true.
    assert_eq!(zf[0][0], cpu.contains_x(0) || cpu.contains_y(0), "z_flip 0");
    assert_eq!(xf[0][0], cpu.contains_z(0) || cpu.contains_y(0), "x_flip 0");
}

#[test]
fn cross_check_x_faults() {
    for seed in 0u64..10 {
        for n in [3usize, 5, 8] {
            for &fq in &[0usize, 1] {
                if fq < n {
                    run_cross_check(seed, n, 30, fq, "x");
                }
            }
        }
    }
}

#[test]
fn cross_check_z_faults() {
    for seed in 100u64..110 {
        for n in [3usize, 5, 8] {
            for &fq in &[0usize, 1] {
                if fq < n {
                    run_cross_check(seed, n, 30, fq, "z");
                }
            }
        }
    }
}

#[test]
fn cross_check_y_faults() {
    for seed in 200u64..210 {
        for n in [3usize, 5, 8] {
            for &fq in &[0usize, 1] {
                if fq < n {
                    run_cross_check(seed, n, 30, fq, "y");
                }
            }
        }
    }
}

#[test]
fn cross_check_longer_circuits() {
    // Stress test with longer circuits and more qubits.
    for seed in 300u64..305 {
        for n in [10usize, 16] {
            for &fq in &[0usize, 3] {
                if fq < n {
                    run_cross_check(seed, n, 100, fq, "x");
                    run_cross_check(seed, n, 100, fq, "z");
                }
            }
        }
    }
}
