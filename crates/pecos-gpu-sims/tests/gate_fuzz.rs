// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
// except in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0

//! Randomized correctness fuzz: random circuits of mixed 1q/2q Clifford and
//! rotation gates on random qubit pairs with random angles, cross-checked
//! against the CPU `StateVecSoA` reference. Fills coverage gaps that the
//! single-gate audit doesn't: gate queue fusion/reorder interactions,
//! different (control, target) pairs, theta edge cases.

use pecos_core::{Angle64, QubitId};
use pecos_gpu_sims::{GpuStateVec32, GpuStateVec64};
use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable, StateVecSoA};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

const TOL_F32: f64 = 5e-3;
const TOL_F64: f64 = 5e-5;

// --- RNG-driven gate emission applied to a generic simulator ---

#[derive(Clone, Copy)]
enum Op {
    H(usize),
    X(usize),
    Y(usize),
    Z(usize),
    S(usize),
    Sdg(usize),
    Sx(usize),
    Sxdg(usize),
    Sy(usize),
    Sydg(usize),
    T(usize),
    Tdg(usize),
    Rx(usize, f64),
    Ry(usize, f64),
    Rz(usize, f64),
    Cx(usize, usize),
    Cy(usize, usize),
    Cz(usize, usize),
    Swap(usize, usize),
    Szz(usize, usize),
    Szzdg(usize, usize),
    Sxx(usize, usize),
    Sxxdg(usize, usize),
    Syy(usize, usize),
    Syydg(usize, usize),
    Rxx(usize, usize, f64),
    Ryy(usize, usize, f64),
    Rzz(usize, usize, f64),
}

fn pick_two(rng: &mut StdRng, n: usize) -> (usize, usize) {
    assert!(n >= 2);
    let a = rng.random_range(0..n);
    let mut b = rng.random_range(0..n);
    while b == a {
        b = rng.random_range(0..n);
    }
    (a, b)
}

fn gen_op(rng: &mut StdRng, n: usize) -> Op {
    let kind = rng.random_range(0u32..27);
    match kind {
        0 => Op::H(rng.random_range(0..n)),
        1 => Op::X(rng.random_range(0..n)),
        2 => Op::Y(rng.random_range(0..n)),
        3 => Op::Z(rng.random_range(0..n)),
        4 => Op::S(rng.random_range(0..n)),
        5 => Op::Sdg(rng.random_range(0..n)),
        6 => Op::Sx(rng.random_range(0..n)),
        7 => Op::Sxdg(rng.random_range(0..n)),
        8 => Op::Sy(rng.random_range(0..n)),
        9 => Op::Sydg(rng.random_range(0..n)),
        10 => Op::T(rng.random_range(0..n)),
        11 => Op::Tdg(rng.random_range(0..n)),
        12 => Op::Rx(rng.random_range(0..n), rng.random_range(-3.3..3.3)),
        13 => Op::Ry(rng.random_range(0..n), rng.random_range(-3.3..3.3)),
        14 => Op::Rz(rng.random_range(0..n), rng.random_range(-3.3..3.3)),
        15 => {
            let (a, b) = pick_two(rng, n);
            Op::Cx(a, b)
        }
        16 => {
            let (a, b) = pick_two(rng, n);
            Op::Cy(a, b)
        }
        17 => {
            let (a, b) = pick_two(rng, n);
            Op::Cz(a, b)
        }
        18 => {
            let (a, b) = pick_two(rng, n);
            Op::Swap(a, b)
        }
        19 => {
            let (a, b) = pick_two(rng, n);
            Op::Szz(a, b)
        }
        20 => {
            let (a, b) = pick_two(rng, n);
            Op::Szzdg(a, b)
        }
        21 => {
            let (a, b) = pick_two(rng, n);
            Op::Sxx(a, b)
        }
        22 => {
            let (a, b) = pick_two(rng, n);
            Op::Sxxdg(a, b)
        }
        23 => {
            let (a, b) = pick_two(rng, n);
            Op::Syy(a, b)
        }
        24 => {
            let (a, b) = pick_two(rng, n);
            Op::Syydg(a, b)
        }
        25 => {
            let (a, b) = pick_two(rng, n);
            Op::Rxx(a, b, rng.random_range(-3.3..3.3))
        }
        26 => {
            let (a, b) = pick_two(rng, n);
            Op::Ryy(a, b, rng.random_range(-3.3..3.3))
        }
        _ => {
            let (a, b) = pick_two(rng, n);
            Op::Rzz(a, b, rng.random_range(-3.3..3.3))
        }
    }
}

fn apply_cpu(sim: &mut StateVecSoA, op: Op) {
    match op {
        Op::H(q) => {
            sim.h(&[QubitId(q)]);
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
        Op::S(q) => {
            sim.sz(&[QubitId(q)]);
        }
        Op::Sdg(q) => {
            sim.szdg(&[QubitId(q)]);
        }
        Op::Sx(q) => {
            sim.sx(&[QubitId(q)]);
        }
        Op::Sxdg(q) => {
            sim.sxdg(&[QubitId(q)]);
        }
        Op::Sy(q) => {
            sim.sy(&[QubitId(q)]);
        }
        Op::Sydg(q) => {
            sim.sydg(&[QubitId(q)]);
        }
        Op::T(q) => {
            sim.t(&[QubitId(q)]);
        }
        Op::Tdg(q) => {
            sim.tdg(&[QubitId(q)]);
        }
        Op::Rx(q, t) => {
            sim.rx(Angle64::from_radians(t), &[QubitId(q)]);
        }
        Op::Ry(q, t) => {
            sim.ry(Angle64::from_radians(t), &[QubitId(q)]);
        }
        Op::Rz(q, t) => {
            sim.rz(Angle64::from_radians(t), &[QubitId(q)]);
        }
        Op::Cx(a, b) => {
            sim.cx(&[(QubitId(a), QubitId(b))]);
        }
        Op::Cy(a, b) => {
            sim.cy(&[(QubitId(a), QubitId(b))]);
        }
        Op::Cz(a, b) => {
            sim.cz(&[(QubitId(a), QubitId(b))]);
        }
        Op::Swap(a, b) => {
            sim.swap(&[(QubitId(a), QubitId(b))]);
        }
        Op::Szz(a, b) => {
            sim.szz(&[(QubitId(a), QubitId(b))]);
        }
        Op::Szzdg(a, b) => {
            sim.szzdg(&[(QubitId(a), QubitId(b))]);
        }
        Op::Sxx(a, b) => {
            sim.sxx(&[(QubitId(a), QubitId(b))]);
        }
        Op::Sxxdg(a, b) => {
            sim.sxxdg(&[(QubitId(a), QubitId(b))]);
        }
        Op::Syy(a, b) => {
            sim.syy(&[(QubitId(a), QubitId(b))]);
        }
        Op::Syydg(a, b) => {
            sim.syydg(&[(QubitId(a), QubitId(b))]);
        }
        Op::Rxx(a, b, t) => {
            sim.rxx(Angle64::from_radians(t), &[(QubitId(a), QubitId(b))]);
        }
        Op::Ryy(a, b, t) => {
            sim.ryy(Angle64::from_radians(t), &[(QubitId(a), QubitId(b))]);
        }
        Op::Rzz(a, b, t) => {
            sim.rzz(Angle64::from_radians(t), &[(QubitId(a), QubitId(b))]);
        }
    }
}

macro_rules! apply_gpu_impl {
    ($fn_name:ident, $sv:ty) => {
        fn $fn_name(sim: &mut $sv, op: Op) {
            match op {
                Op::H(q) => {
                    sim.h(&[QubitId(q)]);
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
                Op::S(q) => {
                    sim.sz(&[QubitId(q)]);
                }
                Op::Sdg(q) => {
                    sim.szdg(&[QubitId(q)]);
                }
                Op::Sx(q) => {
                    sim.sx(&[QubitId(q)]);
                }
                Op::Sxdg(q) => {
                    sim.sxdg(&[QubitId(q)]);
                }
                Op::Sy(q) => {
                    sim.sy(&[QubitId(q)]);
                }
                Op::Sydg(q) => {
                    sim.sydg(&[QubitId(q)]);
                }
                Op::T(q) => {
                    sim.t(&[QubitId(q)]);
                }
                Op::Tdg(q) => {
                    sim.tdg(&[QubitId(q)]);
                }
                Op::Rx(q, t) => {
                    sim.rx(Angle64::from_radians(t), &[QubitId(q)]);
                }
                Op::Ry(q, t) => {
                    sim.ry(Angle64::from_radians(t), &[QubitId(q)]);
                }
                Op::Rz(q, t) => {
                    sim.rz(Angle64::from_radians(t), &[QubitId(q)]);
                }
                Op::Cx(a, b) => {
                    sim.cx(&[(QubitId(a), QubitId(b))]);
                }
                Op::Cy(a, b) => {
                    sim.cy(&[(QubitId(a), QubitId(b))]);
                }
                Op::Cz(a, b) => {
                    sim.cz(&[(QubitId(a), QubitId(b))]);
                }
                Op::Swap(a, b) => {
                    sim.swap(&[(QubitId(a), QubitId(b))]);
                }
                Op::Szz(a, b) => {
                    sim.szz(&[(QubitId(a), QubitId(b))]);
                }
                Op::Szzdg(a, b) => {
                    sim.szzdg(&[(QubitId(a), QubitId(b))]);
                }
                Op::Sxx(a, b) => {
                    sim.sxx(&[(QubitId(a), QubitId(b))]);
                }
                Op::Sxxdg(a, b) => {
                    sim.sxxdg(&[(QubitId(a), QubitId(b))]);
                }
                Op::Syy(a, b) => {
                    sim.syy(&[(QubitId(a), QubitId(b))]);
                }
                Op::Syydg(a, b) => {
                    sim.syydg(&[(QubitId(a), QubitId(b))]);
                }
                Op::Rxx(a, b, t) => {
                    sim.rxx(Angle64::from_radians(t), &[(QubitId(a), QubitId(b))]);
                }
                Op::Ryy(a, b, t) => {
                    sim.ryy(Angle64::from_radians(t), &[(QubitId(a), QubitId(b))]);
                }
                Op::Rzz(a, b, t) => {
                    sim.rzz(Angle64::from_radians(t), &[(QubitId(a), QubitId(b))]);
                }
            }
        }
    };
}

apply_gpu_impl!(apply_gpu32, GpuStateVec32);
apply_gpu_impl!(apply_gpu64, GpuStateVec64);

fn cpu_state(sim: &mut StateVecSoA) -> Vec<[f64; 2]> {
    sim.state().into_iter().map(|c| [c.re, c.im]).collect()
}

fn max_diff(gpu: &[[f64; 2]], cpu: &[[f64; 2]]) -> f64 {
    gpu.iter()
        .zip(cpu.iter())
        .map(|([gr, gi], [cr, ci])| {
            let dr = gr - cr;
            let di = gi - ci;
            (dr * dr + di * di).sqrt()
        })
        .fold(0.0, f64::max)
}

// --- Fuzz harness ---

fn fuzz_one_seed(seed: u64, n: usize, gates: usize) -> Result<(), String> {
    let mut rng = StdRng::seed_from_u64(seed);
    let ops: Vec<Op> = (0..gates).map(|_| gen_op(&mut rng, n)).collect();

    // CPU reference
    let mut cpu = StateVecSoA::new(n);
    for &op in &ops {
        apply_cpu(&mut cpu, op);
    }
    let cpu_s = cpu_state(&mut cpu);

    // f32
    if let Ok(mut g32) = GpuStateVec32::new(u32::try_from(n).expect("test N fits in u32")) {
        for &op in &ops {
            apply_gpu32(&mut g32, op);
        }
        let s: Vec<[f64; 2]> = g32
            .state()
            .into_iter()
            .map(|[re, im]| [f64::from(re), f64::from(im)])
            .collect();
        let d = max_diff(&s, &cpu_s);
        if d > TOL_F32 {
            return Err(format!("seed={seed} N={n} G={gates} f32 diff={d:.3e}"));
        }
    }

    // f64
    if let Ok(mut g64) = GpuStateVec64::new(u32::try_from(n).expect("test N fits in u32")) {
        for &op in &ops {
            apply_gpu64(&mut g64, op);
        }
        let s = g64.state();
        let d = max_diff(&s, &cpu_s);
        if d > TOL_F64 {
            return Err(format!("seed={seed} N={n} G={gates} f64 diff={d:.3e}"));
        }
    }

    Ok(())
}

#[test]
fn fuzz_persistent_path() {
    // N=4 is below persistent_max_qubits on any realistic GPU.
    let mut fails: Vec<String> = Vec::new();
    for seed in 0..20u64 {
        if let Err(e) = fuzz_one_seed(seed, 4, 40) {
            fails.push(e);
        }
    }
    assert!(
        fails.is_empty(),
        "{} persistent-path fuzz failures:\n  {}",
        fails.len(),
        fails.join("\n  ")
    );
}

#[test]
fn fuzz_dispatched_path() {
    // N=14 is well above typical persistent_max_qubits (~12) -- forces dispatched path.
    let mut fails: Vec<String> = Vec::new();
    for seed in 100..115u64 {
        if let Err(e) = fuzz_one_seed(seed, 14, 30) {
            fails.push(e);
        }
    }
    assert!(
        fails.is_empty(),
        "{} dispatched-path fuzz failures:\n  {}",
        fails.len(),
        fails.join("\n  ")
    );
}

#[test]
fn fuzz_small_n_stress() {
    // Stress 2q-gate qubit-pair mask handling with N=2 and N=3 where
    // off-by-one / low-stride scalar-vs-SIMD selection would show up.
    let mut fails: Vec<String> = Vec::new();
    for n in [2usize, 3] {
        for seed in 200..210u64 {
            if let Err(e) = fuzz_one_seed(seed, n, 30) {
                fails.push(e);
            }
        }
    }
    assert!(
        fails.is_empty(),
        "{} small-N fuzz failures:\n  {}",
        fails.len(),
        fails.join("\n  ")
    );
}

// --- Angle edge cases ---

#[test]
fn angle_edge_cases() {
    use std::f64::consts::PI;
    // Angles that commonly hit edge behavior in trig implementations.
    let angles = [0.0, PI, -PI, PI / 2.0, -PI / 2.0, 2.0 * PI, 1e-10, -1e-10];
    let n: usize = 5;
    let qubits: Vec<QubitId> = (0..n).map(QubitId).collect();

    let gate_kind = ["rx", "ry", "rz", "rxx", "ryy", "rzz"];
    for (gi, gate) in gate_kind.iter().enumerate() {
        for (ai, &theta) in angles.iter().enumerate() {
            let t = Angle64::from_radians(theta);
            let mut cpu = StateVecSoA::new(n);
            cpu.h(&qubits);

            let apply =
                |is_cpu: bool, c: &mut StateVecSoA, gs: &mut Option<GpuStateVec64>| match *gate {
                    "rx" => {
                        if is_cpu {
                            c.rx(t, &[QubitId(0)]);
                        } else if let Some(g) = gs {
                            g.rx(t, &[QubitId(0)]);
                        }
                    }
                    "ry" => {
                        if is_cpu {
                            c.ry(t, &[QubitId(0)]);
                        } else if let Some(g) = gs {
                            g.ry(t, &[QubitId(0)]);
                        }
                    }
                    "rz" => {
                        if is_cpu {
                            c.rz(t, &[QubitId(0)]);
                        } else if let Some(g) = gs {
                            g.rz(t, &[QubitId(0)]);
                        }
                    }
                    "rxx" => {
                        if is_cpu {
                            c.rxx(t, &[(QubitId(0), QubitId(1))]);
                        } else if let Some(g) = gs {
                            g.rxx(t, &[(QubitId(0), QubitId(1))]);
                        }
                    }
                    "ryy" => {
                        if is_cpu {
                            c.ryy(t, &[(QubitId(0), QubitId(1))]);
                        } else if let Some(g) = gs {
                            g.ryy(t, &[(QubitId(0), QubitId(1))]);
                        }
                    }
                    "rzz" => {
                        if is_cpu {
                            c.rzz(t, &[(QubitId(0), QubitId(1))]);
                        } else if let Some(g) = gs {
                            g.rzz(t, &[(QubitId(0), QubitId(1))]);
                        }
                    }
                    _ => {}
                };

            let mut none: Option<GpuStateVec64> = None;
            apply(true, &mut cpu, &mut none);
            let cpu_s = cpu_state(&mut cpu);

            let mut g = GpuStateVec64::new(u32::try_from(n).expect("test N fits in u32")).ok();
            if let Some(g) = g.as_mut() {
                g.h(&qubits);
            }
            apply(false, &mut cpu /* unused */, &mut g);
            if let Some(g) = g.as_mut() {
                let gs = g.state();
                let d = max_diff(&gs, &cpu_s);
                assert!(
                    d < 1e-4,
                    "angle {theta} gate {gate} ({gi},{ai}): f64 diff={d:.3e}"
                );
            }
        }
    }
}

// --- Measurement determinism ---

#[test]
fn measurement_deterministic_on_basis_states() {
    use pecos_simulators::CliffordGateable;
    let n: usize = 4;
    // Prepare |0101> and check mz outcomes match.
    let prep_bits = [false, true, false, true];

    let Ok(mut g32) = GpuStateVec32::new(u32::try_from(n).expect("test N fits in u32")) else {
        return;
    };
    for (q, &b) in prep_bits.iter().enumerate() {
        if b {
            g32.x(&[QubitId(q)]);
        }
    }
    let results = g32.mz(&[QubitId(0), QubitId(1), QubitId(2), QubitId(3)]);
    for (q, r) in results.iter().enumerate() {
        assert_eq!(
            r.outcome,
            prep_bits[q],
            "mz on prepared |{}> qubit {q} got {}",
            to_bits(&prep_bits),
            r.outcome
        );
        assert!(
            r.is_deterministic,
            "|basis state> measurement must be deterministic"
        );
    }

    let Ok(mut g64) = GpuStateVec64::new(u32::try_from(n).expect("test N fits in u32")) else {
        return;
    };
    for (q, &b) in prep_bits.iter().enumerate() {
        if b {
            g64.x(&[QubitId(q)]);
        }
    }
    let results = g64.mz(&[QubitId(0), QubitId(1), QubitId(2), QubitId(3)]);
    for (q, r) in results.iter().enumerate() {
        assert_eq!(r.outcome, prep_bits[q], "f64 qubit {q}");
        assert!(r.is_deterministic);
    }
}

fn to_bits(bits: &[bool]) -> String {
    bits.iter()
        .rev()
        .map(|&b| if b { '1' } else { '0' })
        .collect()
}
