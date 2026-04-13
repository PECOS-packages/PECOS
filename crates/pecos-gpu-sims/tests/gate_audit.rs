// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
// except in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0

//! Systematic correctness audit: every GPU 2-qubit gate, on every backend
//! (f32 / f64), on every dispatch path (persistent / dispatched), cross-checked
//! against the CPU `StateVecSoA` reference.
//!
//! Seed state is H^N|0> (uniform |+>^N), which populates every 2-qubit
//! subspace pair with a nonzero amplitude -- this exposes bugs where a gate
//! only updates half the basis pairs (e.g. the existing RXX/RYY
//! bit_a==bit_b-only bug).

use pecos_core::{Angle64, QubitId};
use pecos_gpu_sims::{GpuStateVec32, GpuStateVec64};
use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable, StateVecSoA};

const TOL_F32: f64 = 1e-3;
const TOL_F64: f64 = 1e-5;

// --- CPU reference ---

fn cpu_seed(n: usize) -> StateVecSoA {
    let mut sv = StateVecSoA::new(n);
    let qubits: Vec<QubitId> = (0..n).map(QubitId).collect();
    sv.h(&qubits);
    sv
}

fn cpu_state(sv: &mut StateVecSoA) -> Vec<[f64; 2]> {
    sv.state().into_iter().map(|c| [c.re, c.im]).collect()
}

// --- backend harness ---

fn run_f32<F>(n: u32, apply: F) -> Option<Vec<[f64; 2]>>
where
    F: Fn(&mut GpuStateVec32),
{
    let mut sv = GpuStateVec32::new(n).ok()?;
    let qubits: Vec<QubitId> = (0..usize::try_from(n).unwrap()).map(QubitId).collect();
    sv.h(&qubits);
    apply(&mut sv);
    Some(
        sv.state()
            .into_iter()
            .map(|[re, im]| [f64::from(re), f64::from(im)])
            .collect(),
    )
}

fn run_f64<F>(n: u32, apply: F) -> Option<Vec<[f64; 2]>>
where
    F: Fn(&mut GpuStateVec64),
{
    let mut sv = GpuStateVec64::new(n).ok()?;
    let qubits: Vec<QubitId> = (0..usize::try_from(n).unwrap()).map(QubitId).collect();
    sv.h(&qubits);
    apply(&mut sv);
    Some(sv.state())
}

fn diff(gpu: &[[f64; 2]], cpu: &[[f64; 2]]) -> f64 {
    gpu.iter()
        .zip(cpu.iter())
        .map(|([gr, gi], [cr, ci])| {
            let dr = gr - cr;
            let di = gi - ci;
            (dr * dr + di * di).sqrt()
        })
        .fold(0.0, f64::max)
}

// --- checks ---

struct Case {
    name: &'static str,
    apply_cpu: fn(&mut StateVecSoA),
    apply_f32: fn(&mut GpuStateVec32),
    apply_f64: fn(&mut GpuStateVec64),
}

fn theta() -> Angle64 {
    Angle64::from_radians(0.37)
}

macro_rules! case_1q {
    ($name:literal, $m:ident) => {
        Case {
            name: $name,
            apply_cpu: |sv| {
                sv.$m(&[QubitId(0)]);
            },
            apply_f32: |sv| {
                sv.$m(&[QubitId(0)]);
            },
            apply_f64: |sv| {
                sv.$m(&[QubitId(0)]);
            },
        }
    };
}

macro_rules! case_2q {
    ($name:literal, $m:ident) => {
        Case {
            name: $name,
            apply_cpu: |sv| {
                sv.$m(&[(QubitId(0), QubitId(1))]);
            },
            apply_f32: |sv| {
                sv.$m(&[(QubitId(0), QubitId(1))]);
            },
            apply_f64: |sv| {
                sv.$m(&[(QubitId(0), QubitId(1))]);
            },
        }
    };
}

macro_rules! case_2q_rot {
    ($name:literal, $m:ident) => {
        Case {
            name: $name,
            apply_cpu: |sv| {
                sv.$m(theta(), &[(QubitId(0), QubitId(1))]);
            },
            apply_f32: |sv| {
                sv.$m(theta(), &[(QubitId(0), QubitId(1))]);
            },
            apply_f64: |sv| {
                sv.$m(theta(), &[(QubitId(0), QubitId(1))]);
            },
        }
    };
}

fn all_cases() -> Vec<Case> {
    vec![
        // 1q for regression baseline
        case_1q!("h", h),
        case_1q!("x", x),
        case_1q!("y", y),
        case_1q!("z", z),
        case_1q!("sx", sx),
        case_1q!("sxdg", sxdg),
        case_1q!("sy", sy),
        case_1q!("sydg", sydg),
        case_1q!("sz", sz),
        case_1q!("szdg", szdg),
        // 2q Clifford
        case_2q!("cx", cx),
        case_2q!("cy", cy),
        case_2q!("cz", cz),
        case_2q!("swap", swap),
        case_2q!("szz", szz),
        case_2q!("szzdg", szzdg),
        case_2q!("sxx", sxx),
        case_2q!("sxxdg", sxxdg),
        case_2q!("syy", syy),
        case_2q!("syydg", syydg),
        // 2q rotations
        case_2q_rot!("rxx", rxx),
        case_2q_rot!("ryy", ryy),
        case_2q_rot!("rzz", rzz),
    ]
}

fn check_backends(label: &str, n: usize, cases: &[Case]) {
    // CPU reference
    for case in cases {
        let mut cpu = cpu_seed(n);
        (case.apply_cpu)(&mut cpu);
        let cpu_state = cpu_state(&mut cpu);

        // f32
        if let Some(gpu) = run_f32(u32::try_from(n).unwrap(), case.apply_f32) {
            let d = diff(&gpu, &cpu_state);
            if d > TOL_F32 {
                println!("FAIL [{label}] f32 {} (N={n}): max_diff={d:.3e}", case.name);
            } else {
                println!(" ok  [{label}] f32 {} (N={n}): max_diff={d:.3e}", case.name);
            }
        }

        // f64
        if let Some(gpu) = run_f64(u32::try_from(n).unwrap(), case.apply_f64) {
            let d = diff(&gpu, &cpu_state);
            if d > TOL_F64 {
                println!("FAIL [{label}] f64 {} (N={n}): max_diff={d:.3e}", case.name);
            } else {
                println!(" ok  [{label}] f64 {} (N={n}): max_diff={d:.3e}", case.name);
            }
        }
    }
}

// The persistent_max_qubits threshold on typical desktop GPUs is 10..12.
// N=4 forces persistent kernel. N=14 forces dispatched path.
#[test]
fn audit_persistent_path() {
    let cases = all_cases();
    check_backends("persistent", 4, &cases);
}

#[test]
fn audit_dispatched_path() {
    let cases = all_cases();
    check_backends("dispatched", 14, &cases);
}

/// Boundary qubit counts around the persistent/dispatched threshold.
/// `persistent_max_qubits` on RTX 4090 is ~12, so N=11 is still persistent,
/// N=13 is dispatched; N=12 is right at the edge.
#[test]
fn audit_persistent_dispatched_boundary() {
    let cases = all_cases();
    for n in [11usize, 12, 13] {
        check_backends("boundary", n, &cases);
    }
}

/// Summary assertion: every tested path in one test. Any shader bug surfaces
/// as a FAIL line -- run with `--nocapture` to see per-gate verdicts.
#[test]
fn audit_strict() {
    let cases = all_cases();
    let mut failures: Vec<String> = Vec::new();
    for (label, n) in [
        ("persistent", 4usize),
        ("boundary-under", 11usize),
        ("boundary-at", 12usize),
        ("boundary-over", 13usize),
        ("dispatched", 14usize),
    ] {
        for case in &cases {
            let mut cpu = cpu_seed(n);
            (case.apply_cpu)(&mut cpu);
            let cpu_state = cpu_state(&mut cpu);

            if let Some(gpu) = run_f32(u32::try_from(n).unwrap(), case.apply_f32) {
                let d = diff(&gpu, &cpu_state);
                if d > TOL_F32 {
                    failures.push(format!("{label}/f32/{}: diff={d:.3e}", case.name));
                }
            }
            if let Some(gpu) = run_f64(u32::try_from(n).unwrap(), case.apply_f64) {
                let d = diff(&gpu, &cpu_state);
                if d > TOL_F64 {
                    failures.push(format!("{label}/f64/{}: diff={d:.3e}", case.name));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "GPU shader correctness failures ({}):\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
}
