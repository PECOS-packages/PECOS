// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0.

//! Release-mode timing and outcome counts for the tricycle-CCZ benchmark circuit.

use pecos_phase_poly::PhasePoly;
use pecos_phase_poly::runner::Program;
use pecos_phase_poly::tricycle_ccz::magic_state_circuit;
use pecos_phase_poly::tricycle_ccz_noisy::{CczModel, MetaChecks, NoisyCcz};
use pecos_qec::BivariateTricycleCode;
use pecos_random::PecosRng;
use std::error::Error;
use std::time::Instant;

const USAGE: &str = "usage: tricycle_ccz <27|81> <p> <shots> <seed> <output.json> [exact|surrogate] [complete|paper]";

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(5..=7).contains(&args.len()) {
        return Err(USAGE.into());
    }
    // The [[27,3,3]] and [[81,3,5]] codes of arXiv:2510.07269.
    let code = match args[0].as_str() {
        "27" => BivariateTricycleCode::new(
            3,
            3,
            &[(2, 1), (2, 2)],
            &[(0, 0), (1, 2)],
            &[(1, 0), (2, 1)],
        )?,
        "81" => BivariateTricycleCode::new(
            3,
            9,
            &[(1, 3), (2, 1)],
            &[(0, 0), (1, 8)],
            &[(2, 4), (2, 6)],
        )?,
        _ => return Err(USAGE.into()),
    };
    let p: f64 = args[1].parse()?;
    let shots: u32 = args[2].parse()?;
    let seed: u64 = args[3].parse()?;

    if let Some(mode) = args.get(5) {
        let model = match mode.as_str() {
            "exact" => CczModel::Exact,
            "surrogate" => CczModel::PaperSurrogate,
            _ => return Err(USAGE.into()),
        };
        let meta_checks = match args.get(6).map_or("complete", String::as_str) {
            "complete" => MetaChecks::Complete,
            "paper" => MetaChecks::Paper,
            _ => return Err(USAGE.into()),
        };
        let summary = noisy_summary(&code, p, shots, seed, model, meta_checks)?;
        std::fs::write(
            &args[4],
            format!("{}\n", serde_json::to_string_pretty(&summary)?),
        )?;
        println!("{summary}");
        return Ok(());
    }

    let started = Instant::now();
    let circuit = magic_state_circuit(&code, p, true)?;
    let build_s = started.elapsed().as_secs_f64();
    let started = Instant::now();
    let program = Program::compile(&circuit)?;
    let compile_s = started.elapsed().as_secs_f64();

    let mut seeds = PecosRng::seed_from_u64(seed);
    let mut rng = PecosRng::seed_from_u64(seeds.next_u64());
    let mut sim = PhasePoly::with_seed(program.num_qubits(), seeds.next_u64());
    let (mut accepted, mut target_pass) = (0_u32, 0_u32);
    let started = Instant::now();
    for _ in 0..shots {
        let shot = program.run_shot(&mut sim, &mut rng)?;
        if shot.detectors.iter().all(|bit| !bit) {
            accepted += 1;
            target_pass += u32::from(shot.observables.iter().all(|bit| !bit));
        }
    }
    let sample_s = started.elapsed().as_secs_f64();

    let summary = serde_json::json!({
        "n": code.num_qubits(), "k": code.num_logical_qubits(), "p": p,
        "shots": shots, "seed": seed, "accepted": accepted, "target_pass": target_pass,
        "num_qubits": program.num_qubits(), "detectors": program.detectors().len(),
        "observables": program.observables().len(),
        "build_s": build_s, "compile_s": compile_s, "sample_s": sample_s,
        "sample_s_per_shot": sample_s / f64::from(shots.max(1)),
    });
    std::fs::write(
        &args[4],
        format!("{}\n", serde_json::to_string_pretty(&summary)?),
    )?;
    println!("{summary}");
    Ok(())
}

fn noisy_summary(
    code: &BivariateTricycleCode,
    p: f64,
    shots: u32,
    seed: u64,
    model: CczModel,
    meta_checks: MetaChecks,
) -> Result<serde_json::Value, Box<dyn Error>> {
    let protocol = NoisyCcz::new(code, p, model, meta_checks)?;
    let (build_time, compile_time) = protocol.construction_times();
    let started = Instant::now();
    let counts = protocol.run_counts(shots as usize, seed)?;
    let sample_s = started.elapsed().as_secs_f64();
    Ok(serde_json::json!({
        "n": code.num_qubits(), "k": code.num_logical_qubits(), "p": p,
        "model": match model { CczModel::Exact => "exact", CczModel::PaperSurrogate => "surrogate" },
        "shots": shots, "seed": seed, "attempted": counts.attempted,
        "meta_checks": match meta_checks { MetaChecks::Complete => "complete", MetaChecks::Paper => "paper" },
        "meta_rows": protocol.meta_rows(),
        "repair_failures": counts.repair_failures, "accepted": counts.accepted,
        "logical_failures": counts.logical_failures,
        "accepted_with_residual_x": counts.accepted_with_residual_x,
        "accepted_z_clean": counts.accepted_z_clean,
        "logical_failures_z_clean": counts.logical_failures_z_clean,
        "x_detector_rejections": counts.attempted - counts.repair_failures - counts.accepted,
        "target_pass": counts.accepted - counts.logical_failures,
        "num_qubits": protocol.prefix().num_qubits(),
        "detectors": protocol.suffix().detectors().len(),
        "observables": protocol.suffix().observables().len(),
        "build_s": build_time.as_secs_f64(), "compile_s": compile_time.as_secs_f64(), "sample_s": sample_s,
        "sample_s_per_shot": sample_s / f64::from(shots.max(1)),
    }))
}
