// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0.

//! Fault-count reference for arXiv:2510.07269, Table II.
//! Both codes take an explicit hook-safe or coloring schedule; C27 also accepts paper.

use pecos_phase_poly::tricycle_ccz_noisy::{ExtractionSchedule, PAPER_C27_SCHEDULE};
use pecos_phase_poly::tricycle_ccz_reference::{estimate, paper_catalog};
use pecos_qec::BivariateTricycleCode;
use std::{error::Error, time::Instant};

const USAGE: &str = "usage: tricycle_ccz_reference 27 <hook-safe|coloring|paper> <p> <k_exact> <k_max> <samples_per_stratum> <seed> <output.json>\n       tricycle_ccz_reference 81 <hook-safe|coloring> <p> <k_exact> <k_max> <samples_per_stratum> <seed> <output.json>";

fn main() -> Result<(), Box<dyn Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 8 {
        return Err(USAGE.into());
    }
    let mode = args[1].as_str();
    let started = Instant::now();
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
    let p = args[2].parse()?;
    let k_exact = args[3].parse()?;
    let k_max = args[4].parse()?;
    let samples = args[5].parse()?;
    let seed = args[6].parse()?;
    let schedule = match mode {
        "hook-safe" => ExtractionSchedule::hook_safe(&code)?,
        "coloring" => ExtractionSchedule::edge_coloring(&code)?,
        "paper" if args[0] == "27" => {
            ExtractionSchedule::from_layers(&code, PAPER_C27_SCHEDULE.map(Vec::from).into())?
        }
        _ => return Err(USAGE.into()),
    };
    let catalog = paper_catalog(&code, &schedule, p)?;
    let build_s = started.elapsed().as_secs_f64();
    let sampling = Instant::now();
    let result = estimate(&catalog, k_exact, k_max, samples, seed)?;
    let estimate_s = sampling.elapsed().as_secs_f64();
    let e = result.estimate;
    let strata = result.strata.iter().map(|s| serde_json::json!({
        "k": s.k, "exact": s.exact, "configurations": s.configurations, "mass": s.mass,
        "success_probability": s.success_probability, "failure_probability": s.failure_probability,
        "ler_contribution": s.failure_probability / e.survival_probability,
    })).collect::<Vec<_>>();
    let output = serde_json::json!({
        "n": code.num_qubits(), "schedule": mode, "p": p, "k_exact": k_exact, "k_max": k_max,
        "samples_per_stratum": samples, "seed": seed,
        "success_probability": e.survival_probability, "failure_probability": e.failure_probability,
        "ler": e.failure_given_survival, "success_standard_error": e.survival_standard_error,
        "failure_standard_error": e.failure_standard_error, "ler_standard_error": e.ratio_standard_error,
        "unsampled_mass": e.unsampled_mass, "tail_bound": e.tail_bound, "strata": strata,
        "build_s": build_s, "estimate_s": estimate_s, "wall_s": started.elapsed().as_secs_f64(),
    });
    std::fs::write(
        &args[7],
        format!("{}\n", serde_json::to_string_pretty(&output)?),
    )?;
    println!("{output}");
    Ok(())
}
