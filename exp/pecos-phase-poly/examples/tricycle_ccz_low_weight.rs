// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0.

//! Low-weight logical-set reference for arXiv:2510.07269, Table II.
//! Both codes take an explicit hook-safe or coloring schedule; C27 also accepts paper.

use pecos_phase_poly::tricycle_ccz_noisy::{ExtractionSchedule, PAPER_C27_SCHEDULE};
use pecos_phase_poly::tricycle_ccz_reference::{low_weight_estimate, paper_catalog};
use pecos_qec::BivariateTricycleCode;
use std::{error::Error, time::Instant};

const USAGE: &str = "usage: tricycle_ccz_low_weight 27 <hook-safe|coloring|paper> <p> <output.json>\n       tricycle_ccz_low_weight 81 <hook-safe|coloring> <p> <output.json>";

fn main() -> Result<(), Box<dyn Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 4 {
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
    let result = low_weight_estimate(&catalog, 5)?;
    let terms = result
        .terms
        .iter()
        .map(|term| {
            serde_json::json!({
                "weight": term.weight,
                "number_of_sets": term.number_of_sets,
                "coefficient_sum": term.coefficient_sum,
            })
        })
        .collect::<Vec<_>>();
    let output = serde_json::json!({
        "n": code.num_qubits(), "p": p, "schedule": mode, "distance": result.distance,
        "terms": terms, "total": result.total,
        "build_s": build_s, "wall_s": started.elapsed().as_secs_f64(),
    });
    std::fs::write(
        &args[3],
        format!("{}\n", serde_json::to_string_pretty(&output)?),
    )?;
    println!("{output}");
    Ok(())
}
