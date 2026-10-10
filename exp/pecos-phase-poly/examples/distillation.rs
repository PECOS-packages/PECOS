// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0.

//! Release-mode circuit timing and portable distillation artifacts.

use pecos_phase_poly::PhasePoly;
use pecos_phase_poly::distillation::{Op, TriorthogonalMatrix, run_shot_sampled, to_stim};
use pecos_random::PecosRng;
use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable, QuantumSimulator};
use pecos_stab_active::StabActive;
use std::error::Error;
use std::time::Instant;

const USAGE: &str = "usage:\n  distillation scaling <max-even-k> <shots> <p> <seed> [active-width-limit=16]\n  distillation emit <even-k|rm15> <p> <shots> <seed> <output-prefix>";

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("scaling") if (5..=6).contains(&args.len()) => scaling(&args),
        Some("emit") if args.len() == 6 => emit(&args),
        _ => Err(USAGE.into()),
    }
}

fn shots(text: &str) -> Result<u32, Box<dyn Error>> {
    let value = text.parse()?;
    if value == 0 {
        return Err("shots must be positive".into());
    }
    Ok(value)
}

fn scaling(args: &[String]) -> Result<(), Box<dyn Error>> {
    let max_k: usize = args[1].parse()?;
    let shots = shots(&args[2])?;
    let p: f64 = args[3].parse()?;
    let seed = args[4].parse()?;
    let limit: usize = args.get(5).map_or(Ok(16), |s| s.parse())?;
    // Match the setter's addressability condition before invoking it.
    let platform_limit = (isize::MAX.unsigned_abs() / size_of::<num_complex::Complex64>()).ilog2();
    if limit > usize::try_from(platform_limit)? {
        return Err("active width limit cannot index a complex vector".into());
    }
    TriorthogonalMatrix::bravyi_haah(max_k)?.circuit(p)?;
    println!("shots={shots} p={p} seed={seed} active_width_limit={limit}");
    println!(
        "Timing includes reset, gates, noise, and syndrome; excludes logical expectation queries for both simulators."
    );
    println!("k\tn\tPhasePoly us/shot\tStabActive us/shot\tpeak active width\tacceptance rate");
    for k in (2..=max_k).step_by(2) {
        let matrix = TriorthogonalMatrix::bravyi_haah(k)?;
        let ops: Vec<_> = matrix
            .circuit(p)?
            .into_iter()
            .take_while(|op| !matches!(op, Op::ExpectW(_)))
            .collect();
        let mut phase = PhasePoly::with_seed(matrix.n(), seed);
        let mut phase_rng = PecosRng::seed_from_u64(seed);
        let mut outcomes = Vec::new();
        let start = Instant::now();
        for _ in 0..shots {
            let result = run_shot_sampled(&mut phase, &ops, &mut phase_rng)?;
            outcomes.push(
                result
                    .syndrome
                    .iter()
                    .map(|r| r.outcome)
                    .collect::<Vec<_>>(),
            );
        }
        let phase_us = start.elapsed().as_secs_f64() * 1e6 / f64::from(shots);
        let accepted = u32::try_from(
            outcomes
                .iter()
                .filter(|s| s.iter().all(|&bit| !bit))
                .count(),
        )?;
        let rate = f64::from(accepted) / f64::from(shots);
        // Before T, the encoder has m free pivot coordinates. Physical Z
        // operators act as parities of those m coordinates. Transversal T
        // therefore promotes at most m active coordinates, and all pivot T's
        // attain m. Subsequent Clifford gates/measurements cannot increase it.
        // No amplitudes are allocated for an over-limit circuit.
        if matrix.m() > limit {
            println!(
                "{k}\t{}\t{phase_us:.3}\tskipped (width limit)\t-\t{rate:.6}",
                matrix.n()
            );
            continue;
        }
        let mut active = StabActive::with_seed(matrix.n(), seed).with_max_active_width(limit);
        let mut active_rng = PecosRng::seed_from_u64(seed);
        let mut active_outcomes = Vec::new();
        let mut peak = 0;
        let start = Instant::now();
        for _ in 0..shots {
            active_outcomes.push(active_shot(&mut active, &ops, &mut active_rng));
            peak = peak.max(active.peak_active_width());
        }
        let active_us = start.elapsed().as_secs_f64() * 1e6 / f64::from(shots);
        assert_eq!(outcomes, active_outcomes, "syndrome mismatch at k={k}");
        assert_eq!(peak, matrix.m(), "unexpected active width at k={k}");
        println!(
            "{k}\t{}\t{phase_us:.3}\t{active_us:.3}\t{peak}\t{rate:.6}",
            matrix.n()
        );
    }
    Ok(())
}

fn active_shot(sim: &mut StabActive, ops: &[Op], rng: &mut PecosRng) -> Vec<bool> {
    sim.reset();
    let mut syndrome = Vec::new();
    for op in ops {
        match op {
            Op::PZ(q) => {
                sim.pz(&[*q]);
            }
            Op::PX(q) => {
                sim.px(&[*q]);
            }
            Op::CX(a, b) => {
                sim.cx(&[(*a, *b)]);
            }
            Op::CZ(a, b) => {
                sim.cz(&[(*a, *b)]);
            }
            Op::S(q) => {
                sim.sz(&[*q]);
            }
            Op::Sdg(q) => {
                sim.szdg(&[*q]);
            }
            Op::Z(q) => {
                sim.z(&[*q]);
            }
            Op::T(q) => {
                sim.t(&[*q]);
            }
            Op::ZError(q, p) => {
                if rng.random_bool(*p) {
                    sim.z(&[*q]);
                }
            }
            Op::MeasureX(support) => {
                let (&first, rest) = support
                    .split_first()
                    .expect("independent syndrome row is nonzero");
                for &q in rest {
                    sim.cx(&[(first, q)]);
                }
                syndrome.push(sim.mx(&[first])[0].outcome);
                for &q in rest {
                    sim.cx(&[(first, q)]);
                }
            }
            Op::ExpectW(_) => unreachable!("queries excluded from both timing streams"),
        }
    }
    syndrome
}

fn emit(args: &[String]) -> Result<(), Box<dyn Error>> {
    let matrix = if args[1] == "rm15" {
        TriorthogonalMatrix::rm15()
    } else {
        TriorthogonalMatrix::bravyi_haah(args[1].parse()?)?
    };
    let p = args[2].parse()?;
    let shots = shots(&args[3])?;
    let seed = args[4].parse()?;
    let ops = matrix.circuit(p)?;
    let oracle = matrix.weight_enumerators()?.oracles(p)?;
    let mut sim = PhasePoly::with_seed(matrix.n(), seed);
    let mut rng = PecosRng::seed_from_u64(seed);
    let mut accepted = 0_u32;
    let mut sums = vec![0.0; matrix.k()];
    for _ in 0..shots {
        let result = run_shot_sampled(&mut sim, &ops, &mut rng)?;
        if let Some(w) = result.logical_w {
            accepted += 1;
            for (sum, w) in sums.iter_mut().zip(w) {
                *sum += w;
            }
        }
    }
    let mean_w: Vec<_> = sums
        .iter()
        .map(|sum| (accepted > 0).then(|| sum / f64::from(accepted)))
        .collect();
    let summary = serde_json::json!({
        "k": matrix.k(), "n": matrix.n(), "p": p, "shots": shots, "seed": seed,
        "accepted": accepted, "acceptance_rate": f64::from(accepted) / f64::from(shots),
        "mean_w": mean_w, "P_s": oracle.p_s, "q_a": oracle.q_a,
    });
    let circuit_path = format!("{}.stim", args[5]);
    let json_path = format!("{}.json", args[5]);
    std::fs::write(&circuit_path, to_stim(&ops))?;
    std::fs::write(
        &json_path,
        format!("{}\n", serde_json::to_string_pretty(&summary)?),
    )?;
    println!("wrote {circuit_path} and {json_path}");
    Ok(())
}
