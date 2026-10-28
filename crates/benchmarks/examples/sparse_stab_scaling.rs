// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0.

//! One surface-code size per process. Usage:
//! `sparse_stab_scaling SET D ROUNDS [P [F [SEED]]] [--stats-every K]`.

use pecos_core::{AdaptiveIndexSet, BitSet, IndexSet, QubitId, SortedVecSet, VecSet};
use pecos_random::PecosRng;
use pecos_simulators::{CliffordGateable, GensGeneric, SparseStabGeneric};
use rand::RngExt;
use std::time::Instant;

struct Config {
    d: usize,
    rounds: usize,
    p: f64,
    f: f64,
    seed: u64,
    stats_every: usize,
}

struct Schedule {
    x_ancillas: Vec<QubitId>,
    ancillas: Vec<QubitId>,
    is_x: Vec<bool>,
    layers: [Vec<(QubitId, QubitId)>; 4],
}

fn schedule(d: usize) -> Schedule {
    let side = isize::try_from(d).expect("distance fits isize");
    let mut result = Schedule {
        x_ancillas: Vec::new(),
        ancillas: Vec::new(),
        is_x: Vec::new(),
        layers: std::array::from_fn(|_| Vec::new()),
    };
    for r in -1..side {
        for c in -1..side {
            let interior = (0..side - 1).contains(&r) && (0..side - 1).contains(&c);
            let boundary = (r == -1 && (0..side - 1).contains(&c) && c % 2 == 1)
                || (r == side - 1 && (0..side - 1).contains(&c) && c % 2 == 0)
                || (c == -1 && (0..side - 1).contains(&r) && r % 2 == 0)
                || (c == side - 1 && (0..side - 1).contains(&r) && r % 2 == 1);
            if !interior && !boundary {
                continue;
            }
            let x_type = (r + c) % 2 == 0;
            let ancilla = QubitId(d * d + result.ancillas.len());
            result.ancillas.push(ancilla);
            result.is_x.push(x_type);
            if x_type {
                result.x_ancillas.push(ancilla);
            }
            let order = if x_type {
                [(0, 0), (0, 1), (1, 0), (1, 1)]
            } else {
                [(0, 0), (1, 0), (0, 1), (1, 1)]
            };
            let corners = order.into_iter().filter_map(|(dr, dc)| {
                let (row, col) = (r + dr, c + dc);
                ((0..side).contains(&row) && (0..side).contains(&col))
                    .then(|| QubitId(usize::try_from(row * side + col).unwrap()))
            });
            // Layer k applies each check's k-th existing corner, the same schedule
            // as crates/pecos-simulators/tests/sparse_stab_oracle.rs.
            for (layer, data) in corners.enumerate() {
                result.layers[layer].push(if x_type {
                    (ancilla, data)
                } else {
                    (data, ancilla)
                });
            }
        }
    }
    assert_eq!(result.ancillas.len(), d * d - 1);
    result
}

fn stats<S: IndexSet>(gens: &GensGeneric<S>) -> (usize, usize) {
    let nnz = gens
        .row_x
        .iter()
        .chain(&gens.row_z)
        .map(IndexSet::len)
        .sum();
    let max_column = gens
        .col_x
        .iter()
        .zip(&gens.col_z)
        .map(|(x, z)| x.len() + z.len())
        .max()
        .unwrap_or(0);
    (nnz, max_column)
}

fn run<S: IndexSet>(config: &Config) {
    let construction = Instant::now();
    let ops = schedule(config.d);
    let mut rng = PecosRng::seed_from_u64(config.seed);
    // Fix stochastic operation lists outside all round timers.
    let draws: Vec<_> = (0..config.rounds)
        .map(|_| {
            let (mut x, mut z, mut resets) = (Vec::new(), Vec::new(), Vec::new());
            for q in 0..config.d * config.d {
                if rng.random_bool(config.p) {
                    if rng.random() {
                        x.push(QubitId(q));
                    } else {
                        z.push(QubitId(q));
                    }
                }
                if rng.random_bool(config.f) {
                    resets.push(QubitId(q));
                }
            }
            (x, z, resets)
        })
        .collect();
    let mut sim = SparseStabGeneric::<S>::with_rng(
        2 * config.d * config.d - 1,
        PecosRng::seed_from_u64(config.seed),
    );
    let construction_seconds = construction.elapsed().as_secs_f64();
    println!("# VmHWM is the process peak, including precomputed operation lists.");
    println!(
        "# stats outside timers; BitSet scan O(n^2/64); max column = col_x.len + col_z.len; unsampled=NA"
    );
    println!(
        "round,gate_seconds,measurement_reset_seconds,nondeterministic,stabilizer_nnz,destabilizer_nnz,max_stabilizer_column,max_destabilizer_column"
    );
    let mut first = Vec::new();
    for (round, (x, z, resets)) in draws.iter().enumerate() {
        let start = Instant::now();
        sim.x(x).z(z);
        sim.h(&ops.x_ancillas);
        for layer in &ops.layers {
            sim.cx(layer);
        }
        sim.h(&ops.x_ancillas);
        let gate_seconds = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let measurements = sim.mz(&ops.ancillas);
        sim.pz(&ops.ancillas);
        let data_measurements = sim.mz(resets);
        sim.pz(resets);
        let measurement_seconds = start.elapsed().as_secs_f64();
        let nondeterministic = measurements
            .iter()
            .chain(&data_measurements)
            .filter(|m| !m.is_deterministic)
            .count();
        if config.p == 0.0 && config.f == 0.0 {
            if round == 0 {
                for (&is_x, m) in ops.is_x.iter().zip(&measurements) {
                    if is_x {
                        assert!(!m.is_deterministic);
                    } else {
                        assert!(m.is_deterministic && !m.outcome);
                    }
                }
                first = measurements.iter().map(|m| m.outcome).collect();
            } else {
                for (m, &initial) in measurements.iter().zip(&first) {
                    assert!(m.is_deterministic);
                    assert_eq!(m.outcome, initial);
                }
            }
        }
        if round % config.stats_every == 0 {
            let (s, sc) = stats(sim.stabs());
            let (d, dc) = stats(sim.destabs());
            println!(
                "{round},{gate_seconds:.9},{measurement_seconds:.9},{nondeterministic},{s},{d},{sc},{dc}"
            );
        } else {
            println!(
                "{round},{gate_seconds:.9},{measurement_seconds:.9},{nondeterministic},NA,NA,NA,NA"
            );
        }
    }
    println!("construction_seconds={construction_seconds:.9}");
    let peak = std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find_map(|line| line.strip_prefix("VmHWM:").map(|v| v.trim().to_owned()))
        });
    println!("peak_rss={}", peak.as_deref().unwrap_or("unavailable"));
}

fn main() {
    let mut args: Vec<_> = std::env::args().skip(1).collect();
    let mut stats_every = 1;
    if let Some(i) = args.iter().position(|s| s == "--stats-every") {
        stats_every = args
            .get(i + 1)
            .expect("--stats-every requires K")
            .parse()
            .expect("integer K");
        args.drain(i..=i + 1);
    }
    assert!(
        (3..=6).contains(&args.len()),
        "usage: SET D ROUNDS [P [F [SEED]]] [--stats-every K]"
    );
    let config = Config {
        d: args[1].parse().expect("integer distance"),
        rounds: args[2].parse().expect("integer rounds"),
        p: args
            .get(3)
            .map_or(0.0, |s| s.parse().expect("noise probability")),
        f: args
            .get(4)
            .map_or(0.0, |s| s.parse().expect("reset probability")),
        seed: args
            .get(5)
            .map_or(1434, |s| s.parse().expect("integer seed")),
        stats_every,
    };
    assert!(config.d >= 3 && config.d % 2 == 1 && config.rounds > 0 && stats_every > 0);
    assert!((0.0..=1.0).contains(&config.p) && (0.0..=1.0).contains(&config.f));
    match args[0].as_str() {
        "bitset" => run::<BitSet>(&config),
        "adaptive" => run::<AdaptiveIndexSet>(&config),
        "sorted" => run::<SortedVecSet>(&config),
        "unsorted" => run::<VecSet<usize>>(&config),
        other => panic!("unknown set {other}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pecos_simulators::SparseStabBitSet;
    use std::hint::black_box;

    #[test]
    fn reset_timing_at_7937() {
        // Identically initialized simulators; construction is outside both timers.
        // A separate release test keeps the scaling process's peak RSS comparable.
        let mut adaptive =
            SparseStabGeneric::<AdaptiveIndexSet>::with_rng(7937, PecosRng::seed_from_u64(1434));
        let mut bitset = SparseStabBitSet::with_rng(7937, PecosRng::seed_from_u64(1434));
        let start = Instant::now();
        black_box(&mut adaptive).reset();
        let adaptive_seconds = start.elapsed().as_secs_f64();
        let start = Instant::now();
        black_box(&mut bitset).reset();
        let bitset_seconds = start.elapsed().as_secs_f64();
        black_box((&adaptive, &bitset));
        println!(
            "reset_n=7937 adaptive_seconds={adaptive_seconds:.9} bitset_seconds={bitset_seconds:.9}"
        );
    }
}
