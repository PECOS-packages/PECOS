// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
// except in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the
// License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
// express or implied. See the License for the specific language governing permissions and
// limitations under the License.

//! QEC-like benchmark: syndrome-extraction rounds with small RZ noise.
//!
//! Structure per round:
//!   1. CX ladder entangles each data qubit with its ancilla (syndrome extraction).
//!   2. Small-angle RZ noise on every data qubit (decoherence model).
//!   3. CX ladder in reverse.
//!   4. Ancilla measurements (in Z basis).
//!   5. Ancilla resets (prep |0>).
//!
//! Compares wall time + max bond dim across builder knob combinations:
//!   - default (eager measure, no adaptive truncation)
//!   - `lazy_measure`
//!   - `max_truncation_error`
//!   - both
//!
//! Usage: `cargo run --release --example qec_bench`.

use pecos_core::{Angle64, QubitId};
use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable};
use pecos_stab_tn::stab_mps::StabMps;
use pecos_stab_tn::stab_mps::mast::Mast;
use std::time::Instant;

struct BenchConfig {
    num_data: usize,
    num_rounds: usize,
    noise_angle: Angle64,
    lazy: bool,
    max_trunc: Option<f64>,
    merge_rz: bool,
    max_bond: usize,
    seed: u64,
}

fn build_and_run(cfg: &BenchConfig) -> (f64, usize, u64) {
    let BenchConfig {
        num_data,
        num_rounds,
        noise_angle,
        lazy,
        max_trunc,
        merge_rz,
        max_bond,
        seed,
    } = *cfg;
    // Layout: data qubits [0..num_data), ancilla qubits [num_data..2*num_data).
    let n = num_data * 2;
    let mut builder = StabMps::builder(n)
        .seed(seed)
        .max_bond_dim(max_bond)
        .lazy_measure(lazy)
        .merge_rz(merge_rz);
    if let Some(e) = max_trunc {
        builder = builder.max_truncation_error(e);
    }
    let mut stn = builder.build();

    let start = Instant::now();
    let mut outcome_parity: u64 = 0;

    // Simple xorshift for reproducible pseudo-random gate choices.
    let mut rng_state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    let next_u64 = |s: &mut u64| -> u64 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        *s
    };

    // Initial H on all data to spread into superposition.
    for i in 0..num_data {
        stn.h(&[QubitId(i)]);
    }

    for _round in 0..num_rounds {
        // 1. Long-range CX cascade between random data pairs (mixes entanglement).
        for _ in 0..num_data {
            let a = (next_u64(&mut rng_state) as usize) % num_data;
            let b = (next_u64(&mut rng_state) as usize) % num_data;
            if a != b {
                stn.cx(&[(QubitId(a), QubitId(b))]);
            }
        }
        // 2. T gates (non-Clifford) on random data qubits.
        for _ in 0..num_data {
            let q = (next_u64(&mut rng_state) as usize) % num_data;
            stn.rz(noise_angle, &[QubitId(q)]);
        }
        // 3. Entangle each data with its ancilla (syndrome extraction).
        for i in 0..num_data {
            stn.cx(&[(QubitId(i), QubitId(num_data + i))]);
        }
        // 4. Ancilla measurements (Z-basis).
        for i in 0..num_data {
            let outcome = stn.mz(&[QubitId(num_data + i)])[0].outcome;
            if outcome {
                outcome_parity ^= 1 << (i % 64);
            }
        }
    }
    let elapsed = start.elapsed().as_secs_f64();
    let max_bond_dim = stn.max_bond_dim();
    (elapsed, max_bond_dim, outcome_parity)
}

/// Ion-trap-memory-noise scenario: many small-angle RZs per round on
/// every data qubit (modeling per-step dephasing). RZ batching should
/// merge these consecutive same-qubit RZs into one non-Clifford op.
fn ion_trap_memory_scenario(
    num_data: usize,
    num_rounds: usize,
    noise_per_round: usize,
    noise_angle: Angle64,
    merge_rz: bool,
    seed: u64,
) -> (f64, usize) {
    let mut stn = StabMps::builder(num_data)
        .seed(seed)
        .merge_rz(merge_rz)
        .build();

    // Initial superposition.
    for q in 0..num_data {
        stn.h(&[QubitId(q)]);
    }

    let start = Instant::now();
    for _round in 0..num_rounds {
        // Many small-angle RZ noise per qubit (memory error each timestep).
        for _ in 0..noise_per_round {
            for q in 0..num_data {
                stn.rz(noise_angle, &[QubitId(q)]);
            }
        }
        // One Clifford layer per round (e.g., a syndrome-extraction-like CX).
        for q in 0..num_data - 1 {
            stn.cx(&[(QubitId(q), QubitId(q + 1))]);
        }
    }
    stn.flush();
    let elapsed = start.elapsed().as_secs_f64();
    let bond = stn.max_bond_dim();
    (elapsed, bond)
}

/// MAST-style: T-injection using ancilla pattern, final measurement.
fn mast_scenario(num_qubits: usize, num_t_gates: usize, lazy: bool, seed: u64) -> (f64, usize) {
    let mut mast = Mast::with_seed(num_qubits, num_t_gates, seed).with_lazy_measure(lazy);
    let t = Angle64::QUARTER_TURN / 2u64;

    let start = Instant::now();

    // Random Clifford + T circuit + measurement.
    for q in 0..num_qubits {
        mast.h(&[QubitId(q)]);
    }
    for q in 0..num_qubits - 1 {
        mast.cx(&[(QubitId(q), QubitId(q + 1))]);
    }
    let mut rng_state = 30000u64 + seed;
    for _ in 0..num_t_gates {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        let q = (rng_state % num_qubits as u64) as usize;
        mast.rz(t, &[QubitId(q)]);
    }
    for q in (0..num_qubits - 1).rev() {
        mast.cx(&[(QubitId(q + 1), QubitId(q))]);
    }
    let _ = mast.mz(&[QubitId(0)]);

    let elapsed = start.elapsed().as_secs_f64();
    let bond = mast.mps().max_bond_dim();
    (elapsed, bond)
}

fn main() {
    // Magic-state-distillation-like: T gates per round (non-Clifford-heavy).
    let t_angle = Angle64::QUARTER_TURN / 2u64; // T = RZ(π/4)
    let num_data = 8;
    let num_rounds = 20;
    let max_bond = 64;
    let seed = 42;

    println!(
        "QEC-like bench: {num_data} data qubits, {num_rounds} rounds, T-gate per data per round"
    );
    let _ = t_angle;
    println!("{:-<90}", "");
    println!(
        "{:<40} {:>12} {:>12} {:>20}",
        "config", "time (s)", "max bond", "outcome parity"
    );
    println!("{:-<90}", "");

    let configs: &[(&str, bool, Option<f64>, bool)] = &[
        ("default", false, None, false),
        ("lazy_measure", true, None, false),
        ("max_truncation_error=1e-8", false, Some(1e-8), false),
        ("merge_rz", false, None, true),
        (
            "merge_rz + max_truncation_error=1e-8",
            false,
            Some(1e-8),
            true,
        ),
        ("for_qec()", false, Some(1e-8), true),
    ];

    for &(name, lazy, trunc, merge) in configs {
        let (t, bond, parity) = build_and_run(&BenchConfig {
            num_data,
            num_rounds,
            noise_angle: t_angle,
            lazy,
            max_trunc: trunc,
            merge_rz: merge,
            max_bond,
            seed,
        });
        println!("{name:<40} {t:>12.4} {bond:>12} {parity:>20x}");
    }

    println!("{:-<90}", "");
    println!(
        "\nNote: outcome parities differ between lazy/eager because the RNG is consumed in\n      different sequences (not a correctness issue — both give the right distribution)."
    );

    // -------------------------------------------------------------------------
    // MAST-style scenario: where lazy_measure actually helps.
    // -------------------------------------------------------------------------
    println!();
    println!("MAST-like scenario: deep random Clifford+T measured via ancilla injection");
    println!("{:-<70}", "");
    println!("{:<30} {:>12} {:>12}", "config", "time (s)", "max bond");
    println!("{:-<70}", "");

    let n_q = 8;
    let n_t = 8;
    let num_trials = 20;
    for (name, lazy) in [("eager", false), ("lazy", true)] {
        let mut total_time = 0.0;
        let mut total_bond = 0usize;
        for trial in 0..num_trials {
            let (t, b) = mast_scenario(n_q, n_t, lazy, 20000 + trial as u64);
            total_time += t;
            total_bond += b;
        }
        println!(
            "{name:<30} {:>12.4} {:>12.1}",
            total_time / f64::from(num_trials),
            total_bond as f64 / f64::from(num_trials)
        );
    }
    println!("{:-<70}", "");

    // -------------------------------------------------------------------------
    // Ion-trap memory noise scenario: where merge_rz actually helps.
    // -------------------------------------------------------------------------
    println!();
    println!("Ion-trap memory noise: many small RZs per qubit each round");
    println!("(6 data qubits, 10 rounds, 50 noise RZs/qubit/round, θ=0.01 rad)");
    println!("{:-<70}", "");
    println!("{:<30} {:>12} {:>12}", "config", "time (s)", "max bond");
    println!("{:-<70}", "");
    let small_angle = Angle64::from_radians(0.01);
    for (name, merge) in [("default", false), ("merge_rz", true)] {
        let (t, b) = ion_trap_memory_scenario(6, 10, 50, small_angle, merge, 42);
        println!("{name:<30} {t:>12.4} {b:>12}");
    }
    println!("{:-<70}", "");
}
