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

//! Benchmark disent firing rate across random fuzz circuits.
//!
//! Reports what fraction of non-Clifford RZ gates successfully avoid the
//! multi-site CNOT cascade path via: (a) single-site decomposition,
//! (b) Stabilizer branch (no bond-dim growth), or (c) multi-site disent
//! (one MPS op + tableau right-compose).
//!
//! The remaining fraction hits the std multi-site path which applies CNOTs
//! on the MPS -- the case OFD would replace.

use pecos_core::{Angle64, QubitId};
use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable};
use pecos_stab_tn::stab_mps::StabMps;
use pecos_stab_tn::stab_mps::compile::StabMpsCompile;
use pecos_stab_tn::stab_mps::mast::Mast;
use std::f64::consts::TAU;

/// Same xorshift generator as fuzz tests.
fn next_rng(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// Distribution of gate types to sample.
#[derive(Clone, Copy)]
enum GateMix {
    /// Full random: H/S/X/CX/CZ/T/RZ/RX with equal weights.
    Random,
    /// Clifford + T only (research target for MAST).
    CliffT,
}

fn fuzz_circuit(num_qubits: usize, num_gates: usize, seed: u64, mix: GateMix) -> StabMps {
    let mut stn = StabMps::with_seed(num_qubits, seed);
    // xorshift state 0 stays 0 forever — skip seed 0 by adding offset.
    let mut rng_state = seed.wrapping_add(1);

    for _ in 0..num_gates {
        let n_types: u64 = match mix {
            GateMix::Random => 8,
            GateMix::CliffT => 6,
        };
        let gate_type = next_rng(&mut rng_state) % n_types;
        let q0 = (next_rng(&mut rng_state) % num_qubits as u64) as usize;
        let q1 = loop {
            let q = (next_rng(&mut rng_state) % num_qubits as u64) as usize;
            if q != q0 {
                break q;
            }
        };
        match gate_type {
            0 => {
                stn.h(&[QubitId(q0)]);
            }
            1 => {
                stn.sz(&[QubitId(q0)]);
            }
            2 => {
                stn.x(&[QubitId(q0)]);
            }
            3 => {
                stn.cx(&[(QubitId(q0), QubitId(q1))]);
            }
            4 => {
                stn.cz(&[(QubitId(q0), QubitId(q1))]);
            }
            5 => {
                let t = Angle64::QUARTER_TURN / 2u64;
                stn.rz(t, &[QubitId(q0)]);
            }
            6 => {
                let ab = next_rng(&mut rng_state);
                let a = Angle64::from_radians((ab % 1000) as f64 * 0.001 * TAU);
                stn.rz(a, &[QubitId(q0)]);
            }
            _ => {
                let ab = next_rng(&mut rng_state);
                let a = Angle64::from_radians((ab % 1000) as f64 * 0.001 * TAU);
                stn.rx(a, &[QubitId(q0)]);
            }
        }
    }
    stn
}

/// Run scenario with optional auto-disentangle every N gates.
fn run_scenario_with_auto(
    label: &str,
    n_qubits: usize,
    n_gates: usize,
    n_seeds: u64,
    mix: GateMix,
    auto_disent_every: Option<usize>,
) {
    let mut max_bond_sum = 0u64;
    let mut gates_disent_total = 0u64;
    for seed in 0..n_seeds {
        let mut stn = pecos_stab_tn::stab_mps::StabMps::with_seed(n_qubits, seed);
        let mut rng_state = seed.wrapping_add(1);
        let mut gate_count = 0;
        for _ in 0..n_gates {
            let n_types: u64 = match mix {
                GateMix::Random => 8,
                GateMix::CliffT => 6,
            };
            let gate_type = next_rng(&mut rng_state) % n_types;
            let q0 = (next_rng(&mut rng_state) % n_qubits as u64) as usize;
            let q1 = loop {
                let q = (next_rng(&mut rng_state) % n_qubits as u64) as usize;
                if q != q0 {
                    break q;
                }
            };
            match gate_type {
                0 => {
                    stn.h(&[QubitId(q0)]);
                }
                1 => {
                    stn.sz(&[QubitId(q0)]);
                }
                2 => {
                    stn.x(&[QubitId(q0)]);
                }
                3 => {
                    stn.cx(&[(QubitId(q0), QubitId(q1))]);
                }
                4 => {
                    stn.cz(&[(QubitId(q0), QubitId(q1))]);
                }
                5 => {
                    stn.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(q0)]);
                }
                6 => {
                    let ab = next_rng(&mut rng_state);
                    stn.rz(
                        Angle64::from_radians((ab % 1000) as f64 * 0.001 * TAU),
                        &[QubitId(q0)],
                    );
                }
                _ => {
                    let ab = next_rng(&mut rng_state);
                    stn.rx(
                        Angle64::from_radians((ab % 1000) as f64 * 0.001 * TAU),
                        &[QubitId(q0)],
                    );
                }
            }
            gate_count += 1;
            if let Some(every) = auto_disent_every
                && gate_count % every == 0
            {
                gates_disent_total += stn.disentangle(2) as u64;
            }
        }
        // Final disent sweep
        if auto_disent_every.is_some() {
            gates_disent_total += stn.disentangle(3) as u64;
        }
        max_bond_sum += stn.max_bond_dim() as u64;
    }
    let avg_bond = max_bond_sum as f64 / n_seeds as f64;
    let avg_disent = gates_disent_total as f64 / n_seeds as f64;
    println!(
        "{label:<28} n={n_qubits} gates={n_gates} auto_every={auto_disent_every:?} | avg_bond={avg_bond:.2} avg_disent_gates={avg_disent:.1}",
    );
}

fn run_scenario(label: &str, n_qubits: usize, n_gates: usize, n_seeds: u64, mix: GateMix) {
    use std::io::Write;
    let mut total = 0u64;
    let mut single = 0u64;
    let mut disent = 0u64;
    let mut std_path = 0u64;
    let mut stabilizer = 0u64;
    let mut max_bond_sum = 0u64;
    let mut theoretical_bond_sum = 0u64;
    let mut ofd_in_span = 0u64;
    let mut ofd_wins = 0u64; // in_span gates heuristic sent through std path

    for seed in 0..n_seeds {
        let stn = fuzz_circuit(n_qubits, n_gates, seed, mix);
        theoretical_bond_sum += stn.gf2_matrix().theoretical_min_bond_dim() as u64;
        ofd_in_span += stn.stats.ofd_in_span;
        ofd_wins += stn.stats.ofd_in_span_std;
        let s = stn.stats;
        total += s.total_nonclifford;
        single += s.single_site;
        disent += s.multi_disent;
        std_path += s.multi_std;
        stabilizer += s.stabilizer;
        max_bond_sum += stn.max_bond_dim() as u64;
    }

    let pct = |x: u64| {
        if total == 0 {
            0.0
        } else {
            100.0 * x as f64 / total as f64
        }
    };
    let avg_bond = max_bond_sum as f64 / n_seeds as f64;
    let avg_theo = theoretical_bond_sum as f64 / n_seeds as f64;
    println!(
        "{label:<24} n={n_qubits} gates={n_gates} | total={total} \
        heur: stab={:.0}% single={:.0}% disent={:.0}% std={:.0}% | \
        OFD in_span={:.0}% | OFD wins (in_span but heur-std) ={ofd_wins}/{} ({:.0}%) | \
        bond={avg_bond:.2}/{avg_theo:.2}",
        pct(stabilizer),
        pct(single),
        pct(disent),
        pct(std_path),
        pct(ofd_in_span),
        std_path,
        if std_path == 0 {
            0.0
        } else {
            100.0 * ofd_wins as f64 / std_path as f64
        },
    );
    let _ = std::io::stdout().flush();
}

fn main() {
    println!("Disent firing rate benchmark. Runs random fuzz circuits and reports");
    println!("what fraction of non-Clifford RZs take each code path.");
    println!();
    println!("  stab    = Stabilizer branch (Z_q already a stabilizer product: no MPS site ops)");
    println!("  single  = single-site decomposition (trivial 1-qubit gate on MPS)");
    println!("  disent  = multi-site disent fires (1-qubit MPS op + tableau right-compose)");
    println!("  std     = multi-site CNOT cascade on MPS (OFD target to replace)");
    println!();

    // Random gate mix: Cliffords + rotations + T.
    println!("=== Random gate mix (H/S/X/CX/CZ/T/RZ/RX) ===");
    run_scenario("2q shallow", 2, 10, 100, GateMix::Random);
    run_scenario("2q medium", 2, 20, 100, GateMix::Random);
    run_scenario("2q deep", 2, 50, 50, GateMix::Random);
    run_scenario("3q shallow", 3, 10, 50, GateMix::Random);
    run_scenario("3q medium", 3, 20, 30, GateMix::Random);
    run_scenario("4q shallow", 4, 10, 30, GateMix::Random);
    run_scenario("4q medium", 4, 20, 20, GateMix::Random);

    // T-heavy: research target for MAST / Clifford+T simulation.
    println!("\n=== Clifford + T only (H/S/X/CX/CZ/T) ===");
    run_scenario("2q T 10g", 2, 10, 100, GateMix::CliffT);
    run_scenario("2q T 30g", 2, 30, 50, GateMix::CliffT);
    run_scenario("3q T 15g", 3, 15, 50, GateMix::CliffT);
    run_scenario("3q T 30g", 3, 30, 30, GateMix::CliffT);
    run_scenario("4q T 20g", 4, 20, 20, GateMix::CliffT);
    run_scenario("5q T 20g", 5, 20, 10, GateMix::CliffT);
    run_scenario("8q T 30g", 8, 30, 5, GateMix::CliffT);
    run_scenario("10q T 40g", 10, 40, 3, GateMix::CliffT);
    run_scenario("15q T 50g", 15, 50, 2, GateMix::CliffT);

    // Test auto-heuristic-disentangle: compare to baseline on 2q deep (bond 2)
    // where std path fires heavily.
    // Pre-analysis with StabMpsCompile: same circuit, no MPS cost. Verifies
    // that the compile-only pass gives matching OFD predictions.
    println!("\n=== StabMpsCompile pre-analysis (no MPS cost) ===");
    bench_compile("5q T 20g", 5, 20, 50, GateMix::CliffT);
    bench_compile("10q T 30g", 10, 30, 20, GateMix::CliffT);
    bench_compile("20q T 50g", 20, 50, 10, GateMix::CliffT);
    bench_compile("50q T 100g", 50, 100, 5, GateMix::CliffT);
    bench_compile("100q T 200g", 100, 200, 2, GateMix::CliffT);

    println!("\n=== Auto heuristic disentangle (on 2q deep) ===");
    run_scenario_with_auto("baseline (no auto)", 2, 50, 20, GateMix::Random, None);
    run_scenario_with_auto("auto every 5 gates", 2, 50, 20, GateMix::Random, Some(5));
    run_scenario_with_auto("auto every 10 gates", 2, 50, 20, GateMix::Random, Some(10));
    run_scenario_with_auto("auto every 20 gates", 2, 50, 20, GateMix::Random, Some(20));

    println!("\n=== Auto heuristic disentangle (on 3q T 30g) ===");
    run_scenario_with_auto("baseline (no auto)", 3, 30, 20, GateMix::CliffT, None);
    run_scenario_with_auto("auto every 5 gates", 3, 30, 20, GateMix::CliffT, Some(5));
    run_scenario_with_auto("auto every 10 gates", 3, 30, 20, GateMix::CliffT, Some(10));

    // MAST: magic-state injection scheme. Targets 20-200 qubits with bond ~1.
    println!("\n=== MAST (Clifford+T, deferred ancilla projection) ===");
    run_mast_scenario("10q T 10", 10, 10, 10);
    run_mast_scenario("10q T 50", 10, 50, 5);
    run_mast_scenario("20q T 20", 20, 20, 5);
    run_mast_scenario("20q T 100", 20, 100, 3);
    run_mast_scenario("50q T 20", 50, 20, 3);
    run_mast_scenario("50q T 100", 50, 100, 1);
    run_mast_scenario("100q T 30", 100, 30, 2);
    run_mast_scenario("100q T 100", 100, 100, 1);
}

/// Benchmark `StabMpsCompile` on a fuzz circuit: reports timing and nullity.
fn bench_compile(label: &str, n_qubits: usize, n_gates: usize, n_seeds: u64, mix: GateMix) {
    let mut total_nullity = 0usize;
    let mut total_absorbed = 0u64;
    let mut total_grown = 0u64;
    let start = std::time::Instant::now();
    for seed in 0..n_seeds {
        let mut c = StabMpsCompile::new(n_qubits);
        let mut rng_state = seed.wrapping_add(1);
        for _ in 0..n_gates {
            let n_types: u64 = match mix {
                GateMix::Random => 8,
                GateMix::CliffT => 6,
            };
            let gate_type = next_rng(&mut rng_state) % n_types;
            let q0 = (next_rng(&mut rng_state) % n_qubits as u64) as usize;
            let q1 = loop {
                let q = (next_rng(&mut rng_state) % n_qubits as u64) as usize;
                if q != q0 {
                    break q;
                }
            };
            match gate_type {
                0 => {
                    c.h(&[QubitId(q0)]);
                }
                1 => {
                    c.sz(&[QubitId(q0)]);
                }
                2 => {
                    c.x(&[QubitId(q0)]);
                }
                3 => {
                    c.cx(&[(QubitId(q0), QubitId(q1))]);
                }
                4 => {
                    c.cz(&[(QubitId(q0), QubitId(q1))]);
                }
                5 => {
                    c.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(q0)]);
                }
                6 => {
                    let ab = next_rng(&mut rng_state);
                    c.rz(
                        Angle64::from_radians((ab % 1000) as f64 * 0.001 * TAU),
                        &[QubitId(q0)],
                    );
                }
                _ => {
                    let ab = next_rng(&mut rng_state);
                    c.rx(
                        Angle64::from_radians((ab % 1000) as f64 * 0.001 * TAU),
                        &[QubitId(q0)],
                    );
                }
            }
        }
        total_nullity += c.nullity();
        total_absorbed += c.absorbed();
        total_grown += c.grown();
    }
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    let avg_nullity = total_nullity as f64 / n_seeds as f64;
    let avg_absorbed = total_absorbed as f64 / n_seeds as f64;
    let avg_grown = total_grown as f64 / n_seeds as f64;
    println!(
        "{label:<20} n={n_qubits} g={n_gates} | absorbed={avg_absorbed:.1} grown={avg_grown:.1} nullity={avg_nullity:.1} bound=2^{avg_nullity:.1} | elapsed={elapsed_ms:.1}ms ({n_seeds} seeds)",
    );
}

fn run_mast_scenario(label: &str, n_data: usize, n_t: usize, n_seeds: u64) {
    let mut total = 0u64;
    let mut single = 0u64;
    let mut disent = 0u64;
    let mut std_path = 0u64;
    let mut stabilizer = 0u64;
    let mut max_bond_sum = 0u64;

    for seed in 0..n_seeds {
        let mut mast = Mast::with_seed(n_data, n_t, seed);
        let mut rng_state = seed.wrapping_add(1);
        // Scatter H/CX and T gates so ancillas get diverse inputs.
        for _ in 0..n_t {
            // Random Clifford layer
            for _ in 0..3 {
                let gt = next_rng(&mut rng_state) % 3;
                let q0 = (next_rng(&mut rng_state) % n_data as u64) as usize;
                let q1 = loop {
                    let q = (next_rng(&mut rng_state) % n_data as u64) as usize;
                    if q != q0 {
                        break q;
                    }
                };
                match gt {
                    0 => {
                        mast.h(&[QubitId(q0)]);
                    }
                    1 => {
                        mast.sz(&[QubitId(q0)]);
                    }
                    _ => {
                        mast.cx(&[(QubitId(q0), QubitId(q1))]);
                    }
                }
            }
            // T gate on random qubit
            let tq = (next_rng(&mut rng_state) % n_data as u64) as usize;
            mast.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(tq)]);
        }
        mast.project_all();

        let s = mast.stats;
        total += s.total_nonclifford;
        single += s.single_site;
        disent += s.multi_disent;
        std_path += s.multi_std;
        stabilizer += s.stabilizer;
        max_bond_sum += mast.max_bond_dim() as u64;
    }

    let pct = |x: u64| {
        if total == 0 {
            0.0
        } else {
            100.0 * x as f64 / total as f64
        }
    };
    let avg_bond = max_bond_sum as f64 / n_seeds as f64;
    println!(
        "{label:<24} n={n_data} T={n_t} seeds={n_seeds} | \
        total={total} stab={stabilizer} ({:.1}%) single={single} ({:.1}%) disent={disent} ({:.1}%) std={std_path} ({:.1}%) | avg_max_bond={avg_bond:.2}",
        pct(stabilizer),
        pct(single),
        pct(disent),
        pct(std_path),
    );
}
