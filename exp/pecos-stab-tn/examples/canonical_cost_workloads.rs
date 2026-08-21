//! Fixed-seed workloads for measuring canonical-form overhead.
//!
//! This is deliberately an example rather than a library benchmark so the
//! same source can be copied into historical worktrees without changing the
//! library under test. Run one case at a time with, for example:
//!
//! `CANON_WORKLOAD=qec8 cargo run --release -p pecos-stab-tn --example canonical_cost_workloads`

use pecos_core::{Angle64, QubitId};
use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable};
use pecos_stab_tn::stab_mps::{PauliKind, StabMps, StabMpsStats};
use std::time::Instant;

#[derive(Clone, Copy, Default)]
struct Counts {
    multi_disent: u64,
    multi_std: u64,
    single_site: u64,
}

impl Counts {
    fn add(&mut self, stats: StabMpsStats) {
        self.multi_disent += stats.multi_disent;
        self.multi_std += stats.multi_std;
        self.single_site += stats.single_site;
    }
}

fn next(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn distinct_pair(state: &mut u64, n: usize) -> (usize, usize) {
    let first = (next(state) % n as u64) as usize;
    let mut second = (next(state) % (n - 1) as u64) as usize;
    if second >= first {
        second += 1;
    }
    (first, second)
}

fn mix_hash(hash: &mut u64, value: u64) {
    *hash ^= value;
    *hash = hash.wrapping_mul(0x100_0000_01b3);
}

fn mix_measurement(hash: &mut u64, outcome: bool, deterministic: bool) {
    mix_hash(hash, (u64::from(outcome) << 1) | u64::from(deterministic));
}

fn qec_generators(num_data: usize, num_checks: usize) -> Vec<Vec<(usize, PauliKind)>> {
    (0..num_checks)
        .map(|check| {
            // Sparse, overlapping parity checks with both local and long-range
            // data couplings. All generators are Z-type and commute.
            let first = (2 * check) % num_data;
            let second = (first + 1) % num_data;
            let third = (first + num_data / 2) % num_data;
            vec![
                (first, PauliKind::Z),
                (second, PauliKind::Z),
                (third, PauliKind::Z),
            ]
        })
        .collect()
}

fn run_qec(total_qubits: usize, repeats: usize) -> (Counts, u64, usize, usize, f64) {
    let num_checks = total_qubits / 4;
    let num_data = total_qubits - num_checks;
    let rounds = if total_qubits == 8 { 12 } else { 8 };
    let generators = qec_generators(num_data, num_checks);
    let ancillas: Vec<QubitId> = (num_data..total_qubits).map(QubitId).collect();
    let circuits = 4 * repeats;
    let mut counts = Counts::default();
    let mut trajectory_hash = 0xcbf2_9ce4_8422_2325;
    let start = Instant::now();

    for circuit in 0..circuits {
        let seed = 0x0cec_0000_0000_0000 ^ (total_qubits as u64) << 32 ^ circuit as u64;
        let mut random = seed.wrapping_add(1);
        let mut stn = StabMps::builder(total_qubits)
            .seed(seed)
            .max_bond_dim(64)
            .merge_rz(false)
            .build();

        // Prepare an entangled data block before repeated extraction.
        stn.h(&[QubitId(0)]);
        for q in 1..num_data {
            stn.cx(&[(QubitId(0), QubitId(q))]);
        }

        for round in 0..rounds {
            // Coherent memory noise forces the stabilizer-tensor path while
            // retaining the repeated-syndrome shape of the workload.
            for q in 0..num_data {
                let angle = if (q + round) & 1 == 0 {
                    Angle64::QUARTER_TURN / 2_u64
                } else {
                    Angle64::from_radians(0.03125)
                };
                stn.rz(angle, &[QubitId(q)]);
            }
            let (control, target) = distinct_pair(&mut random, num_data);
            stn.cx(&[(QubitId(control), QubitId(target))]);

            for bit in stn.extract_syndromes(&generators, &ancillas) {
                mix_hash(&mut trajectory_hash, u64::from(bit));
            }

            // Explicit data-qubit measure/reset in the middle, in addition to
            // the ancilla measurements and resets inside extract_syndromes.
            if round + 1 == rounds / 2 {
                let q = (next(&mut random) % num_data as u64) as usize;
                let outcome = stn.reset_qubit(QubitId(q));
                mix_hash(&mut trajectory_hash, u64::from(outcome));
                stn.h(&[QubitId(q)]);
            }
        }
        counts.add(stn.stats);
    }

    (
        counts,
        trajectory_hash,
        circuits,
        rounds,
        start.elapsed().as_secs_f64(),
    )
}

fn run_deep(
    num_qubits: usize,
    t_multiplier: usize,
    repeats: usize,
) -> (Counts, u64, usize, usize, f64) {
    let t_count = t_multiplier * num_qubits;
    let circuits = 4 * repeats;
    let mut counts = Counts::default();
    let mut trajectory_hash = 0xcbf2_9ce4_8422_2325;
    let start = Instant::now();

    for circuit in 0..circuits {
        let seed = 0xc11f_0000_0000_0000
            ^ (num_qubits as u64) << 32
            ^ (t_multiplier as u64) << 24
            ^ circuit as u64;
        let mut random = seed.wrapping_add(1);
        let mut stn = StabMps::builder(num_qubits)
            .seed(seed)
            .max_bond_dim(64)
            .merge_rz(false)
            .build();

        let all_qubits: Vec<QubitId> = (0..num_qubits).map(QubitId).collect();
        stn.h(&all_qubits);
        for injection in 0..t_count {
            // Three Clifford operations per T, including a deliberately
            // long-range two-qubit layer.
            let q = (next(&mut random) % num_qubits as u64) as usize;
            if next(&mut random) & 1 == 0 {
                stn.h(&[QubitId(q)]);
            } else {
                stn.sz(&[QubitId(q)]);
            }
            let (control, target) = distinct_pair(&mut random, num_qubits);
            stn.cx(&[(QubitId(control), QubitId(target))]);
            let (first, second) = distinct_pair(&mut random, num_qubits);
            stn.cz(&[(QubitId(first), QubitId(second))]);

            let tq = (next(&mut random) % num_qubits as u64) as usize;
            stn.rz(Angle64::QUARTER_TURN / 2_u64, &[QubitId(tq)]);

            if injection + 1 == t_count / 2 {
                for _ in 0..2 {
                    let measured = (next(&mut random) % num_qubits as u64) as usize;
                    let outcome = stn.reset_qubit(QubitId(measured));
                    mix_hash(&mut trajectory_hash, u64::from(outcome));
                    stn.h(&[QubitId(measured)]);
                }
            }
        }
        for result in stn.mz(&all_qubits) {
            mix_measurement(
                &mut trajectory_hash,
                result.outcome,
                result.is_deterministic,
            );
        }
        counts.add(stn.stats);
    }

    (
        counts,
        trajectory_hash,
        circuits,
        t_count,
        start.elapsed().as_secs_f64(),
    )
}

fn main() {
    let workload = std::env::var("CANON_WORKLOAD").unwrap_or_else(|_| "all".to_owned());
    let repeats = std::env::var("CANON_REPEATS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1);

    for n in [8_usize, 16] {
        let name = format!("qec{n}");
        if workload == "all" || workload == name {
            let (counts, hash, circuits, rounds, seconds) = run_qec(n, repeats);
            println!(
                "QEC n={n} circuits={circuits} rounds={rounds} multi_disent={} multi_std={} single_site={} trajectory_hash={hash:016x} wall_s={seconds:.6}",
                counts.multi_disent, counts.multi_std, counts.single_site
            );
        }
    }

    for n in [12_usize, 20] {
        for multiplier in [1_usize, 2] {
            let name = format!("deep{n}_{multiplier}n");
            if workload == "all" || workload == name {
                let (counts, hash, circuits, t_count, seconds) = run_deep(n, multiplier, repeats);
                println!(
                    "DEEP n={n} circuits={circuits} t_count={t_count} multi_disent={} multi_std={} single_site={} trajectory_hash={hash:016x} wall_s={seconds:.6}",
                    counts.multi_disent, counts.multi_std, counts.single_site
                );
            }
        }
    }
}
