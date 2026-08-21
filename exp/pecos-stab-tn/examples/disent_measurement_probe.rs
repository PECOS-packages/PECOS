//! Reproducible firing-rate probe with mid-circuit measurements.
//!
//! Unlike `disent_firing_rate`, every non-MAST scenario here measures and
//! resets during the circuit.  The MAST leg records only RZ routing performed
//! after an initial `project_all`, so projection-installed proofs are exercised.

use nalgebra::DMatrix;
use num_complex::Complex64;
use pecos_core::{Angle64, QubitId};
use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable};
use pecos_stab_tn::mps::{Mps, MpsConfig};
use pecos_stab_tn::stab_mps::StabMps;
use pecos_stab_tn::stab_mps::StabMpsStats;
use pecos_stab_tn::stab_mps::mast::Mast;
use std::hint::black_box;
use std::time::Instant;

#[derive(Clone, Copy, Default)]
struct Counts {
    single_site: u64,
    multi_disent: u64,
    multi_std: u64,
    numerical_redetect: u64,
}

impl Counts {
    fn add_stats(&mut self, stats: StabMpsStats) {
        self.single_site += stats.single_site;
        self.multi_disent += stats.multi_disent;
        self.multi_std += stats.multi_std;
        self.numerical_redetect += stats.numerical_redetect;
    }

    fn add_delta(&mut self, after: StabMpsStats, before: StabMpsStats) {
        self.single_site += after.single_site - before.single_site;
        self.multi_disent += after.multi_disent - before.multi_disent;
        self.multi_std += after.multi_std - before.multi_std;
        self.numerical_redetect += after.numerical_redetect - before.numerical_redetect;
    }

    fn fast_rate(self) -> f64 {
        let multi = self.multi_disent + self.multi_std;
        if multi == 0 {
            0.0
        } else {
            100.0 * self.multi_disent as f64 / multi as f64
        }
    }
}

fn next(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn distinct_pair(state: &mut u64, num_qubits: usize) -> (usize, usize) {
    let first = (next(state) % num_qubits as u64) as usize;
    let mut second = (next(state) % (num_qubits - 1) as u64) as usize;
    if second >= first {
        second += 1;
    }
    (first, second)
}

fn mix_hash(hash: &mut u64, value: u64) {
    *hash ^= value;
    *hash = hash.wrapping_mul(0x100_0000_01b3);
}

fn apply_random_clifford_stn(stn: &mut StabMps, state: &mut u64, num_qubits: usize) {
    let q = (next(state) % num_qubits as u64) as usize;
    match next(state) % 5 {
        0 => {
            stn.h(&[QubitId(q)]);
        }
        1 => {
            stn.sz(&[QubitId(q)]);
        }
        2 => {
            stn.x(&[QubitId(q)]);
        }
        3 => {
            let (control, target) = distinct_pair(state, num_qubits);
            stn.cx(&[(QubitId(control), QubitId(target))]);
        }
        _ => {
            let (first, second) = distinct_pair(state, num_qubits);
            stn.cz(&[(QubitId(first), QubitId(second))]);
        }
    }
}

fn run_stn(repeats: usize) -> (Counts, u64, u64, f64) {
    let mut counts = Counts::default();
    let mut trajectory_hash = 0xcbf2_9ce4_8422_2325;
    let mut rotations = 0;
    let start = Instant::now();

    for repeat in 0..repeats {
        for &num_qubits in &[4_usize, 6, 8, 10] {
            for &steps in &[60_usize, 90, 120, 150] {
                for seed in 0..2_u64 {
                    let simulator_seed = 0x5100_0000
                        ^ (repeat as u64) << 24
                        ^ (num_qubits as u64) << 16
                        ^ (steps as u64) << 4
                        ^ seed;
                    let mut random = simulator_seed.wrapping_add(1);
                    let mut stn = StabMps::builder(num_qubits)
                        .seed(simulator_seed)
                        .max_bond_dim(64)
                        .merge_rz(false)
                        .build();

                    for step in 0..steps {
                        apply_random_clifford_stn(&mut stn, &mut random, num_qubits);
                        let q = (next(&mut random) % num_qubits as u64) as usize;
                        let angle = match next(&mut random) % 3 {
                            0 => Angle64::QUARTER_TURN / 2_u64,
                            1 => Angle64::from_radians(0.37),
                            _ => Angle64::from_radians(-0.61),
                        };
                        stn.rz(angle, &[QubitId(q)]);
                        rotations += 1;

                        if step + 1 == steps / 2 {
                            // The reset outcomes may be stochastic, but resetting
                            // every qubit gives every revision the same physical
                            // continuation.  The immediately following measurements
                            // make trajectory equality directly observable.
                            for q in 0..num_qubits {
                                let _ = stn.reset_qubit(QubitId(q));
                            }
                            let qubits: Vec<QubitId> = (0..num_qubits).map(QubitId).collect();
                            for result in stn.mz(&qubits) {
                                mix_hash(
                                    &mut trajectory_hash,
                                    (u64::from(result.outcome) << 1)
                                        | u64::from(result.is_deterministic),
                                );
                            }
                        }
                    }
                    counts.add_stats(stn.stats);
                }
            }
        }
    }

    (
        counts,
        rotations,
        trajectory_hash,
        start.elapsed().as_secs_f64(),
    )
}

fn apply_random_clifford_mast(mast: &mut Mast, state: &mut u64, num_qubits: usize) {
    let q = (next(state) % num_qubits as u64) as usize;
    match next(state) % 4 {
        0 => {
            mast.h(&[QubitId(q)]);
        }
        1 => {
            mast.sz(&[QubitId(q)]);
        }
        2 => {
            let (control, target) = distinct_pair(state, num_qubits);
            mast.cx(&[(QubitId(control), QubitId(target))]);
        }
        _ => {
            let (first, second) = distinct_pair(state, num_qubits);
            mast.cz(&[(QubitId(first), QubitId(second))]);
        }
    }
}

fn run_mast_post_projection(repeats: usize) -> (Counts, u64, f64) {
    const NUM_DATA: usize = 3;
    const PRE_INJECTIONS: usize = 3;
    const POST_INJECTIONS: usize = 5;
    const SEEDS: u64 = 16;

    let mut counts = Counts::default();
    let mut post_rotations = 0;
    let mut elapsed = 0.0;

    for repeat in 0..repeats {
        for seed in 0..SEEDS {
            let simulator_seed = 0x5a00_0000 ^ (repeat as u64) << 16 ^ seed;
            let mut random = simulator_seed.wrapping_add(1);
            let mut mast =
                Mast::with_seed(NUM_DATA, PRE_INJECTIONS + POST_INJECTIONS, simulator_seed)
                    .with_merge_rz(false);

            for _ in 0..PRE_INJECTIONS {
                apply_random_clifford_mast(&mut mast, &mut random, NUM_DATA);
                let q = (next(&mut random) % NUM_DATA as u64) as usize;
                mast.rz(Angle64::QUARTER_TURN / 2_u64, &[QubitId(q)]);
            }
            mast.project_all();
            let before = mast.stats;
            let post_start = Instant::now();

            for injection in 0..POST_INJECTIONS {
                apply_random_clifford_mast(&mut mast, &mut random, NUM_DATA);
                let q = (next(&mut random) % NUM_DATA as u64) as usize;
                let angle = if injection & 1 == 0 {
                    Angle64::QUARTER_TURN / 2_u64
                } else {
                    Angle64::from_radians(0.37)
                };
                mast.rz(angle, &[QubitId(q)]);
                post_rotations += 1;
                if injection % 2 == 1 {
                    mast.project_all();
                }
            }
            mast.project_all();
            elapsed += post_start.elapsed().as_secs_f64();
            counts.add_delta(mast.stats, before);
        }
    }

    (counts, post_rotations, elapsed)
}

fn run_fallback_microbenchmark() -> (usize, usize, f64, f64) {
    const NUM_SITES: usize = 12;
    const ITERATIONS: usize = 2_000;

    // Build a seeded, non-canonical random MPS with chi=6 at every internal
    // bond. Every site is measured directly here, so this isolates the exact
    // fallback tier rather than its cheap eligibility checks.
    let product = Mps::new(NUM_SITES, MpsConfig::default());
    let bond_two = product.add(&product);
    let bond_four = bond_two.add(&bond_two);
    let mut mps = bond_four.add(&bond_two);
    let mut random = 0xfabb_ac1e_2026_0818_u64;
    for tensor in mps.tensors_mut() {
        for value in tensor.iter_mut() {
            let real = (next(&mut random) as f64 / u64::MAX as f64) - 0.5;
            let imag = (next(&mut random) as f64 / u64::MAX as f64) - 0.5;
            *value = Complex64::new(real, imag);
        }
    }
    let projector_one = DMatrix::from_row_slice(
        2,
        2,
        &[
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ],
    );

    let reference_start = Instant::now();
    for _ in 0..ITERATIONS {
        let norm_squared = black_box(&mps).norm_squared();
        for site in 0..NUM_SITES {
            let weight = black_box(&mps)
                .expectation_product(&[(site, projector_one.clone())])
                .re;
            black_box(weight / norm_squared);
        }
    }
    let reference_ns_per_candidate =
        reference_start.elapsed().as_secs_f64() * 1e9 / (ITERATIONS * NUM_SITES) as f64;

    let batched_start = Instant::now();
    let sites: Vec<usize> = (0..NUM_SITES).collect();
    for _ in 0..ITERATIONS {
        let environments = black_box(&mps).environment_cache(&sites);
        for site in 0..NUM_SITES {
            black_box(environments.one_site_basis_marginal(site, 1));
        }
    }
    let batched_ns_per_candidate =
        batched_start.elapsed().as_secs_f64() * 1e9 / (ITERATIONS * NUM_SITES) as f64;

    (
        NUM_SITES,
        mps.max_bond_dim(),
        reference_ns_per_candidate,
        batched_ns_per_candidate,
    )
}

fn main() {
    let repeats = std::env::var("PROBE_REPEATS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1);
    let mode = std::env::var("PROBE_MODE").unwrap_or_else(|_| "all".to_owned());

    if mode == "all" || mode == "stn" {
        let (stn, rotations, trajectory_hash, stn_seconds) = run_stn(repeats);
        println!(
            "STN repeats={repeats} configs=16 rotations={rotations} multi_disent={} multi_std={} single_site={} numerical_redetect={} fast_rate={:.3}% trajectory_hash={trajectory_hash:016x} wall_s={stn_seconds:.6}",
            stn.multi_disent,
            stn.multi_std,
            stn.single_site,
            stn.numerical_redetect,
            stn.fast_rate(),
        );
    }

    if mode == "all" || mode == "mast" {
        let (mast, post_rotations, mast_seconds) = run_mast_post_projection(repeats);
        println!(
            "MAST_POST repeats={repeats} rotations={post_rotations} multi_disent={} multi_std={} single_site={} numerical_redetect={} fast_rate={:.3}% wall_s={mast_seconds:.6}",
            mast.multi_disent,
            mast.multi_std,
            mast.single_site,
            mast.numerical_redetect,
            mast.fast_rate(),
        );
    }

    if mode == "all" || mode == "micro" {
        let (sites, bond, reference_ns, batched_ns) = run_fallback_microbenchmark();
        println!(
            "FALLBACK_MICRO sites={sites} bond={bond} candidates={sites} reference_ns_per_candidate={reference_ns:.3} batched_ns_per_candidate={batched_ns:.3}"
        );
    }
}
