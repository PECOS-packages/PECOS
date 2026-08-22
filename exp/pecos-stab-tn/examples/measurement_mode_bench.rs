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

//! Frozen Stage-B benchmark for the single-qubit measurement policy.
//!
//! Run pinned to one CPU with:
//! `taskset -c 2 cargo run --release -p pecos-stab-tn --example measurement_mode_bench`.
//!
//! Each repetition-code syndrome circuit is expanded here rather than using
//! `extract_syndromes`, so the independently sampled two-qubit Pauli noise has
//! an explicit insertion point after every entangler.
//! Bond telemetry is the maximum across a timed run's shots; discarded weight
//! and event counters are summed across those shots. Each reported field is
//! the median of the seven timed-run aggregates.

use pecos_core::{Angle64, QubitId};
use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable};
use pecos_stab_tn::stab_mps::{MeasurementMode, StabMps};
use rand_chacha::ChaCha12Rng;
use rand_chacha::rand_core::{RngCore, SeedableRng};
use std::hint::black_box;
use std::time::Instant;

const ROUNDS: usize = 8;
const COHERENT_RZ_RADIANS: f64 = 0.1;
const TWO_QUBIT_DEPOLARIZING_PROBABILITY: f64 = 1e-3;
// Largest multiple of 15 below 2^32; reject the one remaining u32 value.
const DEPOLARIZING_CHOICE_ZONE: u32 = 4_294_967_295;
const TIMED_RUN_SEEDS: std::ops::RangeInclusive<u64> = 7001..=7007;
const WARMUP_SEED: u64 = 7000;

#[derive(Clone, Copy, Debug)]
enum CheckBasis {
    Z,
    X,
}

impl CheckBasis {
    const fn label(self) -> &'static str {
        match self {
            Self::Z => "z",
            Self::X => "x",
        }
    }
}

#[derive(Clone, Copy)]
struct Workload {
    distance: usize,
    shots: usize,
    basis: CheckBasis,
}

impl Workload {
    fn label(self) -> String {
        format!("d{}_{}", self.distance, self.basis.label())
    }

    const fn num_qubits(self) -> usize {
        2 * self.distance - 1
    }

    const fn entanglers_per_shot(self) -> usize {
        ROUNDS * (self.distance - 1) * 2
    }
}

const WORKLOADS: [Workload; 4] = [
    Workload {
        distance: 3,
        shots: 2_000,
        basis: CheckBasis::Z,
    },
    Workload {
        distance: 3,
        shots: 2_000,
        basis: CheckBasis::X,
    },
    Workload {
        distance: 5,
        shots: 500,
        basis: CheckBasis::Z,
    },
    Workload {
        distance: 5,
        shots: 500,
        basis: CheckBasis::X,
    },
];

#[derive(Clone, Copy, Debug)]
enum Pauli {
    I,
    X,
    Y,
    Z,
}

#[derive(Clone, Copy, Debug)]
struct TwoQubitNoise {
    first: Pauli,
    second: Pauli,
}

#[derive(Clone)]
struct ShotNoise {
    after_entanglers: Vec<TwoQubitNoise>,
}

#[derive(Clone, Copy, Debug, Default)]
struct RunMetrics {
    shots_per_second: f64,
    lifetime_peak_bond: usize,
    final_bond: usize,
    summed_discarded_weight: f64,
    cap_hit_count: u64,
    branch_vanish_retry_count: u64,
    uncompensated_pre_reduction_count: u64,
}

#[derive(Clone, Copy, Debug)]
struct MedianMetrics {
    shots_per_second: f64,
    lifetime_peak_bond: usize,
    final_bond: usize,
    summed_discarded_weight: f64,
    cap_hit_count: u64,
    branch_vanish_retry_count: u64,
    uncompensated_pre_reduction_count: u64,
}

fn depolarizing_error(rng: &mut ChaCha12Rng) -> TwoQubitNoise {
    let threshold = (TWO_QUBIT_DEPOLARIZING_PROBABILITY * (u64::MAX as f64)) as u64;
    if rng.next_u64() > threshold {
        return TwoQubitNoise {
            first: Pauli::I,
            second: Pauli::I,
        };
    }

    // Rejection sampling makes the fifteen non-identity two-qubit Paulis
    // equiprobable instead of introducing a modulo bias.
    let index = loop {
        let word = rng.next_u32();
        if word < DEPOLARIZING_CHOICE_ZONE {
            break word % 15 + 1;
        }
    };
    let decode = |digit| match digit {
        0 => Pauli::I,
        1 => Pauli::X,
        2 => Pauli::Y,
        3 => Pauli::Z,
        _ => unreachable!(),
    };
    TwoQubitNoise {
        first: decode(index / 4),
        second: decode(index % 4),
    }
}

fn pregenerate_noise(workload: Workload, run_seed: u64) -> Vec<ShotNoise> {
    let mut rng = ChaCha12Rng::seed_from_u64(run_seed);
    (0..workload.shots)
        .map(|_| ShotNoise {
            after_entanglers: (0..workload.entanglers_per_shot())
                .map(|_| depolarizing_error(&mut rng))
                .collect(),
        })
        .collect()
}

fn apply_pauli(stn: &mut StabMps, qubit: QubitId, pauli: Pauli) {
    match pauli {
        Pauli::I => {}
        Pauli::X => {
            stn.x(&[qubit]);
        }
        Pauli::Y => {
            stn.y(&[qubit]);
        }
        Pauli::Z => {
            stn.z(&[qubit]);
        }
    }
}

fn apply_entangler(
    stn: &mut StabMps,
    basis: CheckBasis,
    ancilla: QubitId,
    data: QubitId,
    noise: TwoQubitNoise,
) {
    match basis {
        CheckBasis::Z => {
            stn.cz(&[(ancilla, data)]);
        }
        CheckBasis::X => {
            stn.cx(&[(ancilla, data)]);
        }
    }
    apply_pauli(stn, ancilla, noise.first);
    apply_pauli(stn, data, noise.second);
}

fn prepare_code_state(stn: &mut StabMps, workload: Workload) {
    match workload.basis {
        CheckBasis::Z => {
            // The encoded logical |+> is a GHZ state stabilized by every
            // adjacent ZZ check and is sensitive to the prescribed RZ noise.
            stn.h(&[QubitId(0)]);
            for q in 1..workload.distance {
                stn.cx(&[(QubitId(0), QubitId(q))]);
            }
        }
        CheckBasis::X => {
            // The encoded logical |0> is |+>^d and satisfies every adjacent
            // XX check while remaining sensitive to RZ noise.
            for q in 0..workload.distance {
                stn.h(&[QubitId(q)]);
            }
        }
    }
}

fn run_shot(
    workload: Workload,
    mode: MeasurementMode,
    simulator_seed: u64,
    noise: &ShotNoise,
) -> (StabMps, u64) {
    let mut stn = StabMps::builder(workload.num_qubits())
        .seed(simulator_seed)
        .measurement(mode)
        .merge_rz(false)
        .build();
    prepare_code_state(&mut stn, workload);

    let rz = Angle64::from_radians(COHERENT_RZ_RADIANS);
    let mut noise_cursor = 0;
    let mut syndrome_checksum = 0_u64;
    for round in 0..ROUNDS {
        for data in 0..workload.distance {
            stn.rz(rz, &[QubitId(data)]);
        }

        for check in 0..workload.distance - 1 {
            let ancilla = QubitId(workload.distance + check);
            stn.h(&[ancilla]);
            for data in [check, check + 1] {
                apply_entangler(
                    &mut stn,
                    workload.basis,
                    ancilla,
                    QubitId(data),
                    noise.after_entanglers[noise_cursor],
                );
                noise_cursor += 1;
            }
            stn.h(&[ancilla]);
            let outcome = stn.mz(&[ancilla])[0].outcome;
            syndrome_checksum ^= u64::from(outcome) << ((round + check) % 64);
            stn.reset_qubit(ancilla);
        }
    }
    debug_assert_eq!(noise_cursor, noise.after_entanglers.len());
    (stn, syndrome_checksum)
}

fn run_once(
    workload: Workload,
    mode: MeasurementMode,
    run_seed: u64,
    noise: &[ShotNoise],
) -> RunMetrics {
    assert_eq!(noise.len(), workload.shots);
    let start = Instant::now();
    let mut metrics = RunMetrics::default();
    let mut checksum = 0_u64;
    for (shot, shot_noise) in noise.iter().enumerate() {
        let simulator_seed = 10_000 * run_seed + shot as u64;
        let (stn, shot_checksum) = run_shot(workload, mode, simulator_seed, shot_noise);
        checksum ^= shot_checksum;
        metrics.lifetime_peak_bond = metrics.lifetime_peak_bond.max(stn.lifetime_peak_bond());
        metrics.final_bond = metrics.final_bond.max(stn.max_bond_dim());
        metrics.summed_discarded_weight += stn.summed_discarded_weight();
        metrics.cap_hit_count += stn.bond_cap_hits();
        metrics.branch_vanish_retry_count += stn.branch_vanish_retry_count();
        metrics.uncompensated_pre_reduction_count += stn.uncompensated_pre_reduction_count();
    }
    metrics.shots_per_second = workload.shots as f64 / start.elapsed().as_secs_f64();
    black_box(checksum);
    metrics
}

fn median_f64(values: impl Iterator<Item = f64>) -> f64 {
    let mut values = values.collect::<Vec<_>>();
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn median_usize(values: impl Iterator<Item = usize>) -> usize {
    let mut values = values.collect::<Vec<_>>();
    values.sort_unstable();
    values[values.len() / 2]
}

fn median_u64(values: impl Iterator<Item = u64>) -> u64 {
    let mut values = values.collect::<Vec<_>>();
    values.sort_unstable();
    values[values.len() / 2]
}

fn medians(runs: &[RunMetrics]) -> MedianMetrics {
    MedianMetrics {
        shots_per_second: median_f64(runs.iter().map(|r| r.shots_per_second)),
        lifetime_peak_bond: median_usize(runs.iter().map(|r| r.lifetime_peak_bond)),
        final_bond: median_usize(runs.iter().map(|r| r.final_bond)),
        summed_discarded_weight: median_f64(runs.iter().map(|r| r.summed_discarded_weight)),
        cap_hit_count: median_u64(runs.iter().map(|r| r.cap_hit_count)),
        branch_vanish_retry_count: median_u64(runs.iter().map(|r| r.branch_vanish_retry_count)),
        uncompensated_pre_reduction_count: median_u64(
            runs.iter().map(|r| r.uncompensated_pre_reduction_count),
        ),
    }
}

fn mode_label(mode: MeasurementMode) -> &'static str {
    match mode {
        MeasurementMode::Exact => "exact",
        MeasurementMode::Pragmatic => "pragmatic",
        MeasurementMode::Lazy => unreachable!("the frozen benchmark compares two modes"),
    }
}

fn print_metrics(workload: Workload, mode: MeasurementMode, metrics: MedianMetrics) {
    println!(
        "MEASUREMENT_MODE_BENCH workload={} mode={} shots={} runs=7 shots_per_s={:.6} lifetime_peak_bond={} final_bond={} summed_discarded_weight={:.16e} cap_hit_count={} branch_vanish_retry_count={} uncompensated_pre_reduction_count={}",
        workload.label(),
        mode_label(mode),
        workload.shots,
        metrics.shots_per_second,
        metrics.lifetime_peak_bond,
        metrics.final_bond,
        metrics.summed_discarded_weight,
        metrics.cap_hit_count,
        metrics.branch_vanish_retry_count,
        metrics.uncompensated_pre_reduction_count,
    );
}

fn main() {
    let mut slowdowns = Vec::with_capacity(WORKLOADS.len());
    for workload in WORKLOADS {
        let warmup_noise = pregenerate_noise(workload, WARMUP_SEED);
        for mode in [MeasurementMode::Exact, MeasurementMode::Pragmatic] {
            black_box(run_once(workload, mode, WARMUP_SEED, &warmup_noise));
        }

        let mut exact_runs = Vec::with_capacity(TIMED_RUN_SEEDS.count());
        let mut pragmatic_runs = Vec::with_capacity(TIMED_RUN_SEEDS.count());
        for run_seed in TIMED_RUN_SEEDS {
            let noise = pregenerate_noise(workload, run_seed);
            // Alternate the execution order to balance any drift between the
            // two modes without changing their shared workload.
            if run_seed % 2 == 0 {
                pragmatic_runs.push(run_once(
                    workload,
                    MeasurementMode::Pragmatic,
                    run_seed,
                    &noise,
                ));
                exact_runs.push(run_once(workload, MeasurementMode::Exact, run_seed, &noise));
            } else {
                exact_runs.push(run_once(workload, MeasurementMode::Exact, run_seed, &noise));
                pragmatic_runs.push(run_once(
                    workload,
                    MeasurementMode::Pragmatic,
                    run_seed,
                    &noise,
                ));
            }
        }

        let exact = medians(&exact_runs);
        let pragmatic = medians(&pragmatic_runs);
        print_metrics(workload, MeasurementMode::Exact, exact);
        print_metrics(workload, MeasurementMode::Pragmatic, pragmatic);
        slowdowns.push(pragmatic.shots_per_second / exact.shots_per_second);
    }

    let geometric_mean =
        (slowdowns.iter().map(|value| value.ln()).sum::<f64>() / slowdowns.len() as f64).exp();
    let maximum = slowdowns.iter().copied().fold(0.0_f64, f64::max);
    let decision = if geometric_mean > 10.0 {
        "stop"
    } else if geometric_mean <= 3.0 && maximum <= 5.0 {
        "exact_everywhere"
    } else {
        "middle"
    };
    println!(
        "MEASUREMENT_MODE_DECISION slowdowns={:.6},{:.6},{:.6},{:.6} geometric_mean_slowdown={geometric_mean:.6} max_slowdown={maximum:.6} decision={decision}",
        slowdowns[0], slowdowns[1], slowdowns[2], slowdowns[3]
    );
}
