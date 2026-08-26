//! Stage-1 telemetry harness for the five saturated-regime campaign cells.
//!
//! This is a Rust port of the deterministic `sparse_t_circuit`,
//! `dense_rotation_circuit`, and `_queries` generators in pecos-perf's STN
//! campaign. It deliberately implements `CPython`'s integer-seeded MT19937 and
//! `_randbelow` rules so the gate streams and query rows remain identical.
//!
//! ```text
//! taskset -c 2 cargo run --release -p pecos-stab-tn --example saturated_regime_profile
//! ```
//!
//! Select one cell with `SATURATION_CELL` and control the campaign protocol
//! with `SATURATION_WARMUPS` (default 1) and `SATURATION_REPETITIONS` (default 5).

use pecos_core::{Angle64, QubitId};
use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable};
use pecos_stab_tn::stab_mps::{
    MultiStdSubtype, ProbabilityQueryTelemetry, SaturationTelemetry, StabMps, StabMpsStats,
};
use std::collections::HashSet;
use std::time::Instant;

const QUERY_COUNT: usize = 16;

#[derive(Clone, Copy)]
enum Gate {
    H(usize),
    S(usize),
    Cx(usize, usize),
    Cz(usize, usize),
    T(usize),
    Tdg(usize),
    Rz(f64, usize),
}

#[derive(Clone, Copy)]
enum Family {
    SparseT { n_t: usize },
    DenseRotation { layers: usize },
}

#[derive(Clone, Copy)]
struct Cell {
    name: &'static str,
    n: usize,
    seed: u64,
    family: Family,
    campaign_content_hash: &'static str,
    query_available: bool,
}

const CELLS: [Cell; 5] = [
    Cell {
        name: "sparse-n16-t2n",
        n: 16,
        seed: 21_602,
        family: Family::SparseT { n_t: 32 },
        campaign_content_hash: "8ca2806e891b128cfc50fbaa7d33e2617ce96ad783c0f0e02a4f4d0eb69de3a7",
        query_available: true,
    },
    Cell {
        name: "sparse-n32-tn",
        n: 32,
        seed: 23_201,
        family: Family::SparseT { n_t: 32 },
        campaign_content_hash: "31866e8b94c073969e8be2a50e897bd5e078eb520663768e5c63bf3307b15a84",
        query_available: true,
    },
    Cell {
        name: "sparse-n32-t2n",
        n: 32,
        seed: 23_202,
        family: Family::SparseT { n_t: 64 },
        campaign_content_hash: "9f66034d85dc8f4296f04704f153ec0dd1aa290b8d6fedcd3811b1d3b5f6ca09",
        query_available: true,
    },
    Cell {
        name: "sparse-n64-t2n",
        n: 64,
        seed: 26_402,
        family: Family::SparseT { n_t: 128 },
        campaign_content_hash: "f1f4846a69f8a6d5aac115f27e653e5dd490de1459e10049151f5ec78cdacee7",
        query_available: false,
    },
    Cell {
        name: "dense-n16-l6",
        n: 16,
        seed: 40_016,
        family: Family::DenseRotation { layers: 6 },
        campaign_content_hash: "690dee8fb46dacd2aafae818b1d99efe29fa451c300f22ffe3b2eb1eadf830c6",
        query_available: true,
    },
];

struct PythonRandom {
    state: [u32; 624],
    index: usize,
}

impl PythonRandom {
    fn new(seed: u64) -> Self {
        let mut random = Self {
            state: [0; 624],
            index: 624,
        };
        random.init_genrand(19_650_218);
        let low = seed as u32;
        let high = (seed >> 32) as u32;
        let key = if high == 0 {
            vec![low]
        } else {
            vec![low, high]
        };
        random.init_by_array(&key);
        random
    }

    fn init_genrand(&mut self, seed: u32) {
        self.state[0] = seed;
        for i in 1..624 {
            self.state[i] = 1_812_433_253_u32
                .wrapping_mul(self.state[i - 1] ^ (self.state[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        self.index = 624;
    }

    fn init_by_array(&mut self, key: &[u32]) {
        let mut i = 1;
        let mut j = 0;
        for _ in 0..624.max(key.len()) {
            self.state[i] = (self.state[i]
                ^ (self.state[i - 1] ^ (self.state[i - 1] >> 30)).wrapping_mul(1_664_525))
            .wrapping_add(key[j])
            .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= 624 {
                self.state[0] = self.state[623];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
        }
        for _ in 0..623 {
            self.state[i] = (self.state[i]
                ^ (self.state[i - 1] ^ (self.state[i - 1] >> 30)).wrapping_mul(1_566_083_941))
            .wrapping_sub(i as u32);
            i += 1;
            if i >= 624 {
                self.state[0] = self.state[623];
                i = 1;
            }
        }
        self.state[0] = 0x8000_0000;
    }

    fn next_u32(&mut self) -> u32 {
        if self.index >= 624 {
            const UPPER_MASK: u32 = 0x8000_0000;
            const LOWER_MASK: u32 = 0x7fff_ffff;
            for i in 0..624 {
                let y = (self.state[i] & UPPER_MASK) | (self.state[(i + 1) % 624] & LOWER_MASK);
                self.state[i] = self.state[(i + 397) % 624]
                    ^ (y >> 1)
                    ^ if y & 1 == 0 { 0 } else { 0x9908_b0df };
            }
            self.index = 0;
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    fn getrandbits(&mut self, bits: u32) -> usize {
        debug_assert!((1..=32).contains(&bits));
        (self.next_u32() >> (32 - bits)) as usize
    }

    fn randbelow(&mut self, limit: usize) -> usize {
        assert!(limit > 0);
        let bits = usize::BITS - limit.leading_zeros();
        loop {
            let value = self.getrandbits(bits);
            if value < limit {
                return value;
            }
        }
    }

    fn random(&mut self) -> f64 {
        let high = u64::from(self.next_u32() >> 5);
        let low = u64::from(self.next_u32() >> 6);
        ((high << 26) + low) as f64 / 9_007_199_254_740_992.0
    }
}

fn sparse_t_circuit(n: usize, n_t: usize, seed: u64) -> Vec<Gate> {
    let mut random = PythonRandom::new(seed);
    let mut gates = Vec::new();
    gates.extend((0..n).map(Gate::H));
    gates.extend((0..n - 1).map(|q| Gate::Cx(q, q + 1)));
    for injection in 0..n_t {
        let target = random.randbelow(n);
        let other_index = random.randbelow(n - 1);
        let other = if other_index >= target {
            other_index + 1
        } else {
            other_index
        };
        gates.push(if random.randbelow(2) == 0 {
            Gate::T(target)
        } else {
            Gate::Tdg(target)
        });
        gates.push(Gate::H(target));
        gates.push(if injection % 2 == 0 {
            Gate::Cx(target, other)
        } else {
            Gate::Cz(target, other)
        });
        if injection % 4 == 3 {
            gates.push(Gate::S(random.randbelow(n)));
        }
    }
    gates
}

fn dense_rotation_circuit(n: usize, layers: usize, seed: u64) -> Vec<Gate> {
    let mut random = PythonRandom::new(seed);
    let mut gates = Vec::new();
    for layer in 0..layers {
        gates.extend((0..n).map(Gate::H));
        for parity in [layer % 2, 1 - layer % 2] {
            gates.extend((parity..n - 1).step_by(2).map(|q| Gate::Cx(q, q + 1)));
        }
        for q in 0..n {
            let angle = (0.13 + 0.74 * random.random()) * std::f64::consts::PI;
            gates.push(Gate::Rz(angle, q));
        }
    }
    gates
}

fn gates(cell: Cell) -> Vec<Gate> {
    match cell.family {
        Family::SparseT { n_t } => sparse_t_circuit(cell.n, n_t, cell.seed),
        Family::DenseRotation { layers } => dense_rotation_circuit(cell.n, layers, cell.seed),
    }
}

fn queries(n: usize, seed: u64) -> Vec<Vec<bool>> {
    let mut random = PythonRandom::new(1_000_000 + seed);
    let mut seen = HashSet::new();
    let mut rows = Vec::new();
    while rows.len() < QUERY_COUNT {
        let row: Vec<bool> = (0..n).map(|_| random.randbelow(2) != 0).collect();
        if seen.insert(row.clone()) {
            rows.push(row);
        }
    }
    rows
}

fn apply_gate(simulator: &mut StabMps, gate: Gate) {
    match gate {
        Gate::H(q) => {
            simulator.h(&[QubitId(q)]);
        }
        Gate::S(q) => {
            simulator.sz(&[QubitId(q)]);
        }
        Gate::Cx(control, target) => {
            simulator.cx(&[(QubitId(control), QubitId(target))]);
        }
        Gate::Cz(first, second) => {
            simulator.cz(&[(QubitId(first), QubitId(second))]);
        }
        Gate::T(q) => {
            simulator.t(&[QubitId(q)]);
        }
        Gate::Tdg(q) => {
            simulator.tdg(&[QubitId(q)]);
        }
        Gate::Rz(angle, q) => {
            simulator.rz(Angle64::from_radians(angle), &[QubitId(q)]);
        }
    }
}

fn simulate(cell: Cell, collect_telemetry: bool) -> (StabMps, f64) {
    let started = Instant::now();
    let mut simulator = StabMps::builder(cell.n)
        .seed(cell.seed)
        .max_bond_dim(64)
        .svd_cutoff(1e-12)
        .max_truncation_error(0.0)
        .merge_rz(true)
        .numerical_flag_redetection(true)
        .saturation_telemetry(collect_telemetry)
        .build();
    for gate in gates(cell) {
        apply_gate(&mut simulator, gate);
    }
    simulator.flush();
    (simulator, started.elapsed().as_secs_f64())
}

fn mix_hash(hash: &mut u64, value: u64) {
    *hash ^= value;
    *hash = hash.wrapping_mul(0x100_0000_01b3);
}

fn probability_hash(probabilities: &[f64]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325;
    for probability in probabilities {
        mix_hash(&mut hash, probability.to_bits());
    }
    hash
}

#[derive(Clone, Default)]
struct RunSummary {
    sim_seconds: f64,
    query_seconds: Option<f64>,
    expectation_seconds: f64,
    pre_reduction_seconds: f64,
    projection_seconds: f64,
    post_projection_seconds: f64,
    cascade_seconds: f64,
    add_seconds: f64,
    disent_seconds: f64,
    ofd_avoidable_seconds: f64,
    stats: StabMpsStats,
    output_hash: Option<u64>,
}

fn query_totals(profile: &ProbabilityQueryTelemetry) -> (f64, f64, f64, f64, u64, u64) {
    profile.by_depth.iter().fold(
        (0.0, 0.0, 0.0, 0.0, 0, 0),
        |(expectation, pre, projection, post, svds, capped), depth| {
            (
                expectation + depth.expectation.wall_time_seconds,
                pre + depth.pre_reduction.wall_time_seconds,
                projection + depth.projection.wall_time_seconds,
                post + depth.post_projection.wall_time_seconds,
                svds + depth.expectation.svd_operations
                    + depth.pre_reduction.svd_operations
                    + depth.projection.svd_operations
                    + depth.post_projection.svd_operations,
                capped
                    + depth.expectation.capped_svd_operations
                    + depth.pre_reduction.capped_svd_operations
                    + depth.projection.capped_svd_operations
                    + depth.post_projection.capped_svd_operations,
            )
        },
    )
}

fn print_query_depths(cell: Cell, run: usize, profile: &ProbabilityQueryTelemetry) {
    for (depth, bucket) in profile.by_depth.iter().enumerate() {
        println!(
            "DEPTH cell={} run={} depth={} calls={} expectation_s={:.9} expectation_svds={} expectation_capped={} pre_s={:.9} pre_svds={} pre_capped={} projection_s={:.9} projection_svds={} projection_capped={} post_s={:.9} post_svds={} post_capped={}",
            cell.name,
            run,
            depth,
            bucket.expectation.calls,
            bucket.expectation.wall_time_seconds,
            bucket.expectation.svd_operations,
            bucket.expectation.capped_svd_operations,
            bucket.pre_reduction.wall_time_seconds,
            bucket.pre_reduction.svd_operations,
            bucket.pre_reduction.capped_svd_operations,
            bucket.projection.wall_time_seconds,
            bucket.projection.svd_operations,
            bucket.projection.capped_svd_operations,
            bucket.post_projection.wall_time_seconds,
            bucket.post_projection.svd_operations,
            bucket.post_projection.capped_svd_operations,
        );
    }
}

fn print_events(cell: Cell, run: usize, telemetry: &SaturationTelemetry) {
    for (event_index, event) in telemetry.multi_std_events.iter().enumerate() {
        let subtype = match event.subtype {
            MultiStdSubtype::Add => "add",
            MultiStdSubtype::Cascade => "cascade",
        };
        let bonds = event
            .bond_profile
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",");
        println!(
            "EVENT cell={} run={} kind=multi_std_{} index={} span={} bonds={} ofd_in_span={} wall_s={:.9}",
            cell.name,
            run,
            subtype,
            event_index,
            event.span,
            bonds,
            event.ofd_in_span,
            event.wall_time_seconds,
        );
    }
    for (event_index, event) in telemetry.multi_disent_events.iter().enumerate() {
        let bonds = event
            .bond_profile
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",");
        println!(
            "EVENT cell={} run={} kind=multi_disent index={} span={} bonds={} wall_s={:.9}",
            cell.name, run, event_index, event.span, bonds, event.wall_time_seconds,
        );
    }
}

fn run_profiled(cell: Cell, run: usize) -> RunSummary {
    let (simulator, sim_seconds) = simulate(cell, true);
    let telemetry = simulator.saturation_profile();
    let cascade_seconds = telemetry
        .multi_std_events
        .iter()
        .filter(|event| event.subtype == MultiStdSubtype::Cascade)
        .map(|event| event.wall_time_seconds)
        .sum();
    let add_seconds = telemetry
        .multi_std_events
        .iter()
        .filter(|event| event.subtype == MultiStdSubtype::Add)
        .map(|event| event.wall_time_seconds)
        .sum();
    let disent_seconds = telemetry
        .multi_disent_events
        .iter()
        .map(|event| event.wall_time_seconds)
        .sum();
    let ofd_avoidable_seconds = telemetry
        .multi_std_events
        .iter()
        .filter(|event| event.ofd_in_span)
        .map(|event| event.wall_time_seconds)
        .sum();
    print_events(cell, run, telemetry);

    let mut summary = RunSummary {
        sim_seconds,
        cascade_seconds,
        add_seconds,
        disent_seconds,
        ofd_avoidable_seconds,
        stats: simulator.stats,
        ..RunSummary::default()
    };
    if cell.query_available {
        let query_started = Instant::now();
        let (probabilities, profile) =
            simulator.prob_bitstrings_profiled(&queries(cell.n, cell.seed));
        summary.query_seconds = Some(query_started.elapsed().as_secs_f64());
        summary.output_hash = Some(probability_hash(&probabilities));
        let (expectation, pre, projection, post, svds, capped) = query_totals(&profile);
        summary.expectation_seconds = expectation;
        summary.pre_reduction_seconds = pre;
        summary.projection_seconds = projection;
        summary.post_projection_seconds = post;
        print_query_depths(cell, run, &profile);
        println!(
            "QUERY_OPS cell={} run={} svds={} capped_svds={}",
            cell.name, run, svds, capped
        );
    }
    println!(
        "RUN cell={} run={} sim_s={:.9} query_s={} multi_std={} multi_std_add={} multi_std_cascade={} multi_disent={} signed_candidates={} ofd_in_span_std={} expectation_s={:.9} pre_s={:.9} projection_s={:.9} post_s={:.9} cascade_s={:.9} add_s={:.9} disent_s={:.9} ofd_avoidable_s={:.9} output_hash={}",
        cell.name,
        run,
        summary.sim_seconds,
        summary.query_seconds.map_or_else(
            || "unavailable-issue-586".to_owned(),
            |value| format!("{value:.9}")
        ),
        summary.stats.multi_std,
        summary.stats.multi_std_add,
        summary.stats.multi_std_cascade,
        summary.stats.multi_disent,
        summary.stats.signed_eigenstate_candidates,
        summary.stats.ofd_in_span_std,
        summary.expectation_seconds,
        summary.pre_reduction_seconds,
        summary.projection_seconds,
        summary.post_projection_seconds,
        summary.cascade_seconds,
        summary.add_seconds,
        summary.disent_seconds,
        summary.ofd_avoidable_seconds,
        summary.output_hash.map_or_else(
            || "unavailable-issue-586".to_owned(),
            |hash| format!("{hash:016x}")
        ),
    );
    summary
}

fn median(values: impl Iterator<Item = f64>) -> f64 {
    let mut values: Vec<f64> = values.collect();
    values.sort_by(f64::total_cmp);
    if values.len().is_multiple_of(2) {
        f64::midpoint(values[values.len() / 2 - 1], values[values.len() / 2])
    } else {
        values[values.len() / 2]
    }
}

fn print_summary(cell: Cell, runs: &[RunSummary]) {
    let query_seconds = cell
        .query_available
        .then(|| median(runs.iter().filter_map(|run| run.query_seconds)));
    let first = &runs[0];
    assert!(runs.iter().all(|run| run.output_hash == first.output_hash));
    println!(
        "SUMMARY cell={} repetitions={} sim_median_s={:.9} query_median_s={} expectation_median_s={:.9} pre_median_s={:.9} projection_median_s={:.9} post_median_s={:.9} cascade_median_s={:.9} add_median_s={:.9} disent_median_s={:.9} ofd_avoidable_median_s={:.9} multi_std={} multi_std_add={} multi_std_cascade={} multi_disent={} signed_candidates={} ofd_in_span_std={} output_hash={}",
        cell.name,
        runs.len(),
        median(runs.iter().map(|run| run.sim_seconds)),
        query_seconds.map_or_else(
            || "unavailable-issue-586".to_owned(),
            |value| format!("{value:.9}")
        ),
        median(runs.iter().map(|run| run.expectation_seconds)),
        median(runs.iter().map(|run| run.pre_reduction_seconds)),
        median(runs.iter().map(|run| run.projection_seconds)),
        median(runs.iter().map(|run| run.post_projection_seconds)),
        median(runs.iter().map(|run| run.cascade_seconds)),
        median(runs.iter().map(|run| run.add_seconds)),
        median(runs.iter().map(|run| run.disent_seconds)),
        median(runs.iter().map(|run| run.ofd_avoidable_seconds)),
        first.stats.multi_std,
        first.stats.multi_std_add,
        first.stats.multi_std_cascade,
        first.stats.multi_disent,
        first.stats.signed_eigenstate_candidates,
        first.stats.ofd_in_span_std,
        first.output_hash.map_or_else(
            || "unavailable-issue-586".to_owned(),
            |hash| format!("{hash:016x}")
        ),
    );
}

fn selected_cells() -> Vec<Cell> {
    let selection = std::env::var("SATURATION_CELL").unwrap_or_else(|_| "all".to_owned());
    let cells: Vec<Cell> = CELLS
        .iter()
        .copied()
        .filter(|cell| selection == "all" || selection == cell.name)
        .collect();
    assert!(
        !cells.is_empty(),
        "SATURATION_CELL did not match a campaign cell"
    );
    cells
}

fn env_count(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn main() {
    // Stable anchors from CPython 3.13's random.Random. These fail before a
    // campaign run if the ported generator ever drifts.
    let mut random_anchor = PythonRandom::new(21_602);
    assert_eq!(random_anchor.randbelow(100), 86);
    let mut float_anchor = PythonRandom::new(40_016);
    assert_eq!(
        float_anchor.random().to_bits(),
        0.085_653_475_532_001_87_f64.to_bits()
    );

    let warmups = env_count("SATURATION_WARMUPS", 1);
    let repetitions = env_count("SATURATION_REPETITIONS", 5);
    assert!(repetitions > 0, "at least one timed repetition is required");
    for cell in selected_cells() {
        let gate_count = gates(cell).len();
        println!(
            "CELL cell={} n={} seed={} gates={} campaign_content_hash={} query_status={}",
            cell.name,
            cell.n,
            cell.seed,
            gate_count,
            cell.campaign_content_hash,
            if cell.query_available {
                "available"
            } else {
                "unavailable-issue-586"
            },
        );
        for warmup in 0..warmups {
            let (simulator, sim_seconds) = simulate(cell, false);
            let query_seconds = if cell.query_available {
                let started = Instant::now();
                let probabilities = simulator.prob_bitstrings(&queries(cell.n, cell.seed));
                let elapsed = started.elapsed().as_secs_f64();
                println!(
                    "WARMUP cell={} warmup={} sim_s={:.9} query_s={:.9} output_hash={:016x}",
                    cell.name,
                    warmup,
                    sim_seconds,
                    elapsed,
                    probability_hash(&probabilities),
                );
                Some(elapsed)
            } else {
                None
            };
            if query_seconds.is_none() {
                println!(
                    "WARMUP cell={} warmup={} sim_s={:.9} query_s=unavailable-issue-586",
                    cell.name, warmup, sim_seconds
                );
            }
        }
        let runs: Vec<RunSummary> = (0..repetitions)
            .map(|run| run_profiled(cell, run))
            .collect();
        print_summary(cell, &runs);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_random_port_matches_campaign_anchors() {
        let expected = [86, 57, 26, 48, 93, 52, 89, 74];
        let mut random = PythonRandom::new(21_602);
        assert_eq!(expected, expected.map(|_| random.randbelow(100)),);
    }

    #[test]
    fn ported_campaign_gate_counts_match_source() {
        assert_eq!(
            CELLS.map(|cell| gates(cell).len()),
            [135, 167, 271, 543, 282]
        );
    }
}
