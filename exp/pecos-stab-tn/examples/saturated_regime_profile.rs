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
    MultiStdSubtype, ProbabilityQueryTelemetry, ProjectionConstruction, SaturationTelemetry,
    SignedEigenstateBranchTelemetry, SignedEigenstateTelemetry, StabMps, StabMpsStats,
};
use std::collections::{BTreeMap, HashSet};
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
}

const CELLS: [Cell; 5] = [
    Cell {
        name: "sparse-n16-t2n",
        n: 16,
        seed: 21_602,
        family: Family::SparseT { n_t: 32 },
        campaign_content_hash: "8ca2806e891b128cfc50fbaa7d33e2617ce96ad783c0f0e02a4f4d0eb69de3a7",
    },
    Cell {
        name: "sparse-n32-tn",
        n: 32,
        seed: 23_201,
        family: Family::SparseT { n_t: 32 },
        campaign_content_hash: "31866e8b94c073969e8be2a50e897bd5e078eb520663768e5c63bf3307b15a84",
    },
    Cell {
        name: "sparse-n32-t2n",
        n: 32,
        seed: 23_202,
        family: Family::SparseT { n_t: 64 },
        campaign_content_hash: "9f66034d85dc8f4296f04704f153ec0dd1aa290b8d6fedcd3811b1d3b5f6ca09",
    },
    Cell {
        name: "sparse-n64-t2n",
        n: 64,
        seed: 26_402,
        family: Family::SparseT { n_t: 128 },
        campaign_content_hash: "f1f4846a69f8a6d5aac115f27e653e5dd490de1459e10049151f5ec78cdacee7",
    },
    Cell {
        name: "dense-n16-l6",
        n: 16,
        seed: 40_016,
        family: Family::DenseRotation { layers: 6 },
        campaign_content_hash: "690dee8fb46dacd2aafae818b1d99efe29fa451c300f22ffe3b2eb1eadf830c6",
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
    profiled_sim_seconds: f64,
    query_seconds: f64,
    expectation_seconds: f64,
    pre_reduction_seconds: f64,
    decomposition_seconds: f64,
    projection_seconds: f64,
    post_projection_qr_seconds: f64,
    post_projection_svd_seconds: f64,
    survival_seconds: f64,
    normalization_seconds: f64,
    bookkeeping_seconds: f64,
    query_residual_seconds: f64,
    cascade_seconds: f64,
    add_seconds: f64,
    disent_seconds: f64,
    ofd_avoidable_seconds: f64,
    stats: StabMpsStats,
    signed_eigenstates: SignedEigenstateTelemetry,
    output_hash: u64,
}

#[derive(Clone, Copy, Default)]
struct QueryTotals {
    expectation: f64,
    pre_reduction: f64,
    decomposition: f64,
    projection: f64,
    post_projection_qr: f64,
    post_projection_svd: f64,
    survival: f64,
    normalization: f64,
    bookkeeping: f64,
    svds: u64,
    capped_svds: u64,
}

fn query_totals(profile: &ProbabilityQueryTelemetry) -> QueryTotals {
    let mut totals = QueryTotals::default();
    for depth in &profile.by_depth {
        totals.expectation += depth.expectation.wall_time_seconds;
        totals.pre_reduction += depth.pre_reduction.wall_time_seconds;
        totals.decomposition += depth.decomposition.wall_time_seconds;
        totals.projection += depth.projection.wall_time_seconds;
        totals.post_projection_qr += depth.post_projection_qr.wall_time_seconds;
        totals.post_projection_svd += depth.post_projection_svd.wall_time_seconds;
        totals.survival += depth.survival.wall_time_seconds;
        totals.normalization += depth.normalization.wall_time_seconds;
        totals.bookkeeping += depth.bookkeeping.wall_time_seconds;
        for phase in [
            &depth.expectation,
            &depth.pre_reduction,
            &depth.decomposition,
            &depth.projection,
            &depth.post_projection_qr,
            &depth.post_projection_svd,
            &depth.survival,
            &depth.normalization,
            &depth.bookkeeping,
        ] {
            totals.svds += phase.svd_operations;
            totals.capped_svds += phase.capped_svd_operations;
        }
    }
    totals
}

fn projection_construction_label(construction: ProjectionConstruction) -> &'static str {
    match construction {
        ProjectionConstruction::ScalarScale => "scalar_scale",
        ProjectionConstruction::LocalBlockWrite => "block_write",
        ProjectionConstruction::DirectSum => "direct_sum",
    }
}

fn print_query_depths(cell: Cell, run: usize, profile: &ProbabilityQueryTelemetry) {
    for (depth, bucket) in profile.by_depth.iter().enumerate() {
        println!(
            "DEPTH cell={} run={} depth={} calls={} expectation_s={:.9} expectation_svds={} expectation_capped={} pre_s={:.9} pre_svds={} pre_capped={} decomposition_s={:.9} decomposition_svds={} decomposition_capped={} projection_s={:.9} projection_svds={} projection_capped={} qr_s={:.9} qr_svds={} qr_capped={} svd_s={:.9} svd_svds={} svd_capped={} survival_s={:.9} survival_svds={} survival_capped={} normalization_s={:.9} normalization_svds={} normalization_capped={} bookkeeping_s={:.9} bookkeeping_svds={} bookkeeping_capped={} attributed_s={:.9}",
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
            bucket.decomposition.wall_time_seconds,
            bucket.decomposition.svd_operations,
            bucket.decomposition.capped_svd_operations,
            bucket.projection.wall_time_seconds,
            bucket.projection.svd_operations,
            bucket.projection.capped_svd_operations,
            bucket.post_projection_qr.wall_time_seconds,
            bucket.post_projection_qr.svd_operations,
            bucket.post_projection_qr.capped_svd_operations,
            bucket.post_projection_svd.wall_time_seconds,
            bucket.post_projection_svd.svd_operations,
            bucket.post_projection_svd.capped_svd_operations,
            bucket.survival.wall_time_seconds,
            bucket.survival.svd_operations,
            bucket.survival.capped_svd_operations,
            bucket.normalization.wall_time_seconds,
            bucket.normalization.svd_operations,
            bucket.normalization.capped_svd_operations,
            bucket.bookkeeping.wall_time_seconds,
            bucket.bookkeeping.svd_operations,
            bucket.bookkeeping.capped_svd_operations,
            bucket.attributed_wall_time_seconds(),
        );
        for (event_index, event) in bucket.projection_qr_locality.iter().enumerate() {
            let option = |value: Option<usize>| {
                value.map_or_else(|| "none".to_owned(), |value| value.to_string())
            };
            let construction = projection_construction_label(event.construction);
            println!(
                "LOCALITY cell={} run={} depth={} event={} n={} center_pre_write={} center_pre_write_valid={} center_qr={} center_qr_valid={} construction={} touched_min={} touched_max={} touched_sites={} changed_tensor_min={} changed_tensor_max={} changed_tensors={} changed_bond_min={} changed_bond_max={} changed_bonds={} qr_sites={} qr_skippable={} normalization_preserved={}",
                cell.name,
                run,
                depth,
                event_index,
                event.chain_length,
                option(event.center_before_projection_write),
                event.center_before_projection_write_is_valid,
                option(event.center_before_qr),
                event.center_before_qr_is_valid,
                construction,
                option(event.touched_site_min),
                option(event.touched_site_max),
                event.touched_sites,
                option(event.changed_tensor_min),
                option(event.changed_tensor_max),
                event.changed_tensors,
                option(event.changed_bond_min),
                option(event.changed_bond_max),
                event.changed_bonds,
                event.qr_sites,
                event.qr_sites_skippable_by_locality,
                event
                    .normalization_preserved_center
                    .is_some_and(|preserved| preserved),
            );
        }
        if !bucket.projection_qr_locality.is_empty() {
            let mut constructions = BTreeMap::<&str, usize>::new();
            let mut distributions = BTreeMap::<_, usize>::new();
            let mut qr_sites = 0;
            let mut qr_skippable = 0;
            let mut normalization_losses = 0;
            for event in &bucket.projection_qr_locality {
                *constructions
                    .entry(projection_construction_label(event.construction))
                    .or_default() += 1;
                *distributions
                    .entry((
                        event.construction,
                        event.center_before_projection_write,
                        event.center_before_projection_write_is_valid,
                        event.center_before_qr,
                        event.center_before_qr_is_valid,
                        (
                            event.touched_site_min,
                            event.touched_site_max,
                            event.touched_sites,
                            event.changed_tensor_min,
                            event.changed_tensor_max,
                            event.changed_tensors,
                            event.changed_bond_min,
                            event.changed_bond_max,
                            event.changed_bonds,
                            event.qr_sites,
                            event.qr_sites_skippable_by_locality,
                        ),
                    ))
                    .or_default() += 1;
                qr_sites += event.qr_sites;
                qr_skippable += event.qr_sites_skippable_by_locality;
                normalization_losses +=
                    usize::from(event.normalization_preserved_center == Some(false));
            }
            let construction_counts = constructions
                .iter()
                .map(|(construction, count)| format!("{construction}:{count}"))
                .collect::<Vec<_>>()
                .join(",");
            println!(
                "LOCALITY_DEPTH cell={} run={} depth={} events={} constructions={} qr_sites={} qr_skippable={} headroom_fraction={:.9} normalization_losses={}",
                cell.name,
                run,
                depth,
                bucket.projection_qr_locality.len(),
                construction_counts,
                qr_sites,
                qr_skippable,
                qr_skippable as f64 / qr_sites as f64,
                normalization_losses,
            );
            for (distribution, count) in distributions {
                let (
                    construction,
                    center_pre_write,
                    center_pre_write_valid,
                    center_qr,
                    center_qr_valid,
                    (
                        touched_min,
                        touched_max,
                        touched_sites,
                        changed_min,
                        changed_max,
                        changed_tensors,
                        changed_bond_min,
                        changed_bond_max,
                        changed_bonds,
                        event_qr_sites,
                        event_qr_skippable,
                    ),
                ) = distribution;
                let option = |value: Option<usize>| {
                    value.map_or_else(|| "none".to_owned(), |value| value.to_string())
                };
                println!(
                    "LOCALITY_DIST cell={} run={} depth={} count={} construction={} center_pre_write={} center_pre_write_valid={} center_qr={} center_qr_valid={} touched_min={} touched_max={} touched_sites={} changed_tensor_min={} changed_tensor_max={} changed_tensors={} changed_bond_min={} changed_bond_max={} changed_bonds={} qr_sites={} qr_skippable={}",
                    cell.name,
                    run,
                    depth,
                    count,
                    projection_construction_label(construction),
                    option(center_pre_write),
                    center_pre_write_valid,
                    option(center_qr),
                    center_qr_valid,
                    option(touched_min),
                    option(touched_max),
                    touched_sites,
                    option(changed_min),
                    option(changed_max),
                    changed_tensors,
                    option(changed_bond_min),
                    option(changed_bond_max),
                    changed_bonds,
                    event_qr_sites,
                    event_qr_skippable,
                );
            }
        }
    }
}

fn print_locality_total(cell: Cell, run: usize, profile: &ProbabilityQueryTelemetry) {
    let events = profile
        .by_depth
        .iter()
        .flat_map(|depth| &depth.projection_qr_locality)
        .collect::<Vec<_>>();
    if events.is_empty() {
        return;
    }
    let mut constructions = BTreeMap::<&str, usize>::new();
    let mut qr_sites = 0;
    let mut qr_skippable = 0;
    let mut normalization_losses = 0;
    for event in &events {
        *constructions
            .entry(projection_construction_label(event.construction))
            .or_default() += 1;
        qr_sites += event.qr_sites;
        qr_skippable += event.qr_sites_skippable_by_locality;
        normalization_losses += usize::from(event.normalization_preserved_center == Some(false));
    }
    let construction_counts = constructions
        .iter()
        .map(|(construction, count)| format!("{construction}:{count}"))
        .collect::<Vec<_>>()
        .join(",");
    println!(
        "LOCALITY_TOTAL cell={} run={} events={} constructions={} qr_sites={} qr_skippable={} headroom_fraction={:.9} normalization_losses={}",
        cell.name,
        run,
        events.len(),
        construction_counts,
        qr_sites,
        qr_skippable,
        qr_skippable as f64 / qr_sites as f64,
        normalization_losses,
    );
}

fn print_candidate_branch(
    cell: Cell,
    run: usize,
    branch: &str,
    telemetry: SignedEigenstateBranchTelemetry,
) {
    let candidates = telemetry.candidates;
    println!(
        "CANDIDATE cell={} run={} branch={} events={} sites_tested={} candidates={} x_plus={} x_minus={} y_plus={} y_minus={} z_minus={}",
        cell.name,
        run,
        branch,
        telemetry.events,
        telemetry.sites_tested,
        candidates.total(),
        candidates.x_plus,
        candidates.x_minus,
        candidates.y_plus,
        candidates.y_minus,
        candidates.z_minus,
    );
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
    print_candidate_branch(
        cell,
        run,
        "multi_disent",
        telemetry.signed_eigenstates.multi_disent,
    );
    print_candidate_branch(
        cell,
        run,
        "multi_std_add",
        telemetry.signed_eigenstates.multi_std_add,
    );
    print_candidate_branch(
        cell,
        run,
        "multi_std_cascade",
        telemetry.signed_eigenstates.multi_std_cascade,
    );
}

fn run_profiled(cell: Cell, run: usize) -> RunSummary {
    let (_, sim_seconds) = simulate(cell, false);
    let (simulator, profiled_sim_seconds) = simulate(cell, true);
    let telemetry = simulator.saturation_profile();
    let cascade_seconds = telemetry
        .multi_std_events
        .iter()
        .filter(|event| event.subtype == MultiStdSubtype::Cascade)
        .map(|event| event.wall_time_seconds)
        .fold(0.0_f64, |total, seconds| total + seconds);
    let add_seconds = telemetry
        .multi_std_events
        .iter()
        .filter(|event| event.subtype == MultiStdSubtype::Add)
        .map(|event| event.wall_time_seconds)
        .fold(0.0_f64, |total, seconds| total + seconds);
    let disent_seconds = telemetry
        .multi_disent_events
        .iter()
        .map(|event| event.wall_time_seconds)
        .fold(0.0_f64, |total, seconds| total + seconds);
    let ofd_avoidable_seconds = telemetry
        .multi_std_events
        .iter()
        .filter(|event| event.ofd_in_span)
        .map(|event| event.wall_time_seconds)
        .fold(0.0_f64, |total, seconds| total + seconds);
    print_events(cell, run, telemetry);

    let mut summary = RunSummary {
        sim_seconds,
        profiled_sim_seconds,
        cascade_seconds,
        add_seconds,
        disent_seconds,
        ofd_avoidable_seconds,
        stats: simulator.stats,
        signed_eigenstates: telemetry.signed_eigenstates,
        ..RunSummary::default()
    };
    let query_set = queries(cell.n, cell.seed);
    let projection_locality =
        std::env::var("SATURATION_PROJECTION_LOCALITY").is_ok_and(|value| value != "0");
    let (probabilities, profile) = if projection_locality {
        simulator.prob_bitstrings_profiled_with_projection_locality(&query_set)
    } else {
        simulator.prob_bitstrings_profiled(&query_set)
    };
    assert!(profile.phase_scopes_disjoint());
    let attributed_seconds = profile.attributed_wall_time_seconds();
    assert!(
        attributed_seconds <= profile.whole_call_wall_time_seconds,
        "disjoint phase time cannot exceed the complete query call"
    );
    // Bound the residual as well as the total. This catches unscoped work,
    // but only that: a dropped scope whose share is below the tolerance still
    // passes, so the per-bucket `calls > 0` assertions in the randomized test
    // are the guard against a scope disappearing outright. 0.99 is set from
    // measurement -- healthy residuals run 0.0009%-0.05%, the instrumentation
    // floor at this gate is ~0.2%, and deleting the survival scope produces
    // 1.65%-4.09%, which 0.95 would have admitted. Gated on call length
    // because a microsecond-scale call spends ~100 ns per phase scope in
    // `Instant`, which is a large percentage of a small call and none of a
    // real one.
    if !projection_locality && profile.whole_call_wall_time_seconds > 0.1 {
        assert!(
            attributed_seconds >= 0.99 * profile.whole_call_wall_time_seconds,
            "attribution lost {:.2}% of a {:.3} s query; a phase scope is missing",
            100.0 * (1.0 - attributed_seconds / profile.whole_call_wall_time_seconds),
            profile.whole_call_wall_time_seconds
        );
    }
    summary.query_seconds = profile.whole_call_wall_time_seconds;
    summary.query_residual_seconds = profile.whole_call_wall_time_seconds - attributed_seconds;
    summary.output_hash = probability_hash(&probabilities);
    let totals = query_totals(&profile);
    summary.expectation_seconds = totals.expectation;
    summary.pre_reduction_seconds = totals.pre_reduction;
    summary.decomposition_seconds = totals.decomposition;
    summary.projection_seconds = totals.projection;
    summary.post_projection_qr_seconds = totals.post_projection_qr;
    summary.post_projection_svd_seconds = totals.post_projection_svd;
    summary.survival_seconds = totals.survival;
    summary.normalization_seconds = totals.normalization;
    summary.bookkeeping_seconds = totals.bookkeeping;
    print_query_depths(cell, run, &profile);
    print_locality_total(cell, run, &profile);
    println!(
        "QUERY_OPS cell={} run={} svds={} capped_svds={} attributed_s={:.9} whole_call_s={:.9} trie_clone_residual_s={:.9}",
        cell.name,
        run,
        totals.svds,
        totals.capped_svds,
        attributed_seconds,
        profile.whole_call_wall_time_seconds,
        summary.query_residual_seconds,
    );
    println!(
        "RUN cell={} run={} sim_s={:.9} profiled_sim_s={:.9} telemetry_on_overhead_s={:.9} query_s={:.9} multi_std={} multi_std_add={} multi_std_cascade={} multi_disent={} signed_candidates={} ofd_in_span_std={} expectation_s={:.9} pre_s={:.9} decomposition_s={:.9} projection_s={:.9} qr_s={:.9} svd_s={:.9} survival_s={:.9} normalization_s={:.9} bookkeeping_s={:.9} trie_clone_residual_s={:.9} cascade_s={:.9} add_s={:.9} disent_s={:.9} ofd_avoidable_s={:.9} output_hash={:016x}",
        cell.name,
        run,
        summary.sim_seconds,
        summary.profiled_sim_seconds,
        summary.profiled_sim_seconds - summary.sim_seconds,
        summary.query_seconds,
        summary.stats.multi_std,
        summary.stats.multi_std_add,
        summary.stats.multi_std_cascade,
        summary.stats.multi_disent,
        summary.stats.signed_eigenstate_candidates,
        summary.stats.ofd_in_span_std,
        summary.expectation_seconds,
        summary.pre_reduction_seconds,
        summary.decomposition_seconds,
        summary.projection_seconds,
        summary.post_projection_qr_seconds,
        summary.post_projection_svd_seconds,
        summary.survival_seconds,
        summary.normalization_seconds,
        summary.bookkeeping_seconds,
        summary.query_residual_seconds,
        summary.cascade_seconds,
        summary.add_seconds,
        summary.disent_seconds,
        summary.ofd_avoidable_seconds,
        summary.output_hash,
    );
    summary
}

fn median_run_by(runs: &[RunSummary], key: impl Fn(&RunSummary) -> f64) -> (usize, &RunSummary) {
    let mut indexed: Vec<(usize, &RunSummary)> = runs.iter().enumerate().collect();
    indexed.sort_by(|(_, left), (_, right)| key(left).total_cmp(&key(right)));
    indexed[(indexed.len() - 1) / 2]
}

fn print_summary(cell: Cell, runs: &[RunSummary]) {
    let (sim_run_index, sim_run) = median_run_by(runs, |run| run.sim_seconds);
    let (query_run_index, query_run) = median_run_by(runs, |run| run.query_seconds);
    let first = &runs[0];
    assert!(runs.iter().all(|run| run.output_hash == first.output_hash));
    assert!(
        runs.iter()
            .all(|run| run.signed_eigenstates == first.signed_eigenstates),
        "candidate populations must be deterministic"
    );
    println!(
        "SUMMARY cell={} repetitions={} sim_median_run={} query_median_run={} sim_median_s={:.9} profiled_sim_s={:.9} telemetry_on_overhead_s={:.9} query_median_s={:.9} expectation_s={:.9} pre_s={:.9} decomposition_s={:.9} projection_s={:.9} qr_s={:.9} svd_s={:.9} survival_s={:.9} normalization_s={:.9} bookkeeping_s={:.9} trie_clone_residual_s={:.9} cascade_s={:.9} add_s={:.9} disent_s={:.9} ofd_avoidable_s={:.9} multi_std={} multi_std_add={} multi_std_cascade={} multi_disent={} signed_candidates={} ofd_in_span_std={} output_hash={:016x}",
        cell.name,
        runs.len(),
        sim_run_index,
        query_run_index,
        sim_run.sim_seconds,
        sim_run.profiled_sim_seconds,
        sim_run.profiled_sim_seconds - sim_run.sim_seconds,
        query_run.query_seconds,
        query_run.expectation_seconds,
        query_run.pre_reduction_seconds,
        query_run.decomposition_seconds,
        query_run.projection_seconds,
        query_run.post_projection_qr_seconds,
        query_run.post_projection_svd_seconds,
        query_run.survival_seconds,
        query_run.normalization_seconds,
        query_run.bookkeeping_seconds,
        query_run.query_residual_seconds,
        sim_run.cascade_seconds,
        sim_run.add_seconds,
        sim_run.disent_seconds,
        sim_run.ofd_avoidable_seconds,
        first.stats.multi_std,
        first.stats.multi_std_add,
        first.stats.multi_std_cascade,
        first.stats.multi_disent,
        first.stats.signed_eigenstate_candidates,
        first.stats.ofd_in_span_std,
        first.output_hash,
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
            "CELL cell={} n={} seed={} gates={} campaign_content_hash={} query_status=available",
            cell.name, cell.n, cell.seed, gate_count, cell.campaign_content_hash,
        );
        for warmup in 0..warmups {
            let (simulator, sim_seconds) = simulate(cell, false);
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
