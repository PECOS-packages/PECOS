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
//! Set `SATURATION_PROJECTION_LOCALITY=1` for the runtime-gated projection and
//! pre-reduction diagnostic events. With that outer flag enabled, set
//! `SATURATION_DIRECTION_B_SHADOW=0` to retain the eager diagnostics while
//! disabling only the Direction-B shadow for an overhead A/B.

use pecos_core::{Angle64, QubitId};
use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable};
use pecos_stab_tn::stab_mps::{
    MultiStdSubtype, PreReductionTelemetry, ProbabilityQueryTelemetry, ProjectionConstruction,
    SaturationTelemetry, SignedEigenstateBranchTelemetry, SignedEigenstateTelemetry, StabMps,
    StabMpsStats,
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
        .max_bond_dim(env_count("SATURATION_MAX_BOND_DIM", 64))
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

fn joined_usizes(values: &[usize]) -> String {
    if values.is_empty() {
        "none".to_owned()
    } else {
        values
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",")
    }
}

fn five_number(mut values: Vec<usize>) -> String {
    if values.is_empty() {
        return "none".to_owned();
    }
    values.sort_unstable();
    let last = values.len() - 1;
    [0, last / 4, last / 2, 3 * last / 4, last]
        .map(|index| values[index].to_string())
        .join("/")
}

fn bond_walk_cost(profile: &[usize], from_site: usize, to_site: usize) -> f64 {
    let minimum = from_site.min(to_site);
    let maximum = from_site.max(to_site);
    profile[minimum..maximum]
        .iter()
        .map(|&rank| (rank as f64).powi(3))
        .sum()
}

fn projection_walk_model(
    event: &pecos_stab_tn::stab_mps::ProjectionQrLocalityTelemetry,
) -> Option<(f64, f64, f64)> {
    if event.construction != ProjectionConstruction::DirectSum
        || !event.center_before_positioning_is_valid
    {
        return None;
    }
    projection_walk_model_for_support(event, event.projector_site_min?, event.projector_site_max?)
}

fn projection_walk_model_for_support(
    event: &pecos_stab_tn::stab_mps::ProjectionQrLocalityTelemetry,
    support_min: usize,
    support_max: usize,
) -> Option<(f64, f64, f64)> {
    if !event.center_before_positioning_is_valid {
        return None;
    }
    let center = event.center_before_positioning?;
    let profile = &event.projection_entry_bond_profile;
    let current = profile.iter().map(|&rank| (rank as f64).powi(3)).sum();
    let to_min = bond_walk_cost(profile, center, support_min);
    let to_max = bond_walk_cost(profile, center, support_max);
    let edge_cost_order = to_min.total_cmp(&to_max);
    let (position, terminal_edge) = if edge_cost_order.is_lt()
        || (edge_cost_order.is_eq() && center.abs_diff(support_min) <= center.abs_diff(support_max))
    {
        (to_min, support_max)
    } else {
        (to_max, support_min)
    };
    let span = bond_walk_cost(profile, support_min, support_max);
    let bounded = position + span;
    let mandatory_zero = bounded + bond_walk_cost(profile, terminal_edge, 0);
    Some((current, bounded, mandatory_zero))
}

fn pre_reduction_cnot_chi3_units(event: &PreReductionTelemetry) -> f64 {
    event
        .cnot_steps
        .iter()
        .filter(|cnot| !cnot.structural_identity && !cnot.unconditional_x)
        .map(|cnot| {
            let minimum = cnot.control.min(cnot.target);
            let maximum = cnot.control.max(cnot.target);
            let first = (cnot.input_bond_profile[minimum] as f64).powi(3);
            let return_walk = cnot.input_bond_profile[minimum + 1..maximum]
                .iter()
                .map(|&rank| 2.0 * (rank as f64).powi(3))
                .sum::<f64>();
            first + return_walk
        })
        .sum()
}

fn print_pre_reduction_summary(
    cell: Cell,
    run: usize,
    depth: Option<usize>,
    events: &[&PreReductionTelemetry],
) {
    if events.is_empty() {
        return;
    }
    let mut fingerprints = BTreeMap::<u64, Vec<&PreReductionTelemetry>>::new();
    let mut sibling_pairs = BTreeMap::<u64, Vec<&PreReductionTelemetry>>::new();
    let mut wall_seconds = 0.0;
    let mut svd_seconds = 0.0;
    let mut qr_seconds = 0.0;
    let mut tensor_seconds = 0.0;
    let mut bookkeeping_seconds = 0.0;
    let mut input_bond_sum = 0_usize;
    let mut input_bonds = 0_usize;
    let mut cap_saturated_bonds = 0_usize;
    let mut calls_with_cap = 0_usize;
    let mut calls_with_cap_seconds = 0.0;
    let mut no_op_calls = 0_usize;
    let mut no_op_seconds = 0.0;
    let mut profile_scan_seconds = 0.0;
    let mut svd_operations = 0_u64;
    let mut capped_svd_operations = 0_u64;
    let mut compensating_cnots = 0_u64;
    let mut structural_identity_cnots = 0_u64;
    let mut unconditional_x_cnots = 0_u64;
    let mut replacement_events = 0_usize;
    let mut tied_replacement_events = 0_usize;
    let mut chosen_compensation_cost = 0_usize;
    let mut minimum_weight_optimal_cost = 0_usize;
    let mut weight_plus_one_available_events = 0_usize;
    let mut weight_plus_one_available_chosen_cost = 0_usize;
    let mut weight_plus_one_only_optimal_cost = 0_usize;
    let mut minimum_or_weight_plus_one_optimal_cost = 0_usize;
    let mut maximum_entry_bond = 1_usize;
    for &event in events {
        fingerprints
            .entry(event.input_fingerprint)
            .or_default()
            .push(event);
        if let Some(pair_id) = event.sibling_pair_id {
            sibling_pairs.entry(pair_id).or_default().push(event);
        }
        wall_seconds += event.wall_time_seconds;
        svd_seconds += event.svd_compute_seconds;
        qr_seconds += event.qr_gauge_seconds;
        tensor_seconds += event.tensor_contraction_seconds;
        bookkeeping_seconds += event.bookkeeping_seconds;
        input_bond_sum += event.input_bond_sum;
        input_bonds += event.input_bond_profile.len();
        cap_saturated_bonds += event.input_cap_saturated_bonds;
        maximum_entry_bond = maximum_entry_bond.max(event.input_max_bond);
        if event.input_cap_saturated_bonds > 0 {
            calls_with_cap += 1;
            calls_with_cap_seconds += event.wall_time_seconds;
        }
        if event.output_profile_unchanged {
            no_op_calls += 1;
            no_op_seconds += event.wall_time_seconds;
        }
        profile_scan_seconds +=
            event.input_profile_scan_seconds + event.output_profile_scan_seconds;
        svd_operations += event.svd_operations;
        capped_svd_operations += event.capped_svd_operations;
        compensating_cnots += event.compensating_cnot_count;
        structural_identity_cnots += event.structural_identity_cnot_count;
        unconditional_x_cnots += event.unconditional_x_cnot_count;
        if event.chosen_stabilizer.is_some() {
            replacement_events += 1;
            tied_replacement_events += usize::from(event.minimum_weight_candidate_count > 1);
            chosen_compensation_cost += event.chosen_compensation_cost;
            minimum_weight_optimal_cost += event.minimum_weight_optimal_cost;
            let best_at_most_plus_one = event
                .weight_plus_one_optimal_cost
                .map_or(event.minimum_weight_optimal_cost, |cost| {
                    cost.min(event.minimum_weight_optimal_cost)
                });
            minimum_or_weight_plus_one_optimal_cost += best_at_most_plus_one;
            if let Some(cost) = event.weight_plus_one_optimal_cost {
                weight_plus_one_available_events += 1;
                weight_plus_one_available_chosen_cost += event.chosen_compensation_cost;
                weight_plus_one_only_optimal_cost += cost;
            }
        }
    }

    let repeated_fingerprint_calls = fingerprints
        .values()
        .map(|group| group.len().saturating_sub(1))
        .sum::<usize>();
    let repeated_fingerprint_wall_ceiling = fingerprints
        .values()
        .map(|group| {
            let retained = group
                .iter()
                .map(|event| event.wall_time_seconds)
                .fold(f64::INFINITY, f64::min);
            group
                .iter()
                .map(|event| event.wall_time_seconds)
                .sum::<f64>()
                - retained
        })
        .sum::<f64>();
    let mut sibling_calls = 0_usize;
    let mut sibling_fingerprint_match_calls = 0_usize;
    let mut sibling_share_wall_ceiling = 0.0;
    for pair in sibling_pairs.values() {
        assert_eq!(
            pair.len(),
            2,
            "each populated trie sibling pair has two calls"
        );
        sibling_calls += 2;
        if pair[0].input_fingerprint == pair[1].input_fingerprint
            && pair[0].input_bond_profile == pair[1].input_bond_profile
        {
            sibling_fingerprint_match_calls += 2;
            sibling_share_wall_ceiling += pair[0].wall_time_seconds.min(pair[1].wall_time_seconds);
        }
    }
    let depth = depth.map_or_else(|| "all".to_owned(), |value| value.to_string());
    println!(
        "PRERED_SUMMARY cell={} run={} depth={} calls={} wall_s={:.9} svd_compute_s={:.9} qr_gauge_s={:.9} tensor_s={:.9} bookkeeping_s={:.9} svds={} capped_svds={} max_entry_bond={} mean_entry_bond={:.9} entry_bonds={} cap_saturated_bonds={} cap_bond_fraction={:.9} calls_with_cap={} calls_with_cap_s={:.9} unique_fingerprints={} repeated_fingerprint_calls={} repeated_fingerprint_wall_ceiling_s={:.9} sibling_calls={} sibling_fingerprint_match_calls={} sibling_share_wall_ceiling_s={:.9} no_op_calls={} no_op_s={:.9} profile_scan_s={:.9} compensating_cnots={} structural_identity_cnots={} unconditional_x_cnots={} replacement_events={} tied_replacement_events={} chosen_cost={} minimum_weight_optimal_cost={} plus_one_available_events={} plus_one_available_chosen_cost={} plus_one_only_optimal_cost={} minimum_or_plus_one_optimal_cost={}",
        cell.name,
        run,
        depth,
        events.len(),
        wall_seconds,
        svd_seconds,
        qr_seconds,
        tensor_seconds,
        bookkeeping_seconds,
        svd_operations,
        capped_svd_operations,
        maximum_entry_bond,
        input_bond_sum as f64 / input_bonds.max(1) as f64,
        input_bonds,
        cap_saturated_bonds,
        cap_saturated_bonds as f64 / input_bonds.max(1) as f64,
        calls_with_cap,
        calls_with_cap_seconds,
        fingerprints.len(),
        repeated_fingerprint_calls,
        repeated_fingerprint_wall_ceiling,
        sibling_calls,
        sibling_fingerprint_match_calls,
        sibling_share_wall_ceiling,
        no_op_calls,
        no_op_seconds,
        profile_scan_seconds,
        compensating_cnots,
        structural_identity_cnots,
        unconditional_x_cnots,
        replacement_events,
        tied_replacement_events,
        chosen_compensation_cost,
        minimum_weight_optimal_cost,
        weight_plus_one_available_events,
        weight_plus_one_available_chosen_cost,
        weight_plus_one_only_optimal_cost,
        minimum_or_weight_plus_one_optimal_cost,
    );
}

fn print_pre_reduction_depth(
    cell: Cell,
    run: usize,
    depth: usize,
    events: &[PreReductionTelemetry],
) {
    for (event_index, event) in events.iter().enumerate() {
        let input_bonds = event
            .input_bond_profile
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",");
        let output_bonds = event
            .output_bond_profile
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",");
        let sibling_pair = event
            .sibling_pair_id
            .map_or_else(|| "none".to_owned(), |value| value.to_string());
        let option = |value: Option<usize>| {
            value.map_or_else(|| "none".to_owned(), |value| value.to_string())
        };
        println!(
            "PRERED_EVENT cell={} run={} depth={} event={} sibling_pair={} accumulated_projectors={} measured_qubit={} fingerprint={:016x} cap={} input_max={} input_sum={} input_cap_bonds={} input_bonds={} output_bonds={} no_op={} wall_s={:.9} svd_compute_s={:.9} qr_gauge_s={:.9} tensor_s={:.9} bookkeeping_s={:.9} svds={} capped_svds={} input_scan_s={:.9} output_scan_s={:.9} compensating_cnots={} structural_identity_cnots={} unconditional_x_cnots={} chosen_stabilizer={} chosen_weight={} replacement_candidates={} minimum_weight_candidates={} chosen_cost={} minimum_weight_optimal_cost={} plus_one_candidates={} plus_one_optimal_cost={} target_min={} target_max={} target_span={} support_span={}",
            cell.name,
            run,
            depth,
            event_index,
            sibling_pair,
            event.accumulated_projector_count,
            event.measured_qubit,
            event.input_fingerprint,
            event.bond_cap,
            event.input_max_bond,
            event.input_bond_sum,
            event.input_cap_saturated_bonds,
            input_bonds,
            output_bonds,
            event.output_profile_unchanged,
            event.wall_time_seconds,
            event.svd_compute_seconds,
            event.qr_gauge_seconds,
            event.tensor_contraction_seconds,
            event.bookkeeping_seconds,
            event.svd_operations,
            event.capped_svd_operations,
            event.input_profile_scan_seconds,
            event.output_profile_scan_seconds,
            event.compensating_cnot_count,
            event.structural_identity_cnot_count,
            event.unconditional_x_cnot_count,
            option(event.chosen_stabilizer),
            option(event.chosen_stabilizer_weight),
            event.replacement_candidate_count,
            event.minimum_weight_candidate_count,
            event.chosen_compensation_cost,
            event.minimum_weight_optimal_cost,
            event.weight_plus_one_candidate_count,
            option(event.weight_plus_one_optimal_cost),
            option(event.target_min),
            option(event.target_max),
            event.target_span,
            event.compensation_support_span,
        );
        for (cnot_index, cnot) in event.cnot_steps.iter().enumerate() {
            let cnot_input_bonds = cnot
                .input_bond_profile
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(",");
            let cnot_output_bonds = cnot
                .output_bond_profile
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(",");
            println!(
                "PRERED_CNOT cell={} run={} depth={} event={} cnot={} control={} target={} distance={} input_max={} output_max={} peak_rank={} svd_start={} svds={} structural_identity={} unconditional_x={} input_bonds={} output_bonds={}",
                cell.name,
                run,
                depth,
                event_index,
                cnot_index,
                cnot.control,
                cnot.target,
                cnot.distance,
                cnot.input_bond_profile.iter().copied().max().unwrap_or(1),
                cnot.output_bond_profile.iter().copied().max().unwrap_or(1),
                cnot.peak_bond_rank,
                cnot.svd_start,
                cnot.svd_count,
                cnot.structural_identity,
                cnot.unconditional_x,
                cnot_input_bonds,
                cnot_output_bonds,
            );
            for (svd_in_cnot, svd) in event.svd_steps
                [cnot.svd_start..cnot.svd_start + cnot.svd_count]
                .iter()
                .enumerate()
            {
                println!(
                    "PRERED_SVD cell={} run={} depth={} event={} cnot={} svd_in_cnot={} svd={} input_rows={} input_columns={} output_rank={} capped={}",
                    cell.name,
                    run,
                    depth,
                    event_index,
                    cnot_index,
                    svd_in_cnot,
                    cnot.svd_start + svd_in_cnot,
                    svd.input_rows,
                    svd.input_columns,
                    svd.output_rank,
                    svd.cap_binding,
                );
            }
        }
    }
    print_pre_reduction_summary(cell, run, Some(depth), &events.iter().collect::<Vec<_>>());
}

fn print_pre_reduction_total(cell: Cell, run: usize, profile: &ProbabilityQueryTelemetry) {
    let events = profile
        .by_depth
        .iter()
        .flat_map(|depth| &depth.pre_reduction_diagnostics)
        .collect::<Vec<_>>();
    print_pre_reduction_summary(cell, run, None, &events);
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
        print_pre_reduction_depth(cell, run, depth, &bucket.pre_reduction_diagnostics);
        for (event_index, event) in bucket.projection_qr_locality.iter().enumerate() {
            let option = |value: Option<usize>| {
                value.map_or_else(|| "none".to_owned(), |value| value.to_string())
            };
            let construction = projection_construction_label(event.construction);
            println!(
                "LOCALITY cell={} run={} depth={} event={} n={} center_pre_position={} center_pre_position_valid={} center_pre_write={} center_pre_write_valid={} center_qr={} center_qr_valid={} construction={} touched_min={} touched_max={} touched_sites={} changed_tensor_min={} changed_tensor_max={} changed_tensors={} changed_bond_min={} changed_bond_max={} changed_bonds={} qr_sites={} qr_skippable={} qr_skippable_center_ceiling={} normalization_preserved={}",
                cell.name,
                run,
                depth,
                event_index,
                event.chain_length,
                option(event.center_before_positioning),
                event.center_before_positioning_is_valid,
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
                event.qr_sites_skippable_by_center_ceiling,
                event
                    .normalization_preserved_center
                    .is_some_and(|preserved| preserved),
            );
            if event.construction == ProjectionConstruction::DirectSum {
                let external = event
                    .external_bonds
                    .iter()
                    .map(|bond| {
                        format!(
                            "{}:{}:{}:{}:{:.17e}",
                            bond.bond,
                            bond.pre_projection_rank,
                            bond.compression_input_rank,
                            bond.post_compression_rank,
                            bond.discarded_weight,
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(";");
                println!(
                    "PROJECTOR_EVENT cell={} run={} depth={} event={} n={} center_pre_position={} center_pre_position_valid={} center_pre_write={} center_pre_write_valid={} s_min={} s_max={} span={} flip_sites={} sign_sites={} gauge_sites={} entry_bonds={} qr_s={:.17e} compression_bonds={} external_bonds={} external_discarded={:.17e}",
                    cell.name,
                    run,
                    depth,
                    event_index,
                    event.chain_length,
                    option(event.center_before_positioning),
                    event.center_before_positioning_is_valid,
                    option(event.center_before_projection_write),
                    event.center_before_projection_write_is_valid,
                    option(event.projector_site_min),
                    option(event.projector_site_max),
                    event.projector_span,
                    joined_usizes(&event.projector_flip_sites),
                    joined_usizes(&event.projector_sign_sites),
                    joined_usizes(&event.gauge_compensation_sites),
                    joined_usizes(&event.projection_entry_bond_profile),
                    event.post_projection_qr_wall_time_seconds,
                    event.compression_bonds_observed,
                    external,
                    event.external_discarded_weight,
                );
            }
        }
        if !bucket.projection_qr_locality.is_empty() {
            let mut constructions = BTreeMap::<&str, usize>::new();
            let mut distributions = BTreeMap::<_, usize>::new();
            let mut qr_sites = 0;
            let mut qr_skippable = 0;
            let mut qr_skippable_center_ceiling = 0;
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
                            event.qr_sites_skippable_by_center_ceiling,
                        ),
                    ))
                    .or_default() += 1;
                qr_sites += event.qr_sites;
                qr_skippable += event.qr_sites_skippable_by_locality;
                qr_skippable_center_ceiling += event.qr_sites_skippable_by_center_ceiling;
                normalization_losses +=
                    usize::from(event.normalization_preserved_center == Some(false));
            }
            let construction_counts = constructions
                .iter()
                .map(|(construction, count)| format!("{construction}:{count}"))
                .collect::<Vec<_>>()
                .join(",");
            println!(
                "LOCALITY_DEPTH cell={} run={} depth={} events={} constructions={} qr_sites={} qr_skippable={} headroom_fraction={:.9} qr_skippable_center_ceiling={} headroom_fraction_center_ceiling={:.9} normalization_losses={}",
                cell.name,
                run,
                depth,
                bucket.projection_qr_locality.len(),
                construction_counts,
                qr_sites,
                qr_skippable,
                qr_skippable as f64 / qr_sites as f64,
                qr_skippable_center_ceiling,
                qr_skippable_center_ceiling as f64 / qr_sites as f64,
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
                        event_qr_skippable_center_ceiling,
                    ),
                ) = distribution;
                let option = |value: Option<usize>| {
                    value.map_or_else(|| "none".to_owned(), |value| value.to_string())
                };
                println!(
                    "LOCALITY_DIST cell={} run={} depth={} count={} construction={} center_pre_write={} center_pre_write_valid={} center_qr={} center_qr_valid={} touched_min={} touched_max={} touched_sites={} changed_tensor_min={} changed_tensor_max={} changed_tensors={} changed_bond_min={} changed_bond_max={} changed_bonds={} qr_sites={} qr_skippable={} qr_skippable_center_ceiling={}",
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
                    event_qr_skippable_center_ceiling,
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
    let mut qr_skippable_center_ceiling = 0;
    let mut normalization_losses = 0;
    for event in &events {
        *constructions
            .entry(projection_construction_label(event.construction))
            .or_default() += 1;
        qr_sites += event.qr_sites;
        qr_skippable += event.qr_sites_skippable_by_locality;
        qr_skippable_center_ceiling += event.qr_sites_skippable_by_center_ceiling;
        normalization_losses += usize::from(event.normalization_preserved_center == Some(false));
    }
    let construction_counts = constructions
        .iter()
        .map(|(construction, count)| format!("{construction}:{count}"))
        .collect::<Vec<_>>()
        .join(",");
    println!(
        "LOCALITY_TOTAL cell={} run={} events={} constructions={} qr_sites={} qr_skippable={} headroom_fraction={:.9} qr_skippable_center_ceiling={} headroom_fraction_center_ceiling={:.9} normalization_losses={}",
        cell.name,
        run,
        events.len(),
        construction_counts,
        qr_sites,
        qr_skippable,
        qr_skippable as f64 / qr_sites as f64,
        qr_skippable_center_ceiling,
        qr_skippable_center_ceiling as f64 / qr_sites as f64,
        normalization_losses,
    );
}

fn print_projector_span_total(cell: Cell, run: usize, profile: &ProbabilityQueryTelemetry) {
    let events = profile
        .by_depth
        .iter()
        .flat_map(|depth| &depth.projection_qr_locality)
        .collect::<Vec<_>>();
    if events.is_empty() {
        return;
    }
    let scalar = events
        .iter()
        .filter(|event| event.construction == ProjectionConstruction::ScalarScale)
        .count();
    let local = events
        .iter()
        .filter(|event| event.construction == ProjectionConstruction::LocalBlockWrite)
        .count();
    let direct = events
        .iter()
        .copied()
        .filter(|event| event.construction == ProjectionConstruction::DirectSum)
        .collect::<Vec<_>>();
    println!(
        "PROJECTOR_MIX cell={} run={} all={} direct_sum={} scalar_scale={} local_block={}",
        cell.name,
        run,
        events.len(),
        direct.len(),
        scalar,
        local,
    );
    if direct.is_empty() {
        return;
    }

    let projector_min = direct
        .iter()
        .map(|event| event.projector_site_min.unwrap())
        .collect();
    let projector_max = direct
        .iter()
        .map(|event| event.projector_site_max.unwrap())
        .collect();
    let projector_span = direct.iter().map(|event| event.projector_span).collect();
    let left_margin = direct
        .iter()
        .map(|event| event.projector_site_min.unwrap())
        .collect();
    let right_margin = direct
        .iter()
        .map(|event| event.chain_length - 1 - event.projector_site_max.unwrap())
        .collect();
    let centers_before_positioning = direct
        .iter()
        .filter_map(|event| event.center_before_positioning)
        .collect();
    let centers_before_write = direct
        .iter()
        .filter_map(|event| event.center_before_projection_write)
        .collect();
    let gauge_counts = direct
        .iter()
        .map(|event| event.gauge_compensation_sites.len())
        .collect();
    let gauge_min = direct
        .iter()
        .filter_map(|event| event.gauge_compensation_sites.first().copied())
        .collect();
    let gauge_max = direct
        .iter()
        .filter_map(|event| event.gauge_compensation_sites.last().copied())
        .collect();
    let gauge_span = direct
        .iter()
        .filter_map(|event| {
            event
                .gauge_compensation_sites
                .first()
                .zip(event.gauge_compensation_sites.last())
                .map(|(minimum, maximum)| maximum - minimum)
        })
        .collect();
    let entry_bonds = direct
        .iter()
        .flat_map(|event| event.projection_entry_bond_profile.iter().copied())
        .collect();
    println!(
        "PROJECTOR_QUINTILES cell={} run={} order=min/q25/median/q75/max s_min={} s_max={} span={} left_margin={} right_margin={} center_pre_position={} center_pre_write={} entry_bond_rank={} gauge_count={} gauge_min={} gauge_max={} gauge_span={} invalid_center_pre_position={} invalid_center_pre_write={} events_with_gauge={}",
        cell.name,
        run,
        five_number(projector_min),
        five_number(projector_max),
        five_number(projector_span),
        five_number(left_margin),
        five_number(right_margin),
        five_number(centers_before_positioning),
        five_number(centers_before_write),
        five_number(entry_bonds),
        five_number(gauge_counts),
        five_number(gauge_min),
        five_number(gauge_max),
        five_number(gauge_span),
        direct
            .iter()
            .filter(|event| !event.center_before_positioning_is_valid)
            .count(),
        direct
            .iter()
            .filter(|event| !event.center_before_projection_write_is_valid)
            .count(),
        direct
            .iter()
            .filter(|event| !event.gauge_compensation_sites.is_empty())
            .count(),
    );

    let mut joint = BTreeMap::<(Option<usize>, usize, usize), usize>::new();
    for event in &direct {
        *joint
            .entry((
                event.center_before_positioning,
                event.projector_site_min.unwrap(),
                event.projector_site_max.unwrap(),
            ))
            .or_default() += 1;
    }
    for ((center, support_min, support_max), count) in joint {
        println!(
            "PROJECTOR_JOINT cell={} run={} center={} s_min={} s_max={} count={}",
            cell.name,
            run,
            center.map_or_else(|| "none".to_owned(), |value| value.to_string()),
            support_min,
            support_max,
            count,
        );
    }

    let qr_bucket_seconds = profile
        .by_depth
        .iter()
        .map(|depth| depth.post_projection_qr.wall_time_seconds)
        .sum::<f64>();
    let mut current_units = 0.0;
    let mut bounded_units = 0.0;
    let mut mandatory_zero_units = 0.0;
    let mut current_seconds = 0.0;
    let mut bounded_seconds = 0.0;
    let mut mandatory_zero_seconds = 0.0;
    let mut model_events = 0;
    for event in &direct {
        if let Some((current, bounded, mandatory_zero)) = projection_walk_model(event)
            && current > 0.0
        {
            current_units += current;
            bounded_units += bounded;
            mandatory_zero_units += mandatory_zero;
            current_seconds += event.post_projection_qr_wall_time_seconds;
            bounded_seconds += event.post_projection_qr_wall_time_seconds * bounded / current;
            mandatory_zero_seconds +=
                event.post_projection_qr_wall_time_seconds * mandatory_zero / current;
            model_events += 1;
        }
    }
    println!(
        "PROJECTOR_MODEL cell={} run={} label=chi3_model calibration=per_event_current_qr direct_events={} modeled_events={} qr_bucket_s={:.17e} a_units={:.17e} b_units={:.17e} c_units={:.17e} a_estimated_s={:.17e} b_estimated_s={:.17e} c_estimated_s={:.17e} a_fraction_qr={:.17e} b_fraction_qr={:.17e} c_fraction_qr={:.17e} c_over_a={:.17e}",
        cell.name,
        run,
        direct.len(),
        model_events,
        qr_bucket_seconds,
        current_units,
        bounded_units,
        mandatory_zero_units,
        current_seconds,
        bounded_seconds,
        mandatory_zero_seconds,
        current_seconds / qr_bucket_seconds,
        bounded_seconds / qr_bucket_seconds,
        mandatory_zero_seconds / qr_bucket_seconds,
        mandatory_zero_units / current_units,
    );

    let external_bonds = direct
        .iter()
        .flat_map(|event| &event.external_bonds)
        .collect::<Vec<_>>();
    let changed_bonds = external_bonds
        .iter()
        .filter(|bond| bond.retained_rank_changed)
        .count();
    let decreased_bonds = external_bonds
        .iter()
        .filter(|bond| bond.post_compression_rank < bond.pre_projection_rank)
        .count();
    let increased_bonds = external_bonds
        .iter()
        .filter(|bond| bond.post_compression_rank > bond.pre_projection_rank)
        .count();
    let changed_events = direct
        .iter()
        .filter(|event| {
            event
                .external_bonds
                .iter()
                .any(|bond| bond.retained_rank_changed)
        })
        .count();
    let discarded_events = direct
        .iter()
        .filter(|event| event.external_discarded_weight > 0.0)
        .count();
    let discarded_weight = direct
        .iter()
        .map(|event| event.external_discarded_weight)
        .sum::<f64>();
    println!(
        "PROJECTOR_EXTERNAL cell={} run={} direct_events={} changed_events={} changed_event_fraction={:.17e} external_bonds={} changed_bonds={} changed_bond_fraction={:.17e} decreased_bonds={} increased_bonds={} discarded_events={} discarded_event_fraction={:.17e} summed_discarded_weight={:.17e}",
        cell.name,
        run,
        direct.len(),
        changed_events,
        changed_events as f64 / direct.len() as f64,
        external_bonds.len(),
        changed_bonds,
        changed_bonds as f64 / external_bonds.len() as f64,
        decreased_bonds,
        increased_bonds,
        discarded_events,
        discarded_events as f64 / direct.len() as f64,
        discarded_weight,
    );
}

fn print_direction_b_shadow(cell: Cell, run: usize, profile: &ProbabilityQueryTelemetry) {
    let all_events = profile
        .by_depth
        .iter()
        .flat_map(|depth| &depth.direction_b_shadow)
        .collect::<Vec<_>>();
    if all_events.is_empty() {
        return;
    }

    for (depth, bucket) in profile.by_depth.iter().enumerate() {
        if bucket.direction_b_shadow.is_empty() {
            continue;
        }
        let queue_before = bucket
            .direction_b_shadow
            .iter()
            .map(|event| event.queue_len_before_pre_reduction)
            .collect();
        let queue_projection = bucket
            .direction_b_shadow
            .iter()
            .map(|event| event.queue_len_at_projection)
            .collect();
        let queue_after = bucket
            .direction_b_shadow
            .iter()
            .map(|event| event.queue_len_after_projection)
            .collect();
        let compensation = bucket
            .direction_b_shadow
            .iter()
            .map(|event| event.compensation_cnot_count)
            .collect();
        let frame_seconds = bucket
            .direction_b_shadow
            .iter()
            .map(|event| event.frame_wall_time_seconds)
            .sum::<f64>()
            + bucket.direction_b_shadow_clone_wall_time_seconds;
        let flush_reads = bucket
            .direction_b_shadow
            .iter()
            .filter(|event| event.flush_read_required)
            .count();
        println!(
            "B_QUEUE_DEPTH cell={} run={} depth={} events={} order=min/q25/median/q75/max queue_before={} compensation_cnots={} queue_projection={} queue_after={} flush_reads={} shadow_clone_calls={} shadow_clone_measured_s={:.17e} frame_measured_s={:.17e}",
            cell.name,
            run,
            depth,
            bucket.direction_b_shadow.len(),
            five_number(queue_before),
            five_number(compensation),
            five_number(queue_projection),
            five_number(queue_after),
            flush_reads,
            bucket.direction_b_shadow_clone_calls,
            bucket.direction_b_shadow_clone_wall_time_seconds,
            frame_seconds,
        );
        for (event_index, event) in bucket.direction_b_shadow.iter().enumerate() {
            println!(
                "B_SHADOW_EVENT cell={} run={} depth={} event={} sibling_pair={} outcome={} queue_before={} compensation_cnots={} queue_projection={} queue_after={} flip_sites={} sign_sites={} sites={} s_min={} s_max={} span={} flush_read={} flush_queue={} flush_chi3_units={:.17e} frame_measured_s={:.17e} eager_event={}",
                cell.name,
                run,
                depth,
                event_index,
                event
                    .sibling_pair_id
                    .map_or_else(|| "none".to_owned(), |value| value.to_string()),
                usize::from(event.outcome),
                event.queue_len_before_pre_reduction,
                event.compensation_cnot_count,
                event.queue_len_at_projection,
                event.queue_len_after_projection,
                joined_usizes(&event.projector_flip_sites),
                joined_usizes(&event.projector_sign_sites),
                joined_usizes(&event.projector_sites),
                event
                    .projector_site_min
                    .map_or_else(|| "none".to_owned(), |value| value.to_string()),
                event
                    .projector_site_max
                    .map_or_else(|| "none".to_owned(), |value| value.to_string()),
                event.projector_span,
                event.flush_read_required,
                event.flush_queue_length,
                event.flush_walk_chi3_units,
                event.frame_wall_time_seconds,
                event
                    .eager_projection_event_index
                    .map_or_else(|| "none".to_owned(), |value| value.to_string()),
            );
        }
    }

    let eager_direct_spans = profile
        .by_depth
        .iter()
        .flat_map(|depth| &depth.projection_qr_locality)
        .filter(|event| event.construction == ProjectionConstruction::DirectSum)
        .map(|event| event.projector_span)
        .collect::<Vec<_>>();
    let b_matched_spans = profile
        .by_depth
        .iter()
        .flat_map(|depth| &depth.direction_b_shadow)
        .filter(|event| event.eager_projection_event_index.is_some())
        .map(|event| event.projector_span)
        .collect::<Vec<_>>();
    let b_on_eager_direct_spans = profile
        .by_depth
        .iter()
        .flat_map(|depth| {
            depth.direction_b_shadow.iter().filter_map(|shadow| {
                let eager_index = shadow.eager_projection_event_index?;
                (depth.projection_qr_locality[eager_index].construction
                    == ProjectionConstruction::DirectSum)
                    .then_some(shadow.projector_span)
            })
        })
        .collect::<Vec<_>>();
    let queue_projection = all_events
        .iter()
        .map(|event| event.queue_len_at_projection)
        .collect();
    let queue_after = all_events
        .iter()
        .map(|event| event.queue_len_after_projection)
        .collect();
    println!(
        "B_SUPPORT_QUINTILES cell={} run={} order=min/q25/median/q75/max eager_direct_events={} eager_direct_span={} b_on_eager_direct_events={} b_on_eager_direct_span={} b_all_matched_events={} b_all_conjugated_span={} queue_projection={} queue_after={}",
        cell.name,
        run,
        eager_direct_spans.len(),
        five_number(eager_direct_spans),
        b_on_eager_direct_spans.len(),
        five_number(b_on_eager_direct_spans),
        b_matched_spans.len(),
        five_number(b_matched_spans),
        five_number(queue_projection),
        five_number(queue_after),
    );

    let pre_reduction_seconds = profile
        .by_depth
        .iter()
        .map(|depth| depth.pre_reduction.wall_time_seconds)
        .sum::<f64>();
    let pre_reduction_chi3_units = profile
        .by_depth
        .iter()
        .flat_map(|depth| &depth.pre_reduction_diagnostics)
        .map(pre_reduction_cnot_chi3_units)
        .sum::<f64>();
    let pre_reduction_seconds_per_chi3 = if pre_reduction_chi3_units > 0.0 {
        pre_reduction_seconds / pre_reduction_chi3_units
    } else {
        0.0
    };
    let frame_seconds = all_events
        .iter()
        .map(|event| event.frame_wall_time_seconds)
        .sum::<f64>()
        + profile
            .by_depth
            .iter()
            .map(|depth| depth.direction_b_shadow_clone_wall_time_seconds)
            .sum::<f64>();
    let flush_reads = all_events
        .iter()
        .filter(|event| event.flush_read_required)
        .count();
    let flush_chi3_units = all_events
        .iter()
        .map(|event| event.flush_walk_chi3_units)
        .sum::<f64>();
    let flush_seconds = flush_chi3_units * pre_reduction_seconds_per_chi3;

    let mut extra_projection_seconds = 0.0;
    let mut modeled_projection_events = 0_usize;
    let mut unmodeled_projection_events = 0_usize;
    for bucket in &profile.by_depth {
        for shadow in &bucket.direction_b_shadow {
            let Some(eager_index) = shadow.eager_projection_event_index else {
                continue;
            };
            let eager = &bucket.projection_qr_locality[eager_index];
            let current_units = eager
                .projection_entry_bond_profile
                .iter()
                .map(|&rank| (rank as f64).powi(3))
                .sum::<f64>();
            let eager_bounded_units = if eager.construction == ProjectionConstruction::DirectSum {
                projection_walk_model(eager).map(|(_, bounded, _)| bounded)
            } else {
                Some(0.0)
            };
            let b_bounded_units = match (shadow.projector_site_min, shadow.projector_site_max) {
                (Some(minimum), Some(maximum)) => {
                    projection_walk_model_for_support(eager, minimum, maximum)
                        .map(|(_, bounded, _)| bounded)
                }
                (None, None) => Some(0.0),
                _ => unreachable!("Direction-B support endpoints must be paired"),
            };
            match (eager_bounded_units, b_bounded_units) {
                (Some(eager_bounded), Some(b_bounded)) if current_units > 0.0 => {
                    let calibration_seconds = eager.post_projection_qr_wall_time_seconds
                        + eager.post_projection_svd_wall_time_seconds;
                    extra_projection_seconds +=
                        calibration_seconds * (b_bounded - eager_bounded) / current_units;
                    modeled_projection_events += 1;
                }
                _ => unmodeled_projection_events += 1,
            }
        }
    }
    let delta = pre_reduction_seconds - (extra_projection_seconds + frame_seconds + flush_seconds);
    println!(
        "B_NET_MODEL cell={} run={} formula=Delta_T_B=T_prereduction_removed-(T_extra_projection+T_frame+T_flush_read) T_prereduction_removed_s={:.17e} T_prereduction_removed_label=MEASURED T_extra_projection_s={:.17e} T_extra_projection_label=MODEL_chi3_walk_svd T_frame_s={:.17e} T_frame_label=MEASURED_shadow_algebra T_flush_read_s={:.17e} T_flush_read_label=MODEL_chi3_from_measured_prereduction flush_reads={} flush_chi3_units={:.17e} prereduction_chi3_units={:.17e} modeled_projection_events={} unmodeled_projection_events={} Delta_T_B_s={:.17e}",
        cell.name,
        run,
        pre_reduction_seconds,
        extra_projection_seconds,
        frame_seconds,
        flush_seconds,
        flush_reads,
        flush_chi3_units,
        pre_reduction_chi3_units,
        modeled_projection_events,
        unmodeled_projection_events,
        delta,
    );
    println!(
        "B_MAST cell={} run={} exercised=false reason=campaign_uses_StabMps_and_performs_no_MAST_injections",
        cell.name, run,
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
    let direction_b_shadow = projection_locality
        && std::env::var("SATURATION_DIRECTION_B_SHADOW").map_or(true, |value| value != "0");
    let (probabilities, profile) = if direction_b_shadow {
        simulator.prob_bitstrings_profiled_with_projection_locality(&query_set)
    } else if projection_locality {
        simulator.prob_bitstrings_profiled_with_projection_locality_without_direction_b_shadow(
            &query_set,
        )
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
    print_pre_reduction_total(cell, run, &profile);
    print_locality_total(cell, run, &profile);
    print_projector_span_total(cell, run, &profile);
    print_direction_b_shadow(cell, run, &profile);
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
        "RUN cell={} run={} direction_b_shadow={} sim_s={:.9} profiled_sim_s={:.9} telemetry_on_overhead_s={:.9} query_s={:.9} multi_std={} multi_std_add={} multi_std_cascade={} multi_disent={} signed_candidates={} ofd_in_span_std={} expectation_s={:.9} pre_s={:.9} decomposition_s={:.9} projection_s={:.9} qr_s={:.9} svd_s={:.9} survival_s={:.9} normalization_s={:.9} bookkeeping_s={:.9} trie_clone_residual_s={:.9} cascade_s={:.9} add_s={:.9} disent_s={:.9} ofd_avoidable_s={:.9} output_hash={:016x}",
        cell.name,
        run,
        direction_b_shadow,
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
    let bond_cap = env_count("SATURATION_MAX_BOND_DIM", 64);
    assert!(bond_cap > 0, "SATURATION_MAX_BOND_DIM must be positive");
    assert!(repetitions > 0, "at least one timed repetition is required");
    for cell in selected_cells() {
        let gate_count = gates(cell).len();
        println!(
            "CELL cell={} n={} seed={} gates={} bond_cap={} campaign_content_hash={} query_status=available",
            cell.name, cell.n, cell.seed, gate_count, bond_cap, cell.campaign_content_hash,
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

    #[cfg(feature = "direction-b-phase0-test")]
    #[test]
    #[ignore = "decisive n=64 captured Direction-B replay; run explicitly in release mode"]
    fn captured_n64_direction_b_replay() {
        let cell = CELLS
            .iter()
            .copied()
            .find(|cell| cell.name == "sparse-n64-t2n")
            .unwrap();
        let (simulator, _) = simulate(cell, false);
        let query_set = queries(cell.n, cell.seed);
        let replay = simulator.direction_b_phase0_replay(&query_set);
        let pre_svd_dims = replay
            .eager_pre_reduction_svds
            .iter()
            .map(|step| {
                format!(
                    "{}x{}->{}:{}",
                    step.input_rows, step.input_columns, step.output_rank, step.cap_binding
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        let projection_dims = |steps: &[pecos_stab_tn::stab_mps::ProjectionSvdTelemetry]| {
            steps
                .iter()
                .map(|step| {
                    format!(
                        "{}:{}x{}->{}:{}",
                        step.bond,
                        step.input_rows,
                        step.input_columns,
                        step.output_rank,
                        step.cap_binding
                    )
                })
                .collect::<Vec<_>>()
                .join(",")
        };
        println!(
            "B_REPLAY cell={} depth={} outcome={} input_cap_bonds={} input_bonds={} eager_measured_s={:.17e} b_measured_s={:.17e} b_read_flush_measured_s={:.17e} eager_probability={:.17e} b_probability={:.17e} eager_prered_svds={} eager_projection_svds={} b_prered_svds=0 b_projection_svds={} eager_bonds={} b_virtual_bonds={} b_flushed_bonds={} b_queue={} amplitude_samples={} sampled_overlap={:.17e} sampled_max_aligned_residual={:.17e}",
            cell.name,
            replay.depth,
            usize::from(replay.outcome),
            replay.input_cap_saturated_bonds,
            joined_usizes(&replay.input_bond_profile),
            replay.eager_wall_time_seconds,
            replay.direction_b_wall_time_seconds,
            replay.direction_b_flush_wall_time_seconds,
            replay.eager_probability,
            replay.direction_b_probability,
            pre_svd_dims,
            projection_dims(&replay.eager_projection_svds),
            projection_dims(&replay.direction_b_projection_svds),
            joined_usizes(&replay.eager_bond_profile),
            joined_usizes(&replay.direction_b_bond_profile),
            joined_usizes(&replay.direction_b_flushed_bond_profile),
            replay.direction_b_queue_length,
            replay.amplitude_samples,
            replay.sampled_state_overlap,
            replay.sampled_max_aligned_residual,
        );
        assert!(replay.input_cap_saturated_bonds > 0);
        assert!(!replay.eager_pre_reduction_svds.is_empty());
        assert!(!replay.eager_projection_svds.is_empty());
        assert!(!replay.direction_b_projection_svds.is_empty());
        assert!(replay.eager_probability.is_finite());
        assert!(replay.direction_b_probability.is_finite());
        assert!(replay.sampled_state_overlap.is_finite());
        assert!(replay.sampled_state_overlap <= 1.0 + 1e-10);
    }
}
