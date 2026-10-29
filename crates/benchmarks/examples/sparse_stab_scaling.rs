// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0.

//! One surface- or color-code size per process. Usage:
//! `sparse_stab_scaling SET D ROUNDS [P [F [SEED]]] [--stats-every K] [--code surface|color] [--rule ag|refactor] [--count-work]`.
//!
//! SET is bitset, sorted, unsorted, adaptive, or cpp (requires feature cppsparsestab).

use benchmarks::color_code::color_code;
use pecos_core::{AdaptiveIndexSet, BitSet, IndexSet, QubitId, SortedVecSet, VecSet};
use pecos_random::PecosRng;
use pecos_simulators::{
    CliffordGateable, GensGeneric, MeasurementResult, MeasurementRule, SparseStabGeneric,
};
use rand::RngExt;
use std::time::Instant;

struct Config {
    d: usize,
    rounds: usize,
    p: f64,
    f: f64,
    seed: u64,
    stats_every: usize,
    code: Code,
    count_work: bool,
    rule: MeasurementRule,
}

#[derive(Clone, Copy)]
enum Code {
    Surface,
    Color,
}

struct Schedule {
    data_qubits: usize,
    x_ancillas: Vec<QubitId>,
    ancillas: Vec<QubitId>,
    is_x: Vec<bool>,
    layers: Vec<Vec<(QubitId, QubitId)>>,
    z_layers: Vec<Vec<(QubitId, QubitId)>>,
}

fn surface_schedule(d: usize) -> Schedule {
    let side = isize::try_from(d).expect("distance fits isize");
    let mut result = Schedule {
        data_qubits: d * d,
        x_ancillas: Vec::new(),
        ancillas: Vec::new(),
        is_x: Vec::new(),
        layers: vec![Vec::new(); 4],
        z_layers: Vec::new(),
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

fn color_schedule(d: usize) -> Schedule {
    let code = color_code(d);
    let data = code.num_data_qubits;
    let faces = code.faces.len();
    let x_ancillas: Vec<_> = (data..data + faces).map(QubitId).collect();
    let ancillas: Vec<_> = (data..data + 2 * faces).map(QubitId).collect();
    let mut is_x = vec![true; faces];
    is_x.extend(vec![false; faces]);
    let edges: Vec<_> = code
        .faces
        .iter()
        .enumerate()
        .flat_map(|(i, face)| {
            face.qubits
                .iter()
                .map(move |&q| (QubitId(data + i), QubitId(q)))
        })
        .collect();
    let layers = greedy_layers(&edges, data + 2 * faces);
    let z_edges: Vec<_> = edges
        .iter()
        .map(|&(ancilla, q)| (q, QubitId(ancilla.index() + faces)))
        .collect();
    let z_layers = greedy_layers(&z_edges, data + 2 * faces);
    assert_eq!(data, (3 * d * d + 1) / 4);
    assert_eq!(faces, (3 * d * d - 3) / 8);
    assert_eq!(data + ancillas.len(), (3 * d * d - 1) / 2);
    Schedule {
        data_qubits: data,
        x_ancillas,
        ancillas,
        is_x,
        layers,
        z_layers,
    }
}

fn greedy_layers(edges: &[(QubitId, QubitId)], n: usize) -> Vec<Vec<(QubitId, QubitId)>> {
    let mut layers: Vec<Vec<_>> = Vec::new();
    let mut occupied: Vec<Vec<bool>> = Vec::new();
    for &(a, b) in edges {
        let layer = occupied
            .iter()
            .position(|used| !used[a.index()] && !used[b.index()])
            .unwrap_or(layers.len());
        if layer == layers.len() {
            layers.push(Vec::new());
            occupied.push(vec![false; n]);
        }
        occupied[layer][a.index()] = true;
        occupied[layer][b.index()] = true;
        layers[layer].push((a, b));
    }
    layers
}

#[derive(Default, Debug, PartialEq, Eq)]
struct WorkCounters {
    gates: usize,
    gate_touches: usize,
    meas_det_work: usize,
    meas_rand_work: usize,
    meas_count_det: usize,
    meas_count_rand: usize,
    meas_refactor_work: usize,
    meas_decouple_work: usize,
}

impl WorkCounters {
    fn measurement<S: IndexSet>(&mut self, sim: &SparseStabGeneric<S>, q: QubitId) {
        let q = q.index();
        let (stabs, destabs) = (sim.stabs(), sim.destabs());
        if stabs.col_x[q].is_empty() {
            self.meas_count_det += 1;
            self.meas_det_work += destabs.col_x[q]
                .iter()
                .map(|r| stabs.generator_weight(r))
                .sum::<usize>();
        } else {
            self.meas_count_rand += 1;
            let (weight, _) = stabs.col_x[q]
                .iter()
                .map(|r| (stabs.generator_weight(r), r))
                .min()
                .expect("nondeterministic measurement has a pivot");
            self.meas_rand_work += (stabs.col_x[q].len() + destabs.col_x[q].len()) * weight;
        }
    }

    fn measure<S: IndexSet>(
        &mut self,
        sim: &mut SparseStabGeneric<S>,
        q: QubitId,
        seconds: &mut f64,
    ) -> MeasurementResult {
        self.measurement(sim, q);
        let start = Instant::now();
        let (result, executed) = sim.mz_with_work(q.index());
        *seconds += start.elapsed().as_secs_f64();
        self.meas_refactor_work += executed.refactor;
        self.meas_decouple_work += executed.decouple;
        result
    }

    fn print(&self) {
        print!(
            ",{},{},{},{},{},{},{},{}",
            self.gates,
            self.gate_touches,
            self.meas_det_work,
            self.meas_rand_work,
            self.meas_count_det,
            self.meas_count_rand,
            self.meas_refactor_work,
            self.meas_decouple_work
        );
    }
}

enum GateLayer<'a> {
    H(&'a [QubitId]),
    Cx(&'a [(QubitId, QubitId)]),
}

impl GateLayer<'_> {
    fn apply<T: CliffordGateable>(&self, sim: &mut T) {
        match self {
            Self::H(qubits) => {
                sim.h(qubits);
            }
            Self::Cx(pairs) => {
                sim.cx(pairs);
            }
        }
    }

    fn count_and_apply<S: IndexSet>(
        &self,
        sim: &mut SparseStabGeneric<S>,
        work: &mut WorkCounters,
    ) -> f64 {
        let mut seconds = 0.0;
        match self {
            Self::H(qubits) => {
                work.gates += qubits.len();
                for &q in *qubits {
                    for gens in [sim.stabs(), sim.destabs()] {
                        work.gate_touches +=
                            gens.col_x[q.index()].len() + gens.col_z[q.index()].len();
                    }
                    let start = Instant::now();
                    sim.h(&[q]);
                    seconds += start.elapsed().as_secs_f64();
                }
            }
            Self::Cx(pairs) => {
                work.gates += pairs.len();
                for &(c, t) in *pairs {
                    for gens in [sim.stabs(), sim.destabs()] {
                        work.gate_touches +=
                            gens.col_x[c.index()].len() + gens.col_z[t.index()].len();
                    }
                    let start = Instant::now();
                    sim.cx(&[(c, t)]);
                    seconds += start.elapsed().as_secs_f64();
                }
            }
        }
        seconds
    }
}

impl Schedule {
    fn gate_layers(&self) -> impl Iterator<Item = GateLayer<'_>> {
        std::iter::once(GateLayer::H(&self.x_ancillas))
            .chain(self.layers.iter().map(|layer| GateLayer::Cx(layer)))
            .chain(std::iter::once(GateLayer::H(&self.x_ancillas)))
            .chain(self.z_layers.iter().map(|layer| GateLayer::Cx(layer)))
    }
}

fn counted_gates<S: IndexSet>(
    sim: &mut SparseStabGeneric<S>,
    ops: &Schedule,
    x: &[QubitId],
    z: &[QubitId],
    work: &mut WorkCounters,
) -> f64 {
    work.gates += x.len() + z.len();
    let start = Instant::now();
    sim.x(x).z(z);
    let mut seconds = start.elapsed().as_secs_f64();
    for layer in ops.gate_layers() {
        seconds += layer.count_and_apply(sim, work);
    }
    seconds
}

fn counted_measure_reset<S: IndexSet>(
    sim: &mut SparseStabGeneric<S>,
    qubits: &[QubitId],
    work: &mut WorkCounters,
    seconds: &mut f64,
) -> Vec<MeasurementResult> {
    let mut results = Vec::with_capacity(qubits.len());
    for &q in qubits {
        results.push(work.measure(sim, q, seconds));
    }
    // PZ measures again before its conditional X; preserve MZ(all), PZ(all).
    for &q in qubits {
        let result = work.measure(sim, q, seconds);
        if result.outcome {
            let start = Instant::now();
            sim.x(&[q]);
            *seconds += start.elapsed().as_secs_f64();
        }
    }
    results
}

fn print_header(count_work: bool) {
    println!("# VmHWM is the process peak, including precomputed operation lists.");
    println!(
        "# stats outside timers; BitSet scan O(n^2/64); max column = col_x.len + col_z.len; unsampled=NA"
    );
    if count_work {
        println!(
            "# gates = number of X, Z, H and CX applications in gate stages (excludes reset corrections)"
        );
        println!(
            "# gate_touches = sum over both tableaux of col_x[q].len + col_z[q].len for H(q), col_x[c].len + col_z[t].len for CX(c,t); X/Z only change signs"
        );
        println!(
            "# meas_det_work = sum over r in destabs.col_x[q] of stabs.row_x[r].len + stabs.row_z[r].len for deterministic Z_q"
        );
        println!(
            "# meas_rand_work = (stabs.col_x[q].len + destabs.col_x[q].len) * pivot weight, minimizing (row_x.len + row_z.len, row id)"
        );
        println!(
            "# meas_refactor_work = executed source support scans + dual product support visits; two-source shortcut counts replaced S support + one X toggle; singleton shortcut counts zero"
        );
        println!(
            "# meas_decouple_work = executed Z removals + discarded D support, counted at each update (after the random update); already isolated partners cost zero; |row| = row_x.len + row_z.len"
        );
        println!(
            "# meas_count_det = number of Z_q measurements with empty stabs.col_x[q], including PZ's internal MZ"
        );
        println!(
            "# meas_count_rand = number of Z_q measurements with nonempty stabs.col_x[q], including PZ's internal MZ"
        );
        println!(
            "# work counted every round; refactor/decouple counters run inside measurement timers; counted runs apply H/CX gates one at a time in batch order and are not timing runs"
        );
    }
    print!(
        "round,gate_seconds,measurement_reset_seconds,nondeterministic,stabilizer_nnz,destabilizer_nnz,max_stabilizer_column,max_destabilizer_column"
    );
    if count_work {
        print!(
            ",gates,gate_touches,meas_det_work,meas_rand_work,meas_count_det,meas_count_rand,meas_refactor_work,meas_decouple_work"
        );
    }
    println!();
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

// Only the Rust simulator exposes the tableau needed for statistics and work counts.
trait ScalingSimulator: CliffordGateable {
    const REPORTS_DETERMINISM: bool = true;

    fn tableau_stats(&self) -> Option<(usize, usize, usize, usize)> {
        None
    }

    fn counted_gates(
        &mut self,
        _ops: &Schedule,
        _x: &[QubitId],
        _z: &[QubitId],
        _work: &mut WorkCounters,
    ) -> f64 {
        unreachable!("--count-work requires SparseStabGeneric")
    }

    fn counted_measure_reset(
        &mut self,
        _qubits: &[QubitId],
        _work: &mut WorkCounters,
        _seconds: &mut f64,
    ) -> Vec<MeasurementResult> {
        unreachable!("--count-work requires SparseStabGeneric")
    }
}

impl<S: IndexSet> ScalingSimulator for SparseStabGeneric<S> {
    fn tableau_stats(&self) -> Option<(usize, usize, usize, usize)> {
        let (s, sc) = stats(self.stabs());
        let (d, dc) = stats(self.destabs());
        Some((s, d, sc, dc))
    }

    fn counted_gates(
        &mut self,
        ops: &Schedule,
        x: &[QubitId],
        z: &[QubitId],
        work: &mut WorkCounters,
    ) -> f64 {
        counted_gates(self, ops, x, z, work)
    }

    fn counted_measure_reset(
        &mut self,
        qubits: &[QubitId],
        work: &mut WorkCounters,
        seconds: &mut f64,
    ) -> Vec<MeasurementResult> {
        counted_measure_reset(self, qubits, work, seconds)
    }
}

#[cfg(feature = "cppsparsestab")]
impl ScalingSimulator for pecos_cppsparsestab::CppSparseStab {
    const REPORTS_DETERMINISM: bool = false;
}

fn run<S: IndexSet>(config: &Config) {
    println!("# measurement_rule={:?}", config.rule);
    run_simulator(config, |n| {
        SparseStabGeneric::<S>::with_rng(n, PecosRng::seed_from_u64(config.seed))
            .with_measurement_rule(config.rule)
    });
}

fn run_cpp(config: &Config) {
    #[cfg(not(feature = "cppsparsestab"))]
    {
        let _ = config;
        panic!(
            "cpp requires the benchmarks feature cppsparsestab; build with --features cppsparsestab"
        );
    }
    #[cfg(feature = "cppsparsestab")]
    {
        assert!(
            !config.count_work,
            "cpp does not support --count-work: its tableau internals are not exposed"
        );
        assert_eq!(
            config.rule,
            MeasurementRule::AaronsonGottesman,
            "--rule does not apply to cpp; only the default ag value is accepted"
        );
        println!("# simulator=cpp; measurement_rule=NA; tableau_statistics=NA");
        run_simulator(config, |n| {
            pecos_cppsparsestab::CppSparseStab::with_seed(n, config.seed)
        });
    }
}

fn run_simulator<T: ScalingSimulator>(config: &Config, make_sim: impl FnOnce(usize) -> T) {
    let construction = Instant::now();
    let ops = match config.code {
        Code::Surface => surface_schedule(config.d),
        Code::Color => color_schedule(config.d),
    };
    let mut rng = PecosRng::seed_from_u64(config.seed);
    // Fix stochastic operation lists outside all round timers.
    let draws: Vec<_> = (0..config.rounds)
        .map(|_| {
            let (mut x, mut z, mut resets) = (Vec::new(), Vec::new(), Vec::new());
            for q in 0..ops.data_qubits {
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
    let mut sim = make_sim(ops.data_qubits + ops.ancillas.len());
    let construction_seconds = construction.elapsed().as_secs_f64();
    if matches!(config.code, Code::Color) {
        println!(
            "# color666 d{}: data={}, faces={}, simulator={}, x_cx_layers={}, z_cx_layers={}",
            config.d,
            ops.data_qubits,
            ops.x_ancillas.len(),
            ops.data_qubits + ops.ancillas.len(),
            ops.layers.len(),
            ops.z_layers.len()
        );
    }
    print_header(config.count_work);
    let mut first = Vec::new();
    for (round, (x, z, resets)) in draws.iter().enumerate() {
        let mut work = WorkCounters::default();
        let (gate_seconds, measurement_seconds, measurements, data_measurements) = if config
            .count_work
        {
            let gates = sim.counted_gates(&ops, x, z, &mut work);
            let mut seconds = 0.0;
            let measurements = sim.counted_measure_reset(&ops.ancillas, &mut work, &mut seconds);
            let data_measurements = sim.counted_measure_reset(resets, &mut work, &mut seconds);
            (gates, seconds, measurements, data_measurements)
        } else {
            let start = Instant::now();
            sim.x(x).z(z);
            for layer in ops.gate_layers() {
                layer.apply(&mut sim);
            }
            let gates = start.elapsed().as_secs_f64();
            let start = Instant::now();
            let measurements = sim.mz(&ops.ancillas);
            sim.pz(&ops.ancillas);
            let data_measurements = sim.mz(resets);
            sim.pz(resets);
            (
                gates,
                start.elapsed().as_secs_f64(),
                measurements,
                data_measurements,
            )
        };
        let nondeterministic = if T::REPORTS_DETERMINISM {
            measurements
                .iter()
                .chain(&data_measurements)
                .filter(|m| !m.is_deterministic)
                .count()
                .to_string()
        } else {
            "NA".to_owned()
        };
        if config.p == 0.0 && config.f == 0.0 {
            // The C++ wrapper hardcodes is_deterministic=false, so skip only its determinism assertions.
            if T::REPORTS_DETERMINISM {
                for (&is_x, m) in ops.is_x.iter().zip(&measurements) {
                    assert_eq!(m.is_deterministic, round > 0 || !is_x);
                }
            }
            if round == 0 {
                for (&is_x, m) in ops.is_x.iter().zip(&measurements) {
                    if !is_x {
                        assert!(!m.outcome);
                    }
                }
                first = measurements.iter().map(|m| m.outcome).collect();
            } else {
                for (m, &initial) in measurements.iter().zip(&first) {
                    assert_eq!(m.outcome, initial);
                }
            }
        }
        let tableau_stats = (round % config.stats_every == 0)
            .then(|| sim.tableau_stats())
            .flatten();
        if let Some((s, d, sc, dc)) = tableau_stats {
            print!(
                "{round},{gate_seconds:.9},{measurement_seconds:.9},{nondeterministic},{s},{d},{sc},{dc}"
            );
        } else {
            print!(
                "{round},{gate_seconds:.9},{measurement_seconds:.9},{nondeterministic},NA,NA,NA,NA"
            );
        }
        if config.count_work {
            work.print();
        }
        println!();
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
    let mut code = Code::Surface;
    if let Some(i) = args.iter().position(|s| s == "--code") {
        code = match args
            .get(i + 1)
            .expect("--code requires surface|color")
            .as_str()
        {
            "surface" => Code::Surface,
            "color" => Code::Color,
            other => panic!("unknown code {other}; expected surface|color"),
        };
        args.drain(i..=i + 1);
    }
    let mut rule = MeasurementRule::AaronsonGottesman;
    if let Some(i) = args.iter().position(|s| s == "--rule") {
        rule = match args
            .get(i + 1)
            .expect("--rule requires ag|refactor")
            .as_str()
        {
            "ag" => MeasurementRule::AaronsonGottesman,
            "refactor" => MeasurementRule::RefactorDecouple,
            other => panic!("unknown rule {other}; expected ag|refactor"),
        };
        args.drain(i..=i + 1);
    }
    let count_work = if let Some(i) = args.iter().position(|s| s == "--count-work") {
        args.remove(i);
        true
    } else {
        false
    };
    assert!(
        (3..=6).contains(&args.len()),
        "usage: SET D ROUNDS [P [F [SEED]]] [--stats-every K] [--code surface|color] [--rule ag|refactor] [--count-work]"
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
        code,
        count_work,
        rule,
    };
    assert!(config.d >= 3 && config.d % 2 == 1 && config.rounds > 0 && stats_every > 0);
    assert!((0.0..=1.0).contains(&config.p) && (0.0..=1.0).contains(&config.f));
    match args[0].as_str() {
        "cpp" => run_cpp(&config),
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

    #[test]
    fn bell_pair_work_includes_reset_measurements() {
        let mut sim =
            SparseStabGeneric::<AdaptiveIndexSet>::with_rng(2, PecosRng::seed_from_u64(1434));
        let mut work = WorkCounters::default();
        let h = GateLayer::H(&[QubitId(0)]);
        h.count_and_apply(&mut sim, &mut work);
        let cx = GateLayer::Cx(&[(QubitId(0), QubitId(1))]);
        cx.count_and_apply(&mut sim, &mut work);
        let results =
            counted_measure_reset(&mut sim, &[QubitId(0), QubitId(1)], &mut work, &mut 0.0);
        assert!(!results[0].is_deterministic);
        assert!(results[1].is_deterministic);
        assert_eq!(results[0].outcome, results[1].outcome);
        assert_eq!(
            work,
            WorkCounters {
                gates: 2,
                gate_touches: 4,
                meas_det_work: 7,
                meas_rand_work: 2,
                meas_count_det: 3,
                meas_count_rand: 1,
                meas_refactor_work: 0,
                meas_decouple_work: 0,
            }
        );
    }

    #[test]
    fn refactor_bell_pair_work_includes_reset_measurements() {
        let mut sim = SparseStabGeneric::<SortedVecSet>::with_seed(2, 0)
            .with_measurement_rule(MeasurementRule::RefactorDecouple);
        sim.h(&[QubitId(0)]).cx(&[(QubitId(0), QubitId(1))]);
        let mut work = WorkCounters::default();
        let results =
            counted_measure_reset(&mut sim, &[QubitId(0), QubitId(1)], &mut work, &mut 0.0);
        assert!(!results[0].is_deterministic && results[1].is_deterministic);
        assert_eq!(results[0].outcome, results[1].outcome);
        assert_eq!(work.meas_count_rand, 1);
        assert_eq!(work.meas_count_det, 3);
        assert_eq!(work.meas_rand_work, 2);
        assert_eq!(work.meas_det_work, 3);
        assert_eq!(work.meas_refactor_work, 0);
        // Post-update S1=Z0 Z1 and D0=X0 X1: one strip and two discarded entries.
        // Subsequent singleton measurements are already isolated.
        assert_eq!(work.meas_decouple_work, 3);
    }

    #[test]
    fn deterministic_refactor_work_uses_input_basis() {
        let mut sim = SparseStabGeneric::<SortedVecSet>::with_seed(2, 0)
            .with_measurement_rule(MeasurementRule::RefactorDecouple);
        sim.cx(&[(QubitId(0), QubitId(1))]);
        let mut work = WorkCounters::default();
        work.measure(&mut sim, QubitId(1), &mut 0.0);
        // R={0,1}, weights S=(1,2), D=(2,1), so r0=1.
        assert_eq!(work.meas_det_work, 3);
        assert_eq!(work.meas_refactor_work, 3);
        assert_eq!(work.meas_decouple_work, 0);
        assert!(sim.mz_forced(1, false).is_deterministic);
        work.measure(&mut sim, QubitId(1), &mut 0.0);
        assert_eq!(work.meas_det_work, 4);
        assert_eq!(work.meas_refactor_work, 3);
        assert_eq!(work.meas_decouple_work, 0);
    }

    fn check_random_decouple_work(n: usize, expected: usize) {
        let mut sim = SparseStabGeneric::<SortedVecSet>::with_seed(n, 0)
            .with_measurement_rule(MeasurementRule::RefactorDecouple);
        sim.h(&[QubitId(0)]);
        for q in 1..n {
            sim.cx(&[(QubitId(0), QubitId(q))]);
        }
        let mut work = WorkCounters::default();
        let result = work.measure(&mut sim, QubitId(0), &mut 0.0);
        assert!(!result.is_deterministic);
        // The new D0 is X on all n qubits; n-1 remaining S rows lose Z0.
        assert_eq!(work.meas_decouple_work, expected, "n={n}");
        assert_eq!(work.meas_refactor_work, 0);
    }

    #[test]
    fn random_decouple_work_ghz() {
        check_random_decouple_work(3, 5);
    }

    #[test]
    fn random_decouple_work_single_h() {
        check_random_decouple_work(1, 1);
    }

    #[test]
    fn overlapping_cx_counts_use_each_gates_input_state() {
        let mut sim =
            SparseStabGeneric::<AdaptiveIndexSet>::with_rng(3, PecosRng::seed_from_u64(1434));
        sim.h(&[QubitId(0)]);
        let mut work = WorkCounters::default();
        GateLayer::Cx(&[(QubitId(0), QubitId(1)), (QubitId(1), QubitId(2))])
            .count_and_apply(&mut sim, &mut work);
        // The first CX touches two rows. It adds X support on qubit 1,
        // making the second CX touch three rows instead of its initial two.
        assert_eq!(work.gates, 2);
        assert_eq!(work.gate_touches, 5);
    }

    fn check_counting_preserves_trajectory<S: IndexSet>(ops: &Schedule, rule: MeasurementRule) {
        let n = ops.data_qubits + ops.ancillas.len();
        let mut counted = SparseStabGeneric::<S>::with_rng(n, PecosRng::seed_from_u64(1434))
            .with_measurement_rule(rule);
        let mut batched = SparseStabGeneric::<S>::with_rng(n, PecosRng::seed_from_u64(1434))
            .with_measurement_rule(rule);
        for round in 0..4 {
            let x = [QubitId(round)];
            let z = [QubitId(round + 1)];
            let resets = [QubitId(round + 2), QubitId(round + 3)];
            let mut work = WorkCounters::default();
            counted_gates(&mut counted, ops, &x, &z, &mut work);
            batched.x(&x).z(&z);
            for layer in ops.gate_layers() {
                layer.apply(&mut batched);
            }
            for qubits in [&ops.ancillas[..], &resets[..]] {
                let actual = counted_measure_reset(&mut counted, qubits, &mut work, &mut 0.0);
                let expected = batched.mz(qubits);
                batched.pz(qubits);
                let outcomes = |results: Vec<MeasurementResult>| {
                    results
                        .into_iter()
                        .map(|m| (m.outcome, m.is_deterministic))
                        .collect::<Vec<_>>()
                };
                assert_eq!(outcomes(actual), outcomes(expected));
            }
            assert_eq!(
                work.meas_count_det + work.meas_count_rand,
                2 * (ops.ancillas.len() + resets.len())
            );
            assert_eq!(counted.stab_tableau(), batched.stab_tableau());
            assert_eq!(counted.destab_tableau(), batched.destab_tableau());
        }
    }

    #[test]
    fn counting_preserves_noisy_reset_trajectory() {
        for rule in [
            MeasurementRule::AaronsonGottesman,
            MeasurementRule::RefactorDecouple,
        ] {
            for ops in [surface_schedule(5), color_schedule(5)] {
                check_counting_preserves_trajectory::<AdaptiveIndexSet>(&ops, rule);
                check_counting_preserves_trajectory::<BitSet>(&ops, rule);
                check_counting_preserves_trajectory::<SortedVecSet>(&ops, rule);
                check_counting_preserves_trajectory::<VecSet<usize>>(&ops, rule);
            }
        }
    }

    #[test]
    fn color_layers_cover_each_face_once() {
        for d in [3, 5, 15, 31, 63] {
            let code = color_code(d);
            let ops = color_schedule(d);
            let n = ops.data_qubits + ops.ancillas.len();
            for (layers, offset, x_type) in [
                (&ops.layers, ops.data_qubits, true),
                (&ops.z_layers, ops.data_qubits + code.faces.len(), false),
            ] {
                let mut supports = vec![Vec::new(); code.faces.len()];
                for layer in layers {
                    let mut seen = vec![false; n];
                    for &(c, t) in layer {
                        assert!(!seen[c.index()] && !seen[t.index()]);
                        seen[c.index()] = true;
                        seen[t.index()] = true;
                        let (a, q) = if x_type { (c, t) } else { (t, c) };
                        supports[a.index() - offset].push(q.index());
                    }
                }
                for (support, face) in supports.iter_mut().zip(&code.faces) {
                    support.sort_unstable();
                    assert_eq!(*support, face.qubits);
                }
            }
        }
    }
}
