// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0.

//! Repeated memory shots, including ancilla reset.

use criterion::{BenchmarkId, Criterion};
use pecos_core::{BitSet, IndexSet, PauliOperator};
use pecos_qec::{
    BivariateBicycleCode, BivariateTricycleCode, HypergraphProductCode, MemoryBasis,
    ParityCheckMatrix, SurfaceCode, bb_memory_circuit, coloration_memory_circuit,
    connected_cluster_code_distance, discover_logical_operators,
};
use pecos_quantum::{AnnotationKind, GateType, TickCircuit};
use pecos_random::PecosRng;
use pecos_simulators::{CircuitExecutor, CliffordGateable, SparseStabGeneric, SparseStabSet};
use rand::seq::SliceRandom;
use std::{collections::BTreeMap, hint::black_box, ops::Range, time::Duration};

const SEED: u64 = 1434;

struct Workload {
    family: &'static str,
    instance: String,
    circuit: TickCircuit,
    data: Range<usize>,
    n: usize,
    rounds: usize,
}

fn css(
    family: &'static str,
    instance: String,
    hx: &ParityCheckMatrix,
    hz: &ParityCheckMatrix,
    rounds: usize,
) -> Workload {
    let start = hx.num_checks();
    let circuit = coloration_memory_circuit(hx, hz, rounds, MemoryBasis::Z)
        .unwrap_or_else(|error| panic!("{family}/{instance} memory circuit: {error}"));
    Workload {
        family,
        instance,
        circuit,
        data: start..start + hx.num_qubits(),
        n: start + hx.num_qubits() + hz.num_checks(),
        rounds,
    }
}

// Triangular hexagonal patch of arXiv:1911.00355, Fig. 1(a), using the
// integer-coordinate convention documented at https://qecsim.github.io/api/models/color.html.
// Inside 0 <= column <= row <= 3(d-1)/2, sites with (row+column) mod 3 = 2
// are face centres; the remaining sites are data qubits. The six neighbouring
// sites form each hexagon, clipped to four vertices at the three boundaries.
fn color_checks(d: usize) -> ParityCheckMatrix {
    assert!(d >= 3 && d % 2 == 1, "color666 d{d}: odd distance >= 3");
    let bound = i32::try_from(3 * (d - 1) / 2).expect("color666 coordinate bound fits i32");
    let mut qubits = BTreeMap::new();
    let mut centres = Vec::new();
    for row in 0..=bound {
        for col in 0..=row {
            if (row + col) % 3 == 2 {
                centres.push((row, col));
            } else {
                qubits.insert((row, col), qubits.len());
            }
        }
    }
    let n = qubits.len();
    assert_eq!(n, (3 * d * d + 1) / 4, "color666 d{d}: data count");
    assert_eq!(
        centres.len(),
        (3 * d * d - 3) / 8,
        "color666 d{d}: face count"
    );
    for side in 0..3 {
        let count = qubits
            .keys()
            .filter(|&&(r, c)| match side {
                0 => c == 0,
                1 => r == c,
                _ => r == bound,
            })
            .count();
        assert_eq!(count, d, "color666 d{d}: boundary {side} length");
    }
    let mut incidence = vec![0; n];
    let mut colours_at_qubit = vec![0u8; n];
    let rows = centres
        .iter()
        .map(|&(r, c)| {
            let mut row = vec![0u8; n];
            let colour = 1u8 << (r % 3);
            for (dr, dc) in [(-1, -1), (-1, 0), (0, -1), (0, 1), (1, 0), (1, 1)] {
                if let Some(&q) = qubits.get(&(r + dr, c + dc)) {
                    row[q] = 1;
                    incidence[q] += 1;
                    assert_eq!(
                        colours_at_qubit[q] & colour,
                        0,
                        "color666 d{d}: adjacent faces share a colour at qubit {q}"
                    );
                    colours_at_qubit[q] |= colour;
                }
            }
            let weight: usize = row.iter().copied().map(usize::from).sum();
            assert!(
                matches!(weight, 4 | 6),
                "color666 d{d}: face ({r},{c}) weight {weight}"
            );
            row
        })
        .collect::<Vec<_>>();
    assert!(
        incidence.iter().all(|&count| (1..=3).contains(&count)),
        "color666 d{d}: qubit incidence"
    );
    assert_eq!(
        incidence.iter().filter(|&&count| count == 1).count(),
        3,
        "color666 d{d}: corner count"
    );
    for (i, a) in rows.iter().enumerate() {
        for (j, b) in rows.iter().enumerate() {
            let overlap = a.iter().zip(b).fold(0, |parity, (x, z)| parity ^ (x & z));
            assert_eq!(overlap, 0, "color666 d{d}: Hx Hz^T at ({i},{j})");
        }
    }
    let h = ParityCheckMatrix::from_dense(rows).expect("color666 binary incidence matrix");
    assert_eq!(n - 2 * h.rank(), 1, "color666 d{d}: logical count");
    if d <= 7 {
        color_distance(&h, d);
    }
    eprintln!(
        "color666 d{d}: n={n}, faces={}, simulator={}, rounds={d}, distance={}",
        h.num_checks(),
        n + 2 * h.num_checks(),
        if d <= 7 { "verified" } else { "not searched" }
    );
    h
}

fn color_distance(h: &ParityCheckMatrix, d: usize) {
    let n = h.num_qubits();
    let mut stabilizers = h.to_x_stabilizers();
    stabilizers.extend(h.to_z_stabilizers());
    let logicals = discover_logical_operators(n, &stabilizers)
        .unwrap_or_else(|error| panic!("color666 d{d}: logical discovery: {error}"));
    assert_eq!(
        logicals.num_logical_qubits, 1,
        "color666 d{d}: discovered logical count"
    );
    // Test commutation with the entire discovered logical basis, so this also
    // works if discovery chooses mixed X/Z representatives. For pure X errors
    // use their Z components, and conversely for pure Z errors. The CSS code
    // distance is the minimum of these two pure-error distances.
    for x_errors in [true, false] {
        let rows = logicals
            .logical_xs
            .iter()
            .chain(&logicals.logical_zs)
            .map(|logical| {
                let mut row = vec![0; n];
                let positions = if x_errors {
                    logical.z_positions()
                } else {
                    logical.x_positions()
                };
                for q in positions {
                    row[q] = 1;
                }
                row
            })
            .collect();
        let l = ParityCheckMatrix::from_dense(rows).expect("color666 logical support matrix");
        let result = connected_cluster_code_distance(h, &l, d).unwrap_or_else(|| {
            panic!("color666 d{d}: no logical found through weight {d}, X errors={x_errors}")
        });
        assert_eq!(
            result.distance, d,
            "color666 d{d}: distance, X errors={x_errors}"
        );
    }
}

fn corpus() -> Vec<Workload> {
    let mut workloads = Vec::new();
    for d in [3, 5, 7, 11, 15, 21] {
        let code = SurfaceCode::rotated(d).expect("valid surface-code distance");
        let matrix = |checks: &[pecos_qec::StabilizerCheck]| {
            ParityCheckMatrix::from_dense(
                checks
                    .iter()
                    .map(|check| {
                        let mut row = vec![0; code.num_data_qubits()];
                        for q in check.qubits() {
                            row[q] = 1;
                        }
                        row
                    })
                    .collect(),
            )
            .expect("valid surface-code parity check matrix")
        };
        let (hx, hz) = (matrix(code.x_stabilizers()), matrix(code.z_stabilizers()));
        assert_eq!(hx.num_qubits(), d * d);
        assert_eq!(hx.num_qubits() - hx.rank() - hz.rank(), 1);
        workloads.push(css("surface", format!("d{d}"), &hx, &hz, d));
    }
    for d in [3, 5, 7, 9, 11] {
        let h = color_checks(d);
        workloads.push(css("color666", format!("d{d}"), &h, &h, d));
    }
    // Bravyi et al., Nature 627, 778 (2024), cross-checked against
    // https://errorcorrectionzoo.org/c/bb108, https://errorcorrectionzoo.org/c/bb288
    // and arXiv:2511.13560.
    let a = [(3, 0), (0, 1), (0, 2)];
    let b = [(0, 3), (1, 0), (2, 0)];
    for (n, k, l, m, a, b) in [
        (72, 12, 6, 6, a, b),
        (
            90,
            8,
            15,
            3,
            [(9, 0), (0, 1), (0, 2)],
            [(0, 0), (2, 0), (7, 0)],
        ),
        (108, 8, 9, 6, a, b),
        (144, 12, 12, 6, a, b),
        (288, 12, 12, 12, [(3, 0), (0, 2), (0, 7)], b),
    ] {
        let code = BivariateBicycleCode::new(l, m, &a, &b).expect("valid BB parameters");
        assert_eq!(
            (code.num_qubits(), code.num_logical_qubits()),
            (n, k),
            "BB n{n} ({l}x{m}) parameters"
        );
        workloads.push(Workload {
            family: "bb",
            instance: format!("n{n}"),
            circuit: bb_memory_circuit(l, m, &a, &b, 6, MemoryBasis::Z)
                .expect("valid BB memory circuit"),
            data: l * m..3 * l * m,
            n: 2 * n,
            rounds: 6,
        });
    }
    let hamming = ParityCheckMatrix::from_dense(vec![
        vec![1, 0, 1, 0, 1, 0, 1],
        vec![0, 1, 1, 0, 0, 1, 1],
        vec![0, 0, 0, 1, 1, 1, 1],
    ])
    .expect("valid Hamming parity check matrix");
    let mut rng = PecosRng::seed_from_u64(SEED);
    let regular = loop {
        let mut sockets: Vec<_> = (0..15).flat_map(|r| [r; 4]).collect();
        sockets.shuffle(&mut rng);
        let mut rows = vec![vec![0; 20]; 15];
        let mut simple = true;
        for (q, neighbors) in sockets.as_chunks::<3>().0.iter().enumerate() {
            for &r in neighbors {
                simple &= rows[r][q] == 0;
                rows[r][q] = 1;
            }
        }
        if simple {
            break ParityCheckMatrix::from_dense(rows).expect("valid regular parity check matrix");
        }
    };
    for (name, h, n, total, expected_k) in [
        ("hamming", hamming, 58, 100, Some(16)),
        ("regular20", regular, 625, 1225, None),
    ] {
        let code = HypergraphProductCode::new(&h, &h).expect("valid HGP matrices");
        assert_eq!(code.num_qubits(), n);
        assert!(code.num_logical_qubits() > 0);
        if let Some(k) = expected_k {
            assert_eq!(code.num_logical_qubits(), k, "HGP {name} logical count");
        }
        eprintln!("HGP {name}: n={n}, k={}", code.num_logical_qubits());
        let workload = css("hgp", name.into(), code.hx(), code.hz(), 6);
        assert_eq!(workload.n, total);
        workloads.push(workload);
    }
    let code = BivariateTricycleCode::new(
        3,
        9,
        &[(1, 3), (2, 1)],
        &[(0, 0), (1, 8)],
        &[(2, 4), (2, 6)],
    )
    .expect("valid tricycle parameters");
    assert_eq!((code.num_qubits(), code.num_logical_qubits()), (81, 3));
    let circuit = coloration_memory_circuit(code.hx(), code.hz(), 6, MemoryBasis::Z)
        .expect("tricycle 3x9 memory circuit must accept dependent check rows");
    assert_eq!(
        (code.hx().num_checks(), code.hz().num_checks()),
        (27, 81),
        "tricycle 3x9 checks"
    );
    workloads.push(Workload {
        family: "tricycle",
        instance: "3x9".into(),
        circuit,
        data: 27..108,
        n: 189,
        rounds: 6,
    });
    workloads
}

fn reset_arm(work: &Workload) -> TickCircuit {
    let mut circuit = work.circuit.clone();
    let ends: Vec<_> = circuit
        .ticks()
        .iter()
        .enumerate()
        .filter_map(|(i, tick)| {
            tick.gate_batches()
                .iter()
                .any(|g| {
                    g.gate_type == GateType::MX
                        && g.qubits.iter().all(|q| q.index() < work.data.start)
                })
                .then_some(i)
        })
        .collect();
    assert_eq!(ends.len(), work.rounds);
    let mut rng = PecosRng::seed_from_u64(SEED);
    let draws: Vec<Vec<_>> = ends
        .iter()
        .map(|_| {
            work.data
                .clone()
                .filter(|_| rng.random_bool(0.01))
                .collect()
        })
        .collect();
    for (&idx, qubits) in ends.iter().zip(&draws).rev() {
        // PZ measures and conditionally flips internally, without allocating a
        // MeasId or adding to the executor's result vector. Existing annotation
        // IDs and their execution-order indices therefore remain valid.
        if !qubits.is_empty() {
            circuit.insert_tick(idx + 1).pz(qubits);
        }
    }
    assert_eq!(circuit.annotations(), work.circuit.annotations());
    let ids = |c: &TickCircuit| {
        c.ticks()
            .iter()
            .flat_map(pecos_quantum::Tick::gate_batches)
            .flat_map(|g| g.meas_ids.iter().copied())
            .collect::<Vec<_>>()
    };
    assert_eq!(ids(&circuit), ids(&work.circuit));
    circuit
}

fn outcomes<S: IndexSet>(circuit: &TickCircuit, n: usize) -> Vec<(bool, bool)> {
    let mut sim = SparseStabGeneric::<S>::with_rng(n, PecosRng::seed_from_u64(SEED));
    CircuitExecutor::new(circuit)
        .run(&mut sim)
        .expect("supported workload circuit")
        .into_iter()
        .map(|r| (r.outcome, r.is_deterministic))
        .collect()
}

fn replay<S: CliffordGateable>(sim: &mut S, circuit: &TickCircuit) -> Vec<(bool, bool)> {
    let mut results = Vec::new();
    for tick in circuit.ticks() {
        for gate in tick.gate_batches() {
            match gate.gate_type {
                GateType::PZ => {
                    sim.pz(&gate.qubits);
                }
                GateType::PX => {
                    sim.pz(&gate.qubits);
                    sim.h(&gate.qubits);
                }
                GateType::CX => {
                    let pairs: Vec<_> = gate
                        .qubits
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|p| (p[0], p[1]))
                        .collect();
                    sim.cx(&pairs);
                }
                GateType::MX => {
                    // Match CircuitExecutor: MX rotates into Z and leaves the
                    // measured ancilla there; the next PX prepares it afresh.
                    sim.h(&gate.qubits);
                    results.extend(
                        sim.mz(&gate.qubits)
                            .into_iter()
                            .map(|r| (r.outcome, r.is_deterministic)),
                    );
                }
                GateType::MZ => results.extend(
                    sim.mz(&gate.qubits)
                        .into_iter()
                        .map(|r| (r.outcome, r.is_deterministic)),
                ),
                GateType::Idle => {}
                other => panic!("unsupported workload gate {other:?}"),
            }
        }
    }
    results
}

fn bench_set<S: IndexSet>(
    c: &mut Criterion,
    work: &Workload,
    circuit: &TickCircuit,
    arm: &str,
    set: &str,
) {
    let mut group = c.benchmark_group(format!("SparseStab Workloads/{}", work.family));
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(2))
        .warm_up_time(Duration::from_millis(500));
    let mut sim = SparseStabGeneric::<S>::with_rng(work.n, PecosRng::seed_from_u64(SEED));
    let executor = CircuitExecutor::new(circuit);
    group.bench_function(
        BenchmarkId::new(&work.instance, format!("f={arm}/{set}")),
        |b| {
            b.iter(|| {
                sim.reset();
                black_box(
                    executor
                        .run(&mut sim)
                        .expect("validated workload execution"),
                );
            });
        },
    );
    if (work.family == "surface" && work.instance == "d3")
        || (work.family == "bb" && work.instance == "n144")
    {
        let mut direct = SparseStabGeneric::<S>::with_rng(work.n, PecosRng::seed_from_u64(SEED));
        group.bench_function(
            BenchmarkId::new(&work.instance, format!("f={arm}/{set}/direct")),
            |b| {
                b.iter(|| {
                    direct.reset();
                    black_box(replay(&mut direct, circuit));
                });
            },
        );
    }
    group.finish();
}

fn validate<S: IndexSet>(work: &Workload, circuit: &TickCircuit, expected: &[(bool, bool)]) {
    assert_eq!(
        outcomes::<S>(circuit, work.n),
        expected,
        "{}/{}: outcomes differ",
        work.family,
        work.instance
    );
    if (work.family == "surface" && work.instance == "d3")
        || (work.family == "bb" && work.instance == "n144")
    {
        let mut direct = SparseStabGeneric::<S>::with_rng(work.n, PecosRng::seed_from_u64(SEED));
        assert_eq!(replay(&mut direct, circuit), expected);
    }
}

pub fn benchmarks(c: &mut Criterion) {
    let start = std::time::Instant::now();
    let workloads = corpus();
    let mut prepared = Vec::new();
    for work in &workloads {
        let clean = outcomes::<BitSet>(&work.circuit, work.n);
        for annotation in work.circuit.annotations() {
            if let AnnotationKind::Detector {
                measurement_ids, ..
            } = &annotation.kind
            {
                assert!(
                    !measurement_ids
                        .iter()
                        .fold(false, |p, id| p ^ clean[id.index()].0),
                    "{}/{} detector",
                    work.family,
                    work.instance
                );
            }
        }
        for (arm, circuit) in [("0", work.circuit.clone()), ("0.01", reset_arm(work))] {
            let expected = outcomes::<BitSet>(&circuit, work.n);
            validate::<BitSet>(work, &circuit, &expected);
            validate::<SparseStabSet>(work, &circuit, &expected);
            prepared.push((work, arm, circuit));
        }
    }
    eprintln!(
        "SparseStab Workloads setup time (corpus and validation): {:.3}s",
        start.elapsed().as_secs_f64()
    );
    for (work, arm, circuit) in prepared {
        bench_set::<BitSet>(c, work, &circuit, arm, "bitset");
        bench_set::<SparseStabSet>(c, work, &circuit, arm, "adaptive");
    }
    eprintln!(
        "SparseStab Workloads module wall time: {:.3}s",
        start.elapsed().as_secs_f64()
    );
}
