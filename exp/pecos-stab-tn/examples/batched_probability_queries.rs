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

//! Release-mode performance falsifier for shared-prefix probability queries.
//!
//! This mirrors the measured cross-implementation workload shape: n=32,
//! T=n basis-mixed sparse Clifford+T, chi=64, cutoff 1e-12, adaptive
//! truncation disabled, and 16 deterministic random bitstring queries. It
//! compares the shared `prob_bitstrings` trie with the mutation baseline of a
//! fresh singular forced-projection walk per query.
//!
//! Run pinned and single-threaded when possible:
//! `RAYON_NUM_THREADS=1 taskset -c 2 cargo run --release -p pecos-stab-tn --example batched_probability_queries`.

use std::hint::black_box;
use std::time::{Duration, Instant};

use pecos_core::{Angle64, QubitId};
use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable};
use pecos_stab_tn::stab_mps::StabMps;

const NUM_QUBITS: usize = 32;
const NUM_QUERIES: usize = 16;
const CIRCUIT_SEED: u64 = 23_201;
const TIMED_REPETITIONS: usize = 3;

fn sparse_clifford_t_state() -> StabMps {
    let mut simulator = StabMps::builder(NUM_QUBITS)
        .seed(CIRCUIT_SEED)
        .max_bond_dim(64)
        .svd_cutoff(1e-12)
        .max_truncation_error(0.0)
        .merge_rz(false)
        .build();
    for q in 0..NUM_QUBITS {
        simulator.h(&[QubitId(q)]);
    }
    for q in 0..NUM_QUBITS - 1 {
        simulator.cx(&[(QubitId(q), QubitId(q + 1))]);
    }

    // Exact gate choices produced by pecos-perf's Python `random.Random`
    // stream for sparse_t_circuit(n=32, n_t=32, seed=23201).
    let injections = [
        (0, 26, true),
        (19, 11, true),
        (27, 7, false),
        (10, 25, false),
        (12, 25, false),
        (6, 14, false),
        (25, 11, false),
        (28, 31, false),
        (22, 0, false),
        (19, 21, true),
        (9, 21, false),
        (28, 7, false),
        (14, 24, false),
        (5, 6, true),
        (21, 5, false),
        (4, 14, false),
        (28, 2, false),
        (2, 18, true),
        (7, 6, true),
        (11, 25, false),
        (12, 4, false),
        (11, 18, false),
        (28, 26, true),
        (18, 15, false),
        (19, 7, true),
        (16, 12, false),
        (4, 29, true),
        (18, 20, false),
        (25, 8, true),
        (10, 9, false),
        (29, 7, true),
        (19, 17, true),
    ];
    let s_targets = [28, 8, 5, 16, 0, 14, 7, 1];
    for (injection, &(target, other, dagger)) in injections.iter().enumerate() {
        let angle = if dagger {
            -std::f64::consts::FRAC_PI_4
        } else {
            std::f64::consts::FRAC_PI_4
        };
        simulator.rz(Angle64::from_radians(angle), &[QubitId(target)]);
        simulator.h(&[QubitId(target)]);
        if injection & 1 == 0 {
            simulator.cx(&[(QubitId(target), QubitId(other))]);
        } else {
            simulator.cz(&[(QubitId(target), QubitId(other))]);
        }
        if injection % 4 == 3 {
            let q = s_targets[injection / 4];
            simulator.sz(&[QubitId(q)]);
        }
    }
    simulator.flush();
    simulator
}

fn random_queries() -> Vec<Vec<bool>> {
    // Exact unique queries produced by pecos-perf's `_queries(32, 23201)`;
    // each mask bit q becomes query[q].
    let masks = [
        0x2244_c5d0_u32,
        0x88d7_f354,
        0x37f3_360d,
        0xacaa_0389,
        0xaa6b_4fd0,
        0x1b30_c687,
        0x049f_ecbd,
        0x2932_a1fb,
        0x3bac_8b5b,
        0xf42b_e99c,
        0xdf9c_7891,
        0x42f1_2ff2,
        0x4fe1_0f33,
        0x4caa_06b3,
        0xf8d3_f1c8,
        0x1381_bcc5,
    ];
    masks
        .into_iter()
        .map(|mask| (0..NUM_QUBITS).map(|q| mask >> q & 1 != 0).collect())
        .collect()
}

fn median(samples: &mut [Duration]) -> Duration {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn distinct_nonempty_prefixes(queries: &[Vec<bool>]) -> usize {
    let mut prefixes = std::collections::HashSet::new();
    for query in queries {
        for depth in 1..=query.len() {
            prefixes.insert(query[..depth].to_vec());
        }
    }
    prefixes.len()
}

fn main() {
    let simulator = sparse_clifford_t_state();
    let queries = random_queries();

    // Prime numerical-library and allocator one-time work outside the table.
    let singular_warmup = queries
        .iter()
        .map(|bits| simulator.prob_bitstring(bits))
        .collect::<Vec<_>>();
    let batched_warmup = simulator.prob_bitstrings(&queries);
    assert_eq!(
        singular_warmup
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        batched_warmup
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
    black_box((&singular_warmup, &batched_warmup));

    let mut individual_totals = Vec::with_capacity(TIMED_REPETITIONS);
    let mut individual_by_query = (0..NUM_QUERIES)
        .map(|_| Vec::with_capacity(TIMED_REPETITIONS))
        .collect::<Vec<_>>();
    let mut batched_totals = Vec::with_capacity(TIMED_REPETITIONS);
    for _ in 0..TIMED_REPETITIONS {
        let individual_start = Instant::now();
        let mut individual = Vec::with_capacity(NUM_QUERIES);
        for (query_index, bits) in queries.iter().enumerate() {
            let query_start = Instant::now();
            individual.push(simulator.prob_bitstring(bits));
            individual_by_query[query_index].push(query_start.elapsed());
        }
        individual_totals.push(individual_start.elapsed());

        let batched_start = Instant::now();
        let batched = simulator.prob_bitstrings(&queries);
        batched_totals.push(batched_start.elapsed());
        assert_eq!(
            individual
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            batched
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        black_box((individual, batched));
    }

    let individual = median(&mut individual_totals).as_secs_f64() * 1_000.0;
    let batched = median(&mut batched_totals).as_secs_f64() * 1_000.0;
    let slowest_singular = individual_by_query
        .iter_mut()
        .map(|samples| median(samples).as_secs_f64() * 1_000.0)
        .fold(0.0_f64, f64::max);
    let speedup = individual / batched;
    let distinct_prefixes = distinct_nonempty_prefixes(&queries);

    println!(
        "| n | T | queries | path | forced-projection prefixes | median query time | relative to batch |"
    );
    println!("|---:|---:|---:|:---|---:|---:|---:|");
    println!(
        "| {NUM_QUBITS} | {NUM_QUBITS} | {NUM_QUERIES} | 16 singular fresh walks | \
         {} | {individual:.3} ms | {speedup:.2}x |",
        NUM_QUBITS * NUM_QUERIES
    );
    println!(
        "| {NUM_QUBITS} | {NUM_QUBITS} | {NUM_QUERIES} | shared-prefix batch | \
         {distinct_prefixes} | {batched:.3} ms | 1.00x |"
    );
    println!(
        "| {NUM_QUBITS} | {NUM_QUBITS} | 1 | slowest singular walk | \
         {NUM_QUBITS} | {slowest_singular:.3} ms | {:.2}x |",
        slowest_singular / batched
    );
    println!(
        "batch/fresh-sum={:.3}; batch/slowest-singular={:.3}; agreement=bit-for-bit",
        batched / individual,
        batched / slowest_singular
    );
    assert!(
        speedup > 1.05,
        "shared query trie did not beat fresh singular walks: speedup={speedup:.3}x"
    );
}
