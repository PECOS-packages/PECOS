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

//! Compare a per-shot clone-and-MZ loop with prefix-sharing bitstring sampling.

use std::collections::HashSet;
use std::hint::black_box;
use std::time::Instant;

use pecos_core::{Angle64, QubitId, RngManageable};
use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable};
use pecos_stab_tn::stab_mps::StabMps;

fn next_seeded(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}

fn seeded_circuit(num_qubits: usize, t_count: usize, seed: u64) -> StabMps {
    let mut order = (0..num_qubits).collect::<Vec<_>>();
    let mut random_state = seed;
    for i in (1..num_qubits).rev() {
        let j = next_seeded(&mut random_state) as usize % (i + 1);
        order.swap(i, j);
    }

    let mut stn = StabMps::with_seed(num_qubits, seed);
    stn.h(&[QubitId(order[0])]);
    for pair in order.windows(2) {
        stn.cx(&[(QubitId(pair[0]), QubitId(pair[1]))]);
    }

    let t = Angle64::QUARTER_TURN / 2u64;
    for _ in 0..t_count {
        let q = next_seeded(&mut random_state) as usize % num_qubits;
        stn.rz(t, &[QubitId(q)]);
        let a = next_seeded(&mut random_state) as usize % num_qubits;
        let mut b = next_seeded(&mut random_state) as usize % num_qubits;
        if a == b {
            b = (b + 1) % num_qubits;
        }
        stn.cz(&[(QubitId(a), QubitId(b))]);
    }
    stn
}

fn distinct_internal_prefixes(shots: &[Vec<bool>]) -> usize {
    let mut prefixes = HashSet::new();
    if let Some(first) = shots.first() {
        for shot in shots {
            for depth in 0..first.len() {
                prefixes.insert(shot[..depth].to_vec());
            }
        }
    }
    prefixes.len()
}

fn sample_with_fresh_clones(base: &StabMps, num_qubits: usize, num_shots: usize) -> Vec<Vec<bool>> {
    (0..num_shots)
        .map(|shot| {
            let mut simulator = base.clone();
            simulator.set_seed(0xF2E5_0000_u64.wrapping_add(shot as u64));
            simulator
                .mz(&(0..num_qubits).map(QubitId).collect::<Vec<_>>())
                .into_iter()
                .map(|result| result.outcome)
                .collect()
        })
        .collect()
}

fn main() {
    println!(
        "| n | T | shots | fresh clone + MZ loop | sample_bitstrings | speedup | distinct internal prefixes |"
    );
    println!("|---:|---:|---:|---:|---:|---:|---:|");

    for num_qubits in [8usize, 12, 16, 20] {
        let t_count = num_qubits / 2;
        let base = seeded_circuit(num_qubits, t_count, 0x5eed_u64 + num_qubits as u64);
        for num_shots in [100usize, 1_000, 10_000] {
            let start = Instant::now();
            let per_shot = sample_with_fresh_clones(&base, num_qubits, num_shots);
            let per_shot_elapsed = start.elapsed();
            black_box(&per_shot);

            let mut prefix = base.clone();
            let start = Instant::now();
            let prefix_shots = prefix.sample_bitstrings(num_shots);
            let prefix_elapsed = start.elapsed();
            let distinct_prefixes = distinct_internal_prefixes(&prefix_shots);
            black_box(&prefix_shots);

            let per_shot_ms = per_shot_elapsed.as_secs_f64() * 1_000.0;
            let prefix_ms = prefix_elapsed.as_secs_f64() * 1_000.0;
            let speedup = per_shot_ms / prefix_ms;
            println!(
                "| {num_qubits} | {t_count} | {num_shots} | {per_shot_ms:.3} ms | \
                 {prefix_ms:.3} ms | {speedup:.2}x | {distinct_prefixes} |"
            );
        }
    }
}
