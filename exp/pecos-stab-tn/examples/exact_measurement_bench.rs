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

//! Time one exact MZ on a fresh GHZ state with coefficient-MPS bond dimension one.
//! Run: `cargo run --release -p pecos-stab-tn --example exact_measurement_bench`.

use pecos_core::QubitId;
use pecos_simulators::CliffordGateable;
use pecos_stab_tn::stab_mps::{MeasurementMode, StabMps};
use std::hint::black_box;
use std::time::{Duration, Instant};

fn run(num_qubits: usize, samples: u64) -> f64 {
    let rank_ceiling = (1usize << (num_qubits / 2)).min(usize::MAX / 8);
    let mut elapsed = Duration::ZERO;
    for seed in 0..samples {
        let mut simulator = StabMps::builder(num_qubits)
            .seed(seed)
            .measurement(MeasurementMode::Exact)
            .max_bond_dim(rank_ceiling)
            .svd_cutoff(0.0)
            .max_truncation_error(0.0)
            .build();
        simulator.h(&[QubitId(0)]);
        for q in 1..num_qubits {
            simulator.cx(&[(QubitId(0), QubitId(q))]);
        }
        assert_eq!(simulator.max_bond_dim(), 1);
        let start = Instant::now();
        black_box(simulator.mz(black_box(&[QubitId(0)])));
        elapsed += start.elapsed();
        black_box(&simulator);
    }
    elapsed.as_secs_f64() * 1e6 / samples as f64
}

fn main() {
    for num_qubits in [26, 64, 118] {
        black_box(run(num_qubits, 100));
        let mut runs: Vec<_> = (0..7).map(|_| run(num_qubits, 1_000)).collect();
        runs.sort_by(f64::total_cmp);
        println!(
            "qubits={num_qubits} exact_mz_us={:.3} min_us={:.3} max_us={:.3} runs=7 samples_per_run=1000",
            runs[3], runs[0], runs[6]
        );
    }
}
