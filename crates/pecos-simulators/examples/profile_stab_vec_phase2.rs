// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the
// License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
// either express or implied. See the License for the specific language governing permissions and
// limitations under the License.

//! Reproducible first-measurement benchmark for the `StabVec` dense crossover.

use pecos_core::{Angle64, QubitId};
use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable, StabVec, StateVecSoA};
use std::hint::black_box;
use std::time::{Duration, Instant};

fn build_case(structure: &str, target_terms: usize) -> StabVec {
    let (num_qubits, rotations, pruning_threshold) = match target_terms {
        256 => (10, 8, 0.0),
        1024 => (10, 10, 0.0),
        4096 => (11, 12, 0.0),
        16_200 => (12, 14, 1e-8),
        _ => panic!("unsupported target term count {target_terms}"),
    };
    let mut sim = StabVec::builder(num_qubits)
        .seed(42)
        .pruning_threshold(pruning_threshold)
        .build();
    let all_qubits: Vec<_> = (0..num_qubits).map(QubitId).collect();
    sim.h(&all_qubits);
    black_box(sim.state_vector());

    let theta = Angle64::from_radians(std::f64::consts::FRAC_PI_4);
    for step in 0..rotations {
        let q = if structure == "shared" {
            QubitId(0)
        } else {
            QubitId(step % (num_qubits - 1))
        };
        sim.rz(theta, &[q]);
        match structure {
            "shared" => sim.flush_all_pending_rz(),
            "divergent" => {
                sim.h(&[q]);
            }
            _ => panic!("structure must be shared or divergent"),
        }
    }
    sim.flush_all_pending_rz();
    assert_eq!(sim.num_terms(), target_terms);
    assert_eq!(sim.has_shared_projection_structure(), structure == "shared");
    sim
}

fn median(samples: &mut [Duration]) -> Duration {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn main() {
    let structure = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "shared".to_owned());
    let target_terms = std::env::args()
        .nth(2)
        .and_then(|value| value.parse().ok())
        .unwrap_or(256);
    let runs = std::env::args()
        .nth(3)
        .and_then(|value| value.parse().ok())
        .unwrap_or(5);
    let template = build_case(&structure, target_terms);
    let measurement_qubit = QubitId(template.num_qubits() - 1);
    let expect_dense = structure == "divergent" && matches!(target_terms, 4096 | 16_200);

    let mut automatic_times = Vec::with_capacity(runs);
    let mut conversion_times = Vec::with_capacity(runs);
    for run in 0..runs {
        let mut automatic = template.clone();
        let start = Instant::now();
        black_box(automatic.mz(&[measurement_qubit]));
        automatic_times.push(start.elapsed());
        assert_eq!(automatic.is_dense(), expect_dense);

        let mut source = template.clone();
        let start = Instant::now();
        let state = source.state_vector();
        let mut dense =
            StateVecSoA::from_complex_state(&state, pecos_random::PecosRng::seed_from_u64(42));
        black_box(dense.mz(&[measurement_qubit]));
        conversion_times.push(start.elapsed());
        eprintln!("completed run {} of {runs}", run + 1);
    }

    eprintln!(
        "structure={structure} qubits={} terms={target_terms} runs={runs} automatic_median={:?} conversion_median={:?}",
        template.num_qubits(),
        median(&mut automatic_times),
        median(&mut conversion_times),
    );
    eprintln!("automatic_samples={automatic_times:?}");
    eprintln!("conversion_samples={conversion_times:?}");
}
