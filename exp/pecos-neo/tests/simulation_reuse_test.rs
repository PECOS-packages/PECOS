// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! Seeded outcomes, named-register records and weights survive reuse and worker partitioning.

use pecos_neo::prelude::*;
use pecos_neo::tool::{
    SimNeoBuilder, SimulationResults, importance_sampling, monte_carlo, sim_neo, sparse_stab,
};
use pecos_qasm::qasm_engine;

fn builder(classical: bool) -> SimNeoBuilder {
    let builder = if classical {
        let qasm = "OPENQASM 2.0; include \"qelib1.inc\"; qreg q[3]; creg c[3]; \
                    reset q; h q[0]; cx q[0],q[1]; cx q[1],q[2]; measure q -> c;";
        sim_neo(qasm).classical(qasm_engine())
    } else {
        let circuit = CommandBuilder::new()
            .pz(&[0, 1, 2])
            .h(&[0])
            .cx(&[(0, 1)])
            .cx(&[(1, 2)])
            .mz(&[0, 1, 2])
            .build();
        sim_neo(circuit)
    };
    builder
        .quantum(sparse_stab())
        .depolarizing(0.07)
        .seed(12345)
}

fn assert_same_prefix(actual: &SimulationResults, reference: &SimulationResults, shots: usize) {
    assert_eq!(actual.len(), shots);
    for (actual, expected) in actual.outcomes.iter().zip(&reference.outcomes) {
        assert_eq!(actual.as_slice(), expected.as_slice());
    }
    match (&actual.shots, &reference.shots) {
        (Some(actual), Some(expected)) => {
            assert_eq!(actual.shots.len(), shots);
            assert_eq!(actual.shots, expected.shots[..shots]);
        }
        (None, None) => {}
        _ => panic!("named-register record availability must match"),
    }
    assert!(actual.weights.is_none());
    assert!(reference.weights.is_none());
    assert!(actual.subset.is_none());
}

#[test]
fn noisy_seeded_outcomes_and_records_match_across_workers_and_reuse() {
    for classical in [false, true] {
        let reference = builder(classical)
            .sampling(monte_carlo(12))
            .run()
            .expect("reference simulation should succeed");
        assert_eq!(reference.shots.is_some(), classical);
        for workers in [1, 2, 4, 16] {
            let mut sim = builder(classical)
                .sampling(monte_carlo(12).workers(workers))
                .build();
            for shots in [12, 5, 12, 0] {
                let result = sim.shots(shots).run().expect("simulation should succeed");
                if shots == 0 {
                    assert!(result.is_empty());
                    assert!(result.shots.is_none());
                } else {
                    assert_same_prefix(&result, &reference, shots);
                }
            }
        }
    }
}

#[test]
fn seeded_importance_outcomes_and_weights_match_exactly_across_workers_and_reuse() {
    let build = |workers| {
        builder(false)
            .sampling(
                importance_sampling(12)
                    .with_uniform_error(0.01)
                    .with_boost(10.0)
                    .workers(workers),
            )
            .build()
    };
    let reference = build(1)
        .run()
        .expect("reference importance simulation should succeed");
    assert_eq!(reference.len(), 12);
    let reference_weights: Vec<_> = reference
        .weights
        .as_ref()
        .expect("importance sampling should produce weights")
        .iter()
        .map(|weight| weight.log_weight().to_bits())
        .collect();
    assert_eq!(reference_weights.len(), 12);

    for workers in [1, 2, 4, 16] {
        let mut sim = build(workers);
        for shots in [12, 5, 12, 0] {
            let result = sim
                .shots(shots)
                .run()
                .expect("importance simulation should succeed");
            assert_eq!(result.len(), shots);
            for (actual, expected) in result.outcomes.iter().zip(&reference.outcomes[..shots]) {
                assert_eq!(actual.as_slice(), expected.as_slice());
            }
            // Zero-shot runs must retain Some(empty), not discard weight tracking.
            let weights: Vec<_> = result
                .weights
                .as_ref()
                .expect("importance sampling should produce weights even for zero shots")
                .iter()
                .map(|weight| weight.log_weight().to_bits())
                .collect();
            assert_eq!(weights, reference_weights[..shots]);
            assert!(result.shots.is_none());
            assert!(result.subset.is_none());
        }
    }
}
