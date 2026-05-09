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

//! Integration tests for `MemStabSim`.
//!
//! Parity: `MemStabSim` must produce identical raw-measurement shots to the raw
//! `DagFaultAnalyzer` + `MemBuilder` + `MeasurementNoiseModel` pipeline given equal
//! inputs and seeds.

use pecos_qec::fault_tolerance::dem_builder::{MemBuilder, NoiseConfig};
use pecos_qec::fault_tolerance::propagator::DagFaultAnalyzer;
use pecos_qec::mem_stab::{MemStabError, MemStabSim};
use pecos_quantum::DagCircuit;
use rand::SeedableRng;
use rand::rngs::SmallRng;

fn repetition_code_circuit() -> DagCircuit {
    let mut dag = DagCircuit::new();
    dag.pz(&[3]);
    dag.pz(&[4]);
    dag.cx(&[(0, 3)]);
    dag.cx(&[(1, 3)]);
    dag.cx(&[(1, 4)]);
    dag.cx(&[(2, 4)]);
    dag.mz(&[3]);
    dag.mz(&[4]);
    dag
}

#[test]
fn builder_rejects_missing_circuit() {
    let err = MemStabSim::builder().build().unwrap_err();
    assert!(matches!(err, MemStabError::MissingCircuit));
}

#[test]
fn zero_noise_produces_zero_mechanisms() {
    let sim = MemStabSim::builder()
        .circuit(repetition_code_circuit())
        .noise(NoiseConfig::uniform(0.0))
        .build()
        .unwrap();

    assert_eq!(sim.num_mechanisms(), 0);
    assert_eq!(sim.num_measurements(), 2);
}

#[test]
fn parity_with_raw_pipeline() {
    let noise = NoiseConfig::uniform(0.01);
    let shots = 512;
    let seed = 0xFEED_FACE_u64;

    // Path 1: MemStabSim.
    let sim = MemStabSim::builder()
        .circuit(repetition_code_circuit())
        .noise(noise.clone())
        .build()
        .unwrap();
    let mut rng1 = SmallRng::seed_from_u64(seed);
    let batch1 = sim.sample_batch(shots, &mut rng1);

    // Path 2: raw pipeline, identical inputs + seed.
    let dag = repetition_code_circuit();
    let analyzer = DagFaultAnalyzer::new(&dag);
    let influence_map = analyzer.build_influence_map();
    let mnm = MemBuilder::new(&influence_map)
        .with_noise(noise.p1, noise.p2, noise.p_meas, noise.p_prep)
        .build();
    let mut rng2 = SmallRng::seed_from_u64(seed);
    let mut batch2 = Vec::with_capacity(shots);
    let mut buf = vec![false; mnm.num_measurements];
    for _ in 0..shots {
        mnm.sample_into(&mut buf, &mut rng2);
        batch2.push(buf.clone());
    }

    assert_eq!(batch1, batch2);
}

#[test]
fn sample_and_sample_batch_agree() {
    let sim = MemStabSim::builder()
        .circuit(repetition_code_circuit())
        .noise(NoiseConfig::uniform(0.02))
        .build()
        .unwrap();

    let seed = 0xABCD_EF01_u64;
    let shots = 32;

    let mut rng_single = SmallRng::seed_from_u64(seed);
    let singles: Vec<Vec<bool>> = (0..shots).map(|_| sim.sample(&mut rng_single)).collect();

    let mut rng_batch = SmallRng::seed_from_u64(seed);
    let batch = sim.sample_batch(shots, &mut rng_batch);

    assert_eq!(singles, batch);
}

#[test]
fn shot_shape_is_correct() {
    let sim = MemStabSim::builder()
        .circuit(repetition_code_circuit())
        .noise(NoiseConfig::uniform(0.005))
        .build()
        .unwrap();

    let mut rng = SmallRng::seed_from_u64(11);
    let batch = sim.sample_batch(20, &mut rng);
    assert_eq!(batch.len(), 20);
    for row in &batch {
        assert_eq!(row.len(), sim.num_measurements());
    }
}

#[test]
fn nonzero_noise_yields_some_flips() {
    let sim = MemStabSim::builder()
        .circuit(repetition_code_circuit())
        .noise(NoiseConfig::uniform(0.1))
        .build()
        .unwrap();

    let mut rng = SmallRng::seed_from_u64(123);
    let batch = sim.sample_batch(1000, &mut rng);

    let total_flips: usize = batch
        .iter()
        .map(|row| row.iter().filter(|&&b| b).count())
        .sum();
    assert!(total_flips > 0);
}
