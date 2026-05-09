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

//! Integration tests for `DemStabSim`.
//!
//! Parity: `DemStabSim` must produce identical shot batches to the raw
//! `DagFaultAnalyzer` + `DemSamplerBuilder` pipeline given equal inputs and seeds.

use pecos_qec::dem_stab::{DemStabError, DemStabSim};
use pecos_qec::fault_tolerance::dem_builder::{
    DemOutput, DemSamplerBuilder, DetectorDef, NoiseConfig,
};
use pecos_qec::fault_tolerance::propagator::DagFaultAnalyzer;
use pecos_quantum::DagCircuit;
use rand::SeedableRng;
use rand::rngs::SmallRng;

fn repetition_code_circuit() -> DagCircuit {
    let mut dag = DagCircuit::new();
    // 3 data qubits (0, 1, 2), 2 ancillas (3, 4)
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

fn detectors() -> Vec<DetectorDef> {
    vec![
        DetectorDef::new(0).with_records([-2]),
        DetectorDef::new(1).with_records([-1]),
    ]
}

fn observables() -> Vec<DemOutput> {
    vec![DemOutput::new(0).with_records([-2, -1])]
}

#[test]
fn builder_rejects_missing_circuit() {
    let err = DemStabSim::builder().build().unwrap_err();
    assert!(matches!(err, DemStabError::MissingCircuit));
}

#[test]
fn zero_noise_produces_zero_mechanisms() {
    let sim = DemStabSim::builder()
        .circuit(repetition_code_circuit())
        .noise(NoiseConfig::uniform(0.0))
        .detectors(detectors())
        .observables(observables())
        .build()
        .unwrap();

    assert_eq!(sim.num_mechanisms(), 0);
    assert_eq!(sim.num_detectors(), 2);
    assert_eq!(sim.num_observables(), 1);
}

#[test]
fn parity_with_raw_pipeline() {
    let noise = NoiseConfig::uniform(0.01);
    let shots = 512;
    let seed = 0xDEAD_BEEF_u64;

    // Path 1: DemStabSim.
    let sim = DemStabSim::builder()
        .circuit(repetition_code_circuit())
        .noise(noise.clone())
        .detectors(detectors())
        .observables(observables())
        .build()
        .unwrap();
    let mut rng1 = SmallRng::seed_from_u64(seed);
    let batch = sim.sample_batch(shots, &mut rng1);

    // Path 2: raw pipeline, identical inputs + identical RNG seed.
    let dag = repetition_code_circuit();
    let analyzer = DagFaultAnalyzer::new(&dag);
    let influence_map = analyzer.build_influence_map();
    let det_records: Vec<Vec<i32>> = detectors().iter().map(|d| d.records.to_vec()).collect();
    let obs_records: Vec<Vec<i32>> = observables().iter().map(|o| o.records.to_vec()).collect();
    let sampler = DemSamplerBuilder::new(&influence_map)
        .with_noise(noise.p1, noise.p2, noise.p_meas, noise.p_prep)
        .with_detector_records(det_records)
        .with_observable_records(obs_records)
        .build()
        .unwrap();
    let mut rng2 = SmallRng::seed_from_u64(seed);
    let (det_raw, obs_raw) = sampler.sample_batch(shots, &mut rng2);

    assert_eq!(batch.detector_flips, det_raw);
    assert_eq!(batch.observable_flips, obs_raw);
}

#[test]
fn shot_batch_shape_is_correct() {
    let sim = DemStabSim::builder()
        .circuit(repetition_code_circuit())
        .noise(NoiseConfig::uniform(0.005))
        .detectors(detectors())
        .observables(observables())
        .build()
        .unwrap();

    let mut rng = SmallRng::seed_from_u64(7);
    let batch = sim.sample_batch(16, &mut rng);

    assert_eq!(batch.detector_flips.len(), 16);
    assert_eq!(batch.observable_flips.len(), 16);
    for row in &batch.detector_flips {
        assert_eq!(row.len(), sim.num_detectors());
    }
    for row in &batch.observable_flips {
        assert_eq!(row.len(), sim.num_observables());
    }
}

#[test]
fn nonzero_noise_yields_some_flips() {
    // Sanity check: at p=0.1 with 1000 shots we should see plenty of flips.
    let sim = DemStabSim::builder()
        .circuit(repetition_code_circuit())
        .noise(NoiseConfig::uniform(0.1))
        .detectors(detectors())
        .observables(observables())
        .build()
        .unwrap();

    let mut rng = SmallRng::seed_from_u64(123);
    let batch = sim.sample_batch(1000, &mut rng);

    let total_det_flips: usize = batch
        .detector_flips
        .iter()
        .map(|row| row.iter().filter(|&&b| b).count())
        .sum();

    assert!(
        total_det_flips > 0,
        "expected some detector flips at p=0.1 over 1000 shots"
    );
}
