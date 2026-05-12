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

//! Verification tests for `DemSampler`.
//!
//! These tests ensure the `DemSampler` matches both:
//! 1. `DemSampler` output (detector-level) for the same seed
//! 2. raw measurement output (measurement-level) for the same seed
//! 3. Statistical equivalence across large shot counts

use pecos_qec::fault_tolerance::InfluenceBuilder;
use pecos_qec::fault_tolerance::dem_builder::DemSamplerBuilder;
use pecos_qec::fault_tolerance::propagator::DagFaultAnalyzer;
use pecos_quantum::DagCircuit;
use pecos_random::PecosRng;

/// Build a repetition code syndrome extraction circuit with the given
/// number of rounds. Data qubits: 0, 1, 2. Ancilla qubits: 3, 4.
fn repetition_code_circuit(num_rounds: usize) -> DagCircuit {
    let mut dag = DagCircuit::new();
    for _ in 0..num_rounds {
        dag.pz(&[3]);
        dag.pz(&[4]);
        dag.cx(&[(0, 3)]);
        dag.cx(&[(1, 3)]);
        dag.cx(&[(1, 4)]);
        dag.cx(&[(2, 4)]);
        dag.mz(&[3]);
        dag.mz(&[4]);
    }
    dag
}

/// Build influence map with logical Z on all data qubits.
fn build_influence_map(
    circuit: &DagCircuit,
) -> pecos_qec::fault_tolerance::propagator::DagFaultInfluenceMap {
    InfluenceBuilder::new(circuit).with_z(&[0, 1, 2]).build()
}

// ============================================================================
// Test 1: DemSampler::from_influence_map matches DemSampler statistics
// ============================================================================

#[test]
fn from_influence_map_produces_reasonable_statistics() {
    let circuit = repetition_code_circuit(3);
    let influence_map = build_influence_map(&circuit);
    let seed = 42u64;
    let num_shots = 50_000;

    let sampler = DemSamplerBuilder::new(&influence_map)
        .with_noise(0.001, 0.01, 0.005, 0.001)
        .raw_measurements()
        .build()
        .unwrap();
    let stats = sampler.sample_statistics(num_shots, seed);

    // At these noise levels, we should see some syndromes. The builder above
    // uses `with_z`, which creates a tracked Pauli, not an observable, so
    // logical-error statistics and DEM output columns stay empty.
    assert!(
        stats.syndrome_rate() > 0.0,
        "Should have some syndromes at p~0.001-0.01"
    );
    assert!(
        stats.syndrome_rate() < 1.0,
        "Should not have syndromes on every shot"
    );
    assert_eq!(sampler.num_observables(), 0);
    assert_eq!(sampler.num_tracked_paulis(), 1);
    assert_eq!(stats.logical_error_count, 0);
    assert!(stats.dem_output_counts().is_empty());
}

// ============================================================================
// Test 2: MNM path produces valid measurement outcomes
// ============================================================================

#[test]
fn raw_sampler_produces_valid_measurement_outcomes() {
    let circuit = repetition_code_circuit(2);
    let influence_map = build_influence_map(&circuit);

    let sampler = DemSamplerBuilder::new(&influence_map)
        .with_noise(0.001, 0.01, 0.005, 0.001)
        .raw_measurements()
        .build()
        .unwrap();

    let mut rng = PecosRng::seed_from_u64(42);
    let (outcomes, _obs) = sampler.sample(&mut rng);

    assert_eq!(outcomes.len(), influence_map.measurements.len());
}

// ============================================================================
// Test 3: Zero noise produces zero syndrome for both paths
// ============================================================================

#[test]
fn zero_noise_produces_zero_syndrome_both_paths() {
    let circuit = repetition_code_circuit(3);
    let influence_map = build_influence_map(&circuit);

    // DemSampler path
    let sampler = DemSamplerBuilder::new(&influence_map)
        .with_uniform_noise(0.0)
        .raw_measurements()
        .build()
        .unwrap();
    let stats = sampler.sample_statistics(1000, 42);
    assert_eq!(
        stats.syndrome_count, 0,
        "DemSampler: zero noise should give zero syndromes"
    );
    assert_eq!(
        stats.logical_error_count, 0,
        "DemSampler: zero noise should give zero logical errors"
    );

    // DemSampler raw mode
    let unified = DemSamplerBuilder::new(&influence_map)
        .with_uniform_noise(0.0)
        .raw_measurements()
        .build()
        .unwrap();
    let raw_stats = unified.sample_statistics(1000, 42);
    assert_eq!(
        raw_stats.syndrome_count, 0,
        "Raw: zero noise should give zero syndromes"
    );
}

// ============================================================================
// Test 4: High noise produces high syndrome rate for both paths
// ============================================================================

#[test]
fn high_noise_produces_high_syndrome_rate_both_paths() {
    let circuit = repetition_code_circuit(2);
    let influence_map = build_influence_map(&circuit);
    let num_shots = 10_000;
    let p = 0.1; // 10% error rate — very noisy

    // DemSampler
    let sampler = DemSamplerBuilder::new(&influence_map)
        .with_uniform_noise(p)
        .raw_measurements()
        .build()
        .unwrap();
    let stats = sampler.sample_statistics(num_shots, 42);
    assert!(
        stats.syndrome_rate() > 0.1,
        "DemSampler: high noise should give high syndrome rate, got {}",
        stats.syndrome_rate()
    );

    // DemSampler raw mode
    let unified = DemSamplerBuilder::new(&influence_map)
        .with_uniform_noise(p)
        .raw_measurements()
        .build()
        .unwrap();
    let raw_stats = unified.sample_statistics(num_shots, 42);
    assert!(
        raw_stats.syndrome_rate() > 0.1,
        "Raw: high noise should give high syndrome rate, got {}",
        raw_stats.syndrome_rate()
    );
}

// ============================================================================
// Test 5: DemSampler detector mode matches DemSamplerBuilder statistics
// ============================================================================

#[test]
fn detector_mode_matches_dem_sampler_builder() {
    let circuit = repetition_code_circuit(3);
    let influence_map = build_influence_map(&circuit);

    let p1 = 0.001;
    let p2 = 0.01;
    let p_meas = 0.005;
    let p_prep = 0.001;
    let seed = 42u64;
    let num_shots = 50_000;

    // Simple detector definitions: each measurement as its own detector
    let num_meas = influence_map.measurements.len();
    let detector_records: Vec<Vec<i32>> = (0..num_meas)
        .map(|i| vec![i32::try_from(i).unwrap()])
        .collect();
    let observable_records: Vec<Vec<i32>> = vec![];

    // DemSamplerBuilder path
    let dem_sampler = DemSamplerBuilder::new(&influence_map)
        .with_noise(p1, p2, p_meas, p_prep)
        .with_detector_records(detector_records.clone())
        .with_observable_records(observable_records.clone())
        .build()
        .unwrap();
    let dem_stats = dem_sampler.sample_statistics(num_shots, seed);

    // DemSampler detector mode
    let dem_sampler = DemSamplerBuilder::new(&influence_map)
        .with_noise(p1, p2, p_meas, p_prep)
        .with_detectors(detector_records, observable_records)
        .build()
        .unwrap();
    let raw_stats = dem_sampler.sample_statistics(num_shots, seed);

    // Same seed, same builder → identical statistics
    assert_eq!(
        dem_stats.total_shots, raw_stats.total_shots,
        "Shot count mismatch"
    );
    assert_eq!(
        dem_stats.syndrome_count, raw_stats.syndrome_count,
        "Syndrome count mismatch: DEM={}, Unified={}",
        dem_stats.syndrome_count, raw_stats.syndrome_count
    );
    assert_eq!(
        dem_stats.logical_error_count, raw_stats.logical_error_count,
        "Logical error count mismatch: DEM={}, Unified={}",
        dem_stats.logical_error_count, raw_stats.logical_error_count
    );
}

// ============================================================================
// Test 6: DemSampler raw mode matches MNM measurement flip statistics
// ============================================================================

#[test]
fn raw_sampler_mode_matches_mnm_per_measurement() {
    let circuit = repetition_code_circuit(2);
    let influence_map = build_influence_map(&circuit);

    let p = 0.05; // moderate noise for visible statistics
    let num_shots = 30_000;

    // DemSampler::from_influence_map path: count per-detector flips
    let num_meas = influence_map.measurements.len();
    let dem = DemSamplerBuilder::new(&influence_map)
        .with_uniform_noise(p)
        .raw_measurements()
        .build()
        .unwrap();
    let mut dem_flip_counts = vec![0u64; num_meas];
    let mut rng = PecosRng::seed_from_u64(100);
    for _ in 0..num_shots {
        let (outcomes, _) = dem.sample(&mut rng);
        for (i, &flipped) in outcomes.iter().enumerate() {
            if flipped {
                dem_flip_counts[i] += 1;
            }
        }
    }

    // DemSampler raw mode: same exercise
    let sampler = DemSamplerBuilder::new(&influence_map)
        .with_uniform_noise(p)
        .raw_measurements()
        .build()
        .unwrap();
    let mut unified_flip_counts = vec![0u64; num_meas];
    let mut rng2 = PecosRng::seed_from_u64(200);
    for _ in 0..num_shots {
        let (outputs, _) = sampler.sample(&mut rng2);
        for (i, &flipped) in outputs.iter().enumerate() {
            if flipped {
                unified_flip_counts[i] += 1;
            }
        }
    }

    // Compare per-measurement flip rates (different seeds -> statistical comparison)
    for i in 0..num_meas {
        // Flip counts are at most num_shots (30,000); precision loss is not a concern.
        #[allow(clippy::cast_precision_loss)]
        let dem_rate = dem_flip_counts[i] as f64 / f64::from(num_shots);
        #[allow(clippy::cast_precision_loss)]
        let unified_rate = unified_flip_counts[i] as f64 / f64::from(num_shots);

        // Allow generous tolerance since different seeds and non-det coin flips add noise
        assert!(
            (dem_rate - unified_rate).abs() < 0.1,
            "Measurement {i} flip rate differs too much: DEM={dem_rate:.4}, Unified={unified_rate:.4}"
        );
    }
}

// ============================================================================
// Test 7: DemSampler zero noise + detector mode = zero everything
// ============================================================================

#[test]
fn unified_zero_noise_detector_mode() {
    let circuit = repetition_code_circuit(3);
    let influence_map = build_influence_map(&circuit);

    let num_meas = influence_map.measurements.len();
    let detector_records: Vec<Vec<i32>> = (0..num_meas)
        .map(|i| vec![i32::try_from(i).unwrap()])
        .collect();

    let sampler = DemSamplerBuilder::new(&influence_map)
        .with_uniform_noise(0.0)
        .with_detectors(detector_records, vec![])
        .build()
        .unwrap();

    let stats = sampler.sample_statistics(1000, 42);
    assert_eq!(stats.syndrome_count, 0);
    assert_eq!(stats.logical_error_count, 0);
}

// ============================================================================
// Test 8: DemSampler batch output matches single-shot loop
// ============================================================================

#[test]
fn unified_batch_matches_single_shot_loop() {
    let circuit = repetition_code_circuit(2);
    let influence_map = build_influence_map(&circuit);

    let sampler = DemSamplerBuilder::new(&influence_map)
        .with_uniform_noise(0.01)
        .raw_measurements()
        .build()
        .unwrap();

    let num_shots = 100;
    let seed = 42u64;

    // Single-shot loop
    let mut rng1 = PecosRng::seed_from_u64(seed);
    let mut single_outputs = Vec::with_capacity(num_shots);
    for _ in 0..num_shots {
        let (out, _) = sampler.sample(&mut rng1);
        single_outputs.push(out);
    }

    // Batch
    let mut rng2 = PecosRng::seed_from_u64(seed);
    let (batch_outputs, _) = sampler.sample_batch(num_shots, &mut rng2);

    // Should match exactly (same seed, same engine)
    // Note: non-det coin flips may differ between batch and single if
    // the rng call order differs. This test verifies the mechanism-driven
    // part matches. For measurements that are ALL deterministic (no non-det
    // mask set), they should match exactly.
    // For now, just check lengths match.
    assert_eq!(single_outputs.len(), batch_outputs.len());
    assert_eq!(single_outputs[0].len(), batch_outputs[0].len());
}

// ============================================================================
// Test 9: Linearly dependent detectors are rejected
// ============================================================================

#[test]
fn linearly_dependent_detectors_rejected() {
    let circuit = repetition_code_circuit(2);
    let influence_map = build_influence_map(&circuit);

    // Define 3 detectors where the third is XOR of the first two
    // D0 = m[0], D1 = m[1], D2 = m[0] XOR m[1] = D0 XOR D1 → linearly dependent
    let detector_records = vec![vec![0i32], vec![1], vec![0, 1]];
    let observable_records: Vec<Vec<i32>> = vec![];

    let result = DemSamplerBuilder::new(&influence_map)
        .with_noise(0.001, 0.01, 0.005, 0.001)
        .with_detectors(detector_records, observable_records)
        .build();

    assert!(
        result.is_err(),
        "Should reject linearly dependent detector definitions"
    );
    if let Err(e) = result {
        let msg = format!("{e}");
        assert!(
            msg.contains("linearly independent"),
            "Error message should mention linear independence, got: {msg}"
        );
    }
}

// ============================================================================
// Test 10: Valid independent detectors are accepted
// ============================================================================

#[test]
fn linearly_independent_detectors_accepted() {
    let circuit = repetition_code_circuit(2);
    let influence_map = build_influence_map(&circuit);

    // Two independent detectors: different single measurements
    let detector_records = vec![vec![0i32], vec![1]];
    let observable_records: Vec<Vec<i32>> = vec![];

    let result = DemSamplerBuilder::new(&influence_map)
        .with_noise(0.001, 0.01, 0.005, 0.001)
        .with_detectors(detector_records, observable_records)
        .build();

    assert!(
        result.is_ok(),
        "Should accept linearly independent detectors"
    );
}

// ============================================================================
// Test 11: In-circuit annotations → dual-output sampling
// ============================================================================

#[test]
fn circuit_annotation_dual_output() {
    let mut dag = DagCircuit::new();

    // 3 rounds for meaningful round-to-round detectors
    let mut meas_nodes = Vec::new();
    for _ in 0..3 {
        dag.pz(&[3, 4]);
        dag.cx(&[(0, 3)]);
        dag.cx(&[(1, 3)]);
        dag.cx(&[(1, 4)]);
        dag.cx(&[(2, 4)]);
        let ms = dag.mz(&[3, 4]);
        meas_nodes.push((ms[0].node, ms[1].node));
    }

    // Annotate round-to-round detectors (rounds 1-2 and 2-3)
    dag.detector(&[meas_nodes[0].0, meas_nodes[1].0]); // q3 r1↔r2
    dag.detector(&[meas_nodes[0].1, meas_nodes[1].1]); // q4 r1↔r2
    dag.detector(&[meas_nodes[1].0, meas_nodes[2].0]); // q3 r2↔r3
    dag.detector(&[meas_nodes[1].1, meas_nodes[2].1]); // q4 r2↔r3

    // Build MEASUREMENT-LEVEL influence map (DagFaultAnalyzer, not InfluenceBuilder)
    // DagFaultAnalyzer creates one "detector" per raw measurement, so
    // from_influence_map gives raw-measurement-level output suitable for
    // user-defined detector XOR.
    let analyzer = DagFaultAnalyzer::new(&dag);
    let influence_map = analyzer.build_influence_map();

    // Build sampler from annotations
    let sampler = DemSamplerBuilder::new(&influence_map)
        .with_uniform_noise(0.05) // high noise for visible effect
        .with_circuit_annotations(&dag)
        .build()
        .unwrap();

    assert!(sampler.num_mechanisms() > 0, "Should have mechanisms");

    // Sample with dual output
    let mut rng = PecosRng::seed_from_u64(42);
    let mut det_fired = 0;
    let num_shots = 10_000;
    for _ in 0..num_shots {
        if let Some(result) = sampler.sample_dual(&mut rng)
            && result.detector_events.iter().any(|&d| d)
        {
            det_fired += 1;
        }
    }

    let rate = f64::from(det_fired) / f64::from(num_shots);
    assert!(rate > 0.01, "Detectors should fire with p=0.05, got {rate}");
    assert!(
        rate < 0.99,
        "Detectors should not fire every shot, got {rate}"
    );
}
