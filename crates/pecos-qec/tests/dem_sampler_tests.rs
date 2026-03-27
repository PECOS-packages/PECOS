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

//! Integration tests for `DemSampler`.
//!
//! These tests verify:
//! 1. Correct mechanism aggregation
//! 2. Statistical properties of sampling
//! 3. Consistency with MNM (Measurement Noise Model)
//! 4. Edge cases and boundary conditions

use pecos_qec::fault_tolerance::dem_builder::{DemSamplerBuilder, MemBuilder};
use pecos_qec::fault_tolerance::propagator::DagFaultAnalyzer;
use pecos_quantum::DagCircuit;
use rand::SeedableRng;
use rand::rngs::SmallRng;

// ============================================================================
// Test Helpers
// ============================================================================

/// Build a simple syndrome extraction circuit (2 data qubits, 1 ancilla).
fn build_parity_check_circuit() -> DagCircuit {
    let mut dag = DagCircuit::new();
    dag.pz(&[2]); // Ancilla
    dag.cx(&[(0, 2)]);
    dag.cx(&[(1, 2)]);
    dag.mz(&[2]);
    dag
}

/// Build a repetition code-like circuit (3 data qubits, 2 ancillas).
fn build_repetition_code_circuit() -> DagCircuit {
    let mut dag = DagCircuit::new();
    // Prepare ancillas
    dag.pz(&[3]);
    dag.pz(&[4]);
    // Check Z0*Z1
    dag.cx(&[(0, 3)]);
    dag.cx(&[(1, 3)]);
    // Check Z1*Z2
    dag.cx(&[(1, 4)]);
    dag.cx(&[(2, 4)]);
    // Measure
    dag.mz(&[3]);
    dag.mz(&[4]);
    dag
}

// ============================================================================
// Basic Functionality Tests
// ============================================================================

#[test]
fn test_zero_noise_produces_no_errors() {
    let dag = build_parity_check_circuit();
    let analyzer = DagFaultAnalyzer::new(&dag);
    let influence_map = analyzer.build_influence_map();

    let detectors_json = r#"[{"id": 0, "records": [-1]}]"#;

    let sampler = DemSamplerBuilder::new(&influence_map)
        .with_noise(0.0, 0.0, 0.0, 0.0)
        .with_detectors_json(detectors_json)
        .unwrap()
        .build();

    // Zero noise should produce zero mechanisms
    assert_eq!(sampler.num_mechanisms(), 0);

    let mut rng = SmallRng::seed_from_u64(42);
    let stats = sampler.sample_statistics_with_rng(1000, &mut rng);

    assert_eq!(stats.logical_error_count, 0);
    assert_eq!(stats.syndrome_count, 0);
}

#[test]
fn test_mechanism_count_scales_with_circuit() {
    // Larger circuit should have more mechanisms
    let dag1 = build_parity_check_circuit();
    let dag2 = build_repetition_code_circuit();

    let analyzer1 = DagFaultAnalyzer::new(&dag1);
    let analyzer2 = DagFaultAnalyzer::new(&dag2);

    let im1 = analyzer1.build_influence_map();
    let im2 = analyzer2.build_influence_map();

    let sampler1 = DemSamplerBuilder::new(&im1)
        .with_noise(0.01, 0.01, 0.01, 0.01)
        .with_detectors_json(r#"[{"id": 0, "records": [-1]}]"#)
        .unwrap()
        .build();

    let sampler2 = DemSamplerBuilder::new(&im2)
        .with_noise(0.01, 0.01, 0.01, 0.01)
        .with_detectors_json(r#"[{"id": 0, "records": [-2]}, {"id": 1, "records": [-1]}]"#)
        .unwrap()
        .build();

    // Larger circuit should have more mechanisms
    assert!(
        sampler2.num_mechanisms() >= sampler1.num_mechanisms(),
        "Larger circuit should have at least as many mechanisms"
    );
}

#[test]
fn test_deterministic_sampling_with_seed() {
    let dag = build_parity_check_circuit();
    let analyzer = DagFaultAnalyzer::new(&dag);
    let influence_map = analyzer.build_influence_map();

    let detectors_json = r#"[{"id": 0, "records": [-1]}]"#;

    let sampler = DemSamplerBuilder::new(&influence_map)
        .with_noise(0.1, 0.1, 0.1, 0.1)
        .with_detectors_json(detectors_json)
        .unwrap()
        .build();

    // Same seed should produce same results
    let mut rng1 = SmallRng::seed_from_u64(12345);
    let mut rng2 = SmallRng::seed_from_u64(12345);

    let (det1, obs1) = sampler.sample_batch(100, &mut rng1);
    let (det2, obs2) = sampler.sample_batch(100, &mut rng2);

    assert_eq!(det1, det2);
    assert_eq!(obs1, obs2);
}

#[test]
fn test_different_seeds_produce_different_results() {
    let dag = build_parity_check_circuit();
    let analyzer = DagFaultAnalyzer::new(&dag);
    let influence_map = analyzer.build_influence_map();

    let detectors_json = r#"[{"id": 0, "records": [-1]}]"#;

    let sampler = DemSamplerBuilder::new(&influence_map)
        .with_noise(0.1, 0.1, 0.1, 0.1)
        .with_detectors_json(detectors_json)
        .unwrap()
        .build();

    let mut rng1 = SmallRng::seed_from_u64(12345);
    let mut rng2 = SmallRng::seed_from_u64(54321);

    let (det1, _) = sampler.sample_batch(100, &mut rng1);
    let (det2, _) = sampler.sample_batch(100, &mut rng2);

    // Different seeds should produce different results (with high probability)
    assert_ne!(det1, det2);
}

// ============================================================================
// Statistical Validation Tests
// ============================================================================

#[test]
fn test_syndrome_rate_scales_with_noise() {
    let dag = build_parity_check_circuit();
    let analyzer = DagFaultAnalyzer::new(&dag);
    let influence_map = analyzer.build_influence_map();

    let detectors_json = r#"[{"id": 0, "records": [-1]}]"#;

    let sampler_low = DemSamplerBuilder::new(&influence_map)
        .with_noise(0.001, 0.001, 0.001, 0.001)
        .with_detectors_json(detectors_json)
        .unwrap()
        .build();

    let sampler_high = DemSamplerBuilder::new(&influence_map)
        .with_noise(0.05, 0.05, 0.05, 0.05)
        .with_detectors_json(detectors_json)
        .unwrap()
        .build();

    let mut rng = SmallRng::seed_from_u64(42);

    let stats_low = sampler_low.sample_statistics_with_rng(10000, &mut rng);
    let stats_high = sampler_high.sample_statistics_with_rng(10000, &mut rng);

    // Higher noise should produce higher syndrome rate
    assert!(
        stats_high.syndrome_rate() > stats_low.syndrome_rate(),
        "Higher noise ({:.3}) should produce higher syndrome rate than lower noise ({:.3})",
        stats_high.syndrome_rate(),
        stats_low.syndrome_rate()
    );
}

#[test]
fn test_syndrome_rate_reasonable_magnitude() {
    let dag = build_parity_check_circuit();
    let analyzer = DagFaultAnalyzer::new(&dag);
    let influence_map = analyzer.build_influence_map();

    let detectors_json = r#"[{"id": 0, "records": [-1]}]"#;
    let p = 0.01;

    let sampler = DemSamplerBuilder::new(&influence_map)
        .with_noise(p, p, p, p)
        .with_detectors_json(detectors_json)
        .unwrap()
        .build();

    let mut rng = SmallRng::seed_from_u64(42);
    let stats = sampler.sample_statistics_with_rng(100_000, &mut rng);

    // With p=0.01, syndrome rate should be in a reasonable range
    // For this circuit, multiple error sources can trigger the detector
    // so syndrome rate should be roughly O(p) to O(10p)
    assert!(
        stats.syndrome_rate() > 0.001,
        "Syndrome rate too low: {}",
        stats.syndrome_rate()
    );
    assert!(
        stats.syndrome_rate() < 0.5,
        "Syndrome rate too high: {}",
        stats.syndrome_rate()
    );
}

#[test]
fn test_observable_tracking() {
    let dag = build_parity_check_circuit();
    let analyzer = DagFaultAnalyzer::new(&dag);
    let influence_map = analyzer.build_influence_map();

    let detectors_json = r#"[{"id": 0, "records": [-1]}]"#;
    // Observable tracking the measurement (will flip when measurement errors occur)
    let observables_json = r#"[{"id": 0, "records": [-1]}]"#;

    let sampler = DemSamplerBuilder::new(&influence_map)
        .with_noise(0.01, 0.01, 0.01, 0.01)
        .with_detectors_json(detectors_json)
        .unwrap()
        .with_observables_json(observables_json)
        .unwrap()
        .build();

    assert_eq!(sampler.num_observables(), 1);

    let mut rng = SmallRng::seed_from_u64(42);
    let stats = sampler.sample_statistics_with_rng(10000, &mut rng);

    // With observable tracking, we should see some logical errors
    assert!(
        stats.logical_error_rate() > 0.0,
        "Expected some logical errors with noise"
    );
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_empty_detector_definitions() {
    let dag = build_parity_check_circuit();
    let analyzer = DagFaultAnalyzer::new(&dag);
    let influence_map = analyzer.build_influence_map();

    let sampler = DemSamplerBuilder::new(&influence_map)
        .with_noise(0.01, 0.01, 0.01, 0.01)
        .with_detectors_json("[]")
        .unwrap()
        .build();

    assert_eq!(sampler.num_detectors(), 0);

    let mut rng = SmallRng::seed_from_u64(42);
    let (det_events, _) = sampler.sample(&mut rng);

    assert!(det_events.is_empty());
}

#[test]
fn test_single_qubit_circuit() {
    let mut dag = DagCircuit::new();
    dag.pz(&[0]);
    dag.h(&[0]);
    dag.mz(&[0]);

    let analyzer = DagFaultAnalyzer::new(&dag);
    let influence_map = analyzer.build_influence_map();

    let sampler = DemSamplerBuilder::new(&influence_map)
        .with_noise(0.01, 0.01, 0.01, 0.01)
        .with_detectors_json(r#"[{"id": 0, "records": [-1]}]"#)
        .unwrap()
        .build();

    // Should have mechanisms from prep, H gate, and measurement
    assert!(sampler.num_mechanisms() > 0);

    let mut rng = SmallRng::seed_from_u64(42);
    let stats = sampler.sample_statistics_with_rng(1000, &mut rng);

    // Should produce some syndromes
    assert!(stats.syndrome_count > 0);
}

#[test]
fn test_only_measurement_noise() {
    let dag = build_parity_check_circuit();
    let analyzer = DagFaultAnalyzer::new(&dag);
    let influence_map = analyzer.build_influence_map();

    let sampler = DemSamplerBuilder::new(&influence_map)
        .with_noise(0.0, 0.0, 0.1, 0.0) // Only measurement noise
        .with_detectors_json(r#"[{"id": 0, "records": [-1]}]"#)
        .unwrap()
        .build();

    assert!(
        sampler.num_mechanisms() > 0,
        "Should have mechanisms from measurement noise"
    );

    let mut rng = SmallRng::seed_from_u64(42);
    let stats = sampler.sample_statistics_with_rng(10000, &mut rng);

    // Should produce syndromes from measurement errors
    assert!(
        stats.syndrome_rate() > 0.01,
        "Expected syndromes from measurement noise"
    );
}

#[test]
fn test_only_two_qubit_noise() {
    let dag = build_parity_check_circuit();
    let analyzer = DagFaultAnalyzer::new(&dag);
    let influence_map = analyzer.build_influence_map();

    let sampler = DemSamplerBuilder::new(&influence_map)
        .with_noise(0.0, 0.1, 0.0, 0.0) // Only two-qubit noise
        .with_detectors_json(r#"[{"id": 0, "records": [-1]}]"#)
        .unwrap()
        .build();

    assert!(
        sampler.num_mechanisms() > 0,
        "Should have mechanisms from two-qubit noise"
    );

    let mut rng = SmallRng::seed_from_u64(42);
    let stats = sampler.sample_statistics_with_rng(10000, &mut rng);

    // Should produce syndromes from CX errors
    assert!(
        stats.syndrome_rate() > 0.01,
        "Expected syndromes from two-qubit noise"
    );
}

// ============================================================================
// Consistency Tests (DemSampler vs MNM)
// ============================================================================

#[test]
fn test_dem_sampler_vs_mnm_mechanism_structure() {
    let dag = build_parity_check_circuit();
    let analyzer = DagFaultAnalyzer::new(&dag);
    let influence_map = analyzer.build_influence_map();

    let p1 = 0.01;
    let p2 = 0.01;
    let p_meas = 0.01;
    let p_init = 0.01;

    // Build MNM for comparison
    let mnm = MemBuilder::new(&influence_map)
        .with_noise(p1, p2, p_meas, p_init)
        .build();

    // Build DemSampler
    let sampler = DemSamplerBuilder::new(&influence_map)
        .with_noise(p1, p2, p_meas, p_init)
        .with_detectors_json(r#"[{"id": 0, "records": [-1]}]"#)
        .unwrap()
        .build();

    // MNM mechanisms are at measurement level, DemSampler at detector level
    // They should both have non-zero counts
    assert!(mnm.num_mechanisms() > 0);
    assert!(sampler.num_mechanisms() > 0);
}

#[test]
fn test_multi_detector_circuit() {
    let dag = build_repetition_code_circuit();
    let analyzer = DagFaultAnalyzer::new(&dag);
    let influence_map = analyzer.build_influence_map();

    let detectors_json = r#"[
        {"id": 0, "records": [-2]},
        {"id": 1, "records": [-1]}
    ]"#;

    let sampler = DemSamplerBuilder::new(&influence_map)
        .with_noise(0.01, 0.01, 0.01, 0.01)
        .with_detectors_json(detectors_json)
        .unwrap()
        .build();

    assert_eq!(sampler.num_detectors(), 2);

    let mut rng = SmallRng::seed_from_u64(42);
    let (det_events, _) = sampler.sample_batch(1000, &mut rng);

    // Should get events on both detectors
    let det0_fired = det_events.iter().filter(|d| d[0]).count();
    let det1_fired = det_events.iter().filter(|d| d[1]).count();

    assert!(det0_fired > 0, "Detector 0 should fire sometimes");
    assert!(det1_fired > 0, "Detector 1 should fire sometimes");
}

// ============================================================================
// Performance Sanity Tests
// ============================================================================

#[test]
fn test_batch_sampling_performance() {
    let dag = build_repetition_code_circuit();
    let analyzer = DagFaultAnalyzer::new(&dag);
    let influence_map = analyzer.build_influence_map();

    let sampler = DemSamplerBuilder::new(&influence_map)
        .with_noise(0.01, 0.01, 0.01, 0.01)
        .with_detectors_json(r#"[{"id": 0, "records": [-2]}, {"id": 1, "records": [-1]}]"#)
        .unwrap()
        .build();

    let mut rng = SmallRng::seed_from_u64(42);

    // Should be able to sample many shots quickly
    let num_shots = 100_000;
    let start = std::time::Instant::now();
    let _ = sampler.sample_statistics_with_rng(num_shots, &mut rng);
    let elapsed = start.elapsed();

    // Should complete in reasonable time (< 1 second for 100k shots)
    assert!(
        elapsed.as_secs() < 1,
        "100k shots took {elapsed:?}, should be < 1s"
    );
}

#[test]
fn test_statistics_vs_batch_consistency() {
    // Note: sample_statistics uses geometric skip sampling while sample_batch uses
    // per-shot threshold sampling. These are different algorithms that produce
    // statistically equivalent but not bit-identical results, even with the same seed.
    let dag = build_parity_check_circuit();
    let analyzer = DagFaultAnalyzer::new(&dag);
    let influence_map = analyzer.build_influence_map();

    let detectors_json = r#"[{"id": 0, "records": [-1]}]"#;
    let observables_json = r#"[{"id": 0, "records": [-1]}]"#;

    let sampler = DemSamplerBuilder::new(&influence_map)
        .with_noise(0.05, 0.05, 0.05, 0.05)
        .with_detectors_json(detectors_json)
        .unwrap()
        .with_observables_json(observables_json)
        .unwrap()
        .build();

    let num_shots = 10000;

    // Sample with statistics method (uses geometric skip)
    let mut rng1 = SmallRng::seed_from_u64(42);
    let stats = sampler.sample_statistics_with_rng(num_shots, &mut rng1);

    // Sample with batch method (uses per-shot threshold)
    let mut rng2 = SmallRng::seed_from_u64(123); // Different seed since algorithms differ
    let (det_events, obs_flips) = sampler.sample_batch(num_shots, &mut rng2);

    // Count from batch results
    let batch_syndromes = det_events.iter().filter(|d| d.iter().any(|&x| x)).count();
    let batch_logical = obs_flips.iter().filter(|o| o.iter().any(|&x| x)).count();

    // Should be statistically similar (within 10% relative difference)
    let stats_rate = stats.syndrome_count as f64 / num_shots as f64;
    let batch_rate = batch_syndromes as f64 / num_shots as f64;
    let rel_diff = (stats_rate - batch_rate).abs() / stats_rate.max(batch_rate).max(0.001);
    assert!(
        rel_diff < 0.1,
        "Syndrome rates should be similar: stats={stats_rate:.4} batch={batch_rate:.4} rel_diff={rel_diff:.2}"
    );

    let stats_logical_rate = stats.logical_error_count as f64 / num_shots as f64;
    let batch_logical_rate = batch_logical as f64 / num_shots as f64;
    let logical_rel_diff = (stats_logical_rate - batch_logical_rate).abs()
        / stats_logical_rate.max(batch_logical_rate).max(0.001);
    assert!(
        logical_rel_diff < 0.1,
        "Logical error rates should be similar: stats={stats_logical_rate:.4} batch={batch_logical_rate:.4} rel_diff={logical_rel_diff:.2}"
    );
}
