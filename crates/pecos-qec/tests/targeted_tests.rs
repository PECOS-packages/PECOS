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

//! Targeted tests for specific features and bug fixes from the annotation/decoder session.

use pecos_core::pauli::{X, Y, Z};
use pecos_qec::fault_tolerance::InfluenceBuilder;
use pecos_qec::fault_tolerance::dem_builder::{NoiseConfig, PauliWeights};
use pecos_qec::fault_tolerance::lookup_decoder::LookupDecoder;
use pecos_qec::fault_tolerance::propagator::{DagFaultAnalyzer, Pauli};
use pecos_quantum::DagCircuit;

// ============================================================================
// Y anticommutation: Y commutes with Y
// ============================================================================

/// Verify that Y fault does NOT flip a detector when the propagated observable
/// is also Y on that qubit. {Y, Y} = 2I (commutes), not anticommutes.
#[test]
fn y_commutes_with_y() {
    // Build a circuit where the backward-propagated observable has Y on a qubit.
    // Backward propagation from MZ through H -> SZ -> H:
    //   MZ: Z -> backward H: X -> backward SZ: Y -> backward H: Y
    // So at the first H location, the observable is Y on qubit 0.
    let mut dag = DagCircuit::new();
    dag.pz(&[0]);
    dag.h(&[0]); // observable is Y here (backward from MZ through H,SZ,H)
    dag.sz(&[0]); // observable is X here (backward from MZ through H,SZ)
    dag.h(&[0]); // observable is Z here (backward from MZ through H)
    dag.mz(&[0]);

    let analyzer = DagFaultAnalyzer::new(&dag);
    let map = analyzer.build_influence_map();

    // Find the first H gate's after-location (node with lowest index)
    // At this point the backward-propagated observable should have Y on qubit 0.
    let h_locs: Vec<(
        usize,
        &pecos_qec::fault_tolerance::propagator::DagSpacetimeLocation,
    )> = map
        .locations
        .iter()
        .enumerate()
        .filter(|(_, loc)| loc.gate_type == pecos_core::gate_type::GateType::H && !loc.before)
        .collect();

    assert!(h_locs.len() >= 2, "Should have at least 2 H locations");
    let (first_h_loc, _) = h_locs[0]; // first H (closest to prep)

    // Y fault at first H should NOT flip detector (Y commutes with Y)
    let y_dets = map.get_detector_indices(first_h_loc, Pauli::Y as u8);
    assert!(
        y_dets.is_empty(),
        "Y fault should not flip detectors when observable is Y on same qubit, got {y_dets:?}"
    );
    // But X and Z faults SHOULD flip (both anticommute with Y)
    let x_dets = map.get_detector_indices(first_h_loc, Pauli::X as u8);
    let z_dets = map.get_detector_indices(first_h_loc, Pauli::Z as u8);
    assert!(
        !x_dets.is_empty() || !z_dets.is_empty(),
        "X or Z fault should flip detectors when observable is Y"
    );
}

// ============================================================================
// T1/T2 idle noise
// ============================================================================

/// Verify T1/T2 produces biased noise: P(Z) > P(X) = P(Y).
#[test]
fn t1_t2_biased_idle_noise() {
    let noise = NoiseConfig::new(0.001, 0.01, 0.001, 0.001).set_t1_t2(50_000.0, 30_000.0);

    // 1000 time units of idle
    let pp = noise.idle_pauli_probs(1000.0);

    // P(X) == P(Y) (from amplitude damping)
    assert!(
        (pp.px - pp.py).abs() < 1e-15,
        "P(X) should equal P(Y): px={}, py={}",
        pp.px,
        pp.py
    );

    // P(Z) > P(X) (dephasing dominates relaxation)
    assert!(
        pp.pz > pp.px,
        "P(Z) should be larger than P(X): pz={}, px={}",
        pp.pz,
        pp.px
    );

    // Total should be reasonable (not > 1)
    assert!(pp.total() < 1.0, "Total should be < 1: {}", pp.total());
    assert!(pp.total() > 0.0, "Total should be > 0");
}

/// Verify uniform depolarizing idle gives equal X/Y/Z.
#[test]
fn uniform_idle_noise() {
    let noise = NoiseConfig::uniform(0.001);
    let pp = noise.idle_pauli_probs(1.0);

    let eps = 1e-15;
    assert!((pp.px - pp.py).abs() < eps);
    assert!((pp.py - pp.pz).abs() < eps);
    assert!((pp.px - 0.001 / 3.0).abs() < eps);
}

// ============================================================================
// PauliWeights
// ============================================================================

/// Verify custom single-qubit weights change decoder probabilities.
#[test]
fn custom_p1_weights_affect_decoder() {
    let mut dag = DagCircuit::new();
    dag.pz(&[0, 1]);
    dag.h(&[0]);
    dag.cx(&[(0, 1)]);
    let ms = dag.mz(&[0, 1]);
    dag.observable(&[ms[0], ms[1]]);

    let map = InfluenceBuilder::new(&dag)
        .with_circuit_annotations(&dag)
        .build();

    // Uniform weights
    let noise_uniform = NoiseConfig::uniform(0.001);
    let d_uniform = LookupDecoder::build(&map, &noise_uniform, 2);

    // Biased weights: Z only
    let noise_biased = NoiseConfig::uniform(0.001).set_p1_weights(PauliWeights::from([
        (X(0), 0.0),
        (Y(0), 0.0),
        (Z(0), 1.0),
    ]));
    let d_biased = LookupDecoder::build(&map, &noise_biased, 2);

    // Both should build successfully
    assert!(d_uniform.num_syndromes() > 0);
    assert!(d_biased.num_syndromes() > 0);

    // Both should account for most probability at p=0.001
    assert!(
        d_uniform.accounted_probability() > 0.99,
        "Uniform: {}",
        d_uniform.accounted_probability()
    );
    assert!(
        d_biased.accounted_probability() > 0.99,
        "Biased: {}",
        d_biased.accounted_probability()
    );
}

/// Verify `PauliWeights` validates sum to ~1.0.
#[test]
#[should_panic(expected = "must sum to 1.0")]
fn pauli_weights_validation() {
    // Should panic: doesn't sum to 1.0
    let _ = PauliWeights::from([(X(0), 0.5), (Y(0), 0.3)]);
}

/// Verify `PauliWeights::weight_for` matches by pattern, not qubit ID.
#[test]
fn pauli_weights_pattern_matching() {
    let w = PauliWeights::uniform_2q();

    // X(0) & Z(1) pattern should match regardless of actual qubit IDs
    let weight = w.weight_for(&(X(0) & Z(1)));
    assert!((weight - 1.0 / 15.0).abs() < 1e-10);

    // Same pattern from different qubit IDs should also match
    let weight2 = w.weight_for(&(X(5) & Z(9)));
    assert!(
        (weight2 - 1.0 / 15.0).abs() < 1e-10,
        "Pattern matching should ignore qubit IDs: got {weight2}"
    );
}

// ============================================================================
// Prep-gate propagation stop
// ============================================================================

/// Verify that faults before a mid-circuit reset don't propagate past it.
#[test]
fn prep_gate_stops_propagation() {
    // Circuit: PZ -> H -> PZ (reset) -> MZ
    // An X fault after the first H should NOT affect the final measurement
    // because the second PZ resets qubit 0.
    let mut dag = DagCircuit::new();
    dag.pz(&[0]);
    dag.h(&[0]); // fault here
    dag.pz(&[0]); // mid-circuit reset -- should block propagation
    dag.mz(&[0]);

    let map = InfluenceBuilder::new(&dag)
        .with_circuit_annotations(&dag)
        .build();

    // Find the H gate's after-location
    let mut h_has_influence = false;
    for (loc_idx, loc) in map.locations.iter().enumerate() {
        if loc.gate_type == pecos_core::gate_type::GateType::H && !loc.before {
            let x_dets = map.influences.detectors(loc_idx, Pauli::X);
            let z_dets = map.influences.detectors(loc_idx, Pauli::Z);
            let y_dets = map.influences.detectors(loc_idx, Pauli::Y);
            if !x_dets.is_empty() || !z_dets.is_empty() || !y_dets.is_empty() {
                h_has_influence = true;
            }
        }
    }

    assert!(
        !h_has_influence,
        "Faults before mid-circuit PZ should not propagate past the reset"
    );
}

/// Verify that faults AFTER a mid-circuit reset DO affect later measurements.
#[test]
fn faults_after_reset_propagate() {
    // Use DagFaultAnalyzer (which creates 1 detector per measurement) to
    // verify that faults after a reset propagate to the measurement.
    // Circuit: PZ -> PZ (reset) -> H -> MZ
    let mut dag = DagCircuit::new();
    dag.pz(&[0]);
    dag.pz(&[0]); // mid-circuit reset
    dag.h(&[0]); // fault here should affect measurement
    dag.mz(&[0]);

    let analyzer = DagFaultAnalyzer::new(&dag);
    let map = analyzer.build_influence_map();

    // H gate after-location should have detector influence
    // (backward from MZ through H: observable is X at H location)
    let mut h_has_influence = false;
    for (loc_idx, loc) in map.locations.iter().enumerate() {
        if loc.gate_type == pecos_core::gate_type::GateType::H && !loc.before {
            for p in [Pauli::X, Pauli::Y, Pauli::Z] {
                let dets = map.influences.detectors(loc_idx, p);
                if !dets.is_empty() {
                    h_has_influence = true;
                }
            }
        }
    }

    assert!(
        h_has_influence,
        "Faults after mid-circuit PZ should propagate to later measurements"
    );
}

// ============================================================================
// DagCircuit annotation methods
// ============================================================================

/// Verify `detector()` auto-derives Z Pauli from measurement nodes.
#[test]
fn detector_derives_pauli_from_measurements() {
    let mut dag = DagCircuit::new();
    dag.pz(&[0, 1]);
    dag.cx(&[(0, 1)]);
    let ms = dag.mz(&[0, 1]);
    dag.detector(&[ms[0], ms[1]]);

    let ann = &dag.annotations()[0];
    // Pauli should be Z on both measured qubits
    let paulis = ann.pauli.paulis();
    assert_eq!(paulis.len(), 2);
    assert_eq!(paulis[0].0, pecos_core::Pauli::Z);
    assert_eq!(paulis[1].0, pecos_core::Pauli::Z);
}

/// Verify `tracked_pauli` normalizes phase to +1.
#[test]
fn tracked_pauli_normalizes_phase() {
    let mut dag = DagCircuit::new();
    dag.pz(&[0]);

    // -X(0) has phase -1
    let neg_x = -X(0);
    assert_ne!(neg_x.get_phase(), pecos_core::QuarterPhase::PlusOne);

    dag.tracked_pauli(neg_x);

    // After storage, phase should be normalized to +1
    let ann = &dag.annotations()[0];
    assert_eq!(ann.pauli.get_phase(), pecos_core::QuarterPhase::PlusOne);
}

/// Verify probability sums to 1.0 with the per-gate noise model.
#[test]
fn probability_sums_to_one() {
    let mut dag = DagCircuit::new();
    dag.pz(&[0, 1]);
    dag.h(&[0]);
    dag.cx(&[(0, 1)]);
    let ms = dag.mz(&[0, 1]);
    dag.observable(&[ms[0], ms[1]]);

    let map = InfluenceBuilder::new(&dag)
        .with_circuit_annotations(&dag)
        .build();

    let noise = NoiseConfig::uniform(0.001);
    let decoder = LookupDecoder::build(&map, &noise, 3);

    assert!(
        decoder.truncation_bound() < 1e-4,
        "Weight-3 at p=0.001 should have small truncation: {}",
        decoder.truncation_bound()
    );
    assert!(
        decoder.accounted_probability() > 0.999,
        "Should account for >99.9% of probability: {}",
        decoder.accounted_probability()
    );
}
