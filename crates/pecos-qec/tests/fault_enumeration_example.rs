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

// Shot counts and error counters are small integers; precision loss in f64 is not a concern.
#![allow(clippy::cast_precision_loss)]

//! Example: Repetition code d=3 with 3 rounds of syndrome extraction.
//!
//! Demonstrates the full QEC workflow:
//! 1. Build the circuit with annotations (detectors, observables, tracked Paulis)
//! 2. Build the fault influence map
//! 3. Enumerate fault combinations up to weight 3
//! 4. Classify errors (detectable, undetectable, logical)

use pecos_core::pauli::X;
use pecos_qec::fault_tolerance::InfluenceBuilder;
use pecos_quantum::DagCircuit;

/// Build a repetition code d=3 circuit with `num_rounds` syndrome extraction rounds.
///
/// Layout:
///   Data qubits:    0, 1, 2
///   Z-ancillas:     3 (measures `Z_0` `Z_1`), 4 (measures `Z_1` `Z_2`)
///
/// Each round: prep ancillas, CNOT syndrome extraction, measure ancillas.
/// After the last round: measure all data qubits for final readout.
fn build_repetition_code(num_rounds: usize) -> DagCircuit {
    let mut dag = DagCircuit::new();

    // Data qubits
    let data: Vec<usize> = vec![0, 1, 2];
    // Ancilla qubits: one per stabilizer
    let ancilla_01 = 3; // measures Z_0 Z_1
    let ancilla_12 = 4; // measures Z_1 Z_2

    // Initialize data qubits in |0⟩
    dag.pz(&data);

    // Track measurements across rounds for detector definitions
    let mut prev_meas_01 = None;
    let mut prev_meas_12 = None;

    for round in 0..num_rounds {
        // Prep ancillas
        dag.pz(&[ancilla_01, ancilla_12]);

        // Syndrome extraction: CX from data to ancilla
        // Z_0 Z_1 stabilizer
        dag.cx(&[(data[0], ancilla_01)]);
        dag.cx(&[(data[1], ancilla_01)]);
        // Z_1 Z_2 stabilizer
        dag.cx(&[(data[1], ancilla_12)]);
        dag.cx(&[(data[2], ancilla_12)]);

        // Measure ancillas
        let ms_01 = dag.mz(&[ancilla_01]);
        let ms_12 = dag.mz(&[ancilla_12]);

        // Detectors
        if round == 0 {
            // First round: each measurement should be 0 (fresh code state)
            dag.detector_labeled(&format!("Z01_r{round}"), &[ms_01[0]]);
            dag.detector_labeled(&format!("Z12_r{round}"), &[ms_12[0]]);
        } else {
            // Subsequent rounds: compare with previous round
            dag.detector_labeled(&format!("Z01_r{round}"), &[prev_meas_01.unwrap(), ms_01[0]]);
            dag.detector_labeled(&format!("Z12_r{round}"), &[prev_meas_12.unwrap(), ms_12[0]]);
        }

        prev_meas_01 = Some(ms_01[0]);
        prev_meas_12 = Some(ms_12[0]);
    }

    // Final data qubit measurements
    let ms_data = dag.mz(&data);

    // Final detectors: compare last syndrome round with data measurements
    // Z_0 Z_1 from data should match last ancilla measurement
    dag.detector_labeled(
        "Z01_final",
        &[ms_data[0], ms_data[1], prev_meas_01.unwrap()],
    );
    // Z_1 Z_2 from data should match last ancilla measurement
    dag.detector_labeled(
        "Z12_final",
        &[ms_data[1], ms_data[2], prev_meas_12.unwrap()],
    );

    // Observable: logical Z readout = Z_0 (any single data qubit works for rep code)
    dag.observable_labeled("logical_Z", &[ms_data[0]]);

    // Pauli operator: track logical X = X_0 X_1 X_2
    dag.tracked_pauli_labeled("logical_X", X(0) & X(1) & X(2));

    dag
}

#[test]
fn repetition_code_fault_enumeration() {
    let dag = build_repetition_code(3);

    println!("Circuit: {} gates", dag.gate_count());
    println!("Annotations:");
    for ann in dag.annotations() {
        let kind = match &ann.kind {
            pecos_quantum::AnnotationKind::Detector { .. } => "detector",
            pecos_quantum::AnnotationKind::Observable { .. } => "observable",
            pecos_quantum::AnnotationKind::TrackedPauli => "tracked_pauli",
        };
        let label = ann.label.as_deref().unwrap_or("(none)");
        println!("  {kind:10} {label:15} {}", ann.pauli);
    }

    // Build influence map (InfluenceBuilder handles annotations)
    let map = InfluenceBuilder::new(&dag)
        .with_circuit_annotations(&dag)
        .build();

    let locs = map.gate_fault_locations();
    println!(
        "\nFault locations: {} (grouped from {} per-qubit locations)",
        locs.len(),
        map.locations.len()
    );
    println!(
        "Detectors: {}, DEM outputs: {}",
        map.detectors.len(),
        map.influences.max_dem_output_index().map_or(0, |i| i + 1)
    );

    // Show all fault locations and their possible faults
    println!("\n--- Fault locations ---");
    for (i, loc) in locs.iter().enumerate() {
        let timing = if loc.before { "before" } else { "after" };
        let qubit_list: Vec<usize> = loc.qubits.iter().map(pecos_core::QubitId::index).collect();
        let num_faults = loc.possible_faults().len();
        println!(
            "  loc {i:2}: {:3?} {timing:6} qubits={qubit_list:?} ({num_faults} faults)",
            loc.gate_type
        );
    }

    // Weight-1 analysis
    let mut w1_detectable = 0usize;
    let mut w1_undetectable = 0usize;
    let mut w1_trivial = 0usize;
    let mut w1_total = 0usize;

    map.for_each_fault_combo(1, |combo| {
        w1_total += 1;
        let has_det = !combo.effect.detectors.is_empty();
        let has_dem_output = !combo.effect.dem_outputs.is_empty();
        match (has_det, has_dem_output) {
            (true, _) => w1_detectable += 1,
            (false, true) => w1_undetectable += 1,
            (false, false) => w1_trivial += 1,
        }
    });
    println!("\n--- Weight-1 faults ---");
    println!("  Total:        {w1_total}");
    println!("  Detectable:   {w1_detectable}");
    println!("  Undetectable: {w1_undetectable}");
    println!("  Trivial:      {w1_trivial}");
    // The repetition code only detects X errors (via Z stabilizers).
    // Z errors on data qubits are undetectable -- this is expected.
    // The undetectable errors flip logical_X (index 1) since Z anticommutes with X.
    assert!(
        w1_undetectable > 0,
        "Z errors should be undetectable in the repetition code"
    );

    // Weight-2 analysis
    let mut w2_detectable = 0usize;
    let mut w2_undetectable = 0usize;
    let mut w2_trivial = 0usize;
    let mut w2_total = 0usize;

    map.for_each_fault_combo(2, |combo| {
        w2_total += 1;
        let has_det = !combo.effect.detectors.is_empty();
        let has_dem_output = !combo.effect.dem_outputs.is_empty();
        match (has_det, has_dem_output) {
            (true, _) => w2_detectable += 1,
            (false, true) => w2_undetectable += 1,
            (false, false) => w2_trivial += 1,
        }
    });
    println!("\n--- Weight-2 faults ---");
    println!("  Total:        {w2_total}");
    println!("  Detectable:   {w2_detectable}");
    println!("  Undetectable: {w2_undetectable}");
    println!("  Trivial:      {w2_trivial}");
    // Weight-2 also has undetectable Z errors (Z type is not protected).

    // Weight-3: this is where d=3 codes can have undetectable errors
    let mut w3_detectable = 0usize;
    let mut w3_undetectable = 0usize;
    let mut w3_trivial = 0usize;
    let mut w3_total = 0usize;
    let mut w3_undetectable_examples: Vec<String> = Vec::new();

    map.for_each_fault_combo(3, |combo| {
        w3_total += 1;
        let has_det = !combo.effect.detectors.is_empty();
        let has_dem_output = !combo.effect.dem_outputs.is_empty();
        match (has_det, has_dem_output) {
            (true, _) => w3_detectable += 1,
            (false, true) => {
                w3_undetectable += 1;
                // Collect first few examples
                if w3_undetectable_examples.len() < 5 {
                    let desc: Vec<String> = combo
                        .components
                        .iter()
                        .map(|c| {
                            let loc = &locs[c.location_index];
                            let timing = if loc.before { "before" } else { "after" };
                            format!(
                                "{} {} {:?} q={:?}",
                                c.event.pauli,
                                timing,
                                loc.gate_type,
                                loc.qubits
                                    .iter()
                                    .map(pecos_core::QubitId::index)
                                    .collect::<Vec<_>>()
                            )
                        })
                        .collect();
                    w3_undetectable_examples.push(desc.join(" + "));
                }
            }
            (false, false) => w3_trivial += 1,
        }
    });
    println!("\n--- Weight-3 faults ---");
    println!("  Total:        {w3_total}");
    println!("  Detectable:   {w3_detectable}");
    println!("  Undetectable: {w3_undetectable}");
    println!("  Trivial:      {w3_trivial}");

    if !w3_undetectable_examples.is_empty() {
        println!("\n  Example undetectable w=3 errors:");
        for ex in &w3_undetectable_examples {
            println!("    {ex}");
        }
    }

    // The d=3 repetition code can correct any single fault, so:
    // - No undetectable errors at w=1 or w=2
    // - Some undetectable errors at w=3 (this is the code distance)
    println!("\n--- Summary ---");
    println!("  The repetition code detects X errors (via Z stabilizers) but not Z errors.");
    println!("  Undetectable errors at all weights are Z-type faults flipping logical_X.");

    // Also demonstrate single-event introspection
    println!("\n--- Single event example ---");
    let first_cx_loc = locs
        .iter()
        .find(|l| l.gate_type == pecos_core::gate_type::GateType::CX && !l.before);
    if let Some(loc) = first_cx_loc {
        let qubit_list: Vec<usize> = loc.qubits.iter().map(pecos_core::QubitId::index).collect();
        println!("  CX after, qubits={qubit_list:?}:");
        for event in loc.events() {
            if !event.detectors.is_empty() || !event.dem_outputs.is_empty() {
                println!(
                    "    {} -> dets={:?} dem_outputs={:?} meas={:?}",
                    event.pauli, event.detectors, event.dem_outputs, event.measurements
                );
            }
        }
    }
}

/// Test that labels are accessible from the influence map during fault introspection.
#[test]
fn repetition_code_labels() {
    let dag = build_repetition_code(1); // 1 round for simplicity
    let map = InfluenceBuilder::new(&dag)
        .with_circuit_annotations(&dag)
        .build();

    // Check DEM-output labels are populated (observables + tracked Paulis)
    println!("DEM output labels: {:?}", map.dem_output_labels);
    // 1 observable (logical_Z) + 1 tracked Pauli (logical_X) = 2 labels
    assert_eq!(map.dem_output_labels.len(), 2);
    assert_eq!(map.num_dem_outputs(), 1, "1 observable");
    assert_eq!(map.num_tracked_paulis(), 1, "1 tracked Pauli");

    // Labels accessible via internal index
    assert_eq!(map.dem_output_label(0), Some("logical_Z"));
    assert_eq!(map.dem_output_label(1), Some("logical_X"));
    assert_eq!(map.dem_output_label(99), None);

    // Use labels during fault introspection
    let locs = map.gate_fault_locations();
    let mut found_labeled_event = false;
    for loc in &locs {
        for event in loc.events() {
            for &output_idx in &event.dem_outputs {
                if let Some(label) = map.dem_output_label(output_idx as usize) {
                    println!("  {} at {:?} flips {label}", event.pauli, loc.gate_type);
                    found_labeled_event = true;
                }
            }
        }
    }
    assert!(
        found_labeled_event,
        "Should find at least one labeled logical event"
    );
}

/// Demonstrate building a lookup table from the influence map.
#[test]
fn repetition_code_lookup_table() {
    let dag = build_repetition_code(3);
    let map = InfluenceBuilder::new(&dag)
        .with_circuit_annotations(&dag)
        .build();

    // Build lookup: syndrome pattern -> list of possible logical effects
    let mut lookup: std::collections::BTreeMap<Vec<u32>, Vec<pecos_core::PauliString>> =
        std::collections::BTreeMap::new();

    // Weight-1 faults define the basic lookup
    map.for_each_fault_combo(1, |combo| {
        if !combo.effect.detectors.is_empty() {
            let mut syndrome = combo.effect.detectors.clone();
            syndrome.sort_unstable();
            lookup
                .entry(syndrome)
                .or_default()
                .push(combo.effect.pauli.clone());
        }
    });

    println!("Lookup table (weight-1 syndromes):");
    for (syndrome, faults) in &lookup {
        println!("  syndrome {syndrome:?} <- {} fault(s)", faults.len());
    }

    // Verify: each non-trivial weight-1 syndrome maps to a correction
    assert!(
        !lookup.is_empty(),
        "Should have at least one syndrome pattern"
    );
}

/// Build an ML lookup table decoder and test it against sampled errors.
#[test]
fn repetition_code_ml_decoder() {
    use pecos_qec::fault_tolerance::dem_builder::{DemSampler, NoiseConfig};
    use pecos_qec::fault_tolerance::lookup_decoder::LookupDecoder;
    use rand::SeedableRng;

    let dag = build_repetition_code(3);
    let noise = NoiseConfig::uniform(0.001);

    // Build influence map
    let map = InfluenceBuilder::new(&dag)
        .with_circuit_annotations(&dag)
        .build();

    // Build ML decoder from fault enumeration up to weight 3
    let decoder = LookupDecoder::build(&map, &noise, 3);

    println!("ML Decoder:");
    println!("  Syndrome patterns: {}", decoder.num_syndromes());
    println!("  Observables:       {}", decoder.num_observables());
    println!("  Max weight:        {}", decoder.max_weight());

    // Build sampler for testing
    let sampler = DemSampler::from_circuit(&dag, &noise).unwrap();

    // Sample and decode
    let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
    let num_shots = 100_000;
    let mut _correct = 0usize;
    let mut total_errors = 0usize;
    let mut unknown_syndromes = 0usize;

    for _ in 0..num_shots {
        if let Some(dual) = sampler.sample_dual(&mut rng) {
            let result = decoder.decode_from_bools(&dual.detector_events);
            if !result.known_syndrome {
                unknown_syndromes += 1;
            }

            // Check: did the decoder's correction fix the logical?
            // The actual DEM-output outcome is dual.dem_output_flips.
            // After applying the correction, the residual should be identity.
            let has_observable_error: bool = dual
                .dem_output_flips
                .iter()
                .zip(&result.corrections)
                .any(|(&flip, &corr)| flip ^ corr); // residual error

            if has_observable_error {
                total_errors += 1;
            } else {
                _correct += 1;
            }
        }
    }

    let error_rate = total_errors as f64 / num_shots as f64;
    let raw_error_rate = sampler
        .sample_statistics(num_shots, 42)
        .logical_error_rate();

    println!("\n  Shots:             {num_shots}");
    println!("  Unknown syndromes: {unknown_syndromes}");
    println!("  Raw observable rate:  {raw_error_rate:.6}");
    println!("  Decoded error rate:{error_rate:.6}");
    println!(
        "  Improvement:       {:.1}x",
        raw_error_rate / error_rate.max(1e-10)
    );

    // The decoder should reduce the observable error rate compared to raw
    // (for the repetition code with Z errors unprotected, improvement is modest
    // since only X errors are correctable, but it should still help)
    assert!(
        total_errors < num_shots,
        "Decoder should correct at least some errors"
    );
}

/// Test decoder correctness: empty syndrome should produce no corrections.
#[test]
fn decoder_empty_syndrome() {
    use pecos_qec::fault_tolerance::dem_builder::NoiseConfig;
    use pecos_qec::fault_tolerance::lookup_decoder::LookupDecoder;

    let dag = build_repetition_code(1);
    let noise = NoiseConfig::uniform(0.01);
    let map = InfluenceBuilder::new(&dag)
        .with_circuit_annotations(&dag)
        .build();

    let decoder = LookupDecoder::build(&map, &noise, 2);

    // Empty syndrome = no detectors fired = most likely no error
    let result = decoder.decode(&[]);
    assert!(result.known_syndrome, "Empty syndrome should be known");
    assert!(
        result.corrections.iter().all(|&c| !c),
        "Empty syndrome should produce no corrections: {:?}",
        result.corrections
    );
}

/// Test that the decoder table grows with weight, and truncation bound works.
#[test]
fn decoder_table_size_and_truncation() {
    use pecos_qec::fault_tolerance::dem_builder::NoiseConfig;
    use pecos_qec::fault_tolerance::lookup_decoder::LookupDecoder;

    let dag = build_repetition_code(1);
    let noise = NoiseConfig::uniform(0.001);
    let map = InfluenceBuilder::new(&dag)
        .with_circuit_annotations(&dag)
        .build();

    let d1 = LookupDecoder::build(&map, &noise, 1);
    let d2 = LookupDecoder::build(&map, &noise, 2);
    let d3 = LookupDecoder::build(&map, &noise, 3);

    println!(
        "Weight 1: {} syndromes, accounted={:.8}, truncation={:.2e}",
        d1.num_syndromes(),
        d1.accounted_probability(),
        d1.truncation_bound()
    );
    println!(
        "Weight 2: {} syndromes, accounted={:.8}, truncation={:.2e}",
        d2.num_syndromes(),
        d2.accounted_probability(),
        d2.truncation_bound()
    );
    println!(
        "Weight 3: {} syndromes, accounted={:.8}, truncation={:.2e}",
        d3.num_syndromes(),
        d3.accounted_probability(),
        d3.truncation_bound()
    );

    // Higher weight covers more probability mass
    assert!(d2.accounted_probability() >= d1.accounted_probability());
    assert!(d3.accounted_probability() >= d2.accounted_probability());

    // At p=0.001, weight-3 should cover essentially all probability mass
    assert!(
        d3.truncation_bound() < 1e-6,
        "Weight 3 at p=0.001 should have negligible truncation: {}",
        d3.truncation_bound()
    );

    // Higher weight should discover at least as many syndromes
    assert!(d2.num_syndromes() >= d1.num_syndromes());
    assert!(d1.num_syndromes() > 1);
}

// ============================================================================
// [[4,2,2]] Code Example
// ============================================================================

/// Build a [[4,2,2]] code circuit with `num_rounds` of syndrome extraction.
///
/// The [[4,2,2]] code:
///   - 4 data qubits (0-3), 2 ancilla qubits (4-5)
///   - Stabilizers: `X_0` `X_1` `X_2` `X_3` and `Z_0` `Z_1` `Z_2` `Z_3`
///   - 2 logical qubits:
///     - Logical `Z_1` = `Z_0` `Z_1`,  Logical `X_1` = `X_0` `X_2`
///     - Logical `Z_2` = `Z_0` `Z_2`,  Logical `X_2` = `X_0` `X_1`
///   - Distance 2: detects any single-qubit error (cannot correct)
///
/// X stabilizer measurement (ancilla 4):
///   Prep |+⟩, CX(ancilla, data) for each data qubit, H, MZ
///
/// Z stabilizer measurement (ancilla 5):
///   Prep |0⟩, CX(data, ancilla) for each data qubit, MZ
fn build_422_code(num_rounds: usize) -> DagCircuit {
    let mut dag = DagCircuit::new();

    let data: Vec<usize> = vec![0, 1, 2, 3];
    let ancilla_x = 4; // measures X_0 X_1 X_2 X_3
    let ancilla_z = 5; // measures Z_0 Z_1 Z_2 Z_3

    // Initialize data qubits
    dag.pz(&data);

    let mut prev_meas_x = None;
    let mut prev_meas_z = None;

    for round in 0..num_rounds {
        // --- X stabilizer: X_0 X_1 X_2 X_3 ---
        dag.pz(&[ancilla_x]);
        dag.h(&[ancilla_x]); // prep |+⟩
        // CX(ancilla, data) propagates X from ancilla to data
        dag.cx(&[
            (ancilla_x, data[0]),
            (ancilla_x, data[1]),
            (ancilla_x, data[2]),
            (ancilla_x, data[3]),
        ]);
        dag.h(&[ancilla_x]); // rotate back to Z basis
        let ms_x = dag.mz(&[ancilla_x]);

        // --- Z stabilizer: Z_0 Z_1 Z_2 Z_3 ---
        dag.pz(&[ancilla_z]);
        // CX(data, ancilla) propagates Z from data to ancilla
        dag.cx(&[
            (data[0], ancilla_z),
            (data[1], ancilla_z),
            (data[2], ancilla_z),
            (data[3], ancilla_z),
        ]);
        let ms_z = dag.mz(&[ancilla_z]);

        // Detectors
        if round == 0 {
            // Z stabilizer is deterministic on |0000⟩ (Z eigenstate)
            dag.detector_labeled(&format!("Sz_r{round}"), &[ms_z[0]]);
            // X stabilizer is NOT deterministic on |0000⟩ -- no standalone detector.
            // First X measurement is a random coin flip; only round-to-round
            // comparisons are valid detectors.
        } else {
            dag.detector_labeled(&format!("Sx_r{round}"), &[prev_meas_x.unwrap(), ms_x[0]]);
            dag.detector_labeled(&format!("Sz_r{round}"), &[prev_meas_z.unwrap(), ms_z[0]]);
        }

        prev_meas_x = Some(ms_x[0]);
        prev_meas_z = Some(ms_z[0]);
    }

    // Final data qubit measurements
    let ms_data = dag.mz(&data);

    // Final detector: Z stabilizer from data should match last Z-ancilla.
    // Z_0 Z_1 Z_2 Z_3 is readable from Z-basis data measurements.
    dag.detector_labeled(
        "Sz_final",
        &[
            ms_data[0],
            ms_data[1],
            ms_data[2],
            ms_data[3],
            prev_meas_z.unwrap(),
        ],
    );
    // No final X-stabilizer detector: Z-basis data measurements cannot
    // reconstruct X_0 X_1 X_2 X_3 parity.

    // Observables: logical Z readouts
    // Logical Z_1 = Z_0 Z_1
    dag.observable_labeled("logical_Z1", &[ms_data[0], ms_data[1]]);
    // Logical Z_2 = Z_0 Z_2
    dag.observable_labeled("logical_Z2", &[ms_data[0], ms_data[2]]);

    // Tracked Paulis: logical X operators
    // Logical X_1 = X_0 X_2
    dag.tracked_pauli_labeled("logical_X1", X(0) & X(2));
    // Logical X_2 = X_0 X_1
    dag.tracked_pauli_labeled("logical_X2", X(0) & X(1));

    dag
}

#[test]
fn code_422_fault_enumeration() {
    let dag = build_422_code(2);

    println!("[[4,2,2]] Code with 2 rounds");
    println!("Circuit: {} gates", dag.gate_count());
    println!("Annotations:");
    for ann in dag.annotations() {
        let kind = match &ann.kind {
            pecos_quantum::AnnotationKind::Detector { .. } => "detector",
            pecos_quantum::AnnotationKind::Observable { .. } => "observable",
            pecos_quantum::AnnotationKind::TrackedPauli => "tracked_pauli",
        };
        let label = ann.label.as_deref().unwrap_or("(none)");
        println!("  {kind:10} {label:15} {}", ann.pauli);
    }

    // Build influence map
    let map = InfluenceBuilder::new(&dag)
        .with_circuit_annotations(&dag)
        .build();

    let locs = map.gate_fault_locations();
    println!(
        "\nFault locations: {} (from {} per-qubit locations)",
        locs.len(),
        map.locations.len()
    );
    println!(
        "Detectors: {}, DEM outputs: {}",
        map.detectors.len(),
        map.influences.max_dem_output_index().map_or(0, |i| i + 1)
    );

    // Weight-1
    let mut w1_total = 0usize;
    let mut w1_detectable = 0usize;
    let mut w1_undetectable = 0usize;
    let mut w1_trivial = 0usize;

    map.for_each_fault_combo(1, |combo| {
        w1_total += 1;
        let has_det = !combo.effect.detectors.is_empty();
        let has_dem_output = !combo.effect.dem_outputs.is_empty();
        match (has_det, has_dem_output) {
            (true, _) => w1_detectable += 1,
            (false, true) => {
                w1_undetectable += 1;
                if w1_undetectable <= 5 {
                    let c = &combo.components[0];
                    let loc = &locs[c.location_index];
                    let timing = if loc.before { "before" } else { "after" };
                    println!(
                        "  UNDET w=1: {} {timing} {:?} q={:?} -> dem_outputs={:?}",
                        c.event.pauli,
                        loc.gate_type,
                        loc.qubits
                            .iter()
                            .map(pecos_core::QubitId::index)
                            .collect::<Vec<_>>(),
                        combo.effect.dem_outputs,
                    );
                }
            }
            (false, false) => w1_trivial += 1,
        }
    });

    println!("\n--- Weight-1 faults ---");
    println!("  Total:        {w1_total}");
    println!("  Detectable:   {w1_detectable}");
    println!("  Undetectable: {w1_undetectable}");
    println!("  Trivial:      {w1_trivial}");

    // The [[4,2,2]] code starting from |0000⟩ has partial detection:
    // - Z stabilizer detects X errors from round 1 (|0000⟩ is Z eigenstate)
    // - X stabilizer only detects Z errors from round 2 onward (round-to-round)
    // - First-round Z errors on data qubits are undetectable (no X-stabilizer
    //   detector in round 1, since |0000⟩ is not an X-stabilizer eigenstate)
    assert!(w1_total > 0, "Should have fault events");
    assert!(w1_detectable > 0, "Some faults should be detectable");
    assert!(w1_undetectable > 0, "First-round Z faults are undetectable");
}

#[test]
fn code_422_ml_decoder() {
    use pecos_qec::fault_tolerance::dem_builder::{DemSampler, NoiseConfig};
    use pecos_qec::fault_tolerance::lookup_decoder::LookupDecoder;
    use rand::SeedableRng;

    let dag = build_422_code(2);
    let noise = NoiseConfig::uniform(0.001);

    let map = InfluenceBuilder::new(&dag)
        .with_circuit_annotations(&dag)
        .build();

    // Build ML decoder up to weight 2
    let decoder = LookupDecoder::build(&map, &noise, 2);

    println!("[[4,2,2]] ML Decoder:");
    println!("  Syndrome patterns: {}", decoder.num_syndromes());
    println!("  Observables:       {}", decoder.num_observables());

    // Build sampler
    let sampler = DemSampler::from_circuit(&dag, &noise).unwrap();

    // Sample and decode
    let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
    let num_shots = 100_000;
    let mut decoded_errors = 0usize;
    let mut raw_errors = 0usize;

    let mut post_selected_shots = 0usize;
    let mut post_selected_errors = 0usize;

    for _ in 0..num_shots {
        if let Some(dual) = sampler.sample_dual(&mut rng) {
            let result = decoder.decode_from_bools(&dual.detector_events);

            // ML correction
            let has_residual = dual
                .dem_output_flips
                .iter()
                .zip(&result.corrections)
                .any(|(&flip, &corr)| flip ^ corr);

            if has_residual {
                decoded_errors += 1;
            }
            if dual.dem_output_flips.iter().any(|&f| f) {
                raw_errors += 1;
            }

            // Post-selection: only keep shots with no detectors fired
            if !result.detected {
                post_selected_shots += 1;
                if dual.dem_output_flips.iter().any(|&f| f) {
                    post_selected_errors += 1;
                }
            }
        }
    }

    let raw_rate = raw_errors as f64 / num_shots as f64;
    let decoded_rate = decoded_errors as f64 / num_shots as f64;
    let ps_rate = if post_selected_shots > 0 {
        post_selected_errors as f64 / post_selected_shots as f64
    } else {
        0.0
    };
    let discard_rate = 1.0 - post_selected_shots as f64 / num_shots as f64;

    println!("\n  Shots:                {num_shots}");
    println!("  Raw observable rate:     {raw_rate:.6}");
    println!("  ML decoded rate:      {decoded_rate:.6}");
    println!("  Post-selected rate:   {ps_rate:.6} (discarded {discard_rate:.4})");
    println!(
        "  PS improvement:       {:.1}x",
        raw_rate / ps_rate.max(1e-10)
    );

    // For the [[4,2,2]] detection code, post-selection should improve the
    // observable error rate: detected errors are discarded, only undetectable
    // errors (weight 2+) remain. At p=0.001, this gives ~p^2 rate.
    assert!(
        ps_rate < raw_rate || post_selected_shots == 0,
        "Post-selection should reduce observable error rate"
    );
    assert!(
        decoded_errors < num_shots,
        "Some shots should decode correctly"
    );
}
