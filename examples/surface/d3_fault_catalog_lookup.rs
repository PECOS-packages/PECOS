// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0

//! Build a truncated maximum-likelihood lookup table from the Rust fault catalog.
//!
//! This example keeps the expensive loop in Rust:
//! - build a d=3 rotated surface-code Z-memory experiment,
//! - enumerate all k-fault configurations for k <= `max_faults`,
//! - XOR detector / observable effects via `fault_configurations(k)`,
//! - aggregate `configuration_probability` into a lookup table.
//!
//! The circuit builder below intentionally uses a simple sequential stabilizer
//! extraction schedule. The point of the example is the fault-catalog lookup
//! aggregation path, not the optimized surface-code scheduling used by the
//! larger sweep scripts.
//!
//! Run from the PECOS repo root:
//!
//! ```text
//! cargo run -p pecos-qec --example surface_d3_fault_catalog_lookup
//! ```

use pecos_qec::SurfaceCode;
use pecos_qec::fault_tolerance::fault_sampler::{
    FaultCatalog, StochasticNoiseParams, build_fault_catalog,
};
use pecos_quantum::{Attribute, TickCircuit, TickMeasRef};
use std::collections::BTreeMap;
use std::time::Instant;

type Syndrome = Vec<usize>;
type Logical = Vec<usize>;
type LogicalWeights = BTreeMap<Logical, f64>;
type LookupWeights = BTreeMap<Syndrome, LogicalWeights>;

#[derive(Debug)]
struct MemoryCircuit {
    circuit: TickCircuit,
    num_detectors: usize,
    num_observables: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let distance = 3;
    let rounds = 3;
    let max_faults = 2;
    let p = 0.001;

    let memory = build_d3_z_memory_circuit(rounds)?;
    let noise = StochasticNoiseParams {
        p1: p / 10.0,
        p2: p,
        p_meas: p,
        p_prep: p,
    };

    println!("d={distance} rotated surface-code Z-memory experiment");
    println!(
        "rounds={rounds}, detectors={}, observables={}",
        memory.num_detectors, memory.num_observables
    );
    println!(
        "noise: p1={:.3e}, p2={:.3e}, p_meas={:.3e}, p_prep={:.3e}",
        noise.p1, noise.p2, noise.p_meas, noise.p_prep
    );

    let catalog = build_fault_catalog(&memory.circuit, &noise)?;
    let total_alternatives: usize = catalog
        .locations
        .iter()
        .map(|loc| loc.num_alternatives)
        .sum();

    println!(
        "catalog: {} locations, {total_alternatives} single-location alternatives",
        catalog.locations.len()
    );

    let started = Instant::now();
    let (weights, configs_by_weight) = build_lookup_weights(&catalog, max_faults);
    let decoder = choose_most_likely_logicals(&weights);
    let elapsed = started.elapsed();

    println!("enumerated configurations:");
    for (k, count) in configs_by_weight {
        println!("  k={k}: {count}");
    }
    println!(
        "lookup table: {} syndromes covered, built in {:.3?}",
        decoder.len(),
        elapsed
    );

    print_top_syndromes(&weights, &decoder, 10);

    Ok(())
}

fn build_lookup_weights(
    catalog: &FaultCatalog,
    max_faults: usize,
) -> (LookupWeights, Vec<(usize, usize)>) {
    let mut weights: LookupWeights = BTreeMap::new();
    let mut configs_by_weight = Vec::new();

    for k in 0..=max_faults {
        let mut count = 0usize;
        for event in catalog.fault_configurations(k) {
            add_lookup_weight(
                &mut weights,
                event.affected_detectors,
                event.affected_observables,
                event.configuration_probability,
            );
            count += 1;
        }
        configs_by_weight.push((k, count));
    }

    (weights, configs_by_weight)
}

fn add_lookup_weight(
    weights: &mut LookupWeights,
    syndrome: Syndrome,
    logical: Logical,
    probability: f64,
) {
    weights
        .entry(syndrome)
        .or_default()
        .entry(logical)
        .and_modify(|p| *p += probability)
        .or_insert(probability);
}

fn choose_most_likely_logicals(weights: &LookupWeights) -> BTreeMap<Syndrome, Logical> {
    weights
        .iter()
        .map(|(syndrome, logical_weights)| {
            let best_logical = logical_weights
                .iter()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(logical, _)| logical.clone())
                .unwrap_or_default();
            (syndrome.clone(), best_logical)
        })
        .collect()
}

fn print_top_syndromes(
    weights: &LookupWeights,
    decoder: &BTreeMap<Syndrome, Logical>,
    limit: usize,
) {
    let mut rows: Vec<_> = weights
        .iter()
        .map(|(syndrome, logical_weights)| {
            let total_p: f64 = logical_weights.values().sum();
            (total_p, syndrome, logical_weights)
        })
        .collect();
    rows.sort_by(|(a, _, _), (b, _, _)| b.total_cmp(a));

    println!();
    println!("top {limit} syndrome classes by truncated probability:");
    for (rank, (total_p, syndrome, logical_weights)) in rows.into_iter().take(limit).enumerate() {
        let correction = decoder.get(syndrome).cloned().unwrap_or_default();
        let no_logical = logical_weights.get(&Vec::new()).copied().unwrap_or(0.0);
        let logical_total = total_p - no_logical;
        println!(
            "  {:>2}. syndrome={:?} total={:.6e} logical_weight={:.6e} correction={:?}",
            rank + 1,
            syndrome,
            total_p,
            logical_total,
            correction
        );
    }
}

fn build_d3_z_memory_circuit(rounds: usize) -> Result<MemoryCircuit, String> {
    let code = SurfaceCode::rotated(3)?;
    let num_data = code.num_data_qubits();
    let x_ancilla_offset = num_data;
    let z_ancilla_offset = x_ancilla_offset + code.num_x_stabilizers();

    let x_ancilla = |idx: usize| x_ancilla_offset + idx;
    let z_ancilla = |idx: usize| z_ancilla_offset + idx;

    let mut circuit = TickCircuit::new();
    let data_qubits: Vec<usize> = (0..num_data).collect();
    circuit.tick().pz(&data_qubits);

    let mut x_round_measurements: Vec<Vec<TickMeasRef>> = Vec::with_capacity(rounds);
    let mut z_round_measurements: Vec<Vec<TickMeasRef>> = Vec::with_capacity(rounds);

    for _round in 0..rounds {
        let x_ancillas: Vec<usize> = (0..code.num_x_stabilizers()).map(x_ancilla).collect();
        let z_ancillas: Vec<usize> = (0..code.num_z_stabilizers()).map(z_ancilla).collect();

        circuit.tick().pz(&x_ancillas);
        circuit.tick().pz(&z_ancillas);
        circuit.tick().h(&x_ancillas);

        for check in code.x_stabilizers() {
            let anc = x_ancilla(check.index);
            for data in check.qubits() {
                circuit.tick().cx(&[(anc, data)]);
            }
        }

        for check in code.z_stabilizers() {
            let anc = z_ancilla(check.index);
            for data in check.qubits() {
                circuit.tick().cx(&[(data, anc)]);
            }
        }

        circuit.tick().h(&x_ancillas);

        let x_refs = circuit.tick().mz(&x_ancillas);
        let z_refs = circuit.tick().mz(&z_ancillas);
        x_round_measurements.push(x_refs);
        z_round_measurements.push(z_refs);
    }

    let final_data_measurements = circuit.tick().mz(&data_qubits);
    let num_measurements = circuit.num_measurements();

    let mut detectors: Vec<Vec<i32>> = Vec::new();

    // Initial Z-basis boundary detectors: data starts in |0...0>, so the first
    // Z-check round is deterministic. Without these, an initial data X fault can
    // flip every repeated Z-check round and the final data parity, cancelling all
    // later detectors.
    for &meas_ref in z_round_measurements[0]
        .iter()
        .take(code.num_z_stabilizers())
    {
        detectors.push(relative_records(num_measurements, &[meas_ref]));
    }

    // Repeated syndrome detectors: current stabilizer measurement XOR previous
    // stabilizer measurement. These are deterministic after the first round.
    for round in 1..rounds {
        for (&current, &previous) in x_round_measurements[round]
            .iter()
            .zip(x_round_measurements[round - 1].iter())
            .take(code.num_x_stabilizers())
        {
            detectors.push(relative_records(num_measurements, &[current, previous]));
        }
        for (&current, &previous) in z_round_measurements[round]
            .iter()
            .zip(z_round_measurements[round - 1].iter())
            .take(code.num_z_stabilizers())
        {
            detectors.push(relative_records(num_measurements, &[current, previous]));
        }
    }

    // Final Z-basis detectors: last Z-stabilizer result XOR the final data
    // measurements in that stabilizer support.
    let last_round = rounds - 1;
    for check in code.z_stabilizers() {
        let mut refs = vec![z_round_measurements[last_round][check.index]];
        refs.extend(
            check
                .qubits()
                .into_iter()
                .map(|q| final_data_measurements[q]),
        );
        detectors.push(relative_records(num_measurements, &refs));
    }

    let logical_z_refs: Vec<TickMeasRef> = code
        .logical_z()
        .data_qubits
        .iter()
        .map(|&q| final_data_measurements[q])
        .collect();
    let observables = vec![relative_records(num_measurements, &logical_z_refs)];

    circuit.set_meta(
        "num_measurements",
        Attribute::String(num_measurements.to_string()),
    );
    circuit.set_meta("detectors", Attribute::String(records_json(&detectors)));
    circuit.set_meta("observables", Attribute::String(records_json(&observables)));

    Ok(MemoryCircuit {
        circuit,
        num_detectors: detectors.len(),
        num_observables: observables.len(),
    })
}

fn relative_records(num_measurements: usize, refs: &[TickMeasRef]) -> Vec<i32> {
    let num_measurements = i32::try_from(num_measurements).expect("measurement count fits in i32");
    refs.iter()
        .map(|m| {
            i32::try_from(m.record_idx).expect("measurement record index fits in i32")
                - num_measurements
        })
        .collect()
}

fn records_json(records: &[Vec<i32>]) -> String {
    let entries: Vec<String> = records
        .iter()
        .map(|rs| {
            let values = rs.iter().map(i32::to_string).collect::<Vec<_>>().join(",");
            format!(r#"{{"records":[{values}]}}"#)
        })
        .collect();
    format!("[{}]", entries.join(","))
}

#[cfg(test)]
mod tests {
    use super::*;
    use pecos_qec::fault_tolerance::targeted_lookup_decoder::TargetedLookupDecoder;
    use std::collections::BTreeSet;

    #[test]
    fn accumulates_repeated_logical_weights_for_a_syndrome() {
        let mut weights = LookupWeights::new();

        add_lookup_weight(&mut weights, vec![1, 3], vec![0], 0.125);
        add_lookup_weight(&mut weights, vec![1, 3], vec![0], 0.375);
        add_lookup_weight(&mut weights, vec![1, 3], Vec::new(), 0.250);

        assert_close(weights[&vec![1, 3]][&vec![0]], 0.500);
        assert_close(weights[&vec![1, 3]][&Vec::new()], 0.250);
    }

    #[test]
    fn decoder_picks_most_likely_logical_per_syndrome() {
        let mut weights = LookupWeights::new();

        add_lookup_weight(&mut weights, Vec::new(), Vec::new(), 0.900);
        add_lookup_weight(&mut weights, Vec::new(), vec![0], 0.100);
        add_lookup_weight(&mut weights, vec![2], Vec::new(), 0.200);
        add_lookup_weight(&mut weights, vec![2], vec![0], 0.700);

        let decoder = choose_most_likely_logicals(&weights);

        assert_eq!(decoder[&Vec::new()], Vec::<usize>::new());
        assert_eq!(decoder[&vec![2]], vec![0]);
    }

    #[test]
    fn small_catalog_lookup_matches_hand_calculation() {
        let mut circuit = TickCircuit::new();
        circuit.tick().h(&[0]);
        circuit.tick().mz(&[0]);
        circuit.set_meta("num_measurements", Attribute::String("1".to_string()));
        circuit.set_meta(
            "detectors",
            Attribute::String(r#"[{"records":[-1]}]"#.to_string()),
        );
        circuit.set_meta(
            "observables",
            Attribute::String(r#"[{"records":[-1]}]"#.to_string()),
        );

        let noise = StochasticNoiseParams {
            p1: 0.03,
            p2: 0.0,
            p_meas: 0.01,
            p_prep: 0.0,
        };
        let catalog = build_fault_catalog(&circuit, &noise).unwrap();
        let (weights, counts) = build_lookup_weights(&catalog, 1);
        let decoder = choose_most_likely_logicals(&weights);

        assert_eq!(counts, vec![(0, 1), (1, 4)]);

        // k=0 no fault: 0.97 * 0.99 = 0.9603.
        // k=1 no-effect H alternative: (0.03 / 3) * 0.99 = 0.0099.
        assert_close(weights[&Vec::new()][&Vec::new()], 0.9702);

        // k=1 detector+logical events:
        // two H alternatives flip MZ: 2 * (0.03 / 3) * 0.99 = 0.0198.
        // one MZ flip: 0.01 * 0.97 = 0.0097.
        assert_close(weights[&vec![0]][&vec![0]], 0.0295);

        assert_eq!(decoder[&Vec::new()], Vec::<usize>::new());
        assert_eq!(decoder[&vec![0]], vec![0]);
    }

    #[test]
    fn small_catalog_decoder_corrects_every_truncated_event() {
        let mut circuit = TickCircuit::new();
        circuit.tick().h(&[0]);
        circuit.tick().mz(&[0]);
        circuit.set_meta("num_measurements", Attribute::String("1".to_string()));
        circuit.set_meta(
            "detectors",
            Attribute::String(r#"[{"records":[-1]}]"#.to_string()),
        );
        circuit.set_meta(
            "observables",
            Attribute::String(r#"[{"records":[-1]}]"#.to_string()),
        );

        let noise = StochasticNoiseParams {
            p1: 0.03,
            p2: 0.0,
            p_meas: 0.01,
            p_prep: 0.0,
        };
        let catalog = build_fault_catalog(&circuit, &noise).unwrap();
        let (weights, _) = build_lookup_weights(&catalog, 1);
        let decoder = choose_most_likely_logicals(&weights);

        let mut decoded = 0usize;
        for k in 0..=1 {
            for event in catalog.fault_configurations(k) {
                let correction = decoder
                    .get(&event.affected_detectors)
                    .expect("decoder should cover every truncated syndrome");
                let residual = xor_parity(&event.affected_observables, correction);
                assert!(
                    residual.is_empty(),
                    "failed to decode syndrome {:?}: event logical {:?}, correction {:?}",
                    event.affected_detectors,
                    event.affected_observables,
                    correction
                );
                decoded += 1;
            }
        }

        assert_eq!(decoded, 5);
    }

    #[test]
    fn d3_surface_lookup_builds_nontrivial_weight_one_table() {
        let memory = build_d3_z_memory_circuit(3).unwrap();
        let noise = StochasticNoiseParams {
            p1: 0.0001,
            p2: 0.001,
            p_meas: 0.001,
            p_prep: 0.001,
        };
        let catalog = build_fault_catalog(&memory.circuit, &noise).unwrap();
        let (weights, counts) = build_lookup_weights(&catalog, 1);
        let decoder = choose_most_likely_logicals(&weights);

        assert_eq!(memory.num_detectors, 24);
        assert_eq!(memory.num_observables, 1);
        assert_eq!(counts[0], (0, 1));
        assert_eq!(counts[1].0, 1);
        assert!(counts[1].1 > 1_000);
        assert!(weights.len() > 10);
        assert_eq!(decoder[&Vec::new()], Vec::<usize>::new());
    }

    #[test]
    fn d3_surface_weight_one_decoder_corrects_weight_one_events() {
        let memory = build_d3_z_memory_circuit(3).unwrap();
        let noise = StochasticNoiseParams {
            p1: 0.0001,
            p2: 0.001,
            p_meas: 0.001,
            p_prep: 0.001,
        };
        let catalog = build_fault_catalog(&memory.circuit, &noise).unwrap();
        let (weights, _) = build_lookup_weights(&catalog, 1);
        let decoder = choose_most_likely_logicals(&weights);

        let mut checked = 0usize;
        for k in 0..=1 {
            for event in catalog.fault_configurations(k) {
                let correction = decoder
                    .get(&event.affected_detectors)
                    .expect("decoder should cover every weight-one syndrome");
                let residual = xor_parity(&event.affected_observables, correction);
                assert!(
                    residual.is_empty(),
                    "failed to decode syndrome {:?}: event logical {:?}, correction {:?}",
                    event.affected_detectors,
                    event.affected_observables,
                    correction
                );
                checked += 1;
            }
        }

        assert_eq!(
            checked,
            1 + catalog
                .locations
                .iter()
                .map(|loc| loc.num_alternatives)
                .sum::<usize>()
        );
    }

    #[test]
    fn targeted_decoder_matches_bruteforce_on_real_d3_surface_catalog() {
        let memory = build_d3_z_memory_circuit(3).unwrap();
        let noise = StochasticNoiseParams {
            p1: 0.0001,
            p2: 0.001,
            p_meas: 0.001,
            p_prep: 0.001,
        };
        let catalog = build_fault_catalog(&memory.circuit, &noise).unwrap();
        let decoder = TargetedLookupDecoder::new(&catalog).max_faults(1);
        let base_probability = decoder.base_probability();

        let mut syndromes = BTreeSet::new();
        syndromes.insert(Vec::new());
        for event in catalog.fault_configurations(1) {
            if !event.affected_detectors.is_empty() {
                syndromes.insert(event.affected_detectors);
            }
            if syndromes.len() >= 8 {
                break;
            }
        }
        assert!(
            syndromes.len() >= 4,
            "surface catalog should expose several non-empty weight-one syndromes"
        );

        for syndrome in syndromes {
            let result = decoder.decode(&syndrome);
            let expected = brute_force_odds_for_syndrome(&catalog, 1, &syndrome, base_probability);
            assert_eq!(result.syndrome, syndrome);
            assert_logical_weights_close(&result.logical_weights, &expected);

            let expected_best = expected
                .iter()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(logical, _)| logical.clone())
                .unwrap_or_default();
            assert_eq!(result.best_logical, expected_best);
        }
    }

    fn brute_force_odds_for_syndrome(
        catalog: &FaultCatalog,
        max_faults: usize,
        syndrome: &[usize],
        base_probability: f64,
    ) -> LogicalWeights {
        let mut weights = LogicalWeights::new();
        for k in 0..=max_faults {
            for event in catalog.fault_configurations(k) {
                if event.affected_detectors == syndrome {
                    let odds = event.configuration_probability / base_probability;
                    weights
                        .entry(event.affected_observables)
                        .and_modify(|w| *w += odds)
                        .or_insert(odds);
                }
            }
        }
        weights
    }

    fn assert_logical_weights_close(actual: &LogicalWeights, expected: &LogicalWeights) {
        assert_eq!(
            actual.keys().collect::<Vec<_>>(),
            expected.keys().collect::<Vec<_>>()
        );
        for (logical, expected_weight) in expected {
            let actual_weight = actual[logical];
            let scale = expected_weight.abs().max(1e-15);
            assert!(
                (actual_weight - expected_weight).abs() / scale < 1e-10,
                "logical={logical:?}: expected {expected_weight:.12e}, got {actual_weight:.12e}"
            );
        }
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-12,
            "expected {expected}, got {actual}"
        );
    }

    fn xor_parity(a: &[usize], b: &[usize]) -> Vec<usize> {
        let mut out = std::collections::BTreeSet::new();
        for value in a.iter().chain(b.iter()) {
            if !out.remove(value) {
                out.insert(*value);
            }
        }
        out.into_iter().collect()
    }
}
