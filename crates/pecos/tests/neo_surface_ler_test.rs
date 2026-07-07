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

//! Surface-code memory LER equivalence between the engines and neo stacks
//! (validation-gate item V5).
//!
//! Builds a rotated surface-code Z-memory experiment once, emitting the
//! same circuit as both a QASM program (run through `sim()` on each stack)
//! and a `TickCircuit` (fed to the Rust DEM builder for decoding). Both
//! stacks' samples are decoded with the same MWPM decoder against the
//! same DEM, and the logical error rates are compared with Jeffreys
//! credible intervals.

#![cfg(feature = "neo")]

use pecos::{SimStack, sim};
use pecos_decoder_core::ObservableDecoder;
use pecos_engines::shot_results::ShotVec;
use pecos_fusion_blossom::FusionBlossomDecoder;
use pecos_num::jeffreys_interval;
use pecos_programs::Qasm;
use pecos_qec::SurfaceCode;
use pecos_qec::fault_tolerance::dem_builder::DemBuilder;
use pecos_quantum::{Attribute, TickCircuit, TickMeasRef};
use std::fmt::Write as _;

/// A surface-code memory experiment in both program representations.
struct MemoryExperiment {
    qasm: String,
    tick: TickCircuit,
    /// Detector definitions as relative measurement records (Stim style:
    /// record -k is the k-th most recent measurement).
    detectors: Vec<Vec<i32>>,
    /// The logical-Z observable as relative measurement records.
    observable: Vec<i32>,
    num_measurements: usize,
    /// Classical registers in declaration order: (name, width).
    registers: Vec<(String, usize)>,
    /// Global measurement record index -> (register index, bit index).
    record_map: Vec<(usize, usize)>,
}

/// Build a rotated surface-code Z-memory experiment of the given distance,
/// emitting the identical circuit as QASM and as a `TickCircuit`.
///
/// Mirrors `examples/surface/d3_fault_catalog_lookup.rs`: data qubits are
/// reset to |0>, each round prepares ancillas, runs a sequential
/// CX-per-check schedule (X checks via H-conjugated ancilla controls,
/// Z checks via data controls), and measures the ancillas; the experiment
/// ends with a transversal Z-basis data measurement.
fn build_surface_memory(distance: usize, rounds: usize) -> MemoryExperiment {
    let code = SurfaceCode::rotated(distance).expect("valid distance");
    let num_data = code.num_data_qubits();
    let nx = code.num_x_stabilizers();
    let nz = code.num_z_stabilizers();
    let num_qubits = num_data + nx + nz;
    let x_anc = |idx: usize| num_data + idx;
    let z_anc = |idx: usize| num_data + nx + idx;

    let mut tick = TickCircuit::new();
    let mut body = String::new();
    let mut registers: Vec<(String, usize)> = Vec::new();
    let mut record_map: Vec<(usize, usize)> = Vec::new();

    let data_qubits: Vec<usize> = (0..num_data).collect();
    let x_ancillas: Vec<usize> = (0..nx).map(x_anc).collect();
    let z_ancillas: Vec<usize> = (0..nz).map(z_anc).collect();

    tick.tick().pz(&data_qubits);
    for q in &data_qubits {
        writeln!(body, "reset q[{q}];").unwrap();
    }

    let mut x_round: Vec<Vec<TickMeasRef>> = Vec::with_capacity(rounds);
    let mut z_round: Vec<Vec<TickMeasRef>> = Vec::with_capacity(rounds);

    for round in 0..rounds {
        tick.tick().pz(&x_ancillas);
        tick.tick().pz(&z_ancillas);
        for q in x_ancillas.iter().chain(&z_ancillas) {
            writeln!(body, "reset q[{q}];").unwrap();
        }

        tick.tick().h(&x_ancillas);
        for q in &x_ancillas {
            writeln!(body, "h q[{q}];").unwrap();
        }

        for check in code.x_stabilizers() {
            let anc = x_anc(check.index);
            for data in check.qubits() {
                tick.tick().cx(&[(anc, data)]);
                writeln!(body, "cx q[{anc}],q[{data}];").unwrap();
            }
        }
        for check in code.z_stabilizers() {
            let anc = z_anc(check.index);
            for data in check.qubits() {
                tick.tick().cx(&[(data, anc)]);
                writeln!(body, "cx q[{data}],q[{anc}];").unwrap();
            }
        }

        tick.tick().h(&x_ancillas);
        for q in &x_ancillas {
            writeln!(body, "h q[{q}];").unwrap();
        }

        let reg_idx = registers.len();
        registers.push((format!("s{round}"), nx + nz));
        x_round.push(tick.tick().mz(&x_ancillas));
        for (bit, q) in x_ancillas.iter().enumerate() {
            writeln!(body, "measure q[{q}] -> s{round}[{bit}];").unwrap();
            record_map.push((reg_idx, bit));
        }
        z_round.push(tick.tick().mz(&z_ancillas));
        for (offset, q) in z_ancillas.iter().enumerate() {
            let bit = nx + offset;
            writeln!(body, "measure q[{q}] -> s{round}[{bit}];").unwrap();
            record_map.push((reg_idx, bit));
        }
    }

    let reg_idx = registers.len();
    registers.push(("f".to_string(), num_data));
    let final_data = tick.tick().mz(&data_qubits);
    for (bit, q) in data_qubits.iter().enumerate() {
        writeln!(body, "measure q[{q}] -> f[{bit}];").unwrap();
        record_map.push((reg_idx, bit));
    }

    let num_measurements = tick.num_measurements();
    assert_eq!(
        record_map.len(),
        num_measurements,
        "QASM measurement emission must track TickCircuit records one-to-one"
    );

    // Detector definitions, identical to the fault-catalog example:
    // first-round Z checks are deterministic for |0...0> initialization,
    // consecutive rounds compare like checks, and the final round compares
    // each Z check against the data measurements in its support.
    let mut detectors: Vec<Vec<i32>> = Vec::new();
    for &meas_ref in &z_round[0] {
        detectors.push(relative_records(num_measurements, &[meas_ref]));
    }
    for round in 1..rounds {
        for (&current, &previous) in x_round[round].iter().zip(&x_round[round - 1]) {
            detectors.push(relative_records(num_measurements, &[current, previous]));
        }
        for (&current, &previous) in z_round[round].iter().zip(&z_round[round - 1]) {
            detectors.push(relative_records(num_measurements, &[current, previous]));
        }
    }
    for check in code.z_stabilizers() {
        let mut refs = vec![z_round[rounds - 1][check.index]];
        refs.extend(check.qubits().into_iter().map(|q| final_data[q]));
        detectors.push(relative_records(num_measurements, &refs));
    }

    let logical_refs: Vec<TickMeasRef> = code
        .logical_z()
        .data_qubits
        .iter()
        .map(|&q| final_data[q])
        .collect();
    let observable = relative_records(num_measurements, &logical_refs);

    tick.set_meta(
        "num_measurements",
        Attribute::String(num_measurements.to_string()),
    );
    tick.set_meta("detectors", Attribute::String(records_json(&detectors)));
    tick.set_meta(
        "observables",
        Attribute::String(records_json(std::slice::from_ref(&observable))),
    );

    let mut qasm = String::new();
    writeln!(qasm, "OPENQASM 2.0;").unwrap();
    writeln!(qasm, "include \"qelib1.inc\";").unwrap();
    writeln!(qasm, "qreg q[{num_qubits}];").unwrap();
    for (name, width) in &registers {
        writeln!(qasm, "creg {name}[{width}];").unwrap();
    }
    qasm.push_str(&body);

    MemoryExperiment {
        qasm,
        tick,
        detectors,
        observable,
        num_measurements,
        registers,
        record_map,
    }
}

fn relative_records(num_measurements: usize, refs: &[TickMeasRef]) -> Vec<i32> {
    let num_measurements = i32::try_from(num_measurements).expect("measurement count fits in i32");
    refs.iter()
        .map(|m| i32::try_from(m.record_idx).expect("record index fits in i32") - num_measurements)
        .collect()
}

fn records_json(records: &[Vec<i32>]) -> String {
    let entries: Vec<String> = records
        .iter()
        .enumerate()
        .map(|(id, rs)| {
            let values = rs.iter().map(i32::to_string).collect::<Vec<_>>().join(",");
            format!(r#"{{"id":{id},"records":[{values}]}}"#)
        })
        .collect();
    format!("[{}]", entries.join(","))
}

/// Extract the flat measurement-record bits of one shot.
fn shot_record_bits(
    shot: &pecos_engines::shot_results::Shot,
    experiment: &MemoryExperiment,
) -> Vec<u8> {
    let register_bit = |reg: usize, bit: usize| -> u8 {
        let (name, _) = &experiment.registers[reg];
        let data = &shot.data[name.as_str()];
        match data {
            pecos_engines::shot_results::Data::BitVec(bv) => u8::from(bv[bit]),
            pecos_engines::shot_results::Data::U8(v) => u8::from((v >> bit) & 1 == 1),
            pecos_engines::shot_results::Data::U16(v) => u8::from((v >> bit) & 1 == 1),
            pecos_engines::shot_results::Data::U32(v) => u8::from((v >> bit) & 1 == 1),
            pecos_engines::shot_results::Data::U64(v) => u8::from((v >> bit) & 1 == 1),
            other => panic!("unexpected register data type for {name}: {other:?}"),
        }
    };

    experiment
        .record_map
        .iter()
        .map(|&(reg, bit)| register_bit(reg, bit))
        .collect()
}

/// XOR a relative-record definition over a shot's measurement bits.
fn xor_records(bits: &[u8], records: &[i32], num_measurements: usize) -> u8 {
    records.iter().fold(0u8, |acc, &rec| {
        let idx = i64::try_from(num_measurements).unwrap() + i64::from(rec);
        let idx = usize::try_from(idx).expect("record index in range");
        acc ^ bits[idx]
    })
}

/// Convert a `ShotVec` into per-shot detector syndromes and observable masks.
fn shots_to_syndromes(
    results: &ShotVec,
    experiment: &MemoryExperiment,
) -> (Vec<Vec<u8>>, Vec<u64>) {
    let mut syndromes = Vec::with_capacity(results.shots.len());
    let mut masks = Vec::with_capacity(results.shots.len());
    for shot in &results.shots {
        let bits = shot_record_bits(shot, experiment);
        let syndrome: Vec<u8> = experiment
            .detectors
            .iter()
            .map(|records| xor_records(&bits, records, experiment.num_measurements))
            .collect();
        let mask = u64::from(xor_records(
            &bits,
            &experiment.observable,
            experiment.num_measurements,
        ));
        syndromes.push(syndrome);
        masks.push(mask);
    }
    (syndromes, masks)
}

/// Uniform circuit-level depolarizing noise for the engines/neo mapping.
fn depolarizing_noise(p: f64) -> pecos_engines::noise::DepolarizingNoiseModelBuilder {
    pecos_engines::noise::DepolarizingNoiseModel::builder()
        .with_prep_probability(p)
        .with_meas_probability(p)
        .with_p1_probability(p)
        .with_p2_probability(p)
}

/// Run the experiment on one stack and return its `ShotVec`.
fn run_stack(
    experiment: &MemoryExperiment,
    stack: SimStack,
    p: f64,
    shots: usize,
    seed: u64,
) -> ShotVec {
    sim(Qasm::from_string(&experiment.qasm))
        .stack(stack)
        .noise(depolarizing_noise(p))
        .seed(seed)
        .workers(4)
        .shots(shots)
        .run()
        .expect("simulation run")
}

/// Decode both stacks' samples with one MWPM decoder over the same DEM,
/// returning (engines errors, neo errors).
fn decode_logical_errors(
    experiment: &MemoryExperiment,
    p: f64,
    engines: &ShotVec,
    neo: &ShotVec,
) -> (u64, u64) {
    let dem = DemBuilder::try_from_tick_circuit(&experiment.tick, p, p, p, p)
        .expect("DEM from tick circuit")
        .to_string_decomposed();
    let mut decoder = FusionBlossomDecoder::from_dem(&dem).expect("decoder from DEM");

    let mut count = |results: &ShotVec| -> u64 {
        let (syndromes, masks) = shots_to_syndromes(results, experiment);
        let num_detectors = experiment.detectors.len();
        let flat: Vec<u8> = syndromes.concat();
        let predicted = decoder
            .decode_batch_to_observables(&flat, masks.len(), num_detectors)
            .expect("batch decode");
        predicted
            .iter()
            .zip(&masks)
            .filter(|(pred, actual)| pred != actual)
            .count() as u64
    };

    (count(engines), count(neo))
}

#[test]
fn noiseless_surface_memory_is_silent_on_both_stacks() {
    // Validates the generator end-to-end on each stack independently: the
    // X-ancilla outcomes are individually random, so every detector and
    // the logical observable XOR to zero only if the QASM, the record
    // bookkeeping, and the register bit mapping all line up.
    let experiment = build_surface_memory(3, 3);

    for stack in [SimStack::Engines, SimStack::Neo] {
        let results = sim(Qasm::from_string(&experiment.qasm))
            .stack(stack)
            .seed(11)
            .shots(25)
            .run()
            .expect("noiseless run");
        let (syndromes, masks) = shots_to_syndromes(&results, &experiment);
        for (shot_idx, syndrome) in syndromes.iter().enumerate() {
            assert!(
                syndrome.iter().all(|&bit| bit == 0),
                "stack {stack:?} shot {shot_idx}: noiseless detectors must be silent, got {syndrome:?}"
            );
        }
        assert!(
            masks.iter().all(|&m| m == 0),
            "stack {stack:?}: noiseless logical observable must be trivial"
        );
    }
}

#[test]
fn surface_memory_ler_matches_across_stacks() {
    // V5: d=3 and d=5 Z-memory under uniform depolarizing noise. Both
    // stacks' LERs must have overlapping Jeffreys intervals, and each
    // stack must show d=5 suppressing the LER below d=3.
    //
    // Calibration (20k shots, this circuit's sequential schedule):
    // threshold sits near p = 0.004; at p = 0.003 the LERs are roughly
    // 4.1e-3 (d=3) and 1.4e-3 (d=5) with ~3x suppression on both stacks.
    // The sequential schedule's hook errors limit the suppression
    // steepness; that affects both stacks identically.
    // 20k shots: at 10k the per-stack error counts (~20-40) fluctuate too
    // much for the suppression margin to be decisive.
    let p = 0.003;
    let shots = 20_000;
    // High-confidence intervals so stack disagreement, not sampling
    // noise, is what fails the equivalence check (~4.4 sigma per side).
    //
    // Sensitivity, stated honestly: at these counts the overlap criterion
    // only fails for LER ratios beyond roughly 2.4x at d=3, so this test
    // is a coarse end-to-end guard — its value is exercising a real QEC
    // circuit with decoding through both stacks. Fine-grained discrepancy
    // detection belongs to the V1 matrix's analytic-anchor cells (power
    // ~1 against convention bugs like the 2/3 and 8/15 factors or the
    // measurement state-flip/record-flip distinction). The seed-42 d=3
    // draw (engines 85 vs neo 60, ~2.1 sigma) was settled as sampling
    // noise by an independent 6-seed 120k-shot-per-stack run (engines
    // 517 vs neo 482, z = 1.11).
    let equivalence_confidence = 0.99999;
    // The suppression margin is smaller than the equivalence margin, so
    // it gets its own (still strict) confidence. Pooling the two stacks
    // for suppression is justified by that 120k-shot equivalence run,
    // not by this test's own (weaker) overlap check.
    let suppression_confidence = 0.99;

    let mut pooled_intervals = Vec::new();
    for distance in [3, 5] {
        let experiment = build_surface_memory(distance, distance);
        let engines = run_stack(&experiment, SimStack::Engines, p, shots, 42);
        let neo = run_stack(&experiment, SimStack::Neo, p, shots, 42);
        let (engines_errors, neo_errors) = decode_logical_errors(&experiment, p, &engines, &neo);

        let engines_ci = jeffreys_interval(engines_errors, shots as u64, equivalence_confidence);
        let neo_ci = jeffreys_interval(neo_errors, shots as u64, equivalence_confidence);
        println!(
            "d={distance}: engines {engines_errors}/{shots} LER CI [{:.5}, {:.5}], \
             neo {neo_errors}/{shots} LER CI [{:.5}, {:.5}]",
            engines_ci.0, engines_ci.1, neo_ci.0, neo_ci.1
        );

        assert!(
            engines_ci.0 <= neo_ci.1 && neo_ci.0 <= engines_ci.1,
            "d={distance}: stack LERs are statistically incompatible: \
             engines {engines_errors}/{shots} vs neo {neo_errors}/{shots}"
        );
        // With per-stack equivalence established, pool the stacks for the
        // suppression physics check (doubles the statistics).
        pooled_intervals.push(jeffreys_interval(
            engines_errors + neo_errors,
            2 * shots as u64,
            suppression_confidence,
        ));
    }

    // Error suppression: the pooled d=5 interval must sit strictly below
    // the pooled d=3 interval (p = 0.003 is below threshold).
    let d3 = pooled_intervals[0];
    let d5 = pooled_intervals[1];
    assert!(
        d5.1 < d3.0,
        "d=5 LER must be suppressed below d=3: \
         d5 CI [{:.5}, {:.5}] vs d3 CI [{:.5}, {:.5}]",
        d5.0,
        d5.1,
        d3.0,
        d3.1
    );
}
