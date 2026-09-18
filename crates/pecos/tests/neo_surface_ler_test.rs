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
use pecos_decoder_core::obs_mask::ObsMask;
use pecos_engines::shot_results::ShotVec;
use pecos_fusion_blossom::FusionBlossomDecoder;
use pecos_num::jeffreys_interval;
use pecos_programs::Qasm;
use pecos_qec::SurfaceCode;
use pecos_qec::fault_tolerance::dem_builder::DemBuilder;
use pecos_quantum::{Attribute, TickCircuit, TickMeasRef};
use std::fmt::Write as _;
use std::sync::OnceLock;

/// A surface-code memory experiment in both program representations.
struct MemoryExperiment {
    qasm: String,
    tick: TickCircuit,
    /// Detector definitions as relative measurement records (Stim style:
    /// record -k is the k-th most recent measurement).
    detectors: Vec<Vec<usize>>,
    /// The logical-Z observable as stable measurement ids.
    observable: Vec<usize>,
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
    let mut detectors: Vec<Vec<usize>> = Vec::new();
    for &meas_ref in &z_round[0] {
        detectors.push(ref_meas_ids(&[meas_ref]));
    }
    for round in 1..rounds {
        for (&current, &previous) in x_round[round].iter().zip(&x_round[round - 1]) {
            detectors.push(ref_meas_ids(&[current, previous]));
        }
        for (&current, &previous) in z_round[round].iter().zip(&z_round[round - 1]) {
            detectors.push(ref_meas_ids(&[current, previous]));
        }
    }
    for check in code.z_stabilizers() {
        let mut refs = vec![z_round[rounds - 1][check.index]];
        refs.extend(check.qubits().into_iter().map(|q| final_data[q]));
        detectors.push(ref_meas_ids(&refs));
    }

    let logical_refs: Vec<TickMeasRef> = code
        .logical_z()
        .data_qubits
        .iter()
        .map(|&q| final_data[q])
        .collect();
    let observable = ref_meas_ids(&logical_refs);

    tick.set_meta(
        "num_measurements",
        Attribute::String(num_measurements.to_string()),
    );
    tick.set_meta("detectors", Attribute::String(meas_ids_json(&detectors)));
    tick.set_meta(
        "observables",
        Attribute::String(meas_ids_json(std::slice::from_ref(&observable))),
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
        registers,
        record_map,
    }
}

fn ref_meas_ids(refs: &[TickMeasRef]) -> Vec<usize> {
    refs.iter().map(|m| m.meas_id.index()).collect()
}

fn meas_ids_json(annotations: &[Vec<usize>]) -> String {
    let entries: Vec<String> = annotations
        .iter()
        .enumerate()
        .map(|(id, ids)| {
            let values = ids
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(",");
            format!(r#"{{"id":{id},"meas_ids":[{values}]}}"#)
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
fn xor_meas_bits(bits: &[u8], meas_ids: &[usize]) -> u8 {
    // Ids in this test are minted positionally by `mz()`, so they coincide
    // with the record order of the shot bits the engines produce.
    meas_ids.iter().fold(0u8, |acc, &id| acc ^ bits[id])
}

/// Convert a `ShotVec` into per-shot detector syndromes and observable masks.
fn shots_to_syndromes(
    results: &ShotVec,
    experiment: &MemoryExperiment,
) -> (Vec<Vec<u8>>, Vec<ObsMask>) {
    let mut syndromes = Vec::with_capacity(results.shots.len());
    let mut masks = Vec::with_capacity(results.shots.len());
    for shot in &results.shots {
        let bits = shot_record_bits(shot, experiment);
        let syndrome: Vec<u8> = experiment
            .detectors
            .iter()
            .map(|meas_ids| xor_meas_bits(&bits, meas_ids))
            .collect();
        let mask = ObsMask::from_u64(u64::from(xor_meas_bits(&bits, &experiment.observable)));
        syndromes.push(syndrome);
        masks.push(mask);
    }
    (syndromes, masks)
}

/// Uniform circuit-level depolarizing noise for the engines/neo mapping.
fn depolarizing_noise(p: f64) -> pecos_engines::noise::DepolarizingNoiseModelBuilder {
    pecos_engines::noise::DepolarizingNoiseModel::builder()
        .with_p_prep(p)
        .with_p_meas(p)
        .with_p1(p)
        .with_p2(p)
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
            masks.iter().all(|m| m.to_u64() == Some(0)),
            "stack {stack:?}: noiseless logical observable must be trivial"
        );
    }
}

/// Uniform depolarizing rate for the d=3 and d=5 memory experiments. The
/// threshold sits near p = 0.004 for this circuit's sequential schedule;
/// at p = 0.003 the LERs are roughly 4.1e-3 (d=3) and 1.4e-3 (d=5) with
/// ~3x suppression on both stacks. The sequential schedule's hook errors
/// limit the suppression steepness; that affects both stacks identically.
const P: f64 = 0.003;
/// 20k shots per stack and distance: at 10k the per-stack error counts
/// (~20-40) fluctuate too much for the suppression margin to be decisive.
const SHOTS: usize = 20_000;
const SEED: u64 = 42;

/// Logical error counts of one distance's memory experiment on both stacks.
struct LogicalErrors {
    engines: u64,
    neo: u64,
}

/// Simulate and decode the memory experiment at `distance`, once per test
/// binary. The two distances are the expensive part of this file, so each
/// has its own test (libtest runs them on separate threads) and the
/// suppression test reads both cached results instead of resampling.
fn logical_errors(distance: usize) -> &'static LogicalErrors {
    static D3: OnceLock<LogicalErrors> = OnceLock::new();
    static D5: OnceLock<LogicalErrors> = OnceLock::new();
    let slot = match distance {
        3 => &D3,
        5 => &D5,
        other => panic!("no cached run for distance {other}"),
    };
    slot.get_or_init(|| {
        let experiment = build_surface_memory(distance, distance);
        let engines = run_stack(&experiment, SimStack::Engines, P, SHOTS, SEED);
        let neo = run_stack(&experiment, SimStack::Neo, P, SHOTS, SEED);
        let (engines, neo) = decode_logical_errors(&experiment, P, &engines, &neo);
        LogicalErrors { engines, neo }
    })
}

/// V5 equivalence at one distance: both stacks' LERs must have overlapping
/// Jeffreys intervals.
fn assert_stacks_agree(distance: usize) {
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
    let alpha = 1.0 - 0.99999;
    let errors = logical_errors(distance);
    let shots = SHOTS as u64;
    let engines_ci = jeffreys_interval(errors.engines, shots, alpha)
        .expect("Jeffreys interval for engines errors k and shots n");
    let neo_ci = jeffreys_interval(errors.neo, shots, alpha)
        .expect("Jeffreys interval for neo errors k and shots n");
    println!(
        "d={distance}: engines {}/{shots} LER CI [{:.5}, {:.5}], \
         neo {}/{shots} LER CI [{:.5}, {:.5}]",
        errors.engines, engines_ci.lo, engines_ci.hi, errors.neo, neo_ci.lo, neo_ci.hi
    );
    assert!(
        engines_ci.lo <= neo_ci.hi && neo_ci.lo <= engines_ci.hi,
        "d={distance}: stack LERs are statistically incompatible: \
         engines {}/{shots} vs neo {}/{shots}",
        errors.engines,
        errors.neo
    );
}

#[test]
fn d3_ler_matches_across_stacks() {
    assert_stacks_agree(3);
}

#[test]
fn d5_ler_matches_across_stacks() {
    assert_stacks_agree(5);
}

#[test]
fn d5_suppresses_ler_below_d3() {
    // Error suppression: the pooled d=5 interval must sit strictly below
    // the pooled d=3 interval (p = 0.003 is below threshold). Pooling the
    // stacks doubles the statistics; that is justified by the independent
    // 120k-shot equivalence run described in `assert_stacks_agree`, not by
    // this file's (weaker) overlap checks. The suppression margin is
    // smaller than the equivalence margin, so it gets its own (still
    // strict) confidence.
    let alpha = 1.0 - 0.99;
    let pooled = |distance: usize| {
        let errors = logical_errors(distance);
        jeffreys_interval(errors.engines + errors.neo, 2 * SHOTS as u64, alpha)
            .expect("Jeffreys interval for pooled errors k and pooled shots n")
    };
    let d3 = pooled(3);
    let d5 = pooled(5);
    assert!(
        d5.hi < d3.lo,
        "d=5 LER must be suppressed below d=3: \
         d5 CI [{:.5}, {:.5}] vs d3 CI [{:.5}, {:.5}]",
        d5.lo,
        d5.hi,
        d3.lo,
        d3.hi
    );
}
