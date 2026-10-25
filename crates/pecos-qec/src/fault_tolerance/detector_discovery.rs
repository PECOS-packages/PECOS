// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0

//! Code-agnostic detector discovery from latest deterministic tick suffixes.

use std::collections::BTreeMap;

use pecos_core::{BitSet, Pauli, PauliString};
use pecos_quantum::TickCircuit;
use pecos_simulators::symbolic_sparse_stab::{MeasurementHistory, SymbolicMeasurementResult};
use pecos_simulators::{BitmaskPauliProp, SymbolicSparseStab};

use super::fault_sampler::{
    MeasurementHistoryError, flatten_tick_circuit, measurement_history_num_qubits,
    propagate_forward, replay_measurement_history, symbolic_measurement_history,
};

/// Record parities and their noiseless reference values.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DiscoveredDetectors {
    /// Ascending absolute records and reference, ordered by newest record after cleaning.
    pub detectors: Vec<(Vec<usize>, bool)>,
    /// Completed observables, with ascending absolute records, in input order.
    pub observables: Vec<(Vec<usize>, bool)>,
}

/// A conjugate logical applied just before `before_tick` (1 means after tick 0).
/// Its phase is ignored; only its Pauli support affects record flips.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Conjugate {
    /// Tick boundary at which the operator is injected, less than `num_ticks`.
    pub before_tick: usize,
    /// Logical operator conjugate to the corresponding declared observable.
    pub pauli: PauliString,
}

/// Invalid observable declarations or a circuit that cannot be replayed.
#[derive(Debug, thiserror::Error)]
pub enum DetectorDiscoveryError {
    /// The circuit cannot be replayed symbolically (an unsupported gate).
    #[error("{0}")]
    MeasurementHistory(#[from] MeasurementHistoryError),
    /// An observable must declare at least one terminal record.
    #[error("observable {observable} is empty; supply at least one terminal record")]
    EmptyObservable { observable: usize },
    /// Absolute record indices must lie within the full history.
    #[error("observable {observable} record {record} is out of range for {num_records} records")]
    RecordOutOfRange {
        observable: usize,
        record: usize,
        num_records: usize,
    },
    /// A record listed twice is ambiguous: as a parity it would cancel.
    #[error("observable {observable} lists record {record} more than once; list each record once")]
    DuplicateRecord { observable: usize, record: usize },
    /// Completing this parity would cancel one of its declared records.
    #[error(
        "observable {observable} is non-deterministic: declared record {record} is a random source of its own parity"
    )]
    NonDeterministicObservable { observable: usize, record: usize },
    /// Every observable needs exactly one conjugate.
    #[error(
        "{observables} observables but {conjugates} conjugates; supply one conjugate per observable"
    )]
    ConjugateCountMismatch {
        observables: usize,
        conjugates: usize,
    },
    /// Injection must precede a tick in the circuit.
    #[error(
        "conjugate {conjugate} before_tick {before_tick} is out of range for {num_ticks} ticks; choose a boundary less than {num_ticks}"
    )]
    ConjugateTickOutOfRange {
        conjugate: usize,
        before_tick: usize,
        num_ticks: usize,
    },
    /// Pauli support must be within the circuit's qubit register.
    #[error(
        "conjugate {conjugate} qubit {qubit} is out of range for {num_qubits} qubits; use qubits less than {num_qubits}"
    )]
    ConjugateQubitOutOfRange {
        conjugate: usize,
        qubit: usize,
        num_qubits: usize,
    },
    /// A conjugate must flip precisely its own completed observable.
    #[error(
        "conjugate {conjugate} flips observables {flipped_observables:?}; supply a logical conjugate that flips only observable {conjugate}"
    )]
    ConjugatePairing {
        conjugate: usize,
        flipped_observables: Vec<usize>,
    },
}

/// Discover detector parities for a Clifford tick circuit, without code metadata.
///
/// `observables[i]` is the set of absolute terminal readout records for observable
/// `i`, in the same record order as [`symbolic_measurement_history`]. Observables
/// are completed using the full history's random sources. `conjugates[i]` is
/// propagated from its tick boundary and must flip only completed observable i.
/// This pairing check also rejects dependent observables. Each canonical detector
/// is cleaned by taking the XOR with the completed observables whose conjugates flip it,
/// including their references. Independent cleaned detectors are retained, giving
/// priority to unchanged canonical detectors to preserve locality.
/// **Undeclared deterministic logical parities still become detectors.**
///
/// # Preconditions
///
/// Each conjugate must be a genuine logical operator: it commutes with every
/// check the circuit measures and anticommutes with its own observable only.
/// Records alone cannot identify logical content or verify this precondition.
/// A wrong conjugate can pass pairing and silently lower the fault distance.
/// The Pauli phase is ignored. With no observables, supply no conjugates; all
/// canonical detectors are returned.
///
/// Full-history determinism uses the real implicit |0^n> initial state. Each
/// detector uses the latest tick boundary at which its newest record remains
/// deterministic under a Bell-purified unknown input, then newest-first
/// elimination within that suffix. Boundary zero always means the real initial
/// state. References include the constants of every substituted member.
///
/// # Errors
///
/// Returns an actionable [`DetectorDiscoveryError`] for an empty, out-of-range,
/// duplicated, or non-deterministic observable; then a conjugate count mismatch,
/// out-of-range tick, out-of-range qubit, or invalid pairing, in that order.
/// Returns a shared [`MeasurementHistoryError`] if replay cannot represent the circuit.
pub fn discover_detectors(
    tc: &TickCircuit,
    observables: &[Vec<usize>],
    conjugates: &[Conjugate],
) -> Result<DiscoveredDetectors, DetectorDiscoveryError> {
    let full = symbolic_measurement_history(tc)?;
    let completed = complete_observables(&full, observables)?;
    let num_qubits = measurement_history_num_qubits(tc);
    let flips = conjugate_record_flips(tc, conjugates, &completed, num_qubits)?;
    let mut bases = Vec::with_capacity(tc.num_ticks());
    let mut record_ticks = Vec::with_capacity(full.len());
    for (tick_idx, tick) in tc.iter_ticks() {
        bases.push(record_ticks.len());
        for gate in tick.iter_gate_batches() {
            if gate.gate_type.consumes_measurement_record() {
                record_ticks.extend(std::iter::repeat_n(tick_idx, gate.qubits.len()));
            }
        }
    }
    debug_assert_eq!(
        record_ticks.len(),
        full.len(),
        "record-consuming gates and replayed record columns must agree"
    );
    // Canonical order is ascending newest record before cleaning.
    let mut detectors = Vec::new();
    for result in full.iter().filter(|result| result.is_deterministic) {
        let j = result.index;
        let end = record_ticks[j];
        let (start, suffix) = latest_suffix(tc, num_qubits, j, end, &bases)?;
        let history = suffix.as_ref().unwrap_or(&full);
        let base = bases[start];
        let (records, reference) = canonical_relation(history, j - base, base);
        detectors.push((records, reference));
    }
    Ok(DiscoveredDetectors {
        detectors: clean_detectors(detectors, &completed, &flips),
        observables: completed,
    })
}

fn random_vector(result: &SymbolicMeasurementResult) -> BitSet {
    if result.is_deterministic {
        result.outcome.clone()
    } else {
        BitSet::single(result.index)
    }
}

fn complete_observables(
    history: &MeasurementHistory,
    observables: &[Vec<usize>],
) -> Result<Vec<(Vec<usize>, bool)>, DetectorDiscoveryError> {
    observables
        .iter()
        .enumerate()
        .map(|(observable, records)| {
            if records.is_empty() {
                return Err(DetectorDiscoveryError::EmptyObservable { observable });
            }
            // Validate before allocating a bitset indexed by user input.
            for &record in records {
                if record >= history.len() {
                    return Err(DetectorDiscoveryError::RecordOutOfRange {
                        observable,
                        record,
                        num_records: history.len(),
                    });
                }
            }
            let mut declared = BitSet::new();
            for &record in records {
                if declared.contains(record) {
                    return Err(DetectorDiscoveryError::DuplicateRecord { observable, record });
                }
                declared.insert(record);
            }
            let mut sources = BitSet::new();
            let mut reference = false;
            for r in &declared {
                let result = &history.as_slice()[r];
                sources ^= &random_vector(result);
                reference ^= result.is_deterministic && result.flip;
            }
            if let Some(record) = sources.iter().find(|&r| declared.contains(r)) {
                return Err(DetectorDiscoveryError::NonDeterministicObservable {
                    observable,
                    record,
                });
            }
            declared ^= &sources; // Disjoint, so this is R union S.
            Ok((declared.iter().collect(), reference))
        })
        .collect()
}

fn conjugate_record_flips(
    tc: &TickCircuit,
    conjugates: &[Conjugate],
    completed: &[(Vec<usize>, bool)],
    num_qubits: usize,
) -> Result<Vec<BitSet>, DetectorDiscoveryError> {
    if completed.len() != conjugates.len() {
        return Err(DetectorDiscoveryError::ConjugateCountMismatch {
            observables: completed.len(),
            conjugates: conjugates.len(),
        });
    }
    for (conjugate, operator) in conjugates.iter().enumerate() {
        if operator.before_tick >= tc.num_ticks() {
            return Err(DetectorDiscoveryError::ConjugateTickOutOfRange {
                conjugate,
                before_tick: operator.before_tick,
                num_ticks: tc.num_ticks(),
            });
        }
    }
    for (conjugate, operator) in conjugates.iter().enumerate() {
        for (_, qubit) in operator.pauli.iter_pairs() {
            if qubit.index() >= num_qubits {
                return Err(DetectorDiscoveryError::ConjugateQubitOutOfRange {
                    conjugate,
                    qubit: qubit.index(),
                    num_qubits,
                });
            }
        }
    }
    let (gates, measurements) = flatten_tick_circuit(tc);
    conjugates
        .iter()
        .enumerate()
        .map(|(conjugate, operator)| {
            let mut prop = BitmaskPauliProp::new();
            for (pauli, qubit) in operator.pauli.iter_pairs() {
                match pauli {
                    Pauli::X => prop.track_x(&[qubit.index()]),
                    Pauli::Y => prop.track_y(&[qubit.index()]),
                    Pauli::Z => prop.track_z(&[qubit.index()]),
                    Pauli::I => {}
                }
            }
            let start = gates.partition_point(|gate| gate.tick < operator.before_tick);
            let flipped: BitSet = propagate_forward(&mut prop, start, &gates, &measurements)
                .into_iter()
                .collect();
            let flipped_observables: Vec<_> = completed
                .iter()
                .enumerate()
                .filter_map(|(i, (records, _))| flips_parity(&flipped, records).then_some(i))
                .collect();
            if flipped_observables != [conjugate] {
                return Err(DetectorDiscoveryError::ConjugatePairing {
                    conjugate,
                    flipped_observables,
                });
            }
            Ok(flipped)
        })
        .collect()
}

fn flips_parity(flipped: &BitSet, records: &[usize]) -> bool {
    records
        .iter()
        .fold(false, |parity, &record| parity ^ flipped.contains(record))
}

fn clean_detectors(
    detectors: Vec<(Vec<usize>, bool)>,
    completed: &[(Vec<usize>, bool)],
    flips: &[BitSet],
) -> Vec<(Vec<usize>, bool)> {
    let mut unchanged = Vec::new();
    let mut cleaned = Vec::new();
    for (records, mut reference) in detectors {
        let mut vector: BitSet = records.iter().copied().collect();
        let mut changed = false;
        for (flipped, (observable, obs_reference)) in flips.iter().zip(completed) {
            if flips_parity(flipped, &records) {
                changed = true;
                for &record in observable {
                    vector.toggle(record);
                }
                reference ^= obs_reference;
            }
        }
        if changed {
            cleaned.push((vector.iter().collect(), reference));
        } else {
            unchanged.push((records, reference));
        }
    }
    let mut basis: BTreeMap<usize, BitSet> = BTreeMap::new();
    let mut kept = Vec::new();
    // Seed with every unflipped canonical detector before considering cleaned ones.
    for (records, reference) in unchanged.into_iter().chain(cleaned) {
        let mut vector: BitSet = records.iter().copied().collect();
        while let Some(pivot) = vector.iter().next() {
            if let Some(row) = basis.get(&pivot) {
                vector ^= row;
            } else {
                basis.insert(pivot, vector);
                kept.push((records, reference));
                break;
            }
        }
    }
    kept.sort_by_key(|(records, _)| records.last().copied());
    kept
}

/// Search backwards exponentially, then bisect the deterministic/unknown bracket.
/// Retain the successful history so extraction needs no additional replay.
fn latest_suffix(
    tc: &TickCircuit,
    num_qubits: usize,
    record: usize,
    end: usize,
    bases: &[usize],
) -> Result<(usize, Option<MeasurementHistory>), MeasurementHistoryError> {
    let probe = |start: usize| -> Result<MeasurementHistory, MeasurementHistoryError> {
        let mut sim = SymbolicSparseStab::new(2 * num_qubits);
        for q in 0..num_qubits {
            sim.h(&[q]);
            sim.cx(&[(q, num_qubits + q)]);
        }
        replay_measurement_history(tc, start..end + 1, sim)
    };
    let mut lo = 0;
    let mut hi = end + 1;
    let mut best = None;
    let mut candidate = end;
    let mut distance = 1;
    while candidate > 0 {
        let history = probe(candidate)?;
        if history.as_slice()[record - bases[candidate]].is_deterministic {
            lo = candidate;
            best = Some(history);
            break;
        }
        hi = candidate;
        candidate = end.saturating_sub(distance);
        distance = distance.saturating_mul(2);
    }
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        let history = probe(mid)?;
        if history.as_slice()[record - bases[mid]].is_deterministic {
            lo = mid;
            best = Some(history);
        } else {
            hi = mid;
        }
    }
    Ok((lo, best))
}

fn canonical_relation(history: &MeasurementHistory, j: usize, base: usize) -> (Vec<usize>, bool) {
    let records = history.as_slice();
    let mut basis: BTreeMap<usize, (BitSet, BitSet)> = BTreeMap::new();
    for k in (0..j).rev() {
        let mut vector = random_vector(&records[k]);
        let mut combo = BitSet::single(k);
        while let Some(pivot) = vector.iter().next() {
            if let Some((row, members)) = basis.get(&pivot) {
                vector ^= row;
                combo ^= members;
            } else {
                basis.insert(pivot, (vector, combo));
                break;
            }
        }
    }
    let mut vector = records[j].outcome.clone();
    let mut combo = BitSet::single(j);
    while let Some(pivot) = vector.iter().next() {
        let (row, members) = basis
            .get(&pivot)
            .expect("a deterministic suffix measurement lies in the span of preceding records");
        vector ^= row;
        combo ^= members;
    }
    let reference = combo.iter().fold(false, |reference, k| {
        reference ^ (records[k].is_deterministic && records[k].flip)
    });
    (combo.iter().map(|k| base + k).collect(), reference)
}
