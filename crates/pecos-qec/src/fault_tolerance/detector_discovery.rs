// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0

//! Code-agnostic detector discovery from latest deterministic tick suffixes.

use std::collections::BTreeMap;

use pecos_core::BitSet;
use pecos_quantum::TickCircuit;
use pecos_simulators::SymbolicSparseStab;
use pecos_simulators::symbolic_sparse_stab::{MeasurementHistory, SymbolicMeasurementResult};

use super::fault_sampler::{
    MeasurementHistoryError, measurement_history_num_qubits, replay_measurement_history,
    symbolic_measurement_history,
};

/// Record parities and their noiseless reference values.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DiscoveredDetectors {
    /// Ascending absolute records and reference, ordered by newest record.
    pub detectors: Vec<(Vec<usize>, bool)>,
    /// Completed observables, with ascending absolute records, in input order.
    pub observables: Vec<(Vec<usize>, bool)>,
}

/// Invalid observable declarations or a circuit that cannot be replayed.
#[derive(Debug, thiserror::Error)]
pub enum DetectorDiscoveryError {
    /// The circuit cannot be replayed symbolically (unsupported gate or a
    /// record that depends on a hidden `MeasureLeaked` outcome).
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
    /// Completing this parity would cancel one of its declared records.
    #[error(
        "observable {observable} is non-deterministic: declared record {record} is a random source of its own parity"
    )]
    NonDeterministicObservable { observable: usize, record: usize },
}

/// Discover detector parities for a Clifford tick circuit, without code metadata.
///
/// `observables[i]` is the set of absolute terminal readout records for observable
/// `i`, in the same record order as [`symbolic_measurement_history`]. Duplicate
/// entries are treated as one record. Observables are completed using the full
/// history's random sources; their newest independent pivots exclude detector
/// relations. **A deterministic logical parity that is not declared becomes a
/// detector.** No detector is added to a completed observable.
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
/// or non-deterministic observable, or a shared [`MeasurementHistoryError`] if
/// replay cannot represent the circuit. A suffix can introduce an unrecorded
/// `MeasureLeaked` dependency even when the full history is representable; this
/// also returns the shared history error.
pub fn discover_detectors(
    tc: &TickCircuit,
    observables: &[Vec<usize>],
) -> Result<DiscoveredDetectors, DetectorDiscoveryError> {
    let full = symbolic_measurement_history(tc)?;
    let completed = complete_observables(&full, observables)?;
    let excluded = observable_pivots(&full, &completed);
    let num_qubits = measurement_history_num_qubits(tc);
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
    let mut detectors = Vec::new();
    for result in full.iter() {
        if !result.is_deterministic || excluded.contains(result.index) {
            continue;
        }
        let j = result.index;
        let (start, suffix) = latest_suffix(tc, num_qubits, j, record_ticks[j], &bases)?;
        let history = suffix.as_ref().unwrap_or(&full);
        let base = bases[start];
        detectors.push(canonical_relation(history, j - base, base));
    }
    Ok(DiscoveredDetectors {
        detectors,
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
            let mut declared: BitSet = records.iter().copied().collect();
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

fn observable_pivots(history: &MeasurementHistory, completed: &[(Vec<usize>, bool)]) -> BitSet {
    let mut basis: BTreeMap<usize, BitSet> = BTreeMap::new();
    for (records, _) in completed {
        let mut vector: BitSet = records.iter().copied().collect();
        while let Some(pivot) = vector.iter().last() {
            if let Some(row) = basis.get(&pivot) {
                vector ^= row;
            } else {
                history
                    .get(pivot)
                    .filter(|result| result.is_deterministic)
                    .expect("a completed observable's newest echelon pivot is a deterministic measurement");
                basis.insert(pivot, vector);
                break;
            }
        }
    }
    basis.keys().copied().collect()
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
