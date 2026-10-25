// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0

//! Code-agnostic detector discovery from latest deterministic tick suffixes.

use std::cmp::Reverse;
use std::collections::BTreeMap;

use pecos_core::BitSet;
use pecos_core::gate_type::GateType;
use pecos_quantum::TickCircuit;
use pecos_simulators::SymbolicSparseStab;
use pecos_simulators::symbolic_sparse_stab::{MeasurementHistory, SymbolicMeasurementResult};

use super::fault_sampler::{
    FaultCatalog, FaultCatalogError, FaultLocation, MeasurementHistoryError,
    measurement_history_num_qubits, record_effect_fault_catalog, replay_measurement_history,
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
    /// The structural single-fault catalog, used to choose replaced detectors,
    /// cannot be built for this circuit.
    #[error("{0}")]
    FaultCatalog(#[from] FaultCatalogError),
    /// No joint choice of replaced detectors exists: every detector this
    /// observable could replace is the only kept detector that some single fault
    /// on a declared observable flips.
    #[error(
        "observable {observable} cannot replace any detector it is built from without leaving a \
         single fault undetected: a fault on {gate_type:?} at tick {tick}, qubits {qubits:?}, flips \
         a declared observable and, of the kept detectors, only {detector:?}. Either no check \
         protects the observable against this fault, its protecting checks are not relations of \
         single deterministic records (for example, no check round before the readout), or it \
         differs from another declared observable by a check; add such a check round, declare \
         independent observables, or supply detectors by hand"
    )]
    UndetectableObservableFault {
        observable: usize,
        tick: usize,
        gate_type: GateType,
        qubits: Vec<usize>,
        detector: Vec<usize>,
    },
}

/// Discover detector parities for a Clifford tick circuit, without code metadata.
///
/// `observables[i]` is the set of absolute terminal readout records for observable
/// `i`, in the same record order as [`symbolic_measurement_history`]. Observables
/// are completed using the full history's random sources. Each independent
/// observable replaces one detector of its expansion over the detectors. The
/// replaced detectors are chosen jointly so that every structural single fault
/// that flips a declared observable still flips a kept detector; each observable
/// prefers, in declaration order, the longest window, then the newest record. **A deterministic logical parity that is not declared
/// becomes a detector.** No detector is added to a completed observable.
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
/// duplicated, or non-deterministic observable, or one that cannot replace a
/// detector without leaving a single fault undetected; a shared
/// [`MeasurementHistoryError`] if replay cannot represent the circuit; or a
/// [`FaultCatalogError`] if observables are declared and the single-fault
/// catalog cannot be built.
pub fn discover_detectors(
    tc: &TickCircuit,
    observables: &[Vec<usize>],
) -> Result<DiscoveredDetectors, DetectorDiscoveryError> {
    let full = symbolic_measurement_history(tc)?;
    let completed = complete_observables(&full, observables)?;
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
    // Newest record -> (records, reference, ticks spanned by the window).
    let mut detectors: BTreeMap<usize, (Vec<usize>, bool, usize)> = BTreeMap::new();
    for result in full.iter().filter(|result| result.is_deterministic) {
        let j = result.index;
        let end = record_ticks[j];
        let (start, suffix) = latest_suffix(tc, num_qubits, j, end, &bases)?;
        let history = suffix.as_ref().unwrap_or(&full);
        let base = bases[start];
        let (records, reference) = canonical_relation(history, j - base, base);
        detectors.insert(j, (records, reference, end - start));
    }
    for j in replaced_detectors(tc, &detectors, &completed)? {
        detectors.remove(&j);
    }
    Ok(DiscoveredDetectors {
        detectors: detectors
            .into_values()
            .map(|(records, reference, _)| (records, reference))
            .collect(),
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

/// Choose the detector each independent observable replaces, so that no
/// observable lies in the span of the kept detectors.
///
/// Which member of an observable's expansion is the logical one is not
/// determined by the records alone, so the choice is checked against the
/// circuit's structural single faults: a detector that is the only kept detector
/// flipped by some fault on a declared observable must stay. The joint choice is
/// searched depth-first in declaration order, trying each observable's
/// candidates least local first (longest window, then newest record), so the
/// first choice is kept whenever it leads to a full assignment.
fn replaced_detectors(
    tc: &TickCircuit,
    detectors: &BTreeMap<usize, (Vec<usize>, bool, usize)>,
    completed: &[(Vec<usize>, bool)],
) -> Result<Vec<usize>, DetectorDiscoveryError> {
    if completed.is_empty() {
        return Ok(Vec::new());
    }
    let catalog = record_effect_fault_catalog(tc)?;
    let exposures = observable_fault_exposures(&catalog, detectors, completed);
    let expansions: Vec<BitSet> = completed
        .iter()
        .map(|(records, _)| expansion_over_detectors(detectors, records))
        .collect();
    let mut chosen = Vec::new();
    let mut refusal = None;
    if choose_replacements(
        0,
        &expansions,
        &exposures,
        detectors,
        &mut chosen,
        &mut refusal,
    ) {
        Ok(chosen.into_iter().map(|(detector, _)| detector).collect())
    } else {
        Err(refusal.expect("a failed search records the dead end that ended it"))
    }
}

/// Every detector ends at a distinct newest record, so reducing a deterministic
/// parity by newest record expands it exactly over the detectors.
fn expansion_over_detectors(
    detectors: &BTreeMap<usize, (Vec<usize>, bool, usize)>,
    records: &[usize],
) -> BitSet {
    let mut vector: BitSet = records.iter().copied().collect();
    let mut expansion = BitSet::new();
    while let Some(newest) = vector.iter().last() {
        let (relation, _, _) = detectors.get(&newest).expect(
            "a deterministic parity's newest record is deterministic, so a detector ends there",
        );
        for &record in relation {
            vector.toggle(record);
        }
        expansion.insert(newest);
    }
    expansion
}

/// Assign replaced detectors to observables `observable..`, given `chosen`
/// (each earlier replaced detector with its observable's reduced expansion).
/// Earlier observables stand in for the detectors they replaced. Records the
/// first dead end in `refusal` and returns whether a full assignment exists.
fn choose_replacements(
    observable: usize,
    expansions: &[BitSet],
    exposures: &[(&FaultLocation, BitSet)],
    detectors: &BTreeMap<usize, (Vec<usize>, bool, usize)>,
    chosen: &mut Vec<(usize, BitSet)>,
    refusal: &mut Option<DetectorDiscoveryError>,
) -> bool {
    let Some(original) = expansions.get(observable) else {
        return true;
    };
    let mut expansion = original.clone();
    for (detector, earlier) in chosen.iter() {
        if expansion.contains(*detector) {
            expansion ^= earlier;
        }
    }
    if expansion.is_empty() {
        // Dependent on earlier observables: nothing to replace.
        return choose_replacements(
            observable + 1,
            expansions,
            exposures,
            detectors,
            chosen,
            refusal,
        );
    }
    let replaced: BitSet = chosen.iter().map(|(detector, _)| *detector).collect();
    // A fault seen by exactly one kept detector pins that detector.
    let mut pinned: BTreeMap<usize, &FaultLocation> = BTreeMap::new();
    for (location, flipped) in exposures {
        let mut kept = flipped.iter().filter(|&k| !replaced.contains(k));
        if let (Some(j), None) = (kept.next(), kept.next()) {
            pinned.entry(j).or_insert(location);
        }
    }
    let rank = |&j: &usize| (detectors[&j].2, j);
    let mut candidates: Vec<usize> = expansion
        .iter()
        .filter(|j| !pinned.contains_key(j))
        .collect();
    if candidates.is_empty() {
        if refusal.is_none() {
            let best = expansion
                .iter()
                .max_by_key(rank)
                .expect("the expansion is not empty");
            let location = pinned[&best];
            *refusal = Some(DetectorDiscoveryError::UndetectableObservableFault {
                observable,
                tick: location.tick,
                gate_type: location.gate_type,
                qubits: location.qubits.clone(),
                detector: detectors[&best].0.clone(),
            });
        }
        return false;
    }
    candidates.sort_by_key(|j| Reverse(rank(j)));
    for detector in candidates {
        chosen.push((detector, expansion.clone()));
        if choose_replacements(
            observable + 1,
            expansions,
            exposures,
            detectors,
            chosen,
            refusal,
        ) {
            return true;
        }
        chosen.pop();
    }
    false
}

/// Detectors flipped by each structural single fault that flips an observable.
fn observable_fault_exposures<'a>(
    catalog: &'a FaultCatalog,
    detectors: &BTreeMap<usize, (Vec<usize>, bool, usize)>,
    completed: &[(Vec<usize>, bool)],
) -> Vec<(&'a FaultLocation, BitSet)> {
    let mut record_detectors: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (&j, (records, _, _)) in detectors {
        for &record in records {
            record_detectors.entry(record).or_default().push(j);
        }
    }
    let mut record_observables: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (observable, (records, _)) in completed.iter().enumerate() {
        for &record in records {
            record_observables
                .entry(record)
                .or_default()
                .push(observable);
        }
    }
    let toggled = |index: &BTreeMap<usize, Vec<usize>>, flipped: &[usize]| {
        let mut set = BitSet::new();
        for record in flipped {
            for &member in index.get(record).into_iter().flatten() {
                set.toggle(member);
            }
        }
        set
    };
    let mut exposures = Vec::new();
    for location in &catalog.locations {
        for fault in &location.faults {
            if !toggled(&record_observables, &fault.affected_measurements).is_empty() {
                exposures.push((
                    location,
                    toggled(&record_detectors, &fault.affected_measurements),
                ));
            }
        }
    }
    exposures
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
