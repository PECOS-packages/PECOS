// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0.

use super::{Instruction, Operation, Program};
use pecos_core::{Angle64, MeasId, QubitId, gate_type::GateType};
use pecos_quantum::{AnnotationKind, Gate, TickCircuit, channel::pauli_mixture};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
};

#[derive(Clone, Copy)]
struct Batch<'a> {
    gate: &'a Gate,
    index: usize,
}
impl std::ops::Deref for Batch<'_> {
    type Target = Gate;
    fn deref(&self) -> &Gate {
        self.gate
    }
}
impl Batch<'_> {
    fn batch_index(self) -> usize {
        self.index
    }
}

/// Failure to compile a static circuit, before any shot takes place.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CompileError {
    /// Unsupported or malformed gate, with its original batch location.
    Gate {
        /// Offending gate kind.
        gate_type: GateType,
        /// Physical qubits in the batch.
        qubits: Vec<usize>,
        /// Zero-based tick index.
        tick: usize,
        /// Zero-based batch index within the tick.
        batch: usize,
        /// Explanation of rejection.
        reason: String,
    },
    /// A typed annotation refers to a record absent from the circuit.
    Annotation {
        /// Index in the circuit's annotation list.
        annotation: usize,
        /// Unresolved stable measurement ID.
        measurement: MeasId,
    },
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Gate {
                gate_type,
                qubits,
                tick,
                batch,
                reason,
            } => write!(
                f,
                "{gate_type:?} on {qubits:?} at tick {tick}, batch {batch}: {reason}"
            ),
            Self::Annotation {
                annotation,
                measurement,
            } => write!(
                f,
                "annotation {annotation} references unknown measurement {measurement:?}"
            ),
        }
    }
}
impl std::error::Error for CompileError {}

fn error(gate: Batch<'_>, tick: usize, reason: impl Into<String>) -> CompileError {
    CompileError::Gate {
        gate_type: gate.gate_type,
        qubits: gate.qubits.iter().map(QubitId::index).collect(),
        tick,
        batch: gate.batch_index(),
        reason: reason.into(),
    }
}

fn validate_gate(gate: Batch<'_>, tick: usize) -> Result<Operation, CompileError> {
    let kind = gate.gate_type;
    if matches!(
        kind,
        GateType::I | GateType::Idle | GateType::TrackedPauliMeta
    ) {
        return Ok(Operation::Gate(kind));
    }
    let stored: BTreeSet<_> = gate.qubits.iter().map(QubitId::index).collect();
    if kind == GateType::Channel {
        let channel = gate
            .channel
            .as_ref()
            .ok_or_else(|| error(gate, tick, "missing channel expression"))?;
        let alternatives = pauli_mixture(channel).map_err(|e| error(gate, tick, e.to_string()))?;
        if channel.qubits().into_iter().collect::<BTreeSet<_>>() != stored
            || stored.len() != gate.qubits.len()
            || !gate.angles.is_empty()
        {
            return Err(error(
                gate,
                tick,
                "channel support disagrees with stored qubits",
            ));
        }
        return Ok(Operation::Channel(alternatives));
    }
    match kind {
        GateType::X
        | GateType::Y
        | GateType::Z
        | GateType::SZ
        | GateType::SZdg
        | GateType::T
        | GateType::Tdg
        | GateType::CX
        | GateType::CZ
        | GateType::RZ
        | GateType::RZZ
        | GateType::SZZ
        | GateType::SZZdg
        | GateType::PZ
        | GateType::PX
        | GateType::MZ
        | GateType::MX
        | GateType::MPZ => {}
        _ => return Err(error(gate, tick, "unsupported gate")),
    }
    if gate.qubits.is_empty()
        || !gate.qubits.len().is_multiple_of(kind.quantum_arity())
        || gate.angles.len() != kind.angle_arity()
        || stored.len() != gate.qubits.len()
    {
        return Err(error(gate, tick, "invalid angle count or qubit batch"));
    }
    if kind.consumes_measurement_record() && gate.meas_ids.len() != gate.qubits.len() {
        return Err(error(
            gate,
            tick,
            "each recorded measurement needs one stable ID",
        ));
    }
    match kind {
        GateType::RZ | GateType::RZZ => {
            let power = (0_u8..8)
                .find(|&k| gate.angles[0] == Angle64::from_turns(f64::from(k) / 8.0))
                .ok_or_else(|| error(gate, tick, "angle must be an exact multiple of pi/4"))?;
            Ok(if kind == GateType::RZ {
                Operation::Phase(power)
            } else {
                Operation::ZZ(power)
            })
        }
        GateType::SZZ => Ok(Operation::ZZ(2)),
        GateType::SZZdg => Ok(Operation::ZZ(6)),
        _ => Ok(Operation::Gate(kind)),
    }
}

/// Compile a circuit, validating every batch and resolving all annotation IDs.
/// Record ordinals follow execution order, not stable-ID allocation order.
///
/// # Errors
/// Returns the original gate or annotation location on rejection.
pub fn compile(circuit: &TickCircuit) -> Result<Program, CompileError> {
    let mut ids = BTreeMap::new();
    let mut instructions = Vec::new();
    let mut num_qubits = 0;
    for (tick, row) in circuit.iter_ticks() {
        for gate in row.iter_gate_batches() {
            let gate = Batch {
                gate: gate.as_gate(),
                index: gate.batch_index(),
            };
            let operation = validate_gate(gate, tick)?;
            for q in &gate.qubits {
                num_qubits = num_qubits.max(
                    q.index()
                        .checked_add(1)
                        .ok_or_else(|| error(gate, tick, "qubit count overflows usize"))?,
                );
            }
            if gate.gate_type.consumes_measurement_record() {
                for &id in &gate.meas_ids {
                    let ordinal = ids.len();
                    if ids.insert(id, ordinal).is_some() {
                        return Err(error(gate, tick, "duplicate measurement ID"));
                    }
                }
            }
            instructions.push(Instruction {
                operation,
                qubits: gate.qubits.to_vec(),
                tick,
                batch: gate.batch_index(),
            });
        }
    }
    let mut detectors = Vec::new();
    let mut observables = Vec::new();
    for (annotation, item) in circuit.annotations().iter().enumerate() {
        let (measurements, destination) = match &item.kind {
            AnnotationKind::Detector {
                measurement_ids, ..
            } => (measurement_ids, &mut detectors),
            AnnotationKind::Observable { measurement_ids } => (measurement_ids, &mut observables),
            AnnotationKind::TrackedPauli => continue,
        };
        destination.push(
            measurements
                .iter()
                .map(|measurement| {
                    ids.get(measurement)
                        .copied()
                        .ok_or(CompileError::Annotation {
                            annotation,
                            measurement: *measurement,
                        })
                })
                .collect::<Result<_, _>>()?,
        );
    }
    Ok(Program {
        instructions,
        num_qubits,
        num_records: ids.len(),
        detectors,
        observables,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use pecos_core::{ChannelExpr, channel};

    #[test]
    fn noops_skip_shape_validation() {
        for kind in [GateType::I, GateType::Idle, GateType::TrackedPauliMeta] {
            for qubits in [vec![], vec![QubitId(0), QubitId(0)]] {
                let mut gate = Gate::simple(kind, qubits);
                gate.angles.push(Angle64::ZERO);
                assert!(matches!(
                    validate_gate(Batch { gate: &gate, index: 0 }, 0),
                    Ok(Operation::Gate(actual)) if actual == kind
                ));
            }
        }
    }

    // TickCircuit insertion already rejects most malformed gates. Exercise the
    // compiler's validation boundary directly too, without weakening insertion.
    #[test]
    fn malformed_batch_validation() {
        let mut bad = Vec::new();
        for kind in [GateType::X, GateType::CX] {
            let mut gate = Gate::simple(kind, vec![QubitId(0), QubitId(0)]);
            bad.push(gate.clone());
            gate.qubits = vec![QubitId(0)].into();
            gate.angles.push(Angle64::ZERO);
            bad.push(gate);
        }
        bad.push(Gate::simple(GateType::CX, vec![QubitId(0)]));
        bad.push(Gate::simple(GateType::X, vec![]));
        let mut rz = Gate::rz(Angle64::ZERO, &[0]);
        rz.angles.clear();
        bad.push(rz);
        for kind in [GateType::MZ, GateType::MX, GateType::MPZ] {
            let mut gate = Gate::simple(kind, vec![QubitId(0), QubitId(1)]);
            gate.meas_ids.push(MeasId::from_raw(7));
            bad.push(gate);
        }
        let channel = Gate::channel(channel::BitFlip(0.5, 2));
        let mut gate = channel.clone();
        gate.qubits = vec![QubitId(1)].into();
        bad.push(gate);
        let mut gate = channel.clone();
        gate.channel = None;
        bad.push(gate);
        let mut gate = channel.clone();
        gate.qubits.push(QubitId(2));
        bad.push(gate);
        let mut gate = channel;
        gate.angles.push(Angle64::ZERO);
        bad.push(gate);
        for gate in bad {
            let error = validate_gate(
                Batch {
                    gate: &gate,
                    index: 3,
                },
                5,
            )
            .unwrap_err();
            assert!(matches!(
                error,
                CompileError::Gate {
                    tick: 5,
                    batch: 3,
                    ..
                }
            ));
        }
        // Empty channels and the identity tracked-Pauli annotation are valid.
        for gate in [
            Gate::channel(ChannelExpr::Compose(vec![])),
            Gate::simple(GateType::TrackedPauliMeta, vec![]),
        ] {
            assert!(
                validate_gate(
                    Batch {
                        gate: &gate,
                        index: 0
                    },
                    0
                )
                .is_ok()
            );
        }
    }
}
