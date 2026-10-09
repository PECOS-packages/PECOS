// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0.

use super::{AffineSign, HeisenbergOp, HeisenbergProgram, NoiseChannel, VirtualPauli, dispatch};
use crate::PauliKindForDecomp;
use crate::structure::{clifford_turns, rotate_tableau};
use num_complex::Complex64;
use pecos_core::{BitmaskStorage, MeasId, PauliBitmaskVec, gate_type::GateType};
use pecos_quantum::{AnnotationKind, Gate, TickCircuit};
use pecos_simulators::SparseStabY;
use pecos_stab_tn::stab_mps::pauli_decomp::decompose_pauli_string;
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

/// Failure to compile a static circuit, before any frame replay takes place.
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
        qubits: gate.qubits.iter().map(pecos_core::QubitId::index).collect(),
        tick,
        batch: gate.batch_index(),
        reason: reason.into(),
    }
}

fn validate_gate(gate: Batch<'_>, tick: usize) -> Result<(), CompileError> {
    match gate.gate_type {
        g if dispatch::named_clifford(g) || dispatch::rotation_axis(g).is_some() => {}
        GateType::MZ | GateType::MX | GateType::MPZ | GateType::PZ | GateType::PX => {}
        GateType::I | GateType::Idle | GateType::TrackedPauliMeta => return Ok(()),
        GateType::Channel => {
            let channel = gate
                .channel
                .as_ref()
                .ok_or_else(|| error(gate, tick, "missing channel expression"))?;
            pecos_quantum::channel::pauli_mixture(channel)
                .map_err(|e| error(gate, tick, e.to_string()))?;
            let support: Vec<_> = channel.qubits().into_iter().collect();
            let stored: BTreeSet<_> = gate.qubits.iter().map(pecos_core::QubitId::index).collect();
            if support.into_iter().collect::<BTreeSet<_>>() != stored
                || stored.len() != gate.qubits.len()
                || !gate.angles.is_empty()
            {
                return Err(error(
                    gate,
                    tick,
                    "channel support disagrees with stored qubits",
                ));
            }
            return Ok(());
        }
        _ => return Err(error(gate, tick, "unsupported gate")),
    }
    let arity = gate.gate_type.quantum_arity();
    if gate.qubits.is_empty()
        || !gate.qubits.len().is_multiple_of(arity)
        || gate.angles.len() != gate.gate_type.angle_arity()
        || gate.qubits.iter().collect::<BTreeSet<_>>().len() != gate.qubits.len()
    {
        return Err(error(gate, tick, "invalid angle count or qubit batch"));
    }
    if gate.gate_type.consumes_measurement_record() && gate.meas_ids.len() != gate.qubits.len() {
        return Err(error(
            gate,
            tick,
            "each recorded measurement needs one stable ID",
        ));
    }
    Ok(())
}

fn record_map(circuit: &TickCircuit) -> Result<BTreeMap<MeasId, usize>, CompileError> {
    let mut ids = BTreeMap::new();
    for (tick, row) in circuit.iter_ticks() {
        for gate in row.iter_gate_batches() {
            let gate = Batch {
                gate: gate.as_gate(),
                index: gate.batch_index(),
            };
            validate_gate(gate, tick)?;
            if gate.gate_type.consumes_measurement_record() {
                for &id in &gate.meas_ids {
                    let ordinal = ids.len();
                    if ids.insert(id, ordinal).is_some() {
                        return Err(error(gate, tick, "duplicate measurement ID"));
                    }
                }
            }
        }
    }
    Ok(ids)
}

pub(super) fn pullback(
    frame: &SparseStabY,
    physical: &[(usize, PauliKindForDecomp)],
) -> (VirtualPauli, bool) {
    let (flips, signs, mut phase) =
        decompose_pauli_string(frame.stabs(), frame.destabs(), physical);
    let mut bits = PauliBitmaskVec::identity();
    for &q in &flips {
        bits.x_bits.set_bit(q);
    }
    for &q in &signs {
        bits.z_bits.set_bit(q);
    }
    // The decomposition is phi X^F Z^G. Replacing every XZ by Y/i
    // gives the Hermitian tensor body with real coefficient phi i^(-k).
    for _ in flips.iter().filter(|q| signs.contains(q)) {
        phase *= Complex64::new(0.0, -1.0);
    }
    assert!(
        phase == Complex64::new(1.0, 0.0) || phase == Complex64::new(-1.0, 0.0),
        "Hermitian pullback must have real unit sign"
    );
    let factors = flips
        .iter()
        .chain(&signs)
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(|q| {
            (
                q,
                match (bits.has_x(q), bits.has_z(q)) {
                    (true, true) => PauliKindForDecomp::Y,
                    (true, false) => PauliKindForDecomp::X,
                    _ => PauliKindForDecomp::Z,
                },
            )
        })
        .collect();
    (VirtualPauli { bits, factors }, phase.re < 0.0)
}

enum Symbol {
    Noise(usize),
    Measurement(usize),
}
struct Builder {
    frame: SparseStabY,
    effects: Vec<(VirtualPauli, Symbol)>,
    program: HeisenbergProgram,
}

impl Builder {
    fn signed(&self, physical: &[(usize, PauliKindForDecomp)]) -> (VirtualPauli, AffineSign) {
        let (pauli, constant) = pullback(&self.frame, physical);
        let mut sign = AffineSign {
            constant,
            ..AffineSign::default()
        };
        for (component, symbol) in &self.effects {
            if !component.bits.commutes_with(&pauli.bits) {
                match symbol {
                    Symbol::Noise(i) => sign.noise.push(*i),
                    Symbol::Measurement(i) => sign.measurements.push(*i),
                }
            }
        }
        (pauli, sign)
    }

    fn channel(&mut self, gate: Batch<'_>) {
        let expression = gate.channel.as_ref().expect("validated channel");
        let qubits: Vec<_> = gate.qubits.iter().map(pecos_core::QubitId::index).collect();
        let first_symbol = self.program.num_noise_symbols;
        for &q in &qubits {
            for axis in [PauliKindForDecomp::X, PauliKindForDecomp::Z] {
                let (component, _) = pullback(&self.frame, &[(q, axis)]);
                self.effects
                    .push((component, Symbol::Noise(self.program.num_noise_symbols)));
                self.program.num_noise_symbols += 1;
            }
        }
        let alternatives = pecos_quantum::channel::pauli_mixture(expression)
            .expect("validated channel")
            .into_iter()
            .map(|(p, pauli)| {
                (
                    p,
                    qubits
                        .iter()
                        .flat_map(|&q| [pauli.has_x(q), pauli.has_z(q)])
                        .collect(),
                )
            })
            .collect();
        self.program.noise_channels.push(NoiseChannel {
            qubits,
            first_symbol,
            alternatives,
        });
    }

    fn measurement(&mut self, gate: Batch<'_>, ids: &BTreeMap<MeasId, usize>) {
        let x_basis = matches!(gate.gate_type, GateType::MX | GateType::PX);
        let axis = if x_basis {
            PauliKindForDecomp::X
        } else {
            PauliKindForDecomp::Z
        };
        for (i, q) in gate.qubits.iter().enumerate() {
            let (pauli, sign) = self.signed(&[(q.index(), axis)]);
            let symbol = self.program.num_measurements;
            let record = gate
                .gate_type
                .consumes_measurement_record()
                .then(|| ids[&gate.meas_ids[i]]);
            self.program.operations.push(HeisenbergOp::Measurement {
                pauli,
                sign,
                symbol,
                record,
            });
            self.program.num_measurements += 1;
            if matches!(gate.gate_type, GateType::PZ | GateType::PX | GateType::MPZ) {
                let correction = if x_basis {
                    PauliKindForDecomp::Z
                } else {
                    PauliKindForDecomp::X
                };
                let (component, _) = pullback(&self.frame, &[(q.index(), correction)]);
                self.effects.push((component, Symbol::Measurement(symbol)));
            }
        }
    }

    fn gate(&mut self, gate: Batch<'_>, ids: &BTreeMap<MeasId, usize>) {
        if let Some(axis) = dispatch::rotation_axis(gate.gate_type) {
            let angle = gate.angles[0];
            for qs in gate.qubits.chunks_exact(gate.gate_type.quantum_arity()) {
                let physical: Vec<_> = qs.iter().map(|q| (q.index(), axis)).collect();
                if let Some(turns) = clifford_turns(angle) {
                    rotate_tableau(&mut self.frame, &physical, turns);
                } else {
                    let (pauli, sign) = self.signed(&physical);
                    self.program
                        .operations
                        .push(HeisenbergOp::Rotation { pauli, sign, angle });
                }
            }
        } else if dispatch::named_clifford(gate.gate_type) {
            dispatch::apply_clifford(&mut self.frame, gate.gate_type, &gate.qubits);
        } else {
            match gate.gate_type {
                GateType::Channel => self.channel(gate),
                GateType::MZ | GateType::MX | GateType::MPZ | GateType::PZ | GateType::PX => {
                    self.measurement(gate, ids);
                }
                GateType::I | GateType::Idle | GateType::TrackedPauliMeta => {}
                _ => unreachable!("validated gate"),
            }
        }
    }
}

pub(super) fn compile(circuit: &TickCircuit) -> Result<HeisenbergProgram, CompileError> {
    let ids = record_map(circuit)?;
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
    let num_qubits = circuit
        .iter_gate_batches()
        .flat_map(|g| g.as_gate().qubits.iter())
        .map(|q| q.index() + 1)
        .max()
        .unwrap_or(0);
    let mut builder = Builder {
        frame: SparseStabY::with_seed(num_qubits, 0).with_destab_sign_tracking(),
        effects: Vec::new(),
        program: HeisenbergProgram {
            num_qubits,
            operations: Vec::new(),
            noise_channels: Vec::new(),
            num_noise_symbols: 0,
            num_measurements: 0,
            num_records: ids.len(),
            detectors,
            observables,
        },
    };
    for (_, tick) in circuit.iter_ticks() {
        for gate in tick.iter_gate_batches() {
            builder.gate(
                Batch {
                    gate: gate.as_gate(),
                    index: gate.batch_index(),
                },
                &ids,
            );
        }
    }
    let operations = std::mem::take(&mut builder.program.operations);
    let program = builder
        .program
        .with_operations(operations)
        .unwrap_or_else(|error| {
            panic!("compile produced an invalid program (builder bug): {error}")
        });
    Ok(program)
}
