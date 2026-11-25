use super::{Location, LossDemError, Side, invalid};
use crate::dem_build::evaluate_record;
use crate::fault_tolerance::circuit_definitions::{
    CircuitDefinitions, MeasurementEmission, definitions_from_tick_circuit,
    tick_circuit_measurement_emission,
};
use pecos_core::{QubitId, gate_type::GateType};
use pecos_decoder_core::obs_mask::ObsMask;
use pecos_engines::noise::GeneralNoiseModel;
use pecos_engines::quantum::SparseStabEngine;
use pecos_engines::{ByteMessage, Engine, EngineSystem, QuantumSystem};
use pecos_quantum::TickCircuit;
use std::collections::BTreeMap;

/// Probe index in engine outcome order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProbeId(pub usize);

/// Classification of one returned engine outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputSlot {
    /// Emission index of a detector/observable record.
    Record(usize),
    /// Loss-sensitive detect operand.
    Probe(ProbeId),
    /// Record-free Z herald or leakage-only detect.
    Herald,
}

/// Location and next same-qubit record for a loss probe.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Probe {
    /// Probe identifier.
    pub id: ProbeId,
    /// Detect operand (before its nondestructive readout).
    pub location: Location,
    /// Physical qubit.
    pub qubit: QubitId,
    /// Next record on this qubit, if one exists.
    pub next_record: Option<usize>,
}

/// Immutable layout constructed independently of message emission.
#[derive(Debug, Clone)]
pub struct OutputTable {
    /// Every outcome, in batch then operand order.
    pub slots: Vec<OutputSlot>,
    /// Probe metadata indexed by `ProbeId`.
    pub probes: Vec<Probe>,
    /// Record emission descriptors.
    pub records: Vec<MeasurementEmission>,
    /// Gate coordinates for each emission-indexed record.
    pub record_locations: Vec<Location>,
}

impl OutputTable {
    /// Classify an annotated circuit's engine output stream.
    ///
    /// # Panics
    /// Panics if output-table records disagree with canonical emission order.
    ///
    /// # Errors
    /// Rejects unsupported gates with their tick and batch coordinates.
    pub fn new(circuit: &TickCircuit) -> Result<Self, LossDemError> {
        validate_circuit(circuit)?;
        let mut table = Self {
            slots: Vec::new(),
            probes: Vec::new(),
            records: Vec::new(),
            record_locations: Vec::new(),
        };
        let mut waiting: BTreeMap<QubitId, Vec<usize>> = BTreeMap::new();
        for (tick, gates) in circuit.iter_ticks() {
            for (batch, gate) in gates.iter_gate_batches().enumerate() {
                for (operand, &qubit) in gate.qubits.iter().enumerate() {
                    if gate.gate_type.consumes_measurement_record() {
                        let record = table.records.len();
                        if let Some(probes) = waiting.remove(&qubit) {
                            for index in probes {
                                table.probes[index].next_record = Some(record);
                            }
                        }
                        table.slots.push(OutputSlot::Record(record));
                        table.record_locations.push(Location {
                            tick,
                            batch,
                            operand,
                            side: Side::Before,
                        });
                        table.records.push(MeasurementEmission {
                            qubit,
                            meas_id: gate.meas_ids.get(operand).copied(),
                        });
                    } else if matches!(
                        gate.gate_type,
                        GateType::DetectLost | GateType::DetectLeakedLost
                    ) {
                        let id = ProbeId(table.probes.len());
                        waiting.entry(qubit).or_default().push(id.0);
                        table.slots.push(OutputSlot::Probe(id));
                        table.probes.push(Probe {
                            id,
                            location: Location {
                                tick,
                                batch,
                                operand,
                                side: Side::Before,
                            },
                            qubit,
                            next_record: None,
                        });
                    } else if gate.gate_type.is_recordless_z_measurement()
                        || gate.gate_type.is_detection()
                    {
                        table.slots.push(OutputSlot::Herald);
                    }
                }
            }
        }
        assert_eq!(
            table.records,
            tick_circuit_measurement_emission(circuit),
            "output records must match canonical measurement emission order"
        );
        Ok(table)
    }

    pub(super) fn decode(
        &self,
        outcomes: &[u32],
        context: &str,
    ) -> Result<(Vec<u8>, Vec<bool>), LossDemError> {
        if outcomes.len() != self.slots.len() {
            return Err(LossDemError::Record {
                message: format!(
                    "engine returned {} outcomes, expected {} for {}; check the circuit's output-producing gates",
                    outcomes.len(),
                    self.slots.len(),
                    context
                ),
                source: None,
            });
        }
        let mut records = vec![0; self.records.len()];
        let mut flags = vec![false; self.probes.len()];
        for (&value, slot) in outcomes.iter().zip(&self.slots) {
            match *slot {
                OutputSlot::Record(record) => {
                    if value > 1 {
                        return Err(LossDemError::Record {
                            message: format!(
                                "record {record} at {:?} has nonbinary value {value}; use a binary record readout",
                                self.record_locations[record]
                            ),
                            source: None,
                        });
                    }
                    records[record] = u8::from(value == 1);
                }
                OutputSlot::Probe(id) => flags[id.0] = value == 3,
                OutputSlot::Herald => {}
            }
        }
        Ok((records, flags))
    }
}

pub(super) fn validate_circuit(circuit: &TickCircuit) -> Result<(), LossDemError> {
    let capacity = circuit
        .iter_gate_batches()
        .map(|g| g.qubits.iter().map(|q| q.index() + 1).max().unwrap_or(0))
        .max()
        .unwrap_or(0);
    let mut engine = SparseStabEngine::with_seed(capacity, 0);
    for (tick, gates) in circuit.iter_ticks() {
        for (batch, gate) in gates.iter_gate_batches().enumerate() {
            let g = gate.as_gate();
            if g.gate_type.is_meta() {
                continue;
            }
            let accepted = matches!(
                g.gate_type,
                GateType::PZ
                    | GateType::MZ
                    | GateType::MPZ
                    | GateType::MeasureZLeaked
                    | GateType::MeasureZLost
                    | GateType::MeasureZLeakedLost
                    | GateType::DetectLeaked
                    | GateType::DetectLost
                    | GateType::DetectLeakedLost
                    | GateType::Idle
                    | GateType::H
                    | GateType::F
                    | GateType::Fdg
                    | GateType::SZ
                    | GateType::SZdg
                    | GateType::SX
                    | GateType::SXdg
                    | GateType::SY
                    | GateType::SYdg
                    | GateType::X
                    | GateType::Y
                    | GateType::Z
                    | GateType::CX
                    | GateType::CY
                    | GateType::CZ
                    | GateType::SWAP
                    | GateType::SXX
                    | GateType::SXXdg
                    | GateType::SYY
                    | GateType::SYYdg
                    | GateType::SZZ
                    | GateType::SZZdg
            ) || pecos_core::try_lower_rotation_to_clifford(g).is_some();
            let context = || {
                format!(
                    "gate {:?} at (tick {tick}, batch {batch}), angles {:?}",
                    g.gate_type, g.angles
                )
            };
            if !accepted {
                return Err(invalid(format!(
                    "{} is not supported by the loss-conditioned DEM runner; use supported Clifford gates",
                    context()
                )));
            }
            g.validate()
                .map_err(|e| invalid(format!("{}: {e}; correct the gate payload", context())))?;
            let mut message = ByteMessage::quantum_operations_builder();
            message.add_gate_command(g);
            engine
                .process(message.build())
                .map_err(|source| LossDemError::Engine {
                    context: context(),
                    source,
                })?;
        }
    }
    Ok(())
}

/// One shot's physical bits and loss flags.
#[derive(Debug, Clone)]
pub struct Shot {
    /// Binary record values indexed by emission position.
    pub records: Vec<u8>,
    /// True exactly when a probe returned outcome 3.
    pub flags: Vec<bool>,
    /// Detector parities from the shared runtime evaluator.
    pub detectors: Vec<u8>,
    /// Observable parities from the shared runtime evaluator.
    pub observables: ObsMask,
}

/// Whole-circuit engine runner. Each shot resets once and submits one message.
pub struct TickRunner {
    system: QuantumSystem,
    message: ByteMessage,
    table: OutputTable,
    definitions: CircuitDefinitions,
    num_detectors: usize,
    execution_context: String,
}

impl TickRunner {
    /// Compile a circuit and seed both noise and ideal stabilizer engines.
    ///
    /// # Errors
    /// Rejects unsupported gates and invalid detector/observable definitions.
    pub fn new(
        circuit: &TickCircuit,
        noise: GeneralNoiseModel,
        seed: u64,
    ) -> Result<Self, LossDemError> {
        let table = OutputTable::new(circuit)?;
        let definitions = definitions_from_tick_circuit(circuit)?;
        let num_detectors = definitions
            .detectors
            .iter()
            .map(|d| d.id as usize + 1)
            .max()
            .unwrap_or(0);
        let capacity = circuit
            .iter_gate_batches()
            .map(|g| g.qubits.iter().map(|q| q.index() + 1).max().unwrap_or(0))
            .max()
            .unwrap_or(0);
        let mut message = ByteMessage::quantum_operations_builder();
        for (_, tick) in circuit.iter_ticks() {
            for gate in tick.iter_gate_batches() {
                if !gate.gate_type.is_meta() {
                    message.add_gate_command(gate.as_gate());
                }
            }
        }
        let mut system = QuantumSystem::new(
            Box::new(noise),
            Box::new(SparseStabEngine::with_seed(capacity, seed)),
        );
        system.set_seed(seed);
        Ok(Self {
            system,
            message: message.build(),
            table,
            definitions,
            num_detectors,
            execution_context: format!(
                "whole shot over tick indices 0..{} and {} stored batches (exact failing batch unavailable from whole-message execution)",
                circuit.num_ticks(),
                circuit.iter_gate_batches().count()
            ),
        })
    }

    /// Output layout shared by every shot.
    #[must_use]
    pub fn table(&self) -> &OutputTable {
        &self.table
    }

    /// Run one independently reset shot.
    ///
    /// # Errors
    /// Returns engine errors, outcome-count errors, and nonbinary record values.
    pub fn shot(&mut self) -> Result<Shot, LossDemError> {
        self.system.reset().map_err(|source| LossDemError::Engine {
            context: format!("reset before {}", self.execution_context),
            source,
        })?;
        let result = self
            .system
            .process_as_system(self.message.clone())
            .map_err(|source| LossDemError::Engine {
                context: self.execution_context.clone(),
                source,
            })?;
        let outcomes = result.outcomes().map_err(|source| LossDemError::Engine {
            context: format!("decode outputs for {}", self.execution_context),
            source,
        })?;
        let (records, flags) = self.table.decode(&outcomes, &self.execution_context)?;
        let (detectors, observables) =
            evaluate_record(&self.definitions, self.num_detectors, &records)
                .map_err(|source| LossDemError::Record {
                    message: format!(
                        "record {records:?} for {}: {source}; correct the record values or readout definitions",
                        self.execution_context
                    ),
                    source: Some(Box::new(source)),
                })?;
        Ok(Shot {
            records,
            flags,
            detectors,
            observables,
        })
    }
}
