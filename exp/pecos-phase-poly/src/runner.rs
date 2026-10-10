// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0.

//! Validated `TickCircuit` execution with sampled Pauli-mixture noise.
//!
//! Records follow execution order, independent of stable measurement-ID allocation.
//! Detector and observable values are raw XOR parities, without reference normalization.
//! Noise uses the caller's RNG; measurements use the simulator's independent RNG.
//!
//! There is no up-front compatibility certification. arXiv:2610.06811 Section 6
//! (iterative-measurements theorem) guarantees compatibility under Pauli noise when
//! every noiseless X measurement is deterministic or case 1 on every noiseless
//! branch. Otherwise an incompatible X measurement fails loudly for that shot,
//! with its original tick, batch and qubit. The state retains the executed prefix.

use crate::{IncompatibleMeasurement, PhasePoly};
use pecos_core::{PauliBitmaskSmall, QubitId, gate_type::GateType};
use pecos_quantum::TickCircuit;
use pecos_random::PecosRng;
use pecos_simulators::QuantumSimulator;
use std::fmt;

mod builder;
pub use builder::{CompileError, compile};
#[cfg(test)]
mod tests;

#[derive(Clone, Debug)]
enum Operation {
    Gate(GateType),
    Phase(u8),
    ZZ(u8),
    Channel(Vec<(f64, PauliBitmaskSmall)>),
}

#[derive(Clone, Debug)]
struct Instruction {
    operation: Operation,
    qubits: Vec<QubitId>,
    tick: usize,
    batch: usize,
}

/// A validated, reusable circuit. Compilation does not simulate measurements.
#[derive(Clone, Debug)]
pub struct Program {
    instructions: Vec<Instruction>,
    num_qubits: usize,
    num_records: usize,
    detectors: Vec<Vec<usize>>,
    observables: Vec<Vec<usize>>,
}

/// Measurement records and raw annotation parities from a completed shot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShotResult {
    /// Outcomes in execution order; false denotes eigenvalue +1.
    pub records: Vec<bool>,
    /// Raw detector XORs, in detector annotation order.
    pub detectors: Vec<bool>,
    /// Raw observable XORs, in observable annotation order.
    pub observables: Vec<bool>,
}

/// A shot could not execute on the supplied simulator.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RunError {
    /// The simulator must have exactly the compiled circuit's qubit count.
    QubitCount {
        /// Compiled qubit count.
        expected: usize,
        /// Supplied simulator's qubit count.
        actual: usize,
    },
    /// An X measurement has a branch outside the phase-polynomial family.
    Measurement {
        /// Zero-based tick index.
        tick: usize,
        /// Zero-based batch index within the tick.
        batch: usize,
        /// Physical qubit measured.
        qubit: QubitId,
        /// Underlying compatibility diagnostic.
        source: IncompatibleMeasurement,
    },
}

impl fmt::Display for RunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::QubitCount { expected, actual } => {
                write!(f, "program needs {expected} qubits, simulator has {actual}")
            }
            Self::Measurement {
                tick,
                batch,
                qubit,
                source,
            } => {
                write!(
                    f,
                    "MX on {} at tick {tick}, batch {batch}: {source}",
                    qubit.0
                )
            }
        }
    }
}
impl std::error::Error for RunError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Measurement { source, .. } => Some(source),
            Self::QubitCount { .. } => None,
        }
    }
}

impl Program {
    /// Validate all batches and resolve annotation IDs before execution.
    ///
    /// # Errors
    /// Returns the original location of an unsupported or malformed batch or annotation.
    pub fn compile(circuit: &TickCircuit) -> Result<Self, CompileError> {
        compile(circuit)
    }

    /// Required physical qubit count (largest stored index plus one).
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Number of recorded outcomes per completed shot.
    #[must_use]
    pub fn num_records(&self) -> usize {
        self.num_records
    }

    /// Record ordinals for each detector, retaining repeated references.
    #[must_use]
    pub fn detectors(&self) -> &[Vec<usize>] {
        &self.detectors
    }

    /// Record ordinals for each observable, retaining repeated references.
    #[must_use]
    pub fn observables(&self) -> &[Vec<usize>] {
        &self.observables
    }

    /// Reset the supplied simulator and run one shot, preserving both RNG streams.
    ///
    /// # Errors
    /// Rejects a mismatched qubit count before modifying the simulator or RNG.
    /// An incompatible X measurement aborts at its location, retaining the prefix
    /// state and consumed randomness. No partial result is returned.
    pub fn run_shot(
        &self,
        sim: &mut PhasePoly,
        rng: &mut PecosRng,
    ) -> Result<ShotResult, RunError> {
        if sim.num_qubits() != self.num_qubits {
            return Err(RunError::QubitCount {
                expected: self.num_qubits,
                actual: sim.num_qubits(),
            });
        }
        sim.reset();
        let mut records = Vec::with_capacity(self.num_records);
        for instruction in &self.instructions {
            instruction.execute(sim, rng, &mut records)?;
        }
        let parity = |indices: &Vec<usize>| indices.iter().fold(false, |p, &i| p ^ records[i]);
        Ok(ShotResult {
            detectors: self.detectors.iter().map(parity).collect(),
            observables: self.observables.iter().map(parity).collect(),
            records,
        })
    }

    /// Run `shots` using independent reproducible noise and measurement streams.
    /// Both stream seeds are drawn, in that order, from a seed-initialized RNG.
    ///
    /// # Errors
    /// Stops at the first incompatible measurement and returns its location.
    pub fn run_shots(&self, shots: usize, seed: u64) -> Result<Vec<ShotResult>, RunError> {
        let mut seeds = PecosRng::seed_from_u64(seed);
        let mut noise = PecosRng::seed_from_u64(seeds.next_u64());
        let mut sim = PhasePoly::with_seed(self.num_qubits, seeds.next_u64());
        (0..shots)
            .map(|_| self.run_shot(&mut sim, &mut noise))
            .collect()
    }
}

impl Instruction {
    fn execute(
        &self,
        sim: &mut PhasePoly,
        rng: &mut PecosRng,
        records: &mut Vec<bool>,
    ) -> Result<(), RunError> {
        let qs = &self.qubits;
        match &self.operation {
            Operation::Phase(power) => {
                sim.diagonal(qs, *power);
            }
            Operation::ZZ(power) => {
                for pair in qs.as_chunks::<2>().0 {
                    sim.cx(&[(pair[0], pair[1])]);
                    sim.diagonal(&[pair[1]], *power);
                    sim.cx(&[(pair[0], pair[1])]);
                }
            }
            Operation::Channel(alternatives) => {
                // Same normalized cumulative sampling as StabActive, preserving tiny weights.
                let total: f64 = alternatives.iter().map(|(p, _)| p).sum();
                let draw = rng.next_f64();
                let mut cumulative = 0.0;
                let (_, pauli) = alternatives
                    .iter()
                    .find(|(p, _)| {
                        cumulative += p;
                        draw < cumulative / total
                    })
                    .expect("validated nonempty distribution with finite positive total");
                for &q in qs {
                    match (pauli.has_x(q.0), pauli.has_z(q.0)) {
                        (true, true) => {
                            sim.y(&[q]);
                        }
                        (true, false) => {
                            sim.x(&[q]);
                        }
                        (false, true) => {
                            sim.z(&[q]);
                        }
                        (false, false) => {}
                    }
                }
            }
            Operation::Gate(kind) => match kind {
                GateType::I | GateType::Idle | GateType::TrackedPauliMeta => {}
                GateType::X => {
                    sim.x(qs);
                }
                GateType::Y => {
                    sim.y(qs);
                }
                GateType::Z => {
                    sim.z(qs);
                }
                GateType::SZ => {
                    sim.sz(qs);
                }
                GateType::SZdg => {
                    sim.szdg(qs);
                }
                GateType::T => {
                    sim.t(qs);
                }
                GateType::Tdg => {
                    sim.tdg(qs);
                }
                GateType::CX | GateType::CZ => {
                    for pair in qs.as_chunks::<2>().0 {
                        if *kind == GateType::CX {
                            sim.cx(&[(pair[0], pair[1])]);
                        } else {
                            sim.cz(&[(pair[0], pair[1])]);
                        }
                    }
                }
                GateType::PZ => {
                    sim.pz(qs);
                }
                GateType::PX => {
                    sim.px(qs);
                }
                GateType::MZ | GateType::MPZ => {
                    for &q in qs {
                        let outcome = sim.mz(&[q])[0].outcome;
                        records.push(outcome);
                        if *kind == GateType::MPZ && outcome {
                            sim.x(&[q]);
                        }
                    }
                }
                GateType::MX => {
                    for &q in qs {
                        records.push(
                            sim.mx(q)
                                .map_err(|source| RunError::Measurement {
                                    tick: self.tick,
                                    batch: self.batch,
                                    qubit: q,
                                    source,
                                })?
                                .outcome,
                        );
                    }
                }
                _ => unreachable!("validated gate"),
            },
        }
        Ok(())
    }
}
