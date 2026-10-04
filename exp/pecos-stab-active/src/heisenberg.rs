// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0.

//! Unscheduled Heisenberg programs in the virtual basis of the initial `|0^n>`.
//! Cliffords are replayed only at compilation. Pauli faults and reset corrections
//! enter affine signs through symplectic anticommutation with later operations.

mod builder;
mod dispatch;
mod noise;
#[cfg(test)]
mod tests;
mod validation;

use crate::{ActiveStructure, PauliKindForDecomp, StabActive};
pub use builder::CompileError;
pub use noise::NoiseChannel;
use pecos_core::{Angle64, PauliBitmaskVec};
use pecos_quantum::TickCircuit;
pub(crate) use validation::ProgramError;

/// A constant XOR noise symbols XOR measurement symbols (including hidden resets).
/// Each symbol occurs at most once in its set; outcomes are never flattened.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AffineSign {
    /// Constant real minus sign after Hermitian phase normalization.
    pub constant: bool,
    /// Sorted, distinct indices of sampled noise component bits.
    pub noise: Vec<usize>,
    /// Sorted, distinct indices of earlier raw signed measurement outcomes.
    pub measurements: Vec<usize>,
}

impl AffineSign {
    /// Evaluate `constant XOR noise XOR measurements`.
    /// The measurement slice is indexed by symbol, which equals execution
    /// position in compiled order.
    ///
    /// # Panics
    /// Panics if a referenced symbol has no entry in the supplied slices.
    #[must_use]
    pub fn evaluate(&self, noise: &[bool], measurements: &[bool]) -> bool {
        self.noise.iter().fold(self.constant, |s, &i| s ^ noise[i])
            ^ self
                .measurements
                .iter()
                .fold(false, |s, &i| s ^ measurements[i])
    }
}

/// Hermitian body `H = i^|F intersect G| X^F Z^G`, with no scalar sign.
/// Word-level symplectic products on `bits` determine commutation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VirtualPauli {
    bits: PauliBitmaskVec,
    factors: Vec<(usize, PauliKindForDecomp)>,
}

impl VirtualPauli {
    /// The X and Z planes, with Hermitian Y at overlapping bits.
    #[must_use]
    pub fn bits(&self) -> &PauliBitmaskVec {
        &self.bits
    }

    /// Distinct virtual qubits and Hermitian single-qubit Pauli factors.
    #[must_use]
    pub fn factors(&self) -> &[(usize, PauliKindForDecomp)] {
        &self.factors
    }
}

/// An operation in the program's execution order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HeisenbergOp {
    /// Apply `exp(-i (-1)^sign angle H / 2)`.
    Rotation {
        /// Unsigned Hermitian virtual Pauli body H.
        pauli: VirtualPauli,
        /// Exact fixed-point rotation angle.
        angle: Angle64,
        /// Affine exponent of the real sign.
        sign: AffineSign,
    },
    /// Measure `(-1)^sign H`; the raw result defines `symbol`.
    Measurement {
        /// Unsigned Hermitian virtual Pauli body H.
        pauli: VirtualPauli,
        /// Affine exponent of the real sign.
        sign: AffineSign,
        /// Measurement symbol, including hidden reset measurements.
        symbol: usize,
        /// Visible record ordinal, or none for a hidden reset measurement.
        record: Option<usize>,
    },
}

/// A program whose operations run in list order.
/// `compile` preserves physical order; every program is checked at construction.
#[derive(Clone, Debug)]
pub struct HeisenbergProgram {
    num_qubits: usize,
    operations: Vec<HeisenbergOp>,
    noise_channels: Vec<NoiseChannel>,
    num_noise_symbols: usize,
    num_measurements: usize,
    num_records: usize,
    detectors: Vec<Vec<usize>>,
    observables: Vec<Vec<usize>>,
}

/// Visible records and their annotated XORs from one execution of a program.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShotResult {
    /// Raw visible outcomes in record ordinal order.
    pub records: Vec<bool>,
    /// Detector parities in detector annotation order.
    pub detectors: Vec<bool>,
    /// Observable parities in observable annotation order.
    pub observables: Vec<bool>,
    /// Maximum number of active coordinates during this shot.
    pub peak_active_width: usize,
}

impl HeisenbergProgram {
    /// Pull back operations by the Clifford frame `C` as `C† P C`.
    /// The entire input is validated before replaying any Clifford.
    ///
    /// # Errors
    /// Returns a location-bearing error for unsupported or malformed gates,
    /// invalid channels, duplicate measurement IDs, or dangling annotations.
    ///
    /// # Panics
    /// Panics if a builder bug produces an invalid program.
    pub fn compile(circuit: &TickCircuit) -> Result<Self, CompileError> {
        builder::compile(circuit)
    }

    /// Replace the operation list, keeping every table, and check the result.
    /// This is the only way to obtain a program from a builder or rewrite.
    pub(crate) fn with_operations(
        self,
        operations: Vec<HeisenbergOp>,
    ) -> Result<Self, ProgramError> {
        let program = Self { operations, ..self };
        program.validate()?;
        Ok(program)
    }

    /// Size of the virtual register, including all physical gate support.
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Operations in execution order.
    #[must_use]
    pub fn operations(&self) -> &[HeisenbergOp] {
        &self.operations
    }

    /// Active width after each operation, without allocating amplitudes or
    /// imposing a width limit. The peak is the maximum of the returned list.
    /// An empty program returns an empty list and has peak width zero.
    /// All signs are evaluated as positive and raw measurement outcomes as zero;
    /// projection values still include the sign of the shared measurement basis.
    /// The tableau carries an internal RNG, but profiling makes no sampling draws.
    #[must_use]
    pub fn width_profile(&self) -> Vec<usize> {
        let mut structure = ActiveStructure::with_seed(self.num_qubits, 0);
        self.operations
            .iter()
            .map(|operation| {
                match operation {
                    HeisenbergOp::Rotation { pauli, angle, .. } => {
                        let rotation = structure.rotation(*angle, pauli.factors(), false);
                        structure.apply_rotation(rotation);
                    }
                    HeisenbergOp::Measurement { pauli, .. } => {
                        let measurement = structure.measurement(pauli.factors(), false);
                        if let Some(active) = structure.apply_measurement(measurement, false) {
                            let projection = active.projection;
                            structure.demote(projection);
                        }
                    }
                }
                structure.width()
            })
            .collect()
    }

    /// Independent channel instances, each with a joint component distribution.
    #[must_use]
    pub fn noise_channels(&self) -> &[NoiseChannel] {
        &self.noise_channels
    }

    /// Detector XORs over visible record ordinals.
    #[must_use]
    pub fn detectors(&self) -> &[Vec<usize>] {
        &self.detectors
    }

    /// Observable XORs over visible record ordinals.
    #[must_use]
    pub fn observables(&self) -> &[Vec<usize>] {
        &self.observables
    }

    /// Execute from `|0^n>` using one seeded RNG stream for noise and measurement.
    ///
    /// # Panics
    /// Panics if the active width exceeds `StabActive`'s default limit (26).
    #[must_use]
    pub fn run(&self, seed: u64) -> ShotResult {
        let mut state = StabActive::with_seed(self.num_qubits, seed);
        let mut noise = vec![false; self.num_noise_symbols];
        for channel in &self.noise_channels {
            channel.sample(&mut state.rng, &mut noise);
        }
        self.execute(state, &noise, |_, _, _, _, _| None)
    }

    fn execute<F>(&self, state: StabActive, noise: &[bool], force: F) -> ShotResult
    where
        F: FnMut(&StabActive, &VirtualPauli, bool, usize, Option<usize>) -> Option<bool>,
    {
        self.execute_with_outcomes(state, noise, force).0
    }

    fn execute_with_outcomes<F>(
        &self,
        state: StabActive,
        noise: &[bool],
        force: F,
    ) -> (ShotResult, Vec<bool>)
    where
        F: FnMut(&StabActive, &VirtualPauli, bool, usize, Option<usize>) -> Option<bool>,
    {
        self.execute_observed(state, noise, force, |_, _, _| {})
    }

    // Tests inject noise, a pre-measurement force hook, and an observer after
    // every operation through this production loop. Outcomes are stored by
    // symbol and records by ordinal. Construction checked that every symbol
    // is produced before any operation reads it.
    fn execute_observed<F, O>(
        &self,
        mut state: StabActive,
        noise: &[bool],
        mut force: F,
        mut observe: O,
    ) -> (ShotResult, Vec<bool>)
    where
        F: FnMut(&StabActive, &VirtualPauli, bool, usize, Option<usize>) -> Option<bool>,
        O: FnMut(&StabActive, &HeisenbergOp, &[bool]),
    {
        let mut measurements = vec![false; self.num_measurements];
        let mut records = vec![false; self.num_records];
        for operation in &self.operations {
            match operation {
                HeisenbergOp::Rotation { pauli, angle, sign } => {
                    state.rotate_pauli(
                        *angle,
                        pauli.factors(),
                        sign.evaluate(noise, &measurements),
                    );
                }
                HeisenbergOp::Measurement {
                    pauli,
                    sign,
                    symbol,
                    record,
                } => {
                    let negative = sign.evaluate(noise, &measurements);
                    let forced = force(&state, pauli, negative, *symbol, *record);
                    let result = state.measure_pauli_inner(pauli.factors(), negative, forced);
                    measurements[*symbol] = result.outcome;
                    if let Some(ordinal) = record {
                        records[*ordinal] = result.outcome;
                    }
                }
            }
            observe(&state, operation, &measurements);
        }
        let parity = |ids: &Vec<usize>| ids.iter().fold(false, |s, &i| s ^ records[i]);
        let shot = ShotResult {
            detectors: self.detectors.iter().map(parity).collect(),
            observables: self.observables.iter().map(parity).collect(),
            records,
            peak_active_width: state.peak_active_width(),
        };
        (shot, measurements)
    }
}
