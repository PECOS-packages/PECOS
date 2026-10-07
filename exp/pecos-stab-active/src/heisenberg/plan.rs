// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0.

//! Model A lowering: `C_real = P C_plan`. Corrections change signs only.

mod sampler;
mod validation;

use super::{AffineSign, HeisenbergOp, HeisenbergProgram, NoiseChannel, ShotResult, VirtualPauli};
use crate::{ActiveStructure, kernels, structure::clifford_turns};
use num_complex::Complex64;
use pecos_core::{Angle64, PauliBitmaskVec};
use pecos_stab_tn::stab_mps::coordinate_tableau::{CoordinateGate, MeasurementCase};
pub use sampler::Sampler;
use std::fmt;

/// A failure to lower a program into a bounded, causal sampling plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlanError {
    /// The limit cannot index a complex vector on this platform.
    InvalidMaxWidth { limit: usize },
    /// A promotion would exceed the supplied limit.
    WidthExceeded {
        operation: usize,
        width: usize,
        limit: usize,
    },
    /// An invalid lowered instruction (also detects internal planner errors).
    InvalidPlan {
        operation: usize,
        width: usize,
        reason: &'static str,
    },
}

impl fmt::Display for PlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidMaxWidth { limit } => write!(
                f,
                "active width limit {limit} cannot index a complex vector"
            ),
            Self::WidthExceeded {
                operation,
                width,
                limit,
            } => write!(
                f,
                "operation {operation}: active width {width} exceeds limit {limit}"
            ),
            Self::InvalidPlan {
                operation,
                width,
                reason,
            } => write!(f, "operation {operation}, width {width}: {reason}"),
        }
    }
}
impl std::error::Error for PlanError {}

#[derive(Clone, Debug)]
pub(super) struct Descriptor {
    pub(super) flip: usize,
    pub(super) sign: usize,
    pub(super) phase: Complex64,
}

impl Descriptor {
    fn new(parts: &pecos_stab_tn::stab_mps::coordinate_tableau::CoordinateDecomposition) -> Self {
        Self {
            flip: kernels::mask(&parts.active_flips),
            sign: kernels::mask(&parts.active_signs),
            phase: parts.phase,
        }
    }
}

#[derive(Clone, Debug)]
pub(super) struct Projection {
    pub(super) gates: Vec<CoordinateGate>,
    pub(super) pivot: usize,
    // Separate from the pre-draw sign: may reference this measurement's symbol.
    pub(super) value: AffineSign,
}

#[derive(Clone, Debug)]
pub(super) enum Instruction {
    Clifford,
    Rotation {
        angle: Angle64,
        double: bool,
        parts: Descriptor,
    },
    Measurement {
        case: MeasurementCase,
        parts: Descriptor,
        symbol: usize,
        record: Option<usize>,
        projection: Option<Projection>,
    },
}

#[derive(Clone, Debug)]
pub(super) struct PlannedOp {
    pub(super) sign: AffineSign,
    pub(super) instruction: Instruction,
}

/// Immutable instructions and annotations shared by any number of samplers.
/// Contains no tableau; all structural work is completed during planning.
#[derive(Clone, Debug)]
pub struct SamplingPlan {
    pub(super) operations: Vec<PlannedOp>,
    noise_channels: Vec<NoiseChannel>,
    num_noise_symbols: usize,
    num_measurements: usize,
    num_records: usize,
    detectors: Vec<Vec<usize>>,
    observables: Vec<Vec<usize>>,
    widths: Vec<usize>,
    peak: usize,
}

impl SamplingPlan {
    /// Active width after each source operation, including Clifford rotations.
    #[must_use]
    pub fn width_profile(&self) -> &[usize] {
        &self.widths
    }

    /// Symbol terms in each effective sign (the constant is not a term).
    pub fn sign_term_counts(&self) -> impl Iterator<Item = usize> + '_ {
        self.operations
            .iter()
            .map(|op| op.sign.noise.len() + op.sign.measurements.len())
    }

    /// Allocate reusable buffers sized to this plan's peak active width.
    #[must_use]
    pub fn sampler(&self) -> Sampler<'_> {
        Sampler::new(self)
    }

    /// Run one shot with a freshly allocated sampler.
    #[must_use]
    pub fn run(&self, seed: u64) -> ShotResult {
        self.sampler().run(seed)
    }
}

pub(super) type Corrections = Vec<(PauliBitmaskVec, AffineSign)>;

fn effective(sign: &AffineSign, body: &VirtualPauli, corrections: &Corrections) -> AffineSign {
    corrections
        .iter()
        .filter(|(row, _)| !row.commutes_with(body.bits()))
        .fold(sign.xor(&AffineSign::default()), |sign, (_, bit)| {
            sign.xor(bit)
        })
}

fn outcome_bit(sign: &AffineSign, symbol: usize, eta: bool) -> AffineSign {
    sign.xor(&AffineSign {
        constant: eta,
        measurements: vec![symbol],
        noise: Vec::new(),
    })
}

impl HeisenbergProgram {
    /// Lower once, retaining the signed executor as an independent reference.
    ///
    /// # Errors
    /// Returns a platform-limit error or an operation/width-bearing planning error.
    pub fn plan(&self, max_width: usize) -> Result<SamplingPlan, PlanError> {
        self.plan_observed(max_width, |_, _, _, _| {})
    }

    // Tests observe the live frame and reference tableau without storing either
    // in SamplingPlan. The no-op production observer is monomorphized away.
    pub(super) fn plan_observed(
        &self,
        max_width: usize,
        mut observe: impl FnMut(&ActiveStructure, &Corrections, &PlannedOp, usize),
    ) -> Result<SamplingPlan, PlanError> {
        if max_width > (isize::MAX.unsigned_abs() / size_of::<Complex64>()).ilog2() as usize {
            return Err(PlanError::InvalidMaxWidth { limit: max_width });
        }
        let mut structure = ActiveStructure::with_seed(self.num_qubits, 0);
        // Retain emitted corrections in source order, even identically zero
        // promotion bits, and validate every newly constructed expression.
        let mut corrections = Corrections::new();
        let mut produced = vec![false; self.num_measurements];
        let mut operations = Vec::with_capacity(self.operations.len());
        let mut widths = Vec::with_capacity(self.operations.len());
        for (index, operation) in self.operations.iter().enumerate() {
            let correction_start = corrections.len();
            let op = match operation {
                HeisenbergOp::Rotation { pauli, angle, sign } => {
                    let mut prime = effective(sign, pauli, &corrections);
                    // The constant sign fixes the reference Clifford. Symbolic
                    // quarter/three-quarter differences become body corrections.
                    let constant = clifford_turns(*angle).is_some() && prime.constant;
                    let rotation = structure.rotation(*angle, pauli.factors(), constant);
                    if rotation.needs_promotion() && rotation.width() + 1 > max_width {
                        return Err(PlanError::WidthExceeded {
                            operation: index,
                            width: rotation.width() + 1,
                            limit: max_width,
                        });
                    }
                    let (work, row) = rotation.plan_rotation();
                    if let Some(row) = row {
                        let bit = prime.xor(sign);
                        corrections.push((row, bit));
                        // Effective sign at kernel time, AFTER installing S_h.
                        prime = effective(sign, pauli, &corrections);
                    }
                    let instruction = if let Some(work) = work {
                        Instruction::Rotation {
                            angle: *angle,
                            double: work.double,
                            parts: Descriptor::new(&work.parts),
                        }
                    } else {
                        if !pauli.factors().is_empty()
                            && matches!(clifford_turns(*angle), Some(1 | 3))
                            && (!prime.noise.is_empty() || !prime.measurements.is_empty())
                        {
                            let mut bit = prime.clone();
                            bit.constant = false;
                            corrections.push((pauli.bits().clone(), bit));
                        }
                        Instruction::Clifford
                    };
                    PlannedOp {
                        sign: prime,
                        instruction,
                    }
                }
                HeisenbergOp::Measurement {
                    pauli,
                    sign,
                    symbol,
                    record,
                } => {
                    let prime = effective(sign, pauli, &corrections);
                    let measurement = structure.measurement(pauli.factors(), false);
                    let case = measurement.data().case();
                    let parts = Descriptor::new(measurement.data().parts());
                    let work = measurement.plan_measurement();
                    let value = outcome_bit(&prime, *symbol, work.eta);
                    if let Some(row) = work.row {
                        corrections.push((row, value.clone()));
                    }
                    let projection = work.active.map(|active| Projection {
                        gates: active.gates().to_vec(),
                        pivot: active.projection().pivot_bit(),
                        value,
                    });
                    // The final validator checks uniqueness and pre-draw causality.
                    if let Some(entry) = produced.get_mut(*symbol) {
                        *entry = true;
                    }
                    PlannedOp {
                        sign: prime,
                        instruction: Instruction::Measurement {
                            case,
                            parts,
                            symbol: *symbol,
                            record: *record,
                            projection,
                        },
                    }
                }
            };
            for (_, bit) in &corrections[correction_start..] {
                validation::check_sign(
                    bit,
                    &produced,
                    self.num_noise_symbols,
                    index,
                    structure.width(),
                )?;
            }
            observe(&structure, &corrections, &op, index);
            widths.push(structure.width());
            operations.push(op);
        }
        let plan = SamplingPlan {
            operations,
            widths,
            peak: structure.peak_width(),
            noise_channels: self.noise_channels.clone(),
            num_noise_symbols: self.num_noise_symbols,
            num_measurements: self.num_measurements,
            num_records: self.num_records,
            detectors: self.detectors.clone(),
            observables: self.observables.clone(),
        };
        plan.validate(max_width)?;
        Ok(plan)
    }
}
