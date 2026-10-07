// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0.

use super::{
    AffineSign, CoordinateGate, Descriptor, Instruction, MeasurementCase, PlanError, SamplingPlan,
};

fn invalid(operation: usize, width: usize, reason: &'static str) -> PlanError {
    PlanError::InvalidPlan {
        operation,
        width,
        reason,
    }
}

pub(super) fn check_sign(
    sign: &AffineSign,
    produced: &[bool],
    noise: usize,
    operation: usize,
    width: usize,
) -> Result<(), PlanError> {
    if !sign.noise.windows(2).all(|w| w[0] < w[1])
        || !sign.measurements.windows(2).all(|w| w[0] < w[1])
    {
        return Err(invalid(operation, width, "noncanonical affine sign"));
    }
    if sign.noise.iter().any(|&i| i >= noise) {
        return Err(invalid(operation, width, "noise symbol out of range"));
    }
    if sign
        .measurements
        .iter()
        .any(|&i| produced.get(i) != Some(&true))
    {
        return Err(invalid(
            operation,
            width,
            "measurement symbol not yet produced",
        ));
    }
    Ok(())
}

fn check_parts(parts: &Descriptor, operation: usize, width: usize) -> Result<(), PlanError> {
    if parts.flip >= 1 << width || parts.sign >= 1 << width {
        return Err(invalid(
            operation,
            width,
            "amplitude mask outside active width",
        ));
    }
    Ok(())
}

impl SamplingPlan {
    pub(in crate::heisenberg) fn validate(&self, limit: usize) -> Result<(), PlanError> {
        let mut produced = vec![false; self.num_measurements];
        let mut records = vec![false; self.num_records];
        let mut width = 0;
        let mut peak = 0;
        if self.operations.len() != self.widths.len() {
            return Err(invalid(0, 0, "width profile length mismatch"));
        }
        for (index, (op, &expected)) in self.operations.iter().zip(&self.widths).enumerate() {
            check_sign(&op.sign, &produced, self.num_noise_symbols, index, width)?;
            match &op.instruction {
                Instruction::Clifford => {}
                Instruction::Rotation { double, parts, .. } => {
                    width += usize::from(*double);
                    if width > limit {
                        return Err(PlanError::WidthExceeded {
                            operation: index,
                            width,
                            limit,
                        });
                    }
                    check_parts(parts, index, width)?;
                }
                Instruction::Measurement {
                    case,
                    parts,
                    symbol,
                    record,
                    projection,
                } => {
                    check_parts(parts, index, width)?;
                    let Some(entry) = produced.get_mut(*symbol) else {
                        return Err(invalid(index, width, "measurement symbol out of range"));
                    };
                    if *entry {
                        return Err(invalid(index, width, "duplicate measurement symbol"));
                    }
                    *entry = true;
                    if let Some(ordinal) = record {
                        let Some(entry) = records.get_mut(*ordinal) else {
                            return Err(invalid(index, width, "record ordinal out of range"));
                        };
                        if *entry {
                            return Err(invalid(index, width, "duplicate record ordinal"));
                        }
                        *entry = true;
                    }
                    if (*case == MeasurementCase::Active) != projection.is_some() {
                        return Err(invalid(
                            index,
                            width,
                            "measurement case/projection mismatch",
                        ));
                    }
                    if let Some(projection) = projection {
                        check_sign(
                            &projection.value,
                            &produced,
                            self.num_noise_symbols,
                            index,
                            width,
                        )?;
                        if projection.pivot >= width {
                            return Err(invalid(
                                index,
                                width,
                                "projection pivot outside active width",
                            ));
                        }
                        for gate in &projection.gates {
                            let valid = match *gate {
                                CoordinateGate::H(bit) | CoordinateGate::Sdg(bit) => bit < width,
                                CoordinateGate::Cx(a, b) => a < width && b < width && a != b,
                            };
                            if !valid {
                                return Err(invalid(
                                    index,
                                    width,
                                    "coordinate gate outside active width",
                                ));
                            }
                        }
                        width -= 1;
                    }
                }
            }
            peak = peak.max(width);
            if width != expected {
                return Err(invalid(index, width, "width transition mismatch"));
            }
        }
        let end = self.operations.len();
        if produced.contains(&false) || records.contains(&false) {
            return Err(invalid(end, width, "missing symbol or record producer"));
        }
        if peak != self.peak {
            return Err(invalid(end, width, "peak width mismatch"));
        }
        Ok(())
    }
}
