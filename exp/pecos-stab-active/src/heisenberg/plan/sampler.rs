// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0.

use super::{Instruction, SamplingPlan, ShotResult};
use crate::kernels;
use num_complex::Complex64;
use pecos_random::PecosRng;

/// Reusable shot storage borrowing an immutable plan.
/// Each run allocates only the three vectors owned by its returned `ShotResult`.
pub struct Sampler<'a> {
    plan: &'a SamplingPlan,
    amplitudes: Vec<Complex64>,
    noise: Vec<bool>,
    measurements: Vec<bool>,
    records: Vec<bool>,
}

impl<'a> Sampler<'a> {
    pub(super) fn new(plan: &'a SamplingPlan) -> Self {
        Self {
            plan,
            amplitudes: Vec::with_capacity(1 << plan.peak),
            noise: vec![false; plan.num_noise_symbols],
            measurements: vec![false; plan.num_measurements],
            records: vec![false; plan.num_records],
        }
    }

    /// Reuse the buffers for a shot, drawing noise before measurements.
    #[must_use]
    pub fn run(&mut self, seed: u64) -> ShotResult {
        self.run_observed(seed, |_, _, _| None, |_, _, _| {})
    }

    pub(in crate::heisenberg) fn run_observed<F, O>(
        &mut self,
        seed: u64,
        force: F,
        observe: O,
    ) -> ShotResult
    where
        F: FnMut(f64, usize, Option<usize>) -> Option<bool>,
        O: FnMut(&[Complex64], usize, &[bool]),
    {
        let mut rng = PecosRng::seed_from_u64(seed);
        self.noise.fill(false);
        for channel in &self.plan.noise_channels {
            channel.sample(&mut rng, &mut self.noise);
        }
        self.execute_observed(&mut rng, force, observe)
    }

    #[cfg(test)]
    pub(in crate::heisenberg) fn fixed_noise_observed<F, O>(
        &mut self,
        seed: u64,
        noise: &[bool],
        force: F,
        observe: O,
    ) -> ShotResult
    where
        F: FnMut(f64, usize, Option<usize>) -> Option<bool>,
        O: FnMut(&[Complex64], usize, &[bool]),
    {
        self.noise.copy_from_slice(noise);
        self.execute_observed(&mut PecosRng::seed_from_u64(seed), force, observe)
    }

    fn execute_observed<F, O>(
        &mut self,
        rng: &mut PecosRng,
        mut force: F,
        mut observe: O,
    ) -> ShotResult
    where
        F: FnMut(f64, usize, Option<usize>) -> Option<bool>,
        O: FnMut(&[Complex64], usize, &[bool]),
    {
        self.amplitudes.clear();
        self.amplitudes.push(Complex64::new(1.0, 0.0));
        self.measurements.fill(false);
        self.records.fill(false);
        for (op, &width) in self.plan.operations.iter().zip(&self.plan.widths) {
            let negative = op.sign.evaluate(&self.noise, &self.measurements);
            match &op.instruction {
                Instruction::Clifford => {}
                Instruction::Rotation {
                    angle,
                    double,
                    parts,
                } => {
                    if *double {
                        kernels::double(&mut self.amplitudes);
                    }
                    kernels::rotate(
                        &mut self.amplitudes,
                        if negative { -*angle } else { *angle },
                        parts.flip,
                        parts.sign,
                        parts.phase,
                    );
                }
                Instruction::Measurement {
                    case,
                    parts,
                    symbol,
                    record,
                    projection,
                } => {
                    let phase = if negative { -parts.phase } else { parts.phase };
                    let probability = kernels::measurement_probability(
                        &self.amplitudes,
                        *case,
                        parts.flip,
                        parts.sign,
                        phase,
                    );
                    let forced = force(probability, *symbol, *record);
                    let deterministic = probability <= 0.0 || probability >= 1.0;
                    let outcome = if deterministic {
                        probability >= 1.0
                    } else {
                        forced.unwrap_or_else(|| rng.random_bool(probability))
                    };
                    self.measurements[*symbol] = outcome;
                    if let Some(ordinal) = record {
                        self.records[*ordinal] = outcome;
                    }
                    if let Some(projection) = projection {
                        for &gate in &projection.gates {
                            kernels::coordinate_gate(&mut self.amplitudes, gate);
                        }
                        let value = projection.value.evaluate(&self.noise, &self.measurements);
                        kernels::project(&mut self.amplitudes, projection.pivot, value);
                        kernels::normalize(&mut self.amplitudes);
                    }
                }
            }
            observe(&self.amplitudes, width, &self.measurements);
        }
        let parity = |ids: &Vec<usize>| ids.iter().fold(false, |s, &i| s ^ self.records[i]);
        ShotResult {
            records: self.records.clone(),
            detectors: self.plan.detectors.iter().map(parity).collect(),
            observables: self.plan.observables.iter().map(parity).collect(),
            peak_active_width: self.plan.peak,
        }
    }
}
