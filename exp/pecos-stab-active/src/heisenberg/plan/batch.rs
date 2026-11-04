// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0.

use super::{Instruction, PlannedOp, SamplingPlan, ShotResult};
use crate::kernels;
use num_complex::Complex64;
use pecos_random::PecosRng;
use rayon::prelude::*;
use std::ops::Range;

pub(crate) fn chunk_shots(peak: usize, lane_budget_bytes: usize) -> usize {
    // plan.rs: HeisenbergProgram::plan_observed bounds max_width by the platform's
    // complex-vector limit; plan/validation.rs: SamplingPlan::validate checks
    // every width <= limit and peak equals the maximum width. Thus this shift
    // cannot overflow for a constructed plan.
    (lane_budget_bytes / (size_of::<Complex64>() << peak)).max(1)
}

impl SamplingPlan {
    /// One shot per seed. Result i is bit-identical to `self.run(seeds[i])` for
    /// every budget. Shots that agree on every rotation sign and projection value
    /// so far share one amplitude vector. Amplitude memory is at most
    /// `max(lane_budget_bytes, 16 * 2^peak)` bytes. Any budget below `16 << peak`,
    /// including 0, runs shots one at a time, at the cost of `run`.
    #[must_use]
    pub fn run_batch(&self, seeds: &[u64], lane_budget_bytes: usize) -> Vec<ShotResult> {
        seeds
            .chunks(chunk_shots(self.peak, lane_budget_bytes))
            .flat_map(|seeds| BatchSampler::new(self, seeds).run())
            .collect()
    }

    /// As [`Self::run_batch`], with chunks run on the rayon global pool. Results
    /// are identical to `run_batch` and returned in seed order. Amplitude memory
    /// is up to the thread count times the `run_batch` bound.
    #[must_use]
    pub fn par_run_batch(&self, seeds: &[u64], lane_budget_bytes: usize) -> Vec<ShotResult> {
        seeds
            .par_chunks(chunk_shots(self.peak, lane_budget_bytes))
            .flat_map_iter(|seeds| BatchSampler::new(self, seeds).run())
            .collect()
    }
}

pub(in crate::heisenberg) struct Shot {
    rng: PecosRng,
    pub(in crate::heisenberg) noise: Vec<bool>,
    pub(in crate::heisenberg) measurements: Vec<bool>,
    records: Vec<bool>,
}

pub(in crate::heisenberg) struct Lane {
    pub(in crate::heisenberg) amplitudes: Vec<Complex64>,
    pub(in crate::heisenberg) shots: Range<usize>,
}

pub(in crate::heisenberg) struct BatchSampler<'a> {
    plan: &'a SamplingPlan,
    pub(in crate::heisenberg) shots: Vec<Shot>,
    pub(in crate::heisenberg) lanes: Vec<Lane>,
    // Membership ranges partition this arena. Each range stays in ascending
    // shot order. Splits retain the majority (false on ties), cloning the
    // minority's amplitudes; stable scratch partitioning allocates no index lists.
    pub(in crate::heisenberg) indices: Vec<usize>,
    scratch: Vec<usize>,
    bits: Vec<bool>,
}

impl<'a> BatchSampler<'a> {
    pub(in crate::heisenberg) fn new(plan: &'a SamplingPlan, seeds: &[u64]) -> Self {
        let shots = seeds
            .iter()
            .map(|&seed| {
                let mut rng = PecosRng::seed_from_u64(seed);
                let mut noise = vec![false; plan.num_noise_symbols];
                for channel in &plan.noise_channels {
                    channel.sample(&mut rng, &mut noise);
                }
                Shot {
                    rng,
                    noise,
                    measurements: vec![false; plan.num_measurements],
                    records: vec![false; plan.num_records],
                }
            })
            .collect();
        // Reserve by actual chunk length, never by the nominal budget quotient.
        let mut lanes = Vec::with_capacity(seeds.len());
        if !seeds.is_empty() {
            let mut amplitudes = Vec::with_capacity(1 << plan.peak);
            amplitudes.push(Complex64::new(1.0, 0.0));
            lanes.push(Lane {
                amplitudes,
                shots: 0..seeds.len(),
            });
        }
        Self {
            plan,
            shots,
            lanes,
            indices: (0..seeds.len()).collect(),
            scratch: vec![0; seeds.len()],
            bits: vec![false; seeds.len()],
        }
    }

    fn run(self) -> Vec<ShotResult> {
        self.run_observed(
            #[cfg(test)]
            |_, _| {},
        )
    }

    pub(in crate::heisenberg) fn run_observed(
        mut self,
        #[cfg(test)] mut observe: impl FnMut(usize, &Self),
    ) -> Vec<ShotResult> {
        for (operation, op) in self.plan.operations.iter().enumerate() {
            let lane_count = self.lanes.len();
            for index in 0..lane_count {
                self.execute(index, op);
            }
            #[cfg(test)]
            observe(operation, &self);
            #[cfg(not(test))]
            let _ = operation;
        }
        self.shots
            .into_iter()
            .map(|shot| {
                let parity = |ids: &Vec<usize>| ids.iter().fold(false, |s, &i| s ^ shot.records[i]);
                let detectors = self.plan.detectors.iter().map(parity).collect();
                let observables = self.plan.observables.iter().map(parity).collect();
                ShotResult {
                    records: shot.records,
                    detectors,
                    observables,
                    peak_active_width: self.plan.peak,
                }
            })
            .collect()
    }

    fn execute(&mut self, index: usize, op: &PlannedOp) {
        match &op.instruction {
            Instruction::Clifford => {}
            Instruction::Rotation {
                angle,
                double,
                parts,
            } => {
                for &id in &self.indices[self.lanes[index].shots.clone()] {
                    let shot = &self.shots[id];
                    self.bits[id] = op.sign.evaluate(&shot.noise, &shot.measurements);
                }
                self.apply_partition(index, |amplitudes, negative| {
                    if *double {
                        kernels::double(amplitudes);
                    }
                    kernels::rotate(
                        amplitudes,
                        if negative { -*angle } else { *angle },
                        parts.flip,
                        parts.sign,
                        parts.phase,
                    );
                });
            }
            Instruction::Measurement {
                case,
                parts,
                symbol,
                record,
                projection,
            } => {
                let lane = &mut self.lanes[index];
                // Compute each sign group's probability on its first shot. The
                // amplitudes stay unchanged until every shot has drawn, so this
                // preserves shot-index order without separate group buffers.
                let mut probabilities = [None; 2];
                for &id in &self.indices[lane.shots.clone()] {
                    let shot = &mut self.shots[id];
                    let negative = op.sign.evaluate(&shot.noise, &shot.measurements);
                    let probability =
                        *probabilities[usize::from(negative)].get_or_insert_with(|| {
                            let phase = if negative { -parts.phase } else { parts.phase };
                            kernels::measurement_probability(
                                &lane.amplitudes,
                                *case,
                                parts.flip,
                                parts.sign,
                                phase,
                            )
                        });
                    let deterministic = probability <= 0.0 || probability >= 1.0;
                    let outcome = if deterministic {
                        probability >= 1.0
                    } else {
                        shot.rng.random_bool(probability)
                    };
                    shot.measurements[*symbol] = outcome;
                    if let Some(ordinal) = record {
                        shot.records[*ordinal] = outcome;
                    }
                    if let Some(projection) = projection {
                        self.bits[id] = projection.value.evaluate(&shot.noise, &shot.measurements);
                    }
                }
                if let Some(projection) = projection {
                    // Fixed gates commute with partitioning the shots. Apply
                    // them before cloning so each gate runs once per input lane.
                    for &gate in &projection.gates {
                        kernels::coordinate_gate(&mut lane.amplitudes, gate);
                    }
                    self.apply_partition(index, |amplitudes, value| {
                        kernels::project(amplitudes, projection.pivot, value);
                        kernels::normalize(amplitudes);
                    });
                }
            }
        }
    }

    fn apply_partition(&mut self, index: usize, mut apply: impl FnMut(&mut Vec<Complex64>, bool)) {
        let range = self.lanes[index].shots.clone();
        let ones = self.indices[range.clone()]
            .iter()
            .filter(|&&id| self.bits[id])
            .count();
        let zeros = range.len() - ones;
        let retained = ones > zeros;
        if ones == 0 || zeros == 0 {
            apply(&mut self.lanes[index].amplitudes, retained);
            return;
        }
        let middle = range.start + ones.max(zeros);
        let mut destinations = [range.start, middle];
        for &id in &self.indices[range.clone()] {
            let group = usize::from(self.bits[id] != retained);
            self.scratch[destinations[group]] = id;
            destinations[group] += 1;
        }
        self.indices[range.clone()].copy_from_slice(&self.scratch[range.clone()]);
        let mut amplitudes = Vec::with_capacity(1 << self.plan.peak);
        amplitudes.extend_from_slice(&self.lanes[index].amplitudes);
        self.lanes[index].shots.end = middle;
        apply(&mut self.lanes[index].amplitudes, retained);
        apply(&mut amplitudes, !retained);
        self.lanes.push(Lane {
            amplitudes,
            shots: middle..range.end,
        });
    }
}
