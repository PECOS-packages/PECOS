// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

//! Independent overlapping windows with earliest-detector ownership and joint commits.
//!
//! This adapter starts a fresh forward/backward committee for each window. It is
//! distinct from the persistent beam in [`crate::StreamingMode::FixedLag`]. A
//! chronological detector-block partition is explicit; coordinates alone are not
//! assumed to identify measurement rounds.
//!
//! ```
//! use pecos_frontier::{FrontierConfig, MetricMode, SparseDem};
//! use pecos_frontier::windowed::{WindowConfig, WindowedDecoder, ZeroSyndromePolicy};
//! use std::collections::BTreeMap;
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let dem = SparseDem::from_dem_str("error(0.1) D0 D8 L0\n")?;
//! let window = WindowConfig {
//!     detector_blocks: (0..9).map(|d| vec![d]).collect(),
//!     window_blocks: 8,
//!     stride_blocks: 2,
//!     zero_policy: ZeroSyndromePolicy::SkipAndCommitZero,
//!     detector_free_error_owners: BTreeMap::new(),
//! };
//! let config = FrontierConfig {
//!     k: 512, delta: 1e9, score_alpha: 0.8, column_order: None,
//!     merge_indistinguishable: false, bp_score_iterations: 0,
//!     metric_mode: MetricMode::LogSumExpFloat, int_metric_scale: 1024,
//! };
//! let mut decoder = WindowedDecoder::from_sparse_dem(&dem, window, config, 32)?;
//! let result = decoder.decode(&[1, 0, 0, 0, 0, 0, 0, 0, 1])?;
//! assert_eq!(result.predicted.to_u64(), Some(1));
//! # Ok(())
//! # }
//! ```

use crate::{
    CommitteeDirection, DecoderError, FrontierBoundedProgress, FrontierCommittee,
    FrontierCommitteeResult, FrontierConfig, FrontierResumableCommittee, ObsMask, SparseDem,
    deadline_column_order,
};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

/// Policy for a window whose current residual syndrome is zero.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ZeroSyndromePolicy {
    /// Run both committee members, including for a zero syndrome.
    Decode,
    /// Skip DP and commit class zero. This is an explicit approximation: a zero
    /// syndrome need not have logical MAP zero for an arbitrary model.
    SkipAndCommitZero,
}

/// Experiment-specific window geometry and ownership, independent of pruning settings.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WindowConfig {
    /// Chronological blocks, including initial, terminal and empty blocks.
    /// Each global detector ID must occur exactly once. Order within each block
    /// determines the projected model's local detector order.
    pub detector_blocks: Vec<Vec<u32>>,
    /// Maximum number of blocks observed by each window.
    pub window_blocks: usize,
    /// Number of blocks committed by ordinary windows; positive and at most the window size.
    pub stride_blocks: usize,
    /// Explicit zero-residual policy.
    pub zero_policy: ZeroSyndromePolicy,
    /// Optional owner block for each detector-free mechanism, keyed by its
    /// original sparse-DEM mechanism index. Overrides for detector-supported
    /// mechanisms are rejected. Missing detector-free ownership is an error.
    pub detector_free_error_owners: BTreeMap<usize, usize>,
}

/// One independent window and the mapping needed to apply its selected joint class.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Window {
    /// First observed block, inclusive.
    pub start: usize,
    /// Last observed block, exclusive.
    pub stop: usize,
    /// Exclusive commit boundary. The last window commits its entire remaining tail.
    pub commit_stop: usize,
    /// Index in [`WindowPlan::models`]. Identical numerical models share compilation.
    pub model_id: usize,
    /// Local detector ID to global detector ID, in explicit block order.
    pub detector_ids: Vec<u32>,
    /// Ascending global future-detector IDs. Their class bits follow the original observables.
    pub future_detector_ids: Vec<u32>,
    /// Included mechanism indices, in original DEM order.
    pub mechanism_indices: Vec<usize>,
    /// Included mechanisms whose owners are before the commit boundary.
    pub committed_mechanism_indices: Vec<usize>,
}

/// Validated projected models and their global commit mappings.
#[derive(Clone, Debug)]
pub struct WindowPlan {
    config: WindowConfig,
    models: Vec<SparseDem>,
    windows: Vec<Window>,
    num_detectors: usize,
    num_observables: usize,
}

#[derive(Eq, Ord, PartialEq, PartialOrd)]
struct ModelKey(usize, usize, Vec<(u64, Vec<u32>, Vec<u32>)>);

fn invalid(message: impl Into<String>) -> DecoderError {
    DecoderError::InvalidConfiguration(message.into())
}

fn xor_targets(targets: &[u32], dimension: usize) -> Result<Vec<u32>, DecoderError> {
    let mut parity = BTreeSet::new();
    for &target in targets {
        if target as usize >= dimension {
            return Err(invalid(format!(
                "target {target} exceeds dimension {dimension}"
            )));
        }
        if !parity.insert(target) {
            parity.remove(&target);
        }
    }
    Ok(parity.into_iter().collect())
}

impl WindowPlan {
    /// Project independent binary mechanisms without decomposition, probability
    /// fitting or mechanism merging/reordering.
    ///
    /// An error is owned by the earliest block containing a detector in its
    /// XOR-normalized support. A window includes owners in `[start, stop)` and
    /// restricts detector support to that window. For committed errors only,
    /// class bits are the original observables followed by effects on future
    /// detectors. The maximum *joint* class is committed, not per-bit marginal MAP.
    ///
    /// # Errors
    /// Rejects invalid probabilities/targets, ambiguous partitions, invalid window
    /// sizes, missing detector-free owners and unsupported ownership overrides.
    pub fn from_sparse_dem(dem: &SparseDem, config: WindowConfig) -> Result<Self, DecoderError> {
        if config.window_blocks == 0
            || config.stride_blocks == 0
            || config.stride_blocks > config.window_blocks
            || config.detector_blocks.is_empty()
        {
            return Err(invalid(
                "require nonempty blocks and 1 <= stride_blocks <= window_blocks",
            ));
        }
        let mut block_of = vec![None; dem.num_detectors];
        for (block, detectors) in config.detector_blocks.iter().enumerate() {
            for &detector in detectors {
                let position = block_of.get_mut(detector as usize).ok_or_else(|| {
                    invalid(format!("detector D{detector} is outside the DEM dimension"))
                })?;
                if position.replace(block).is_some() {
                    return Err(invalid(format!(
                        "detector D{detector} appears in multiple block positions"
                    )));
                }
            }
        }
        let block_of = block_of
            .into_iter()
            .enumerate()
            .map(|(detector, block)| {
                block.ok_or_else(|| {
                    invalid(format!(
                        "detector D{detector} is missing from detector_blocks"
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mechanisms = dem
            .mechanisms
            .iter()
            .enumerate()
            .map(|(index, (p, d, o))| {
                if !p.is_finite() || !(0.0..=1.0).contains(p) {
                    return Err(invalid(format!(
                        "mechanism {index} probability must be in [0, 1]"
                    )));
                }
                Ok((
                    *p,
                    xor_targets(d, dem.num_detectors)?,
                    xor_targets(o, dem.num_observables)?,
                ))
            })
            .collect::<Result<Vec<_>, DecoderError>>()?;
        for (&index, &owner) in &config.detector_free_error_owners {
            let mechanism = mechanisms
                .get(index)
                .ok_or_else(|| invalid("owner mechanism index is out of range"))?;
            if !mechanism.1.is_empty() || owner >= config.detector_blocks.len() {
                return Err(invalid(
                    "owner overrides require a detector-free mechanism and a valid block",
                ));
            }
        }
        let owners = mechanisms.iter().enumerate().map(|(index, (_, detectors, _))| {
            detectors.iter().map(|&d| block_of[d as usize]).min()
                .or_else(|| config.detector_free_error_owners.get(&index).copied())
                .ok_or_else(|| invalid(format!("detector-free mechanism {index} needs an explicit detector_free_error_owners entry")))
        }).collect::<Result<Vec<_>, _>>()?;
        let mut models = Vec::new();
        let mut windows = Vec::new();
        let mut unique = BTreeMap::new();
        let mut start = 0_usize;
        loop {
            let stop = start
                .saturating_add(config.window_blocks)
                .min(config.detector_blocks.len());
            let commit_stop = if stop == config.detector_blocks.len() {
                stop
            } else {
                start + config.stride_blocks
            };
            let detector_ids = config.detector_blocks[start..stop]
                .iter()
                .flatten()
                .copied()
                .collect::<Vec<_>>();
            let mut local = vec![None; dem.num_detectors];
            for (index, &detector) in detector_ids.iter().enumerate() {
                local[detector as usize] =
                    Some(u32::try_from(index).map_err(|_| invalid("too many local detectors"))?);
            }
            let included = owners
                .iter()
                .enumerate()
                .filter_map(|(index, &owner)| (start <= owner && owner < stop).then_some(index))
                .collect::<Vec<_>>();
            let committed = included
                .iter()
                .copied()
                .filter(|&index| owners[index] < commit_stop)
                .collect::<Vec<_>>();
            let future = committed
                .iter()
                .flat_map(|&index| mechanisms[index].1.iter().copied())
                .filter(|&d| block_of[d as usize] >= commit_stop)
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>();
            let future_index = future
                .iter()
                .enumerate()
                .map(|(index, &d)| (d, index))
                .collect::<BTreeMap<_, _>>();
            let mut projected = Vec::new();
            for &index in &included {
                let (probability, detectors, observables) = &mechanisms[index];
                let mut detectors_local = detectors
                    .iter()
                    .filter_map(|&d| local[d as usize])
                    .collect::<Vec<_>>();
                detectors_local.sort_unstable();
                let mut joint = Vec::new();
                if owners[index] < commit_stop {
                    joint.extend_from_slice(observables);
                    for detector in detectors {
                        if let Some(&future_bit) = future_index.get(detector) {
                            let bit = dem
                                .num_observables
                                .checked_add(future_bit)
                                .and_then(|b| u32::try_from(b).ok())
                                .ok_or_else(|| invalid("joint observable index exceeds u32"))?;
                            joint.push(bit);
                        }
                    }
                    joint.sort_unstable();
                }
                projected.push((*probability, detectors_local, joint));
            }
            let model = SparseDem {
                mechanisms: projected,
                detector_coords: BTreeMap::new(),
                num_detectors: detector_ids.len(),
                num_observables: dem
                    .num_observables
                    .checked_add(future.len())
                    .ok_or_else(|| invalid("joint dimension overflow"))?,
            };
            let key = ModelKey(
                model.num_detectors,
                model.num_observables,
                model
                    .mechanisms
                    .iter()
                    .map(|(p, d, o)| (p.to_bits(), d.clone(), o.clone()))
                    .collect(),
            );
            let model_id = if let Some(&id) = unique.get(&key) {
                id
            } else {
                let id = models.len();
                unique.insert(key, id);
                models.push(model);
                id
            };
            windows.push(Window {
                start,
                stop,
                commit_stop,
                model_id,
                detector_ids,
                future_detector_ids: future,
                mechanism_indices: included,
                committed_mechanism_indices: committed,
            });
            if stop == config.detector_blocks.len() {
                break;
            }
            start += config.stride_blocks;
        }
        Ok(Self {
            config,
            models,
            windows,
            num_detectors: dem.num_detectors,
            num_observables: dem.num_observables,
        })
    }

    /// Validated experiment configuration.
    #[must_use]
    pub fn config(&self) -> &WindowConfig {
        &self.config
    }
    /// Unique numerical models in first-window occurrence order. Coordinates are
    /// omitted; each window's global detector mapping is retained separately.
    #[must_use]
    pub fn models(&self) -> &[SparseDem] {
        &self.models
    }
    /// Ordered window and commit descriptors.
    #[must_use]
    pub fn windows(&self) -> &[Window] {
        &self.windows
    }
    /// Number of original detector inputs.
    #[must_use]
    pub fn num_detectors(&self) -> usize {
        self.num_detectors
    }
    /// Number of original logical observables, before future-detector class bits.
    #[must_use]
    pub fn num_observables(&self) -> usize {
        self.num_observables
    }
}

/// One committed window, including full retained telemetry when DP ran.
#[derive(Clone, Debug, PartialEq)]
pub struct WindowResult {
    /// Index in the plan's ordered windows.
    pub window_index: usize,
    /// Current residual syndrome in the window's local detector order.
    pub observed: Vec<u8>,
    /// Selected joint class, including original logical and future-detector bits.
    pub selected_joint_class: ObsMask,
    /// Full selected result and both members' status/evidence summaries;
    /// absent for an explicit zero skip.
    pub committee: Option<FrontierCommitteeResult>,
    /// Absolute marginal LLR for each original logical bit, folded over retained
    /// joint classes in engine order. Absent for a zero skip; not a selection rule.
    pub logical_abs_llrs: Option<Vec<f64>>,
    /// Number of bounded steps, including phase-boundary calls.
    pub step_calls: usize,
    /// Diagnostic commit trace after this window.
    pub trace_hash: u32,
}

/// A complete independently decoded shot.
#[derive(Clone, Debug, PartialEq)]
pub struct WindowedResult {
    /// XOR of committed original-observable bits.
    pub predicted: ObsMask,
    /// Every processed window, including zero skips.
    pub windows: Vec<WindowResult>,
    /// Per-original-logical minimum absolute retained-mass LLR; positive infinity
    /// when no finite value was seen. Skipped windows are excluded.
    pub min_abs_llrs: Vec<f64>,
    /// First window attaining a strictly smaller LLR, independently for each bit.
    pub weakest_windows: Vec<Option<usize>>,
    /// Final diagnostic commit trace.
    pub trace_hash: u32,
}

/// A failed shot, retaining completed windows rather than substituting a prediction.
#[derive(Debug)]
pub struct WindowDecodeError {
    /// Original core or input error.
    pub error: DecoderError,
    /// Failed window, or `None` for invalid whole-shot input.
    pub failed_window: Option<usize>,
    /// Windows completed before the failure. They are not a complete prediction.
    pub completed_windows: Vec<WindowResult>,
}
impl fmt::Display for WindowDecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.error.fmt(f)
    }
}
impl std::error::Error for WindowDecodeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

/// Model-reusing independent-window committee adapter. No beam or answer is reused.
pub struct WindowedDecoder {
    plan: WindowPlan,
    decoders: Vec<FrontierResumableCommittee>,
    max_columns: usize,
    config: FrontierConfig,
}
impl WindowedDecoder {
    /// Build validated window models and both committee members for each unique model.
    ///
    /// Each model uses its own explicit deadline permutation and plain reverse.
    /// The provided `FrontierConfig.column_order` must be `None`; a global DEM
    /// permutation is not a permutation of every projected model. Other pruning
    /// and arithmetic fields pass through unchanged. Full dropped-mass bookkeeping
    /// remains enabled. A column budget bounds work, not elapsed milliseconds.
    ///
    /// # Errors
    /// Returns model/configuration errors or a zero column-budget error.
    pub fn from_sparse_dem(
        dem: &SparseDem,
        window: WindowConfig,
        config: FrontierConfig,
        max_columns: usize,
    ) -> Result<Self, DecoderError> {
        Self::from_plan(
            WindowPlan::from_sparse_dem(dem, window)?,
            config,
            max_columns,
        )
    }

    /// Construct a decoder from an already validated plan.
    ///
    /// # Errors
    /// Returns the same configuration errors as [`Self::from_sparse_dem`].
    pub fn from_plan(
        plan: WindowPlan,
        config: FrontierConfig,
        max_columns: usize,
    ) -> Result<Self, DecoderError> {
        if max_columns == 0 || config.column_order.is_some() {
            return Err(invalid(
                "require a positive column budget and no global column_order",
            ));
        }
        let decoders = plan
            .models
            .iter()
            .map(|model| {
                let mut per_model = config.clone();
                per_model.column_order = Some(deadline_column_order(model)?);
                Ok(FrontierCommittee::from_sparse_dem(model, per_model)?.into_resumable())
            })
            .collect::<Result<Vec<_>, DecoderError>>()?;
        Ok(Self {
            plan,
            decoders,
            max_columns,
            config,
        })
    }
    /// Auditable geometry, numerical models and commit mappings.
    #[must_use]
    pub fn plan(&self) -> &WindowPlan {
        &self.plan
    }
    /// Configured numerical settings. The global order is `None`; each projected
    /// model uses its own deadline order and plain reverse.
    #[must_use]
    pub fn frontier_config(&self) -> &FrontierConfig {
        &self.config
    }
    /// Maximum DP columns per bounded step, excluding phase-boundary work.
    #[must_use]
    pub fn column_budget(&self) -> usize {
        self.max_columns
    }

    /// Synchronously decode one fresh full detector vector, draining all windows
    /// before returning. Logical truth is not an input. `max_columns` limits
    /// internal committee steps, not the work or duration of this whole-shot call.
    ///
    /// Errors are never retried or recovered. A later call starts fresh attempts
    /// using the compiled models, including after a failed shot.
    ///
    /// # Errors
    /// Returns invalid input or a core window failure, retaining completed windows.
    pub fn decode(&mut self, syndrome: &[u8]) -> Result<WindowedResult, WindowDecodeError> {
        for decoder in &mut self.decoders {
            decoder.cancel();
        }
        if syndrome.len() != self.plan.num_detectors || syndrome.iter().any(|&bit| bit > 1) {
            return Err(WindowDecodeError {
                error: invalid("syndrome dimension or bit value is invalid"),
                failed_window: None,
                completed_windows: Vec::new(),
            });
        }
        let mut residual = syndrome.to_vec();
        let mut predicted = ObsMask::new();
        let mut minimum = vec![f64::INFINITY; self.plan.num_observables];
        let mut weakest = vec![None; self.plan.num_observables];
        let mut trace = 2_166_136_261;
        let mut windows = Vec::new();
        for (index, window) in self.plan.windows.iter().enumerate() {
            let observed = window
                .detector_ids
                .iter()
                .map(|&d| residual[d as usize])
                .collect::<Vec<_>>();
            let mut step_calls = 0;
            let committee = if observed.iter().any(|&b| b != 0)
                || self.plan.config.zero_policy == ZeroSyndromePolicy::Decode
            {
                let decoder = &mut self.decoders[window.model_id];
                let attempt = (|| {
                    decoder.begin(&observed)?;
                    while !matches!(decoder.progress(), Some(FrontierBoundedProgress::Ready)) {
                        decoder.step(self.max_columns)?;
                        step_calls += 1;
                    }
                    decoder.finish()
                })();
                match attempt {
                    Ok(result) => Some(result),
                    Err(error) => {
                        for decoder in &mut self.decoders {
                            decoder.cancel();
                        }
                        return Err(WindowDecodeError {
                            error,
                            failed_window: Some(index),
                            completed_windows: windows,
                        });
                    }
                }
            } else {
                None
            };
            let selected = committee
                .as_ref()
                .map_or_else(ObsMask::new, |r| r.selected.predicted.clone());
            for bit in 0..self.plan.num_observables {
                if selected.get(bit) {
                    predicted.toggle(bit);
                }
            }
            for (bit, &detector) in window.future_detector_ids.iter().enumerate() {
                residual[detector as usize] ^=
                    u8::from(selected.get(self.plan.num_observables + bit));
            }
            let llrs = committee
                .as_ref()
                .map(|r| logical_llrs(r, self.plan.num_observables));
            if let Some(llrs) = &llrs {
                for (bit, &llr) in llrs.iter().enumerate() {
                    if llr < minimum[bit] {
                        minimum[bit] = llr;
                        weakest[bit] = Some(index);
                    }
                }
            }
            let direction = committee.as_ref().map_or(0, |r| match r.direction {
                CommitteeDirection::Forward => 1,
                CommitteeDirection::Backward => 2,
            });
            trace = commit_trace(
                trace,
                index,
                &selected,
                self.plan.models[window.model_id].num_observables,
                direction,
            );
            windows.push(WindowResult {
                window_index: index,
                observed,
                selected_joint_class: selected,
                committee,
                logical_abs_llrs: llrs,
                step_calls,
                trace_hash: trace,
            });
        }
        Ok(WindowedResult {
            predicted,
            windows,
            min_abs_llrs: minimum,
            weakest_windows: weakest,
            trace_hash: trace,
        })
    }
}

fn log_add(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let high = a.max(b);
    high + libm::log1p(libm::exp(a.min(b) - high))
}
fn logical_llrs(result: &FrontierCommitteeResult, width: usize) -> Vec<f64> {
    let mut masses = vec![[f64::NEG_INFINITY; 2]; width];
    for class in &result.selected.logical_masses {
        for (bit, mass) in masses.iter_mut().enumerate() {
            let value = usize::from(class.logical.get(bit));
            mass[value] = log_add(mass[value], class.log_mass);
        }
    }
    masses.iter().map(|m| (m[1] - m[0]).abs()).collect()
}
fn hash_bytes(mut hash: u32, bytes: &[u8]) -> u32 {
    for &byte in bytes {
        hash = (hash ^ u32::from(byte)).wrapping_mul(16_777_619);
    }
    hash
}
fn hash_words(mut hash: u32, words: &[u64]) -> u32 {
    for word in words {
        hash = hash_bytes(hash, &word.to_le_bytes());
    }
    hash
}

/// Diagnostic FNV-1a trace: narrow classes use `(window, class, direction)` as
/// three little-endian u64 words. Classes wider than 64 bits are domain-separated
/// with `FRONTIER-WIDE-CLASS-V1\0`, then window, class width, fixed word count,
/// every little-endian class word and direction. High class bits are never truncated.
fn commit_trace(
    mut hash: u32,
    window: usize,
    class: &ObsMask,
    width: usize,
    direction: u64,
) -> u32 {
    if width <= 64 {
        return hash_words(
            hash,
            &[
                window as u64,
                class.to_u64().expect("narrow class"),
                direction,
            ],
        );
    }
    hash = hash_bytes(hash, b"FRONTIER-WIDE-CLASS-V1\0");
    hash = hash_words(
        hash,
        &[window as u64, width as u64, width.div_ceil(64) as u64],
    );
    for word in 0..width.div_ceil(64) {
        hash = hash_words(hash, &[class.words().get(word).copied().unwrap_or(0)]);
    }
    hash_words(hash, &[direction])
}
