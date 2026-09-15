// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! Sequential windows with whole-component commit and global residual carry.

use pecos_decoder_core::streaming::StreamingDecoder;
use pecos_decoder_core::window::{CommitColumn, CommitWindow, StructuredDem, min_buffer_rounds};
use pecos_decoder_core::{DecoderError, EdgeDecoder, ObservableDecoder, obs_mask::ObsMask};
use std::collections::BTreeSet;
use std::ops::Range;

pub use crate::beam_windowed::{BeamSearchConfig, BeamSearchWindowedDecoder, BeamWindowConfig};

/// Window engine parameters. The buffer is explicit; `buffer = d` is recommended.
#[derive(Clone, Copy, Debug)]
pub struct WindowedConfig {
    /// Number of commit rounds. Zero estimates the code distance from the model.
    pub step: usize,
    /// Forward context in rounds, at least [`min_buffer_rounds`].
    pub buffer: usize,
}

struct Window<D> {
    model: CommitWindow,
    decoder: D,
    commit: Range<u32>,
    end: u32,
    last: bool,
}

/// Per-shot diagnostics, reset with the decoder.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WindowDiagnostics {
    /// Number of forced commits containing at least one future edge.
    pub forced_future_components: usize,
    /// Number of deferred components across all processed windows.
    pub deferred_components: usize,
    /// Global column indices committed so far, retaining repeated toggles.
    pub committed_columns: Vec<usize>,
}

/// Finite, predeclared window decoder. Batch and incremental input share one residual.
///
/// Correlation groups are retained within each window; correlation evidence does not
/// cross windows. Finite windows need not reproduce the monolithic correction.
pub struct StreamingWindowedDecoder<D> {
    windows: Vec<Window<D>>,
    columns: Vec<CommitColumn>,
    times: Vec<u32>,
    raw: Vec<u8>,
    residual: Vec<u8>,
    next_decode: usize,
    accumulated: u64,
    diagnostics: WindowDiagnostics,
}

impl<D: EdgeDecoder> StreamingWindowedDecoder<D> {
    /// Construct from flattened DEM text and a checked-window decoder factory.
    ///
    /// # Errors
    /// Returns an error for invalid models, incomplete or unsolvable windows, or factory errors.
    pub fn from_dem<F>(dem: &str, config: WindowedConfig, factory: F) -> Result<Self, DecoderError>
    where
        F: FnMut(&CommitWindow) -> Result<D, DecoderError>,
    {
        Self::from_structured_dem(&StructuredDem::from_dem_str(dem)?, config, factory)
    }

    /// Construct from a structured model without a top-level render/parse cycle.
    ///
    /// # Errors
    /// Returns an error for invalid models, incomplete or unsolvable windows, or factory errors.
    pub fn from_structured_dem<F>(
        dem: &StructuredDem,
        config: WindowedConfig,
        mut factory: F,
    ) -> Result<Self, DecoderError>
    where
        F: FnMut(&CommitWindow) -> Result<D, DecoderError>,
    {
        dem.ensure_observables_fit_u64()?;
        let times = dem.commit_detector_times()?;
        let columns = dem.commit_columns()?;
        let minimum = min_buffer_rounds(dem)?;
        if config.buffer < minimum as usize {
            return Err(DecoderError::InvalidConfiguration(format!(
                "buffer {} is below max_forward_span {minimum}",
                config.buffer
            )));
        }
        let total = times.iter().copied().max().map_or(0, |t| t + 1);
        let step = if config.step == 0 {
            let stabilizers = dem.num_detectors.checked_div(total as usize).unwrap_or(0);
            ((stabilizers as f64).sqrt().ceil() as usize).max(3)
        } else {
            config.step
        };
        let step = u32::try_from(step)
            .map_err(|_| DecoderError::InvalidConfiguration("step exceeds u32::MAX".into()))?;
        let buffer = u32::try_from(config.buffer)
            .map_err(|_| DecoderError::InvalidConfiguration("buffer exceeds u32::MAX".into()))?;
        let mut windows = Vec::new();
        let mut start = 0u32;
        while start < total {
            let end = start.saturating_add(step);
            let last = u64::from(end) + u64::from(step) > u64::from(total);
            let end = if last { total } else { end };
            let rows = start.saturating_sub(step)..if last {
                total
            } else {
                end.saturating_add(buffer).min(total)
            };
            let model = dem
                .commit_window(rows.clone(), start..end)
                .map_err(|error| {
                    DecoderError::InvalidConfiguration(format!("window {}: {error}", windows.len()))
                })?;
            let decoder = factory(&model)?;
            windows.push(Window {
                model,
                decoder,
                commit: start..end,
                end: rows.end,
                last,
            });
            if last {
                break;
            }
            start = end;
        }
        Ok(Self {
            windows,
            columns,
            times,
            raw: vec![0; dem.num_detectors],
            residual: vec![0; dem.num_detectors],
            next_decode: 0,
            accumulated: 0,
            diagnostics: WindowDiagnostics::default(),
        })
    }

    /// Number of windows, including the merged tail exactly once.
    #[must_use]
    pub fn num_windows(&self) -> usize {
        self.windows.len()
    }

    /// Index of the next undecoded window.
    #[must_use]
    pub fn next_window(&self) -> usize {
        self.next_decode
    }

    /// Current global residual, `raw XOR incidence(committed columns)`.
    #[must_use]
    pub fn residual(&self) -> &[u8] {
        &self.residual
    }

    /// Diagnostics for the current shot.
    #[must_use]
    pub fn diagnostics(&self) -> &WindowDiagnostics {
        &self.diagnostics
    }

    /// Reset and load a complete shot. Windows are advanced by [`Self::decode_next_window`].
    ///
    /// # Errors
    /// Returns an error for incorrect width or non-binary input, before changing state.
    pub fn start_shot(&mut self, syndrome: &[u8]) -> Result<(), DecoderError> {
        if syndrome.len() != self.raw.len() {
            return Err(DecoderError::InvalidDimensions {
                expected: self.raw.len(),
                actual: syndrome.len(),
            });
        }
        if syndrome.iter().any(|&s| s > 1) {
            return Err(DecoderError::InvalidSyndrome(
                "window input must be binary".into(),
            ));
        }
        self.reset();
        self.raw.copy_from_slice(syndrome);
        self.residual.copy_from_slice(syndrome);
        Ok(())
    }

    /// Decode and commit the next window. Incidence is checked before state advances.
    ///
    /// # Errors
    /// Returns an error naming the window and mismatching row if the correction is incomplete.
    pub fn decode_next_window(&mut self) -> Result<u64, DecoderError> {
        let index = self.next_decode;
        let Some(window) = self.windows.get_mut(index) else {
            return Ok(0);
        };
        let syndrome: Vec<_> = window
            .model
            .local_to_global_detector
            .iter()
            .map(|&d| self.residual[d as usize])
            .collect();
        let selected = window.decoder.decode_to_edges(&syndrome)?;
        let mut incidence = vec![0; syndrome.len()];
        let mut incident = vec![Vec::new(); syndrome.len()];
        for (position, &edge_index) in selected.iter().enumerate() {
            let edge = window.model.edges.get(edge_index).ok_or_else(|| {
                DecoderError::DecodingFailed(format!("window {index}: invalid edge {edge_index}"))
            })?;
            for node in [Some(edge.node1), edge.node2].into_iter().flatten() {
                incidence[node as usize] ^= 1;
                incident[node as usize].push(position);
            }
        }
        if let Some(row) = incidence.iter().zip(&syndrome).position(|(a, b)| a != b) {
            return Err(DecoderError::DecodingFailed(format!(
                "window {index}: incidence mismatch at row {row} (global detector {})",
                window.model.local_to_global_detector[row]
            )));
        }
        let mut visited = vec![false; selected.len()];
        let mut committed = Vec::new();
        let mut forced_future = 0;
        let mut deferred = 0;
        for first in 0..selected.len() {
            if visited[first] {
                continue;
            }
            visited[first] = true;
            let mut stack = vec![first];
            let mut resolved = Vec::new();
            let mut sigma = BTreeSet::new();
            let mut future = false;
            while let Some(position) = stack.pop() {
                let edge = &window.model.edges[selected[position]];
                future |= edge.future;
                let column = if edge.future {
                    edge.rep_any
                } else {
                    edge.rep_nonprojected.ok_or_else(|| {
                        DecoderError::InternalError(format!(
                            "window {index}: non-future edge lacks a non-projected representative"
                        ))
                    })?
                };
                resolved.push(column);
                for &detector in &self.columns[column].detectors {
                    if !sigma.insert(detector) {
                        sigma.remove(&detector);
                    }
                }
                for node in [Some(edge.node1), edge.node2].into_iter().flatten() {
                    for &next in &incident[node as usize] {
                        if !visited[next] {
                            visited[next] = true;
                            stack.push(next);
                        }
                    }
                }
            }
            let forced = sigma
                .iter()
                .any(|&d| self.times[d as usize] < window.commit.start);
            let ordinary = !future
                && sigma
                    .iter()
                    .all(|&d| self.times[d as usize] < window.commit.end);
            if window.last || forced || ordinary {
                forced_future += usize::from(forced && future);
                committed.extend(resolved);
            } else {
                deferred += 1;
            }
        }
        let mut obs = 0;
        for &id in &committed {
            for &d in &self.columns[id].detectors {
                self.residual[d as usize] ^= 1;
            }
            for &o in &self.columns[id].observables {
                obs ^= 1u64 << o;
            }
        }
        self.diagnostics.committed_columns.extend(committed);
        self.diagnostics.forced_future_components += forced_future;
        self.diagnostics.deferred_components += deferred;
        self.accumulated ^= obs;
        self.next_decode += 1;
        Ok(obs)
    }
}

impl<D: EdgeDecoder> ObservableDecoder for StreamingWindowedDecoder<D> {
    fn num_detectors(&self) -> Option<usize> {
        Some(self.raw.len())
    }

    fn decode_obs(&mut self, syndrome: &[u8]) -> Result<ObsMask, DecoderError> {
        self.start_shot(syndrome)?;
        self.finish()?;
        Ok(ObsMask::from_u64(self.accumulated))
    }
}

impl<D: EdgeDecoder> StreamingDecoder for StreamingWindowedDecoder<D> {
    fn feed_round(&mut self, round: usize, detectors: &[(u32, u8)]) -> Result<u64, DecoderError> {
        if let Some(window) = self
            .next_decode
            .checked_sub(1)
            .and_then(|index| self.windows.get(index))
            && round < window.end as usize
        {
            return Err(DecoderError::InvalidSyndrome(format!(
                "round {round} has already been decoded through round {}",
                window.end - 1
            )));
        }
        for &(detector, value) in detectors {
            if self
                .times
                .get(detector as usize)
                .is_none_or(|&time| time as usize != round)
                || value > 1
            {
                return Err(DecoderError::InvalidSyndrome(format!(
                    "invalid detector {detector} for round {round}"
                )));
            }
        }
        for &(detector, value) in detectors {
            let row = detector as usize;
            self.residual[row] ^= self.raw[row] ^ value;
            self.raw[row] = value;
        }
        let mut obs = 0;
        while self
            .windows
            .get(self.next_decode)
            .is_some_and(|w| !w.last && round >= (w.end - 1) as usize)
        {
            obs ^= self.decode_next_window()?;
        }
        Ok(obs)
    }

    fn finish(&mut self) -> Result<u64, DecoderError> {
        let mut obs = 0;
        while self.next_decode < self.windows.len() {
            obs ^= self.decode_next_window()?;
        }
        Ok(obs)
    }

    fn accumulated_obs(&self) -> u64 {
        self.accumulated
    }

    fn reset(&mut self) {
        self.raw.fill(0);
        self.residual.fill(0);
        self.next_decode = 0;
        self.accumulated = 0;
        self.diagnostics = WindowDiagnostics::default();
    }
}
