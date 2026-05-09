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

//! Sliding-window decoder for real-time surface code decoding.
//!
//! Two modes:
//!
//! - **Non-overlapping (`buf=0`)**: sub-DEM per window, any inner decoder,
//!   observable XOR across windows. Converges to ~1.03x penalty at large r
//!   (matching Tan et al.). Inner decoder is pluggable via factory.
//!
//! - **Overlapping (`buf>0`)**: uses `UfDecoder` for edge tracking. Each
//!   window is extended by buffer rounds for matching context. Only corrections
//!   with both endpoints in the core region are committed. No artificial defect
//!   injection — the buffer just provides graph context (Tan et al.).
//!
//! Reference: Tan et al., PRX Quantum 2023 (arXiv:2209.09219).

use pecos_decoder_core::ObservableDecoder;
use pecos_decoder_core::correlated_decoder::EdgeTrackingDecoder;
use pecos_decoder_core::dem::DemMatchingGraph;
use pecos_decoder_core::errors::DecoderError;
use std::fmt::Write as _;

/// Configuration for the windowed decoder.
#[derive(Debug, Clone, Copy, Default)]
pub struct WindowedConfig {
    /// Commit rounds per window (step size). 0 = auto (code distance).
    pub step_size: usize,
    /// Buffer rounds on each side of the core. 0 = non-overlapping.
    /// Recommended: set equal to code distance for near-zero penalty.
    pub buffer_size: usize,
    /// Half-width of Type-2 seam windows in rounds. 0 = auto (step/2).
    pub seam_half_width: usize,
    /// Extend core by this many layers into the buffer on each side.
    /// Committed edges can touch the extended core, capturing more
    /// boundary corrections. 0 = strict core only (default).
    pub core_extend: usize,
    /// Maximum edge weight for Phase-1 commit. Only correction edges
    /// with weight below this are committed (high-confidence corrections).
    /// 0.0 = no threshold (commit all core edges, default).
    pub commit_weight_max: f64,
}

// =============================================================================
// Non-overlapping windowed decoder (buf=0)
// =============================================================================

/// Pre-built window with a generic inner decoder.
struct PrebuiltWindow {
    decoder: Box<dyn ObservableDecoder>,
    local_to_global: Vec<u32>,
    num_local: usize,
}

/// Non-overlapping windowed decoder. Any `ObservableDecoder` as inner decoder.
pub struct WindowedDecoder {
    windows: Vec<PrebuiltWindow>,
}

impl WindowedDecoder {
    /// Create from a DEM string with a decoder factory.
    ///
    /// # Errors
    ///
    /// Returns `DecoderError` if the DEM is malformed or the factory fails.
    pub fn from_dem<F>(
        dem: &str,
        config: WindowedConfig,
        mut decoder_factory: F,
    ) -> Result<Self, DecoderError>
    where
        F: FnMut(&str) -> Result<Box<dyn ObservableDecoder>, DecoderError>,
    {
        let (det_times, num_detectors, step_size, total_t) = parse_dem_params(dem, &config)?;
        let mut windows = Vec::new();
        let mut t_start = 0.0f64;

        while t_start < total_t {
            let is_last = t_start + 2.0 * step_size as f64 > total_t;
            let t_end = if is_last {
                total_t + 1.0
            } else {
                t_start + step_size as f64
            };

            let (local_to_global, window_dem) =
                extract_window_dem(dem, &det_times, num_detectors, t_start, t_end);

            let num_local = local_to_global.len();
            if num_local > 0 && !window_dem.is_empty() {
                let decoder = decoder_factory(&window_dem)?;
                windows.push(PrebuiltWindow {
                    decoder,
                    local_to_global,
                    num_local,
                });
            }

            t_start += step_size as f64;
        }

        Ok(Self { windows })
    }

    /// Number of windows.
    #[must_use]
    pub fn num_windows(&self) -> usize {
        self.windows.len()
    }
}

impl ObservableDecoder for WindowedDecoder {
    fn decode_to_observables(&mut self, syndrome: &[u8]) -> Result<u64, DecoderError> {
        let mut obs_mask = 0u64;
        for window in &mut self.windows {
            let mut window_syn = vec![0u8; window.num_local];
            for (local_id, &global_id) in window.local_to_global.iter().enumerate() {
                let gid = global_id as usize;
                if gid < syndrome.len() {
                    window_syn[local_id] = syndrome[gid];
                }
            }
            obs_mask ^= window.decoder.decode_to_observables(&window_syn)?;
        }
        Ok(obs_mask)
    }
}

// =============================================================================
// Overlapping windowed decoder (buf>0, Tan et al.)
// =============================================================================

/// Pre-built overlapping window with an edge-tracking inner decoder.
struct OverlappingWindow<D> {
    decoder: D,
    local_to_global: Vec<u32>,
    /// Per local detector: true = core region, false = buffer.
    is_core: Vec<bool>,
    num_local: usize,
}

/// Overlapping windowed decoder using any `EdgeTrackingDecoder` for edge tracking.
///
/// Each window is extended by buffer rounds for matching context.
/// Only core corrections are committed; buffer corrections are discarded.
pub struct OverlappingWindowedDecoder<D> {
    windows: Vec<OverlappingWindow<D>>,
}

impl<D: EdgeTrackingDecoder> OverlappingWindowedDecoder<D> {
    /// Create from a DEM string with a factory for the inner decoder.
    ///
    /// # Errors
    ///
    /// Returns `DecoderError` if the DEM is malformed or the factory fails.
    pub fn from_dem<F>(
        dem: &str,
        config: WindowedConfig,
        mut factory: F,
    ) -> Result<Self, DecoderError>
    where
        F: FnMut(&str) -> Result<D, DecoderError>,
    {
        let (det_times, num_detectors, step_size, total_t) = parse_dem_params(dem, &config)?;
        let buffer_size = config.buffer_size;
        let mut windows = Vec::new();
        let mut t_start = 0.0f64;

        while t_start < total_t {
            let is_last = t_start + 2.0 * step_size as f64 > total_t;
            let t_core_end = if is_last {
                total_t + 1.0
            } else {
                t_start + step_size as f64
            };
            let t_win_start = (t_start - buffer_size as f64).max(0.0);
            let t_win_end = if is_last {
                total_t + 1.0
            } else {
                t_core_end + buffer_size as f64
            };

            let (local_to_global, window_dem) =
                extract_window_dem(dem, &det_times, num_detectors, t_win_start, t_win_end);

            let ext = config.core_extend as f64;
            let is_core: Vec<bool> = local_to_global
                .iter()
                .map(|&gid| {
                    let t = det_times[gid as usize];
                    t >= (t_start - ext) && t < (t_core_end + ext)
                })
                .collect();

            let num_local = local_to_global.len();
            if num_local > 0 && !window_dem.is_empty() {
                let decoder = factory(&window_dem)?;
                windows.push(OverlappingWindow {
                    decoder,
                    local_to_global,
                    is_core,
                    num_local,
                });
            }

            t_start += step_size as f64;
        }

        Ok(Self { windows })
    }

    /// Number of windows.
    #[must_use]
    pub fn num_windows(&self) -> usize {
        self.windows.len()
    }
}

impl<D: EdgeTrackingDecoder> ObservableDecoder for OverlappingWindowedDecoder<D> {
    fn decode_to_observables(&mut self, syndrome: &[u8]) -> Result<u64, DecoderError> {
        let mut obs_mask = 0u64;

        for window in &mut self.windows {
            let mut window_syn = vec![0u8; window.num_local];
            for (local_id, &global_id) in window.local_to_global.iter().enumerate() {
                let gid = global_id as usize;
                if gid < syndrome.len() {
                    window_syn[local_id] = syndrome[gid];
                }
            }

            // Use MatchingDecoder trait for edge tracking.
            let (_, matched_edges) = window.decoder.decode_with_matching(&window_syn)?;

            let boundary = window.num_local as u32;
            for &edge_idx in &matched_edges {
                let n1 = window.decoder.edge_node1(edge_idx);
                let n2 = window.decoder.edge_node2(edge_idx);

                let n1_core = n1 >= boundary
                    || ((n1 as usize) < window.is_core.len() && window.is_core[n1 as usize]);
                let n2_core = n2 >= boundary
                    || ((n2 as usize) < window.is_core.len() && window.is_core[n2 as usize]);

                if n1_core && n2_core {
                    obs_mask ^= window.decoder.edge_obs_mask(edge_idx);
                }
            }
        }

        Ok(obs_mask)
    }
}

// =============================================================================
// Sandwich windowed decoder (Tan et al. two-phase)
// =============================================================================

/// Sandwich windowed decoder: two-phase decoding for reduced boundary penalty.
///
/// Phase 1 (Type-1): Overlapping windows with core-only commit, same as
/// `OverlappingWindowedDecoder`. Independent, can run in parallel.
///
/// Phase 2 (Type-2): Small seam windows at core boundaries decode the
/// residual syndrome left by Type-1. The residual is computed as
/// `original XOR correction_effect` where `correction_effect` tracks
/// which syndrome bits were flipped by Type-1's committed edges.
///
/// This gives Type-2 bidirectional boundary information from both
/// flanking Type-1 windows, reducing the boundary penalty.
pub struct SandwichWindowedDecoder<D> {
    type1_windows: Vec<OverlappingWindow<D>>,
    residual_decoder: Box<dyn ObservableDecoder>,
    num_detectors: usize,
    commit_weight_max: f64,
}

impl<D: EdgeTrackingDecoder> SandwichWindowedDecoder<D> {
    /// Create from a DEM string with factories for Phase-1 and Phase-2 decoders.
    ///
    /// # Errors
    ///
    /// Returns `DecoderError` if the DEM is malformed or factories fail.
    pub fn from_dem<F1, F2>(
        dem: &str,
        config: WindowedConfig,
        mut phase1_factory: F1,
        mut phase2_factory: F2,
    ) -> Result<Self, DecoderError>
    where
        F1: FnMut(&str) -> Result<D, DecoderError>,
        F2: FnMut(&str) -> Result<Box<dyn ObservableDecoder>, DecoderError>,
    {
        let (det_times, num_detectors, step_size, total_t) = parse_dem_params(dem, &config)?;
        let buffer_size = config.buffer_size;

        let mut type1_windows = Vec::new();
        let mut t_start = 0.0f64;

        while t_start < total_t {
            let is_last = t_start + 2.0 * step_size as f64 > total_t;
            let t_core_end = if is_last {
                total_t + 1.0
            } else {
                t_start + step_size as f64
            };
            let t_win_start = (t_start - buffer_size as f64).max(0.0);
            let t_win_end = if is_last {
                total_t + 1.0
            } else {
                t_core_end + buffer_size as f64
            };

            let (local_to_global, window_dem) =
                extract_window_dem(dem, &det_times, num_detectors, t_win_start, t_win_end);

            let ext = config.core_extend as f64;
            let is_core: Vec<bool> = local_to_global
                .iter()
                .map(|&gid| {
                    let t = det_times[gid as usize];
                    t >= (t_start - ext) && t < (t_core_end + ext)
                })
                .collect();

            let num_local = local_to_global.len();
            if num_local > 0 && !window_dem.is_empty() {
                let decoder = phase1_factory(&window_dem)?;
                type1_windows.push(OverlappingWindow {
                    decoder,
                    local_to_global,
                    is_core,
                    num_local,
                });
            }

            t_start += step_size as f64;
        }

        let residual_decoder = phase2_factory(dem)?;

        Ok(Self {
            type1_windows,
            residual_decoder,
            num_detectors,
            commit_weight_max: config.commit_weight_max,
        })
    }

    /// Number of Type-1 windows.
    #[must_use]
    pub fn num_windows(&self) -> usize {
        self.type1_windows.len()
    }

    /// Decode with parallel Phase-1 windows using rayon.
    ///
    /// Requires `D: Send` for thread safety. Phase-1 windows run on rayon's
    /// thread pool; Phase-2 residual runs sequentially after.
    ///
    /// # Errors
    ///
    /// Returns `DecoderError` if any window decoder fails.
    pub fn decode_parallel(&mut self, syndrome: &[u8]) -> Result<u64, DecoderError>
    where
        D: Send,
    {
        use rayon::prelude::*;

        let commit_weight_max = self.commit_weight_max;
        let num_detectors = self.num_detectors;

        // Phase 1: Decode Type-1 windows in parallel.
        let window_results: Result<Vec<_>, DecoderError> = self
            .type1_windows
            .par_iter_mut()
            .map(|window| {
                let mut window_syn = vec![0u8; window.num_local];
                for (local_id, &global_id) in window.local_to_global.iter().enumerate() {
                    let gid = global_id as usize;
                    if gid < syndrome.len() {
                        window_syn[local_id] = syndrome[gid];
                    }
                }

                let (_, matched_edges) = window.decoder.decode_with_matching(&window_syn)?;

                let mut obs = 0u64;
                let mut corrections: Vec<(usize, u8)> = Vec::new();
                let boundary = window.num_local as u32;

                for &edge_idx in &matched_edges {
                    let n1 = window.decoder.edge_node1(edge_idx);
                    let n2 = window.decoder.edge_node2(edge_idx);

                    let n1_core = n1 >= boundary
                        || ((n1 as usize) < window.is_core.len() && window.is_core[n1 as usize]);
                    let n2_core = n2 >= boundary
                        || ((n2 as usize) < window.is_core.len() && window.is_core[n2 as usize]);

                    let weight_ok = commit_weight_max <= 0.0
                        || window.decoder.edge_weight(edge_idx) <= commit_weight_max;

                    if n1_core && n2_core && weight_ok {
                        obs ^= window.decoder.edge_obs_mask(edge_idx);
                        if (n1 as usize) < window.num_local {
                            corrections.push((window.local_to_global[n1 as usize] as usize, 1));
                        }
                        if (n2 as usize) < window.num_local {
                            corrections.push((window.local_to_global[n2 as usize] as usize, 1));
                        }
                    }
                }

                Ok((obs, corrections))
            })
            .collect();

        // Merge results (XOR is order-independent).
        let mut obs_mask = 0u64;
        let mut correction_effect = vec![0u8; num_detectors];
        for (window_obs, corrections) in window_results? {
            obs_mask ^= window_obs;
            for (gid, bit) in corrections {
                correction_effect[gid] ^= bit;
            }
        }

        // Phase 2: Residual decode (sequential).
        let mut residual_syn = vec![0u8; num_detectors];
        for (i, &s) in syndrome.iter().enumerate() {
            if i < num_detectors {
                residual_syn[i] = s ^ correction_effect[i];
            }
        }
        obs_mask ^= self.residual_decoder.decode_to_observables(&residual_syn)?;

        Ok(obs_mask)
    }
}

impl<D: EdgeTrackingDecoder> ObservableDecoder for SandwichWindowedDecoder<D> {
    fn decode_to_observables(&mut self, syndrome: &[u8]) -> Result<u64, DecoderError> {
        let mut obs_mask = 0u64;
        let mut correction_effect = vec![0u8; self.num_detectors];
        let commit_weight_max = self.commit_weight_max;

        // Phase 1: Decode Type-1 windows.
        for window in &mut self.type1_windows {
            let mut window_syn = vec![0u8; window.num_local];
            for (local_id, &global_id) in window.local_to_global.iter().enumerate() {
                let gid = global_id as usize;
                if gid < syndrome.len() {
                    window_syn[local_id] = syndrome[gid];
                }
            }

            let (_, matched_edges) = window.decoder.decode_with_matching(&window_syn)?;

            let boundary = window.num_local as u32;
            for &edge_idx in &matched_edges {
                let n1 = window.decoder.edge_node1(edge_idx);
                let n2 = window.decoder.edge_node2(edge_idx);

                let n1_core = n1 >= boundary
                    || ((n1 as usize) < window.is_core.len() && window.is_core[n1 as usize]);
                let n2_core = n2 >= boundary
                    || ((n2 as usize) < window.is_core.len() && window.is_core[n2 as usize]);

                let weight_ok = commit_weight_max <= 0.0
                    || window.decoder.edge_weight(edge_idx) <= commit_weight_max;

                if n1_core && n2_core && weight_ok {
                    obs_mask ^= window.decoder.edge_obs_mask(edge_idx);

                    if (n1 as usize) < window.num_local {
                        let gid = window.local_to_global[n1 as usize] as usize;
                        correction_effect[gid] ^= 1;
                    }
                    if (n2 as usize) < window.num_local {
                        let gid = window.local_to_global[n2 as usize] as usize;
                        correction_effect[gid] ^= 1;
                    }
                }
            }
        }

        // Phase 2: Decode residual syndrome on the full graph.
        let mut residual_syn = vec![0u8; self.num_detectors];
        for (i, &s) in syndrome.iter().enumerate() {
            if i < self.num_detectors {
                residual_syn[i] = s ^ correction_effect[i];
            }
        }
        obs_mask ^= self.residual_decoder.decode_to_observables(&residual_syn)?;

        Ok(obs_mask)
    }
}

// =============================================================================
// Shared helpers
// =============================================================================

/// Parse DEM parameters for windowing.
fn parse_dem_params(
    dem: &str,
    config: &WindowedConfig,
) -> Result<(Vec<f64>, usize, usize, f64), DecoderError> {
    let graph = DemMatchingGraph::from_dem_str(dem)?;
    let num_detectors = graph.num_detectors;

    let mut det_times = vec![0.0f64; num_detectors];
    let mut max_time = 0.0f64;
    for (i, coord) in graph.detector_coords.iter().enumerate() {
        if let Some(c) = coord {
            let t = c.get(2).copied().unwrap_or(0.0);
            if i < det_times.len() {
                det_times[i] = t;
            }
            if t > max_time {
                max_time = t;
            }
        }
    }

    let num_rounds = (max_time + 1.0) as usize;
    let num_stab = num_detectors
        .checked_div(num_rounds)
        .unwrap_or(num_detectors);
    let d_est = ((num_stab as f64).sqrt().ceil() as usize).max(3);
    let step_size = if config.step_size > 0 {
        config.step_size
    } else {
        d_est
    };
    let total_t = num_rounds as f64;

    Ok((det_times, num_detectors, step_size, total_t))
}

/// Extract a window sub-DEM by filtering the original DEM text.
///
/// Detectors in `[t_start, t_end)` are included and remapped to local IDs.
/// Detectors outside the window are dropped from error mechanisms, creating
/// implicit boundary edges.
fn extract_window_dem(
    dem: &str,
    det_times: &[f64],
    num_det: usize,
    t_start: f64,
    t_end: f64,
) -> (Vec<u32>, String) {
    let mut in_window = vec![false; num_det];
    let mut local_to_global: Vec<u32> = Vec::new();
    let mut global_to_local: Vec<Option<u32>> = vec![None; num_det];

    for (i, &t) in det_times.iter().enumerate() {
        if t >= t_start && t < t_end {
            in_window[i] = true;
            global_to_local[i] = Some(local_to_global.len() as u32);
            local_to_global.push(i as u32);
        }
    }

    let mut out = String::new();

    for line in dem.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        if trimmed.starts_with("error(") {
            let Some(close) = trimmed.find(')') else {
                continue;
            };
            let prob_str = &trimmed[6..close];
            let rest = &trimmed[close + 1..];
            let tokens: Vec<&str> = rest.split_whitespace().collect();

            // Split by ^ into decomposed segments.
            let mut segments: Vec<Vec<&str>> = vec![Vec::new()];
            for tok in &tokens {
                if *tok == "^" {
                    segments.push(Vec::new());
                } else {
                    segments.last_mut().unwrap().push(tok);
                }
            }

            let mut remapped_segments: Vec<String> = Vec::new();
            for seg in &segments {
                let mut seg_dets: Vec<String> = Vec::new();
                let mut seg_obs: Vec<String> = Vec::new();
                let mut seg_any_in = false;

                for tok in seg {
                    if let Some(d_str) = tok.strip_prefix('D') {
                        if let Ok(d) = d_str.parse::<usize>()
                            && d < num_det
                            && in_window[d]
                        {
                            seg_any_in = true;
                            if let Some(local) = global_to_local[d] {
                                seg_dets.push(format!("D{local}"));
                            }
                        }
                    } else if tok.starts_with('L') {
                        seg_obs.push((*tok).to_string());
                    }
                }

                if seg_any_in {
                    let mut seg_str = seg_dets.join(" ");
                    for obs in &seg_obs {
                        seg_str.push(' ');
                        seg_str.push_str(obs);
                    }
                    remapped_segments.push(seg_str);
                }
            }

            if !remapped_segments.is_empty() {
                let _ = write!(out, "error({prob_str}) ");
                out.push_str(&remapped_segments.join(" ^ "));
                out.push('\n');
            }
        } else if trimmed.starts_with("detector(")
            && let Some(d_start) = trimmed.rfind('D')
            && let Ok(d) = trimmed[d_start + 1..].trim().parse::<usize>()
            && d < num_det
            && in_window[d]
            && let Some(local) = global_to_local[d]
        {
            let coords_end = trimmed.find(')').unwrap_or(trimmed.len());
            out.push_str(&trimmed[..=coords_end]);
            let _ = writeln!(out, " D{local}");
        }
    }

    (local_to_global, out)
}

// =============================================================================
// Streaming windowed decoder
// =============================================================================

use std::collections::BTreeMap;

/// Streaming windowed decoder that accepts syndrome data round-by-round.
///
/// Precomputes round-to-detector mapping from DEM coordinates. As rounds
/// arrive via `feed_round`, buffers syndrome data and triggers window
/// decoding when each window's extended region is complete. Emits partial
/// observable corrections as windows commit.
pub struct StreamingWindowedDecoder<D> {
    /// Prebuilt windows, ordered by start time.
    windows: Vec<OverlappingWindow<D>>,
    /// Round number → list of (`local_window_idx`, `local_detector_idx`) for each window.
    round_to_dets: BTreeMap<usize, Vec<(usize, usize)>>,
    /// Per-window syndrome buffers.
    window_syndromes: Vec<Vec<u8>>,
    /// Round at which each window becomes decodable (all data received).
    window_ready_round: Vec<usize>,
    /// Index of next window to decode.
    next_decode: usize,
    /// Accumulated observable corrections.
    accumulated: u64,
}

impl<D: EdgeTrackingDecoder> StreamingWindowedDecoder<D> {
    /// Create from a DEM string with factory.
    ///
    /// # Errors
    ///
    /// Returns `DecoderError` if the DEM is malformed or factory fails.
    pub fn from_dem<F>(
        dem: &str,
        config: WindowedConfig,
        mut factory: F,
    ) -> Result<Self, DecoderError>
    where
        F: FnMut(&str) -> Result<D, DecoderError>,
    {
        let (det_times, num_detectors, step_size, total_t) = parse_dem_params(dem, &config)?;
        let buffer_size = config.buffer_size;

        // Build windows (same as OverlappingWindowedDecoder).
        let mut windows = Vec::new();
        let mut window_ranges: Vec<(f64, f64, f64)> = Vec::new(); // (win_start, win_end, core_end)
        let mut t_start = 0.0f64;

        while t_start < total_t {
            let is_last = t_start + 2.0 * step_size as f64 > total_t;
            let t_core_end = if is_last {
                total_t + 1.0
            } else {
                t_start + step_size as f64
            };
            let t_win_start = (t_start - buffer_size as f64).max(0.0);
            let t_win_end = if is_last {
                total_t + 1.0
            } else {
                t_core_end + buffer_size as f64
            };

            let (local_to_global, window_dem) =
                extract_window_dem(dem, &det_times, num_detectors, t_win_start, t_win_end);

            let ext = config.core_extend as f64;
            let is_core: Vec<bool> = local_to_global
                .iter()
                .map(|&gid| {
                    let t = det_times[gid as usize];
                    t >= (t_start - ext) && t < (t_core_end + ext)
                })
                .collect();

            let num_local = local_to_global.len();
            if num_local > 0 && !window_dem.is_empty() {
                let decoder = factory(&window_dem)?;
                window_ranges.push((t_win_start, t_win_end, t_core_end));
                windows.push(OverlappingWindow {
                    decoder,
                    local_to_global,
                    is_core,
                    num_local,
                });
            }

            t_start += step_size as f64;
        }

        // Build round → (window_idx, local_det) mapping.
        let mut round_to_dets: BTreeMap<usize, Vec<(usize, usize)>> = BTreeMap::new();
        for (win_idx, window) in windows.iter().enumerate() {
            for (local_id, &global_id) in window.local_to_global.iter().enumerate() {
                let round = det_times[global_id as usize] as usize;
                round_to_dets
                    .entry(round)
                    .or_default()
                    .push((win_idx, local_id));
            }
        }

        // Compute when each window has all its data.
        let window_ready_round: Vec<usize> = window_ranges
            .iter()
            .map(|&(_, t_end, _)| (t_end.ceil() as usize).saturating_sub(1))
            .collect();

        let window_syndromes = windows.iter().map(|w| vec![0u8; w.num_local]).collect();

        Ok(Self {
            windows,
            round_to_dets,
            window_syndromes,
            window_ready_round,
            next_decode: 0,
            accumulated: 0,
        })
    }

    /// Decode a ready window and return its observable contribution.
    fn decode_window(&mut self, win_idx: usize) -> Result<u64, DecoderError> {
        let window = &mut self.windows[win_idx];
        let syn = &self.window_syndromes[win_idx];

        let (_, matched_edges) = window.decoder.decode_with_matching(syn)?;

        let mut obs = 0u64;
        let boundary = window.num_local as u32;
        for &edge_idx in &matched_edges {
            let n1 = window.decoder.edge_node1(edge_idx);
            let n2 = window.decoder.edge_node2(edge_idx);

            let n1_core = n1 >= boundary
                || ((n1 as usize) < window.is_core.len() && window.is_core[n1 as usize]);
            let n2_core = n2 >= boundary
                || ((n2 as usize) < window.is_core.len() && window.is_core[n2 as usize]);

            if n1_core && n2_core {
                obs ^= window.decoder.edge_obs_mask(edge_idx);
            }
        }
        Ok(obs)
    }
}

impl<D: EdgeTrackingDecoder> pecos_decoder_core::streaming::StreamingDecoder
    for StreamingWindowedDecoder<D>
{
    fn feed_round(&mut self, round: usize, detectors: &[(u32, u8)]) -> Result<u64, DecoderError> {
        // Store detection events into each window's syndrome buffer.
        for &(det, val) in detectors {
            if let Some(entries) = self.round_to_dets.get(&round) {
                for &(win_idx, local_id) in entries {
                    // Check if this detector matches
                    if self.windows[win_idx].local_to_global.get(local_id) == Some(&det) {
                        self.window_syndromes[win_idx][local_id] = val;
                    }
                }
            }
        }

        // Also store by global detector index for windows that contain this detector.
        for &(det, val) in detectors {
            for (win_idx, window) in self.windows.iter().enumerate() {
                for (local_id, &global_id) in window.local_to_global.iter().enumerate() {
                    if global_id == det {
                        self.window_syndromes[win_idx][local_id] = val;
                    }
                }
            }
        }

        // Check if any window became ready.
        let mut new_obs = 0u64;
        while self.next_decode < self.windows.len() {
            if round < self.window_ready_round[self.next_decode] {
                break;
            }
            new_obs ^= self.decode_window(self.next_decode)?;
            self.next_decode += 1;
        }

        self.accumulated ^= new_obs;
        Ok(new_obs)
    }

    fn flush(&mut self) -> Result<u64, DecoderError> {
        let mut new_obs = 0u64;
        while self.next_decode < self.windows.len() {
            new_obs ^= self.decode_window(self.next_decode)?;
            self.next_decode += 1;
        }
        self.accumulated ^= new_obs;
        Ok(new_obs)
    }

    fn accumulated_obs(&self) -> u64 {
        self.accumulated
    }

    fn reset(&mut self) {
        for syn in &mut self.window_syndromes {
            syn.fill(0);
        }
        self.next_decode = 0;
        self.accumulated = 0;
    }
}

// =============================================================================
// Beam search windowed decoder
// =============================================================================

/// Configuration for the beam search windowed decoder.
#[derive(Debug, Clone, Copy)]
pub struct BeamSearchConfig {
    /// Windowed decoder parameters.
    pub window: WindowedConfig,
    /// Number of beam hypotheses (K). Default 5.
    pub beam_width: usize,
    /// Perturbation sigma for log-normal weight noise. Default 0.5.
    pub perturbation_sigma: f64,
    /// RNG seed for reproducibility.
    pub seed: u64,
}

impl Default for BeamSearchConfig {
    fn default() -> Self {
        Self {
            window: WindowedConfig::default(),
            beam_width: 5,
            perturbation_sigma: 0.5,
            seed: 42,
        }
    }
}

/// One beam hypothesis: accumulated state from windows processed so far.
struct Hypothesis {
    correction_effect: Vec<u8>,
    obs_mask: u64,
    total_weight: f64,
}

/// Per-window storage: K decoders (1 unperturbed + K-1 perturbed).
struct BeamWindow<D> {
    decoders: Vec<D>,
    local_to_global: Vec<u32>,
    is_core: Vec<bool>,
    num_local: usize,
}

/// Beam search windowed decoder.
///
/// Maintains K correction hypotheses across window boundaries. Each window
/// expands K hypotheses × K perturbed decoders = K² candidates, pruned
/// to K by total correction weight. After all windows, picks the
/// lowest-weight hypothesis and optionally runs a Phase-2 residual decode.
///
/// The key insight: different hypotheses propagate different
/// `correction_effect` vectors to subsequent windows, so each hypothesis
/// sees a different modified syndrome. This explores different string
/// continuations across window boundaries.
pub struct BeamSearchWindowedDecoder<D> {
    windows: Vec<BeamWindow<D>>,
    num_detectors: usize,
    beam_width: usize,
    commit_weight_max: f64,
    residual_decoder: Option<Box<dyn ObservableDecoder>>,
}

impl<D: EdgeTrackingDecoder> BeamSearchWindowedDecoder<D> {
    /// Create from a DEM string.
    ///
    /// `phase1_factory` builds the inner edge-tracking decoder from a sub-DEM.
    /// `phase2_factory` (optional) builds the full-graph residual decoder.
    ///
    /// # Errors
    ///
    /// Returns `DecoderError` if the DEM is malformed or factories fail.
    pub fn from_dem<F1, F2>(
        dem: &str,
        config: BeamSearchConfig,
        mut phase1_factory: F1,
        mut phase2_factory: Option<F2>,
    ) -> Result<Self, DecoderError>
    where
        F1: FnMut(&str) -> Result<D, DecoderError>,
        F2: FnMut(&str) -> Result<Box<dyn ObservableDecoder>, DecoderError>,
    {
        let (det_times, num_detectors, step_size, total_t) = parse_dem_params(dem, &config.window)?;
        let buffer_size = config.window.buffer_size;
        let k = config.beam_width;

        let mut windows = Vec::new();
        let mut t_start = 0.0f64;

        while t_start < total_t {
            let is_last = t_start + 2.0 * step_size as f64 > total_t;
            let t_core_end = if is_last {
                total_t + 1.0
            } else {
                t_start + step_size as f64
            };
            let t_win_start = (t_start - buffer_size as f64).max(0.0);
            let t_win_end = if is_last {
                total_t + 1.0
            } else {
                t_core_end + buffer_size as f64
            };

            let (local_to_global, window_dem) =
                extract_window_dem(dem, &det_times, num_detectors, t_win_start, t_win_end);

            let ext = config.window.core_extend as f64;
            let is_core: Vec<bool> = local_to_global
                .iter()
                .map(|&gid| {
                    let t = det_times[gid as usize];
                    t >= (t_start - ext) && t < (t_core_end + ext)
                })
                .collect();

            let num_local = local_to_global.len();
            if num_local > 0 && !window_dem.is_empty() {
                let mut decoders = Vec::with_capacity(k);

                // Decoder 0: unperturbed anchor
                decoders.push(phase1_factory(&window_dem)?);

                // Decoders 1..K-1: perturbed weights
                for member_idx in 1..k {
                    let mut rng = pecos_random::PecosRng::seed_from_u64(
                        config.seed.wrapping_add(member_idx as u64),
                    );
                    let mut next_f64 = || rng.next_f64();
                    let perturbed = pecos_decoder_core::perturbed::perturb_dem(
                        &window_dem,
                        config.perturbation_sigma,
                        &mut next_f64,
                    );
                    if let Ok(dec) = phase1_factory(&perturbed) {
                        decoders.push(dec);
                    }
                }

                windows.push(BeamWindow {
                    decoders,
                    local_to_global,
                    is_core,
                    num_local,
                });
            }

            t_start += step_size as f64;
        }

        let residual_decoder = if let Some(ref mut f2) = phase2_factory {
            Some(f2(dem)?)
        } else {
            None
        };

        Ok(Self {
            windows,
            num_detectors,
            beam_width: k,
            commit_weight_max: config.window.commit_weight_max,
            residual_decoder,
        })
    }

    /// Number of windows.
    #[must_use]
    pub fn num_windows(&self) -> usize {
        self.windows.len()
    }
}

impl<D: EdgeTrackingDecoder> ObservableDecoder for BeamSearchWindowedDecoder<D> {
    fn decode_to_observables(&mut self, syndrome: &[u8]) -> Result<u64, DecoderError> {
        let k = self.beam_width;
        let commit_weight_max = self.commit_weight_max;

        // Initialize beam with K identical empty hypotheses.
        let mut beam: Vec<Hypothesis> = (0..k)
            .map(|_| Hypothesis {
                correction_effect: vec![0u8; self.num_detectors],
                obs_mask: 0,
                total_weight: 0.0,
            })
            .collect();

        // Process each window: expand K hypotheses × K decoders → prune to K.
        for window in &mut self.windows {
            let actual_k = window.decoders.len();
            let mut candidates: Vec<Hypothesis> = Vec::with_capacity(beam.len() * actual_k);

            // Build window syndrome from the original (Phase-1 windows are
            // independent — correction_effect is only used for Phase-2 residual).
            let mut window_syn = vec![0u8; window.num_local];
            for (local_id, &global_id) in window.local_to_global.iter().enumerate() {
                let gid = global_id as usize;
                if gid < syndrome.len() {
                    window_syn[local_id] = syndrome[gid];
                }
            }

            for hyp in &beam {
                // Decode with each perturbed decoder.
                for decoder in &mut window.decoders {
                    let (_, matched_edges) = decoder.decode_with_matching(&window_syn)?;

                    let mut new_obs = hyp.obs_mask;
                    let mut new_correction = hyp.correction_effect.clone();
                    let mut new_weight = hyp.total_weight;
                    let boundary = window.num_local as u32;

                    for &edge_idx in &matched_edges {
                        let n1 = decoder.edge_node1(edge_idx);
                        let n2 = decoder.edge_node2(edge_idx);

                        let n1_core = n1 >= boundary
                            || ((n1 as usize) < window.is_core.len()
                                && window.is_core[n1 as usize]);
                        let n2_core = n2 >= boundary
                            || ((n2 as usize) < window.is_core.len()
                                && window.is_core[n2 as usize]);

                        let weight_ok = commit_weight_max <= 0.0
                            || decoder.edge_weight(edge_idx) <= commit_weight_max;

                        if n1_core && n2_core && weight_ok {
                            new_obs ^= decoder.edge_obs_mask(edge_idx);
                            new_weight += decoder.edge_weight(edge_idx);

                            if (n1 as usize) < window.num_local {
                                let gid = window.local_to_global[n1 as usize] as usize;
                                new_correction[gid] ^= 1;
                            }
                            if (n2 as usize) < window.num_local {
                                let gid = window.local_to_global[n2 as usize] as usize;
                                new_correction[gid] ^= 1;
                            }
                        }
                    }

                    candidates.push(Hypothesis {
                        correction_effect: new_correction,
                        obs_mask: new_obs,
                        total_weight: new_weight,
                    });
                }
            }

            // Prune: sort by total weight (lower = more likely), dedup, truncate.
            candidates.sort_by(|a, b| {
                a.total_weight
                    .partial_cmp(&b.total_weight)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            candidates.dedup_by(|a, b| a.correction_effect == b.correction_effect);
            candidates.truncate(k);
            beam = candidates;
        }

        // Pick the result via majority vote across surviving hypotheses.
        // Each hypothesis may have a different Phase-1 obs_mask; we also run
        // Phase-2 on each to get the complete observable prediction.
        if beam.is_empty() {
            return Ok(0);
        }

        // Collect final observable predictions from each hypothesis.
        let mut predictions: Vec<u64> = Vec::with_capacity(beam.len());
        if let Some(ref mut residual_dec) = self.residual_decoder {
            for hyp in &beam {
                let mut residual_syn = vec![0u8; self.num_detectors];
                for (i, &s) in syndrome.iter().enumerate() {
                    if i < self.num_detectors {
                        residual_syn[i] = s ^ hyp.correction_effect[i];
                    }
                }
                let phase2_obs = residual_dec.decode_to_observables(&residual_syn)?;
                predictions.push(hyp.obs_mask ^ phase2_obs);
            }
        } else {
            for hyp in &beam {
                predictions.push(hyp.obs_mask);
            }
        }

        // Majority vote across hypotheses (per observable bit).
        let half = predictions.len() / 2;
        let mut result = 0u64;
        for bit in 0..64u32 {
            let mask = 1u64 << bit;
            let count = predictions.iter().filter(|&&p| p & mask != 0).count();
            if count > half {
                result |= mask;
            }
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const D3_DEM: &str =
        include_str!("../../../examples/surface_code_circuits/surface_code_d3_z_stim.dem");

    fn uf_factory(dem: &str) -> Result<Box<dyn ObservableDecoder>, DecoderError> {
        Ok(Box::new(crate::UfDecoder::from_dem(
            dem,
            crate::UfDecoderConfig::fast(),
        )?))
    }

    fn uf_edge_factory(dem: &str) -> Result<crate::UfDecoder, DecoderError> {
        crate::UfDecoder::from_dem(dem, crate::UfDecoderConfig::windowed())
    }

    #[test]
    fn test_windowed_construction() {
        let dec = WindowedDecoder::from_dem(D3_DEM, WindowedConfig::default(), uf_factory);
        assert!(dec.is_ok());
        assert!(dec.unwrap().num_windows() > 0);
    }

    #[test]
    fn test_windowed_no_errors() {
        let graph = DemMatchingGraph::from_dem_str(D3_DEM).unwrap();
        let mut dec =
            WindowedDecoder::from_dem(D3_DEM, WindowedConfig::default(), uf_factory).unwrap();
        let obs = dec.decode_to_observables(&vec![0u8; graph.num_detectors]);
        assert!(obs.is_ok());
        assert_eq!(obs.unwrap(), 0);
    }

    #[test]
    fn test_single_window_matches_full() {
        let graph = DemMatchingGraph::from_dem_str(D3_DEM).unwrap();
        let config = WindowedConfig {
            step_size: 100,
            buffer_size: 0,
            ..Default::default()
        };
        let mut wdec = WindowedDecoder::from_dem(D3_DEM, config, uf_factory).unwrap();
        let mut udec = crate::UfDecoder::from_dem(D3_DEM, crate::UfDecoderConfig::fast()).unwrap();

        let syn = vec![0u8; graph.num_detectors];
        assert_eq!(
            wdec.decode_to_observables(&syn).unwrap(),
            udec.decode_to_observables(&syn).unwrap(),
        );
    }

    #[test]
    fn test_overlapping_construction() {
        let config = WindowedConfig {
            step_size: 3,
            buffer_size: 2,
            ..Default::default()
        };
        let dec = OverlappingWindowedDecoder::from_dem(D3_DEM, config, uf_edge_factory);
        assert!(dec.is_ok());
        assert!(dec.unwrap().num_windows() > 0);
    }

    #[test]
    fn test_overlapping_no_errors() {
        let graph = DemMatchingGraph::from_dem_str(D3_DEM).unwrap();
        let config = WindowedConfig {
            step_size: 3,
            buffer_size: 2,
            ..Default::default()
        };
        let mut dec =
            OverlappingWindowedDecoder::from_dem(D3_DEM, config, uf_edge_factory).unwrap();
        let obs = dec.decode_to_observables(&vec![0u8; graph.num_detectors]);
        assert!(obs.is_ok());
        assert_eq!(obs.unwrap(), 0);
    }

    #[test]
    fn test_overlapping_single_window() {
        let graph = DemMatchingGraph::from_dem_str(D3_DEM).unwrap();
        let config = WindowedConfig {
            step_size: 100,
            buffer_size: 5,
            ..Default::default()
        };
        let mut dec =
            OverlappingWindowedDecoder::from_dem(D3_DEM, config, uf_edge_factory).unwrap();
        assert_eq!(dec.num_windows(), 1);
        let syn = vec![0u8; graph.num_detectors];
        assert_eq!(dec.decode_to_observables(&syn).unwrap(), 0);
    }

    #[test]
    fn test_sandwich_construction() {
        let config = WindowedConfig {
            step_size: 3,
            buffer_size: 3,
            ..Default::default()
        };
        let dec = SandwichWindowedDecoder::from_dem(D3_DEM, config, uf_edge_factory, uf_factory);
        assert!(dec.is_ok());
        let dec = dec.unwrap();
        assert!(dec.num_windows() > 0);
    }

    #[test]
    fn test_sandwich_no_errors() {
        let graph = DemMatchingGraph::from_dem_str(D3_DEM).unwrap();
        let config = WindowedConfig {
            step_size: 3,
            buffer_size: 3,
            ..Default::default()
        };
        let mut dec =
            SandwichWindowedDecoder::from_dem(D3_DEM, config, uf_edge_factory, uf_factory).unwrap();
        let obs = dec.decode_to_observables(&vec![0u8; graph.num_detectors]);
        assert!(obs.is_ok());
        assert_eq!(obs.unwrap(), 0);
    }

    #[test]
    fn test_sandwich_parallel_matches_sequential() {
        let graph = DemMatchingGraph::from_dem_str(D3_DEM).unwrap();
        let config = WindowedConfig {
            step_size: 3,
            buffer_size: 3,
            ..Default::default()
        };
        let mut dec =
            SandwichWindowedDecoder::from_dem(D3_DEM, config, uf_edge_factory, uf_factory).unwrap();

        let syn = vec![0u8; graph.num_detectors];
        let seq = dec.decode_to_observables(&syn).unwrap();
        let par = dec.decode_parallel(&syn).unwrap();
        assert_eq!(seq, par);
    }

    #[test]
    fn test_streaming_construction() {
        let config = WindowedConfig {
            step_size: 3,
            buffer_size: 2,
            ..Default::default()
        };
        let dec = StreamingWindowedDecoder::from_dem(D3_DEM, config, uf_edge_factory);
        assert!(dec.is_ok());
    }

    #[test]
    fn test_streaming_no_errors() {
        use pecos_decoder_core::streaming::StreamingDecoder;

        let config = WindowedConfig {
            step_size: 3,
            buffer_size: 2,
            ..Default::default()
        };
        let mut dec = StreamingWindowedDecoder::from_dem(D3_DEM, config, uf_edge_factory).unwrap();

        // Feed empty rounds — no detectors fire.
        let graph = DemMatchingGraph::from_dem_str(D3_DEM).unwrap();
        let max_round = graph
            .detector_coords
            .iter()
            .filter_map(|c| c.as_ref().and_then(|v| v.get(2)).copied())
            .fold(0.0f64, f64::max) as usize;

        for r in 0..=max_round {
            dec.feed_round(r, &[]).unwrap();
        }
        dec.flush().unwrap();
        assert_eq!(dec.accumulated_obs(), 0);
    }

    #[test]
    fn test_beam_k1_matches_sandwich_nonzero() {
        // K=1 beam with non-zero syndrome should match sandwich.
        let graph = DemMatchingGraph::from_dem_str(D3_DEM).unwrap();
        let wconfig = WindowedConfig {
            step_size: 3,
            buffer_size: 3,
            commit_weight_max: 2.5,
            ..Default::default()
        };

        let mut sandwich =
            SandwichWindowedDecoder::from_dem(D3_DEM, wconfig, uf_edge_factory, uf_factory)
                .unwrap();

        let bconfig = BeamSearchConfig {
            window: wconfig,
            beam_width: 1,
            perturbation_sigma: 0.0,
            seed: 42,
        };
        let mut beam =
            BeamSearchWindowedDecoder::from_dem(D3_DEM, bconfig, uf_edge_factory, Some(uf_factory))
                .unwrap();

        // Test with single-defect syndrome.
        let mut syn = vec![0u8; graph.num_detectors];
        syn[0] = 1;
        let sw_obs = sandwich.decode_to_observables(&syn).unwrap();
        let bm_obs = beam.decode_to_observables(&syn).unwrap();
        assert_eq!(
            sw_obs, bm_obs,
            "K=1 beam should match sandwich. sw={sw_obs}, bm={bm_obs}"
        );

        // Test with two defects.
        syn[0] = 1;
        syn[1] = 1;
        let sw_obs = sandwich.decode_to_observables(&syn).unwrap();
        let bm_obs = beam.decode_to_observables(&syn).unwrap();
        assert_eq!(
            sw_obs, bm_obs,
            "K=1 beam should match sandwich on 2 defects. sw={sw_obs}, bm={bm_obs}"
        );
    }

    #[test]
    fn test_beam_search_construction() {
        let config = BeamSearchConfig {
            window: WindowedConfig {
                step_size: 3,
                buffer_size: 3,
                ..Default::default()
            },
            beam_width: 3,
            perturbation_sigma: 0.5,
            seed: 42,
        };
        let dec =
            BeamSearchWindowedDecoder::from_dem(D3_DEM, config, uf_edge_factory, Some(uf_factory));
        assert!(dec.is_ok());
        assert!(dec.unwrap().num_windows() > 0);
    }

    #[test]
    fn test_beam_k1_matches_sandwich() {
        // K=1 beam with no perturbation should match the sandwich decoder.
        let graph = DemMatchingGraph::from_dem_str(D3_DEM).unwrap();
        let wconfig = WindowedConfig {
            step_size: 3,
            buffer_size: 3,
            commit_weight_max: 2.5,
            ..Default::default()
        };

        // Sandwich
        let mut sandwich =
            SandwichWindowedDecoder::from_dem(D3_DEM, wconfig, uf_edge_factory, uf_factory)
                .unwrap();

        // Beam K=1
        let bconfig = BeamSearchConfig {
            window: wconfig,
            beam_width: 1,
            perturbation_sigma: 0.0,
            seed: 42,
        };
        let mut beam =
            BeamSearchWindowedDecoder::from_dem(D3_DEM, bconfig, uf_edge_factory, Some(uf_factory))
                .unwrap();

        let syn = vec![0u8; graph.num_detectors];
        let sw_obs = sandwich.decode_to_observables(&syn).unwrap();
        let bm_obs = beam.decode_to_observables(&syn).unwrap();
        assert_eq!(
            sw_obs, bm_obs,
            "K=1 beam should match sandwich on zero syndrome"
        );
    }

    #[test]
    fn test_beam_search_no_errors() {
        let graph = DemMatchingGraph::from_dem_str(D3_DEM).unwrap();
        let config = BeamSearchConfig {
            window: WindowedConfig {
                step_size: 3,
                buffer_size: 3,
                ..Default::default()
            },
            beam_width: 3,
            perturbation_sigma: 0.5,
            seed: 42,
        };
        let mut dec =
            BeamSearchWindowedDecoder::from_dem(D3_DEM, config, uf_edge_factory, Some(uf_factory))
                .unwrap();
        let obs = dec.decode_to_observables(&vec![0u8; graph.num_detectors]);
        assert!(obs.is_ok());
        assert_eq!(obs.unwrap(), 0);
    }
}
