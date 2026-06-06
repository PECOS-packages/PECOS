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

//! Coord-preserving per-observable subgraph/window plan with honest
//! windowing-mode introspection.
//!
//! Both windowed logical-subgraph decoders (the standalone
//! `WindowedLogicalSubgraphDecoder` in `pecos-uf-decoder`, and the streaming
//! `WindowedLogicalSubgraphStrategy` here) need the same inputs: per-observable
//! graphlike sub-DEMs that PRESERVE detector coordinates (so time-based
//! windowing has real times), the local↔global detector maps, and -- crucially
//! -- a way to report whether real time-windowing will actually happen or the
//! decode silently degenerates to a single full window.
//!
//! Subgraph matching graphs drop detector coordinates
//! ([`crate::logical_subgraph::subgraphs_from_membership`] sets
//! `detector_coords: Vec::new()`), so a sub-DEM serialized from the graph alone
//! has no `detector(...)` lines; any windowed inner then sees `total_t = 1` and
//! builds a single window. This plan injects the full-DEM coordinates (mapped to
//! subgraph-local indices) and exposes the resulting window structure, so
//! callers can FAIL LOUD instead of silently full-decoding behind a
//! bounded-latency API.
//!
//! This lives in `pecos-decoder-core` as shared data/modeling so both the
//! downstream UF decoder and the logical-circuit strategy consume one plan
//! (avoiding a `decoder-core -> pecos-uf-decoder` dependency).

use crate::logical_subgraph::LogicalSubgraph;
use std::fmt::Write as _;

/// Whether a windowed logical-subgraph decode actually time-windows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffectiveWindowing {
    /// Every per-observable subgraph fits in a single window: a full
    /// (non-windowed) decode -- accurate, but unbounded latency. Selecting this
    /// when bounded latency was requested is a silent fallback unless surfaced.
    FullFallback,
    /// At least one subgraph spans multiple time windows: real sliding-window
    /// decoding (bounded latency, subject to the windowed-LOM accuracy limit).
    RealWindowed,
}

impl EffectiveWindowing {
    /// Stable string label for APIs / tests.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            EffectiveWindowing::FullFallback => "full_fallback",
            EffectiveWindowing::RealWindowed => "real_windowed",
        }
    }
}

/// One per-observable subgraph with detector coordinates preserved.
pub struct PlanEntry {
    /// Global observable (logical) index this subgraph decodes.
    pub observable_idx: usize,
    /// Subgraph-local detector index -> full-DEM detector index.
    pub detector_map: Vec<usize>,
    /// Coord-preserving sub-DEM: `detector(...)` lines + `error(...)` lines.
    pub sub_dem: String,
    /// Per-local-detector time (coordinate element 2; `0.0` if unknown).
    pub detector_times: Vec<f64>,
}

/// Coord-preserving per-observable subgraph/window plan.
pub struct LogicalSubgraphWindowPlan {
    entries: Vec<PlanEntry>,
}

impl LogicalSubgraphWindowPlan {
    /// Build from per-observable subgraphs and the full-DEM detector
    /// coordinates (indexed by global detector id). Empty-region observables
    /// (no detectors) are skipped -- they never flip and contribute nothing.
    #[must_use]
    pub fn new(subgraphs: &[LogicalSubgraph], full_coords: &[Option<Vec<f64>>]) -> Self {
        let mut entries = Vec::new();
        for sg in subgraphs {
            if sg.detector_map.is_empty() {
                continue;
            }
            let mut detector_times = Vec::with_capacity(sg.detector_map.len());
            let mut sub_dem = String::new();
            for (local, &global) in sg.detector_map.iter().enumerate() {
                let coords = full_coords.get(global).and_then(|c| c.as_ref());
                let t = coords.and_then(|c| c.get(2).copied()).unwrap_or(0.0);
                detector_times.push(t);
                if let Some(c) = coords {
                    let cs: Vec<String> = c.iter().map(|v| format!("{v}")).collect();
                    let _ = writeln!(sub_dem, "detector({}) D{local}", cs.join(", "));
                }
            }
            for edge in &sg.graph.edges {
                let _ = write!(sub_dem, "error({})", edge.probability);
                let _ = write!(sub_dem, " D{}", edge.node1);
                if let Some(n2) = edge.node2 {
                    let _ = write!(sub_dem, " D{n2}");
                }
                for &obs in &edge.observables {
                    let _ = write!(sub_dem, " L{obs}");
                }
                let _ = writeln!(sub_dem);
            }
            entries.push(PlanEntry {
                observable_idx: sg.observable_idx,
                detector_map: sg.detector_map.clone(),
                sub_dem,
                detector_times,
            });
        }
        Self { entries }
    }

    /// Number of non-empty per-observable subgraphs in the plan.
    #[must_use]
    pub fn num_observables(&self) -> usize {
        self.entries.len()
    }

    /// The per-observable plan entries.
    #[must_use]
    pub fn entries(&self) -> &[PlanEntry] {
        &self.entries
    }

    /// Coord-preserving sub-DEM strings (one per non-empty observable).
    #[must_use]
    pub fn sub_dems(&self) -> Vec<String> {
        self.entries.iter().map(|e| e.sub_dem.clone()).collect()
    }

    /// Local->global detector maps (one per non-empty observable).
    #[must_use]
    pub fn detector_maps(&self) -> Vec<Vec<usize>> {
        self.entries.iter().map(|e| e.detector_map.clone()).collect()
    }

    /// Estimated number of time windows observable `i` would use at `step`
    /// rounds per window.
    ///
    /// This is an ESTIMATE, not a guaranteed match of the exact window count an
    /// `OverlappingWindowedDecoder` builds: it counts core ranges that contain a
    /// detector and ignores the buffer overlap, and it requires an explicit
    /// `step` (the real decoder auto-derives `step` from the graph when none is
    /// given). It is sufficient for the load-bearing use here -- the
    /// [`Self::effective_windowing`] FullFallback-vs-RealWindowed *boolean*,
    /// which depends only on `total_t` vs `step`, not on buffer details. Exact
    /// counts should single-source the decoder's own loop (a Layer C item when
    /// the windowing construction is revisited; see the proper-solution design
    /// doc).
    #[must_use]
    pub fn window_count(&self, i: usize, step: usize) -> usize {
        self.entries
            .get(i)
            .map_or(0, |e| window_count_for_times(&e.detector_times, step))
    }

    /// Total windows across all observables at `step`.
    #[must_use]
    pub fn total_windows(&self, step: usize) -> usize {
        (0..self.entries.len())
            .map(|i| self.window_count(i, step))
            .sum()
    }

    /// Whether real time-windowing happens at `step`, or it degenerates to a
    /// single-window full decode for every observable.
    #[must_use]
    pub fn effective_windowing(&self, step: usize) -> EffectiveWindowing {
        if (0..self.entries.len()).any(|i| self.window_count(i, step) > 1) {
            EffectiveWindowing::RealWindowed
        } else {
            EffectiveWindowing::FullFallback
        }
    }
}

/// Count the time windows for a set of detector times at `step` rounds per
/// window, matching the sliding-window loop's core ranges. Only windows that
/// contain at least one detector are counted. With no coordinates (all times
/// `0.0`) this returns 1 -- the silent-fallback signal.
fn window_count_for_times(times: &[f64], step: usize) -> usize {
    if times.is_empty() {
        return 0;
    }
    let max_time = times.iter().copied().fold(0.0f64, f64::max);
    let total_t = max_time + 1.0;
    let step = step.max(1) as f64;

    let mut count = 0usize;
    let mut t_start = 0.0f64;
    while t_start < total_t {
        let is_last = t_start + 2.0 * step > total_t;
        let t_core_end = if is_last { total_t + 1.0 } else { t_start + step };
        if times.iter().any(|&t| t >= t_start && t < t_core_end) {
            count += 1;
        }
        if is_last {
            break;
        }
        t_start += step;
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coordless_times_are_single_window() {
        // All detectors at time 0 (the subgraph-graph / no-coords case).
        assert_eq!(window_count_for_times(&[0.0, 0.0, 0.0], 4), 1);
        assert_eq!(window_count_for_times(&[0.0], 1), 1);
    }

    #[test]
    fn empty_times_are_zero_windows() {
        assert_eq!(window_count_for_times(&[], 4), 0);
    }

    #[test]
    fn multi_round_times_window_by_step() {
        // Times 0..=23 (24 rounds), step 4 -> several windows (> 1).
        let times: Vec<f64> = (0..24).map(f64::from).collect();
        assert!(window_count_for_times(&times, 4) > 1);
        // A step covering the whole range -> a single window.
        assert_eq!(window_count_for_times(&times, 1000), 1);
    }
}
