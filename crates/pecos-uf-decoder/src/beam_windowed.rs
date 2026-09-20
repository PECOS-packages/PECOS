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

//! Beam search with independent windows and a full residual pass.

use pecos_decoder_core::ObservableDecoder;
use pecos_decoder_core::correlated_decoder::EdgeTrackingDecoder;
use pecos_decoder_core::errors::DecoderError;
use pecos_decoder_core::window::{DemBoundaryKind, StructuredDem};

/// Configuration for the windowed decoder.
#[derive(Debug, Clone, Copy, Default)]
pub struct BeamWindowConfig {
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

fn window_parameters(
    dem: &StructuredDem,
    config: &BeamWindowConfig,
) -> Result<(Vec<f64>, usize, usize, f64), DecoderError> {
    // Every edge-tracking window family and its residual/beam aggregation use
    // u64 observable masks internally. Reject a wider model before creating
    // any sub-decoder so no global observable bit can be truncated.
    dem.ensure_observables_fit_u64()?;
    let num_detectors = dem.num_detectors;
    let det_times = dem.detector_times();
    let max_time = det_times.iter().copied().fold(0.0f64, f64::max);

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

/// Extract a soft-boundary window from the shared structured DEM.
///
/// Detectors in `[t_start, t_end)` are included and remapped to local IDs.
/// Detectors outside the window are dropped from error mechanisms, creating
/// implicit boundary edges.
fn render_soft_window(
    dem: &StructuredDem,
    t_start: f64,
    t_end: f64,
) -> Result<(Vec<u32>, String), DecoderError> {
    let window = dem.window_by_time(t_start, t_end, DemBoundaryKind::Soft)?;
    Ok((window.local_to_global, window.model.to_dem_string()))
}

/// Configuration for the beam search windowed decoder.
#[derive(Debug, Clone, Copy)]
pub struct BeamSearchConfig {
    /// Windowed decoder parameters.
    pub window: BeamWindowConfig,
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
            window: BeamWindowConfig::default(),
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
    /// Returns `DecoderError` if the DEM is outside
    /// [`StructuredDem::from_dem_str`]'s strict flattened grammar or factories fail.
    pub fn from_dem<F1, F2>(
        dem: &str,
        config: BeamSearchConfig,
        phase1_factory: F1,
        phase2_factory: Option<F2>,
    ) -> Result<Self, DecoderError>
    where
        F1: FnMut(&str) -> Result<D, DecoderError>,
        F2: FnMut(&str) -> Result<Box<dyn ObservableDecoder>, DecoderError>,
    {
        let dem = StructuredDem::from_dem_str(dem)?;
        let phase2_factory = phase2_factory
            .map(|mut factory| move |model: &StructuredDem| factory(&model.to_dem_string()));
        Self::from_structured_dem(&dem, config, phase1_factory, phase2_factory)
    }

    /// Create a beam-search decoder from an already parsed structured DEM.
    ///
    /// # Errors
    ///
    /// Returns `DecoderError` if the model is incompatible or a factory fails.
    pub fn from_structured_dem<F1, F2>(
        dem: &StructuredDem,
        config: BeamSearchConfig,
        mut phase1_factory: F1,
        mut phase2_factory: Option<F2>,
    ) -> Result<Self, DecoderError>
    where
        F1: FnMut(&str) -> Result<D, DecoderError>,
        F2: FnMut(&StructuredDem) -> Result<Box<dyn ObservableDecoder>, DecoderError>,
    {
        let (det_times, num_detectors, step_size, total_t) =
            window_parameters(dem, &config.window)?;
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

            let (local_to_global, window_dem) = render_soft_window(dem, t_win_start, t_win_end)?;

            let ext = config.window.core_extend as f64;
            let is_core: Vec<bool> = local_to_global
                .iter()
                .map(|&gid| {
                    let t = det_times[gid as usize];
                    t >= (t_start - ext) && t < (t_core_end + ext)
                })
                .collect();

            let num_local = local_to_global.len();
            if num_local > 0 {
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
                    )?;
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
    fn decode_obs(
        &mut self,
        syndrome: &[u8],
    ) -> Result<pecos_decoder_core::obs_mask::ObsMask, DecoderError> {
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
            return Ok(pecos_decoder_core::obs_mask::ObsMask::new());
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
        Ok(pecos_decoder_core::obs_mask::ObsMask::from_u64(result))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{UfDecoder, UfDecoderConfig};

    #[test]
    fn beam_residual_and_constructor_contracts() {
        let dem =
            include_str!("../../../examples/surface_code_circuits/surface_code_d3_z_stim.dem");
        let model = StructuredDem::from_dem_str(dem).unwrap();
        let config = BeamSearchConfig {
            window: BeamWindowConfig {
                step_size: 3,
                buffer_size: 3,
                commit_weight_max: 2.5,
                ..BeamWindowConfig::default()
            },
            beam_width: 1,
            perturbation_sigma: 0.0,
            seed: 42,
        };
        let factory = |text: &str| UfDecoder::from_dem(text, UfDecoderConfig::windowed());
        let full_factory = |text: &str| -> Result<Box<dyn ObservableDecoder>, DecoderError> {
            Ok(Box::new(UfDecoder::from_dem(
                text,
                UfDecoderConfig::fast(),
            )?))
        };
        let mut text =
            BeamSearchWindowedDecoder::from_dem(dem, config, factory, Some(full_factory)).unwrap();
        let mut direct = BeamSearchWindowedDecoder::from_structured_dem(
            &model,
            config,
            factory,
            Some(|m: &StructuredDem| full_factory(&m.to_dem_string())),
        )
        .unwrap();
        assert_eq!(text.num_windows(), direct.num_windows());
        let mut mono = full_factory(dem).unwrap();
        let mut syndrome = vec![0; model.num_detectors];
        for d in 0..model.num_detectors {
            syndrome[d] = 1;
            let expected = mono.decode_to_observables(&syndrome).unwrap();
            assert_eq!(text.decode_to_observables(&syndrome).unwrap(), expected);
            assert_eq!(direct.decode_to_observables(&syndrome).unwrap(), expected);
            syndrome[d] = 0;
        }
    }
}
