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

//! Frontier approximate logical maximum-likelihood decoding.
//!
//! The decoder performs ordered dynamic programming over independent binary
//! fault mechanisms. Prefixes with identical active detector boundary and
//! logical labels are merged by log-sum-exp, preserving degeneracy mass. The
//! configured frontier width and log-mass window provide deterministic pruning
//! for a fixed build and platform; underlying `ln`/`exp` implementations may
//! differ across platforms.
//! [`FrontierDecoder`] is the parity port of Leverrier and Urbanke's Frontier
//! decoder. Its input-order defaults remain those of that port.

use pecos_decoder_core::ObservableDecoder;
pub use pecos_trellis::{
    DecoderError, ObsMask, SparseDem, backward_deadline_column_order, deadline_column_order,
};
use std::cmp::Ordering;
use std::time::Instant;

/// Parity-port name for [`pecos_trellis::TrellisDecoder`].
pub type FrontierDecoder = pecos_trellis::TrellisDecoder;
/// Parity-port name for [`pecos_trellis::TrellisConfig`].
pub type FrontierConfig = pecos_trellis::TrellisConfig;
/// Parity-port name for [`pecos_trellis::TrellisResult`].
pub type FrontierResult = pecos_trellis::TrellisResult;
/// Parity-port name for [`pecos_trellis::TrellisStatus`].
pub type FrontierStatus = pecos_trellis::TrellisStatus;
/// Parity-port name for [`pecos_trellis::TrellisDecodeAttempt`].
pub type FrontierDecodeAttempt = pecos_trellis::TrellisDecodeAttempt;
/// Parity-port name for [`pecos_trellis::TrellisLogicalMass`].
pub type FrontierLogicalMass = pecos_trellis::TrellisLogicalMass;

/// Direction selected by [`FrontierCommittee`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CommitteeDirection {
    /// The configured processing order.
    Forward,
    /// The plain reverse of the configured processing order.
    Backward,
}

/// Decode status for one committee leg.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CommitteeStatus {
    /// The leg retained at least one terminal state.
    Ok,
    /// The leg found no retained path for the syndrome.
    NoPath,
}

/// Summary of one committee leg.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CommitteeMember {
    /// Decode status for this leg.
    pub status: CommitteeStatus,
    /// Total retained log evidence, or negative infinity for no path.
    pub log_evidence: f64,
}

/// Result selected from the forward/backward committee.
#[derive(Clone, Debug, PartialEq)]
pub struct FrontierCommitteeResult {
    /// Full result from the selected leg.
    pub selected: FrontierResult,
    /// Direction of the selected leg.
    pub direction: CommitteeDirection,
    /// Forward-leg status and evidence.
    pub forward: CommitteeMember,
    /// Backward-leg status and evidence.
    pub backward: CommitteeMember,
}

/// Two-leg Frontier decoder using a processing order and its plain reverse.
#[derive(Clone, Debug)]
pub struct FrontierCommittee {
    forward: FrontierDecoder,
    backward: FrontierDecoder,
    build_seconds: f64,
}

impl FrontierCommittee {
    /// Construct a forward/backward committee from a sparse DEM.
    ///
    /// The forward leg uses `config.column_order` (or DEM order when absent).
    /// The backward leg uses the plain reverse of that same sequence.
    ///
    /// # Errors
    ///
    /// Returns [`DecoderError::InvalidConfiguration`] when the configuration or
    /// DEM is invalid.
    pub fn from_sparse_dem(dem: &SparseDem, config: FrontierConfig) -> Result<Self, DecoderError> {
        let build_started = Instant::now();
        let FrontierConfig {
            k,
            delta,
            score_alpha,
            label_diverse_retention,
            column_order,
            merge_indistinguishable,
            bp_score_iterations,
        } = config;
        let mut forward_order = column_order.unwrap_or_else(|| (0..dem.mechanisms.len()).collect());
        let forward = FrontierDecoder::from_sparse_dem(
            dem,
            FrontierConfig {
                k,
                delta,
                score_alpha,
                label_diverse_retention,
                column_order: Some(forward_order.clone()),
                merge_indistinguishable,
                bp_score_iterations,
            },
        )?;
        forward_order.reverse();
        let backward_config = FrontierConfig {
            k,
            delta,
            score_alpha,
            label_diverse_retention,
            column_order: Some(forward_order),
            merge_indistinguishable,
            bp_score_iterations,
        };
        let backward = FrontierDecoder::from_sparse_dem(dem, backward_config)?;
        let build_seconds = build_started.elapsed().as_secs_f64();
        Ok(Self {
            forward,
            backward,
            build_seconds,
        })
    }

    /// Wall-clock seconds spent constructing both committee legs in
    /// [`Self::from_sparse_dem`].
    #[must_use]
    pub fn build_seconds(&self) -> f64 {
        self.build_seconds
    }

    /// Parse a Stim-format detector error model and construct a committee.
    ///
    /// # Errors
    ///
    /// Returns [`DecoderError`] if parsing or committee construction fails.
    pub fn from_dem_str(dem_str: &str, config: FrontierConfig) -> Result<Self, DecoderError> {
        let dem = SparseDem::from_dem_str(dem_str)?;
        Self::from_sparse_dem(&dem, config)
    }

    /// Decode with both processing directions and select the stronger result.
    ///
    /// # Errors
    ///
    /// Returns the standard unexplainable-syndrome error if both legs find no
    /// retained path.
    pub fn decode(&mut self, syndrome: &[u8]) -> Result<FrontierCommitteeResult, DecoderError> {
        // Each leg owns its BP graph and scratch and runs independently on the
        // same syndrome. Their beliefs are expected to match, but are never
        // shared by reference between the two code paths.
        let forward_result = self.forward.decode(syndrome);
        let backward_result = self.backward.decode(syndrome);
        let forward = committee_member(&forward_result);
        let backward = committee_member(&backward_result);
        let (selected, direction) = match (forward_result, backward_result) {
            (Err(_), Err(_)) => return Err(unexplainable_error()),
            (Ok(selected), Err(_)) => (selected, CommitteeDirection::Forward),
            (Err(_), Ok(selected)) => (selected, CommitteeDirection::Backward),
            (Ok(forward_result), Ok(backward_result)) => {
                if compare_committee_legs(Some(&forward_result), Some(&backward_result))
                    == Ordering::Less
                {
                    (backward_result, CommitteeDirection::Backward)
                } else {
                    (forward_result, CommitteeDirection::Forward)
                }
            }
        };

        Ok(FrontierCommitteeResult {
            selected,
            direction,
            forward,
            backward,
        })
    }
}

impl ObservableDecoder for FrontierCommittee {
    fn decode_obs(&mut self, syndrome: &[u8]) -> Result<ObsMask, DecoderError> {
        Ok(self.decode(syndrome)?.selected.predicted)
    }
}

fn committee_member(result: &Result<FrontierResult, DecoderError>) -> CommitteeMember {
    match result {
        Ok(decoded) => CommitteeMember {
            status: CommitteeStatus::Ok,
            log_evidence: decoded.log_evidence,
        },
        Err(_) => CommitteeMember {
            status: CommitteeStatus::NoPath,
            log_evidence: f64::NEG_INFINITY,
        },
    }
}

fn compare_committee_legs(
    forward: Option<&FrontierResult>,
    backward: Option<&FrontierResult>,
) -> Ordering {
    let forward_rank = committee_rank(forward, true);
    let backward_rank = committee_rank(backward, false);
    forward_rank
        .iter()
        .zip(backward_rank)
        .find_map(|(forward_component, backward_component)| {
            let ordering = forward_component.total_cmp(&backward_component);
            (ordering != Ordering::Equal).then_some(ordering)
        })
        .unwrap_or(Ordering::Equal)
}

fn committee_rank(result: Option<&FrontierResult>, is_forward: bool) -> [f64; 6] {
    let forward_bonus = if is_forward { 1.0 } else { 0.0 };
    let Some(result) = result else {
        return [
            1.0,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
            0.0,
            forward_bonus,
        ];
    };

    let terminal_gap = result.runner_up_gap.unwrap_or(f64::INFINITY);
    let terminal_gap = if terminal_gap.is_nan() {
        f64::NEG_INFINITY
    } else {
        terminal_gap
    };
    let top_one_posterior = result
        .logical_masses
        .first()
        .map_or(f64::NEG_INFINITY, |winner| {
            winner.log_mass - result.log_evidence
        });
    let top_one_posterior = if top_one_posterior.is_finite() {
        top_one_posterior
    } else {
        f64::NEG_INFINITY
    };
    [
        2.0,
        result.log_evidence,
        terminal_gap,
        top_one_posterior,
        0.0,
        forward_bonus,
    ]
}

fn unexplainable_error() -> DecoderError {
    DecoderError::DecodingFailed("syndrome is unexplainable at the given pruning parameters".into())
}

#[cfg(test)]
mod tests {
    use super::{
        FrontierCommittee, FrontierConfig, FrontierLogicalMass, FrontierResult, FrontierStatus,
        SparseDem, committee_rank,
    };
    use pecos_decoder_core::obs_mask::ObsMask;
    use std::collections::BTreeMap;

    #[test]
    fn committee_merges_both_legs() {
        let dem = SparseDem {
            mechanisms: vec![
                (0.3, vec![0], vec![0]),
                (0.2, vec![1], vec![]),
                (0.4, vec![0], vec![0]),
            ],
            detector_coords: BTreeMap::new(),
            num_detectors: 2,
            num_observables: 1,
        };
        let mut committee = FrontierCommittee::from_sparse_dem(
            &dem,
            FrontierConfig {
                k: usize::MAX,
                delta: f64::INFINITY,
                score_alpha: 0.8,
                label_diverse_retention: false,
                column_order: None,
                merge_indistinguishable: true,
                bp_score_iterations: 0,
            },
        )
        .unwrap();
        let forward = committee.forward.decode(&[0, 0]).unwrap();
        let backward = committee.backward.decode(&[0, 0]).unwrap();

        assert_eq!(forward.processed_columns, 2);
        assert_eq!(backward.processed_columns, 2);
        assert_eq!(forward.processed_columns, backward.processed_columns);
    }

    #[test]
    fn committee_bp_legs_do_not_share_bp_state() {
        let dem = SparseDem {
            mechanisms: vec![
                (0.15, vec![0], vec![]),
                (0.15, vec![0, 1], vec![0]),
                (0.08, vec![1], vec![]),
            ],
            detector_coords: BTreeMap::new(),
            num_detectors: 2,
            num_observables: 1,
        };
        let committee = FrontierCommittee::from_sparse_dem(
            &dem,
            FrontierConfig {
                k: 1,
                delta: f64::INFINITY,
                score_alpha: 0.8,
                label_diverse_retention: false,
                column_order: None,
                merge_indistinguishable: false,
                bp_score_iterations: 5,
            },
        )
        .unwrap();
        let (forward_graph, forward_scratch) = committee.forward.bp_state_addrs().unwrap();
        let (backward_graph, backward_scratch) = committee.backward.bp_state_addrs().unwrap();

        assert_ne!(forward_graph, forward_scratch);
        assert_ne!(forward_graph, backward_graph);
        assert_ne!(forward_graph, backward_scratch);
        assert_ne!(forward_scratch, backward_graph);
        assert_ne!(forward_scratch, backward_scratch);
        assert_ne!(backward_graph, backward_scratch);
    }

    #[test]
    fn committee_rank_maps_special_terminal_statistics() {
        let no_runner_up = FrontierResult {
            predicted: ObsMask::new(),
            log_evidence: -1.0,
            runner_up_gap: None,
            peak_retained_states: 1,
            processed_columns: 0,
            transitions: 0,
            dropped_states: 0,
            dropped_log_mass: f64::NEG_INFINITY,
            bp_seconds: 0.0,
            escalation_rungs_used: 0,
            status: FrontierStatus::Exact,
            logical_masses: vec![FrontierLogicalMass {
                logical: ObsMask::new(),
                log_mass: f64::NAN,
            }],
        };
        let rank = committee_rank(Some(&no_runner_up), true);
        assert_eq!(rank[2].to_bits(), f64::INFINITY.to_bits());
        assert_eq!(rank[3].to_bits(), f64::NEG_INFINITY.to_bits());
        assert_eq!(rank[5].to_bits(), 1.0_f64.to_bits());

        let nan_gap = FrontierResult {
            runner_up_gap: Some(f64::NAN),
            logical_masses: vec![FrontierLogicalMass {
                logical: ObsMask::new(),
                log_mass: -1.5,
            }],
            ..no_runner_up
        };
        let rank = committee_rank(Some(&nan_gap), false);
        assert_eq!(rank[2].to_bits(), f64::NEG_INFINITY.to_bits());
        assert_eq!(rank[3].to_bits(), (-0.5_f64).to_bits());

        let no_path_rank = committee_rank(None, false);
        assert_eq!(no_path_rank[0].to_bits(), 1.0_f64.to_bits());
        assert_eq!(no_path_rank[1].to_bits(), f64::NEG_INFINITY.to_bits());
    }
}
