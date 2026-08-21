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

//! logical-subgraph decoder with software commitment for streaming decoding.
//!
//! Wraps an `LogicalSubgraphDecoder` with per-detector commitment
//! tracking. Committed detectors are masked during future decodes,
//! implementing the "software commitment" concept from Cain et al.
//! (arXiv:2505.13587).
//!
//! This enables streaming: decode a region, commit it, decode the next
//! region. Only uncommitted detectors participate in matching.

use crate::ObservableDecoder;
use crate::decode_budget::{DecodeStrategy, DetectorRegion};
use crate::errors::DecoderError;
use crate::logical_subgraph::LogicalSubgraphDecoder;
use crate::obs_mask::ObsMask;

/// Observable subgraph decoder with software commitment.
///
/// After decoding a region, call `commit_range()` to mark those
/// detectors as finalized. Future decodes will mask committed
/// detectors (treat as syndrome=0), preventing re-matching of
/// already-corrected errors.
///
/// The total correction is `committed_obs ^ active_obs`: the XOR
/// of committed corrections and the latest active decode.
pub struct CommittedLogicalSubgraphDecoder {
    /// The underlying logical-subgraph decoder (unchanged).
    inner: LogicalSubgraphDecoder,
    /// Per-detector commitment state. True = committed.
    committed: Vec<bool>,
    /// Accumulated observable correction from committed regions.
    committed_obs: ObsMask,
    /// Total number of detectors.
    num_detectors: usize,
    /// Reusable masked syndrome buffer.
    masked_syndrome: Vec<u8>,
}

impl CommittedLogicalSubgraphDecoder {
    /// Wrap an existing logical-subgraph decoder with commitment tracking.
    #[must_use]
    pub fn new(inner: LogicalSubgraphDecoder, num_detectors: usize) -> Self {
        Self {
            inner,
            committed: vec![false; num_detectors],
            committed_obs: ObsMask::new(),
            num_detectors,
            masked_syndrome: vec![0u8; num_detectors],
        }
    }

    /// Decode only uncommitted detectors.
    ///
    /// Committed detectors are masked to 0 before passing to the
    /// inner logical-subgraph decoder. Returns the correction for the active (uncommitted)
    /// region.
    pub fn decode_active(&mut self, syndrome: &[u8]) -> Result<u64, DecoderError> {
        self.decode_active_obs(syndrome)?.to_u64().ok_or_else(|| {
            DecoderError::InvalidConfiguration(
                "decoder has more than 64 observables; use decode_obs() for the wide mask".into(),
            )
        })
    }

    fn decode_active_obs(&mut self, syndrome: &[u8]) -> Result<ObsMask, DecoderError> {
        // Build masked syndrome: zero out committed detectors
        let len = syndrome.len().min(self.num_detectors);
        self.masked_syndrome[..len].copy_from_slice(&syndrome[..len]);
        for i in 0..len {
            if self.committed[i] {
                self.masked_syndrome[i] = 0;
            }
        }
        self.inner.decode_obs(&self.masked_syndrome[..len])
    }

    /// Mark detectors in [start, end) as committed.
    ///
    /// Before committing, decodes the full syndrome to get the
    /// correction that includes the about-to-be-committed region.
    /// The committed correction is stored for accumulation.
    pub fn commit_range(
        &mut self,
        syndrome: &[u8],
        region: &DetectorRegion,
    ) -> Result<u64, DecoderError> {
        // Decode with current syndrome (including uncommitted detectors)
        let wide_obs = self.decode_active_obs(syndrome)?;
        let obs = wide_obs.to_u64().ok_or_else(|| {
            DecoderError::InvalidConfiguration(
                "commit_range supports at most 64 observables; use decode_obs() for wide decoding"
                    .into(),
            )
        })?;

        // Mark detectors as committed
        for i in region.start..region.end.min(self.num_detectors) {
            self.committed[i] = true;
        }

        // Accumulate the correction
        self.committed_obs ^= &wide_obs;
        Ok(obs)
    }

    /// Total correction: committed + latest active.
    ///
    /// Call `decode_active` first to get the active correction,
    /// then XOR with `committed_obs` for the full correction.
    ///
    /// # Errors
    ///
    /// Returns [`DecoderError::InvalidConfiguration`] when the accumulated
    /// correction has an observable above bit 63.
    pub fn committed_obs(&self) -> Result<u64, DecoderError> {
        self.committed_obs.to_u64().ok_or_else(|| {
            DecoderError::InvalidConfiguration(
                "committed_obs() supports at most 64 observables; use decode_obs() for the wide mask"
                    .into(),
            )
        })
    }

    /// Number of committed detectors.
    #[must_use]
    pub fn num_committed(&self) -> usize {
        self.committed.iter().filter(|&&c| c).count()
    }

    /// Reset all commitment state for the next shot.
    pub fn reset(&mut self) {
        self.committed.fill(false);
        self.committed_obs = ObsMask::new();
    }
}

impl ObservableDecoder for CommittedLogicalSubgraphDecoder {
    fn decode_obs(&mut self, syndrome: &[u8]) -> Result<ObsMask, DecoderError> {
        // Full decode: committed XOR active
        let mut active = self.decode_active_obs(syndrome)?;
        active ^= &self.committed_obs;
        Ok(active)
    }
}

impl DecodeStrategy for CommittedLogicalSubgraphDecoder {
    fn decode(&mut self, syndrome: &[u8]) -> Result<u64, DecoderError> {
        self.decode_active(syndrome)
    }

    fn decode_obs(&mut self, syndrome: &[u8]) -> Result<ObsMask, DecoderError> {
        ObservableDecoder::decode_obs(self, syndrome)
    }

    fn commit(&mut self, region: &DetectorRegion) -> Result<u64, DecoderError> {
        // Commit with zeros — the actual syndrome was already decoded
        // via decode(). Just mark the region.
        for i in region.start..region.end.min(self.num_detectors) {
            self.committed[i] = true;
        }
        self.committed_obs.to_u64().ok_or_else(|| {
            DecoderError::InvalidConfiguration(
                "streaming commit supports at most 64 observables; use decode_obs() for wide decoding"
                    .into(),
            )
        })
    }

    fn committed_obs(&self) -> Result<u64, DecoderError> {
        Self::committed_obs(self)
    }

    fn reset(&mut self) {
        CommittedLogicalSubgraphDecoder::reset(self);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FixedSingleObservable;

    impl ObservableDecoder for FixedSingleObservable {
        fn decode_obs(&mut self, syndrome: &[u8]) -> Result<ObsMask, DecoderError> {
            Ok(if syndrome.contains(&1) {
                ObsMask::from_u64(1)
            } else {
                ObsMask::new()
            })
        }
    }

    fn wide_decoder(num_observables: usize) -> CommittedLogicalSubgraphDecoder {
        use std::fmt::Write as _;

        let mut dem = String::from("detector(0, 0, 0) D0\n");
        for observable in 0..num_observables {
            writeln!(dem, "error(0.1) D0 L{observable}").unwrap();
        }
        let membership = vec![vec![0]; num_observables];
        let inner = LogicalSubgraphDecoder::from_membership(&dem, &membership, |_| {
            Ok(Box::new(FixedSingleObservable) as Box<dyn ObservableDecoder + Send + Sync>)
        })
        .unwrap();
        CommittedLogicalSubgraphDecoder::new(inner, 1)
    }

    #[test]
    fn test_detector_region() {
        let r = DetectorRegion { start: 5, end: 15 };
        assert_eq!(r.len(), 10);
        assert!(r.contains(5));
        assert!(!r.contains(15));
    }

    #[test]
    fn test_decode_strategy_trait() {
        // Verify the trait exists and has the right methods
        // (compile-time check via trait bound)
        fn _assert_strategy<T: DecodeStrategy>() {}
    }

    #[test]
    fn observable_wrapper_is_correct_at_64_and_65_observables() {
        for num_observables in [64, 65] {
            let mut decoder = wide_decoder(num_observables);
            let prediction = ObservableDecoder::decode_obs(&mut decoder, &[1]).unwrap();
            assert_eq!(
                prediction.count_ones(),
                u32::try_from(num_observables).unwrap()
            );
            assert!(prediction.get(num_observables - 1));
        }
    }

    #[test]
    fn wide_committed_getter_returns_an_error_instead_of_panicking() {
        let mut decoder = wide_decoder(65);
        decoder.committed_obs.set(64);
        assert!(matches!(
            decoder.committed_obs(),
            Err(DecoderError::InvalidConfiguration(_))
        ));
    }
}
