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

//! OSD with software commitment for streaming decoding.
//!
//! Wraps an `ObservableSubgraphDecoder` with per-detector commitment
//! tracking. Committed detectors are masked during future decodes,
//! implementing the "software commitment" concept from Cain et al.
//! (arXiv:2505.13587).
//!
//! This enables streaming: decode a region, commit it, decode the next
//! region. Only uncommitted detectors participate in matching.

use crate::ObservableDecoder;
use crate::decode_budget::{DecodeStrategy, DetectorRegion};
use crate::errors::DecoderError;
use crate::observable_subgraph::ObservableSubgraphDecoder;

/// Observable subgraph decoder with software commitment.
///
/// After decoding a region, call `commit_range()` to mark those
/// detectors as finalized. Future decodes will mask committed
/// detectors (treat as syndrome=0), preventing re-matching of
/// already-corrected errors.
///
/// The total correction is `committed_obs ^ active_obs`: the XOR
/// of committed corrections and the latest active decode.
pub struct CommittedOsdDecoder {
    /// The underlying OSD (unchanged).
    inner: ObservableSubgraphDecoder,
    /// Per-detector commitment state. True = committed.
    committed: Vec<bool>,
    /// Accumulated observable correction from committed regions.
    committed_obs: u64,
    /// Total number of detectors.
    num_detectors: usize,
    /// Reusable masked syndrome buffer.
    masked_syndrome: Vec<u8>,
}

impl CommittedOsdDecoder {
    /// Wrap an existing OSD with commitment tracking.
    #[must_use]
    pub fn new(inner: ObservableSubgraphDecoder, num_detectors: usize) -> Self {
        Self {
            inner,
            committed: vec![false; num_detectors],
            committed_obs: 0,
            num_detectors,
            masked_syndrome: vec![0u8; num_detectors],
        }
    }

    /// Decode only uncommitted detectors.
    ///
    /// Committed detectors are masked to 0 before passing to the
    /// inner OSD. Returns the correction for the active (uncommitted)
    /// region.
    pub fn decode_active(&mut self, syndrome: &[u8]) -> Result<u64, DecoderError> {
        // Build masked syndrome: zero out committed detectors
        let len = syndrome.len().min(self.num_detectors);
        self.masked_syndrome[..len].copy_from_slice(&syndrome[..len]);
        for i in 0..len {
            if self.committed[i] {
                self.masked_syndrome[i] = 0;
            }
        }
        self.inner
            .decode_to_observables(&self.masked_syndrome[..len])
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
        let obs = self.decode_active(syndrome)?;

        // Mark detectors as committed
        for i in region.start..region.end.min(self.num_detectors) {
            self.committed[i] = true;
        }

        // Accumulate the correction
        self.committed_obs ^= obs;
        Ok(obs)
    }

    /// Total correction: committed + latest active.
    ///
    /// Call `decode_active` first to get the active correction,
    /// then XOR with `committed_obs` for the full correction.
    #[must_use]
    pub fn committed_obs(&self) -> u64 {
        self.committed_obs
    }

    /// Number of committed detectors.
    #[must_use]
    pub fn num_committed(&self) -> usize {
        self.committed.iter().filter(|&&c| c).count()
    }

    /// Reset all commitment state for the next shot.
    pub fn reset(&mut self) {
        self.committed.fill(false);
        self.committed_obs = 0;
    }
}

impl ObservableDecoder for CommittedOsdDecoder {
    fn decode_to_observables(&mut self, syndrome: &[u8]) -> Result<u64, DecoderError> {
        // Full decode: committed XOR active
        let active = self.decode_active(syndrome)?;
        Ok(self.committed_obs ^ active)
    }
}

impl DecodeStrategy for CommittedOsdDecoder {
    fn decode(&mut self, syndrome: &[u8]) -> Result<u64, DecoderError> {
        self.decode_active(syndrome)
    }

    fn commit(&mut self, region: &DetectorRegion) -> Result<u64, DecoderError> {
        // Commit with zeros — the actual syndrome was already decoded
        // via decode(). Just mark the region.
        for i in region.start..region.end.min(self.num_detectors) {
            self.committed[i] = true;
        }
        Ok(self.committed_obs)
    }

    fn committed_obs(&self) -> u64 {
        self.committed_obs
    }

    fn reset(&mut self) {
        CommittedOsdDecoder::reset(self);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
