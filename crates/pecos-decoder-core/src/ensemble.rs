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

//! Multi-decoder ensemble with per-observable majority vote.
//!
//! Runs multiple decoders on the same syndrome, then combines their
//! predictions via (optionally weighted) majority vote on each
//! observable bit independently.

use crate::ObservableDecoder;
use crate::errors::DecoderError;

/// Voting strategy for combining decoder predictions.
#[derive(Debug, Clone)]
pub enum VotingStrategy {
    /// Each decoder gets one vote. Ties go to 0 (no flip).
    Majority,
    /// Each decoder gets a weight. Higher weight = more influence.
    /// Ties (equal total weight for 0 and 1) go to 0.
    Weighted(Vec<f64>),
}

/// Multi-decoder ensemble that votes on observable predictions.
///
/// Each member decoder runs independently on the same syndrome,
/// then the ensemble combines their observable masks via majority
/// vote (per bit).
pub struct EnsembleDecoder {
    decoders: Vec<Box<dyn ObservableDecoder>>,
    strategy: VotingStrategy,
    /// Reusable buffer for collecting predictions.
    predictions: Vec<u64>,
}

impl EnsembleDecoder {
    /// Create an ensemble with uniform (majority) voting.
    #[must_use]
    pub fn new(decoders: Vec<Box<dyn ObservableDecoder>>) -> Self {
        let n = decoders.len();
        Self {
            decoders,
            strategy: VotingStrategy::Majority,
            predictions: Vec::with_capacity(n),
        }
    }

    /// Create an ensemble with weighted voting.
    ///
    /// # Panics
    ///
    /// Panics if `weights.len() != decoders.len()`.
    #[must_use]
    pub fn with_weights(decoders: Vec<Box<dyn ObservableDecoder>>, weights: Vec<f64>) -> Self {
        assert_eq!(
            decoders.len(),
            weights.len(),
            "number of weights must match number of decoders"
        );
        let n = decoders.len();
        Self {
            decoders,
            strategy: VotingStrategy::Weighted(weights),
            predictions: Vec::with_capacity(n),
        }
    }

    /// Number of decoders in the ensemble.
    #[must_use]
    pub fn len(&self) -> usize {
        self.decoders.len()
    }

    /// Whether the ensemble has no decoders.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.decoders.is_empty()
    }
}

impl ObservableDecoder for EnsembleDecoder {
    fn decode_to_observables(&mut self, syndrome: &[u8]) -> Result<u64, DecoderError> {
        if self.decoders.is_empty() {
            return Ok(0);
        }

        // Collect predictions from all decoders.
        self.predictions.clear();
        for decoder in &mut self.decoders {
            self.predictions
                .push(decoder.decode_to_observables(syndrome)?);
        }

        // Vote on each observable bit independently.
        let mut result = 0u64;
        for bit in 0..64 {
            let mask = 1u64 << bit;

            // Check if any decoder cares about this bit.
            let any_set = self.predictions.iter().any(|&p| p & mask != 0);
            if !any_set {
                continue;
            }

            let vote_for_flip = match &self.strategy {
                VotingStrategy::Majority => {
                    let count = self.predictions.iter().filter(|&&p| p & mask != 0).count();
                    // Strict majority: more than half must vote flip.
                    count * 2 > self.decoders.len()
                }
                VotingStrategy::Weighted(weights) => {
                    let mut weight_flip = 0.0;
                    let mut weight_no_flip = 0.0;
                    for (i, &pred) in self.predictions.iter().enumerate() {
                        if pred & mask != 0 {
                            weight_flip += weights[i];
                        } else {
                            weight_no_flip += weights[i];
                        }
                    }
                    weight_flip > weight_no_flip
                }
            };

            if vote_for_flip {
                result |= mask;
            }
        }

        Ok(result)
    }
}

/// Thread-safe ensemble decoder that decodes K members in parallel using rayon.
///
/// Same majority-vote logic as `EnsembleDecoder` but runs all members
/// concurrently. Requires inner decoders to be `Send`.
pub struct ParallelEnsembleDecoder {
    decoders: Vec<Box<dyn ObservableDecoder + Send>>,
}

impl ParallelEnsembleDecoder {
    /// Create a parallel ensemble with majority voting.
    #[must_use]
    pub fn new(decoders: Vec<Box<dyn ObservableDecoder + Send>>) -> Self {
        Self { decoders }
    }

    /// Number of decoders.
    #[must_use]
    pub fn len(&self) -> usize {
        self.decoders.len()
    }

    /// Whether the ensemble is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.decoders.is_empty()
    }
}

impl ObservableDecoder for ParallelEnsembleDecoder {
    fn decode_to_observables(&mut self, syndrome: &[u8]) -> Result<u64, DecoderError> {
        use rayon::prelude::*;

        if self.decoders.is_empty() {
            return Ok(0);
        }

        // Decode all members in parallel.
        let predictions: Result<Vec<u64>, DecoderError> = self
            .decoders
            .par_iter_mut()
            .map(|decoder| decoder.decode_to_observables(syndrome))
            .collect();
        let predictions = predictions?;

        // Majority vote.
        let half = predictions.len() / 2;
        let mut result = 0u64;
        for bit in 0..64 {
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

    /// Fake decoder that always returns a fixed observable mask.
    struct FixedDecoder(u64);

    impl ObservableDecoder for FixedDecoder {
        fn decode_to_observables(&mut self, _syndrome: &[u8]) -> Result<u64, DecoderError> {
            Ok(self.0)
        }
    }

    #[test]
    fn test_majority_unanimous() {
        let decoders: Vec<Box<dyn ObservableDecoder>> = vec![
            Box::new(FixedDecoder(0b101)),
            Box::new(FixedDecoder(0b101)),
            Box::new(FixedDecoder(0b101)),
        ];
        let mut ensemble = EnsembleDecoder::new(decoders);
        assert_eq!(ensemble.decode_to_observables(&[]).unwrap(), 0b101);
    }

    #[test]
    fn test_majority_split() {
        let decoders: Vec<Box<dyn ObservableDecoder>> = vec![
            Box::new(FixedDecoder(0b11)),
            Box::new(FixedDecoder(0b01)),
            Box::new(FixedDecoder(0b01)),
        ];
        let mut ensemble = EnsembleDecoder::new(decoders);
        // Bit 0: 3/3 vote flip -> flip. Bit 1: 1/3 vote flip -> no flip.
        assert_eq!(ensemble.decode_to_observables(&[]).unwrap(), 0b01);
    }

    #[test]
    fn test_majority_tie_goes_to_zero() {
        // With 2 decoders, need >50% for flip. 1/2 is not >50%.
        let decoders: Vec<Box<dyn ObservableDecoder>> =
            vec![Box::new(FixedDecoder(1)), Box::new(FixedDecoder(0))];
        let mut ensemble = EnsembleDecoder::new(decoders);
        assert_eq!(ensemble.decode_to_observables(&[]).unwrap(), 0);
    }

    #[test]
    fn test_weighted_vote() {
        let decoders: Vec<Box<dyn ObservableDecoder>> = vec![
            Box::new(FixedDecoder(1)), // votes flip, weight 3.0
            Box::new(FixedDecoder(0)), // votes no flip, weight 1.0
            Box::new(FixedDecoder(0)), // votes no flip, weight 1.0
        ];
        let mut ensemble = EnsembleDecoder::with_weights(decoders, vec![3.0, 1.0, 1.0]);
        // Weight for flip: 3.0, weight for no flip: 2.0. Flip wins.
        assert_eq!(ensemble.decode_to_observables(&[]).unwrap(), 1);
    }

    #[test]
    fn test_empty_ensemble() {
        let mut ensemble = EnsembleDecoder::new(vec![]);
        assert_eq!(ensemble.decode_to_observables(&[]).unwrap(), 0);
        assert!(ensemble.is_empty());
    }

    #[test]
    fn test_single_decoder() {
        let decoders: Vec<Box<dyn ObservableDecoder>> = vec![Box::new(FixedDecoder(42))];
        let mut ensemble = EnsembleDecoder::new(decoders);
        assert_eq!(ensemble.decode_to_observables(&[]).unwrap(), 42);
    }
}
