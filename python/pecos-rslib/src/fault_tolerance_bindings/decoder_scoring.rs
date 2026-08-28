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

//! Shared fail-loud scoring for observable decoders.

use pecos_decoder_core::obs_mask::ObsMask;
use pecos_decoder_core::{DecoderError, ObservableDecoder};
use std::fmt;
use std::ops::Range;

/// A decoder failure annotated with the shot that caused it.
#[derive(Debug)]
pub(super) struct ShotDecodeError {
    pub(super) shot_index: usize,
    pub(super) source: DecoderError,
}

impl ShotDecodeError {
    pub(super) const fn new(shot_index: usize, source: DecoderError) -> Self {
        Self { shot_index, source }
    }
}

/// Scored output from one contiguous sequential worker range.
pub(super) struct DecodeRangeResult {
    pub(super) mismatches: usize,
    pub(super) predictions: Vec<ObsMask>,
    pub(super) per_shot_seconds: Vec<f64>,
}

impl fmt::Display for ShotDecodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "decoder failed on shot {}: {}",
            self.shot_index, self.source
        )
    }
}

impl std::error::Error for ShotDecodeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

/// Decode, score, and optionally retain predictions and per-shot elapsed times.
///
/// The syndrome allocation belongs to the caller and is reused for every shot
/// in the range. Predictions and timings are allocated only when requested.
pub(super) fn decode_and_score_range(
    shots: Range<usize>,
    syndrome: &mut [u8],
    mut access_shot: impl FnMut(usize, &mut [u8]) -> ObsMask,
    decoder: &mut dyn ObservableDecoder,
    collect_predictions: bool,
    collect_timings: bool,
) -> Result<DecodeRangeResult, ShotDecodeError> {
    let capacity = shots.len();
    let mut result = DecodeRangeResult {
        mismatches: 0,
        predictions: if collect_predictions {
            Vec::with_capacity(capacity)
        } else {
            Vec::new()
        },
        per_shot_seconds: if collect_timings {
            Vec::with_capacity(capacity)
        } else {
            Vec::new()
        },
    };
    for shot_index in shots {
        let truth = access_shot(shot_index, syndrome);
        let start = collect_timings.then(std::time::Instant::now);
        let decoded = decoder.decode_obs(syndrome);
        if let Some(start) = start {
            result.per_shot_seconds.push(start.elapsed().as_secs_f64());
        }
        let prediction = decoded.map_err(|source| ShotDecodeError::new(shot_index, source))?;
        result.mismatches += usize::from(prediction != truth);
        if collect_predictions {
            result.predictions.push(prediction);
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    enum StubResult {
        Prediction(ObsMask),
        Error,
    }

    struct StubDecoder {
        expected_syndromes: Vec<Vec<u8>>,
        results: Vec<StubResult>,
        next: usize,
    }

    impl StubDecoder {
        fn new(expected_syndromes: &[Vec<u8>], results: Vec<StubResult>) -> Self {
            assert_eq!(expected_syndromes.len(), results.len());
            Self {
                expected_syndromes: expected_syndromes.to_vec(),
                results,
                next: 0,
            }
        }
    }

    impl ObservableDecoder for StubDecoder {
        fn decode_obs(&mut self, syndrome: &[u8]) -> Result<ObsMask, DecoderError> {
            assert_eq!(syndrome, self.expected_syndromes[self.next]);
            let result = match &self.results[self.next] {
                StubResult::Prediction(mask) => Ok(mask.clone()),
                StubResult::Error => Err(DecoderError::DecodingFailed("stub error".into())),
            };
            self.next += 1;
            result
        }
    }

    fn mask(bits: &[usize]) -> ObsMask {
        let mut mask = ObsMask::new();
        for &bit in bits {
            mask.set(bit);
        }
        mask
    }

    fn score(
        shots: &[(Vec<u8>, ObsMask)],
        results: Vec<StubResult>,
    ) -> Result<usize, ShotDecodeError> {
        let syndromes: Vec<Vec<u8>> = shots.iter().map(|(syndrome, _)| syndrome.clone()).collect();
        let mut decoder = StubDecoder::new(&syndromes, results);
        let mut syndrome = vec![0; syndromes.first().map_or(0, Vec::len)];
        decode_and_score_range(
            0..shots.len(),
            &mut syndrome,
            |shot_index, buffer| {
                buffer.copy_from_slice(&shots[shot_index].0);
                shots[shot_index].1.clone()
            },
            &mut decoder,
            false,
            false,
        )
        .map(|result| result.mismatches)
    }

    #[test]
    fn decoder_error_aborts_with_shot_index_and_source() {
        let shots = vec![
            (vec![0], mask(&[])),
            (vec![1], mask(&[0])),
            (vec![0], mask(&[])),
        ];
        let error = score(
            &shots,
            vec![
                StubResult::Prediction(mask(&[])),
                StubResult::Error,
                StubResult::Prediction(mask(&[])),
            ],
        )
        .unwrap_err();

        assert_eq!(error.shot_index, 1);
        assert!(error.to_string().contains("shot 1"));
        assert!(error.to_string().contains("stub error"));
    }

    #[test]
    fn healthy_decoder_preserves_exact_mismatch_count() {
        let shots = vec![
            (vec![0, 0], mask(&[])),
            (vec![1, 0], mask(&[0])),
            (vec![0, 1], mask(&[1])),
            (vec![1, 1], mask(&[0, 1])),
        ];
        let count = score(
            &shots,
            vec![
                StubResult::Prediction(mask(&[])),
                StubResult::Prediction(mask(&[])),
                StubResult::Prediction(mask(&[1])),
                StubResult::Prediction(mask(&[1])),
            ],
        )
        .unwrap();

        assert_eq!(count, 2);
    }

    #[test]
    fn always_erroring_decoder_does_not_match_all_observables_flipped_truth() {
        let all_observables = mask(&[0, 1, 2]);
        let shots = vec![(vec![1, 1], all_observables)];

        // The old parallel sampler substituted the full observable-selection
        // mask on error, which incorrectly scored this exact truth as correct.
        let error = score(&shots, vec![StubResult::Error]).unwrap_err();

        assert_eq!(error.shot_index, 0);
        assert!(error.to_string().contains("stub error"));
    }
}
