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
    shot_index: usize,
    source: DecoderError,
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

/// Decode and score a contiguous range of shots, aborting on the first failure.
///
/// `access_shot` fills the reusable syndrome buffer and returns that shot's
/// true observable mask. Shot indices are kept absolute so callers can combine
/// independently scored worker ranges without losing error context.
pub(super) fn count_decoder_mismatches(
    shots: Range<usize>,
    syndrome: &mut [u8],
    mut access_shot: impl FnMut(usize, &mut [u8]) -> ObsMask,
    decoder: &mut dyn ObservableDecoder,
) -> Result<usize, ShotDecodeError> {
    let mut mismatches = 0;
    for shot_index in shots {
        let truth = access_shot(shot_index, syndrome);
        let prediction = decoder
            .decode_obs(syndrome)
            .map_err(|source| ShotDecodeError { shot_index, source })?;
        mismatches += usize::from(prediction != truth);
    }
    Ok(mismatches)
}

/// Observable-decoder adapter that keeps only caller-selected observables.
pub(super) struct MaskedObservableDecoder {
    inner: Box<dyn ObservableDecoder>,
    mask: ObsMask,
}

impl MaskedObservableDecoder {
    pub(super) fn new(inner: Box<dyn ObservableDecoder>, mask: ObsMask) -> Self {
        Self { inner, mask }
    }
}

impl ObservableDecoder for MaskedObservableDecoder {
    fn decode_obs(&mut self, syndrome: &[u8]) -> Result<ObsMask, DecoderError> {
        let mut prediction = self.inner.decode_obs(syndrome)?;
        prediction &= &self.mask;
        Ok(prediction)
    }
}

/// Observable-decoder adapter that records one elapsed time per decode call.
pub(super) struct TimedObservableDecoder {
    inner: Box<dyn ObservableDecoder>,
    per_shot_seconds: Vec<f64>,
}

impl TimedObservableDecoder {
    pub(super) fn new(inner: Box<dyn ObservableDecoder>, capacity: usize) -> Self {
        Self {
            inner,
            per_shot_seconds: Vec::with_capacity(capacity),
        }
    }

    pub(super) fn into_times(self) -> Vec<f64> {
        self.per_shot_seconds
    }
}

impl ObservableDecoder for TimedObservableDecoder {
    fn decode_obs(&mut self, syndrome: &[u8]) -> Result<ObsMask, DecoderError> {
        let start = std::time::Instant::now();
        let result = self.inner.decode_obs(syndrome);
        self.per_shot_seconds.push(start.elapsed().as_secs_f64());
        result
    }
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
        count_decoder_mismatches(
            0..shots.len(),
            &mut syndrome,
            |shot_index, buffer| {
                buffer.copy_from_slice(&shots[shot_index].0);
                shots[shot_index].1.clone()
            },
            &mut decoder,
        )
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
