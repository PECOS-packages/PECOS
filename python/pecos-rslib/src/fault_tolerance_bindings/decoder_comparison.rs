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

//! Paired DUT/reference decoder comparison over a shared sequence of shots.

use pecos_decoder_core::obs_mask::ObsMask;
use pecos_decoder_core::{DecoderError, ObservableDecoder};
use pecos_num::stats::{JeffreysError, JeffreysInterval, jeffreys_interval};
use pyo3::prelude::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DecoderOutcome {
    Correct,
    Mismatch,
    Error,
}

impl DecoderOutcome {
    const fn index(self) -> usize {
        match self {
            Self::Correct => 0,
            Self::Mismatch => 1,
            Self::Error => 2,
        }
    }
}

fn classify(result: Result<ObsMask, DecoderError>, truth: &ObsMask) -> DecoderOutcome {
    match result {
        Ok(prediction) if prediction == *truth => DecoderOutcome::Correct,
        Ok(_) => DecoderOutcome::Mismatch,
        Err(_) => DecoderOutcome::Error,
    }
}

/// Counts indexed by DUT outcome first, then reference outcome.
///
/// In each dimension the order is correct, mismatch, decode error.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(super) struct DecoderComparisonCounts {
    cells: [[u64; 3]; 3],
}

impl DecoderComparisonCounts {
    fn record(&mut self, dut: DecoderOutcome, reference: DecoderOutcome) {
        self.cells[dut.index()][reference.index()] += 1;
    }

    pub(super) const fn cells(&self) -> &[[u64; 3]; 3] {
        &self.cells
    }

    fn total_shots(&self) -> u64 {
        self.cells.iter().flatten().sum()
    }

    const fn dut_only_failures(&self) -> u64 {
        self.cells[DecoderOutcome::Mismatch.index()][DecoderOutcome::Correct.index()]
    }

    const fn both_failed(&self) -> u64 {
        self.cells[DecoderOutcome::Mismatch.index()][DecoderOutcome::Mismatch.index()]
    }
}

/// Compare two decoders on the same shots in the same order.
///
/// `prepare_shot` writes the selected syndrome into the reusable buffer and
/// returns that shot's wide true-observable mask.
pub(super) fn compare_decoder_outcomes(
    num_shots: usize,
    syndrome: &mut [u8],
    mut prepare_shot: impl FnMut(usize, &mut [u8]) -> ObsMask,
    dut: &mut dyn ObservableDecoder,
    reference: &mut dyn ObservableDecoder,
) -> DecoderComparisonCounts {
    let mut counts = DecoderComparisonCounts::default();
    for shot in 0..num_shots {
        let truth = prepare_shot(shot, syndrome);
        // Run both decoders before classifying either result. In particular, a
        // DUT error must not prevent the reference from seeing this shot.
        let dut_result = dut.decode_obs(syndrome);
        let reference_result = reference.decode_obs(syndrome);
        counts.record(
            classify(dut_result, &truth),
            classify(reference_result, &truth),
        );
    }
    counts
}

#[derive(Clone, Copy, Debug)]
struct HeadlineProportion {
    point: f64,
    interval: JeffreysInterval,
}

impl HeadlineProportion {
    fn new(count: u64, total_shots: u64, alpha: f64) -> Result<Self, JeffreysError> {
        let interval = jeffreys_interval(count, total_shots, alpha)?;
        Ok(Self {
            point: interval.point,
            interval,
        })
    }
}

/// Python-facing paired decoder contingency counts and headline proportions.
#[pyclass(
    name = "DecoderComparisonResult",
    module = "pecos_rslib.qec",
    skip_from_py_object
)]
#[derive(Clone, Debug)]
pub(super) struct PyDecoderComparisonResult {
    counts: DecoderComparisonCounts,
    total_shots: u64,
    alpha: f64,
    dut_only_failure: HeadlineProportion,
    both_failed: HeadlineProportion,
}

impl PyDecoderComparisonResult {
    pub(super) fn new(counts: DecoderComparisonCounts, alpha: f64) -> Result<Self, JeffreysError> {
        let total_shots = counts.total_shots();
        let dut_only_failure =
            HeadlineProportion::new(counts.dut_only_failures(), total_shots, alpha)?;
        let both_failed = HeadlineProportion::new(counts.both_failed(), total_shots, alpha)?;
        Ok(Self {
            counts,
            total_shots,
            alpha,
            dut_only_failure,
            both_failed,
        })
    }
}

#[pymethods]
impl PyDecoderComparisonResult {
    /// Raw 3x3 counts in correct, mismatch, error order on both axes.
    #[getter]
    fn counts(&self) -> Vec<Vec<u64>> {
        self.counts.cells().iter().map(|row| row.to_vec()).collect()
    }

    /// Number of shots compared.
    #[getter]
    const fn total_shots(&self) -> u64 {
        self.total_shots
    }

    /// Tail probability used for the equal-tailed Jeffreys intervals.
    #[getter]
    const fn alpha(&self) -> f64 {
        self.alpha
    }

    #[getter]
    const fn dut_correct_reference_correct(&self) -> u64 {
        self.counts.cells[0][0]
    }

    #[getter]
    const fn dut_correct_reference_mismatch(&self) -> u64 {
        self.counts.cells[0][1]
    }

    #[getter]
    const fn dut_correct_reference_error(&self) -> u64 {
        self.counts.cells[0][2]
    }

    #[getter]
    const fn dut_mismatch_reference_correct(&self) -> u64 {
        self.counts.cells[1][0]
    }

    #[getter]
    const fn dut_mismatch_reference_mismatch(&self) -> u64 {
        self.counts.cells[1][1]
    }

    #[getter]
    const fn dut_mismatch_reference_error(&self) -> u64 {
        self.counts.cells[1][2]
    }

    #[getter]
    const fn dut_error_reference_correct(&self) -> u64 {
        self.counts.cells[2][0]
    }

    #[getter]
    const fn dut_error_reference_mismatch(&self) -> u64 {
        self.counts.cells[2][1]
    }

    #[getter]
    const fn dut_error_reference_error(&self) -> u64 {
        self.counts.cells[2][2]
    }

    /// DUT mismatches on shots where the reference was correct.
    #[getter]
    const fn dut_only_failures(&self) -> u64 {
        self.counts.dut_only_failures()
    }

    /// Jeffreys posterior-mean proportion for DUT-only failures.
    #[getter]
    const fn dut_only_failure_proportion(&self) -> f64 {
        self.dut_only_failure.point
    }

    /// Equal-tailed Jeffreys interval for the DUT-only-failure proportion.
    #[getter]
    const fn dut_only_failure_interval(&self) -> (f64, f64) {
        (
            self.dut_only_failure.interval.lo,
            self.dut_only_failure.interval.hi,
        )
    }

    /// Shots on which both decoders returned mismatching predictions.
    #[getter]
    const fn both_failed(&self) -> u64 {
        self.counts.both_failed()
    }

    /// Jeffreys posterior-mean proportion for shots where both decoders failed.
    #[getter]
    const fn both_failed_proportion(&self) -> f64 {
        self.both_failed.point
    }

    /// Equal-tailed Jeffreys interval for the both-failed proportion.
    #[getter]
    const fn both_failed_interval(&self) -> (f64, f64) {
        (self.both_failed.interval.lo, self.both_failed.interval.hi)
    }

    fn __repr__(&self) -> String {
        format!(
            "DecoderComparisonResult(shots={}, dut_only_failures={}, both_failed={})",
            self.total_shots,
            self.counts.dut_only_failures(),
            self.counts.both_failed(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug)]
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

    fn predictions(masks: &[ObsMask]) -> Vec<StubResult> {
        masks.iter().cloned().map(StubResult::Prediction).collect()
    }

    fn compare(
        shots: &[(Vec<u8>, ObsMask)],
        dut_results: Vec<StubResult>,
        reference_results: Vec<StubResult>,
    ) -> DecoderComparisonCounts {
        let syndromes: Vec<Vec<u8>> = shots.iter().map(|(s, _)| s.clone()).collect();
        let mut dut = StubDecoder::new(&syndromes, dut_results);
        let mut reference = StubDecoder::new(&syndromes, reference_results);
        let mut syndrome = vec![0; syndromes.first().map_or(0, Vec::len)];
        compare_decoder_outcomes(
            shots.len(),
            &mut syndrome,
            |shot, buffer| {
                buffer.copy_from_slice(&shots[shot].0);
                shots[shot].1.clone()
            },
            &mut dut,
            &mut reference,
        )
    }

    fn sample_shots() -> Vec<(Vec<u8>, ObsMask)> {
        vec![
            (vec![0, 0], mask(&[])),
            (vec![1, 0], mask(&[0])),
            (vec![0, 1], mask(&[1])),
            (vec![1, 1], mask(&[0, 1])),
        ]
    }

    #[test]
    fn both_decoders_correct_puts_all_mass_in_correct_correct() {
        let shots = sample_shots();
        let truths: Vec<ObsMask> = shots.iter().map(|(_, truth)| truth.clone()).collect();
        let counts = compare(&shots, predictions(&truths), predictions(&truths));

        assert_eq!(counts.cells(), &[[4, 0, 0], [0, 0, 0], [0, 0, 0]]);
        assert_eq!(counts.dut_only_failures(), 0);
    }

    #[test]
    fn dut_only_failures_count_a_known_wrong_subset() {
        let shots = sample_shots();
        let truths: Vec<ObsMask> = shots.iter().map(|(_, truth)| truth.clone()).collect();
        let mut dut = truths.clone();
        dut[1] = mask(&[]);
        dut[3] = mask(&[]);

        let counts = compare(&shots, predictions(&dut), predictions(&truths));

        // Shots 1 and 3 are deliberately wrong for the DUT: 2 DUT-only failures.
        assert_eq!(counts.dut_only_failures(), 2);
        assert_eq!(counts.cells(), &[[2, 0, 0], [2, 0, 0], [0, 0, 0]]);
    }

    #[test]
    fn dut_errors_are_not_mismatches_and_do_not_abort() {
        let shots = sample_shots();
        let truths: Vec<ObsMask> = shots.iter().map(|(_, truth)| truth.clone()).collect();
        let mut dut = predictions(&truths);
        dut[1] = StubResult::Error;
        dut[3] = StubResult::Error;

        let counts = compare(&shots, dut, predictions(&truths));

        assert_eq!(counts.cells(), &[[2, 0, 0], [0, 0, 0], [2, 0, 0]]);
        assert_eq!(counts.cells()[DecoderOutcome::Mismatch.index()][0], 0);
    }

    #[test]
    fn reference_errors_are_counted_and_do_not_abort() {
        let shots = sample_shots();
        let truths: Vec<ObsMask> = shots.iter().map(|(_, truth)| truth.clone()).collect();
        let mut reference = predictions(&truths);
        reference[0] = StubResult::Error;
        reference[2] = StubResult::Error;

        let counts = compare(&shots, predictions(&truths), reference);

        assert_eq!(counts.cells(), &[[2, 0, 2], [0, 0, 0], [0, 0, 0]]);
    }

    #[test]
    fn wide_observable_difference_above_bit_63_is_preserved() {
        let wide_truth = mask(&[70]);
        let shots = vec![(vec![1], wide_truth.clone())];
        let counts = compare(
            &shots,
            predictions(&[ObsMask::new()]),
            predictions(&[wide_truth]),
        );

        assert_eq!(counts.cells(), &[[0, 0, 0], [1, 0, 0], [0, 0, 0]]);
        assert_eq!(counts.dut_only_failures(), 1);
    }

    #[test]
    fn headline_interval_matches_pecos_num_helper() {
        let shots = sample_shots();
        let truths: Vec<ObsMask> = shots.iter().map(|(_, truth)| truth.clone()).collect();
        let mut dut = truths.clone();
        dut[1] = mask(&[]);
        let summary = PyDecoderComparisonResult::new(
            compare(&shots, predictions(&dut), predictions(&truths)),
            0.05,
        )
        .expect("valid Jeffreys inputs");
        let expected = jeffreys_interval(1, 4, 0.05).expect("valid direct helper inputs");

        assert_eq!(summary.dut_only_failure.interval, expected);
        // Both sides come from the same helper call, so the point estimate must be
        // bit-identical; compare bit patterns rather than floats.
        assert_eq!(
            summary.dut_only_failure.point.to_bits(),
            expected.point.to_bits()
        );
    }

    #[test]
    fn comparison_is_deterministic_for_the_same_batch() {
        let shots = sample_shots();
        let truths: Vec<ObsMask> = shots.iter().map(|(_, truth)| truth.clone()).collect();
        let mut dut = truths.clone();
        dut[2] = mask(&[]);

        let first = compare(&shots, predictions(&dut), predictions(&truths));
        let second = compare(&shots, predictions(&dut), predictions(&truths));

        assert_eq!(first, second);
    }
}
