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

use pecos_decoder_core::ObservableDecoder;
use pecos_trellis::{DecoderError, TrellisResult};

pub fn assert_errors_equal(left: &DecoderError, right: &DecoderError) {
    assert_eq!(std::mem::discriminant(left), std::mem::discriminant(right));
    assert_eq!(left.to_string(), right.to_string());
}

pub fn assert_results_equal(left: &TrellisResult, right: &TrellisResult) {
    assert_eq!(left.predicted, right.predicted);
    assert_eq!(left.log_evidence.to_bits(), right.log_evidence.to_bits());
    assert_eq!(
        left.runner_up_gap.map(f64::to_bits),
        right.runner_up_gap.map(f64::to_bits)
    );
    assert_eq!(left.peak_retained_states, right.peak_retained_states);
    assert_eq!(left.processed_columns, right.processed_columns);
    assert_eq!(left.transitions, right.transitions);
    assert_eq!(left.dropped_states, right.dropped_states);
    assert_eq!(
        left.dropped_log_mass.to_bits(),
        right.dropped_log_mass.to_bits()
    );
    assert_eq!(left.bp_runs, right.bp_runs);
    assert_eq!(left.escalation_rungs_used, right.escalation_rungs_used);
    assert_eq!(left.status, right.status);
    assert_eq!(left.logical_masses.len(), right.logical_masses.len());
    for (left, right) in left.logical_masses.iter().zip(&right.logical_masses) {
        assert_eq!(left.logical, right.logical);
        assert_eq!(left.log_mass.to_bits(), right.log_mass.to_bits());
    }
    // Actual BP timings vary between calls; zero timings are deterministic.
    if left.bp_runs == 0 {
        assert_eq!(left.bp_seconds.to_bits(), right.bp_seconds.to_bits());
    } else {
        assert!(left.bp_seconds.is_finite() && left.bp_seconds >= 0.0);
        assert!(right.bp_seconds.is_finite() && right.bp_seconds >= 0.0);
    }
}

pub fn check<D: ObservableDecoder>(
    decoder: &mut D,
    fresh: &mut D,
    syndrome: &[u8],
    decode: impl Fn(&mut D, &[u8]) -> Result<TrellisResult, DecoderError>,
    wide: bool,
) -> u32 {
    let expected = decode(fresh, syndrome);
    let predicted = decoder.decode_obs(syndrome);
    let after = decode(decoder, syndrome);
    match (&expected, predicted, &after) {
        (Ok(expected), Ok(predicted), Ok(after)) => {
            assert_eq!(predicted, expected.predicted);
            assert_results_equal(after, expected);
        }
        (Err(expected), Err(predicted), Err(after)) => {
            assert_errors_equal(&predicted, expected);
            assert_errors_equal(after, expected);
        }
        values => panic!("prediction-only/detail mismatch: {values:?}"),
    }
    if !wide {
        let narrow = decoder.decode_to_observables(syndrome);
        match (&expected, narrow) {
            (Ok(expected), Ok(narrow)) => assert_eq!(Some(narrow), expected.predicted.to_u64()),
            (Err(expected), Err(narrow)) => assert_errors_equal(&narrow, expected),
            values => panic!("narrow/detail mismatch: {values:?}"),
        }
        match (&expected, decode(decoder, syndrome)) {
            (Ok(expected), Ok(after)) => assert_results_equal(&after, expected),
            (Err(expected), Err(after)) => assert_errors_equal(&after, expected),
            values => panic!("detailed after narrow mismatch: {values:?}"),
        }
    }
    expected.map_or(0, |result| result.escalation_rungs_used)
}
