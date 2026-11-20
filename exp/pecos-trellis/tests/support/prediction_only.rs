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
use pecos_trellis::{DecoderError, TrellisLogicalMass, TrellisResult};

pub fn assert_errors_equal(left: &DecoderError, right: &DecoderError) {
    assert_eq!(std::mem::discriminant(left), std::mem::discriminant(right));
    assert_eq!(left.to_string(), right.to_string());
}

pub fn assert_results_equal(left: &TrellisResult, right: &TrellisResult) {
    // Exhaustive patterns: a new result field fails to compile until it is compared.
    let TrellisResult {
        predicted: left_predicted,
        log_evidence: left_log_evidence,
        logical_masses: left_logical_masses,
        runner_up_gap: left_runner_up_gap,
        peak_retained_states: left_peak_retained_states,
        processed_columns: left_processed_columns,
        transitions: left_transitions,
        dropped_states: left_dropped_states,
        dropped_log_mass: left_dropped_log_mass,
        bp_runs: left_bp_runs,
        bp_seconds: left_bp_seconds,
        escalation_rungs_used: left_escalation_rungs_used,
        status: left_status,
    } = left;
    let TrellisResult {
        predicted: right_predicted,
        log_evidence: right_log_evidence,
        logical_masses: right_logical_masses,
        runner_up_gap: right_runner_up_gap,
        peak_retained_states: right_peak_retained_states,
        processed_columns: right_processed_columns,
        transitions: right_transitions,
        dropped_states: right_dropped_states,
        dropped_log_mass: right_dropped_log_mass,
        bp_runs: right_bp_runs,
        bp_seconds: right_bp_seconds,
        escalation_rungs_used: right_escalation_rungs_used,
        status: right_status,
    } = right;
    assert_eq!(left_predicted, right_predicted);
    assert_eq!(left_log_evidence.to_bits(), right_log_evidence.to_bits());
    assert_eq!(
        left_runner_up_gap.map(f64::to_bits),
        right_runner_up_gap.map(f64::to_bits)
    );
    assert_eq!(left_peak_retained_states, right_peak_retained_states);
    assert_eq!(left_processed_columns, right_processed_columns);
    assert_eq!(left_transitions, right_transitions);
    assert_eq!(left_dropped_states, right_dropped_states);
    assert_eq!(
        left_dropped_log_mass.expect("enabled diagnostic").to_bits(),
        right_dropped_log_mass
            .expect("enabled diagnostic")
            .to_bits()
    );
    assert_eq!(left_bp_runs, right_bp_runs);
    assert_eq!(left_escalation_rungs_used, right_escalation_rungs_used);
    assert_eq!(left_status, right_status);
    assert_eq!(left_logical_masses.len(), right_logical_masses.len());
    for (left, right) in left_logical_masses.iter().zip(right_logical_masses) {
        let TrellisLogicalMass {
            logical: left_logical,
            log_mass: left_log_mass,
        } = left;
        let TrellisLogicalMass {
            logical: right_logical,
            log_mass: right_log_mass,
        } = right;
        assert_eq!(left_logical, right_logical);
        assert_eq!(left_log_mass.to_bits(), right_log_mass.to_bits());
    }
    // Actual BP timings vary between calls; zero timings are deterministic.
    if *left_bp_runs == 0 {
        assert_eq!(left_bp_seconds.to_bits(), right_bp_seconds.to_bits());
    } else {
        assert!(left_bp_seconds.is_finite() && *left_bp_seconds >= 0.0);
        assert!(right_bp_seconds.is_finite() && *right_bp_seconds >= 0.0);
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
