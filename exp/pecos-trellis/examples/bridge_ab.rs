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

//! Cross-implementation A/B harness: decode upstream-frontier sample shots
//! with `FrontierDecoder` on the identical model and column order.
//!
//! Input JSON (produced by an external extraction script from the upstream
//! `frontier` package): `{num_detectors, num_observables, mechanisms:
//! [[p, [detectors], [observables]], ...], shots: [{fired, truth_logical}]}`
//! where mechanism order IS the processing order and `fired` lists the indices
//! of the detectors that fired.
//!
//! Usage: `bridge_ab <model.json> <k> <delta> <score_alpha> [bp_score_iterations] [workers] [stream=<chunk>]`
//! Prints one `shot,predicted,truth,status,gap,log_evidence,seconds` line per
//! shot (no-path rows leave the gap and evidence fields empty) plus a summary
//! line. Streaming reports the zero-based index of the last processed column
//! at first non-reset commitment; NaN means no such commitment occurred.
//! Both `committed_before_flush` and `early_committed_bits` exclude bits that never toggle.
//! Lookahead includes logical-only columns, whose own detector index is -1.

use pecos_decoder_core::dem::SparseDem;
use pecos_trellis::frontier::{FrontierConfig, FrontierDecoder, TrellisStreamingDecoder};
use serde::Deserialize;
use std::collections::BTreeMap;

#[derive(Deserialize)]
struct BridgeModel {
    num_detectors: usize,
    num_observables: usize,
    mechanisms: Vec<(f64, Vec<u32>, Vec<u32>)>,
    shots: Vec<Shot>,
}

#[derive(Deserialize)]
struct Shot {
    /// Fired detector indices (supports arbitrary detector counts).
    fired: Vec<u32>,
    truth_logical: u128,
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: bridge_ab <model.json> <k> <delta> <score_alpha>");
    let k: usize = args.next().expect("missing k").parse().expect("k");
    let delta: f64 = args.next().expect("missing delta").parse().expect("delta");
    let score_alpha: f64 = args
        .next()
        .expect("missing score_alpha")
        .parse()
        .expect("score_alpha");
    let bp_score_iterations: usize = args
        .next()
        .map_or(0, |raw| raw.parse().expect("bp_score_iterations"));

    let mut workers = None;
    let mut stream_chunk = None;
    for raw in args {
        if let Some(chunk) = raw.strip_prefix("stream=") {
            let chunk: usize = chunk.parse().expect("stream chunk size");
            assert!(chunk > 0, "stream chunk size must be positive");
            assert!(
                stream_chunk.replace(chunk).is_none(),
                "duplicate stream argument"
            );
        } else {
            assert!(
                workers.replace(raw.parse().expect("workers")).is_none(),
                "duplicate workers argument"
            );
        }
    }
    assert!(
        workers.is_none() || stream_chunk.is_none(),
        "workers and stream modes are exclusive"
    );

    let model: BridgeModel =
        serde_json::from_str(&std::fs::read_to_string(&path).expect("read model json"))
            .expect("parse model json");
    let dem = SparseDem {
        mechanisms: model.mechanisms,
        detector_coords: BTreeMap::new(),
        num_detectors: model.num_detectors,
        num_observables: model.num_observables,
    };
    let config = FrontierConfig {
        k,
        delta,
        score_alpha,
        column_order: None,
        merge_indistinguishable: false,
        bp_score_iterations,
        metric_mode: pecos_trellis::frontier::MetricMode::default(),
        int_metric_scale: 1024,
    };
    let mut stream = stream_chunk.map(|_| {
        TrellisStreamingDecoder::from_sparse_dem(&dem, config.clone())
            .expect("build streaming decoder")
    });
    let mut decoder = stream_chunk
        .is_none()
        .then(|| FrontierDecoder::from_sparse_dem(&dem, config).expect("build decoder"));
    let mut committed_before_flush = 0_u64;
    let mut early_committed_bits = 0_u64;
    let mut first_commit_columns = 0.0;
    let mut shots_with_commitments = 0_u32;

    let mut failures = 0_u32;
    let mut no_path = 0_u32;
    let started = std::time::Instant::now();
    assert!(
        model.num_observables <= 128,
        "bridge truth_logical is u128; wider observables need a format change"
    );
    let mut batch = workers.map(|workers| {
        let shots: Vec<_> = model
            .shots
            .iter()
            .map(|entry| {
                let mut syndrome = vec![0; model.num_detectors];
                for &fired in &entry.fired {
                    syndrome[fired as usize] = 1;
                }
                syndrome
            })
            .collect();
        decoder
            .as_ref()
            .expect("batch mode has a decoder")
            .decode_batch(&shots, workers)
            .expect("decode batch")
            .into_iter()
    });
    let mut syndrome = vec![0_u8; model.num_detectors];
    for (shot, entry) in model.shots.iter().enumerate() {
        syndrome.fill(0);
        for &fired in &entry.fired {
            syndrome[fired as usize] = 1;
        }
        let shot_started = std::time::Instant::now();
        let outcome = if let Some(stream) = &mut stream {
            stream.reset();
            let reset_mask = stream.committed().1;
            let mut first_commit = None;
            let outcome = (|| {
                for chunk in syndrome.chunks(stream_chunk.expect("stream mode has a chunk size")) {
                    stream.feed_prefix(chunk)?;
                    let progress = stream.advance()?;
                    let count = u64::try_from(
                        progress
                            .newly_committed
                            .iter()
                            .filter(|(logical, _)| !reset_mask.get(*logical))
                            .count(),
                    )
                    .expect("commit count fits u64");
                    if progress.columns_processed < stream.column_lookahead().len() {
                        early_committed_bits += count;
                    }
                    if count != 0 {
                        first_commit.get_or_insert(progress.columns_processed);
                    }
                }
                committed_before_flush +=
                    u64::from(stream.committed().1.count_ones() - reset_mask.count_ones());
                stream.flush()
            })();
            if let Some(column) = first_commit {
                first_commit_columns +=
                    f64::from(u32::try_from(column).expect("column count fits u32")) - 1.0;
                shots_with_commitments += 1;
            }
            outcome
        } else if let Some(batch) = &mut batch {
            match batch.next().expect("one result per shot") {
                pecos_trellis::frontier::FrontierDecodeAttempt::Success(result) => Ok(result),
                pecos_trellis::frontier::FrontierDecodeAttempt::NoPath { error, .. }
                | pecos_trellis::frontier::FrontierDecodeAttempt::Error(error) => Err(error),
            }
        } else {
            decoder
                .as_mut()
                .expect("sequential mode has a decoder")
                .decode(&syndrome)
        };
        // Batch mode has no per-shot wall-clock measurement.
        let shot_seconds = if workers.is_some() {
            f64::NAN
        } else {
            shot_started.elapsed().as_secs_f64()
        };
        if let Err(error) = &outcome {
            // Only a genuine no-path is a shot outcome. Anything else is an
            // engine fault, and recording it as no_path would silently skew
            // the A/B comparison.
            assert!(
                matches!(
                    error,
                    pecos_trellis::frontier::DecoderError::DecodingFailed(_)
                ),
                "engine fault on shot {shot}: {error}"
            );
        }
        if let Ok(result) = outcome {
            let words = result.predicted.words();
            assert!(words.iter().skip(2).all(|&w| w == 0), "label fits u128");
            let predicted = u128::from(words.first().copied().unwrap_or(0))
                | (u128::from(words.get(1).copied().unwrap_or(0)) << 64);
            let status = if predicted == entry.truth_logical {
                "ok"
            } else {
                failures += 1;
                "logical_fail"
            };
            let gap = result
                .runner_up_gap
                .map_or(String::from("inf"), |g| format!("{g:.6}"));
            println!(
                "{shot},{predicted},{},{status},{gap},{:.6},{shot_seconds:.6}",
                entry.truth_logical, result.log_evidence
            );
        } else {
            failures += 1;
            no_path += 1;
            println!(
                "{shot},,{},no_path,,,{shot_seconds:.6}",
                entry.truth_logical
            );
        }
    }
    let elapsed = started.elapsed().as_secs_f64();
    let trials = u32::try_from(model.shots.len()).expect("shot count fits u32");
    let streaming_summary = stream.as_ref().map_or_else(String::new, |stream| {
        let lookahead = stream.column_lookahead();
        let maximum = lookahead.iter().copied().max().unwrap_or(0);
        let total: f64 = lookahead
            .iter()
            .map(|&value| f64::from(u32::try_from(value).expect("lookahead fits u32")))
            .sum();
        let mean = total
            / f64::from(
                u32::try_from(lookahead.len())
                    .expect("column count fits u32")
                    .max(1),
            );
        let first_mean = if shots_with_commitments == 0 {
            f64::NAN
        } else {
            first_commit_columns / f64::from(shots_with_commitments)
        };
        format!(
            " committed_before_flush={committed_before_flush} \
             early_committed_bits={early_committed_bits} \
             shots_with_commitments={shots_with_commitments} \
             first_commit_column_mean={first_mean} \
             lookahead_max={maximum} lookahead_mean={mean}"
        )
    });
    println!(
        "SUMMARY trials={trials} fail={failures} no_path={no_path} fer={} k={k} delta={delta} alpha={score_alpha} decode_s_mean={}{streaming_summary}",
        f64::from(failures) / f64::from(trials),
        elapsed / f64::from(trials),
    );
}
