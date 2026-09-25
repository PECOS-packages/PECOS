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

//! Shared worker loop for trellis decoding policies.

use crate::DecoderError;
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Mirrors `PARALLEL_CHUNK_SHOTS` in `crates/pecos-decoders/src/batch.rs` without
/// depending on the higher-level decoder aggregator.
/// The per-run chunk cap spreads small batches across workers.
const PARALLEL_CHUNK_SHOTS: usize = 64;

/// Decode dense shots with fresh worker state, returning outcomes in input order.
///
/// The factory must share immutable models and allocate independent scratch.
/// Workers are capped at one per shot, with one worker for an empty batch.
///
/// # Errors
/// Returns `InvalidConfiguration` for zero workers and `InternalError` for pool construction failure.
pub fn decode_batch<W, R: Send>(
    shots: &[Vec<u8>],
    workers: usize,
    worker: impl Fn() -> W + Sync,
    decode: impl Fn(&mut W, &[u8]) -> R + Sync,
) -> Result<Vec<R>, DecoderError> {
    if workers == 0 {
        return Err(DecoderError::InvalidConfiguration(
            "workers must be at least 1".into(),
        ));
    }
    let workers = workers.min(shots.len().max(1));
    if workers == 1 {
        let mut state = worker();
        return Ok(shots.iter().map(|shot| decode(&mut state, shot)).collect());
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .map_err(|error| DecoderError::InternalError(error.to_string()))?;
    let chunk_shots = PARALLEL_CHUNK_SHOTS
        .min(shots.len().div_ceil(workers))
        .max(1);
    let cursor = AtomicUsize::new(0);
    let mut chunks: Vec<_> = pool.install(|| {
        (0..workers)
            .into_par_iter()
            .map(|_| {
                let mut state = worker();
                let mut chunks = Vec::new();
                loop {
                    let chunk = cursor.fetch_add(1, Ordering::Relaxed);
                    if chunk >= shots.len().div_ceil(chunk_shots) {
                        break;
                    }
                    let start = chunk * chunk_shots;
                    let end = (start + chunk_shots).min(shots.len());
                    let results: Vec<_> = shots[start..end]
                        .iter()
                        .map(|shot| decode(&mut state, shot))
                        .collect();
                    chunks.push((chunk, results));
                }
                chunks
            })
            .flatten()
            .collect()
    });
    chunks.sort_unstable_by_key(|(index, _)| *index);
    Ok(chunks
        .into_iter()
        .flat_map(|(_, results)| results)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructs_exactly_the_capped_worker_states() {
        for workers in [1, 4] {
            for shots in [vec![], vec![vec![7]], vec![vec![7]; 2], vec![vec![7]; 129]] {
                let expected_workers = workers.min(shots.len().max(1));
                let factories = AtomicUsize::new(0);
                let results = decode_batch(
                    &shots,
                    workers,
                    || {
                        factories.fetch_add(1, Ordering::Relaxed);
                    },
                    |(), shot| {
                        if expected_workers > 1 {
                            assert_eq!(rayon::current_num_threads(), expected_workers);
                        }
                        shot.to_vec()
                    },
                )
                .unwrap();
                assert_eq!(factories.load(Ordering::Relaxed), expected_workers);
                assert_eq!(results, shots);
            }
        }
    }
}
