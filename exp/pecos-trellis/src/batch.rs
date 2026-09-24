//! Shared worker loop for trellis decoding policies.

use crate::DecoderError;
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Mirrors `PARALLEL_CHUNK_SHOTS` in `crates/pecos-decoders/src/batch.rs` without
/// depending on the higher-level decoder aggregator.
const PARALLEL_CHUNK_SHOTS: usize = 64;

/// Decode dense shots with fresh worker state, returning outcomes in input order.
///
/// The factory must share immutable models and allocate independent scratch.
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
    fn constructs_exactly_the_requested_worker_states() {
        for workers in [1, 4] {
            for shots in [vec![], vec![vec![7]; 129]] {
                let factories = AtomicUsize::new(0);
                let results = decode_batch(
                    &shots,
                    workers,
                    || {
                        factories.fetch_add(1, Ordering::Relaxed);
                    },
                    |(), shot| shot.to_vec(),
                )
                .unwrap();
                assert_eq!(factories.load(Ordering::Relaxed), workers);
                assert_eq!(results, shots);
            }
        }
    }
}
