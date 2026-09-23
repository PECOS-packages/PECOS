//! Shared worker loop for trellis decoding policies.

use crate::DecoderError;
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Decode dense shots with fresh worker state, returning outcomes in input order.
///
/// The factory must share immutable models and allocate independent scratch.
///
/// # Errors
/// Returns `InvalidConfiguration` for zero threads or pool construction failure.
pub fn decode_batch<W, R: Send>(
    shots: &[Vec<u8>],
    threads: usize,
    worker: impl Fn() -> W + Sync,
    decode: impl Fn(&mut W, &[u8]) -> R + Sync,
) -> Result<Vec<R>, DecoderError> {
    if threads == 0 {
        return Err(DecoderError::InvalidConfiguration(
            "threads must be at least 1".into(),
        ));
    }
    if threads == 1 {
        let mut state = worker();
        return Ok(shots.iter().map(|shot| decode(&mut state, shot)).collect());
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .map_err(|error| DecoderError::InvalidConfiguration(error.to_string()))?;
    // Mirror pecos-decoders/src/batch.rs:51-65 without a dependency on the
    // higher-level decoder aggregator: 64 shots, reduced for small batches.
    let chunk_shots = 64.min(shots.len().div_ceil(threads)).max(1);
    let cursor = AtomicUsize::new(0);
    let mut chunks: Vec<_> = pool.install(|| {
        (0..threads)
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
