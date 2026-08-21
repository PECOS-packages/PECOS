# Changelog

PECOS uses GitHub to manage both Python and Rust releases.

Please see our [GitHub releases page](https://github.com/PECOS-packages/PECOS/releases) for the changelog.

## Unreleased

### Python batch decoding

The legacy batch-decode entry points have been removed in favor of the unified
`SampleBatch.decode(...)` and `DemSampler.decode(...)` APIs:

- `SampleBatch.decode_count(dem, decoder)` becomes `SampleBatch.decode(dem, decoder).num_errors`.
- `SampleBatch.decode_each(dem, decoder)` becomes
  `SampleBatch.decode(dem, decoder, predictions=True).predictions`.
- `SampleBatch.decode_count_parallel(dem, decoder, num_workers=N)` becomes
  `SampleBatch.decode(dem, decoder, workers=N).num_errors`.
- `SampleBatch.decode_count_batch(dem)` becomes
  `SampleBatch.decode(dem, pymatching(correlated=False)).num_errors`.
- `SampleBatch.decode_stats(dem, decoder)` becomes
  `SampleBatch.decode(dem, decoder, timing=True)`; read counts from the result and timing from `.stats`.
- `SampleBatch.decode_stats_parallel(dem, decoder, num_workers=N)` becomes
  `SampleBatch.decode(dem, decoder, workers=N, timing=True)`.
- `DemSampler.sample_decode_count(dem, shots, decoder, seed=seed)` becomes
  `DemSampler.decode(dem, shots, decoder, seed=seed).num_errors`.
- `DemSampler.sample_decode_count_parallel(dem, shots, decoder, seed=seed, num_workers=N)` becomes
  `DemSampler.decode(dem, shots, decoder, seed=seed, workers=N).num_errors`.

`DemSampler.decode` uses a new canonical sampling ABI. A fixed seed therefore
produces a deliberately different shot stream from the removed sampling
methods, so seeded results are not comparable across this change. In return,
the new stream is reproducible independently of worker count.

The legacy `"pymatching"` type string continues to mean correlated matching.
The typed `pymatching(correlated=...)` factory requires callers to choose the
correlation mode explicitly.
