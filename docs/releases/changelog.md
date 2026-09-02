# Changelog

PECOS uses GitHub to manage both Python and Rust releases.

Please see our [GitHub releases page](https://github.com/PECOS-packages/PECOS/releases) for the changelog.

## Unreleased

### Rust breaking changes

- `CliffordGateable::apply_global_phase` replaces the former
  `ArbitraryRotationGateable::apply_global_phase` hook. This is source-breaking
  for out-of-tree implementors that override or call the hook through the old
  trait. Projective backends may retain the no-op default; amplitude-exposing
  backends must implement it.

### Rust bug fixes

- Dense rotation-family matrices now use the signed `(-pi, pi]` angle
  representative and agree exactly with the simulators. This changes
  `ToMatrix` output by a global `-1` for stored negative rotation angles,
  negative `theta` in `RXY1Q` and `U3`, composites containing those rotations,
  and the named `SXXdg`, `SYYdg`, and `SZZdg` gates.
- The `CliffordGateable` default decompositions now deliver their residual
  global phases through `apply_global_phase`. Amplitude-exposing backends that
  inherit these defaults therefore change state by the required global phase;
  projective backends continue to use the no-op hook.
- `Angle::to_radians_signed`, `to_turns_signed`, and
  `to_half_turns_signed` now choose the principal-value sign from the stored
  fraction instead of a rounded floating-point value. Exactly `HALF_TURN`
  remains positive and maps to `+pi`, `+0.5`, and `+1.0`, respectively; stored
  fractions strictly above it map to the negative representative.

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
