# Experimental decoders

Decoders in `exp/` that are not yet part of the unified `pecos.decoders` surface.
They are reached through `pecos_rslib_exp` and may change without notice.

Two capabilities live here that the production decoders do not offer:

- **Complementary-gap output** — how much more likely the winning logical class
  was than the runner-up, per shot. Production decoders return a correction and,
  for some backends, a weight or cost; none return a gap.
- **Hyperedge decoding with a confidence signal** — Frontier and BP-Trellis
  decode a model containing mechanisms that touch three or more detectors,
  which matching-style decoders cannot represent at all.

## Frontier and BP-Trellis

Both take a Stim-format DEM string and expose the same result surface.
`FrontierDecoder` and `BpTrellisDecoder` are each a single trellis decoder over
the same machinery (BP-Trellis adds BP-informed pruning scores); the separate
`FrontierCommitteeDecoder` runs a two-leg forward/backward committee over them.
Start with whichever you like and compare.

<!--test-name: experimental_decoders_gap-->
```python
from pecos_rslib_exp import BpTrellisDecoder, FrontierDecoder

# The only mechanism carrying L0 touches three detectors, so this model is not
# graphlike: matching-style decoders (pymatching, fusion_blossom, pecos_uf,
# k_mwpm, ...) reject it (see below).
dem = "error(0.1) D0 D1 D2 L0\nerror(0.05) D0\nerror(0.05) D1\nerror(0.05) D2\n"

result = FrontierDecoder.from_dem(dem).decode_syndrome([1, 1, 1])

assert list(result.observable_flips) == [True]
assert result.status == "exact"
print(f"status={result.status} gap={result.runner_up_gap:.3f}")
```

`decode_batch` takes a list of syndromes and returns one result per shot:

<!--continuation-->
```python
batch = FrontierDecoder.from_dem(dem).decode_batch([[1, 1, 1], [0, 0, 0]])

assert [list(shot.observable_flips) for shot in batch] == [[True], [False]]
```

BP-Trellis exposes the identical result surface:

<!--continuation-->
```python
trellis = BpTrellisDecoder.from_dem(dem).decode_syndrome([1, 1, 1])

assert list(trellis.observable_flips) == [True]
assert trellis.status == "exact"
```

### Reading the complementary gap

`logical_masses` holds `(observable_mask, log_mass)` pairs ordered by decreasing
mass, and `runner_up_gap` is the log-mass difference between the first two --
`None` whenever fewer than two logical classes survive to the end, which
includes a fully exact decode with a single reachable class, so treat a
missing gap as "no runner-up existed", not as a pruning signal, and handle it
before comparing.
With the default float metric, when `status` is `"exact"` (nothing was pruned),
the masses are the true unnormalized posteriors: a large gap means one logical
class really was overwhelmingly more likely, and a gap near zero means the
shot was nearly a coin flip -- a natural candidate for post-selection or a
soft-output pipeline. Under the integer `maxlog_int` metric (Rust API only for
now), each terminal mass is instead the best-route (Viterbi) mass for that
logical class, the gap is a route-mass margin, and `log_evidence` is the winning
route mass rather than evidence.
When the result was pruned, the gap and the masses describe only what the
search retained, not a certified confidence -- treat them as search
diagnostics, not posteriors.

<!--continuation-->
```python
assert result.status == "exact"
if result.runner_up_gap is not None:
    winner, runner_up = result.logical_masses[0], result.logical_masses[1]
    assert winner[1] - runner_up[1] == result.runner_up_gap
    print(f"winner mask={winner[0]} log-mass={winner[1]:.3f}")
```

The remaining fields describe the search itself rather than the answer:
`log_evidence` (total retained log evidence under the default float metric),
`status` (`"exact"` when nothing was pruned), `dropped_states` and
`dropped_log_mass` (what pruning discarded),
`peak_retained_states`, `processed_columns`, `escalation_rungs_used`, and
`bp_seconds`. `dropped_log_mass` counts only the mass states carried when they
were pruned -- a state dropped early would have branched through later columns,
so it is not a bound on the probability the answer is missing.

## Decoding a hyperedge model with a matching decoder

Matching-style decoders build a graph whose edges carry at most two detectors,
so PECOS rejects a hyperedge model rather than dropping the mechanisms it cannot
represent:

```
Invalid configuration: fusion_blossom needs a graphlike model, but this DEM has
113 mechanism(s) touching three or more detectors. Decoding it here would
silently ignore them. Pass a decomposed model
(DetectorErrorModel.to_string_terminal_graphlike_decomposed() or
to_string_source_graphlike_decomposed()), or use a decoder that accepts
hyperedges such as bp_osd or tesseract.
```

There are two ways forward. The production option is a decoder that represents
hyperedges directly — `bp_osd()` or `tesseract()` from `pecos.decoders`, or the
Frontier and BP-Trellis decoders above.

The experimental option is to build a genuinely decomposed model.
`coherent_dem_decomposed` traces a `TickCircuit` and uses Pauli provenance to
split each hyperedge into its X and Z components, fitting the probabilities to
Heisenberg-exact marginals. It needs a circuit whose detectors are defined --
one built by the QEC layer rather than a bare `TickCircuit`, which yields an
empty model:

<!--skip: needs a detector-bearing circuit from the QEC layer-->
```python
from pecos_rslib_exp import coherent_dem_decomposed

raw_dem, decomposed_dem = coherent_dem_decomposed(
    tick_circuit,
    p1=0.001,
    p2=0.001,
    p_meas=0.001,
)
```

The second return value is graphlike and can go to any matching decoder.
`noise_characterization` exposes the same machinery with an explicit
`max_order`. Note the distinction from `ParsedDem.to_string_decomposed()`, which
splits only mechanisms whose provenance is already recorded and raises for the
rest — it has no circuit to re-derive provenance from.

## What is not here yet

These decoders are not reachable through `DecoderSpec` / `pecos.decoders`, so
they cannot be passed to `SampleBatch.decode(...)` or `DemSampler.decode(...)`
and do not participate in unified execution planning, batching, or timing. Use
their own `decode_syndrome` / `decode_batch` methods directly.
