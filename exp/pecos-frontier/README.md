# PECOS Frontier Decoder

Native Rust implementation of the Frontier approximate logical maximum-likelihood
decoder (Leverrier & Urbanke, arXiv:2606.20513). Not a wrap of the upstream
`frontier` package; the upstream implementation is used as a verification oracle.

**Experimental** (`exp/`): the algorithm core is enumeration- and upstream-verified
(per-shot parity on matched models), but the crate has not yet accumulated real-user
mileage. Python exposes `pecos_rslib_exp.frontier()` for parallel `SampleBatch.decode`
and `DemSampler.decode` execution. The experimental extension is optional;
standard `pecos-rslib` and `pecos-decoders` do not depend on this crate. The
detailed per-shot API remains available as `pecos_rslib_exp.FrontierDecoder`.
Graduation to `crates/` awaits broader real-world use.

Pruning ranks accumulated prefix log mass plus a `score_alpha`-weighted
suffix-compatibility estimate. Unpruned results are exact and upstream-verified.

The discarded-prefix-mass diagnostic is off by default. Set
`FrontierConfig::dropped_mass` to `true`, or pass `dropped_mass=True` to a Python
detailed decoder constructor, to include it. Float metrics accumulate it at
extra pruning cost; the integer max-log metric always keeps its cheap running
maximum.
`dropped_log_mass` is `Option<f64>` in Rust and `float | None` in Python.
The flag applies to every entry point and never changes predictions, retained
logical masses, evidence, or other result fields.
Opt-in `incremental_score` carries suffix scores for binary float decoding, speeding up wide boundaries; rounding can change near-tie pruning outside snapshot/reference guarantees, while unpruned results are unchanged.
With incremental scoring enabled, construction fails for non-finite static suffix values when BP scoring is off; fixed-lag streaming rejects this option, and it can be slower on narrow models with few active detectors.

Deterministic ordering and tie-breaking are bitwise reproducible, and the
engine's `libm` transcendentals make outputs bit-identical across platforms.

## Saved detector events and independent windows

`windowed::WindowedDecoder` projects a sparse DEM into overlapping windows and
commits a maximum retained **joint** class. It accepts sparse hyperedges and
multiple logical observables. `WindowPlan` exposes every numerical model, included
mechanism index and global detector mapping for auditing or independent use.

Supply a chronological partition of detector IDs, including initial, terminal and
empty blocks. An error belongs to its earliest detector block. Each window
includes errors owned in its observed block range, in original DEM order. For
committed errors, class bits contain the original logical effects followed by
effects on ascending global future-detector IDs. The chosen class XORs into the
logical prediction and future residual. The last window commits its entire tail.
The adapter does not decompose hyperedges, fit probabilities or merge errors.

`ZeroSyndromePolicy::Decode` runs ordinary committee decoding on zero syndromes.
`SkipAndCommitZero` explicitly skips those windows; this approximation can change
accuracy when a model's zero syndrome has nonzero logical MAP. Detector-free
mechanisms require explicit owner blocks. Full-DEM decoding needs no block or owner
metadata. A persistent fixed-lag beam remains available through the separate
`StreamingMode::FixedLag` API.

The `decode_dem` example reads an existing flattened DEM and JSONL detector events.
It uses K=512, Delta=1e9, score_alpha=0.8, exact float log-sum-exp, no
indistinguishable-error merging, BP iterations=0 and int_metric_scale=1024 (unused
by the float metric). Each projected model uses deadline order and its plain
reverse committee. The example enables `dropped_mass`; full retained classes, evidence,
discarded mass and work telemetry are computed. The step budget is 32 columns; it is not a wall-time limit.
`WindowedDecoder::decode` is a synchronous whole-shot call that drains every
window before returning. The internal step budget does not bound that public
call or a future FFI wrapper to 32 columns or a controller latency deadline.

```sh
cargo run -p pecos-frontier --release --example decode_dem -- \
  windowed model.dem windows.json events.jsonl new-results
```

For window width 8 and stride 2, `windows.json` contains:

```json
{
  "schema": "frontier-earliest-owner-joint-window.v1",
  "window_blocks": 8,
  "stride_blocks": 2,
  "detector_blocks": [[0], [1], [2], [3], [4], [5], [6], [7], [8]],
  "zero_policy": "skip_and_commit_zero",
  "detector_free_error_owners": {}
}
```

Use the experiment's actual block partition. The example above has one detector
per block and is not a convention inferred from a DEM. Owner-map keys, when
needed, are flattened zero-based error-mechanism indices, including probability-zero
errors. Overrides on detector-supported errors and unknown configuration fields
are rejected. Repeat/shift syntax must be flattened to absolute detector IDs before
input; correlated `^` components retain their shared probability and XOR effects.

Each event row has a unique string or nonnegative integer `shot_id`, exactly one
integer bit per DEM detector, and optional logical truth:

```json
{"shot_id": "a", "detectors": [1,0,0,0,0,0,0,0,1], "truth": [1]}
{"shot_id": "b", "detectors": [0,0,0,0,0,0,0,0,0]}
```

Truth has exactly one bit per original observable and is used only after decoding.
Unknown truth is omitted; it is not assumed zero. Input order is preserved without
deduplication or answer caching. Decoder failures remain error rows and are counted
separately from successful known-truth accuracy. Malformed inputs produce a nonzero
exit; a partial output is not a completed cohort. Outputs must be new directories.

`predictions.jsonl` contains predictions, full selected-class telemetry and per-window
logical marginal LLRs. LLRs and minima are diagnostics, not postselection rules.
Finite values and IEEE bits are both saved; infinities use a null numeric value
and an explicit tag. `summary.json` separates decoder errors and truth denominators;
failure counts/rates are null when no successful shot has truth.
Window mode also writes `resolved_window_models.json` with the projected models,
mappings and exact orders.
Full-DEM mode writes `full_dem_config.json` with schema
`frontier-full-dem-config.v1`, model dimensions, all eight numerical settings,
the committee policy and the internal column budget. `numeric_config.column_order`
is the exact forward order; `backward_column_order` is its plain reverse. Both
refer to zero-based mechanisms in the input DEM. Keep the input DEM and source
revision with these outputs when archiving a run. Configuration files are written
before event decoding; only `summary.json` indicates a completed cohort.

To inspect geometry without running DP, use `resolve-windows MODEL CONFIG OUTPUT`.
To decode the entire DEM, use `full-dem MODEL EVENTS OUTPUT`. These are local native
interfaces; native per-call durations are not controller or WASM latency measurements.

## Benchmarks

Run the Criterion decode benchmark from the repository root:

```sh
cargo bench -p pecos-frontier --bench decode
# Filter by any part of the benchmark id (DEM, regime, or flag):
cargo bench -p pecos-frontier --bench decode -- 'd7_z_stim/k-capped'
cargo bench -p pecos-frontier --bench decode -- committee
```

Each iteration decodes all 32 deterministically sampled shots (Xoshiro256PlusPlus,
seed `0x5eed`) from a committed surface-code DEM. Decoder cases use d5 and d7 Z
Stim DEMs, crossing `dropped_mass=false/true` with `wide` (default K and delta),
`k-capped` (K=8, delta=infinity), and `delta-capped` (K=65536, delta=2).
The committee group uses d5 with default settings and both flag values.
Construction, sampling, and setup validation are outside timing. Setup reports
dropped-state totals and decode-error counts and checks flag parity; failed shots
remain in the timed workload. Decoder totals include no-path attempts; committee
totals cover only selected successful legs. Criterion reports time per 32-shot
batch and throughput in shots/second; divide batch time by 32 for time per shot.

Latency numbers must not be taken with the `pecos-trellis/runtime-checks` feature
enabled: it adds reference-path validation work. This bench is not run in CI.
