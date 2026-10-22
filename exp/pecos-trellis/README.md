# PECOS Trellis Decoder

Trellis dynamic-programming engine for logical coset-mass decoding.
The `bp_trellis` module provides PECOS's BP-guided preset.

## PECOS BP Trellis Decoder

PECOS's degeneracy-aware BP-guided trellis decoder is an approximate logical
maximum-likelihood decoder that is exact in the unpruned limit. Its optimality
is relative to the supplied detector error model, not the underlying physics,
and pruned results have no certified bound on discarded posterior mass. Belief
propagation guides only which states pruning retains; it does not change branch
probabilities or mass arithmetic. This is not a wrap or port of an external
project.

**Experimental** (`exp/`): the defaults and optional no-path escalation ladder
remain provisional pending broader validation. The `bp_trellis` module holds
PECOS's configuration and decoder facade on the engine in this crate.

The optional `pecos-rslib-exp` package provides `pecos_rslib_exp.bp_trellis(...)`
for `SampleBatch.decode` and `DemSampler.decode`, including parallel Rust workers.
Both ladder forms and all native configuration options are exposed. Standard `pecos-rslib` and
`pecos-decoders` do not depend on this crate. The direct
`pecos_rslib_exp.BpTrellisDecoder` API additionally returns detailed per-shot
confidence and retry telemetry.

Rust callers use `BpTrellisConfig::escalation: Vec<EscalationRung>` with one model
and `decode_outcome` / `decode_batch_outcomes` for explicit outcomes; `decode`,
`decode_batch`, and `ObservableDecoder` stay strict. See the
[BP-Trellis user guide](../../docs/user-guide/decoders.md#rust-backed-bp-trellis-batch-decoding)
for Python ladder forms, no-path reporting, and telemetry.

## Exact runtime fast paths

The binary kernel reuses compact sort records and retention buffers, emits
paired branches together, and groups suffix scoring without reassociating its
floating-point additions. Compact keys include the arrival index, preserving
full detector/logical key order and the order in which equal-key masses fold.
Scratch records and prefix plans trade additional retained memory for less
per-column work. Unsupported key shapes use the general comparison path. Fixed-lag streams
cache scoring-prefix ends for their configured schedule; delayed advances use
the original prefix discovery whenever the cached range does not match.

These paths are shared by native Frontier and the bare-WASM adapter. Decoder
settings, scheduling defaults, and public result semantics are unchanged.
They come from the accepted round61 runtime; the separately frozen round63
and round68 experiments are not included. Historical benchmark gains do not
establish the speed of a new compiler or WASM runtime.

Unit tests compare the fast paths with scalar references, including exact mass
and score bits. To enable the expensive comparisons throughout integration
workloads as well:

```console
cargo test -p pecos-trellis -p pecos-frontier -p pecos-frontier-wasm --features pecos-trellis/runtime-checks
```

`runtime-checks` is disabled by default. It is a validation feature and should
not be enabled in production or used for latency measurements.
