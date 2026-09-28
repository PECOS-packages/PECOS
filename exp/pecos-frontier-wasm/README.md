# PECOS Frontier bare-WebAssembly adapter

This crate compiles the PECOS Frontier decoder into a WebAssembly module with
no imports. Its exported functions use only `i32` parameters and at most one
`i32` result. That lowest-common-denominator ABI allows the same module to run
on Quantinuum hardware, which requires those integer-only signatures.

## Build

The default build embeds `model.dem`, a tiny smoke-test model:

```console
just build-frontier-wasm
```

To embed a real, flattened Stim detector error model:

```console
just build-frontier-wasm /path/to/model.dem
```

The output is `dist/pecos_frontier_wasm.wasm`. The live-call adapter supports
at most 128 detectors and 128 logical observables. Embedded replay fixtures
support wider detector models while retaining the 128-observable result limit.
Detector and observable bit `i` is word `i / 32`, bit `i % 32`. The build
validates the model with PECOS's Stim DEM parser and rejects malformed,
unflattened, or oversized-observable models.

The underlying command is also cross-platform:

```console
uv run --frozen python scripts/build_frontier_wasm.py [flattened-model.dem]
```

### Compile hardware syndromes into the module

An NPZ containing binary `detection_events` and `observable_flips` arrays can
be packed into the module for self-contained replay. No syndrome file is read
while the Wasm module runs.

```console
uv run --frozen --group numpy-compat python scripts/build_frontier_wasm.py \
  /path/to/model.dem \
  --syndromes /path/to/hardware_shots.npz \
  --replay-output /path/to/hardware_shots.fwr \
  --output dist/hardware_frontier.wasm
```

Rebuild the same module from the compact FWR1 fixture without NumPy:

```console
uv run --frozen python scripts/build_frontier_wasm.py \
  /path/to/model.dem --replay /path/to/hardware_shots.fwr \
  --output dist/hardware_frontier.wasm
```

Replay mode supports per-shot calls and an in-module range mode that amortizes
the host/Wasm boundary:

```console
node scripts/benchmark_frontier_wasm.mjs dist/hardware_frontier.wasm \
  --shots 16 --mode shot
node scripts/benchmark_frontier_wasm.mjs dist/hardware_frontier.wasm \
  --shots 16 --mode range
```

Use `scripts/validate_frontier_wasm_predictions.mjs` to record the full
prediction SHA-256 and logical-error count. Passing `--fixture` also validates
older modules that expose `frontier_decode` but do not embed replay data.

## Streaming detector rounds

The adapter uses PECOS's allocation-reusing streaming trellis engine. Start a
shot, append contiguous detector blocks as they arrive, and flush after the
last detector. The retained frontier advances between calls, moving most work
out of the final correction interval.

```console
node scripts/benchmark_frontier_wasm.mjs dist/hardware_frontier.wasm \
  --shots 8 --mode stream --rounds 21

# Time only the final detector block through the returned correction.
node scripts/benchmark_frontier_wasm.mjs dist/hardware_frontier.wasm \
  --shots 8 --mode correction --rounds 21
```

Correction reports include p50, p95, p99, maximum, and the fraction below 50
ms. Use `--stride` to sample syndromes across a large fixture instead of
benchmarking only adjacent shots.

## WebAssembly ABI

- `init() -> ()`: constructs the embedded decoder.
- `frontier_decode(i32, i32, i32, i32) -> ()`: asynchronous-friendly decode.
- `frontier_stream_begin() -> ()`: reset the persistent frontier for a shot.
- `frontier_stream_push(i32, i32, i32, i32, i32) -> i32`: append
  `bit_count` contiguous detector bits packed into four words; returns the
  number of newly processed DEM columns, or -1 on failure.
- `frontier_stream_finish() -> ()`: flush after all detector bits arrive.
- `frontier_result_0..3() -> i32`: four observable-mask words.
- `frontier_status() -> i32`: 0 success, 1 model error, 2 model too wide,
  3 decode error, 4 replay error.
- `frontier_reset() -> ()`: clears per-shot output.
- `frontier_replay_shot_count() -> i32`: number of embedded shots.
- `frontier_replay_shot(i32) -> i32`: decode one embedded shot and return its
  logical mismatch bit, or -1 on failure.
- `frontier_replay_stream_prepare(i32, i32) -> i32` and
  `frontier_replay_stream_finish() -> i32`: process prior rounds outside the
  timing interval and then measure final-block-to-correction latency.
- `frontier_replay_range(i32, i32) -> i32`: decode an embedded range and
  return its logical-error count.

`frontier_decode` rejects any set syndrome bit at or above the embedded model's
detector count with status 3 instead of silently discarding it.

Call `frontier_reset` at the end of each shot when the host persists module
state between shots. Quantinuum requires this reset for in-memory Wasm state.

## Hardware latency

Before running a production model on hardware, tune the decoder configuration
and re-measure its latency on the target system. In a 2,000-shot Wasmtime JIT
benchmark on a fast desktop, a rotated-memory-Z model at physical error rate
0.001 took 4.1 ms median / 8.9 ms p99 / 42 ms maximum at distance 3 with three
rounds, and 57.9 ms median / 124.8 ms p99 / 158 ms maximum at distance 5 with
five rounds. The latter is close enough to Quantinuum's approximately 250 ms
per-call hardware limit that desktop measurements should not be treated as a
hardware safety margin.
