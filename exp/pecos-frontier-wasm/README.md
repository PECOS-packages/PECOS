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

The output is `dist/pecos_frontier_wasm.wasm`. `frontier_decode` accepts at most
128 detectors. Streaming supports wider models through blocks of at most 128
detectors per call. All paths retain the 128-observable result limit.
Detector and observable bit `i` is word `i / 32`, bit `i % 32`. The build
validates the model with PECOS's Stim DEM parser and rejects malformed,
unflattened, or oversized-observable models.

The underlying command is also cross-platform:

```console
uv run --frozen python scripts/build_frontier_wasm.py [flattened-model.dem]
```

### Compile hardware syndromes into the module

Use a PECOS `SampleBatch` corpus containing the DEM and hardware shots. The
builder loads it with `SampleBatch.load`, verifies its integrity, and requires
its embedded DEM to exactly match the DEM being compiled (including whitespace).
No syndrome file is read while the Wasm module runs. This option requires an
installed PECOS Python package with the corpus API.

```console
uv run --frozen python scripts/build_frontier_wasm.py \
  /path/to/model.dem \
  --corpus /path/to/hardware_shots.pecos \
  --replay-output /path/to/hardware_shots.fwr \
  --output dist/hardware_frontier.wasm
```

In your own analysis environment, convert existing NPZ data at the interop
boundary before using the builder:

```python
from pathlib import Path

import numpy as np
from pecos.qec import SampleBatch

with np.load("hardware_shots.npz", allow_pickle=False) as shots:
    detectors = shots["detection_events"].tolist()
    flips = shots["observable_flips"].tolist()
    width = shots["observable_flips"].shape[1]
    masks = [sum(int(bit) << i for i, bit in enumerate(row)) for row in flips]
SampleBatch(detectors, masks, num_observables=width).save("hardware_shots.pecos", dem=Path("model.dem").read_text())
```

Rebuild from a prepacked FWR1 **version 2** fixture:

```console
uv run --frozen python scripts/build_frontier_wasm.py \
  /path/to/model.dem --replay /path/to/hardware_shots.fwr \
  --output dist/hardware_frontier.wasm
```

Prepacked FWR1 stores widths, not DEM identity; the caller must pair it with
the correct model. Version 1 is unsupported. Replay exports and the fixture
are behind the `replay` Cargo feature, which the builder enables automatically
for `--corpus` or `--replay`. Default production builds omit both.

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

For a non-sampled report, omit `--shots` (or set it to the fixture size), use
`--repeat 1`, and bound startup work independently with `--warmup-shots`. For
example, `--warmup 1 --warmup-shots 32` warms the JIT on 32 records and then
times every selected hardware shot exactly once.

## WebAssembly ABI

- `init() -> ()`: constructs the embedded decoder.
- `frontier_decode(i32, i32, i32, i32) -> ()`: asynchronous-friendly decode.
- `frontier_stream_begin() -> ()`: reset the persistent frontier for a shot.
- `frontier_stream_push(i32, i32, i32, i32, i32) -> i32`: append
  `bit_count` contiguous detector bits packed into four words; returns the
  number of newly processed DEM columns, or -1 on failure.
- `frontier_stream_finish() -> ()`: flush after all detector bits arrive.
- `frontier_stream_finish_round(i32, i32, i32, i32, i32) -> i32`: append
  the final detector block, flush, and return the first correction word in one
  call. A return of -1 can mean failure **or a valid all-ones correction**;
  always check `frontier_status()`. This is the preferred hardware latency boundary.
- `frontier_result_0..3() -> i32`: four observable-mask words.
- `frontier_status() -> i32`: 0 success, 1 model error, 2 model too wide,
  3 decode error, 4 replay error.
- `frontier_reset() -> ()`: clears per-shot output.
The following exports require the `replay` feature:

- `frontier_replay_shot_count() -> i32`: number of embedded shots.
- `frontier_replay_shot(i32) -> i32`: decode one embedded shot and return its
  logical mismatch bit, or -1 on failure.
- `frontier_replay_stream_prepare(i32, i32) -> i32` and
  `frontier_replay_stream_finish() -> i32`: process prior rounds outside the
  timing interval and then measure final-block-to-correction latency.
- `frontier_replay_range(i32, i32) -> i32`: decode an embedded range and
  return its logical-error count.

`frontier_decode` rejects any set syndrome bit at or above the embedded model's
detector count with status 3 instead of silently discarding it. Streaming calls
likewise reject set bits at or above `bit_count`, including nonzero unused
words. Failed decode/stream calls clear the correction words.

Compared with the original adapter in PR #680, decoding now uses
`TrellisOrdering::Deadline` instead of natural mechanism order. With bounded
pruning (default `k = 64`), this can change predictions for the same DEM and
syndrome. Revalidate archived predictions before upgrading a deployed module.
`init` now accepts models wider than 128 detectors for streaming/replay;
`frontier_decode` reports status 2 when called on such a model. The 128-observable
limit still applies at initialization.

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
