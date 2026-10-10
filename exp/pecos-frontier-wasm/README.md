# PECOS Frontier bare-WebAssembly adapter

This crate compiles the PECOS Frontier decoder into a WebAssembly module with
no imports. The original API uses `i32` parameters and results. Helios callers
must instead use the explicit `i64` streaming wrappers below: the H2-style
32-bit boundary is not compatible with the Helios call interface.

### Helios streaming boundary

Use `WasmPlatform.Helios` in Guppy and `WasmFileHandler(..., int_size=64)`
when uploading. Call `frontier_stream_push_i64` and
`frontier_stream_finish_round_i64` with five `i64` arguments and an `i64`
result; query `frontier_status_i64`. Read correction words with
`frontier_result_0_i64..3_i64`; observables 32 and above are only available
there. The no-argument, no-result `init`,
`frontier_stream_begin`, and `frontier_reset` exports are shared.

These wrappers preserve the decoder's existing 32-bit packed-word layout.
Both sign-extended `i32` and zero-extended `u32` words are accepted; values
outside those ranges return -1, set status 3, and clear correction words.
`frontier_stream_finish_round_i64` returns successful corrections zero-extended
to `0..=u32::MAX`, so every negative result is unambiguously an error. The
original `i32` exports remain available for existing callers. Use only the
`_i64` integer-valued exports from a Helios HUGR.

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

The native example lives in the adapter crate alongside the FWR1 reader:

```console
cargo run --release -p pecos-frontier-wasm --example replay_fwr -- \
  /path/to/model.dem /path/to/hardware_shots.fwr
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
- `frontier_stream_push_i64`, `frontier_stream_finish_round_i64`,
  `frontier_status_i64`, `frontier_result_0_i64..3_i64`: Helios `i64`
  counterparts described above; correction words are zero-extended.
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

Replay operations return -1 and set status 4 on invalid inputs or replay
failure, clearing correction words and any prepared shot. Successful replay
operations set status 0. `frontier_reset` also discards prepared replay work;
calling `frontier_replay_stream_finish` after reset requires a new preparation.

Call `frontier_reset` at the end of each shot when the host persists module
state between shots. Quantinuum requires this reset for in-memory Wasm state.

## Hardware latency

The streaming input path uses a fixed-size stack buffer for each detector
block. The shared trellis branch emitter checks closing-detector compatibility
before copying a rejected state, and skips that check when no detectors close.
These changes preserve branch arrival order and probability arithmetic; they
do not reduce the beam or introduce approximate pruning. Validate every
correction against a frozen reference for the target dataset before deploying
a rebuilt module. Faster mean latency alone does not establish a hardware
deadline: report push and final-call tails separately, excluding initialization.

The binary floating-point kernel also precomputes per-column closing checks
and stores only the contiguous span of live detector words in each state.
Omitted words are exactly zero, including when a detector word becomes live
again. Detector-word order, branch arrival order, floating-point accumulation,
and pruning tie-breaks are preserved. This changes the state representation,
not the noise model, beam size, or decoding approximation. The integer and
general N-ary kernels retain their full-width representation.

Before running a production model on hardware, tune the decoder configuration
and re-measure its latency on the target system. In a 2,000-shot Wasmtime JIT
benchmark on a fast desktop, a rotated-memory-Z model at physical error rate
0.001 took 4.1 ms median / 8.9 ms p99 / 42 ms maximum at distance 3 with three
rounds, and 57.9 ms median / 124.8 ms p99 / 158 ms maximum at distance 5 with
five rounds. The latter is close enough to Quantinuum's approximately 250 ms
per-call hardware limit that desktop measurements should not be treated as a
hardware safety margin.

For a reproducible native throughput comparison on a checked-in public model,
run the same command at the base and candidate revisions:

```console
cargo run --release -p pecos-frontier --example public_dem_benchmark -- \
  examples/surface_code_circuits/surface_code_d7_z_stim.dem 256 5 24301
```

The output includes the model dimensions, seed, failures, and prediction
checksum alongside mean decode time. Treat it as a throughput probe rather
than a hardware-latency measurement.
