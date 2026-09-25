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

The output is `dist/pecos_frontier_wasm.wasm`. The adapter supports at most
128 detectors and 128 logical observables. Detector and observable bit `i` is
word `i / 32`, bit `i % 32`. The build validates the model with PECOS's Stim
DEM parser and rejects malformed, unflattened, or oversized models.

The underlying command is also cross-platform:

```console
uv run --frozen python scripts/build_frontier_wasm.py [flattened-model.dem]
```

## WebAssembly ABI

- `init() -> ()`: constructs the embedded decoder.
- `frontier_decode(i32, i32, i32, i32) -> ()`: asynchronous-friendly decode.
- `frontier_result_0..3() -> i32`: four observable-mask words.
- `frontier_status() -> i32`: 0 success, 1 model error, 2 model too wide,
  3 decode error.
- `frontier_reset() -> ()`: clears per-shot output.

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
