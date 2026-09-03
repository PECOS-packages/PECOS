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
word `i / 32`, bit `i % 32`.

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

Call `frontier_reset` at the end of each shot when the host persists module
state between shots. Quantinuum requires this reset for in-memory Wasm state.
