# PECOS Frontier for Quantinuum WAVM

This crate compiles the PECOS Frontier decoder into a Quantinuum-compatible
WebAssembly module. Its exported functions use only `i32` parameters and at
most one `i32` result.

## Build

The default build embeds `model.dem`, a tiny smoke-test model. On Windows:

```powershell
.\scripts\build-frontier-wasm.ps1
```

On Linux or macOS:

```sh
./scripts/build-frontier-wasm.sh
```

To embed a real, flattened Stim detector error model, pass its path to the
platform's script:

```powershell
.\scripts\build-frontier-wasm.ps1 -DemPath C:\path\to\model.dem
```

```sh
./scripts/build-frontier-wasm.sh /path/to/model.dem
```

The output is `dist/pecos_frontier_wasm.wasm`. The adapter supports at most
128 detectors and 128 logical observables. Detector and observable bit `i` is
word `i / 32`, bit `i % 32`.

## Quantinuum ABI

- `init() -> ()`: required WAVM load hook; constructs the embedded decoder.
- `frontier_decode(i32, i32, i32, i32) -> ()`: asynchronous-friendly decode.
- `frontier_result_0..3() -> i32`: four observable-mask words.
- `frontier_status() -> i32`: 0 success, 1 model error, 2 model too wide,
  3 decode error.
- `frontier_reset() -> ()`: clears per-shot output.

Call `frontier_reset` at the end of each shot, per Quantinuum's guidance for
in-memory Wasm state.
