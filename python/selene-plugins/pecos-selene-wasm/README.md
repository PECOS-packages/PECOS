# Public WASM support for Selene

This package lowers Guppy's public `tket.wasm` operations and supplies a Selene
Utility plugin backed by Wasmtime. It supports mixed signed 64-bit integer and double-precision float arguments,
with zero or one scalar result and explicit module bytes. Branches and loops run
with the released public compiler; no upstream source changes are required.

```python
from pecos_selene_wasm import build

runner = build(program, {module_identifier: wasm_bytes})
```

Pass a Guppy function directly so the plugin selects its safe frontend. For
precompiled input, use `compile_guppy(program)` from this package and pass the
returned package or its serialized bytes directly to lowering, without further
optimization. Ordinary external calls in loops can be miscompiled by default
Guppy optimization; unprepared mixed-extern loop packages are rejected.

Use the runner's normal Selene `run_shots` API. No PECOS simulator or Python
bindings are required for standalone Selene execution. PECOS `sim()` uses the
same lowering and runtime through `Guppy(program, wasm_modules=modules)`.

From the PECOS checkout, install this package with
`uv pip install -e 'python/selene-plugins/pecos-selene-wasm[test]'`. Building from
source requires Rust and compiles the native utility. See
[the Guppy WASM guide](https://github.com/PECOS-packages/PECOS/blob/dev/docs/user-guide/guppy-wasm.md)
for examples, supported operations, and limitations.

Each source build asks Cargo to check the native library and its dependencies
for changes, then packages the library reported by Cargo. Existing staged
libraries do not bypass this check. Cargo's incremental cache is reused, and
custom target directories configured through `CARGO_TARGET_DIR` or Cargo's
`build.target-dir` are supported.
