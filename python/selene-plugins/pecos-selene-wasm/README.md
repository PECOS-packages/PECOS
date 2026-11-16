# Public WASM support for Selene

This package lowers Guppy's public `tket.wasm` operations and supplies a Selene
Utility plugin backed by Wasmtime. It supports signed 64-bit arguments and zero
or one signed 64-bit result, with explicit module bytes.

```python
from pecos_selene_wasm import build
runner = build(program.compile(), {module_identifier: wasm_bytes})
```

Use the runner's normal Selene `run_shots` API. No PECOS simulator or Python
bindings are required for standalone Selene execution. PECOS `sim()` uses the
same lowering and runtime through `Guppy(program, wasm_modules=modules)`.

From the PECOS checkout, install this package with
`uv pip install -e 'python/selene-plugins/pecos-selene-wasm[test]'`. Building from
source requires Rust and compiles the native utility. See
[the Guppy WASM guide](https://github.com/PECOS-packages/PECOS/blob/dev/docs/user-guide/guppy-wasm.md)
for examples, supported operations, and limitations.
