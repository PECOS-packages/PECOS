# Public WASM support for Selene

This package lowers Guppy's public `tket.wasm` operations and supplies a Selene
Utility plugin backed by Wasmtime. It supports signed integer and double-precision float arguments,
with zero or one scalar result and explicit module bytes. Exports can be selected
by name or zero-based position in the complete export list (`@wasm(index)`). Branches and loops run
with the released public compiler; no upstream source changes are required.
Integer signatures default to strict i64. Select `wasm_int_width=32` for strict
i32, or `wasm_int_width="auto"` to resolve each integer lane from the actual
export signature, including programs that call both i32 and i64 exports.

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

## Repeated-shot execution

The shared public runtime retains up to eight compiled WASM modules per worker
thread, with an aggregate input-byte budget of 32 MiB. Cache identity uses the
complete module bytes, so reusing a module identifier with different bytes cannot
reuse the wrong code. Oversized modules run without being retained in the cache.

Only compiled code is reused. Each context in each shot receives a fresh store
and instance, including fresh globals and linear memory, execution of the start
function, and (by default) the optional `init` export. Trap recovery clears
mutable state without discarding reusable compiled code. Resource limits and
execution deadlines remain independent for each instance.

These are cache retention bounds, not a process-wide memory cap: compiled code
and engine metadata have additional costs, and live instances can retain code
after cache eviction. Worker threads maintain separate caches.

A standalone public benchmark separates preparation from call execution:

```bash
cargo run --release -p pecos-wasm --example compilation_reuse
```

It compares 20 fresh instances of a generated 1,000-function module with and
without compilation reuse, including the first compilation in both totals.
Every shot checks fresh global state. This is a compilation microbenchmark,
not an end-to-end Guppy throughput or full-stack parity claim.

## Initialization policy

`build(..., wasm_auto_init=False)` and `compile_wasm(..., wasm_auto_init=False)`
skip automatic validation and invocation of the optional exported `init` function.
The default remains `True`. WASM start functions always run, and explicit Guppy
calls to `init` work with either policy. The setting applies independently to each
instance; compiled code can still be shared across initialization policies.
Opting out requires the updated native `pecos_wasm_v3_register` ABI. See
`docs/user-guide/guppy-wasm.md` and `examples/guppy_wasm/memory_lifecycle.py`.


## Managed objects inside WASM

The native runtime enables Wasmtime GC with its copying collector. Modules may
use managed structs and arrays internally while exporting the supported scalar
signatures. Managed state belongs to each store, independently of cached code.
See `examples/guppy_wasm/managed_objects.py` for a scalar interface and explicit
GC payload that also work with older Guppy frontends. Linear-memory limits do
not cover the managed heap.


Use `wasm_state_scope="worker"` to retain WASM state across successful shots on
one execution worker. The default `"shot"` creates fresh instances. Each worker
has independent state, so results can depend on worker count and scheduling.
New `run_shots` batches start fresh. Custom native callers control their worker
lifetime and can explicitly reset it. See the user guide for lifecycle details.
