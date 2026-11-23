# Guppy WASM calls

Guppy programs can call WebAssembly through PECOS `sim()` or standalone public
Selene using `pecos-selene-wasm`. The package supplies a lowering pass for the
public `tket.wasm` extension and a Wasmtime-backed Selene Utility plugin. PECOS
uses the same lowering and WASM runtime through its QIS FFI library.

## Setup from a checkout

Build PECOS as described in the [development guide](../development/DEVELOPMENT.md), then
install the utility from the same checkout:

```bash
uv pip install -e 'python/selene-plugins/pecos-selene-wasm[test]'
```

The install builds the native utility with Cargo. Standalone Selene needs only
this utility, Guppy, and `selene-sim`; it does not require PECOS Python bindings.
For PECOS, rebuild both the bindings and `pecos-qis-ffi` from this revision so
the running QIS library exports the WASM ABI. An older installed runtime cannot
execute these calls. The `quantum-pecos[wasm]` extra selects the utility package.

The lowering requires public compiler version 0.4.3 or newer (below 0.6), and
has been tested with 0.4.3 and 0.5.2, with HUGR
0.18 and `tket.wasm` 0.5.0. Guppy currently exposes the WASM decorators through
`guppylang_internals`; these imports may change in a future Guppy release.

## Addition

Save this as `arithmetic.wat` next to your Python script:

```wat
(module
  (func (export "add") (param i64 i64) (result i64)
    local.get 0
    local.get 1
    i64.add)
)
```

Declare the export in Guppy. The default Helios WASM platform represents Guppy
`int` as a signed 64-bit WASM integer:

```hidden-python
import re
from pathlib import Path

# Stage the WAT shown above beside the temporary script used by the doc tests.
_guide = Path("docs/user-guide/guppy-wasm.md").read_text()
_wat = re.search(r"`{3}wat\n(.*?)`{3}", _guide, re.DOTALL).group(1)
Path(__file__).with_name("arithmetic.wat").write_text(_wat)
```

```python
from pathlib import Path
from guppylang import guppy
from guppylang.std.builtins import nat, result
from guppylang_internals.decorator import wasm, wasm_module

WASM_PATH = Path(__file__).with_name("arithmetic.wat")


@wasm_module(str(WASM_PATH))
class Arithmetic:
    @wasm
    def add(self, a: int, b: int) -> int: ...


@guppy
def addition() -> int:
    arithmetic = Arithmetic(nat(0))
    total = arithmetic.add(2, 3)
    arithmetic.discard()
    result("sum", total)
    return total


modules = {str(WASM_PATH): WASM_PATH.read_bytes()}
```

Guppy reads the declaration's module to validate exports while compiling. At
execution, **the supplied bytes determine the module**. Each mapping key must
match the exact module identifier in the HUGR, including a path if Guppy used
one. The lowering does not read files named by a HUGR program. Both WAT bytes
and binary WASM bytes are accepted.

## PECOS `sim()`

<!--continuation-->
```python
import pecos as pc

data = pc.sim(pc.Guppy(addition, wasm_modules=modules)).qubits(1).quantum(pc.state_vector()).run(4).to_dict()
assert data["sum"] == [5, 5, 5, 5]
```

For precompiled HUGR, use `from pecos_selene_wasm import compile_guppy` and
`pc.Hugr(compile_guppy(addition).to_bytes(), wasm_modules=modules)`, including the
`Hugr.from_bytes` and `Hugr.from_file` constructors. Supply modules when creating
the wrapper, before `sim()` lowers the program.

## Standalone Selene

<!--continuation-->
```python
from pecos_selene_wasm import build
from selene_sim import Quest

runner = build(addition, modules)
shots = [
    dict(shot)
    for shot in runner.run_shots(
        simulator=Quest(),
        n_qubits=1,
        n_shots=4,
        n_processes=1,
    )
]
assert shots == [{"sum": 5}] * 4
```

Consume each shot's streamed records before advancing to the next shot.
For lower-level integration, `compile_wasm(program, modules)` returns LLVM;
pass that LLVM to `selene_sim.build(..., utilities=[WasmPlugin()])`.
The utility supplies classical WASM calls and works alongside Selene's normal
quantum runtime and simulator plugins.

## Repeated quantum correction

The independently written `examples/guppy_wasm/repetition.py` demonstrates a
three-bit repetition code with a WASM syndrome lookup. Each round injects one
bit flip, measures `q0 XOR q1` and `q1 XOR q2` using two ancillas, calls the WASM
decoder, and applies the returned correction to the surviving data qubits.
It covers all three error positions and a final round without an error.

```bash
python examples/guppy_wasm/repetition.py --backend selene --shots 8 --workers 2
python examples/guppy_wasm/repetition.py --backend pecos --shots 8 --workers 2
```

Every shot should report `calls=4`, `correction_sum=7`, and `q0=q1=q2=0`.
The ordered syndromes `[1, 3, 2, 0]` are encoded in base 4 as
`syndrome_trace=120`; the corrections `[1, 2, 4, 0]` are encoded in base 8 as
`correction_trace=672`. These traces check every round, including its order.
The call counter also checks that WASM state starts fresh on each shot. This is
a small bit-flip correction example with perfect syndrome measurements, not a
general decoder or a fault-tolerant noise benchmark. It requires five qubits.

## Examples and state

From the repository root:

```bash
python examples/guppy_wasm/arithmetic.py --run pecos
python examples/guppy_wasm/arithmetic.py --run selene
```

Both commands run addition (`5`), chained addition/multiplication (`20`), and a
WASM-controlled X gate (measurement `1`).
`examples/guppy_wasm/stateful.py` also demonstrates repeated calls, a void export, and
independent contexts using `counter.wat`.

A context slot can be leased once at a time. Release it with `.discard()`.
Reacquiring the same slot within a shot preserves its module state. Different
slots and different modules have independent instances. Every shot starts with
fresh instances; workers do not share state. By default, a module's optional
`init` export must have signature `() -> ()` and runs once when that instance is
first used; the next section describes how to turn this off.

## Choosing initialization behavior

Pass `wasm_auto_init=False` to `pc.Guppy`, `pc.Hugr` (including its file/bytes
constructors), `pecos_selene_wasm.build`, or `compile_wasm` when your program
controls initialization explicitly. `compile_hugr_to_qis` accepts the same option.
The default is `True`, preserving automatic execution of an optional `init` export.
The option must be a boolean and applies to every supplied module. PECOS requires
an explicit `wasm_modules` mapping when opting out.

With automatic initialization disabled, the runtime neither validates nor calls
an export named `init` merely because the instance was created. Guppy can still
declare and call that export normally. A WebAssembly **start function always
runs** during instantiation, under either policy. Releasing and reacquiring a
context preserves its instance without repeating startup or initialization.

The memory example uses a start function to store `11`, and an optional `init`
export to add `31`. It also checks memory growth, zero-filled new pages, signed
loads/stores, indirect calls, and rejection of growth beyond the module maximum:

```bash
python examples/guppy_wasm/memory_lifecycle.py --backend selene
python examples/guppy_wasm/memory_lifecycle.py --backend pecos --no-auto-init
python examples/guppy_wasm/memory_lifecycle.py --backend selene --no-auto-init --explicit-init
```

The initial value is `42` by default and `11` with `--no-auto-init`. The last
command calls `init` explicitly, recording `before=11` and `after=42`.
Opting out requires the native `pecos_wasm_v3_register` symbol: rebuild or upgrade
the Selene utility or PECOS QIS FFI together with the Python package. Default
programs retain v1 registration; custom-timeout-only programs retain v2.

## Lookup by export position

Guppy's `@wasm(index)` selects an export by its zero-based position in the
module's **complete export list**. Memories, globals and tables occupy positions
as well; this is not the internal WebAssembly function index. The Python method
name can differ from the export name. For example, the checked-in
`indexed_exports.wat` exports a memory at position 0 and the `sum` function at
position 1, so its declaration uses `@wasm(1)` on a method named `add`.

```bash
python examples/guppy_wasm/indexed_exports.py --backend selene
python examples/guppy_wasm/indexed_exports.py --backend pecos
```

These examples cover indexed integer arithmetic, mixed float/integer arguments,
quantum feedback, stateful loops, void returns, and named calls in the same
program. They produce sum `4`, product `20`, a named-call result `22`, float value
`1.5`, measurement `1`, and counter results `3` then `1` after reset.

The explicit module bytes supplied at execution determine export positions and
signatures. The runtime resolves the position on a call and rejects missing
positions, non-function exports and signature mismatches. Reordering exports
therefore changes indexed calls even if their names remain unchanged. Prefer
named calls when export order is not part of your module's interface.

Indices must fit an unsigned 64-bit integer. Rebuild the native utility and PECOS
QIS FFI for this feature: indexed lookup adds `pecos_wasm_v2_lookup_by_id`; the
existing named lookup ABI remains unchanged.

## Floating-point calls

Guppy `float` maps to WASM `f64` on the Helios WASM platform. Integer and float
arguments can be mixed in one signature, and the result can be either type.
For example, `affine(self, value: float, scale: int, offset: float) -> float`
matches a WAT export with `(param f64 i64 f64) (result f64)`.

Run the mixed arithmetic, state, and quantum-feedback examples:

```bash
python examples/guppy_wasm/floating_point.py --backend selene
python examples/guppy_wasm/floating_point.py --backend pecos
```

The examples produce `-4.25` from mixed arithmetic, `-4` from an explicit WASM
float-to-integer conversion, and measurement `1` from float-controlled feedback.
Float state is retained across context leases and reset between shots. The loop
example finishes at `3.75` and produces measurement `1`.

Float arguments and results are transported by their IEEE 754 bits. The adapter
does not convert them to integers or clamp them. Runtime signature checks reject
a supplied export with different argument or result types, even when arity matches.
Actual arithmetic follows WebAssembly semantics, including traps from invalid
float-to-integer conversions. Rebuild the native utility and PECOS QIS FFI when
upgrading: float signatures need the new typed lookup export. Existing integer
programs continue to use the original lookup ABI.

## Measure compilation and execution

```bash
python examples/guppy_wasm/benchmark.py --backend selene --calls 0 1 10 100
python examples/guppy_wasm/benchmark.py --backend both --shots 100 --repeats 5 > wasm-timings.json
```

The benchmark writes JSON containing package versions, platform, Guppy compilation
time, backend preparation time, first-run time, and repeated-run samples. Every
run checks the expected result. Zero calls provide a no-WASM baseline; other
cases perform a dependent chain of additions. The generated Guppy is straight-line
code to measure a fixed number of calls without adding loop-control overhead.

Compare repeated measurements on the same machine and package versions. Times
include shot setup, module compilation/instantiation, runtime communication and
result collection. “Amortized seconds per call” divides the complete run time by
the number of calls; it is not an isolated measurement of the WASM call boundary.
The first run is reported separately because backend initialization and caches
can make it substantially slower. These measurements do not establish parity
with other compiler/runtime implementations.

## Supported scope

- Exports selected by name or export-list position, with mixed scalar `i64`/`f64` arguments and zero or one scalar result; explicit 32-bit mode supports `i32` integer exports.
- `get_context`, `dispose_context`, `lookup_by_name`, `lookup_by_id`, `call`, and `read_result`.
- One HUGR module per program, with any explicitly supplied WASM modules.
- Synchronous WASM calls on 64-bit hosts, including branches and loops.

`f32`, aggregate and multiple results,
WASI and host imports are not supported by this adapter. Unsupported
signatures and operations fail explicitly. Traps and runtime signature errors
propagate as simulation errors. Calls default to a one-second execution timeout;
memory limits are currently the Wasmtime wrapper's default.

For longer classical calculations, pass `wasm_timeout_seconds=10.0` to
`pecos_selene_wasm.build(...)` or to `pc.Guppy(..., wasm_modules=modules)`.
The same option is available on `compile_wasm`, `Hugr`, `Hugr.from_bytes`, and
`Hugr.from_file`. Nondefault limits require explicit WASM module mappings.

The timeout must be a positive finite number of seconds. It applies separately
to the module start function, optional `init`, and each exported call, not to
compilation or the complete shot. The epoch clock has approximately 10 ms
resolution; positive fractional ticks round up, and very large values saturate
with counter headroom. Selene's overall run timeout remains a separate limit.

Default programs retain the existing registration ABI. Custom limits require
the updated native runtime exporting `pecos_wasm_v2_register`: rebuild or upgrade
both the Python lowering package and the native Selene utility or PECOS QIS FFI.
Standalone `build` checks this capability before launching execution.

For a public reproduction, run
`python examples/guppy_wasm/long_call.py --backend selene --timeout 10` or select
`--backend pecos`. It checks a bounded two-billion-step integer recurrence
against an independent logarithmic-time oracle. Runtime depends on the machine;
`--timeout 1` can reproduce the default limit. This deliberately slow example
is excluded from the normal comparison suite and from CI timing assertions.

Loops execute through the PECOS-owned lowering with released public compiler
packages. No upstream patch or custom compiler build is required. Examples in
`stateful.py` cover counted loops, zero iterations, nested loops, result-dependent
termination, and calls through helpers. `floating_point.py` also carries float
results through a loop and uses the final value for quantum feedback.

The lowering protects external function values during HUGR optimization using
public barrier and indirect-call operations. This avoids a released-compiler
constant-analysis bug that otherwise replaces loop results with initial values.
The same protection is used by PECOS and standalone Selene.

Pass Guppy source directly to `Guppy(function, wasm_modules=modules)` or
`pecos_selene_wasm.build(function, modules)`. Both use Guppy's minimal-optimization
frontend before applying the lowering. Normal Guppy compilation can already
miscompile loops containing ordinary `@guppy.declare` external calls; LLVM
optimization settings cannot repair those results.

To store or transport a program, use `pecos_selene_wasm.compile_guppy(function)`
and serialize the returned package with `.to_bytes()`. Pass that package or its
bytes directly to WASM lowering without running additional optimizers. Its
metadata records the safe frontend choice; it does not certify later transforms.
Unprepared packages with visible external declarations and loops are rejected
with instructions to recompile from source. Already optimized arbitrary HUGR
cannot be repaired, and a compiler may have erased evidence of the original
calls. Use the safe frontend whenever preparing Guppy for WASM execution.

The implementation follows the public
[tket.wasm specification](https://github.com/Quantinuum/tket2/blob/main/tket/tket-qsystem/src/extension/classical_compute/wasm.rs)
and Selene's public
[Utility interface](https://github.com/Quantinuum/selene/blob/main/selene-core/python/selene_core/utility.py).

## 32-bit WASM integer exports

Run `python examples/guppy_wasm/integer32.py --backend selene` or use
`--backend pecos` for the PECOS stack.

Guppy's `WasmPlatform.H2` declaration mode accepts `i32` exports. Select
`wasm_int_width=32` when executing these declarations through either public
backend. This selects the WASM integer width; it does not change the quantum
platform or the 64-bit host requirement.

```python
from pathlib import Path
from guppylang import guppy
from guppylang.std.builtins import nat, result
from guppylang_internals.decorator import wasm, wasm_module
from guppylang_internals.std._internal.wasm import WasmPlatform
from pecos import Guppy, sim, state_vector
from pecos_selene_wasm import build
from selene_sim import Quest

# Run from the PECOS checkout root, using the example's public WASM module.
WASM_PATH = Path("examples/guppy_wasm/integer32.wat").resolve()


@wasm_module(str(WASM_PATH), wasm_platform=WasmPlatform.H2)
class Arithmetic32:
    @wasm
    def add(self, left: int, right: int) -> int: ...


@guppy
def addition32() -> None:
    arithmetic = Arithmetic32(nat(0))
    value = arithmetic.add(2, 3)
    arithmetic.discard()
    result("sum", value)


modules = {str(WASM_PATH): WASM_PATH.read_bytes()}
runner = build(addition32, modules, wasm_int_width=32)
shots = [dict(shot) for shot in runner.run_shots(simulator=Quest(), n_qubits=1, n_shots=4, n_processes=1)]
assert shots == [{"sum": 5}] * 4
results = sim(Guppy(addition32, wasm_modules=modules, wasm_int_width=32)).qubits(1).quantum(state_vector()).run(4)
assert results.to_dict()["sum"] == [5] * 4
```

The [32-bit example](https://github.com/PECOS-packages/PECOS/blob/dev/examples/guppy_wasm/integer32.py)
covers named/indexed calls, signed overflow, state and quantum feedback. Declare
its module with `@wasm_module(path, wasm_platform=WasmPlatform.H2)`, importing
`WasmPlatform` from `guppylang_internals.std._internal.wasm`.

Arguments are truncated to their low 32 bits, and results are sign-extended to
Guppy's integer type. Thus `2147483647 + 1` returns `-2147483648`, and passing
`4294967301` to an identity function returns `5`. This differs from the legacy
foreign-object API's clamping behavior.

The width applies to every integer WASM signature in the program, including
precompiled programs supplied through `Hugr(..., wasm_int_width=32)` or
`compile_wasm(..., wasm_int_width=32)`. Mixed 32/64-bit integer modules in one
program are not supported by this option. The default remains 64: supplying an
`i32` export to a default-width program, or an `i64` export in 32-bit mode, fails
signature validation. Float signatures retain their existing `f64` behavior.

## Compare against another Guppy/WASM stack

The [public comparison bundle](https://github.com/PECOS-packages/PECOS/tree/dev/examples/guppy_wasm/parity)
provides 22 workloads with independent expected answers, public PECOS and Selene
runners, and an adapter contract for a separately installed reference stack.
It checks scalar calls, state, loops, quantum feedback and traps, and separates
preparation, first-run and repeat timings. The handoff archive contains only
allowlisted public sources; reference adapters and environments stay local.
Passing this suite demonstrates conformance on its workloads, not full stack
or performance parity.
