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

The lowering has been tested with public compiler versions 0.4.2, 0.4.3, and 0.5.2, with HUGR
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

Precompiled HUGR works with
`pc.Hugr(addition.compile().to_bytes(), wasm_modules=modules)`, including the
`Hugr.from_bytes` and `Hugr.from_file` constructors. Supply modules when creating
the wrapper, before `sim()` lowers the program.

## Standalone Selene

<!--continuation-->
```python
from pecos_selene_wasm import build
from selene_sim import Quest

runner = build(addition.compile(), modules)
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
fresh instances; workers do not share state. A module's optional `init` export
must have signature `() -> ()` and runs once when that instance is first used.

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

- Named exports with mixed scalar `i64`/`f64` arguments and zero or one scalar result.
- `get_context`, `dispose_context`, `lookup_by_name`, `call`, and `read_result`.
- One HUGR module per program, with any explicitly supplied WASM modules.
- Synchronous WASM calls on 64-bit hosts, including branches and loops.

`i32`, `f32`, aggregate and multiple results, lookup by numeric ID,
WASI and host imports are not supported by this adapter. Unsupported
signatures and operations fail explicitly. Traps and runtime signature errors
propagate as simulation errors. Calls use PECOS Wasmtime's default one-second
execution timeout; memory limits are currently the Wasmtime wrapper's default.

Loops execute through the PECOS-owned lowering with released public compiler
packages. No upstream patch or custom compiler build is required. Examples in
`stateful.py` cover counted loops, zero iterations, nested loops, result-dependent
termination, and calls through helpers. `floating_point.py` also carries float
results through a loop and uses the final value for quantum feedback.

The lowering protects external function values during HUGR optimization using
public barrier and indirect-call operations. This avoids a released-compiler
constant-analysis bug that otherwise replaces loop results with initial values.
The same protection is used by PECOS and standalone Selene.

The implementation follows the public
[tket.wasm specification](https://github.com/Quantinuum/tket2/blob/main/tket/tket-qsystem/src/extension/classical_compute/wasm.rs)
and Selene's public
[Utility interface](https://github.com/Quantinuum/selene/blob/main/selene-core/python/selene_core/utility.py).
