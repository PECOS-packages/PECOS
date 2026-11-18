# WASM calls in Guppy

Guppy can declare functions exported by a WebAssembly module and compile calls to
them into HUGR. The small examples in
[`examples/guppy_wasm/arithmetic.py`](https://github.com/PECOS-packages/PECOS/blob/dev/examples/guppy_wasm/arithmetic.py)
cover addition, chained addition/multiplication, and a WASM result controlling an
`X` gate. The examples use public dependencies and provide execution probes for
PECOS and standalone Selene.

## Run the examples

From a PECOS checkout with its Python development environment installed:

```bash
uv run --no-sync python examples/guppy_wasm/arithmetic.py
```

This compiles all three Guppy programs through `pecos.Guppy` and then executes
the arithmetic module directly through `pecos.WasmForeignObject`. It prints
`2 + 3 = 5` and `(2 + 3) * 4 = 20`. These are separate checks: the default command
does not execute the Guppy programs.

No C/Rust compiler, external decoder, or downloaded WASM binary is needed. The
example uses the readable
[`arithmetic.wat`](https://github.com/PECOS-packages/PECOS/blob/dev/examples/guppy_wasm/arithmetic.wat) source directly.
Both the Guppy decorator and PECOS's WASM loader accept WAT text in the tested
versions. Paths are resolved relative to the example file, so it can also be
launched from another working directory.

## Declare the module

The module exports signed 64-bit integer functions:

```wat
(module
  (func (export "init"))
  (func (export "add") (param i64 i64) (result i64)
    local.get 0
    local.get 1
    i64.add)
  (func (export "multiply") (param i64 i64) (result i64)
    local.get 0
    local.get 1
    i64.mul)
)
```

The empty `init` export satisfies PECOS's `WasmForeignObject` initialization
contract. This module has no mutable state and needs no shot reset.

The Guppy declarations must match the exported names and signatures. Run this
code from the repository root:

```python
from pathlib import Path

from guppylang import guppy
from guppylang.std.builtins import nat, result
from guppylang_internals.decorator import wasm, wasm_module

wasm_path = Path("examples/guppy_wasm/arithmetic.wat").resolve()


@wasm_module(str(wasm_path))
class Arithmetic:
    @wasm
    def add(self, a: int, b: int) -> int: ...

    @wasm
    def multiply(self, a: int, b: int) -> int: ...
```

The decorators currently come from `guppylang_internals`, as in the reference
demo. These examples use the default Helios WASM platform, where Guppy `int`
maps to WASM `i64`. The `i32` functions used in some QASM examples do not match
this declaration. Missing exports and mismatched signatures are compilation
errors.

## Compile arithmetic calls

<!--continuation-->
```python
from pecos import Guppy


@guppy
def chained_arithmetic() -> int:
    arithmetic = Arithmetic(nat(0))
    total = arithmetic.add(2, 3)
    product = arithmetic.multiply(total, 4)
    arithmetic.discard()
    result("product", product)
    return product


program = Guppy(chained_arithmetic)
assert program.hugr_bytes
```

`Arithmetic(nat(0))` acquires WASM context slot 0. Calls borrow that context;
`discard()` releases it. The context is a linear Guppy value, so release it even
when the WASM functions are stateless. `result(...)` gives the output a stable
name for result collection.

HUGR contains the WASM module filename and the `tket.wasm` operations for context
acquisition, export lookup, calls, result reads, and disposal. Compilation alone
does not prove that a simulation backend can lower or execute those operations.
Keep the referenced module available if you save or move the compiled HUGR.

## Use a WASM result in a quantum program

<!--continuation-->
```python
from guppylang.std.quantum import measure, qubit, x


@guppy
def conditional_x() -> bool:
    arithmetic = Arithmetic(nat(0))
    total = arithmetic.add(2, 3)
    arithmetic.discard()
    q = qubit()
    if total == 5:
        x(q)
    measured = measure(q).read()
    result("measured", measured)
    return measured


program = Guppy(conditional_x)
assert program.hugr_bytes
```

On a backend supporting these WASM operations, this should produce
`measured = True`: the WASM addition selects the `X` gate before measurement.

## Execute the WASM module directly

This works independently of Guppy execution support:

```python
from pathlib import Path
from pecos import WasmForeignObject

arithmetic = WasmForeignObject.from_file(Path("examples/guppy_wasm/arithmetic.wat"))
arithmetic.init()
try:
    total = arithmetic.exec("add", [2, 3])
    product = arithmetic.exec("multiply", [total, 4])
    assert total == 5
    assert product == 20
finally:
    arithmetic.teardown()
```

See [WASM Foreign Objects](wasm-foreign-objects.md) for lifecycle management,
timeouts, and QASM integration.

## Probe Guppy execution with PECOS or Selene

These commands probe the existing PECOS and default standalone Selene paths:

```bash
uv run --no-sync python examples/guppy_wasm/arithmetic.py --run pecos
uv run --no-sync python examples/guppy_wasm/arithmetic.py --run selene
```

The PECOS path calls `sim(Guppy(function)).qubits(1)` with a state-vector
simulator. The default standalone path calls `selene_sim.build(function.compile())`
using its public HUGR/QIS compiler. Both propagate build and runtime errors.

These default-path compiler checks were run on macOS arm64 on 2026-11-13:

| Guppy | Selene | HUGR/QIS compiler | Guppy to HUGR | Default Selene build |
|---|---|---|---|---|
| 1.0.1 | 0.3.0 | 0.4.2 | Pass | `Unknown type: module` |
| 1.2.0 | 0.3.7 | 0.5.1 | Pass | `Unknown type: module` |
| 1.2.1 | 0.3.8 | 0.5.2 | Pass | `Unknown type: module` |
| 0.21.15 | 0.2.15 | 0.2.10 | Addition passes | `Unknown type: module` |

The second combination checks the dependency versions proposed in
[PECOS #1627](https://github.com/PECOS-packages/PECOS/pull/1627), not that PR's
rebuilt native stack. The latest-version check also tested a binary `.wasm`
input as well as WAT; both reached the same LLVM emission error.

These failures are specific to the tested public compiler route. They do **not**
mean that Selene cannot execute WASM locally. A separate local experiment with
additional compiler/runtime components passed addition, chained calls, and
WASM-controlled quantum execution. Those components are not dependencies of
these public examples, and those results do not establish support through
`sim(Guppy(...))`. PECOS's `.foreign_object(...)` QASM API does not configure this
Guppy execution path.

### Guidance for public PECOS integration

The intended public implementation has two responsibilities:

1. **Compilation:** lower the public `tket.wasm` operations and types to calls
   supported by a public runtime. This can be implemented in the public compiler
   or a PECOS compilation pass, independently of proprietary compiler code.
2. **Execution:** attach the selected WASM module and dispatch those calls through
   PECOS's existing Wasmtime-backed `WasmForeignObject`. Define context ownership,
   argument/result conversion, trap propagation, and per-shot/worker state.

`Guppy._to_program()` currently invokes `compile_hugr_to_qis` before the Rust
simulation builder is returned. Module/compiler configuration must therefore be
available at that stage, or lowering must be deferred until the simulation
configuration is complete. A builder option added after eager compilation cannot
fix the current module-type error.

For standalone Selene, the same public runtime could be exposed through a Selene
utility plugin. That plugin must be paired with compatible public lowering;
changing the quantum runtime plugin or runtime-library lookup alone is insufficient.
A shared WASM implementation would let PECOS's `sim()` path and a Selene utility
adapter share execution semantics without requiring proprietary packages.

Selene's public `Utility` interface links libraries exposing functions callable by
the compiled program. Its `Runtime` interface schedules quantum operations. The
utility interface is therefore the proposed integration point for classical WASM
calls; a quantum-runtime change should only be added if a demonstrated timing or
lifecycle requirement needs it. Lowering should be available to standalone Selene
as well as PECOS, ideally through the public HUGR/QIS compiler.

Start with scalar integer arithmetic, which the existing PECOS foreign-object
interface supports. Other WASM signatures need explicit support or a clear error;
passing these examples does not establish complete `tket.wasm` coverage.

The arithmetic examples are independent acceptance tests for this work. First
verify addition and chained calls; then verify a WASM return value controlling a
quantum gate. The current probes retain expected failures until the public paths
can execute them. Proprietary compiler/runtime implementations should not be
copied into this public implementation.

### Hosted Selene through Nexus

The reference demo's saved outputs record successful Guppy+WASM jobs on
`Helios-1E`. Quantinuum documents its Helios emulators as
[distributions of Selene configured with the Helios runtime and error model](https://docs.quantinuum.com/systems/trainings/helios/getting_started/quantinuum_emulator.html).
The demo compiles HUGR locally, uploads HUGR and the WASM binary separately, and
attaches the uploaded module with `qnx.start_execute_job(..., wasm_module=wasm_ref)`.
Those are recorded results, not newly submitted cloud jobs. The hosted submission
API and local execution are different routes.

## Tests

```bash
uv run --no-sync pytest python/quantum-pecos/tests/guppy/test_guppy_wasm.py -q -rx
```

The tests execute arithmetic exports (including zero, negative, and 64-bit inputs),
round-trip HUGR and check its WASM operations, reject bad declarations, and run the
example outside the repository's working directory. Non-WASM control programs
verify the default runners and result parsing.

The PECOS and default-Selene execution probes mark only the observed
`Unknown type: module` LLVM-emission error as `xfail`; other failures remain
failures. They will check the expected results as public support lands.
