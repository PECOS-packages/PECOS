# Rust PhirClassicalInterpreter -- Drop-in Replacement Spec

## Goal

Build a Rust `PhirClassicalInterpreter` that is a drop-in replacement for the Python
`PhirClassicalInterpreter` in `HybridEngine`. The Python PHIR code is the spec -- the
Rust version must have identical behavior.

Eventually we move to the full Rust sim() system, but this is the first step: replace
the Python classical interpreter internals while keeping the Python `HybridEngine`
orchestration.

## Architecture

```
crates/pecos-phir-json/     -- Rust logic lives here (reuse existing internals)
python/pecos-rslib/          -- PyO3 wrapper lives here (new pyclass)
```

The Rust interpreter reuses existing components from `pecos-phir-json/src/v0_1/`:
- `ast.rs` -- PHIR JSON parsing
- `environment.rs` -- variable storage, bit ops, types
- `expression.rs` -- expression evaluation
- `operations.rs` -- classical op handling logic (adapt)

The PyO3 layer in `pecos-rslib` exposes it as a `#[pyclass]` implementing the
`ClassicalInterpreterProtocol`.

## Python Protocol

From `python/quantum-pecos/src/pecos/protocols.py`, the interpreter must satisfy:

```python
class ClassicalInterpreterProtocol(Protocol):
    program: Any
    foreign_obj: Any
    phir_validate: bool

    def reset(self) -> None: ...
    def init(self, program: str | dict | QuantumCircuit, foreign_obj: object | None = None) -> int: ...
    def shot_reinit(self) -> None: ...
    def execute(self, sequence: Sequence | None) -> Generator[list[QOp | MOp], Any, None]: ...
    def receive_results(self, qsim_results: list[dict]) -> None: ...
    def results(self, *, return_int: bool = True) -> dict: ...
```

Additional methods called by HybridEngine but not in the protocol:
```python
def add_cvar(self, cvar: str, dtype: type, size: int) -> None: ...
def result_bits(self, bits: Iterable, *, filter_private: bool = True) -> dict: ...
```

## Call Ordering in HybridEngine

```
INITIALIZATION:
  outer = PhirClassicalInterpreter()
  inner = PhirClassicalInterpreter()
  inner.phir_validate = outer.phir_validate

  num_qubits = outer.init(program, foreign_object)
  inner.init(program, foreign_object)    # same program
  machine.init(num_qubits)

PER-SHOT:
  outer.shot_reinit()
  inner.shot_reinit()
  for i in range(num_qubits):
      inner.add_cvar(f"__q{i}__", pc.dtypes.i64, 1)    # private qubit vars

  EXECUTION LOOP:
    for buffered_ops in outer.execute(outer.program.ops):
        noisy_ops = op_processor.process(buffered_ops)
        measurements.clear()
        for noisy_qops in inner.execute(noisy_ops):
            temp_meas = qsim.run(noisy_qops)
            inner.receive_results(temp_meas)
            measurements.extend(temp_meas)
        transmit_meas = inner.result_bits(measurements)
        outer.receive_results([transmit_meas])

  RESULTS:
    shot_results = outer.results(return_int=return_int)
```

## Data Flow

```
PHIR JSON str/dict
     |
     v
outer.init()  -->  parse JSON, validate, build internal AST, init env
     |                return num_qubits
     v
outer.execute(program.ops)
     |
     | yields: list[QOp | MOp]  (batches ending at measurements)
     v
op_processor.process(buffered_ops)
     |
     | returns: list[QOp]  (noisy operations)
     v
inner.execute(noisy_ops)
     |
     | yields: list[QOp]
     v
qsim.run(noisy_qops)
     |
     | returns: list[dict]   e.g. [{("m", 0): 1, ("m", 1): 0}]
     v
inner.receive_results(temp_meas)   -->  stores via assign_int
     |
inner.result_bits(measurements)    -->  extracts bits, filters __private__ vars
     |
     | returns: dict[(str, int), int]
     v
outer.receive_results([transmit_meas])  -->  stores via assign_int
     |
outer.results(return_int=...)
     |
     | returns: dict[str, int_or_bitstring]
```

## What `execute()` Yields

`execute()` is a generator/iterator. It yields `list[QOp | MOp]`.

Behavior:
- Walks the op list, recursively flattening SeqBlock and evaluating IfBlock conditions
- Buffers QOps and MOps
- Executes COps inline (never yielded)
- Yields the buffer when a measurement QOp is encountered (name in {"measure Z", "Measure", "Measure +Z"})
- Yields remaining buffer at end of program
- Classical ops (assignment, Result mapping, FFCall) are handled during the walk

### QOp Fields (what consumers read)

```
name: str                        # e.g. "H", "Measure", "RZ"
sim_name: str                    # resolved name for simulator
args: list[int] | list[tuple]    # qubit IDs
returns: list | None             # measurement targets, e.g. [["m", 0], ["m", 1]]
metadata: dict | None            # includes "angle", "angles", "var_output"
angles: tuple[float, ...] | None # rotation angles in radians
```

Fields read by QuantumSimulator: `sim_name`, `args`, `metadata`, `returns`
Fields read by GenericOpProc: only `isinstance()` checks (QOp vs MOp routing)

### MOp Fields

```
name: str
args: list | None
returns: list | None
metadata: dict | None            # may contain "duration"
```

## The Generator Problem

Python `execute()` is a coroutine -- yields batches, caller runs quantum sim, feeds
measurements back via `receive_results()`, then execution resumes. Conditional branches
may depend on those measurement results.

**Solution: PyO3 iterator class.** A `#[pyclass]` that holds interpreter state and
advances to the next yield point on each `__next__()`. The Rust struct holds program
state, and each `__next__` call processes ops until the next measurement batch.

## Dual Interpreter Pattern

HybridEngine uses TWO interpreter instances:

**Outer interpreter:**
- Drives the full program
- Has only program-declared variables
- Receives filtered measurement bits from inner via `receive_results()`

**Inner interpreter:**
- Processes noisy ops from error model
- Gets extra `__q{i}__` private vars via `add_cvar()` (one per qubit, i64, size 1)
- `result_bits()` filters out `__`-prefixed vars when transmitting back

Both use the same code, same class. The inner just handles a flat list of QOps
(no blocks to flatten) and has extra private vars.

## Method Details

### `init(program, foreign_obj) -> int`

1. Parse program: JSON string -> dict, or accept dict directly
2. Validate format ("PHIR/JSON" or "PHIR") and version (< 0.2.0)
3. Optionally validate against PHIR schema (if `phir_validate` is True)
4. Build internal AST / operation list
5. Extract variable definitions, initialize environment (all vars to 0)
6. Check foreign function calls against foreign object
7. Return num_qubits

### `shot_reinit()`

Reset all variable values to 0. Keep variable definitions.

### `execute(sequence) -> Iterator[list[QOp | MOp]]`

Walk ops, flatten blocks, execute classical ops, yield quantum op batches.
See "What execute() Yields" above.

### `receive_results(qsim_results: list[dict])`

Each dict maps `cvar` or `(cvar, idx)` to a value.
For each key/value, calls `assign_int(key, value)`.

### `result_bits(bits, filter_private=True) -> dict`

`bits` is a list of measurement dicts from qsim.run().
Iterates all (cvar, bit_idx) pairs, filters out `__`-prefixed vars,
returns `{(cvar, bit_idx): self.get_bit(cvar, bit_idx)}`.

Important: reads from own env (after receive_results), not from input.

### `results(return_int=True) -> dict`

Returns ALL variables in csym2id.
- return_int=True: values are integers
- return_int=False: values are zero-padded binary strings ("{:0{width}b}")

### `add_cvar(cvar, dtype, size)`

Dynamically add a new classical variable after init. Used by HybridEngine
for the inner interpreter's private qubit vars.

### `assign_int(cvar, val)`

Assign integer value to variable or specific bit.
- `cvar` is a string: assign whole variable
- `cvar` is (string, int): assign to bit at index

## Name Resolver

`sim_name_resolver(qop)` translates PHIR gate names to simulator names:
- `RZZ(0.0)` -> `"I"`
- `RZZ(pi/2)` -> `"SZZ"`
- `RZZ(3pi/2)` -> `"SZZdg"`
- `RZ(angle)` -> tries clifford match
- `R1XY(angles)` -> tries clifford match
- Otherwise returns `qop.name`

Applied during PHIR parsing when building QOp objects. The Rust side needs this
for yielded QOps to have correct `sim_name` values.

## Foreign Function Calls

FFCalls are COps handled during `execute()` -- never yielded. The foreign object
is a Python object implementing `ForeignObjectProtocol`:

    def exec(func_name: str, args: Sequence) -> tuple | int

The Rust side must call back into this Python object for FFCalls. This requires
holding a `Py<PyAny>` reference.

## Edge Cases

- **Empty programs**: `execute([])` yields nothing. `results()` returns `{}`
- **No measurements**: buffer yielded at end (if non-empty). No receive_results() calls.
- **Only classical ops**: all handled inline. Nothing yielded. results() has computed values.
- **"Result" cop**: maps internal register to external name, copies value, creates dest var if needed.

## What Exists vs What Needs Building

### Reuse from `pecos-phir-json/src/v0_1/`:

| Component | File | Notes |
|-----------|------|-------|
| PHIR JSON parsing | `ast.rs` | Complete |
| Variable storage | `environment.rs` | Complete -- DataType, TypedValue, Environment |
| Expression eval | `expression.rs` | Complete -- all operators |
| Classical ops | `operations.rs` | Adapt -- different interface needed |
| Block flattening | `block_iterative_executor.rs` | Adapt -- yield pattern differs |

### New code needed:

| Component | Location | Description |
|-----------|----------|-------------|
| Rust interpreter struct | `pecos-phir-json` | State machine wrapping existing internals |
| PyO3 wrapper class | `pecos-rslib` | `#[pyclass]` with protocol methods |
| PyO3 iterator | `pecos-rslib` | `__iter__`/`__next__` for execute() generator |
| QOp/MOp pyclass | `pecos-rslib` | Lightweight attribute bags for yielded ops |
| Name resolver | `pecos-phir-json` | Port of sim_name_resolver |
| result_bits() | `pecos-phir-json` | Bit extraction with private var filtering |
| receive_results() | `pecos-phir-json` | Handle list[dict] format |
| results() | `pecos-phir-json` | Return dict with return_int flag |

## Validation Strategy

The Rust interpreter must produce identical results to the Python one. Test by:
1. Running existing PHIR test programs through both interpreters
2. Comparing shot-by-shot results
3. Testing edge cases (empty programs, no measurements, only classical ops)
4. Testing the dual-interpreter pattern (outer + inner with private vars)
