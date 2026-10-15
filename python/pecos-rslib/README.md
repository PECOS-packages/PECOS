# pecos-rslib

`pecos-rslib` provides Rust extensions for the Python version of PECOS.

## Building wheels

Build distributable wheels through the package's PEP 517 backend:

```sh
uv build --wheel python/pecos-rslib
```

The backend uses Maturin for the extension and bundles the QIS FFI library and
Selene runtime plugins selected from Cargo's artifact messages. These libraries
are loaded dynamically, so they are not discovered by scanning the extension's
link-time dependencies. The installed package registers its own library directory
with the executor; explicit runtime overrides still take precedence.

Direct `maturin build` builds only the extension and bypasses this packaging step.
Use `uv build` (or another PEP 517 frontend) for complete wheels. Release wheels
still pass through the platform's wheel repair step before distribution.

## Optional HUGR compiler

Install `pecos-rslib[hugr]` to lower HUGR programs with Selene's compiler.
The base wheel does not install the compiler or depend on `quantum-pecos`.
`quantum-pecos` installs the compiler as a required dependency.

```python
from pecos_rslib.hugr_lowering import compile_hugr_to_qis

# llvm_ir = compile_hugr_to_qis(hugr_bytes)
```

The pure-Python `hugr_lowering` module ships inside the mixed maturin package.
It imports `selene_hugr_qis_compiler` only when compilation is requested.

## Qudit reference simulators

The multilevel state-vector and density-matrix simulators are implemented in
Rust and exposed through thin Python classes. They accept ordinary Python
sequences of complex numbers and do not require NumPy:

```python
from pecos_rslib.simulators import QutritDensityMatrix, qutrit_leakage_channel

state = QutritDensityMatrix(1, seed=42)
state.apply_kraus([0], qutrit_leakage_channel(0.01))
print(state.outcome_probabilities(0))
```

Pass `seed=` when a stochastic trajectory or measurement must be reproducible.
Omitting it uses entropy-derived randomness.

## Tick request insertion

Generic `TickHandle.add_gate` broadcasts preflight every chunk in the Rust tick
before inserting any. `.meta()` and `.metas()` apply to every batch of that
insertion. Typed broadcasts validate their complete batch before insertion too.

`TickHandle.gate_on_qubit`, `get_gate`, and `get_gate_attr` inspect an individual
stored batch without cloning its tick. `TickCircuit.gate_locations` shares the
generic insertion arity rules with Python request preparation. Prepared Python
requests preserve native fields, metadata, and custom signatures when committed.

QuantumCircuit reserves one tick-level symbol/parameter pair for a zero-qubit
operation. Identical repeats (symbol and canonical JSON parameters) are no-ops;
a different operation raises `ValueError` before the request is committed.
Identical repeats preserve the original metadata encoding, even when the same
request also inserts physical gates.

Deleting a QuantumCircuit tick clears its gates and reserved zero-qubit operation
attributes while retaining unrelated tick metadata, such as a builder phase.
