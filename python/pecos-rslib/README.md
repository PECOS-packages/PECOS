# pecos-rslib

`pecos-rslib` provides Rust extensions for the Python version of PECOS.

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
