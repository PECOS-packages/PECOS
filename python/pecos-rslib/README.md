# pecos-rslib

`pecos-rslib` provides Rust extensions for the Python version of PECOS.

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
