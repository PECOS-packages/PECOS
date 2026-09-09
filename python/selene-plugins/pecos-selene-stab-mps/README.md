# pecos-selene-stab-mps

PECOS StabMps (stabilizer tableau + MPS) simulator plugin for the Selene quantum emulator.

Stabilizer gates are O(n) on the tableau; non-Clifford rotations decompose in the stabilizer basis and apply to the MPS. Cost is polynomial when non-Clifford count is bounded.

## Usage

```python
from pecos_selene_stab_mps import StabMpsPlugin

sim = StabMpsPlugin()
```

The plugin inherits `StabMps::for_qec()`'s Exact measurement mode. The frozen
Stage-B repetition-code benchmark in
`exp/pecos-stab-tn/examples/measurement_mode_bench.rs` measured a 1.497x
geometric-mean slowdown and a 1.548x per-workload maximum versus Pragmatic on
that noisy repetition-code family,
inside the decided Exact-everywhere limits. Exact avoids Pragmatic mode's
known conditional-state bias on honest non-Clifford circuit families.
