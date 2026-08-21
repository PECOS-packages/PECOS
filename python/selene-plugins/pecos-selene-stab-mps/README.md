# pecos-selene-stab-mps

PECOS StabMps (stabilizer tableau + MPS) simulator plugin for the Selene quantum emulator.

Stabilizer gates are O(n) on the tableau; non-Clifford rotations decompose in the stabilizer basis and apply to the MPS. Cost is polynomial when non-Clifford count is bounded.

## Usage

```python
from pecos_selene_stab_mps import StabMpsPlugin

sim = StabMpsPlugin()
```

The plugin inherits `StabMps::for_qec()`'s Pragmatic measurement mode pending
the Stage B performance measurement. That mode preserves the current QEC
throughput behavior, but its conditional states are known to be biased on
honest non-Clifford circuit families; use the general `StabMps` interface with
Exact measurement when exact continuation or state reads are required.
