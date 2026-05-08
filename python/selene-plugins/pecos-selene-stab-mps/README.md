# pecos-selene-stab-mps

PECOS StabMps (stabilizer tableau + MPS) simulator plugin for the Selene quantum emulator.

Stabilizer gates are O(n) on the tableau; non-Clifford rotations decompose in the stabilizer basis and apply to the MPS. Cost is polynomial when non-Clifford count is bounded.

## Usage

```python
from pecos_selene_stab_mps import StabMpsPlugin

sim = StabMpsPlugin()
```
