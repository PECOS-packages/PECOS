# pecos-selene-mast

PECOS Mast (magic state injection) simulator plugin for the Selene quantum emulator.

Handles non-Clifford gates via deferred ancilla projection. Bond dimension stays bounded for Clifford+T circuits.

## Usage

```python
from pecos_selene_mast import MastPlugin

sim = MastPlugin()
```
