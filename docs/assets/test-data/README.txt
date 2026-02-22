# Test Data Directory

This directory contains test data files (HUGR programs, WASM modules, etc.) used by documentation code examples.

## Usage

When documentation code examples need external files, place them here. The doc test generator (`scripts/docs/generate_doc_tests.py`) can copy these files to the test environment.

## Example Files

- `repetition_code.hugr` - A compiled repetition code circuit (TODO: generate)

## Generating HUGR Files

HUGR files can be generated from Guppy programs:

```python
from guppylang import GuppyModule
from guppylang.std.quantum import qubit, cx, measure

@guppy
def my_circuit() -> None:
    q = qubit()
    # ... circuit logic ...

# Export to HUGR
my_circuit.compile().save("my_circuit.hugr")
```

## Note

If a documentation example requires a file that doesn't exist here, mark it with `<!--skip: Requires filename.hugr-->` in the documentation.
