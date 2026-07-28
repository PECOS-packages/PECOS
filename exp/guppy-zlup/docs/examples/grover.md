# Example: Grover's Algorithm

This example demonstrates a more complex program with loops: a simplified
Grover's search iteration.

## The Guppy Program

```python
# grover.py
def grover_iteration(q: qubit[4], iterations: int) -> None:
    """
    Perform Grover iterations on a 4-qubit register.

    This is a simplified version that applies:
    1. Oracle (placeholder: CZ on first two qubits)
    2. Diffusion operator (H gates + phase flip + H gates)
    """
    for i in range(iterations):
        # Oracle (simplified)
        cz(q[0], q[1])

        # Diffusion: H on all qubits
        h(q[0])
        h(q[1])
        h(q[2])
        h(q[3])

        # Phase flip would go here

        # Diffusion: H on all qubits again
        h(q[0])
        h(q[1])
        h(q[2])
        h(q[3])


def main() -> None:
    """Run Grover's algorithm."""
    q = qubit[4]

    # Initialize superposition
    h(q[0])
    h(q[1])
    h(q[2])
    h(q[3])

    # Run iterations (optimal is ~π/4 * √N for N=16)
    grover_iteration(q, 3)

    # Measure and emit results
    m = measure(q)
    result("measurements", m)
```

## Linting Results

```bash
$ guppy-zlup check grover.py
No issues found.
All checks passed!
```

This program passes all checks:

- ✓ **ZLUP001**: The `for` loop is bounded by `iterations` parameter
- ✓ **ZLUP002**: No recursive calls
- ✓ **ZLUP003**: No allocations inside the loop
- ✓ **ZLUP006**: All types annotated

## What Would Fail?

### Unbounded Loop (ZLUP001)

```python
def bad_grover(q: qubit[4]) -> None:
    while True:  # ZLUP001: unbounded loop
        cz(q[0], q[1])
        if converged():
            break
```

**Fix**: Use a bounded loop with maximum iterations.

### Allocation in Loop (ZLUP003)

```python
def bad_grover(iterations: int) -> None:
    for i in range(iterations):
        q = qubit[4]  # ZLUP003: allocation in loop
        h(q[0])
```

**Fix**: Allocate qubits outside the loop.

### Missing Types (ZLUP006)

```python
def bad_grover(q, iterations):  # ZLUP006: missing types
    for i in range(iterations):
        h(q[0])
```

**Fix**: Add type annotations.

## Generated IR

```bash
$ guppy-zlup emit grover.py -o grover.json
```

Key parts of the IR for `grover_iteration`:

```json
{
  "name": "grover_iteration",
  "params": [
    {"name": "q", "type": {"kind": "qalloc", "size": {"kind": "literal", "value": 4}}},
    {"name": "iterations", "type": {"kind": "primitive", "name": "int"}}
  ],
  "body": [
    {
      "kind": "for",
      "var": "i",
      "range": {
        "start": {"kind": "literal", "value": 0},
        "end": {"kind": "ident", "name": "iterations"}
      },
      "body": [
        {"kind": "gate", "gate": "cz", "targets": [...]},
        {"kind": "gate", "gate": "h", "targets": [...]},
        ...
      ]
    }
  ]
}
```

## Generated Zlup

```bash
$ guppy-zlup compile grover.json --stdout
```

```zlup
fn grover_iteration(q: &mut [4]qubit, iterations: i64) -> unit {
    for 0..iterations |i| {
        cz (q[0], q[1]);
        h q[0];
        h q[1];
        h q[2];
        h q[3];
        h q[0];
        h q[1];
        h q[2];
        h q[3];
    }
}

fn main() -> unit {
    mut q := qalloc(4);
    h q[0];
    h q[1];
    h q[2];
    h q[3];
    grover_iteration(&mut q, 3);
    m := mz([4]u1) q;
    result("measurements", m);
    return;
}
```

## Key Observations

### Loop Translation

| Guppy                          | Zlup                    |
|--------------------------------|-------------------------|
| `for i in range(iterations):`  | `for 0..iterations |i|` |
| `for i in range(0, n):`        | `for 0..n |i|`          |
| `for i in range(1, n, 2):`     | `for 1..n:2 |i|`        |

### Type Mapping

| Guppy          | Zlup        |
|----------------|-------------|
| `int`          | `i64`       |
| `float`        | `f64`       |
| `bool`         | `bool`      |
| `qubit[4]`     | `qalloc(4)` |
| `None`         | `unit`      |

## Circuit Visualization

One Grover iteration:

```
q[0] ──●──H──────H──
       │
q[1] ──Z──H──────H──

q[2] ─────H──────H──

q[3] ─────H──────H──
       │    │
     Oracle Diffusion
```

The full algorithm:
1. Initialize all qubits to |+⟩ (H gates)
2. Repeat O(√N) times:
   - Apply oracle (marks solution)
   - Apply diffusion (amplifies marked state)
3. Measure to get solution with high probability
