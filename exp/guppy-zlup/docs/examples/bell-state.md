# Example: Bell State

This example walks through compiling a simple Bell state preparation circuit
from Guppy to Zlup.

## The Guppy Program

A Bell state is a maximally entangled two-qubit state. Here's the Guppy code:

```python
# bell.py
def bell() -> None:
    """Prepare and measure a Bell state |Φ+⟩ = (|00⟩ + |11⟩) / √2"""
    q = qubit[2]          # Allocate 2 qubits
    h(q[0])               # Hadamard on first qubit
    cx(q[0], q[1])        # CNOT: control=q[0], target=q[1]
    m = measure(q)        # Measure both qubits
    result("measurements", m)  # Emit results to runtime
```

## Step 1: Lint the Code

First, check for any rule violations:

```bash
$ guppy-zlup check bell.py
No issues found.
All checks passed!
```

The program passes all checks:
- ✓ No unbounded loops (ZLUP001)
- ✓ No recursion (ZLUP002)
- ✓ No dynamic allocation in loops (ZLUP003)
- ✓ No dynamic dispatch (ZLUP004)
- ✓ Return type annotated (ZLUP006)

## Step 2: Emit IR

Generate the intermediate representation:

```bash
$ guppy-zlup emit bell.py -o bell.json
Wrote IR to bell.json
```

The generated IR:

```json
{
  "version": "0.1.0",
  "source_file": "bell.py",
  "functions": [
    {
      "name": "bell",
      "return_type": {"kind": "primitive", "name": "None"},
      "body": [
        {
          "kind": "qalloc",
          "name": "q",
          "size": {"kind": "literal", "value": 2}
        },
        {
          "kind": "gate",
          "gate": "h",
          "targets": [
            {"kind": "index", "array": "q", "index": {"kind": "literal", "value": 0}}
          ]
        },
        {
          "kind": "gate",
          "gate": "cx",
          "targets": [
            {"kind": "index", "array": "q", "index": {"kind": "literal", "value": 0}},
            {"kind": "index", "array": "q", "index": {"kind": "literal", "value": 1}}
          ]
        },
        {
          "kind": "assign",
          "target": {"kind": "ident", "name": "m"},
          "value": {
            "kind": "call",
            "callee": "measure",
            "args": [{"kind": "ident", "name": "q"}]
          }
        },
        {
          "kind": "result",
          "tag": "measurements",
          "value": {"kind": "ident", "name": "m"}
        }
      ]
    }
  ]
}
```

## Step 3: Compile to Zlup

Transform the IR to Zlup source code:

```bash
$ guppy-zlup compile bell.py -o bell.zlp
```

The generated Zlup code:

```zlup
fn bell() -> unit {
    mut q := qalloc(2);
    h q[0];
    cx (q[0], q[1]);
    m := mz([2]u1) q;
    result("measurements", m);
    return;
}
```

Note: Entry/main functions return `unit`. Use explicit `result(tag, value)` calls to emit outputs to the quantum runtime.

## Understanding the Transformation

| Guppy                       | Zlup                       | Notes                        |
|-----------------------------|----------------------------|------------------------------|
| `q = qubit[2]`              | `mut q := qalloc(2);`      | Qubit allocation             |
| `h(q[0])`                   | `h q[0];`                  | Gate as statement            |
| `cx(q[0], q[1])`            | `cx (q[0], q[1]);`         | Multi-qubit gate             |
| `m = measure(q)`            | `m := mz([2]u1) q;`        | Measurement with result      |
| `result("tag", val)`        | `result("tag", val);`      | Emit value to runtime        |
| `-> None`                   | `-> unit`                  | Entry functions return unit  |

### Measurement Variants

Zlup supports flexible measurement syntax:

```zlup
// Per-qubit into array (default)
m := mz([2]u1) q;

// Single qubit
bit := mz(u1) q[0];

// Pack into custom struct (for QEC syndromes, etc.)
syndrome := mz(pack Syndrome) [ancilla[0], ancilla[1], ancilla[2]];
```

## The Quantum Circuit

The Bell state circuit:

```
q[0] ──H──●──M──
          │
q[1] ─────X──M──
```

1. **H gate** puts q[0] in superposition: |0⟩ → (|0⟩ + |1⟩) / √2
2. **CNOT** entangles the qubits: creates |Φ+⟩ = (|00⟩ + |11⟩) / √2
3. **Measure** collapses to either |00⟩ or |11⟩ with equal probability

## What's Next?

The Zlup code can now be:

- Compiled to QASM for hardware execution
- Optimized by the Zlup compiler
- Analyzed for resource usage
