# Stabilizer-Code Verification

This guide covers designing, verifying, and analyzing stabilizer codes with the
Rust-backed types in `pecos.quantum`. The workflow starts with Pauli checks,
discovers a compatible logical basis, and searches for low-weight logical
operators.

## What You'll Learn

- Building a `StabilizerCodeSpec` from Pauli checks
- Diagnosing anticommuting and dependent generators
- Discovering logical operators and calculating code distance
- Searching a range of low-weight logical operators
- Importing CSS and symplectic check matrices
- Choosing between the two exact distance methods

```hidden-python
import re

import numpy as np
from pecos.quantum import (
    ParityCheckMatrix,
    StabilizerCode,
    StabilizerCodeSpec,
    SymplecticMatrix,
    X,
    Xs,
    Y,
    Ys,
    Z,
    Zs,
    pauli_string,
)


def add_checks(builder, checks):
    for check in checks:
        builder.check(check)
    return builder


def original_checks():
    return [
        Xs([3, 4, 7, 8]),
        Xs([5, 6, 7, 9]),
        Zs([2, 4, 5, 7]),
        Zs([7, 8, 9]),
        Zs([0, 1]) * Y(2),
        pauli_string("X0 X2 Z3 Y4"),
        pauli_string("X1 X2 Z6 Y5"),
    ]


def fixed_checks():
    checks = original_checks()
    checks[4] = Zs([0, 1, 2])
    return checks


def final_checks():
    return [
        Zs([2, 4, 5, 7]),
        Xs([3, 4, 7, 8]),
        Xs([5, 6, 7, 9]),
        pauli_string("X0 X2 Z3 Y4"),
        pauli_string("X1 X2 Z6 Y5"),
        Zs([0, 1, 2]),
        Xs([0, 1]),
        Zs([3, 8]),
        Zs([6, 9]),
    ]
```

## Overview

`StabilizerCodeSpec.builder(num_qubits)` collects stabilizer checks and,
optionally, explicitly chosen logical operators. A check is a `PauliString`.
The single-qubit `X`, `Y`, and `Z` constructors compose with `&`; `Xs`, `Ys`,
and `Zs` construct one Pauli type on several qubits at once. Multiplication
with `*` provides Pauli multiplication, while `pauli_string` parses sparse
text.

```python
first = Xs([3, 4, 7, 8])
assert first == X(3) & X(4) & X(7) & X(8)

mixed = Zs([0, 1]) * Y(2)
assert mixed == Z(0) & Z(1) & Y(2)
assert Ys([1, 5]) == Y(1) & Y(5)
assert pauli_string("X1 X2 Z6 Y5") == X(1) & X(2) & Z(6) & Y(5)

builder = StabilizerCodeSpec.builder(10)
builder.check(first)
builder.check(mixed)
```

Use `build()` when only count and independence validation is needed,
`build_verified()` to validate a supplied stabilizer and logical basis, or
`build_with_discovered_logicals()` to verify the checks and discover paired
logical operators and destabilizers.

## Developing a Ten-Qubit Code

Consider a ten-qubit design with seven proposed checks. The fifth check has a
`Y` on qubit 2:

```python
checks = original_checks()
builder = add_checks(StabilizerCodeSpec.builder(10), checks)

try:
    builder.build_verified()
except ValueError as error:
    message = str(error)
else:
    raise AssertionError("the original checks should not verify")

pair = re.search(r"generators (\d+) and (\d+) anticommute", message)
assert pair is not None
first_index, second_index = (int(index) for index in pair.groups())
assert checks[first_index].anticommutes_with(checks[second_index])
print(message)
```

```text
Stabilizer generators 2 and 4 anticommute
```

The indices identify the offending entries in insertion order. Replacing that
mixed check with `Zs([0, 1, 2])` produces a valid `[[10, 3]]` code. Building
with discovered logicals also supplies the destabilizer and paired-logical
generators displayed by `print(spec)`:

```python
builder = add_checks(StabilizerCodeSpec.builder(10), fixed_checks())
spec = builder.build_with_discovered_logicals()
result = spec.distance()

assert spec.num_logical_qubits == 3
assert result is not None
assert result.distance == 2
assert result.min_weight_operator.weight() == 2
print(spec)
print(result)
```

```text
[[10, 3]]
Stabilizer generators:
+IIIXXIIXXI
+IIIIIXXXIX
+IIZIZZIZII
+IIIIIIIZZZ
+ZZZIIIIIII
+XIXZYIIIII
+IXXIIYZIII
Destabilizer generators:
+IIIZIIIIII
+IZIIIZIIII
+ZIIIXIIIII
+IIIIIIIIXI
+ZIXIXIIIII
+ZIIIIIIIII
+IZIIIIIIII
Logical operators:
Z1: +IZIIIZZIII
X1: +IZIIIIXIII
Z2: +IZIZIZIZII
X2: +ZIIIXIIXXI
Z3: +IZIIIZIIIZ
X3: +IIIIIIIIXX

DistanceResult(distance=2, min_weight_operator=X_0 X_1)
```

The parameters line is `[[n, k]]`; the distance result adds the minimum
logical weight and one operator attaining it. Adding checks that detect the
weight-two logicals, while removing the original `Zs([7, 8, 9])` check, gives
the final nine-check design:

```python
builder = add_checks(StabilizerCodeSpec.builder(10), final_checks())
spec = builder.build_with_discovered_logicals()
result = spec.distance()

assert spec.num_logical_qubits == 1
assert result is not None
assert result.distance == 3
assert result.min_weight_operator.weight() == 3
assert str(spec).splitlines()[0] == "[[10, 1]]"
print(str(spec).splitlines()[0])
print(result)
```

```text
[[10, 1]]
DistanceResult(distance=3, min_weight_operator=X_0 X_2 X_7)
```

This is a `[[10, 1, 3]]` code: it encodes one logical qubit into ten physical
qubits and has distance three.

## Exploring Low-Weight Logicals

`min_weight_logicals()` returns every logical operator found at the minimum
weight. Each `LogicalOperatorInfo` records the Pauli operator, its weight, and
which chosen logical generators it is equivalent to modulo stabilizers.
`equivalence_string()` formats that last field compactly.

`shortest_logicals(delta)` continues through `delta` weights above the minimum.
For the five-qubit code there are 30 weight-three logical operators. No
weight-four logicals exist, and `delta=2` exposes another 18 at weight five:

```python
spec = StabilizerCodeSpec.from_stabilizer_code(StabilizerCode.five_qubit())
minimum = spec.min_weight_logicals()
spectrum = spec.shortest_logicals(delta=2)

assert len(minimum) == 30
assert [info.operator for info in spectrum[: len(minimum)]] == [
    info.operator for info in minimum
]
assert {info.weight for info in spectrum} == {3, 5}
assert sum(info.weight == 5 for info in spectrum) == 18
assert len(spectrum) == 48
assert all(info.equivalence_string() for info in spectrum)

first = minimum[0]
print(first.operator, first.weight, first.equivalence_string())
```

```text
X_0 Y_1 X_2 3 X0*Z0
```

Only genuine logical operators are returned; stabilizers and detected
operators are excluded even when their weights fall inside the requested
range.

## Matrix Input

For CSS codes, `ParityCheckMatrix` represents a role-neutral binary
checks-by-qubits matrix. `checks_from_css(x_stabilizers, z_stabilizers)` chooses
the role: rows in the first matrix become X-type stabilizers, and rows in the
second become Z-type stabilizers.

### CSS Parity-Check Matrices

The Steane code uses the classical Hamming parity-check matrix for both
blocks. Nested Python sequences and NumPy integer arrays are accepted:

```python
hamming_h = [
    [1, 0, 1, 0, 1, 0, 1],
    [0, 1, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 1, 1, 1],
]

plain = ParityCheckMatrix(hamming_h)
int64_matrix = ParityCheckMatrix(np.asarray(hamming_h, dtype=np.int64))
uint8_matrix = ParityCheckMatrix(np.asarray(hamming_h, dtype=np.uint8))
assert plain.rows() == int64_matrix.rows() == uint8_matrix.rows() == hamming_h

builder = StabilizerCodeSpec.builder(7)
builder.checks_from_css(int64_matrix, uint8_matrix)
steane = builder.build_with_discovered_logicals()
result = steane.distance(css=True)

assert steane.num_logical_qubits == 1
assert result is not None
assert result.distance == StabilizerCode.steane().distance() == 3
```

The builder checks CSS orthogonality before appending rows. It reports the
first X-row and Z-row pair with an odd overlap. The code-spec constructor then
protects the independent-generator invariant and reports both rank and count:

```python
builder = StabilizerCodeSpec.builder(2)
try:
    builder.checks_from_css(
        ParityCheckMatrix([[1, 0]]),
        ParityCheckMatrix([[1, 0]]),
    )
except ValueError as error:
    orthogonality_message = str(error)
else:
    raise AssertionError("non-orthogonal CSS rows should be rejected")
assert "X row 0 and Z row 0" in orthogonality_message

dependent = ParityCheckMatrix(
    [
        [1, 1, 0],
        [0, 1, 1],
        [1, 0, 1],
    ]
)
builder = StabilizerCodeSpec.builder(3)
builder.checks_from_css(dependent, ParityCheckMatrix.zeros(0, 3))
try:
    builder.build()
except ValueError as error:
    dependence_message = str(error)
else:
    raise AssertionError("dependent stabilizers should be rejected")
assert "rank 2, count 3" in dependence_message
```

`ParityCheckMatrix.zeros(0, n)` carries the width that an empty nested list
cannot express. It is useful for a code with only one stabilizer type:

```python
x_stabilizers = ParityCheckMatrix([[1, 1]])
z_stabilizers = ParityCheckMatrix.zeros(0, 2)
assert z_stabilizers.rows() == []
assert z_stabilizers.num_qubits() == 2

builder = StabilizerCodeSpec.builder(2)
builder.checks_from_css(x_stabilizers, z_stabilizers)
spec = builder.build_with_discovered_logicals()
assert spec.stabilizers == x_stabilizers.to_x_stabilizers()
assert spec.num_logical_qubits == 1
```

### Symplectic Matrices

`SymplecticMatrix` stores each Pauli row as `[X block | Z block]`. A set bit in
both blocks represents `Y`; phase information is not present, so
`to_positive_paulis()` always returns positive-phase operators.

These are the four stabilizer rows of the five-qubit code:

```python
rows = [
    [1, 0, 0, 1, 0, 0, 1, 1, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 1, 1, 0],
    [1, 0, 1, 0, 0, 0, 0, 0, 1, 1],
    [0, 1, 0, 1, 0, 1, 0, 0, 0, 1],
]
matrix = SymplecticMatrix.from_dense(rows)
assert matrix.x_block() == [row[:5] for row in rows]
assert matrix.z_block() == [row[5:] for row in rows]

builder = StabilizerCodeSpec.builder(5)
builder.checks_from_symplectic(matrix)
five_qubit = builder.build_with_discovered_logicals()
result = five_qubit.distance()

assert matrix.to_positive_paulis() == five_qubit.stabilizers
assert result is not None
assert result.distance == 3
```

As with CSS ingestion, width mismatches and anticommuting rows raise
`ValueError` before a spec is built.

## Choosing a Distance Method

Two exact distance calculations serve different regimes:

| Method | Search strategy | Best use |
|--------|-----------------|----------|
| `StabilizerCode.distance()` | Enumerates stabilizer/logical cosets | Tiny codes with small generator counts |
| `StabilizerCodeSpec.distance()` | Enumerates Paulis by increasing weight | Codes whose distance is small relative to their length |

The coset method is a useful oracle for tiny built-in codes. The spec method
supports `max_weight` as a search budget and `verbose=True` to print
`Checking weight N...` progress to standard error. It returns `None` if no
logical operator is found within the budget:

```python
code = StabilizerCode.five_qubit()
spec = StabilizerCodeSpec.from_stabilizer_code(code)

result = spec.distance()
assert result is not None
assert result.distance == code.distance() == 3
assert spec.distance(max_weight=2, verbose=True) is None
```

The same `max_weight`, `css`, and `verbose` controls are available on
`min_weight_logicals()` and `shortest_logicals()`.

## Next Steps

- **[Pauli Algebra and QEC in Python](python-pauli-qec.md)** - Work with Pauli strings, sequences, and stabilizer groups
- **[Stabilizer Codes](stabilizer-codes.md)** - Understand the Rust stabilizer-code model
- **[QEC Geometry](qec-geometry.md)** - Describe layouts and check supports for code families
