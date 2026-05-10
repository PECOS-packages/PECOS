# Pauli Algebra and QEC in Python

This guide covers using PECOS's Pauli algebra and stabilizer code types from Python via `pecos_rslib`. These are Python bindings to the same Rust types documented in the [Quantum Operator Algebra](quantum-operator-algebra.md) and [Stabilizer Codes](stabilizer-codes.md) guides.

## What You'll Learn

- Working with Pauli strings in Python
- Building and analyzing stabilizer groups
- Using `PauliSequence` for GF(2) linear algebra
- Creating stabilizer codes and computing code parameters
- Computing syndromes, logical operators, and distance

## Installation

The bindings are included in the `quantum-pecos` package:

```bash
pip install quantum-pecos
```

```python
from pecos_rslib import (
    Pauli,
    PauliString,
    PauliStabilizerGroup,
    PauliSequence,
    StabilizerCode,
)
```

## Pauli Operators

### Single-Qubit Paulis

```python
from pecos_rslib import Pauli

# The four single-qubit Pauli operators
i = Pauli.I
x = Pauli.X
y = Pauli.Y
z = Pauli.Z

# From string
x = Pauli.from_str("X")

# To/from integer (I=0, X=1, Z=2, Y=3)
assert Pauli.from_int(1) == Pauli.X
assert Pauli.X.to_int() == 1
```

### Pauli Strings

`PauliString` represents a multi-qubit Pauli operator with a phase from {+1, -1, +i, -i}.
For inline code, prefer constructor syntax such as `X(0) & Z(1)`. Use sparse
strings like `"X0 Z1"` when text input is useful and dense strings like `"XZ"`
for compact table-like input.

```python
from pecos_rslib import Pauli, PauliString, X, Z

# Constructor syntax
p = X(0) & Z(1)
q = Z(0) & X(1)

# From list of (Pauli, qubit) pairs
same_p = PauliString([(Pauli.X, 0), (Pauli.Z, 1)])
assert p == same_p

# From string notation
sparse_text = PauliString.from_sparse_str("X0 Z1")
dense_text = PauliString.from_dense_str("XZ")
auto_detected = PauliString.from_str("X0 Z1")
assert p == sparse_text
assert p == dense_text
assert p == auto_detected

# To string notation
assert p.to_sparse_str() == "+X0 Z1"
assert p.to_dense_str() == "+XZ"

# Get components
print(p.get_paulis())  # [(Pauli.X, 0), (Pauli.Z, 1)]
print(p.get_phase())  # 0 (0=+1, 1=+i, 2=-1, 3=-i)

# String representation
print(p)  # Shows sparse representation with non-identity operators
```

### Matrix Representation

```python
from pecos_rslib import X, Z

p = X(0) & Z(1)
matrix = p.to_matrix()  # Returns complex matrix as list of lists
# Each element is a (real, imag) tuple
```

## Stabilizer Groups

`PauliStabilizerGroup` represents a group of mutually commuting Pauli strings with real phases (+1 or -1). This is the standard stabilizer group used in QEC.

```python
from pecos_rslib import PauliStabilizerGroup, Z

# Create from generators
g1 = Z(0) & Z(1)
g2 = Z(1) & Z(2)
group = PauliStabilizerGroup([g1, g2])

# Basic properties
print(group.rank())  # 2
print(group.num_qubits())  # 3
print(group.num_generators())  # 2
print(group.is_independent())  # True

# Membership testing (GF(2) span)
ziz = Z(0) & Z(2)
print(group.contains(ziz))  # True (ZIZ = ZZI * IZZ)
print(group.contains_with_phase(ziz))  # True (with correct +1 phase)

# Get generators
for stab in group.stabilizers():
    print(stab)
```

### From String Notation

```python
from pecos_rslib import PauliStabilizerGroup

# Dense notation (one string per line)
group = PauliStabilizerGroup.from_str("ZZI\nIZZ")

# Sparse notation also supported
group = PauliStabilizerGroup.from_str("Z0 Z1\nZ1 Z2")
```

### Modifying Groups

```python
from pecos_rslib import PauliStabilizerGroup, X

group = PauliStabilizerGroup.from_str("ZZI\nIZZ")

# Add a generator (must commute with existing generators)
group.add_generator(X(0) & X(1) & X(2))

# Remove a generator by index
removed = group.remove_generator(2)  # Returns the removed PauliString

# Merge two groups
other = PauliStabilizerGroup.from_str("XXXX")
group.merge(other)
```

## Pauli Sequences

`PauliSequence` is an ordered list of Pauli strings with no constraints (they can anticommute). Provides GF(2) symplectic analysis.

```python
from pecos_rslib import PauliSequence, X, Y, Z

p1 = X(0) & Z(1)
p2 = Z(0) & X(1)
seq = PauliSequence([p1, p2])

# Analysis
print(seq.rank())  # 2 (linearly independent)
print(seq.is_abelian())  # False (XZ and ZX anticommute)

# Commutation matrix
comm = seq.commutation_matrix()
# comm[i][j] is 1 if seq[i] anticommutes with seq[j]

# GF(2) membership
print(seq.contains(Y(0) & Y(1)))  # True (XZ * ZX = -YY in GF(2) span)

# Row reduction to independent subset
reduced = seq.row_reduce()
print(len(reduced))  # Number of independent generators
```

## Stabilizer Codes

`StabilizerCode` wraps a `PauliStabilizerGroup` with an explicit qubit count and provides QEC analysis.

### Standard Code Constructors

```python
from pecos_rslib import StabilizerCode

# Built-in codes
rep = StabilizerCode.repetition(3)  # [[3, 1, 1]] bit-flip code
steane = StabilizerCode.steane()  # [[7, 1, 3]] Steane code
five = StabilizerCode.five_qubit()  # [[5, 1, 3]] perfect code
shor = StabilizerCode.shor()  # [[9, 1, 3]] Shor code
four = StabilizerCode.four_two_two()  # [[4, 2, 2]] detection code
toric = StabilizerCode.toric(3)  # [[18, 2, 3]] toric code
```

### From a Stabilizer Group

```python
from pecos_rslib import PauliStabilizerGroup, StabilizerCode

group = PauliStabilizerGroup.from_str("ZZI\nIZZ")
code = StabilizerCode(group)

# With explicit qubit count (when generators don't touch all qubits)
code = StabilizerCode(group, num_qubits=5)
# Now k = 5 - 2 = 3 logical qubits instead of 1
```

### Code Parameters

```python
from pecos_rslib import StabilizerCode

code = StabilizerCode.steane()
print(code.num_qubits())  # 7
print(code.num_logical_qubits())  # 1
print(code.code_parameters())  # "[[7, 1]]"
```

### Distance

```python
from pecos_rslib import StabilizerCode

code = StabilizerCode.steane()
d = code.distance()
print(d)  # 3

# Returns None if there are no logical qubits
# Only suitable for small codes (exponential complexity)
```

### Logical Operators

```python
from pecos_rslib import StabilizerCode

code = StabilizerCode.steane()
logicals = code.logical_operators()
print(len(logicals))  # 2 (one X-type and one Z-type for k=1)

for op in logicals:
    print(op)
```

### Syndrome Computation

```python
from pecos_rslib import StabilizerCode, X, Z

code = StabilizerCode.repetition(3)

# X error on qubit 0
error = X(0)
syndrome = code.syndrome(error)
print(syndrome)  # [True, False] -- triggers first stabilizer

# X error on qubit 1
error = X(1)
syndrome = code.syndrome(error)
print(syndrome)  # [True, True] -- triggers both stabilizers

# Z error (undetectable by Z-stabilizers)
error = Z(0)
syndrome = code.syndrome(error)
print(syndrome)  # [False, False]
```

### Accessing the Group

```python
from pecos_rslib import StabilizerCode

code = StabilizerCode.steane()
group = code.group()
print(group.rank())  # 6
print(group.num_generators())  # 6
```

## Type Relationships

```text
Python                           Rust
------                           ----
Pauli                        ->  pecos_core::Pauli
PauliString                  ->  pecos_core::PauliString
PauliSequence                ->  pecos_quantum::PauliSequence
PauliStabilizerGroup         ->  pecos_quantum::PauliStabilizerGroup
StabilizerCode               ->  pecos_qec::StabilizerCode
```

For the full Rust API (which includes additional types like `PauliSet`, `PauliGroup`, `CliffordRep`, `Op`, `StabilizerCodeSpec`, etc.), see the [Rust API docs](https://docs.rs/pecos).
