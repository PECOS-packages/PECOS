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

```python
from pecos_rslib import Pauli, PauliString

# From string notation
p = PauliString.from_str("XZI")  # X on qubit 0, Z on qubit 1, I on qubit 2
q = PauliString.from_str("ZXI")

# From list of (Pauli, qubit) pairs
p = PauliString([(Pauli.X, 0), (Pauli.Z, 1)])

# Get components
print(p.get_paulis())  # [(Pauli.X, 0), (Pauli.Z, 1)]
print(p.get_phase())  # 0 (0=+1, 1=+i, 2=-1, 3=-i)

# String representation
print(p)  # Shows sparse representation with non-identity operators
```

### Matrix Representation

```python
from pecos_rslib import PauliString

p = PauliString.from_str("XZ")
matrix = p.to_matrix()  # Returns complex matrix as list of lists
# Each element is a (real, imag) tuple
```

## Stabilizer Groups

`PauliStabilizerGroup` represents a group of mutually commuting Pauli strings with real phases (+1 or -1). This is the standard stabilizer group used in QEC.

```python
from pecos_rslib import PauliString, PauliStabilizerGroup

# Create from generators
g1 = PauliString.from_str("ZZI")
g2 = PauliString.from_str("IZZ")
group = PauliStabilizerGroup([g1, g2])

# Basic properties
print(group.rank())  # 2
print(group.num_qubits())  # 3
print(group.num_generators())  # 2
print(group.is_independent())  # True

# Membership testing (GF(2) span)
ziz = PauliString.from_str("ZIZ")
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
from pecos_rslib import PauliString, PauliStabilizerGroup

group = PauliStabilizerGroup.from_str("ZZI\nIZZ")

# Add a generator (must commute with existing generators)
group.add_generator(PauliString.from_str("XXX"))

# Remove a generator by index
removed = group.remove_generator(2)  # Returns the removed PauliString

# Merge two groups
other = PauliStabilizerGroup.from_str("XXXX")
group.merge(other)
```

## Pauli Sequences

`PauliSequence` is an ordered list of Pauli strings with no constraints (they can anticommute). Provides GF(2) symplectic analysis.

```python
from pecos_rslib import PauliString, PauliSequence

p1 = PauliString.from_str("XZ")
p2 = PauliString.from_str("ZX")
seq = PauliSequence([p1, p2])

# Analysis
print(seq.rank())  # 2 (linearly independent)
print(seq.is_abelian())  # False (XZ and ZX anticommute)

# Commutation matrix
comm = seq.commutation_matrix()
# comm[i][j] is True if seq[i] commutes with seq[j]

# GF(2) membership
print(seq.contains(PauliString.from_str("YY")))  # True (XZ * ZX = -YY in GF(2) span)

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
from pecos_rslib import StabilizerCode, PauliString

code = StabilizerCode.repetition(3)

# X error on qubit 0
error = PauliString.from_str("XII")
syndrome = code.syndrome(error)
print(syndrome)  # [True, False] -- triggers first stabilizer

# X error on qubit 1
error = PauliString.from_str("IXI")
syndrome = code.syndrome(error)
print(syndrome)  # [True, True] -- triggers both stabilizers

# Z error (undetectable by Z-stabilizers)
error = PauliString.from_str("ZII")
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
