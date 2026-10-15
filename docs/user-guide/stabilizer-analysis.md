# Pauli and stabilizer analysis

`pecos.analysis` provides conversion and analysis helpers backed by `PauliString`.
Dense inputs contain only `I`, `X`, `Y`, and `Z`, with the leftmost character
addressing qubit zero. Input phases are rejected; product phases are returned
separately. Sparse mappings and single-tick, metadata-free Pauli circuits are
also accepted by `commutes` and `pauli_to_string`.

```python
from pecos.analysis import (
    commutes,
    compatible_stabilizer_groups,
    iter_stabilizer_products,
    measurement_compatible,
    pauli_to_circuit,
    pauli_to_dict,
    pauli_to_string,
)

assert pauli_to_dict("IXYX") == {"X": {1, 3}, "Y": {2}}
assert pauli_to_string({"X": {1}}, 3) == "IXI"
assert commutes(pauli_to_circuit("XX"), "ZZ")
assert not measurement_compatible("XX", "ZZ")
assert list(iter_stabilizer_products(["XX", "ZZ"])) == [("+", "II"), ("+", "ZZ"), ("+", "XX"), ("-", "YY")]
assert compatible_stabilizer_groups(["XX", "ZZ"], "II")["+II"] == [{"+XX"}, {"+ZZ"}, {"-YY"}]
```

`measurement_compatible` means that operators can share individual-qubit
measurement bases, a stronger condition than commutation. It requires equal
string lengths. `commutes` instead treats missing qubits as identities.
Conversions require disjoint nonnegative integer locations and never silently
truncate an operator to fit a requested width.

`iter_stabilizer_products(generators, operator=None)` enumerates every subset in
binary order, with the last generator varying fastest. Products are ordered as
`operator * generator[0] * generator[1] * ...`. Noncommuting generators are
allowed here, and imaginary phases are retained. Duplicate products from
dependent generators are retained. With no generators, it yields the supplied
operator or the zero-qubit identity `("+", "")`.

`compatible_stabilizer_groups` requires commuting generators and returns all
maximal mutually compatible groups of nonidentity stabilizers for each operator
variant. Groups may overlap; they are not a minimum partition of measurement
settings. Every group is jointly compatible with its operator variant. Group
ordering is deterministic, although each group is represented as a set.

Enumeration is intended for small codes: product counts grow as `2**n`, and
finding all maximal compatible groups can also take exponential time and memory.
The iterator avoids retaining all products; the grouping function materializes
them. Generator independence and the absence of negative identity in the
generated group are not checked; these utilities do not validate a stabilizer code.
