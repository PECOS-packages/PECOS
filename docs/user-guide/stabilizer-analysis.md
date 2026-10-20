# Pauli and stabilizer analysis

Use the Rust-backed `PauliString` and `PauliStabilizerGroup` for Pauli algebra.
`pecos.analysis` provides bridges to legacy Python mappings and circuits, and
maximal local-measurement grouping.

## Converting legacy inputs

`to_pauli_string` accepts a `PauliString` unchanged, a signed dense or sparse
string, a mapping of I/X/Y/Z symbols to qubit indices, or a single-tick
`QuantumCircuit` without circuit or operation metadata. Strings use
`PauliString.from_str` auto-detection. The leftmost dense character is qubit zero.
Mapping inputs use the binding constructor's integer extraction and duplicate
nonidentity-qubit checks. A circuit cannot contain two gates on one qubit in
one tick. Identity entries are omitted.

`pauli_to_dict` and `pauli_to_circuit` accept a `PauliString`. They reject
every phase other than +1 because the legacy representations cannot preserve it.
The circuit bridge always produces one tick, including for identity.

```python
from pecos.analysis import pauli_to_circuit, pauli_to_dict, to_pauli_string

pauli = to_pauli_string("IXYX")
assert pauli_to_dict(pauli) == {"X": {1, 3}, "Y": {2}}
assert to_pauli_string(pauli_to_circuit(pauli)) == pauli
assert to_pauli_string(pauli) is pauli
assert to_pauli_string("-XZ") == -to_pauli_string("X0 Z1")
```

Qubit indices must be nonnegative integers in the platform index range; bool
and non-integers raise `TypeError`, while out-of-range integers raise
`ValueError`. Integer types supporting `__index__`, including NumPy integers,
work without conversion:

<!-- numpy-interop: passing NumPy integer qubit indices from an existing NumPy workflow -->
```python
import numpy as np
from pecos.analysis import to_pauli_string

assert to_pauli_string({"X": {np.int64(1)}}).to_dense_str(3) == "+IXI"
```

## Rust algebra methods

Use `commutes_with` for commutation, `qubitwise_commutes_with` for local
measurement compatibility, and `to_dense_str(num_qubits)` for dense strings.
Dense output includes the phase. Both commutation methods ignore phase and treat absent qubits as
identity; qubitwise commutation requires matching Paulis wherever both act
nontrivially.

```python
from pecos_rslib import PauliString, PauliStabilizerGroup

xx = PauliString.from_str("XX")
zz = PauliString.from_str("ZZ")
assert xx.commutes_with(zz)
assert not xx.qubitwise_commutes_with(zz)
assert PauliString.X(0).qubitwise_commutes_with(PauliString.Z(1))
assert (xx * zz).to_dense_str(2) == "-YY"
```

Use `PauliStabilizerGroup.elements()` to enumerate the group.
It returns a list of all `2**rank` distinct elements in Rust enumeration order.
Dependent generators are accepted. All generators must commute, have real
phases, and must not generate -I; the group constructor validates these rules.
Signed generators are supported and products retain their absolute phases.

`element(mask)` multiplies the original generators selected by the mask's bits,
with bit zero selecting the first generator. Unlike `elements()`, different
masks can give the same element for dependent generators. Masks must fit an
unsigned 64-bit integer and cannot select beyond the supplied generators.
With 64 or more generators, generators at index >= 64 cannot be selected.
Bool and non-integer masks raise `TypeError`; negative, oversized, or
out-of-generator-range masks raise `ValueError`.
`elements()` raises `ValueError` for rank > 30.

```python
from pecos_rslib import PauliStabilizerGroup, PauliString

group = PauliStabilizerGroup.from_str("XX\nZZ\n-YY")
assert group.rank() == 2
assert [p.to_dense_str(2) for p in group.elements()] == ["+II", "+XX", "+ZZ", "-YY"]
assert group.element(3) == group.element(4) == PauliString.from_str("-YY")
signed = PauliStabilizerGroup.from_str("-XX\nZZ")
assert signed.element(3) == PauliString.from_str("YY")
```

## Compatible measurement groups

`compatible_stabilizer_groups(group, operator)` accepts a
`PauliStabilizerGroup` and a `PauliString`. The operator must have real phase
(+1 or -1) and commute with every generator, otherwise it raises `ValueError`.
Identity and members of the stabilizer group are valid operators.

Variants are `operator * s` for each distinct element `s`. Each returned group
is a maximal clique of nonidentity stabilizer elements that are qubitwise
compatible with each other and with the variant. Variants with no compatible
candidate are omitted. Keys and set members are `PauliString` objects with
absolute phases. Groups may overlap; this is not a minimum partition.

Variant keys are sorted lexicographically by `str(pauli)`. The list of groups is
sorted by tuples of their members' sorted `str(pauli)` values; the members are
sets and have no iteration-order guarantee.

```python
from pecos.analysis import compatible_stabilizer_groups
from pecos_rslib import PauliStabilizerGroup, PauliString

xx = PauliString.from_str("XX")
zz = PauliString.from_str("ZZ")
group = PauliStabilizerGroup.from_str("XX\nZZ")
identity = PauliString.from_str("II")
yy = PauliString.from_str("-YY")
groups = compatible_stabilizer_groups(group, identity)
assert groups[identity] == [{yy}, {xx}, {zz}]
assert groups[xx] == [{xx}]
assert groups[zz] == [{zz}]
assert groups[yy] == [{yy}]
assert compatible_stabilizer_groups(PauliStabilizerGroup.from_str("ZZ"), PauliString.from_str("XX")) == {}
```

These APIs are intended for small codes. Element enumeration materializes
`2**rank` Paulis, building the compatibility graph compares pairs, and maximal
clique enumeration can itself require exponential time and space. The rank-30
binding limit prevents unsupported enumeration; it is not a practical memory
budget.
