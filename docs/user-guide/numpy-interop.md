# NumPy Interoperability

PECOS ships its own native array layer (`pecos.array`, `pecos.zeros`,
`pecos.dtypes`, and the surrounding NumPy-compatible functions), and PECOS
internals use only that layer -- NumPy is not a runtime dependency of any
non-test PECOS code. That boundary is enforced by lint (see issue #458).

None of that restricts *your* code. NumPy is a first-class citizen at the
boundary: PECOS arrays convert to and from NumPy arrays cheaply and exactly,
so you can hand results to matplotlib, SciPy, pandas, or any other
NumPy-consuming library, and feed NumPy data into PECOS APIs.

This page documents the two conversion directions and the one asymmetry worth
remembering: **PECOS to NumPy is a zero-copy view; NumPy to PECOS is a copy.**

## NumPy to PECOS: `pecos.asarray` (copies)

`pecos.asarray` accepts any object exposing the NumPy array interface,
including non-contiguous views, and copies the data into a native PECOS array:

```python
import numpy as np
import pecos

n = np.arange(6, dtype=np.float64).reshape(2, 3)
a = pecos.asarray(n)

assert a.shape == (2, 3)
assert str(a.dtype) == "float64"

# The PECOS array owns its own data: later NumPy mutations do not leak in.
n[0, 0] = -1.0
assert a.tolist() == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]

# Non-contiguous NumPy views are handled correctly.
t = np.arange(9).reshape(3, 3).T
assert pecos.asarray(t).tolist() == t.tolist()
```

Dtypes round-trip exactly for the numeric types both libraries share:

```python
import numpy as np
import pecos

for dt in [np.bool_, np.uint8, np.int64, np.float32, np.float64]:
    src = np.array([0, 1], dtype=dt)
    back = np.asarray(pecos.asarray(src))
    assert back.dtype == src.dtype
    assert (back == src).all()
```

## PECOS to NumPy: `np.asarray` (zero-copy view)

PECOS arrays implement `__array_interface__`, so `np.asarray` wraps the PECOS
array's memory directly -- no copy, regardless of size:

```python
import numpy as np
import pecos
from pecos import dtypes

a = pecos.array([1, 2, 3], dtype=dtypes.uint8)
view = np.asarray(a)

assert view.dtype == np.uint8
assert view.base is a  # a view into the PECOS array's memory, not a copy
```

Because it is a writable view, mutations flow through -- in both directions:

```python
import numpy as np
import pecos
from pecos import dtypes

a = pecos.array([1, 2, 3], dtype=dtypes.uint8)
view = np.asarray(a)

view[0] = 99
assert a.tolist() == [99, 2, 3]
```

The view keeps the PECOS array alive: dropping your last reference to the
PECOS array while the NumPy view survives is safe (`view.base` holds it).

When you want an independent snapshot instead of a live view, use `np.array`,
which copies by default:

```python
import numpy as np
import pecos
from pecos import dtypes

a = pecos.array([1, 2, 3], dtype=dtypes.uint8)
snapshot = np.array(a)

snapshot[0] = 42
assert a.tolist() == [1, 2, 3]
```

## Scalars

Reductions return PECOS scalars that support the standard Python numeric
protocols, so they flow into NumPy and plain Python arithmetic directly:

```python
import pecos

total = pecos.sum(pecos.array([1.5, 2.5]))
assert float(total) + 1.0 == 5.0
```

## Pauli and PauliString arrays

Arrays of symbolic Pauli operators have no NumPy dtype and do not convert;
use `tolist()` to extract the elements:

```python
import numpy as np
import pecos

paulis = pecos.array([pecos.Pauli.X, pecos.Pauli.Z])

try:
    np.asarray(paulis)
    raise AssertionError("expected TypeError")
except TypeError:
    pass

assert paulis.tolist() == [pecos.Pauli.X, pecos.Pauli.Z]
```

## Semantic differences from NumPy

The PECOS array layer is NumPy-*compatible*, not NumPy-*identical*. Two
deliberate or known divergences matter in practice.

### Basic slicing returns copies, not views

In NumPy, `a[1:3]` is a view: writing through it mutates `a`. In PECOS,
basic slicing returns an independent copy (see issue #500):

```python
import numpy as np
import pecos
from pecos import dtypes

a = pecos.array([1, 2, 3, 4], dtype=dtypes.uint8)
s = a[1:3]
s[0] = 99
assert a.tolist() == [1, 2, 3, 4]  # original untouched

n = np.array([1, 2, 3, 4], dtype=np.uint8)
v = n[1:3]
v[0] = 99
assert n.tolist() == [1, 99, 3, 4]  # NumPy view wrote through
```

Copy semantics are the safer default for QEC workloads -- no
action-at-a-distance through forgotten aliases -- at the cost of NumPy's
in-place slice-mutation idiom. To mutate a region in place, assign to the
slice directly (`a[1:3] = new_values`), which works in both libraries.

### Truthiness is container-style, not NumPy-style

`bool()` on a PECOS array currently reports "is it non-empty", like a Python
list (issue #531) -- so `bool(pecos.array([0]))` is `True`, where NumPy would
say `False`,
and multi-element arrays are truthy where NumPy raises the ambiguity error.
`pecos.bool_` implements the exact NumPy semantics if you need them. Prefer
explicit `len(a) > 0`, `pecos.any(a)`, or `pecos.all(a)` over `if a:` so the
intent survives either behavior.

## Choosing a boundary

Keep hot loops on PECOS-native operations -- on release builds the native
layer runs at NumPy speed or better for the common operations -- and convert
at the edges:

- **Producing results for analysis or plotting**: finish the computation in
  PECOS, then `np.asarray(result)` (free) or `np.array(result)` (isolated
  copy) at the hand-off point.
- **Consuming external data**: `pecos.asarray(external)` once at the entry
  point, then stay native.
- **Writing PECOS library or example code**: do not import NumPy outside
  tests; the lint ban will remind you.
