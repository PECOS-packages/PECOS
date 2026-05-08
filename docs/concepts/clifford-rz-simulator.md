# Clifford+RZ Simulator

The `StabVec` simulator represents quantum states as weighted sums of stabilizer
states using the CH-form representation from Bravyi et al. ([arXiv:1808.00128](https://arxiv.org/abs/1808.00128)). This
enables efficient simulation of circuits with many Clifford gates and a moderate
number of RZ rotations.

## CH-form: phase-aware stabilizer states

### The problem with standard tableaux

Standard stabilizer tableaux (Aaronson-Gottesman style) can simulate Clifford circuits
in polynomial time, but they discard global phase. This means you cannot compute inner
products `<phi_1|phi_2>` between two stabilizer states. If you want to represent a
quantum state as a *sum* of stabilizer states, inner products are essential for
normalization, measurement probabilities, and pruning small terms.

### The CH-form factoring

The CH-form represents any stabilizer state as:

```
|psi> = omega * U_C * U_H * |s>
```

Reading right to left:

1. **`|s>` -- computational basis state.** A bitstring. Qubit q is |1> if q is in
   set s, otherwise |0>.

2. **`U_H` -- Hadamard layer.** A tensor product of Hadamard gates on qubits marked
   in set v. Turns |0> into |+> and |1> into |-> on those qubits, creating
   superpositions.

3. **`U_C` -- C-type Clifford.** A Clifford built exclusively from {CX, CZ, S, Z}
   (no Hadamard). Decomposes into three layers:
   - A CNOT network, encoded by binary matrix **F**
   - Per-qubit S/Z phases, encoded by vector **gamma** (values mod 4)
   - A CZ network, encoded by binary matrix **M** (upper-triangular entries = CZ
     pairs, diagonal entries contribute S gates)

   The matrix **G** is the GF(2) inverse-transpose of F, maintained for inner product
   computation.

4. **`omega` -- global phase.** An exact complex scalar, tracked precisely to enable
   inner products.

### Why this factoring works

Gate updates reduce to cheap binary arithmetic on F, G, M, gamma, s, v:

| Gate | Update |
|------|--------|
| CX(a,b) | Row XOR on F and related updates to G, M |
| CZ(a,b) | Toggle a bit in M |
| S(q) | Increment gamma[q] mod 4 |
| Z(q) | Increment gamma[q] by 2 mod 4 |
| H(q) | Most complex -- requires the `update_s_vector` subroutine |

No matrix multiplication, no floating point arithmetic for Clifford gates.

### Inner products

Given two CH-form states, `<phi_1|phi_2>` reduces to a sum over a GF(2) linear system
determined by their F, G, M, v, s values. This requires at most O(n^3) work (Gaussian
elimination over GF(2)), often less. This is the key capability that standard stabilizer
tableaux lack.

## Sum-over-Cliffords decomposition

The `StabVec` simulator represents the full quantum state as:

```
|psi> = sum_k  alpha_k |phi_k>
```

where each `|phi_k>` is a CH-form stabilizer state and `alpha_k` is a complex
coefficient. The number of terms T determines the simulation cost.

### How RZ creates terms

An RZ gate decomposes as:

```
RZ(theta) = cos(theta/2) * I  -  i * sin(theta/2) * Z
```

Both I and Z are Clifford operations, so applying RZ to a stabilizer state produces
two stabilizer states:

```
RZ(theta)|phi> = cos(theta/2)|phi>  -  i * sin(theta/2) * Z|phi>
```

Each RZ gate doubles the term count: T -> 2T. This is the fundamental cost -- the
simulation is polynomial in qubit count but exponential in the number of RZ gates.

### Measurement

Computing measurement probabilities requires the norm `<psi|psi>`:

```
<psi|psi> = sum_{j,k}  alpha_j* alpha_k <phi_j|phi_k>
```

This involves O(T^2) inner products between CH-form pairs. For large T, Monte Carlo
sampling over terms provides an O(T) approximate alternative.

### Cost hierarchy

| Operation | Term count | Work per operation |
|-----------|------------|--------------------|
| Clifford gate | T unchanged | O(T) -- update each term |
| RZ gate | T -> 2T | O(T) -- clone and modify terms |
| Measurement | T unchanged (post-projection) | O(T^2) exact, O(T) Monte Carlo |
| Inner product (two CH-forms) | -- | O(n^3) worst case |

## Implementation optimizations

### Lazy evaluation layer

The simulator maintains a per-qubit lazy state sitting between the gate API and the
expensive term-level CH-form updates:

```
|state> = cliff_frame * pending_rz * |stored_terms>
```

- **Clifford frame:** A per-qubit single-qubit Clifford (one of 24 elements).
  Single-qubit Cliffords compose into this frame in O(1) via lookup table, avoiding
  O(T) updates on every term.

- **Pending RZ:** A per-qubit rotation angle buffer. Consecutive RZ gates on the same
  qubit fuse: `RZ(a) * RZ(b) = RZ(a+b)`. Only one decomposition (one doubling) happens
  instead of two. Uses fixed-point angle arithmetic for exact fusion (e.g., T+T=S,
  8T=I).

The frame and pending RZ are flushed (applied to all terms) only when necessary:

| Incoming gate | Flush behavior |
|---------------|----------------|
| Diagonal Clifford (Z, S, Sdg) | No flush. Commutes with RZ, composes into frame. |
| Anti-diagonal Clifford (X, Y) | No flush. Negates pending RZ (anticommutes with Z), composes into frame. |
| H gate (no pending RZ) | No flush. Composes into frame. |
| H gate (with pending RZ) | Flush frame and pending RZ on that qubit. |
| Two-qubit gate (Pauli frame) | Flush pending RZ on target. Frame propagates through in O(1). |
| Two-qubit gate (non-Pauli frame) | Flush both frame and pending RZ on affected qubits. |
| Measurement | Flush frame and pending RZ on measured qubit. |

### Sparse binary matrices

The Bravyi paper uses dense n x n binary matrices for F, G, M. This implementation
uses sparse binary matrices with a dual row/column representation:

- Each row and column stored as a `BitSet` (bit-vector with hardware popcount)
- Row XOR is O(weight) instead of O(n) -- significant when matrices are sparse
- Dual representation provides O(1) column membership queries alongside fast row
  operations
- Both access patterns are needed: gate updates walk rows, inner products walk columns

### Structural sharing (Arc + copy-on-write)

When RZ splits a term into two, the new terms often share identical F, G, M, v, s
matrices -- only gamma and omega differ. The matrices are wrapped in `Arc`, so both
terms reference the same data. Mutation uses `Arc::make_mut` for copy-on-write
semantics. This substantially reduces memory when T is large.

### Shared constraints

When computing O(T^2) inner products for measurement, if all terms share the same v, F,
and s (common after RZ-only growth), the GF(2) constraint system is identical for every
pair. The system is solved once and reused for all T^2/2 pair evaluations.

### Pruning

Terms with negligible coefficients are dropped:

```
drop if |alpha_k|^2 < threshold * max_j(|alpha_j|^2)
```

Default threshold: 1e-8. This trades exactness for keeping T manageable.

### Monte Carlo measurement

When T exceeds a threshold (default: 2048), measurement uses O(T) term sampling instead
of exact O(T^2) pairwise inner products.

## When to use StabVec

| Scenario                                     | Best approach                                                                  |
|----------------------------------------------|--------------------------------------------------------------------------------|
| Pure Clifford, no branching, pure depolarizing noise | DEM sampling (detector error model -- fastest, but limited to this restricted case) |
| Pure Clifford circuits (general)             | SparseStab (single tableau, O(n^2) worst case, typically less due to sparsity) |
| Few qubits, arbitrary gates                  | StateVec (full 2^n vector)                                                     |
| Many qubits, mostly Clifford, some rotations | StabVec                                                                     |
| Deep circuits with many rotations            | StateVec or cuQuantum (term count explodes)                                    |

For pure Clifford QEC circuits without branching or conditional logic, and with only
pure depolarizing noise, DEM sampling avoids full state simulation
entirely and is significantly faster. However, stabilizer tableaux like SparseStab are
full state simulators that handle the general case -- branching, conditional operations,
arbitrary Clifford circuits, and complex noise models.

The sweet spot for StabVec is large qubit counts with circuits dominated by Clifford
gates and a moderate number of non-Clifford rotations -- typical of error correction
circuits with T gates or variational circuits with few parameterized layers.

## References

- Bravyi, Browne, Calpin, Campbell, Gosset, Howard.
  "Simulation of quantum circuits by low-rank stabilizer decompositions."
  [arXiv:1808.00128](https://arxiv.org/abs/1808.00128) (2018).
  Published in [Quantum 3, 181](https://doi.org/10.22331/q-2019-09-02-181) (2019).
