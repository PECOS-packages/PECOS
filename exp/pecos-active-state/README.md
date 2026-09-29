# PECOS ActiveState

Experimental per-shot near-Clifford simulation based on ideas from Clifft
(arXiv:2604.27058) and SymFT (arXiv:2607.28600), implemented from the mathematics
against PECOS types.

A signed Hermitian-Y stabilizer tableau defines a reference state. Ordered active
generator indices and `2^k` complex amplitudes describe its destabilizer orbit.
Only this vector has exponential storage cost; the tableau remains polynomial
in the physical qubit count. Clifford gates leave coordinates unchanged.
Non-Clifford Pauli rotations promote at most one dormant coordinate; measurements
can remove one active coordinate. Independent commuting rotations can reach `k=n`.

`ActiveState::new(n)` and `ActiveState::with_seed(n, seed)` start at zero active
width. `with_max_active_width(limit)` configures the default limit of 26;
rotations panic before promotion if the requested width exceeds the limit.
`active_width()` and `peak_active_width()` expose width telemetry. `reset()`
restores the zero state and clears both widths, retaining the RNG stream and limit.

Exactness means relative phases are retained, up to one overall global phase
(and floating-point roundoff). This crate deliberately does not implement
`StateVectorSimulator`, whose contract includes global phase. Angles are classified
as Clifford only by exact `Angle64` equality. Forced measurement only forces
nondeterministic outcomes. Batch calls require distinct qubits, as in the simulator
traits. The implementation is scalar and prioritizes correctness over performance.

The representation is

```text
|psi> = sum_x a[x] D_active[0]^x_0 ... D_active[k-1]^x_{k-1} |phi>,
S_j |phi> = |phi>.
```

Bit zero is the least significant amplitude-index bit. Generator indices need
not be contiguous; demotion preserves the order of surviving coordinates.

For a signed physical Pauli, Layer 0 returns
`P = omega product(D_F) product(S_G)` with separate active bit positions and
dormant generator indices. The stabilizer block acts first on a ket. Its active
part supplies input parity; dormant factors supply no amplitude parity but remain
part of every tableau change.

Promotion chooses a dormant `h` in `F`, multiplies each non-pivot `S_j` by
`S_h` when `j` is in `F`, multiplies each non-pivot `D_j` by `S_h` when `j` is
in `G`, and replaces `D_h` with the entire signed `P`. These products retain
both row sign bits. All non-pivot rows now commute with `P`, the canonical
pairing is preserved, and `S_h` acts trivially on every old active basis vector.
Appending a zero bit therefore leaves the state unchanged. Rotation uses
`cos(theta/2) I - i sin(theta/2) P` on paired or diagonal amplitudes.

A random measurement uses the same dormant localization, then exchanges `S_h`
and `D_h` and sets `S_h = (-1)^outcome P`. Its reference state is the normalized
projection of the old reference. Updated active destabilizers intertwine the
projector with the old destabilizers, leaving all active coefficients unchanged.

A deterministic measurement has no flips or active signs; its decomposition
phase gives its eigenvalue. An active measurement computes `(1 - <P>)/2` with
the same routine used by the tests. For compaction, Layer 0 right-composes
coordinate H, S, and CX gates until `P = +/- S_h`; Layer 1 applies their inverses
to amplitudes. Dormant controls carry residual dormant stabilizers into the pivot
without changing amplitudes. Selecting the measured bit, normalizing, and
absorbing a bit-one value into the pivot stabilizer sign removes one coordinate.
This also runs for certain active outcomes; an impossible forced outcome is ignored.

The amplitude-independent API lives in
`pecos_stab_tn::stab_mps::coordinate_tableau`: `decompose`, `promote`,
`measure_random`, `measurement_basis`, and `demote`. `MeasurementBasis` returns
inverse coordinate gates indexed by active bit position, a pivot bit, and the
remaining Pauli sign. The API takes distinct physical Pauli factors plus an
explicit overall sign, and requires destabilizer sign tracking.

Tests reconstruct the state from only the representation, compare 200 circuits
(12,000 gate boundaries) against an independent dense reference, and exercise
all standard measurement/rotation/stabilizer conformance suites. The reference
uses `StateVec` to form physical gate matrices and keeps a plain dense vector
for synchronized forced projections. Layer 0 additionally checks complete dense
matrices at up to five qubits.
