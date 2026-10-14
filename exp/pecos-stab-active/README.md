# PECOS StabActive

Experimental per-shot near-Clifford simulation based on ideas from Clifft
(arXiv:2604.27058) and SymFT (arXiv:2607.28600), implemented from the mathematics
against PECOS types.

A signed Hermitian-Y stabilizer tableau defines a reference state. Ordered active
generator indices and `2^k` complex amplitudes describe its destabilizer orbit.
Only this vector has exponential storage cost; the tableau remains polynomial
in the physical qubit count. Clifford gates leave coordinates unchanged.
Non-Clifford Pauli rotations promote at most one dormant coordinate; measurements
can remove one active coordinate. Independent commuting rotations can reach `k=n`.

`StabActive::new(n)` and `StabActive::with_seed(n, seed)` start at zero active
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

`HeisenbergProgram::compile(&TickCircuit)` validates the complete input, replays
its Cliffords once, and retains only ordered virtual Pauli rotations and
measurements. `program.run(seed)` starts a fresh zero state and returns visible
records, detector and observable XORs, and peak active width. There is no
scheduler, rotation fusion, or reordering. Each shot uses one continuous RNG
stream for joint Pauli-noise alternatives and quantum measurements.

The accepted gates are named Cliffords, exact fixed-axis RX/RY/RZ/RXX/RYY/RZZ
rotations at any angle, Pauli channels, MZ/MX/MPZ/PZ/PX, and I/Idle/TrackedPauliMeta.
Other gates produce errors with their tick, batch index, type, and support.
Quarter-turn multiples take the Clifford path using exact `Angle64` equality.
Pauli channel expressions preserve all positive alternative weights, including
those below the cleanup tolerance of `PauliChannel`. Symbolic Pauli products,
adjoints, tensor products, and channel compositions are supported; non-Pauli
unitaries and channels are rejected.

For `C† P C = phi X^F Z^G`, the virtual Hermitian body is
`H = i^|F intersect G| X^F Z^G`. Its real sign is therefore
`phi i^(-|F intersect G|)`. Each operation carries that constant XOR earlier
noise bits XOR earlier measurement symbols selected by anticommutation.
A reset's hidden measurement is a symbol; its conditional correction contributes
to later signs. Measurements are never flattened into noise dependencies.
Stable measurement IDs are mapped to record ordinals for annotations.

`StabActive::rotate_pauli(angle, factors, negative)` and
`StabActive::measure_pauli(factors, negative)` expose arbitrary Hermitian Pauli
tensors, where factors are distinct `(usize, PauliKindForDecomp)` pairs and
`negative` specifies an overall minus sign. Empty support is the signed identity;
rotation omits its global phase. Existing simulator traits retain their behavior.
