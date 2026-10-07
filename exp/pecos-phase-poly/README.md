# pecos-phase-poly

An experimental exact simulator for third-order phase-polynomial (3PP) states,
following arXiv:2610.06811, *Polynomial-time simulation of non-Clifford quantum
error correction*, Sections 2–3 and the appendices on X measurements, efficient
simulation, and phase polynomials.

A fixed number of physical qubits is represented as

```text
|psi> = 2^(-r/2) sum_{y in F2^r} exp(i*pi*p(y)/4) |x0 xor B y>.
```

The binary matrix B has full column rank. The phase is a sparse, ordered sum of
nonzero parities with nonzero coefficients modulo eight. All masks have the same
fixed word count, independent of the current support dimension. Constants in the
phase polynomial are discarded. `amplitude` and `state_vector` return the stored
phase convention; comparisons with another simulation should ignore global phase.

`PhasePoly` supports X, Y, Z, S (`sz`), S dagger (`szdg`), T, T dagger, CX, CZ,
controlled S and its inverse, and CCZ. `pz` and `px` reset qubits by sampling Z and
preparing zero or plus. Gate batches have distinct qubits across the whole batch;
CCZ takes `&[(QubitId, QubitId, QubitId)]`. The qubit count never changes.
`QuantumSimulator::reset` restores all-zero while retaining the RNG stream.
There is no Hadamard, `CliffordGateable`, or `StateVectorSimulator` implementation.

All Z-string measurements are supported. X-string measurements are supported
exactly when both nonzero branches remain 3PP (Theorem 3.2):

- Case 1 enlarges the affine support.
- Case 2a restricts it by an affine equation, or returns a deterministic result.
- Case 2b changes the phase using a quadratic Boolean function of rank at most two.
- Case 2c changes the phase using a parity and has irrational outcome probabilities.

Other X measurements return an error describing the failed condition and leave
the state and RNG untouched. X methods accept a single qubit or a single string,
not batches. False/true outcomes mean eigenvalues +1/-1. Empty strings measure
identity; the X classification is case 2a. Forced variants override only random
outcomes and consume no randomness. `mz_forced(usize, bool)` is supplied through
`pecos_simulators::ForcedMeasurement`.

`z_probabilities` and `x_probabilities` are non-mutating and return probabilities
in `[Pr(0), Pr(1)]` order. The latter also returns `XMeasurementCase` or an
incompatibility error. Probability-1/2 outcomes use a fair random bit. Case-2c
outcomes use a uniform `f64` comparison with the outcome probability.

`amplitude(&[bool])` accepts one bit per physical qubit. `state_vector()` uses bit
q of a basis index for qubit q, matching `pecos_simulators::StateVec`. The vector
query is exponential and intended for small-system tests. `expectation` accepts
`pecos_core::PauliString` with real sign, uses the Hermitian convention `Y = i X Z`,
and rejects imaginary overall signs. It evaluates the quadratic Gauss sum by
variable elimination, retaining an exact root of unity and power of sqrt(2), or
zero, until conversion to `f64`. It never enumerates support assignments.

For n qubits and K stored parities, representation storage is O(n² + Kn) bits.
A restriction costs O((n+K)n + Kn log(K+1)) bit operations. Derivative construction,
compatibility checks, and expectation queries take O(Kn² + n³) arithmetic
operations. Each gate/measurement adds at most a constant number of parity terms,
so K is O(m) after m circuit operations. Runtime and memory are polynomial in
circuit size; sparse storage is not asserted to be bounded by O(n³) independently
of circuit length. Floating-point query results have ordinary rounding and
underflow limitations; internal phase and support updates are exact.

Tests use independent gate matrices from `StateVec`, dense projectors, a dense
3PP-membership oracle based on affine support and a Boolean Moebius transform,
random signed Pauli expectations, brute-force quadratic sums, and a 256-qubit
smoke circuit.

## Triorthogonal distillation

The public `distillation` module implements the circuit and finite-sum oracles
of Bravyi and Haah, arXiv:1209.2426, Sections “Triorthogonal matrices”,
“Distillation subroutine”, and “A family of triorthogonal matrices”.
`TriorthogonalMatrix::new(&rows, k)` validates a rectangular binary
matrix, odd/even row order, pair/triple overlap parity, and row independence.
The first `k` rows are logical rows; the remaining rows are syndrome rows.
Zero logical rows or zero syndrome rows are supported; an empty row list
represents the 0×0 matrix. Constructors are
`TriorthogonalMatrix::bravyi_haah(k)` for even `k >= 2`, and
`TriorthogonalMatrix::rm15()` for the 15-to-1 matrix. `n()`, `m()`, `k()`, and
`rows()` expose dimensions and the original matrix.

`matrix.circuit(p)` returns public `Op` values: PZ, PX, CX, CZ, S, Sdg, Z, T,
ZError, MeasureX, and ExpectW. Gaussian elimination tracks `R = A G`, so each
original coefficient is a parity of physical pivot bits. The encoder prepares
`|G>`, physical T gates are each followed by a Z error, and the diagonal Clifford
correction cancels the unwanted linear and quadratic phases. The weight
inclusion-exclusion expansion is a congruence modulo eight, not an integer
identity truncated at degree three. Fourth and higher terms vanish modulo eight;
triorthogonality also removes the cubic term. S powers and CZ parities are
accumulated before emission. Syndrome measurements use the original even rows.
Logical W queries use `(Xbar + Ybar)/sqrt(2)`, including the sign
`Ybar = i (-i)^weight Y(row)`.

`run_shot(&mut sim, &ops, &mut errors)` resets and runs a complete circuit.
The callback `errors(op_index, qubit_index, p) -> bool` decides every Z error.
`run_shot_sampled(&mut sim, &ops, &mut rng)` samples with a caller-owned
`PecosRng`; its error RNG is separate from the simulator's measurement RNG.
`ShotResult` contains syndrome `MeasurementResult`s (including determinism),
acceptance, and optional logical W expectations. Rejected shots have no output
expectations. Generated circuits always give deterministic syndromes for a
specified physical error pattern. Unsupported measurements in manually edited
operation streams propagate `IncompatibleMeasurement`.

`matrix.weight_enumerators()` returns `WeightEnumerators { even, cosets }`,
integer coefficient vectors indexed by weight. Each `cosets[a]` enumerates
only `G0 + f^a`; adding `even` gives the paper's `G0 + {0,f^a}` enumerator.
Enumeration takes exponential time in the number of even rows and polynomial
memory. Counts use checked `u64` arithmetic and report overflow.
`weights.oracles(p)` evaluates the paper's full `P_s` and `qdual` expressions:

```text
x = 1 - 2p
P_s = W_even(x) / W_even(1)
q_a = (1 - W_coset_a(x) / W_even(x)) / 2
```

These are finite-sum formulas, with ordinary floating-point rounding at
numerical evaluation (including cancellation near p=0). `pattern_oracle(&bits)`
returns the syndrome from even-row dot products and conditional logical signs
from odd-row dot products. It is independent of the circuit and simulator.

Run the example in release mode:

```sh
cargo run -p pecos-phase-poly --release --example distillation -- scaling 20 100 0.05 2026 16
cargo run -p pecos-phase-poly --release --example distillation -- emit 4 0.05 20000 2026 /tmp/distillation-k4
cargo run -p pecos-phase-poly --release --example distillation -- emit rm15 0.05 20000 2027 /tmp/distillation-rm15
```

Scaling arguments are maximum even k, shots, p, seed, and optional active-width
limit (default 16). Timings include reset, all preparations/gates/noise, and
syndrome measurements; logical expectation queries are excluded from both
simulators because `StabActive` does not expose an expectation query. The same
error RNG seed is used, and complete syndrome streams are compared. Each X
string in `StabActive` uses CX from its first qubit to every other qubit, MX on
the first, then the same CX gates. The peak active width is `m = k+3`: physical
Z operators are parities of m free encoder coordinates, and transversal T
includes every pivot. Subsequent Clifford gates and measurements cannot
increase the width. The example configures `with_max_active_width(limit)` and
skips circuits with `m > limit` before execution, without catching panics.
Initialization is outside the timer; each shot's reset is inside it.

Emit arguments are even k (or `rm15`), p, shots, seed, and output prefix. It
writes `<prefix>.stim` and `<prefix>.json`. JSON contains k, n, p, shots, seed,
accepted, acceptance_rate, per-logical mean_w, P_s, and q_a. If no shots are
accepted, each mean_w entry is null. Output expectations are computed on every
accepted shot, without collapsing the logical state.

`to_stim(&ops)` writes R/RX, CX/CZ, S/S_DAG/Z, T, Z_ERROR, and X-string MPP.
A DETECTOR rec[-1] immediately follows each syndrome measurement. This is a
Stim dialect: T denotes diag(1, exp(i*pi/4)); `EXP_VAL P Q` is an extension that
reports the expectation of each listed Pauli product separately, without measuring
or collapsing the state and without adding a measurement record. Each logical
output gets one `EXP_VAL X(f) Y(f)` line; its W check is `(<X(f)> + <Y(f)>)/sqrt(2)`,
meaningful on shots with all detectors zero. Products use `*`; a leading `!` on
their first factor negates the entire product. These semantics are also documented
in the emitted file's comments.

Tests transcribe the paper's 5×14 example, check exact enumerators and integer
series coefficients, compare small ideal circuits with `StateVec` and independent
target amplitudes, exhaust errors of weight at most two, and sample 300 distinct
patterns at each weight three through five per matrix. Statistical tests use
20,000 shots per case and a two-sided Bernoulli Bernstein bound, conditional on
the accepted count for output errors. A union bound over all 18 comparisons
limits the joint failure probability to 1.8e-7. The 128-qubit G(40) test runs
three ideal shots and prints elapsed time with `-- --nocapture`.
