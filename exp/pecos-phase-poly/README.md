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
