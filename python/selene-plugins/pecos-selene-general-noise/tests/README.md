# General-noise conformance tests

This directory tests the PECOS general noise model through its public Selene
plugin boundary. The framework is deliberately device-neutral: circuits use only
standard one- and two-qubit operations, and local crosstalk topology is supplied
as ordinary qubit groups rather than inferred from a hardware layout.

`general_noise_conformance.py` provides three reusable pieces:

- `ExpectedDistribution` describes a small analytic reference distribution.
- `ConformanceExperiment` samples the native plugin with explicit component seeds.
- A Hoeffding bound compares every observed frequency with its reference without
  adding SciPy as a test dependency.

`basis_state_reference.py` is an independent exact oracle for arbitrary circuits
that remain in the computational basis. It propagates a full distribution over
``0``, ``1``, and leaked states and independently specifies preparation, custom
Pauli channels, emission gate replacement, leakage, seepage, and asymmetric
readout. This makes combined-channel tests possible without copying PECOS's Rust
implementation or relying on a device gate set.

`qutrit_reference.py` extends that independent specification to coherent density-
matrix evolution. It embeds standard quantum rotations into a qutrit space and
models computational/leaked projections, non-unitary leakage and reset, emission
gate replacement, seepage, and noisy readout. The associated circuit matrix uses
only `guppylang.std.quantum` operations; it does not import a hardware-specific
gate library or runtime. It covers deep and parallel single-qubit sequences plus
correlated and anti-correlated Bell-state circuits, including the full two-qubit
Pauli channel.

Every behavioral case also supplies a comparison distribution. Before taking
shots, the framework verifies that the circuit is sensitive enough to distinguish
the configured channel from that comparison. This prevents a test from passing
merely because its circuit cannot observe the configured noise.

The current suite covers preparation and asymmetric readout, process and average
gate infidelity, custom Pauli and emission channels, two-qubit angle scaling, all
three idle families, leakage and seepage, preparation crosstalk, global and
topology-defined local measurement crosstalk, family/global/noiseless controls,
combined-channel behavior over several seeds, and both Stim and PECOS StateVec
simulator boundaries.

Keep conformance circuits shallow and use elevated probabilities. These tests are
for correctness of channel semantics, not estimation of realistic device error
rates. New cases should state an analytic distribution and a distinct comparison;
snapshots of one random sequence are not an adequate statistical reference.
