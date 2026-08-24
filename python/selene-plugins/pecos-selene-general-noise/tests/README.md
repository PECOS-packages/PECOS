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

`test_generated_conformance.py` adds a reproducible matrix of mixed-axis and
parity-sensitive entangling circuits. The generator retains only circuits whose
ideal and noisy distributions are statistically distinguishable, and compiles
them entirely from standard `rx`, `ry`, `rz`, and `cx` operations. Direct Rust
adapter tests separately exercise Selene's leakage-valued measurement operation,
so that path does not require a hardware gate library.

Rust adapter-contract tests complement the statistical cases. They verify exact
RXY, RZ, RZZ, reset, and measurement translation; per-qubit runtime ordering;
nanosecond-to-second idle insertion for every idle family; and clear rejection of
invalid batches, qubit indices, custom operations, and abstract crosstalk groups.
These tests are deterministic and remain in the fast lane.

`test_qec_workload.py` runs one round of the three-qubit repetition code using
only standard operations. It checks the noiseless syndrome, a known middle-data
fault, analytic noisy-readout syndromes, and agreement across sequential and
parallel Selene workers. Its larger statistical checks carry the `slow` marker.

The generated matrix and additional statistical seed repetitions carry the
repository's `slow` marker. The default fast lane retains one seed for every
qutrit circuit family. Run the two layers explicitly with:

```console
uv run pytest python/selene-plugins/pecos-selene-general-noise/tests -m "not slow"
uv run pytest python/selene-plugins/pecos-selene-general-noise/tests -m slow
```

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

The in-test qutrit density matrix remains an independent oracle rather than a
Selene simulator plugin. End-to-end outcomes `0`, `1`, and leaked `2` are covered
at the Rust adapter boundary today. Moving those tests onto native qutrit
state-vector and density-matrix simulator plugins is tracked by
[PECOS issue #585](https://github.com/PECOS-packages/PECOS/issues/585).

Keep conformance circuits shallow and use elevated probabilities. These tests are
for correctness of channel semantics, not estimation of realistic device error
rates. New cases should state an analytic distribution and a distinct comparison;
snapshots of one random sequence are not an adequate statistical reference.
