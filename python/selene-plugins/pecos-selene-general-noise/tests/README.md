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

`test_generated_conformance.py` adds a reproducible matrix of 15 mixed-axis
one-qubit circuits and 25 primitive-RZZ two-qubit circuits. It crosses those
circuits with isolated symmetric, asymmetric, emission, leakage, seepage, SPAM,
and combined profiles, and runs every retained pair with three deterministic
component-seed combinations. The generator retains only profile/circuit pairs
whose target and mutation-comparison distributions are statistically
distinguishable. It uses Guppy's public standard rotations and qsystem RZZ
operation without importing a hardware gate library. Testing the primitive RZZ
is important: a high-level CX is decomposed before the error model sees it, so an
atomic-CX oracle would specify the wrong replacement semantics for emission.

Rust adapter-contract tests complement the statistical cases. They verify exact
RXY, RZ, RZZ, reset, and measurement translation; per-qubit runtime ordering;
nanosecond-to-second idle insertion for every idle family; and clear rejection of
invalid batches, qubit indices, custom operations, and abstract crosstalk groups.
These tests are deterministic and remain in the fast lane.

A seeded randomized differential trace also sends the same 97 runtime batches
through the Selene adapter and directly through `GeneralNoiseModel`. It compares
every operation delivered to the simulator and every Boolean or leakage-valued
result while mixing timing gaps, parallel batches, all supported runtime gates,
all three idle families, leakage/seepage, and measurement crosstalk.

`test_qec_workload.py` runs one round of the three-qubit repetition code using
only standard operations. It checks the noiseless syndrome, a known middle-data
fault, analytic noisy-readout syndromes, and agreement across sequential and
parallel Selene workers. A two-round history localizes a known fault between
rounds and checks the full analytic distribution under asymmetric readout noise.
Its larger statistical checks carry the `slow` marker.

`test_idle_matrix.py` independently evaluates the linear, sine-squared, and
coherent idle laws at normalized schedule depths 100, 200, 300, 400, and 500.
The coherent cases run against the PECOS StateVec Selene plugin because arbitrary
rotations are intentionally outside a stabilizer simulator's domain.

`test_layered_matrix.py` adds seeded one- and five-layer workloads on three and
four qubits. Isolated `p1+SPAM`, `p2+SPAM`, `p1+p2`, and full profiles are checked
against the qutrit oracle with three component-seed combinations. As elsewhere,
only profile/workload pairs that can distinguish the target model from the
noiseless comparison are retained.

Device-neutral crosstalk probes exercise three- and four-qubit local topologies
with one victim, multiple victims, and an unaffected spectator. The qubit groups
are supplied directly by the test rather than copied from any device layout. A
second matrix tests global crosstalk after one and five mid-circuit measurements
on both qubit counts, using three deterministic component-seed pairs per circuit.

`test_coverage_contract.py` makes semantic breadth an executable contract. Each
current channel must retain at least three sensitive circuits backed by an
analytic, basis-state, or qutrit oracle. Collection prints a channel-by-channel
count and oracle summary. Seed repetitions do not inflate those circuit counts.

The generated matrix and additional statistical seed repetitions carry the
repository's `slow` marker. The default fast lane retains one seed for every
qutrit circuit family. Run the two layers explicitly with:

```console
uv run pytest python/selene-plugins/pecos-selene-general-noise/tests -m "not slow"
uv run pytest python/selene-plugins/pecos-selene-general-noise/tests -m slow
```

The extended matrix also runs weekly and on demand in
`.github/workflows/selene-general-noise-semantics.yml`; the workflow publishes
the channel summary and complete pytest log as CI artifacts. Pull requests keep
the deterministic and representative statistical fast lane on all supported
platforms.

Every behavioral case also supplies a comparison distribution. Before taking
shots, the framework verifies that the circuit is sensitive enough to distinguish
the configured channel from that comparison. This prevents a test from passing
merely because its circuit cannot observe the configured noise.

The current suite covers preparation and asymmetric readout, process and average
gate infidelity, custom Pauli and emission channels, two-qubit angle scaling, all
three idle families, leakage and seepage, preparation crosstalk, global and
topology-defined local measurement crosstalk through four qubits,
family/global/noiseless controls, combined-channel behavior over several seeds,
and both Stim and PECOS StateVec simulator boundaries.

The in-test qutrit density matrix remains an independent oracle rather than a
Selene simulator plugin. End-to-end outcomes `0`, `1`, and leaked `2` are covered
at the Rust adapter boundary today. Moving those tests onto native qutrit
state-vector and density-matrix simulator plugins is tracked by
[PECOS issue #585](https://github.com/PECOS-packages/PECOS/issues/585).

`MUTATION_AUDIT.md` records a representative defect-injection audit and the
specific test that rejected each mutation. It also documents the focused
one-qubit emission-replacement test added after that mutation initially survived
the broader statistical matrix.

Keep conformance circuits shallow and use elevated probabilities. These tests are
for correctness of channel semantics, not estimation of realistic device error
rates. New cases should state an analytic distribution and a distinct comparison;
snapshots of one random sequence are not an adequate statistical reference.
