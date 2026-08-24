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

Every behavioral case also supplies a comparison distribution. Before taking
shots, the framework verifies that the circuit is sensitive enough to distinguish
the configured channel from that comparison. This prevents a test from passing
merely because its circuit cannot observe the configured noise.

The current suite covers preparation and asymmetric readout, process and average
gate infidelity, custom Pauli and non-leaking emission channels, two-qubit angle
scaling, all three idle families, leakage and seepage, global and topology-defined
local measurement crosstalk, scaling/noiseless controls, seed reproducibility, and
both Stim and PECOS StateVec simulator boundaries.

Keep conformance circuits shallow and use elevated probabilities. These tests are
for correctness of channel semantics, not estimation of realistic device error
rates. New cases should state an analytic distribution and a distinct comparison;
snapshots of one random sequence are not an adequate statistical reference.
