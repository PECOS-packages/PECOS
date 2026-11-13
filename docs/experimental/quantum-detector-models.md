# Quantum detector models

`pecos.experimental.detector_models` represents detector-conditioned quantum
evolution and supplies an exact coherent repetition-code generator. It preserves
complex amplitudes and persistent leakage or vacancy sectors. The API is
experimental and may change.

Use this model when discarding coherence or hidden occupancy changes the
conditional logical prediction. The existing independent-Pauli
`DetectorErrorModel` API continues to serve conventional DEM construction and
decoding. This module has its own representation and serialization; it does not
change the meaning of an existing `.dem` file.

## Source process and compiled instrument

`DetectorProcess` describes an ordered sequence of local channels and measurement
instruments on persistent systems:

| Type | Meaning |
| --- | --- |
| `SystemSpec` | Ordered basis, sector partition, and physical or explicit-memory role |
| `ChannelFactor` | Unobserved completely positive, trace-preserving evolution |
| `InstrumentFactor` | An actual observation with a CP map for each outcome |
| `DetectorDef` | Parity of declared observations, including explicit unavailable outcomes |
| `DetectorProcess` | Systems, operations, detectors, retained records, and optional input isometry |
| `KrausMap` | A trace-nonincreasing CP map, including rectangular input/output maps |
| `DetectorInstrument` | Compiled maps indexed by detector values and retained observations |

`compile_instrument` is a small exact reference backend. Its default limits are
64 Hilbert-space dimensions and 8,192 paths. It embeds local operators into the
full space and enumerates measurement/environment branches; its cost grows
exponentially. Storing a local process does not require this expansion.

For a detector value `d` and retained record `r`, the compiled map is

```text
J[d,r](rho) = sum_a K[d,r,a] rho K[d,r,a]^dagger
```

The output is an unnormalized physical density matrix. Its trace is the branch
probability. The complete instrument preserves trace. An input isometry may
encode a logical input into the physical systems; recovery and logical output
reduction require their own declared conventions.

Coherent alternatives stay inside a Kraus operator. Independent unobserved
environment branches are combined as CP maps, not as amplitudes. Discarding raw
observations similarly combines their backaction incoherently. Tensor products
are most-significant system first, with explicit local target order.

### Leakage, loss, and observation

A system can have basis `("0", "1", "L", "vac")` and sectors
`{"computational": (0, 1), "leakage": (2,), "vacancy": (3,)}`. Channels can
transfer population or preserve allowed coherence across those subspaces.
The same system persists between factors; refill or replacement is an explicit
channel.

Sector membership and Kraus indices do not create observations. Ordinary readout
outside the computational space must be specified by an `InstrumentFactor`.
For example, a declared readout can produce an unavailable result for vacuum,
or a leaked state can produce an ordinary binary result. Neither behavior is
inferred from the sector name. No additional leakage/loss flags are required.

Processes, compiled instruments, and compact repetition models save and load
JSON with explicit complex real/imaginary pairs. This experimental format uses
PECOS arrays and the standard library, with no runtime NumPy or pickle dependency.

## Exact coherent repetition generator

The first specialized generator covers odd-distance repetition memory with:

- Bit protection: `Rx` data rotations, ideal `ZZ` checks, and `Z` readout.
- Phase protection: `Rz` data rotations, ideal `XX` checks, and `X` readout.
- Distances 3 through 15 and 1 through 1,000 syndrome rounds.
- One declared, fixed rotation angle per data site, repeated each round.

The two bases are related by transversal Hadamards. They are separate experiments
protecting one error axis each. They do not provide simultaneous protection
against arbitrary bit and phase noise.

For each adjacent-check syndrome increment, choose the representative error
string `e` with `e[0] = 0`. The exact round map in that virtual reference frame is

```text
K_e = a_e I + b_e X
a_e = product_q (cos(theta_q/2) if e[q] == 0 else -i sin(theta_q/2))
b_e = a_complement(e)
```

At odd distance, `K_e^dagger K_e = q_e I`. There are only `2**(distance-1)` round
outcomes: 4, 16, and 64 at distances 3, 5, and 7. Successive conditional quantum
maps retain logical interference across rounds. The independent-Pauli baseline
instead propagates the incoherent alternatives `abs(a_e)**2` and `abs(b_e)**2`.
In this family, both models have the same detector-history marginal but can have
different conditional logical predictions.

```python
from pecos.experimental.detector_models import (
    DetectorProcess,
    RepetitionConfig,
    RepetitionModel,
    compile_instrument,
)

model = RepetitionModel(RepetitionConfig())  # d=3, R=3, bit protection
checks = [[[0, 0], [0, 0], [0, 0]]]  # [shot][round][adjacent check]
prediction = model.query(checks)
print(prediction["posterior"].tolist())
print(prediction["canonical_posterior"].tolist())

# Choose new paths when saving artifacts.
model.save("coherent_repetition.json")
restored = RepetitionModel.load("coherent_repetition.json")

# Export the full conditional quantum maps, not just a classical posterior.
process = restored.as_detector_process()
process.save("repetition_process.json")
instrument = compile_instrument(DetectorProcess.load("repetition_process.json"))
```

`query` accepts ordinary binary check histories. Optional final data are used
only to verify their parity against the last ideal extraction. They never enter
the logical posterior as a target label. Preparation labels and the target
logical flip are reserved for scoring. A nonzero final-boundary detector is
outside this model's ideal-extraction assumptions and is rejected.

`posterior` and `canonical_posterior` contain the conditional probabilities of
the two logical-flip classes. `log_joint`, `canonical_log_joint`, and
`log_evidence` retain evidence in log space. `detectors` is a round-major vector
of adjacent-parity increments followed by the known-zero terminal boundary.
There is no 64-bit detector-mask restriction.

Logical class zero/one means the first data readout agrees/disagrees with the
initial logical bit, in the specified readout basis. The syndrome representative
has its first bit zero, so this convention needs no physical correction or
feedback. `as_detector_process` retains the quantum logical output in this
reference frame. `canonical_dem()` exports the ordinary Pauli-twirled baseline.

## State-vector example

From an installed PECOS environment with Guppy, Selene, and its QuEST plugin:

```bash
python examples/coherent_repetition_detector_model.py --basis bit --shots 2000 --seed 20261113
python examples/coherent_repetition_detector_model.py --basis phase --shots 2000 --seed 20261113
```

The example builds the model before sampling, runs a small distance-3 memory,
and compares coherent and canonical MAP predictions on the same state-vector
shots. Only ordinary parity measurements and final data readout are collected.
It reports actual counts and uncertainty; a finite sample need not improve at
every setting. The example uses the model's posteriors directly and has no
Frontier dependency.

### Precursor benchmark evidence

The precursor implementation ran 960,000 Guppy/Selene/QuEST state-vector shots
over 48 predeclared settings: both bases, distances 3/5/7, rounds 1/3/5/10, and
angle scales 0.5/1.0. All shots were retained, with 19,823--20,169 per setting.
The full-scale spatial angles interpolate `(0.6, 0.9, 0.94)` radians across the
data sites. These illustrative angles are not calibrated hardware error rates.

At full scale with ten rounds:

| Distance | Bit: canonical / coherent LER | Phase: canonical / coherent LER |
| --- | --- | --- |
| 3 | 56.17% / 18.37% | 56.69% / 19.01% |
| 5 | 30.72% / 16.14% | 31.00% / 15.79% |
| 7 | 15.30% / 11.13% | 15.19% / 11.08% |

All 12 one-round controls produced identical corrections. Every half-scale
paired 95% interval included zero improvement. Bit/phase model predictions are
exactly dual; differences between sampled curves reflect sampling. The
canonical decoder can exceed 50% error under model mismatch; the matched
Bayes-risk bound does not apply to that mismatched rule.

The [48-setting reference counts and intervals](data/coherent-repetition-reference.csv)
are historical prototype evidence, not a fresh run of this PR's numerical port.
The CSV SHA-256 is
`99a5d134bb68b10bf39e645202268f25917b526d749fdcf90d4e66545ea88745`
(LF-normalized copy).
It includes every setting, including null or slightly reversed sample estimates.
Rate intervals are Wilson intervals; paired improvement intervals use two
97.5% Wilson components with a Bonferroni difference construction. These are
per-comparison intervals, with no familywise efficacy claim.

The historical study checked native Frontier posteriors against fixed quantum
models through bounded query-conditioned classical projections. Frontier did
not consume quantum instruments directly. That adapter is a separate integration
and is not included here. It provides no generic coherent-decoder throughput
claim. The tests for this module instead use independent dense-state NumPy
oracles and real state-vector example controls.

## Boundaries and next steps

This is a source representation plus an exact reference compiler and one
specialized generator. It does not automatically compile arbitrary Guppy or
non-Clifford circuits into a scalable decoder model. General contraction,
adaptive schedules, and logical recovery require additional APIs or backends.

The data model represents coherent leakage, stochastic leakage, absorbing loss,
explicit refill, and declared readout behavior. Unit tests exercise those
semantics. The repetition decoding benchmark validates coherent qubit errors;
it does not establish a leakage/loss decoding improvement. Noisy extraction and
hardware calibration are further benchmarks.
