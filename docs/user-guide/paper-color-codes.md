# Paper color-code constructions

PECOS provides executable X/Z memories for selected constructions from
Noah Shutty, [*Denser Planar Color Codes* (arXiv:2609.21376v3)](https://arxiv.org/abs/2609.21376v3),
and Ali Lavasani et al., [*Quantum error correction well below the color code
threshold* (arXiv:2610.24077v1)](https://arxiv.org/abs/2610.24077v1).

Code geometry and checks, syndrome-extraction implementation, memory assembly,
scheduling, and backend export are separate objects. These static Python
construction adapters provide a working path to Guppy and Stim; they do not
implement the proposed general Rust instruction graph or replace the existing
surface-code, SLR, or color-code factories.

The paper memories now compose through the [named gadget authoring
interface](named-qec-gadgets.md). Use `color_memory_program(...)` to inspect
preparation, extraction, readout, and named detector expressions before
selecting a schedule with `program.compile(...)`. The `color_memory(...)`
convenience API compiles this same path.

## Build and run a memory

```python
from pecos import sim, stabilizer
from pecos.qec.color import (
    ColorExtraction,
    color_code_spec,
    color_code_implementation,
    color_memory,
)
from pecos.qec.generic import SchedulePolicy

code = color_code_spec("666", distance=3)
implementation = color_code_implementation(code, ColorExtraction.SWAP_SUPERDENSE)
plan = color_memory(
    implementation, rounds=3, basis="Z", start_phase=0,
    schedule=SchedulePolicy.REFERENCE,
)
program = plan.guppy_program()
columns = sim(program).qubits(plan.num_qubits).quantum(stabilizer()).seed(42).run(32).to_dict()
for shot in range(32):
    detectors, observables = plan.evaluate_named({tag: values[shot] for tag, values in columns.items()})
    assert not any(detectors)
    assert not any(observables)

source = plan.guppy_source()  # Includes the construction's paper citation.
stim_circuit = plan.to_stim()  # Requires the optional Stim dependency.
ticks = plan.to_tick_circuit()  # Native PECOS circuit with explicit moments.
```

Each `qec:m:<ordinal>` output is a raw scalar measurement. Detector and logical
parities are evaluated separately; a raw syndrome need not be zero. Use event
names when reading Guppy results because runtime output order is not the
canonical measurement order. The memory builder carries measurement-induced
Pauli frames through the parities without inserting physical corrections.

## Select a construction

| Family | Extraction choice | Distances accepted | Recurring period |
| --- | --- | --- | --- |
| Planar 4.8.8 | `ColorExtraction.BRICKWORK_CX` | 3, 7, 11, … | 2 phases |
| Planar 4.8.8 | `ColorExtraction.BRICKWORK_CZ` | 3, 7, 11, … | 2 phases |
| Triangular 6.6.6 | `ColorExtraction.SUPERDENSE` | Odd distances ≥ 3 | 1 phase |
| Triangular 6.6.6 | `ColorExtraction.SWAP_SUPERDENSE` | Odd distances ≥ 3 | 3 phases |

Incompatible combinations raise an error. The 4.8.8 geometry uses Shutty's
brickwork construction and indexing, distinct from the existing
`ColorCode488` geometry. Its CX implementation has twelve CX layers per cycle.
The native-CZ implementation uses the boundary Bell-pair polarity optimization
and alternating data Hadamard frames from Appendix E.2; its extraction cycle
has 23 moments at distance 3 and 24 at distance 7 and above. Memory preparation
and final readout add their own moments. Input and output checks can have
different parities in the raw records. The implementation keeps both maps,
logical signs, and record constraints.

The 6.6.6 memory first projects with the Bell-flag circuit from Supplement B
of Lavasani et al. Its flags remain available as raw events and detectors.
Recurring extraction uses either superdense or SWAP-superdense cycles. The
latter rotates its orientation through three phases and has seven CX layers
per cycle. Its data/ancilla role exchange is represented by `RELABEL`, a
zero-gate change of carrier bindings. It does **not** add physical SWAP gates.
`plan.output_carriers` records the final reference-carrier permutation.
Across three phases the bulk, boundary, and corner data roles are refreshed
three, two, and one times respectively, as in the paper's circuit construction.

`start_phase` selects the first recurring phase of a fresh memory; it does not
resume a previous experiment's carrier mapping or Pauli frame.

## Combining the superdense constructions

The 6.6.6 and Shutty 4.8.8 implementations belong under the same extraction
interface. Shutty already uses superdense extraction: both Pauli check types are
acquired in a cycle and measurement-induced signs are tracked classically.
The physical arrangement differs: 6.6.6 assigns an ancilla pair per face, while
Shutty shares ancillas between square and octagon checks. See
[Shutty, Section 2.2](https://arxiv.org/html/2609.21376v3#S2.SS2).

The reusable description consists of named check/logical supports, separate
input and output syndrome parities, logical-sign updates, intrinsic record
constraints, and boundary frame/carrier bindings. `PauliFlow` and the gadget
audit express these without depending on face size. Both families then use
the same named memory composition, scheduling policies, and Guppy/TickCircuit
lowering. The periodic adapter extends this to four logical pairs and dependent
global face checks. This is the implemented combination at the representation
and verification levels; the published physical recipes remain explicit choices.

For a future physical combination, the published 6.6.6 SWAP-superdense work
supplies a concrete starting point: a local CNOT identity and a three-ordering
data-refresh schedule. Its software relabeling relies on the hardware's
connectivity, and its boundary/corner refresh pattern follows its own face
geometry. See [Lavasani et al., Fig. 1 and Supplement B](https://arxiv.org/html/2610.24077v1).
Transferring that approach to 4.8.8 would require identifying compatible local
interactions in Shutty's shared-ancilla circuit, deriving the resulting carrier
and signed-check maps, and checking subsequent couplers and preparation/readout
boundaries. Passing ideal flows would then be followed by physical fault-distance
and appropriate noise/leakage studies. Improved performance does not follow from
the gate identity alone.

Such a 4.8.8 refresh hybrid is a research proposal, not an implemented published
construction or an option silently enabled by rescheduling. Current additions
use public papers and their released circuits; unpublished local color-code
improvements are outside this implementation scope.

## Choose a schedule independently

`SchedulePolicy.REFERENCE` retains the construction's moments.
`SchedulePolicy.ASAP_UNITARIES` packs unitary regions as soon as their per-wire
dependencies permit, while keeping reset, measurement, and relabel moments as
boundaries. For example:

```python
packed = plan.rescheduled(SchedulePolicy.ASAP_UNITARIES)
assert packed.detectors == plan.detectors
assert packed.output_carriers == plan.output_carriers
```

This preserves event identities and ideal behavior, but changes idle locations.
Build the memory again to recover its reference schedule.

Stim export implements the selected moments with `TICK`s. Default Guppy export is a
**functional circuit contract**: source comments identify moments, but the
compiler/runtime may reschedule operations and choose physical sites. These
comments are not timing barriers. A Guppy noise experiment must therefore use
the actual lowered trace. The typed event specifications integrate with the
existing audited DEM builder:

```python
from pecos.qec import build_dem_from_guppy

detectors, observables = plan.dem_spec()
build = build_dem_from_guppy(
    program, num_qubits=plan.num_qubits,
    detectors=detectors, observables=observables,
    p1=0.001, p2=0.002, p_meas=0.003, p_prep=0.004,
)
dem_text = build.dem.to_string()
```

These example probabilities define a simple Pauli noise experiment, not either
paper's benchmark noise model. See [the Guppy DEM workflow](../workflows/guppy-dem-decoding.md)
for decoding and trace-based noise conventions.

## Run the comparison example

From the repository root:

```sh
uv run python examples/python_examples/paper_color_memories.py --output /tmp/paper-color-memories --dem
uv run python examples/python_examples/paper_color_memories.py --basis X --schedule asap_unitaries
```

The example executes all four implementations through Guppy and PECOS, checks
their ideal detector/logical parities, and optionally writes Guppy source,
Stim circuits, and DEMs from the actual Guppy trace.

## Verification, provenance, and scope

Tests compare 16 planar 4.8.8 circuits with fixtures generated by the
unmodified [author source release](https://doi.org/10.5281/zenodo.22820614):
distances 3/7, both bases, both phases, and CX/native-CZ. Selected Apache-2.0
source files are vendored unchanged under `pecos/qec/color/_shutty488`, with
LICENSE, NOTICE, original paths, and hashes. PECOS supplies memory composition,
the concrete circuit representation, and Guppy/`TickCircuit` export.

The 6.6.6 geometry has a separate migration fixture from the existing local
color-code experiments. Its extraction and Bell-flag instruments are tested
against independently constructed ideal Pauli projections on entangled inputs,
including conditional measurement branches and carrier permutations. These
checks cover extraction at distances 3/5/7 and Bell-flag branches at distance 3;
memory tests cover phases, round boundaries, both bases, and both schedules.
The local candidate code is not represented as an author-distributed release.

The test environment uses Stim 1.15.0; Shutty's release pins 1.16.0. The fixtures
are reconstruction comparisons, not a claim of byte-identical serialization
of the published artifacts. Accepted distances beyond those tested do not
carry a separate distance or fault-tolerance certification.

The original planar recipes cover static X/Z memories. The separate periodic
4.8.8 CNOT factory is described below. These recipes do not reproduce randomized benchmarking, Helios leakage/repump behavior, hardware timing,
or either paper's reported
logical-error curves. Ideal instrument checks and noiseless execution do not
establish circuit distance or a threshold. Decoder geometry annotations,
schedule-conforming Guppy timing, and matched physical noise/decoder benchmarks
remain separate work.

The 6.6.6 family additionally supports all six signed logical Pauli states and
static transversal Clifford experiments through `color_state_memory_program`
and `ColorCliffordExperiment`. See the [six-state and Clifford guide](color-code-cliffords.md).
These use Bell-flag zero preparation, logical rotations, phase-aware extraction,
and inverse-preparation readout. Existing X/Z reference recipes are preserved.

## Lacroix superconducting memory construction

Lacroix et al., [*Scaling and logic in the color code on a superconducting
quantum processor*, arXiv:2412.14256v1](https://arxiv.org/abs/2412.14256v1),
Supplement C.3/D.1, gives a different superdense implementation adapted to
readout lines. Its ten-CZ-layer extraction uses separate first, recurring,
and final gadgets, with preparation/readout rotations merged into the X-memory
boundary rounds. It is available separately from `ColorExtraction.SUPERDENSE`:

```python
from pecos.qec.color import lacroix_memory_program

program = lacroix_memory_program(patch="d3_west", rounds=3, basis="X")
print(program.describe())
plan = program.compile()
function = plan.guppy_program()
ticks = plan.to_tick_circuit()
stim_circuit = plan.to_stim()
```

Supported released patches are `d3_west`, `d3_east`, `d3_north`, and `d5`, with
13 or 37 active qubits, respectively; bases are X/Z and rounds must be at least
two. Named roles retain the authors' qubit IDs and square-grid coordinates.
These hardware patch tables do not identify their indexing with the canonical
triangular 666 geometry used by `ColorCliffordExperiment`.

The construction is extracted from the authors' [released ideal memory
circuits](https://doi.org/10.5281/zenodo.14238944). Sixteen unchanged reference
files check every per-wire gate sequence, measurement ordinal, detector parity,
and accumulated logical parity for three and five rounds. Other lengths repeat
the same recurring gadget; two, four, and seven rounds are also checked for
ideal detector determinism under both schedule policies. Tests execute both
bases and all four patches through Guppy and inspect native TickCircuit output.

This is a **normalized ideal construction**: all sweep bits are zero, implicit
initial states become explicit preparations, spectator-only resets and repeated
fresh-zero resets are omitted, empty timing slots are removed, and final
readout is a separate gadget. These changes affect noise locations and timing.
`REFERENCE` preserves the normalized gadget moments, not the device schedule
or the original SI1000 circuit. Hardware readout, leakage reduction, noise,
and decoder benchmarking remain separate work. The implementation does not yet
include this paper's randomized-benchmarking experiment, magic-state injection,
postselection, or lattice-surgery teleportation.

Run `examples/python_examples/lacroix_color_memories.py` to export and execute
both lower-level forms. The coverage record in `design/qec-guppy-lowering-assessment.md` distinguishes
implemented circuits from remaining paper and benchmark work.

## Periodic 4.8.8 memory with four logical qubits

The periodic construction in [Shutty's paper](https://arxiv.org/abs/2609.21376v3)
uses a different topology and logical basis from the planar code. It has four
logical qubits, d² data qubits, and d² ancillas for distances 4, 8, 12, … .
Its released mixed-polarity extraction has twelve CX layers and one recurring
phase. Global color dependencies make some face checks redundant; the memory
also retains independent intrinsic record constraints from the author builder.

```python
from pecos.qec.color import (
    periodic_488_code, periodic_488_implementation, periodic_488_memory_program,
)

code = periodic_488_code(distance=4)
implementation = periodic_488_implementation(code)
print(code.describe())                    # Includes four paired X/Z logicals.
print(implementation.gadget().describe())  # Named input/output and logical flows.
experiment = periodic_488_memory_program(implementation, rounds=3, basis="Z")
plan = experiment.compile()
function = plan.guppy_program()
ticks = plan.to_tick_circuit()
assert len(plan.observables) == 4
```

`Periodic488CodeSpec` exposes named checks, the author's canonical logical
supports, coordinates, and period vectors. `Periodic488Implementation` adds
ancilla positions, allowed couplers, and the extraction gadget. Recurring
memory calls reuse that named gadget. Boundary variants retain the author's
preparation/readout overlap with disjoint ancilla operations, including a
single-round memory. This API is separate from the single-logical-qubit planar
factories and does not add torus support to `ColorCliffordExperiment`.

Twelve independent author-generator fixtures cover d=4/8/12, X/Z, and one/three
rounds. Tests compare exact physical moments, measurement order, detector
parities, and all four logical parities. Independent checks verify CSS ranks and
logical commutation, signed Choi correlations at d=4/8/12, every named signed
flow at d=4/8/12/16, and completeness of the deterministic parity span. Guppy
execution covers d=4/8, both bases, and both scheduling policies. Both native
and Guppy-trace DEM paths retain all four observable IDs.

Choose the released 25-moment native-CZ construction explicitly:

```python
implementation = periodic_488_implementation(code, gate_set="native-cz")
experiment = periodic_488_memory_program(implementation, rounds=3, basis="Z")
plan = experiment.compile()
assert len(plan.moments) == 25 * 3
```

This preserves the native schedule, including overlapped memory boundaries,
and the stationary data Hadamard frame. Its `C_XYZ`, `C_ZYX`, and `SQRT_X`
primitives lower to native TickCircuit `F`, `Fdg`, and `SX`. Guppy uses the
corresponding Clifford implementations; its runtime scheduling remains functional.
Signed input/output/logical flows determine the detector expression constants
without running a reference simulator. These constants are visible as XOR-one
terms in named expressions and survive raw-result evaluation through both backends.

Twelve additional native-CZ author-reference circuits check exact physical
moments, raw record order, detector/logical supports, and independently sampled
reference signs at d=4/8/12. Signed framed-flow checks extend through d=16.
Tests cover one-round boundaries, alternate scheduling, actual Guppy and native
execution, and both DEM paths. Use the [signed-parity evaluation methods](named-qec-gadgets.md#xor-with-a-constant)
when consuming raw results, especially Stim's synthetic constant record.

Both periodic constructions preserve the released generator's detector coordinates
through named declarations, both schedules, Guppy DEM specifications, native
TickCircuit, and Stim/DEM exports. The CNOT generator retains four axes, including
the color/basis label. The public native-CZ generator uses spatial `(x, y, t)`:
it removes the label axis and replaces placeholder positions with centroids of
the measured support. PECOS applies that same final conversion without a runtime
Stim dependency. Coordinates do not promise Guppy runtime timing.

Decoder error-rate benchmark reproduction, larger-instance lower-bound
certification, and other periodic families remain separate work. Neither
construction introduces a speculative SWAP-style 4.8.8 data-refresh variant.

Run the inspectable example with:

```sh
uv run python examples/python_examples/periodic_color_memories.py --distance 4 --rounds 3 --dem
uv run python examples/python_examples/periodic_color_memories.py --gate-set native-cz --output /tmp/periodic-native-cz --dem
```

It writes named code/gadget/schedule descriptions, Guppy and Stim sources, native
measurement definitions, and optional example-noise DEMs, then checks all four
logical parities through Guppy execution.

## Matched SI1000 and physical fault bounds

The shared `SI1000(p).apply(plan)` layer now reproduces the noise convention in
[Shutty's simulation methods and source release](https://doi.org/10.5281/zenodo.22820614).
It preserves the selected native primitives and moments, including overlapping
boundary operations. Each idle channel and additional reset/measurement wait
channel is inspectable. See [schedule noise](named-qec-gadgets.md#noise-attached-to-a-physical-schedule)
for the rates and supported backend.

Eighteen unchanged published noisy circuits verify every per-wire gate/noise
sequence within each physical moment and every undecomposed DEM support and
probability. They cover planar d3, both bases/phases/rates; periodic d4/d8, both
bases/rates; and periodic d12, both bases at p=.0005. Periodic and planar decoder coordinates now match the release. Planar native-CZ
face detectors use data-face centroids and constraint detectors use measured
physical-support centroids; CNOT retains the original index/color coordinates.

All 18 released witnesses replay as undetected logical faults at distinct
physical noise locations. The 12 small instances also exhaustively exclude
lower-weight faults: planar d3 excludes weights one/two, periodic d4 excludes
one/two/three. Their lower and upper bounds therefore coincide. The d8/d12
checks establish only upper bounds; they do not reproduce the release's larger
lower-bound calculations or prove distance preservation for the entire family.

To compare explicit SI1000 schedules after repacking, run:

```sh
uv run python examples/python_examples/compare_si1000_schedules.py \
  --case torus_488_d4_r4_Z_phase0_p0p001 --output output/si1000-schedule-comparison
```

This transfers the published operation-attached witness to ASAP and ordered
compaction, verifies finite-memory bounds, and compares independent DEMs at the
selected rate. It independently checks Stim and noisy TickCircuit exports.
These are scheduled native-CZ noise circuits; Guppy runtime noise equivalence
and logical-error-rate performance remain separate questions. See
[fault comparison semantics](named-qec-gadgets.md#compare-si1000-fault-behavior-after-rescheduling).

Run from a checkout to inspect the schedule, channels, witness, and bounds:

```sh
uv run python examples/python_examples/si1000_color_memories.py
uv run python examples/python_examples/si1000_color_memories.py \
    --case planar_488_d3_r3_X_phase0_p0p001 --exclude-through 2 \
    --output output/si1000-planar-d3
uv run python examples/python_examples/si1000_color_memories.py \
    --case torus_488_d8_r8_Z_phase0_p0p001 --exclude-through 0 \
    --output output/si1000-periodic-d8
```

The default case is periodic d4/r4/Z at p=.001 and reports bounds `[4, 4]`.
Artifacts include the noisy Stim circuit, ideal Guppy source and native definitions,
the selected schedule, every noise channel, and a JSON result with source hashes.
The example also verifies the witness through native PECOS propagation and native
stabilizer replay, and saves a full-precision native DEM plus 64 noisy trajectories
(`--native-shots` and `--seed` control sampling). Native DEM mechanisms match all
18 published references. The example also runs 64 shots through a standalone
noisy TickCircuit using native inline channels and saves its samples and physical
moment/execution-step map. Its DEM is built directly from the inline TickCircuit
with `DetectorErrorModel.from_circuit()` and cross-checked against the scheduled
native adapter. `memory.inline.stim` exports the noisy TickCircuit through the
generic compatibility exporter, with execution-step TICKs; `memory.si1000.stim`
retains physical-moment TICKs from the schedule adapter. Guppy exports and explicitly requested
`noisy.ideal.to_tick_circuit()` exports remain ideal. This is a finite-instance physical-fault check, not a decoder benchmark or
a published logical-error-rate reproduction. No speculative refresh schedule is
introduced by this baseline.

Add `--audit-guppy` to save a compiler-bound measurement audit and source
allocation summary. Its
runtime circuit is Clifford-normalized for DEM analysis, so its gate counts and
replay ticks are not hardware pulses or physical schedule durations. See the
[Guppy audit contract](named-qec-gadgets.md) before interpreting these counters.

Add `--guppy-ordering unitary_regions` to request live-wire fences within unitary
regions in both the emitted Guppy source and the audit. The example saves
`guppy-ordering.json`, listing covered boundaries, uncovered boundaries, and
runtime helper-call cost. Measurement/reset boundaries remain uncovered. This
option does not apply SI1000 noise to Guppy or establish physical timing; see
[explicit Guppy ordering](named-qec-gadgets.md#explicit-guppy-ordering).

Use `--guppy-ordering all_moments` to include measurement/reset boundaries and
empty live intervals in the compiler dependency contract. This requires the
matching native ordering helpers and uses one runtime barrier request per source
cut. The ordering receipt separately counts the dependency-only helper calls.
The physical timing and Guppy noise limitations still apply.

`examples/python_examples/inspect_guppy_schedule.py --family 488 --plot` records
the emitted runtime batch clock, original/adapted operations, and source barrier
segments. It also supports `swap666`, `lacroix`, and `periodic`. Its supplied
durations are illustrative serial-runtime inputs, not calibrated hardware values
or the paper noise model. See [the batch-clock audit](named-qec-gadgets.md#inspect-the-emitted-batch-clock)
for the distinction between site gaps and live-qubit idle exposure.
Add `--include-lifetimes` for preparation/readout labels and source-bound site
windows that exclude gaps between site uses. This records a tagged execution
whose runtime scheduling can differ; see [endpoint binding and its limits](named-qec-gadgets.md#bind-preparation-and-readout-to-runtime-site-windows).


## Reproducible decoder comparisons

The `examples/python_examples/benchmark_si1000_memories.py` example
compares freshly sampled PECOS circuits and the unchanged published references.
It requires the paper-pinned Tesseract CLI and a provenance JSON binding its
executable hash to source revision `024db1d3b5b038f565c476dd1b51885271f7b0bf`.
The executable is supplied explicitly; the example does not download it or infer
a source revision from its filename. The provenance JSON must contain
`source_commit` and `binary_sha256`; retain the source archive and build recipe
with that declaration. Use Stim 1.16.0 to match the released
simulation environment. Every run saves its command, seed, circuit, DEM, native
statistics, hashes, and Python Stim version.
The estimator requires SciPy. The optional native sampling path additionally
requires the experimental `pecos_rslib_exp` extension available in this workspace.

```sh
python examples/python_examples/benchmark_si1000_memories.py \
    --tesseract /path/to/tesseract \
    --decoder-provenance /path/to/provenance.json \
    --shots 1000 --output output/si1000-decoder-benchmark
```

The defaults select periodic d4, both bases, at p=.0005/.001. Repeat `--case`
to select other pinned fixture stems. The runner applies all 63 published
BFS/coordinate/index detector orderings, beam 20, beam climbing, no revisits,
queue limit 1,000,000, and no sparsification. It checks exact decoder coordinates
and matching DEM mechanisms before sampling, and rejects incomplete or
low-confidence decoder runs. Circuit text is normalized before DEM generation
so both decoder paths receive the same serialized coordinate precision.

The manifest covers 42 released planar/periodic 488 instances. For circuits not
included in the small checked-in fixture set, supply `--reference-dir` pointing
to the release's `circuits/simulated` directory. Every selected file is verified
against its released SHA-256 before use.

By default, both variants sample through the pinned Tesseract/Stim executable.
Use `--sampling-backend pecos` to sample the PECOS noisy TickCircuit with the
native stabilizer engine and feed detector/observable deviations to the same
decoder. An ideal native reference removes authored affine signs. The report
retains the sample-file hash, reference values, and native-extension hashes.
The unchanged reference circuit still samples through Tesseract/Stim.

`--wall-time-budget 1800` caps the run at approximately 30 minutes, while
`--timeout` limits each sampling or decoding subprocess separately. Circuit
preparation is checked between cases and can slightly overrun a deadline.
Native sampling runs in a terminable subprocess. Atomic checkpoints retain
completed fixed-shot runs and list pending runs; timed-out partial samples
never contribute to estimates. Use a fresh output directory for each run.
`--threads` selects independent decoder shot workers, defaulting to one. It does
not reduce the 63 detector orderings or beam size. Thread count is retained and
checked in the run receipt; decoder-reported time sums shot times rather than
measuring parallel wall time, so use the outer elapsed time for budget planning.

To measure worker scaling before committing to a longer run, use a completed
case from a saved benchmark:

```sh
python examples/python_examples/profile_si1000_decoder.py \
    --benchmark output/si1000-decoder-benchmark \
    --case torus_488_d4_r4_Z_phase0_p0p001 \
    --threads 1 4 --shots 16 --repetitions 2 \
    --timeout 150 --wall-time-budget 600 \
    --output output/si1000-decoder-profile
```

The profiler checks the circuit, DEM, and pinned decoder hashes, draws one new
Python Stim sample file, and feeds exactly those samples to every worker count.
It alternates worker-count order between repetitions, records wall time and
summed shot time separately, and compares every predicted observable bit. It
reports median speedups only when all planned runs finish with identical
predictions. Timeouts, unresolved outcomes, and missing output cannot become
speedup evidence. Sampling and decoding run in bounded subprocesses; completed
measurements survive checkpoints. Input validation and small checkpoint overhead
are outside the strict subprocess deadlines.

These repeated decodes are timing measurements, not extra independent shots for
an error-rate estimate. Python Stim samples need not match an earlier C++ sample
stream with the same seed. Small batches can be dominated by one difficult shot;
scaling measured on one basis or noise rate does not establish throughput for
the remaining cases.

`pecos.qec.memory_benchmark.MemoryCounts` and `equal_weight_memory_estimate()`
provide the released equal-basis/phase estimator. It averages stratum failure
fractions, not pooled counts; constructs Bonferroni-adjusted Clopper-Pearson
intervals; then applies `1 - (1 - P)**(1/(k*r))`. Complete groups require both
X/Z bases and, for planar 488, both phases. Partial groups remain explicitly
incomplete in the example's JSON. Other papers may require different exposure
normalizations; this convention is not applied to them automatically.

Tests reproduce all 50 frozen plot points from the release's 116 count rows.
Fresh small local runs validate the benchmark workflow; they do not reproduce
publication precision, suppression factors, or thresholds. Zero observed errors
still produce a nonzero upper bound. For adaptive sampling, these intervals are
descriptive and are not confidence sequences. See the release's
[statistical and decoder configuration](https://doi.org/10.5281/zenodo.22820614).

Lavasani's memory experiments use a separate convention:
`pooled_wilson_memory_estimate()` pools failure and trial counts, computes a
Wilson interval, and divides the estimate and interval by `rounds +
initialization_rounds`. Its defaults are `initialization_rounds=0.5` and
`z_score=1`, matching the paper's nominal one-sigma coverage (approximately
68.27%). For example, the published d3 Tesseract counts can be evaluated with:

```python
from pecos.qec.memory_benchmark import MemoryCounts, pooled_wilson_memory_estimate

estimate = pooled_wilson_memory_estimate([MemoryCounts(15840, 178)], rounds=20)
print(estimate.rate_per_cycle, estimate.interval)
```

Supply counts of logical-memory trials, including any pooling across signs,
bases, or parallel patches already performed; do not multiply published trial
counts by the patch count again. The interval assumes independent trials and
does not include device/emulator mismatch or training variability. Tests retain
all 12 published hardware rows and an explicit rounding discrepancy: d5
Frontier prints `4.0e-5`, whereas its counts and stated normalization imply
`18/(21600*20.5) = 4.0650406504e-5`. The other 11 central values and all 12
uncertainty increments agree at their displayed precision. This is a check of
published statistics, not reproduction of the hardware experiment. See
[Lavasani et al., Table I and supplementary decoder analysis](https://arxiv.org/abs/2610.24077v1).
