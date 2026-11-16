# Authoring and inspecting QEC gadgets

The named gadget interface separates the code you want to protect, the gadgets
that implement an experiment, and their physical schedule. Guppy and TickCircuit generation
happen after these descriptions have been checked and lowered.

| Level | Main objects | Questions you can inspect |
| --- | --- | --- |
| Code | `CSSCodeSpec`, named checks and logical operators | Which data roles support each operator? |
| Gadget | `Gadget`, `GadgetCall`, `GadgetProgram` | Which roles enter and leave? What do measured outcomes mean? |
| Schedule | `ScheduledCircuit`, `ScheduleRow`, `ScheduleResources` | What runs in each moment, on which carrier, and after which predecessors? |
| Backend | `guppy_program()`, `guppy_source()`, `to_tick_circuit()`, `to_stim()` | Which authoring operation produced this source statement or measurement? |

The generic gadget interface is available from `pecos.qec.generic`. It supports
static Clifford gadgets with reset, measurement, and carrier relabeling. The
[paper color-code constructions](paper-color-codes.md) use this interface for
their preparation, initialization, recurring extraction, and readout.
For signed Pauli-state preparation and gates between extraction rounds, see
[Six-state memories and logical Clifford gadgets](color-code-cliffords.md).

## Inspect an existing code and its gadgets

```python
from pecos.qec.color import (
    ColorExtraction, color_code_spec, color_code_implementation,
    color_memory_program,
)
from pecos.qec.generic import SchedulePolicy

code = color_code_spec("666", distance=3)
print(code.describe())
print(code.checks[0])
# X[0] = X(data[...]) * ...

implementation = color_code_implementation(code, ColorExtraction.SWAP_SUPERDENSE)
print(implementation.gadget(phase=0).describe())

experiment = color_memory_program(implementation, rounds=3, basis="Z")
print(experiment.describe())

reference = experiment.compile(SchedulePolicy.REFERENCE)
packed = experiment.compile(SchedulePolicy.ASAP_UNITARIES)
ordered = experiment.compile(SchedulePolicy.COMPACT_UNITARIES)
print(reference.describe())
print(packed.resources)
print(reference.explain_detector("round:1:X:0"))
```

Choose the scheduling policy separately from the code and extraction gadget:

| Policy | Source scheduling rule |
| --- | --- |
| `REFERENCE` | Retain the authored paper schedule, including empty moments. |
| `ASAP_UNITARIES` | Place each operation as early as its per-role dependencies allow within unitary regions. Disjoint operations can change listing order. |
| `COMPACT_UNITARIES` | Pack consecutive disjoint operations while preserving the entire flattened source order and gadget origins. |

Both alternatives retain preparation, measurement, and relabel boundary moments
exactly, including unitary operations sharing those moments. Empty unitary
moments can disappear. Ordered compaction starts a new moment when the next
operation conflicts with an operation in the current moment. This preserves
ideal operations, named events, and parities, but changes possible idle locations.
It does not guarantee better runtime timing or preserve the paper's noise model.
The policy applies before either Guppy or TickCircuit lowering. Source listing
order is distinct from the compiler's emitted operation order.

The extraction description exposes `input:X[face]`, `output:X[face]`, their Z
counterparts, logical signs, and any record constraints as named XOR expressions.
Input and output maps are separate because extraction byproducts can change
their relation to raw measurements. The experiment composes these expressions
across rounds without exposing bit masks or measurement offsets to its caller.

You can inspect each call through `experiment.calls`, including its gadget,
local-to-program bindings, and boundary frames. `experiment.detectors` and
`experiment.observables` contain the named parity expressions before lowering.
The existing `color_memory(...)` convenience API now compiles this same path.

## Author a reusable measurement gadget

This gadget measures ZZ on two live data ports using a scratch ancilla:

```python
from pecos.qec.generic import Gadget, GadgetFlow, Gate, NamedParity

measure_zz = Gadget(
    name="measure ZZ",
    roles=("left", "right", "ancilla"),
    inputs=("left", "right"),
    outputs=("left", "right"),
    moments=(
        (Gate("R", ("ancilla",)),),
        (Gate("CX", ("left", "ancilla")),),
        (Gate("CX", ("right", "ancilla")),),
        (Gate("M", ("ancilla",), outcome="syndrome"),),
    ),
    flows=(GadgetFlow("ZZ", NamedParity(("syndrome",))),),
)
```

Each inner tuple is a reference moment of disjoint operations. `inputs` and
`outputs` declare which ports are live at the gadget boundaries. The ancilla
is prepared inside this gadget and consumed by its measurement. Every
measurement has a unique local name.

`GadgetFlow` attaches an author-declared meaning to a parity of local outcomes.
The compiler checks references and composition contracts; it does not prove
that an arbitrary circuit measures its declared operator. New gadgets need
independent instrument tests. The paper gadgets retain their existing
reference-circuit and conditional-channel checks.

## Bind and compose gadgets

The same definition can be used repeatedly, or on different data roles:

```python
from pecos.qec.generic import GadgetProgram

experiment = GadgetProgram(("data[0]", "data[1]", "probe"))
experiment.append(
    Gadget(
        name="prepare 00",
        roles=("data[0]", "data[1]"), inputs=(), outputs=("data[0]", "data[1]"),
        moments=((Gate("R", ("data[0]",)), Gate("R", ("data[1]",))),),
    ),
    label="prepare",
)
bindings = {"left": "data[0]", "right": "data[1]", "ancilla": "probe"}
first = experiment.append(measure_zz, label="first", bindings=bindings)
second = experiment.append(measure_zz, label="second", bindings=bindings)

experiment.detector("repeat ZZ", first["syndrome"] ^ second.flow("ZZ"))

readout = experiment.append(
    Gadget(
        name="readout Z",
        roles=("data[0]", "data[1]"), inputs=("data[0]", "data[1]"), outputs=(),
        moments=((Gate("M", ("data[0]",), "left"), Gate("M", ("data[1]",), "right")),),
    ),
    label="readout",
)
experiment.detector("final ZZ", second["syndrome"] ^ readout["left"] ^ readout["right"])
plan = experiment.compile()
guppy_program = plan.guppy_program()
```

Call labels scope the measurements: `first/syndrome` and `second/syndrome` are
different events. XOR cancels repeated terms. Binding dictionaries must cover
every gadget role and cannot alias distinct ports. An omitted dictionary uses
the gadget's role names directly.

Appending validates live roles and basis conventions before changing the
program. A complete experiment must consume every live role before compilation.
Recompiling the authored program with another policy retains access to the
original reference schedule. Appending a gadget itself introduces no extra
timing barrier.

## Frames and carrier changes

`input_h_frame` and `output_h_frame` specify the physical Hadamard-basis
convention on live ports. Composition checks them by bound role name. A
mismatch raises an error; an explicit conversion gadget can apply the needed
H gates and declare the new convention.

These tuples are author-declared basis conventions. They do not claim that a
qubit is in a particular state, and they are separate from measurement-dependent
Pauli signs. The latter remain named parity expressions, including the paper
gadgets' `logical:X` and `logical:Z` flow entries.

`Gate("RELABEL", ("data", "ancilla"))` exchanges the roles' reference carriers
without adding a physical quantum gate. Both roles must be live. The compiler
carries this mapping across gadget boundaries, and the Guppy emitter renames
its qubit variables accordingly. Inspection shows the carriers before each
operation; a RELABEL row shows the carriers immediately before exchange.

## Explain a schedule and follow its provenance

```python
print(plan.describe())
print(plan.resources)
for row in plan.inspect():
    print(row.id, row.moment, row.roles, row.carriers, row.dependencies)
print(plan.explain_operation("second/moment[1]/op[0]"))
print(plan.explain_detector("repeat ZZ"))
print(plan.measurement_aliases)
```

Every lowered operation has a stable identity containing its call, original
gadget moment, and operation index. Rescheduling retains that identity. Rows
include per-role predecessors and the reset/measurement/relabel boundaries used
by the scheduler. Reference mode retains authored moment positions; ASAP mode
packs unitary regions according to these dependencies. There is no device
connectivity solver or duration-aware scheduling in this interface yet.

Resource counts distinguish allocated roles, peak live roles, physical
operations, two-qubit operations, measurements, and relabels. `moments` includes
zero-time relabel steps; `physical_moments` follows the selected Stim tick
convention. These are static reference-allocation counts, not Guppy runtime site
IDs or measured hardware durations.

Generated Guppy source includes the same operation identities, role names, and
measurement aliases in comments. Runtime outputs retain `qec:m:<ordinal>` tags
for compatibility with existing measurement auditing and DEM construction.
`plan.measurement_aliases` maps authored names to those tags. Comments provide
source traceability; they do not enforce timing or survive as a general runtime
debug-information contract. Guppy may reschedule and choose different sites.

## Raw measurement outputs and separate QEC definitions

The scheduled Guppy backend emits each physical readout as a direct scalar
result, with no detector or logical XOR performed on that output:

```python
m0 = measure(q0).read()
result("qec:m:0", m0)
```

Every measurement has a unique canonical tag, including preparation flags,
extraction records, and final data readout. `plan.measurement_aliases` associates
authored names such as `round[0]/ancilla[3]` with those tags. `plan.inspect()`
retains measurement roles, gadget origins, schedule moments, and reference
carriers; these are source metadata rather than runtime hardware-site identities.
Keep that mapping with exported Guppy and recorded shots.

`plan.measurement_manifest()` returns a detached, JSON-ready dictionary with a
versioned format (`pecos.scheduled-measurements`, version 1):

```python
import json
from pathlib import Path

manifest = plan.measurement_manifest()
Path("memory.measurements.json").write_text(json.dumps(manifest, indent=2))
```

`raw_events` lists measurement ordinals, tags, authored names, roles, reference
carriers, and source locations. `derived_parities` lists detectors and observables
separately, with their measurement tags, explicit Boolean XOR constants, and
detector coordinates. Evaluate each definition by XORing its referenced raw bits
with its constant, including constant-only definitions. There is no implicit
subtraction of an ideal baseline. Tags are unique within one experiment, not
across independently generated experiments; every column stores shot occurrences
of that event. JSON shot values from PECOS are integer 0/1 encodings of the
Boolean readouts.

The manifest describes the selected source schedule. It does not certify compiled
measurement binding, physical timing, or noise equivalence; use `audit_guppy()`
separately for compiler-bound measurement evidence. The planar, periodic, and
Lacroix memory examples save `.measurements.json` beside `.raw-results.json`.
Each result file records the manifest filename and SHA-256, simulation seed,
shot count, and unmodified raw columns. Keep each pair together when moving
results. These examples record ideal stabilizer simulations.

Detector and observable definitions are separate derived data:

```python
aliases = plan.measurement_aliases
raw_one_shot = {tag: values[shot] for tag, values in columns.items()}
detector_bits, observable_bits = plan.evaluate_named(raw_one_shot)
detector_definitions, observable_definitions = plan.dem_spec()
```

Changing a parity definition, its coordinates, or its XOR constant leaves the
generated Guppy program unchanged. The same raw records can therefore be
reanalyzed using a different detector/observable definition. Compiler-bound
measurement auditing verifies that tags name direct measurements, not computed
XORs. No `DETECTOR` result stream is required by this path.

For future reusable Guppy components, preserve the separation between raw
outcomes, event metadata, and decoding/frame-processing state. A single physical
outcome can contribute to several checks, so a face/check label is metadata,
not a substitute for a unique measurement identity. Leakage/validity and
postselection information should likewise remain explicit if those output types
are added; the current scheduled backend's result contract is Boolean readout.

## Verify a gadget's signed Pauli contracts

`GadgetFlow` names an outcome expression. To check its quantum meaning, supply
explicit `PauliFlow` contracts with input/output operators and the local outcome
parity. For a ZZ measurement gadget with ports `left`, `right`, and outcome
`syndrome`:

```python
from pecos.qec.generic import NamedPauli, NamedParity, PauliFlow

zz = NamedPauli("ZZ", (("left", "Z"), ("right", "Z")))
bit = NamedParity("syndrome")
audit = measure_zz.audit_flows([
    PauliFlow("measures ZZ", input=zz, parity=bit),
    PauliFlow("projects ZZ", output=zz, parity=bit),
])
print(audit.describe())
audit.require_valid()
```

`None` at a boundary means identity. A flow states that the input Pauli becomes
the output Pauli times the sign `(-1)**parity`. Use `parity ^ 1` for a negative
relation, including one without measurements. Only live input/output roles may
appear in the corresponding operators. Measurement names are local to this
gadget, rather than names qualified by a program call.

Verification treats inputs as arbitrary quantum states and checks measurement
backaction. Operators use the gadget's declared boundary H frames: X/Z exchange,
and Y acquires a minus sign. `RELABEL` updates output carrier bindings and emits
no physical SWAP. Destructive measurement followed by reset begins a fresh
lifetime; a flow cannot silently carry the old state through it.

The optional Stim backend supplies a complete set of signed flow generators.
PECOS checks requested supports by binary elimination, then multiplies the
selected generators and checks the sign exactly. This uses no sampled shots or
randomized `has_flow` test. The receipt retains each requested contract, its
physical flow, the analysis circuit, generator witnesses, carrier bindings, and
Stim version. See [Stim's flow API](https://github.com/quantumlib/Stim/blob/main/doc/python_api_reference_vDev.md#stim.Circuit.flow_generators).

The planar 4.8.8 and triangular 6.6.6 implementations supply explicit contracts
for their published input/output checks, logical transport, and record constraints:

```python
contracts = implementation.flow_contracts(phase=0)
audit = implementation.audit_flows(phase=0)
audit.require_valid()
```

These contracts use the existing cited paper parity maps and are checked against
the concrete extraction gates. Wrong signs and missing entangling gates are
tested as failures. Inspect every phase of an alternating or three-phase
construction. The audit checks only the requested ideal relations; it does not
claim a complete channel specification, fault tolerance, calibrated timing, or
noise equivalence. Merely adding a `GadgetFlow` annotation does not invoke it.

Periodic 4.8.8 exposes the same `flow_contracts()` and `audit_flows()` methods
without a phase argument. Both the published CNOT and native-CZ constructions
retain every face check (including global dependencies), all four canonical
logical X/Z pairs, and intrinsic record constraints. Native-CZ affine signs and
boundary H frames are included. Check names retain their face-color annotations.

## Check a complete memory's parities

For a closed experiment, use the selected schedule directly:

```python
plan = experiment.compile(SchedulePolicy.COMPACT_UNITARIES)
audit = plan.audit_parities()
print(audit.describe())
audit.require_valid()
```

Every declared detector and observable must be identically zero, including its
explicit XOR constant. Reports distinguish **always zero**, **always one**, and
**nondeterministic** expressions. An always-one result is a sign mismatch; the
audit does not subtract an ideal baseline to make it pass. An experiment without
any detector or observable definitions is rejected instead of reporting success
for an empty check list.

This is exact signed-flow verification of the closed ideal circuit. It covers
initialization, repeated extraction, and final readout together, and works for
Lacroix's published memory parities without assuming open extraction contracts.
The report retains authored names, indices, coordinates, XOR constants,
operation origins, and explicit bindings to the verifier's internal names.
Names containing slashes and punctuation remain supported. A random individual
measurement is permitted when the declared parity of correlated records is fixed.

Closed memory checks do not establish a gadget's action on arbitrary inputs, or
validate the circuit emitted by Guppy. Use the open-gadget flow audit and compiled
execution checks for those separate questions. The periodic example offers
`--audit-flows` for both extraction and memory checks; the Lacroix example offers
`--audit-parities` for its closed memory checks. Both save JSON and text receipts.

## Run the examples

```sh
uv run python examples/python_examples/named_qec_gadgets.py
uv run python examples/python_examples/paper_color_memories.py --audit-flows --output /tmp/paper-color-memories
```

The first example starts from a state with a random first ZZ outcome and checks
repeated/final parity agreement under both schedules through actual Guppy
execution. The second writes code, gadget, and schedule descriptions beside
the four paper constructions' generated circuits. With `--audit-flows`, it also
checks every extraction phase and saves signed-contract receipts. Neither requires changes to
the Guppy emitter when adding or composing a gadget.

This interface currently covers flat, static gadget sequences. Hierarchical
protocols, adaptive feed-forward, automatic flow inference, arbitrary logical
gadgets, and the proposed Rust instruction graph remain separate extensions.
Explicit affine parities are supported as described below; automatic flow
inference remains a separate feature.

## Lower the same plan to TickCircuit or Guppy

```python
plan = experiment.compile(SchedulePolicy.REFERENCE)
function = plan.guppy_program()
source = plan.guppy_source(function_name="my_memory")
ticks = plan.to_tick_circuit()
print(ticks.circuit_definitions())
print(ticks.render("ascii"))
```

`TickCircuit` preserves the selected physical moments, including explicit empty
moments. A pure `RELABEL` step updates carriers and adds neither a SWAP nor a
tick. The exporter assigns canonical event ordinals as native measurement IDs
and resolves detector/observable references against native emission order.
Use `ticks.measurement_emission()` to map each emitted record to its canonical
ID; this is distinct from a qubit ID, which can be measured repeatedly.

Tick metadata `qec_moment` and `qec_operations` retain the authored locations,
named roles, gates, and physical carriers. Circuit metadata includes
`qec_roles`, `qec_measurement_names`, `qec_measurement_tags`,
`qec_output_carriers`, `qec_relabels`, and `qec_citations`. Authored observable
labels are preserved in `qec_observable_labels`; labels reserved by the native
`L<n>` decoder-ID syntax are escaped for the native definition.

The native export supports the existing `DetectorErrorModel.from_circuit(...)`
path. Both Guppy and TickCircuit come from the same scheduled plan; Guppy may
reschedule and reallocate sites, so a Guppy noise analysis must still use its
actual trace. TickCircuit retains affine constants in parity metadata. Native
detector/observable definitions still require nonempty measurement supports.


## Noise attached to a physical schedule

The noise profile is a separate layer after choosing the physical schedule:

```python
from pecos.qec.generic import SI1000

noisy = SI1000(0.001).apply(plan)  # plan must use the native-CZ alphabet
print(noisy.describe())
for channel in noisy.channels[:5]:
    print(channel.id, channel.reason, channel.qubits, channel.probability)
stim_circuit = noisy.to_stim()
```

This implements [Shutty's released SI1000 convention](https://doi.org/10.5281/zenodo.22820614):
depolarization `p` after CZ, `p/10` after single-qubit Cliffords and on idle sites,
`X_ERROR(2p)` after reset, and `X_ERROR(5p)` before measurement. Every noncollapsed
site receives an additional independent `DEPOLARIZE1(2p)` during a reset or
measurement moment, including sites performing concurrent gates. Idle and wait
noise use the full allocation, including sites before preparation or after
retirement. Empty moments receive idle noise; synthetic MPAD records do not.

Each channel identifies one independent physical location. A two-qubit
depolarizing branch is one location, even if it affects both qubits. Reapply the
profile after rescheduling: different idle locations can change error behavior.
Applying the profile does not lower CNOTs or reinterpret free carrier relabels;
unsupported operations fail explicitly.

`compile_native()` uses PECOS's native stabilizer simulator for noisy trajectories
and native backward Pauli propagation for an independent DEM:

```python
native = noisy.compile_native()
records = native.sample_records(100, seed=32)
values = [plan.evaluate(record) for record in records]
dem = native.detector_error_model()
mechanisms = dem.independent_mechanisms()  # full precision: (probability, detectors, observables)
samples = dem.to_sampler().sample_batch(100, seed=32)
```

Trajectory sampling chooses one categorical branch per channel. The DEM converts
depolarization exactly to independent Bernoulli mechanisms, then combines identical
effects by XOR probability. It preserves declared detector/observable dimensions
and all coordinate axes, including zero-noise cases. Native DEM construction rejects
nondeterministic ideal parities. Neither native path imports or converts through Stim.
Python orchestrates trajectories around native stabilizer kernels; this is not yet
a compiled batch execution path or a performance benchmark.

`native.replay_faults(faults)` returns physical measurement records;
`native.fault_effect(faults)` independently propagates their detector/observable
flips. Records contain no synthetic MPAD and use canonical scheduled event order.
`plan.evaluate(record)` includes authored XOR constants; propagated faults and DEM
samples are deviations from the ideal reference, where those constants cancel.
Use physical channel IDs for fault witnesses: the aggregate DEM no longer records
which branches were mutually exclusive at a physical location.

For execution entirely inside the native runtime loop, lower the channels into
an executable `TickCircuit`:

```python
from pecos_rslib_exp import sim_neo, stabilizer, monte_carlo
from pecos.qec.generic import evaluate_tick_circuit

ticks = noisy.to_tick_circuit()
result = sim_neo(ticks).quantum(stabilizer()).sampling(monte_carlo(100)).seed(32).run()
values = [evaluate_tick_circuit(ticks, row) for row in result.to_list()]
```

Use an explicit stabilizer backend for these large Clifford memories. The current
inline path supports one worker. Do not attach another `.noise(...)` model;
the runtime rejects that combination when channels are present. Typed channel payloads drive execution,
while each gate's `qec_noise_channel` metadata retains its original location ID,
probability, reason, and placement.

Ticks here are **execution steps**, not additional hardware time. A physical
moment has one `qec_execution_phase="gates"` step, plus nonempty `before`, `after`,
`idle`, and `collapse_wait` steps. Their common `qec_moment` preserves the selected
schedule. `qec_physical_moments` records its original length. Do not infer hardware
duration from `ticks.num_ticks()` or apply a new per-tick idle model. Empty physical
moments, simultaneous gates and measurement/reset, and separate idle/wait
channels are retained. At p=0 there are no added channel steps.

For direct circuit authoring, `tick.x_error(p, qubit)`,
`tick.depolarize1(p, qubit)`, and `tick.depolarize2(p, (q0, q1))` insert one typed
categorical channel each. Probabilities must be finite in [0,1]; two-qubit targets
must differ, and ordinary tick qubit-conflict checks apply. The 15 nonidentity
two-qubit Paulis each have probability p/15; they are not sampled independently.

The generic native DEM and sampler now consume the actual channel payloads:

```python
from pecos_rslib.qec import DetectorErrorModel, DemSampler

model = DetectorErrorModel.from_circuit(ticks)
sampler = DemSampler.from_circuit(ticks)
flips = sampler.sample_batch(1000, seed=32)
```

This also works after `ticks.to_dag_circuit()`. Omitted gate-noise rates default
to zero for inline circuits; supplying additional nonzero gate, idle, or readout
noise is rejected. Ideal circuits retain their previous default gate-noise rates.
Both `noisy.to_tick_circuit()` and `ideal_ticks.with_noise(...)` declare
`pecos_noise_source="inline"`, including at zero noise, so a noiseless export does
not accidentally acquire default noise. Actual channel payloads also identify an
inline circuit without this marker.

Exact conversion supports binary Pauli channels on one or two qubits and uniform
depolarization up to p=3/4 (one qubit) or p=15/16 (two qubits), covering this entire
SI1000 parameter range. Other distributions fail explicitly. Categorical branches
are converted to an equivalent independent-error distribution before propagation;
DEM mechanisms are not physical fault locations. Detectors and observables must
be deterministic in the ideal circuit. Affine constants cancel in error flips.

The earlier `noisy.compile_native().detector_error_model()` remains available as
an independent path. Generic Stim export also accepts inline Pauli channels:

```python
from pecos.qec.surface import tick_circuit_to_stim

stim_text = tick_circuit_to_stim(ticks)
```

TickCircuit remains the source representation and may support features beyond
Stim. Export is a compatibility path: supported operations are converted faithfully;
unsupported channels fail with their tick location. Binary errors, uniform
one/two-qubit depolarization over the full [0,1] range, and nonuniform one/two-qubit
Pauli mixtures retain their categorical distributions. Nonuniform mixtures use
Stim's `PAULI_CHANNEL_1/2`; they are never split into independent Pauli errors.
The exporter reads physical targets from the payload and does not trust noise
receipt metadata. See the [Stim channel definitions](https://github.com/quantumlib/Stim/blob/main/doc/gates.md).

Exported `TICK`s preserve execution-step boundaries, including channel-only steps;
they do not encode physical duration. Affine constants use a trailing `MPAD 1`,
so an exported measurement record may have one extra constant bit. No extra noise
model may be attached during export, including to an explicitly zero-noise circuit.
`generate_dem_from_tick_circuit_via_stim()` uses zero omitted rates for inline
circuits and retains its legacy 0.01 defaults for ideal circuits. Stim may reject
DEMs for some otherwise exportable channels; the wrapper does not enable a
silent approximation. For very small depolarizing probabilities, Stim's own DEM
analysis can round away terms even though the exported circuit retains them;
the native exact DEM path preserves these tiny probabilities.

Constant-only empty-support parities retain the ideal TickCircuit lowering's
explicit rejection.

`noisy.ideal.guppy_program()` and `noisy.ideal.to_tick_circuit()` still produce ideal
implementations. Passing them to a generic runtime does not apply this adapter's
noise. A Guppy trace requires a separate timing/allocation audit before claiming
schedule-matched noise. Inspect the actual implementation with:

```python
audit = noisy.ideal.audit_guppy(seed=32)  # optional runtime=plugin selector
print(audit.describe())
receipt = audit.to_dict()
normalized_runtime_ticks = audit.build.circuit
```

This compiles and traces the ideal program, validates its compiler-bound named
measurement identities, and retains the annotated runtime analysis circuit, ledger,
source resources, and normalized operation counts. The existing DEM builder lowers
Clifford rotations and simplifies single-qubit chains; use
`capture_qis_operation_trace()` for the original emitted pulse stream. Noise rates are explicitly zero, and
runtime idles are retained. The receipt marks physical schedule and noise
equivalence as unverified: matching measurements is not a timing certificate.
Reference moments and runtime replay ticks are separate units. For example, a
local d3/r1 native-CZ memory had 26 reference moments but 151 runtime replay ticks,
with H/CZ instructions lowered into rotations. A device timing model must address
that emitted operation stream before applying noise. `si1000_color_memories.py
--audit-guppy` saves this receipt alongside the other artifacts.

### Explicit Guppy ordering

Ordering is an opt-in lowering choice, independent of the authored schedule:

```python
from pecos.qec.generic import GuppyOrdering

ordering = GuppyOrdering.UNITARY_REGIONS
print(plan.guppy_ordering_plan(ordering).describe())
source = plan.guppy_source(ordering=ordering)
program = plan.guppy_program(ordering=ordering)
audit = plan.audit_guppy(ordering=ordering, seed=32)
```

| Mode | Enforced compiler dependencies |
| --- | --- |
| `FUNCTIONAL` (default) | Ordinary Guppy data dependencies; no added fences |
| `UNITARY_REGIONS` | Joins all live roles between adjacent moments that both contain no reset, measurement, or relabel |
| `FIXED_LIFETIME` | Joins every moment boundary; requires initial reset on every role, terminal measurement on every role, and no intermediate reset or measurement |
| `ALL_MOMENTS` | Joins every moment boundary, including measurement, fresh allocation, relabel, and empty live sets, using opaque classical dependencies |

`FIXED_LIFETIME` permits interior unitaries, empty moments, and carrier relabels.
It rejects circuits with repeated ancilla lifetimes. Use `UNITARY_REGIONS` for
partial ordering in the paper memories, or `ALL_MOMENTS` to cover every cut.
Inspect the report's `uncovered_cuts` rather than assuming coverage. Empty moments
carry no duration.

The qubit-only modes collect all incoming live-qubit dependencies into one anchor and
distributes them to every outgoing wire. It uses one native runtime barrier helper
for one live role, or `2*n - 3` two-qubit helper calls for `n >= 2`. The helpers
return their qubit handles unchanged and request runtime barriers; they add no
quantum gates, qubits, or SWAPs. The ordering report exposes each fence's roles
and the total helper-call count. This overhead can be substantial for large codes.

`ALL_MOMENTS` instead carries an internal classical token through external native
helpers. Each cut collects live qubits and outcomes measured in that moment,
requests one runtime barrier, and distributes the dependency to every surviving
qubit. New allocations consume the token and perform the usual allocate/reset
sequence. Measurements remain ordinary destructive Guppy measurements with the
same named result tags. The token has no quantum state and adds no ancilla.
The helpers must remain opaque external calls; inlining their identity bodies
would erase the compiler dependency. The report distinguishes runtime barriers,
dependency-only helper calls, and ordered allocation calls. Build the Python
extension and QIS FFI library together when adopting these native helpers.
The current two-value native helper ABI is unsupported on Windows; compilation
rejects those helpers before linking. A Windows aggregate-return adapter remains
future work. The local execution validation uses macOS arm64.

These are **compiler dependency cuts and runtime barrier requests**, not calibrated
physical timing. Runtime support determines barrier synchronization behavior;
virtual phase updates can still be carried across a barrier. No mode certifies
pulse durations or equivalence to the paper's noise model. The audit records the
requested plan but keeps physical schedule and noise equivalence unverified.
Its normalized DEM analysis circuit removes barriers and simplifies Clifford
operations; use `capture_qis_operation_trace()` to inspect emitted barrier requests.
Compiled tests separately check raw operation ordering, barrier counts, and named
measurement bindings for planar 4.8.8, SWAP-superdense 6.6.6, Lacroix, and periodic
4.8.8 memories. `ALL_MOMENTS` also tests lifetime changes and empty intervals
through LLVM optimization levels 0–3. These tests establish ordering on the
supported PECOS compilation/execution path; they do not establish calibrated
hardware timing or general portability to other Guppy execution backends.

### Inspect the emitted batch clock

Use a separate schedule audit to retain the runtime's timestamped batches:

```python
from selene_simple_runtime_plugin import SimpleRuntimePlugin

# Illustrative inputs, not measured hardware durations or the paper noise model.
runtime = SimpleRuntimePlugin(duration_ns_reset=5, duration_ns_rxy=10,
                              duration_ns_rzz=25, duration_ns_measure=40)
schedule_audit = plan.audit_guppy_schedule(runtime=runtime)
print(schedule_audit.describe())
summary = schedule_audit.to_dict()
evidence = schedule_audit.evidence()
```

This API requires an explicit plugin object with `library_file` and
`get_init_args()`. It preserves the plugin's event adapter, or an explicitly
supplied `batch_adapter_factory`, and records original and normalized operations
separately. Uninterpreted custom events are rejected. Snapshot accessors return
independent copies. The default ordering is `ALL_MOMENTS`; the other ordering
modes are also accepted.

The report includes batch-clock span, operation counts, and site occupancy
assigned by each batch. Multiple operations on one site can share a batch
interval, which counts once toward that site's occupancy. Between-operation
gaps exclude time before the first and after the last operation and can cross
site reuse. They are **not live-qubit idle-noise exposure**. Physical runtime site
IDs are not authored role IDs. Zero-duration runtimes and experiments with no
batches are reported explicitly rather than treated as calibrated timing.

Source QIS barrier counts are checked against the ordering plan. With
`ALL_MOMENTS`, `qis_moments` also lists source gate counts and emitted QIS quantum
operation counts between barriers. These segments are separate from timestamped
physical batches; the audit does not establish a gate-by-gate equivalence or a
source-moment-to-batch mapping. Virtual phases, gate decomposition, and physical
site allocation still matter.

Capture executes one coin-toss trace, so its measurement outcomes are provenance
evidence, not ideal stabilizer statistics. Continue using `audit_guppy()` for
compiler-bound measurement/DEM analysis and stabilizer simulation for ideal
parity checks. Schedule auditing keeps calibration, physical schedule, measurement
binding, and noise-equivalence claims unverified.

Run `examples/python_examples/inspect_guppy_schedule.py --plot` for a planar
4.8.8 example with JSON receipts, raw evidence, source, runtime arguments/library
hash, and a timeline PNG. `--family` also accepts `swap666`, `lacroix`, and
`periodic`. Its serial-runtime durations are explicitly illustrative. Even
preparations in one authored moment can be emitted serially by this runtime.

### Bind preparation and readout to runtime site windows

Enable `include_lifetimes=True` to exclude gaps between a site's destructive
measurement and its next preparation:

```python
schedule_audit = plan.audit_guppy_schedule(runtime=runtime, include_lifetimes=True)
windows = schedule_audit.to_dict()["lifetime_windows"]
for window in windows["windows"]:
    print(window["measurement_tag"], window["role_history"],
          window["runtime_site"], window["gap_ns"])
```

This requires `ALL_MOMENTS` and uses an opt-in allocator that attaches required
provenance metadata to each reset. No quantum gates or ancillas are added, but
metadata can introduce local runtime barriers and drains. **The recorded clocks
describe this tagged execution**; do not assume its timing equals the ordinary
untagged program. Reproduce it with
`plan.guppy_program(ordering=GuppyOrdering.ALL_MOMENTS, allocation_provenance=True)`.
The example's `--include-lifetimes` flag saves the corresponding tagged source.

Each window links a required reset label and a measured program result ID to
the source preparation/readout moments and role history, including `RELABEL`.
Runtime future IDs remain distinct from program result IDs. Gaps run from the
end of preparation to the start of destructive readout. Native Selene handles
may be recycled: `allocation_occurrence` distinguishes successive allocations
of a handle, and `source_qubit_kind="native_handle"` identifies that convention.
Legacy QIR allocation IDs remain unique. Tagged provenance uses flat tracing;
scheduled transport rejects source metadata rather than silently dropping it.
The reported intervals exclude gate intervals
and gaps between successive uses of a site. Batch durations count once
per participating site. All normalized gate kinds and site operands must match
the lowered emission sequence; this check does not compare rotation angles.
Incomplete traces, missing or conflicting endpoint labels, adapters that change
reset/readout boundaries, and boundaries sharing their site and batch with
another operation are rejected.

`bind_runtime_lifetime_windows(plan, schedule_audit.evidence())` from
`pecos.qec.generic` can repeat the binding offline. Endpoint provenance does not
prove that logical state remained at one physical site under runtime routing.
These are normalized site windows, not calibrated idle-noise exposure; logical
carrier residency, calibrated durations, an idle-noise model, and paper-noise
equivalence remain unverified. Compiler-bound parity analysis remains separate.

### Compare Guppy scheduling policies

Start from the original reference plan:

```python
comparison = plan.compare_guppy_schedules(runtime=runtime, seed=17)
print(comparison.describe(include_windows=True))
report = comparison.to_dict()
reference_evidence = comparison.reference.evidence()
candidate_evidence = comparison.candidate.evidence()

ordered_comparison = plan.compare_guppy_schedules(
    runtime=runtime, seed=17, candidate_policy=SchedulePolicy.COMPACT_UNITARIES,
)
```

The default candidate is `ASAP_UNITARIES`; `COMPACT_UNITARIES` selects ordered
compaction. Before executing either
variant, it checks each wire's ordered unitary operations within each region,
all preparation/measurement/relabel boundary moments, and the named event and
parity definitions. It accepts only an original `REFERENCE` plan, because an
optimized plan cannot recover the original schedule.
For ordered compaction it also checks the full source listing, including gadget
origin annotations. `source_listing_order_preserved` records whether the selected
candidate preserves that listing, independently of runtime emission order.

Both runs use `ALL_MOMENTS` and tagged preparation endpoints. The runtime library
hash is checked before and after execution. Init arguments, native gate support,
measurement deferral, library search paths, and seed are snapshotted; the report
retains this profile. Each execution constructs a fresh adapter from the same
factory. This does not fingerprint transitive shared libraries or external
configuration files, or prove that plugin/adapter state is deterministic.

The table separates source moment count, emitted batch count, reported wall-clock
span, and the sum of gaps inside site windows. Differences are **candidate minus
reference**. Per-window rows match canonical source measurements and display
human-readable event names and role histories; runtime site numbers can differ
between the two runs. A sum over site gaps is not wall-clock duration. Fewer
source moments can leave a serial runtime's reported span unchanged, and a
smaller gap sum does not establish better logical error rates.

`examples/python_examples/inspect_guppy_schedule.py --compare --plot` saves the
comparison JSON and text table, plus separate reference/candidate sources, complete
evidence and timelines. Add `--candidate compact` for ordered compaction; the
default `--candidate asap` retains the previous behavior and output directory
names. All four `--family` choices are supported. These paired
coin-toss traces remain observations of tagged executions, with uncalibrated
timing and unverified logical residency/noise equivalence. Each candidate is
an alternative schedule, not a reproduction of the paper's noise benchmark.

Use `comparison.explain(top_windows=5, top_operations=5)` to diagnose a difference. It reports
changes in source listing order and moment membership, the numbers of runtime
batches grouped by normalized operation kinds and duration, and the largest
absolute measurement-window gap changes. It shows the first source operations
whose order or moment membership changed, retaining gadget origin labels. Full source-operation placements and
batch accounting are available under `comparison.to_dict()["diagnostics"]`.
The example also saves `diagnostics.txt`.

The clock accounting uses the exact identity
`span = sum(batch durations) - overlap excess + uncovered span`. Overlap excess
is the duration sum minus the union of batch intervals; uncovered span is the
part of the reported span outside that union. This handles runtimes that overlap
disjoint batches, zero-duration timestamps, and batches with no normalized
quantum operations. Group contributions refer to summed durations; they can be
interpreted as contributions to the span only together with the overlap and
uncovered-clock terms.

Source identities use per-role operation occurrence numbers and preserve gadget
origin labels when available. Removing empty moments can renumber operations
without changing which operations share a moment. Neither this source comparison
nor equal runtime operation-kind counts establishes a gate-to-batch mapping or
equality of rotation angles and physical pulses. `runtime_causality_verified`
remains false: the report accounts for observed timing changes but does not
attribute them to a specific source gate or predict logical error rates.

Native propagation uses identity markers only in its
internal analysis DAG; it adds no physical gates or ticks to the selected schedule.

For physical distance checks, select explicit `PauliFault(channel.id, "X")`
branches and call `verify_logical_fault(noisy, faults)`. Duplicate locations,
invalid/zero-probability branches, detected faults, and zero logical effects are
rejected. A two-qubit branch can be written as `"XY"` and still costs one location.
`fault_distance_bounds(noisy, faults, exclude_through=3)` additionally excludes
small undetected combinations using full, undecomposed detector/logical supports.
It reports exactness only when that lower bound meets the replayed upper bound.

The exclusion allows mutually exclusive branches together, a conservative
relaxation for lower bounds. It uses a fixed interior noise rate to prevent tiny
numerical probabilities from disappearing during support extraction. The actual
witness must still have positive probability in the supplied profile. Analysis
requires `0 < p < 0.2`; export also supports both endpoints. The maximum exclusion
weight is three, and an explicit pair-check budget prevents an unbounded search.
See the [paper validation example](paper-color-codes.md#matched-si1000-and-physical-fault-bounds).

### Compare SI1000 fault behavior after rescheduling

Given a physical logical witness `reference_faults` on a reference native-CZ
memory, compare its explicitly scheduled noise model with an alternative:

```python
noisy = SI1000(0.001).apply(plan)
comparison = noisy.compare_schedule(
    reference_faults,
    candidate_policy=SchedulePolicy.COMPACT_UNITARIES,
    exclude_through=2,
)
print(comparison.describe())
report = comparison.to_dict()
noisy_candidate_ticks = comparison.candidate.to_tick_circuit()
```

The default candidate is `ASAP_UNITARIES`. The input must use `REFERENCE` and
the existing physical native-CZ SI1000 alphabet; CX and RELABEL are not silently
lowered or assigned an inferred noise convention. Analysis requires `0 < p < 0.2`.

`remap_operation_faults(noisy, candidate, faults)` transfers operation-attached
Pauli branches using per-role occurrence identities, including repeated gates
without origin labels. It requires the same noise profile and source contract.
Channel IDs contain moment/operation positions and are **not stable across
rescheduling**. Idle and collapse-wait faults are rejected by this mapping: their
locations require an explicit new physical choice. A two-qubit Pauli branch
continues to count as one physical location.

The comparison replays both witnesses with Stim, checks them with PECOS native
propagation and trajectory execution, and computes bounded distance exclusions.
Trajectory checks subtract the ideal affine parity baseline; a deterministic
nonzero parity is not mistaken for a fault. The pair-check budget fails explicitly
when exceeded. An exact bound describes this finite memory and noise-location
convention, not the whole code family.

The report also retains channel counts and native independent-error DEMs, with
added/removed signatures and changed probabilities. `matches_at_p` compares their
aggregated Bernoulli mechanisms at the selected rate (`rel_tol=1e-10`, `abs_tol=0`).
This is not a physical-channel bijection or an all-rate noise-equivalence proof.
Changing noise probabilities need not change the distance bound. Logical error
rates require decoder experiments, and this analysis does not apply the SI1000
model to compiler-lowered Guppy executions.

Use `comparison.explain(top_changes=3, max_channels=5)` to trace changed DEM terms
back to named parities and contributing noise channels. For complete structured
evidence on one entry of `report["independent_dem_comparison"]["changes"]`, use
`comparison.explain_change(index)`. This reconstructs both independent factor
weights and checks them against the stored comparison. Each side uses its own
channel locations; channel IDs are not matched across rescheduling.

An individual support can also be inspected without a logical witness or a
schedule comparison:

```python
explanation = noisy.compile_native().explain_signature(detectors=[0])
print(explanation.describe(max_channels=5))
contributors = explanation.to_dict()
```

Add `observables=[0]` to select a joint detector/logical support, or use only
`observables=[0]` for an undetected logical factor. Indices select the **exact**
nonempty support, not every term that touches a chosen detector. Parity records,
measurement names, XOR constants, and coordinates are retained. Effects are flips
relative to ideal, so affine constants do not alter which branches contribute.

Each contributing channel lists its matching Pauli words, source gadget origin
when available, roles, and two different probabilities. The **local physical
branch mass** sums mutually exclusive matching branches within that one channel,
with other channels absent. The **independent factor probability** belongs to
the equivalent Bernoulli DEM representation; these factors combine by XOR.
Neither summing local masses across channels nor treating a DEM term's weight
as the marginal probability of observing that signature is valid. Other DEM
supports can also combine to produce the same signature.

The structured explanation retains every nonzero converted contributor; display
limits only shorten the text. An absent support or zero-rate model returns no
contributors and weight zero. Explanation accepts the SI1000 export endpoints
`p=0` and `p=0.2`, while physical fault-distance analysis retains its stricter
interior-rate requirement. Ideal parity determinism is checked even at zero rate.

`examples/python_examples/compare_si1000_schedules.py` loads a pinned published
witness, compares both policies, and cross-checks the reference with its unchanged
published circuit. It checks the candidate DEM independently through Stim and
noisy TickCircuit export, then saves the mapped witnesses, bounds, DEMs, schedules,
and noise circuits, including a text explanation of changed terms. See
[the paper comparisons](paper-color-codes.md#matched-si1000-and-physical-fault-bounds).

## Detector coordinates

Attach decoder coordinates directly to a named detector:

```python
program.detector("repeat", first["syndrome"] ^ second["syndrome"], coords=(2, 3, 1, 0))
```

Coordinates are optional finite numerical vectors. They survive compilation,
rescheduling, `plan.dem_spec()` for Guppy, TickCircuit metadata, and Stim/DEM
export. Their axes retain the author's meaning: a time coordinate is not
automatically changed to a physical tick index when a schedule is optimized.
Extra axes can carry labels such as color or Pauli basis.

The native terminal-pairing heuristic uses only the first three coordinates
(missing axes are zero); additional labels do not contribute to its distance.
The existing DEM slice layout requires exactly `(x, y, round)` and explicitly
rejects other dimensions. Generic circuit and DEM export preserve all axes.

## XOR with a constant

Add a literal one to a typed definition, or XOR a named expression with one:

```python
from pecos.qec import Detector, Observable, rec
from pecos.qec.generic import NamedParity

check = Detector("m0", "m1", 1)
logical = Observable(rec[-1], 1, id=0)
named = NamedParity(("first/syndrome", "second/syndrome")) ^ 1
```

`constant=True` and `definition ^ 1` are equivalent alternatives. Literal `0`
does nothing and repeated ones cancel. Integers other than 0/1 are invalid;
use measurement references for indices. These constants specify expressions,
not inferred noiseless values: `Detector("m", 1)` evaluates `m XOR 1` even when
that choice gives one in the absence of noise. It does not certify determinism.

`plan.evaluate(...)`, `plan.evaluate_named(...)`, and the audited Guppy build's
`evaluate_*` methods apply the constants. For native emitted records use:

```python
from pecos.qec.generic import evaluate_tick_circuit

detectors, observables = evaluate_tick_circuit(ticks, raw_record)
```

The native `circuit_definitions()` API returns **support-only** definitions.
The helper combines them with `parity_constant` (integer 0/1) in the circuit's
`detectors`/`observables` JSON metadata. Constants affect raw evaluation but
cancel in fault differences, so the DEM's error mechanisms remain unchanged.
An existing `reference` metadata value is preserved separately; it is not an
instruction to XOR a bit and is not inferred from the constant.

Stim export appends one `MPAD 1` synthetic record only when needed, and adds
that record to signed parities. It adds no physical gate or tick. To evaluate
raw samples from that export, use `plan.evaluate_stim_record(record)`, which
checks/removes the synthetic trailing bit before applying the authored parities.
Canonical Guppy/native measurement IDs and `plan.num_measurements` exclude it.
Stim's detector sampler reports deviations from its own noiseless reference,
which is distinct from evaluating the authored expression on raw records.
Constant-only supports remain unsupported by typed/native DEM definitions.
