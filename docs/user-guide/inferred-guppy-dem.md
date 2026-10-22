# Inferring a DEM from Guppy Outputs

Use `DetectorErrorModel.builder().with_inference(...)` when a Guppy program already emits raw
physical measurements, detector bits, and logical-observable bits with
`output()`, but does not separately expose the measurement parities that
define those detectors and observables. The program does not need to be
edited: PECOS infers the parities, binds them to the runtime QIS trace, and
builds the detector error model (DEM) with native PECOS fault propagation.
Stim is not used.

## Choose the right entry point

| What the application already has | Use |
| --- | --- |
| Raw measurements plus detector and observable **values** computed by Guppy | `DetectorErrorModel.builder().with_inference(...)` (this guide) |
| Audited detector and observable **definitions** using records, measurement IDs, or scalar result tags | [`DetectorErrorModel.from_guppy`](dem-from-guppy.md) |
| An annotated `TickCircuit` | `DetectorErrorModel.from_circuit` |

The inferred-output workflow is especially useful for generated Guppy that
collects measurements into arrays or computes round-to-round parities inside
the program.

## Use inference in the DEM builder

Inference supplies the detector and observable definitions. Noise, idle placement,
sampling and decoding use the same pipeline as an explicit-definition build:

<!--skip: API template requires a caller-supplied Guppy program; covered by test_dem_inference_builder.py.-->
```python
from pecos import NoiseParameters
from pecos.qec import DemInferenceSettings, DetectorErrorModel

inference = (
    DemInferenceSettings()
    .with_raw_tag("raw measurements")
    .with_detector_tag("DETECTOR")
    .with_observable_tags(("obs",))
    .with_provenance_shots(32)
    .with_validation_rows(32)
)

build = (
    DetectorErrorModel.builder()
    .with_program(program)  # Your Guppy program satisfying the contract below.
    .with_qubits(num_qubits)
    .with_inference(inference)
    .with_seed(7)
    .with_noise(NoiseParameters().with_p1(0.001).with_p2(0.005))
    .with_idle_after_2q(1.0)
    .build()
)
dem = build.dem
report = build.inference
```

`DemInferenceSettings()` is sufficient for the default tags. Every `with_*`
method returns a new settings value; keep the returned value when configuring
settings outside a chain. `.with_probe_shots(n)` overrides automatic sizing.
Seed and runtime belong to the DEM builder and apply to tracing and probing.
Inference settings can accompany explicit definitions. An unspecified role is
inferred using the settings' default tags; an explicitly empty list means none.
`with_num_measurements()` remains incompatible with inference settings.

The returned `GuppyDemBuild` supports the same `evaluate_runtime_record`,
`evaluate_measurements`, and `evaluate_result_columns` methods as an explicit
build. The last method evaluates the bound raw output columns, including arrays
and reordered outputs. Pass columns from one simulation run. Its audit includes
`inference`, with probe counts, the raw measurement binding, and schedule evidence.
Explicit builds have `build.inference is None`.

Idle rates use the durations in the resulting circuit. With the default simple
runtime, explicitly insert idles when requesting idle noise; the unified builder
rejects idle-noise settings when there are no idle gates. Use
`.with_idle_after_2q(duration)` for a uniform post-two-qubit convention, or
`.with_runtime(runtime)` for runtime-provided scheduling. The former inserts
noise locations on the two operands; it does not set physical gate times.
See [rates, probabilities, and time units](dem-from-guppy.md#rates-probabilities-and-time-units)
before copying rates from a shot-simulation configuration. In particular,
`.with_idle_after_2q(1.0)` uses unit-duration idles, while runtime-emitted idles
are replayed in nanoseconds.

The low-level `infer_guppy_dem_annotations` helper remains available when you
want to inspect the inferred annotations without building a DEM; the examples
below demonstrate that interface.

## Mix explicit and inferred definitions

Each detector or observable can use explicit references or select an output to
infer. Settings control the inference algorithm; the definitions select its
inputs. `element` indexes the tag's flattened per-shot output: repeated calls
are concatenated in execution order, with array elements in array order.

<!--skip: API template requires caller-supplied program; covered by test_dem_inference_builder.py.-->
```python
from pecos.qec import Detector, Observable, rec

build = (
    DetectorErrorModel.builder()
    .with_program(program)
    .with_qubits(num_qubits)
    .with_detectors(
        [
            Detector(rec[-2], rec[-1]),
            Detector.inferred("checks", element=3),
        ]
    )
    .with_observables(
        [
            Observable.inferred("logical", element=0, id=0),
            Observable(rec[-1], id=1),
        ]
    )
    .with_inference(DemInferenceSettings().with_validation_rows(32))
    .build()
)
```

An inferred definition activates default settings if `.with_inference(...)` is
omitted. In that case, omitted roles are empty; they do not request default tags.
Supplying `.with_inference(...)` additionally enables the all-output shorthand
for unspecified roles. Supplied role lists always take precedence, regardless
of setter order. IDs default to list positions; detector IDs must be unique and
dense, and observable IDs must be unique.

Only selected outputs are fitted or required. Fully explicit lists skip probes,
raw-output binding and the inference solver, even if settings are supplied.
Inference-only builds skip compiler named-reference analysis. Mixed definitions
using `rec[...]` also avoid that analysis; explicit named references retain the
existing compiler certification requirements. All inferred outputs share probes
and one GF(2) solve, and the combined definitions feed one fault-propagation pass.
Explicit JSON definitions can also be combined with an inferred opposite role.

`build.audit["definition_origins"]` records each final ID as explicit or inferred,
including the selected tag and element. Inferred entries share the build's
empirical inference evidence. Selecting fewer outputs may still require the same
number of probes to establish rank over the raw measurements.

## Rust settings and circuit/probe builds

Rust exposes the same settings and inference algorithms through `pecos_qec`.
For an already captured circuit and bound coin-toss probes:

<!--skip: API template requires caller-supplied circuit and probes; covered by dem_inference_builder.rs.-->
```rust
use pecos_qec::{DemInferenceSettings, DetectorErrorModel, NoiseConfig};

let settings = DemInferenceSettings::default()
    .with_raw_tag("raw measurements")
    .with_detector_tag("DETECTOR")
    .with_observable_tags(["obs"])
    .with_provenance_shots(32)
    .with_validation_rows(32);

let build = DetectorErrorModel::builder()
    .with_circuit(circuit)
    .with_inference(settings)
    .with_inference_data(probes)
    .with_noise(NoiseConfig::new(0.001, 0.005, 0.001, 0.001))
    .with_idle_after_2q(1.0)
    .build()?;
let shots = build.evaluate_result_columns(&columns)?;
```

`probes` is a `DemInferenceData` containing `raw_measurement_ids`, `raw_rows`,
`detector_rows`, and `observable_rows`. IDs follow raw-output column order and
must cover every measurement in the circuit exactly once. Each row contains
`u8` bits. Supplying bound data is a trusted boundary: the caller establishes
measurement provenance and a fixed quantum schedule. The Rust
`correlate_measurement_ids` function is available for empirical binding.

Rust program execution connects through the `DemProgram` trait and
`.with_program(adapter).with_qubits(n).with_seed(seed)`. The adapter owns its
runtime and receives the inference settings before execution. The built-in
`SeleneDemProgram` accepts QIS LLVM IR with the `pecos-qec/selene` feature and
HUGR envelopes with `pecos-qec/hugr`. Through the `pecos` facade these features
are named `qec-selene` and `qec-hugr`. Circuit/probe builds require neither
feature. Native HUGR lowering requires LLVM 21 at build time and uses a pinned
compiler revision compatible with the workspace's HUGR version; its Python
bindings are disabled. QIS execution requires the usual PECOS QIS FFI and Selene
runtime libraries and LLVM tools. All Rust paths run without Python.

<!--skip: API template requires caller-supplied HUGR bytes and native runtime; covered by dem_selene_builder.rs.-->
```rust
use pecos_qec::{DemInferenceSettings, DetectorErrorModel, SeleneDemProgram};

let build = DetectorErrorModel::builder()
    .with_program(SeleneDemProgram::from_hugr(hugr_bytes))
    .with_qubits(7)
    .with_seed(42)
    .with_inference(DemInferenceSettings::default().with_validation_rows(32))
    .build()?;
```

For QIS use `SeleneDemProgram::new(pecos_programs::Qis::from_string(llvm_ir))`.
The QIS adapter currently accepts IR text, not bitcode. The default runtime is
Selene simple; `.with_runtime(runtime)` selects another configured
`pecos_qis::SeleneRuntime`. Runtime initialization is deferred until building.
Explicit Rust builds use `.with_detectors(...)` and `.with_observables(...)`
with stable measurement-ID supports, or JSON setters.

The native adapter replays the runtime-lowered trace, preserving stable
measurement IDs, metadata, and idle durations. It requires complete traces and
compares the quantum operation schedule across all sampled shots. **This is an
empirical check, not a compiler proof that every possible execution has the
same schedule.** Rare measurement-dependent paths may escape sampling; supply
programs known to have a fixed quantum schedule. The circuit records this as
`dem_schedule_validation = "empirical_trace_comparison"` and records the number
of additional checked shots in `dem_schedule_validation_shots`.

Explicit program builds perform 32 additional schedule-check shots by default,
configurable with `SeleneDemProgram::with_schedule_validation_shots(n)`; `n` must
be positive. They require no raw-output tags and perform no binding or parity inference.
Inferred builds reuse their inference and provenance shots for schedule
validation. Additional provenance shots run only when raw outputs lack direct
runtime IDs and provenance checking is enabled. Inferred builds record
`dem_raw_measurement_binding` as `runtime_result_ids`,
`probe_correlated_result_ids`, or `assumed_canonical_result_order`. A failed
rank, binding, or schedule check stops the build. Only requested output tags
are inspected; their widths must remain fixed across inference shots.

Rust supports mixed lists through typed sources:

<!--skip: API template requires caller-supplied adapter; covered by dem_inference_builder.rs.-->
```rust
use pecos_qec::DemDefinition;

let build = DetectorErrorModel::builder()
    .with_program(adapter)
    .with_qubits(7)
    .with_detector_definitions(vec![
        DemDefinition::explicit(0, vec![2, 3]).with_coords([0.0, 0.0, 0.0]),
        DemDefinition::inferred(1, "checks", 3)
            .with_coords([1.0, 0.0, 0.0])
            .with_metadata("label", "parity check"),
    ])
    .with_observable_definitions(vec![
        DemDefinition::inferred(0, "logical", 0),
    ])
    .with_inference(DemInferenceSettings::default().with_validation_rows(32))
    .build()?;
```

Both definition sources support `.with_coords([x, y, t])` for detectors and
`.with_metadata(key, value)` for custom JSON fields. Metadata is retained in
`build.circuit`; known fields such as labels and observable references also
appear in `build.definitions`. Coordinates must be finite. Identity, support,
coordinates, and inference provenance keys are reserved in custom metadata;
invalid settings fail before running a program.

`DemProgramRequest` supplies the required `detector_outputs` and
`observable_outputs`. `Some([])` skips a role; `Some(selectors)` requests exactly
those columns in list order; `None` requests all outputs under that role's default
tags. The adapter returns only selected inferred columns, not explicit definition
columns. With no inferred definitions, `request.inference` is `None`, even when
the caller supplied settings. No parity or provenance probes should be generated in that case; runtime
schedule validation is a separate responsibility.
Circuit/probe builds follow the same selected-column contract. They use empty
rows for an unrequested role; the builder ignores any supplied values for that
role. Existing circuit annotations must be supplied as builder definitions when
mixing with inference, so they cannot silently collide with inferred definitions.

`build.definition_origins` records explicit, selected inferred, or default-tag
inferred origins. The Selene adapter resolves default tags into exact
`Inferred(selector)` origins, including each tag and element. Its definition JSON
matches Python: default-expanded detectors carry `inferred_from_result_tag`, and
observables also carry `result_element`. Probe data without resolved selectors
(supplied through `with_inference_data`, or from an adapter that leaves
`DemBuildInput::resolved_detector_outputs`/`resolved_observable_outputs` as
`None`) has no per-tag widths, so it records `InferredDefaults` columns in the
concatenation of the default tags.

## Program contract

Before calling the tool, check all of the following:

1. Every physical measurement is emitted exactly once through one or more
   `output(raw_tag, ...)` calls. Do not omit initialization, syndrome, flag,
   postselection, or final-readout measurements.
2. Selected detector and observable outputs are Boolean XOR parities of those raw
   measurements. AND, OR, nonlinear expressions, constant-one offsets, and
   outputs independent of every measurement are rejected.
3. Measurement values may affect classical parity calculations, but must not
   change the quantum gate schedule. Measurement-dependent quantum branches
   and repeated-until-success loops need a different analysis.
4. The supplied `num_qubits` is large enough to run the program through the
   selected Selene runtime.

Tag names are case-sensitive. The defaults are `"raw measurements"`,
`"DETECTOR"`, and `"obs"`; all are configurable. Repeated calls with the same
tag are concatenated in execution order. An array-valued call contributes its
elements in array order.

## Quick start

This two-measurement example is the smallest complete workflow. Strict
provenance determines the physical identity of each raw output automatically.

<!--test-name: inferred_guppy_dem_quick_start-->
```python
from guppylang import guppy
from guppylang.std.builtins import output
from guppylang.std.quantum import measure, qubit

from pecos.qec import infer_guppy_dem_annotations


@guppy
def parity_readout() -> None:
    m0 = measure(qubit()).read()
    m1 = measure(qubit()).read()

    # Emit each physical result exactly once under the raw tag.
    output("raw measurements", m0)
    output("raw measurements", m1)

    # These are computed values, not additional measurements.
    output("DETECTOR", m0 ^ m1)
    output("obs", m0)


inferred = infer_guppy_dem_annotations(
    parity_readout,
    num_qubits=2,
    seed=7,
)

assert inferred.raw_measurement_ids == (0, 1)
assert inferred.detector_supports == ((0, 1),)
assert inferred.observable_supports == ((0,),)
assert inferred.raw_binding in {
    "runtime_result_ids",
    "probe_correlated_result_ids",
}

dem = inferred.build_dem(
    p1=0.001,
    p2=0.005,
    p_meas=0.005,
    p_prep=0.001,
)
assert dem.num_detectors == 1
assert dem.num_observables == 1
print(dem.to_string())
```

An inferred DEM is an ordinary `DetectorErrorModel`, so it samples and decodes
exactly like one built from explicit detectors:

<!--continuation-->
```python
from pecos.decoders import bp_osd

batch = dem.to_sampler().sample_batch(500, seed=3)
result = batch.decode(dem.to_string(), bp_osd())

assert result.num_shots == 500
print(f"logical errors: {result.num_errors} ({result.execution_path})")
```

`bp_osd()` consumes the raw model directly. Matching-style decoders need a
graph-like projection instead — see
[Decoders](decoders.md#hyperedge-models-and-matching-decoders) — and the
experimental Frontier and BP-Trellis decoders additionally report a per-shot
confidence gap; see [Experimental Decoders](../experimental/decoders.md).

The two accepted `raw_binding` values are both identity-preserving. A compiler
may retain a directly tagged measurement ID, or it may erase the ID while
duplicating the value into raw and computed outputs; strict probe correlation
handles either lowering.

The returned supports contain QIS `MeasId` values. PECOS writes equivalent
`meas_ids` entries into `inferred.detectors_json` and
`inferred.observables_json`, attaches them to `inferred.circuit`, and passes
the annotated circuit to `DetectorErrorModel.from_circuit` when `build_dem`
is called.

For an existing program, the integration itself is only this call:

<!--skip: illustrative fragment uses an application-provided program and qubit count-->
```python
inferred = infer_guppy_dem_annotations(
    existing_guppy_program,
    num_qubits=program_qubit_count,
    raw_tag="raw measurements",
    detector_tag="DETECTOR",
    observable_tags=("obs",),
)
dem = inferred.build_dem(p1=0.001, p2=0.005, p_meas=0.005, p_prep=0.001)
```

No separate trace-capture call is required. Python delegates coin-toss capture,
trace validation, measurement binding, and circuit replay to the shared Rust
Selene analysis. Python then validates hosted-operation order when requested,
normalizes the circuit, and fits the parities with the Rust solver.

## Example: rounds and aggregate arrays

Real memory experiments commonly keep earlier syndromes, emit measurements in
arrays, and form final-boundary detectors from the last syndrome and data
readout. This two-data-qubit repetition memory demonstrates that pattern with
four physical measurements.

<!--test-name: inferred_guppy_dem_round_arrays-->
```python
from guppylang import guppy
from guppylang.std.builtins import array, output
from guppylang.std.quantum import cx, measure, qubit

from pecos.qec import infer_guppy_dem_annotations


@guppy
def repetition_memory() -> None:
    d0, d1 = qubit(), qubit()

    a0 = qubit()
    cx(d0, a0)
    cx(d1, a0)
    s0 = measure(a0).read()
    output("DETECTOR", s0)

    a1 = qubit()
    cx(d0, a1)
    cx(d1, a1)
    s1 = measure(a1).read()
    output("DETECTOR", s0 ^ s1)

    m0, m1 = measure(d0).read(), measure(d1).read()
    output("DETECTOR", s1 ^ m0 ^ m1)
    output("obs", m0)

    # Raw arrays can be emitted after the parities have been computed.
    output("raw measurements", array(s0, s1))
    output("raw measurements", array(m0, m1))


inferred = infer_guppy_dem_annotations(
    repetition_memory,
    num_qubits=4,
    probe_shots=64,
    provenance_shots=32,
    validation_rows=16,
    seed=11,
)

assert inferred.raw_measurement_ids == (0, 1, 2, 3)
assert inferred.detector_supports == (
    (0,),
    (0, 1),
    (1, 2, 3),
)
assert inferred.observable_supports == ((2,),)
assert inferred.raw_binding == "probe_correlated_result_ids"

dem = inferred.build_dem(p1=0.001, p2=0.005, p_meas=0.005, p_prep=0.001)
assert dem.num_detectors == 3
assert dem.num_observables == 1
```

Array indexing, copying, and aggregation can erase element-level result IDs in
the compiled path. In strict mode, PECOS recovers them by correlating each raw
output column with result-ID-keyed physical outcomes over independent
coin-toss traces. `probe_correlated_result_ids` records that stronger binding.

## Example: raw outputs in a different order

Do not assume raw-array order is physical measurement order. Strict provenance
tracks the identity of each element even when an array is reordered.

<!--test-name: inferred_guppy_dem_reordered_array-->
```python
from guppylang import guppy
from guppylang.std.builtins import array, output
from guppylang.std.quantum import measure, qubit

from pecos.qec import infer_guppy_dem_annotations


@guppy
def reordered_readout() -> None:
    m0 = measure(qubit()).read()
    m1 = measure(qubit()).read()
    m2 = measure(qubit()).read()
    output("DETECTOR", m2 ^ m0)
    output("obs", m1)
    output("raw measurements", array(m2, m0, m1))


inferred = infer_guppy_dem_annotations(
    reordered_readout,
    num_qubits=3,
    probe_shots=32,
    provenance_shots=24,
    validation_rows=8,
    seed=13,
)

# Array order is m2, m0, m1; the IDs preserve physical identity.
assert inferred.raw_measurement_ids == (2, 0, 1)
assert inferred.detector_supports == ((2, 0),)
assert inferred.observable_supports == ((1,),)
assert inferred.raw_binding == "probe_correlated_result_ids"
```

This distinction matters whenever an intermediate array is assembled in an
order that differs from the runtime measurement record.

## Example: custom tags and several observables

Pass every logical-result tag in the desired DEM observable order. An
array-valued observable expands into one DEM observable per element.

<!--test-name: inferred_guppy_dem_custom_tags-->
```python
from guppylang import guppy
from guppylang.std.builtins import array, output
from guppylang.std.quantum import collect_measurements, measure_array, qubit

from pecos.qec import infer_guppy_dem_annotations


@guppy
def tagged_readout() -> None:
    bits = collect_measurements(measure_array(array(qubit(), qubit(), qubit())))
    m0 = bits[0]
    m1 = bits[1]
    m2 = bits[2]
    output("physical", bits)
    output("events", m0 ^ m1)
    output("logical_z", m0)
    output("logical_x", array(m1, m2))


inferred = infer_guppy_dem_annotations(
    tagged_readout,
    num_qubits=3,
    raw_tag="physical",
    detector_tag="events",
    observable_tags=("logical_z", "logical_x"),
    probe_shots=32,
    provenance_shots=24,
    validation_rows=8,
    seed=17,
)

assert inferred.detector_supports == ((0, 1),)
assert inferred.observable_supports == ((0,), (1,), (2,))
assert inferred.observable_labels == (
    ("logical_z", 0),
    ("logical_x", 0),
    ("logical_x", 1),
)
```

## Measurement identity modes

Leave `require_raw_provenance=True`, the default, whenever possible:

| `raw_binding` | Meaning |
| --- | --- |
| `runtime_result_ids` | Every raw output retained its QIS measurement ID directly. |
| `probe_correlated_result_ids` | PECOS recovered a unique complete ID mapping from independent probe signatures. |
| `assumed_canonical_result_order` | Strict identity was disabled and PECOS used positional order. |

Both APIs check the circuit-producing schedule against every correlation and
fitting shot. When direct runtime IDs bind the raw outputs, they reuse fitting
shots for schedule validation and skip correlation shots. With strict provenance
enabled, the raw-to-physical binding is also checked on every fitted row. Python's
`inference.audit` records the actual `provenance_shots` used (zero for direct
bindings) and the total `schedule_validation_shots`, excluding the baseline.
Its `schedule_validation` uses the circuit's `dem_schedule_validation` value,
`empirical_trace_comparison`; the circuit records the binding as
`dem_raw_measurement_binding`.
Correlation fails loudly if the quantum schedule changes, a raw element is a
computed value instead of a direct measurement, a measurement is omitted or
duplicated, or two physical signatures collide. Increasing
`provenance_shots` resolves a rare signature collision; it cannot repair an
incomplete or computed raw record.

The weak fallback is explicit:

<!--skip: illustrative fragment uses an application-provided program-->
```python
inferred = infer_guppy_dem_annotations(
    program,
    num_qubits=7,
    require_raw_provenance=False,
)
assert inferred.raw_binding == "assumed_canonical_result_order"
```

Use it only when the application independently guarantees that the
concatenated raw values occur exactly once each in physical measurement order.
It checks the measurement count, but it cannot detect a permutation.

## Parameters and outputs

`infer_guppy_dem_annotations` accepts:

| Argument | Default | Purpose |
| --- | --- | --- |
| `program` | required | Compiled or compilable Guppy entry point. |
| `num_qubits` | required keyword | Runtime qubit capacity. |
| `raw_tag` | `"raw measurements"` | Tag containing every physical measurement exactly once. |
| `detector_tag` | `"DETECTOR"` | Tag containing all computed detection-event bits. |
| `observable_tags` | `("obs",)` | Logical result tags, in DEM observable order. |
| `probe_shots` | derived | Coin-toss rows used to infer and validate affine parities. Defaults to one row per raw measurement plus the affine constant, the validation rows, and a rank margin. |
| `provenance_shots` | `32` | Rows used to correlate array elements with QIS IDs. |
| `validation_rows` | `32` | Extra probe rows required beyond the square system, as additional consistency constraints on the fit. |
| `seed` | `0` | Reproducible trace and probe seed. |
| `runtime` | `None` | Selene runtime selection forwarded to PECOS. |
| `require_raw_provenance` | `True` | Require a complete identity-preserving raw-measurement binding. |

The result is an `InferredGuppyDemAnnotations` with:

| Attribute | Contents |
| --- | --- |
| `circuit` | Runtime-lowered `TickCircuit` with detector and observable metadata attached. |
| `detectors_json`, `observables_json` | Serialized definitions using QIS `meas_ids`. |
| `raw_measurement_ids` | One physical ID per raw output element, in emitted order. |
| `detector_supports`, `observable_supports` | Inferred parity supports expressed as physical IDs. |
| `observable_labels` | `(tag, element_index)` for each DEM observable. |
| `raw_binding` | Identity mode from the table above. |
| `probe_shots` | Number of parity-inference rows used. |
| `build_dem(**noise)` | Build a PECOS `DetectorErrorModel` from the annotated circuit. |

`probe_shots` must provide at least one row per raw measurement, one affine
constant column, and the requested validation rows. Left unset it is derived
from the traced circuit, so a larger experiment no longer fails for want of a
bigger number; pass an explicit count only to override that.

**More probe shots do not make a better DEM.** Inference solves an affine
system over GF(2): once the probe matrix reaches full rank the solution is exact
and unique, and further rows can only re-validate it. The derived default adds a
margin because probes are random -- a square system of exactly one row per
unknown is full rank only about 29% of the time, and each row beyond it halves
the shortfall probability, so with the validation rows and margin included a
rank failure is vanishingly unlikely. Elimination uses every probe row;
validation rows are additional consistency constraints on the same fit, not a
held-out set. Raise `probe_shots` to buy validation confidence against a
mis-inferred parity, not resolution. Note that the derived floor grows with
the raw measurement count and the GF(2) elimination is quadratic in it, so a
program with many thousands of raw measurements pays a noticeable inference
cost where it previously failed fast asking for more shots.

## Failure guide

| Error or symptom | Meaning | Action |
| --- | --- | --- |
| `missing required tag(s)` | A configured tag was never emitted. | Check spelling and case, or pass the matching tag arguments. |
| `must expose every physical measurement exactly once` | The raw stream omitted or duplicated a measurement. | Include initialization, syndrome, postselection, and final-readout measurements exactly once. |
| `not a direct physical measurement` | A raw element is computed from measurements. | Emit the original measurement value; keep computed values under detector/observable tags. |
| `signatures are ambiguous` | Too few provenance probes caused an identity collision. | Increase `provenance_shots` or change `seed`. |
| `not affine` or `constant-one` | An output is not a representable XOR parity. | Replace nonlinear/offset post-processing with explicit parity outputs, or provide audited static definitions. |
| `quantum operation schedule changed` | A measurement changed which quantum operations ran. | Build separate justified models per static path; do not use a single inferred DEM. |
| Native fault propagation rejects a gate such as `T` | The traced circuit is outside PECOS's Pauli/Clifford propagation support. | Supply a separately justified Clifford model or use a different analysis. |

## Soundness and leakage boundaries

The prototype establishes two empirical facts:

1. Emitted detector and observable bits fit unique affine GF(2) parities of
   the raw physical measurements, including independent validation rows.
2. The raw measurements bind to one runtime-lowered QIS circuit trace.

It is an empirical certificate, not a compiler proof over every possible
classical-control path. The probability of accidental probe-signature
collisions decreases exponentially with the number of probes.

The QIS tracer safely preserves leakage-aware measurement outcomes `0`, `1`,
and `2`, including Guppy's `is_leaked()` check. Parity inference itself is
Boolean and `build_dem` performs Pauli fault propagation, so the returned DEM
represents the accepted no-leakage path. It does **not** model leakage rate,
postselection probability, or rejected-shot behavior. A leakage check that
changes later quantum operations also violates the static-schedule contract.

Repeated-until-success loops are outside this workflow because different
attempt counts produce different quantum schedules. Non-Clifford protocols,
including magic-state preparation or cultivation containing native `T` gates,
may have detector parities that can be inferred, but PECOS will reject DEM
construction unless the traced circuit has a separately justified supported
fault-propagation model.

When definitions are already known rather than computed only as outputs, use
the audited typed workflow in [Detector Error Models from Guppy
Programs](dem-from-guppy.md).
