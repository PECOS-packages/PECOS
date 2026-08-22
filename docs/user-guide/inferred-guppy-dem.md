# Inferring a DEM from Guppy Outputs

Use `infer_guppy_dem_annotations` when a Guppy program already emits raw
physical measurements, detector bits, and logical-observable bits with
`output()`, but does not separately expose the measurement parities that
define those detectors and observables. The program does not need to be
edited: PECOS infers the parities, binds them to the runtime QIS trace, and
builds the detector error model (DEM) with native PECOS fault propagation.
Stim is not used.

## Choose the right entry point

| What the application already has | Use |
| --- | --- |
| Raw measurements plus detector and observable **values** computed by Guppy | `infer_guppy_dem_annotations` (this guide) |
| Audited detector and observable **definitions** using records, measurement IDs, or scalar result tags | [`DetectorErrorModel.from_guppy`](dem-from-guppy.md) |
| An annotated `TickCircuit` | `DetectorErrorModel.from_circuit` |

The inferred-output workflow is especially useful for generated Guppy that
collects measurements into arrays or computes round-to-round parities inside
the program.

## Program contract

Before calling the tool, check all of the following:

1. Every physical measurement is emitted exactly once through one or more
   `output(raw_tag, ...)` calls. Do not omit initialization, syndrome, flag,
   postselection, or final-readout measurements.
2. Detector and observable outputs are Boolean XOR parities of those raw
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

No separate trace-capture call is required. The function runs the coin-toss
probes and captures the QIS trace internally.

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
