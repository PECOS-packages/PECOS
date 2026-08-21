# Detector Error Models from Guppy Programs

This guide covers `DetectorErrorModel.from_guppy`, which builds a
circuit-level detector error model (DEM) from a Guppy program by tracing it
through the Selene QIS engine. Use this entry point when detector and
observable definitions are already available as records, measurement IDs, or
scalar measurement-result tags.

If the program has only computed detector and observable **values**, choose the
inferred-output workflow instead:

| Available information | Entry point |
| --- | --- |
| Static parity definitions | `DetectorErrorModel.from_guppy` (this guide) |
| Raw measurements plus computed detector/observable outputs | [`infer_guppy_dem_annotations`](inferred-guppy-dem.md) |
| An annotated `TickCircuit` | `DetectorErrorModel.from_circuit` |

The inferred-output workflow leaves an existing Guppy program unchanged and
recovers the missing static definitions from its `result()` outputs.

## What You'll Learn

- Building a DEM from a hand-written Guppy program
- Referencing measurements with `records`, `meas_ids`, and `result_tags`
- Building a DEM for a generated surface-code memory experiment
- Sampling and decoding from the resulting DEM
- Adding explicit idle gates so idle-noise parameters take effect
- Exporting native Stim DEM text and graph-like projections
- Comparing PyMatching, Tesseract, and BP-OSD on the same samples
- Choosing the Selene runtime and understanding the limitations

## Overview

`DetectorErrorModel.from_guppy(program, ...)` runs the Guppy program once,
ideally (noise-free), under the Selene QIS engine with operation tracing,
replays the captured gate stream into a PECOS `TickCircuit`, attaches your
detector/observable definitions, and builds the DEM with native PECOS fault
propagation.

The program is compiled Guppy → HUGR → QIS before execution, and the trace
is captured at the QIS boundary. The DEM therefore models the circuit **as
the Selene runtime actually scheduled and lowered it**, not the abstract
circuit you wrote. This matters: abstract-circuit DEMs can significantly
misestimate logical error rates compared to DEMs built from the
runtime-lowered circuit.

## Quick Start: A Repetition-Code Round

One round of the 3-qubit repetition code, with detectors referenced by
Guppy `result()` tags:

<!--test-name: dem_from_guppy_repetition_code-->
```python
from guppylang import guppy
from guppylang.std.builtins import result
from guppylang.std.quantum import cx, measure, qubit

from pecos.qec import DetectorErrorModel


@guppy
def rep_code_round() -> None:
    d0, d1, d2 = qubit(), qubit(), qubit()
    a0, a1 = qubit(), qubit()
    # Z-type parity checks: (d0, d1) and (d1, d2)
    cx(d0, a0)
    cx(d1, a0)
    cx(d1, a1)
    cx(d2, a1)
    result("s0", measure(a0).read())
    result("s1", measure(a1).read())
    # Deferred tagging: bind now, tag later -- the tag->measurement
    # binding follows dataflow, not source order.
    m0 = measure(d0).read()
    m1 = measure(d1).read()
    m2 = measure(d2).read()
    result("m2", m2)
    result("m0", m0)
    result("m1", m1)


dem = DetectorErrorModel.from_guppy(
    rep_code_round,
    num_qubits=5,
    detectors_json="""[
        {"id": "D0", "result_tags": ["s0"]},
        {"id": "D1", "result_tags": ["s1"]},
        {"id": "D2", "result_tags": ["s0", "m0", "m1"]},
        {"id": "D3", "result_tags": ["s1", "m1", "m2"]}
    ]""",
    observables_json='[{"id": "L0", "result_tags": ["m0"]}]',
    p1=0.001,
    p2=0.005,
    p_meas=0.005,
    p_prep=0.005,
    seed=0,
)

assert dem.num_detectors == 4
assert dem.num_observables == 1
print(dem.to_string())
```

Each detector is the **parity** of the measurements it references:
`D0`/`D1` fire when a syndrome measurement flips, and `D2`/`D3` compare each
syndrome against the corresponding data-readout parity.

## Referencing Measurements

Detector and observable definitions are JSON lists. Each entry is
`{"id": ..., <references>}`, where `id` may be a bare integer or the
DEM-label form `"D0"` / `"L0"`. Three reference forms are supported
(provide one; co-presence is allowed only when the forms resolve to the
same measurements):

- `"records": [-1, -5]` — negative positional offsets into the traced
  measurement record (Stim convention). Works for any *accepted* program,
  including the built-in generators' runtime loops; note that generic
  (non-generator) programs with loops or branches are rejected outright by
  the static-schedule guard, whatever reference form they use.
- `"meas_ids": [0, 4]` — stable stamped `MeasId`s, resolved against the
  traced circuit, robust to measurement reordering.
- `"result_tags": ["s0", "m1"]` — Guppy `result(tag, ...)` names, recovered
  structurally from the compiled HUGR. The tag→measurement binding follows
  **dataflow, not syntax**: `m = measure(q).read()` followed later by
  `result("tag", m)` is fully supported, in any order. Restrictions: the
  tagged value must be a raw scalar measurement (computed, constant, and
  array-valued results are rejected), and programs with runtime loops must
  use `records`/`meas_ids` instead (each loop body has one static measure
  op in the HUGR, not one per iteration — `result_tags` fails loudly).

A practical split: `result_tags` for small straight-line programs,
`records`/`meas_ids` for round-looped QEC programs.

Note that the negative-offset spelling appears in three places with three
reference frames: `"records"` here indexes the **traced runtime record**
(Stim convention); `rec[-k]` in `build_dem_from_guppy` indexes the
**canonical Guppy result-id stream** (before runtime scheduling); and the
surface builder's metadata `records` index its **abstract circuit order**.
They coincide only when the runtime preserves canonical order — when moving
a detector spec between entry points under a reordering runtime, re-derive
the offsets rather than copying them.

## Surface-Code Memory DEM

For generated QEC programs (see [QEC with Guppy](qec-guppy.md)), the
surface-code circuit builder already produces matching detector/observable
metadata, so you do not need to author it by hand:

<!--test-name: dem_from_guppy_surface_memory-->
```python
from pecos.guppy_gen import get_num_qubits, make_surface_code
from pecos.qec import DetectorErrorModel
from pecos.qec.surface import SurfacePatch
from pecos.qec.surface.circuit_builder import generate_tick_circuit_from_patch
from pecos_rslib.decoders import PyMatchingDecoder

# The abstract surface builder provides the detector/observable metadata.
patch = SurfacePatch.create(distance=3)
meta_tc = generate_tick_circuit_from_patch(patch, num_rounds=3, basis="Z")

# The DEM itself is built by tracing the generated Guppy program through
# the Selene QIS engine.
dem = DetectorErrorModel.from_guppy(
    make_surface_code(distance=3, num_rounds=3, basis="Z"),
    num_qubits=get_num_qubits(3),
    detectors_json=meta_tc.get_meta("detectors"),
    observables_json=meta_tc.get_meta("observables"),
    num_measurements=int(meta_tc.get_meta("num_measurements")),
    p1=0.005,
    p2=0.005,
    p_meas=0.005,
    p_prep=0.005,
)
assert dem.num_detectors == 28
assert dem.num_observables == 1

# Sample syndromes/observables and decode them, all PECOS-native.
sampler = dem.to_sampler()
batch = sampler.sample_batch(1000, 0)
assert batch.num_shots == 1000

decoder = PyMatchingDecoder.from_dem(dem.to_string_decomposed())
errors = 0
for shot in range(batch.num_shots):
    predicted = decoder.decode_syndrome(batch.get_syndrome(shot)).observable_flips[0]
    actual = batch.get_observable_flips(shot)[0]
    errors += predicted != actual
print(f"logical error rate: {errors / batch.num_shots:.4f}")
```

The DEM built this way is identical to the reference DEM produced by the
surface traced-QIS pipeline — the abstract builder's metadata and the
traced Guppy program agree on measurement order.

## Sampling and Comparing Decoders with `SampleBatch`

`DemSampler.sample_batch` and `ParsedDem.sample_batch` return a
`SampleBatch`. The detector events and observable flips remain in Rust memory,
so the same shots can be passed to several decoders without copying them
through Python. When Python data is needed, `detector_events()` and
`observable_flips()` return shots-major `list[list[bool]]` values.

<!--test-name: dem_sample_batch_workflow-->
```python
from pecos.decoders import pymatching
from pecos_rslib.qec import ParsedDem, SampleBatch

dem_text = "error(0.1) D0 L0"
sampler = ParsedDem.from_string(dem_text).to_dem_sampler()
batch = sampler.sample_batch(32, seed=42)

assert isinstance(batch, SampleBatch)
assert batch.num_shots == 32

# Materialize shots-major Python data only when it is needed.
detector_events = batch.detector_events()
observable_flips = batch.observable_flips()
assert len(detector_events) == len(observable_flips) == 32
assert all(len(shot) == 1 for shot in detector_events)
assert all(len(shot) == 1 for shot in observable_flips)

# Request counts, individual predictions, and timings from one entry point.
result = batch.decode(
    dem_text,
    pymatching(correlated=True),
    predictions=True,
    timing=True,
)
error_count = result.num_errors
predictions = result.predictions
stats = result.stats

assert 0 <= error_count <= batch.num_shots
assert predictions is not None
assert len(predictions) == batch.num_shots
assert stats is not None
assert stats.num_shots == batch.num_shots
```

`batch.decode(...)` returns a `DecodeResult`: `.num_errors` and
`.logical_error_rate` hold aggregates, optional `.predictions` preserves shot
order, and optional `.stats` holds timing statistics. With the default
`workers=None`, the execution planner selects the native-batch, sequential, or
parallel path. Pass `workers=N` to request a specific worker count and inspect
`.execution_path` to see which path ran. A former raw-list call such as
`detectors, observables = sampler.sample_batch(...)` becomes a batch call
followed by the two bulk accessors shown above.

## Choosing the Selene Runtime

`from_guppy(..., runtime=...)` forwards to `pecos.selene_engine(runtime)`,
which accepts:

- `None` — the default `selene_simple_runtime`
- a built runtime name (string)
- a shared-library path to a runtime plugin
- any Selene runtime plugin object exposing the standard protocol
  (`library_file`, optional `get_init_args()`, `library_search_dirs`)

Because the trace records the runtime-lowered QIS operation stream, a
runtime that schedules or lowers differently produces a (correctly)
different DEM.

## Grouping Noise Parameters

Both Guppy DEM entry points accept either the existing flat noise keywords or
one `NoiseParameters` instance containing the complete noise configuration.
`NoiseParameters` is available from the `pecos` top level, and supports both
its dataclass constructor and immutable family/setter chaining.
The grouped and flat forms below are equivalent. Do not mix them in one call:
even explicitly passing a flat parameter at its default value conflicts with
`noise`. When `noise` is present, its defaults fully replace the entry point's
defaults — `NoiseParameters().p1` is `0.0`, not the flat `p1=0.001` default.

Each `with_<field_name>` returns a new `NoiseParameters`, so chains never mutate
the object they start from. The idle families are the one exception to the
one-method-per-field rule: each takes its rate and model **together**, because a
model without a rate is inert and the two halves cannot be set in separate
calls.

```python
from pecos import NoiseParameters

noise = (
    NoiseParameters()
    .with_p1(0.002)
    .with_p_idle_linear(0.01, {"X": 0.25, "Y": 0.25, "Z": 0.5})
    .with_p_idle_sin_squared(0.03, {"Z": 1.0})
)
```

<!--test-name: dem_from_guppy_grouped_noise-->
```python
from guppylang import guppy
from guppylang.std.builtins import result
from guppylang.std.quantum import cx, measure, qubit

from pecos import NoiseParameters
from pecos.qec import DetectorErrorModel


@guppy
def noisy_pair() -> None:
    q0, q1 = qubit(), qubit()
    cx(q0, q1)
    result("m0", measure(q0).read())
    result("m1", measure(q1).read())


common = {
    "num_qubits": 2,
    "detectors_json": '[{"id": "D0", "result_tags": ["m0"]}]',
    "observables_json": '[{"id": "L0", "result_tags": ["m1"]}]',
    "seed": 0,
}
noise = NoiseParameters().with_p1(0.002).with_p2(0.004).with_p_meas(0.006).with_p_prep(0.008)

grouped = DetectorErrorModel.from_guppy(noisy_pair, noise=noise, **common)
flat = DetectorErrorModel.from_guppy(
    noisy_pair,
    p1=0.002,
    p2=0.004,
    p_meas=0.006,
    p_prep=0.008,
    **common,
)
assert grouped.to_string() == flat.to_string()

try:
    DetectorErrorModel.from_guppy(noisy_pair, noise=noise, p1=0.002, **common)
except ValueError as exc:
    assert "p1" in str(exc)
else:
    raise AssertionError("grouped and flat noise must not be mixed")
```

`NoiseParameters.p2_szz` and `p2_szzdg` are gate-type total-rate overrides.
When either is unset, that gate inherits the shared `p2` rate, so omitting both
causes no DEM/simulator divergence. A non-default override is a documented API
gap: it is not expressible through the engines `general_noise()` builder. Neo
can represent the distinction with a `PerGatePauliChannel`, but the standard
Guppy DEM entry points reject explicit `p2_szz` or `p2_szzdg` values rather than
silently dropping them.

## Idle Noise

The recommended structured interface has three rate-and-model families. Every
model value is a **relative-rate multiplier**: for each channel,
`channel_rate = family_rate * channel_multiplier`. The linear law is additive, so
its multipliers are exactly the engines relative-probability distribution and
must sum to 1. The nonlinear laws are not additive, so their finite,
non-negative multipliers have no sum constraint.

- Linear: `p_idle_linear` with `p_idle_linear_model`. An axis fault has
  probability `(p_idle_linear * m_axis) * t`. The model keys are `X`, `Y`, and
  `Z`, plus the engines leakage key `L`. The default is the uniform
  `{X: 1/3, Y: 1/3, Z: 1/3}` engines model. An explicit `L` weight participates
  in the sum-to-1 requirement. This is a categorical Pauli channel. DEM
  construction first propagates its Pauli branches to detector/observable flip
  signatures, discards empty signatures, and adds probabilities for aliases.
  Two or more distinct signatures are converted to independent mechanisms; a
  non-negative boundary fit and its quantified residual are reported when the
  exact conversion would require a negative mechanism.
- Sine-squared: `p_idle_sin_squared` with `p_idle_sin_squared_model`. A Pauli
  fault has probability `sin((p_idle_sin_squared * m_axis) * t) ** 2`. The
  model keys are `X`, `Y`, `Z`, and `L`; there is no sum constraint. The
  symmetric default `{X: 1.0, Y: 1.0, Z: 1.0}` applies the full family rate to
  every Pauli axis—these are multipliers, not shares of a normalized total.
  To request pure dephasing instead, pass the explicit model `{"Z": 1.0}`.
- Coherent: `p_idle_coherent` with `p_idle_coherent_model`. An axis rotation has
  angle `(p_idle_coherent * m_axis) * t`. The symmetric default is
  `{RX: 1.0, RY: 1.0, RZ: 1.0}`. The standard DEM builder cannot represent
  coherent idle noise, so it rejects every nonzero family rate at call time;
  its previous lowering silently stored the Pauli twirl and discarded exactly
  the coherence requested. The EEG coherent route in `exp/pecos-eeg` is the
  consumer that can represent coherent idle noise, and only with an RZ
  generator even there. For an honest stochastic equivalent, the exact Pauli
  twirl of `RZ(rate * t)`, use `p_idle_sin_squared=rate/2` with
  `p_idle_sin_squared_model={"Z": 1.0}`. A coherent rate of zero or `None` has
  no effect. The `RX`, `RY`, and `RZ` model keys are validation-only on this DEM
  route; `L` and `U` are not valid coherent-model keys.

### Sine-squared is not the coefficient-quadratic law

Two different things are easy to confuse here, so state them side by side:

| Law | Probability | Where it lives |
| --- | --- | --- |
| Sine-squared | `sin((rate * m_axis) * t) ** 2` | `p_idle_sin_squared`, the structured family above |
| Coefficient quadratic | `(rate * m_axis) * t**2` | `p_idle_quadratic` and the axis-specific `p_idle_{x,y,z}_quadratic_rate` DEM fields |

Both scale as `t**2` for small angles, but equal numeric rates are never
interchangeable: the sine law's small-angle limit is `(rate * m_axis)**2 * t**2`
-- the rate enters squared -- while the coefficient law is
`(rate * m_axis) * t**2` with the rate entering linearly. They produce the same
small-angle probabilities only if the coefficient rate is set to the square of
the sine rate (per axis, `r_quad * m_quad = (r_sin * m_sin)**2`). The sine law
is also bounded and oscillates, so it saturates and then falls as the duration
grows; the coefficient law is an unbounded polynomial that only ever increases.
Their rates carry different units: the sine rate is radians per time, the
coefficient rate is inverse time squared.

Prefer the sine-squared family. It is the one the structured builders expose,
and it is the exact Pauli twirl of a coherent `RZ` rotation, which the
coefficient law is not. The coefficient fields remain on the low-level DEM
surface for models expressed that way already.

The engine-level spelling `with_p_idle_quadratic_rate(r)` has been removed. Its
input was in cycles per time, so it maps to the sine family with a factor of
pi -- see the conversion table below, which is easy to get wrong by omission.

The engines simulators can consume leakage models, such as an engines-bound
linear model `{"X": 0.8, "L": 0.2}`. DEM fault propagation is Pauli-only:
these DEM entry points accept `L` in linear and sine-squared models for model
compatibility, but reject it at call time when its weight is nonzero. A zero
`L` weight is silently accepted. Multi-qubit idle faults are outside the scope
of these keyword arguments and will arrive through a typed channel interface.

These rates match the engines family setters and runtime application semantics.
The removed quadratic builder spelling was the exception: its input was in
cycles per time and was converted before the runtime saw it. Family sine rates
are in radians per time and are not converted.

Every residual is readable from `dem.idle_noise_residuals` as a dictionary
containing `channel_kind`, `location_index`, the concrete
`detectors`/`dem_outputs`/`tracked_paulis` signature, `magnitude`,
`channel_weight`, and `relative_magnitude`. Gate and idle categorical Pauli
channels share this list. The channel weight is the sum of the requested
channel's non-identity probabilities before conversion, including branches
whose propagated signature is empty. The magnitude is the total-variation
distance between the requested categorical channel and the emitted independent
mechanisms, and `relative_magnitude = magnitude / channel_weight`; in the
two-dimensional boundary case the absolute magnitude is also the excess on the
reported signature and the matching identity deficit. Audited Guppy builds
copy this list to
`dem_build.audit["idle_noise_residuals"]`. An empty list certifies that all
categorical signature conversions were exact.

`DetectorErrorModel.builder().with_residual_warning_threshold(fraction)` sets
a relative physics tolerance. For example, `fraction=0.002` accepts an inexact
conversion whose total-variation residual is at most 0.2% of that requested
channel's total error weight. The default is zero, and a build warns when any
residual is greater than the accepted fraction. The threshold gates only that
warning: every exact figure remains in `dem.idle_noise_residuals` and the audit
entry regardless of the tolerance. To suppress warnings wholesale, use
`warnings.filterwarnings`; the builder setter encodes an accepted channel
approximation, not a blanket quiet mode.

The families are the only public way to configure these idle channels on
`NoiseParameters`. They translate into underscore-prefixed canonical per-axis
fields internally; those fields are implementation details consumed by the
Rust DEM boundary, not public constructor arguments or fluent setters.

Migration from the removed setters is mechanical:

| Removed | Replacement |
|---|---|
| `with_p_idle_z_linear_rate(r)` | `with_p_idle_linear(r, {"Z": 1.0})` |
| `with_p_idle_x_quadratic_sine_rate(r)` | `with_p_idle_sin_squared(r, {"X": 1.0})` |
| `with_p_idle_linear_rate(r)` | `with_p_idle_linear(r, {"Z": 1.0})` |

The last row is intentionally Z-only: despite its axis-free name,
`NoiseParameters.with_p_idle_linear_rate` configured only the Z channel. The
identically named setter on `general_noise()` configured a total linear rate
split according to its model and has now also been removed. Migrate that
engines spelling to `with_p_idle_linear(r, model)`, passing the symmetric
`{"X": 1/3, "Y": 1/3, "Z": 1/3}` model if the old model setter was not used.
Copying a numeric value between the two old interfaces did not preserve the
channel.

The engines quadratic migration has a unit conversion that must not be
omitted. At the removed path's default coherent-to-incoherent factor of `1.0`:

```text
with_p_idle_quadratic_rate(r)  ==  with_p_idle_sin_squared(r * PI, {"Z": 1.0})
```

The left-hand rate was in cycles per time; the family rate is in radians per
time. The orphaned `with_p_idle_coherent_to_incoherent_factor` setter has no
replacement. The two `with_average_p_idle_*_rate` spellings were also removed:
a gate-channel average-error conversion does not define a duration-independent
idle-family rate, especially for the nonlinear sine-squared law.

The default Selene runtime does not emit idle gates. These parameters and
`t1`/`t2` therefore have no locations to attach to unless the runtime supplies
scheduled idles or you insert them explicitly. `from_guppy` raises
`ValueError` when an idle-noise rate is supplied but the final traced circuit
contains no `Idle` gates; it does not silently build a DEM without the
requested noise.

Both `DetectorErrorModel.from_guppy` and `build_dem_from_guppy` accept two
passes for controlling those locations:

- `strip_traced_idles=True` removes only `Idle` gates from the normalized
  trace. It preserves `I` and zero-angle rotations as gate-noise locations.
- `idle_after_2q_duration=<positive float>` inserts an `Idle` of that duration
  on both qubits after every two-qubit gate.

Stripping runs before insertion. By default, setting `idle_after_2q_duration`
also strips first: inserting a uniform idle convention on top of
runtime-emitted idles would double-count idle noise. Pass
`strip_traced_idles=False` explicitly to keep runtime-emitted idles alongside
the inserted ones.

<!--test-name: dem_from_guppy_idle_noise-->
```python
from guppylang import guppy
from guppylang.std.builtins import result
from guppylang.std.quantum import cx, measure, qubit

from pecos.qec import DetectorErrorModel


@guppy
def idle_demo() -> None:
    q0, q1 = qubit(), qubit()
    cx(q0, q1)
    result("m0", measure(q0).read())
    result("m1", measure(q1).read())


common = {
    "num_qubits": 2,
    "detectors_json": '[{"id": "D0", "result_tags": ["m0"]}]',
    "observables_json": '[{"id": "L0", "result_tags": ["m1"]}]',
    "p1": 0.0,
    "p2": 0.0,
    "p_meas": 0.0,
    "p_prep": 0.0,
    "seed": 0,
}

without_idle_noise = DetectorErrorModel.from_guppy(
    idle_demo,
    idle_after_2q_duration=1.0,
    **common,
)
with_idle_noise = DetectorErrorModel.from_guppy(
    idle_demo,
    idle_after_2q_duration=1.0,
    p_idle_linear=0.01,
    p_idle_linear_model={"X": 0.25, "Z": 0.75},
    p_idle_sin_squared=0.02,
    **common,
)


def count_errors(model: DetectorErrorModel) -> int:
    return model.to_string().count("error(")


assert count_errors(with_idle_noise) > count_errors(without_idle_noise)

try:
    DetectorErrorModel.from_guppy(idle_demo, p_idle_linear=0.01, **common)
except ValueError as exc:
    assert "idle-noise parameters have no idle gates" in str(exc)
else:
    raise AssertionError("idle noise without Idle gates should fail")
```

Runtime-emitted idle durations are replayed as nanosecond `TimeUnits`.
Inserted idles instead carry the duration passed to
`idle_after_2q_duration`, which must be finite and positive. Linear and
sine-law idle rates are per time unit. For example, uniform linear idle noise
uses `(p_idle_linear / 3) * duration` per Pauli axis, clamped to the probability
range. The low-level coefficient-style quadratic rates multiply `duration**2`
and therefore scale as inverse time squared. T1 and T2 values must use the same
units as the idle duration.

## Exporting the DEM as Stim Text

PECOS's native DEM text is the Stim DEM format; there is no separate
`to_stim()` conversion. `dem.to_string()` emits standard
`error(p) D... L...` mechanisms and can be parsed directly as a Stim detector
error model. The export itself has no extra dependency.

`dem.to_string_decomposed()` uses decomposition components attached to the
original fault source, writing `^`-separated components when that provenance
is available. It preserves residual hyperedges when a true hyperedge has no
source-attached decomposition. Graph matchers instead need
`dem.to_string_terminal_graphlike_decomposed()`, an explicitly lossy
hyperedge-to-edge projection based on detector terminals rather than a proof
of source provenance.

<!--test-name: dem_from_guppy_stim_export-->
```python
from guppylang import guppy
from guppylang.std.builtins import result
from guppylang.std.quantum import cx, measure, qubit

from pecos.qec import DetectorErrorModel


@guppy
def idle_demo() -> None:
    q0, q1 = qubit(), qubit()
    cx(q0, q1)
    result("m0", measure(q0).read())
    result("m1", measure(q1).read())


dem = DetectorErrorModel.from_guppy(
    idle_demo,
    num_qubits=2,
    detectors_json='[{"id": "D0", "result_tags": ["m0"]}]',
    observables_json='[{"id": "L0", "result_tags": ["m1"]}]',
    idle_after_2q_duration=1.0,
    p1=0.0,
    p2=0.0,
    p_meas=0.0,
    p_prep=0.0,
    p_idle_linear=0.01,
    seed=0,
)

raw_text = dem.to_string()
source_decomposed_text = dem.to_string_decomposed()
graphlike_text = dem.to_string_terminal_graphlike_decomposed()

print(raw_text)
print(source_decomposed_text)
print(graphlike_text)
assert "error(" in raw_text
```

To verify interoperability against Stim itself, install the optional extra
(`pip install "quantum-pecos[stim]"`) — the base install does not depend on
stim:

<!--test-name: dem_from_guppy_stim_parse_check-->
```python
import stim
from guppylang import guppy
from guppylang.std.builtins import result
from guppylang.std.quantum import cx, measure, qubit

from pecos.qec import DetectorErrorModel


@guppy
def idle_demo() -> None:
    q0, q1 = qubit(), qubit()
    cx(q0, q1)
    result("m0", measure(q0).read())
    result("m1", measure(q1).read())


dem = DetectorErrorModel.from_guppy(
    idle_demo,
    num_qubits=2,
    detectors_json='[{"id": "D0", "result_tags": ["m0"]}]',
    observables_json='[{"id": "L0", "result_tags": ["m1"]}]',
    idle_after_2q_duration=1.0,
    p1=0.0,
    p2=0.0,
    p_meas=0.0,
    p_prep=0.0,
    p_idle_linear=0.01,
    seed=0,
)

stim.DetectorErrorModel(dem.to_string())
stim.DetectorErrorModel(dem.to_string_decomposed())
stim.DetectorErrorModel(dem.to_string_terminal_graphlike_decomposed())
```

## Decoding: PyMatching, Tesseract, and BP-OSD

A sampled `SampleBatch` provides the uniform `batch.decode(dem_text, spec)`
interface. Typed specs make choices such as correlated versus uncorrelated
PyMatching explicit. Passing the same batch to each decoder compares them on
identical shots rather than on three independently sampled experiments.

<!--test-name: dem_from_guppy_decoder_comparison-->
```python
from pecos.decoders import (
    BpOsdDecoder,
    TesseractDecoder,
    bp_osd as bp_osd_spec,
    pymatching,
    tesseract as tesseract_spec,
)
from pecos.guppy_gen import get_num_qubits, make_surface_code
from pecos.qec import DetectorErrorModel
from pecos.qec.surface import SurfacePatch
from pecos.qec.surface.circuit_builder import generate_tick_circuit_from_patch

patch = SurfacePatch.create(distance=3)
meta_tc = generate_tick_circuit_from_patch(patch, num_rounds=3, basis="Z")
dem = DetectorErrorModel.from_guppy(
    make_surface_code(distance=3, num_rounds=3, basis="Z"),
    num_qubits=get_num_qubits(3),
    detectors_json=meta_tc.get_meta("detectors"),
    observables_json=meta_tc.get_meta("observables"),
    num_measurements=int(meta_tc.get_meta("num_measurements")),
    p1=0.005,
    p2=0.005,
    p_meas=0.005,
    p_prep=0.005,
)

batch = dem.to_sampler().sample_batch(1000, 0)
error_counts = {
    "pymatching": batch.decode(
        dem.to_string_terminal_graphlike_decomposed(),
        pymatching(correlated=True),
    ).num_errors,
    "tesseract": batch.decode(
        dem.to_string_source_graphlike_decomposed(),
        tesseract_spec(preset="fast"),
    ).num_errors,
    "bp_osd": batch.decode(dem.to_string(), bp_osd_spec()).num_errors,
}
assert all(0 <= count <= batch.num_shots for count in error_counts.values())
print(error_counts)

# Construct a decoder directly when you need per-shot results. The "fast"
# preset matches the typed Tesseract spec used above.
syndrome = batch.get_syndrome(0)
tesseract = TesseractDecoder.from_dem(dem.to_string(), preset="fast")
tesseract_result = tesseract.decode_syndrome(syndrome)
assert tesseract_result.observable_flips.mask >= 0

bp_osd = BpOsdDecoder.from_dem(dem.to_string())
bp_osd_result = bp_osd.decode_syndrome(syndrome)
assert bp_osd_result.observable_flips.mask >= 0
```

For direct PyMatching construction, use the
`PyMatchingDecoder.from_dem(...)` pattern in the
[surface-memory example](#surface-code-memory-dem). That DEM's source-attached
decomposition is already graph-like; in general, matching decoders require the
terminal-decomposed graph-like projection. Tesseract and BP-OSD can consume the
raw hyperedge DEM directly; the batch comparison above uses the established
source-graphlike form for Tesseract so it matches the QEC-with-Guppy workflow.

## Limitations

- **Measurement-dependent quantum control flow is unsupported and
  rejected.** `from_guppy` traces one ideal execution, so a program whose
  quantum operations depend on a measurement *outcome* (e.g.
  `if measure(q).read(): x(other)`) would yield a DEM built from a single sampled
  branch — wrong and seed-dependent. Guppy programs whose compiled HUGR
  contains branching or looping control flow therefore raise `ValueError`
  before tracing; built-in generators such as `make_surface_code` cross
  this boundary through a trusted program-bound layout certificate.
  Statically-scheduled gates after measurements (every QEC round has them)
  are fine; genuinely conditioned gates are not.
- **Clifford circuits only.** Traced operations must normalize to named
  Clifford gates (`RZZ(±π/2)` is accepted as `SZZ`/`SZZdg`); residual
  non-Clifford rotations are rejected.
- **Only HUGR-certifiable inputs are accepted**: a `@guppy` function, a
  `pecos.Guppy` or `pecos.Hugr` wrapper, or HUGR envelope bytes.
  Already-lowered QIS/QIR inputs — accepted by `pecos.sim` and by earlier
  versions of this constructor — are now rejected, because their control
  flow cannot be certified before tracing; recompile from the Guppy/HUGR
  source instead.
- **`num_qubits` is required** for HUGR-bytes programs; use
  `get_num_qubits(...)` for the built-in generators.
- **Idle noise needs idle gates.** The default simple runtime does not emit
  explicit idles. Use `idle_after_2q_duration` to insert them, optionally after
  `strip_traced_idles` removes runtime-provided `Idle` gates. Passing
  idle-noise parameters without any final `Idle` gates raises `ValueError`; see
  [Idle Noise](#idle-noise).
- **Hand-authored tracked-Pauli observables are rejected** in
  `observables_json`; tracked Paulis come from circuit annotations only.

## Next Steps

- **[QEC with Guppy](qec-guppy.md)** - Generate the QEC programs this page
  builds DEMs from
- **[Fault Tolerance Analysis](fault-tolerance.md)** - The underlying DEM
  builder and fault propagation
- **[Decoders](decoders.md)** - Decode syndromes sampled from a DEM
- **[HUGR & Guppy Simulation](hugr-simulation.md)** - Running Guppy programs
  with noise
- **[Runtime QIS Tracing](runtime-qis-tracing.md)** - Capture, inspect, store,
  and replay the runtime-lowered circuit directly
