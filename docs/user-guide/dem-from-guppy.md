# Detector Error Models from Guppy Programs

This guide covers `DetectorErrorModel.from_guppy`, which builds a
circuit-level detector error model (DEM) from a Guppy program by tracing it
through the Selene QIS engine. This is the recommended way to get a DEM for
a logical circuit you intend to run on a Selene-compatible runtime.

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
    result("s0", measure(a0))
    result("s1", measure(a1))
    # Deferred tagging: bind now, tag later -- the tag->measurement
    # binding follows dataflow, not source order.
    m0 = measure(d0)
    m1 = measure(d1)
    m2 = measure(d2)
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
  **dataflow, not syntax**: `m = measure(q)` followed later by
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
from pecos.guppy import get_num_qubits, make_surface_code
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
batch = sampler.generate_samples(1000, 0)
assert batch.num_shots == 1000

decoder = PyMatchingDecoder.from_dem(dem.to_string_decomposed())
errors = 0
for shot in range(batch.num_shots):
    predicted = decoder.decode(batch.get_syndrome(shot)).correction[0]
    actual = batch.get_observable_mask(shot) & 1
    errors += predicted != actual
print(f"logical error rate: {errors / batch.num_shots:.4f}")
```

The DEM built this way is identical to the reference DEM produced by the
surface traced-QIS pipeline — the abstract builder's metadata and the
traced Guppy program agree on measurement order.

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

## Idle Noise

The recommended structured interface mirrors the engines
`GeneralNoiseModel`:

- `p_idle_linear` is the total stochastic idle rate, linear in duration.
  `p_idle_linear_model` supplies relative `X`, `Y`, and `Z` weights that sum
  to 1; its default is the uniform `{X: 1/3, Y: 1/3, Z: 1/3}` engines model.
  The engines `L` leakage key is reserved but rejected because DEM construction
  cannot represent leakage. `p_idle` remains shorthand for the uniform model.
- `p_idle_quadratic` is the engines-style quadratic dephasing rate. By default,
  an idle of duration `t` has a stochastic Z probability
  `sin(p_idle_quadratic * t) ** 2`. With `p_idle_coherent=True`, it instead
  contributes an `RZ(p_idle_quadratic * t)` angle. The DEM coherently adds
  angles for matching detector sets and uses the RZ half-angle probability
  `sin(total_angle / 2) ** 2`.

The per-axis `p_idle_{x,y,z}_linear_rate`,
`p_idle_{x,y,z}_quadratic_rate`, and
`p_idle_{x,y,z}_quadratic_sine_rate` parameters remain available as low-level
knobs. The bare Z-only aliases `p_idle_linear_rate`,
`p_idle_quadratic_rate`, and `p_idle_quadratic_sine_rate` are deprecated; use
the structured interface or the explicitly named `p_idle_z_*` equivalent.

The default Selene runtime does not emit idle gates. These parameters and
`t1`/`t2` therefore have no locations to attach to unless the runtime supplies
scheduled idles or you insert them explicitly. `from_guppy` raises
`ValueError` when an idle-noise rate is supplied but the final traced circuit
contains no `Idle` gates; it does not silently build a DEM without the
requested noise.

Both `DetectorErrorModel.from_guppy` and `build_dem_from_guppy` accept two
passes for controlling those locations:

- `strip_traced_idles=True` removes identity-like gates from the normalized
  trace, including `I`, `Idle`, and zero-angle rotations.
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
    result("m0", measure(q0))
    result("m1", measure(q1))


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
    p_idle_quadratic=0.02,
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
sine-law idle rates are per time unit (for example, uniform idle noise uses
`p_idle * duration`, clamped to the probability range). The low-level
coefficient-style quadratic rates multiply `duration**2` and therefore scale
as inverse time squared. T1 and T2 values must use the same units as the idle
duration.

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
    result("m0", measure(q0))
    result("m1", measure(q1))


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
    p_idle=0.01,
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
    result("m0", measure(q0))
    result("m1", measure(q1))


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
    p_idle=0.01,
    seed=0,
)

stim.DetectorErrorModel(dem.to_string())
stim.DetectorErrorModel(dem.to_string_decomposed())
stim.DetectorErrorModel(dem.to_string_terminal_graphlike_decomposed())
```

## Decoding: PyMatching, Tesseract, and BP-OSD

A sampled `SampleBatch` provides the uniform
`batch.decode_count(dem_text, name)` interface. The names used here are
`"pymatching"` (correlated matching by default), `"tesseract"`, and
`"bp_osd"`. Passing the same batch to each decoder compares them on identical
shots rather than on three independently sampled experiments.

<!--test-name: dem_from_guppy_decoder_comparison-->
```python
from pecos.decoders import DemAwareDecoder, TesseractDecoder
from pecos.guppy import get_num_qubits, make_surface_code
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

batch = dem.to_sampler().generate_samples(1000, 0)
error_counts = {
    "pymatching": batch.decode_count(
        dem.to_string_terminal_graphlike_decomposed(),
        "pymatching",
    ),
    "tesseract": batch.decode_count(
        dem.to_string_source_graphlike_decomposed(),
        "tesseract",
    ),
    "bp_osd": batch.decode_count(dem.to_string(), "bp_osd"),
}
assert all(0 <= count <= batch.num_shots for count in error_counts.values())
print(error_counts)

# Construct a decoder directly when you need per-shot results. The "fast"
# preset matches the configuration decode_count(..., "tesseract") uses.
syndrome = batch.get_syndrome(0)
tesseract = TesseractDecoder.from_dem(dem.to_string(), preset="fast")
tesseract_result = tesseract.decode_syndrome(syndrome)
assert tesseract_result.observables_mask >= 0

bp_osd = DemAwareDecoder.from_dem(dem.to_string(), decoder_type="bp_osd")
bp_osd_result = bp_osd.decode_syndrome(syndrome)
assert bp_osd_result.observables_mask >= 0
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
  `if measure(q): x(other)`) would yield a DEM built from a single sampled
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
  `strip_traced_idles` removes runtime-provided identity-like gates. Passing
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
