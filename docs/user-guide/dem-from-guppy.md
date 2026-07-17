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
- Choosing the Selene runtime, and the limitations to know about

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
  measurement record (Stim convention). Works for any program, including
  ones with runtime loops.
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
- **`num_qubits` is required** for QIS/HUGR programs; use
  `get_num_qubits(...)` for the built-in generators.
- **Idle noise needs idle gates.** Selene runtimes do not emit idle gates,
  so the idle/T1/T2 noise parameters only apply where idle gates are
  present in the traced circuit.
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
