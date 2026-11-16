# `DetectorErrorModel.from_guppy` Handoff

This note is for future work on the DEM polish path. It captures the current
local fix and the validation target for constrained-ancilla surface-code DEMs.

## Context

The `dem-polish` work adds a Python-level `DetectorErrorModel.from_guppy(...)`
entry point. The intended shape is:

1. Build any Guppy program, including constrained surface-code memory circuits.
2. Trace it through Selene/QIS into a `TickCircuit`.
3. Attach caller-provided detector and observable metadata.
4. Build the native PECOS DEM from that traced circuit.

For new workflows, prefer the additive typed build API over separately tracing
metadata and then calling `from_guppy`:

```python,notest
from pecos.qec import Detector, Observable, build_dem_from_guppy, result_ref

build = build_dem_from_guppy(
    program,
    num_qubits=num_qubits,
    detectors=[Detector(result_ref("m0"), result_ref("m1"))],
    observables=[Observable(result_ref("m1"))],
    runtime=runtime,
)
dem = build.dem
```

The `GuppyDemBuild` owns the one traced circuit, resolved `MeasId` metadata,
measurement ledger, schema fingerprint, and shot evaluators. This prevents a
second runtime trace or a separately maintained detector conversion from
silently permuting decoder inputs.

Typed Guppy specs reject `rec[...]`: compiled measurements do not preserve source
order. Use `result_ref(tag)` or `result_ref(tag, occurrence=...)` for repeated
tags. The retained audit names `runtime_order_is_canonical`,
`runtime_order_mismatch_count` and `canonical_record_index` refer to runtime
positions versus compiled MeasIds, not source measurement order.

The audited path is deliberately fail-closed. A runtime trace must contain a
complete, contiguously framed lowered operation stream with stable measurement
IDs and an explicit terminal marker; raw pre-runtime QIS order is not an
acceptable substitute. The terminal marker is only emitted after the engine
verifies the shot did not fail and the runtime scheduler holds no undelivered
operations. Precisely stated, `lowered_quantum_ops_complete` attests that the
operations the runtime returned parsed consistently (counts, metadata,
measurement IDs); dropped measurements are additionally caught by
measurement-mapping conservation. Completion additionally forces a terminal
scheduler flush via a required runtime global barrier and fails the shot,
stickily, if any operation surfaces after the final lowered batch — or if the
plugin cannot provide the barrier at all. A runtime that internally discards a non-measurement gate while
reporting a consistent stream remains undetectable in principle — that
residual trust lives in the runtime plugin itself.

Shot completion runs three gates in order before the terminal marker: the
sticky terminal-failure check, drain verification (which requires the plugin
to export `selene_runtime_global_barrier` — a plugin that cannot prove a
terminal flush fails closed), and the runtime's `shot_end` finalization hook
with its error propagated and latched. A plugin that only detects an invalid
final schedule at shot end therefore fails the shot rather than receiving a
certified trace. Named shot conversion accepts compiler-certified direct
scalar `result()` provenance from the executed lowered QIS and built-in
generator slots whose identities are resolved from runtime provenance or compiled
scalar analysis. Both sources must agree when available, and every layout slot
requires evidence. The certificate digest binds both the HUGR and layout. Aggregate arrays
and transformed booleans must not be treated as generic measurement identity
until the compiler exposes an explicit element-level provenance ABI.

Generic Guppy branching and looping control flow is rejected. One sampled
runtime branch cannot certify a static DEM; built-in surface generators
`make_surface_code` and `make_surface_memory` cross that boundary only through
their program-bound static-layout certificate.

The certificate is an integrity mechanism, not an authentication mechanism.
Its digest binds a layout to the exact compiled HUGR and detects changes to
either without a matching digest update. Measurement identity is resolved
from runtime provenance or compiled scalar analysis for every slot; both must
agree when available, and distinct slots must resolve to distinct measurements.
Together with the entry-count and traced-ID membership checks, this establishes
a bijection from slots to traced measurements. Reordering the layout and recomputing its digest preserves
these identities: layout order is informational. The certificate still supplies
control-flow admission, entry count against the trace, slot inventory against
runtime value counts, required tags and producer integrity. Per-tag occurrence k
assumes the k-th compiled print is the k-th abstract occurrence: `result()` calls
retain source order even when measurements reorder. No check compares the
executed trace's measurement order against the abstract schedule; identity rests
on the tags and the evidence above. The certificate does not defend against
deliberate in-process forgery: any Python code that can set the attribute can
also recompute the public digest (or monkeypatch the checker), and no
in-process scheme changes that. The certificate makes no claim that compiled measurement order matches layout order.

Generic scalar tags now derive their measurement ordinals from the exact lowered
QIS text used for execution. CFG dominance and cycle checks certify measurement
and print order across direct calls, independently of LLVM block layout. Only
direct scalar measurement reads carry provenance; unsupported values preserve
an unresolved occurrence, and uncertifiable schedules fail closed. The runtime
measurement count and identities are checked against the certified program.

This should support calls like:

```python,notest
from pecos.guppy_gen import get_num_qubits, make_surface_code
from pecos.qec import DetectorErrorModel

program = make_surface_code(
    distance=9,
    num_rounds=18,
    basis="Z",
    ancilla_budget=17,
)

dem = DetectorErrorModel.from_guppy(
    program,
    num_qubits=get_num_qubits(9, ancilla_budget=17),
    detectors_json=detectors_json,
    observables_json=observables_json,
    num_measurements=num_measurements,
    p1=p,
    p2=p,
    p_meas=p,
    p_prep=p,
)
```

## Import-Time Issue

`pecos_rslib.qec.DetectorErrorModel` is not currently subclassable from Python.
Defining:

```python,notest
class DetectorErrorModel(_RustDetectorErrorModel):
    ...
```

causes `import pecos` to fail with:

```text
TypeError: type 'pecos_rslib.qec.DetectorErrorModel' is not an acceptable base type
```

The current local fix is to re-export the Rust class directly and attach the
Python convenience constructor:

```python,notest
DetectorErrorModel = _RustDetectorErrorModel
DetectorErrorModel.from_guppy = classmethod(...)
```

This keeps the public API as `pecos.qec.DetectorErrorModel.from_guppy(...)`
while preserving the Rust class identity for objects returned by
`from_circuit(...)` and `from_guppy(...)`.

## Constrained-Ancilla Surface DEM Target

The key surface-code use case is Helios-sized rotated surface code memory:

- `distance=9`
- `ancilla_budget=17`
- `num_qubits=98`
- both X and Z memory bases
- DEMs built from the traced Guppy/Selene/QIS path

Important checks:

```bash
uv run python -c "from pecos.guppy_gen import make_surface_code, get_num_qubits; from pecos.qec import DetectorErrorModel; print(get_num_qubits(9, ancilla_budget=17)); print(hasattr(DetectorErrorModel, 'from_guppy')); make_surface_code(distance=9, num_rounds=18, basis='Z', ancilla_budget=17); print('ok')"
```

Expected output includes:

```text
98
True
ok
```

In the downstream `surface-memory-helios` repo, this smoke test currently
generates a constrained d=9 traced DEM:

```bash
uv run python -c "from surface_memory_helios import surface_memory_dem; dem = surface_memory_dem(distance=9, rounds=18, basis='Z', p=0.01, decoder='pymatching', dem_source='traced_qis', ancilla_budget=17); print(len(dem.splitlines())); print(dem.splitlines()[0])"
```

The latest local run produced 37,066 DEM lines and a detector metadata first
line.

## Follow-Up Guidance

- Prefer `build_dem_from_guppy(...)` for new audited workflows. Keep
  `DetectorErrorModel.from_guppy(...)` as the lower-level JSON compatibility
  constructor rather than adding more surface-specific tracing plumbing.
- Runtime plugins are intentionally generic: pass any Selene-compatible runtime
  plugin object through `pecos.selene_engine(runtime)` or the higher-level
  `runtime=...` arguments on traced Guppy/DEM helpers. PECOS should depend only
  on the public shape of those plugin objects; experiment-specific runtimes and
  package sources belong in downstream projects.
- Runtime-produced `Idle` gates are preserved in the QIS operation trace and
  replayed into QEC circuits as `TimeUnits` with the convention
  `1 TimeUnit = 1 ns`. They only affect DEMs when an idle-noise parameter such
  as an idle family, `t1`, or `t2` is set.
- Keep fail-closed regression coverage for entirely raw traces, transformed
  scalar results, and aggregate arrays. Generated adapters may expose direct
  scalar sideband tags while retaining aggregate results for researcher-facing
  analysis.
- Keep the surface helper path compatible with constrained ancilla budgets:
  pass `ancilla_budget` into both `make_surface_code(...)` and
  `get_num_qubits(...)` when tracing surface Guppy.
- Avoid reintroducing any `circuit_source="traced_qis"` rejection for
  `ancilla_budget`; constrained Guppy programs are valid and traceable.
- Ensure PyMatching users can get decomposed DEM text from the `from_guppy(...)`
  result, e.g. via `to_string_decomposed()`.
- Keep or add regression coverage for constrained d=9, `ancilla_budget=17`
  through the Guppy/from_guppy route.
