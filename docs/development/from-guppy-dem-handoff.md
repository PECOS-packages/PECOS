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
from pecos.qec import Detector, Observable, build_dem_from_guppy, rec

build = build_dem_from_guppy(
    program,
    num_qubits=num_qubits,
    detectors=[Detector(rec[-2], rec[-1])],
    observables=[Observable(rec[-1])],
    runtime=runtime,
)
dem = build.dem
```

The `GuppyDemBuild` owns the one traced circuit, resolved `MeasId` metadata,
measurement ledger, schema fingerprint, and shot evaluators. This prevents a
second runtime trace or a separately maintained detector conversion from
silently permuting decoder inputs.

The audited path is deliberately fail-closed. A runtime trace must contain a
complete, contiguously framed lowered operation stream with stable measurement
IDs and an explicit terminal marker; raw pre-runtime QIS order is not an
acceptable substitute. The terminal marker is only emitted after the engine
verifies the shot did not fail and the runtime scheduler holds no undelivered
operations. Precisely stated, `lowered_quantum_ops_complete` attests that the
operations the runtime returned parsed consistently (counts, metadata,
measurement IDs); dropped measurements are additionally caught by
measurement-mapping conservation. Completion additionally forces a terminal
scheduler flush (a runtime global barrier where the plugin supports one) and
fails the shot, stickily, if any operation surfaces after the final lowered
batch. A runtime that internally discards a non-measurement gate while
reporting a consistent stream remains undetectable in principle — that
residual trust lives in the runtime plugin itself.

Shot completion runs three gates in order before the terminal marker: the
sticky terminal-failure check, drain verification (which requires the plugin
to export `selene_runtime_global_barrier` — a plugin that cannot prove a
terminal flush fails closed), and the runtime's `shot_end` finalization hook
with its error propagated and latched. A plugin that only detects an invalid
final schedule at shot end therefore fails the shot rather than receiving a
certified trace. Named shot conversion accepts compiler-certified direct
scalar `result()` dataflow and trusted built-in generator layouts whose digest
binds both the HUGR and layout. Aggregate arrays and transformed booleans
must not be treated as generic measurement identity until the compiler exposes
an explicit element-level provenance ABI.

Generic Guppy branching and looping control flow is rejected. One sampled
runtime branch cannot certify a static DEM; built-in surface generators cross
that boundary only through their program-bound static-layout certificate.

The certificate is an integrity mechanism, not an authentication mechanism.
Its digest binds a layout to the exact compiled HUGR, so a stale, permuted,
or accidentally re-attached layout fails closed. It does not defend against
deliberate in-process forgery: any Python code that can set the attribute can
also recompute the public digest (or monkeypatch the checker), and no
in-process scheme changes that. The trust statement is "this layout was
computed for exactly this program", nothing more.

The remaining generic scalar trust boundary is cross-pipeline measurement
ordinal agreement: HUGR traversal ordinals and source-QIS measurement emission
are regression-tested to agree for supported straight-line programs, but do not
yet share an explicit compiler origin-ID ABI.

This should support calls like:

```python,notest
from pecos.guppy import get_num_qubits, make_surface_code
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
uv run python -c "from pecos.guppy import make_surface_code, get_num_qubits; from pecos.qec import DetectorErrorModel; print(get_num_qubits(9, ancilla_budget=17)); print(hasattr(DetectorErrorModel, 'from_guppy')); make_surface_code(distance=9, num_rounds=18, basis='Z', ancilla_budget=17); print('ok')"
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
  as `p_idle`, `t1/t2`, `p_idle_linear_rate`, or `p_idle_quadratic_rate` is set.
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
