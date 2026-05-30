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

This should support calls like:

```python
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

```python
class DetectorErrorModel(_RustDetectorErrorModel):
    ...
```

causes `import pecos` to fail with:

```text
TypeError: type 'pecos_rslib.qec.DetectorErrorModel' is not an acceptable base type
```

The current local fix is to re-export the Rust class directly and attach the
Python convenience constructor:

```python
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

- Prefer the generic `from_guppy(...)` abstraction for future DEM construction
  rather than adding more surface-specific tracing plumbing.
- Runtime plugins are intentionally generic: pass a Selene runtime plugin object
  through `pecos.selene_engine(runtime)` or the higher-level
  `runtime=...` arguments on traced Guppy/DEM helpers. Anduril should live in
  downstream experiment projects, not as a PECOS dependency.
- Runtime-produced `Idle` gates are preserved in the QIS operation trace and
  replayed into QEC circuits as `TimeUnits` with the convention
  `1 TimeUnit = 1 ns`. They only affect DEMs when an idle-noise parameter such
  as `p_idle`, `t1/t2`, `p_idle_linear_rate`, or `p_idle_quadratic_rate` is set.
- Keep the surface helper path compatible with constrained ancilla budgets:
  pass `ancilla_budget` into both `make_surface_code(...)` and
  `get_num_qubits(...)` when tracing surface Guppy.
- Avoid reintroducing any `circuit_source="traced_qis"` rejection for
  `ancilla_budget`; constrained Guppy programs are valid and traceable.
- Ensure PyMatching users can get decomposed DEM text from the `from_guppy(...)`
  result, e.g. via `to_string_decomposed()`.
- Keep or add regression coverage for constrained d=9, `ancilla_budget=17`
  through the Guppy/from_guppy route.
