# Guppy DEM preparation and building

The Python `GuppyDemBuilder`, `GuppyDemBuild`, and
`infer_guppy_dem_annotations` APIs wrap one Rust implementation:
`DemBuildBuilder` → `prepare()` → `PreparedDem` → `build_dem(noise)`.
Preparation resolves measurement identities, validates definitions, computes
noiseless references, and records schedule and binding evidence separately.
Rust also owns hosted-operation validation, QIS result-tag analysis,
helper-ABI normalization, noise-option validation, and structured idle translation.

Use `build_dem_from_guppy(...)` for typed definitions and shot evaluators, or
`DetectorErrorModel.from_guppy(...)` for JSON definitions. See the
[decoding walkthrough](../workflows/guppy-dem-decoding.md) for an executable example.

## Prepared schemas and references

Preparation computes the exact noiseless parity of every detector and
observable and writes it into the returned metadata as `reference`. Random
parities and declared references that disagree are errors. Shot evaluators
return detection events and observable flips relative to those references;
the `*_parities` methods return raw parities.

`InferredGuppyDemAnnotations` wraps the prepared schema. Its `.circuit` is a
detached snapshot: modifying it does not affect subsequent builds.
`build_dem(**noise)` accepts the full `DetectorErrorModel.from_circuit` keyword
set. In mixed builds its JSON, circuit, and DEM include the whole schema;
inference supports and labels describe only the inferred definitions.

Native Rust callers can reuse `PreparedDem::build_dem(noise)` with different
noise configurations. `DemBuildBuilder::build()` requires `with_noise(...)`;
`prepare()` needs no noise. `SeleneDemProgram::from_hugr` takes a typed
`QSystemPlatform` and `OptimizationLevel`.

Schema fingerprint version 2 includes computed references. Fingerprints differ
from earlier releases and have no cross-version compatibility.

## Schedule and measurement evidence

Explicit Python builds require a HUGR static-schedule preflight or a trusted,
program-bound generator layout on an eligible definition carrier. Native Rust
builds accept the HUGR verdict, a front-end assertion, or a trusted generator
layout. Mixed builds with named references also require static evidence.
Pure inference and positional-only mixed builds check schedules empirically
across probes; their audits identify this weaker schedule evidence separately
from measurement binding.

The layout digest binds the layout to the compiled HUGR. It detects stale or
accidentally reassigned layouts, but does not authenticate their producer.
Generic direct scalar result tags are analyzed against the lowered QIS used
for execution. Aggregate arrays require a supported generator layout or the
inference raw-output binding.

Replay requires a complete runtime-lowered operation stream with stable
measurement identities. Raw compatibility replay and its fallback flag are
removed. Malformed replay fields are rejected. Rust validation retains the
error wording while formatting values as Rust does, including double-quoted
strings and Rust-formatted lists.

## Idle noise and symbolic propagation

Runtime idle durations use `1 TimeUnit = 1 ns`. Stripping idle gates runs before
optional insertion after two-qubit gates. Active idle noise needs idle gates;
explicit zero rates do not. Relaxation noise requires both `t1` and `t2`.

Symbolic stabilizer propagation accounts for random partner signs after
entangled resets, restores the X basis after MX, and treats recordless
`MeasureLeaked` outcomes as hidden random sources. These rules prevent random
noiseless parities from being accepted as deterministic references.
