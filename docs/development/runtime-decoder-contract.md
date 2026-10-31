# Versioned runtime decoder declarations

The supplied-model decoder path supports an additive `pecos_runtime_decode_v1`
import. It binds a call to a declared program/compiler revision, decode source
site and detector/frame schema. It also transports bounded, caller-supplied
branch, loop and classical-input context to the shot-local window provider.
This is declaration checking, **not compiler-verified data flow or adaptive DEM
certification**. Full JIT generation and runtime-batch stitching remain separate.

## Native and session interfaces

```llvm
declare i64 @pecos_runtime_decode_v1(i64, i64, ptr, i64, ptr, i64)
; window_id, instance_id, observed detector bytes, detector count,
; UTF-8 context JSON, context byte count. Return: uint64 XOR frame update.
```

Detector bytes retain the [existing contract](runtime-decoding.md): binary,
dense, at most 1,048,576 bytes. The context is 1–65,536 bytes, copied before
suspension. A nonempty buffer needs a valid readable pointer. Native transport
checks lengths and UTF-8; the standard Python provider additionally parses and
validates the v1 contract below. The QIS host/FFI data ABI is now **7**; rebuild
both artifacts together. The intrinsic's v1 is the JSON contract version, not
the host/FFI data ABI version.

The original `pecos_runtime_decode` import remains supported for unbound windows.
All decode imports share the same shot-local contiguous instance counter and failure
latch. Rust `RuntimeDecoderSession::decode_v1` defaults to an error. Custom Python
sessions must retain a callable `decode(window_id, instance_id, syndrome)`
method because the factory validates that existing protocol on creation. To
handle versioned calls, also implement
`decode_v1(window_id, instance_id, syndrome, context)`; its last two arguments
arrive as bytes. A session intended only for v1 can implement `decode` by raising
an unsupported-call error. There is no fallback from `decode_v1` to `decode`.
Custom session implementations are trusted to check their contract.

Versioned requests follow the same synchronous host path as legacy requests.
They do not insert barriers, force unread futures, advance native timestamps,
replace physical operations or expose internal noise measurements. Observed
measurements must have been read through the selected runtime before the call.

## Schema and binding

`RuntimeDecodeSchema` contains:

- `detectors`: one `RuntimeDetectorBinding` per **program byte**, in byte order.
  Each has a semantic detector identity, named read references and a Boolean
  reference sign. The declared byte is the XOR of those reads and the sign.
- `frame_ids`: semantic destination frame identities in bit order, at most 64.

`RuntimeRecordRef(name, occurrence)` names a program read within the current
independently closed decode region. Occurrences start at zero per name in read
order. These are frontend-declared labels, not QIR slots, native future handles,
trace measurement IDs or `result()` outputs. Repeated reads need distinct
occurrences; a producer must state which occurrence its parity uses. This
region-local convention lets repeated independent regions reuse a schema.
It does not authorize references across independently decoded boundaries.

This declaration differs from `dem_spec.result_ref`, which resolves named
`result()` outputs against a captured trace. No second circuit, DEM, composer or
decoder is introduced; models still come from the existing APIs. A future
instrumented frontend must supply source expectations. The optional observed-read
path below verifies executed native reads against those expectations.

The schema's `schema_id` is lowercase SHA-256 of canonical ASCII JSON:
`{"version":1,"detectors":...,"frame_ids":...}`. Serialize with sorted object
keys, no whitespace, separators `,` and `:`, JSON escaping for non-ASCII text and
no NaN/infinity. Detector/frame order matters. Within each parity, sort terms
by `(name, occurrence)`; reject repeated terms, requiring the producer to perform
GF(2) cancellation explicitly. Empty parity terms represent the reference
constant. Fields inside a detector are exactly `detector_id`, `records` and
`reference`; a record has `name` and `occurrence`. The fixed test vector in
`test_runtime_decode_context.py` specifies the bytes independently of the classes.

A `RuntimeDecodeBinding` couples the schema with nonempty `program_revision`,
`compiler_revision` and `decode_site`. Revisions must identify the semantics and
compiler configuration used to produce the program; human labels in the example
are only synthetic fixture identities. Source sites have a `qis:` or `hugr:`
namespace and must be stable within those revisions. A frontend needing both
representations should retain its revision-specific QIS-to-HUGR mapping.

Set `RuntimeDecodeWindow(binding=...)` to require this contract. Validation
checks that `window.detector_ids[i]` equals the schema detector identity at
`window.detector_order[i]`, and every `output_map` target names an actual schema
frame bit. Thus a same-width swapped mapping is rejected. Model output labels
and their physical/logical meanings remain caller declarations.

At each versioned call, the returned window must be bound and the four binding
identities must match exactly. A bound window rejects a legacy call; an unbound
window rejects a versioned call. Wrong identities, malformed contexts and decoder
failures abort the shot before a frame is returned. Factories, counters and
provider state reset for the next shot and remain isolated across workers.

## Observed native reads (opt-in)

```llvm
declare i1 @pecos_runtime_read_bool_v1(ptr, i64, i64)
; UTF-8 semantic name, name byte count, native future handle.
declare i64 @pecos_runtime_decode_observed_v1(i64, i64, ptr, i64, ptr, i64)
; Same arguments and v1 JSON context as decode_v1 above.
```

The read wrapper delegates to the selected runtime's existing Boolean getter,
including its existing lazy-force behavior. It records the returned bit and the
QIS FFI's shot-local measurement occurrence ID. This ID is assigned at source
measurement creation, before runtime scheduling; it is neither a native future
handle nor a simulator-private fault/logical-truth value. Handle reuse replaces
the handle association without changing already captured observations. Opaque
custom calls invalidate source associations: subsequent checked reads with
unknown provenance fail. Already captured reads remain usable.

Each label's read occurrence starts at zero after **every decode call**, including
legacy and declared-only calls. Those older calls discard their observed-read
interval. Shot reset clears both completed and interrupted reads. Observations
are separate from caller JSON and named `result()` output bookkeeping; printing
a named result does not consume decoder observations.

Opt in with `RuntimeDecodeWindow(binding=..., read_sources=...)`. Each immutable
`RuntimeReadSource(record=RuntimeRecordRef(name="readout", occurrence=0),
source_id=0)` supplies one expected source measurement occurrence. The mapping
must cover exactly the unique schema read terms; repeated reads have distinct
occurrences even when they read the same source. All named reads in the interval
must participate in that schema, with occurrences contiguous from zero per name.
For this restricted checked subset, use ordinary getters for reads unrelated to
detector terms. An empty tuple enables checked constant-reference detectors;
`None` retains declared-only/legacy behavior. The observed import rejects windows
without `read_sources`, and checked windows reject the other decode imports.

Before constructing the existing model decoder, the standard session checks:

1. The v1 binding and the complete, exact set of observed named occurrences.
2. Each observed source ID against its expected `RuntimeReadSource`.
3. Each **program-order** detector byte against the XOR of the observed values
   and its declared reference sign. Only then is `detector_order` applied.

A wrong source is rejected even if it returns the same bit. Missing/extra reads,
wrong occurrence numbers, wrong reference parities and wrong source mappings
abort the shot before feedback. A supplied provider can select a concrete source
mapping per instance using its existing request, but receives no observation
array, syndrome, future handle or private simulator state. Deriving source IDs
by arithmetic is valid only for a separately established fixed schedule, such as
the synthetic integration fixture; adaptive paths require their own sound
source mapping, not guesses based on a realized trace.

Read names contain 1–256 UTF-8 bytes without control characters below U+0020.
An interval permits at most 65,536 reads and 1,048,576 total name bytes. Bounds
are checked before entering the getter. The runtime owns all pending allocations
so cancellation can safely clear an interrupted read. This wrapper currently
requires dynamic execution with the native Selene protocol; unsupported paths
are rejected. Decoder calls themselves add no forcing, barriers or physical work.

Custom Python sessions implement
`decode_observed_v1(window_id, instance_id, syndrome, context, observations)`;
observations arrive as a list of `(name, occurrence, source_id, bool)` tuples in
read order. The existing callable `decode` method is still required. Rust's
`RuntimeDecoderSession::decode_observed_v1` rejects by default. Neither bridge
falls back to a weaker method. Custom sessions are trusted extensions; directly
calling a Python session with fabricated observations is not an authenticated
execution path.

These checks verify executed getter/source/parity consistency against supplied
expectations. Semantic names, compiler/source-site identity, branch predicates,
loop/input declarations and the DEM's physical meaning remain caller assertions.
Equal parities cannot prove the program's full classical data flow. These checks
also do not certify a single adaptive branch as a sound DEM, establish window
independence or generate JIT fragments.

## Runtime-emitted prefixes (opt-in)

The ordinary dynamic native path can pass runtime batch provenance to an
observed-read provider. This helper accepts the caller's per-shot provider factory;
pass its returned engine builder to `sim(...).classical(...)`:

```python
from pecos import selene_engine
from pecos.qec import RuntimeDecoder


def batch_decoder_engine(provider_factory, runtime="selene_simple_runtime"):
    return selene_engine(runtime).runtime_decoder(
        RuntimeDecoder(provider_factory=provider_factory),
        capture_batches=True,
    )
```

The program still calls `pecos_runtime_decode_observed_v1`. The provider now
receives `RuntimeDecodeBatchRequest`, which extends the existing request with
an immutable `batch_prefix: RuntimeDecodeBatchPrefix`. This is captured by the
host independently of caller JSON. The native import and FFI ABI remain unchanged.
Legacy and declared-only calls are rejected in this mode. Scheduled transport
and runtimes without capture support are rejected at construction.

A prefix contains the engine-local `runtime_shot_id`, `first_batch_index`, an
exclusive `next_batch_index`, and an ordered tuple of `RuntimeDecodeBatch` values.
Indices cover every emitted batch since shot start or the preceding decode,
including empty batches. Each interval is consumed once; repeated, missing or
out-of-shot intervals fail. Empty intervals are valid and have equal watermarks.
Buffers and session state reset for each shot and worker. Shot IDs are not global
worker IDs; keep provider state in its existing shot-local session.

Each batch retains its integer nanosecond start/duration, whether a batch adapter
normalized it, and the runtime-lowered operation sequence after source-result
routing and idle insertion. `RuntimeDecodeOperation` exposes `kind`, `qubits`,
`parameters` and optional `source_id`. Angles are radians; Idle parameters use the
existing lowering's floating-point seconds. Inserted idles may begin before the
batch start. Operations are captured before the host's existing gate encoding
and decompositions; they are not a second simulation or noise realization.
Source IDs on measurements are the same shot-local occurrences used by observed
reads. `None` on a measurement means runtime-owned work without a program source;
it must not be guessed into a detector identity. Native future handles, raw custom
payloads, measurement outcomes, seeds, fault samples and logical truth are excluded.
Original custom payloads are handled by the existing checked adapter/metadata
contract. Unhandled custom events reject capture even under capture-only policy.

The host delivers the prefix only after the existing ready commands and their
measurement feedback have executed. Capture adds no barrier, forcing, clock
advance or extra drain. Unforced work may still be held by the runtime, and
operations after the current decoder-dependent decision are unavailable. The
watermark certifies the emitted prefix only, not a complete circuit or fault
region. It is neither a cache key nor proof of independent closed windows.

Retention is bounded to 4,096 batches and 65,536 runtime-lowered operations
between decode calls (also before the first call and after the last). Exceeding
either bound aborts the shot; the history is never silently truncated. A provider
may retain supplied immutable snapshots at its own cost, but this does not change
window ownership or boundary-state admission.

Custom Python sessions implement
`decode_observed_with_batches_v1(window, instance, syndrome, context, observations, prefix)`.
The last argument is bounded version-one JSON bytes; the standard session validates
and freezes it before provider selection. Rust sessions implement the corresponding
method accepting `&DecoderBatchPrefix`; its default rejects. Neither bridge falls
back to the older observed method. Custom sessions remain trusted extension points.

This prerequisite connects actual batch timing/order and measurement occurrences
to the existing declared QIS/HUGR decode site, branch events, active loop counts
and classical inputs. It **does not verify those declarations**. Existing decorative
source metadata uses matching heuristics and is deliberately not promoted into
an exact compiler-site-to-physical-operation mapping. A future instrumenter must
bind a module/compiler revision and site table, record the actual branch predicate,
enforce loop-entry/backedge/exit counters, and capture typed input dependencies.
Manual hook calls alone cannot establish instrumentation completeness. Full JIT,
compiler-verified control flow and exact source-to-gate attribution remain follow-ups.

## Context v1

`RuntimeDecodeContext.to_json()` produces the bounded envelope. All fields are
required, including explicitly empty arrays:

| Field | Contract |
| --- | --- |
| `version` | Integer exactly 1; Boolean `true` is not version 1. |
| `program_revision`, `compiler_revision`, `decode_site`, `schema_id` | Static binding identities described above. |
| `branches` | Ordered `RuntimeBranchRecord(site_id, occurrence, outcome)` events since the previous decode, or shot start. Boolean outcomes; occurrences contiguous from zero per site within that interval. |
| `loops` | Active loop stack, outermost first, of `RuntimeLoopRecord(site_id, occurrence, iteration)`. Occurrence counts entries to that site from shot start; iteration counts completed iterations of that entry. Future total trip counts are not assumed. |
| `inputs` | Uniquely named `RuntimeClassicalInput(name, kind, value, dependencies)` records. Exact `bool`, `i64` or `u64` values only; unsupported types are rejected. |

Counters are unsigned 64-bit integers. A dependency has `source`, `name`, and
`occurrence`: `record` uses the region-local read convention; `decoder` names a
frame and a prior shot-local decode instance. Current or future decoder instances
are rejected. Frame-name ownership and dependencies are caller assertions;
this change does not reconstruct data flow or verify branch predicates.

The parser rejects unknown or missing fields, duplicate object keys, nonfinite
values, incorrect types, invalid source sites, duplicate event/input identities
and unsupported versions. Nested records are immutable. A malformed envelope
fails before calling the provider. A valid envelope is passed as
`RuntimeDecodeRequest.context`; legacy requests have `context=None`.
The provider receives no syndrome array or simulator-private state. Legitimate
branch outcomes and classical inputs can reveal information about observed
measurements; they are not an information-isolation boundary.

See [the executable example](../../examples/runtime_decoding.py), which offers
both legacy and versioned calls. Its one-bit detector has an independently known
correction action. Integration tests cover same-width wrong-read declarations,
branch/loop context across live iterations, correction projection, failure/reuse,
two runtimes and multiple workers. They do not prove instrumentation completeness
or a noise model for arbitrary adaptive programs.

## Future provider and JIT boundary

This context is **not a physical-fragment cache key** and carries no fabricated
runtime batch, timing, noise or boundary-state certificate. A future producer
must join it with authoritative emitted batches, availability/completion state,
source metadata and the sufficient physical identities listed in
[the runtime-decoding design](runtime-decoding.md#follow-up-providers-driven-by-emitted-physical-batches).
Use bounded caches for normalized physical fragments, including timing, scheduling,
noise/support topology, geometry, schema, horizon and supported boundary state.
Apply supported instance relabelings separately with `DemSliceInstance`.
Neither the schema digest nor the entire Boolean history is sufficient alone.

Declared-only calls cannot detect a program that sends the right schema ID with
values computed from the wrong reads. Observed calls check source identities and
parities, but cannot establish that semantic labels or branch records are complete
or true, prove fault ownership or discharge independence.
No admission rule has been weakened:

- A fault changing a branch requires a proved closed/invariant gadget or a
  supported joint conditional model; a realized branch trace is insufficient.
- Correlations crossing windows still require supported ownership and boundary
  carry. This host provider accepts only independent full commits with empty
  buffers/boundary state, preserving existing composer/decoder rejections.
- Persistent leakage cannot be discarded by declaring a new instance or zero
  state; closure must follow from the actual reset/noise contract.
- Unavailable future operations must not be invented or forced past the decode
  decision. Defer or use a supported boundary/latency protocol. Preserve
  `BufferTooSmall` and unsupported `from_guppy` control-flow failures.
