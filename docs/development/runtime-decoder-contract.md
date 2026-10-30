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
validates the v1 contract below. The QIS host/FFI data ABI is now **6**; rebuild
both artifacts together. The intrinsic's v1 is the JSON contract version, not
the host/FFI data ABI version.

The original `pecos_runtime_decode` import remains supported for unbound windows.
Both imports share the same shot-local contiguous instance counter and failure
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
instrumented frontend must supply and verify the bridge between these semantic
reads and actual runtime measurement provenance.

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

Checked declarations still cannot detect a program that sends the right schema
ID with values computed from the wrong reads. They cannot establish that branch
records are complete or true, prove fault ownership or discharge independence.
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
