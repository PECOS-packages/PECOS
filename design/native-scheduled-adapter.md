# Native scheduled batch adapters

Status: experimental Rust/C implementation. Base: PECOS 8515186d.
Python exposure uses scheduled_native_adapter and the existing scheduled-noise wrapper.
Performance and Windows execution validation remain separate milestones.

Python batch adapters currently construct owned Python snapshots and output Gate
objects at every validation/translation boundary. A native adapter should implement
ScheduledBatchAdapter through a C ABI and feed the same ScheduledEventNoise factory.
The production route remains sim -> MonteCarlo -> HybridEngine -> QuantumSystem.
No alternate simulator loop, timing reconstruction or offline trace is introduced.

## Ownership and lifecycle

The loader belongs in pecos-foreign, which already depends on pecos-engines and
libloading. Python exposes a constructor returning the existing scheduled-event
noise wrapper. Load an explicitly supplied trusted library path, resolve a distinct
symbol, and retain the library in every factory and session until after destruction.
Do not reuse the decoder/simulator plugin descriptor or a runtime's full error-model
interface: a normalizer has no simulator, outcome or RNG access.

A versioned descriptor is filled into a zeroed host-owned structure, with ABI version
and exact structure size passed to the entry point. Reject unsupported version/size,
nonzero reserved fields, missing callbacks or unsupported capability flags before
creating instances. No returned descriptor pointer or Rust layouts cross the ABI.

Immutable configuration bytes are copied once into the factory. The first normalized input for a host ShotContext
lazily creates at most one opaque session; a shot with no such input creates none. The plugin copies any configuration it retains. The host
passes run/worker/shot as checked u64 values; they identify a shot and are not seeds.
Sessions may move across threads; calls on one session are serialized, including
validation, translation and destruction. Different sessions may execute concurrently.
The entry point and creation callbacks must support concurrent hosts. Each create
returns independent mutable state. The plugin owns and destroys that state exactly
once; every non-null create output must be destroyable even when create fails.

Factory clones retain immutable configuration/library; live sessions are not cloned.
Existing host clone/poison/reset rules remain authoritative. Whole-host reset drops
the session, and the next shot constructs a fresh one. Adapter errors are sticky
within a session. No plugin reset callback or active-session snapshot is introduced.

## Borrowed input

Use a callback-scoped batch view carrying native shot ID, batch index, start/duration,
operation count and measurement-map count, plus checked indexed readers. A reader
copies one small C record; custom payload bytes are borrowed until callback return.
No Vec/Gate/trait object, ownership transfer or pointer into movable Rust state is
exposed. No input pointer may be retained. Read failures are latched by the host even
if a plugin ignores the returned status and claims success.

An operation record distinguishes ordinary gates from opaque custom events. Gate
records use explicit ABI codes and five u64 words (code, q0, q1, angle0, angle1), with
exact Angle64 fractions, matching the admitted scheduled vocabulary. Custom records
carry a tag and borrowed bytes. Measurement maps separately carry operation position,
native result ID and source result ID; neither namespace is collapsed.

The view's readers are synchronous, same-thread and non-reentrant. Null pointers with
zero lengths are allowed where stated; all other pointers must be valid and aligned.
Sizes are bounded and checked before slice construction/allocation. This is a trusted
native-code boundary, not a sandbox against forged pointers or arbitrary writes.

## Two phases and output

The existing controller admits the complete source message, then invokes validate
on every batch before translating any batch. Native validate must be deterministic
and side-effect free: no session mutation, emissions, quantum work or RNG consumption.
The plugin must bound its retained state across inputs; the sink only bounds host output.
A C const pointer documents this restriction but cannot enforce it; conformance is
part of the trusted plugin contract. No validation output sink is supplied.

Translation may update shot-local classical state and emit normalized gates via a
host-owned bounded sink. It cannot retime, split or reorder batch boundaries. Provide
three emission operations: emit an exact gate record, copy an indexed original gate,
and emit a typed local/global-excluding crosstalk exposure. The host validates every
record and latches failures before more allocation, even if callbacks are ignored.
Invalid source indices/custom-as-gate copies and reserved/unused fields are errors.
Every sink callback and its context is borrowed only during translation: invoke
synchronously on the calling host thread, with no retention, concurrency or reentry.
Plugin gate/exposure buffers are valid, aligned, non-overlapping with host contexts,
and borrowed only for the emission call; the host copies them before returning.
Successful creation requires a non-null handle.
Gate/exposure emission uses ScheduledGateBuffer, with its existing budget and profile
checks. IDs are checked for target-width conversion before allocation. Measurement
order/kind/targets and source identities remain checked by the existing owner.

No noise sampling or quantum execution occurs until the whole input has normalized
and passed admission. Plugin failure does not promise classical-state rollback;
QuantumSystem remains poisoned until successful whole-host reset. Crosstalk remains
resolved through the existing noise/quantum continuation loop, including internal
measurements, rather than through plugin simulator access.

## Failure boundary

All fallible plugin functions return an integer status (zero success), with phase and numeric
status preserved in host errors. The host catches Rust panics in its own callbacks,
latches a failure and returns an error status. Rust panics/C++ exceptions must never
cross a plugin C boundary: each plugin must contain its own exceptions. The host
cannot recover from native aborts, invalid pointers or a plugin that never returns.
Destroy returns void, is non-failing and must not unwind; it runs on both successful and poisoned
sessions while the library is still alive. Loading uses eager symbol resolution on
POSIX. No result-string ownership or unbounded plugin error allocation is needed.

## Integration acceptance and remaining validation

Use a separately compiled public synthetic C plugin; do not use in-process Rust-only
fixtures as the sole ABI evidence. Cover descriptor rejection, null/failing create,
exactly-once destruction, library retention, concurrent independent sessions and
thread movement. Exercise read/output errors ignored by the plugin, malformed gates,
expansion bounds, custom tags, angle bits, both measurement namespaces, crosstalk,
and whole-message validation before any translation or simulator mutation.

Through QuantumSystem, check metadata insertion/RNG invariance, measured feedback,
poison persistence, reset, clone rejection and multiple inputs per shot. Python tests
must use sim and the existing MonteCarlo route, with repeated run calls and multiple
workers. Reject unsupported stack/tracing routes as existing event profiles do.
Run no timing benchmark until correctness and loaded-binary provenance are established.

This document does not itself declare a stable ABI. The header, implementation,
negative conformance tests and Python exposure must be reviewed together before the
interface is advertised as supported. Avoid promising ABI extensibility beyond the
explicit v1 size/version contract.
