# Scheduled event adapters (Rust)

The checked scheduled path can opt into **mandatory v4** to retain opaque native
events until a shot-local adapter validates and normalizes them. Existing v3 and
ordinary consumers reject v4. The default v3 scheduled path still rejects custom events.

This first adapter is a deterministic batch normalizer with the existing idle-Z
noise profile and 1–16-qubit StateVec admission. It is not a general execution-time
event handler: callbacks cannot read measurement outcomes, access a simulator or RNG,
or configure additional noise channels. Python adapter configuration is not exposed.

## Configuration

Use `QisEngineBuilder::scheduled_event_batches(true)` to opt into the v4 producer,
and `ScheduledEventIdleZ::new(profile, factory)` as the noise configuration on the
ordinary Rust simulation route. The factory receives the explicit host `ShotContext`
and returns a `Box<dyn ScheduledBatchAdapter>`. It must create independent mutable
state for each shot; immutable configuration may be shared. Adapter sessions need
not implement Clone. Cloning a live scheduled host blocks it until whole-host reset.

`ScheduledBatchAdapter::validate` checks supported tags, payload schemas and semantic
requirements for every batch in the input before translation begins. It must not
change state. `translate` receives one original batch and a `ScheduledGateBuffer`.
Emit supported ordinary gates through its checked `push` method, including all source
measurements in their original order, kind and target. Events can update classical
adapter state or expand to gates; unknown events must return an error. A rejected
buffer append stays latched even if the adapter catches its error. The buffer is a
borrowed writer over owner-held output and failure state; adapters cannot construct
a replacement or clear that state.

The handler is trusted Rust code. Signature/type checks do not prove that its physics
are correct, that it avoids interior mutation during validation, or that its factory
does not share state. Adapter implementations must bound their own retained state.
Host bounds cover transport and expanded gates, not arbitrary adapter allocations.
Do not use this API for effects that depend on results within the current input.

## Ordering and execution

1. Decode and validate the entire v4 input, including original timing, identities,
   payload limits and source qubit capacity, before invoking user callbacks.
2. Latch host failure state before the factory or adapter can run or unwind.
3. Validate all event schemas, then normalize batches in native order. Translation
   may update shot-local classical state, but all translation for this input happens
   before its quantum execution. There is no rollback if a callback fails.
4. Validate the whole expanded schedule, preserving measurement order and targets.
   Adapters cannot change batch identity, timing or boundaries through their output.
5. Execute the existing idle-Z model once per nonempty normalized native batch.
   Custom-only batches remain in ordinal/identity accounting. An event that emits no
   gates does not create a noise-model invocation or an extra idle site.

Source and normalized timing histories persist separately across inputs and commit
only after successful execution. Removing a gate does not erase its source interval;
adding a gate affects the normalized timeline without rewriting source history.
Reset clears both histories and discards the adapter, and the next
input creates a fresh session. Callback errors and caught panics keep the execution
owner poisoned until a successful whole-host reset. Malformed wire input rejects
before callbacks or quantum/noise mutation. Measurement results retain the source
mapping established by the classical engine and are fed back through its normal path.

## Wire and retention contract

The outer 16-byte header uses the existing batch magic, version 4 with zero reserved
bytes, a u32 batch count and a u32 exact total length. Integers are little endian.
Each batch contains a u64 length and one complete v3 batch of ordinary gates, followed
by a u64 event count. Each event has u64 original operation position, tag and payload
length, followed by its bytes. A u64 measurement count then precedes triples of u64
original operation position, native result ID and source result ID. Original event
positions must be strictly increasing; measurement positions must match the ordered
measurement gates after interleaving. Both result namespaces are unique per input.
Unknown versions, reserved bits, nested gate types and malformed records reject.

A source or expanded batch permits at most 4096 operations. Opaque payloads total at
most 256 KiB per original batch. Input wire and expanded v3-equivalent size each have
a 64 MiB limit; these are serialization bounds, not a 64 MiB process-RSS guarantee.
The native extraction API still has no aggregate returned-schedule cap. Its existing
per-batch limits apply; this consumer bounds the subsequent transport. No event
history is retained by the host after processing, apart from adapter-owned state.

The envelope uses v3 gate-angle conversion; it does not promise bitwise preservation
of arbitrary floating-point angle encodings. Normalization does not add another
angle serialization round trip before execution.

## Evidence and remaining work

Synthetic tests cover metadata/RNG/idle invariance, event-to-X measurement ordering,
state across inputs, distinct worker contexts, clone/reset isolation, malformed and
oversized input, output expansion and swallowed errors, unsupported consumers, panic
poisoning, and classical terminal-feedback/drain completion. These establish software
contracts, not correctness of an external adapter or a device model. Private adapters,
broader noise profiles, outcome-dependent events and Python exposure need separate
implementation and validation.
