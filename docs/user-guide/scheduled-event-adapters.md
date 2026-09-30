# Scheduled event adapters (Rust)

Mandatory v4 transport preserves opaque native events for a shot-local adapter.
The adapter normalizes complete batches into checked gates **before quantum
execution**. It retains the existing idle-Z profile and 1–16-qubit StateVec limit;
there is no Python adapter configuration or general execution-time event API.

The first intended downstream use is a deterministic adapter that tracks runtime
bookkeeping and virtual phase conventions across batches, adjusting ordinary gates
for a reduced idle-only comparison. That adapter is not implemented or validated
here. Public synthetic fixtures establish transport contracts, not device fidelity.
Outcome-dependent behavior and additional noise channels require separate work.

## Configure and implement

Use `QisEngineBuilder::scheduled_event_batches(true)` with
`ScheduledEventIdleZ::new(profile, factory)`. The factory receives `ShotContext` and
must return an independent `Box<dyn ScheduledBatchAdapter>` for each shot.

The builder has three transport modes: off, v3, v4. Enabling scheduled batches
preserves a selected v4 mode; disabling batches turns transport off. Disabling
events downgrades v4 to v3 and leaves an already-disabled transport off. Existing
v3 and ordinary consumers reject v4; v3 still rejects opaque events.

- `validate` checks every event schema without side effects. Unknown or unsupported
  semantics must reject. All batches are validated before translation starts.
- `translate` receives a complete original batch and a borrowed `ScheduledGateBuffer`.
  Emit gates with `push`, preserving measurement order, kind and targets. Rejected
  appends remain latched; the adapter cannot replace the writer or clear failure.
- Callbacks cannot access outcomes, simulator state or RNG. Their deterministic
  semantics, validation purity, factory isolation and internal memory bounds remain
  trusted responsibilities. Signature checks do not prove physical correctness.

## Execution and recovery

The owner validates the whole wire input and source schedule before callbacks,
latches failure before invoking user code, then validates the entire expanded
schedule before execution. It runs one noise lifecycle per nonempty normalized
batch. Custom-only batches retain timing and identity without creating idle sites.
Adapters cannot split or retime batch boundaries.

Source and normalized timing histories persist independently across inputs and
commit only after successful execution. Removing a gate retains its source interval;
adding one affects normalized timing. Reset clears both histories and the adapter.
A clone cannot resume a live native shot with a fresh adapter, including when its
controller and simulator were cloned separately. Callback/execution failures and
caught panics require successful reset; malformed source input rejects before
callbacks. No rollback of adapter side effects is promised.

## Wire and bounds

The 16-byte header contains the batch magic, version 4, zero reserved bytes, u32 batch
count and u32 total byte length. All integers are little endian. Each batch contains:

1. A u64 length followed by exactly one v3 batch of ordinary gates.
2. A u64 event count and ordered records: u64 original position, tag, payload length,
   then payload bytes.
3. A u64 measurement count and triples of u64 original position, native result ID,
   source result ID. Positions must match the measurement gates; both ID namespaces
   must be unique per input.

Unknown records, versions, reserved bits and malformed lengths reject. Each source
or expanded batch allows 4096 operations; opaque payloads total at most 256 KiB per
batch. Wire and expanded v3-equivalent sizes are each capped at 64 MiB. These bounds
are not a process-RSS limit. Native extraction still has no aggregate output cap,
and adapters must bound their own retained state. The host retains only bounded
per-qubit timing and shot/ordinal state after processing.

V3 angle conversion is reused; arbitrary floating-point encodings are not preserved
bitwise. Normalization does not add another angle serialization round trip.
