# Scheduled event adapters

Mandatory v4 transport preserves opaque native events for a shot-local adapter.
The adapter normalizes complete batches into checked gates **before quantum
execution**. It supports the checked local idle-noise profiles and 1–16-qubit StateVec limit;
Python supports an opt-in adapter factory on the QIS/HUGR engines route. This is
not a general execution-time event API.

The first intended downstream use is a deterministic adapter that tracks runtime
bookkeeping and virtual phase conventions across batches, adjusting ordinary gates
for a reduced idle-only comparison. That adapter is not implemented or validated
here. Public synthetic fixtures establish transport contracts, not device fidelity.
Outcome-dependent adapters and non-idle noise require separate work.

## Configure and implement

Use `QisEngineBuilder::scheduled_event_batches(true)` with
`ScheduledEventIdleNoise::new(profile, factory)` (also available under the
compatibility name `ScheduledEventIdleZ`, retained for existing callers).
The alias wraps either checked profile; `ScheduledIdleZ` itself remains a Z-only
convenience constructor. The factory receives `ShotContext` and
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

## Python factory

Pair `qis_engine().scheduled_event_batches()` with
`scheduled_event_idle_z(qubits, adapter_factory, *, linear=0, sine=0, coherent=0)`
from `pecos_rslib` for Z/RZ noise, or
`scheduled_event_idle_noise(profile, adapter_factory)` with a profile returned by
`scheduled_idle_noise()` for configurable local idle channels including leakage.
Both factories use the same transport, admission and recovery contracts. Omission of idle noise before preparation applies to the
**normalized** gates emitted by the adapter, using the same per-qubit rules as
[scheduled idle-noise simulation](scheduled-idle-simulation.md#timing-and-noise-contract).
Source timing admission remains enforced even when a preparation gap is omitted.
Use StateVec and the explicit physical capacity as for v3.
The normal `pecos.sim(program).classical(...).noise(...).run(shots)` route and
QIS `.build()` simulations support this configuration, including HUGR/Guppy
lowering to QIS. Other stacks and operation tracing do not support this factory.

On the first v4 input of each host shot, `adapter_factory((run, worker, shot))`
must create a fresh object with two methods:

- `validate(batch)` returns `None`, or raises for an unsupported schema/semantics.
  It must not mutate adapter state. A boolean return is rejected.
- `translate(batch)` updates deterministic shot-local state and returns a **list**
  of `pecos_rslib.quantum.Gate` objects, at most 4096. Iterators are not consumed.
  Gates pass through the existing checked Rust writer and measurement validation.

`batch` is an owned, read-only snapshot with `runtime_shot_id`, `batch_index`,
`start_nanos`, and `duration_nanos`. Its `operations` property returns a copy of the
ordered list: ordinary `Gate` objects or `(tag, payload_bytes)` tuples for opaque
events. `measurements` returns `(original_position, native_id, program_id)` triples.
Modifying a returned list does not modify native input. No callback receives outcomes,
RNG or simulator state. A Python extension object may implement the same protocol;
no cross-extension Rust ABI or pointer capsule is required.

Callback errors include their stage and fail the shot through the existing Rust
poisoning contract. QIS execution releases the GIL while workers run, acquiring it
for each callback. External calls to built `run`, `run_with_workers`, and `reset` serialize on the
engine mutex with the GIL released. Calling any of these methods from a scheduled
adapter callback rejects before waiting, including calls to a different built
simulation. A poisoned mutex is reported separately from callback reentry. Factory isolation,
determinism, validation purity, callback termination and retained-memory bounds
remain trusted responsibilities. The reentry guard is thread-local: callbacks must
not start another thread that calls into the active simulation and then wait for
that thread, which would still deadlock. Capturing and sharing mutable state across shot
objects violates the contract even if the signatures are correct.

Python callbacks incur GIL and data-copy overhead. This API establishes a usable
integration path, not a throughput claim. The idle/timing policy and narrow physics
are unchanged; adding a factory does not admit additional noise channels.
