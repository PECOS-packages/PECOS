# Scheduled event adapters

Mandatory v7 transport preserves opaque native events for a shot-local adapter.
The adapter normalizes complete batches into checked gates and optional typed
crosstalk exposures **before quantum execution**. The noise controller resolves
those exposures during execution. Profiles retain the 1–16-qubit StateVec limit;
Python supports an opt-in adapter factory on the QIS/HUGR engines route. This is
not a general execution-time event API.

The first intended downstream use is a deterministic adapter that tracks runtime
bookkeeping and virtual phase conventions across batches, adjusting ordinary gates
for a reduced idle-only comparison. That adapter is not implemented or validated
here. Public synthetic fixtures establish transport contracts, not device fidelity.
Outcome-dependent adapter callbacks and broader physical noise require separate work.

## Configure and implement

Set the Selene runtime policy to `RuntimeCustomEventPolicy::Capture` explicitly
when forwarding custom events to this consumer (Python: `custom_event_policy="capture"`).
The default runtime policy rejects unacknowledged custom events during extraction.

Use `QisEngineBuilder::scheduled_event_batches(true)` with
`ScheduledEventNoise::new(profile, factory)`. `ScheduledEventIdleNoise` and
`ScheduledEventIdleZ` are compatibility aliases for this canonical type.
The aliases preserve the supplied checked profile; `ScheduledIdleZ` itself remains
a Z-only convenience constructor. The factory receives `ShotContext` and
must return an independent `Box<dyn ScheduledBatchAdapter>` for each shot.

The builder has three transport modes: off, v6, v7. Enabling scheduled batches
preserves a selected v7 mode; disabling batches turns transport off. Disabling
events downgrades v7 to v6 and leaves an already-disabled transport off. Existing
v6 and ordinary consumers reject v7; v6 still rejects opaque events.

- `validate` checks every event schema without side effects. Unknown or unsupported
  semantics must reject. All batches are validated before translation starts.
- `translate` receives a complete original batch and a borrowed `ScheduledGateBuffer`.
  Emit gates with `push`, preserving measurement order, kind and targets. Opted-in
  profiles also accept `push_crosstalk` with typed spectator exposures. Rejected
  appends remain latched; the adapter cannot replace the writer or clear failure.
- Callbacks cannot access outcomes, simulator state or RNG. Their deterministic
  semantics, validation purity, factory isolation and internal memory bounds remain
  trusted responsibilities. Signature checks do not prove physical correctness.

## Execution and recovery

The owner validates the whole wire input and source schedule before callbacks,
latches failure before invoking user code, then validates the entire expanded
schedule before execution. It runs one noise lifecycle per nonempty normalized
batch, including any internal noise/quantum exchanges. Custom-only batches retain
timing and identity without creating idle sites.
Adapters cannot split or retime batch boundaries.

Source and normalized timing histories persist independently across inputs and
commit only after successful execution. Removing a gate retains its source interval;
adding one affects normalized timing. Reset clears both histories and the adapter.
A clone cannot resume a live native shot with a fresh adapter, including when its
controller and simulator were cloned separately. Callback/execution failures and
caught panics require successful reset; malformed source input rejects before
callbacks. No rollback of adapter side effects is promised.

## Wire and bounds

The 16-byte header contains the batch magic, version 7, zero reserved bytes, u32 batch
count and u32 total byte length. All integers are little endian. Legacy version 4
embedded float-angle frames and is rejected; rebuild those messages with the
current encoder. Each batch contains:

1. A u64 length followed by exactly one v6 batch of ordinary gates.
2. A u64 event count and ordered records: u64 original position, tag, payload length,
   then payload bytes.
3. A u64 measurement count and triples of u64 original position, native result ID,
   source result ID. Positions must match the measurement gates; both ID namespaces
   must be unique per input.

Unknown records, versions, reserved bits and malformed lengths reject. Each source
or expanded batch allows 4096 operations; opaque payloads total at most 256 KiB per
batch. Normalized exposures also have a 16,384-candidate-target limit per batch;
global exposures charge the full configured capacity, even with exclusions. Wire
and expanded sizes are each capped at 64 MiB, including eight bytes per exposure
ID in addition to the fixed normalized record charge. These bounds
are not a process-RSS limit. Native extraction still has no aggregate output cap,
and adapters must bound their own retained state. The host retains only bounded
per-qubit timing and shot/ordinal state after processing.

Nested v6 angle words are exact little-endian u64 Angle64 fractions; all 64 bits
are preserved. Normalization does not add another angle serialization round trip.

## Python factory

Pair `qis_engine().scheduled_event_batches()` with
`scheduled_event_idle_z(qubits, adapter_factory, *, linear=0, sine=0, coherent=0)`
from `pecos_rslib` for Z/RZ noise, or
`scheduled_event_idle_noise(profile, adapter_factory)` with a profile returned by
`scheduled_idle_noise()` for configurable local idle channels including leakage.
For checked local gate/preparation/readout faults, use
`scheduled_event_local_noise(local, adapter_factory)` with `scheduled_local_noise()`.
All factories enforce the same transport, recovery, batch and measurement-order
contracts described under [shared scheduled limits](scheduled-idle-simulation.md#shared-scheduled-admission-and-execution-limits). Omission of idle noise before preparation applies to the
**normalized** gates emitted by the adapter, using the same per-qubit rules as
[scheduled idle-noise simulation](scheduled-idle-simulation.md#timing-and-noise-contract).
Source timing admission remains enforced even when a preparation gap is omitted.
Use StateVec and the explicit physical capacity as for v6.
The normal `pecos.sim(program).classical(...).noise(...).run(shots)` route and
QIS `.build()` simulations support this configuration, including HUGR/Guppy
lowering to QIS. Other stacks and operation tracing do not support this factory.

On the first v7 input of each host shot, `adapter_factory((run, worker, shot))`
must create a fresh object with two methods:

- `validate(batch)` returns `None`, or raises for an unsupported schema/semantics.
  It must not mutate adapter state. A boolean return is rejected.
- `translate(batch)` updates deterministic shot-local state and returns a **list**
  of `pecos_rslib.quantum.Gate` or `pecos_rslib.ScheduledCrosstalk` objects, at most
  4096 in total. Iterators are not consumed. All items pass through the checked
  Rust writer; program measurement order, kind and targets must remain unchanged.

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
integration path, not a throughput claim. The idle/timing policy is shared; each checked profile determines the admitted
noise channels. A batch adapter itself does not enable arbitrary noise models.

## Measurement-crosstalk exposures

Opt in explicitly while retaining an existing idle or local profile:

```python
import pecos_rslib as pr


class CrosstalkAdapter:
    """Invented schema: event (7, b"") exposes qubit 1 to a nearby readout."""

    def validate(self, batch):
        for op in batch.operations:
            if isinstance(op, tuple) and op != (7, b""):
                message = "unknown synthetic event"
                raise ValueError(message)

    def translate(self, batch):
        return [pr.ScheduledCrosstalk.local([1]) if isinstance(op, tuple) else op for op in batch.operations]


def adapter_factory(_context):
    return CrosstalkAdapter()


base = pr.scheduled_idle_noise(2, linear=0.02)
profile = pr.scheduled_measurement_crosstalk(
    base,
    {"0->1": 1.0, "1->0": 1.0},
    local_probability=0.03,
    global_probability=0.01,
)
# Pass this wrapper to sim(...).noise(...), with scheduled_event_batches enabled.
noise = pr.scheduled_event_idle_noise(profile, adapter_factory)
```

The values are invented examples. The transition map and probabilities use
`GeneralNoiseModel` validation, sampling and leakage semantics. Invalid settings
reject even when their associated probability is zero. Rust exposes
`ScheduledNoise::with_measurement_crosstalk(local, global, model)`.

At an appropriate position in `translate`'s returned list, emit:

- `pr.ScheduledCrosstalk.local([1, 0])` for potential victims in that sampling order.
- `pr.ScheduledCrosstalk.global_excluding([0])` for prepared logical qubits outside
  the exclusion list, in the general model's order.

IDs must be unique within a record and within the profile's capacity. Empty local
lists select no victims; empty global lists exclude none. Unprepared and already
leaked victims are skipped according to the general model. Separate exposures to
the same victim remain separate, ordered effects. Explicit zero-probability
exposures retain the general model's probability draws; they are not equivalent
to ignored metadata for seeded RNG comparisons.

No exposures are inserted automatically. The runtime adapter is responsible for
interpreting its event schema and selecting victims or exclusions. Ordinary gate
output cannot smuggle in exposure payloads; use the typed writer. The v7 source
wire remains unchanged, and v6 cannot encode these effects. Preparation crosstalk,
transition hooks and arbitrary execution-time callbacks remain unsupported.

An exposure is ordered noise metadata: it creates no physical occupancy, closes
no idle gap and does not advance a qubit's busy endpoint. Thus a later physical
operation still receives the complete native idle interval, including nonlinear
idle noise. The same rule means an exposure samples its victim as of the victim's
previous physical operation: idle noise accrued since then is applied only at
the victim's next physical operation, after the exposure and its transition.

Deferring X/Y/leakage or RX/RY idle noise can change the output distribution.
An asymmetric transition map is not required: the internal Z measurement does
not commute with coherent RX/RY evolution, even with symmetric `0->1` / `1->0`
transitions. For example, prepare `|0>`, accrue an RX(pi/2) idle rotation, expose
the qubit with a certain symmetric flip, then apply RX(-pi/2). Deferred idle
execution leaves `|1>` with certainty; applying the idle rotation before the
exposure gives equal probabilities of 0 and 1 at the final Z readout.

Z/RZ idle channels preserve the physical outcome distribution under this
reordering, but identical seeds do not guarantee identical trajectories or later
readouts. In particular, if an exposure leaks a qubit, deferred idle noise skips
that qubit and its stochastic draws. Applying that idle noise before the exposure
would consume those draws, changing subsequent noise RNG state.

Reordering exposure records alone cannot move accrued idle noise ahead of them;
idle is delivered at the victim's next physical operation. Adapters must use
this profile only when its deferred-idle semantics suit the intended model.
These synthetic contracts do not establish device fidelity.

The normal path is `sim() → MonteCarloEngine → HybridEngine`: Selene and its runtime
provide classical control, while `QuantumSystem` exchanges work and internal
measurement results between `GeneralNoiseModel` and the quantum engine. Each
native batch retains one controller lifecycle. Internal spectator measurements
never become program results; ordinary measurements return to the live runtime
for feedback. Shots and workers retain the existing Monte Carlo lifecycle, RNG
and identity handling. This path does not compile a static simulation trace.

## Native scheduling across release

Scheduled extraction submits qubit release to the runtime in source order without
adding a host barrier or draining the queue at that release. Later operations in
the same input reach the scheduler before normal extraction. The runtime owns
queued work and native slot reuse. Result mappings remain available after release.
Source-annotated flat lowering retains its separate provenance boundary.

The shared [timing and noise contract](scheduled-idle-simulation.md#timing-and-noise-contract)
applies to every scheduled consumer: use native timestamps without rescheduling.
Explicit source barriers and terminal draining still apply.
