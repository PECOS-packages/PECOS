# Native schedule extraction and its experimental timing consumer

`QisRuntime` can return original native batches before idle insertion. The experimental
`ScheduledExecutor` lives in `tests/support`, shared by integration tests
and the runnable `scheduled_runtime` example. It is not a supported library API.
It uses the existing QuantumSystem and state-vector engine, with either pass-through
noise or a narrow GeneralNoiseModel idle-Z profile. It is an experimental Rust path,
not the QisEngine/SimBuilder/Python path or a device-noise adapter. The idle profile
consumes native timing; physical event handling and broader noise-model admission
remain unimplemented.

## Consumer contract

- Fixed capacity of 1–16 state-vector qubits. Runtime and simulator are owned
  together; no cloning or mutable component escape is exposed.
- Native RXY, RZ, RZZ, RPP, reset and ordinary/leakage-aware measurements execute.
  All opaque events reject, including events that could be metadata. No custom
  event profile or arbitrary noise controller is admitted. The optional idle-Z
  profile enables only linear stochastic Z, sine-squared stochastic Z and coherent
  RZ idle noise; gate, readout, leakage and crosstalk rates remain zero.
- Caller-supplied host context and runtime-local shot ID are kept distinct. Batch
  ordinals and state persist across submissions. Callers assign unique contexts
  across workers; this path does not introduce a Monte Carlo scheduler.
- Validate the entire extraction result before quantum mutation: identities,
  finite angles, qubit capacity, distinct two-qubit targets, timing overflow and
  per-qubit timing across batches, and measurement mappings. Duplicate source or
  native measurement IDs within one extraction result reject. Preserve operation
  emission order within each batch, including repeated targets.
- Translate ordinary gates and inferred idles into the existing ByteMessage format
  after whole-input admission. Ideal execution makes one QuantumSystem process call
  per extraction result. Idle-Z execution makes one per nonempty original native batch,
  including all its continuations, and returns feedback after each batch. Host
  submission boundaries do not add or remove native noise-controller boundaries.
  Return the original batches alongside outcomes.
- Any submission/feedback/finalization failure or caught panic poisons the owner.
  Both runtime and quantum state must reset successfully before another shot.
  Finish drains and executes deferred work before ending the runtime shot. After
  clean completion, the next shot reuses the native instance while resetting
  quantum state and host bookkeeping. First use and explicit abandonment/recovery
  perform a full runtime reset.

No opaque event is put into an ignorable legacy message. A mandatory scheduled
execution envelope is still needed before a general noise consumer can receive
batch timing and physical events through the normal engine boundary. The narrow
v2 frame is unchanged. Python exposure and Guppy-program integration remain future
work; this consumer takes native Operation inputs directly.

## Idle-Z timing contract

Native timestamps are nanoseconds; inferred idle durations enter GeneralNoiseModel
in seconds. Linear rate is inverse seconds; sine and coherent rates are radians
per second. Rates and rate-duration products must be finite and nonnegative.
For each touched qubit, insert the whole interval from its previous native batch
end to the current batch start, once before that qubit's first operation in the
batch. Repeated targets within the batch share that idle site. Do not insert idle
sites at empty batches or host submission boundaries, split nonlinear intervals,
or invent trailing idle after the final operation. Initial per-qubit end time is
zero. These are this profile's explicit semantics, not a claim about device physics.

A nonempty original native batch is a semantic controller boundary. An artificial
split inside that batch is not: supporting events must not restart the controller at
each event. This distinction remains necessary when broadening the profile.
All opaque events still reject, including possible metadata; metadata insertion
invariance is not claimed for an unsupported input. Empty native batches are tested.

The harness currently preflights the complete extraction result and then prepares
individual messages for the idle profile before executing any of them. This extra
encoding is intentional scaffolding, not a performance-oriented production API.

## Memory and native lifecycle

Each native callback batch admits at most 4096 operations and 256 KiB of opaque
bytes. These are allocation/read limits within an indivisible native batch.
There is no aggregate extraction cap: returned output and the consumer's prepared
commands grow with the schedule. No artificial barriers are inserted to meet a
budget. A streaming/backpressure API remains to be designed for large workloads.

Scheduled shot completion or replacement requires a successful terminal drain
since the last non-empty submission. Reset explicitly abandons work. The consumer
adds execution and feedback to that extraction lifecycle. Runtime-local shot IDs
are not globally unique execution identities.

## Runnable diagnostic

```sh
cargo run -p pecos-qis --example scheduled_runtime -- simple
cargo run -p pecos-qis --example scheduled_runtime -- soft-rz idle-z
```

Both prepare and measure a flipped qubit, with nontrivial program identifiers.
Tests also exercise two-qubit gates, repeated shots and inputs, deferred schedules,
whole-input admission, feedback failures/panics, unsuccessful reset and native
instance reuse. Admission regressions check duplicate/mismatched measurement IDs,
leakage flags, distinct targets, outcome counts and encoded measurement kinds.
These are software checks, not statistical simulator parity or performance
benchmarks.

The CLI is a native-runtime smoke test. Timing sensitivity is verified separately
by a Ramsey test: two inverse pulses with no gap return zero, while a one-second
gap at coherent rate pi radians/second returns one. Further tests inspect the
single encoded idle site, exercise nonlinear and stochastic noise, compare seeded
outcomes and the subsequent underlying noise-RNG draw across extraction groupings,
check native batch dispatches, reject invalid rates/overflow/events before quantum
mutation, and run repeated native shots with deferred measurement feedback.
No statistical device parity or performance comparison has been performed.
