# Native schedule extraction and its noiseless consumer

`QisRuntime` can return original native batches before idle insertion. The first
execution consumer is `NoiselessScheduledExecutor` in pecos-qis, used by the runnable
`scheduled_noiseless` example. It uses the existing QuantumSystem, pass-through
noise controller and state-vector engine. It is a diagnostic Rust path, not the
QisEngine/SimBuilder/Python path or a device-noise adapter. The consumer/API design
still requires review before landing PR #896.

## Consumer contract

- Fixed capacity of 1–16 state-vector qubits. Runtime and simulator are owned
  together; no cloning or mutable component escape is exposed.
- Native RXY, RZ, RZZ, RPP, reset and ordinary/leakage-aware measurements execute.
  All opaque events reject, including events that could be metadata. No custom
  physical profile or arbitrary noise controller is admitted.
- Caller-supplied host context and runtime-local shot ID are kept distinct. Batch
  ordinals and state persist across submissions. Callers assign unique contexts
  across workers; this path does not introduce a Monte Carlo scheduler.
- Validate the entire extraction result before quantum mutation: identities,
  finite angles, qubit capacity, distinct two-qubit targets, timing overflow and
  per-qubit timing across batches, and measurement mappings. Duplicate source or
  native measurement IDs within one extraction result reject. Preserve operation
  emission order within each batch, including repeated targets.
- After admission, translate ordinary gates into the existing ByteMessage format
  and make one QuantumSystem process call for the extraction result. Timing has
  no noise effect in this admitted profile. Return the original batches alongside
  outcomes, and deliver outcomes through the runtime's existing feedback API.
- Any submission/feedback/finalization failure or caught panic poisons the owner.
  Both runtime and quantum state must reset successfully before another shot.
  Finish drains and executes deferred work before ending the runtime shot.

No opaque event is put into an ignorable legacy message. A mandatory scheduled
execution envelope is still needed before a general noise consumer can receive
batch timing and physical events through the normal engine boundary. The narrow
v2 frame is unchanged. Python exposure and Guppy-program integration remain future
work; this consumer takes native Operation inputs directly.

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
cargo run -p pecos-qis --example scheduled_noiseless -- simple
cargo run -p pecos-qis --example scheduled_noiseless -- soft-rz
```

Both prepare and measure a flipped qubit, with nontrivial program identifiers.
Tests also exercise two-qubit gates, repeated shots and inputs, deferred schedules,
whole-input admission, feedback failures/panics and unsuccessful reset. These are
software checks, not statistical simulator parity or performance benchmarks.
