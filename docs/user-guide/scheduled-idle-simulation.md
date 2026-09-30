# Scheduled idle-Z simulation

The opt-in QIS scheduled route carries original native batch timestamps into
`sim()` and applies a narrow idle-Z profile using `GeneralNoiseModel`. It supports
state-vector simulation with a fixed admitted profile of 1–16 physical qubits.
The simulator must have at least that capacity; any extra qubits are inaccessible
to scheduled input. This is an
experimental integration API; it does not enable an arbitrary general noise
configuration or establish agreement with a device model.

<!--skip: API template requires caller-supplied runtime and LLVM program; covered by integration tests.-->
```python
import pecos
import pecos_rslib as pr

classical = (
    pr.qis_engine()
    .selene_runtime_plugin(library_file, init_args, custom_event_policy="reject_unhandled")
    .scheduled_batches()
    .interface(pr.qis_helios_interface())
)
results = (
    pecos.sim(pecos.Qis(llvm_ir))
    .classical(classical)
    .qubits(2)
    .quantum(pr.state_vector())
    .noise(pr.scheduled_idle_z(2, linear=0.01, sine=0.02, coherent=0.03))
    .seed(42)
    .workers(1)
    .run(100)
)
```

The runtime must support scheduled extraction. `library_file`, `init_args`, and
`llvm_ir` above are supplied by the caller. An ordinary noise model rejects this
mandatory transport. Conversely, the scheduled profile rejects ordinary gate
messages; the canonical empty host-wait message is the only legacy exception.
Operation tracing is currently incompatible with this opt-in route.

## Timing and noise contract

Batch start and duration remain integer nanoseconds. For each physical qubit,
its idle interval runs from the end of its previous touching batch to the start
of its next touching batch. The initial end is zero. A qubit receives one idle
operation for that whole gap before its first operation in the batch. Operations
retain emission order, including same-batch measurement boundaries. A batch's
entire duration counts as busy time for each qubit it touches; no final idle tail
is invented after the last operation. Measurement consumes its preceding idle gap
and advances the cursor to the end of its batch, so a later operation cannot count
that earlier gap again. Backward/overlapping use of a qubit rejects.

By default, idle noise is included before preparation (`PZ`, reset-Z). Set
`scheduled_idle_z(..., idle_before_preparation=False)` in Python or call
`ScheduledIdleZ::with_idle_before_preparation(false)` in Rust to omit that gap's
noise when preparation is the first operation on that qubit in the batch. This is
a per-qubit choice: other qubits and gaps before earlier non-preparation gates
remain unchanged. Preparation still occupies the entire batch and advances the
cursor; capacity, overlap and finite-rate checks remain enforced. Omitting a
stochastic idle channel changes RNG consumption even when the reset erases its
physical effect, so the two policies need not produce identical seeded outcomes.

The profile enables only linear stochastic Z, sine-squared stochastic Z and
coherent RZ idle effects. Linear rates are inverse seconds; sine and coherent
rates are radians per second. All rates and duration products must be finite and
nonnegative. Gate faults, readout faults, leakage, crosstalk and custom events are
not supported. Each nonempty native batch goes through one existing general-noise
controller lifecycle; empty batches do not trigger a lifecycle. In particular,
this adapter is not permission to split arbitrary noise controllers.

## Admission, identity and recovery

QIS validates native-to-program measurement mappings and retains them until the
reply is delivered. Native shot IDs and consecutive batch ordinals are validated
across inputs. Host run/worker/shot identity is supplied separately by the existing
Monte Carlo path. IDs are not inferred from one another.

Every input is fully admitted before any of its gates execute. Invalid inputs
leave the quantum state and schedule cursor unchanged. Failures after execution
begins poison the quantum owner; successful whole-host reset is required before
reuse. Cloning a live scheduled owner also requires reset and a new shot context.
Terminal scheduled batches execute before the shot is finalized. Measurement
feedback invalidates earlier drain certification; completion requires a fresh
empty drain after the final feedback and execution.

The mandatory wire envelope is version 3: a 16-byte little-endian header (PECS
magic, version byte and three zero reserved bytes, u32 batch count, u32 total
length), followed by batches. Each batch has five u64 fields: runtime shot ID,
batch ordinal, start, duration, gate count. Each gate has five u64 fields: opcode,
first target, second target, theta IEEE-754 bits, phi IEEE-754 bits. Opcodes are
1 RXY, 2 RZ, 3 RZZ, 4 RXYXY, 5 reset-Z, 6 measure-Z, 7 leakage-aware measure.
Unused fields must be zero. Unknown records, nonfinite angles, invalid targets,
length errors and unsupported envelope versions reject. Each complete envelope
is limited to 64 MiB. This does **not** bound the upstream native drain's aggregate
allocation or guarantee termination of a misbehaving plugin.

## Evidence and limits

The integration tests use a public runtime with an explicitly synthetic timestamp
proxy. A one-second gap and coherent rate π rad/s change a Ramsey result from
zero to one; feedback and worker/shot reset are exercised through the normal
Python builder. These timestamps are test inputs, not device calibration.
The public runtimes tested previously emitted zero timing for the probes; such a
run cannot validate nonzero idle noise. Full-profile parity, experimental-data
agreement, leakage repumping and comparative speed remain separate work.
