# Scheduled idle-noise simulation

The opt-in QIS scheduled route carries original native batch timestamps into
`sim()` and applies checked local idle channels using `GeneralNoiseModel`. It supports
state-vector simulation with a fixed admitted profile of 1–16 physical qubits.
The simulator must have at least that capacity; any extra qubits are inaccessible
to scheduled input. This is an
experimental integration API; it does not enable an arbitrary general noise
configuration or establish agreement with a device model.

"Scheduled" refers to timing supplied by the running runtime. The program runs
dynamically: runtime batches enter the noise model and simulator, and measurement
results return to the program to drive feedback and branches. Validation buffers
the current input; simulation does not require a precomputed whole-shot trace.

Each handle introduced by an explicit qubit-allocation record stays alive through
measurements until the program releases it. A later reset or measurement on that
handle reaches the same runtime allocation, including across feedback
continuations. Implicitly materialized legacy handles still release on
measurement, even when explicit and implicit handles occur in the same stream.

Capacity admission counts handles still live from earlier inputs, including
measured handles that have not been released. Set `.qubits(...)` to the maximum
simultaneously live physical allocation count before running a dynamic program.
The scheduled route requires this capacity explicitly. Flat and metadata routes
can infer it from a preloaded complete collector or the first submitted input,
but cannot safely increase it once the plugin has initialized: the plugin ABI
has no state-preserving resize. A continuation that needs more capacity rejects
before native submission. Reset and configure sufficient capacity before starting
a new shot; automatic reinitialization never discards an in-progress shot.

A failed native release leaves allocation state uncertain and blocks further
execution and shot completion until a successful reset. Shot lifecycle cleanup
remains the plugin's responsibility; the public runtimes clear allocations at
`shot_end`, including measured handles not explicitly released by the program.

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
of its next touching batch. The initial end is zero. Except before preparation
as described below, a qubit receives one idle operation for that whole gap before
its first operation in the batch. Operations retain emission order, including
same-batch measurement boundaries. A batch's
entire duration counts as busy time for each qubit it touches; no final idle tail
is invented after the last operation. Backward/overlapping use of a qubit rejects.

When preparation (`PZ`, reset-Z) is the first operation on a qubit in a batch,
its preceding idle channel is omitted: the ideal reset erases the effect of all
admitted local channels, including leakage. Preparation clears the general-noise
leakage record as well as resetting the simulator state. Other qubits and gaps before
non-preparation gates retain their idle noise. Preparation still occupies the
entire batch and advances the cursor; capacity, overlap and finite-rate checks
remain enforced. The local fault profile below also emits a real reset before its
preparation bit flip, so it uses the same omission. This does not justify omission
for correlated channels or other preparation operations.

Earlier versions sampled idle noise before preparation. Omitting those discarded
channels avoids unnecessary work and changes RNG consumption, so seeded outcomes
can differ from earlier versions even though the output distribution is unchanged.
All scheduled Python factories and Rust profiles use this rule without a
configuration option.

## Idle families and leakage

`scheduled_idle_z()` retains its existing Z/RZ-only behavior. The additive
`scheduled_idle_noise()` factory accepts the existing general-noise idle families:

| Family | Model keys | Per-gap behavior for duration `d` seconds |
| --- | --- | --- |
| Linear | X, Y, Z, L | One event with probability `min(linear * d, 1)`, followed by an axis drawn from the normalized weights |
| Sine-squared | X, Y, Z, L | Independent per-axis probability `sin(sine * multiplier * d)**2` |
| Coherent | RX, RY, RZ | Rotation angle `coherent * multiplier * d`, in RX/RY/RZ order |

`L` means leakage. Sine-squared leakage probability is not monotonic in idle
duration: it can decrease again for longer gaps. Linear leakage instead saturates
at one. These are the existing general-noise family semantics.

Linear weights must sum to one within `1e-10`; sine and
coherent multipliers are deliberately unnormalized. Rates and weights must be
finite and nonnegative, and rate/multiplier/duration products must remain finite.
Validation applies even to disabled families. Empty sine/coherent maps disable
those families; an empty linear distribution rejects. Defaults are Z for the
stochastic models and RZ for coherent evolution, with all rates zero.

```python
import pecos_rslib as pr

profile = pr.scheduled_idle_noise(
    2,
    linear=0.02,
    linear_model={"Z": 0.5, "L": 0.5},
    coherent=0.03,
    coherent_model={"RZ": 1.0},
)
```

These are illustrative process-rate inputs, not calibration values. The factory
performs no average-infidelity conversion, frequency conversion or external
scaling. Linear rates use inverse seconds; sine and coherent rates use radians
per second. Supply any conversions explicitly before constructing the profile.

Leakage state persists across batches, host waits, and measurement feedback.
Ordinary Z measurement of a leaked qubit returns 1; leakage-aware measurement
returns 2. Measurement does not clear leakage. Ideal preparation and whole-shot
reset clear it. These are the existing `GeneralNoiseModel` leakage semantics,
not a new physical leakage model. Native-to-program result handling must support
the selected measurement kind.

Gate faults, readout faults, crosstalk and repumping are not configurable in this
profile. Each nonempty batch goes through one existing general-noise controller
lifecycle; empty batches do not trigger a lifecycle or split an idle interval.
In particular, this adapter is not permission to split arbitrary noise controllers.
The v4 [event adapter](scheduled-event-adapters.md) can use the same profile via
`scheduled_event_idle_noise(profile, adapter_factory)`.

## Local gate, preparation and readout faults

`scheduled_local_noise(idle, *, p1=0, p2=0, prep=0, meas0=0, meas1=0)` adds a
checked subset of `GeneralNoiseModel` to an existing `scheduled_idle_noise()` profile:

- `p1`: a uniform X/Y/Z fault after an emitted single-qubit gate, including `RZ`.
  The profile treats emitted `RZ` as a physical operation. A virtual frame update
  must be resolved by the runtime/adapter without emitting a physical `RZ`; there
  is no per-gate noiseless override in this profile.
- `p2`: a uniform nonidentity two-qubit Pauli fault after a two-qubit gate. This
  event probability is independent of rotation angle.
- `prep`: an X fault after preparation resets the qubit and clears leakage.
- `meas0`, `meas1`: classical readout flips 0→1 and 1→0, respectively.

Preparation faults apply to emitted reset-Z operations. Allocation by itself does
not imply a reset in the scheduled stream; runtimes that prepare on allocation
must expose that operation to model its fault. No hidden preparation is inserted.

All probabilities must be finite and in [0, 1]. They are event probabilities,
not average gate infidelities. Gate-induced emission/leakage, preparation leakage,
seepage, custom fault maps, angle-dependent rates, crosstalk, repumping and automatic
post-two-qubit idle injection are not configurable. Unsupported Python keywords
raise rather than being silently ignored. Existing idle leakage remains supported:
leakage-aware readout returns 2 without readout flips, whereas ordinary leaked
readout starts at 1 and is subject to `meas1`.

```python
import pecos_rslib as pr

idle = pr.scheduled_idle_noise(2, linear=0.02)
local = pr.scheduled_local_noise(idle, p1=0.01, p2=0.02, prep=0.03, meas0=0.04, meas1=0.05)
```

In Python, the `idle` base must be created with `scheduled_idle_noise()`;
`scheduled_idle_z()` and an existing local profile are not accepted. For a Z-only
base, use `scheduled_idle_noise()` with its default Z models and the same rates.

Use `local` as `.noise(local)` with v3, or
`scheduled_event_local_noise(local, adapter_factory)` with v4. These numbers are
invented examples. Rust exposes `ScheduledLocalNoise::new(idle, p1, p2, prep,
meas0, meas1)` and `ScheduledEventNoise::new(local, factory)`. Unlike Python, the
Rust constructor accepts a carrier that already contains local faults: it replaces
all five local probabilities while preserving its idle configuration, rather than
adding or combining the old and new probabilities. The canonical Rust
carrier is `ScheduledNoise`; `ScheduledIdleNoise` is a compatibility alias that
retains the entire configuration, including local faults. `ScheduledEventIdleZ`
and `ScheduledEventIdleNoise` likewise alias `ScheduledEventNoise`. The Python
idle and local factories retain their distinct profile classes.

## Shared scheduled admission and execution limits

Every scheduled profile admits at most 4096 gates per batch. Same-qubit
measurement, reset and reuse (including repeated measurement) execute in the
runtime's original operation order. Each measurement captures the leakage state
at its own position, so a later reset cannot change an earlier readout. This
applies to both original v3 and normalized v4 gates, including idle-only profiles.
Adapters cannot split or retime native batches.

Readout faults retain the general model's batch-level sampling order: gate and
idle faults are sampled first, then readout faults in measurement order. Capturing
leakage consumes no randomness. The general controller also records leakage per
measurement on legacy inputs. Scheduled profiles remain bounded and exclude
measurement-conditioned crosstalk, emission and seepage. Measurement results still
return to the live runtime to resume program feedback.

Each prepared batch retains one noise-controller lifecycle. Inserted idle commands
are included in an expansion budget of sixteen output commands per prepared
command (the admitted channels emit at most eight per idle target and three per
gate). Output bytes and command counts are checked before simulator execution.
No second simulator call is allowed for a scheduled batch; an unexpected
continuation fails and poisons the shot. These limits bound admitted noise
expansion, not native extraction or arbitrary adapter allocations. Existing
shot context, clone and whole-host reset requirements still apply.

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

The extended profile has deterministic Rust regressions for forced leakage,
leakage-aware readout, measurement busy time, reset and clone isolation, and
nonlinear gaps across metadata-only batches. Mixed-channel scheduled execution
is compared with explicit general-noise idle operations for both outcomes and
RNG state. Python native-runtime tests exercise forced leakage, feedback and reset
through both transport routes. These tests do not establish full device parity.
