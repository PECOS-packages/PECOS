# Scheduled idle-noise simulation

The opt-in QIS scheduled route carries original native batch timestamps into
`sim()` and applies checked local idle channels using `GeneralNoiseModel`. It supports
state-vector simulation with a fixed admitted profile of 1–16 physical qubits,
or `pr.stabilizer()` with 1–512 qubits. Stabilizer execution requires zero coherent
idle rate and supported Clifford rotations. It accepts stochastic idle noise,
including the sine-squared dephasing channel, and the checked local/crosstalk profiles.
The 512-qubit bound is an explicit resource policy, separate from the wire format.
Coherent noise is never converted automatically: choosing a stochastic profile
changes the model and must be recorded explicitly in comparisons.
The simulator must have at least the profile's capacity; extra qubits are inaccessible
to scheduled input, and state-vector capacity must never exceed 16. `sim()` checks
the concrete backend and profile before allocating the simulator. Scheduled execution
requires an explicit `state_vector()` or `stabilizer()` builder; parallel/dense state
vectors and other backends remain unsupported. This is an experimental integration API; it does not enable an arbitrary general noise
configuration or establish agreement with a device model.

"Scheduled" refers to timing supplied by the running runtime. The program runs
dynamically: runtime batches enter the noise model and simulator, and measurement
results return to the program to drive feedback and branches. Validation buffers
the current input; simulation does not require a precomputed whole-shot trace.

For allocation lifetimes, capacity inference, cloning and error recovery on all
routes, see [Selene runtime execution](selene-runtime-execution.md). Scheduled
extraction requires an explicit nonzero capacity. Once collection begins, any
error—including capacity or duplicate-allocation admission—requires reset even
when native state was not changed. Initial source-shape validation happens before
collection; it does not by itself poison the runtime.

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

Every scheduled consumer uses the runtime's batch start times and durations to
calculate idle exposure. Consumers must not reschedule operations or replace those
timestamps with a reconstructed timeline. Explicit source barriers and terminal
draining still apply. For release ordering, see
[native scheduling across release](scheduled-event-adapters.md#native-scheduling-across-release).

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
preparation X or leakage fault, so it uses the same omission. This does not justify
omission for correlated channels or other preparation operations.

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

This idle factory does not configure gate faults, readout faults, crosstalk or
repumping. The local profile below adds gate/readout faults; the v7
[measurement-crosstalk wrapper](scheduled-event-adapters.md#measurement-crosstalk-exposures)
adds explicit exposures. Each nonempty batch goes through one existing general-noise controller
lifecycle; empty batches do not trigger a lifecycle or split an idle interval.
In particular, this adapter is not permission to split arbitrary noise controllers.
The v7 [event adapter](scheduled-event-adapters.md) can use the same profile via
`scheduled_event_idle_noise(profile, adapter_factory)`.

## Local gate, preparation and readout faults

`scheduled_local_noise(idle, *, p1=0, p2=0, prep=0, meas0=0, meas1=0,
prep_leak_ratio=0, prep_crosstalk=0, p2_pauli_model=None, p2_angle_params=None, p2_angle_power=None,
p1_emission_ratio=0, p1_emission_model=None, p1_seepage_prob=0,
p2_emission_ratio=0, p2_emission_model=None, p2_seepage_prob=0)`
adds a checked subset of
`GeneralNoiseModel` to an existing `scheduled_idle_noise()` profile:

- `p1`: probability of a fault at an emitted single-qubit gate, including `RZ`.
  Without emission, this applies a uniform X/Y/Z fault after the gate.
  The profile treats emitted `RZ` as a physical operation. A virtual frame update
  must be resolved by the runtime/adapter without emitting a physical `RZ`; there
  is no per-gate noiseless override in this profile.
- `p2`: probability of a fault at a two-qubit gate. Without emission, this applies
  a nonidentity Pauli pair after the gate.
  This event probability is independent of rotation angle unless angle scaling
  is supplied.
- `p2_pauli_model`: optional conditional distribution over the fifteen nonidentity
  I/X/Y/Z pairs. Omit it or use `None` for the existing uniform distribution.
  The first character acts on the emitted gate's first operand and the second
  on its second operand, without sorting qubit IDs. For example, `{"XI": 1.0}`
  applies X only to the first operand when a fault occurs. This follows runtime
  operand order, which may differ from the source program after compilation.
  Missing pairs have zero weight; weights must be finite, nonnegative and sum to
  one within 1e-10. Empty/all-zero maps, `II`, leakage entries and invalid keys
  are rejected even when `p2=0`. Weights select the fault type and do not rescale
  `p2`; values near unit sum follow the existing general-model sampler.
- `p2_angle_params`: optional `(a, b, c, d)` coefficients from `GeneralNoiseModel`.
  For signed principal angle `theta` in `(-pi, pi]` and `x = abs(theta)/pi`,
  the event probability is `p2 * (a*x**power+b)` for negative angles,
  `p2 * (c*x**power+d)` for positive angles, and `p2*(b+d)/2` at zero.
  Both half-turn endpoints use the positive branch. Omit it or use `None` for
  constant coefficients `(0, 1, 0, 1)`. This applies to the admitted `RZZ` and
  `RXYXY2Q` operations.
- `p2_angle_power`: optional finite positive exponent; the default is one.
  Combined probabilities must remain finite and in [0, 1] over the full
  principal-angle range. Validation reuses the general model's checks and
  rejects invalid/nonfinite settings even when `p2=0`.
  Coefficients scale the event probability, not the conditional fault distributions.
  Scaling uses emitted gate angles without rewriting operations or timing.
- `p1_emission_ratio`, `p2_emission_ratio`: conditional fraction of gate faults
  that select spontaneous emission instead of the ordinary Pauli branch. Defaults
  are zero. For healthy operands, emission **replaces** the ideal gate; it does
  not append a fault to that gate. Angle scaling changes the entire two-qubit
  fault probability, before this branch is selected.
- `p1_emission_model`, `p2_emission_model`: conditional emission distributions.
  Single-qubit keys are X/Y/Z/L. Two-qubit keys are ordered I/X/Y/Z/L pairs except
  II; the first character acts on the first emitted operand. L marks that operand
  leaked. Missing keys have zero weight. Omission/`None` uses the general model's
  uniform nonidentity Pauli distribution, which contains no leakage. Explicit
  empty maps, invalid keys, negative/nonfinite weights and unnormalized maps
  are rejected, even when emission is disabled. Validation/normalization uses
  the general model's existing sampler tolerance (1e-5).
- `p1_seepage_prob`, `p2_seepage_prob`: conditional probability that an already
  leaked operand returns to a random computational basis state when a gate fault
  selects emission. Defaults are zero. The ideal gate remains suppressed for
  that event. If either two-qubit operand is leaked, the model attempts seepage
  only on leaked operands; it does not sample a new emission fault on the healthy
  partner. This is the existing general-model convention, not a repump channel.
- `prep`: probability of a fault after preparation resets the qubit and clears
  prior leakage.
- `prep_leak_ratio`: conditional probability that a preparation fault causes leakage;
  otherwise it applies X. The default is zero (all preparation faults are X).
  Thus a fresh preparation leaks with probability `prep * prep_leak_ratio`.
  A later preparation clears old leakage, but can introduce a new preparation fault.
- `meas0`, `meas1`: classical readout flips 0→1 and 1→0, respectively.

Preparation faults apply to emitted reset-Z operations. Allocation by itself does
not imply a reset in the scheduled stream; runtimes that prepare on allocation
must expose that operation to model its fault. No hidden preparation is inserted.

All probabilities and ratios must be finite and in [0, 1], even when their
associated fault probability is zero. These are event probabilities, not average gate infidelities.

`prep_crosstalk` is the probability that a preparation collapses each other
prepared qubit in Z, using the existing `GeneralNoiseModel` behavior. It is
independent of `prep` and defaults to zero. The internal measurements are consumed
by the noise/quantum exchange and never become program readouts. This channel
preserves leakage flags; it does not apply the measurement-crosstalk transition
map. Only qubits prepared in the current shot are eligible.

A spectator's collapse does not make it physically busy or close its idle gap.
The next physical operation still receives the whole runtime-supplied gap,
including nonlinear idle channels. Consequently, idle noise the spectator accrued
before the other qubit's preparation is applied after the collapse, not before
it. Z/RZ idle channels commute with the collapse, but X/Y or RX/RY channels can
change the output distribution under this ordering, as described for
[measurement-crosstalk exposures](scheduled-event-adapters.md#measurement-crosstalk-exposures).
To observe the dephasing itself, prepare the spectator in a superposition and
measure its coherence after resetting another qubit;
computational-basis inputs alone cannot reveal the dephasing.

Other custom fault maps, transition hooks, repumping and
automatic post-two-qubit idle injection are not configurable. Measurement
crosstalk requires the explicit v7
[profile wrapper and typed exposures](scheduled-event-adapters.md#measurement-crosstalk-exposures). Unsupported Python keywords raise rather than being silently ignored.
Existing idle leakage remains supported: leakage-aware readout returns 2 without
readout flips, whereas ordinary leaked readout starts at 1 and is subject to `meas1`.

```python
import pecos_rslib as pr

idle = pr.scheduled_idle_noise(2, linear=0.02)
local = pr.scheduled_local_noise(
    idle,
    p1=0.01,
    p2=0.02,
    p2_pauli_model={"IX": 0.25, "XI": 0.75},
    prep=0.03,
    prep_leak_ratio=0.5,
    prep_crosstalk=0.02,
    meas0=0.04,
    meas1=0.05,
)
```

In Python, the `idle` base must be created with `scheduled_idle_noise()`;
`scheduled_idle_z()` and an existing local profile are not accepted. For a Z-only
base, use `scheduled_idle_noise()` with its default Z models and the same rates.

Use `local` as `.noise(local)` with v6, or
`scheduled_event_local_noise(local, adapter_factory)` with v7. These numbers are
invented examples. Rust exposes `ScheduledLocalNoise::new(idle, p1, p2, prep,
meas0, meas1)` and `ScheduledEventNoise::new(local, factory)`. Unlike Python, the
Rust constructor accepts a carrier that already contains local faults: it replaces
all five local probabilities, resets preparation leakage and crosstalk to zero, and
restores uniform two-qubit faults and constant angle scaling while preserving
idle configuration.
Opt into preparation crosstalk with `.with_prep_crosstalk(probability)?`,
preparation leakage with `.with_prep_leak_ratio(ratio)?`, and select an
ordered Pauli map with `.with_p2_pauli_model(distribution)?`. Each setter replaces
only its own setting. Rust also exposes `.with_p2_angle_params(a, b, c, d)?`
and `.with_p2_angle_power(power)?`. Conversion to `ScheduledNoise` preserves all
these settings for timed and event batches. The canonical Rust carrier is
`ScheduledNoise`; `ScheduledIdleNoise`
is a compatibility alias that retains the entire configuration, including local
faults. `ScheduledEventIdleZ` and `ScheduledEventIdleNoise` likewise alias
`ScheduledEventNoise`. The Python idle and local factories retain their distinct
profile classes.

## Shared scheduled admission and execution limits

Every scheduled profile admits at most 4096 gates per batch. Same-qubit
measurement, reset and reuse (including repeated measurement) execute in the
runtime's original operation order. Each measurement captures the leakage state
at its own position, so a later reset cannot change an earlier readout. This
applies to both original v6 and normalized v7 gates, including idle-only profiles.
Adapters cannot split or retime native batches.

Readout faults retain the general model's sampling order: within each dispatched
segment, gate and idle faults are sampled first, then readout faults in measurement
order. An explicit crosstalk exposure can pause a batch for internal measurements;
its transitions resolve before later source operations are sampled. Capturing
leakage consumes no randomness. The general controller also records leakage per
measurement on legacy inputs. Scheduled profiles remain bounded. Opted-in v7
profiles support explicit measurement-crosstalk exposures through the existing
noise/quantum exchange; transition hooks remain excluded. Measurement results still
return to the live runtime to resume program feedback.

Exposures do not split idle intervals: accrued idle noise is delivered at the
victim's next physical operation, after the exposure. This can change output
distributions for non-Z idle channels even with symmetric crosstalk transitions;
Z/RZ channels also need not preserve seeded trajectories. See the
[exposure timing contract](scheduled-event-adapters.md#measurement-crosstalk-exposures)
before combining crosstalk with idle noise.

Each prepared batch retains one noise-controller lifecycle. Inserted idle commands
are included in an expansion budget of sixteen output commands per prepared
command (the admitted channels emit at most eight per idle target and four per
gate, including seepage of both two-qubit operands). Output bytes and command
counts are checked before simulator execution.

Batches without exposures authorize one simulator call. With E explicit exposures
and F candidate targets, the execution budget adds at most 2F commands to the
ordinary expansion budget (one internal measurement and one transition per
target) and authorizes at most E + 1 simulator calls. Each exposure yields at
most once for internal measurements, with a final continuation for transitions
and the remaining operations. Byte budgets cover every command and each message
header. Exceeding any cumulative budget
fails and poisons the shot. These limits bound dispatched noise expansion, not
native extraction, allocations inside a model/adapter, or callback execution
time. Existing shot context, clone and whole-host reset requirements still apply.

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

The mandatory wire envelope is version 6: a 16-byte little-endian header (PECS
magic, version byte and three zero reserved bytes, u32 batch count, u32 total
length), followed by batches. Each batch has five u64 fields: runtime shot ID,
batch ordinal, start, duration, gate count. Each gate has five u64 fields: opcode,
first target, second target, theta Angle64 fraction, phi Angle64 fraction.
The angle words preserve all 64 bits. Legacy version 3 used f64 radians and is
rejected; rebuild those messages with the current encoder. Opcodes are
1 RXY, 2 RZ, 3 RZZ, 4 RXYXY, 5 reset-Z, 6 measure-Z, 7 leakage-aware measure.
Unused fields must be zero. Unknown records, invalid targets,
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
