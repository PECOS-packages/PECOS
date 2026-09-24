# Execution-ordered runtime effects: focused follow-up to #826

Status: historical reviewed proposal. Current scope and validation are in the
[scoped production Rust milestone](runtime-production-milestone.md). The following
blocked/proposed statements describe earlier investigation. The
test-only review probes do not implement runtime-event support. Base inspected:
`b7b3fb94ef63444dd99cc8042cbcd0ce29788cb0`
(the merge of #826). The proposal is a bounded implementation slice of the
execution/transport direction in RFC #591, not a replacement event framework.

Latest: [the Rust frame prototype](runtime-frame-slice.md) implements a private
test-only vertical slice while keeping production transport disabled. General
physical admission remains restricted. Parse/reset correctness fixes are now
separate in PR #828, on which #827 is stacked. The original review below is
historical context; the prototype page states the current scope and evidence.

Follow-up: [GeneralNoiseModel and mandatory-envelope contracts](runtime-effect-contracts.md)
records concrete general-noise leakage, RNG and idle counterexamples and proposes
the minimum resumable interface. Production transport remains disabled. Two
ordinary general-noise fixes are now implemented/tested: parse errors return
errors instead of panicking, and reset clears abandoned result buffers. The
review below is the initial investigation; the linked follow-up supersedes its
open question about whether GeneralNoiseModel can be safely segmented.

## Transport review: confirmed compatibility blocker

The mandatory-command direction remains preferred. However, the proposed adapter
cannot safely wrap an arbitrary existing `NoiseModel`. `ControlEngine::start`
accepts a whole input, and `Complete` ends that input. There is no distinction
between completing a segment and completing the original input. A model may
legitimately perform noise at completion. Splitting around a trigger invokes that
noise more than once, even if the trigger acknowledges metadata only.

`crates/pecos-engines/tests/runtime_transport_review.rs` demonstrates this with an
independently invented model that emits one X during its result continuation:

- Unsplit `Z; Z`: one completion X, final measurement 1.
- `Z; metadata acknowledgement; Z`: two completion X operations, measurement 0.

The probe waits for every `NeedsProcessing`/`Complete` continuation. Thus simply
waiting for pending measurement processing does not repair the counterexample.
This is a contract counterexample, not a claim that every built-in model has a
per-input fault. It prevents promising generic composition or exactly-once noise
from the existing trait alone.

Concrete alternatives requiring a scope decision:

1. **Recommended: explicit opt-in composition contract.** Require a model to
   declare and test its segment/barrier semantics, non-recursive effect handling,
   fresh worker/shot state, abort/reset behavior, and RNG ordering. Default is
   unsupported. Initially admit only individually verified models; do not infer
   safety from implementing `NoiseModel` or accepting gate messages. Passing
   `GeneralNoiseModel` through this bridge requires its own measurement,
   crosstalk, leakage-state and seeded-RNG parity tests.
2. Introduce resumable segment/batch lifecycle hooks on the controller so it can
   preserve one logical input across event barriers. This is broader work toward
   #591; an adapter cannot synthesize these hooks for arbitrary opaque models.

Production implementation is paused at this decision. No new message type,
reserved Selene tag, public event executor, or Python configuration is introduced.

### Traced execution and rejection paths

| Boundary | Current behavior and required change |
|---|---|
| Selene ABI `runtime_batch_custom` | Copies borrowed payload while the callback is active; validates null/length and uses fallible reservation. No configured byte/event ceiling. Finite limits must be checked **before** copying. A non-null readable allocation remains the plugin's responsibility. |
| `drain_runtime_operations` / `convert_runtime_batch` | Drains native batches into one gate vector; custom events go to capture/metadata handling. Preserve explicit batches and event positions in a new lowering path; reject mixed simultaneous event/gate batches initially. Capture acknowledgement is not execution support. |
| `QuantumOp` / `LoweredQuantumOp` | Gate-only operation enum and trace metadata; neither is mandatory event transport. A future typed command must survive serialization and measurement-result mapping. |
| `QisEngine::quantum_ops_to_lowered_commands` | Builds gate bytes and parallel metadata; measurements have separate result IDs. Segment results must concatenate in exactly this order. |
| `ByteMessageBuilder`, `ByteMessage::new`, `as_bytes`, aligned reconstruction | Bytes are the transport. No sidecar can carry mandatory semantics. Leave ordinary v1 gate messages byte-compatible. |
| `parse_batch_header`, `process_gate_message`, `quantum_ops_into` | Header rejects unknown version; v1 skips unknown record types. An event-bearing envelope needs a distinct mandatory version, with a command parser separate from the gate-only parser. Gate-only parsing must reject the entire event-bearing input before executing any prefix. |
| `HybridEngine` → `QuantumSystem` → `EngineSystem` | The controller may issue arbitrarily many simulator sends before completion. The proposed adapter must finish the preceding segment before dispatching an effect. Simulator failure returns immediately without notifying the controller. A host-level abort/poison contract is needed; controller-local error latching alone is insufficient. |
| Pass-through / depolarizing / biased depolarizing noise | Pass-through forwards bytes; the other two parse gates and propagate parse errors. These are potential individually testable integrations, not an established compatibility list. |
| `GeneralNoiseModel` | The reviewed base used `expect` and panicked on unsupported versions; the follow-up fixes error propagation. It also owns leakage, prepared-qubit and pending-measurement state. Bypassing this model for physical effects must not bypass state semantics. |
| State-vector, sparse-stabilizer, stab-vector dispatch | Gate parsers reject unsupported version. Test probes exercise all three through `QuantumSystem`. Custom gates or ignored crosstalk placeholders are not a mandatory command mechanism. |
| QIS lowered-gate trace / raw byte dumps | Lowered-gate trace propagates gate parse errors; raw dumps preserve bytes. Extend event traces or reject event mode before execution. QIS/QASM debug-only parse attempts do not constitute execution validation. |
| Python byte-message binding / PHIR bridge | Byte-message gate conversion raises a Python error. PHIR bridge catches parse errors and falls back to Python generation: this is not a fail-closed mandatory-event path. Keep event input unavailable there until fallback explicitly excludes mandatory/unsupported input. |
| QASM, PHIR, PHIR-JSON, PHIR-Pliron producers | Existing paths construct ordinary gate messages. They need no event emission change; they must not be advertised as carrying opaque mandatory commands. |

### Lifecycle, bounds, and RFC alignment

Monte Carlo clones a template per worker, seeds each worker, then resets before
each shot. Neither `DynClone` nor `Send + Sync` promises independent state: a
model can clone a shared mutable allocation. `ControlEngine::reset` also provides
no shot identity. The adapter must use an immutable factory and explicit shot
context, not clone a live physical handler. `QuantumSystem::reset` resets noise
before the simulator; a later simulator reset error must leave the whole adapter
unusable. These are required implementation contracts, not tested guarantees.

The retention proposal must bound event bytes, event count, outstanding buffered
bytes, and effect operations separately. Check callback bounds before allocation,
stream native batches rather than drain an entire shot, and latch limit/allocation
failure until successful full reset. Never truncate or downgrade to Capture.
The current source has fallible allocation but no such configured bounds. A
callback cannot force a misbehaving native plugin to return; streaming does not
provide native-process isolation. Numeric limits and their public configuration
remain to be specified before implementation.

RFC #591's actual diff at `95eef838b6be7e6099e0226969456d581132ebda` was reviewed,
particularly **Runtime Batches and
Simultaneous Operations**, **PECOS and Selene Trigger Transport**, **General-Noise
Compatibility Facade**, and **Differential Conformance**. Event-only barriers,
stable ordered targets, schema-versioned payloads, explicit unsupported
capabilities, non-recursive generated effects, and preserved batch timing fit
that direction. An unrestricted segmenting adapter does not establish its
batch-lifecycle or exact RNG-compatibility contracts. Do not reserve a Selene
wire schema independently of that RFC. No leakage-channel change is proposed.

### What the review probes establish

Six Rust tests pass: synthetic execution placement (`H; phase; H; M` differs
from lowering-time placement), `M; flip; M`, explicit unsupported probe rejection,
v1 unknown-record loss, unsupported-version rejection by three simulator paths,
general-noise version rejection, and the segmentation counterexample (some are
assertions within one test). The synthetic decoder and execution sketch exist
only inside the test file; they do not exercise the Selene producer or define a
wire protocol. The original panic probe was converted to a normal-error
regression after the follow-up fix. The silent-skip probe characterizes v1
behavior; it must not be mistaken for a mandatory-event contract.

No FFI-to-executor event path, idle accounting, shot/worker isolation, bounded
retention, failure recovery, or Python event configuration has been implemented
or validated. No fresh Python tests were run for this review-only change. These
probes neither establish simulator parity nor device-model equivalence. The
remaining sections describe the original proposal, subject to the blocker above.

## Objective and sequence

First establish agreement between PECOS `sim()` and a Selene simulation path
using the same circuit, schedule, noise model, parameters, decoder, and explicit
approximations. Only then improve the physical model, including leakage
repumping. Synthetic interface tests establish placement and lifecycle behavior;
they do not establish device-model equivalence or cross-simulator parity.

The first executable slice should support an independently invented event that
applies a deterministic Pauli effect at an explicit command boundary. It must
work before and after measurements, without reapplying ordinary gate noise.
Outcome-dependent effects can use only outcomes that have actually completed.

## Current source constraints

- `crates/pecos-qis/src/selene_runtime.rs`: custom events are owned and handled
  during lowering. The callback cannot observe quantum execution. Strict
  rejection is latched until reset. The shared metadata closure is not a
  per-shot physical-effect handler.
- `crates/pecos-qis-ffi-types/src/operations.rs`: `QuantumOp` carries gates,
  idles, reset and measurements, but no ordered opaque event. Trace metadata
  attached to a gate does not create a standalone execution boundary.
- `crates/pecos-qis/src/ccengine.rs`: lowered operations become a `ByteMessage`
  plus per-gate trace metadata. Result IDs are mapped separately; splitting
  messages must preserve their outcome order and mapping.
- `crates/pecos-engines/src/byte_message/protocol.rs`: the byte protocol has
  gate, outcome and return-value message types, but no effect command.
  `message.rs::process_gate_message` skips unknown types. Adding a type alone
  would therefore permit consumers to silently discard effects.
- `crates/pecos-engines/src/noise.rs`: `NoiseModel` already implements
  `ControlEngine`, allowing multiple sends and result-dependent continuation.
  `QuantumSystem` runs it around the simulator. This is the recommended
  execution boundary, rather than executing effects inside the QIS runtime.
- `GeneralNoiseModel` transforms the incoming gate batch and processes its
  results. Injecting an ordinary X into its input would also apply ordinary
  X-gate noise. Existing crosstalk payloads implement specific established
  channels and are not a general opaque-event transport.
- `GateType::Custom` and crosstalk payloads can be ignored by simulator paths;
  neither provides the required fail-closed contract for an unconfigured
  physical-effect adapter. Typed channel gates are not supported by the current
  byte-message noise path either.

RFC #591 calls for stable trigger identity, typed timing, explicit generation
origin, preservation of runtime batches, and effects that do not recursively
receive ordinary gate noise. It proposes initially isolating triggers in their
own command position/batch and rejecting ambiguous mixes with simultaneous
gates. Its coordinated Selene trigger encoding and barrier semantics remain
open decisions.

## Concrete alternatives

| Choice | Scope and consequence |
|---|---|
| A: typed ordered command plus a `NoiseModel` adapter | Adds transport through QIS and byte messages, explicit mandatory-consumer semantics, and execution dispatch using existing control flow. Supports a genuine before/after measurement boundary and avoids recursive gate noise. Recommended. |
| B: lower events into existing gate/channel payloads | Smaller changes only for an already-supported, explicitly named channel. An ordinary gate would receive noise again; existing payload consumers may ignore unsupported effects. Does not supply a general execution-time handler or completed-outcome context. Not sufficient for the requested generic contract. |
| C: wait for the complete RFC #591 engine | Avoids a provisional transport decision but bundles much broader selector, channel and compatibility work. Unnecessary for a narrowly reviewed bridge. |

The decision requested is whether to implement A now as a deliberately limited
slice of #591, or review the transport contract first. Do not implement B and
present lowering-time decoding as execution-time physical handling.

## Recommended execution contract

1. The Selene edge validates and copies the event, bounds its size, and lowers it
   to an explicit ordered command. It does not execute a physical effect or
   mutate shot-specific physical state. Interpretation stays downstream.
2. Initially accept physical events only in dedicated zero-duration runtime
   batches with an unambiguous boundary. Reject batches mixing physical events
   with gates/measurements, rather than pretending callback order defines an
   order between simultaneous physical operations. Preserve batch boundaries;
   do not split ordinary simultaneous batches incidentally.
3. Define event time relative to implicit idles: target idle intervals up to
   the event boundary must execute before the effect. The adapter must declare
   targets during validation, without executing its stateful handler. Update
   timing bookkeeping so the following gate cannot insert that interval again.
   Reject effects whose timing cannot be placed consistently with their targets.
4. The execution adapter forwards the preceding ordinary segment through the
   selected existing `NoiseModel` and waits for completion, including result
   postprocessing. Only then invoke the event handler with completed outcomes.
5. For the initial vocabulary, accept only explicit Pauli effects on validated
   physical target slots, metadata-only acknowledgement, or unsupported/error.
   Submit effect-generated Pauli operations directly to the quantum engine,
   bypassing the ordinary noise model. Do not synthesize gates with elapsed time,
   idles, resets or measurement outputs under this initial contract.
6. Continue the remaining ordinary segment through the same noise-model instance.
   Preserve accumulated measurement outcomes and program result-ID routing.
   Synthetic effects after a measurement cannot change that earlier result.
7. Latch an unsupported event, malformed payload, handler error, invalid target,
   failed effect execution or invalid lifecycle transition until a successful
   reset. An absent adapter must reject mandatory effect commands in every
   consumer; metadata capture must not downgrade physical commands to success.

An event transport must survive cloning and tracing or explicitly reject the
unsupported trace path. A silent side channel attached to `ByteMessage` is not
acceptable: reconstructing a message from bytes must not lose mandatory effects.
Choose a versioned encoding and fail-closed batch capability/version signal;
old gate-only consumers currently skip unknown individual message types. Do not
choose a provisional shared Selene tag/payload schema without coordinating it
with #591. A plugin-specific decoder may remain downstream, producing the agreed
PECOS command at the adapter edge.

## State, workers and bounds

- Use an immutable adapter configuration/factory that creates a fresh handler
  for each shot. Do not reuse #826's shared `Arc<Fn>` as physical state.
- Worker clones share immutable configuration only. Each owns a separate handler,
  pending segments, completed outcomes and failure state. Reset destroys the old
  handler/queues even after failure. Handler construction receives explicit shot
  identity and an independently derived seed; workers must not infer identity
  from timestamps or local batch counters.
- Define clone behavior for an in-flight shot explicitly: reject it or construct
  an unstarted worker, never copy partially consumed queues as a new shot.
- Enforce limits before FFI payload allocation, before accumulating event batches,
  and before queuing decoded effects. Limit payload bytes, event count and total
  outstanding bytes; exceeding a limit is a terminal error, never truncation.
- Stream/drain commands incrementally. A per-event limit alone does not bound a
  single lowering call that emits arbitrarily many events. The adapter must not
  require retaining an additional full-shot capture history.

## Acceptance tests for the implementation

Use independent synthetic tags and payloads, with no external runtime dependency.

- Ordered Pauli placement: compare H/event/H/measurement with
  event/H/H/measurement; select a Pauli whose placement gives distinct known
  outcomes. Calling the effect during lowering must fail this test.
- Measurement boundary: measure, effect, measure; preserve the first result and
  change only the second where specified. An outcome-dependent handler sees only
  completed results and cannot observe future measurement outcomes.
- Idle placement: execute an idle interval before the event and a later idle
  after it. A counting noise model sees each interval once; a physical oracle
  distinguishes placing the effect before versus after the first idle.
- No duplicate gate noise: a counting/noisy model sees program gates once and
  never sees generated Pauli effects as new program gates.
- Strict handling: no adapter, unknown tag/schema, unsupported effect, invalid
  target, ambiguous simultaneous batch, or unsupported transport consumer fails
  explicitly and remains failed on retry/finalization.
- Lifecycle: repeat shots with handler state, clone workers and run concurrently,
  then reset after a handler failure. No state, outcomes or queued effects leak.
- Bounds: oversized payload, many small events, excessive decoded output and
  handler backlog fail before crossing configured limits; no silent eviction.
- Ordinary execution: without events, preserve original message/batch structure,
  exact seeded outcomes and the selected noise model's RNG consumption.
- Serialization/trace: mandatory commands round-trip or fail explicitly;
  gate-only parsing must never silently discard them.

Run relevant Rust tests, doctests, Clippy, formatting and repository hooks. If a
Python surface is added, rebuild the extension and run fresh integration tests;
record diagnostic skips separately from passes.

## Python and parity follow-up

Defer Python configuration until the Rust execution and transport contract is
reviewed and tested. Specify a `sim(...).classical(selene_engine(...))` adapter
configuration that couples the runtime decoder to the execution adapter and
selected noise model. Validate required capabilities during build, before shots
start. Define per-shot factory/seed behavior, worker ownership, error conversion,
retention limits and trace support. Do not add a lowering-time Python callback
and label it an execution handler. Python integration must test measurement
boundaries, repeated shots and multiple workers with a freshly built extension.

The next parity experiment must pin the same circuit, lowered schedule, decoder,
noise parameters and approximations in both simulation paths; start with the
synthetic effect and a small independently checkable circuit. Record exact seeded
comparisons only where both paths share RNG semantics, otherwise use specified
multi-seed distributional tests. Leakage repumping and model/calibration changes
remain a later, separate step.
