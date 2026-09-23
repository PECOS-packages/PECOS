# Execution-ordered runtime effects: focused follow-up to #826

Status: proposal for an implementation decision. No execution-effect support is
implemented by this document. Base inspected: `b7b3fb94ef63444dd99cc8042cbcd0ce29788cb0`
(the merge of #826). The proposal is a bounded implementation slice of the
execution/transport direction in RFC #591, not a replacement event framework.

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
