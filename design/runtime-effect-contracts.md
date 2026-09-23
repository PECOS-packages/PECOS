# Runtime effect contracts: GeneralNoiseModel investigation

Status: specification and executable investigation, **not enabled transport**.
This supersedes the suggestion that a generic segmenting decorator is sufficient
in [the original proposal](runtime-physical-effects.md). The actual production
changes are only normal parse-error propagation and clearing abandoned result
buffers on general-noise reset. No physical-event capability is granted.

## Evidence and decision

`tests/general_noise_segmentation.rs` in `pecos-engines` tests the real model:

- Start with a classically leaked qubit. Unsplit `MeasureLeaked; Prepare`
  returns 0; split execution returns 2. Start-time preparation changes leakage
  bookkeeping before continuation-time readout in the unsplit path. This is a
  deterministic behavior difference. The test characterizes existing batching,
  not a claim that its historical semantics are physically preferable.
- Splitting `Measure; X; Measure` changes seeded outcomes and the subsequent
  eight RNG words across a fixed 64-seed sample. Gate faults are sampled during
  `start`, but readout faults during continuation. A new completion boundary
  changes their order and sometimes the number of draws.
- Dividing one idle into two changes the sine-squared channel distribution.
  Avoiding duplicate duration is necessary but not sufficient for equivalence.
- Positive control: in the tested preparation/Pauli/linear-idle/readout profile,
  splits of a measurement-free prefix preserve exact outcomes and eight RNG
  words over 64 seeds. This does not certify arbitrary parameters, crosstalk,
  other boundaries, or full RNG-state equality.
- Reset after a failed crosstalk continuation formerly retained user results,
  preventing the next start. Reset now discards those results. Resetting a clone
  leaves the original continuation's results intact. Four concurrent cloned
  model workers, reset and seeded independently, match fresh reference models.
  These test existing model ownership, not a runtime handler or Monte Carlo
  shot-identity protocol.

The generic completion-fault counterexample remains in
`tests/runtime_transport_review.rs`. These counterexamples are regression
requirements: **a future metadata-aware adapter must compare equal on these
fixtures**, not reproduce the differences. No test-only sketch is an admissible
production adapter. None of these findings is resolved by asserting a boolean
capability, matching a type name, or running `start`/`Complete` more carefully.

Recommendation: do not admit GeneralNoiseModel to a physical bridge through a
small segmenting wrapper. Its gate sampling, measurement state and crosstalk
continuations need explicit resumable semantics. Keep ordinary execution as the
reference while implementing the minimum interface below toward RFC #591. Do
not repair its historical batching behavior incidentally in this transport PR.

## Mandatory metadata-insertion invariance

Metadata-only commands do not create model completion, lifecycle reset, sampling,
or timing boundaries. They annotate an existing frame and operation position.
For identical original frames, inserting/removing such annotations must preserve
the simulator command stream, outcome mapping, model state, and RNG consumption.
This includes annotations before/after measurements, inside a waiting
continuation, adjacent/repeated annotations, and annotations during an idle.
They cannot split the idle or alter implicit-idle accounting. If an annotation
requires completed outcomes, report only already completed outcomes; do not
force completion to satisfy the annotation.

Metadata callbacks must not get mutable physical state or the model RNG. A
downstream decoder's metadata disposition must be fixed during validation,
before execution; a physical callback cannot apply effects then retrospectively
claim metadata. Unknown mandatory events are errors, never metadata fallback.

An event with a physical effect may require a genuine execution barrier, but
cannot restart or complete the enclosing logical input. Its contract must name
the supported phase, state updates and RNG order. Do not change legacy RNG
ordering implicitly. If ordered physical effects require a different sampling
profile, expose that explicitly, retain legacy mode, and run the parity study
with the same profile on both paths. Metadata insertion must remain exactly
invariant *within either profile*, including subsequent random draws. Statistical
agreement does not waive this requirement.

## Enforced admission, not a caller assertion

Proposed build boundary: an implementation-owned, sealed compiler converts a
model configuration, complete validated frame plan and execution capabilities
into an opaque `PreparedEffectRun`. Its constructor and capability token are
private. Execution accepts that prepared run, not `Box<dyn NoiseModel>` plus a
boolean, a downcast, or an unchecked caller-supplied capability list.

Initially the compiler rejects **all** physical bridge requests, including
GeneralNoiseModel, until a concrete implementation passes the contracts below.
Adding support requires implementing the resumable controller and registering
its tested profile. Downstream code supplies event interpretation; it cannot
opt an arbitrary model into safe segmentation. Unsupported profiles, records,
effects, simulators, trace modes or consumer routes fail before starting a shot.
This is a proposed API restriction; no token/compiler API is implemented here.

## Minimum resumable extension to the existing controller

Retain `NoiseModel` and its ordinary `ControlEngine` route. An optional sealed
execution interface should add these operations, rather than another noise-rule
engine:

1. `begin_shot(context)` creates fresh state from immutable configuration.
   Context includes explicit run/shot/worker identity and deterministic seed
   domains. Worker topology is part of the seed report; cross-worker-count
   invariance is not inferred from existing worker seeding.
2. `begin_frame(validated_frame)` is called once per original logical input.
   It retains runtime-batch descriptors, stable operation indices, timing,
   pending outcomes and the model's sampling cursor.
3. `advance(simulator_reply)` returns `NeedsProcessing`, `AtEffect(token)`, or
   `FrameComplete(outcomes)`. A token denotes a validated phase and completed
   outcome prefix. Yielding is distinct from completing a frame; zero or many
   yields cannot add completion faults. Existing measurement/crosstalk work
   must be accounted for at its specified phase before exposing outcomes.
4. `resume_effect(token, validated_effects)` consumes that token exactly once.
   Model-owned application updates physical bookkeeping and emits generated
   simulator operations without recursively applying program-gate noise.
   Invalid/stale tokens, unsupported effects or illegal targets fail. Do not
   bypass classically tracked leakage by sending arbitrary simulator gates.
5. `abort(reason)` invalidates the frame, pending tokens and buffered outcomes
   without publishing partial results. The host invokes it on decoder,
   handler, allocation, simulator and continuation failures. Only successful
   reset of **both** controller and simulator allows a new shot; partial reset
   keeps the run poisoned. Cloning a live run is unavailable; workers are
   constructed from the immutable factory, never a live handler snapshot.

Finite payload/event/buffer/effect-operation limits are constructor inputs with
documented finite defaults. Check limits before copying at FFI and before queue
growth. Incremental native draining is bounded by complete frame/segment
budgets; release consumed payloads. Exceeding a limit aborts, never truncates.
Whole-message validation below must fit within that bound before execution.

Required admission tests cover metadata invariance above with seeded outcomes
and full observable RNG/command traces, all measurement/crosstalk continuations,
nonlinear and implicit idles, generated-effect provenance, one-time completion,
repeat shots, concurrent worker factories, rejection of live cloning, partial
reset and injected simulator/handler failures, retry after failure, allocation
and retention limits. Existing reset tests are only partial evidence, not
certification of this proposed interface.

## Separate wire specification: mandatory envelope version 2

This is a proposed PECOS envelope, not a Selene custom-tag reservation. No v2
encoder/parser is enabled. Ordinary gate messages retain their current v1 bytes.

- Retain the 16-byte batch framing: magic, version, flags, reserved, record
  count and total byte length. Version is 2; flags/reserved must be zero.
  Header integers and new payload integers use explicit little-endian encoding.
- Each record retains the 8-byte type/flags/reserved/payload-length header and
  four-byte alignment. Padding must be zero. All v2 records are mandatory;
  there is no optional/ignorable bit. Unknown types, flags or schema versions
  reject the **whole** envelope.
- Proposed record kinds are `GateBatch` (10) and `RuntimeEvent` (30). These are
  proposed wire discriminants only; neither is added to `MessageType` here.
  A `GateBatch` contains batch ordinal, start/duration nanoseconds (u64 each),
  a u32 nested length, then one complete ordinary gate-only v1 message. Retain
  its original simultaneous batch; never merge/split it to accommodate metadata.
  The legacy body retains its existing encoding/portability constraints.
- A `RuntimeEvent` contains batch ordinal, operation ordinal, start/duration
  nanoseconds and stable trigger ID (u64 each), then metadata schema version,
  target count and payload length (u32 each), ordered target IDs (u32 each),
  followed by owned opaque bytes. Physical events initially occupy their own
  zero-duration batch; reject mixed simultaneous physical-event/gate batches.
  Metadata annotations may reference a gate batch/operation position without
  introducing another noise boundary. No host pointers, `usize`, Rust `TypeId`,
  or private runtime schema appears in the stable event payload contract.
- Validate the complete outer length/count, checked arithmetic, every nested
  length and type, duplicates/target bounds, timing/order constraints, schema,
  decoder disposition, output bounds and consumer capability **before** sending
  any gate prefix to a simulator or mutating a noise model. Bound the whole
  envelope before allocating. A nested v1 body must be exhaustively validated;
  do not use permissive `quantum_ops()` to certify absence of unknown records.
  Incomplete, trailing, over-limit or malformed data is an error, not EOF.
- Gate-only parsers reject v2 wholesale even after an event-aware parser exists.
  A v2-aware consumer rejects unknown mandatory records anywhere in a message,
  including after valid gates. Raw-byte round trips preserve the envelope;
  unsupported trace/Python routes reject before execution and cannot fall back
  to gate-only reconstruction. No sidecar carries mandatory execution semantics.

The current unsupported-version probes check rejection before a valid X prefix
through three simulator routes and general noise. The general-noise panic is
fixed in this PR. This cannot retrofit errors into already deployed binaries
that panic or downstream consumers that swallow errors; admission must exclude
such routes. Future v2 parser tests must additionally cover supported-version
unknown-record rejection, truncation, overflow, invalid padding, nested v2,
duplicate targets, batch-order errors and capacity exhaustion. They do not
exist yet, so production transport remains disabled.

No Python event configuration is exposed. Fresh extension/integration tests are
required when that follow-up is implemented. No simulator parity or device-model
equivalence follows from these investigation probes.
