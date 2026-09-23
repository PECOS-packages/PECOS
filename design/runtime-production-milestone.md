# Production Rust event milestone: contract decision record

Status: proposed production contracts; implementation blocked on host ownership.
No production transport is enabled. The nine tests in `runtime_frame_slice.rs`
exercise a test-local prototype, not this production contract. This record does
not extend that prototype. Python and downstream event interpretation are deferred.

Reviewed dev `b7b3fb94ef63444dd99cc8042cbcd0ce29788cb0`, #827 prototype
`2b5494d79dcd196d9b18a71a894a884ff29f320a`, #828
`e20a12bee0bc68fd006aa5949d5ff4f71ca9224b`, and RFC #591
`95eef838b6be7e6099e0226969456d581132ebda`.

## Measurement and state contract for the first slice

Recommended profile: preserve legacy GeneralNoiseModel sampling. One original
input has one start phase and one continuation/completion lifecycle, regardless
of yields. Completion is per original input, not per event or per whole shot.
Multiple completed inputs share the same shot's model, simulator and RNG state.
An event cannot split a simultaneous batch or an idle. Metadata annotations do
not introduce completion, sampling, implicit idle, or reset boundaries.

Use one independently invented unconditional Pauli-X event. It executes after
the preceding complete noisy gate/measurement expansion and before the following
expansion, without ordinary program-gate noise being applied to the injected X.
The event consumes no RNG and observes no measurement values or model state.
In particular, a preceding simulator measurement has physically executed, but its
noise-processed readout is unavailable until the original input completes.
Previously returned inputs' results remain owned by the caller; they are not a
mutable callback context. Metadata likewise cannot inspect or mutate the model.

Physical admission requires the existing checked simple-probabilities profile;
leakage/crosstalk/state-dependent physical effects reject before execution.
Metadata-only inputs must retain the broader legacy profile, including leakage
readout and nonlinear idle. This separation is an enforced compiler decision,
not a caller assertion. It is not full general-noise physical support.

Rationale: GeneralNoiseModel updates leakage/preparation and samples future gate
faults during `apply_noise_on_start`, then interprets raw measurements and samples
readout/crosstalk during `apply_noise_on_continue_processing`. Moving readout to
each yield changes state and RNG order. The existing leakage and segmentation
counterexamples rule that out. A future outcome-dependent event requires a
separately reviewed sampling profile, not retroactive completion of a prefix.

## Host ownership: separate review required

See the [call-site decision record](runtime-session-ownership.md) for exact
production references and the two migration alternatives. It supersedes the
initial blanket characterization of engine clones as live snapshots: QIS and
native runtime clones mix reconstruction with copied bookkeeping. The scheduler
needs fresh workers; other legacy callers preserve selected live state.

#827 remains paused. Do not add a host interface or expand the prototype before
that record receives separate review. The latest assessment recommends a scoped non-Clone executor inside synchronous
QuantumSystem::process, with context/recovery plumbing in the existing host.
External suspension is not required for milestone one. Keep the SimBuilder,
MonteCarlo and HybridEngine route used by Python sim(); no disconnected Rust-only
execution path. This recommendation still awaits separate review.

## Proposed session decisions after host ownership is settled

- A shot key is `(execution namespace, worker index, local shot index)` and is
  explicit at begin-shot. Frame ordinals increase across accepted inputs in that
  session. Tokens additionally carry a private owner/generation and position;
  checked overflow returns an error, never wraps. Identity does not reseed RNG.
- Preserve current worker seeding domains and streams across ordinary resets;
  do not introduce per-shot reseeding as part of transport. Replay uses the same
  worker seed report/topology; worker-count invariance is not promised.
- Only one input may be active. A yield retains that input's controller lifecycle.
  Frame completion releases its storage, publishes results once, and leaves shot
  state intact for the next input. End-shot requires no pending continuation.
- Admission errors mutate neither model, simulator, RNG nor frame ordinal; a
  corrected message can be submitted. Errors after execution starts poison the
  shot and discard pending results/tokens. Recovery requires successful reset of
  the complete host, followed by a new generation; partial reset stays poisoned.
- Configuration clones have independent sessions. Live-session cloning is absent
  from the recommended new API; compile-time coverage must enforce that boundary.

## Wire admission and retention: next, not implemented

Use the proposed mandatory v2 envelope in `runtime-effect-contracts.md`, with an
exhaustive parser that validates every nested record rather than certifying it
through the permissive legacy gate parser. Preserve original batch descriptors.
Reject unknown versions, mandatory records, schema/trigger IDs, measurement IDs
without supported mapping, configuration, targets, ordering, reserved bits,
truncation and trailing bytes before any part of the message executes. Existing
gate-only consumers must reject v2 wholesale. No new ignorable v1 record type.

Proposed finite first-slice limits: 1 MiB encoded input, 4096 records, 64 KiB
aggregate event payload, 65536 expanded simulator commands, and 65536 outcomes
per frame, one retained frame per session, no retained event history. These are
proposed defaults requiring enforcement tests, not measured production bounds.
Validate encoded limits before copies; use checked length arithmetic and fallible
reservations. Bound model expansion buffers as well as parser storage. A proven
worst-case expansion budget must be admitted before model mutation; a profile
without such a bound is unsupported. Unexpected execution-time exhaustion aborts
the shot without publishing partial results; it does not promise rollback of
already executed effects. Retained caller-owned outputs are outside session
storage. Consumed payloads are released; overflow never drops or truncates events.

## Acceptance evidence required before enabling transport

Tests must call the exported Rust host/session path, not duplicate its logic in
a test. Required coverage includes metadata versus ordinary legacy execution
(over seeds, full relevant RNG state, leakage readout, nonlinear idle and
crosstalk completion), `M; event-X; M` placement, one completion across yields,
multiple inputs in one shot, explicit identities across workers and resets,
configuration cloning, stale-token rejection, failed reset and recovery. Wire
negative tests must put invalid records after valid effectful prefixes and check
that simulator/model/RNG state stays unchanged. Exercise exact capacity limits,
one-over-limit inputs and expansion overflow. Ordinary v1 execution must remain
unchanged. These tests do not yet exist for a production event path.

## Alignment with #591 and independent fixes

The recommendation preserves #591's typed phase/outcome availability, distinct
generation origin, non-recursive generated effects, whole simultaneous batches,
strict capability compilation and exact legacy compatibility requirement. It
does not implement the RFC's general event compiler or compatibility facade.
Shot-session ownership is an additional host contract the RFC does not settle.

#828 remains separate at `e20a12bee0bc68fd006aa5949d5ff4f71ca9224b`:
normal parse-error propagation and abandoned-result reset only. Its three
regressions cover rejection before effects/RNG, reset after continuation failure,
and clone independence. Keep it available for independent review; do not merge.
Neither synthetic interface tests nor this decision record establishes simulator
or device-model equivalence.
