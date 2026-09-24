# Scoped production Rust frame milestone

Status: implemented in the Rust library, opt-in, draft #827. This supersedes the
owned-shot-host recommendation. No externally suspended continuation or new host
is introduced. The [ownership record](runtime-session-ownership.md) retains the
source-based comparison and approved choice. #828 remains independent at
`e20a12bee0bc68fd006aa5949d5ff4f71ca9224b`; it contains only parse/reset fixes.

## Production route and admission

`RuntimeNoise::new(general_builder, qubits, limits)` validates configuration and
creates an opaque opt-in configuration accepted by existing `SimBuilder::noise`.
The built wrapper owns GeneralNoiseModel and its checked capabilities. No caller
support flag, arbitrary-model downcast, mutable inner-model access, or callback
can grant admission. An ordinary GeneralNoiseModel does not implicitly opt in.

`encode_frame` produces mandatory version-2 ByteMessages; those enter the existing
SimBuilder → MonteCarlo → HybridEngine → QuantumSystem path. QuantumSystem::process
validates the entire input and borrows model/simulator state into a private,
non-Clone FrameExecutor. All yields are consumed synchronously. The existing
Engine and ControlEngine Clone bounds and v1 consumer format remain unchanged.

Initial consumer: built-in StateVecEngine, with capacity checked before effects.
Initial gates: one target per PZ, X, Z, H, MZ, MeasureLeaked, Idle or local crosstalk
payload record. No simultaneous batches, two-qubit gates, measurement IDs,
channels, arbitrary trigger payloads or processed-outcome callbacks are admitted.
This sequential format does not flatten or claim support for native runtime
simultaneous batches. The Selene producer remains unwired; no Python API is added.

## Execution and measurement contract

One original input has one shared GeneralNoiseModel start phase, with expansion
end offsets recorded without changing sampling, then its existing continuation/
completion lifecycle. Start-phase sampling finishes before simulator execution,
exactly as in ordinary execution. Gate expansions execute in source order. Raw
measurements accumulate and enter readout/crosstalk processing once at the original
input boundary. Generated crosstalk work remains inside that lifecycle.

Invented metadata ID 4101 does nothing: no RNG draw, idle split, additional
completion or reset. Invented physical ID 4102 applies unconditional X after the
previous complete noisy expansion and before the next. It observes no model
state, raw outcomes or processed readout, and receives no program-gate noise.
Physical admission requires the builder's checked simple-probabilities profile;
leakage, idle-noise, emission, crosstalk and other excluded physical configurations
reject before execution. Metadata supports the broader legacy profile for the
listed gates. Idle arithmetic that would become non-finite is rejected.

This preserves legacy measurement-state behavior, including preparation updating
leakage bookkeeping before a preceding measurement's deferred readout. It does not
promise post-noise measurement availability at an internal event. Broader physical
profiles require separate sampling/state decisions under RFC #591.

## Shot identity, cloning and recovery

For runtime-enabled systems only, Monte Carlo allocates a checked process-local run namespace, and supplies worker
and local-shot indices after successful reset. HybridEngine retains that context
in QuantumSystem across all inputs in the shot. Direct runtime-enabled HybridEngine execution
allocates its own run namespace; direct QuantumSystem callers must call begin_shot
with explicit context. Context never reseeds RNG; existing worker seed streams
are preserved. Run namespaces identify execution instances, not portable replay
IDs. Mapping this host context into native runtime-local shot IDs is deferred
with Selene producer integration; no such mapping is claimed here.

Clones retain component clone behavior and copy failure state, but clear event
execution authorization: a successful clone needs an explicit context before
processing event-enabled inputs. No frame/token escapes process, and no live
native-runtime snapshot capability is added. Mutable model/simulator access
invalidates authorization and requires reset. Direct users own uniqueness of
supplied contexts; the scheduler assigns distinct contexts for its workers/shots.

At QuantumSystem level, preflight rejection leaves model, simulator and RNG
unchanged and permits a corrected input. HybridEngine aborts the enclosing shot
on any returned error and requires whole-host reset. The HybridEngine guard covers shot-context admission as well as execution.
Before execution, QuantumSystem latches failure state; success
clears it, while errors or unwind retain it. HybridEngine additionally guards
classical start/continuation failures and failed classical resets. QuantumSystem::reset clears the quantum execution latch after noise and simulator
reset succeed. It cannot clear a failed HybridEngine host latch: classical, noise
and simulator reset must all succeed through HybridEngine::reset. Dropping a frame releases its buffers; reset discards abandoned
controller results without reseeding. No rollback of already executed effects is
promised. Ordinary non-opted-in v1 execution keeps its existing controller lifecycle;
shared parsing now returns Input for non-finite angles instead of unwinding.

## Wire subset and retention bounds

This implemented sequential v2 subset supersedes the earlier nested GateBatch
wire proposal **for this slice only**. That richer timed format is not implemented.
All fields are little-endian; headers and records have no implicit host pointers.

- 16-byte header: existing numeric PECS magic; version 2; zero flags/reserved;
  u32 count and exact total length.
- 8-byte record header: kind u8; three zero bytes; u32 payload length.
- Kind 10, 16-byte gate payload: opcode u32 (PZ=1, X=2, Z=3, H=4, MZ=5,
  MeasureLeaked=6, Idle=7, local crosstalk payload=8), target u32, f64 idle seconds
  (zero for other gates).
- Kind 30, 8-byte event payload: ID u32 and target u32. Only 4101/4102 exist.
- Unknown versions/kinds/opcodes/IDs, flags, targets, malformed lengths, truncation,
  trailing bytes and unsupported profiles reject the entire message. Gate-only
  consumers reject version 2 wholesale. No ignorable v1 event record is added.

Hard limits: 16 configured qubits, 128 records, 3088 input bytes and 65536 expanded
operations/outcomes per frame. FrameLimits can lower record/expansion limits.
Before model mutation, admission reserves a conservative budget of
`record_count * 16 * (qubits + 1)` against the configured expansion bound. For
singleton gates, idle emits at most eight operations, single-qubit faults at most
four, preparation at most three commands with at most qubits crosstalk outcomes,
and local crosstalk at most one measurement plus one continuation command. The
budget includes injected X operations and outcomes. Unsupported gate arities are
rejected rather than assigned an unproven bound.

Opted-in v1 inputs have byte and declared-count budgets checked before parsing
(the parser reserves from the declared count). Shared angle parsing rejects
non-finite values with an input error. Gates, targets and idle arithmetic are
checked once at QuantumSystem admission and must canonically
round-trip, so preceding inputs cannot grow bookkeeping outside this profile.
Both v1 and v2 share the simulator-capacity check, preventing automatic growth
from recreating persistent state. A private validated start avoids repeated
admission within the model.
Only one frame is retained; consumed segments are dropped, with no event history.
Persistent leakage/preparation sets are bounded by configured qubits. Expansion
invariant violations and unexpected continuation growth abort; nothing is silently
truncated. Existing ByteMessage/model allocations remain infallible Rust allocator
operations; this slice does not claim recoverability from process-wide OOM.
New top-level vector reservations return errors on allocation failure.

## Evidence and remaining work

`tests/runtime_frame_production.rs` has 24 tests exercising exported production APIs:

- Metadata versus ordinary legacy execution over 64 seeds, multiple inputs,
  leakage readout, nonlinear idle and complete debug-visible noise/simulator RNG
  states (including RNG caches); deterministic historical leakage and idle cases.
- H;X-event;H;M and M;X-event;M placement; p1=1 proves no duplicate gate noise.
- Exactly-once crosstalk completion, checked through the following input.
- Invalid records after a valid effectful prefix; every truncation of a sample;
  versions, IDs, flags, targets, NaN, unsupported profiles/consumers and limits.
- Clone state/authorization, mutable-access invalidation, simulator execution and
  reset failure through the shared opted-in v1 guard, classical failure after a
  physical input, and failed whole-host reset followed by recovery.
- SimBuilder multi-input shots across four workers, repeated runs, noisy seeded
  metadata replay, and explicit distinct worker contexts.

Validation: full `cargo test -p pecos-engines --offline` passed 412 tests/doctests
with zero failures/ignored tests; all-target Clippy with `-D warnings`, formatting,
changed-file pre-commit and diff checks passed. No fresh Python test run because
no Python path was added. The historical diagnostic library-search skip remains
a skip, not a pass.

The earlier test-local prototype is historical evidence only. No tests are
claimed for private interpretation, Python exposure or a native Selene producer.
Next connect a generic producer with preserved native batch semantics and host/
runtime identity mapping, then expose configuration through existing Python sim()
and run fresh integration tests. Do not add a disconnected executor stack.
Matched circuits, schedules, model parameters, decoders and approximations remain
necessary before studying parity. Synthetic tests do not establish simulator or
device-model equivalence. Keep #827 draft; do not merge either PR.

## Historical production self-review of f3bb9f75

The raw-scan containment below was superseded by the separate-review fixes.

Two confirmed admission issues were reproduced before correction:

- **P1:** Opted-in v1 messages containing an unsupported RZ with a non-finite angle
  unwound in the general parser instead of returning an admission error. A strict
  bounded wire scan now rejects unsupported records before angle conversion.
  NaN and both infinities following a valid X prefix return errors without RNG or
  simulator mutation; a corrected message still executes.
- **P1:** Simulator capacity admission covered v2 only. Opted-in v1 could grow an
  undersized StateVecEngine, recreating its state and losing previous inputs.
  Capacity is now checked before either route. The regression preserves a
  previously prepared state and checks both RNGs and unchanged capacity.

Additional production regressions cover consecutive measurements over 32 seeds,
execution/reset unwind and independently poisoned clones, and 128-record expansion
admission with 16 persistent preparation-crosstalk victims. The latter accepts the
conservative budget exactly and rejects one below it before model/RNG effects.
A suspected joint-measurement batching issue was not reproduced: the admitted
StateVecEngine uses SparseStateVecSoA's sequential measurement implementation.
No dispatch change or additional simulator capability was added.

This is implementation-author self-review, not independent external review.
The fixes do not alter the scope or the RFC #591 limitations above: preserved
native simultaneous batches and outcome-dependent physical profiles remain
unimplemented. Ordinary non-opted-in legacy parser behavior is unchanged.

## Response to separate review of d4eaaa7b

The [separate review](https://github.com/PECOS-packages/PECOS/pull/827#issuecomment-5805975740)
reported a P1 admission-guard hole and shared-parser panic. Both were reproduced
before correction. The new regressions are
`failed_host_readmission_requires_whole_host_reset` and
`shared_parser_rejects_nonfinite_angles_without_panicking`.

The guard now exists before begin_shot can fail; a failed second shot cannot
execute quantum commands or recover through quantum-only reset. Shared parsing
checks angle finiteness before conversion. The hand-written raw gate scan is
removed; bounded count admission and canonical round-trip remain necessary to
prevent excessive parser reservation and silent unknown-record skipping.

Execution now sends the entire expansion directly to the simulator when no
physical event exists, and otherwise splits only at physical events. There is no
per-gate message queue. Outcome retention for physical frames reserves the admitted
per-input expansion bound, rather than the configured maximum. Fallible reservations
remain intentional: their failures can be returned even though other allocator
sites preclude a general process-wide OOM recovery guarantee.

Shot identity remains an explicit contract; ordinary systems no longer allocate
run namespaces. Mutable accessor rustdoc states invalidation/reset requirements.
Malformed input returns Input; owner-state, allocation and internal frame failures
return Processing. Empty/event-only frames and zero-operation spans between adjacent
physical events are covered by production regressions. Underlying component errors retain their original variants, so
an error after execution still requires reset regardless of its variant.

Removed the obsolete test-local executors in runtime_frame_slice.rs and
runtime_transport_review.rs. Production regressions retain physical-event metadata
invariance/reset and unsupported-consumer/unknown-record rejection; scoped frames
have no public continuation tokens to test. Historical counterexamples and design
rationale remain in these design notes and git history. A SimBuilder/four-worker
regression confirms a post-physical-input classical error reaches the caller.

The separate review applies to d4eaaa7b. These corrections and their validation
are implementation-author work awaiting follow-up review, not a new independent
approval. #828 remains unchanged. No producer, Python API or broader profile is added.
