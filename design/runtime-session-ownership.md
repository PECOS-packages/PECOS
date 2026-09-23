# Decision pending: session ownership on the existing simulation path

Status: **awaiting separate review**. #827 is paused; production transport stays
disabled. No prototype expansion or architectural implementation is authorized
before that review. #828 remains independent at
`e20a12bee0bc68fd006aa5949d5ff4f71ca9224b`.

Source links below pin production dev `b7b3fb94ef63444dd99cc8042cbcd0ce29788cb0`.
This refines the [milestone record](runtime-production-milestone.md): DynClone
requires an infallible clone operation, **not a uniform snapshot guarantee**.
Caller intent below is inferred from code unless an explicit comment is noted.

## Actual clone obligations and callers

| Call site | State preserved or recreated; migration significance |
|---|---|
| [Engine:5](https://github.com/PECOS-packages/PECOS/blob/b7b3fb94ef63444dd99cc8042cbcd0ce29788cb0/crates/pecos-engines/src/engine.rs#L5); [ControlEngine:25](https://github.com/PECOS-packages/PECOS/blob/b7b3fb94ef63444dd99cc8042cbcd0ce29788cb0/crates/pecos-engines/src/engine_system.rs#L25) | Both require DynClone. Neither documents live-fork identity, fresh-worker semantics or fallible rejection. A non-Clone session cannot directly implement these traits. |
| [QuantumSystem::clone:177](https://github.com/PECOS-packages/PECOS/blob/b7b3fb94ef63444dd99cc8042cbcd0ce29788cb0/crates/pecos-engines/src/quantum_system.rs#L177); [HybridEngine::clone:220](https://github.com/PECOS-packages/PECOS/blob/b7b3fb94ef63444dd99cc8042cbcd0ce29788cb0/crates/pecos-engines/src/hybrid/engine.rs#L220) | Clone boxed components as they stand. GeneralNoiseModel derives Clone (general.rs:108), copying RNG, leakage/preparation/measurement bookkeeping and buffered results. This preserves pending noise state, not only parameters; component-specific clone semantics still apply. |
| [Monte Carlo worker creation:393](https://github.com/PECOS-packages/PECOS/blob/b7b3fb94ef63444dd99cc8042cbcd0ce29788cb0/crates/pecos-engines/src/monte_carlo/engine.rs#L393); [shot loop:422](https://github.com/PECOS-packages/PECOS/blob/b7b3fb94ef63444dd99cc8042cbcd0ce29788cb0/crates/pecos-engines/src/monte_carlo/engine.rs#L422) | Clones template, reseeds worker, resets before every shot. This caller needs fresh workers with matching configuration and seed behavior, not an active continuation snapshot. Arbitrary supplied engines still need a reconstruction contract; parameters cannot be recovered from any trait object. |
| [builder replacement:133](https://github.com/PECOS-packages/PECOS/blob/b7b3fb94ef63444dd99cc8042cbcd0ce29788cb0/crates/pecos-engines/src/monte_carlo/builder.rs#L133) | with_classical_engine explicitly says preserve the quantum system and clones it. Replacing this with a fresh configuration silently discards preserved state. Keep this legacy behavior or explicitly restrict the new configuration route. |
| [builder helper:107](https://github.com/PECOS-packages/PECOS/blob/b7b3fb94ef63444dd99cc8042cbcd0ce29788cb0/crates/pecos-engines/src/monte_carlo/builder.rs#L107); [noise replacement:197](https://github.com/PECOS-packages/PECOS/blob/b7b3fb94ef63444dd99cc8042cbcd0ce29788cb0/crates/pecos-engines/src/monte_carlo/builder.rs#L197) | Copies classical engine; noise replacement deliberately constructs a fresh StateVec engine. These are component-replacement policies, not evidence of a general live-snapshot requirement. |
| [MonteCarloEngine::clone:699](https://github.com/PECOS-packages/PECOS/blob/b7b3fb94ef63444dd99cc8042cbcd0ce29788cb0/crates/pecos-engines/src/monte_carlo/engine.rs#L699) | Copies template, scheduler RNG and seed/configuration. Replacing worker construction must retain subsequent seed-report behavior; do not reset scheduler RNG when cloning configuration. |
| [QisEngine::clone:956](https://github.com/PECOS-packages/PECOS/blob/b7b3fb94ef63444dd99cc8042cbcd0ce29788cb0/crates/pecos-qis/src/ccengine.rs#L956) | Rebuilds program interface, clones runtime and RNG, but clears measurement results, started flag, dynamic thread and pending operations (980–1009). It is not a resumable live-program snapshot. Interface recreation can fail and leave None. |
| [SeleneRuntime::clone:1908](https://github.com/PECOS-packages/PECOS/blob/b7b3fb94ef63444dd99cc8042cbcd0ce29788cb0/crates/pecos-qis/src/selene_runtime.rs#L1908) | Copies buffered operations, maps, pending measurements and shot bookkeeping; native library/instance are recreated lazily. Metadata handler Arc is shared. This is not a native runtime snapshot and must not implicitly become physical-handler session sharing. |
| [QisEngineBuilder::clone:23](https://github.com/PECOS-packages/PECOS/blob/b7b3fb94ef63444dd99cc8042cbcd0ce29788cb0/crates/pecos-qis/src/engine_builder.rs#L23); [builder output bound:16](https://github.com/PECOS-packages/PECOS/blob/b7b3fb94ef63444dd99cc8042cbcd0ce29788cb0/crates/pecos-engines/src/engine_builder.rs#L16) | Builder clones runtime/interface configuration; output is explicitly Clone. Existing builders can supply future factory inputs but do not currently retain a reusable, fallible construction recipe for every component. |

## Identity and ownership entry points

[SeedReport creation:323](https://github.com/PECOS-packages/PECOS/blob/b7b3fb94ef63444dd99cc8042cbcd0ce29788cb0/crates/pecos-engines/src/monte_carlo/engine.rs#L323) assigns worker indices and
worker seeds. [The worker shot loop:422](https://github.com/PECOS-packages/PECOS/blob/b7b3fb94ef63444dd99cc8042cbcd0ce29788cb0/crates/pecos-engines/src/monte_carlo/engine.rs#L422) assigns local shot indices;
these are recorded with results at line 454. Add a run namespace at this scheduler
boundary; pass context after successful whole-host reset and before run_shot.
Keep the same context across [HybridEngine’s repeated quantum inputs:150](https://github.com/PECOS-packages/PECOS/blob/b7b3fb94ef63444dd99cc8042cbcd0ce29788cb0/crates/pecos-engines/src/hybrid/engine.rs#L150).
A frame ordinal belongs inside that shot, not a new shot per process call.

There is already a separate runtime-local identity: [QisEngine start:1824](https://github.com/PECOS-packages/PECOS/blob/b7b3fb94ef63444dd99cc8042cbcd0ce29788cb0/crates/pecos-qis/src/ccengine.rs#L1824)
draws a shot seed, increments its trace shot index, resets the runtime and calls
shot_start with that index (1839–1842). It is not the scheduler worker/shot tuple.
Unify or explicitly map these identities at shot start; do not infer identity from
seed, thread ID, pointer or clone-copied counters. Preserve existing RNG draws.

## Why existing owners need contract changes

- **Builder:** [SimBuilder::build:263](https://github.com/PECOS-packages/PECOS/blob/b7b3fb94ef63444dd99cc8042cbcd0ce29788cb0/crates/pecos-engines/src/sim_builder.rs#L263) consumes component builders,
  constructs one HybridEngine (330), then stores it as the Monte Carlo template
  (337). Retaining a repeatable factory instead is a construction/storage change,
  not merely placing another field in the consumed builder.
- **Runtime:** QisRuntime owns program lowering and measurement feedback, not the
  quantum simulator or GeneralNoiseModel. Its clone/reset and shot_start do not
  atomically reset those components. A lowering callback cannot own execution
  ordering or implement controller yields.
- **Controller/host:** [process_as_system:129](https://github.com/PECOS-packages/PECOS/blob/b7b3fb94ef63444dd99cc8042cbcd0ce29788cb0/crates/pecos-engines/src/engine_system.rs#L129) has only
  start/continue/complete, with no shot context or abort transaction. Reset of
  noise then simulator can fail partway. Adding owned continuation state requires
  an explicit lifecycle and failure policy; hiding it behind DynClone does not
  settle ownership. It can live within the existing host after that change;
  this is not a reason to invent a separate simulation stack.

## Smallest viable alternatives

| Choice | Minimum change and migration impact |
|---|---|
| **1. Configuration creates owned sessions** | Add repeatable, fallible worker construction to the existing SimBuilder/MonteCarlo path. Separate the cloneable template from the live worker used by the existing HybridEngine shot loop. Give that loop explicit shot context and abort/reset handling; own the non-Clone session there. A session cannot implement today’s DynClone-bound host traits unchanged: use an internal execution core without that bound, retaining legacy Clone wrappers/routes for existing callers. Preserve legacy builders accepting live engines; event mode requires reconstructible configuration. Do not change their clones to silently reset. |
| **2. Explicit snapshot/fork** | Retain cloneable live execution but specify copied simulator/model/RNG/buffer state, fresh branch ownership, token/reply rebinding, fork lineage and reset behavior. Add a fallible explicit fork for components that cannot snapshot. Keep worker construction distinct. QIS dynamic threads and native runtime instances currently lack snapshot support; either add it or reject forks there. An infallible Clone cannot report that rejection, and a poisoned or fresh clone is not an honest snapshot. This requires more than copying the prototype runner. |

Option 1 matches the actual scheduler use and avoids making native runtime
snapshot support a prerequisite for parity. It still requires reviewed changes to
host trait/storage boundaries; it is not an already-compatible drop-in wrapper.
Option 2 is useful only if live branching is a product requirement, which the
parity study does not currently require.

## Scoped synchronous executor evaluation — recommended for milestone one

**External suspension is not required.** Engine::process (engine.rs:13) takes
an exclusive mutable borrow and returns completed output. HybridEngine::run_shot
(hybrid/engine.rs:150–156) waits for that output before providing measurements to
the classical controller. Monte Carlo creates its worker clones before entering
the shot loop (monte_carlo/engine.rs:393, 422). There is no inspected caller needing
an event token to survive process(). The first event is an unconditional synthetic
X, not a request for asynchronous user input or processed measurement feedback.

A private non-Clone executor can borrow QuantumSystem's disjoint noise-model and
simulator fields for one call. It owns bounded frame buffers and consumes effect
yields internally; it returns only completed results or an error. Persistent
noise/simulator state, shot context, frame ordinal and poison status stay in the
existing QuantumSystem across inputs. No frame or borrowing token is stored in
that owner. Rust's exclusive borrow prevents ordinary cloning during execution;
this is not a claim about arbitrary shared state hidden inside downstream engines.

| Contract | Scoped executor | New owned-shot host |
|---|---|---|
| Placement and traits | Extend QuantumSystem::process with an admitted event branch; keep Engine/ControlEngine Clone bounds and the v1 route. Internal yields are not new public EngineStage values. | Requires configuration/worker construction and host-storage changes to accommodate non-Clone live sessions. |
| Identity | Pass scheduler run/worker/local-shot context through the existing HybridEngine once after successful reset; retain it across process calls. Map QIS runtime-local IDs explicitly. Direct Rust users of event mode must establish context or receive an admission error. | Same identity source and QIS mapping required; owning a new session does not supply them automatically. |
| Recovery | Preflight rejection leaves state/RNG/frame ordinal unchanged. Set a persistent in-progress/poison guard before mutation; clear only on successful completion. Drop or unwind releases buffers but leaves the owner poisoned. Only full classical/noise/simulator reset permits reuse. | Session abort encapsulates this, but still needs whole-host reset coordination. Neither option rolls back effects already executed. |
| Cloning | No active frame escapes. Preserve existing model/simulator clone behavior and copy poison status; never turn a failed state into a healthy clone. For opt-in events, clear execution authorization on clone and require a new explicit context before processing, without resetting copied physics/RNG. Legacy v1 clones remain unchanged. No live fork capability is claimed. | Factory clones create fresh sessions. Existing callers preserving live components need a separate legacy route or migration. |
| Python | Existing SimBuilder → MonteCarlo → HybridEngine → QuantumSystem path stays intact. Later expose opt-in configuration through the existing QIS wrapper. | Must migrate that same stack before Python can use it; a standalone host is insufficient. |

The scoped approach is smaller, but not zero API work. QuantumSystem currently
holds erased NoiseModel/QuantumEngine objects and exposes mutable component
accessors (quantum_system.rs:164, 172). Event admission must be implementation-owned,
compiled from supported configuration, with any mutable component access invalidating
that admission before further event execution. A downcast, cached type name or
caller capability flag is insufficient. Add a narrow default-reject capability
entry on the existing noise boundary, or an internal checked wrapper; do not copy
GeneralNoiseModel sampling into a competing engine. These are implementation
choices for review, not new interfaces added by this document.

Preserve one original input's start/continuation/completion lifecycle across all
internal yields. Sample start-phase noise in the legacy order, accumulate raw
measurements in source order, and perform readout/crosstalk continuation once at
the original boundary. Event X observes no pending results or model state. It
executes between intact noisy expansions without program-gate re-noising. Metadata
cannot split idles/batches, advance RNG, or add completion. Require identical
seeded outcomes, leakage readout, nonlinear idle behavior and relevant RNG state;
no statistical relaxation. Physical configurations outside the checked profile
reject. Wire validation and bounded expansion/retention remain prerequisites.

This supersedes the earlier preference for a new owned-shot host as the first
milestone. Retain that option only if a concrete future caller needs external
suspension; live runtime snapshots are unnecessary here. The scoped recommendation
is a source-based design assessment, not a production test result. Keep #827 draft
and implementation paused until separate review resolves this choice.

## Concrete route back to Python sim(), after separate review

Keep one execution implementation:

1. Keep existing worker template construction; add reviewed shot context and
   reset/error plumbing to the MonteCarlo/HybridEngine loop and scoped execution
   inside QuantumSystem::process. Keep GeneralNoiseModel
   sampling and the existing simulator behind that loop. Introduce no second
   Rust-only runner API as the milestone’s endpoint.
2. Carry mandatory events from QisEngine/Selene lowering to that same quantum
   processing path after whole-message admission. Preserve raw-measurement versus
   processed-readout semantics from the milestone record, frame completion once,
   original batches and the worker seed report. Bound all retained frame state.
3. Exercise Rust integration via the same builder, scheduler and QIS host used by
   Python, including multiple feedback inputs and workers. Only then expose the
   opt-in configuration through [PySimBuilder QIS branch:1300](https://github.com/PECOS-packages/PECOS/blob/b7b3fb94ef63444dd99cc8042cbcd0ce29788cb0/python/pecos-rslib/src/sim.rs#L1300),
   which already uses Rust sim_builder; its GeneralNoiseModel extraction is at
   line 1393 and build at 1412. [QisControlSimulation:404](https://github.com/PECOS-packages/PECOS/blob/b7b3fb94ef63444dd99cc8042cbcd0ce29788cb0/python/pecos-rslib/src/engine_builders.rs#L404)
   already stores MonteCarloEngine and delegates run/run_with_workers to it.
   Retain that wrapper and result format, then run fresh Python sim() integration
   tests with matched circuit, schedule, noise parameters and decoder.

RFC #591 at `95eef838b6be7e6099e0226969456d581132ebda` requires typed phase/outcome
availability, preserved batches, non-recursive effects and exact compatibility
sampling. Neither ownership choice waives those requirements; the RFC does not
supply a native-runtime snapshot contract. This is a source-based recommendation
awaiting separate review, not implemented behavior or a parity claim.
