# `pecos.sampling.stab` -- Batch Orchestration for Static Stabilizer Sampling

Status: draft / proposal
Pairs with: `design/qec_sim_literature.md`, `design/dem_stab_sim_skeleton.md`
Date: 2026-04-11 (revised same day)

## Problem

PECOS already has the two batch primitives we need:

- `DemStabSim` (wraps `DemSampler`) -- samples detector + observable flips.
- `MemStabSim` (wraps `MeasurementNoiseModel`) -- samples raw measurement outcomes.

Both expose `sample_batch(N, rng)` that computes the error mechanism table **once** and then draws N shots in a tight loop. `DemSampler` additionally has rayon-parallel fast paths (`sample_statistics`, `sample_statistics_parallel`) that scale across workers.

The current main simulation entry, `sim()` in `pecos-engines`, is built on `MonteCarloEngine`: a per-shot classical-engine loop. Wrapping `DemStabSim` / `MemStabSim` behind the `QuantumEngine` streaming trait (the Path A record-and-replay idea in `dem_stab_sim_skeleton.md`) **throws away the batch win**: we would compute the mechanism table once and then re-enter the classical shot loop N times instead of calling `sample_batch(N)` once.

That is not the right shape for the workload this is built for:

- Circuits are **static** (no classical feed-forward / conditionals / loops).
- Noise is **Pauli-family** (depolarizing, per-location, later Pauli-Lindblad, later channel-lowered).
- The user wants **many shots, fast**, for threshold estimation / memory experiments / decoder benchmarking.

## API architecture: top-level `sim()` plus `sampling.*` catalog

Four options were considered:

- **X1 -- single magic top-level.** `sim(anything)` dispatches internally. Rejected: magic hurts predictability.
- **X2 -- flat method-named top-level.** `monte_carlo()`, `dem_sampling()`, `subset_sampling()`, `importance_sampling()`. Rejected: puts "which method fits my problem" on the user; leaks implementation vocabulary.
- **X3 -- flat intent-named top-level.** `sim()`, `sample()`, `rare_events()`. Rejected: vague naming, multiple top entries still.
- **X4 -- one beginner hook + grouped submodule.** Chosen.

**X4 in one line.** Keep `sim()` and `sim_neo()` as top-level shortcuts for the two most-used strategies; expose the full sampling catalog under a grouping module `pecos.sampling.*` that is IDE-tab-discoverable.

### Why X4 wins

- `sim()` is PECOS's brand entry. Breaking it buys nothing.
- Power users get honest explicit access under a grouping noun that tells the truth: these are **sampling strategies**, not "the sim function and its rivals".
- Submodule invites future entries (`sampling.matchgate`, `sampling.decision_diagram`, ...) without cluttering top-level namespace each time.
- Uses user-language word ("sampling"), not implementation-language word ("orchestrator").

### Two ways to call the same thing (aliases, not duplicates)

`sim()` and `sim_neo()` stay top-level as shortcuts. Inside the catalog, the same strategies are available as re-exports -- one implementation, two paths to it:

| User types | Resolves to |
|---|---|
| `pecos.sim(prog)` | monte-carlo over classical-engine loop |
| `pecos.sampling.monte_carlo(prog)` | **same** code as `sim(prog)` (re-export) |
| `pecos.sim_neo(input)` | pecos-neo tool-framework shot loop |
| `pecos.sampling.neo(input)` | **same** code as `sim_neo(input)` (re-export) |
| `pecos.sampling.stab(dag)` | **new** -- batch DEM/MNM one-shot sample |
| `pecos.sampling.subset(...)` | future |
| `pecos.sampling.importance(...)` | future |

Not rival entry points. Same implementations, two surfaces: friendly shortcut + explicit catalog.

### Graduation rule

A catalog entry graduates to a top-level shortcut only when it's load-bearing enough that users hit it constantly. Current bar is set by `sim` (monte carlo) and `sim_neo` (adaptive / composable noise). If `sampling.stab` becomes as common in a year, promote to `sample_stab()` top-level then. Until it earns promotion, the catalog entry is enough.

### Why not promote "orchestrator" to the top-level concept

Grug considered making orchestration the unifying abstraction: `pecos::orchestrator::monte_carlo()`, `pecos::orchestrator::neo()`, `pecos::orchestrator::batch()`. Rejected:

1. The three orchestrators are **genuinely different shapes**, not three instances of one pattern. MonteCarloEngine (per-shot, classical-driven, streaming), pecos-neo tool framework (per-shot, ECS, rayon, adaptive), batch sampler (no shot loop at all). Unifying under one trait becomes a tagged union with mostly-optional methods -- the abstraction doesn't save code, it just moves the switch statement.
2. **Zero duplication evidence** between pecos-neo and the proposed batch path. Let concrete duplication name the abstraction, not prediction.
3. **"Orchestrator" is implementation-language.** Users ask "how do I get shots" and "what sampling strategy fits my problem", not "which orchestrator runs it". Top-level entries stay named by behavior.

If a real orchestrator abstraction earns its place later (concrete duplication shows up, a user asks to swap orchestrators), that's the moment to extract it -- not now.

## `sampling` vs `orchestration` as a builder verb

When a sampling strategy needs to be selected *inside* an existing entry (e.g. `sim_neo(prog).<method>(strategy)`), the verb is `.sampling(...)`, not `.orchestration(...)`:

```rust
sim_neo(prog)
    .sampling(sampling::monte_carlo())        // default
    .sampling(sampling::importance(config))   // alt
    .sampling(sampling::subset(config))       // alt
    .run()
```

Reasoning:

- Inside `sim_neo`, the **orchestrator is singular** (pecos-neo's tool framework). What varies per call is the **sampling strategy** that plugs into it. The verb names the axis that changes.
- `sampling` is user-language (statistical choice); `orchestration` is implementation-language (execution mechanism). Users pick statistical strategy.
- Matches the catalog noun `pecos.sampling.*`. Same word for the same concept at both call sites.
- Reserves `.orchestrator(...)` for the day (if ever) when swapping orchestrators is a user-visible axis.

## Decision

Build a separate orchestration entry, sibling to `sim()`, living inside the `sampling` catalog:

```rust
pub fn stab(dag: DagCircuit) -> sampling::stab::Builder
```

Called as `pecos::sampling::stab(dag).noise(...).shots(...).run()`. `sim()` / `sim_neo()` are untouched; they become catalog entries as re-exports (`sampling::monte_carlo`, `sampling::neo`).

No retrofit of `MonteCarloEngine`. No record-and-replay through `QuantumEngine`. A clean, separate path that preserves batch semantics end-to-end.

## Builder chain

```rust
use pecos_qec::sampling;

let result = sampling::stab(dag)
    .noise(NoiseConfig::uniform(1e-3))
    .detectors(detectors)           // optional -> DEM path
    .observables(observables)       // optional -> DEM path
    .include_raw_measurements(true) // opt-in; default false
    .shots(100_000)
    .workers(8)                     // rayon; default 1
    .seed(42)
    .run()?;
```

Methods:

- `.noise(NoiseConfig)` -- uniform depolarizing rates. Future: accept any type implementing a `DemStabNoiseModel` trait (see `dem_stab_sim_skeleton.md`).
- `.detectors(Vec<DetectorDef>)` / `.observables(Vec<LogicalObservable>)` -- if either is set, take the DEM path; otherwise take the MEM path.
- `.include_raw_measurements(bool)` -- always available in MEM path; additionally toggleable in DEM path (carries extra cost).
- `.shots(n)` -- required.
- `.workers(n)` -- rayon worker count. `workers(0)` or omitted -> single-threaded. Helper `.auto_workers()` -> `available_parallelism`.
- `.seed(u64)` -- master seed; split deterministically across workers.
- `.run()` -- consumes the builder, runs, returns `SampleResult`.

## Dispatch rule

```
if detectors.is_empty() && observables.is_empty():
    use MemStabSim -> raw measurement outcomes only.
else:
    use DemStabSim -> detector + observable flips (+ optionally raw measurements).
```

The user picks by what they register, not by naming a backend. This reads naturally and removes a spurious choice.

## Result type

```rust
// pecos_qec::sampling::stab::SampleResult
pub struct SampleResult {
    /// Per-shot detector flip vectors. Present when detectors were registered.
    pub detector_flips: Option<Vec<Vec<bool>>>,
    /// Per-shot observable flip vectors. Present when observables were registered.
    pub observable_flips: Option<Vec<Vec<bool>>>,
    /// Per-shot raw measurement outcomes. Always present in MEM path;
    /// present in DEM path iff `.include_raw_measurements(true)`.
    pub raw_measurements: Option<Vec<Vec<bool>>>,
    /// Metadata for reproducibility / debugging.
    pub num_shots: usize,
    pub num_mechanisms: usize,
    pub seed: u64,
}
```

Follow-up helpers (optional, add as the need shows up):
- `.logical_error_rate(observable_id: usize) -> f64`
- `.detector_rates() -> Vec<f64>`
- `.to_shot_vec(...) -> ShotVec` for compat with consumers that expect the engines' shot format.

## Static-only guarantee

For v1 the input is `DagCircuit`. A `DagCircuit` is a pure gate graph -- it has no conditional-gate opcode, no classical predicate, no loop construct. So the type itself is the static guarantee; no extra traversal or rejection check is needed in v1.

When v2 adds program-IR lowering (QASM / QIS / HUGR / PHIR -> DagCircuit), the lowering layer does the static check: reject on any classical predicate, classically-controlled gate, or loop that depends on measurement outcomes. Unconditional loops are fine and get unrolled.

This keeps v1 honest (no pretend-check on a type that can't contain feedback) and v2 honest (check where it actually matters).

## Parallelism

Shots are embarrassingly parallel. Two options:

1. **Leverage the existing `DemSampler::sample_statistics_parallel`.** Already there, already tested. For DEM path this is the single-call fast path. Downside: returns aggregated statistics rather than per-shot bit vectors.
2. **Roll our own rayon split.** Chunk N shots by worker count; each worker gets a seeded RNG split (e.g. `seed ^ worker_idx` or `SplitMix64`). Each worker loops `sample_into_packed` locally. Merge.

Grug recommend: DEM path -> default to `sample_statistics_parallel` when the user asks only for *aggregate* outputs (rates, logical-error counts). When the user asks for per-shot bit vectors, fall back to rolled-rayon. MEM path -> rolled-rayon (MNM does not have a native parallel path yet; if it becomes a bottleneck, add one).

Seeding: master seed -> `PecosRng::seed_from_u64(seed)`; split to `workers` child seeds via a deterministic mixer (`SplitMix64`, `seed_from_u64(worker_id)`, whichever matches PECOS convention). Reproducibility means same `(seed, workers, shots)` returns the same bytes.

## Where it lives

Module layout follows PECOS convention (`foo.rs` + sibling `foo/` directory, no `mod.rs`):

```
crates/pecos-qec/src/sampling.rs           -- parent module, catalog root
crates/pecos-qec/src/sampling/stab.rs      -- stab strategy (this work)
crates/pecos-qec/src/sampling/neo.rs       -- re-export of sim_neo()   (future)
crates/pecos-qec/src/sampling/monte_carlo.rs -- re-export of sim()     (future)
```

Public names:

- `pecos_qec::sampling::stab::stab(dag) -> Builder` -- entry free function.
  Actually cleaner: `pecos_qec::sampling::stab(dag) -> stab::Builder` (the module name doubles as the function when there's one primary constructor). Implemented as:
  ```rust
  // in sampling.rs
  pub mod stab;
  pub use stab::sample as stab;   // or inline: pub fn stab(dag) -> ...
  ```
  Bikeshed -- decide at implementation time.
- `pecos_qec::sampling::stab::Builder`
- `pecos_qec::sampling::stab::SampleResult`
- `pecos_qec::sampling::stab::BuilderError`

Metacrate re-export: `pecos::sampling::stab` via existing `pecos-qec` re-export path.

Python bindings in `pecos-rslib` follow later (v1b / v2), once the Rust API is stable.

## Relationship to `DemStabSim` / `MemStabSim`

`sampling::stab()` is the **user-facing** orchestration. It is implemented on top of `DemStabSim` / `MemStabSim` without giving up direct access to them.

Power users keep the lower-level APIs:

```rust
use pecos_qec::DemStabSim;

let sim = DemStabSim::builder().circuit(dag).noise(n).detectors(d).build()?;
let mut rng = SmallRng::seed_from_u64(seed);
let batch = sim.sample_batch(n_shots, &mut rng);
// ... introspect sim.sampler(), export DEM, etc.
```

`sampling::stab` is for "give me shots, now"; the typed sims are for "I need the sampler object for something else".

## What this deliberately is not

- Not an extension of `MonteCarloEngine`. Per-shot streaming is not the right fit and forcing it throws away the batch primitive.
- Not a replacement for `sim()`. Classical control + adaptive programs still go through `sim()` with `sparse_stab` + `pecos-neo` noise. Two entries, two computational models.
- Not a Stim rewrite. Naming is honest: "sample a stabilizer circuit statically with depolarizing-family noise". Stim's workflow is inspiration only; the entry point is PECOS-shaped.
- Not an abstraction-first design. If a higher layer above `sampling::stab()` and `sim()` turns out to be useful later, it earns that place once concrete duplication shows up. Not before.
- Not a promotion of "orchestrator" to a top-level concept. See the X4 reasoning above.

## v1 deliverables

1. `crates/pecos-qec/src/sampling.rs` + `crates/pecos-qec/src/sampling/stab.rs`:
   - Free function `stab(dag) -> Builder`.
   - `Builder` + chain.
   - `SampleResult`.
   - `BuilderError` (e.g. missing shots, missing noise, builder misuse).
   - Dispatch logic: DEM vs MEM on detector/observable presence.
   - Rayon parallel shot split; deterministic per-worker seed.
2. Re-exports in `pecos-qec/src/lib.rs`.
3. Integration test: distance-3 repetition code, 10k shots, both DEM and MEM paths, assert logical-error rate sits within binomial CI around the analytic expectation.
4. Doctest on the module header example.
5. Clippy clean.

## v2+ ideas (not blocking v1)

- **Catalog completeness**: add `sampling::monte_carlo` (alias to `sim()`) and `sampling::neo` (alias to `sim_neo()`) so the `pecos.sampling.*` namespace is a full catalog from day one.
- Accept program IRs (QASM / QIS / HUGR / PHIR) with static-check + lowering to `DagCircuit`.
- Accept `TickCircuit` directly.
- Streamed output API (`stream_batches(chunk_size)`).
- Richer result type: `SampleResult::logical_error_rate(observable)`, `detector_rates()`, `export_dem()`.
- PyO3 bindings for `sampling::stab` in `pecos-rslib`; Python helper `pecos.sampling.stab(dag, noise=..., detectors=..., shots=...)`.
- Noise-model trait hierarchy (see `dem_stab_sim_skeleton.md`: `Uniform`, `PerLocation`, `PauliLindblad`, `FromChannel`, eventually `FromLindblad`).
- GPU batch sampler (wgpu) once CPU numbers motivate it.
- Future sampling strategies: `sampling::subset`, `sampling::importance`, `sampling::rare_events`, ... each as its own submodule under `sampling/`.

## Open questions

- [ ] Exact free-function name: `sampling::stab(dag)` vs `sampling::stab::sample(dag)` vs `sampling::stab::new(dag)`. Decide at implementation. Grug lean `sampling::stab(dag)` -- one token, reads cleanly.
- [ ] Where does `sampling::neo` live? In `pecos-qec/src/sampling/neo.rs` as a re-export of the `pecos-neo` crate, or should `pecos-neo` itself expose the module path? Defer until `sampling::stab` lands; decide then.
- [ ] `sampling::monte_carlo` as an alias of `sim()` implies a dep from `pecos-qec` -> `pecos-engines`. Currently it's the other way. If keeping `pecos-qec` free of `pecos-engines`, the alias lives in the metacrate `pecos` instead. Likely the right answer.
