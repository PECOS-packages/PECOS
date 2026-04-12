# `sample_stab` -- Batch Orchestration for Static Stabilizer Sampling

Status: draft / proposal
Pairs with: `design/qec_sim_literature.md`, `design/dem_stab_sim_skeleton.md`
Date: 2026-04-11

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

## Decision

Build a separate orchestration entry point, sibling to `sim()`, that preserves batch semantics end-to-end:

```rust
pub fn sample_stab(dag: DagCircuit) -> StabSampleBuilder
```

This mirrors how `pecos-neo` sits next to `sim()` with its own `sim_neo()` entry: each orchestration is honest about the computational model it serves. `sim()` = per-shot classical-control Monte Carlo. `sample_stab()` = one-shot compile + batch sample.

No retrofit of `MonteCarloEngine`. No record-and-replay through `QuantumEngine`. A clean, separate path.

## Builder chain

```rust
let result = sample_stab(dag)
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
- `.workers(n)` -- rayon worker count. `workers(0)` or omitted -> single-threaded. `workers(None)` / helper `.auto_workers()` -> `available_parallelism`.
- `.seed(u64)` -- master seed; split deterministically across workers.
- `.run()` -- consumes the builder, runs, returns `StabSampleResult`.

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
pub struct StabSampleResult {
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

- Module: `crates/pecos-qec/src/sample.rs`.
- Re-export: `pecos-qec/src/lib.rs` -> `pub use sample::{sample_stab, StabSampleBuilder, StabSampleResult, StabSampleError};`.
- Metacrate: `pecos::sample_stab` via existing pecos-qec re-export path.

Python bindings in `pecos-rslib` follow later (v1b / v2), once the Rust API is stable.

## Relationship to `DemStabSim` / `MemStabSim`

`sample_stab()` is the **user-facing** orchestration. It is implemented on top of `DemStabSim` / `MemStabSim` without giving up direct access to them.

Power users keep the lower-level APIs:

```rust
let sim = DemStabSim::builder().circuit(dag).noise(n).detectors(d).build()?;
let mut rng = SmallRng::seed_from_u64(seed);
let batch = sim.sample_batch(n_shots, &mut rng);
// ... introspect sim.sampler(), export DEM, etc.
```

`sample_stab` is for "give me shots, now"; the typed sims are for "I need the sampler object for something else".

## What this deliberately is not

- Not an extension of `MonteCarloEngine`. Per-shot streaming is not the right fit and forcing it throws away the batch primitive.
- Not a replacement for `sim()`. Classical control + adaptive programs still go through `sim()` with `sparse_stab` + `pecos-neo` noise. Two entries, two computational models.
- Not a Stim rewrite. Naming is honest: "sample a stabilizer circuit statically with depolarizing-family noise". Stim's workflow is inspiration only; the entry point is PECOS-shaped.
- Not an abstraction-first design. If a higher layer above `sample_stab()` and `sim()` turns out to be useful later, it earns that place once concrete duplication shows up. Not before.

## v1 deliverables

1. `crates/pecos-qec/src/sample.rs`:
   - `StabSampleBuilder` + chain.
   - `StabSampleResult`.
   - `StabSampleError` (e.g. missing shots, missing noise, builder misuse).
   - Dispatch logic: DEM vs MEM on detector/observable presence.
   - Rayon parallel shot split; deterministic per-worker seed.
2. Re-exports in `pecos-qec/src/lib.rs`.
3. Integration test: distance-3 repetition code, 10k shots, both DEM and MEM paths, assert logical-error rate sits within binomial CI around the analytic expectation.
4. Doctest on the module header example.
5. Clippy clean.

## v2+ ideas (not blocking v1)

- Accept program IRs (QASM / QIS / HUGR / PHIR) with static-check + lowering to `DagCircuit`.
- Accept `TickCircuit` directly.
- Streamed output API (`stream_batches(chunk_size)`).
- Richer result type: `StabSampleResult::logical_error_rate(observable)`, `detector_rates()`, `export_dem()`.
- PyO3 bindings for `sample_stab` in `pecos-rslib`; Python helper `pecos.sample_stab(dag, noise=..., detectors=..., shots=...)`.
- Noise-model trait hierarchy (see `dem_stab_sim_skeleton.md`: `Uniform`, `PerLocation`, `PauliLindblad`, `FromChannel`, eventually `FromLindblad`).
- GPU batch sampler (wgpu) once CPU numbers motivate it.
