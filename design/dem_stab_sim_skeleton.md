# DemStabSim Module Skeleton

Status: draft / skeleton
Target crate: `pecos-simulators` (parent) + engine integration in `pecos-engines`
Pairs with: `design/qec_sim_literature.md` (rationale, literature, build order)

## Goals in one paragraph

Make the existing `pecos-qec::fault_tolerance::dem_builder` pipeline (DagFaultAnalyzer -> DemBuilder / DemSamplerBuilder) selectable through `sim(program).quantum(dem_stab()).noise(...)` as a first-class quantum backend. It behaves as "Clifford + depolarizing-family noise, sampled via precomputed fault influence". Non-adaptive circuits only. No Stim dependency.

## Crate layout

Two parts:

```
crates/pecos-qec/src/dem_stab.rs                         (pure sim type; peer of fault_tolerance)
crates/pecos-engines/src/dem_stab_engine.rs              (QuantumEngine impl + builder)
```

**Crate pivot from initial skeleton (2026-04-11).** `pecos-qec` depends on `pecos-simulators` (one-way), so a `dem_stab` module inside `pecos-simulators` would have to pull `pecos-qec` upward and create a cycle. The DemSampler / DemBuilder / DagFaultAnalyzer machinery already lives in `pecos-qec`, and DemStabSim is a thin wrapper over that machinery -- so `pecos-qec::dem_stab` is both the only legal home and the honest one. It sits as a peer of `fault_tolerance`, not inside it (it *uses* fault tolerance rather than being part of it).

Other backends keep the `pecos-simulators` + `pecos-engines` split (`SparseStab` + `SparseStabEngine`); DemStabSim's split is `pecos-qec` + `pecos-engines` instead. Same orchestration pattern either way.

## Public surface -- Rust

### `pecos_simulators::dem_stab`

```rust
pub struct DemStabSim {
    dag: DagCircuit,
    noise: Arc<dyn DemStabNoiseModel>,
    detectors: Vec<DetectorDef>,
    observables: Vec<LogicalObservable>,

    // Lazy-built, cached across shots.
    sampler: OnceLock<DemSampler>,
    influence_map: OnceLock<DagFaultInfluenceMap>,
}

impl DemStabSim {
    pub fn builder() -> DemStabSimBuilder { DemStabSimBuilder::default() }

    /// Consume N shots.
    pub fn sample_batch(&mut self, shots: usize, rng: &mut impl Rng)
        -> DemStabShotBatch;

    pub fn detector_error_model(&mut self) -> &DetectorErrorModel;
}

pub struct DemStabSimBuilder { /* private */ }

impl DemStabSimBuilder {
    pub fn circuit(mut self, dag: DagCircuit) -> Self;
    pub fn tick_circuit(mut self, tc: TickCircuit) -> Self;        // convenience
    pub fn noise<N: DemStabNoiseModel + 'static>(mut self, n: N) -> Self;
    pub fn detectors(mut self, d: Vec<DetectorDef>) -> Self;
    pub fn observables(mut self, o: Vec<LogicalObservable>) -> Self;
    pub fn build(self) -> Result<DemStabSim, DemStabError>;
}

pub struct DemStabShotBatch {
    pub detector_flips: PackedBits2D,         // shots x num_detectors
    pub observable_flips: PackedBits2D,       // shots x num_observables
    pub measurement_record: Option<PackedBits2D>, // opt-in, via MemBuilder
    pub stats: SamplingStatistics,
}

#[derive(Debug, thiserror::Error)]
pub enum DemStabError {
    #[error("circuit contains classical feed-forward; use sparse_stab() for adaptive circuits")]
    AdaptiveCircuit,
    #[error("unsupported non-Clifford gate {0:?}; use CliffordRz or STN/MAST")]
    NonClifford(GateType),
    #[error(transparent)]
    Builder(#[from] DemBuilderError),
}
```

### `DemStabNoiseModel` trait -- unified noise input

```rust
/// Lowers to per-fault-location Pauli rates consumed by DemBuilder.
pub trait DemStabNoiseModel: Send + Sync + Debug {
    fn noise_config(&self, circuit: &DagCircuit) -> NoiseConfig;
}

// Concrete structs (same convention as DepolarizingNoise / BiasedDepolarizingNoise):
pub struct Uniform { pub p_1q: f64, pub p_2q: f64, pub p_meas: f64, pub p_prep: f64 }
pub struct PerLocation { /* HashMap<SpacetimeLocation, PauliChannel> */ }
pub struct PauliLindblad { pub generators: Vec<(PauliString, f64)> }   // IBM-style learned
pub struct FromChannel { pub channel: ChannelMatrix }                  // Pauli-twirled lowering

impl DemStabNoiseModel for Uniform { /* trivial */ }
impl DemStabNoiseModel for PerLocation { /* trivial */ }
impl DemStabNoiseModel for PauliLindblad { /* decompose generators */ }
impl DemStabNoiseModel for FromChannel { /* PTM -> Pauli rates, error on residual non-Pauli */ }
```

Future additions (no API impact today): `FromLindblad { op, duration }`, `FromTrajectorySamples { .. }` once item #7 (Lindblad sim) lands.

## Engine integration -- Path A (record-and-replay)

### `pecos_engines::dem_stab_engine`

```rust
pub struct DemStabEngine {
    n_qubits: usize,
    dag: DagCircuit,
    detectors: Vec<DetectorDef>,
    observables: Vec<LogicalObservable>,
    noise: Option<Arc<dyn DemStabNoiseModel>>,
    seed: u64,

    // Built lazily on first shot_end.
    sim: Option<DemStabSim>,
    shot_rng: PecosRng,
}

impl QuantumEngine for DemStabEngine {
    fn process(&mut self, msg: ByteMessage) -> Result<ByteMessage, PecosError> {
        // 1. Decode ByteMessage into gates / measurements / shot-boundary.
        // 2. If it is a gate -> push into self.dag (+ validate: no non-Clifford, no feedback).
        // 3. If it is a measurement request:
        //      - on first call: lazy-build self.sim via DagFaultAnalyzer + DemSamplerBuilder.
        //      - sample one shot, return packed measurement outcomes as ByteMessage.
        //      - if circuit has NO measurements-used-classically, we can also defer to shot-end.
        // 4. If it is a shot-reset: clear per-shot scratch (but NOT dag / sampler caches).
        //    (Re-entrant input = error: we only accept a single static program.)
    }
    fn set_seed(&mut self, seed: u64) { self.shot_rng = PecosRng::from_seed(seed); ... }
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
}

pub struct DemStabEngineBuilder {
    noise: Option<Arc<dyn DemStabNoiseModel>>,
    detectors: Vec<DetectorDef>,
    observables: Vec<LogicalObservable>,
    num_qubits: Option<usize>,
}

impl QuantumEngineBuilder for DemStabEngineBuilder {
    fn build(&mut self) -> Result<Box<dyn QuantumEngine>, PecosError> { ... }
    fn set_qubits_if_needed(&mut self, n: usize) { self.num_qubits.get_or_insert(n); }
}

/// Free-function backend constructor, matching sparse_stab() / state_vector() convention.
#[must_use]
pub fn dem_stab() -> DemStabEngineBuilder { DemStabEngineBuilder::default() }
```

### Usage (Rust)

```rust
use pecos_engines::{sim, dem_stab, ...};

let results = sim(program)
    .quantum(dem_stab()
        .detectors(detectors)
        .observables(observables))
    .noise(dem_stab::Uniform { p_1q: 1e-3, p_2q: 5e-3, p_meas: 1e-3, p_prep: 1e-3 })
    .seed(42)
    .run(100_000)?;
```

Note: the noise is set via `.noise(...)` at the `SimBuilder` level **but** DemStabSim needs it at circuit-build time, not per-gate. Solution: `DemStabEngine::build()` pulls the `NoiseModel` out of the orchestrator wiring and downcasts it to `DemStabNoiseModel` (using `as_any`). If the noise model isn't a `DemStabNoiseModel`, return `DemStabError::Builder(...)` with a clear message. Alternative: add `.dem_stab_noise(...)` on the builder to bypass the shared `.noise()` slot. Grug prefers the downcast -- keeps one noise API.

### Python usage

Mirror on the Python side in `pecos-rslib` / `pecos` package:

```python
from pecos import sim
from pecos.backends import dem_stab
from pecos.noise import Uniform

results = (
    sim(program)
    .quantum(dem_stab().detectors(dets).observables(obs))
    .noise(Uniform(p_1q=1e-3, p_2q=5e-3, p_meas=1e-3, p_prep=1e-3))
    .seed(42)
    .run(100_000)
)
```

## Rejection / validation

DemStabEngine must reject circuits it cannot honestly handle. Two classes:

1. **Adaptive (classical feed-forward).** If any gate's application depends on a prior measurement outcome, reject.
2. **Non-Clifford.** T / RZ / RX(theta) / etc. reject.

Rejection happens in `process()` at ByteMessage decode time -- not at `build()` -- because the DAG is streamed in. Clear error messages that name the offending gate and point to `sparse_stab()` + `pecos-neo` or `CliffordRz` / STN as appropriate.

## Tests (initial)

1. **Parity test (core).** Small distance-3 repetition-code / surface-code memory experiment. Run both:
   - `sim(prog).quantum(sparse_stab()).noise(DepolarizingNoise{p}).run(N)` with N = 1e5 shots.
   - `sim(prog).quantum(dem_stab().detectors(...).observables(...)).noise(Uniform{...})`.
   Assert detector-flip and logical-error-rate distributions match within binomial CI. Reuse `compare_dems_statistical` machinery.
2. **Rejection tests.** Assert `DemStabError::AdaptiveCircuit` on a feed-forward circuit and `NonClifford` on a T-gate circuit.
3. **DEM export parity.** Build the DEM via `DemBuilder` directly and via `DemStabSim::detector_error_model()`; assert equal via existing `compare_dems_exact`.
4. **Determinism.** Same seed, same shots -- identical results across runs.
5. **Benchmark.** `criterion` bench vs `sparse_stab` Monte Carlo for increasing N (expect crossover ~100 shots, large asymptotic speedup).

## Out of scope for v1

- **Path B batch-mode fast-path** -- planned v2. Does the orchestrator surgery to let DemStabSim skip the per-shot streaming loop entirely. Grug do this *after* Path A lands so v1 doesn't drag along an orchestrator redesign.
- Non-Clifford hybrid escape (falling back to `CliffordRz` for RZ slices). Land as v2+ once basic path proven.
- GPU sampling via wgpu. Natural follow-up once CPU numbers are known.
- Lindblad-derived noise input. Blocked on trajectory sim (item #7 in literature survey).

## v2: Path B batch-mode fast-path (sketch)

Kept in this doc so the v1 design does not paint v2 into a corner.

**Goal.** When `SimBuilder` is configured with a batch-capable quantum backend, bypass the per-shot classical-engine loop and feed the whole compiled program to the backend in one call.

**Mechanism.**
- New trait in `pecos-engines`, e.g. `BatchQuantumEngine: QuantumEngine` with `run_shots(&mut self, shots: usize, rng: &mut dyn RngCore) -> ShotVec`. Default impl falls through to the current streaming loop for backends that do not implement it natively.
- `MonteCarloEngine::run(shots)` downcasts on its `QuantumEngine` trait object: if batch-capable and circuit is static (no classical feed-forward from the classical engine), call `run_shots` once; otherwise current per-shot loop.
- DemStabEngine implements `BatchQuantumEngine` natively: the internal sampler already does `sample_batch`, so `run_shots` is one call into `DemSampler::sample_batch` with a single RNG split.
- Preserves the `sim(program).quantum(dem_stab()).noise(...).run(N)` user API -- the path split is internal.

**Design constraints v1 must preserve:**
- `DemStabEngine` already owns the built `DemSampler` across shots (Path A caches it). Path B just exposes a batch entry point alongside the streaming one. No duplication.
- DAG must be fully captured before batch execution begins -- identical to Path A's lazy-build trigger. Path A's record step is reused.
- `DemStabShotBatch` (the v1 return type of the sim) becomes the natural backing type for the batch return path.

**What v2 does not need v1 to commit to:**
- Final form of `BatchQuantumEngine` trait (single method vs. split detect/observable/measurement).
- Whether batch fast-path is opt-in via config or auto-enabled on downcast success.

Decide at the start of v2, with Path A's numbers as evidence.

## Open questions

- [ ] Should the detectors/observables live on the `DemStabEngine` builder (as drafted) or on the program IR itself (HUGR/QASM annotations)? The latter is cleaner long-term but out of scope for v1.
- [ ] `Path B` batch fast-path: useful right away (bypass classical engine) or premature? Grug lean premature -- do Path A, measure, revisit.
- [ ] How do we want to surface the DEM object for decoder hand-off? Return it from `run()` alongside shots? Provide a sidecar `.build_dem_only()` method? Likely both.
- [ ] Seeding semantics for parallel shot batches (`rayon`): per-shot seed = master_seed XOR shot_idx is deterministic and trivially parallelizable. Default to that.
