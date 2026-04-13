# Lindblad / Trajectory Simulator -- Module Skeleton

Status: draft (2026-04-12)
Target crate: new `pecos-lindblad` (or module inside `pecos-neo`; decision below)
Pairs with: `design/qec_sim_literature.md` (#7), `design/lindblad_magnus_algorithm.md` (math spec + closed forms), `design/dem_stab_sim_skeleton.md`, `design/stab_sample_orchestration.md`

## Goals in one paragraph

Add a continuous-time open-system simulator to PECOS that bridges device physics
to the existing DEM / stabilizer pipeline via a **gate + duration -> effective
Pauli-Lindblad rates** lowering (arXiv:2502.03462). Offer two surfaces:
(a) `Gate -> Magnus-on-superoperator -> PauliLindbladModel -> DemStabSim`
for fast Pauli-twirled noise used in QEC threshold sweeps, and
(b) a direct `LindbladTrajectorySim` (MCWF / quantum jumps) for small circuits
(<= 10 qubits) to validate the twirl and to expose coherent / non-Pauli effects
that a twirled DEM misses (arXiv:2402.16727, 2510.23797).
Keep scope minimal: adaptive Dormand-Prince solver, dense `faer::Mat`, rayon
fan-out on trajectories. No MPS, no Krylov, no GPU on v1.

## Why now (one-liner per literature scout)

Nobody in the mature Python/Julia stack (QuTiP, Dynamiqs, QuantumToolbox.jl)
exposes the `Lindbladian + gate_duration -> PauliLindbladModel` API. This is
the wedge. Everything else is the 80/20 ODE plumbing.

## Two-path architecture

```
                                                   +----------------------+
Gate { H_ideal, H_err, c_ops, duration }           |   PauliLindbladModel |
       |                                           |   { supports, rates }|
       | Magnus / Dyson on superoperator           +----------+-----------+
       | (arXiv:2502.03462)                                   |
       v                                                      | feeds
(a) Pauli-twirled -> PauliLindbladModel  ------------------->  DemStabSim
                                                              (fast, threshold sweeps)

Gate { ... } + initial |psi>
       |
       | MCWF / quantum jumps (Daley 2014)
       | dense n x n, adaptive RK (Dormand-Prince)
       v
(b) LindbladTrajectorySim  ------> syndrome samples -> learned DEM
                                                      (captures coherent
                                                       hyperedges twirl misses)
```

Both paths share `Gate`, `Lindbladian`, and `PauliBasis` primitives. Path (a)
is the MVP and the integration with `DemStabSim`. Path (b) is the reference
validator -- small but honest.

## Crate placement (decision)

**Proposal: new crate `crates/pecos-lindblad`** peer of `pecos-neo`, not inside
it. Reasons:

1. `pecos-neo`'s `NoiseChannel` trait is sample-only (`NoiseResponse` is
   a Pauli-injection enum). Lindblad synthesis lowers *to* rates; trajectory
   sim lowers *to* a stream of pure-state wavefunctions. Neither fits the
   current trait without widening it. Keep trait lean; add a sibling crate
   that *feeds* it.
2. Lindblad sim pulls in `faer` + `ode_solvers` (or `diffsol`). `pecos-neo`
   today only has `rand`, `rayon`, `smallvec`. Don't fatten the base crate.
3. Clean cut: `pecos-lindblad` produces `PauliLindbladModel` and
   `TrajectoryShot`; `pecos-neo` and `pecos-qec::dem_stab` consume them.

Alternative considered: put behind a feature flag inside `pecos-neo`.
Rejected -- feature flags on numerics crates become build-matrix purgatory.

## Crate layout

```
crates/pecos-lindblad/
  Cargo.toml            # faer, ode_solvers (or diffsol), rand, rayon, thiserror
  src/
    lib.rs
    basis.rs            # PauliBasis, PauliString, SparsePauliOp
    lindbladian.rs      # Lindbladian { H: Matrix, c_ops: Vec<Matrix> }
    gate.rs             # Gate { ideal: UnitaryRep, H_drive(t), c_ops, duration }
    magnus.rs           # Magnus/Dyson on superoperator -> effective generator
    pauli_twirl.rs      # Liouvillian -> diagonal in Pauli basis -> rates
    trajectory.rs       # MCWF / quantum jumps unraveling (Path b)
    solver.rs           # thin Dormand-Prince wrapper over faer state
    api.rs              # public builders: MagnusSynth, TrajectorySim
  tests/
    parity_small.rs     # symbolic case from arXiv:2502.03462 Appendix C
    trajectory_vs_me.rs # mcwf(N) -> rho_avg vs mesolve, within statistical CI
    pauli_twirl_roundtrip.rs
```

Integration-side additions (kept in owning crates):

```
crates/pecos-qec/src/dem_stab.rs
    + DemStabNoiseModel impl for pecos_lindblad::PauliLindbladModel
crates/pecos-neo/src/noise/lindblad_derived.rs
    + Cached LindbladChannel(table: HashMap<(GateId, OrderedFloat<f64>), PauliChannel>)
```

## Public surface -- Rust

### Core types

```rust
/// Sparse Pauli-basis decomposition (arXiv:2201.09866 generator form).
pub struct PauliLindbladModel {
    pub supports: Vec<PauliString>,
    pub rates: Vec<f64>,              // lambda_k >= 0
}

impl PauliLindbladModel {
    /// p_flip for Pauli k over duration t: (1 - exp(-2 lambda_k t)) / 2.
    pub fn flip_probs(&self, t: f64) -> Vec<f64>;

    /// Sample one realization over duration t.
    pub fn sample(&self, t: f64, rng: &mut impl Rng) -> SparsePauliOp;
}

/// Time-independent collapse-operator Lindbladian.
/// Time-dependent H is handled on `Gate` via H_drive(t) closure.
pub struct Lindbladian {
    pub hamiltonian: FaerMat<Complex64>,  // n x n
    pub collapse_ops: Vec<(FaerMat<Complex64>, f64)>,   // (c_k, gamma_k)
}

pub struct Gate {
    pub label: &'static str,
    pub ideal: UnitaryRep,                // for sanity-check / inverse
    pub drive_hamiltonian: Option<Box<dyn Fn(f64) -> FaerMat<Complex64> + Send + Sync>>,
    pub static_lindbladian: Lindbladian,  // H_err + c_ops during the gate
    pub duration: f64,
}
```

### Path (a): Magnus synthesis

```rust
pub struct MagnusSynth {
    order: u8,               // 1, 2, 3, 4 (paper goes to 4)
    twirl: bool,             // default true for PauliLindbladModel output
    basis: PauliBasis,       // sparse by default, up to weight-2
}

impl MagnusSynth {
    pub fn synthesize(&self, gate: &Gate) -> Result<PauliLindbladModel, SynthError>;

    /// Untwirled variant: full Liouville generator (useful for (b) validation).
    pub fn synthesize_generator(&self, gate: &Gate)
        -> Result<FaerMat<Complex64>, SynthError>;
}

/// Grug-fallback: numerical integration of the Liouvillian directly, no Magnus.
/// Use as gold standard; slow but unambiguous.
pub fn synthesize_numerical(gate: &Gate, rtol: f64, atol: f64)
    -> Result<FaerMat<Complex64>, SynthError>;
```

### Path (b): trajectory simulator

```rust
pub struct TrajectorySim {
    initial_state: StateVec,
    gate_sequence: Vec<Gate>,
    num_trajectories: usize,
    seed: u64,
}

impl TrajectorySim {
    pub fn builder() -> TrajectorySimBuilder { ... }

    pub fn run(&self) -> TrajectoryBatch;          // rayon fan-out
}

pub struct TrajectoryBatch {
    pub final_states: Vec<StateVec>,               // one per trajectory
    pub jump_records: Vec<Vec<JumpEvent>>,         // when / which c_op fired
    pub measurement_outcomes: Option<PackedBits2D>,
}
```

### Glue into DemStabSim

```rust
// in pecos-qec::dem_stab
impl DemStabNoiseModel for pecos_lindblad::PauliLindbladModel { ... }

// Usage:
let pl_noise: PauliLindbladModel = MagnusSynth::order(2).synthesize(&gate_cx)?;
let sim = DemStabSim::builder()
    .circuit(dag)
    .noise(pl_noise)           // directly consumed, no conversion layer
    .detectors(...)
    .observables(...)
    .build()?;
```

### Glue into pecos-neo (non-DEM stabilizer Monte Carlo)

For researchers who want realistic noise on a `sparse_stab()` run without the
DEM detour: cache the Magnus output per gate and expose it as a
`NoiseChannel` that injects Pauli samples.

```rust
// in pecos-neo::noise::lindblad_derived
pub struct LindbladChannel {
    // (GateId, duration_ns) -> precomputed PauliLindbladModel
    table: HashMap<(GateId, OrderedFloat<f64>), PauliLindbladModel>,
}

impl NoiseChannel for LindbladChannel {
    fn apply(&self, event: &NoiseEvent, ctx: &mut NoiseContext, rng: &mut PecosRng)
        -> NoiseResponse {
        // look up (gate_id, duration) in table; sample Pauli; InjectGates
    }
}
```

**Pre-req path (audit 2026-04-12, revised).** No first-class duration field
exists on `GateCommand` / `NoiseEvent::AfterGate`. Instead of a schema change,
use the existing `TickCircuit` / `DagCircuit` `Attribute` metadata dictionary
(`crates/pecos-quantum/src/tick_circuit.rs:1147`): standardize on the key
`"gate_duration"` = `Attribute::Float(nanoseconds)`. The `pecos-neo` circuit
converter reads this key at translation time into `GateCommand`, and
`LindbladChannel` queries it via `ctx` at apply time. Zero breaking changes
to core circuit types. The schema-extension option (adding
`duration: Option<TimeUnits>` to `NoiseEvent::AfterGate`) is reserved for a
later PR if the metadata convention proves insufficient -- grug do not prepay
that complexity.

## Noise-model input hierarchy

| Input shape | Path to DEM | Who produces it |
|---|---|---|
| Ideal + per-qubit T1/T2 + gate duration | Magnus (1st order enough) | user spec |
| + coherent over-rotation / miscalibration | Magnus (2nd-4th order) | user spec or fit |
| + 2Q ZZ crosstalk | Magnus (closed-form from 2502.03462 Appendix D) | user spec |
| Learned sparse PL from device (PEC) | direct -- already PauliLindbladModel | cycle-benchmarking fit (future `pecos-char` crate?) |
| General CPTP / Choi channel | Pauli-twirl -> rates | `synthesize_numerical` then twirl |

## Solver choice

Default: **adaptive Dormand-Prince 5(4)** via `ode_solvers` (nalgebra-native,
simple) or `diffsol` (more features, BDF for stiff). Tolerances `rtol=1e-6`,
`atol=1e-9`. Pack `Complex64` as interleaved `[re, im, re, im, ...]`
`Vec<f64>` for solver input (both crates are real-only).

Magnus integrand evaluation: closed-form first order, trapezoidal second-order
nested, adaptive Gauss-Kronrod for third/fourth order (or punt to
`synthesize_numerical` for order > 2). Grug vote: ship order-2 as default; add
higher orders only when a test case demands.

## State representation

Keep density matrix as `faer::Mat<Complex64>` of size `n x n` (QuTiP's
`matrix_form` trick). Apply `-i[H, rho] + sum_k (c_k rho c_k^dag -
(1/2){c_k^dag c_k, rho})` directly. Avoid materializing the `n^2 x n^2`
superoperator except for Pauli-transfer-matrix extraction. Crossover to
vectorized superop at `n <= 4 qubits` via a cfg-gated fast path; do not
implement v1.

Pauli basis representation: `PauliString` = pair of `BitVec` (x-part, z-part)
plus sign/phase. Sparse storage `Vec<(PauliString, f64)>` for
`PauliLindbladModel`. Up to weight-2 by default; weight-3+ opt-in.

## Trajectory parallelization

- `rayon::par_iter` over `0..num_trajectories`.
- Each trajectory gets its own `rand_chacha::ChaCha12Rng` seeded from
  `master_seed.wrapping_add(trajectory_idx as u64)`. Deterministic under
  parallel execution.
- No GPU on v1. (Dynamiqs-style `vmap+jit` is the future win but requires
  wgpu/CUDA ODE integrator -- defer until CPU numbers demand.)

## Tests (initial)

1. **Magnus parity vs paper.** Reproduce the amplitude-damping-under-identity
   and the cross-resonance CX symbolic rates from arXiv:2502.03462 Appendix C
   within `< 1e-10`. Freeze numerical values as test fixtures.
2. **Magnus vs numerical.** For random Lindbladians with `beta/omega_g < 0.1`,
   compare `MagnusSynth::order(4)` against `synthesize_numerical` -- should
   match within `< 1e-6`.
3. **Magnus out-of-regime detection.** At `beta/omega_g = 1.0`, Magnus order-2
   vs order-4 should diverge; `SynthError::OutOfConvergenceRegime` must fire.
4. **Trajectory vs master equation.** For 1-qubit T1 decay, 10k trajectories
   averaged should match `mesolve` output within binomial CI.
5. **Pauli twirl round-trip.** `Liouvillian -> twirl -> Pauli rates -> sample
   -> average` must match the diagonal of the twirled generator.
6. **End-to-end DemStabSim glue.** Small rep-code memory experiment: feed
   `MagnusSynth` output to `DemStabSim`, compare logical error rate against
   the trajectory path (path b) on the same circuit.
7. **Integration regression.** Once the `"gate_duration"` metadata convention
   lands in `pecos-neo`'s circuit converter, add a parity test with
   `LindbladChannel` vs `DemStabSim + PauliLindbladModel` on the same
   circuit annotated with per-gate durations.

## Rejection / validation

- Non-Hermitian `H_drive`: reject at `Gate` construction.
- Non-CP `c_ops` (gamma_k < 0): reject. Pseudo-Lindblad (arXiv:2306.14876)
  opt-in only.
- Magnus convergence check: estimate `||beta|| * duration` vs `||H_ideal||`;
  emit warning if > 0.3, error if > 1.0 (tunable).
- Time-ordered integrals of `H_drive(t)`: require user-supplied `Fn(f64)`
  plus optional `sample_points` hint; otherwise adaptive.

## Out of scope for v1

- **GPU solver path.** Natural follow-up once CPU numbers are known.
- **MPS / tensor-network Lindblad.** Too far from the wedge.
- **Non-Markovian (Redfield, HEOM) solvers.** Separate design.
- **Stochastic master equation (SME) / diffusive unravelings.** Separate design.
- **Krylov / expm-based propagators.** Default is adaptive RK; add later if
  stiffness demands.
- **Magnus as time-stepping integrator.** Different use (arXiv:2407.03576);
  here Magnus is for effective-generator synthesis only.
- **Leakage-aware Lindblad.** Needs 3-level model; design separately
  (see scout open question on leakage).

## Open questions

- [ ] Crate name: `pecos-lindblad` vs `pecos-open-system` vs fold into
  `pecos-neo`. Grug prefers `pecos-lindblad` -- says what it is.
- [ ] Magnus order default: 2 (cheap, closed-form) or 4 (paper's highest).
      Probably 2; 4 as opt-in for research.
- [ ] Should `PauliLindbladModel` live in `pecos-lindblad` or promoted to
      `pecos-qec` so `DemStabSim` can consume it without a dependency flip?
      Lean `pecos-qec` -- it's a noise format, not Lindblad-specific.
- [ ] Symbolic vs numerical Appendix-D (ZZ crosstalk) formulae:
      can we parse paper's LaTeX / Mathematica output, or transcribe? Manual
      transcription of 3-4 closed forms is honest work; skip symbolic
      pipelines.
- [x] Gate-duration data path (**audit 2026-04-12**): no first-class field
      on `GateCommand`, `TickCircuit`, `DagCircuit`, or `NoiseEvent::AfterGate`.
      Only `GateCommand::idle(qubit, duration)` (stashed in `angles`, see
      `exp/pecos-neo/src/command.rs:296`) and `NoiseEvent::IdleTime { duration }`
      (`exp/pecos-neo/src/noise.rs:203-206`) carry duration today. **Decision:**
      use `TickCircuit` / `DagCircuit` `Attribute` metadata dictionary
      (`crates/pecos-quantum/src/tick_circuit.rs:1147`) with standardized key
      `"gate_duration"` = `Attribute::Float(ns)`. Lower-risk than a schema
      change; zero breaking changes to core circuit types. The
      `pecos-neo/src/circuit.rs` converter reads this key when translating
      to `GateCommand`; `LindbladChannel` queries it at lookup time. Promote
      to a first-class field on `NoiseEvent::AfterGate` only if the metadata
      convention proves itself insufficient in practice.
- [ ] Seeding semantics for `MagnusSynth` (deterministic, no RNG needed) vs
      `TrajectorySim` (per-trajectory seed). Document clearly.

## Build order

1. **`pecos-lindblad` crate scaffold** + `PauliBasis` + `Lindbladian` +
   `Gate` types. Tests: round-trip Pauli ops, Lindbladian constructor sanity.
2. **`synthesize_numerical`** (gold-standard, slow). One integrator, one pass.
   Tests: trajectory-vs-mesolve (test #4) uses this.
3. **`MagnusSynth::order(1)` + `order(2)`** with twirl. Tests: parity vs
   `synthesize_numerical` + paper fixtures (tests #1, #2, #5).
4. **Glue into `DemStabSim`** -- implement `DemStabNoiseModel` for
   `PauliLindbladModel`. Tests: #6.
5. **`TrajectorySim`** (Path b) -- MCWF, rayon fan-out. Tests: #4 properly,
   small rep-code validation run.
6. **`"gate_duration"` metadata convention + `LindbladChannel`** in
   `pecos-neo` (converter reads `TickCircuit` attribute; channel looks up
   cached Pauli rates). Tests: #7.
7. **Higher-order Magnus (3, 4)** + convergence detection. Tests: #3.

Stop after step 4 if that's all that's needed for the next research run.
Steps 5-7 unlock honest coherent-error studies but are not on the critical
path for Pauli-noise threshold sweeps.

## References

Must-read:
- arXiv:2502.03462 -- Magnus/Dyson Lindblad synthesis (**the algorithm**)
- arXiv:2201.09866 -- sparse Pauli-Lindblad generator form
- arXiv:2311.15408 -- learning sparse PL models
- arXiv:2402.16727 -- when Pauli approximation underestimates QEC failure
- arXiv:2510.23797 -- coherent-error hyperedges missing from twirled DEMs
- arXiv:2407.03576 -- 4th-order commutator-free Magnus (gold-standard cross-check)
- Daley 2014 (Adv. Phys.) -- MCWF review

Should-read:
- arXiv:2406.08981 -- Bayesian CPTP learning from syndromes
- arXiv:2512.10814 -- decoder-free DEM estimation on Willow
- arXiv:2504.21440 -- QuantumToolbox.jl architecture

Nice-to-have:
- arXiv:2306.14876 -- pseudo-Lindblad trajectories for non-GKSL
- arXiv:2405.12925 -- Magnus superconvergence (Hamiltonian sim)
- arXiv:2107.10054 -- periodic-Lindblad high-frequency expansion

Rust crates evaluated:
- `diffsol` (stiff + non-stiff, dense+sparse, nalgebra/faer)
- `ode_solvers` (simpler, nalgebra-native)
- `faer` (fast dense linalg)
- `rand_chacha` (per-trajectory seeding)
