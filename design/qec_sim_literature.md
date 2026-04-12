# QEC Simulator Literature Survey: Candidate Gaps in PECOS

Status: draft / scaffold
Scope: quantum **simulators** (not decoders) useful for QEC research that PECOS does not currently implement or is not actively working on.

## Current PECOS simulator coverage (for reference)

| Family | PECOS impl |
|---|---|
| Stabilizer tableau (sparse/dense) | `SparseStab`, `DenseStab`, `GpuStab` (wgpu), `CuStabilizer`, `pecos-cppsparsestab` |
| State vector | `StateVec` (SoA/AoS), `CuStateVec`, Qulacs, QuEST |
| Density matrix | `DensityMatrix`, `CuDensityMat`, QuestDensityMatrix |
| Clifford + T / CH-form | `CliffordRz` |
| Stabilizer + tensor network / magic states | `STN` (stabilizer tableau + MPS), `MAST` (magic-state injection + deferred ancilla projection, Clifford disentangling) -- active on branch/worktree `study/tensor-network-clifford-rz` |
| Tensor network | `CuTensorNet` |
| Pauli propagation | `PauliProp` |
| Graph state | `GraphStateSim` |
| ZX calculus | `pecos-zx` (exp) |
| Composable noise | `pecos-neo` (exp) |
| **Detector Error Model (Stim-compat)** | `pecos-qec::fault_tolerance::dem_builder` -- `DemBuilder` + `DemSamplerBuilder` (SoA batch sampler) |
| **Stabilizer-rank / sum-over-Cliffords** | `CliffordRz` via CH-form sum decomposition (Bravyi 1808.00128) |

## Out-of-scope (already covered or actively developed)

Do **not** propose these as gaps:

- **Stim-style DEM generation and sampling** -- `pecos-qec::fault_tolerance::dem_builder` (`DemBuilder`, `DemSamplerBuilder`).
- **Stabilizer-rank / sum-over-Cliffords simulation** -- `CliffordRz` (CH-form, Bravyi 1808.00128).
- **Stabilizer + tensor-network hybrids / magic-state injection via MPS** -- `STN` and `MAST` on branch/worktree `study/tensor-network-clifford-rz`.
- **Composable gate-level noise channels** -- `pecos-neo` (depolarizing, measurement, idle, crosstalk, leakage, custom).

Sections below that originally proposed any of these have been revised to focus on *refinements* or *wrapper backends* rather than net-new simulators.

## Candidate gap families

Each entry below is a simulation family that appears **not covered** by the above. TODOs mark items still to validate.

---

### 1. (REVISED) Alternatives to PECOS's existing DEM sampler

**Correction.** PECOS **already has** a Stim-compatible DEM generator and fast batch sampler at `crates/pecos-qec/src/fault_tolerance/dem_builder/`: `DemBuilder` (per-qubit fault model, 15 Pauli combos for 2Q gates, Stim-format output, hyperedge decomposition for MWPM) and `DemSamplerBuilder` (SoA / CSR / bit-packed u64 / rayon batch sampling).

**So the real question is:** what *simulator backends* could feed or complement the existing DemSampler?

The DemSampler today operates on a `DagFaultInfluenceMap` produced by circuit-level *fault propagation analysis* (not a full quantum sim). That is deliberate and fast, but it assumes Pauli/depolarizing-tractable noise. Gaps worth a literature hunt:

- **Stim as a cross-validation backend.** Wrap `stim.Circuit -> DEM` via PyO3/FFI so PECOS circuits can be round-tripped through Stim and diffed against PECOS's own DEM. Test harness already exists (`test_dem_sampler_vs_stim.py`). Worth formalizing as a first-class optional backend.
- **Coherent / non-Pauli noise -> effective DEM.** For circuits with rotations, coherent over-rotations, leakage, crosstalk, the Pauli-twirled DEM loses information. Candidates to compute an *approximate* DEM from a richer sim:
  - Lindblad / trajectory sim (see #7 below) -> tomographic per-location Pauli channel -> feed into DemBuilder.
  - Pauli-Lindblad learned models (see #9) -> sparse correlated error mechanisms as DEM hyperedges.
  - Matchgate / FLO (see #2) where tractable, for exact coherent error rates on small gadgets.
- **Non-Clifford logical gadgets.** For T / magic-state circuits, the influence-map pipeline must handle non-Pauli effects. `CliffordRz` (CH-form sum) is a candidate backend for an "exact small-DEM" oracle used to validate approximations.

**Seminal / anchor refs.**
- C. Gidney, *Stim: a fast stabilizer circuit simulator*, Quantum 5, 497 (2021). arXiv:2103.02202
- Stim's DEM spec + Sinter sampling harness (docs).

**Action items.**
- [ ] Survey how Stim, qec_lib, and qsim's circuit-level noise adapters handle non-Pauli noise lowering.
- [ ] Decide whether PECOS's DemBuilder gains a `from_channel(qubit_op, ...)` entry point that accepts PTM / Choi / Lindblad and lowers to Pauli rates.
- [ ] Evaluate PyO3 `stim` bridge as an optional cross-check backend.

---

### 2. Matchgate / Free-fermion / Fermionic-linear-optics (FLO) simulators

**What it is.** Efficient classical simulation of matchgate circuits (nearest-neighbor 2-qubit gates satisfying matchgate identities) via covariance-matrix evolution. Equivalent to non-interacting fermion dynamics.

**Why QEC needs it.** Majorana / fermionic codes, certain LDPC constructions built from free-fermion layers, boundary-matching benchmarks, noise models where errors are Gaussian-fermionic. Also useful as a sanity-check oracle for small non-Clifford circuits that happen to be matchgate-reducible.

**PECOS status.** Not present.

**Seminal refs.**
- Valiant, *Quantum circuits that can be simulated classically in polynomial time*, SIAM J. Comput. 31 (2002).
- Knill, *Fermionic Linear Optics and Matchgates* (2001), arXiv:quant-ph/0108033.
- Terhal, DiVincenzo (2002), arXiv:quant-ph/0108010.
- Jozsa, Miyake, *Matchgates and classical simulation of quantum circuits*, Proc. R. Soc. A 464 (2008).

**Existing OSS.**
- `Flo-simulator` (GitHub academic repo) -- matchgate + non-Gaussian gate simulator (Cudby-Strelchuk 2024, arXiv:2307.12702 / Quantum 2024).
- OpenFermion-FQE (fermionic quantum emulator; second-quantized, not matchgate-specialized but adjacent).
- No widely-used production library -- niche for PECOS.

**Newer ref worth tracking.**
- Cudby, Strelchuk, *Improved simulation of quantum circuits dominated by free fermionic operations*, Quantum 8 (2024), DOI 10.22331/q-2024-12-04-1549.

---

### 3. Decision-diagram simulators (QMDD / DDSIM)

**What it is.** Represent state/operator as a reduced decision diagram (QMDD, TDD, LIMDD). Exponential compression on structured circuits, exact non-Clifford.

**Why QEC needs it.** Exact verification of small logical gadgets (magic-state distillation circuits, small code blocks), cross-checking approximate sims, equivalence checking of compiled vs logical circuits.

**PECOS status.** Not present.

**Seminal refs.**
- Miller, Thornton, *QMDD: A Decision Diagram Structure for Reversible and Quantum Circuits* (2006).
- Zulehner, Wille, *Advanced Simulation of Quantum Computations*, IEEE TCAD (2019).
- Vinkhuijzen et al., *LIMDD: A Decision Diagram for Simulation of Quantum Computing Including Stabilizer States*, Quantum 7 (2023).

**Existing OSS.**
- MQT DDSIM (Munich Quantum Toolkit) -- C++20/Python, actively maintained: https://github.com/munich-quantum-toolkit/ddsim
- LIMDD branch of DDSIM: https://github.com/munich-quantum-toolkit/ddsim/tree/limdd -- first LIMDD implementation, compactly represents stabilizer states *and* DD-friendly non-stabilizer states.
- Q-Sylvan (parallel DD package for quantum, Springer 2025).

**Newer ref worth tracking.**
- Vinkhuijzen et al., *LIMDD*, Quantum 7 (2023) 1108.
- Tutorial: Quantum Inf. Process. (2025), https://doi.org/10.1007/s11128-025-04917-0.

---

### 4. (REVISED) Stabilizer-rank / sparsification refinements to CliffordRz

**Correction.** `CliffordRz` is a stabilizer-rank simulator -- it represents states as sum of CH-form stabilizer states and cites Bravyi et al. (arXiv:1808.00128). Each RZ doubles the term count; norm computation uses CH-form inner products. See `docs/concepts/clifford-rz-simulator.md`.

**Open research directions** (things CliffordRz may not already cover):

- **Sparsification / random stabilizer decomposition.** Bravyi-Smith-Smolin style *sampling* from the sum rather than carrying all 2^t terms -- runtime scales with *stabilizer extent* / *robustness of magic*, not 2^t.
- **Low-extent magic state decompositions.** Replace per-RZ doubling with structured multi-T decompositions (|T><T|^k at lower-than-2^k extent). See Bravyi-Browne-Calpin-Campbell-Gosset-Howard (2019).
- **Stabilizer rank lower/upper bound benchmarks.** Use as a research playground for resource-theoretic magic monotones.
- **Heuristic pruning.** Drop small-amplitude terms (already present?) with principled error bounds.

**Seminal refs.**
- Bravyi, Gosset, *Improved Classical Simulation of Quantum Circuits Dominated by Clifford Gates*, PRL 116 (2016). arXiv:1601.07601.
- Bravyi, Smith, Smolin, *Trading classical and quantum computational resources*, PRX 6 (2016). arXiv:1506.01396.
- Bravyi, Browne, Calpin, Campbell, Gosset, Howard, *Simulation of quantum circuits by low-rank stabilizer decompositions*, Quantum 3 (2019). arXiv:1808.00128.
- Qassim, Pashayan, Gosset, *Improved upper bounds on the stabilizer rank of magic states* (2021).

**Audit result (2026-04-11).** `crates/pecos-simulators/src/clifford_rz/` contains **no** pruning / sparsification / random-sampling / extent-based truncation code. Each RZ unconditionally doubles the term count. Randomized stabilizer-extent sampling (Bravyi-Gosset, Bravyi-Smith-Smolin) is a genuine refinement gap.

**Action items.**
- [x] Audit CliffordRz for existing sparsification / sampling knobs -- none present.
- [ ] Prototype extent-based randomized sampling; compare variance vs full-sum cost.
- [ ] Principled small-amplitude pruning with error bounds (deterministic counterpart).

---

### 5. Pauli-based computation (PBC) simulators

**What it is.** Replace Clifford-dominated circuits with a sequence of commuting/anticommuting Pauli measurements on magic states. Simulation cost is in measurements, not gates.

**Why QEC needs it.** Matches the native FTQC protocol model (Litinski's game of surface codes, magic-state teleportation). Makes cost of logical algorithms transparent in the same language as the hardware.

**PECOS status.** Not present as a first-class simulator. MAST is adjacent but not PBC.

**Seminal refs.**
- Bravyi, Smith, Smolin, *Trading classical and quantum computational resources*, PRX 6 (2016) [introduces PBC].
- Litinski, *A Game of Surface Codes: Large-Scale Quantum Computing with Lattice Surgery*, Quantum 3 (2019). arXiv:1808.02892.

**Existing OSS.**
- `latticesurgery-com/lattice-surgery-compiler` -- QASM -> Pauli-rotation IR -> lattice surgery ops + visualizer.
- PennyLane has a PBC compilation module.

---

### 6. Bosonic / continuous-variable simulators (GKP, cat, binomial codes)

**What it is.** Fock-truncated or phase-space (Wigner, Husimi) simulation of bosonic modes with Gaussian + non-Gaussian operations.

**Why QEC needs it.** GKP codes, cat codes, binomial codes, dual-rail, concatenated bosonic-qubit codes. Increasingly central to neutral-atom / superconducting bosonic / photonic QEC. PECOS is qubit-only.

**PECOS status.** Not present.

**Seminal refs.**
- Gottesman, Kitaev, Preskill, *Encoding a qubit in an oscillator*, PRA 64 (2001).
- Mirrahimi et al., *Dynamically protected cat-qubits*, NJP 16 (2014).
- Michael et al., *New Class of Quantum Error-Correcting Codes for a Bosonic Mode*, PRX 6 (2016).

**Existing OSS.**
- Bosonic Qiskit (C2QA / IBM-NQI) -- qumode + qubit hybrid circuits: https://github.com/C2QA/bosonic-qiskit
- Mr Mustard (Xanadu) -- differentiable Gaussian + Fock, phase-space <-> Fock bridge: https://github.com/XanaduAI/MrMustard
- Dynamiqs -- JAX-based GPU Lindblad / SME solvers; used by Alice & Bob for cat-qubit chips: https://github.com/dynamiqs/dynamiqs
- Strawberry Fields (Xanadu, photonic CV).
- `EQuS/bosonic` -- deprecated, now `jaxquantum.circuits` / `jaxquantum.codes` (2025-07-13).
- Piquasso (photonic QC platform, Quantum 2025).

**Newer ref worth tracking.**
- *Bosonic Pauli+*: efficient simulation of concatenated GKP codes, arXiv:2402.09333.
- *Classical simulation of circuits with realistic odd-dimensional GKP states*, arXiv:2412.13136.
- *Fast simulation of bosonic qubits via Gaussian functions in phase space*, PRX Quantum 2 040315 (2021).
- *Universal gate set for GKP logical qubits*, Nat. Phys. (2025).

---

### 7. Lindblad / master-equation + quantum-trajectory simulators

**What it is.** Continuous-time evolution under Lindbladians, optionally unraveled as stochastic quantum trajectories (Monte Carlo wavefunction / quantum jumps).

**Why QEC needs it.** Realistic noise: T1/T2, coherent errors, leakage, crosstalk, cross-resonance dynamics; non-Markovian extensions; studying Pauli-twirl approximation error; modeling syndrome extraction in the analog regime.

**PECOS status.** `pecos-neo` has composable noise channels at the gate/Pauli level, but no continuous-time Lindblad or trajectory solver (TODO: verify).

**Seminal refs.**
- Dalibard, Castin, Molmer (1992) [MCWF].
- Plenio, Knight (1998) [review].
- Modern: Johansson, Nation, Nori, *QuTiP* (2012/2013).

**Existing OSS.**
- QuTiP (Python) -- reference implementation, `mesolve` / `mcsolve`.
- Dynamiqs (JAX, GPU-accelerated, differentiable) -- 30-60x speedup on dissipative cat CNOT: https://github.com/dynamiqs/dynamiqs
- QuantumToolbox.jl (Julia, QuTiP-like syntax, distributed + GPU): https://github.com/qutip/QuantumToolbox.jl -- arXiv:2504.21440.
- C3 (characterization / control / calibration framework).

**Newer ref worth tracking.**
- Lambert et al., *QuantumToolbox.jl* (2025), arXiv:2504.21440.
- *Efficient Lindblad synthesis for noise model construction*, npj QI 11 (2025), arXiv:2502.03462 -- bridges Lindblad sims to Pauli-noise models (useful for feeding DEM pipelines).

---

### 8. Fermion-native simulators (no Jordan-Wigner cost)

**What it is.** Second-quantized fermion operators simulated directly (matrix-product-fermion states, Gaussian fermion + low-rank non-Gaussian, etc.) without qubit mapping overhead.

**Why QEC needs it.** Fermionic codes (Majorana fermion code, Bravyi-Kitaev-style fermionic LDPC), QEC for fermionic simulation itself (fault-tolerant chemistry), tensor-network methods on fermionic Hilbert spaces.

**PECOS status.** Not present.

**Anchor refs.**
- Bravyi, Kitaev, *Fermionic Quantum Computation*, Ann. Phys. 298 (2002).
- Corboz, Vidal, *Fermionic multiscale entanglement renormalization ansatz* (2009).

**Existing OSS.** OpenFermion (operator-level, not fast sim), ITensor fermionic MPS, TeNPy.

---

### 9. Correlated / Pauli-Lindblad noise model simulators

**What it is.** Circuit-level noise with correlated (multi-qubit) Pauli-Lindblad generators, learned from device tomography (IBM's PEC/PEA pipeline). Not pure iid depolarizing.

**Why QEC needs it.** Accurate threshold / pseudo-threshold estimates on real hardware; studying impact of correlations on matching decoders; PEC-assisted QEC experiments.

**PECOS status.** `pecos-neo` supports crosstalk/leakage channels (good start); unclear if Pauli-Lindblad learned models are a first-class input. TODO: verify with `pecos-neo` docs.

**Seminal refs.**
- van den Berg, Minev, Kandala, Temme, *Probabilistic error cancellation with sparse Pauli-Lindblad models*, Nat. Phys. 19 (2023). arXiv:2201.09866.
- Cai et al., *Quantum error mitigation*, RMP (2023).

**Newer refs worth tracking.**
- Chen et al., *Techniques for learning sparse Pauli-Lindblad noise models*, Quantum 8 (2024), DOI 10.22331/q-2024-12-10-1556. arXiv:2311.15408.
- *Efficient Lindblad synthesis for noise model construction*, npj QI (2025), arXiv:2502.03462.
- *Bayesian inference of general noise-model parameters from surface-code syndrome statistics*, arXiv:2406.08981 (couples learned noise -> DEM-style decoder input).

---

### 10. Weak-simulation samplers via quasi-probability / negativity

**What it is.** Sample measurement outcomes of near-Clifford circuits via quasi-probability decompositions (Wigner negativity, Howard-Campbell robustness of magic). Runtime scales with negativity/robustness, not 2^n.

**Why QEC needs it.** Resource-theoretic analysis of magic injection, distillation cost lower bounds, benchmarking distillation protocols.

**PECOS status.** Not present.

**Anchor refs.**
- Pashayan, Wallman, Bartlett, *Estimating outcome probabilities of quantum circuits using quasiprobabilities*, PRL 115 (2015).
- Howard, Campbell, *Application of a Resource Theory for Magic States to Fault-Tolerant Quantum Computing*, PRL 118 (2017).

---

### 11. Fusion-based / measurement-based photonic QEC sims

**What it is.** Discrete-variable photonic FBQC / MBQC: cluster-state construction via fusion gates, loss + dephasing noise, percolation-threshold analysis.

**Why QEC needs it.** PsiQuantum-style FBQC, measurement-based surface codes, photonic interfaces. PECOS `GraphStateSim` has the states but not the fusion/loss sim layer.

**Anchor refs.**
- Bartolucci et al. (PsiQuantum), *Fusion-based quantum computation*, Nat. Commun. 14 (2023).
- Raussendorf, Harrington, Goyal, *A fault-tolerant one-way quantum computer*, Ann. Phys. 321 (2006).

---

## Ranking (grug's revised first-cut opinion)

Given PECOS already has DemSampler, CliffordRz, STN/MAST, pecos-neo:

1. **DemStabSim** (new module; wraps existing DemSampler/influence-map as a first-class backend via `sim()`) -- see proposal below. Build first.
2. **Lindblad / trajectory** (#7) -- only honest way to study coherent / non-Pauli noise; feeds DemStabSim via channel->Pauli-rate lowering (arXiv:2502.03462). Build second.
3. **Pauli-Lindblad correlated noise -> DEM** (#9) -- learned IBM-style noise as a DemStabSim input; small addition once #1 lands.
4. **CliffordRz sparsification** (revised #4) -- confirmed absent; standalone refinement, do when researcher hits the 2^t wall.
5. **Pauli-based computation** (#5) -- matches FTQC resource accounting; pairs with MAST.

**Out of scope.** Bosonic / CV (#6), photonic FBQC (#11) -- PECOS does not work on these areas.

**Rest** (matchgate, decision diagrams, fermion-native, quasiprobability) are narrower or more research-y and not on the near-term path.

---

## Proposal: DemSampler-backed "fast stabilizer + depolarizing" simulator

**Goal.** Expose the existing DemSampler / fault-influence-map machinery as a **first-class simulator** alongside `SparseStab`, `CliffordRz`, `StateVec`, etc. Stim is inspiration only -- PECOS stays self-contained.

**What it gives the user.** A drop-in sim for the most common QEC research workload: Clifford circuit + per-location depolarizing-family noise -> detector + observable + raw-measurement samples at Stim-competitive speeds. Reuses every piece PECOS already has (`DagFaultAnalyzer`, `DemBuilder`, `DemSamplerBuilder`, `NoisySampler`) instead of adding a new algorithm.

### Why this is a real simulator, not just sugar

Stim's core algorithm *is* "Pauli-frame propagation through a Clifford circuit with per-location Pauli noise, aggregated into detector/observable signatures, then sampled shot-wise". That is exactly what PECOS's fault-influence + DemSampler pipeline does today. Wrapping it behind a simulator-shaped API makes the equivalence visible and reusable. Calling it what it is also keeps the story honest: it is a Clifford + Pauli-noise sim, not a general sim.

### Proposed location and name

- Crate: **`pecos-simulators`** (same place as `SparseStab`, `StateVec`).
- Module: `src/dem_stab.rs` (or `fault_influence_sim.rs`).
- Public type: `DemStabSim` (bikeshed: `InfluenceSampler`, `FaultFrameSim`).

### SimBuilder audit (2026-04-11)

Findings from `crates/pecos-engines/src/sim_builder.rs` and `python/pecos-rslib/src/sim.rs`:

**Rust shape.** `SimBuilder` holds four pieces: `classical_builder` (required), `quantum_builder` (default `SparseStabEngine`), `noise_builder` (default `PassThroughNoiseModel`), `config`. Backend registration convention is a free function returning a builder:

```rust
.quantum(sparse_stab())           // IntoQuantumEngineBuilder
.quantum(state_vector())          // same pattern
.noise(DepolarizingNoise { p: 0.001 })   // IntoNoiseModel
```

**`QuantumEngine` trait is streaming.** It's a `process(ByteMessage) -> ByteMessage` interface driven per-tick by the classical engine, with the noise model intercepting `ByteMessage`s before they hit the quantum side. This is a fundamental fit-shape constraint for DemStabSim.

**Python shape.** `sim(program)` dispatches on program type (QASM / QIS / HUGR / PHIR) and each variant carries a `quantum_engine_builder: Option<...>` slot -- same backend-selection pattern.

### Implication: two honest integration paths

Because `QuantumEngine` is streaming and DemStabSim is batch-by-nature, grug sees two options:

**Path A -- Record-and-replay `QuantumEngine` impl (recommended first step).**
`DemStabSimEngine` implements `QuantumEngine` by buffering all incoming `ByteMessage`s into an internal `DagCircuit`. On the first "end of circuit" signal (or lazy on first measurement query), it runs `DagFaultAnalyzer` -> `DemSamplerBuilder::build` once and caches the sampler. Subsequent shots short-circuit via `DemSampler::sample_batch`. Zero orchestrator changes; slots straight into `sim(program).quantum(dem_stab()).noise(...)`.

Hard limitation: **only valid for non-adaptive circuits** (no classical feed-forward affecting gate sequence across shots). That is fine for static syndrome-extraction memory experiments, which is most standard QEC research. DemStabSim must *reject* circuits where the ByteMessage stream depends on mid-circuit measurement outcomes, and clearly redirect users to `sparse_stab()` + `pecos-neo` for adaptive circuits.

**Path B -- Batch-mode fast-path (later).**
Extend `SimBuilder` with a batch-execution branch that, when a batch-capable backend is set, bypasses the per-shot classical loop entirely and hands the whole compiled program to the backend once. More invasive, semantically honest, unlocks GPU batch. Do this only after Path A proves out and numbers justify the orchestrator surgery.

Action items:
- [ ] Confirm how end-of-shot is signalled to `QuantumEngine` today (look for `reset()` / shot-boundary markers in `ByteMessage`).
- [ ] Confirm DemSampler cache can be safely reused across `MonteCarloEngine::run()` calls with fresh RNG (should be yes).
- [ ] Decide rejection mechanics: return `PecosError::Input` on first classical-feedback instruction, or pre-scan once at build.

### Integration with the `sim()` entry point

`sim()` is the main simulation entry on both sides:

- Rust: `crates/pecos-engines/src/sim_builder.rs:418` -- `pub fn sim<I: SimInput>(input: I) -> SimBuilder`.
- Python: `python/pecos-rslib/src/sim.rs:66` -- `pub fn sim(py, program) -> PySimBuilder`.

DemStabSim must be selectable through `sim(circuit).backend(...)` / equivalent, not live as a sidecar API. Concretely:

- Register `DemStabSim` as a backend variant in whatever enum / dispatch `SimBuilder` uses today (check `engine_builder::SimInput` and existing backends like `SparseStab`, `StateVec`).
- `sim(dag).dem_stab().noise(...).detectors(...).sample(n)` reads naturally at both call sites.
- The builder path is the ergonomic home for the noise-model hierarchy and detector/observable definitions.
- Python mirror: `pecos.sim(program).dem_stab().noise(...).sample(n)` via PyO3 bindings in `pecos-rslib`.

**Action item.** Audit `SimBuilder` / `PySimBuilder` to confirm the shape of the backend-selection API before committing to a method name, so DemStabSim slots in next to existing backends consistently.

### Two API shapes (offer both)

#### Shape A -- batch / circuit-at-a-time (primary, honest API)

Takes a fully-specified `DagCircuit` (or `TickCircuit`) + noise model + detector/observable definitions, returns batch shot results. This is the *true* shape of the algorithm; no per-gate illusion.

```rust
let mut sim = DemStabSim::builder()
    .circuit(&dag)
    .noise(DepolarizingModel::uniform(p = 1e-3))  // or PauliLindblad, per-location, ...
    .detectors(&detectors)
    .observables(&observables)
    .build()?;

let shots = sim.sample_batch(num_shots, &mut rng);
// shots: { detector_flips, observable_flips, [optional] raw_measurement_record }
```

Internally: `DagFaultAnalyzer -> DagFaultInfluenceMap -> DemSamplerBuilder -> DemSampler::sample_batch`.

#### Shape B -- `CliffordGateable` facade (compat shim)

A thin record-and-replay wrapper that implements `CliffordGateable` / `QuantumSimulator`. Gate calls append to an internal `DagCircuit`. First measurement / `end_shot` / explicit `.finalize()` triggers one-time influence-map build; subsequent shots reuse the cached analysis.

```rust
let mut sim = DemStabSimFacade::new(n_qubits)
    .with_noise(DepolarizingModel::uniform(1e-3));

sim.h(&[q0]).cx(&[(q0, q1)]).mz(&[q0, q1]);   // records
let result = sim.run_shot(&mut rng);           // builds influence map lazily
```

Trade-off: per-gate method-chain is *not* cheap here (allocates into DAG). Document clearly: "prefer Shape A; Shape B exists only for trait-compatibility with code that assumes a streaming sim".

### Noise model input

Not locked to depolarizing. Start simple, extend via traits:

- `UniformNoiseModel::depolarizing(p)` (already exists).
- `PerLocationNoiseModel { cx: (px, py, pz, pxx, ...), mz: p_flip, idle: (t1, t2 + tick_duration), ... }`.
- `PauliLindbladNoiseModel { generators: &[(support, rate)] }` -- maps learned IBM-style sparse Pauli-Lindblad to per-location effective Pauli rates; covers correlated noise. (See arXiv:2201.09866, 2311.15408.)
- `FromChannelOp(ChannelMatrix)` -- lowers an arbitrary CPTP Pauli-twirled channel to rates; rejects non-Pauli parts with a warning (keeps honest).
- Future: `FromLindblad(LindbladOp, gate_duration)` -- feeds a trajectory/exp-midpoint solver (item #7) to produce the Pauli channel per location.

### Outputs

- Detector flips (`Vec<u32>` per shot or bit-packed `PackedBits`).
- Observable flips.
- Optional: raw measurement record (toggleable via `MemBuilder` / measurement-noise-model path already present).
- Sampling statistics (already exposed via `SamplingStatistics`).
- Circuit-level Pauli error record per shot (useful for decoder dev / syndrome studies) -- TODO: confirm whether `NoisySampler::ShotResult::faults_fired` is exposed or internal.

### Where PECOS should deliberately differ from Stim

Stim is the inspiration; these are places to diverge on purpose:

1. **First-class DAG / `TickCircuit` ingestion** -- no text round-trip, no external IR. PECOS's circuit types are the canonical input.
2. **No text DEM format as the API boundary.** Expose `DetectorErrorModel` (typed Rust) directly; string form is a serialization detail.
3. **Richer noise model hierarchy.** Pauli-Lindblad / channel / Lindblad-derived inputs as above, with clean trait plumbing instead of Stim's circuit-instruction-annotation-only model.
4. **Native hybrid escape hatch.** When the circuit has T / RZ / RX / ... gates outside the Clifford subgroup, either (a) refuse and suggest `CliffordRz` / STN / MAST, or (b) fall back to `CliffordRz`-driven DEM generation for those slices. Not a Stim feature.
5. **GPU path.** `pecos-gpu-sims` already has wgpu; DemSampler sampling is embarrassingly parallel (independent shots, independent mechanisms per shot). A wgpu backend for the batch sampler is natural.
6. **Tighter decoder handoff.** PECOS controls its own decoder stack, so the DEM type can carry extra metadata (detector spacetime coords, hypergraph decomposition hints) without standard-format constraints.

### Implementation plan

1. New module `pecos-simulators/src/dem_stab.rs`.
2. Re-export `DemStabSim` (Shape A) from `pecos-simulators` prelude.
3. Implement `QuantumSimulator` on a facade type `DemStabSimFacade` (Shape B) with internal `DagCircuit` accumulator.
4. Parity tests: Clifford + depolarizing circuit on `SparseStab + pecos-neo noise + Monte Carlo` vs `DemStabSim` (with and without statistical `compare_dems_statistical`) -- distributions must match.
5. Micro-benchmarks vs direct stabilizer Monte Carlo (expect large speedup once shots > ~100).
6. Python bindings through `pecos-rslib` (follow the `DemSamplerBuilder` existing path).
7. Docs: `docs/concepts/dem-stabilizer-simulator.md` explaining the algorithm and the Stim parallel explicitly.

### Open questions

- [ ] Name: `DemStabSim` vs `InfluenceSampler` vs `FaultFrameSim`. (Grug prefers names that say what it is; `DemStabSim` is fine.)
- [ ] Should Shape B exist at all, or is documentation + Shape A enough? (Grug lean: skip Shape B until a concrete consumer asks. Record-and-replay smells.)
- [ ] Per-shot raw measurement record: always on (matches Stim) or opt-in (memory cost)? Default off, opt-in.
- [ ] Seeding semantics when shots run in parallel (`rayon`): use per-shot seed derived from master seed (deterministic, embarrassingly parallel) vs thread-local split.

---

## Build order

**Confirmed build order (2026-04-11):**

1. **DemStabSim first.** Wraps existing infrastructure, slots into `sim()` as a new backend, highest near-term leverage for QEC research. No new algorithmic risk.
2. **Lindblad + quantum-trajectory second.** Closes the device-characterization -> effective-Pauli-channel -> DemStabSim loop. Anchored to `pecos-neo`.

Out of scope for this roadmap (per project direction, 2026-04-11): **photonic / fusion-based**, **GKP / cat / bosonic codes**. Sections #6 and #11 remain documented for completeness but are not proposals.

## Next simulator proposal (after DemStabSim)

Given the current stack, grug recommend **open-system / Lindblad + quantum-trajectory simulator** as the next build, for these reasons:

1. **Bridges device physics to PECOS's existing DEM pipeline.** Efficient-Lindblad-synthesis techniques (arXiv:2502.03462) lower a per-gate Lindbladian to an effective Pauli channel per location, which feeds straight into the `DemStabSim` noise-model input above. This turns every real-device characterization run into a honest PECOS noise model.
2. **Only way to study coherent / non-Pauli errors honestly.** Pauli-twirling assumptions underlie every stabilizer-based sim; a trajectory sim is the validator.
3. **Leverages existing `pecos-neo` scaffolding.** `pecos-neo` already has composable channels and Monte Carlo parallel execution -- a Lindblad / MCWF solver slots in as a new channel-evaluation backend.
4. **Narrow, well-understood scope.** QuTiP, Dynamiqs, QuantumToolbox.jl are mature references; algorithm risk is low, the gain is PECOS-native performance + tight coupling with DEM generation.
5. **Practical leverage right now.** Researchers using PECOS on near-term hardware benchmarks pay a lot for Pauli-twirled approximations when what they actually want is "what does this T1/T2/over-rotation budget imply for my logical error rate". This closes that loop.

Bosonic / CV and photonic FBQC are **out of scope** for the PECOS roadmap and are not candidates here.

Concrete next steps (separate design doc):
- Pick solver family (adaptive RK vs Magnus vs Krylov exp) for mid-size (N <= 10 qubits) Lindbladians.
- Define `LindbladOp` + `TrajectoryResult` types.
- Bridge API: `LindbladBackend::gate_channel(op, duration) -> PauliChannel` for DemStabSim consumption.
- Trajectory mode (MCWF / quantum jumps) for variance-reduced sampling.
- GPU path: start with rayon-parallel trajectories on CPU; wgpu later if cost justifies.

## TODOs before formalizing

- [x] Confirm `PauliProp` does not already extract DEMs -- correction: `pecos-qec::fault_tolerance::dem_builder` already does.
- [x] Audit CliffordRz for sparsification -- none present.
- [ ] Verify `pecos-neo` scope in detail (Pauli-Lindblad learned input format? continuous-time?).
- [ ] Check `STN/MAST` docs on `study/tensor-network-clifford-rz` for overlap with stabilizer-rank sampling (may already cover some of the CliffordRz refinements).
- [ ] Ask maintainer which roadmap items (if any) already claim Lindblad / bosonic / PBC.
- [ ] Add OSS licence notes per wrapper candidate (Dynamiqs: Apache-2.0? Bosonic Qiskit: Apache-2.0? Stim: Apache-2.0; confirm).
