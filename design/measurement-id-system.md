<!-- Ported from the pecos-docs vault (design/measurement-id-system.md,
     written 2026-05-02) so the repo's design records can cite it. The vault
     copy is superseded by this one. -->

# Measurement result identity via MLIR-aligned SSA

## Problem

Measurement results are referenced by position-dependent indices throughout
PECOS. Each component invents its own ordering (tick-sequential, topological,
propagation-walk). Detector definitions use Stim-style negative offsets
that assume a specific ordering. Any reordering breaks the mapping.

This caused 2-7x detection rate errors and 19GB memory explosions in the
2026-05-02 DEM audit.

## Direction: MLIR-aligned SSA values

Rather than inventing a custom ID system, align with MLIR's proven approach:

- Measurement results are **SSA values** — defined once, unique within scope
- Loops use **regions with block arguments** — no PHI nodes, no flattening
- Cross-iteration data flows through **explicit yield** — no ordering conventions
- Identity is structural (region, block, value) — not positional

### MLIR patterns to adopt

**1. Block arguments instead of PHI nodes**

Values flow into blocks as explicit arguments. No magic, no ordering
dependency. In PECOS: measurement results from previous rounds arrive
as block arguments to the current round's region.

**2. Regions as structured control flow**

A loop is an operation that contains a region (the body). The region has
block arguments for the induction variable and loop-carried values. The
region yields values to the next iteration. The loop body is a template
written once — each iteration instantiates it with fresh SSA values.

**3. SSA values as operation results**

Every operation produces SSA values. A measurement is an operation that
produces a result value. The value IS the identity. No separate ID system.

### How it looks for QEC

```
// Structured form — the loop body is a template, not unrolled
%final = qec.syndrome_rounds(%patch, rounds=d) -> region {
    ^body(%iter: index, %prev_measurements: !qec.meas_set):
        %m0 = qec.mz %ancilla_0 : !qec.meas_result
        %m1 = qec.mz %ancilla_1 : !qec.meas_result
        %current = qec.collect(%m0, %m1, ...)

        // Detector: XOR of current and previous iteration
        qec.detector %m0, %prev_measurements[0]

        qec.yield %current   // passes to next iteration as %prev_measurements
}
```

- `%m0` is scoped to the region body. Defined once in the template.
- Each iteration gets fresh SSA values (by the semantics of the loop op).
- Cross-iteration references flow through block arguments and yield.
- No measurement ordering. No negative offsets. No ID mapping.

### Two levels of representation

**Structured (pre-unroll):** SSA values scoped to regions. Detectors are
templates referencing block arguments. The loop body is written once.
This is the program representation.

**Flat (post-unroll):** Each iteration instantiated with unique SSA values.
Detectors are concrete, referencing specific values. This is what the
DEM builder works on. Standard SSA numbering — sequential, unique by
construction.

The unroller transforms structured → flat. No manual ID assignment.
No ordering conventions. Values carry their identity through the
transformation.

### Metadata as a side table

SSA values are lightweight (pointer-sized). Any metadata (qubit, basis,
coordinates, human-readable labels) lives in a side table on the circuit,
not on the values. The hot path (DEM builder, sampler, decoder) works
with values only.

```rust
// Side table, opt-in
circuit.measurement_info: BTreeMap<MeasResult, MeasurementInfo>

struct MeasurementInfo {
    qubit: QubitId,
    basis: Basis,
    label: Option<String>,
    coords: Option<Vec<f64>>,
}
```

## Why MLIR-aligned

MLIR solved exactly our problems:
- Value identity across transformations (lowering, optimization passes)
- Loops without unrolling (scf.for with regions)
- Nested structures (regions within regions)
- Branching (scf.if with region per branch, yielding results)

Their solution is battle-tested across the LLVM ecosystem. Rather than
inventing something custom, we should adopt their patterns:

- Regions for structured control flow
- Block arguments for value passing at boundaries
- Scoped SSA values for identity
- Yield for cross-iteration data flow

This also positions PECOS well for future MLIR integration (quantum
dialects, compilation pipelines) if we ever go that direction.

## What changes in PECOS

### Principle: no assumption of unrolling

The structured form (regions, loops, branches) IS the representation —
not a convenience layer that gets flattened. The DEM builder, simulators,
and analysis tools must work on regions directly. Unrolling is one
possible lowering, not a required step.

### Phase 1: SSA measurement results (immediate)

- Each MZ operation produces a `MeasResult` value (lightweight, Copy)
- Assigned sequentially during circuit construction
- Detector definitions reference `MeasResult` values directly
- Influence map, DEM builder, sampler all use `MeasResult`
- Negative-offset convention becomes import/export format only
- Fixes the current measurement ordering bugs
- Works for both flat and structured circuits

### Phase 2: Regions and structured loops

- Circuit representation supports regions (loop body as sub-circuit)
- Block arguments pass measurement results across iterations
- Yield passes current results to next iteration
- Detectors defined as templates within regions

**DEM builder on regions (not unrolled):**

A fault in the loop body template has a LOCAL effect — it flips some
measurements in the current iteration and possibly adjacent iterations.
The detector template says "XOR current and previous." The DEM entry
is: "this fault at position X flips detector D at iteration offsets
[0, -1]."

The DEM builder for a loop region:
1. Analyze the template body ONCE (which faults flip which measurements)
2. Apply detector templates (which measurement XORs define detectors)
3. Produce ONE DEM entry per fault mechanism in the template
4. Handle boundary detectors (first/last iteration) as special cases

This is O(template_size), not O(template_size × rounds). For a d=100
surface code: same analysis cost as d=3. The 16-distinct-detector-value
observation from the Heisenberg audit confirms this — the template
analysis produces those 16 values directly.

Boundary detectors (first round compares against initialization, last
round includes data readout) are the only iteration-dependent entries.
These are handled at the region entry/exit, not by unrolling.

### Phase 3: Branching

- scf.if-style regions for conditional execution
- Each branch has its own region with fault analysis
- The DEM has branch-conditional entries: "if branch A taken, this fault
  flips these detectors"
- For QEC Pauli frame updates: the branch affects the Pauli frame but
  not the detection events (frame tracking is a classical side-effect)
- For more complex branching: enumerate branch paths up to some depth

### How it fits PECOS's existing architecture

**TickCircuit (current, flat):** Works with Phase 1 MeasResult values.
No change to structure. Existing code continues to work.

**DagCircuit (current, flat):** Same — Phase 1 MeasResult on gates.

**StructuredCircuit (new, Phase 2):** Operations can contain regions.
A `SyndromeRound` op has a region for the body. This is the authoring
format from LogicalCircuitBuilder/Selene. CAN lower to flat TickCircuit
(unrolling), but doesn't have to.

**Selene/HUGR (existing program model):** Already has structured control
flow. The structured circuit is the natural compilation target from
Selene, before any lowering.

**sim_neo (execution):** Can execute structured circuits by iterating
regions. Each iteration produces fresh MeasResult values. The simulator
doesn't need the circuit to be unrolled.

**DEM builder (analysis):** Works on the template body for loop regions.
Produces template DEM entries. Boundary handling at region boundaries.
No unrolling required.

**Backward Heisenberg (analysis):** Template analysis: walk the template
body once, accumulate per-detector interference pattern. The 16-distinct-
value result is the template analysis. Boundary detectors analyzed
separately.

## DEMs for programs with loops and branches

### Can we still compile DEMs?

**Loops (no measurement-dependent branching):** Yes. The loop body is a
fixed circuit template. Each iteration has the same fault mechanisms. The
DEM is the template DEM repeated per iteration, plus boundary terms.
Standard QEC syndrome extraction is this case.

**Branches not depending on measurements:** Yes. The branch is resolved
at compile time or by classical input. Each branch path has its own DEM.
Select the right one at runtime.

**Pauli frame corrections (the common QEC feed-forward):** Yes. The branch
affects the classical Pauli frame, not the physical gates. The circuit
is the same regardless of measurement outcome — only the interpretation
of future measurements changes. The DEM is identical for both branches.
Frame tracking is a decoder-side post-processing step. This is how Stim
handles it.

**Truly adaptive circuits (e.g., T-injection with conditional S gate):**
The DEM is branch-dependent. Options:
1. **Enumerate branches** — for k branch points, 2^k DEMs. Tractable
   for small k (typical: 1-3 per magic state distillation).
2. **Conditional DEM entries** — single DEM with entries conditioned on
   measurement outcomes. Sampler evaluates conditions during sampling.
3. **Streaming/incremental DEM** — extend the DEM at runtime as
   measurements arrive and branches resolve.

### Incremental DEM extension at runtime

For adaptive programs, the DEM grows as the program executes:

```
Round 1: execute syndrome extraction → get measurements
         DEM slice covers round 1 faults (precomputed from template)
         Decoder processes round 1 syndrome

Branch:  measurement outcome determines next operation
         NOW we know which branch was taken
         Select the precomputed DEM slice for that branch arm

Round 2: execute → get measurements
         DEM slice for round 2 (precomputed)
         Decoder processes incrementally
```

Each step, the DEM grows by one "slice" — the faults from the most
recent region. The decoder processes the new slice. Old slices don't
change.

### Program path as execution context

The path through the program IS the execution context for DEM selection.
You don't need individual measurement values — just the sequence of
branch decisions. Two shots taking the same path have the same DEM.

```
path = [loop_iter=0, branch=LEFT, loop_iter=1, branch=RIGHT, ...]
```

For QEC: loop iterations are fixed (d rounds), branches are rare (one
per magic state injection). The path space is small.

### Precomputation and caching

The path representation enables aggressive caching:

1. **Compile time:** For each region template, precompute the DEM slice.
   For each branch, precompute both arms. Store in a table keyed by
   (region_type, branch_decision).

2. **Runtime:** As the program executes, the path accumulates decisions.
   Look up each DEM slice from the precomputed table. Concatenate slices
   to form the full DEM for this path. Feed to decoder.

3. **Cross-shot caching:** Many shots take the same path (especially when
   branches are rare). Cache the full concatenated DEM keyed by the path
   (a small bitvector — one bit per branch point). First shot on a new
   path: assemble from slices. Subsequent shots: cache hit.

For a d=13 surface code + 1 T-injection (2 branches): 2 paths, 2 cached
DEMs. Every shot hits the cache.

For a deep algorithm with k branch points: 2^k paths worst case. But
most QEC branches are Pauli frame corrections (same DEM either way), so
effective path space is much smaller.

### Latency considerations

For superconducting qubits with ~1us cycle times, the DEM slice for the
next round must be available in microseconds:

- **Fixed loops (syndrome rounds):** DEM slice precomputed from template.
  Zero runtime cost. Just index into the table.
- **Branches with 2 arms:** Precompute BOTH slices. Select at runtime
  based on measurement outcome. Zero latency, 2x memory (negligible).
- **Deep branching (rare):** Precompute the tree up to depth k. For
  k=10: 1024 cached slices. Still fits in memory.
- **Unbounded branching (theoretical):** Fall back to on-the-fly DEM
  generation. Higher latency. Only for exotic programs.

### Connection to MLIR region model

The region structure tells you exactly what to precompute:

- Each loop region → one template DEM slice (reused every iteration)
- Each branch region → one DEM slice per arm (select at runtime)
- Nested regions → compose slices hierarchically
- Region boundaries → boundary detector handling

The measurement SSA values are scoped to regions, so each slice's
internal identities are self-contained. The path determines which slices
connect. Block arguments and yield handle cross-slice data flow (previous
iteration's measurements flowing to current iteration's detectors).

### What this means for decoders

The decoder interface becomes:

```rust
trait StreamingDecoder {
    /// Process a new DEM slice (one region's worth of fault mechanisms)
    fn push_slice(&mut self, slice: &DemSlice);

    /// Process new syndrome data for the latest slice
    fn push_syndrome(&mut self, syndrome: &[bool]);

    /// Get current correction estimate
    fn correction(&self) -> PauliFrame;
}
```

This is compatible with windowed decoding architectures (Cain et al.,
Turner et al.) where the decoder processes syndrome data in windows.
Each window corresponds to one region execution. The DEM slice IS the
window's fault model.

## Motivation from audit (2026-05-02)

The root cause of ALL non-EEG DEM accuracy issues was measurement index
mismatch between components. The influence map orders measurements by
backward propagation walk. The TickCircuit orders by tick. Detector
records use negative offsets assuming TickCircuit order. Any transformation
that reorders (gate splitting, DAG construction) breaks the mapping.

With SSA values: impossible by construction. Values carry their identity.
No ordering to mismatch.

## PHIR as the canonical representation

PECOS already has a prototype MLIR-inspired IR: **PHIR** (`pecos-phir`).
It has regions, blocks, block arguments, SSA values, structured control
flow, yield, and a dialect system. The measurement identity problem is
solved by construction if PHIR is the representation everything works on.

Rather than patching TickCircuit and DagCircuit with measurement IDs,
the proper long-term path:

1. **All input formats convert to PHIR** — TickCircuit, Selene/HUGR,
   OpenQASM, Stim circuits all lower to PHIR.

2. **PHIR is the canonical IR** — the DEM builder, backward Heisenberg,
   simulator, and decoder all operate on PHIR. No TickCircuit → DagCircuit
   → InfluenceMap pipeline with ordering mismatches at every boundary.

3. **Measurement results are SSA values in PHIR** — defined by Measure
   ops, scoped to regions, flowed through block arguments and yield.
   Detectors reference SSA values directly. No position-dependent indices
   anywhere in the pipeline.

4. **Loop and branch analysis on PHIR regions** — the DEM builder
   analyzes region templates directly. The backward Heisenberg walks
   region bodies. No unrolling required (but available as a lowering).

### What PHIR already has

From `pecos-phir`:
- `Region` with `SSACFG` and `Graph` kinds
- `Block` with `BlockArgument` (replaces PHI nodes)
- `SSAValue` for all operation results
- `Measure` / `MeasurePauli` ops producing SSA results
- `Branch` / `ConditionalBranch` / `Switch` terminators
- `Yield` for passing values out of regions
- Dialect system for QEC-specific ops (detector, syndrome round, etc.)

### What needs to be added

- **QEC detector definitions** as first-class PHIR operations:
  `qec.detector %m_current, %m_previous` referencing measurement SSA values
- **DEM builder** that operates on PHIR regions (template analysis)
- **Backward Heisenberg** walk on PHIR (region-aware)
- **Lowering passes** from TickCircuit/Selene to PHIR
- **DEM slice extraction** from region templates for streaming decoders

### Migration path

Phase 1 (immediate): fix current bugs with `MeasResult(usize)` on flat
circuits. Minimal change, unblocks correct DEM generation.

Phase 2: build PHIR ← TickCircuit lowering. Run DEM builder on PHIR.
Verify results match the flat pipeline.

Phase 3: add PHIR ← Selene/HUGR lowering. Region-aware DEM analysis.
Template-based DEM for loops. Incremental DEM for branches.

Phase 4: PHIR becomes the primary representation. TickCircuit/DagCircuit
become lowering targets (for hardware backends), not analysis targets.

## References

- [MLIR Language Reference](https://mlir.llvm.org/docs/LangRef/)
- [MLIR SCF Dialect](https://mlir.llvm.org/docs/Dialects/SCFDialect/)
  (scf.for, scf.if, scf.while with regions and yield)
- [MLIR Rationale: Block Arguments vs PHI](https://mlir.llvm.org/docs/Rationale/Rationale/)
- [QIR](https://github.com/microsoft/qdk/wiki/QIR) — adds %Result type
  to LLVM IR for measurement results (similar direction, LLVM-based)
- `non-eeg-dem-sensitivity-audit.md` — the audit that motivated this
- `eeg-heisenberg-approach.md` — EEG expansion also has measurement ordering
