# Parallelism in Zlup: Design Notes

> **Note:** Zlup is an experimental toy language for exploring quantum programming
> language design. These notes are exploratory design discussions, not specifications
> for a production system.

This document explores how Zlup's scope-aware allocator tracking could enable
automatic parallelization of quantum programs.

## Design Philosophy: Parallelism Through Constraints

**Zlup provides parallelism without threads, locks, or explicit parallel syntax.**

But this isn't "implicit" parallelism where the compiler magically figures things out.
It's **constraint-based** parallelism: the type system explicitly constrains what
functions can access, and parallelism follows directly from those constraints.

### The Core Principle

> **Expressiveness through constraints, not features.**

Instead of adding `@parallel` annotations or `spawn` keywords, we constrain the
language so that parallelism is **obvious and verifiable** from the code structure:

| Constraint | What It Enables |
|------------|-----------------|
| No allocator param → can't touch qubits | Classical functions are trivially parallel with quantum |
| Allocators are owned, not aliased | Different names = different qubits = independent |
| Scopes define lifetimes | Explicit sync points without barrier syntax |
| Static allocation only | All parallelism decidable at compile time |

This aligns with NASA Power of 10: don't add features to enable analysis—remove
possibilities until analysis is trivial.

### Why Not Threads or Parallel Annotations?

Threads and `@parallel` annotations solve problems Zlup doesn't have:

| Traditional Problem | Zlup's Constraint-Based Solution |
|---------------------|----------------------------------|
| Shared mutable state | Allocators are owned, not shared |
| Data races | No aliasing - different names = different resources |
| What can parallelize? | Read the type signature |
| Where are sync points? | Scope boundaries |

Adding explicit parallelism syntax would:

1. **Duplicate information** already in the type system
2. **Add complexity** without enabling anything new
3. **Violate "explicit through constraints"** by adding yet another mechanism

### What the Programmer Writes vs. What Executes

```zlup
// Programmer writes sequential code:
h q[0];
h q[1];
h q[2];
cx (q[0], q[1]);
cx (q[1], q[2]);
```

```
// Compiler sees dependency graph:
[h q[0]] ──────┐
               ├──► [cx (q[0],q[1])] ──┐
[h q[1]] ──────┘                       ├──► [cx (q[1],q[2])]
                                       │
[h q[2]] ─────────────────────────────►┘

// Execution can parallelize independent operations:
Layer 1: h q[0], h q[1], h q[2]    (all parallel)
Layer 2: cx (q[0], q[1])
Layer 3: cx (q[1], q[2])
```

No annotations needed. The syntax already contains all the information.

## The Core Insight

Zlup tracks which qubit allocators are accessible in each scope. This information,
combined with the NASA Power of 10 constraints (no dynamic allocation, bounded loops,
no recursion), enables powerful static analysis for parallelism detection.

```zlup
fn example() -> unit {
    mut q1 := qalloc(2);  // Allocator q1 enters scope
    mut q2 := qalloc(2);  // Allocator q2 enters scope

    // These blocks operate on disjoint qubit sets
    {
        h q1[0];          // Only touches q1
        cx (q1[0], q1[1]);
    }
    {
        h q2[0];          // Only touches q2
        cx (q2[0], q2[1]);
    }
}
```

The compiler knows statically that these two blocks are independent and can execute
in parallel on hardware that supports it.

## Dependency Analysis

### What the Compiler Knows

At compile time, Zlup has complete information about:

1. **Allocator lifetimes** - When each allocator enters and exits scope
2. **Allocator sizes** - Fixed at allocation time (no dynamic sizing)
3. **Operation targets** - Which allocator(s) each gate/measurement touches
4. **Control flow structure** - Bounded loops, no recursion

This enables construction of a precise **operation dependency graph**.

### Dependency Rules

Two operations are **independent** (parallelizable) if:

1. They operate on disjoint qubit sets, AND
2. Neither reads a classical variable written by the other, AND
3. No control flow dependency exists between them

Two operations are **dependent** (must be ordered) if:

1. They share at least one qubit target (read-after-write, write-after-write), OR
2. One reads a classical variable written by the other (data dependency), OR
3. One is control-dependent on the other (branch/loop)

### Example: Building the Dependency Graph

```zlup
fn grover_iteration(mut q: [4]qubit) -> unit {
    // Operation 1: Oracle (all qubits)
    oracle q;

    // Operation 2-5: Diffusion operator
    h q[0]; h q[1]; h q[2]; h q[3];  // Ops 2,3,4,5 - independent of each other

    x q[0]; x q[1]; x q[2]; x q[3];  // Ops 6,7,8,9 - independent of each other

    // Multi-controlled Z
    mcz q;                            // Op 10 - depends on 6,7,8,9

    x q[0]; x q[1]; x q[2]; x q[3];  // Ops 11,12,13,14
    h q[0]; h q[1]; h q[2]; h q[3];  // Ops 15,16,17,18
}
```

Dependency graph (simplified):
```
    [1:oracle]
        |
   +----+----+----+
   v    v    v    v
 [2:h] [3:h] [4:h] [5:h]   <- Parallel layer
   |    |    |    |
   v    v    v    v
 [6:x] [7:x] [8:x] [9:x]   <- Parallel layer
   |    |    |    |
   +----+----+----+
        v
    [10:mcz]
        |
   ... (continues)
```

## Levels of Parallelism

### 1. Gate-Level Parallelism

Gates on independent qubits within the same allocator can execute simultaneously:

```zlup
mut q := qalloc(4);
h q[0];  // Can run in parallel with...
h q[1];  // ...this gate
h q[2];  // ...and this one
h q[3];  // ...and this one
```

This is the finest granularity. Most quantum hardware naturally supports this.

### 2. Block-Level Parallelism

Independent code blocks operating on disjoint allocators:

```zlup
mut ancilla := qalloc(2);
mut data := qalloc(4);

// These blocks could execute on different QPU regions
{
    // Prepare ancilla
    h ancilla[0];
    cx (ancilla[0], ancilla[1]);
}
{
    // Prepare data register (independent)
    for i in 0..4 {
        h data[i];
    }
}
// Sync point: both blocks must complete before continuing
```

### Scopes as Implicit Barriers

Scopes naturally provide synchronization points when you need them:

```zlup
{
    // Region 1: everything here completes...
    h q1[0];
    cx (q1[0], q1[1]);
}
// ...before this point
{
    // Region 2: can depend on Region 1's results
    if some_classical_result {
        ...
    }
}
```

But if you *don't* want a barrier, just don't use a scope:

```zlup
h q1[0];
h q2[0];
// No barrier - compiler free to interleave
cx (q1[0], q1[1]);
cx (q2[0], q2[1]);
// q1 and q2 operations form independent chains
```

This gives programmers control over synchronization *when they need it*, without
requiring explicit barrier syntax for the common case (maximum parallelism).

### 3. Classical-Quantum Parallelism

Classical computation can proceed while quantum operations execute, as long as
no data dependencies exist:

```zlup
mut q := qalloc(4);
classical_prep := expensive_classical_computation();  // Could overlap with qalloc

h q[0];
cx (q[0], q[1]);

// Measurement creates a sync point
m := mz([4]u1) q;

// Classical post-processing can start immediately
processed := process_results(m);
```

### 4. Inter-Function Parallelism

With whole-program analysis, independent function calls could parallelize:

```zlup
fn main() -> unit {
    mut q1 := qalloc(2);
    mut q2 := qalloc(2);

    // These function calls are independent
    prepare_bell(q1);   // Only touches q1
    prepare_ghz(q2);    // Only touches q2
}
```

## Measurements and Sync Points

Measurements introduce **implicit synchronization** because:

1. The quantum state must be fully evolved before measurement
2. Classical variables holding results must be available before use

```zlup
mut q := qalloc(2);
h q[0];
cx (q[0], q[1]);

// SYNC POINT: All operations on q must complete
m := mz([2]u1) q;

// Classical code can now use m
if m[0] == 1 {
    // Conditional logic based on measurement
}
```

### Mid-Circuit Measurement Considerations

Mid-circuit measurement creates **partial sync points**:

```zlup
mut q := qalloc(4);
h q[0]; h q[1]; h q[2]; h q[3];

// Only q[0] needs to sync here
m0 := mz([1]u1) q[0..1];

// These can continue in parallel with processing m0
h q[1]; h q[2]; h q[3];

// Classical processing of m0 can happen concurrently
if m0[0] == 1 {
    // This branch doesn't touch q[1..4]
    log("measured 1");
}
```

## QEC and Classical Processing

Quantum Error Correction (QEC) is a critical use case that stress-tests the ownership
model. QEC involves tight classical-quantum interaction with real-time constraints.

### The QEC Data Flow

```zlup
fn qec_round(mut data: [n]qubit, mut ancilla: [m]qubit) -> unit {
    // 1. Syndrome extraction (quantum operations)
    extract_stabilizers(data, ancilla);

    // 2. Measure ancilla → classical syndrome
    syndrome := mz([m]u1) ancilla;

    // 3. Decode (pure classical computation)
    corrections := decode(syndrome);

    // 4. Apply corrections (quantum operations)
    apply_corrections(data, corrections);

    // 5. Reset ancilla for next round
    reset ancilla;
}
```

### What Works Well

The ownership model handles QEC's core patterns naturally:

| QEC Pattern | How Ownership Helps |
|-------------|---------------------|
| Syndrome arrays | Values after measurement - no aliasing, pass freely |
| Decoder parallelism | Pure functions on values - multiple decoders can run in parallel |
| Clear data flow | Explicit: qubits → measurement → classical → corrections → qubits |
| No race conditions | Syndrome is a value, not shared mutable state |

```zlup
// Parallel decoders work naturally - syndrome is a value
fn fault_tolerant_decode(syndrome: [m]u1) -> Corrections {
    c1 := decoder_mwpm(syndrome);   // These three calls
    c2 := decoder_uf(syndrome);     // are independent
    c3 := decoder_nn(syndrome);     // (same input value)
    return vote(c1, c2, c3);
}
```

### Challenging Patterns

Some QEC patterns require more thought:

#### Syndrome History (Sliding Window Decoders)

MWPM, Union-Find, and other decoders often need multiple rounds of syndrome data:

```zlup
// Option A: Pass history as a value (copied)
fn decode_with_history(
    current: [m]u1,
    history: [[m]u1; k]  // Copied - potentially expensive
) -> Corrections { ... }

// Option B: Borrow history (reference)
fn decode_with_history(
    current: [m]u1,
    history: &[[m]u1; k]  // Borrowed - no copy, but adds reference semantics
) -> Corrections { ... }

// Option C: Decoder owns accumulating state
struct Decoder {
    history: [[m]u1; k],
    // ... decoder state
}
fn decode(self: &mut Decoder, current: [m]u1) -> Corrections {
    self.history.push(current);
    // ... decode using history
}
```

#### Pipelined QEC

Real QEC systems pipeline: decode round N while extracting round N+1.

```zlup
fn pipelined_qec(mut data: [n]qubit, mut ancilla: [m]qubit) {
    syndrome_prev := extract_and_measure(data, ancilla);

    for round in 1..num_rounds {
        // GOAL: These should overlap:
        // - decode(syndrome_prev)     → pure classical
        // - extract_stabilizers(...)  → quantum on ancilla

        // But in sequential code, how does the compiler know?
        extract_stabilizers(data, ancilla);       // Quantum
        corrections := decode(syndrome_prev);      // Classical
        syndrome_curr := mz([m]u1) ancilla;
        apply_corrections(data, corrections);
        syndrome_prev = syndrome_curr;
    }
}
```

### Approaches to Classical-Quantum Overlap

Three approaches for expressing that classical and quantum operations can overlap:

#### Approach A: Implicit Inference

The compiler analyzes data flow and infers what can overlap.

**Problem:** This is implicit, not explicit. The programmer writes code without knowing
what will parallelize. This conflicts with Zlup's "explicit over implicit" philosophy.

#### Approach B: Explicit Annotations

Add syntax like `@parallel` or `overlap { }` blocks.

**Problem:** This adds complexity. The programmer must learn new syntax and manually
annotate parallelism that the type system already expresses.

#### Approach C: Structural Separation

Separate classical and quantum into distinct "lanes" with explicit channels.

**Problem:** Major language addition. Overkill for simple cases. Introduces concurrency
primitives (channels, send/recv) that we're trying to avoid.

### Our Recommendation: Constraint-Based (No New Syntax)

None of the above. Instead, recognize that **the type system already expresses parallelism**.

The key insight, aligned with NASA Power of 10 Rule 6 (minimize scope):

> **If a function doesn't take allocator parameters, it cannot touch qubits.**
> This is explicit, enforced by the compiler, and requires no annotation.

```zlup
// EXPLICIT: "I take classical values, I return classical values, I cannot touch qubits"
fn decode(syndrome: [m]u1) -> Corrections { ... }

// EXPLICIT: "I require mutable qubit access"
fn extract_stabilizers(mut data: [n]qubit, mut ancilla: [m]qubit) -> unit { ... }
```

The function signature IS the parallelism declaration:

| Signature | What It Says | Parallelizable With |
|-----------|--------------|---------------------|
| `fn f(x: int) -> int` | Pure classical | Any quantum op |
| `fn f(syndrome: [m]u1) -> Corrections` | Classical only | Any quantum op |
| `fn f(mut q: [n]qubit) -> unit` | Needs qubit access | Only disjoint allocators |
| `fn f(mut q1: [n]qubit, mut q2: [m]qubit)` | Needs both | Nothing touching q1 or q2 |

This is:
- **Explicit**: The constraint is visible in the type signature
- **Simple**: No new syntax, no annotations
- **Constraint-based**: Expressiveness through what you CAN'T do

### Why This Aligns with NASA Power of 10

| Power of 10 Rule | How It Applies |
|------------------|----------------|
| Rule 3: No dynamic allocation | Allocators are static → known at compile time |
| Rule 6: Minimize scope | Narrow allocator scope → clear parallelism boundaries |
| Rule 9: Limit pointer use | No aliasing → if `q1` and `q2` are different names, they're different allocators |

The Power of 10 philosophy: **Don't add features to enable analysis. Remove possibilities until analysis is trivial.**

We don't add `@parallel`. We constrain functions so parallelism is obvious:
- No allocator param? Can't touch qubits. Done.
- Different allocator params? Different qubits. Done.

### Scopes as Explicit Sync Points

If you need ordering, use a scope:

```zlup
{
    // Everything in this scope completes...
    extract_stabilizers(data, ancilla);
    syndrome := mz([m]u1) ancilla;
}
// ...before this point

corrections := decode(syndrome);  // Can overlap with next scope

{
    apply_corrections(data, corrections);
}
```

Scopes are explicit barriers. No scope = no enforced ordering. This is already in the language.

### Why This Works for QEC

The constraint-based model is perfect for QEC because:

1. **Decoders are pure classical by construction**

```zlup
fn decode(syndrome: [m]u1) -> Corrections {
    // No allocator parameters → CANNOT touch qubits
    // This isn't a promise - it's enforced by the type system
}
```

2. **The classical/quantum boundary is explicit and verified**

```zlup
fn qec_round(mut data: [n]qubit, mut ancilla: [m]qubit) -> unit {
    extract_stabilizers(data, ancilla);  // Signature says: needs qubits
    syndrome := mz([m]u1) ancilla;       // Measurement: quantum → classical
    corrections := decode(syndrome);      // Signature says: classical only
    apply_corrections(data, corrections); // Signature says: needs qubits
}
```

3. **Parallelism is obvious from signatures**

Looking at just the types:
- `decode: [m]u1 -> Corrections` — classical, parallelizable with any quantum
- `extract_stabilizers: (mut [n]qubit, mut [m]qubit) -> unit` — needs both allocators

No analysis needed. No annotations. The constraints tell you everything.

4. **NASA Power of 10 compliance comes free**

The same constraints that enable parallelism also enforce safety:
- Static allocators (Rule 3)
- Minimal scope (Rule 6)
- No aliasing (Rule 9)

## Realistic QEC Examples

Let's work through concrete QEC scenarios to verify the constraint-based model works.

### Example 1: Repetition Code (Simplest Case)

A 3-qubit repetition code with 2 syndrome qubits:

```zlup
// Data qubits: |ψ⟩ encoded as |ψψψ⟩
// Syndrome qubits: measure ZZ stabilizers

fn repetition_round(mut data: [3]qubit, mut syndrome: [2]qubit) -> [2]u1 {
    // Syndrome extraction: ZZ stabilizers
    cx (data[0], syndrome[0]);
    cx (data[1], syndrome[0]);
    cx (data[1], syndrome[1]);
    cx (data[2], syndrome[1]);

    // Measure syndrome qubits
    s := mz([2]u1) syndrome;

    // Reset for next round
    reset syndrome;

    return s;
}

// Decoder: pure classical, no allocator params
fn decode_repetition(s: [2]u1) -> [3]u1 {
    // Simple majority voting
    // s[0] = data[0] ⊕ data[1]
    // s[1] = data[1] ⊕ data[2]
    mut corrections := [0, 0, 0];
    if s[0] == 1 && s[1] == 0 {
        corrections[0] = 1;  // Error on qubit 0
    } else if s[0] == 1 && s[1] == 1 {
        corrections[1] = 1;  // Error on qubit 1
    } else if s[0] == 0 && s[1] == 1 {
        corrections[2] = 1;  // Error on qubit 2
    }
    return corrections;
}

fn apply_corrections(mut data: [3]qubit, corrections: [3]u1) -> unit {
    for i in 0..3 {
        if corrections[i] == 1 {
            x data[i];
        }
    }
}

fn qec_cycle(mut data: [3]qubit, mut syndrome: [2]qubit) -> unit {
    s := repetition_round(data, syndrome);  // Quantum: touches data, syndrome
    corrections := decode_repetition(s);     // Classical: no allocators
    apply_corrections(data, corrections);    // Quantum: touches data
}
```

**Parallelism analysis from signatures:**
- `repetition_round`: needs `data` and `syndrome` → serialized with anything touching those
- `decode_repetition`: takes `[2]u1`, returns `[3]u1` → pure classical → can overlap
- `apply_corrections`: needs `data` → must wait for decode, serialized with data ops

### Example 2: Surface Code Patch

A distance-3 surface code with X and Z stabilizers:

```zlup
// Surface code layout (distance 3):
//   D0 -- Z0 -- D1
//   |           |
//   X0    D4    X1
//   |           |
//   D2 -- Z1 -- D3

struct SurfaceCodePatch {
    data: [5]qubit,      // 5 data qubits
    x_ancilla: [2]qubit, // 2 X stabilizer ancillas
    z_ancilla: [2]qubit, // 2 Z stabilizer ancillas
}

fn extract_x_stabilizers(
    mut data: [5]qubit,
    mut x_ancilla: [2]qubit
) -> unit {
    // X0 stabilizer: X on D0, D2, D4
    h x_ancilla[0];
    cx (x_ancilla[0], data[0]);
    cx (x_ancilla[0], data[2]);
    cx (x_ancilla[0], data[4]);
    h x_ancilla[0];

    // X1 stabilizer: X on D1, D3, D4
    h x_ancilla[1];
    cx (x_ancilla[1], data[1]);
    cx (x_ancilla[1], data[3]);
    cx (x_ancilla[1], data[4]);
    h x_ancilla[1];
}

fn extract_z_stabilizers(
    mut data: [5]qubit,
    mut z_ancilla: [2]qubit
) -> unit {
    // Z0 stabilizer: Z on D0, D1, D4
    cx (data[0], z_ancilla[0]);
    cx (data[1], z_ancilla[0]);
    cx (data[4], z_ancilla[0]);

    // Z1 stabilizer: Z on D2, D3, D4
    cx (data[2], z_ancilla[1]);
    cx (data[3], z_ancilla[1]);
    cx (data[4], z_ancilla[1]);
}

fn surface_code_round(mut patch: SurfaceCodePatch) -> SurfaceSyndrome {
    // X and Z extraction both need data qubits
    // But they use different ancilla allocators
    extract_x_stabilizers(patch.data, patch.x_ancilla);
    extract_z_stabilizers(patch.data, patch.z_ancilla);

    // Measure all ancillas
    x_syndrome := mz([2]u1) patch.x_ancilla;
    z_syndrome := mz([2]u1) patch.z_ancilla;

    reset patch.x_ancilla;
    reset patch.z_ancilla;

    return SurfaceSyndrome { x: x_syndrome, z: z_syndrome };
}
```

**Key observation:** X and Z stabilizer extraction both touch `data`, so they serialize.
But within each extraction, gates on different ancilla qubits can parallelize.

### Example 3: Pipelined QEC with Decoder

The real challenge: overlapping decode(round N) with extract(round N+1):

```zlup
fn pipelined_qec(
    mut data: [5]qubit,
    mut ancilla: [4]qubit,
    num_rounds: int
) -> unit {
    // First round: no previous syndrome to decode
    syndrome_prev := extract_and_measure(data, ancilla);

    for round in 1..num_rounds {
        // KEY INSIGHT: These have different signatures
        //
        // extract_and_measure: (mut [5]qubit, mut [4]qubit) -> Syndrome
        //                      ^^^^^^^^^^^^  ^^^^^^^^^^^^^
        //                      needs qubit allocators
        //
        // decode: (Syndrome, SyndromeHistory) -> Corrections
        //         ^^^^^^^^^^^^^^^^^^^^^^^^^
        //         pure classical values, NO allocators

        // The compiler sees:
        // - extract touches {data, ancilla}
        // - decode touches {} (no allocators)
        // Therefore: decode can overlap with extract

        {
            // Scope groups the quantum operations
            extract_stabilizers(data, ancilla);
            syndrome_curr := mz([4]u1) ancilla;
            reset ancilla;
        }

        // decode runs on previous syndrome - can overlap with scope above
        corrections := decode_surface(syndrome_prev);

        // apply must wait for both decode AND current extraction
        apply_corrections(data, corrections);

        syndrome_prev = syndrome_curr;
    }

    // Final decode and correct
    corrections := decode_surface(syndrome_prev);
    apply_corrections(data, corrections);
}
```

**The constraint-based parallelism:**

```
Timeline:
          Round 1           |        Round 2           |
  extract(data,ancilla)     |  extract(data,ancilla)   |
          |                 |          |               |
          v                 |          v               |
  syndrome_1                |  syndrome_2              |
          |                 |          |               |
          +---- decode(s0) -+          +--- decode(s1) +
                   |                          |
                   v                          v
            corrections_1              corrections_2
                   |                          |
          apply(data, c1)             apply(data, c2)
```

Decode overlaps with extract because their signatures prove they're independent.

### Example 4: Multi-Patch Logical Operations

Multiple surface code patches with independent QEC:

```zlup
struct LogicalQubit {
    patch: SurfaceCodePatch,
    // ... other metadata
}

fn parallel_qec_rounds(
    mut logical_qubits: [n]LogicalQubit
) -> unit {
    // Each logical qubit has its own allocators
    // They're completely independent

    for i in 0..n {
        // These iterations touch disjoint allocators
        // Compiler can parallelize across logical qubits
        qec_round(logical_qubits[i].patch);
    }
}

fn transversal_cnot(
    mut control: LogicalQubit,
    mut target: LogicalQubit
) -> unit {
    // Transversal gate: independent physical CNOTs
    for i in 0..5 {
        // Each CNOT is on different physical qubits
        // within the same logical operation
        cx (control.patch.data[i], target.patch.data[i]);
    }
}
```

**Parallelism from structure:**
- Different `LogicalQubit` instances have different allocators → independent
- Loop iterations on independent allocators → parallelizable
- Within transversal gate: different qubit indices → parallelizable

### Example 5: Decoder with Syndrome History

Realistic decoders need history. How does ownership handle this?

```zlup
// Option A: Decoder owns its history (stateful decoder)
struct MWPMDecoder {
    history: [[4]u1; window_size],
    window_pos: int,
    // ... matching graph, etc.
}

impl MWPMDecoder {
    // Decoder is a classical object with classical methods
    // No allocator params anywhere → all classical
    fn decode(self: &mut MWPMDecoder, current: [4]u1) -> Corrections {
        // Add to history
        self.history[self.window_pos] = current;
        self.window_pos = (self.window_pos + 1) % window_size;

        // Run MWPM on history window
        // ... matching algorithm ...

        return corrections;
    }
}

fn qec_with_stateful_decoder(
    mut data: [5]qubit,
    mut ancilla: [4]qubit,
    decoder: &mut MWPMDecoder,  // Classical reference, no qubits
    num_rounds: int
) -> unit {
    for round in 0..num_rounds {
        syndrome := extract_and_measure(data, ancilla);

        // decoder.decode takes classical values only
        // Signature: (&mut MWPMDecoder, [4]u1) -> Corrections
        // No qubit allocators → can overlap with quantum
        corrections := decoder.decode(syndrome);

        apply_corrections(data, corrections);
    }
}
```

**Key insight:** The decoder struct contains classical data only. Its methods
take `&mut self` (classical reference) and classical values. No allocator
parameters → provably doesn't touch qubits → safe to overlap.

### What These Examples Demonstrate

1. **Simple cases work naturally** — repetition code shows basic pattern
2. **Complex extraction parallelizes** — within stabilizer extraction, independent gates parallelize
3. **Pipelining works** — decode/extract overlap follows from signatures
4. **Multi-patch parallelizes** — different logical qubits are independent
5. **Stateful decoders work** — classical state doesn't affect qubit access

All parallelism follows from reading the type signatures. No annotations needed.

## Compiler Passes for Parallelism

### Pass 1: Allocator Scope Analysis

Build a map of allocator lifetimes and which scopes can access them:

```
allocator "q1": defined at line 5, scope depth 1, size 4
  accessible in: main (lines 5-50), helper (lines 20-30)
allocator "q2": defined at line 10, scope depth 1, size 2
  accessible in: main (lines 10-50)
```

### Pass 2: Operation Tagging

Tag each operation with the allocators it touches:

```
h q1[0]          -> touches: {q1}
cx (q1[0], q2[0]) -> touches: {q1, q2}
m := mz q1       -> touches: {q1}, defines: {m}
```

### Pass 3: Dependency Graph Construction

Build edges between dependent operations:

```rust
struct DepGraph {
    nodes: Vec<Operation>,
    edges: Vec<(OpId, OpId, DepKind)>,  // (from, to, kind)
}

enum DepKind {
    QubitDep(String),      // Shared qubit allocator
    DataDep(String),       // Shared classical variable
    ControlDep,            // Control flow
}
```

### Pass 4: Parallelism Detection

Find maximal independent sets (parallel layers):

```rust
fn find_parallel_layers(graph: &DepGraph) -> Vec<Vec<OpId>> {
    // Topological sort with level assignment
    // Operations at the same level are parallelizable
}
```

### Pass 5: Schedule Generation

Generate a schedule respecting dependencies while maximizing parallelism:

```
Layer 0: [qalloc q1, qalloc q2]
Layer 1: [h q1[0], h q2[0]]           <- parallel
Layer 2: [cx (q1[0],q1[1]), cx (q2[0],q2[1])]  <- parallel
Layer 3: [mz q1]                       <- sync point for q1
Layer 4: [mz q2, classical_op(m1)]    <- q2 measure + classical parallel
```

## Hardware Considerations

### CAN vs SHOULD Parallelize

The constraint-based model tells us what **CAN** parallelize. But hardware
constraints determine what **SHOULD** parallelize.

```
Constraints (language level):     Scheduling (compiler level):
─────────────────────────────     ────────────────────────────
"These operations are             "Given hardware limits,
 independent"                      what's the best schedule?"
     │                                      │
     ▼                                      ▼
Type signatures                   Architecture + Noise model
Allocator ownership               Control system parallelism
Scope boundaries                  Decoherence times
```

The language expresses independence. The compiler decides how to exploit it.

### Limited Parallel Control

Some hardware architectures have constrained parallelism due to:

- Limited control resources (only N operations simultaneously)
- Shared control channels across qubit subsets
- Control electronics bottlenecks

In these cases, **serializing a QEC gadget may be better** than spreading
resources across parallel operations:

```
Option A: Parallelize across patches (spread resources)
────────────────────────────────────────────────────────
Patch 1: ░░░ extract ░░░░░░░░░░░░░░░░░░░░
Patch 2: ░░░ extract ░░░░░░░░░░░░░░░░░░░░
         ^^^
         Shared control resources → both run slow

Option B: Serialize patches (concentrate resources)
────────────────────────────────────────────────────────
Patch 1: ▓▓ extract ▓▓
Patch 2:              ▓▓ extract ▓▓
                      ^^^
                      Full resources → each runs fast
```

For QEC with decoherence pressure, Option B may win: finish each syndrome
extraction quickly rather than have both running slowly.

### What the Language Expresses vs What the Compiler Decides

| Concern | Language (Constraints) | Compiler (Scheduling) |
|---------|------------------------|----------------------|
| Independence | Type signatures | — |
| Ordering requirements | Scope boundaries | — |
| Hardware parallelism limits | — | Architecture model |
| Decoherence optimization | — | Noise model |
| Resource allocation | — | Control system model |

The language should express **what's possible**. The compiler should decide
**what's optimal** given the target hardware.

### Philosophy: Start Simple, Add Knobs If Needed

Our approach:

1. **Language stays simple** — constraints express independence, nothing more
2. **Compiler has architecture knowledge** — scheduling decisions are backend-specific
3. **No premature optimization knobs** — don't add hints until we know we need them

If the compiler has enough information (architecture + noise model), it should
make good scheduling decisions without programmer hints. If not, we can add
optional annotations later:

```zlup
// Hypothetical future annotation (only if needed):
@schedule_hint(serialize)  // "Run this gadget fast, don't spread resources"
fn syndrome_extraction(mut data: [n]qubit, mut ancilla: [m]qubit) -> [m]u1 {
    // ...
}
```

But we should try to avoid this. The constraint-based model gives the compiler
freedom to schedule optimally. Adding hints constrains that freedom and burdens
the programmer.

**Principle:** Let the compiler be smart. Add knobs only when proven necessary.

### Connectivity Constraints

Not all qubit pairs can interact directly. The compiler must:

1. Respect hardware topology (coupling map)
2. Insert SWAP gates for non-adjacent interactions
3. Re-analyze parallelism after routing

```
Logical:  cx (q[0], q[3])   <- May not be directly connected
Physical: swap q[1], q[2]; cx (q[0], q[1]); swap q[1], q[2]
```

### Execution Zones

Some architectures have independent execution zones:

```
Zone A: qubits 0-15
Zone B: qubits 16-31

Operations in different zones are naturally parallel.
Allocator assignment can optimize for zone boundaries.
```

### Timing Constraints

Real hardware has gate timing constraints:

- Different gates take different times
- Idle qubits may decohere
- Some parallelism is limited by control electronics

The scheduler must balance parallelism against timing.

## Connection to NASA Power of 10

The Power of 10 rules enable parallelism analysis:

| Rule | How It Enables Parallelism |
|------|---------------------------|
| No unbounded loops | Loop iterations are finite → can unroll for analysis |
| No recursion | Call graph is a DAG → simpler interprocedural analysis |
| No dynamic allocation | All allocators known at compile time |
| Static dispatch | No runtime polymorphism to complicate analysis |
| Bounded complexity | Functions small enough for precise analysis |

Without these constraints, parallelism analysis would require:
- Runtime profiling
- Conservative assumptions
- Complex pointer/alias analysis

With them, we can do everything statically.

## Design Decisions (Resolved)

### Do we need explicit parallelism syntax?

**No.** Parallelism is expressed through constraints already in the type system:

- Function signature shows allocator access → shows what qubits it touches
- No allocator param = pure classical = parallel with any quantum op
- Different allocator params = different qubits = independent

Adding `@parallel` would duplicate information already in the types.

### Do we need explicit barriers?

**No.** Scopes are explicit barriers:

```zlup
{
    // This scope is a barrier
    do_stuff();
}
// Everything above completes before here
```

No scope = no enforced ordering = maximum parallelism.

This is "explicit through constraints": the presence or absence of a scope
explicitly controls synchronization. No new syntax needed.

### Is this "implicit" parallelism?

**No.** It's constraint-based parallelism. The parallelism isn't hidden or
inferred—it's directly visible in the type signatures and scope structure.

```zlup
fn decode(s: [m]u1) -> Corrections  // ← This signature EXPLICITLY says "classical only"
```

The compiler doesn't guess. It reads the constraints you wrote.

## Open Questions

1. **Compiler architecture knowledge**: What information does the compiler need
   about the target hardware? (Parallelism limits, noise model, connectivity)

2. **Scheduling heuristics**: For constrained hardware, when should the compiler
   serialize vs parallelize? (Decoherence vs. resource contention trade-off)

3. **Do we need scheduling hints?**: Can the compiler make good decisions with
   just architecture knowledge, or will we need programmer hints? (Start without,
   add if proven necessary)

4. **Allocator placement**: How should allocators be assigned to physical
   qubits to maximize parallelism while respecting connectivity?

5. **Syndrome history management**: What's the best pattern for decoders that
   need sliding windows? (Value copy vs. borrowing vs. stateful decoder)

6. **Real-time verification**: Should we add optional annotations to verify
   timing constraints? (e.g., decoder must complete before next extraction)

7. **Cross-backend portability**: If we add scheduling hints, can they be
   portable across architectures or must they be backend-specific?

## Future Work

- [x] Implement allocator scope tracking in Zlup semantic analysis
  - Done: `zlup::analysis::AllocatorAnalysis`
- [x] Add dependency graph construction pass
  - Done: `zlup::analysis::DependencyGraph`
- [x] Prototype parallel layer extraction
  - Done: `zlup::analysis::DependencyGraph::parallel_layers()`
- [x] Integrate analysis into compilation pipeline (CLI flags)
  - Done: `zlup analyze` and `guppy-zlup analyze` commands
  - Done: `--analyze` flag on compile commands
- [ ] Benchmark on representative quantum algorithms
- [ ] Explore integration with PECOS simulator for validation

## Why This Works for Quantum

Implicit parallelism is particularly well-suited for quantum computing:

1. **Quantum operations are naturally pure** - A gate transforms its target qubits
   and nothing else. No hidden state, no side effects.

2. **Qubit identity is explicit** - Unlike classical memory (where pointers can alias),
   qubits are always explicitly named. `q[0]` is unambiguously qubit 0 of allocator q.

3. **Independence is common** - Quantum algorithms frequently apply the same operation
   to many qubits (Hadamard on all qubits, etc.). These are trivially parallel.

4. **Hardware wants parallelism** - Quantum hardware can naturally execute independent
   gates simultaneously. The programming model should expose this, not hide it.

5. **Coherence time pressure** - Faster execution = less decoherence. The compiler
   should maximize parallelism automatically, not rely on programmer hints.

## Summary

Zlup is an experimental language exploring constraint-based parallelism:

- **No threads** — ownership model makes them unnecessary
- **No locks** — no shared mutable state to protect
- **No `@parallel`** — type signatures already express parallelizability
- **No barrier syntax** — scopes are explicit sync points

The key insight: **the function signature IS the parallelism declaration.**

```zlup
fn decode(syndrome: [m]u1) -> Corrections    // No allocators → classical only → parallel with any quantum
fn extract(mut q: [n]qubit) -> unit          // Has allocator → needs qubit access → serialized with same allocator
```

This is:
- **Explicit** — constraints are visible in the type signature
- **Simple** — no new syntax to learn
- **Verifiable** — compiler enforces constraints, parallelism follows

**For QEC specifically:**

- Decoder signatures prove they can't touch qubits → safe overlap with extraction
- Classical/quantum boundary is explicit and compiler-verified
- Pipelining emerges from constraints, no concurrency primitives needed

**Separation of concerns:**

- **Language** (constraints) → expresses what CAN parallelize
- **Compiler** (scheduling) → decides what SHOULD parallelize given hardware

The language stays simple. Architecture-specific optimization lives in the compiler.

**Aligned with NASA Power of 10:**

Don't add features to enable analysis. Remove possibilities until analysis is trivial.
The same constraints that make Zlup safe make parallelism obvious.

## References

- NASA Power of 10 Rules: https://spinroot.com/gerard/pdf/P10.pdf
- Quantum circuit scheduling: [various papers on ASAP/ALAP scheduling]
- QISKIT transpiler passes: routing, optimization, scheduling
