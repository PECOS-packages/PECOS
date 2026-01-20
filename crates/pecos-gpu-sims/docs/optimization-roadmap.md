# GPU Stabilizer Simulation Optimization Roadmap

## Current State (January 2026)

At d=21 surface code (881 qubits, 42 rounds):
- **wgpu GPU**: ~65,000 us/iter (15.75x slower than CPU)
- **DenseStab (CPU)**: ~4,100 us/iter (baseline)
- **GpuStabOpt (CPU)**: ~7,100 us/iter (1.73x slower)

### Profiling Breakdown (single round, d=21)

| Phase | Time | Percentage |
|-------|------|------------|
| Gate flush | 892.8 us | 81.4% |
| GPU dispatch (measurements) | 199.7 us | 18.2% |
| Buffer write | 3.6 us | 0.3% |
| Buffer read | 1.1 us | 0.1% |

**Key insight**: Gate processing dominates. With only 56 threads (one per generator word) processing 1741 gates sequentially, GPU utilization is very low.

---

## Optimization 1: Shot Parallelism (Multi-Instance Batching)

**Impact: High** | **Complexity: Medium** | **Status: IMPLEMENTED**

### Results (d=21, 881 qubits, 42 rounds)

| Shots | GPU us/shot | CPU us/shot | Speedup |
|-------|-------------|-------------|---------|
| 1     | 25,066      | 4,259       | 0.17x   |
| 16    | 7,168       | 4,133       | 0.58x   |
| 64    | 3,664       | 4,091       | 1.12x   |
| 256   | 2,231       | 4,086       | 1.83x   |
| 512   | 2,203       | 4,095       | 1.86x   |

**Key findings:**
- Crossover point: ~64 shots (GPU becomes faster than CPU)
- Maximum speedup: ~1.86x at 256-512 shots (appears to plateau)
- Per-shot cost reduction: 11.4x (from 25,066 us at 1 shot to 2,203 us at 512 shots)
- Buffer size limit: ~600 shots at d=21 due to 128MB max_buffer_binding_size

**Implementation:** `GpuStabMulti` in `gpu_stab_multi.rs`

### Concept

Run N independent stabilizer simulations in parallel, each with its own tableau but processing the same circuit. This is ideal for Monte Carlo sampling where we need many independent shots.

### Current vs Proposed

| Metric | Current | With 64 shots |
|--------|---------|---------------|
| Threads | 56 | 3,584 |
| GPU utilization | ~1% | ~60% |
| Memory | ~200KB | ~13MB |

### Implementation Approach

1. Allocate tableau arrays with extra dimension for shot index
2. Modify shader to index by `shot_id * tableau_size + original_index`
3. Each thread processes one (shot, word) pair
4. All shots process the same gate queue simultaneously

### Memory Layout Options

**Option A: Shot-major (better for gate processing)**
```
stab_x[shot][qubit][word] -> stab_x[shot * num_qubits * gen_words + qubit * gen_words + word]
```

**Option B: Word-major (better for coalescing)**
```
stab_x[word][shot][qubit] -> stab_x[word * num_shots * num_qubits + shot * num_qubits + qubit]
```

### Expected Speedup

For N shots, amortize fixed overhead across all shots. If gate processing is 80% of time:
- Single shot: 1000 us
- 64 shots naive: 64,000 us
- 64 shots parallel: ~2,000-5,000 us (estimated 12-30x speedup per shot)

### API Design

```rust
// Create multi-shot simulator
let mut sim = GpuStabMulti::new(num_qubits, num_shots)?;

// Apply gates (same circuit to all shots)
sim.h(&qubits);
sim.cx(&[control, target]);

// Measure (returns results for all shots)
let results: Vec<Vec<bool>> = sim.mz(&qubits);
```

---

## Optimization 2: Gate-Parallel Processing

**Impact: NONE (NEGATIVE)** | **Complexity: High** | **Status: IMPLEMENTED, BENCHMARKED - NOT BENEFICIAL**

### Implementation Summary

Implemented gate-parallel processing that partitions gates into independent batches (no shared qubits)
and processes each batch with more parallelism (num_gates * gen_words threads instead of just gen_words).

**Thread count comparison at d=21:**
- Original: 56 threads (gen_words)
- Gate-parallel with 100-gate batch: 5,600 threads
- Gate-parallel with full round (~400 independent gates): 22,400 threads

**Implementation:**
- Created `stab_gate_shader_parallel.wgsl` with atomic sign updates
- Added `partition_gates_into_batches()` function
- Added `enable_parallel()` / `disable_parallel()` methods to GpuStab
- Uses atomicXor for concurrent sign updates

### Benchmark Results (NEGATIVE)

| d | Qubits | Sequential us/round | Parallel us/round | Speedup |
|---|--------|---------------------|-------------------|---------|
| 5 | 49 | 50.5 | 2616.6 | **0.02x** |
| 7 | 97 | 56.2 | 5066.6 | **0.01x** |
| 11 | 241 | 96.7 | 12893.7 | **0.01x** |
| 15 | 449 | 249.0 | 25142.5 | **0.01x** |

### Analysis - Why It's Slower

**The fundamental problem**: Gate partitioning creates multiple batches, but we cannot write
multiple batches to the same GPU buffer without synchronization. This requires:
1. Write batch to buffer
2. Dispatch compute shader
3. **Wait for GPU to complete** before writing next batch
4. Repeat for each batch

This per-batch synchronization overhead (submitting, polling, waiting) completely dominates the
potential parallelism gains. The sequential approach does just ONE write and ONE dispatch per sync.

**Possible fixes (not worth implementing):**
- Use multiple gate queue buffers (one per batch) - significant memory overhead
- Use staging buffers with GPU-side copies - complex, still has per-batch dispatch overhead
- Process ALL gates in single batch - only possible if all gates are independent (rare)

### Conclusion

Gate-parallel processing is **NOT BENEFICIAL** for this use case because:
1. Surface code circuits have many gate dependencies (CX gates create batches)
2. Per-batch synchronization overhead is 50-100x worse than sequential
3. Shot parallelism (GpuStabMulti) already provides 1.86x speedup for Monte Carlo

**Recommendation:** Use shot parallelism (GpuStabMulti) instead. It achieves speedup without
the synchronization overhead since all shots process the same gate queue.

### Concept

Instead of one thread per generator word, use one thread per qubit. Each thread processes all gates affecting its qubit.

### Current vs Proposed

| Metric | Current | Qubit-parallel |
|--------|---------|----------------|
| Threads | 56 | 881 |
| Work per thread | All gates, one word | Gates on one qubit, all words |

### Challenges

1. **Two-qubit gates**: CX touches two qubits, need synchronization
2. **Sign updates**: Signs are per-generator, not per-qubit
3. **Memory access**: Would access non-contiguous words

### Implementation Approach

**Multi-pass strategy:**
1. Pass 1: Process single-qubit gates (fully parallel)
2. Pass 2: Process CX gates with atomics or careful ordering

**Alternative: Dependency graph**
- Build graph of gate dependencies
- Process independent gates in parallel
- Barrier between dependent groups

### Expected Speedup

Modest improvement (2-4x) due to synchronization overhead for two-qubit gates.

---

## Optimization 3: Circuit Compilation / JIT Shader

**Impact: LOW (negative for large circuits)** | **Complexity: High** | **Status: IMPLEMENTED, BENCHMARKED**

### Concept

Generate specialized WGSL shaders for specific circuits, eliminating loop and switch overhead.

### Benchmark Results

| d | Qubits | Gates/round | Compile Time | Compiled | Dynamic | Speedup |
|---|--------|-------------|--------------|----------|---------|---------|
| 3 | 17 | 31 | 3.6 ms | 35.1 us | 44.3 us | 1.26x |
| 5 | 49 | 93 | 10.4 ms | 41.2 us | 43.7 us | 1.06x |
| 7 | 97 | 187 | 24.7 ms | 66.1 us | 56.3 us | **0.85x** |
| 11 | 241 | 471 | 90 ms | 280.2 us | 95.1 us | **0.34x** |
| 15 | 449 | 883 | 241 ms | 743.7 us | 249.7 us | **0.34x** |
| 21 | 881 | 1741 | 797 ms | (very slow) | - | **<0.3x** |

### Analysis

**Compiled circuits are SLOWER for circuits with >100 gates** because:

1. **Large shader size**: The generated WGSL has inlined code for every gate, causing:
   - Instruction cache misses
   - Longer shader compilation time
   - More GPU register pressure

2. **Dynamic approach has better cache behavior**: The tight loop + switch statement
   fits in cache and processes gates efficiently

3. **Compile overhead not amortized**: Even with caching, the initial compile time
   (10-800ms) dominates for practical use cases

### When to Use

- **Use compiled circuits**: Only for very small circuits (<100 gates) that will be
  executed thousands of times
- **Use dynamic gate queue**: For all practical QEC workloads (d>=5)

### Implementation

Files created:
- `circuit_compiler.rs` - JIT shader generation with caching
- Integration in `GpuStab::compile_circuit()` and `execute_compiled()`

### Current Approach

```wgsl
for (var i = 0u; i < num_gates; i++) {
    let gate = decode_gate(queue[i]);
    switch (gate.type) {
        case GATE_H: { ... }
        case GATE_CX: { ... }
        // ...
    }
}
```

### Compiled Approach

```wgsl
// Generated shader for specific circuit
fn process_circuit(word_idx: u32) {
    // Gate 0: H on qubit 5
    let offset_0 = 5u * gen_words + word_idx;
    let x0 = stab_x[offset_0];
    let z0 = stab_z[offset_0];
    stab_x[offset_0] = z0;
    stab_z[offset_0] = x0;
    local_sign_minus ^= (x0 & z0);

    // Gate 1: CX on qubits 3, 7
    let ctrl_1 = 3u * gen_words + word_idx;
    let tgt_1 = 7u * gen_words + word_idx;
    // ... inlined CX code
}
```

### Benefits

- No loop overhead
- No switch statement
- Compiler can optimize memory access patterns
- Constants inlined

### Implementation Approach

1. Create `CircuitCompiler` that generates WGSL from gate list
2. Cache compiled shaders by circuit hash
3. Fall back to generic shader for one-off circuits

### When to Use

- Repeated circuits (syndrome extraction rounds)
- Circuits with regular structure
- Not worth it for single-use circuits (compilation overhead)

---

## Optimization 4: Memory Layout Optimization

**Impact: Low-Medium** | **Complexity: Low** | **Status: ANALYZED**

### Analysis

Created `stab_gate_shader_interleaved.wgsl` with X/Z pairs adjacent in memory:
```
stab[qubit * gen_words * 2 + word_idx * 2 + 0] = X
stab[qubit * gen_words * 2 + word_idx * 2 + 1] = Z
```

**Potential benefits:**
- H, S, SDG gates read both X and Z for same qubit - would be in same cache line
- Reduces buffers from 4 to 2

**Limitations:**
- CX gates access different qubits (ctrl and tgt), so X/Z adjacency doesn't help much
- Surface code is CX-dominated (~95% CX gates)
- Expected improvement: 10-20% at best

**Conclusion:** Memory layout is not the primary bottleneck. The real issue is:
- Only 56 threads (gen_words) at d=21
- Each thread processes ALL 1741 gates sequentially
- Low GPU utilization, high per-thread work

**Recommendation:** Skip this optimization, focus on qubit-parallel processing instead.

### Concept

Optimize memory layout for better cache utilization and coalesced access.

### Current Layout

Separate arrays for each tableau component:
```
stab_x:    [q0w0, q0w1, ..., q1w0, q1w1, ...]
stab_z:    [q0w0, q0w1, ..., q1w0, q1w1, ...]
destab_x:  [q0w0, q0w1, ..., q1w0, q1w1, ...]
destab_z:  [q0w0, q0w1, ..., q1w0, q1w1, ...]
```

### Option A: Interleaved X/Z

```
tableau: [q0w0_x, q0w0_z, q0w1_x, q0w1_z, ..., q1w0_x, q1w0_z, ...]
```

Benefits:
- X and Z for same qubit/word in same cache line
- Most gates read both X and Z

### Option B: Structure of Arrays per Qubit

```
qubit_data[qubit] = { stab_x[], stab_z[], destab_x[], destab_z[] }
```

### Option C: Transposed (Generator-major)

```
stab_x: [g0q0, g0q1, ..., g1q0, g1q1, ...]
```

Benefits:
- Better for operations that scan generators
- Worse for gate operations

### Measurement

Need to benchmark each layout with realistic workloads.

---

## Optimization 5: Async Pipeline / Double Buffering

**Impact: Low** | **Complexity: Medium** | **Status: ANALYZED - NOT BENEFICIAL**

### Concept

Overlap CPU circuit construction with GPU execution using double-buffered command queues.

### Current Flow

```
CPU: [build round 1] [wait] [build round 2] [wait] ...
GPU:                 [exec round 1]         [exec round 2] ...
```

### Pipelined Flow

```
CPU: [build round 1] [build round 2] [build round 3] ...
GPU:                 [exec round 1]  [exec round 2]  ...
```

### Analysis

**Already implemented:** GpuStab has batch mode (`begin_batch()` / `end_batch()`) that
accumulates command buffers and submits them all at once, reducing CPU-GPU sync overhead.

**Profiling shows minimal benefit:**
- Buffer writes: 0.3% of total time
- Gate flush (GPU): 81.4% of total time
- GPU processing is the bottleneck, not CPU-GPU transfer

**Blocking constraints:**
- Measurement results are needed before next round in most use cases
- QEC syndrome extraction requires measurement outcomes to continue

**Conclusion:** Double buffering the gate queue would provide <1% improvement.
The batch mode already provides the main benefit by reducing submission overhead.
Further async pipelining is not worthwhile given the constraints.

---

## Optimization 6: Subgroup Operations

**Impact: Low-Medium** | **Complexity: Medium** | **Status: IMPLEMENTED, BLOCKED BY WGPU**

### Concept

Use WGSL subgroup operations for reductions (e.g., finding first anticommuting generator).

### Current Approach

Reduction operations use multiple passes or CPU readback.

### With Subgroups

```wgsl
let any_anticommuting = subgroupOr(is_anticommuting);
let first_anticommuting = subgroupBallotFindLsb(subgroupBallot(is_anticommuting));
```

### Implementation Status

**Fully implemented** in `stab_subgroup_shader.wgsl`:
- `find_anticommuting_subgroup` - O(1) parallel search for first anticommuting generator
- `measurement_xor_subgroup` - Parallel row multiplication with subgroup reductions
- `compute_deterministic_outcome_subgroup` - Parallel phase accumulation

**Infrastructure exists** in `gpu_stab.rs`:
- Feature detection for subgroups
- Fallback to sequential implementation
- Pipeline and bind group setup

**Blocked by wgpu/Naga:**
- The `enable subgroups;` WGSL directive is not yet supported in Naga
- Hardcoded `has_subgroups = false` until Naga adds support
- When support arrives, change line ~588 in `gpu_stab.rs` to enable

### Expected Impact When Enabled

- Measurement operation speedup: ~2-4x for large qubit counts
- Most benefit at d>30 where many generators must be searched
- Limited impact on overall simulation since gates dominate

---

## Priority Order (Updated)

1. **Shot Parallelism** - IMPLEMENTED, highest impact for Monte Carlo (1.86x at 512 shots)
2. ~~**Gate-Parallel Processing**~~ - IMPLEMENTED, **NOT BENEFICIAL** (50-100x slower due to sync overhead)
3. **Subgroup Operations** - IMPLEMENTED, blocked by wgpu/Naga
4. **Memory Layout** - ANALYZED, marginal benefit for CX-dominated workloads
5. ~~**Circuit Compilation**~~ - IMPLEMENTED, **NOT BENEFICIAL** for d>=7 (slower than dynamic!)
6. **Async Pipeline** - ANALYZED, not beneficial given current constraints

---

## Achieved Results

| Target | Before | After | Status |
|--------|--------|-------|--------|
| Single shot d=21 | 617x slower | 15.75x slower | Baseline optimizations |
| 64 shots d=21 | N/A | 1.12x faster (GPU wins!) | Shot parallelism |
| 256 shots d=21 | N/A | 1.83x faster | Shot parallelism |
| 512 shots d=21 | N/A | 1.86x faster | Shot parallelism (plateau) |

**Key achievements:**
- Multi-shot GPU simulator (`GpuStabMulti`) with adaptive batching
- Automatic GPU limit detection (queries `max_storage_buffer_binding_size`)
- Crossover point at ~64 shots where GPU becomes faster than CPU
- ~11x reduction in per-shot cost (25,000 us -> 2,200 us) at 512 shots

**Files created:**
- `gpu_stab_multi.rs` - Multi-shot simulator with adaptive batching
- `stab_gate_shader_multi.wgsl` - Multi-shot gate processing shader
- `stab_gate_shader_interleaved.wgsl` - Interleaved memory layout (not integrated)
- `stab_gate_shader_parallel.wgsl` - Gate-parallel processing (integrated but not beneficial)
- `circuit_compiler.rs` - JIT shader compilation for specific gate sequences (not beneficial)
- `stab_subgroup_shader.wgsl` - Subgroup-based measurement (blocked by wgpu)

---

## Remaining Opportunities

1. **Higher shot counts** with batched execution
   - Currently limited by 128MB buffer size (~600 shots at d=21)
   - Could implement multi-batch execution for 1000+ shots
   - **Priority: Low** - 512 shots already provides 1.86x speedup

2. **Subgroup operations** when wgpu/Naga adds support
   - Implementation ready, just needs enable flag changed
   - Monitor wgpu releases for `enable subgroups;` directive support
   - **Priority: Low** - measurements are not the bottleneck

## Optimizations Found NOT Beneficial

1. ~~**Gate-parallel processing**~~ - **NOT RECOMMENDED**
   - Implemented and benchmarked: 50-100x SLOWER than sequential
   - Per-batch synchronization overhead dominates any parallelism gains
   - Use shot parallelism (GpuStabMulti) instead for speedup

2. ~~**Circuit compilation**~~ - **NOT RECOMMENDED**
   - Benchmark shows compiled circuits are 3x SLOWER at d>=11
   - Dynamic gate queue has better cache behavior for large circuits
   - Only useful for tiny circuits (<100 gates) executed thousands of times

3. ~~**Memory layout optimization**~~ - **NOT RECOMMENDED**
   - Analyzed: marginal benefit (10-20% at best)
   - CX gates access different qubits, so X/Z adjacency doesn't help much
   - Not worth the implementation complexity

4. ~~**Async pipeline / double buffering**~~ - **NOT RECOMMENDED**
   - Buffer writes are only 0.3% of total time
   - Batch mode already provides the main benefit
   - Measurement results needed before next round in most use cases
