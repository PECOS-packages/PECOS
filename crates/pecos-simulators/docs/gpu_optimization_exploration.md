# GPU Stabilizer Simulation Optimization Exploration

## Current State

We have two CPU-optimized stabilizer simulators designed with GPU-friendly memory layouts:

- **GpuStab**: Column-only storage, u32 word packing, warp-aligned
- **GpuStabOpt**: Dual row+column storage for better measurement performance

Benchmark results (d=17 surface code, 34 rounds, 577 qubits):
```
DenseStab:    1664 us (1.00x baseline)
SparseStab:   2297 us (1.38x)
GpuStabOpt:   2672 us (1.61x)
GpuStab:      9525 us (5.72x)
```

## Research Findings

### STABSim (arxiv:2507.03092)
First GPU stabilizer simulator to outperform CPU for QEC workloads.

Key techniques:
1. **Row-based threading**: Each thread handles one generator row
2. **Warp primitives**: `__shfl_down_sync` for fast measurement reduction
3. **Two-stage measurement**: Warp reduction -> block reduction -> grid atomic
4. **Global memory only**: Avoids shared memory for multi-qubit gates
5. **1024 threads/block**: Maximizes warp reduction efficiency

### NVIDIA cuStabilizer
Different paradigm: Pauli frame simulation.

- Tracks differences from noise-free simulation
- K independent frames simulated in parallel
- Bit table encoding (X, Z bits per qubit)
- Efficient for QEC with known noise-free reference

### cuStateVec Techniques
State vector ideas that could apply:

- Gate fusion to reduce memory transfers
- Bit manipulation for index computation
- Staggered multi-gate parallelism

## Optimization Opportunities

### 1. Row-Based Threading Model

**Current**: Operations iterate through columns/qubits sequentially
**Proposed**: Each GPU thread handles one generator row

Benefits:
- Clifford gates embarrassingly parallel across 2n generators
- Natural mapping to GPU thread hierarchy
- Eliminates sequential loops for gate application

Implementation:
```
// Pseudocode for H gate on qubit q
kernel h_gate(tableau, qubit q):
    row = thread_id  // Each thread = one generator

    x_bit = get_bit(tableau.x[row], q)
    z_bit = get_bit(tableau.z[row], q)

    // H: X <-> Z, phase += X*Z
    set_bit(tableau.x[row], q, z_bit)
    set_bit(tableau.z[row], q, x_bit)

    if x_bit and z_bit:
        flip_phase(tableau.phase[row])
```

### 2. Workgroup Reduction for Measurement

**Current**: Sequential iteration to find anticommuting generators
**Proposed**: Parallel reduction using subgroup operations

In wgpu/WGSL:
```wgsl
// Find first generator anticommuting with Z_q measurement
var found_idx: u32 = 0xFFFFFFFF;
let anticommutes = (x_bits[thread_id] >> qubit) & 1u;

// Subgroup ballot to find any anticommuting generator
let ballot = subgroupBallot(anticommutes != 0u);
if ballot.x != 0u {
    found_idx = firstTrailingBit(ballot.x);
}
```

### 3. Batched Frame Simulation

For QEC workloads, simulate K independent Pauli frames in parallel:

```
// Memory layout: [frame_0_gen_0, frame_1_gen_0, ..., frame_K_gen_0, frame_0_gen_1, ...]
// Or: [frame_0_all_gens, frame_1_all_gens, ...]

// Each workgroup handles one frame
// Within workgroup, threads handle generators
```

Benefits:
- Massive parallelism for Monte Carlo sampling
- Memory coalescing when frames are interleaved
- Natural fit for QEC error sampling

### 4. Gate Fusion

Fuse consecutive single-qubit Clifford gates:

```rust
// Instead of: H(q) then S(q) then H(q)
// Compute: fused_gate = H * S * H and apply once

// For Cliffords, this is a lookup table
// 24 single-qubit Cliffords, so 24x24x24 = 13824 combinations
// But only 24 unique results, so just need composition table
```

### 5. Two-Qubit Gate Parallelism

For CX/CZ/SWAP on disjoint qubit pairs, apply all in parallel:

```
// Given: CX(0,1), CX(2,3), CX(4,5) in same tick
// All can execute simultaneously since qubits don't overlap
// Each thread still handles one generator
// Multiple gates just means more bit operations per thread
```

## Exploration Plan

### Phase 1: Measurement Optimization (CPU)
1. Profile current measurement to confirm it's the bottleneck
2. Implement parallel-friendly measurement algorithm in GpuStabOpt
3. Use rayon for CPU parallelism as a stepping stone

### Phase 2: Row-Based CPU Implementation
1. Create GpuStabRowThreaded variant
2. Each "logical thread" handles one generator
3. Use rayon to parallelize across generators
4. Benchmark against current implementations

### Phase 3: wgpu Prototype
1. Simple compute shader for Clifford gates
2. Implement measurement with subgroup operations
3. Benchmark GPU vs CPU implementations

### Phase 4: Batched Frame Simulation
1. Implement Pauli frame tracking
2. Batch K frames for parallel execution
3. Target QEC error sampling workloads

## Files to Create/Modify

1. `gpu_stab_parallel.rs` - Row-based parallel implementation
2. `gpu_stab_wgpu.rs` - Actual wgpu compute shader implementation
3. `pauli_frame_sim.rs` - Batched Pauli frame simulation
4. Benchmarks comparing all approaches

## Success Metrics

- 10x+ speedup on measurement-heavy QEC circuits
- Competitive with Stim for single-shot simulation
- Significant speedup for batched frame simulation

---

## Exploration Results

### Benchmark Results (Surface Code, 2*d rounds)

| Simulator | d=5 (49q) | d=9 (161q) | d=13 (337q) | d=17 (577q) |
|-----------|-----------|------------|-------------|-------------|
| DenseStab | 10.0 us (1.0x) | 92.1 us (1.0x) | 539.9 us (1.0x) | 1673 us (1.0x) |
| SparseStab | 22.3 us (2.2x) | 157.5 us (1.7x) | 713.5 us (1.3x) | 2169 us (1.3x) |
| GpuStabOpt | 22.1 us (2.2x) | 182.7 us (2.0x) | 786.7 us (1.5x) | 2702 us (1.6x) |
| GpuStab | 50.3 us (5.0x) | 481.9 us (5.2x) | 2682.9 us (5.0x) | 9557 us (5.7x) |
| GpuStabParallel | 79.8 us (8.0x) | 1289 us (14.0x) | 7187 us (13.3x) | 27010 us (16.1x) |

### Key Findings

1. **DenseStab wins on CPU**: Uses 64-bit words with vectorized operations. Word-level parallelism beats row-level on single-threaded CPU.

2. **GpuStabOpt is best GPU-style implementation on CPU**: Dual row+column storage provides good balance for gates (column view) and measurement (row view).

3. **GpuStabParallel is slow on CPU but designed for GPU**: The row-based threading model is inefficient on CPU because:
   - Iterates over generators (n iterations) instead of words (n/64 iterations)
   - Per-bit operations with function calls vs word-level bit manipulation
   - No parallel execution to amortize the iteration overhead

4. **Column-only (GpuStab) has slow measurement**: O(n^2) measurement algorithm due to needing to multiply generators across columns.

### Why Row-Based Works on GPU

STABSim shows row-based threading achieves speedup on GPU because:
- Each of n generators = 1 GPU thread (thousands of threads in parallel)
- Per-bit operations become single instructions in CUDA/wgpu shader
- Warp-level primitives enable fast measurement reduction
- Memory coalescing when threads access adjacent rows

On CPU, you need either:
- Word-level operations (DenseStab approach) for SIMD efficiency
- Explicit parallelism (rayon) to amortize row-based overhead

### Recommended GPU Implementation Path

1. **Start with GpuStabParallel memory layout** - row-major, u32 words
2. **Implement basic wgpu compute shaders**:
   - Single-qubit gate kernel (all rows in parallel)
   - Two-qubit gate kernel
   - Measurement kernel with subgroup reduction
3. **Profile and optimize**:
   - Ensure memory coalescing
   - Use subgroup operations for measurement
   - Consider batched frame simulation for Monte Carlo

---

## Existing wgpu Implementation Analysis (pecos-gpu-sims)

### Architecture Overview

The `pecos-gpu-sims` crate contains a working wgpu implementation with the following structure:

**Memory Layout (Transposed/Qubit-Major):**
```
stab_x[qubit * gen_words + word_idx]   - X bits for generators on this qubit
stab_z[qubit * gen_words + word_idx]   - Z bits for generators on this qubit
destab_x[qubit * gen_words + word_idx] - X bits for destabilizers on this qubit
destab_z[qubit * gen_words + word_idx] - Z bits for destabilizers on this qubit
```

- Rows = qubits, Columns = generators (packed into u32 words)
- Each word contains bits for generators `[word_idx*32, (word_idx+1)*32)`

**Threading Model:**
- Gate shader: Each thread handles one `word_idx` (column of generators)
- Processes all queued gates sequentially within each thread
- 256 threads per workgroup

### Shader Files

1. **`stab_gate_shader.wgsl`** - Persistent kernel gate processor
   - Processes gate queue in single dispatch
   - Sign caching optimization (read once, write once)
   - Handles all Clifford gates: H, S, Sdg, X, Y, Z, CX, CZ, SWAP

2. **`stab_shaders.wgsl`** - Individual gate shaders + measurement
   - Separate kernel per gate type (legacy/alternative approach)
   - Multi-stage measurement pipeline:
     1. `measurement_compute_weights` - Find anticommuting generators
     2. `measurement_extract_chosen` - Select minimum weight
     3. `measurement_xor_rows` - XOR into anticommuting stabilizers
     4. `measurement_xor_destabs` - Update destabilizers
     5. `measurement_finalize` - Replace with Z_q

### Current Strengths

1. **Sign caching**: Gate shader caches `sign_minus` and `sign_i` in local variables
2. **Gate queue batching**: Multiple gates processed in single dispatch
3. **Gate sorting**: Single-qubit gates sorted by target for cache locality

### Optimization Opportunities

1. **Measurement reduction**: Current `find_anticommuting` uses per-generator checks
   - Could use `subgroupBallot` to find first anticommuting in O(1)
   - Would require enabling subgroup features in wgpu

2. **Row-based alternative**: For large systems, row-based threading might help
   - Each thread handles one generator (row) instead of one word (column)
   - Better for STABSim-style workloads with many measurements

3. **Atomic reduction**: Current measurement uses multi-stage atomics
   - Could use subgroup reductions to minimize global atomics

4. **Memory coalescing**: Current transposed layout good for gates
   - But measurement accesses are scattered across qubits

### Implementation Plan

**Phase 1: Subgroup Operations (High Impact)**
- Add subgroup-based `find_anticommuting` kernel
- Use `subgroupBallot` + `firstTrailingBit` for O(1) search
- Requires `wgpu::Features::SUBGROUP` (available on most GPUs)

**Phase 2: Row-Based Variant**
- Create alternative shader with row-based threading
- Each thread = one generator, processes qubit bits sequentially
- Compare performance vs column-based for different workloads

**Phase 3: Batched Frame Simulation**
- Add Pauli frame simulation mode
- K independent frames in parallel for Monte Carlo sampling
- Memory layout: interleaved frames for coalescing

---

## Implementation Progress

### Subgroup Operations (Phase 1) - Infrastructure Ready

**Status:** Code written, disabled pending Naga support

Created `stab_subgroup_shader.wgsl` with:
- `find_anticommuting_subgroup` - Uses `subgroupBallot` + `atomicMin` for O(1) parallel search
- `measurement_xor_subgroup` - Parallel row multiplication with subgroup reductions
- `compute_deterministic_outcome_subgroup` - Parallel phase accumulation

Updated `gpu_stab.rs` with:
- Subgroup feature detection and graceful fallback
- `find_first_anticommuting_subgroup()` method using subgroup ballot
- Infrastructure to switch between column-based and subgroup-based measurement

**Blocker:** Naga (wgpu's WGSL compiler) doesn't support `enable subgroups;` yet.
See: https://github.com/gfx-rs/wgpu/issues/5555

When Naga adds support, enable by changing line ~501 in `gpu_stab.rs`:
```rust
let has_subgroups = false; // -> has_subgroups
```

### Row-Based Threading Analysis (Phase 2)

**Status:** Not applicable with current memory layout

**Key Insight:** The STABSim row-based threading model requires a different memory layout.

Current memory layout (qubit-major/transposed):
```
stab_x[qubit * gen_words + word_idx]
       ^^^^^                ^^^^^^^^
       row                  column (generators packed into words)
```

The column-based threading (one thread per `word_idx`) is correct for this layout because:
- Each thread exclusively owns all bits in its word (generators 32*word_idx to 32*(word_idx+1))
- No race conditions between threads
- Good memory coalescing when threads access adjacent words

Row-based threading would cause race conditions:
- Multiple threads (handling different generators in the same word) would try to modify the same u32
- WGSL doesn't have atomic bit operations

**Alternative for row-based:** Would require generator-major memory layout:
```
stab_x[gen_idx * qubit_words + qubit_word]
       ^^^^^^^                 ^^^^^^^^^^
       row (thread)            column (qubits packed into words)
```

This would enable true row-based threading but requires a complete memory layout redesign.
The current column-based approach is optimal for the existing qubit-major layout.

### Remaining Optimization Opportunities

1. **Gate fusion on CPU** - Before sending to GPU, fuse consecutive single-qubit gates on same qubit
2. **Measurement batching** - Batch measurement commands to reduce sync overhead
3. **Parallel qubit layers** - Multiple gates on disjoint qubits can run in same dispatch
4. **Memory layout variants** - Consider generator-major layout for measurement-heavy workloads

### Latest Benchmark Results (2025-01-18)

CPU implementations on surface code (2*d rounds):

| Simulator | d=5 (49q) | d=9 (161q) | d=13 (337q) | d=17 (577q) |
|-----------|-----------|------------|-------------|-------------|
| DenseStab | 10.0 us (1.00x) | 91.7 us (1.00x) | 528.3 us (1.00x) | 1660 us (1.00x) |
| SparseStab | 22.5 us (2.25x) | 156.2 us (1.70x) | 732.2 us (1.39x) | 2185 us (1.32x) |
| GpuStabOpt | 18.1 us (1.81x) | 177.7 us (1.94x) | 779.0 us (1.47x) | 2699 us (1.63x) |
| GpuStab | 44.5 us (4.45x) | 477.9 us (5.21x) | 2637.8 us (4.99x) | 9528 us (5.74x) |
| GpuStabParallel | 70.7 us (7.08x) | 1228.7 us (13.4x) | 7133 us (13.5x) | 27005 us (16.3x) |

**Conclusions:**
- DenseStab remains fastest on CPU due to 64-bit word operations and SIMD
- SparseStab scales well for larger systems (ratio improves from 2.25x to 1.32x)
- GpuStabOpt is competitive on CPU (1.5-2x slower than DenseStab)

---

## wgpu GPU Benchmark Results (2025-01-18)

### Surface Code with Measurements (Measurement-Heavy Workload)

The wgpu GPU implementation is **extremely slow** for measurement-heavy workloads:

| d | qubits | wgpu GPU | DenseStab | GpuStabOpt |
|---|--------|----------|-----------|------------|
| 5 | 49 | 27,483 us (2602x) | 10.6 us (1.0x) | 17.2 us (1.6x) |
| 9 | 161 | 165,956 us (1496x) | 110.9 us (1.0x) | 188.1 us (1.7x) |
| 13 | 337 | 514,241 us (962x) | 534.8 us (1.0x) | 803.3 us (1.5x) |
| 17 | 577 | 1,253,191 us (684x) | 1,833 us (1.0x) | 2,804 us (1.5x) |
| 21 | 881 | 2,676,844 us (617x) | 4,336 us (1.0x) | 7,643 us (1.8x) |

**Root Cause:** Each measurement requires CPU-GPU synchronization:
- `find_first_anticommuting()` dispatches GPU, copies result, waits for completion
- d=17 surface code with 34 rounds has ~9,800 measurements = ~9,800 syncs

### Gates Only (No Measurement)

Without measurements, the GPU is **competitive at scale**:

| qubits | gates | wgpu GPU | DenseStab | GpuStabOpt |
|--------|-------|----------|-----------|------------|
| 100 | 1,000 | 133 us (7.3x) | 53 us (2.9x) | 18 us (1.0x) |
| 500 | 5,000 | 1,403 us (7.1x) | 2,368 us (11.9x) | 198 us (1.0x) |
| 1000 | 10,000 | 4,331 us (5.9x) | 10,422 us (14.3x) | 731 us (1.0x) |
| 2000 | 20,000 | 9,973 us (4.0x) | 46,500 us (18.6x) | 2,507 us (1.0x) |

**Key Findings:**
1. **wgpu GPU beats DenseStab** at 500+ qubits for gates-only workloads
2. **GpuStabOpt dominates** - 4-7x faster than wgpu GPU, designed for CPU
3. **DenseStab scales poorly** - O(n^2) complexity shows at large scale
4. **GPU overhead is ~4-7x** vs optimal CPU implementation (GpuStabOpt)

### Optimization Priorities

1. **Batch Measurements** (Highest Impact)
   - Run multiple `find_anticommuting` in single dispatch
   - Defer row operations until all measurements known
   - Reduce CPU-GPU syncs from O(measurements) to O(1)

2. **Asynchronous Measurement Pipeline**
   - Start next dispatch while reading previous results
   - Use double-buffering for measurement results

3. **Gate Fusion** (Medium Impact)
   - Fuse consecutive single-qubit gates before sending to GPU
   - Reduces gate count and dispatch overhead

## Measurement Optimization (2025-01-18)

### Two Key Optimizations

#### 1. Batched Anticommuting Check

Added `find_first_anticommuting_batch()` which checks anticommutation for multiple qubits in a single GPU dispatch:

- **Shader**: `find_anticommuting_batch` in `stab_shaders.wgsl`
- Reduces CPU-GPU syncs from O(measurements) to O(1)
- **Result**: ~2x speedup

#### 2. Buffer Read Caching

The critical fix - `compute_deterministic_outcome()` was reading **5 full buffers from GPU for EACH deterministic measurement**. Now we read buffers once per `mz()` call and reuse them.

- **Before**: O(measurements) buffer reads
- **After**: O(1) buffer reads per `mz()` call
- **Result**: ~14x additional speedup

### Combined Results

| d | qubits | Original | After Both | Total Speedup |
|---|--------|----------|------------|---------------|
| 5 | 49 | 27,483 us (2602x) | 1,942 us (214x) | **14x faster** |
| 9 | 161 | 165,956 us (1496x) | 4,885 us (54x) | **34x faster** |
| 13 | 337 | 514,241 us (962x) | 16,779 us (33x) | **31x faster** |
| 17 | 577 | 1,253,191 us (684x) | 41,242 us (24x) | **30x faster** |
| 21 | 881 | 2,676,844 us (617x) | 93,936 us (24x) | **28x faster** |

### Analysis

The GPU is now **only 24x slower** than CPU (DenseStab) instead of **617x slower**. This is in line with expected GPU overhead for stabilizer simulation.

Remaining overhead comes from:
1. CPU-side processing of deterministic outcomes (still O(n^2) work per `mz()` call)
2. Single buffer read per `mz()` call (unavoidable for deterministic outcomes on CPU)

### Further Optimization Opportunities

To get closer to CPU parity:
1. **Compute deterministic outcomes on GPU** - avoid buffer readback entirely
2. **Consider Pauli frame approach** - like cuStabilizer, may be more GPU-friendly for QEC

### Gate Fusion Module

Also implemented `clifford_fusion.rs` - a module for fusing consecutive single-qubit Clifford gates:

- Handles common patterns: H*H=I, S*S=Z, X*Z=Y, etc.
- Ready for integration but not yet connected to GPU pipeline
- Expected minor impact now that measurement is optimized
