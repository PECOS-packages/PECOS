# PecosRng Optimization Analysis

## Current Performance vs RapidRng

### Where PecosRng WINS (must preserve)

| Benchmark | PecosRng | RapidRng | Speedup | Why |
|-----------|--------------|----------|---------|-----|
| Scalar u64 | 4.82 µs | 5.90 µs | 1.22x | Buffering amortizes generation |
| Scalar bool (fast) | 3.95 µs | 5.21 µs | 1.32x | Bit-packed extraction |
| Bulk fill_u64 (1K) | 5.0 ns | 9.4 ns | 1.88x | 4 parallel generators |
| Bulk fill_u64 (10K) | 46.6 ns | 85.6 ns | 1.84x | 4 parallel generators |
| Bulk fill_u64 (100K) | 423 ns | 853 ns | 2.02x | 4 parallel generators |
| Bulk fill_u64 (1M) | 4.14 µs | 8.52 µs | 2.06x | 4 parallel generators |
| StateVec Born Rule | 8.92 µs | 9.71 µs | 1.09x | f64 generation efficiency |
| Stabilizer Measurement | 2.86 µs | 9.17 µs | 3.21x | `random_bool(0.5)` uses `next_bool_fast()` |

### Where RapidRng WINS (need to improve)

| Benchmark | PecosRng | RapidRng | Slowdown | Root Cause |
|-----------|--------------|----------|----------|------------|
| Scalar f64 | 6.77 µs | 4.54 µs | 1.49x | Buffer check overhead in `next_u64()` |
| Scalar range_100 | 6.58 µs | 5.69 µs | 1.16x | rand's range handling + buffer overhead |
| simd_column (100K) | 854 ns | 693 ns | 1.23x | u64x4 construction overhead |
| simd_column (1M) | 9.97 µs | 6.95 µs | 1.43x | u64x4 construction overhead |
| Measurement Sampling | 30.9 ms | 26.3 ms | 1.17x | Mixed bool generation pattern |
| Noise Model | 8.31 µs | 5.70 µs | 1.46x | f64 + range in tight loop |
| Probability scalar | 6.92 µs | 4.49 µs | 1.54x | Buffer check on every `check_probability()` |

## Root Cause Analysis

### 1. Buffer Check Overhead

Current `next_u64()`:
```rust
pub fn next_u64(&mut self) -> u64 {
    if self.buffer_idx >= BUFFER_SIZE as u8 {  // Branch on every call
        self.refill_buffer();                   // Cold path
    }
    let val = self.buffer[self.buffer_idx as usize];
    self.buffer_idx += 1;
    val
}
```

The branch `if buffer_idx >= 16` happens on every call. Even though it's usually predicted correctly, there's overhead from:
- The comparison itself
- Branch predictor state
- Memory access pattern (buffer vs direct state)

### 2. u64x4 Construction Overhead

Current `next_u64x4()`:
```rust
pub fn next_u64x4(&mut self) -> u64x4 {
    u64x4::new([
        self.rngs[0].next_u64(),
        self.rngs[1].next_u64(),
        self.rngs[2].next_u64(),
        self.rngs[3].next_u64(),
    ])
}
```

Creating a `u64x4` from 4 separate values has overhead vs RapidRng's simpler 4 sequential calls.

### 3. Mixed-Use Pattern Inefficiency

In benchmarks like "Noise Model", the pattern is:
```rust
if rng.next_f64() < error_rate {   // Uses buffered next_u64()
    pauli = rng.random_range(0..3); // Uses buffered next_u64() again
}
```

The buffer doesn't help here because we're doing different operations that don't benefit from having pre-generated values.

## Why Buffering Helps for Scalar u64

The scalar u64 benchmark does:
```rust
for _ in 0..10_000 {
    sum = sum.wrapping_add(rng.next_u64());
}
```

Here buffering wins because:
1. We refill 16 values at once (4 chunks of 4)
2. The refill uses all 4 parallel RNGs
3. The 16 values are served with minimal overhead (just array access + increment)
4. The cold refill path is only hit every 16 calls

## Optimization Constraints

**Must preserve:**
- 2x bulk fill_u64 advantage (4 parallel generators)
- 1.2x scalar u64 advantage (buffering)
- 3.2x stabilizer measurement advantage (bit-packed bools)
- Statistical quality (all tests must pass)

**Would like to improve:**
- Scalar f64 (currently 1.5x slower)
- Probability scalar (currently 1.5x slower)
- Noise model pattern (currently 1.5x slower)
- simd_column (currently 1.2-1.4x slower)

## Potential Optimization Strategies

### Strategy A: Dual-Path Design (NO - loses u64 advantage)

Use scalar RNG for scalar ops, parallel for bulk. But this loses the u64 buffering advantage since we'd bypass the buffer.

### Strategy B: Optimized Buffer Check

Reduce buffer check overhead without removing buffering:
- Use wrapping arithmetic instead of comparison
- Prefetch next buffer values
- Unroll buffer access

### Strategy C: Separate f64 Buffer

Add a dedicated f64 buffer that's filled alongside u64:
- When refilling u64 buffer, also fill f64 buffer
- `next_f64()` uses f64 buffer directly (no u64 conversion on hot path)

### Strategy D: Branchless Buffer Access

Use branchless techniques:
```rust
// Instead of: if buffer_idx >= 16 { refill() }
// Use: conditional move or mask
```

### Strategy E: Larger Buffer with Prefetch

Increase buffer size to 64 or 128 values:
- Fewer refills = less branch overhead amortization
- Use prefetch hints for next buffer chunk

## Detailed Analysis: Why u64 Wins but f64/Probability Lose

**Observation:**
- Scalar u64: PecosFastRng 4.82 µs vs RapidRng 5.90 µs (we WIN)
- Probability scalar: PecosFastRng 6.92 µs vs RapidRng 4.49 µs (we LOSE)

Both use the same `next_u64()` internally! The difference is the surrounding code:

**u64 benchmark:**
```rust
for _ in 0..10_000 {
    sum = sum.wrapping_add(rng.next_u64());  // No branch on result
}
```

**Probability benchmark:**
```rust
for _ in 0..10_000 {
    if rng.check_probability(threshold) {   // Branch on result!
        count += 1;
    }
}
```

**Root cause: Branch interference**

The buffer check branch (`if buffer_idx >= 16`) interferes with the outer conditional branch (`if rng.check_probability(...)`). The branch predictor struggles with two related branches in tight succession.

## Improvement Ideas (Preserving Advantages)

### Idea 1: Larger Buffer (64 instead of 16)

**Rationale:** Fewer refills = fewer buffer check branches = less interference

```rust
const BUFFER_CHUNKS: usize = 16;  // Was 4
const BUFFER_SIZE: usize = BUFFER_CHUNKS * 4;  // 64 instead of 16
```

**Expected impact:**
- Reduces refill frequency by 4x
- Branch predictor sees 63 "not taken" then 1 "taken"
- Better amortization of refill cost

**Risk:** Larger memory footprint, might hurt cache

### Idea 2: Countdown Instead of Count-up

**Rationale:** `== 0` is often faster than `>= 16` on x86

```rust
buffer_remaining: u8,  // Counts down from 16 to 0

pub fn next_u64(&mut self) -> u64 {
    if self.buffer_remaining == 0 {
        self.refill_buffer();
    }
    self.buffer_remaining -= 1;
    self.buffer[(BUFFER_SIZE - 1 - self.buffer_remaining as usize)]
}
```

**Expected impact:**
- Potentially faster comparison
- Same memory layout

### Idea 3: Return-Then-Check Pattern

**Rationale:** Move branch after the return value is determined to improve ILP

```rust
cached_value: u64,  // Always-ready value

pub fn next_u64(&mut self) -> u64 {
    let result = self.cached_value;
    // Refill happens AFTER we have our return value
    self.buffer_idx += 1;
    if self.buffer_idx >= BUFFER_SIZE as u8 {
        self.refill_buffer();
    }
    self.cached_value = self.buffer[self.buffer_idx as usize];
    result
}
```

**Expected impact:**
- CPU can work on return value while doing buffer management
- Better instruction-level parallelism

### Idea 4: Branchless Buffer Access

**Rationale:** Eliminate the branch entirely using conditional moves

```rust
pub fn next_u64(&mut self) -> u64 {
    // Branchless: always increment, conditionally reset
    let needs_refill = self.buffer_idx >= BUFFER_SIZE as u8;
    if needs_refill { self.refill_buffer(); }  // Still a branch, but...

    // Use wrapping to avoid bounds check
    let idx = self.buffer_idx & (BUFFER_SIZE - 1) as u8;  // Mask instead of branch
    let val = self.buffer[idx as usize];
    self.buffer_idx = idx + 1;
    val
}
```

**Note:** This only works if BUFFER_SIZE is a power of 2.

### Idea 5: Prefetch Hint Before Refill

**Rationale:** When buffer is running low, prefetch next chunk

```rust
pub fn next_u64(&mut self) -> u64 {
    let idx = self.buffer_idx as usize;

    // Prefetch when we're 4 values from needing refill
    if idx == BUFFER_SIZE - 4 {
        // Prefetch the RNG states
        std::arch::x86_64::_mm_prefetch(
            &self.rngs[0] as *const _ as *const i8,
            std::arch::x86_64::_MM_HINT_T0
        );
    }

    if idx >= BUFFER_SIZE {
        self.refill_buffer();
    }
    self.buffer_idx += 1;
    self.buffer[idx]
}
```

**Risk:** Platform-specific, might not help

## Recommended Approach

Start with **Idea 1 (Larger Buffer)** combined with **Idea 2 (Countdown)**:

1. Increase buffer to 64 values (16 chunks of 4)
2. Use countdown for faster comparison
3. Benchmark to verify no regression in winning cases
4. If that's not enough, try Idea 3 (Return-Then-Check)

## Benchmarks to Run

1. **Regression tests:** scalar u64, bulk fill_u64, stabilizer measurement
2. **Improvement tests:** scalar f64, probability scalar, noise model pattern
3. **Statistical quality:** Ensure all tests still pass

## Experimental Results

### V2: Larger Buffer (64 instead of 16) - FAILED

| Benchmark | V1 | V2 | Change |
|-----------|-----|-----|--------|
| Scalar u64 | 4.85 µs | 6.54 µs | -35% slower |
| Scalar f64 | 6.96 µs | 8.07 µs | -16% slower |
| Prob scalar | 6.23 µs | 7.04 µs | -13% slower |
| count_occurrences | 3.49 µs | 3.54 µs | ~same |

**Conclusion:** Larger buffer made everything worse. The refill cost (16 x4 calls vs 4 x4 calls)
outweighs the reduced refill frequency. Cache pressure from 512 bytes vs 128 bytes also hurts.

### V3: Hybrid Design (Dedicated Scalar RNG) - SUCCESS

| Benchmark | V1 | V3 | RapidRng | V3 vs V1 | V3 vs RapidRng |
|-----------|-----|-----|----------|----------|----------------|
| Scalar u64 | **4.85 µs** | 5.99 µs | 5.91 µs | -24% slower | ~same |
| Scalar f64 | 6.96 µs | **4.74 µs** | 4.71 µs | +32% faster | ties |
| Prob scalar | 6.23 µs | **4.01 µs** | 4.41 µs | +36% faster | +9% faster! |
| count_occurrences | **3.49 µs** | 3.68 µs | - | -5% slower | - |

**Conclusion:** V3 successfully closes the gap on mixed patterns:
- Beats RapidRng for probability checking (4.01 µs vs 4.41 µs)
- Matches RapidRng for f64 generation
- Trade-off: Loses scalar u64 buffering advantage

### Design Trade-offs

| Use Case | Best Version | Reason |
|----------|--------------|--------|
| Tight u64 loops | V1 | Buffering amortizes RNG overhead |
| Mixed patterns (f64, probability) | V3 | No buffer overhead, direct RapidRng |
| Bulk fill operations | Both | 4 parallel generators in both |
| Bit-packed bools | Both | Same implementation |

## Recommendations

### Option A: Keep V1, Document Trade-offs
- V1 is already good for most use cases
- Users who need maximum scalar performance can use RapidRng directly
- Simpler codebase

### Option B: Replace V1 with V3
- V3 is better for the common case (mixed patterns)
- Probability checking is a key operation in noise models
- Slight regression in scalar u64 is acceptable (still faster than most RNGs)

### Option C: Offer Both
- Keep `PecosFastRng` as V1 for backwards compatibility
- Add `PecosFastRngDirect` or similar as V3 for users who prefer it
- More complexity but maximum flexibility

## Actual Usage Patterns in PECOS

### Pattern Analysis

| Use Case | Operation | Type | Frequency | Fixed/Varying Prob |
|----------|-----------|------|-----------|-------------------|
| Noise probability | u64 threshold check | bool | 1 per gate | Fixed |
| Pauli selection (1q) | `random_index_3()` | u8 | 1 per error | Fixed |
| Pauli selection (2q) | `random_index_15()` | u8 | 1 per error | Fixed |
| Born rule (state vec) | f64 comparison | bool | 1 per qubit | Varying |
| Stabilizer measurement | `coin_flip()` | bool | 1 per qubit | Fixed (50%) |
| Columnar sampling | u64x4 bulk | 256 bits | 1 per column | Varying |

### What's Already Optimized
- **Noise models**: Already using u64 thresholds (fixed-point) - good!
- **Columnar sampling**: Already using SIMD u64x4 - good!
- **50% coin flip**: Uses sign bit extraction - good!

### Optimization Opportunities

#### 1. Pauli Selection (Rejection Sampling Waste)

Current `random_index_3()` uses rejection sampling:
```rust
loop {
    let bits = (rng.next_u32() >> 30) as u8;  // 2 bits: 0,1,2,3
    if bits < 3 { return bits; }               // Reject 3
}
```
- Wastes ~25% of generated values
- Each call needs fresh u32

**Optimization idea**: Batch extraction
```rust
fn random_index_3_batch(&mut self, count: usize) -> Vec<u8> {
    // Generate enough u64s to cover expected need (with rejection overhead)
    // Extract multiple 2-bit pairs, filter valid ones
}
```

#### 2. Born Rule (f64 vs u64)

Current:
```rust
self.rng.random::<f64>() < prob_one  // f64 generation + comparison
```

**Optimization idea**: Convert probability to threshold on the fly
```rust
let threshold = (prob_one * u64::MAX as f64) as u64;
self.rng.next_u64() < threshold  // Just u64 generation + comparison
```
- Avoids f64 generation overhead
- u64 comparison is cheaper than f64 comparison

#### 3. Batch Probability Checks for Noise

When processing many gates, could batch:
```rust
// Instead of:
for gate in gates {
    if rng.check_probability(threshold) {
        apply_error(gate);
    }
}

// Could do:
let error_indices = rng.check_probability_indices(threshold, gates.len());
for idx in error_indices {
    apply_error(&gates[idx]);
}
```

**Implementation using SIMD**:
```rust
fn check_probability_indices(&mut self, threshold: u64, count: usize) -> Vec<usize> {
    let mut indices = Vec::new();
    let threshold_x4 = u64x4::splat(threshold);

    for chunk_start in (0..count).step_by(4) {
        let values = self.next_u64x4();
        let mask = values.cmp_lt(threshold_x4);  // SIMD comparison
        // Extract indices where mask is true
        for (i, &hit) in mask.as_array_ref().iter().enumerate() {
            if hit { indices.push(chunk_start + i); }
        }
    }
    indices
}
```

#### 4. Combined Noise Check + Pauli Selection

For noise models, the pattern is always:
1. Check if error occurs (probability check)
2. If yes, select Pauli type

**Optimization idea**: Fused operation
```rust
/// Returns None if no error, Some(pauli) if error occurred
fn noise_sample_1q(&mut self, threshold: u64) -> Option<u8> {
    let val = self.next_u64();
    if val < threshold {
        // Use remaining bits for Pauli selection
        Some(((val >> 62) % 3) as u8)  // Reuse same random value!
    } else {
        None
    }
}
```
- One RNG call instead of two when error occurs
- Slight bias in Pauli selection (acceptable for noise?)

#### 5. SIMD Threshold Comparison

The `wide` crate supports SIMD comparisons:
```rust
use wide::u64x4;

fn check_probability_x4_simd(&mut self, threshold: u64) -> [bool; 4] {
    let values = self.next_u64x4();
    let thresh = u64x4::splat(threshold);
    let mask = values.cmp_lt(thresh);  // Returns i64x4 with -1 or 0
    // Convert to bools
    let arr: [i64; 4] = mask.into();
    [arr[0] != 0, arr[1] != 0, arr[2] != 0, arr[3] != 0]
}
```

## Proposed Action Plan

### Phase 1: Quick Wins (Low Risk) - COMPLETED

1. **Add fused `noise_sample_1q/2q`**: Combine probability check + Pauli selection
   - Implemented in `rng_ext.rs`
   - Updated depolarizing noise model to use fused methods
   - **Result**: Same performance as separate calls at typical error rates (0.1%)
   - The multiply-shift method eliminates rejection sampling overhead
   - Main benefit: cleaner API, single method call instead of if/then pattern

2. **Replaced rejection sampling with multiply-shift** in `random_index_3()` and `random_index_15()`:
   - Before: Loop with rejection (wastes ~25% and ~6.25% of values respectively)
   - After: Single u128 multiply-shift operation (no rejection, unbiased)
   - Impact at 0.1% error rate: minimal (errors are rare)
   - Impact at higher error rates: should help

### Phase 1 Benchmark Results

| Pattern | Separate | Fused | Notes |
|---------|----------|-------|-------|
| 1q (PecosRng) | 6.60 µs | 6.68 µs | ~same |
| 2q (PecosRng) | 6.57 µs | 6.66 µs | ~same |
| 1q (SmallRng) | 7.02 µs | 7.17 µs | ~same |

At 0.1% error rate, error events are rare (~10 per 10K gates), so the optimization
for the error path has minimal overall impact. The main benefit is code clarity.

### Phase 2: Batching (Medium Risk) - COMPLETED

3. **Add `check_probability_indices`**: Return indices of successful events
   - Implemented in `RngProbabilityExt` trait with default implementation
   - Optimized implementations for `PecosRng` and `PecosScalarRng` using parallel RNGs
   - Processes 4 values at a time using `next_u64x4()`

4. **Add `random_index_3_batch`**: Batch Pauli selection with less rejection waste
   - Not yet implemented (lower priority since Pauli selection is cold path)

### Phase 2 Benchmark Results: Batched Probability Checking

Test: 10K probability checks at 0.1% error rate, collecting indices of events

| RNG Type | Scalar Loop | Batched | Speedup |
|----------|-------------|---------|---------|
| PecosQualityRng | 7.22 µs | 6.58 µs | 1.10x |
| SmallRng | 7.17 µs | 6.81 µs | 1.05x |
| **PecosRng** | 6.37 µs | **4.46 µs** | **1.62x** |
| PecosScalarRng | 6.36 µs | 4.54 µs | 1.59x |

**Key insights:**
- The parallel RNGs in `PecosRng`/`PecosScalarRng` provide significant speedup for batched operations
- Scalar RNGs (`PecosQualityRng`, `SmallRng`) see modest improvement from reduced loop overhead
- `PecosRng` batched (4.46 µs) is **1.62x faster** than its scalar loop and **1.52x faster** than SmallRng's batched
- This optimization is most beneficial when processing many gates at once (noise model pattern)

### Phase 3: Architecture (Higher Risk)
5. **Evaluate V3 hybrid design**: Use for mixed patterns
6. **Consider specialized RNG types**: NoiseRng, MeasurementRng with use-case-specific APIs

## Cleanup Completed

- **V2 removed** - Failed experiment (larger buffer made things slower)
- **V3 renamed to `PecosScalarRng`** - Successful hybrid design for scalar-optimized operations
- **V1 is now the default `PecosRng`** - Renamed from `PecosFastRng`
- **Original `PecosRng` renamed to `PecosQualityRng`** - For users who need Xoshiro256++ quality

### File Structure

| File | Type | Description |
|------|------|-------------|
| `rng.rs` | `PecosRng` | Default RNG (parallel RapidRng with buffering) |
| `quality_rng.rs` | `PecosQualityRng` | High-quality SIMD Xoshiro256++ |
| `scalar_rng.rs` | `PecosScalarRng` | Scalar-optimized (no buffer overhead) |

### When to Use Which RNG

| Use Case | Recommended RNG | Reason |
|----------|-----------------|--------|
| General use (default) | `PecosRng` | Best overall performance |
| Scalar probability checks (one at a time) | `PecosScalarRng` | 36% faster, no buffer overhead |
| Scalar f64 generation | `PecosScalarRng` | 32% faster, no buffer overhead |
| Batched probability checks | `PecosRng` | 1.6x faster than scalar loops |
| Tight u64 loops | `PecosRng` | Buffering amortizes RNG overhead |
| Maximum statistical quality | `PecosQualityRng` | Xoshiro256++ algorithm |

**Key insight**: With batched probability checking (`check_probability_indices`), `PecosRng`
achieves the same performance as `PecosScalarRng` for the noise model hot path. This makes
`PecosRng` the best default choice since it also excels at tight u64 loops.

## Future Direction: Adaptive RNG Architecture

The current approach has specialized RNG types for different use cases. A better
architecture would be a single adaptive RNG that uses different internal strategies
depending on the operation:

### Proposed Design

```rust
pub struct PecosAdaptiveRng {
    /// Dedicated scalar RNG for probability checks and f64 generation
    scalar: RapidRng,

    /// Parallel RNGs for bulk operations
    parallel: [RapidRng; 4],

    /// Buffer for scalar u64 (optional, for tight loops)
    u64_buffer: [u64; 16],
    u64_idx: u8,

    /// Bit buffer for bool generation
    bool_bits: u64,
    bool_remaining: u8,
}
```

### Strategy Selection

| Operation | Strategy | Reason |
|-----------|----------|--------|
| `check_probability()` | Direct scalar | No buffer overhead, best for mixed patterns |
| `next_u64()` in tight loop | Buffered | Amortizes RNG overhead |
| `next_f64()` | Direct scalar | Avoid buffer overhead |
| `fill_u64()` | Parallel | 2x throughput |
| `next_bool_fast()` | Bit-packed | 64x efficiency |

### Trait-Based Extensibility

Expand `RngProbabilityExt` to provide sensible defaults for any `RngCore`:

```rust
pub trait RngProbabilityExt: RngCore {
    // Existing methods with default implementations...

    // New: Adaptive methods that specialized RNGs can override
    fn next_u64_for_probability(&mut self) -> u64 {
        self.next_u64()  // Default: use standard method
    }

    fn next_u64_for_bulk(&mut self) -> u64 {
        self.next_u64()  // Default: use standard method
    }
}
```

Specialized RNGs can override to use different internal paths.

### Benefits

1. **Single type** - Users don't need to choose between `PecosRng` and `PecosScalarRng`
2. **Automatic optimization** - Right strategy for each use case
3. **Backwards compatible** - Existing code works unchanged
4. **Extensible** - Trait defaults work for any RNG

## Next Steps

1. Consider implementing `PecosAdaptiveRng` if `PecosRng`/`PecosScalarRng` trade-offs become problematic
2. Implement Phase 2 batching optimizations if needed for specific use cases
3. Statistical quality tests completed - all RNGs pass
