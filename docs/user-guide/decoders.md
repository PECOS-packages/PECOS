# Decoders

```hidden-rust
use pecos_decoders::{SparseMatrix, Decoder, BpMethod, OsdMethod, UfMethod, BpSchedule, BpOsdDecoder, BpLsdDecoder, BeliefFindDecoder, FlipDecoder, UnionFindDecoder, SoftInfoBpDecoder, LdpcError, InputVectorType};
use ndarray::array;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rows: Vec<u32> = vec![0, 0, 1, 1];
    let cols: Vec<u32> = vec![0, 1, 2, 3];
    let pcm = SparseMatrix::from_coo(2, 4, rows, cols)?;
    let error_rate = 0.05;
    let syndrome = array![1u8, 0];
    // CODE
    Ok(())
}
```

PECOS provides quantum error correction decoders through both Python and Rust APIs. The availability of specific decoders varies by language.

## Overview

The decoder system in PECOS is designed around modularity and performance:

- **Optional Components**: Decoders are not built by default to keep PECOS lightweight
- **External Integration**: LDPC decoders come from specialized external projects
- **Unified API**: Consistent interface across different decoder implementations
- **Cross-Language Support**: Some decoders available in both Python and Rust, others Rust-only

## DEM text grammar

PECOS reads flat Stim detector-error-model text using Stim's grammar:

- Instruction names and detector/observable target prefixes are case-insensitive; tags and inline `#` comments are supported.
- Targets and `^` separators require spacing; separators cannot be first, last, or adjacent.
- Parenthesized arguments immediately follow the name or tag. `error` requires one probability in `[0, 1]`, with `error()` meaning zero; detector and logical-observable declarations require exactly one target of the appropriate kind.
- Grammar is validated per instruction; block balance is not checked. `repeat`, `shift_detectors`, and closing braces require flattening, including stray braces and unclosed repeat blocks.
- PECOS extension targets and metadata statements require an explicit opt-in: `ParsedDem` and `DetectorErrorModel::with_pecos_dem_metadata` enable the whole PECOS superset, including `TP` targets and JSON metadata statements.

PECOS-parsed Python constructors report grammar errors as `ValueError` with the `Invalid DEM syntax:` prefix; Stim-backed constructors such as `PyMatchingDecoder` and `TesseractDecoder` retain their native parser errors and exception types.

## Available Decoders

### Python Decoders

The following decoder APIs and supporting types are publicly re-exported from
`pecos.decoders`:

| API | Primary input | Description |
|-----|---------------|-------------|
| `MWPM2D` | QECC object | Legacy minimum-weight perfect matching for 2D codes. |
| `DummyDecoder` | None | No-op decoder for tests and interface benchmarks. |
| `PyMatchingDecoder` | Graph-like DEM text or `CheckMatrix` | PyMatching minimum-weight perfect matching, with optional correlated decoding. |
| `FusionBlossomDecoder` | Check matrix, standard-code parameters, or a manual graph | Pure-Rust minimum-weight perfect matching. |
| `TesseractDecoder` | DEM text | Search-based decoder that accepts raw hyperedges. |
| `TesseractTrellisDecoder` | DEM text | Sums probability mass over a beam of partial syndromes and reports a per-shot observable probability; accepts at most one observable and a 256-detector active frontier. The probability and `low_confidence` flag come only from this class; the `tesseract_trellis()` spec path yields the observable mask. |
| `DemAwareDecoder` | DEM text | Maps DEM mechanisms and observables onto BP-OSD and other check-matrix decoders. |
| `BpOsdBuilder` / `BpOsdDecoder` | `SparseMatrix` check matrix or DEM text | Belief propagation with ordered-statistics post-processing. |
| `BpLsdBuilder` / `BpLsdDecoder` | `SparseMatrix` check matrix or DEM text | Belief propagation with localized-statistics post-processing. |
| `MinSumBpBuilder` / `MinSumBpDecoder` | Dense check matrix and error priors, or DEM text | Min-sum belief propagation. |
| `RelayBpBuilder` / `RelayBpDecoder` | Dense check matrix and error priors, or DEM text | Relay belief propagation. |
| `UnionFindBuilder` / `UnionFindDecoder` | `SparseMatrix` check matrix or DEM text | Union-find decoding with inversion or peeling. |
| `CheckMatrix` / `SparseMatrix` | Dense or coordinate-form matrix data | Matrix containers used by matching and LDPC decoder constructors. |
| `MwpmResult` / `BpResult` / `TesseractResult` / `TesseractTrellisResult` | Decoder output | Result objects for matching, belief-propagation, and Tesseract decoders. |

The experimental `frontier()` and `bp_trellis()` factories are not part of this table: they import from `pecos.decoders` only when the optional `pecos-rslib-exp` package is installed, and are described in the [Rust-backed Frontier](#rust-backed-frontier-batch-decoding) and [Rust-backed BP-Trellis](#rust-backed-bp-trellis-batch-decoding) sections below.

Python decoder inputs name their encoding explicitly: use
`decode_syndrome(...)` for a dense detector vector and
`decode_from_defects(...)` for sparse detector indices. The BP/LDPC classes'
`from_dem(...)` constructors return a `DemAwareDecoder` wrapper so their results
include `observable_flips` and their instances retain the DEM dimensions.

### Rust Decoders

The Rust API provides access to a broader set of decoders:

**LDPC Decoders** (feature: `ldpc`):

- BP-OSD (Belief Propagation with Ordered Statistics Decoding)
- BP-LSD (Belief Propagation with Localized Statistics Decoding)
- MBP (Min-sum Belief Propagation)
- Belief Find decoder
- Flip decoder
- Union Find decoder
- SoftInfoBP decoder

**Other Decoders**:

- Fusion Blossom MWPM (feature: `fusion-blossom`)
- PyMatching MWPM (feature: `pymatching`)
- Tesseract (feature: `tesseract`)
- Chromobius color code decoder (feature: `chromobius`)

## Installation and Setup

=== ":fontawesome-brands-python: Python"

    Install PECOS with decoder support:

    ```bash
    pip install quantum-pecos
    ```

    The Python package exports the decoder APIs listed above.

=== ":fontawesome-brands-rust: Rust"

    Add decoder dependencies to your `Cargo.toml`:

    ```toml
    [dependencies]
    # Option 1: Use the meta-crate with specific features
    pecos-decoders = { version = "0.1.1", features = ["ldpc"] }

    # Option 2: Use individual decoder crate
    pecos-ldpc-decoders = "0.1.1"

    # Core types (always needed for custom decoders)
    pecos-decoder-core = "0.1.1"
    ```

    Build with LDPC decoders:

    ```bash
    # Build LDPC decoders
    cargo build --package pecos-decoders --features ldpc

    # Build all decoders
    cargo build --package pecos-decoders --all-features
    ```

## Python API

### MWPM2D Decoder

The `MWPM2D` decoder implements Minimum Weight Perfect Matching for 2D topological codes.

```python
import pecos as pc

# Create a surface code first
surface = pc.qeccs.Surface4444(distance=3)

# Create decoder with the QECC
decoder = pc.decoders.MWPM2D(surface)
```

### DummyDecoder

A no-op decoder useful for testing decoder interfaces without actual decoding.

```python
from pecos.decoders import DummyDecoder

decoder = DummyDecoder()
```

## Rust API

### Creating Error Correction Codes

Before using decoders, you need a quantum error correction code:

```rust
use pecos_decoders::SparseMatrix;

// Create a parity check matrix in COO (coordinate) format
let rows: Vec<u32> = vec![0, 0, 1, 1];
let cols: Vec<u32> = vec![0, 1, 2, 3];
let pcm = SparseMatrix::from_coo(2, 4, rows, cols)?;
```

### LDPC Decoders

#### BP-OSD Decoder

Combines belief propagation with ordered statistics decoding post-processing.

```rust
use pecos_decoders::{BpOsdDecoder, BpMethod, OsdMethod, BpSchedule};

// Build decoder with configuration
let mut decoder = BpOsdDecoder::builder(&pcm)
    .error_rate(error_rate)
    .max_iter(100)
    .bp_method(BpMethod::MinimumSum)
    .osd_method(OsdMethod::OsdE)
    .osd_order(10)
    .build()?;

// Decode syndrome
let result = decoder.decode(&syndrome.view())?;

println!("Decoding: {:?}", result.decoding);
println!("Converged: {}", result.converged);
```

#### BP-LSD Decoder

Localized version of OSD for better scaling with large codes.

```rust
use pecos_decoders::BpLsdDecoder;

let mut decoder = BpLsdDecoder::builder(&pcm)
    .error_rate(error_rate)
    .bits_per_step(1)
    .lsd_order(10)
    .build()?;

let result = decoder.decode(&syndrome.view())?;
```

#### Belief Find Decoder

Combines belief propagation with union-find algorithm.

```rust
use pecos_decoders::{BeliefFindDecoder, UfMethod};

let mut decoder = BeliefFindDecoder::builder(&pcm)
    .error_rate(error_rate)
    .uf_method(UfMethod::Inversion)
    .max_iter(10)
    .build()?;

let result = decoder.decode(&syndrome.view())?;
```

#### Flip Decoder

Fast bit-flipping decoder suitable for real-time applications.

```rust
use pecos_decoders::FlipDecoder;

let mut decoder = FlipDecoder::builder(&pcm)
    .max_iter(100)
    .build()?;

let result = decoder.decode(&syndrome.view())?;
```

#### Union Find Decoder

Graph-based decoder using union-find data structure.

```rust
use pecos_decoders::{UnionFindDecoder, UfMethod};

let mut decoder = UnionFindDecoder::builder(&pcm)
    .method(UfMethod::Inversion)
    .build()?;

// Union find decode takes syndrome, LLRs, and bits_per_step
let llrs = vec![0.1; 4];  // one LLR per bit column
let result = decoder.decode(&syndrome.view(), &llrs, 1)?;
```

### Advanced Features

#### Soft Information Decoding

Use log-likelihood ratios for improved decoding performance.

```rust
use pecos_decoders::SoftInfoBpDecoder;

let mut decoder = SoftInfoBpDecoder::builder(&pcm)
    .error_rate(error_rate)
    .max_iter(50)
    .build()?;

// Soft decode takes soft syndrome values, cutoff, and sigma
let soft_syndrome = vec![0.9, 0.1];
let result = decoder.decode(&soft_syndrome, 5.0, 0.5)?;
```

#### Batch Decoding

Decode multiple syndromes efficiently.

```rust
use pecos_decoders::BpOsdDecoder;
use ndarray::array;

let mut decoder = BpOsdDecoder::builder(&pcm).error_rate(error_rate).build()?;

let syndromes = vec![
    array![1u8, 0],
    array![0u8, 1],
];

for (i, syn) in syndromes.iter().enumerate() {
    let result = decoder.decode(&syn.view())?;
    println!("Syndrome {}: {:?}", i, result.decoding);
}
```

#### Performance Tuning

```rust
use pecos_decoders::{BpOsdDecoder, BpSchedule};

let mut decoder = BpOsdDecoder::builder(&pcm)
    .error_rate(error_rate)
    .bp_schedule(BpSchedule::Parallel)
    .omp_threads(4)
    .build()?;
```

### Error Handling

```rust
use pecos_decoders::{BpOsdDecoder, LdpcError};

let mut decoder = BpOsdDecoder::builder(&pcm).error_rate(error_rate).build()?;

match decoder.decode(&syndrome.view()) {
    Ok(result) => {
        println!("Decoding: {:?}", result.decoding);
        println!("Converged: {}", result.converged);
    }
    Err(e) => {
        eprintln!("Decoding error: {}", e);
    }
}
```

## Performance Considerations

1. **Algorithm Selection**:
   - BP-OSD: Best overall performance for most codes
   - BP-LSD: Better for very large codes
   - Flip: Fastest but lower performance
   - Union Find: Good for codes with specific structure

2. **Parameter Tuning**:
   - Start with default parameters
   - Increase `max_iterations` for better convergence
   - Adjust `osd_order` based on code size and error rate
   - Use parallel schedules for larger codes

3. **Hardware Optimization**:
   - Enable CPU-specific optimizations in release builds
   - Use multiple threads for batch decoding
   - Consider memory layout for cache efficiency

## Rust-backed Frontier batch decoding

Install the optional `pecos-rslib-exp` package for this section and BP-Trellis
below. Standard `pecos.decoders` imports do not load the experimental extension.
The explicit `from pecos.decoders import frontier, bp_trellis` convenience import
loads it lazily and raises an actionable `ImportError` if it is unavailable.
Experimental factories are excluded from wildcard imports. Their specifications
work with `SampleBatch.decode(...)` and `DemSampler.decode(...)`; the standard
`DecoderSpec.parse` strings and composite-spec factories do not load optional
providers. The experimental calls cross a Python adapter at each shot, with
native model construction and decoding releasing the GIL.

```python
from pecos_rslib_exp import frontier
from pecos_rslib.qec import SampleBatch

dem = "error(0.1) D0 D1 D2 L0\n"
batch = SampleBatch([[1, 1, 1], [0, 0, 0]], [1, 0])
result = batch.decode(dem, frontier(k=64), workers=2, predictions=True)
assert result.predictions == [1, 0]
assert result.num_errors == 0
```

Frontier accepts raw DEMs, including hyperedges. `workers=None` selects the
worker count automatically; `workers=1` runs sequentially. Parallel execution
releases the Python GIL and preserves shot order. At most one Rust decoder per
worker is alive at a time, so more workers and larger `k` increase memory use.
`SampleBatch.decode(...)` builds exactly one decoder per worker;
`DemSampler.decode(...)` builds one up front to check dimensions and then one per
scheduled group of sampling chunks, which can be several times the worker count
on a long run, so a model that is slow to construct pays that cost more than
once per worker there.

Options match `pecos_rslib_exp.FrontierDecoder.from_dem`: `k`, `delta`,
`score_alpha`, `bp_score_iterations`, `column_order`, `merge_indistinguishable`,
`metric_mode`, and `int_metric_scale`. The default ordering is
`"deadline_reorder"`; `"time_order"`, `"backward_deadline_reorder"`, and explicit
column permutations are also accepted. Frontier remains experimental, and
pruning can make predictions approximate. For per-shot logical masses, pruning
status, and complementary gaps, use the direct experimental binding.

## Rust-backed BP-Trellis batch decoding

```python
from pecos_rslib_exp import bp_trellis
from pecos_rslib.qec import SampleBatch

dem = "error(0.1) D0 D1 D2 L0\n"
batch = SampleBatch([[1, 1, 1], [0, 0, 0]], [1, 0])
spec = bp_trellis(
    k=8,
    delta=100.0,
    score_alpha=0.8,
    bp_score_iterations=5,
    merge_indistinguishable=True,
    ordering="deadline",
    escalation_ks=[32, 128],
)
result = batch.decode(dem, spec, workers=2, predictions=True)
assert result.predictions == [1, 0]
assert result.num_errors == 0
```

The ladder uses one engine model and prepares each shot once, including BP.
Use `escalation=[(32, 50.0), (128, 100.0)]` to choose both pruning parameters;
`escalation_ks=[32, 128]` remains shorthand for rungs at the base `delta`.
Pass only one ladder keyword. The default ladder is empty. Rungs need not be
monotone, and an exact base cannot have a ladder. Rungs run only after a no-path
attempt that dropped states: never after a successful but wrong prediction,
and never after a residual or infeasible no-path.

The direct `BpTrellisDecoder.decode_syndrome`, `decode_from_defects`, and
`decode_batch(shots, workers=1)` methods accept `on_no_path="raise"` (default)
or `"report"`. Report mode returns a `BpTrellisNoPath` in place for each failed
shot, preserving batch order. Its `cause` is `"residual"` when a detector's
residual cannot be changed, `"infeasible"` when an attempt proves there is no
path without pruning, or `"exhausted"` when all attempts fail after pruning.
Residual and infeasible outcomes skip remaining rungs. Other errors still raise.

Both outcome classes expose `no_path`, `transitions`, `bp_runs`, and
`bp_seconds`. Only a decoded result exposes `observable_flips`; a report exposes
`placeholder_flips` instead, the forced contribution of probability-one
mechanisms, including wide observables. The names differ on purpose, so code
written for a correction raises `AttributeError` on a report rather than
silently consuming a placeholder. The report also exposes `detector` (only for
residual) and `rungs_tried`.
BP time is counted once, while transitions sum all attempts.
The `bp_trellis(...)` spec route remains strict and accepts no `on_no_path`.

`ordering` also accepts `"backward_deadline"`, `"time_order"`, or an explicit
mechanism permutation. BP-Trellis uses floating-point coset masses and does not
expose Frontier's integer metric options.

Like Frontier, BP-Trellis accepts raw hyperedges and arbitrary-width observables,
releases the GIL during batch decoding, and supports automatic worker selection
and `DemSampler.decode(...)`. It remains experimental. Use
`pecos_rslib_exp.BpTrellisDecoder` for per-shot confidence, pruning status, and
retry telemetry; the unified batch result returns predictions and aggregate scores.

## Hyperedge models and matching decoders

Matching-style decoders (PyMatching, Fusion Blossom and its perturbed
correlated variant, PECOS UF, K-MWPM, belief matching, and `astar` -- though
not `astar_full`, which consumes the full check matrix) represent a model as a
graph whose edges touch at most two detectors. Given a model containing
mechanisms that touch three or more, they reject it rather than silently
ignoring those mechanisms:

```
Invalid configuration: fusion_blossom needs a graphlike model, but this DEM has
113 mechanism(s) touching three or more detectors. Decoding it here would
silently ignore them. Pass a decomposed model
(DetectorErrorModel.to_string_terminal_graphlike_decomposed() or
to_string_source_graphlike_decomposed()), or use a decoder that accepts
hyperedges such as bp_osd or tesseract.
```

Decode such a model with a decoder that represents hyperedges directly --
`bp_osd()`, `tesseract()`, `tesseract_trellis()`, `frontier()`, or `bp_trellis()` -- or supply a
decomposed projection (a model written with `^` separators passes: each component is graphlike). See
[Experimental Decoders](../experimental/decoders.md) for the Frontier and
BP-Trellis decoders, which additionally report a per-shot complementary gap,
and for provenance-based decomposition of a hyperedge model into a graphlike
one. Frontier's exact default-float masses are true unnormalized posteriors.
Under its integer `maxlog_int` metric (accepted by Python `FrontierDecoder` as
`metric_mode="maxlog_int"`; the committee remains float-only), terminal masses
are per-label best-route (Viterbi) masses, the gap is a route-mass margin, and
`log_evidence` is the winner's mass rather than evidence.

## See Also

- [Getting Started Guide](getting-started.md) - Main installation guide
- [LLVM Setup Guide](llvm-setup.md) - For building with LLVM support
- [Experimental Decoders](../experimental/decoders.md) - Frontier, BP-Trellis, complementary gap
