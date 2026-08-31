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
| `DemAwareDecoder` | DEM text | Maps DEM mechanisms and observables onto BP-OSD and other check-matrix decoders. |
| `BpOsdBuilder` / `BpOsdDecoder` | `SparseMatrix` check matrix or DEM text | Belief propagation with ordered-statistics post-processing. |
| `BpLsdBuilder` / `BpLsdDecoder` | `SparseMatrix` check matrix or DEM text | Belief propagation with localized-statistics post-processing. |
| `MinSumBpBuilder` / `MinSumBpDecoder` | Dense check matrix and error priors, or DEM text | Min-sum belief propagation. |
| `RelayBpBuilder` / `RelayBpDecoder` | Dense check matrix and error priors, or DEM text | Relay belief propagation. |
| `UnionFindBuilder` / `UnionFindDecoder` | `SparseMatrix` check matrix or DEM text | Union-find decoding with inversion or peeling. |
| `CheckMatrix` / `SparseMatrix` | Dense or coordinate-form matrix data | Matrix containers used by matching and LDPC decoder constructors. |
| `MwpmResult` / `BpResult` / `TesseractResult` | Decoder output | Result objects for matching, belief-propagation, and Tesseract decoders. |

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
`bp_osd()` or `tesseract()` -- or supply a decomposed projection (a model
written with `^` separators passes: each component is graphlike). See
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
