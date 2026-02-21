# Decoders

```hidden-rust
use pecos_decoders::{
    CssCode, SparseMatrix, Decoder, BpMethod, OsdMethod, UfMethod, BpSchedule,
    BpOsdDecoder, BpLsdDecoder, BeliefFindDecoder, FlipDecoder, UnionFindDecoder,
    SoftInfoBpDecoder, SoftDecoder, LdpcError,
};

fn create_css_code() -> Result<CssCode, Box<dyn std::error::Error>> {
    let hx_rows: Vec<u32> = vec![0, 1, 0, 1];
    let hx_cols: Vec<u32> = vec![0, 2, 1, 3];
    let hx = SparseMatrix::from_coo(2, 4, hx_rows, hx_cols)?;
    let hz_rows: Vec<u32> = vec![0, 1, 0, 1];
    let hz_cols: Vec<u32> = vec![0, 1, 2, 3];
    let hz = SparseMatrix::from_coo(2, 4, hz_rows, hz_cols)?;
    Ok(CssCode::new(hx, hz)?)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let css_code = create_css_code()?;
    let syndrome = vec![1u8, 0, 1, 0];
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

The following decoders are currently available in Python:

| Decoder | Description | Use Case |
|---------|-------------|----------|
| `MWPM2D` | Minimum Weight Perfect Matching for 2D codes | Surface codes, repetition codes |
| `DummyDecoder` | No-op decoder for testing | Testing and benchmarking |

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

    The Python decoders (`MWPM2D`, `DummyDecoder`) are included by default.

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

<!--skip: standalone function definition - included in preamble-->
```rust
use pecos_decoders::{CssCode, SparseMatrix};

// Create a CSS code from parity check matrices
fn create_css_code() -> Result<CssCode, Box<dyn std::error::Error>> {
    // Define Hx and Hz matrices as COO format sparse matrices
    let hx_rows: Vec<u32> = vec![0, 1, 0, 1];
    let hx_cols: Vec<u32> = vec![0, 2, 1, 3];
    let hx = SparseMatrix::from_coo(2, 4, hx_rows, hx_cols)?;

    let hz_rows: Vec<u32> = vec![0, 1, 0, 1];
    let hz_cols: Vec<u32> = vec![0, 1, 2, 3];
    let hz = SparseMatrix::from_coo(2, 4, hz_rows, hz_cols)?;

    Ok(CssCode::new(hx, hz)?)
}
```

### LDPC Decoders

#### BP-OSD Decoder

Combines belief propagation with ordered statistics decoding post-processing.

<!--skip: API illustration - actual implementation uses different constructor signature-->
```rust
use pecos_decoders::{BpOsdDecoder, Decoder, OsdMethod};

// Create CSS code
let css_code = create_css_code()?;

// Create decoder
let mut decoder = BpOsdDecoder::new(css_code);

// Configure parameters
decoder.set_max_iterations(100);
decoder.set_bp_method(BpMethod::MinSum);
decoder.set_osd_method(OsdMethod::Exhaustive);
decoder.set_osd_order(10);

// Decode syndrome
let syndrome = vec![1, 0, 1, 0];
let result = decoder.decode(&syndrome)?;

println!("Correction: {:?}", result.correction);
println!("Converged: {}", result.converged);
```

#### BP-LSD Decoder

Localized version of OSD for better scaling with large codes.

<!--skip: API illustration-->
```rust
use pecos_decoders::{BpLsdDecoder, Decoder};

let mut decoder = BpLsdDecoder::new(css_code);
decoder.set_bits_per_step(1);
decoder.set_lsd_order(10);

let result = decoder.decode(&syndrome)?;
```

#### Belief Find Decoder

Combines belief propagation with union-find algorithm.

<!--skip: API illustration-->
```rust
use pecos_decoders::{BeliefFindDecoder, Decoder, UfMethod};

let mut decoder = BeliefFindDecoder::new(css_code);
decoder.set_uf_method(UfMethod::Inversion);
decoder.set_max_bp_iterations(10);

let result = decoder.decode(&syndrome)?;
```

#### Flip Decoder

Fast bit-flipping decoder suitable for real-time applications.

<!--skip: API illustration-->
```rust
use pecos_decoders::{FlipDecoder, Decoder, BpSchedule};

let mut decoder = FlipDecoder::new(css_code);
decoder.set_max_iterations(100);
decoder.set_schedule(BpSchedule::Parallel);

let result = decoder.decode(&syndrome)?;
```

#### Union Find Decoder

Graph-based decoder using union-find data structure.

<!--skip: API illustration-->
```rust
use pecos_decoders::{UnionFindDecoder, Decoder, UfMethod};

let mut decoder = UnionFindDecoder::new(css_code);
decoder.set_uf_method(UfMethod::Inversion);

let result = decoder.decode(&syndrome)?;
```

### Advanced Features

#### Soft Information Decoding

Use log-likelihood ratios for improved decoding performance.

<!--skip: API illustration-->
```rust
use pecos_decoders::{SoftInfoBpDecoder, SoftDecoder};

let mut decoder = SoftInfoBpDecoder::new(css_code);

// Provide soft information (LLRs)
let llrs = vec![0.1, -0.5, 0.8, -0.2];
let result = decoder.decode_with_llrs(&syndrome, &llrs)?;
```

#### Batch Decoding

Decode multiple syndromes efficiently.

<!--skip: API illustration-->
```rust
use pecos_decoders::BatchDecoder;

let syndromes = vec![
    vec![1, 0, 1, 0],
    vec![0, 1, 0, 1],
    vec![1, 1, 0, 0],
];

let results = decoder.decode_batch(&syndromes)?;
for (i, result) in results.iter().enumerate() {
    println!("Syndrome {}: {:?}", i, result.correction);
}
```

#### Performance Tuning

<!--skip: API illustration-->
```rust
let mut decoder = BpOsdDecoder::new(css_code);
decoder.set_schedule(BpSchedule::Parallel);  // Use parallel BP updates
decoder.set_num_threads(4);  // Set thread count
```

### Error Handling

<!--skip: API illustration-->
```rust
match decoder.decode(&syndrome) {
    Ok(result) => {
        println!("Success: {:?}", result.correction);
    }
    Err(DecoderError::InvalidSyndrome(msg)) => {
        eprintln!("Invalid syndrome: {}", msg);
    }
    Err(DecoderError::DecodingFailed(msg)) => {
        eprintln!("Decoding failed: {}", msg);
    }
    Err(e) => {
        eprintln!("Other error: {}", e);
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

## See Also

- [Getting Started Guide](getting-started.md) - Main installation guide
- [LLVM Setup Guide](llvm-setup.md) - For building with LLVM support
