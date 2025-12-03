# Decoders

PECOS provides access to LDPC (Low-Density Parity-Check) quantum error correction decoders through both Python and Rust APIs. These decoders can be used to correct errors in quantum LDPC codes, surface codes, and other stabilizer codes.

## Overview

The decoder system in PECOS is designed around modularity and performance:

- **Optional Components**: Decoders are not built by default to keep PECOS lightweight
- **External Integration**: LDPC decoders come from specialized external projects
- **Unified API**: Consistent interface across different decoder implementations
- **Cross-Language Support**: Available in both Python and Rust

## Available Decoders

### LDPC Decoders
Advanced belief propagation and ordered statistics decoding algorithms for LDPC codes.

**Algorithms**:
- BP-OSD (Belief Propagation with Ordered Statistics Decoding)
- BP-LSD (Belief Propagation with Localized Statistics Decoding)
- MBP (Min-sum Belief Propagation)
- Belief Find decoder
- Flip decoder
- Union Find decoder
- SoftInfoBP decoder

**Best for**: Quantum LDPC codes, high-rate codes, hypergraph product codes, CSS codes

## Installation and Setup

=== ":fontawesome-brands-python: Python"

    Install PECOS with decoder support:

    ```bash
    # Install base PECOS (decoders are optional)
    pip install quantum-pecos

    # For decoder dependencies (when available):
    pip install quantum-pecos[decoders]
    ```

    !!! note "Decoder Availability"
        Decoder availability in Python depends on the specific Python package.
        Some decoders may only be available through the Rust interface.

=== ":fontawesome-brands-rust: Rust"

    Add decoder dependencies to your `Cargo.toml`:

    ```toml
    [dependencies]
    # Option 1: Use the meta-crate
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

    # Build all decoders (currently just LDPC)
    cargo build --package pecos-decoders --all-features
    ```

## Basic Usage

### Creating Error Correction Codes

Before using decoders, you need a quantum error correction code. Here are common examples:

=== ":fontawesome-brands-python: Python"

    ```python
    import pecos
    import numpy as np


    # Create a surface code
    def create_surface_code(distance):
        """Create a distance-d surface code."""
        # Implementation details...
        return hx, hz


    # Create a repetition code
    def create_repetition_code(n):
        """Create an n-bit repetition code."""
        h = np.zeros((n - 1, n), dtype=np.uint8)
        for i in range(n - 1):
            h[i, i] = 1
            h[i, i + 1] = 1
        return h


    # Create CSS code for LDPC decoders
    distance = 5
    hx, hz = create_surface_code(distance)
    ```

=== ":fontawesome-brands-rust: Rust"

    ```rust
    use pecos_decoders::{CssCode, SparseMatrix};

    // Create a CSS code from parity check matrices
    fn create_css_code() -> Result<CssCode, Box<dyn std::error::Error>> {
        // Define Hx and Hz matrices
        let hx_rows = vec![0, 1, 0, 1];
        let hx_cols = vec![0, 2, 1, 3];
        let hx = SparseMatrix::new(2, 4, hx_rows, hx_cols)?;

        let hz_rows = vec![0, 1, 0, 1];
        let hz_cols = vec![0, 1, 2, 3];
        let hz = SparseMatrix::new(2, 4, hz_rows, hz_cols)?;

        Ok(CssCode::new(hx, hz)?)
    }
    ```

### Using LDPC Decoders

=== ":fontawesome-brands-python: Python"

    ```python
    import pecos.decoders as decoders

    # Create decoder
    decoder = decoders.BpOsdDecoder(hx, hz)

    # Configure parameters
    decoder.set_max_iterations(100)
    decoder.set_bp_method("min_sum")
    decoder.set_osd_order(10)

    # Decode syndrome
    syndrome = [1, 0, 1, 0]  # Example syndrome
    result = decoder.decode(syndrome)

    print(f"Correction: {result.correction}")
    print(f"Converged: {result.converged}")
    print(f"Iterations: {result.iterations}")
    ```

=== ":fontawesome-brands-rust: Rust"

    ```rust
    use pecos_decoders::{BpOsdDecoder, Decoder};

    // Create CSS code
    let css_code = create_css_code()?;

    // Create decoder
    let mut decoder = BpOsdDecoder::new(css_code);

    // Configure parameters
    decoder.set_max_iterations(100);
    decoder.set_bp_method(BpMethod::MinSum);
    decoder.set_osd_order(10);

    // Decode syndrome
    let syndrome = vec![1, 0, 1, 0];
    let result = decoder.decode(&syndrome)?;

    println!("Correction: {:?}", result.correction);
    println!("Converged: {}", result.converged);
    ```

## LDPC Decoder Variants

### BP-OSD Decoder

Combines belief propagation with ordered statistics decoding post-processing.

=== ":fontawesome-brands-python: Python"

    ```python
    decoder = decoders.BpOsdDecoder(hx, hz)
    decoder.set_osd_method("exhaustive")  # or "combination_sweep"
    decoder.set_osd_order(10)
    ```

=== ":fontawesome-brands-rust: Rust"

    ```rust
    let mut decoder = BpOsdDecoder::new(css_code);
    decoder.set_osd_method(OsdMethod::Exhaustive);
    decoder.set_osd_order(10);
    ```

### BP-LSD Decoder

Localized version of OSD for better scaling with large codes.

=== ":fontawesome-brands-python: Python"

    ```python
    decoder = decoders.BpLsdDecoder(hx, hz)
    decoder.set_bits_per_step(1)
    decoder.set_lsd_order(10)
    ```

=== ":fontawesome-brands-rust: Rust"

    ```rust
    let mut decoder = BpLsdDecoder::new(css_code);
    decoder.set_bits_per_step(1);
    decoder.set_lsd_order(10);
    ```

### Belief Find Decoder

Combines belief propagation with union-find algorithm.

=== ":fontawesome-brands-python: Python"

    ```python
    decoder = decoders.BeliefFindDecoder(hx, hz)
    decoder.set_uf_method("inversion")  # or "peeling"
    decoder.set_max_bp_iterations(10)
    ```

=== ":fontawesome-brands-rust: Rust"

    ```rust
    let mut decoder = BeliefFindDecoder::new(css_code);
    decoder.set_uf_method(UfMethod::Inversion);
    decoder.set_max_bp_iterations(10);
    ```

### Flip Decoder

Fast bit-flipping decoder suitable for real-time applications.

=== ":fontawesome-brands-python: Python"

    ```python
    decoder = decoders.FlipDecoder(hx, hz)
    decoder.set_max_iterations(100)
    decoder.set_schedule("parallel")
    ```

=== ":fontawesome-brands-rust: Rust"

    ```rust
    let mut decoder = FlipDecoder::new(css_code);
    decoder.set_max_iterations(100);
    decoder.set_schedule(BpSchedule::Parallel);
    ```

### Union Find Decoder

Graph-based decoder using union-find data structure.

=== ":fontawesome-brands-python: Python"

    ```python
    decoder = decoders.UnionFindDecoder(hx, hz)
    decoder.set_uf_method("inversion")
    ```

=== ":fontawesome-brands-rust: Rust"

    ```rust
    let mut decoder = UnionFindDecoder::new(css_code);
    decoder.set_uf_method(UfMethod::Inversion);
    ```

## Advanced Features

### Soft Information Decoding

Use log-likelihood ratios for improved decoding performance.

=== ":fontawesome-brands-python: Python"

    ```python
    decoder = decoders.SoftInfoBpDecoder(hx, hz)

    # Provide soft information (LLRs)
    llrs = [0.1, -0.5, 0.8, -0.2]  # Log-likelihood ratios
    result = decoder.decode_with_llrs(syndrome, llrs)
    ```

=== ":fontawesome-brands-rust: Rust"

    ```rust
    let mut decoder = SoftInfoBpDecoder::new(css_code);

    // Provide soft information
    let llrs = vec![0.1, -0.5, 0.8, -0.2];
    let result = decoder.decode_with_llrs(&syndrome, &llrs)?;
    ```

### Batch Decoding

Decode multiple syndromes efficiently.

=== ":fontawesome-brands-python: Python"

    ```python
    # Multiple syndromes
    syndromes = [
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 1, 0, 0],
    ]

    results = decoder.decode_batch(syndromes)
    for i, result in enumerate(results):
        print(f"Syndrome {i}: {result.correction}")
    ```

=== ":fontawesome-brands-rust: Rust"

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

### Performance Tuning

#### Parallel Decoding

=== ":fontawesome-brands-python: Python"

    ```python
    decoder = decoders.BpOsdDecoder(hx, hz)
    decoder.set_schedule("parallel")  # Use parallel BP updates
    decoder.set_num_threads(4)  # Set thread count
    ```

=== ":fontawesome-brands-rust: Rust"

    ```rust
    let mut decoder = BpOsdDecoder::new(css_code);
    decoder.set_schedule(BpSchedule::Parallel);
    decoder.set_num_threads(4);
    ```

#### Memory Optimization

For large codes, use sparse representations:

=== ":fontawesome-brands-python: Python"

    ```python
    # Use sparse matrices for large codes
    from scipy.sparse import csr_matrix

    hx_sparse = csr_matrix(hx)
    hz_sparse = csr_matrix(hz)
    decoder = decoders.BpOsdDecoder(hx_sparse, hz_sparse)
    ```

=== ":fontawesome-brands-rust: Rust"

    ```rust
    // Sparse matrices are used by default
    let sparse_code = CssCode::from_sparse_matrices(hx_sparse, hz_sparse)?;
    let decoder = BpOsdDecoder::new(sparse_code);
    ```

## Error Handling

=== ":fontawesome-brands-python: Python"

    ```python
    try:
        result = decoder.decode(syndrome)
    except ValueError as e:
        print(f"Invalid syndrome: {e}")
    except RuntimeError as e:
        print(f"Decoding failed: {e}")
    ```

=== ":fontawesome-brands-rust: Rust"

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
