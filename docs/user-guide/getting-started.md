# Getting Started

This guide will help you get up and running with PECOS quickly.

## Installation

=== ":fontawesome-brands-python: Python"

    ```bash
    pip install quantum-pecos
    ```

    That's it! This installs everything you need to start simulating quantum circuits.

    !!! note "Import Name"
        Import with `import pecos` (not `import quantum_pecos`).

    !!! tip "Pre-release versions"
        To install the latest development version from PyPI:
        ```bash
        pip install quantum-pecos --pre
        ```
        Or a specific version: `pip install quantum-pecos==0.8.0.dev1`

=== ":fontawesome-brands-rust: Rust"

    Add to your `Cargo.toml`:

    ```toml
    [dependencies]
    pecos = { version = "0.1", features = ["hugr"] }
    ```

    The `hugr` feature enables HUGR simulation. For QASM support, add `qasm`. See the [Rust API docs](https://docs.rs/pecos) for all available features.

## Verify Installation

=== ":fontawesome-brands-python: Python"
    ```python
    import pecos

    print(pecos.__version__)
    ```

=== ":fontawesome-brands-rust: Rust"
    ```rust
    use pecos::prelude::*;

    fn main() {
        let sim = SparseStab::new(1);
        println!("PECOS is working! Created a {}-qubit simulator", sim.num_qubits());
    }
    ```

## Your First Simulation

Now that PECOS is installed, let's simulate a quantum error correction circuit. We'll create a **distance-3 repetition code**—a fundamental building block for protecting quantum information from errors.

### What We're Building

A repetition code encodes a single logical qubit across multiple physical qubits:

1. **3 data qubits** store the logical state: $|0\rangle_L = |000\rangle$, $|1\rangle_L = |111\rangle$
2. **2 ancilla qubits** measure parity between adjacent data qubits (syndrome extraction)
3. **Noise** introduces random errors that the syndromes detect

### Running the Simulation

=== ":fontawesome-brands-python: Python"

    We'll use **Guppy**, a Python-embedded quantum programming language that offers type-safe qubit tracking and native control flow:

    ```python
    from pecos import Guppy, sim, state_vector, depolarizing_noise
    from guppylang import guppy
    from guppylang.std.quantum import qubit, cx, measure


    @guppy
    def repetition_code() -> None:
        # 3 data qubits encode logical |0⟩ = |000⟩
        d0 = qubit()
        d1 = qubit()
        d2 = qubit()

        # 2 ancillas for syndrome extraction
        s0 = qubit()
        s1 = qubit()

        # Measure parity between adjacent data qubits
        cx(d0, s0)
        cx(d1, s0)
        cx(d1, s1)
        cx(d2, s1)

        # Measure syndromes (first two measurements)
        _ = measure(s0)
        _ = measure(s1)

        # Measure data qubits (required by Guppy)
        _ = measure(d0), measure(d1), measure(d2)


    # Run 10 shots with 10% depolarizing noise
    noise = depolarizing_noise().with_uniform_probability(0.1)
    results = sim(Guppy(repetition_code)).qubits(5).quantum(state_vector()).noise(noise).seed(42).run(10)

    # Extract syndromes from first two measured qubits (s0, s1)
    d = results.to_dict()
    syndrome = [[d["q0"][i], d["q1"][i]] for i in range(10)]
    print(syndrome)
    # [[0, 0], [1, 0], [0, 0], [0, 0], [0, 0], [0, 1], [0, 1], [0, 0], [0, 0], [0, 0]]
    ```

=== ":fontawesome-brands-rust: Rust"

    In Rust, we load pre-compiled **HUGR** (Hierarchical Unified Graph Representation) files:

    ```hidden-rust
    use pecos_hugr::hugr_sim;
    use std::path::PathBuf;

    fn main() -> Result<(), Box<dyn std::error::Error>> {
        let mut hugr_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        hugr_path.push("../../../../../crates/pecos/tests/test_data/hugr/bell_state.hugr");

        // CODE
        Ok(())
    }
    ```

    ```rust
    use pecos_hugr::hugr_sim;

    // Load and run a pre-compiled HUGR circuit
    let results = hugr_sim(&hugr_path)
        .seed(42)
        .run(10)?;

    println!("Got {} shots", results.shots.len());
    ```

    !!! note "HUGR Files"
        HUGR files are compiled from Guppy programs or other quantum tools.
        See [HUGR & Guppy Simulation](hugr-simulation.md) for how to generate them.

### Understanding the Output

The syndromes tell you which errors occurred:

| Syndrome | Meaning |
|----------|---------|
| `[0, 0]` | No detected errors |
| `[1, 0]` | Error on qubit d0 (left edge) |
| `[0, 1]` | Error on qubit d2 (right edge) |
| `[1, 1]` | Error on qubit d1 (middle) |

A decoder uses these syndromes to identify and correct errors—see the [Decoders](decoders.md) guide.

The `sim()` function is PECOS's unified simulation API. It accepts circuits in various formats (Guppy, HUGR, QASM) and provides a builder pattern for configuration.

The Python example uses a state vector simulator, which supports all quantum gates. For Clifford-only circuits (H, S, CNOT, measurements), PECOS also provides efficient stabilizer simulators—see the [Simulators](simulators.md) guide.

!!! tip "Working with existing OpenQASM code?"
    PECOS also supports OpenQASM 2.0. See the [QASM Simulation](qasm-simulation.md) guide.

## Next Steps

- **[HUGR & Guppy Simulation](hugr-simulation.md)**: Measurement-based control flow and advanced Guppy features
- **[QASM Simulation](qasm-simulation.md)**: Full QASM simulation API for existing OpenQASM code
- **[Simulators](simulators.md)**: Choose the right simulation backend
- **[Noise Model Builders](noise-model-builders.md)**: Add realistic noise to your simulations
- **[Decoders](decoders.md)**: Explore quantum error correction decoding

## Optional Features

Most users won't need these, but they're available for specialized use cases:

| Feature | What it enables | Setup guide |
|---------|-----------------|-------------|
| **LLVM** (Rust only) | QIR/LLVM IR execution | [LLVM Setup](llvm-setup.md) |
| **CUDA** | GPU-accelerated simulation | [CUDA Setup](cuda-setup.md) |
| **QuEST** | Alternative simulator backend | `pip install quantum-pecos[all]` |

!!! tip "Python users"
    Pre-built wheels include LLVM support—no extra setup needed.

## Uninstalling

To remove PECOS:

=== ":fontawesome-brands-python: Python"

    ```bash
    pip uninstall quantum-pecos pecos-rslib
    ```

=== ":fontawesome-brands-rust: Rust"

    Remove the `pecos` dependency from your `Cargo.toml`.
