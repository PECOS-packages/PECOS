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
        Or a specific version: `pip install quantum-pecos==0.8.0.dev0`

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
        let sim = StdSparseStab::new(1);
        println!("PECOS is working! Created a {}-qubit simulator", sim.num_qubits());
    }
    ```

## Your First Simulation

Now that PECOS is installed, let's create a simple quantum circuit. We'll create a **Bell state**—a fundamental entangled state used throughout quantum computing and quantum error correction.

### What We're Building

A Bell state is created by:

1. Applying a Hadamard gate (H) to put a qubit in superposition
2. Applying a CNOT gate to entangle two qubits

The result is the state $\frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$, where measuring either qubit always gives the same result as the other.

### Running the Simulation

=== ":fontawesome-brands-python: Python"

    We'll use **Guppy**, a Python-embedded quantum programming language that offers type-safe qubit tracking and native control flow:

    ```python
    from guppylang import guppy
    from guppylang.std.quantum import h, cx, measure, qubit
    from pecos import sim, Guppy
    from pecos_rslib import state_vector


    @guppy
    def bell_state() -> tuple[bool, bool]:
        """Create and measure a Bell state."""
        q0 = qubit()
        q1 = qubit()
        h(q0)
        cx(q0, q1)
        return measure(q0), measure(q1)


    # Run 10 shots of the simulation
    results = sim(Guppy(bell_state)).qubits(2).quantum(state_vector()).seed(42).run(10)

    # View results
    print(f"Results: {results.to_dict()}")
    ```

=== ":fontawesome-brands-rust: Rust"

    In Rust, we load pre-compiled **HUGR** (Hierarchical Unified Graph Representation) files:

    ```rust
    use pecos_hugr::hugr_sim;

    fn main() -> Result<(), Box<dyn std::error::Error>> {
        // Load and run a pre-compiled Bell state circuit
        let results = hugr_sim("bell_state.hugr")
            .seed(42)
            .run(10)?;

        // View results
        for shot in &results.shots {
            println!("Measurement: {:?}", shot.data);
        }
        Ok(())
    }
    ```

    !!! note "HUGR Files"
        HUGR files are compiled from Guppy programs or other quantum tools.
        See [HUGR & Guppy Simulation](hugr-simulation.md) for how to generate them.

### Understanding the Output

Run the code multiple times (with different seeds). You'll notice:

- Results contain values like `0` (binary `00`) and `3` (binary `11`)
- Both qubits **always** have the same value—this is quantum entanglement!

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
