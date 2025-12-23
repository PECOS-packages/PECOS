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
    pecos = { version = "0.1", features = ["qasm"] }
    ```

    The `qasm` feature enables QASM simulation. For PHIR support, add `phir`. See the [Rust API docs](https://docs.rs/pecos) for all available features.

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

    ```python
    from pecos import sim, Qasm

    # Define a Bell state circuit in OpenQASM
    qasm_code = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    creg c[2];
    h q[0];
    cx q[0], q[1];
    measure q -> c;
    """

    # Run 10 shots of the simulation
    results = sim(Qasm(qasm_code)).seed(42).run(10)

    # View results (0 = both |0⟩, 3 = both |1⟩)
    print(f"Results: {results.to_dict()}")
    ```

=== ":fontawesome-brands-rust: Rust"

    ```rust
    use pecos::prelude::*;

    fn main() -> Result<(), Box<dyn std::error::Error>> {
        // Define a Bell state circuit in OpenQASM
        let qasm_code = r#"
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[2];
            creg c[2];
            h q[0];
            cx q[0], q[1];
            measure q -> c;
        "#;

        let program = Qasm::from_string(qasm_code);

        // Run 10 shots of the simulation
        let results = sim(program)
            .seed(42)
            .run(10)?;

        // View results
        println!("Results: {:?}", results);
        Ok(())
    }
    ```

### Understanding the Output

Run the code multiple times (with different seeds). You'll notice:

- Results contain values like `0` (binary `00`) and `3` (binary `11`)
- Both qubits **always** have the same value—this is quantum entanglement!

The `sim()` function is PECOS's unified simulation API. It accepts circuits in various formats (QASM, HUGR, etc.) and provides a builder pattern for configuration.

This demonstrates PECOS's stabilizer simulator, which efficiently simulates Clifford circuits (circuits using H, S, CNOT, and similar gates). Stabilizer simulation is the foundation for simulating quantum error correction codes.

## Next Steps

- **[QASM Simulation](qasm-simulation.md)**: Learn the full simulation API
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
