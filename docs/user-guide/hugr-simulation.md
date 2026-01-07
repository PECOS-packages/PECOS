# HUGR and Guppy Simulations

This guide walks you through running quantum circuit simulations using PECOS's HUGR interface and the Guppy quantum programming language. HUGR (Hierarchical Unified Graph Representation) is a modern intermediate representation for quantum programs that supports native control flow based on measurement results.

## What You'll Learn

- How to write quantum programs in Guppy
- Running simulations with `sim(Guppy(...))`
- Using pre-compiled HUGR files
- Measurement-based control flow (conditionals and loops)
- Choosing the right simulation engine
- Comparing HUGR vs QASM approaches

## Why HUGR and Guppy?

**Guppy** is a Python-embedded quantum programming language that compiles to HUGR. It offers:

- **Native Python syntax** - Write quantum programs using familiar Python constructs
- **Linear type system** - Catches qubit errors at compile time
- **Control flow** - Natural if/else and loops based on measurement results
- **No string parsing** - Direct compilation from Python functions

**HUGR** provides:

- **Rich control flow** - CFG-based representation for conditionals and loops
- **Composable** - Functions and modular program structure
- **Portable** - Standard format supported by multiple tools

## Getting Started: Your First Guppy Simulation

Let's create a Bell state using Guppy. First, define a quantum function:

=== ":fontawesome-brands-python: Python"

    ```python
    from guppylang import guppy
    from guppylang.std.quantum import h, cx, measure, qubit
    from pecos import sim, Guppy
    from pecos_rslib import state_vector


    # Define a Bell state circuit using Guppy
    @guppy
    def bell_state() -> tuple[bool, bool]:
        """Create and measure a Bell state."""
        q0 = qubit()
        q1 = qubit()

        # Create Bell state: H on q0, then CNOT
        h(q0)
        cx(q0, q1)

        # Measure both qubits
        return measure(q0), measure(q1)


    # Run simulation
    results = (
        sim(Guppy(bell_state)).qubits(2).quantum(state_vector()).seed(42).run(1000)
    )

    print(results.to_dict())
    # Results: always correlated (00 or 11)
    ```

=== ":fontawesome-brands-rust: Rust"

    ```rust
    use pecos_hugr::{hugr_engine, hugr_sim};
    use pecos_engines::{ClassicalControlEngineBuilder, ClassicalEngine};

    // Load a pre-compiled HUGR file
    let results = hugr_sim("bell_state.hugr")
        .seed(42)
        .run(1000)?;

    println!("Results: {:?}", results);
    ```

## Using the Guppy Builder API

The `sim(Guppy(...))` pattern returns a builder for configuration:

=== ":fontawesome-brands-python: Python"

    ```python
    from guppylang import guppy
    from guppylang.std.quantum import h, measure, qubit
    from pecos import sim, Guppy
    from pecos_rslib import state_vector, DepolarizingNoise


    @guppy
    def coin_flip() -> bool:
        """Quantum random bit."""
        q = qubit()
        h(q)
        return measure(q)


    # Simple simulation
    results = sim(Guppy(coin_flip)).qubits(1).quantum(state_vector()).run(100)

    # With configuration
    results = (
        sim(Guppy(coin_flip))
        .qubits(1)
        .quantum(state_vector())
        .seed(42)
        .noise(DepolarizingNoise(p=0.01))
        .run(1000)
    )
    ```

## Working with Pre-compiled HUGR Files

If you have HUGR files (compiled from Guppy or other tools), you can run them directly:

=== ":fontawesome-brands-python: Python"

    ```python
    from pecos import sim, Hugr

    # From file
    results = sim(Hugr.from_file("circuit.hugr")).run(1000)

    # From bytes
    with open("circuit.hugr", "rb") as f:
        hugr_bytes = f.read()
    results = sim(Hugr(hugr_bytes)).run(1000)
    ```

=== ":fontawesome-brands-rust: Rust"

    ```rust
    use pecos_hugr::{hugr_engine, hugr_sim};
    use pecos_engines::{ClassicalControlEngineBuilder, ClassicalEngine};

    // Quick simulation from file
    let results = hugr_sim("circuit.hugr")
        .seed(42)
        .run(1000)?;

    // Or use the builder for more control
    let engine = hugr_engine()
        .hugr_file("circuit.hugr")
        .build()?;

    println!("Circuit uses {} qubits", engine.num_qubits());
    ```

## Measurement-Based Control Flow

One of HUGR's key advantages is native support for control flow based on measurement results. This is natural in Guppy:

### Conditional Gates

=== ":fontawesome-brands-python: Python"

    ```python
    from guppylang import guppy
    from guppylang.std.quantum import h, x, measure, qubit
    from pecos import sim, Guppy
    from pecos_rslib import state_vector


    @guppy
    def conditional_x() -> tuple[bool, bool]:
        """Apply X gate conditionally based on measurement."""
        q0 = qubit()
        q1 = qubit()

        # Put q0 in superposition and measure
        h(q0)
        m0 = measure(q0)

        # Conditionally apply X to q1
        if m0:
            x(q1)

        m1 = measure(q1)
        return m0, m1


    # Run simulation
    results = (
        sim(Guppy(conditional_x)).qubits(2).quantum(state_vector()).seed(42).run(1000)
    )

    # Results: m0 and m1 are always equal!
    # - If m0=0: no X applied, m1=0
    # - If m0=1: X applied, m1=1
    data = results.to_dict()
    ```

### If-Else Branches

=== ":fontawesome-brands-python: Python"

    ```python
    from guppylang import guppy
    from guppylang.std.quantum import h, x, measure, qubit
    from pecos import sim, Guppy
    from pecos_rslib import state_vector


    @guppy
    def if_else_circuit() -> tuple[bool, bool]:
        """Different gates in each branch."""
        q0 = qubit()
        q1 = qubit()

        m0 = measure(q0)  # Always 0 (qubit starts in |0⟩)

        if m0:
            x(q1)  # This branch won't execute
        else:
            h(q1)  # This branch will execute

        m1 = measure(q1)
        return m0, m1


    results = (
        sim(Guppy(if_else_circuit)).qubits(2).quantum(state_vector()).seed(42).run(1000)
    )
    # m0 always 0, m1 is 50/50 (H applied)
    ```

### Loops

=== ":fontawesome-brands-python: Python"

    ```python
    from guppylang import guppy
    from guppylang.std.quantum import h, measure, qubit
    from pecos import sim, Guppy
    from pecos_rslib import state_vector


    @guppy
    def repeat_until_one() -> bool:
        """Repeat H+measure until we get 1."""
        result: bool = False
        while not result:
            q = qubit()
            h(q)
            result = measure(q)
        return result


    results = (
        sim(Guppy(repeat_until_one))
        .qubits(10)  # Allow enough qubits for iterations
        .quantum(state_vector())
        .seed(42)
        .run(100)
    )
    # Always returns True (loop exits when measure=1)
    ```

## Defining Helper Functions

Guppy supports modular quantum programs with helper functions:

=== ":fontawesome-brands-python: Python"

    ```python
    from guppylang import guppy
    from guppylang.std.builtins import owned
    from guppylang.std.quantum import h, cx, measure, qubit
    from pecos import sim, Guppy
    from pecos_rslib import state_vector


    # Define a reusable function
    @guppy
    def apply_h(q: qubit @ owned) -> qubit:
        """Apply Hadamard gate."""
        h(q)
        return q


    # Use it in another function
    @guppy
    def use_helper() -> bool:
        """Use the helper function."""
        q = qubit()
        q = apply_h(q)
        return measure(q)


    results = sim(Guppy(use_helper)).qubits(1).quantum(state_vector()).run(100)
    ```

## Choosing the Right Simulation Engine

HUGR programs work with different quantum backends:

=== ":fontawesome-brands-python: Python"

    ```python
    from pecos import sim, Guppy
    from pecos_rslib import state_vector, sparse_stabilizer

    # State vector - required for non-Clifford gates (T, rotations)
    results = sim(Guppy(my_circuit)).qubits(5).quantum(state_vector()).run(100)

    # Sparse stabilizer - efficient for Clifford circuits
    results = (
        sim(Guppy(clifford_circuit)).qubits(5).quantum(sparse_stabilizer()).run(100)
    )
    ```

| Engine | Best For | Gates Supported |
|--------|----------|-----------------|
| `state_vector()` | Universal circuits | All gates including T, rotations |
| `sparse_stabilizer()` | Clifford circuits | H, S, CNOT, measurements |

## Adding Noise

Add realistic noise to your Guppy simulations:

=== ":fontawesome-brands-python: Python"

    ```python
    from guppylang import guppy
    from guppylang.std.quantum import h, cx, measure, qubit
    from pecos import sim, Guppy
    from pecos_rslib import state_vector, DepolarizingNoise, GeneralNoiseModelBuilder


    @guppy
    def noisy_bell() -> tuple[bool, bool]:
        q0 = qubit()
        q1 = qubit()
        h(q0)
        cx(q0, q1)
        return measure(q0), measure(q1)


    # Simple depolarizing noise
    results = (
        sim(Guppy(noisy_bell))
        .qubits(2)
        .quantum(state_vector())
        .noise(DepolarizingNoise(p=0.01))
        .seed(42)
        .run(1000)
    )

    # Custom noise model
    noise = (
        GeneralNoiseModelBuilder()
        .with_prep_probability(0.001)
        .with_p1_probability(0.0001)
        .with_p2_probability(0.01)
        .with_meas_0_probability(0.02)
        .with_meas_1_probability(0.03)
    )

    results = (
        sim(Guppy(noisy_bell)).qubits(2).quantum(state_vector()).noise(noise).run(1000)
    )
    ```

## HUGR vs QASM: When to Use Each

| Feature | HUGR/Guppy | QASM |
|---------|------------|------|
| **Control flow** | Native if/else, loops | Limited (some extensions) |
| **Type safety** | Linear types catch errors | String-based, runtime errors |
| **Syntax** | Python-native | String DSL |
| **Composability** | Functions, modules | Limited |
| **Tooling** | Guppy compiler | Many parsers |

**Choose HUGR/Guppy when:**

- You need measurement-based control flow
- You want compile-time qubit tracking
- You prefer Python-native syntax
- You're building larger, modular programs

**Choose QASM when:**

- You have existing QASM code
- You need compatibility with other tools
- Your circuits don't need control flow
- You want a simple, portable format

## Understanding Results

Results from Guppy simulations work the same as QASM:

=== ":fontawesome-brands-python: Python"

    ```python
    from collections import Counter

    results = sim(Guppy(bell_state)).qubits(2).quantum(state_vector()).run(1000)

    # Convert to dictionary
    data = results.to_dict()
    # For a Bell state returning tuple[bool, bool]:
    # Results are encoded as integers
    # 0 = (False, False), 3 = (True, True)

    # Count occurrences
    print(Counter(data.values()))
    ```

## Common Issues and Solutions

### Qubit Limit Errors

If you see qubit allocation errors, increase the qubit limit:

```python
# Increase qubit pool for loops or dynamic allocation
results = sim(Guppy(my_circuit)).qubits(20).quantum(state_vector()).run(100)
```

### Missing guppylang

Install guppylang if not already installed:

```bash
pip install guppylang
```

### Type Errors in Guppy

Guppy enforces linear types. Each qubit must be used exactly once:

```python
@guppy
def bad_example() -> bool:
    q = qubit()
    # ERROR: q not consumed!
    return True


@guppy
def good_example() -> bool:
    q = qubit()
    return measure(q)  # q is consumed by measure
```

## Next Steps

- **[Guppy Language Guide](https://github.com/Quantinuum/guppylang)** - Full Guppy documentation
- **[QASM Simulation](qasm-simulation.md)** - Alternative simulation approach
- **[Noise Model Builders](noise-model-builders.md)** - Custom noise configurations
- **[Simulators](simulators.md)** - Available quantum backends

## Further Reading

- [HUGR Specification](https://github.com/Quantinuum/hugr)
- [Guppy GitHub Repository](https://github.com/Quantinuum/guppylang)
- [PECOS Development Guide](../development/DEVELOPMENT.md)
