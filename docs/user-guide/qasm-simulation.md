# QASM Simulations

```hidden-rust
use pecos::prelude::*;
use pecos::simulators::{sparse_stabilizer, state_vector};
use pecos::noise::GeneralNoiseModelBuilder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
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
    // CODE
    Ok(())
}
```

This guide will walk you through running quantum circuit simulations using PECOS's QASM interface. Whether you're simulating ideal quantum circuits or studying the effects of noise, PECOS provides the tools you need.

## What You'll Learn

- How to run your first QASM simulation
- Adding realistic noise models to your circuits
- Optimizing performance for large simulations
- Analyzing simulation results
- Choosing the right simulation engine for your needs

## Getting Started: Your First Simulation

Let's start with a simple example - creating and measuring a Bell state. First, we'll define our QASM code, which creates a Bell state by applying a Hadamard gate to the first qubit and then a CNOT gate to entangle both qubits:

```qasm
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
measure q -> c;
```

Now, let's run this code using PECOS's unified `sim()` function:

=== ":fontawesome-brands-python: Python"

    ```python
    from pecos import sim, Qasm

    # Define the Bell state QASM code
    qasm_code = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0], q[1];
        measure q -> c;
    """

    # Simple simulation
    results = sim(Qasm(qasm_code)).run(1000)

    # With configuration
    results = sim(Qasm(qasm_code)).seed(42).run(1000)
    ```

=== ":fontawesome-brands-rust: Rust"

    ```rust
    use pecos::prelude::*;

    // Define the Bell state QASM code
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

    // Simple simulation
    let results = sim(program.clone()).run(1000)?;

    // With configuration
    let results = sim(program).seed(42).run(1000)?;
    ```

## Using the Builder API

The `sim()` function returns a builder that provides flexibility through method chaining. You can configure seeds, workers, noise models, and more:

=== ":fontawesome-brands-python: Python"

    ```python
    from pecos import sim, Qasm, depolarizing_noise

    qasm_code = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0], q[1];
        measure q -> c;
    """

    # Simple simulation with builder pattern
    results = sim(Qasm(qasm_code)).run(1000)

    # With more configuration options
    results = (
        sim(Qasm(qasm_code))
        .seed(42)
        .noise(depolarizing_noise().with_uniform_probability(0.01))
        .workers(4)  # Explicitly set number of threads
        # .auto_workers()  # Or use all available CPU cores
        .run(1000)
    )
    ```

=== ":fontawesome-brands-rust: Rust"

    ```rust
    use pecos::prelude::*;

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

    // Simple simulation with builder pattern
    let results = sim(program.clone()).run(1000)?;

    // With more configuration options
    let results = sim(program)
        .seed(42)
        .noise(DepolarizingNoiseModel::builder().with_uniform_probability(0.01))
        .workers(4)  // Explicitly set number of threads
        // .auto_workers()  // Or use all available CPU cores
        .run(1000)?;
    ```

## Running Multiple Shots

Real quantum computers run circuits multiple times ("shots") to build up statistics. PECOS simulates this behavior and
lets you build the experiment once and rerun it multiple times:

=== ":fontawesome-brands-python: Python"

    ```python
    from pecos import sim, Qasm, depolarizing_noise

    qasm_code = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0], q[1];
        measure q -> c;
    """

    # Build once, run multiple times
    experiment = sim(Qasm(qasm_code)).seed(42).noise(depolarizing_noise().with_uniform_probability(0.01)).build()

    # Run with different shot counts
    results_100 = experiment.run(100)
    results_1000 = experiment.run(1000)
    ```

=== ":fontawesome-brands-rust: Rust"

    ```rust
    use pecos::prelude::*;

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

    let mut experiment = sim(program)
        .seed(42)
        .workers(4)
        .noise(DepolarizingNoiseModel::builder().with_uniform_probability(0.01))
        .build()?;

    // Run multiple times
    let results_100 = experiment.run(100)?;
    let results_1000 = experiment.run(1000)?;
    ```

## Adding Noise to Your Simulations

Real quantum computers are noisy. PECOS helps you understand how noise affects your circuits by providing several noise models.

### Common Noise Types

=== ":fontawesome-brands-python: Python"

    ```python
    from pecos import depolarizing_noise, biased_depolarizing_noise

    # No noise (ideal simulation) - simply don't add a noise model

    # Standard depolarizing with uniform probability
    depolarizing_noise().with_uniform_probability(0.01)

    # Custom depolarizing per operation type
    (
        depolarizing_noise()
        .with_prep_probability(0.001)  # State preparation error
        .with_meas_probability(0.002)  # Measurement error
        .with_p1_probability(0.003)  # Single-qubit gate error
        .with_p2_probability(0.004)  # Two-qubit gate error
    )

    # Biased depolarizing (asymmetric error distribution)
    biased_depolarizing_noise().with_uniform_probability(0.01)
    ```

=== ":fontawesome-brands-rust: Rust"

    ```rust
    use pecos::prelude::*;

    // No noise (ideal simulation)
    let _pass = PassThroughNoiseModel::builder();

    // Standard depolarizing
    let _depol = DepolarizingNoiseModel::builder()
        .with_uniform_probability(0.01);

    // Custom depolarizing per operation type
    let _custom = DepolarizingNoiseModel::builder()
        .with_prep_probability(0.001)  // State preparation error
        .with_meas_probability(0.002)  // Measurement error
        .with_p1_probability(0.003)    // Single-qubit gate error
        .with_p2_probability(0.004);   // Two-qubit gate error

    // Biased depolarizing (asymmetric error distribution)
    let _biased = BiasedDepolarizingNoiseModel::builder()
        .with_uniform_probability(0.01);
    ```

### Creating Custom Noise Models

For research or to match specific hardware characteristics, you can create detailed noise models:

=== ":fontawesome-brands-python: Python"

    ```python
    from pecos_rslib import GeneralNoiseModelBuilder

    # Direct builder usage
    noise = (
        GeneralNoiseModelBuilder()
        .with_prep_probability(0.001)  # State prep error
        .with_meas_0_probability(0.005)  # Measurement error |0> → |1>
        .with_meas_1_probability(0.01)  # Measurement error |1> → |0>
        .with_p1_probability(0.0001)  # Single-qubit gate error
        .with_p2_probability(0.01)  # Two-qubit gate error
        .with_seed(42)  # Deterministic noise
    )
    ```

=== ":fontawesome-brands-rust: Rust"

    ```rust
    use pecos::noise::GeneralNoiseModelBuilder;

    let noise = GeneralNoiseModelBuilder::new()
        .with_prep_probability(0.001)      // State prep error
        .with_meas_0_probability(0.005)    // Measurement error |0> → |1>
        .with_meas_1_probability(0.01)     // Measurement error |1> → |0>
        .with_p1_probability(0.0001)       // Single-qubit gate error
        .with_p2_probability(0.01)         // Two-qubit gate error
        .with_p_idle_linear_rate(0.0001)     // Idle noise rate
        .with_seed(42);                    // Deterministic noise

    // Use with sim()
    let results = sim(program).noise(noise).run(1000)?;
    ```

The builder provides many configuration options including idle noise rates, leakage probabilities,
Pauli error models, and more. For a comprehensive guide to using noise model builders, see the
[Noise Model Builders Guide](noise-model-builders.md).

## Choosing the Right Simulation Engine

PECOS provides different engines optimized for different types of circuits:

=== ":fontawesome-brands-python: Python"

    ```python
    from pecos import sim, Qasm
    from pecos_rslib import sparse_stabilizer, state_vector

    qasm_code = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0], q[1];
        measure q -> c;
    """

    # Sparse stabilizer (default, efficient for Clifford circuits)
    results = sim(Qasm(qasm_code)).quantum(sparse_stabilizer()).run(1000)

    # State vector (for non-Clifford circuits)
    results = sim(Qasm(qasm_code)).quantum(state_vector()).run(1000)
    ```

=== ":fontawesome-brands-rust: Rust"

    ```rust
    use pecos::prelude::*;
    use pecos::simulators::{sparse_stabilizer, state_vector};

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

    // Sparse stabilizer (default, efficient for Clifford circuits)
    let results = sim(program.clone())
        .quantum(sparse_stabilizer())
        .run(1000)?;

    // State vector (for non-Clifford circuits)
    let results = sim(program)
        .quantum(state_vector())
        .run(1000)?;
    ```

## Understanding Your Results

Simulation results come back as measurement outcomes for each shot. These can be processed in different ways depending on your needs:

=== ":fontawesome-brands-python: Python"

    ```python
    from pecos import sim, Qasm
    from collections import Counter

    qasm_code = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0], q[1];
        measure q -> c;
    """

    results = sim(Qasm(qasm_code)).run(1000)

    # Returns a ShotVec - convert to dict for easy access
    data = results.to_dict()
    print(data)
    # {"c": [0, 3, 0, 3, ...]}  # List of measurement outcomes

    # Each value is the decimal encoding of the binary string:
    # 0 = 00 (both qubits in |0⟩)
    # 1 = 01
    # 2 = 10
    # 3 = 11 (both qubits in |1⟩)

    # Count the occurrences of each measurement outcome
    counts = Counter(data["c"])
    print(counts)  # {0: 492, 3: 508} for an ideal Bell state
    ```

=== ":fontawesome-brands-rust: Rust"

    ```rust
    use pecos::prelude::*;

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
    let results = sim(program).run(1000)?;

    // Results come as ShotVec
    println!("Got {} shots", results.len());

    // Convert to ShotMap for columnar access
    let shot_map = results.try_as_shot_map()?;

    // Access measurement results by register name
    let c_values = shot_map.try_bits_as_u64("c")?;
    // Returns Vec<u64> where each value is the decimal encoding
    ```

## Practical Examples

### Example 1: Studying Noise Effects on Bell States

This example shows how noise affects quantum entanglement:

=== ":fontawesome-brands-python: Python"

    ```python
    from pecos import sim, Qasm, depolarizing_noise
    from collections import Counter

    qasm_code = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0], q[1];
        measure q -> c;
    """

    # Build simulation with depolarizing noise
    experiment = (
        sim(Qasm(qasm_code)).seed(42).workers(4).noise(depolarizing_noise().with_uniform_probability(0.01)).build()
    )

    # Run multiple times
    for shots in [100, 500, 1000]:
        results = experiment.run(shots)
        data = results.to_dict()
        print(f"Results for {shots} shots:")
        print(f"Counts: {Counter(data['c'])}")
    ```

=== ":fontawesome-brands-rust: Rust"

    ```rust
    use pecos::prelude::*;

    fn bell_state_example() -> Result<(), PecosError> {
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

        // Build simulation with depolarizing noise
        let mut experiment = sim(program)
            .seed(42)
            .workers(4)
            .noise(DepolarizingNoiseModel::builder().with_uniform_probability(0.01))
            .build()?;

        // Run multiple times
        for shots in [100, 500, 1000] {
            let results = experiment.run(shots)?;
            println!("Results for {} shots: {:?}", shots, results);
        }

        Ok(())
    }
    ```

### Example 2: Simulating a Noisy GHZ State

Here's how to simulate a GHZ state with realistic noise:

=== ":fontawesome-brands-python: Python"

    ```python
    from pecos import sim, Qasm, GeneralNoiseModelBuilder

    qasm_code = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        creg c[3];
        h q[0];
        cx q[0], q[1];
        cx q[1], q[2];
        measure q -> c;
    """

    # Create advanced noise model with builder
    noise = (
        GeneralNoiseModelBuilder()
        .with_prep_probability(0.001)  # 0.1% state prep error
        .with_p1_probability(0.0001)  # 0.01% single-qubit gate error
        .with_p2_probability(0.01)  # 1% two-qubit gate error
        .with_meas_0_probability(0.02)  # 2% false positive rate
        .with_meas_1_probability(0.03)  # 3% false negative rate
        .with_seed(12345)  # Deterministic noise
    )

    # Run simulation
    results = sim(Qasm(qasm_code)).noise(noise).seed(42).run(1000)
    print(f"GHZ state results: {results.to_dict()}")
    ```

=== ":fontawesome-brands-rust: Rust"

    ```rust
    use pecos::prelude::*;
    use pecos::noise::GeneralNoiseModelBuilder;

    fn ghz_noise_example() -> Result<(), PecosError> {
        let qasm_code = r#"
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[3];
            creg c[3];
            h q[0];
            cx q[0], q[1];
            cx q[1], q[2];
            measure q -> c;
        "#;

        let program = Qasm::from_string(qasm_code);

        // Create advanced noise model with builder
        let noise = GeneralNoiseModelBuilder::new()
            .with_prep_probability(0.001)      // 0.1% state prep error
            .with_p1_probability(0.0001)       // 0.01% single-qubit gate error
            .with_p2_probability(0.01)         // 1% two-qubit gate error
            .with_meas_0_probability(0.02)     // 2% false positive rate
            .with_meas_1_probability(0.03)     // 3% false negative rate
            .with_seed(12345);                 // Deterministic noise

        // Run simulation
        let results = sim(program).noise(noise).seed(42).run(1000)?;
        println!("GHZ state results: {:?}", results);

        Ok(())
    }
    ```

## Optimizing Your Simulations

### Parallel Execution

For many shots, you can use multiple CPU cores to speed up simulation:

=== ":fontawesome-brands-python: Python"

    ```python
    from pecos import sim, Qasm

    qasm_code = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0], q[1];
        measure q -> c;
    """

    # Single-threaded (default)
    results = sim(Qasm(qasm_code)).run(1000)

    # Use 4 worker threads
    results = sim(Qasm(qasm_code)).workers(4).run(1000)

    # Automatically use all available cores
    results = sim(Qasm(qasm_code)).auto_workers().run(1000)
    ```

=== ":fontawesome-brands-rust: Rust"

    ```rust
    use pecos::prelude::*;

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

    // Single threaded (default)
    let results = sim(program.clone()).workers(1).run(1000)?;

    // Explicit thread count
    let results = sim(program.clone()).workers(4).run(1000)?;

    // Automatically use all available cores
    let results = sim(program).auto_workers().run(1000)?;
    ```

### Choosing the Right Engine

- **For Clifford circuits** (H, S, CNOT, measurements): Use `SparseStabilizer` - it's exponentially faster
- **For circuits with T gates or rotations**: Use `StateVector`
- **Not sure?** The engine will be chosen based on the gates in your circuit

## Common Issues and Solutions

### Handling Errors

=== ":fontawesome-brands-python: Python"

    The API raises `RuntimeError` for invalid operations:
    ```python
    from pecos import sim, Qasm

    try:
        results = sim(Qasm("invalid qasm")).run(10)
    except RuntimeError as e:
        print(f"Error: {e}")
    ```

=== ":fontawesome-brands-rust: Rust"

    All methods return `Result<T, PecosError>`:

    - `build()` - Can fail during QASM parsing
    - `run()` - Can fail during simulation execution
    - `try_as_shot_map()` - Can fail during result conversion

## Next Steps

- **Learn more about QASM**: [OpenQASM 2.0 Specification](https://arxiv.org/abs/1707.03429)
- **Explore quantum algorithms**: Try implementing Grover's algorithm or QFT
- **Study noise**: Experiment with different noise models to understand their effects
- **Optimize performance**: Profile your simulations and choose appropriate engines

## Further Reading

- [Getting Started with PECOS](getting-started.md)
- [WASM Foreign Objects](wasm-foreign-objects.md) - Using WebAssembly for classical computation
- [Simulators Guide](simulators.md)
- [Noise Model Builders Guide](noise-model-builders.md)
- [PECOS Development Guide](../development/DEVELOPMENT.md)
