# Simulators

PECOS provides multiple quantum simulation backends optimized for different use cases. This guide helps you choose the right simulator for your needs.

## Setup

Examples in this guide use a Bell state circuit:

=== ":fontawesome-brands-python: Python"

    ```python
    from pecos import sim, Qasm

    circuit = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    creg c[2];
    h q[0];
    cx q[0], q[1];
    measure q -> c;
    """
    ```

=== ":fontawesome-brands-rust: Rust"

    <!--skip-->
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
    ```

```hidden-python
from pecos import sim, Qasm

circuit = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
measure q -> c;
"""
```

```hidden-rust
use pecos::prelude::*;

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

## Quick Reference

| Simulator | Type | Best For | Requirements |
|-----------|------|----------|--------------|
| **SparseSim** | Stabilizer | QEC simulations, Clifford circuits | None (default) |
| **StateVec** | State vector | Arbitrary circuits, small systems | None |
| **Qulacs** | State vector | High-performance state vector | None |
| **PauliProp** | Fault tracking | Error propagation analysis | None |
| **CuStateVec** | State vector (GPU) | Large circuits with GPU | CUDA, cuQuantum |
| **MPS** | Tensor network | Low-entanglement circuits | CUDA, cuQuantum |
| **QuestStateVec** | State vector | Full state simulation | None |
| **QuestDensityMatrix** | Density matrix | Noisy/mixed state simulation | None |

## Choosing a Simulator

```
┌─────────────────────────────────────────────────────────────┐
│                    What are you simulating?                  │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
   Clifford only?      Arbitrary gates?      Error tracking?
        │                     │                     │
        ▼                     ▼                     ▼
   SparseSim ←───┐      ┌─────┴─────┐          PauliProp
   (fastest)     │      │           │
                 │   Small system?  GPU available?
                 │      │           │
                 │      ▼           ▼
                 │   StateVec    CuStateVec
                 │   Qulacs         MPS
                 │      │
                 │      ▼
                 └── Need mixed states? ──→ QuestDensityMatrix
```

## Setup

The examples below use this Bell state circuit:

```python
from pecos import sim, Qasm

circuit = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
measure q -> c;
"""
```

## Stabilizer Simulators

Stabilizer simulators efficiently simulate **Clifford circuits** (H, S, CNOT, CZ, and similar gates). They scale polynomially with qubit count, making them ideal for quantum error correction.

### SparseSim (Recommended)

The default simulator, optimized for QEC workloads with sparse stabilizer tableaux.

=== ":fontawesome-brands-python: Python"

    ```python
    from pecos import sim, Qasm

    # SparseSim is used by default
    results = sim(Qasm(circuit)).run(1000)

    # Or explicitly select it
    from pecos.simulators import SparseSim

    results = sim(Qasm(circuit)).quantum(SparseSim).run(1000)
    ```

=== ":fontawesome-brands-rust: Rust"

    ```rust
    // SparseSim is used by default
    let results = sim(program.clone()).run(1000)?;

    // Or explicitly select it
    let results = sim(program)
        .quantum(sparse_stabilizer())
        .run(1000)?;
    ```

**Strengths:**

- Fastest for LDPC codes and sparse circuits
- Efficient memory usage
- Pure Rust implementation

**Limitations:**

- Only Clifford gates (no T gates or arbitrary rotations)

### SparseSimPy (Python only)

Pure Python reference implementation—useful for learning and debugging but slower than `SparseSim`.

```python
from pecos.simulators import SparseSimPy

results = sim(Qasm(circuit)).quantum(SparseSimPy).run(100)
```

## State Vector Simulators

State vector simulators can simulate **any quantum circuit** but scale exponentially (2^n memory for n qubits). Practical for ~25-30 qubits on typical hardware.

### StateVec

Pure Rust state vector implementation.

=== ":fontawesome-brands-python: Python"

    ```python
    from pecos.simulators import StateVec

    results = sim(Qasm(circuit)).quantum(StateVec).run(100)
    ```

=== ":fontawesome-brands-rust: Rust"

    ```rust
    let results = sim(program)
        .quantum(state_vector())
        .run(100)?;
    ```

**Strengths:**

- Supports arbitrary gates (including T, rotation gates)
- Good baseline performance

### Qulacs (Python only)

High-performance state vector simulator via the Qulacs C++ library.

```python
from pecos.simulators import Qulacs

results = sim(Qasm(circuit)).quantum(Qulacs).run(100)
```

**Strengths:**

- Highly optimized C++ backend
- SIMD acceleration
- Often faster than StateVec for larger circuits

### QuestStateVec (Python only)

State vector simulator powered by the QuEST library.

<!--skip: requires QuEST library to be built-->
```python
from pecos.simulators import QuestStateVec

results = sim(Qasm(circuit)).quantum(QuestStateVec).run(100)
```

**Strengths:**

- Full state vector simulation with high precision
- Thread-safe: each instance operates on independent quantum registers

## GPU-Accelerated Simulators

For large circuits, GPU acceleration can provide significant speedups.

### CuStateVec (Python only)

NVIDIA cuQuantum-powered state vector simulator.

```python
from pecos.simulators import CuStateVec

results = sim(Qasm(circuit)).quantum(CuStateVec).run(100)
```

**Requirements:**

- NVIDIA GPU with CUDA support
- CUDA Toolkit 12+
- cuQuantum and cupy packages

```bash
pip install quantum-pecos[cuda]
```

See [CUDA Setup Guide](cuda-setup.md) for detailed installation instructions.

### MPS (Matrix Product State, Python only)

Tensor network simulator for circuits with limited entanglement.

<!--skip: requires pytket-cutensornet-->
```python
from pecos.simulators import MPS

# Configure bond dimension (higher = more accurate but slower)
mps = MPS(chi=64, truncation_fidelity=0.99)
results = sim(Qasm(circuit)).quantum(mps).run(100)
```

**Strengths:**

- Can handle more qubits than state vector (for low-entanglement circuits)
- Configurable accuracy/speed tradeoff via bond dimension (`chi`)

**Requirements:** Same as CuStateVec (CUDA + cuQuantum)

## Density Matrix Simulators

Density matrix simulators represent mixed quantum states, enabling simulation of decoherence and non-unitary operations.

### QuestDensityMatrix (Python only)

```python
from pecos.simulators import QuestDensityMatrix

results = sim(Qasm(circuit)).quantum(QuestDensityMatrix).run(100)
```

**Use cases:**

- Simulating noisy quantum channels
- Mixed state preparation
- Non-unitary operations

!!! warning "Memory Usage"
    Density matrices scale as 4^n (vs 2^n for state vectors), limiting practical use to ~15 qubits.

## Specialized Simulators

### PauliProp (Pauli Propagation)

Tracks how Pauli errors propagate through Clifford circuits—essential for QEC analysis.

=== ":fontawesome-brands-python: Python"

    ```python
    from pecos.simulators import PauliProp

    # Track how an X error on qubit 0 propagates
    prop = PauliProp(num_qubits=5)
    # ... apply gates ...
    # Check resulting error pattern
    ```

=== ":fontawesome-brands-rust: Rust"

    ```rust
    use pecos::qsim::{StdPauliProp, CliffordGateable};

    // Track how an X error on qubit 0 propagates
    let mut prop = StdPauliProp::new();
    prop.add_x(0);  // Track an X error on qubit 0

    // Apply Hadamard - transforms X to Z
    prop.h(0);

    // Check resulting error pattern
    assert!(prop.contains_z(0));  // X transformed to Z
    assert!(!prop.contains_x(0)); // No longer has X
    ```

**Use cases:**

- Fault tolerance analysis
- Decoder development
- Understanding error propagation in QEC codes

### CoinToss

Returns random measurement results, ignoring all gates. Useful for testing.

=== ":fontawesome-brands-python: Python"

    ```python
    from pecos.simulators import CoinToss

    # Test classical logic with random quantum outcomes
    results = sim(Qasm(circuit)).quantum(CoinToss).run(1000)
    ```

=== ":fontawesome-brands-rust: Rust"

    ```rust
    use pecos_engines::{QuantumEngineBuilder, coin_toss};

    // Test classical logic with random quantum outcomes
    // CoinToss ignores all gates and returns random measurements
    let mut builder = coin_toss().qubits(2);
    let engine = builder.build()?;
    // Engine is ready for processing quantum operations
    ```

**Use cases:**

- Testing error correction decoders
- Debugging classical control flow
- Benchmarking without quantum overhead

## Performance Comparison

Approximate performance characteristics (relative, not absolute):

| Simulator | Speed (Clifford) | Speed (Universal) | Memory | Max Qubits |
|-----------|------------------|-------------------|--------|------------|
| SparseSim | ★★★★★ | N/A | Low | 1000+ |
| StateVec | ★★★ | ★★★ | 2^n | ~25-30 |
| Qulacs | ★★★★ | ★★★★ | 2^n | ~25-30 |
| CuStateVec | ★★★★ | ★★★★★ | 2^n (GPU) | ~30-35 |
| MPS | ★★★ | ★★★ | ~n × chi² | Varies |
| QuestDensityMatrix | ★★ | ★★ | 4^n | ~15 |

## Using Simulators with sim()

The `sim()` API lets you switch simulators easily:

=== ":fontawesome-brands-python: Python"

    ```python
    from pecos import sim, Qasm
    from pecos.simulators import SparseSim, StateVec, Qulacs

    circuit = Qasm(
        """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0], q[1];
        measure q -> c;
    """
    )

    # Default (SparseSim for Clifford circuits)
    results = sim(circuit).run(1000)

    # Explicit simulator selection
    results = sim(circuit).quantum(StateVec).run(1000)
    results = sim(circuit).quantum(Qulacs).run(1000)
    ```

=== ":fontawesome-brands-rust: Rust"

    ```rust
    let circuit = Qasm::from_string(r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0], q[1];
        measure q -> c;
    "#);

    // Default (sparse stabilizer for Clifford circuits)
    let results = sim(circuit.clone()).run(1000)?;

    // Explicit simulator selection
    let results = sim(circuit.clone())
        .quantum(state_vector())
        .run(1000)?;

    let results = sim(circuit)
        .quantum(sparse_stabilizer())
        .run(1000)?;
    ```

## Direct Simulator Access

For fine-grained control, you can use simulators directly:

=== ":fontawesome-brands-python: Python"

    ```python
    from pecos.simulators import SparseSim

    # Create simulator with 5 qubits
    state = SparseSim(5)

    # Apply gates using run_gate (qubits specified as sets)
    state.run_gate("H", {0})
    state.run_gate("CNOT", {(0, 1)})

    # Measure
    result = state.run_gate("Measure", {0})
    print(f"Qubit 0 measured: {result}")
    ```

=== ":fontawesome-brands-rust: Rust"

    ```rust
    // Create simulator with 5 qubits
    let mut state = StdSparseStab::new(5);

    // Apply gates (methods are chainable)
    state.h(0);
    state.cx(0, 1);

    // Measure
    let result = state.mz(0);
    println!("Qubit 0 measured: {}", result.outcome);

    // Inspect stabilizers
    println!("{:?}", state);
    ```

## Next Steps

- [QASM Simulation](qasm-simulation.md): Full guide to the simulation API
- [Noise Model Builders](noise-model-builders.md): Add noise to your simulations
- [CUDA Setup](cuda-setup.md): Configure GPU acceleration
