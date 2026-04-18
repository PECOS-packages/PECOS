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
| **SparseStab** | Stabilizer | QEC simulations, Clifford circuits | None (default) |
| **Stabilizer** | Stabilizer | Dense Clifford circuits | None |
| **StateVec** | State vector | Arbitrary circuits, small systems | None |
| **StabVec** | Clifford + Rz | Clifford circuits with Z rotations | None |
| **PauliProp** | Fault tracking | Error propagation analysis | None |
| **CuStateVec** | State vector (GPU) | Large circuits with GPU | CUDA, cuQuantum |
| **MPS** | Tensor network | Low-entanglement circuits | CUDA, cuQuantum |
| **density_matrix** | Density matrix | Noisy/mixed state simulation | None |

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
   SparseStab ←───┐      ┌─────┴─────┐          PauliProp
   Stabilizer    │      │           │
                 │   Small system?  GPU available?
                 │      │           │
                 │      ▼           ▼
                 │   StateVec    CuStateVec
                 │  StabVec      MPS
                 │      │
                 │      ▼
                 └── Need mixed states? ──→ density_matrix
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

### SparseStab (Recommended)

The default simulator, optimized for QEC workloads with sparse stabilizer tableaux.

=== ":fontawesome-brands-python: Python"

    ```python
    from pecos import sim, Qasm

    # SparseStab is used by default
    results = sim(Qasm(circuit)).run(1000)

    # Or explicitly select it
    from pecos.simulators import SparseStab

    results = sim(Qasm(circuit)).quantum(SparseStab).run(1000)
    ```

=== ":fontawesome-brands-rust: Rust"

    ```rust
    // SparseStab is used by default
    let results = sim(program.clone()).run(1000)?;

    // Or explicitly select it
    let results = sim(program)
        .quantum(sparse_stab())
        .run(1000)?;
    ```

**Strengths:**

- Fastest for LDPC codes and sparse circuits
- Efficient memory usage
- Pure Rust implementation

**Limitations:**

- Only Clifford gates (no T gates or arbitrary rotations)

### SparseStabPy (Python only)

Pure Python reference implementation—useful for learning and debugging but slower than `SparseStab`.

```python
from pecos.simulators import SparseStabPy

results = sim(Qasm(circuit)).quantum(SparseStabPy).run(100)
```

### Stabilizer

Dense Rust stabilizer backend for Clifford circuits.

```python
from pecos.simulators import Stabilizer

results = sim(Qasm(circuit)).quantum(Stabilizer).run(100)
```

**Strengths:**

- Rust backend with a straightforward dense stabilizer representation
- Good compatibility fallback for Clifford-only workloads

**Limitations:**

- Only Clifford gates
- Usually not as memory-efficient as `SparseStab` on sparse QEC circuits

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

### StabVec

Rust backend specialized for Clifford circuits plus Z-axis rotations.

```python
from pecos.simulators import StabVec

results = sim(Qasm(circuit)).quantum(StabVec).run(100)
```

**Strengths:**

- Efficient for Clifford-heavy workloads that need `RZ` support
- Uses the native Rust backend that ships with PECOS

## GPU-Accelerated Simulators

For large circuits, GPU acceleration can provide significant speedups.

### CuStateVec (Python only)

NVIDIA cuQuantum-powered state vector simulator.

<!--skip-if-no-cuda-->
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

<!--skip-if-no-cuda-->
```python
from pecos.simulators import MPS

results = sim(Qasm(circuit)).quantum(MPS).run(100)
```

**Strengths:**

- Can handle more qubits than state vector (for low-entanglement circuits)
- Configurable accuracy/speed tradeoff via bond dimension (`chi`)

**Requirements:** Same as CuStateVec (CUDA + cuQuantum)

## Density Matrix Simulators

Density matrix simulators represent mixed quantum states, enabling simulation of decoherence and non-unitary operations.

### density_matrix

```python
from pecos.simulators import density_matrix

results = sim(Qasm(circuit)).quantum(density_matrix).run(100)
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
    use pecos::simulators::{PauliProp, CliffordGateable};
    use pecos::QubitId;

    // Track how an X error on qubit 0 propagates
    let mut prop = PauliProp::new();
    prop.track_x(&[0]);  // Track an X error on qubit 0

    // Apply Hadamard - transforms X to Z
    prop.h(&[QubitId(0)]);

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
| SparseStab | ★★★★★ | N/A | Low | 1000+ |
| Stabilizer | ★★★★ | N/A | Medium | 1000+ |
| StateVec | ★★★ | ★★★ | 2^n | ~25-30 |
| StabVec | ★★★★ | Limited to Clifford + Rz | Low | 1000+ |
| CuStateVec | ★★★★ | ★★★★★ | 2^n (GPU) | ~30-35 |
| MPS | ★★★ | ★★★ | ~n × chi² | Varies |
| density_matrix | ★★ | ★★ | 4^n | ~15 |

## Using Simulators with sim()

The `sim()` API lets you switch simulators easily:

=== ":fontawesome-brands-python: Python"

    ```python
    from pecos import sim, Qasm
    from pecos.simulators import SparseStab, StateVec, Stabilizer

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

    # Default (SparseStab for Clifford circuits)
    results = sim(circuit).run(1000)

    # Explicit simulator selection
    results = sim(circuit).quantum(StateVec).run(1000)
    results = sim(circuit).quantum(Stabilizer).run(1000)
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
        .quantum(sparse_stab())
        .run(1000)?;
    ```

## Direct Simulator Access

For fine-grained control, you can use simulators directly:

=== ":fontawesome-brands-python: Python"

    ```python
    from pecos.simulators import SparseStab

    # Create simulator with 5 qubits
    state = SparseStab(5)

    # Apply gates using run_gate (qubits specified as sets)
    state.run_gate("H", {0})
    state.run_gate("CNOT", {(0, 1)})

    # Measure
    result = state.run_gate("Measure", {0})
    print(f"Qubit 0 measured: {result}")
    ```

=== ":fontawesome-brands-rust: Rust"

    ```rust
    use pecos::simulators::{SparseStab, CliffordGateable};
    use pecos::QubitId;

    // Create simulator with 5 qubits
    let mut state = SparseStab::new(5);

    // Apply gates
    state.h(&[QubitId(0)]);
    state.cx(&[(QubitId(0), QubitId(1))]);

    // Measure
    let results = state.mz(&[QubitId(0)]);
    println!("Qubit 0 measured: {}", results[0].outcome);

    // Inspect stabilizers
    println!("{:?}", state);
    ```

## Next Steps

- [QASM Simulation](qasm-simulation.md): Full guide to the simulation API
- [Noise Model Builders](noise-model-builders.md): Add noise to your simulations
- [CUDA Setup](cuda-setup.md): Configure GPU acceleration
