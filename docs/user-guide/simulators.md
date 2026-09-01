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
| **QutritStateVec** | Qutrit state vector | Physical leakage trajectories, small systems | Rust / Python API |
| **QutritDensityMatrix** | Qutrit density matrix | Exact leakage/noise references, very small systems | Rust / Python API |
| **StabVec** | Clifford + Rz | Clifford circuits with Z rotations | None |
| **PauliProp** | Fault tracking | Error propagation analysis | None |
| **CuStateVec** | State vector (GPU, Python) | Large circuits with GPU | CUDA, cuQuantum |
| **CudaStateVec** | State vector (GPU, Rust) | Large circuits, reproducible seeded runs | CUDA, cuQuantum, cuda-rust build |
| **CudaStabilizer** | Stabilizer (GPU, Rust) | Very large Clifford circuits (1000s of qubits) | CUDA, cuQuantum, cuda-rust build |
| **MPS** | Tensor network | Low-entanglement circuits | CUDA, cuQuantum |
| **density_matrix** | Density matrix | Noisy/mixed state simulation | None |

Two additional specialized backends—**SparseStabPy** (pure-Python reference implementation for debugging) and **CoinToss** (random measurement outcomes for testing)—are documented below but rarely used in production.

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
    from pecos.simulators import sparse_stab

    results = sim(Qasm(circuit)).quantum(sparse_stab()).run(1000)
    ```

=== ":fontawesome-brands-rust: Rust"

    ```rust
    // SparseStab is used by default
    let results = sim(program.clone()).shots(1000).run()?;

    // Or explicitly select it
    let results = sim(program)
        .quantum(sparse_stab())
        .shots(1000).run()?;
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

state = SparseStabPy(num_qubits=2)
state.run_gate("H", {0})
state.run_gate("CNOT", {(0, 1)})
measurement = state.run_gate("Measure", {0})
```

### Stabilizer

Dense Rust stabilizer backend for Clifford circuits.

```python
from pecos.simulators import stabilizer

results = sim(Qasm(circuit)).quantum(stabilizer()).run(100)
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
    from pecos.simulators import state_vector

    results = sim(Qasm(circuit)).quantum(state_vector()).run(100)
    ```

=== ":fontawesome-brands-rust: Rust"

    ```rust
    let results = sim(program)
        .quantum(state_vector())
        .shots(100).run()?;
    ```

**Strengths:**

- Supports arbitrary gates (including T, rotation gates)
- Good baseline performance

### StabVec

Rust backend specialized for Clifford circuits plus Z-axis rotations.

```python
from pecos.simulators import stab_vec

results = sim(Qasm(circuit)).quantum(stab_vec()).run(100)
```

**Strengths:**

- Efficient for Clifford-heavy workloads that need `RZ` support
- Uses the native Rust backend that ships with PECOS

### Qutrit and Qudit Simulators

The Rust `QuditStateVec` and `QuditDensityMatrix` APIs simulate a uniform local
dimension rather than assuming qubits. Their qutrit wrappers fix the local
dimension to three, use the basis `|0>, |1>, |L>`, and support arbitrary local
unitaries, embedded qubit gates,
full and computational-subspace measurements, preparation and reset, and Kraus
channels. The state-vector backend samples channel trajectories, while the
density-matrix backend evolves mixed states exactly.

State-vector `reset_site` samples a trajectory branch and therefore consumes
randomness. Density-matrix `reset_site` applies the exact reset channel without
sampling.

Both backends support joint basis measurements, coarse-grained projective
partitions, and generalized measurement instruments expressed as outcome-grouped
Kraus operators. Trajectory samples expose branch probabilities, while exact
density-matrix measurements retain mixed conditional states. Imported density
operators are checked for normalization, Hermiticity, and positive
semidefiniteness.

These are reference backends for physical leakage and noise-model verification,
not replacements for PECOS's scalable stabilizer simulators or the classical
leakage bookkeeping used in large QEC studies. Storage scales as `3^N` for a
qutrit state vector and `9^N` for a qutrit density matrix.

The thin `pecos-rslib` bindings expose the same operations without a NumPy
runtime dependency. Python sequences of complex values are accepted directly:

```python
from pecos_rslib.simulators import QutritStateVec, qutrit_leakage_channel

state = QutritStateVec(1, seed=42)
sample = state.apply_kraus([0], qutrit_leakage_channel(0.01))
print(sample.operator_index, state.outcome_probabilities(0))
```

Supplying `seed` makes stochastic trajectories and measurements reproducible;
omitting it uses entropy-derived randomness.

### Errors

Every failure raises a subclass of `QuditError`, which also derives from the
builtin exception it would otherwise have been, so existing `except ValueError`,
`except IndexError`, and `except MemoryError` handlers keep working:

| Exception | Also derives from | Raised for |
| --- | --- | --- |
| `QuditValueError` | `ValueError` | invalid operators, channels, states, partitions, probabilities |
| `QuditIndexError` | `IndexError` | out-of-range target sites and basis states |
| `QuditMemoryError` | `MemoryError` | dense allocations that cannot be reserved |

Each carries a `kind` attribute holding a stable machine-readable tag, so a
caller can branch on a specific physical condition without matching on message
text:

```python
from pecos_rslib.simulators import QuditError, QutritStateVec

state = QutritStateVec(1, seed=42)
try:
    state.measure_computational(0)
except QuditError as error:
    if error.kind == "LeakagePopulation":
        ...  # the site carries population above |1>
    else:
        raise
```

### Index conventions

Both backends use one radix convention throughout, and getting it wrong produces
plausible but incorrect results rather than an error:

- **Site 0 is the least-significant radix digit** of a global basis index. For
  two qutrits, global index `g = digit(site 0) + 3 * digit(site 1)`.
- **`targets[0]` is the least-significant digit** of a local operator's row and
  column indices, and of a joint measurement outcome. `[0, 1]` and `[1, 0]` are
  therefore different operations, not the same one written two ways.
- **Operators, Kraus operators, and reduced density matrices are flat and
  row-major** — a `k`-site operator is a sequence of `d ** (2 * k)` complex
  values, not a nested list of rows.

The same statements are on the class docstrings, so `help(QutritStateVec)`
reaches them from a REPL.

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
# CUDA 13 (recommended); use [cuda12] for CUDA 12 / V100. Do not install both.
pip install "quantum-pecos[cuda13]"
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

results = sim(Qasm(circuit)).quantum(density_matrix()).run(100)
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
    from pecos.simulators import coin_toss

    # Test classical logic with random quantum outcomes
    results = sim(Qasm(circuit)).quantum(coin_toss()).run(1000)
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
| CudaStateVec | ★★★★ | ★★★★★ | 2^n (GPU) | ~30-35 |
| CudaStabilizer | ★★★★★ | N/A | Low | 1000+ (GPU) |
| MPS | ★★★ | ★★★ | ~n × chi² | Varies |
| density_matrix | ★★ | ★★ | 4^n | ~15 |

## Using Simulators with sim()

The `sim()` API lets you switch simulators easily:

=== ":fontawesome-brands-python: Python"

    ```python
    from pecos import sim, Qasm
    from pecos.simulators import sparse_stab, state_vector, stabilizer

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
    results = sim(circuit).quantum(sparse_stab()).run(1000)
    results = sim(circuit).quantum(state_vector()).run(1000)
    results = sim(circuit).quantum(stabilizer()).run(1000)
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
    let results = sim(circuit.clone()).shots(1000).run()?;

    // Explicit simulator selection
    let results = sim(circuit.clone())
        .quantum(state_vector())
        .shots(1000).run()?;

    let results = sim(circuit)
        .quantum(sparse_stab())
        .shots(1000).run()?;
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
