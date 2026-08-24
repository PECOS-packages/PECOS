# pecos-simulators

Quantum simulator traits and implementations for PECOS.

## Purpose

Defines simulator traits and provides native Rust quantum simulator implementations.

## Key Traits

- `QuantumSimulator` - Base simulator trait
- `CliffordGateable` - Clifford gate operations
- `ArbitraryRotationGateable` - Rotation gate operations

## Simulators

- `StateVec` - Full state vector simulator
- `DensityMatrix` - Density matrix simulator
- `QuditStateVec` / `QutritStateVec` - Physical multilevel state-vector trajectories
- `QuditDensityMatrix` / `QutritDensityMatrix` - Exact small-system multilevel evolution
- `SparseStab` - Sparse stabilizer simulator
- `SymbolicSparseStab` - Symbolic sparse stabilizer (tracks measurement history)
- `StabilizerTableauSimulator` - Tableau-based stabilizer simulator
- `CoinToss` - Simple coin-flip simulator for testing

## Utilities

- `MeasurementSampler` - Sample from symbolic measurement distributions
- `PauliProp` - Pauli propagation through circuits
- `Gens`, `SymbolicGens` - Generator representations
- `PhaseSign`, `SignAlgebra` - Sign algebra for stabilizer phases

## Qutrit and Qudit Simulation

The generalized simulators use a uniform local dimension. Their qutrit aliases use
the physical basis `|0>, |1>, |L>` and are designed for leakage studies and
independent noise-model verification:

```rust
use num_complex::Complex64;
use pecos_simulators::{QutritDensityMatrix, qutrit_leakage_channel};

let inv_sqrt_two = 1.0 / 2.0_f64.sqrt();
let h = [
    Complex64::new(inv_sqrt_two, 0.0),
    Complex64::new(inv_sqrt_two, 0.0),
    Complex64::new(inv_sqrt_two, 0.0),
    Complex64::new(-inv_sqrt_two, 0.0),
];

let mut state = QutritDensityMatrix::qutrit_with_seed(1, 42)?;
state
    .apply_embedded_qubit_unitary(0, &h)?
    .apply_kraus(&[0], &qutrit_leakage_channel(0.01)?)?;
let probabilities = state.outcome_probabilities(0)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Site zero is the least-significant radix digit. In a multi-site operation,
`targets[0]` is likewise the least-significant local-matrix digit. Arbitrary
one- and two-site unitaries can include coherent coupling to leakage levels;
embedded qubit unitaries leave all non-computational levels unchanged.

`QuditStateVec` stores `d^N` amplitudes and samples Kraus trajectories.
`QuditDensityMatrix` stores an exact `d^N` by `d^N` operator and applies Kraus
channels without sampling. The density-matrix backend is therefore intended only
for small reference problems. It provides reduced density matrices and numerical
trace, Hermiticity, and positive-semidefiniteness diagnostics.

Trajectory Kraus samples report both the selected operator and its pre-collapse
probability. Generalized measurement instruments group one or more Kraus
operators into each reported outcome; the state-vector backend samples an
individual pure-state branch while the density-matrix backend retains the exact
conditional mixed state. Joint basis measurements and coarse-grained projective
partitions are also supported on arbitrary target sets.

Constructing a density-matrix simulator from external data validates trace,
Hermiticity, and positive semidefiniteness by default. The `required_memory_bytes`
helpers estimate dense storage before construction, and internal dense allocations
return errors when their requested capacity cannot be reserved.

Full measurement reports any local basis level. Computational measurement is
strict: it returns an error when a site has population outside `|0>, |1>`, avoiding
an implicit device-specific rule for assigning leakage to a detector outcome.
