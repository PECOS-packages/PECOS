# pecos-qsim

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
- `SparseStab` - Sparse stabilizer simulator
- `SymbolicSparseStab` - Symbolic sparse stabilizer (tracks measurement history)
- `StabilizerTableauSimulator` - Tableau-based stabilizer simulator
- `CoinToss` - Simple coin-flip simulator for testing

## Utilities

- `MeasurementSampler` - Sample from symbolic measurement distributions
- `PauliProp` - Pauli propagation through circuits
- `Gens`, `SymbolicGens` - Generator representations
- `PhaseSign`, `SignAlgebra` - Sign algebra for stabilizer phases
