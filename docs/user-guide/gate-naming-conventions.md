# Gate Naming Conventions

PECOS uses systematic naming conventions for quantum gates that make it easy to understand what an operation does at a glance. This guide explains these conventions.

## Overview

Gate names in PECOS follow a pattern:

```
[Prefix][Base][Variant][Suffix]
```

Where:
- **Prefix**: Operation type (C for controlled, R for rotation, S for square root, M for measure, P for prepare)
- **Base**: The Pauli axis or gate type (X, Y, Z, H, F, etc.)
- **Variant**: Numbered variants (2, 3, 4, etc.) or negative indicator (N)
- **Suffix**: Modifier (dg for adjoint/dagger)

## Prefixes

### C - Controlled

The `C` prefix indicates a controlled operation where the first qubit controls the operation on the second.

| Gate | Meaning |
|------|---------|
| CX | Controlled-X (CNOT) |
| CY | Controlled-Y |
| CZ | Controlled-Z |

### R - Rotation

The `R` prefix indicates a parameterized rotation by an arbitrary angle θ.

| Gate | Meaning |
|------|---------|
| RX(θ) | Rotation around X-axis by θ |
| RY(θ) | Rotation around Y-axis by θ |
| RZ(θ) | Rotation around Z-axis by θ |
| RXX(θ) | Two-qubit XX rotation by θ |
| RYY(θ) | Two-qubit YY rotation by θ |
| RZZ(θ) | Two-qubit ZZ rotation by θ |

### S - Square Root

The `S` prefix indicates a square root operation (π/2 rotation). Applying the gate twice gives the base operation.

| Gate | Meaning | Relation |
|------|---------|----------|
| SX | Square root of X | SX · SX = X |
| SY | Square root of Y | SY · SY = Y |
| SZ | Square root of Z (also called S gate) | SZ · SZ = Z |
| SXX | Square root of XX interaction | SXX · SXX = XX (up to phase) |
| SYY | Square root of YY interaction | SYY · SYY = YY (up to phase) |
| SZZ | Square root of ZZ interaction | SZZ · SZZ = ZZ (up to phase) |

### M - Measure

The `M` prefix indicates a measurement operation.

| Gate | Meaning |
|------|---------|
| MX | Measure in X basis |
| MY | Measure in Y basis |
| MZ | Measure in Z basis (computational basis) |

### P - Prepare

The `P` prefix indicates state preparation.

| Gate | Meaning | Prepared State |
|------|---------|----------------|
| PX | Prepare +X eigenstate | \|+⟩ = (\|0⟩ + \|1⟩)/√2 |
| PY | Prepare +Y eigenstate | \|+i⟩ = (\|0⟩ + i\|1⟩)/√2 |
| PZ | Prepare +Z eigenstate | \|0⟩ |

### MP - Measure and Prepare

The `MP` prefix indicates a combined measure-and-prepare operation that measures then deterministically prepares the positive eigenstate.

| Gate | Meaning |
|------|---------|
| MPX | Measure X, then prepare \|+⟩ |
| MPY | Measure Y, then prepare \|+i⟩ |
| MPZ | Measure Z, then prepare \|0⟩ |

## Suffixes

### dg - Adjoint (Dagger)

The `dg` suffix indicates the adjoint (inverse) of a gate. For unitary gates, this is the conjugate transpose.

| Gate | Meaning |
|------|---------|
| Tdg | T† (inverse of T gate) |
| SXdg | SX† (inverse of SX) |
| SYdg | SY† (inverse of SY) |
| SZdg | SZ† (inverse of SZ, also called S†) |
| Fdg | F† (inverse of Face gate) |

**Property**: For any gate G with adjoint G†:
```
G · G† = G† · G = I
```

## Infixes

### N - Negative

The `N` infix indicates the negative version of an operation (measuring or preparing the -1 eigenstate instead of +1).

| Gate | Meaning | Eigenstate |
|------|---------|------------|
| MNX | Measure -X | Projects to \|+⟩ or \|-⟩, outcome flipped |
| MNY | Measure -Y | Projects to \|+i⟩ or \|-i⟩, outcome flipped |
| MNZ | Measure -Z | Projects to \|0⟩ or \|1⟩, outcome flipped |
| PNX | Prepare -X eigenstate | \|-⟩ = (\|0⟩ - \|1⟩)/√2 |
| PNY | Prepare -Y eigenstate | \|-i⟩ = (\|0⟩ - i\|1⟩)/√2 |
| PNZ | Prepare -Z eigenstate | \|1⟩ |
| MPNX | Measure -X, prepare \|-⟩ | |
| MPNY | Measure -Y, prepare \|-i⟩ | |
| MPNZ | Measure -Z, prepare \|1⟩ | |

## Numbered Variants

Some gates have numbered variants that represent different sign combinations or axis orientations.

### Hadamard Variants (H, H2-H6)

All six Hadamard-like gates that exchange pairs of Pauli axes:

| Gate | Pauli Transformation | Plane |
|------|---------------------|-------|
| H (H1) | X↔Z, Y→-Y | XZ plane |
| H2 | X↔-Z, Y→-Y | XZ plane (with signs) |
| H3 | X↔Y, Z→-Z | XY plane |
| H4 | X↔-Y, Z→-Z | XY plane (with signs) |
| H5 | Y↔Z, X→-X | YZ plane |
| H6 | Y↔-Z, X→-X | YZ plane (with signs) |

### Face Gate Variants (F, F2-F4)

The Face gates perform cyclic permutations of Pauli operators with different sign patterns:

| Gate | Pauli Transformation |
|------|---------------------|
| F (F1) | X→Y→Z→X (cyclic) |
| F2 | X→-Z, Y→-X, Z→Y |
| F3 | X→Y, Y→-Z, Z→-X |
| F4 | X→Z, Y→-Z, Z→-X |

Each also has an adjoint variant (Fdg, F2dg, F3dg, F4dg) that performs the inverse permutation.

## Base Gate Names

### Single-Letter Pauli Gates

| Gate | Name | Operation |
|------|------|-----------|
| I | Identity | No operation |
| X | Pauli-X | Bit flip (NOT) |
| Y | Pauli-Y | Bit and phase flip |
| Z | Pauli-Z | Phase flip |

### Other Base Gates

| Gate | Name | Description |
|------|------|-------------|
| H | Hadamard | Creates superposition, exchanges X↔Z |
| F | Face | Cyclic permutation X→Y→Z→X |
| G | G gate | Symmetric two-qubit Clifford |
| T | T gate | π/8 phase gate (= RZ(π/4)) |
| U | Universal | General single-qubit unitary U(θ,φ,λ) |
| SWAP | Swap | Exchange two qubit states |
| iSWAP | iSwap | Swap with i phase |

## Examples

Using these conventions, you can decode any PECOS gate name:

| Gate | Breakdown | Meaning |
|------|-----------|---------|
| `SZdg` | S(quare root) + Z + dg(agger) | Inverse of square root of Z |
| `CX` | C(ontrolled) + X | Controlled-X (CNOT) |
| `RZZ` | R(otation) + ZZ | Two-qubit ZZ rotation |
| `MPNX` | M(easure) + P(repare) + N(egative) + X | Measure -X and prepare \|-⟩ |
| `SXXdg` | S(quare root) + XX + dg(agger) | Inverse of square root of XX |
| `H5` | H(adamard variant) + 5 | Hadamard in YZ plane |
| `F2dg` | F(ace variant) + 2 + dg(agger) | Inverse of Face gate variant 2 |

## Historical Names vs Systematic Naming

Some gate names in PECOS come from historical conventions in quantum computing literature rather than the systematic naming scheme described above. This section clarifies these cases.

### The T Gate

The T gate is historically named but fits into the root hierarchy:

| Gate | Rotation | Relation |
|------|----------|----------|
| Z | RZ(π) | Base gate |
| SZ (S) | RZ(π/2) | Square root: SZ² = Z |
| T | RZ(π/4) | Fourth root: T² = SZ, T⁴ = Z |

Under a fully systematic scheme, T might be named `QZ` (quarter/fourth root of Z) or `FRZ` (fourth root of Z), but `T` is universally recognized in quantum computing and QASM.

**Equivalences** (up to global phase):
```
T = RZ(π/4)
T² = SZ = S = RZ(π/2)
T⁴ = Z = RZ(π)
```

### Gate Aliases

Several gates have multiple names due to historical conventions:

| Systematic Name | Historical Aliases | Notes |
|-----------------|-------------------|-------|
| SZ | S, P (phase gate) | All refer to RZ(π/2) |
| SZdg | S†, Sdg, P† | Inverse of S |
| CX | CNOT | Controlled-NOT |

PECOS generally accepts both forms where applicable.

## Design Principles

When extending PECOS gate names, these principles guide decisions:

1. **Consistency**: New names should follow existing patterns where possible
2. **Discoverability**: Names should be guessable from the pattern
3. **Compatibility**: Maintain QASM and literature compatibility for common gates
4. **Brevity**: Prefer short names for frequently-used gates
5. **Precision**: Avoid ambiguity in meaning

## API Consistency

These naming conventions are consistent across PECOS:

- **Rust API**: Methods use lowercase (e.g., `sim.szdg(q)`, `sim.mpnx(q)`)
- **Python API**: Gate names in `run_gate()` use uppercase (e.g., `"SZdg"`, `"CX"`)
- **QASM**: Standard QASM names where applicable (e.g., `h`, `cx`, `t`, `tdg`)

## See Also

- [Gate Reference](gates.md) - Complete gate documentation with matrices
- [Simulators](simulators.md) - Which simulators support which gates
- [Gate Naming: Future Considerations](gate-naming-speculation.md) - Speculative naming conventions under discussion
