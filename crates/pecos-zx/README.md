# pecos-zx

ZX calculus integration for PECOS quantum error correction.

## Purpose

Provides ZX calculus capabilities for analyzing quantum circuits, built on top of [QuiZX](https://github.com/zxcalc/quizx).

## Modules

- `convert` - Circuit <-> ZX graph conversion (PECOS `DagCircuit` to/from QuiZX graphs)
- `graph` - ZX graph helpers and metadata
- `pauli_web` - Pauli web computation and classification
- `noise` - Noise model for annotating edges with error probabilities
- `dem` - Detector Error Model extraction from Pauli webs
- `viz` - SVG visualization of ZX diagrams
- `graph_state` - Graph state representation and entanglement analysis
- `symplectic` - Symplectic representation of Clifford unitaries
- `stabilizer` - Stabilizer <-> ZX connections (requires `stabilizer` feature)

## Re-exports

Key QuiZX types are re-exported for convenience: `ZxGraph`, `VType`, `EType`, `GraphLike`, along with `basic_rules`, `simplify`, and `zx_circuit` modules.
