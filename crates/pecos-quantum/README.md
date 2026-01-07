# pecos-quantum

Quantum circuit representation data structures.

## Purpose

Provides quantum circuit representation data structures for PECOS, including DAG-based and tick-based circuit representations.

## Key Types

- `DagCircuit` - Quantum circuit as a directed acyclic graph
- `TickCircuit` - Quantum circuit as sequences of parallel time slices
- `Circuit`, `CircuitMut` - Circuit traits
- `Gate`, `GateType` - Gate representations

## Usage

```rust
use pecos_quantum::{DagCircuit, Gate, QubitId};

let mut circuit = DagCircuit::new();
let h = circuit.add_gate(Gate::h(&[0]));
let cx = circuit.add_gate(Gate::cx(&[(0, 1)]));
circuit.connect(h, cx, QubitId::from(0)).unwrap();
```
