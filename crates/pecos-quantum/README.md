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

### HUGR angle limitation

HUGR conversion currently requires compile-time constant rotation angles.
Runtime-computed angles, a normal Guppy pattern, are not yet representable by
PECOS `Gate`; conversion rejects these operations rather than dropping the angle.
TODO(dynamic-angles): add representation and execution support for dynamic angles.

`Gate` constructors check native angle arity, and DAG insertion and mutation
validate gate payloads. `Gate` fields remain public: these checks do not make
malformed states unrepresentable. Full encapsulation of both the gate type and
angles is separate work.
