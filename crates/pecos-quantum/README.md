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

### Gate validation

`Gate` constructors check native angle arity, and DAG insertion and mutation
validate gate payloads. `Gate` fields remain public: these checks do not make
malformed states unrepresentable. Full encapsulation of both the gate type and
angles is separate work.

### Atomic tick insertion and lookup

`Tick::try_add_gates` checks the complete request, including overlaps between its
batches, before inserting any gate. It preserves command boundaries and returns
indices for attaching per-command metadata. Single-batch typed broadcasts use
`try_add_gate`, which likewise validates the entire batch before mutation.

`Tick::gate_on_qubit` uses a maintained per-qubit index to locate an occupying
batch in O(log n) time. Insertions, replacements, splitting, coalescing and
removals maintain that index; callers can read the returned batch and its
attributes without cloning the tick. These operations support Python's prepared
requests: only free batches are committed after native fields and decoded views
have established that every overlap is an identical application.

Single removal deletes the removed qubits and scans the index to decrement shifted
batch indices when removing before the last batch. Removing the last batch does
not scan the remaining index, including when a merge has already reassigned its
qubits to a surviving batch. Bulk removal (including discard and optimization
passes) compacts commands and attributes together and remaps the qubit index once.

Custom gate signatures are checked without mutation before insertion and are
registered only when the gate is accepted, in both typed and Python builders.
