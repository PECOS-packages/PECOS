# Circuit Representations (Internal)

This document covers the internal circuit representations used in PECOS's Rust core. For user-facing documentation on building and manipulating circuits, see the [User Guide: Circuit Representation](../user-guide/circuit-representation.md).

## Overview

PECOS uses four internal circuit representations, each optimized for different stages of the compilation and simulation pipeline:

| Representation | Level | Storage | Mutable | Primary Use |
|----------------|-------|---------|---------|-------------|
| `Hugr` | High-level IR | Hierarchical graph | External | Compilation input, interop |
| `SimpleHugr` | Validated wrapper | Pre-processed cache | No | Fast iteration over simple circuits |
| `DagCircuit` | DAG | Nodes + edges | Yes | Optimization, analysis, construction |
| `TickCircuit` | Time-sliced | Vec of ticks | Yes | Hardware scheduling, QEC |

## Hugr (Higher-order Unified Graph Representation)

HUGR is a standard intermediate representation developed alongside tket2 and guppylang. It represents the full semantics of hybrid quantum-classical programs.

### Capabilities

- **Control flow**: Conditionals, CFG nodes, function calls
- **Loops**: TailLoop nodes for iteration
- **Classical computation**: Arithmetic, logic, data structures
- **Hierarchical structure**: Nested regions, modules
- **Type system**: Linear types for qubits, classical types

### When HUGR is Used

1. **Input from Guppy**: Guppy compiles to HUGR bytecode
2. **Interoperability**: Exchange format with tket2 and other tools
3. **Dynamic programs**: Programs with runtime-dependent control flow

### Limitations for Simulation

HUGR's generality makes it complex to simulate directly. For simple quantum circuits (no control flow), we convert to `DagCircuit` or wrap in `SimpleHugr` for efficient access.

### Key Types

```rust
// From the `hugr` crate (external dependency)
use hugr::{Hugr, HugrView, Node, Wire};

// PECOS conversion functions
use pecos_quantum::hugr_convert::{
    hugr_to_dag_circuit,
    dag_circuit_to_hugr,
    SimpleHugr,
};
```

## SimpleHugr

A validated wrapper around HUGR that guarantees the circuit is "simple" (no control flow) and provides efficient access through the `Circuit` trait.

### Validation

Construction fails if the HUGR contains:
- `Conditional` nodes
- `TailLoop` nodes
- `CFG` nodes
- `Case` nodes

```rust
use pecos_quantum::hugr_convert::{SimpleHugr, NotSimpleError};

match SimpleHugr::try_new(hugr) {
    Ok(simple) => {
        // Safe to iterate efficiently
        for gate in simple.iter_gates_topo() {
            // ...
        }
    }
    Err(NotSimpleError::ContainsConditional) => {
        // Fall back to full HUGR execution
    }
    // ...
}
```

### Pre-computed Structure

On construction, `SimpleHugr` caches:
- Topological order of gates
- Predecessor/successor relationships
- Qubit-to-gate mappings
- Root and leaf gates
- Circuit depth

This avoids repeated graph traversals during simulation.

### When to Use

- When you receive a HUGR but expect it to be a simple circuit
- When you need `Circuit` trait compatibility without conversion overhead
- For read-only circuit analysis

## DagCircuit

The primary internal representation for circuit manipulation. Gates are nodes, qubit wires are labeled edges.

### Design

Follows the design of Qiskit's `DAGCircuit` and HUGR's dataflow regions:
- Edges represent qubit wires (not just dependencies)
- Each edge is labeled with the `QubitId` it carries
- Two-qubit gates have two incoming and two outgoing edges

### Capabilities

- **Mutable**: Add/remove gates, rewire connections
- **Rich queries**: Predecessors, successors, layers, qubit timelines
- **Attributes**: Metadata on circuit, gates, and wires
- **Builder API**: Fluent methods with auto-wiring

### Implementation Notes

```rust
pub struct DagCircuit {
    /// The underlying DAG structure (from pecos-num)
    dag: DAG,
    /// Gates stored by node index
    gates: Vec<Option<Gate>>,
    /// Qubit labels for each edge
    edge_qubits: BTreeMap<usize, QubitId>,
    /// Tracks the most recent gate on each qubit (for builder mode)
    qubit_heads: BTreeMap<QubitId, usize>,
    /// Last added node (for .meta() calls)
    last_node: Option<usize>,
}
```

The `qubit_heads` map enables the builder API to automatically wire consecutive gates on the same qubit.

## TickCircuit

A time-sliced representation where each "tick" contains gates that execute in parallel.

### Design

```rust
pub struct TickCircuit {
    ticks: Vec<Tick>,
    next_tick: usize,
    circuit_attrs: BTreeMap<String, Attribute>,
}

pub struct Tick {
    gates: Vec<Gate>,
    gate_attrs: BTreeMap<usize, BTreeMap<String, Attribute>>,
    attrs: BTreeMap<String, Attribute>,  // Tick-level metadata
}
```

### Qubit Conflict Detection

Each tick enforces that no qubit is used by multiple gates:

```rust
impl Tick {
    pub fn try_add_gate(&mut self, gate: Gate) -> Result<usize, QubitConflictError> {
        let conflicts = self.find_conflicts(&gate.qubits);
        if !conflicts.is_empty() {
            return Err(QubitConflictError { conflicting_qubits: conflicts, tick_idx: None });
        }
        Ok(self.add_gate(gate))
    }
}
```

### Use Cases

- **QEC syndrome extraction**: Each tick is a round
- **Hardware scheduling**: Maps to clocked execution
- **Timing metadata**: Attach round numbers, durations to ticks

## The Circuit Trait

Both `DagCircuit` and `SimpleHugr` implement the `Circuit` trait, enabling generic algorithms:

```rust
pub trait Circuit {
    // Basic properties
    fn gate_count(&self) -> usize;
    fn wire_count(&self) -> usize;
    fn qubits(&self) -> Vec<QubitId>;
    fn depth(&self) -> usize;

    // Gate access
    fn gate(&self, index: GateHandle) -> Option<&Gate>;
    fn iter_gates(&self) -> Box<dyn Iterator<Item = GateView<'_>> + '_>;
    fn iter_gates_topo(&self) -> Box<dyn Iterator<Item = GateView<'_>> + '_>;

    // Graph structure
    fn predecessors(&self, gate: GateHandle) -> Vec<GateHandle>;
    fn successors(&self, gate: GateHandle) -> Vec<GateHandle>;
    fn roots(&self) -> Vec<GateHandle>;
    fn leaves(&self) -> Vec<GateHandle>;

    // Qubit queries
    fn gates_on_qubit(&self, qubit: QubitId) -> Vec<GateHandle>;
    fn qubit_timeline(&self, qubit: QubitId) -> Vec<GateHandle>;

    // Attributes
    fn circuit_attrs(&self) -> &BTreeMap<String, Attribute>;
    fn gate_attrs(&self, gate: GateHandle) -> Option<&BTreeMap<String, Attribute>>;
}
```

### CircuitMut Trait

For mutable operations (only `DagCircuit` implements this):

```rust
pub trait CircuitMut: Circuit {
    fn add_gate(&mut self, gate: Gate) -> GateHandle;
    fn remove_gate(&mut self, gate: GateHandle) -> Option<Gate>;
    fn set_circuit_attr(&mut self, key: impl Into<String>, value: Attribute);
    fn set_gate_attr(&mut self, gate: GateHandle, key: impl Into<String>, value: Attribute) -> bool;
}
```

## Conversions

### Conversion Graph

```
         hugr_to_dag_circuit()
    Hugr ─────────────────────> DagCircuit <────> TickCircuit
      │                              ^               │
      │ try_new()                    │               │
      v                              │               │
  SimpleHugr ────────────────────────+               │
         (implements Circuit)                        │
                                                     │
                          From/Into traits ──────────+
```

### HUGR <-> DagCircuit

```rust
// HUGR to DagCircuit
let dag = hugr_to_dag_circuit(&hugr)?;

// DagCircuit to HUGR
let hugr = dag_circuit_to_hugr(&dag)?;
```

**HUGR -> DagCircuit algorithm:**
1. Extract quantum operations from tket.quantum extension
2. Process in topological order (QAlloc nodes first)
3. Track qubit identity through wire connections
4. Build edges based on qubit flow

**DagCircuit -> HUGR algorithm:**
1. Create DFG builder with qubit type signature
2. Process gates in topological order
3. Track wire mappings for each qubit
4. Handle rotation gates specially (add ConstRotation inputs)

### DagCircuit <-> TickCircuit

```rust
// DagCircuit to TickCircuit (layers become ticks)
let tick_circuit = TickCircuit::from(&dag_circuit);

// TickCircuit to DagCircuit (auto-wire by qubit)
let dag_circuit = DagCircuit::from(&tick_circuit);
```

**DagCircuit -> TickCircuit:**
- Each layer of parallel gates becomes a tick
- Gate attributes are preserved
- Tick-level attributes stored with `tick[N].key` prefix in DAG

**TickCircuit -> DagCircuit:**
- Gates added in tick order
- Consecutive gates on same qubit are wired
- Tick attributes restored from prefixed keys

### HUGR -> SimpleHugr

```rust
let simple = SimpleHugr::try_new(hugr)?;

// Access underlying HUGR if needed
let hugr_ref = simple.as_hugr();
let hugr_owned = simple.into_hugr();
```

## Performance Considerations

### When to Convert

| Scenario | Recommendation |
|----------|----------------|
| Single pass over gates | Use `SimpleHugr` (avoids conversion) |
| Multiple optimization passes | Convert to `DagCircuit` once |
| Need to modify circuit | Must use `DagCircuit` |
| Hardware scheduling | Convert to `TickCircuit` |
| Interop with tket | Keep as `Hugr` |

### Conversion Costs

- **HUGR -> DagCircuit**: O(n) where n = nodes, requires graph traversal
- **DagCircuit -> TickCircuit**: O(n + d) where d = depth (layer computation)
- **HUGR -> SimpleHugr**: O(n) validation + structure caching

### Memory

- `DagCircuit`: ~3 allocations per gate (node, gate storage, edge labels)
- `TickCircuit`: 1 Vec per tick + 1 Vec per gate
- `SimpleHugr`: Original HUGR + cached vectors

## Adding New Circuit Types

To add a new circuit representation:

1. **Implement `Circuit` trait** for read-only access
2. **Optionally implement `CircuitMut`** if mutable
3. **Add conversion functions** to/from `DagCircuit`
4. **Consider validation** (like `SimpleHugr::try_new`)

Example skeleton:

```rust
pub struct MyCircuit {
    // Internal storage
}

impl Circuit for MyCircuit {
    fn gate_count(&self) -> usize { /* ... */ }
    fn wire_count(&self) -> usize { /* ... */ }
    // ... implement all required methods
}

impl From<&DagCircuit> for MyCircuit {
    fn from(dag: &DagCircuit) -> Self { /* ... */ }
}

impl From<&MyCircuit> for DagCircuit {
    fn from(my: &MyCircuit) -> Self { /* ... */ }
}
```
