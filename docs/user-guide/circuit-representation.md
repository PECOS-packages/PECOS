# Circuit Representation

```hidden-rust
use pecos::quantum::{DagCircuit, TickCircuit, Attribute};
use pecos::core::{Gate, QubitId};
use pecos::dag::DAG;
use pecos::digraph::DiGraph;

fn main() {
    // CODE
}
```

PECOS provides several ways to represent and work with quantum circuits, from high-level program formats to low-level data structures.

## Quick Guide: What Should I Use?

| I want to... | Use this |
|--------------|----------|
| Simulate a Guppy function | `sim(Guppy(my_func))` |
| Simulate a QASM program | `sim(Qasm("..."))` |
| Build a circuit programmatically | `DagCircuit` |
| Schedule gates with explicit timing | `TickCircuit` |
| Work with QEC syndrome rounds | `TickCircuit` |
| Analyze circuit depth/width | `DagCircuit` |

## Program Types

When using PECOS's `sim()` API, you wrap your program in one of these types:

| Type | Input Format | Use Case |
|------|--------------|----------|
| `Guppy` | Guppy-decorated function | Pythonic circuit construction (recommended) |
| `Qasm` | OpenQASM 2.0 string | Standard quantum circuits |
| `Hugr` | HUGR binary bytes | Compiled programs, interop with tket |
| `Qis` | LLVM IR string | Low-level compiled programs |
| `PhirJson` | PHIR JSON string | Experimental; easily serializable, simulator/QEC friendly |
| `Wasm` / `Wat` | WebAssembly | Foreign functions (e.g., decoders) written in Rust/C/C++ for hybrid execution |

### Example: Different Program Types

=== ":fontawesome-brands-python: Python"
    ```python
    from pecos import sim, Guppy, Qasm, state_vector

    # Guppy - recommended for new code
    from guppylang import guppy
    from guppylang.std.quantum import qubit, h, cx, measure


    @guppy
    def bell_state() -> tuple[bool, bool]:
        q0, q1 = qubit(), qubit()
        h(q0)
        cx(q0, q1)
        return measure(q0), measure(q1)


    results = sim(Guppy(bell_state)).qubits(2).quantum(state_vector()).run(100)

    # QASM - for existing circuits
    results = sim(
        Qasm(
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
    ).run(100)
    ```

    For HUGR files compiled separately (requires actual file):

    <!--expect-error: FileNotFoundError.*program\.hugr-->
    ```python
    from pecos import sim, Hugr

    # HUGR - from compiled output (fails if file doesn't exist)
    results = sim(Hugr.from_file("program.hugr")).run(100)
    ```

## Circuit Data Structures

For programmatic circuit construction and analysis, PECOS provides these data structures:

| Type | Purpose | Use Case |
|------|---------|----------|
| `DagCircuit` | DAG of quantum gates | Circuit optimization, resource estimation, noise modeling |
| `TickCircuit` | Time-sliced circuit | Explicit timing, parallel gate scheduling, QEC |
| `DiGraph` | Directed graph | General directed graph algorithms |
| `DAG` | Directed acyclic graph | Topological ordering, dependency tracking |
| `Graph` | Undirected graph | Matching, shortest paths (see [Graph API](graph-api.md)) |

### DagCircuit vs TickCircuit

**DagCircuit** represents circuits as a directed acyclic graph where:

- Nodes are gates
- Edges are qubit wires connecting gates
- Parallelism is implicit (independent gates can run together)

**TickCircuit** represents circuits as a sequence of time slices where:

- Each tick contains gates that run in parallel
- Qubits cannot be reused within the same tick
- Parallelism is explicit

```
DagCircuit (implicit parallelism):     TickCircuit (explicit parallelism):

    H(q0) ──┐                          Tick 0: H(q0), H(q1)
            ├── CX(q0,q1)              Tick 1: CX(q0,q1)
    H(q1) ──┘                          Tick 2: Mz(q0), Mz(q1)
            │
    Mz(q0) ─┘
    Mz(q1) ─┘
```

Choose `DagCircuit` when you want automatic parallelism detection. Choose `TickCircuit` when you need explicit control over timing (e.g., QEC syndrome extraction rounds).

## DagCircuit

A directed acyclic graph representation where nodes are gates and edges are qubit wires. This design follows HUGR and Qiskit's `DAGCircuit`.

### Quick Start

=== ":fontawesome-brands-python: Python"
    ```python
    from pecos.quantum import DagCircuit

    # Fluent builder API
    circuit = DagCircuit()
    circuit.h([0]).cx([(0, 1)]).rz(0.5, [0]).mz([0])

    # Query properties
    print(f"Gates: {circuit.gate_count()}")
    print(f"Depth: {circuit.depth()}")
    print(f"Width: {circuit.width()}")
    ```

=== ":fontawesome-brands-rust: Rust"
    ```rust
    use pecos::quantum::DagCircuit;

    // Fluent builder API
    let mut circuit = DagCircuit::new();
    circuit.h(&[0]).cx(&[(0, 1)]).rz(0.5, &[0]).mz(&[0]);

    // Query properties
    println!("Gates: {}", circuit.gate_count());
    println!("Depth: {}", circuit.depth());
    println!("Width: {}", circuit.width());
    ```

### Building Circuits

The fluent API automatically wires gates on the same qubit:

=== ":fontawesome-brands-python: Python"
    ```python
    from pecos.quantum import DagCircuit

    circuit = DagCircuit()

    # Single-qubit gates
    circuit.h([0])  # Hadamard
    circuit.x([1])  # Pauli X
    circuit.y([2])  # Pauli Y
    circuit.z([3])  # Pauli Z
    circuit.sz([0])  # S gate (sqrt Z)
    circuit.szdg([0])  # S-dagger
    circuit.t([0])  # T gate
    circuit.tdg([0])  # T-dagger

    # Rotation gates (angle in radians)
    circuit.rx(3.14159, [0])  # RX
    circuit.ry(1.5708, [1])  # RY
    circuit.rz(0.7854, [2])  # RZ

    # Two-qubit gates
    circuit.cx([(0, 1)])  # CNOT (control, target)
    circuit.szz([(0, 1)])  # sqrt ZZ
    circuit.rzz(0.5, [(0, 1)])  # RZZ rotation

    # Measurement and preparation
    circuit.mz([0])  # Measure in Z basis
    circuit.pz([1])  # Prepare in Z basis (|0>)

    # Chaining
    circuit.h([0]).cx([(0, 1)]).h([0]).mz([0])
    ```

=== ":fontawesome-brands-rust: Rust"
    ```rust
    use pecos::quantum::DagCircuit;

    let mut circuit = DagCircuit::new();

    // Single-qubit gates
    circuit.h(&[0]);       // Hadamard
    circuit.x(&[1]);       // Pauli X
    circuit.y(&[2]);       // Pauli Y
    circuit.z(&[3]);       // Pauli Z
    circuit.sz(&[0]);      // S gate
    circuit.szdg(&[0]);    // S-dagger
    circuit.t(&[0]);       // T gate
    circuit.tdg(&[0]);     // T-dagger

    // Rotation gates (angle in radians)
    circuit.rx(3.14159, &[0]);
    circuit.ry(1.5708, &[1]);
    circuit.rz(0.7854, &[2]);

    // Two-qubit gates
    circuit.cx(&[(0, 1)]);
    circuit.szz(&[(0, 1)]);
    circuit.rzz(0.5, &[(0, 1)]);

    // Measurement and preparation
    circuit.mz(&[0]);
    circuit.pz(&[1]);

    // Chaining
    circuit.h(&[0]).cx(&[(0, 1)]).h(&[0]).mz(&[0]);
    ```

### Adding Metadata

Gates can have arbitrary metadata attached:

=== ":fontawesome-brands-python: Python"
    ```python
    from pecos.quantum import DagCircuit

    circuit = DagCircuit()

    # Attach metadata to the last gate
    circuit.h([0]).meta("error_rate", 0.001)

    # Multiple metadata entries
    circuit.cx([(0, 1)]).meta("duration_ns", 50)

    # Measurements return refs (not the circuit), so chain separately
    circuit.mz([0])
    circuit.meta("basis", "Z")
    ```

=== ":fontawesome-brands-rust: Rust"
    ```rust
    use pecos::quantum::{DagCircuit, Attribute};

    let mut circuit = DagCircuit::new();

    // Attach metadata to the last gate
    circuit.h(&[0]).meta("error_rate", Attribute::Float(0.001));

    // Multiple metadata entries
    circuit.cx(&[(0, 1)]).meta("duration_ns", Attribute::Int(50));

    // Measurements return refs (not &mut Self), so chain separately
    circuit.mz(&[0]);
    circuit.meta("basis", Attribute::String("Z".into()));
    ```

### Circuit Analysis

=== ":fontawesome-brands-python: Python"
    ```python
    from pecos.quantum import DagCircuit

    circuit = DagCircuit()
    circuit.h([0]).cx([(0, 1)]).h([1]).cx([(1, 2)])
    circuit.mz([0])
    circuit.mz([1])
    circuit.mz([2])

    # Basic metrics
    print(f"Total gates: {circuit.gate_count()}")
    print(f"Circuit depth: {circuit.depth()}")
    print(f"Circuit width: {circuit.width()}")
    print(f"Qubits used: {circuit.qubits()}")

    # Gate counts
    print(f"Single-qubit gates: {circuit.single_qubit_gate_count()}")
    print(f"Two-qubit gates: {circuit.two_qubit_gate_count()}")

    # Topological iteration
    for node_id in circuit.topological_order():
        gate = circuit.gate(node_id)
        print(f"Node {node_id}: {gate}")

    # Layer iteration (parallel gates)
    for i, layer in enumerate(circuit.layers()):
        print(f"Layer {i}: {layer}")
    ```

=== ":fontawesome-brands-rust: Rust"
    ```rust
    let mut circuit = DagCircuit::new();
    circuit.h(&[0]).cx(&[(0, 1)]).h(&[1]).cx(&[(1, 2)]);
    circuit.mz(&[0]);
    circuit.mz(&[1]);
    circuit.mz(&[2]);

    // Basic metrics
    println!("Total gates: {}", circuit.gate_count());
    println!("Circuit depth: {}", circuit.depth());
    println!("Circuit width: {}", circuit.width());
    println!("Qubits used: {:?}", circuit.qubits());

    // Gate counts
    println!("Single-qubit gates: {}", circuit.single_qubit_gate_count());
    println!("Two-qubit gates: {}", circuit.two_qubit_gate_count());

    // Topological iteration
    for node_id in circuit.topological_order() {
        if let Some(gate) = circuit.gate(node_id) {
            println!("Node {}: {:?}", node_id, gate);
        }
    }

    // Layer iteration
    for (i, layer) in circuit.layers().enumerate() {
        println!("Layer {}: {:?}", i, layer);
    }
    ```

### Manual Wiring

For advanced use cases, you can manually add gates and wire them:

=== ":fontawesome-brands-python: Python"
    ```python
    from pecos.quantum import DagCircuit, Gate

    circuit = DagCircuit()

    # Add gates manually
    h_node = circuit.add_gate(Gate.h([0]))
    cx_node = circuit.add_gate(Gate.cx([(0, 1)]))

    # Connect gates on qubit 0
    circuit.connect(h_node, cx_node, 0)

    # Query connections
    print(f"Predecessors of CX: {circuit.predecessors(cx_node)}")
    print(f"Successors of H: {circuit.successors(h_node)}")
    ```

=== ":fontawesome-brands-rust: Rust"
    ```rust
    use pecos::quantum::DagCircuit;
    use pecos::core::{Gate, QubitId};

    let mut circuit = DagCircuit::new();

    // Add gates manually
    let h_node = circuit.add_gate(Gate::h(&[0]));
    let cx_node = circuit.add_gate(Gate::cx(&[(0, 1)]));

    // Connect gates on qubit 0
    circuit.connect(h_node, cx_node, QubitId::from(0)).unwrap();

    // Query connections
    println!("Predecessors of CX: {:?}", circuit.predecessors(cx_node));
    println!("Successors of H: {:?}", circuit.successors(h_node));
    ```

## TickCircuit

A time-sliced circuit representation where gates are organized into discrete time steps (ticks). Useful for explicit timing control and parallel gate scheduling.

### Quick Start

=== ":fontawesome-brands-python: Python"
    ```python
    from pecos.quantum import TickCircuit

    circuit = TickCircuit()

    # First tick: parallel gates
    circuit.tick().h([0]).h([1]).h([2])

    # Second tick: entangling layer
    circuit.tick().cx([(0, 1)]).cx([(2, 3)])

    # Third tick: measurements (call separately, mz doesn't chain)
    tick = circuit.tick()
    tick.mz([0])
    tick.mz([1])

    print(f"Number of ticks: {circuit.num_ticks()}")
    print(f"Total gates: {circuit.gate_count()}")
    print(f"Gate batches: {circuit.gate_batch_count()}")
    ```

=== ":fontawesome-brands-rust: Rust"
    ```rust
    use pecos::core::Gate;
    use pecos::quantum::TickCircuit;

    let mut circuit = TickCircuit::new();

    // First tick: parallel gates
    circuit.tick().h(&[0, 1, 2]);

    // Second tick: entangling layer
    circuit.tick().cx(&[(0, 1), (2, 3)]);

    // Third tick: measurements
    circuit.tick().mz(&[0, 1]);

    println!("Number of ticks: {}", circuit.num_ticks());
    println!("Total gates: {}", circuit.gate_count());
    println!("Gate batches: {}", circuit.gate_batch_count());
    ```

### Qubit Conflict Detection

TickCircuit prevents scheduling conflicting gates in the same tick:

=== ":fontawesome-brands-python: Python"
    <!--expect-error: QubitConflictError.*already in use-->
    ```python
    from pecos.quantum import TickCircuit

    circuit = TickCircuit()
    tick = circuit.tick()

    tick.h([0])
    tick.cx([(0, 1)])  # Error! Qubit 0 already used in this tick
    ```

=== ":fontawesome-brands-rust: Rust"
    ```rust
    use pecos::quantum::TickCircuit;

    let mut circuit = TickCircuit::new();
    let mut tick = circuit.tick();

    tick.h(&[0]);
    // This would error: qubit 0 already used
    // tick.cx(&[(0, 1)]);

    // Use try_add_gate for fallible operations
    if let Err(pecos::quantum::TickGateError::QubitConflict(e)) =
        tick.try_add_gate(Gate::cx(&[(0, 1)]))
    {
        println!("Conflict on qubits: {:?}", e.conflicting_qubits);
    }
    ```

### Tick Metadata

=== ":fontawesome-brands-python: Python"
    ```python
    from pecos.quantum import TickCircuit

    circuit = TickCircuit()

    # Add metadata to a tick
    tick = circuit.tick()
    tick.meta("round", 1)
    tick.h([0]).meta("error_rate", 0.001)

    # Circuit-level metadata
    circuit.set_meta("name", "Bell state")
    ```

=== ":fontawesome-brands-rust: Rust"
    ```rust
    let mut circuit = TickCircuit::new();

    // Add metadata to a tick
    let mut tick = circuit.tick();
    tick.meta("round", Attribute::Int(1));
    tick.h(&[0]).meta("error_rate", Attribute::Float(0.001));

    // Circuit-level metadata
    circuit.set_meta("name", Attribute::String("Bell state".into()));
    ```

### Conversion

TickCircuit can be converted to and from DagCircuit:

=== ":fontawesome-brands-python: Python"
    ```python
    from pecos.quantum import DagCircuit, TickCircuit

    # TickCircuit -> DagCircuit
    tick_circuit = TickCircuit()
    tick_circuit.tick().h([0]).h([1])
    tick_circuit.tick().cx([(0, 1)])

    dag_circuit = tick_circuit.to_dag_circuit()

    # DagCircuit -> TickCircuit
    tick_circuit2 = dag_circuit.to_tick_circuit()
    ```

=== ":fontawesome-brands-rust: Rust"
    ```rust
    use pecos::quantum::{DagCircuit, TickCircuit};

    // TickCircuit -> DagCircuit
    let mut tick_circuit = TickCircuit::new();
    tick_circuit.tick().h(&[0, 1]);
    tick_circuit.tick().cx(&[(0, 1)]);

    let dag_circuit = DagCircuit::from(tick_circuit);

    // DagCircuit -> TickCircuit
    let tick_circuit2 = TickCircuit::from(dag_circuit);
    ```

## DiGraph and DAG

For general graph algorithms beyond quantum circuits, PECOS provides `DiGraph` (directed graph) and `DAG` (directed acyclic graph).

### DiGraph

A general directed graph with weighted edges and attributes:

=== ":fontawesome-brands-python: Python"
    ```python
    from pecos.graph import DiGraph

    graph = DiGraph()

    # Add nodes
    n0 = graph.add_node()
    n1 = graph.add_node()
    n2 = graph.add_node()

    # Add edges with weights
    graph.add_edge(n0, n1)
    graph.set_weight(n0, n1, 1.0)
    graph.add_edge(n1, n2)
    graph.set_weight(n1, n2, 2.0)
    graph.add_edge(n0, n2)
    graph.set_weight(n0, n2, 5.0)

    # Query structure
    print(f"Predecessors of n2: {graph.predecessors(n2)}")
    print(f"Successors of n0: {graph.successors(n0)}")
    print(f"In-degree of n2: {graph.in_degree(n2)}")
    print(f"Out-degree of n0: {graph.out_degree(n0)}")
    ```

=== ":fontawesome-brands-rust: Rust"
    ```rust
    use pecos::digraph::DiGraph;

    let mut graph = DiGraph::new();

    // Add nodes
    let n0 = graph.add_node();
    let n1 = graph.add_node();
    let n2 = graph.add_node();

    // Add edges with weights
    graph.add_edge(n0, n1).weight(1.0);
    graph.add_edge(n1, n2).weight(2.0);
    graph.add_edge(n0, n2).weight(5.0);

    // Query structure
    println!("Predecessors of n2: {:?}", graph.predecessors(n2));
    println!("Successors of n0: {:?}", graph.successors(n0));
    println!("In-degree of n2: {}", graph.in_degree(n2));
    println!("Out-degree of n0: {}", graph.out_degree(n0));
    ```

### DAG

A directed acyclic graph with topological ordering and cycle prevention:

=== ":fontawesome-brands-python: Python"
    ```python
    from pecos.graph import DAG

    dag = DAG()

    # Add nodes
    n0 = dag.add_node()
    n1 = dag.add_node()
    n2 = dag.add_node()

    # Add edges (cycle detection)
    dag.add_edge(n0, n1)
    dag.add_edge(n1, n2)
    # dag.add_edge(n2, n0)  # Would raise: creates cycle!

    # Topological operations
    print(f"Topological order: {dag.topological_sort()}")
    print(f"Roots: {dag.roots()}")
    print(f"Leaves: {dag.leaves()}")
    print(f"Depth: {dag.depth()}")

    # Ancestry queries
    print(f"Ancestors of n2: {dag.ancestors(n2)}")
    print(f"Descendants of n0: {dag.descendants(n0)}")

    # Layer iteration
    for layer in dag.layers(dag.roots()):
        print(f"Layer: {layer}")
    ```

=== ":fontawesome-brands-rust: Rust"
    ```rust
    use pecos::dag::DAG;

    let mut dag = DAG::new();

    // Add nodes
    let n0 = dag.add_node();
    let n1 = dag.add_node();
    let n2 = dag.add_node();

    // Add edges (cycle detection)
    dag.add_edge(n0, n1).unwrap();
    dag.add_edge(n1, n2).unwrap();
    // dag.add_edge(n2, n0)  // Would return Err: creates cycle!

    // Topological operations
    println!("Topological order: {:?}", dag.topological_sort());
    println!("Roots: {:?}", dag.roots());
    println!("Leaves: {:?}", dag.leaves());
    println!("Depth: {}", dag.depth());

    // Ancestry queries
    println!("Ancestors of n2: {:?}", dag.ancestors(n2));
    println!("Descendants of n0: {:?}", dag.descendants(n0));

    // Layer iteration
    for layer in dag.layers(dag.roots()) {
        println!("Layer: {:?}", layer);
    }
    ```

## API Summary

### DagCircuit Methods

| Method | Description |
|--------|-------------|
| `new()` | Create empty circuit |
| `h(qubits)`, `x(qubits)`, `y(qubits)`, `z(qubits)` | Single-qubit Pauli gates |
| `sz(qubits)`, `szdg(qubits)`, `t(qubits)`, `tdg(qubits)` | Phase gates |
| `rx(theta, qubits)`, `ry(theta, qubits)`, `rz(theta, qubits)` | Rotation gates |
| `cx(pairs)`, `szz(pairs)`, `rzz(theta, pairs)` | Two-qubit gates |
| `mz(qubits)`, `pz(qubits)` | Measurement and preparation |
| `meta(key, value)` | Attach metadata to last gate |
| `gate_count()`, `gate_node_count()`, `depth()`, `width()` | Circuit metrics |
| `qubits()` | List of qubits used |
| `topological_order()` | Gates in dependency order |
| `layers()` | Iterator over parallel gate layers |
| `predecessors(node)`, `successors(node)` | Graph connectivity |

### TickCircuit Methods

| Method | Description |
|--------|-------------|
| `new()` | Create empty circuit |
| `tick()` | Start a new time step |
| `num_ticks()` | Number of time steps |
| `gate_count()` | Total gate applications across all ticks |
| `gate_batch_count()` | Total stored compatible gate batches across all ticks |
| `gate_batches()` | Stored gate batches with tick indices |
| `set_meta(key, value)` | Circuit-level metadata |

### DAG Methods

| Method | Description |
|--------|-------------|
| `add_node()` | Add node, returns ID |
| `add_edge(src, tgt)` | Add edge (fails if creates cycle) |
| `topological_sort()` | Nodes in dependency order |
| `layers(roots)` | Iterator over parallel layers |
| `roots()`, `leaves()` | Entry/exit nodes |
| `ancestors(n)`, `descendants(n)` | Transitive closure |
| `depth()` | Longest path length |

### DiGraph Methods

| Method | Description |
|--------|-------------|
| `add_node()` | Add node, returns ID |
| `add_edge(src, tgt)` | Add directed edge |
| `predecessors(n)`, `successors(n)` | Direct neighbors |
| `in_degree(n)`, `out_degree(n)` | Edge counts |
| `find_edge(src, tgt)` | Get edge ID |
| `get_weight(src, tgt)` | Get edge weight |
