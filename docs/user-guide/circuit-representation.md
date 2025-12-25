# Circuit Representation

PECOS provides several data structures for representing quantum circuits and general graphs, available in both Rust and Python.

## Overview

| Type | Purpose | Use Case |
|------|---------|----------|
| `DagCircuit` | DAG of quantum gates | Circuit optimization, resource estimation, noise modeling |
| `TickCircuit` | Time-sliced circuit | Explicit timing, parallel gate scheduling |
| `DiGraph` | Directed graph | General directed graph algorithms |
| `DAG` | Directed acyclic graph | Topological ordering, dependency tracking |
| `Graph` | Undirected graph | Matching, shortest paths (see [Graph API](graph-api.md)) |

## DagCircuit

A directed acyclic graph representation where nodes are gates and edges are qubit wires. This design follows HUGR and Qiskit's `DAGCircuit`.

### Quick Start

=== ":fontawesome-brands-python: Python"
    ```python
    from pecos.quantum import DagCircuit

    # Fluent builder API
    circuit = DagCircuit()
    circuit.h(0).cx(0, 1).rz(0.5, 0).mz(0)

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
    circuit.h(0).cx(0, 1).rz(0.5, 0).mz(0);

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
    circuit.h(0)  # Hadamard
    circuit.x(1)  # Pauli X
    circuit.y(2)  # Pauli Y
    circuit.z(3)  # Pauli Z
    circuit.sz(0)  # S gate (sqrt Z)
    circuit.szdg(0)  # S-dagger
    circuit.t(0)  # T gate
    circuit.tdg(0)  # T-dagger

    # Rotation gates (angle in radians)
    circuit.rx(3.14159, 0)  # RX
    circuit.ry(1.5708, 1)  # RY
    circuit.rz(0.7854, 2)  # RZ

    # Two-qubit gates
    circuit.cx(0, 1)  # CNOT (control, target)
    circuit.szz(0, 1)  # sqrt ZZ
    circuit.rzz(0.5, 0, 1)  # RZZ rotation

    # Measurement and preparation
    circuit.mz(0)  # Measure in Z basis
    circuit.pz(1)  # Prepare in Z basis (|0>)

    # Chaining
    circuit.h(0).cx(0, 1).h(0).mz(0)
    ```

=== ":fontawesome-brands-rust: Rust"
    ```rust
    use pecos::quantum::DagCircuit;

    let mut circuit = DagCircuit::new();

    // Single-qubit gates
    circuit.h(0);       // Hadamard
    circuit.x(1);       // Pauli X
    circuit.y(2);       // Pauli Y
    circuit.z(3);       // Pauli Z
    circuit.sz(0);      // S gate
    circuit.szdg(0);    // S-dagger
    circuit.t(0);       // T gate
    circuit.tdg(0);     // T-dagger

    // Rotation gates (angle in radians)
    circuit.rx(3.14159, 0);
    circuit.ry(1.5708, 1);
    circuit.rz(0.7854, 2);

    // Two-qubit gates
    circuit.cx(0, 1);
    circuit.szz(0, 1);
    circuit.rzz(0.5, 0, 1);

    // Measurement and preparation
    circuit.mz(0);
    circuit.pz(1);

    // Chaining
    circuit.h(0).cx(0, 1).h(0).mz(0);
    ```

### Adding Metadata

Gates can have arbitrary metadata attached:

=== ":fontawesome-brands-python: Python"
    ```python
    from pecos.quantum import DagCircuit, Attribute

    circuit = DagCircuit()

    # Attach metadata to the last gate
    circuit.h(0).meta("error_rate", Attribute.float(0.001))

    # Multiple metadata entries
    circuit.cx(0, 1).meta("duration_ns", Attribute.int(50))

    # Measurements break the chain but still support metadata
    circuit.mz(0).meta("basis", Attribute.string("Z"))
    ```

=== ":fontawesome-brands-rust: Rust"
    ```rust
    use pecos::quantum::{DagCircuit, Attribute};

    let mut circuit = DagCircuit::new();

    // Attach metadata to the last gate
    circuit.h(0).meta("error_rate", Attribute::Float(0.001));

    // Multiple metadata entries
    circuit.cx(0, 1).meta("duration_ns", Attribute::Int(50));

    // Measurements break the chain but still support metadata
    circuit.mz(0).meta("basis", Attribute::String("Z".into()));
    ```

### Circuit Analysis

=== ":fontawesome-brands-python: Python"
    ```python
    circuit = DagCircuit()
    circuit.h(0).cx(0, 1).h(1).cx(1, 2).mz(0).mz(1).mz(2)

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
    circuit.h(0).cx(0, 1).h(1).cx(1, 2).mz(0).mz(1).mz(2);

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
    from pecos.quantum import DagCircuit, Gate, QubitId

    circuit = DagCircuit()

    # Add gates manually
    h_node = circuit.add_gate(Gate.h([0]))
    cx_node = circuit.add_gate(Gate.cx([(0, 1)]))

    # Connect gates on qubit 0
    circuit.connect(h_node, cx_node, QubitId(0))

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
    circuit.tick().h(0).h(1).h(2)

    # Second tick: entangling layer
    circuit.tick().cx(0, 1).cx(2, 3)

    # Third tick: measurements
    circuit.tick().mz(0).mz(1)

    print(f"Number of ticks: {circuit.num_ticks()}")
    print(f"Total gates: {circuit.gate_count()}")
    ```

=== ":fontawesome-brands-rust: Rust"
    ```rust
    use pecos::quantum::TickCircuit;

    let mut circuit = TickCircuit::new();

    // First tick: parallel gates
    circuit.tick().h(0).h(1).h(2);

    // Second tick: entangling layer
    circuit.tick().cx(0, 1).cx(2, 3);

    // Third tick: measurements
    circuit.tick().mz(0).mz(1);

    println!("Number of ticks: {}", circuit.num_ticks());
    println!("Total gates: {}", circuit.gate_count());
    ```

### Qubit Conflict Detection

TickCircuit prevents scheduling conflicting gates in the same tick:

=== ":fontawesome-brands-python: Python"
    ```python
    from pecos.quantum import TickCircuit

    circuit = TickCircuit()
    tick = circuit.tick()

    tick.h(0)
    tick.cx(0, 1)  # Error! Qubit 0 already used in this tick
    ```

=== ":fontawesome-brands-rust: Rust"
    ```rust
    use pecos::quantum::TickCircuit;

    let mut circuit = TickCircuit::new();
    let mut tick = circuit.tick();

    tick.h(0);
    // This would error: qubit 0 already used
    // tick.cx(0, 1);

    // Use try_add_gate for fallible operations
    if let Err(e) = tick.try_add_gate(Gate::cx(&[(0, 1)])) {
        println!("Conflict on qubits: {:?}", e.conflicting_qubits);
    }
    ```

### Tick Metadata

=== ":fontawesome-brands-python: Python"
    ```python
    circuit = TickCircuit()

    # Add metadata to a tick
    tick = circuit.tick()
    tick.meta("round", Attribute.int(1))
    tick.h(0).meta("error_rate", Attribute.float(0.001))

    # Circuit-level metadata
    circuit.set_meta("name", Attribute.string("Bell state"))
    ```

=== ":fontawesome-brands-rust: Rust"
    ```rust
    let mut circuit = TickCircuit::new();

    // Add metadata to a tick
    let mut tick = circuit.tick();
    tick.meta("round", Attribute::Int(1));
    tick.h(0).meta("error_rate", Attribute::Float(0.001));

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
    tick_circuit.tick().h(0).h(1)
    tick_circuit.tick().cx(0, 1)

    dag_circuit = DagCircuit.from_tick_circuit(tick_circuit)

    # DagCircuit -> TickCircuit
    tick_circuit2 = TickCircuit.from_dag_circuit(dag_circuit)
    ```

=== ":fontawesome-brands-rust: Rust"
    ```rust
    use pecos::quantum::{DagCircuit, TickCircuit};

    // TickCircuit -> DagCircuit
    let mut tick_circuit = TickCircuit::new();
    tick_circuit.tick().h(0).h(1);
    tick_circuit.tick().cx(0, 1);

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
    graph.add_edge(n0, n1).weight(1.0)
    graph.add_edge(n1, n2).weight(2.0)
    graph.add_edge(n0, n2).weight(5.0)

    # Query structure
    print(f"Predecessors of n2: {graph.predecessors(n2)}")
    print(f"Successors of n0: {graph.successors(n0)}")
    print(f"In-degree of n2: {graph.in_degree(n2)}")
    print(f"Out-degree of n0: {graph.out_degree(n0)}")
    ```

=== ":fontawesome-brands-rust: Rust"
    ```rust
    use pecos::graph::DiGraph;

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
    use pecos::graph::DAG;

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
| `h(q)`, `x(q)`, `y(q)`, `z(q)` | Single-qubit Pauli gates |
| `sz(q)`, `szdg(q)`, `t(q)`, `tdg(q)` | Phase gates |
| `rx(theta, q)`, `ry(theta, q)`, `rz(theta, q)` | Rotation gates |
| `cx(ctrl, tgt)`, `szz(q1, q2)`, `rzz(theta, q1, q2)` | Two-qubit gates |
| `mz(q)`, `pz(q)` | Measurement and preparation |
| `meta(key, value)` | Attach metadata to last gate |
| `gate_count()`, `depth()`, `width()` | Circuit metrics |
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
| `gate_count()` | Total gates across all ticks |
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
