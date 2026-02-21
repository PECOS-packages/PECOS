# Graph API

```hidden-python
from pecos.graph import Graph

graph = Graph()
n0 = graph.add_node()
n1 = graph.add_node()
n2 = graph.add_node()
n3 = graph.add_node()
n4 = graph.add_node()
n5 = graph.add_node()
graph.add_edge(n0, n1)
graph.add_edge(n1, n2)
graph.add_edge(n0, n2)
graph.add_edge(n2, n3)
graph.set_weight(n0, n1, 1.0)
graph.set_weight(n1, n2, 2.0)
graph.set_weight(n0, n2, 5.0)
source_node = n0
edge_id = graph.find_edge(n0, n1)

# Set up graph-level attributes
attrs = graph.attrs()
attrs["name"] = "surface_code"
attrs["distance"] = 5
attrs["rounds"] = 100
attrs["version"] = "1.0"
attrs["author"] = "Alice"

# Set up node attributes
graph.node_attrs(n0)["label"] = "qubit_0"
graph.node_attrs(n0)["position"] = [0.0, 1.0, 2.0]
graph.node_attrs(n0)["active"] = True
graph.node_attrs(n0)["x"] = 1.0
graph.node_attrs(n0)["y"] = 2.0
graph.node_attrs(n1)["label"] = "qubit_1"

# Set up edge attributes
graph.edge_attrs(n0, n1)["label"] = "boundary"
graph.edge_attrs(n0, n1)["syn_path"] = [1, 2, 3]
graph.edge_attrs(n0, n1)["path"] = [0, 1]
graph.edge_attrs(n0, n1)["active"] = True
```

```hidden-rust
use pecos::graph::{Graph, Attribute};
use serde_json::json;

fn main() {
    let mut graph = Graph::new();
    let n0 = graph.add_node();
    let n1 = graph.add_node();
    let n2 = graph.add_node();
    let n3 = graph.add_node();
    graph.add_edge(n0, n1);
    graph.add_edge(n1, n2);
    graph.add_edge(n0, n2);
    graph.add_edge(n2, n3);
    graph.set_weight(n0, n1, 1.0);
    graph.set_weight(n1, n2, 2.0);
    graph.set_weight(n0, n2, 5.0);
    let source_node = n0;
    // CODE
}
```

The PECOS Graph API provides a high-performance graph data structure with idiomatic APIs for both Rust and Python.

## Setup

The examples below use this pre-built graph with nodes, edges, and attributes:

=== ":fontawesome-brands-python: Python"
    ```python
    from pecos.graph import Graph

    # Create graph with nodes and edges
    graph = Graph()
    n0 = graph.add_node()
    n1 = graph.add_node()
    n2 = graph.add_node()
    n3 = graph.add_node()
    graph.add_edge(n0, n1)
    graph.add_edge(n1, n2)
    graph.add_edge(n0, n2)
    graph.add_edge(n2, n3)

    # Set edge weights
    graph.set_weight(n0, n1, 1.0)
    graph.set_weight(n1, n2, 2.0)
    graph.set_weight(n0, n2, 5.0)

    source_node = n0
    edge_id = graph.find_edge(n0, n1)
    ```

=== ":fontawesome-brands-rust: Rust"
    ```rust
    use pecos::graph::{Graph, Attribute};

    let mut graph = Graph::new();
    let n0 = graph.add_node();
    let n1 = graph.add_node();
    let n2 = graph.add_node();
    let n3 = graph.add_node();
    graph.add_edge(n0, n1);
    graph.add_edge(n1, n2);
    graph.add_edge(n0, n2);
    graph.add_edge(n2, n3);

    graph.set_weight(n0, n1, 1.0);
    graph.set_weight(n1, n2, 2.0);
    graph.set_weight(n0, n2, 5.0);

    let source_node = n0;
    ```

## Design Principles

- **Language-Idiomatic** - Dict-like in Python, BTreeMap in Rust
- **Node-Pair Operations** - Use `(node_a, node_b)` for edge operations
- **Integer Node IDs** - Nodes identified by indices (0, 1, 2, ...)
- **Mutable Attribute Views** - Direct mutation through views
- **Typed Attributes** - `Attribute` enum: int, float, string, bool, lists
- **Efficient** - BTreeMap-backed with O(log n) lookups

## Quick Start

=== ":fontawesome-brands-python: Python"
    ```python
    from pecos.graph import Graph

    # Create graph and nodes
    graph = Graph()
    n0 = graph.add_node()
    n1 = graph.add_node()

    # Add edge with attributes
    graph.add_edge(n0, n1)
    graph.set_weight(n0, n1, 5.0)

    attrs = graph.edge_attrs(n0, n1)
    attrs["label"] = "boundary"
    attrs["path"] = [1, 2, 3]
    ```

=== ":fontawesome-brands-rust: Rust"
    ```rust
    use pecos::graph::{Graph, Attribute};
    // Or: use pecos_num::graph::{Graph, Attribute};

    // Create graph and nodes
    let mut graph = Graph::new();
    let n0 = graph.add_node();
    let n1 = graph.add_node();

    // Add edge with attributes
    graph.add_edge(n0, n1);
    graph.set_weight(n0, n1, 5.0);

    if let Some(attrs) = graph.edge_attrs_mut(n0, n1) {
        attrs.insert("label".to_string(),
                    Attribute::String("boundary".into()));
        attrs.insert("path".to_string(),
                    Attribute::IntList(vec![1, 2, 3]));
    }
    ```

## Creating Graphs

=== ":fontawesome-brands-python: Python"
    ```python
    from pecos.graph import Graph

    graph = Graph()
    # Or with initial capacity
    graph = Graph.with_capacity(100, 200)  # nodes, edges
    ```

=== ":fontawesome-brands-rust: Rust"
    ```rust
    use pecos::graph::Graph;

    let mut graph = Graph::new();
    // Or with initial capacity
    let mut graph = Graph::with_capacity(100, 200);  // nodes, edges
    ```
## Graph-Level Attributes

The graph itself can store metadata as attributes.

### Setting Graph Attributes

=== ":fontawesome-brands-python: Python"
    ```python
    # Access graph-level attributes
    attrs = graph.attrs()

    # Style 1: Dict-like
    attrs["name"] = "surface_code"
    attrs["distance"] = 5
    attrs["rounds"] = 100

    # Style 2: Chainable insert
    graph.attrs().insert("version", "1.0").insert("author", "Alice")

    # Style 3: Batch update
    graph.attrs().update({"date": "2025-01-26", "tags": ["qec", "surface_code"], "validated": True})
    ```

=== ":fontawesome-brands-rust: Rust"
    ```rust
    // Access graph-level attributes (always available)
    let attrs = graph.attrs_mut();

    // Individual insert
    attrs.insert("name".to_string(),
                Attribute::String("surface_code".into()));
    attrs.insert("distance".to_string(),
                Attribute::Int(5));

    // Batch extend
    attrs.extend([
        ("version".to_string(), Attribute::String("1.0".into())),
        ("author".to_string(), Attribute::String("Alice".into())),
        ("date".to_string(), Attribute::String("2025-01-26".into())),
    ]);
    ```

### Reading Graph Attributes

=== ":fontawesome-brands-python: Python"
    ```python
    attrs = graph.attrs()

    # Get attribute
    name = attrs["name"]

    # Get with default
    version = attrs.get("version", "unknown")

    # Check existence
    if "distance" in attrs:
        print(f"Distance: {attrs['distance']}")
    ```

=== ":fontawesome-brands-rust: Rust"
    ```rust
    let attrs = graph.attrs();

    // Get attribute
    if let Some(name) = attrs.get("name") {
        println!("{:?}", name);
    }

    // Check existence
    if attrs.contains_key("distance") {
        println!("{:?}", attrs["distance"]);
    }
    ```

## Nodes

### Adding Nodes

=== ":fontawesome-brands-python: Python"
    ```python
    n0 = graph.add_node()  # Returns node ID (int)
    n1 = graph.add_node()
    ```

=== ":fontawesome-brands-rust: Rust"
    ```rust
    let n0 = graph.add_node();  // Returns node ID (usize)
    let n1 = graph.add_node();
    ```

### Node Information

=== ":fontawesome-brands-python: Python"
    ```python
    # Count nodes
    count = graph.node_count()

    # List all node IDs
    nodes = graph.nodes()  # Returns list of ints

    # Iterate over nodes
    for node in graph.nodes():
        print(node)
    ```

=== ":fontawesome-brands-rust: Rust"
    ```rust
    // Count nodes
    let count = graph.node_count();

    // Iterate over nodes
    for node in graph.nodes() {
        println!("{}", node);
    }
    ```
## Node Attributes

Nodes can have arbitrary attributes attached to them, similar to edges.

### Setting Node Attributes

=== ":fontawesome-brands-python: Python"
    ```python
    # Create nodes
    n0 = graph.add_node()
    n1 = graph.add_node()

    # Access node attributes
    attrs = graph.node_attrs(n0)

    # Style 1: Dict-like
    attrs["label"] = "qubit_0"
    attrs["position"] = [0.0, 1.0, 2.0]
    attrs["active"] = True

    # Style 2: Chainable insert
    graph.node_attrs(n1).insert("label", "qubit_1").insert("type", "data")

    # Style 3: Batch update
    graph.node_attrs(n0).update({"x": 1.0, "y": 2.0, "state": "initialized"})
    ```

=== ":fontawesome-brands-rust: Rust"
    ```rust
    // Create nodes
    let n0 = graph.add_node();
    let n1 = graph.add_node();

    // Access node attributes
    if let Some(attrs) = graph.node_attrs_mut(n0) {
        // Individual insert
        attrs.insert("label".to_string(),
                    Attribute::String("qubit_0".into()));
        attrs.insert("position".to_string(),
                    Attribute::Json(json!([0.0, 1.0, 2.0])));

        // Batch extend
        attrs.extend([
            ("x".to_string(), Attribute::Float(1.0)),
            ("y".to_string(), Attribute::Float(2.0)),
        ]);
    }
    ```

### Reading Node Attributes

=== ":fontawesome-brands-python: Python"
    ```python
    attrs = graph.node_attrs(n0)

    # Get attribute (raises KeyError if missing)
    label = attrs["label"]

    # Get with default
    label = attrs.get("label", "default")

    # Check existence
    if "label" in attrs:
        print(attrs["label"])
    ```

=== ":fontawesome-brands-rust: Rust"
    ```rust
    if let Some(attrs) = graph.node_attrs(n0) {
        // Get attribute
        if let Some(label) = attrs.get("label") {
            println!("{:?}", label);
        }

        // Check existence
        if attrs.contains_key("label") {
            println!("{:?}", attrs["label"]);
        }
    }
    ```

## Edges

### Adding Edges

=== ":fontawesome-brands-python: Python"
    ```python
    # Add edge
    graph.add_edge(n0, n1)

    # Add edge with weight
    graph.add_edge(n0, n1)
    graph.set_weight(n0, n1, 5.0)
    ```

=== ":fontawesome-brands-rust: Rust"
    ```rust
    // Add edge
    graph.add_edge(n0, n1);

    // Add edge with weight
    graph.add_edge(n0, n1);
    graph.set_weight(n0, n1, 5.0);
    ```

### Edge Attributes - Three Styles

Python provides three ways to set edge attributes:

=== ":fontawesome-brands-python: Python"
    ```python
    graph.add_edge(n0, n1)
    graph.set_weight(n0, n1, 5.0)

    attrs = graph.edge_attrs(n0, n1)

    # Style 1: Dict-like (most Pythonic)
    attrs["label"] = "boundary"
    attrs["syn_path"] = [1, 2, 3]
    attrs["data_path"] = [0, 1]

    # Style 2: Chainable insert
    attrs.insert("weight", 5.0).insert("label", "virtual")

    # Style 3: Batch update from dict
    attrs.update({"key1": 42, "key2": "value", "key3": [1, 2]})

    # Mix styles as needed
    attrs["x"] = 1.0
    attrs.insert("y", 2.0).insert("z", 3.0)
    attrs.update({"a": 1, "b": 2})
    ```

=== ":fontawesome-brands-rust: Rust"
    ```rust
    use pecos::graph::Attribute;

    graph.add_edge(n0, n1);
    graph.set_weight(n0, n1, 5.0);

    // BTreeMap mutable access
    if let Some(attrs) = graph.edge_attrs_mut(n0, n1) {
        // Individual insert
        attrs.insert("label".to_string(),
                    Attribute::String("boundary".into()));
        attrs.insert("syn_path".to_string(),
                    Attribute::IntList(vec![1, 2, 3]));

        // Batch extend (from Extend trait)
        attrs.extend([
            ("key1".to_string(), Attribute::Int(42)),
            ("key2".to_string(), Attribute::String("value".into())),
        ]);
    }
    ```

### Reading Edge Attributes

=== ":fontawesome-brands-python: Python"
    ```python
    # Get weight
    weight = graph.get_weight(n0, n1)  # Returns float or None

    # Get all edge data as dict
    data = graph.get_edge_data(n0, n1)  # Returns dict or None

    # Access individual attributes
    attrs = graph.edge_attrs(n0, n1)
    label = attrs["label"]  # Raises KeyError if not found
    label = attrs.get("label")  # Returns None if not found
    label = attrs.get("label", "default")  # With default value

    # Check if attribute exists
    if "label" in attrs:
        print(attrs["label"])
    ```

=== ":fontawesome-brands-rust: Rust"
    ```rust
    // Get weight
    let weight = graph.get_weight(n0, n1);  // Returns Option<f64>

    // Get all edge data
    let data = graph.get_edge_data(n0, n1);  // Returns Option<BTreeMap>

    // Access individual attributes
    if let Some(attrs) = graph.edge_attrs(n0, n1) {
        let label = attrs.get("label");  // Returns Option<&Attribute>

        // Check if attribute exists
        if attrs.contains_key("label") {
            println!("{:?}", attrs["label"]);
        }
    }
    ```

### Finding Edges

=== ":fontawesome-brands-python: Python"
    ```python
    # Find edge ID from node pair
    edge_id = graph.find_edge(n0, n1)  # Returns int or None

    # Get endpoints from edge ID
    endpoints = graph.edge_endpoints(edge_id)  # Returns tuple or None
    if endpoints:
        a, b = endpoints
    ```

=== ":fontawesome-brands-rust: Rust"
    ```rust
    // Find edge ID from node pair
    if let Some(edge_id) = graph.find_edge(n0, n1) {
        // Get endpoints from edge ID
        if let Some((a, b)) = graph.edge_endpoints(edge_id) {
            println!("Edge {}: {} -> {}", edge_id, a, b);
        }
    }
    ```

### Edge Information

=== ":fontawesome-brands-python: Python"
    ```python
    # Count edges
    count = graph.edge_count()

    # List all edges
    edges = graph.edges()  # Returns list of (node_a, node_b, weight) tuples

    # Iterate over edges
    for a, b, weight in graph.edges():
        print(f"Edge {a}-{b}: weight={weight}")
    ```

=== ":fontawesome-brands-rust: Rust"
    ```rust
    // Count edges
    let count = graph.edge_count();

    // Iterate over edges (returns (node_a, node_b, weight) tuples)
    for (a, b, weight) in graph.edges() {
        println!("Edge {}-{}: weight={}", a, b, weight);
    }
    ```

## Supported Attribute Types

The `Attribute` enum supports these types:

| Python Type | Rust Type | Example | Notes |
|-------------|-----------|---------|-------|
| `int` | `Attribute::Int(i64)` | `42` | Fast path |
| `float` | `Attribute::Float(f64)` | `3.14` | Fast path |
| `str` | `Attribute::String(String)` | `"text"` | Fast path |
| `bool` | `Attribute::Bool(bool)` | `True` | Fast path |
| `list[int]` | `Attribute::IntList(Vec<i64>)` | `[1, 2, 3]` | Fast path |
| `list[str]` | `Attribute::StringList(Vec<String>)` | `["a", "b"]` | Fast path |
| Any JSON-serializable | `Attribute::Json(serde_json::Value)` | `{"key": [1, "mixed"]}` | Fallback for complex types |

!!! note "Automatic Type Selection"
    PECOS automatically selects the most appropriate type. Native types (int, float, str, bool, homogeneous lists) use fast-path variants. Complex structures like nested dicts, mixed-type lists, or arbitrary objects automatically fall back to JSON serialization.

### Complex Attribute Examples

=== ":fontawesome-brands-python: Python"
    ```python
    # Native types (fast path)
    graph.edge_attrs(0, 1)["label"] = "control"  # String
    graph.edge_attrs(0, 1)["weight_factor"] = 2.5  # Float
    graph.edge_attrs(0, 1)["enabled"] = True  # Bool
    graph.edge_attrs(0, 1)["path"] = [1, 2, 3]  # IntList

    # Complex types (automatic JSON fallback)
    graph.node_attrs(5)["metadata"] = {
        "type": "data_qubit",
        "coordinates": [0.5, 1.2],
        "neighbors": [4, 6, 8],
    }

    # Mixed-type list (automatic JSON fallback)
    graph.edge_attrs(2, 3)["mixed"] = [1, "vertex", {"key": "value"}]

    # Nested structures (automatic JSON fallback)
    graph.attrs()["config"] = {
        "version": "2.0",
        "parameters": {"threshold": 0.01, "rounds": [10, 20, 30], "enabled": True},
    }
    ```

=== ":fontawesome-brands-rust: Rust"
    ```rust
    use pecos::graph::{Graph, Attribute};
    use serde_json::json;

    let mut graph = Graph::new();
    let n0 = graph.add_node();
    let n1 = graph.add_node();
    graph.add_edge(n0, n1);

    // Native types (fast path)
    if let Some(attrs) = graph.edge_attrs_mut(n0, n1) {
        attrs.insert("label".into(), Attribute::String("control".into()));
        attrs.insert("weight_factor".into(), Attribute::Float(2.5));
        attrs.insert("enabled".into(), Attribute::Bool(true));
        attrs.insert("path".into(), Attribute::IntList(vec![1, 2, 3]));
    }

    // Complex types (JSON variant)
    if let Some(attrs) = graph.node_attrs_mut(5) {
        attrs.insert("metadata".into(), Attribute::Json(json!({
            "type": "data_qubit",
            "coordinates": [0.5, 1.2],
            "neighbors": [4, 6, 8]
        })));
    }

    // Nested structures
    graph.attrs_mut().insert("config".into(), Attribute::Json(json!({
        "version": "2.0",
        "parameters": {
            "threshold": 0.01,
            "rounds": [10, 20, 30],
            "enabled": true
        }
    })));
    ```

## Graph Algorithms

### Maximum Weight Matching

=== ":fontawesome-brands-python: Python"
    ```python
    # Find maximum weight perfect matching
    matching = graph.max_weight_matching(max_cardinality=True)

    # matching is a dict: {node: matched_node}
    for node, matched in matching.items():
        print(f"{node} matched with {matched}")
    ```

=== ":fontawesome-brands-rust: Rust"
    ```rust
    // Find maximum weight perfect matching
    let matching = graph.max_weight_matching(true);

    // matching is a BTreeMap<usize, usize>
    for (node, matched) in &matching {
        println!("{} matched with {}", node, matched);
    }
    ```

### Shortest Paths

The graph provides two methods for shortest path computation:

- **`shortest_path_distances(source)`** - Fast, returns only distances (uses Dijkstra)
- **`single_source_shortest_path(source)`** - Slower, returns full paths

#### Distances Only (Faster)

=== ":fontawesome-brands-python: Python"
    ```python
    # Get shortest path distances (Dijkstra) - faster if you don't need paths
    distances = graph.shortest_path_distances(source_node)

    # distances is a dict: {node: distance}
    for node, dist in distances.items():
        print(f"Distance to {node}: {dist}")
    ```

=== ":fontawesome-brands-rust: Rust"
    ```rust
    // Get shortest path distances - faster if you don't need paths
    let distances = graph.shortest_path_distances(source_node);

    // distances is a BTreeMap<usize, f64>
    for (node, dist) in &distances {
        println!("Distance to {}: {}", node, dist);
    }
    ```

#### Full Paths (Slower)

=== ":fontawesome-brands-python: Python"
    ```python
    # Get shortest paths with full path reconstruction
    paths = graph.single_source_shortest_path(source_node)

    # paths is a dict: {node: [list of nodes in path]}
    for node, path in paths.items():
        print(f"Path to {node}: {path}")
    ```

=== ":fontawesome-brands-rust: Rust"
    ```rust
    // Get shortest paths with full path reconstruction
    let paths = graph.single_source_shortest_path(source_node);

    // paths is a BTreeMap<usize, Vec<usize>>
    for (node, path) in &paths {
        println!("Path to {}: {:?}", node, path);
    }
    ```

### Subgraphs

=== ":fontawesome-brands-python: Python"
    ```python
    # Create subgraph from node subset
    nodes_to_keep = [0, 2, 5, 7]
    subgraph = graph.subgraph(nodes_to_keep)

    # Note: nodes are renumbered in subgraph
    # Original node N becomes subgraph node M
    ```

=== ":fontawesome-brands-rust: Rust"
    ```rust
    // Create subgraph from node subset
    let nodes_to_keep = vec![0, 2, 5, 7];
    let subgraph = graph.subgraph(&nodes_to_keep);
    ```

## Complete Example

=== ":fontawesome-brands-python: Python"
    ```python
    from pecos.graph import Graph

    # Build a simple graph
    graph = Graph()
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

    # Add edge attributes
    attrs = graph.edge_attrs(n0, n1)
    attrs.update({"label": "virtual", "path": [0, 1], "active": True})

    # Find shortest paths
    paths = graph.single_source_shortest_path(n0)
    print(f"Path n0->n2: {paths[n2]}")  # [0, 1, 2]

    # Find matching
    matching = graph.max_weight_matching(max_cardinality=True)
    print(f"Matching: {matching}")
    ```

=== ":fontawesome-brands-rust: Rust"
    ```rust
    use pecos::graph::{Graph, Attribute};

    // Build a simple graph
    let mut graph = Graph::new();
    let n0 = graph.add_node();
    let n1 = graph.add_node();
    let n2 = graph.add_node();

    // Add edges with weights
    graph.add_edge(n0, n1);
    graph.set_weight(n0, n1, 1.0);

    graph.add_edge(n1, n2);
    graph.set_weight(n1, n2, 2.0);

    graph.add_edge(n0, n2);
    graph.set_weight(n0, n2, 5.0);

    // Add edge attributes
    if let Some(attrs) = graph.edge_attrs_mut(n0, n1) {
        attrs.extend([
            ("label".to_string(), Attribute::String("virtual".into())),
            ("path".to_string(), Attribute::IntList(vec![0, 1])),
            ("active".to_string(), Attribute::Bool(true)),
        ]);
    }

    // Find shortest paths
    let paths = graph.single_source_shortest_path(n0);
    println!("Path n0->n2: {:?}", paths.get(&n2));  // Some([0, 1, 2])

    // Find matching
    let matching = graph.max_weight_matching(true);
    println!("Matching: {:?}", matching);
    ```

## API Summary

### Core Methods

| Method | Python | Rust | Description |
|--------|--------|------|-------------|
| Create | `Graph()` | `Graph::new()` | New empty graph |
| Add node | `add_node()` | `add_node()` | Returns node ID |
| Add edge | `add_edge(a, b)` | `add_edge(a, b)` | Add edge between nodes |
| Set weight | `set_weight(a, b, w)` | `set_weight(a, b, w)` | Set edge weight |
| Get weight | `get_weight(a, b)` | `get_weight(a, b)` | Get edge weight |
| **Graph attrs** | `attrs()` | `attrs()` / `attrs_mut()` | Get graph-level attributes |
| **Node attrs** | `node_attrs(node)` | `node_attrs(node)` / `node_attrs_mut(node)` | Get node attributes |
| **Edge attrs** | `edge_attrs(a, b)` | `edge_attrs(a, b)` / `edge_attrs_mut(a, b)` | Get edge attributes |
| Find edge | `find_edge(a, b)` | `find_edge(a, b)` | Get edge ID |
| Edge data | `get_edge_data(a, b)` | `get_edge_data(a, b)` | Get all edge attrs |
| Node count | `node_count()` | `node_count()` | Number of nodes |
| Edge count | `edge_count()` | `edge_count()` | Number of edges |

### Key Differences

**Python**:
- Attribute methods return view objects: `GraphAttrsView`, `NodeAttrsView`, `EdgeAttrsView`
- Views provide dict-like interface: `attrs["key"] = value`
- Chainable: `attrs.insert("k", v).insert("k2", v2)`
- Batch updates: `attrs.update({"k1": v1, "k2": v2})`

**Rust**:
- `attrs()` returns `&BTreeMap` (immutable)
- `attrs_mut()`, `node_attrs_mut()`, `edge_attrs_mut()` return `Option<&mut BTreeMap>`
- Standard BTreeMap methods: `insert()`, `extend()`, `get()`, etc.
