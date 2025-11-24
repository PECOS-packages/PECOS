"""Type stubs for PECOS Rust library bindings.

This file provides type hints and documentation for IDE support.
"""

from enum import Enum

# Enums
class NoiseModel(Enum):
    """Available noise model types."""

    PassThrough = "PassThrough"
    Depolarizing = "Depolarizing"
    DepolarizingCustom = "DepolarizingCustom"
    BiasedDepolarizing = "BiasedDepolarizing"
    General = "General"

class QuantumEngine(Enum):
    """Available quantum simulation engines."""

    StateVector = "StateVector"
    SparseStabilizer = "SparseStabilizer"

# Main classes
class GeneralNoiseModelBuilder:
    """Builder for constructing complex general noise models with fluent API.

    This builder provides a type-safe way to construct noise models with
    various error types including gate errors, measurement errors, idle noise,
    and state preparation errors.

    Example:
        >>> noise = (GeneralNoiseModelBuilder()
        ...     .with_seed(42)
        ...     .with_p1_probability(0.001)  # Single-qubit error
        ...     .with_p2_probability(0.01)   # Two-qubit error
        ...     .with_meas_0_probability(0.002)  # Measurement 0->1 flip
        ...     .with_meas_1_probability(0.002)) # Measurement 1->0 flip
        >>>
        >>> from pecos_rslib import sim
        >>> from pecos_rslib.programs import QasmProgram
        >>> program = QasmProgram.from_string(qasm)
        >>> simulation = sim(program).noise(noise).build()
    """

    def __init__(self) -> None:
        """Create a new GeneralNoiseModelBuilder with default parameters."""

    def with_seed(self, seed: int) -> GeneralNoiseModelBuilder:
        """Set the random number generator seed for reproducible noise.

        Args:
            seed: Random seed value (must be non-negative)

        Returns:
            Self for method chaining

        Raises:
            ValueError: If seed is negative
        """

    def with_scale(self, scale: float) -> GeneralNoiseModelBuilder:
        """Set global scaling factor for all error rates.

        This multiplies all error probabilities by the given factor,
        useful for studying noise threshold behavior.

        Args:
            scale: Scaling factor (must be non-negative)

        Returns:
            Self for method chaining

        Raises:
            ValueError: If scale is negative
        """

    def with_leakage_scale(self, scale: float) -> GeneralNoiseModelBuilder:
        """Set the leakage vs depolarizing ratio.

        Controls how much of the error budget goes to leakage (qubit
        leaving computational subspace) vs depolarizing errors.

        Args:
            scale: Leakage scale between 0.0 (no leakage) and 1.0 (all leakage)

        Returns:
            Self for method chaining

        Raises:
            ValueError: If scale is not between 0 and 1
        """

    def with_emission_scale(self, scale: float) -> GeneralNoiseModelBuilder:
        """Set scaling factor for spontaneous emission errors.

        Args:
            scale: Emission scaling factor (must be non-negative)

        Returns:
            Self for method chaining

        Raises:
            ValueError: If scale is negative
        """

    def with_noiseless_gate(self, gate: str) -> GeneralNoiseModelBuilder:
        """Mark a specific gate type as noiseless.

        Args:
            gate: Gate name (e.g., "H", "X", "CX", "MEASURE")

        Returns:
            Self for method chaining

        Raises:
            ValueError: If gate type is unknown
        """
    # State preparation noise
    def with_prep_probability(self, p: float) -> GeneralNoiseModelBuilder:
        """Set error probability during qubit state preparation.

        Args:
            p: Error probability between 0.0 and 1.0

        Returns:
            Self for method chaining

        Raises:
            ValueError: If p is not between 0 and 1
        """
    # Single-qubit gate noise
    def with_p1_probability(self, p: float) -> GeneralNoiseModelBuilder:
        """Set total error probability after single-qubit gates.

        This is the total probability of any error occurring after
        a single-qubit gate operation.

        Args:
            p: Total error probability between 0.0 and 1.0

        Returns:
            Self for method chaining

        Raises:
            ValueError: If p is not between 0 and 1
        """

    def with_average_p1_probability(self, p: float) -> GeneralNoiseModelBuilder:
        """Set average error probability for single-qubit gates.

        This sets the average gate infidelity, which is automatically
        converted to total error probability (multiplied by 1.5).

        Args:
            p: Average error probability between 0.0 and 1.0

        Returns:
            Self for method chaining

        Raises:
            ValueError: If p is not between 0 and 1
        """

    def with_p1_pauli_model(
        self,
        model: dict[str, float],
    ) -> GeneralNoiseModelBuilder:
        """Set the distribution of Pauli errors for single-qubit gates.

        Specifies how single-qubit errors are distributed among
        X, Y, and Z Pauli errors. Values should sum to 1.0.

        Args:
            model: Dictionary mapping Pauli operators to probabilities
                   e.g., {"X": 0.5, "Y": 0.3, "Z": 0.2}

        Returns:
            Self for method chaining

        Example:
            >>> builder.with_p1_pauli_model({
            ...     "X": 0.5,  # 50% X errors (bit flips)
            ...     "Y": 0.3,  # 30% Y errors
            ...     "Z": 0.2   # 20% Z errors (phase flips)
            ... })
        """
    # Two-qubit gate noise
    def with_p2_probability(self, p: float) -> GeneralNoiseModelBuilder:
        """Set total error probability after two-qubit gates.

        This is the total probability of any error occurring after
        a two-qubit gate operation (e.g., CX, CZ).

        Args:
            p: Total error probability between 0.0 and 1.0

        Returns:
            Self for method chaining

        Raises:
            ValueError: If p is not between 0 and 1
        """

    def with_average_p2_probability(self, p: float) -> GeneralNoiseModelBuilder:
        """Set average error probability for two-qubit gates.

        This sets the average gate infidelity, which is automatically
        converted to total error probability (multiplied by 1.25).

        Args:
            p: Average error probability between 0.0 and 1.0

        Returns:
            Self for method chaining

        Raises:
            ValueError: If p is not between 0 and 1
        """

    def with_p2_pauli_model(
        self,
        model: dict[str, float],
    ) -> GeneralNoiseModelBuilder:
        """Set the distribution of Pauli errors for two-qubit gates.

        Specifies how two-qubit errors are distributed among
        two-qubit Pauli operators.

        Args:
            model: Dictionary mapping two-qubit Pauli strings to probabilities
                   e.g., {"IX": 0.25, "XI": 0.25, "XX": 0.5}

        Returns:
            Self for method chaining
        """
    # Measurement noise
    def with_meas_0_probability(self, p: float) -> GeneralNoiseModelBuilder:
        """Set probability of 0→1 flip during measurement.

        This is the probability that a qubit in |0⟩ state is
        incorrectly measured as 1.

        Args:
            p: Bit flip probability between 0.0 and 1.0

        Returns:
            Self for method chaining

        Raises:
            ValueError: If p is not between 0 and 1
        """

    def with_meas_1_probability(self, p: float) -> GeneralNoiseModelBuilder:
        """Set probability of 1→0 flip during measurement.

        This is the probability that a qubit in |1⟩ state is
        incorrectly measured as 0.

        Args:
            p: Bit flip probability between 0.0 and 1.0

        Returns:
            Self for method chaining

        Raises:
            ValueError: If p is not between 0 and 1
        """

    def _get_builder(self) -> object:
        """Internal method to get the underlying Rust builder."""

class QasmSimulation:
    """A compiled QASM simulation ready for execution.

    This represents a parsed and compiled quantum circuit that can be
    run multiple times with different shot counts efficiently.
    """

    def run(self, shots: int) -> dict[str, list[int | str]]:
        """Run the simulation with the specified number of shots.

        Args:
            shots: Number of measurement shots to perform

        Returns:
            Dictionary mapping register names to lists of measurement results.
            Results are integers by default, or binary strings if
            with_binary_string_format() was used.

        Example:
            >>> from pecos_rslib import sim
            >>> from pecos_rslib.programs import QasmProgram
            >>> program = QasmProgram.from_string(qasm)
            >>> simulation = sim(program).build()
            >>> results = simulation.run(1000)
            >>> print(results["c"][:5])  # First 5 measurement results
            [0, 3, 0, 3, 0]  # Bell state measurements
        """

# QasmSimulationBuilder has been removed - use sim() API instead
# See sim() function for the modern approach to quantum simulations

class MappedGraph:
    """A graph data structure with support for arbitrary hashable node identifiers.

    This class provides a NetworkX-compatible API for graph operations with
    maximum weight perfect matching capabilities. Nodes can be any hashable
    type (strings, integers, tuples, etc.).

    The graph is undirected and supports weighted edges for use with matching
    algorithms in quantum error correction decoders.

    Example:
        >>> from pecos_rslib.graph import MappedGraph
        >>> g = MappedGraph()
        >>> g.add_edge("v1", "v2", weight=-10.0)
        >>> g.add_edge("v2", "v3", weight=-5.0)
        >>> print(g.nodes())
        ["v1", "v2", "v3"]
        >>> matching = g.max_weight_matching(max_cardinality=True)
        >>> print(matching)
        {"v1": "v2", "v2": "v1"}
    """

    def __init__(self) -> None:
        """Create a new empty MappedGraph."""

    def add_edge(
        self,
        a: object,
        b: object,
        weight: float | None = None,
        **kwargs: object,
    ) -> None:
        """Add an edge between nodes a and b.

        If the nodes do not exist, they will be created. Edge attributes
        (including weight) can be provided as keyword arguments.

        Args:
            a: First node identifier (must be hashable)
            b: Second node identifier (must be hashable)
            weight: Optional edge weight (defaults to 1.0 if not specified)
            **kwargs: Additional edge attributes to store (e.g., syn_path, data_path)

        Example:
            >>> g = MappedGraph()
            >>> g.add_edge("a", "b", weight=-10.0, data_path=[1, 2, 3])
        """

    def nodes(self) -> list[object]:
        """Return a list of all nodes in the graph.

        Returns:
            List of node identifiers in insertion order.

        Example:
            >>> g = MappedGraph()
            >>> g.add_edge("a", "b")
            >>> g.add_edge("b", "c")
            >>> print(g.nodes())
            ["a", "b", "c"]
        """

    def edges(self) -> list[tuple[object, object]]:
        """Return a list of all edges in the graph.

        Returns:
            List of (node1, node2) tuples representing edges.

        Example:
            >>> g = MappedGraph()
            >>> g.add_edge("a", "b")
            >>> g.add_edge("b", "c")
            >>> print(g.edges())
            [("a", "b"), ("b", "c")]
        """

    def node_count(self) -> int:
        """Return the number of nodes in the graph.

        Returns:
            Number of nodes.

        Example:
            >>> g = MappedGraph()
            >>> g.add_edge("a", "b")
            >>> print(g.node_count())
            2
        """

    def edge_count(self) -> int:
        """Return the number of edges in the graph.

        Returns:
            Number of edges.

        Example:
            >>> g = MappedGraph()
            >>> g.add_edge("a", "b")
            >>> g.add_edge("b", "c")
            >>> print(g.edge_count())
            2
        """

    def max_weight_matching(
        self, max_cardinality: bool = False, weight_multiplier: float = 1000.0
    ) -> dict[object, object]:
        """Compute the maximum weight perfect matching of the graph.

        Uses the Blossom algorithm to find a maximum weight matching.
        Weights are interpreted as negative distances (higher weight = better match).

        The matching algorithm internally uses integer weights. Float weights are
        converted by multiplying by weight_multiplier and casting to integers.

        Args:
            max_cardinality: If True, prioritize maximum cardinality over weight.
                           If False, prioritize maximum weight.
            weight_multiplier: Multiplier for converting float weights to integers.
                             Default is 1000.0 (preserves 3 decimal places).
                             Use 1.0 if weights are already integers.
                             Use higher values (10000.0+) for more decimal precision.

        Returns:
            Dictionary mapping nodes to their matched partners. Each edge
            appears twice (both directions).

        Example:
            >>> g = MappedGraph()
            >>> g.add_edge("a", "b", weight=-10.0)
            >>> g.add_edge("c", "d", weight=-5.0)
            >>> matching = g.max_weight_matching(max_cardinality=True)
            >>> print(matching)
            {"a": "b", "b": "a", "c": "d", "d": "c"}

            >>> # For integer weights, use weight_multiplier=1.0
            >>> g2 = MappedGraph()
            >>> g2.add_edge("x", "y", weight=-5)
            >>> g2.add_edge("z", "w", weight=-10)
            >>> matching2 = g2.max_weight_matching(max_cardinality=True, weight_multiplier=1.0)
        """

    def get_edge_data(self, a: object, b: object) -> dict[str, object]:
        """Get the attributes dictionary for an edge.

        Args:
            a: First node identifier
            b: Second node identifier

        Returns:
            Dictionary of edge attributes (including 'weight' and any custom attributes).

        Raises:
            KeyError: If the edge does not exist.

        Example:
            >>> g = MappedGraph()
            >>> g.add_edge("a", "b", weight=-10.0, data_path=[1, 2])
            >>> attrs = g.get_edge_data("a", "b")
            >>> print(attrs["weight"])
            -10.0
            >>> print(attrs["data_path"])
            [1, 2]
        """

    def subgraph(self, nodes: list[object]) -> MappedGraph:
        """Create a subgraph containing only the specified nodes.

        The subgraph includes all edges between the specified nodes.

        Args:
            nodes: List of node identifiers to include in the subgraph.

        Returns:
            A new MappedGraph containing only the specified nodes and their edges.

        Example:
            >>> g = MappedGraph()
            >>> g.add_edge("a", "b", weight=-10.0)
            >>> g.add_edge("b", "c", weight=-5.0)
            >>> subg = g.subgraph(["a", "b"])
            >>> print(subg.nodes())
            ["a", "b"]
            >>> print(subg.edge_count())
            1
        """

    def single_source_shortest_path(self, source: object) -> dict[object, list[object]]:
        """Compute shortest paths from source to all reachable nodes.

        Uses breadth-first search to find unweighted shortest paths.

        Args:
            source: Source node identifier.

        Returns:
            Dictionary mapping target nodes to paths. Each path is a list
            of nodes from source to target (inclusive).

        Example:
            >>> g = MappedGraph()
            >>> g.add_edge("a", "b")
            >>> g.add_edge("b", "c")
            >>> paths = g.single_source_shortest_path("a")
            >>> print(paths["c"])
            ["a", "b", "c"]
        """

class Graph:
    """Integer-indexed graph data structure for quantum error correction decoding.

    This class provides a high-performance graph with integer node IDs and
    a method-based API for edge construction and attribute management. It supports
    named nodes (with string labels) alongside integer IDs.

    The API follows a Rust-inspired pattern where edges are created first, then
    configured using separate method calls with edge IDs rather than kwargs.

    Example:
        >>> from pecos_rslib.graph import Graph
        >>> g = Graph()
        >>> n0 = g.add_node()  # Returns 0
        >>> n1 = g.add_node()  # Returns 1
        >>> g.add_edge(n0, n1)
        >>> edge_id = g.find_edge(n0, n1)
        >>> g.set_edge_weight(edge_id, 5.0)
        >>> g.set_edge_attr(edge_id, "data_path", [1, 2, 3])
    """

    def __init__(self) -> None:
        """Create a new empty Graph."""

    def add_node(self) -> int:
        """Add a new node with an automatically assigned integer ID.

        Returns:
            The integer ID of the newly created node.

        Example:
            >>> g = Graph()
            >>> n0 = g.add_node()  # Returns 0
            >>> n1 = g.add_node()  # Returns 1
        """

    def add_named_node(self, label: str) -> int:
        """Add a new node with a string label.

        Args:
            label: String label for the node.

        Returns:
            The integer ID of the newly created node.

        Example:
            >>> g = Graph()
            >>> v1 = g.add_named_node("virtual_1")
            >>> print(v1)  # Prints the integer ID
        """

    def node_by_label(self, label: str) -> int | None:
        """Look up a node's integer ID by its string label.

        Args:
            label: The string label to search for.

        Returns:
            The integer node ID, or None if not found.

        Example:
            >>> g = Graph()
            >>> v1 = g.add_named_node("virtual_1")
            >>> node_id = g.node_by_label("virtual_1")
            >>> print(node_id == v1)
            True
        """

    def get_label(self, node: int) -> str | None:
        """Get the string label for a node ID.

        Args:
            node: The integer node ID.

        Returns:
            The string label, or None if the node has no label.

        Example:
            >>> g = Graph()
            >>> v1 = g.add_named_node("virtual_1")
            >>> label = g.get_label(v1)
            >>> print(label)
            virtual_1
        """

    def add_edge(self, a: int | str, b: int | str) -> None:
        """Add an edge between two nodes with default weight of 1.0.

        Use set_edge_weight() and set_edge_attr() to configure the edge after creation.
        Nodes can be specified by integer IDs or string labels.

        Args:
            a: First node (integer ID or string label).
            b: Second node (integer ID or string label).

        Example:
            >>> g = Graph()
            >>> n0 = g.add_node()
            >>> n1 = g.add_node()
            >>> g.add_edge(n0, n1)
            >>> edge_id = g.find_edge(n0, n1)
            >>> g.set_edge_weight(edge_id, 5.0)
            >>> g.set_edge_attr(edge_id, "data_path", [1, 2, 3])
        """

    def find_edge(self, a: int | str, b: int | str) -> int | None:
        """Find an edge by its source and destination nodes.

        Args:
            a: First node (integer ID or string label).
            b: Second node (integer ID or string label).

        Returns:
            The integer edge ID, or None if the edge doesn't exist.

        Example:
            >>> g = Graph()
            >>> n0, n1 = g.add_node(), g.add_node()
            >>> g.add_edge(n0, n1)
            >>> edge_id = g.find_edge(n0, n1)
            >>> print(edge_id)
            0
        """

    def edge_weight(self, edge_id: int) -> float | None:
        """Get the weight of an edge by edge ID (low-level API).

        For most use cases, prefer `get_weight(a, b)` which uses node pairs.

        Args:
            edge_id: The integer edge ID.

        Returns:
            The edge weight, or None if the edge doesn't exist.

        Example:
            >>> edge_id = g.find_edge(n0, n1)
            >>> weight = g.edge_weight(edge_id)
            >>> print(weight)
            5.0
        """

    def set_edge_weight(self, edge_id: int, weight: float) -> None:
        """Set the weight of an edge by edge ID (low-level API).

        For most use cases, prefer `set_weight(a, b, weight)` which uses node pairs.

        Args:
            edge_id: The integer edge ID.
            weight: The new weight value.

        Example:
            >>> edge_id = g.find_edge(n0, n1)
            >>> g.set_edge_weight(edge_id, 10.0)
        """

    def set_weight(self, a: int | str, b: int | str, weight: float) -> None:
        """Set the weight of an edge between two nodes (NetworkX-style).

        This is the recommended way to set edge weights - no edge ID lookup needed!

        Args:
            a: First node (integer ID or string label)
            b: Second node (integer ID or string label)
            weight: The new weight value

        Example:
            >>> g.add_edge(n0, n1)
            >>> g.set_weight(n0, n1, 5.0)  # Much simpler!

            >>> # Works with labels too
            >>> g.set_weight("v1", "v2", 3.0)
        """

    def get_weight(self, a: int | str, b: int | str) -> float | None:
        """Get the weight of an edge between two nodes (NetworkX-style).

        This is the recommended way to get edge weights - no edge ID lookup needed!

        Args:
            a: First node (integer ID or string label)
            b: Second node (integer ID or string label)

        Returns:
            The edge weight, or None if the edge doesn't exist.

        Example:
            >>> weight = g.get_weight(n0, n1)
            >>> if weight is not None:
            ...     print(f"Weight: {weight}")
        """

    def edge_attr(self, edge_id: int, key: str) -> object:
        """Get a custom attribute value from an edge by edge ID (low-level API).

        For most use cases, prefer `get_attr(a, b, key)` which uses node pairs.

        Args:
            edge_id: The integer edge ID.
            key: The attribute key.

        Returns:
            The attribute value, or None if not found.

        Example:
            >>> edge_id = g.find_edge(n0, n1)
            >>> data_path = g.edge_attr(edge_id, "data_path")
            >>> print(data_path)
            [1, 2, 3]
        """

    def set_edge_attr(self, edge_id: int, key: str, value: object) -> None:
        """Set a custom attribute on an edge by edge ID (low-level API).

        For most use cases, prefer `set_attr(a, b, key, value)` which uses node pairs.

        Args:
            edge_id: The integer edge ID.
            key: The attribute key.
            value: The attribute value (int, float, str, bool, list[int], list[float]).

        Example:
            >>> edge_id = g.find_edge(n0, n1)
            >>> g.set_edge_attr(edge_id, "label", "boundary")
            >>> g.set_edge_attr(edge_id, "data_path", [1, 2, 3])
        """

    def remove_edge_attr(self, edge_id: int, key: str) -> object:
        """Remove a custom attribute from an edge by edge ID (low-level API).

        For most use cases, prefer `remove_attr(a, b, key)` which uses node pairs.

        Args:
            edge_id: The integer edge ID.
            key: The attribute key to remove.

        Returns:
            The removed value, or None if the attribute didn't exist.

        Example:
            >>> edge_id = g.find_edge(n0, n1)
            >>> old_value = g.remove_edge_attr(edge_id, "old_attr")
        """

    def set_attr(self, a: int | str, b: int | str, key: str, value: object) -> None:
        """Set an attribute on an edge between two nodes (NetworkX-style).

        This is the recommended way to set edge attributes - no edge ID lookup needed!

        Args:
            a: First node (integer ID or string label)
            b: Second node (integer ID or string label)
            key: The attribute key
            value: The attribute value (int, float, str, bool, list[int], list[str], or any JSON-serializable type)

        Example:
            >>> g.add_edge(n0, n1)
            >>> g.set_attr(n0, n1, "label", "boundary")
            >>> g.set_attr(n0, n1, "syn_path", [1, 2, 3])

            >>> # Works with labels too
            >>> g.set_attr("v1", "v2", "custom", "value")
        """

    def get_attr(self, a: int | str, b: int | str, key: str) -> object:
        """Get an attribute from an edge between two nodes (NetworkX-style).

        This is the recommended way to get edge attributes - no edge ID lookup needed!

        Args:
            a: First node (integer ID or string label)
            b: Second node (integer ID or string label)
            key: The attribute key

        Returns:
            The attribute value, or None if not found.

        Example:
            >>> label = g.get_attr(n0, n1, "label")
            >>> if label is not None:
            ...     print(f"Label: {label}")
        """

    def remove_attr(self, a: int | str, b: int | str, key: str) -> object:
        """Remove an attribute from an edge between two nodes (NetworkX-style).

        This is the recommended way to remove edge attributes - no edge ID lookup needed!

        Args:
            a: First node (integer ID or string label)
            b: Second node (integer ID or string label)
            key: The attribute key to remove

        Returns:
            The removed value, or None if the attribute didn't exist.

        Example:
            >>> g.set_attr(n0, n1, "temp", "value")
            >>> removed = g.remove_attr(n0, n1, "temp")
            >>> print(removed)  # "value"
        """

    def get_edge_data(self, a: int | str, b: int | str) -> dict[str, object]:
        """Get all attributes for an edge as a dictionary.

        This is a convenience method for reading multiple edge attributes at once.

        Args:
            a: First node (integer ID or string label).
            b: Second node (integer ID or string label).

        Returns:
            Dictionary of all edge attributes including 'weight'.

        Example:
            >>> g.add_edge(n0, n1)
            >>> edge_id = g.find_edge(n0, n1)
            >>> g.set_edge_weight(edge_id, 5.0)
            >>> g.set_edge_attr(edge_id, "label", "test")
            >>> data = g.get_edge_data(n0, n1)
            >>> print(data)
            {"weight": 5.0, "label": "test"}
        """

    def remove_edge(self, edge_id: int) -> bool:
        """Remove an edge from the graph.

        Args:
            edge_id: The integer edge ID.

        Returns:
            True if the edge was removed, False if it didn't exist.

        Example:
            >>> edge_id = g.find_edge(n0, n1)
            >>> was_removed = g.remove_edge(edge_id)
        """

    def has_node(self, node: int) -> bool:
        """Check if a node exists in the graph.

        Args:
            node: The integer node ID.

        Returns:
            True if the node exists, False otherwise.

        Example:
            >>> g = Graph()
            >>> n0 = g.add_node()
            >>> print(g.has_node(n0))
            True
            >>> print(g.has_node(999))
            False
        """

    def remove_node(self, node: int) -> None:
        """Remove a node and all its connected edges from the graph.

        Args:
            node: The integer node ID.

        Example:
            >>> g = Graph()
            >>> n0 = g.add_node()
            >>> g.remove_node(n0)
        """

    def nodes(self) -> list[int]:
        """Return a list of all node IDs in the graph.

        Returns:
            List of integer node IDs.

        Example:
            >>> g = Graph()
            >>> n0 = g.add_node()
            >>> n1 = g.add_node()
            >>> print(g.nodes())
            [0, 1]
        """

    def edges(self) -> list[tuple[int, int, float]]:
        """Return a list of all edges with their weights.

        Returns:
            List of (source, destination, weight) tuples.

        Example:
            >>> g = Graph()
            >>> n0, n1 = g.add_node(), g.add_node()
            >>> g.add_edge(n0, n1)
            >>> edge_id = g.find_edge(n0, n1)
            >>> g.set_edge_weight(edge_id, 5.0)
            >>> print(g.edges())
            [(0, 1, 5.0)]
        """

    def max_weight_matching(
        self, max_cardinality: bool = False, weight_multiplier: float = 1000.0
    ) -> dict[int, int]:
        """Compute the maximum weight perfect matching of the graph.

        Uses the Blossom algorithm to find a maximum weight matching.
        Weights are interpreted as negative distances (higher weight = better match).

        The matching algorithm internally uses integer weights. Float weights are
        converted by multiplying by weight_multiplier and casting to integers.

        Args:
            max_cardinality: If True, prioritize maximum cardinality over weight.
                           If False, prioritize maximum weight.
            weight_multiplier: Multiplier for converting float weights to integers.
                             Default is 1000.0 (preserves 3 decimal places).
                             Use 1.0 if weights are already integers.
                             Use higher values (10000.0+) for more decimal precision.

        Returns:
            Dictionary mapping node IDs to their matched partners. Each edge
            appears twice (both directions).

        Example:
            >>> g = Graph()
            >>> n0, n1, n2, n3 = [g.add_node() for _ in range(4)]
            >>> g.add_edge(n0, n1)
            >>> e1 = g.find_edge(n0, n1)
            >>> g.set_edge_weight(e1, -10.0)
            >>> g.add_edge(n2, n3)
            >>> e2 = g.find_edge(n2, n3)
            >>> g.set_edge_weight(e2, -5.0)
            >>> matching = g.max_weight_matching(max_cardinality=True)
            >>> print(matching)
            {0: 1, 1: 0, 2: 3, 3: 2}
        """

# Module functions
def run_qasm(
    qasm: str,
    shots: int,
    noise_model: GeneralNoiseModelBuilder | object | None = None,
    engine: QuantumEngine | None = None,
    workers: int | None = None,
    seed: int | None = None,
) -> dict[str, list[int]]:
    """Run a QASM simulation with specified parameters.

    Simple function interface for running quantum simulations without
    using the builder pattern.

    Args:
        qasm: OpenQASM 2.0 code as a string
        shots: Number of measurement shots to perform
        noise_model: Noise model instance or None for ideal simulation
        engine: Quantum engine or None for default (SparseStabilizer)
        workers: Number of worker threads or None for default (1)
        seed: Random seed or None for non-deterministic

    Returns:
        Dictionary mapping register names to measurement results

    Example:
        >>> results = run_qasm(qasm, shots=1000, seed=42)
    """

# qasm_sim has been removed - use sim() API instead
# Example migration:
#   Old: qasm_sim(qasm).seed(42).noise(noise).run(1000)
#   New: sim(QasmProgram.from_string(qasm)).seed(42).noise(noise).run(1000)

def get_noise_models() -> list[str]:
    """Get a list of available noise model names.

    Returns:
        List of noise model names like 'PassThrough', 'Depolarizing', etc.
    """

def get_quantum_engines() -> list[str]:
    """Get a list of available quantum engine names.

    Returns:
        List of engine names like 'StateVector', 'SparseStabilizer'
    """
