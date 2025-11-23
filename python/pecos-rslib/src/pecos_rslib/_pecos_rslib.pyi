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
