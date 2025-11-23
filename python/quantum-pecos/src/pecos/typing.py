# Copyright 2025 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Common type definitions used throughout PECOS."""

from __future__ import annotations

from typing import Protocol, TypedDict

# JSON-like types for gate parameters and metadata
JSONValue = str | int | float | bool | dict[str, "JSONValue"] | list["JSONValue"] | None
JSONDict = dict[str, JSONValue]

# Gate parameter type - used for **params in various gate operations
GateParams = JSONDict

# Simulator gate parameters - these are passed to simulator gate functions
SimulatorGateParams = JSONDict

# Simulator initialization parameters
SimulatorInitParams = (
    JSONDict  # Parameters for simulator initialization (e.g., MPS config)
)

# QECC parameter types
QECCParams = JSONDict  # Parameters for QECC initialization
QECCGateParams = JSONDict  # Parameters for QECC gate operations
QECCInstrParams = JSONDict  # Parameters for QECC instruction operations


# Error model parameter types
class ErrorParams(TypedDict, total=False):
    """Type definition for error parameters."""

    p: float
    p1: float
    p2: float
    p2_mem: float | None
    p_meas: float | tuple[float, ...]
    p_init: float
    scale: float
    noiseless_qubits: set[int]


# Threshold calculation types
class ThresholdResult(TypedDict):
    """Type definition for threshold calculation results."""

    distance: int | list[int]
    error_rates: list[float]
    logical_rates: list[float]
    time_rates: list[float] | None


# Fault tolerance checking types
class SpacetimeLocation(TypedDict):
    """Type definition for spacetime location in fault tolerance checking."""

    tick: int
    location: tuple[int, ...]
    before: bool
    symbol: str
    metadata: dict[str, int | str | bool]


class FaultDict(TypedDict, total=False):
    """Type definition for fault dictionary."""

    faults: list[tuple[int, ...]]
    locations: list[tuple[int, ...]]
    symbols: list[str]


# Stabilizer verification types
class StabilizerCheckDict(TypedDict, total=False):
    """Type definition for stabilizer check dictionary."""

    X: set[int]
    Y: set[int]
    Z: set[int]


class StabilizerVerificationResult(TypedDict):
    """Type definition for stabilizer verification results."""

    stabilizers: list[StabilizerCheckDict]
    destabilizers: list[StabilizerCheckDict]
    logicals_x: list[StabilizerCheckDict]
    logicals_z: list[StabilizerCheckDict]
    distance: int | None


# Circuit execution output types
class OutputDict(TypedDict, total=False):
    """Type definition for output dictionary used in circuit execution."""

    # Common keys based on codebase usage
    syndrome: set[int]
    measurements: dict[str, int | list[int]]
    classical_registers: dict[str, int]


# Logical operator types
LogicalOperator = dict[
    str,
    set[int],
]  # Maps Pauli operator ('X', 'Y', 'Z') to qubit indices

# Gate location types
Location = int | tuple[int, ...]  # Single qubit or multi-qubit gate location
LocationSet = (
    set[Location] | list[Location] | tuple[Location, ...]
)  # Collection of locations


class LogicalOpInfo(TypedDict):
    """Information about a logical operator."""

    X: set[int]
    Z: set[int]
    equiv_ops: tuple[str, ...]


# Graph protocol types
# Node identifiers can be any hashable type (str, int, tuple, etc.)
Node = object
# Edges are represented as tuples of two nodes
Edge = tuple[Node, Node]
# Paths are lists of nodes
Path = list[Node]


class GraphProtocol(Protocol):
    """Protocol for graph objects used in decoder precomputation and algorithms.

    This protocol defines the interface that graph implementations must provide
    to be compatible with PECOS decoders and graph algorithms.
    """

    def nodes(self) -> list[Node]:
        """Return list of nodes in the graph.

        Returns:
            List of node identifiers in the graph.
        """
        ...

    def add_edge(
        self,
        a: Node,
        b: Node,
        weight: float | None = None,
        **kwargs: object,
    ) -> None:
        """Add an edge between nodes a and b.

        Args:
            a: First node identifier.
            b: Second node identifier.
            weight: Optional edge weight.
            **kwargs: Additional edge attributes.
        """
        ...

    def single_source_shortest_path(self, source: Node) -> dict[Node, Path]:
        """Compute shortest paths from source to all other nodes.

        Args:
            source: Source node identifier.

        Returns:
            Dictionary mapping target nodes to paths (list of nodes from source to target).
        """
        ...
