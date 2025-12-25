# Copyright 2025 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Convert HUGR (from Guppy) to PECOS DAG representation.

This module provides utilities to convert compiled Guppy quantum programs
(represented as HUGR - Hierarchical Unified Graph Representation) into
PECOS DAG structures for analysis and optimization.

Currently supports basic quantum circuits without classical control flow,
loops, or conditionals. Will raise UnsupportedHugrStructureError for
unsupported constructs.

Example::

    from guppylang import guppy
    from guppylang.std.quantum import h, cx, qubit, measure
    from pecos.circuit_converters.hugr_to_dag import hugr_to_dag

    @guppy
    def bell() -> tuple[bool, bool]:
        q0 = qubit()
        q1 = qubit()
        h(q0)
        cx(q0, q1)
        return measure(q0), measure(q1)

    package = bell.compile()
    dag = hugr_to_dag(package.modules[0])
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from pecos.graph import DAG

if TYPE_CHECKING:
    from hugr import Hugr


class CompiledPackage(Protocol):
    """Protocol for a compiled Guppy package."""

    @property
    def modules(self) -> list[Hugr]:
        """List of HUGR modules in the compiled package."""
        ...


class GuppyFunction(Protocol):
    """Protocol for a Guppy-decorated function that can be compiled."""

    def compile(self) -> CompiledPackage:
        """Compile the Guppy function to a package containing HUGR modules."""
        ...


class UnsupportedHugrStructureError(Exception):
    """Raised when the HUGR contains unsupported structures.

    This converter only supports basic quantum circuits without:
    - Classical control flow (Conditional nodes with quantum operations)
    - Loops (TailLoop nodes containing quantum operations)
    - Nested CFG structures with quantum operations
    """


# Quantum operation extensions we recognize
QUANTUM_EXTENSIONS = {"tket.quantum"}

# Operations that represent quantum gates (vs allocation/measurement)
GATE_OPERATIONS = {
    # Single-qubit gates
    "H",
    "X",
    "Y",
    "Z",
    "S",
    "Sdg",
    "T",
    "Tdg",
    "SX",
    "SXdg",
    "Rx",
    "Ry",
    "Rz",
    "U1",
    "U2",
    "U3",
    # Two-qubit gates
    "CX",
    "CY",
    "CZ",
    "CH",
    "SWAP",
    "ISWAP",
    "CRx",
    "CRy",
    "CRz",
    "CU1",
    "CU3",
    "ECR",
    "ZZMax",
    # Three-qubit gates
    "CCX",
    "CSWAP",
    "CCZ",
}

# Operations for qubit lifecycle
ALLOC_OPERATIONS = {"QAlloc"}
MEASURE_OPERATIONS = {"Measure", "MeasureFree"}

# All quantum operations
ALL_QUANTUM_OPERATIONS = GATE_OPERATIONS | ALLOC_OPERATIONS | MEASURE_OPERATIONS


def _check_for_unsupported_structures(hugr: Hugr, quantum_op_parents: set[int]) -> None:
    """Check if any quantum operations are inside unsupported control structures.

    Args:
        hugr: The HUGR to check.
        quantum_op_parents: Set of parent node indices that contain quantum ops.

    Raises:
        UnsupportedHugrStructureError: If quantum ops are in unsupported structures.
    """
    # Build parent chain for each quantum op parent
    for parent_idx in quantum_op_parents:
        # Walk up the parent chain looking for problematic structures
        current_idx = parent_idx
        visited = set()

        while current_idx is not None and current_idx not in visited:
            visited.add(current_idx)
            from hugr import Node  # noqa: PLC0415

            node = Node(current_idx)
            try:
                node_data = hugr[node]
            except (KeyError, IndexError):
                break

            op_name = node_data.op.__class__.__name__

            # TailLoop containing quantum ops is not supported
            if op_name == "TailLoop":
                msg = (
                    f"Quantum operations inside TailLoop (node {current_idx}) "
                    "are not supported. Only straight-line quantum circuits "
                    "without loops are currently handled."
                )
                raise UnsupportedHugrStructureError(msg)

            # Conditional containing quantum ops might be problematic
            # (we allow it at the top level for measurement results, but not nested)
            if op_name == "Conditional" and node_data.parent is not None:
                # Check if this conditional is inside another control structure
                parent_node = node_data.parent
                parent_data = hugr[parent_node]
                parent_op = parent_data.op.__class__.__name__
                if parent_op in ("TailLoop", "Conditional", "Case"):
                    msg = (
                        f"Nested Conditional (node {current_idx}) containing "
                        "quantum operations is not supported."
                    )
                    raise UnsupportedHugrStructureError(msg)

            # Move to parent
            if node_data.parent is not None:
                current_idx = node_data.parent.idx
            else:
                break


def _extract_quantum_operations(hugr: Hugr) -> list[dict]:
    """Extract all quantum operations from the HUGR.

    Args:
        hugr: The HUGR to extract operations from.

    Returns:
        List of dicts containing operation info:
        - node_idx: The HUGR node index
        - op_name: Operation name (e.g., "H", "CX")
        - extension: Extension name (e.g., "tket.quantum")
        - parent_idx: Parent node index
        - incoming: List of (source_node_idx, source_port, dest_port)
        - outgoing: List of (source_port, dest_node_idx, dest_port)
    """
    operations = []
    quantum_op_parents = set()

    for node, data in hugr.nodes():
        if data.op.__class__.__name__ != "ExtOp":
            continue

        op_def = data.op.op_def()
        # _extension is the only public way to access the extension name
        ext_name = op_def._extension.name if op_def._extension else None  # noqa: SLF001

        if ext_name not in QUANTUM_EXTENSIONS:
            continue

        op_name = op_def.name
        if op_name not in ALL_QUANTUM_OPERATIONS:
            continue

        node_idx = node.idx
        parent_idx = data.parent.idx if data.parent else None
        if parent_idx is not None:
            quantum_op_parents.add(parent_idx)

        # Extract connectivity
        incoming = []
        for in_port, out_ports in hugr.incoming_links(node):
            for out_port in out_ports:
                # Skip order edges (port offset -1)
                if in_port.offset >= 0 and out_port.offset >= 0:
                    incoming.append(  # noqa: PERF401
                        (out_port.node.idx, out_port.offset, in_port.offset),
                    )

        outgoing = []
        for out_port, in_ports in hugr.outgoing_links(node):
            for in_port in in_ports:
                # Skip order edges
                if out_port.offset >= 0 and in_port.offset >= 0:
                    outgoing.append(  # noqa: PERF401
                        (out_port.offset, in_port.node.idx, in_port.offset),
                    )

        operations.append(
            {
                "node_idx": node_idx,
                "op_name": op_name,
                "extension": ext_name,
                "parent_idx": parent_idx,
                "incoming": incoming,
                "outgoing": outgoing,
            },
        )

    # Check for unsupported structures
    _check_for_unsupported_structures(hugr, quantum_op_parents)

    return operations


def _trace_qubit_dependencies(
    operations: list[dict],
    hugr: Hugr,
) -> list[tuple[int, int]]:
    """Trace qubit data flow to establish dependencies between operations.

    This follows the qubit wires through the HUGR to find which operations
    depend on which other operations.

    Args:
        operations: List of quantum operation info dicts.
        hugr: The HUGR for looking up intermediate nodes.

    Returns:
        List of (source_op_idx, target_op_idx) dependency edges.
    """
    # Build lookup from HUGR node index to operation list index
    node_to_op_idx = {op["node_idx"]: i for i, op in enumerate(operations)}

    # Track edges we've found
    edges = []

    # For each operation, trace its inputs back to find source operations
    for target_op_idx, op in enumerate(operations):
        for src_node_idx, src_port, _dest_port in op["incoming"]:
            # Direct connection to another quantum op?
            if src_node_idx in node_to_op_idx:
                source_op_idx = node_to_op_idx[src_node_idx]
                edges.append((source_op_idx, target_op_idx))
            else:
                # Need to trace back through intermediate nodes (Call, UnpackTuple, etc.)
                source_op_idx = _trace_back_to_quantum_op(
                    src_node_idx,
                    src_port,
                    hugr,
                    node_to_op_idx,
                )
                if source_op_idx is not None:
                    edges.append((source_op_idx, target_op_idx))

    return edges


def _trace_back_to_quantum_op(
    node_idx: int,
    port: int,
    hugr: Hugr,
    node_to_op_idx: dict[int, int],
) -> int | None:
    """Trace backwards from a node/port to find the source quantum operation.

    Args:
        node_idx: Starting node index.
        port: Starting port.
        hugr: The HUGR.
        node_to_op_idx: Mapping from HUGR node index to operation list index.

    Returns:
        The operation list index of the source quantum op, or None if not found.
    """
    from hugr import Node  # noqa: PLC0415

    visited: set[tuple[int, int]] = set()
    stack = [(node_idx, port)]

    while stack:
        current_idx, current_port = stack.pop()

        if (current_idx, current_port) in visited:
            continue
        visited.add((current_idx, current_port))

        # Check if this is a quantum operation
        if current_idx in node_to_op_idx:
            return node_to_op_idx[current_idx]

        # Otherwise, trace back through this node's inputs
        try:
            node = Node(current_idx)
            for _in_port, out_ports in hugr.incoming_links(node):
                stack.extend(
                    (out_port.node.idx, out_port.offset)
                    for out_port in out_ports
                    if out_port.offset >= 0
                )
        except (KeyError, IndexError):
            continue

    return None


def hugr_to_dag(
    hugr: Hugr,
    *,
    include_alloc: bool = True,
    include_measure: bool = True,
) -> DAG:
    """Convert a HUGR quantum circuit to a PECOS DAG.

    The resulting DAG has nodes representing quantum operations with attributes
    accessible via ``dag.node_attrs(node_idx)``:

    - op_name: The operation name (e.g., "H", "CX", "QAlloc", "MeasureFree")
    - extension: The HUGR extension (e.g., "tket.quantum")
    - hugr_node_idx: The original HUGR node index
    - op_type: One of "gate", "alloc", or "measure"

    Edges represent qubit data dependencies (a qubit flows from source to target).

    Args:
        hugr: A HUGR containing a compiled quantum circuit.
        include_alloc: If True, include QAlloc operations as nodes.
        include_measure: If True, include Measure/MeasureFree operations as nodes.

    Returns:
        A DAG representing the quantum circuit.

    Raises:
        UnsupportedHugrStructureError: If the HUGR contains unsupported structures
            like loops or nested conditionals with quantum operations.

    Example:
        >>> from guppylang import guppy
        >>> from guppylang.std.quantum import h, qubit, measure
        >>> @guppy
        ... def simple() -> bool:
        ...     q = qubit()
        ...     q = h(q)
        ...     return measure(q)
        ...
        >>> dag = hugr_to_dag(simple.compile().modules[0])
        >>> len(dag)  # Number of nodes
        3
    """
    # Extract quantum operations
    operations = _extract_quantum_operations(hugr)

    # Filter operations based on options
    filtered_ops = []
    for op in operations:
        op_name = op["op_name"]
        if op_name in ALLOC_OPERATIONS and not include_alloc:
            continue
        if op_name in MEASURE_OPERATIONS and not include_measure:
            continue
        filtered_ops.append(op)

    operations = filtered_ops

    # Trace dependencies
    edges = _trace_qubit_dependencies(operations, hugr)

    # Build the DAG
    dag = DAG()

    # Add nodes with attributes
    for op in operations:
        op_name = op["op_name"]

        # Determine operation type
        if op_name in GATE_OPERATIONS:
            op_type = "gate"
        elif op_name in ALLOC_OPERATIONS:
            op_type = "alloc"
        else:
            op_type = "measure"

        node_idx = dag.add_node()
        # DAG nodes are added sequentially starting from 0
        attrs = dag.node_attrs(node_idx)
        attrs["op_name"] = op_name
        attrs["extension"] = op["extension"]
        attrs["hugr_node_idx"] = op["node_idx"]
        attrs["op_type"] = op_type

    # Add edges
    for src_idx, tgt_idx in edges:
        # Avoid duplicate edges and self-loops
        if src_idx != tgt_idx and dag.find_edge(src_idx, tgt_idx) is None:
            dag.add_edge(src_idx, tgt_idx)

    return dag


def guppy_to_dag(
    guppy_func: GuppyFunction,
    *,
    include_alloc: bool = True,
    include_measure: bool = True,
) -> DAG:
    """Convert a Guppy-decorated function to a PECOS DAG.

    This is a convenience wrapper around hugr_to_dag that handles
    compilation of the Guppy function.

    Args:
        guppy_func: A function decorated with @guppy.
        include_alloc: If True, include QAlloc operations as nodes.
        include_measure: If True, include Measure/MeasureFree operations as nodes.

    Returns:
        A DAG representing the quantum circuit.

    Raises:
        UnsupportedHugrStructureError: If the HUGR contains unsupported structures
            like loops or nested conditionals with quantum operations.

    Example::

        from guppylang import guppy
        from guppylang.std.quantum import h, qubit, measure
        from pecos.circuit_converters import guppy_to_dag

        @guppy
        def simple() -> bool:
            q = qubit()
            h(q)
            return measure(q)

        dag = guppy_to_dag(simple)
        len(dag)  # Number of nodes: 3
    """
    package = guppy_func.compile()
    hugr = package.modules[0]
    return hugr_to_dag(
        hugr,
        include_alloc=include_alloc,
        include_measure=include_measure,
    )


def dag_to_gate_sequence(dag: DAG) -> list[dict]:
    """Convert a DAG back to a topologically-sorted sequence of gates.

    Args:
        dag: A DAG created by hugr_to_dag.

    Returns:
        List of gate dictionaries in topological order, each containing:
        - op_name: The operation name
        - op_type: "gate", "alloc", or "measure"
        - node_idx: The DAG node index
    """
    sorted_nodes = dag.topological_sort()

    gates = []
    for node_idx in sorted_nodes:
        attrs = dag.node_attrs(node_idx)
        gates.append(
            {
                "op_name": attrs.get("op_name"),
                "op_type": attrs.get("op_type"),
                "node_idx": node_idx,
            },
        )

    return gates
