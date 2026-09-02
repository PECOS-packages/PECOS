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

"""Convert HUGR (from Guppy) to SLR-AST representation.

This module provides utilities to convert compiled Guppy quantum programs
(represented as HUGR - Hierarchical Unified Graph Representation) into
SLR-AST (Abstract Syntax Tree) structures for analysis, optimization,
and code generation to other targets.

Supports:
- Straight-line quantum circuits
- One conditional (if/else) whose predicate is the direct read of a measurement
- Two-qubit gates (CX, CZ, etc.)
- Parameterized rotations (Rx/Ry/Rz/Rzz) with a constant angle

A branch predicate is read off the HUGR wire that selects the successor block.
Anything other than a direct measurement read -- a negation, a comparison, a
purely classical expression, or a loop counter -- is rejected rather than
lowered to a fabricated condition, so ``while`` loops are not convertible today
even though the CFG shape itself is recognized.

Will raise UnsupportedHugrStructureError for unsupported CFG patterns.

Examples::

    Basic Bell state circuit:

    >>> from guppylang import guppy
    >>> from guppylang.std.quantum import h, cx, qubit, measure
    >>> from pecos.circuit_converters.hugr_to_ast import guppy_to_ast
    >>>
    >>> @guppy
    ... def bell() -> tuple[bool, bool]:
    ...     q0 = qubit()
    ...     q1 = qubit()
    ...     h(q0)
    ...     cx(q0, q1)
    ...     return measure(q0).read(), measure(q1).read()
    ...
    >>>
    >>> ast = guppy_to_ast(bell)
    >>> # Use ast with SLR-AST analysis, optimization, or code generation

    Conditional circuit with measurement feedback:

    >>> @guppy
    ... def conditional() -> bool:
    ...     q = qubit()
    ...     h(q)
    ...     result = measure(q).read()
    ...     q2 = qubit()
    ...     if result:
    ...         x(q2)
    ...     return measure(q2).read()
    ...
    >>>
    >>> ast = guppy_to_ast(conditional)
    >>> # AST contains IfStmt node for the conditional

    Parameterized rotation:

    >>> from guppylang.std.angles import pi
    >>> from guppylang.std.quantum import rz
    >>>
    >>> @guppy
    ... def rotation() -> bool:
    ...     q = qubit()
    ...     h(q)
    ...     rz(q, pi / 4)
    ...     return measure(q).read()
    ...
    >>>
    >>> ast = guppy_to_ast(rotation)
    >>> # The RZ GateOp carries the angle in its `params`
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from pecos.slr.ast.nodes import (
    AllocatorDecl,
    BitRef,
    GateKind,
    GateOp,
    IfStmt,
    LiteralExpr,
    MeasureOp,
    PrepareOp,
    Program,
    RegisterDecl,
    SlotRef,
    VarExpr,
    WhileStmt,
)

if TYPE_CHECKING:
    from hugr import Hugr, Node

    from pecos.slr.ast.nodes import (
        Declaration,
        Expression,
        Statement,
    )


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

    This converter supports:
    - Straight-line quantum circuits
    - One conditional whose predicate directly reads a measurement

    Unsupported structures include:
    - Sequential, nested, or loop-contained conditionals
    - Branch predicates that are not a direct measurement read (negations,
      comparisons, classical expressions, loop counters)
    - Rotation gates whose angle is not a constant
    - Complex CFG patterns that cannot be mapped to structured control flow
    - Irreducible control flow graphs
    """


# Quantum operation extensions we recognize
QUANTUM_EXTENSIONS = {"tket.quantum"}

# Map from HUGR gate names to SLR-AST GateKind
GATE_KIND_MAP: dict[str, GateKind] = {
    # Single-qubit Clifford gates
    "H": GateKind.H,
    "X": GateKind.X,
    "Y": GateKind.Y,
    "Z": GateKind.Z,
    "S": GateKind.SZ,
    "Sdg": GateKind.SZdg,
    # T gates
    "T": GateKind.T,
    "Tdg": GateKind.Tdg,
    # Sqrt gates
    "SX": GateKind.SX,
    "SXdg": GateKind.SXdg,
    # Two-qubit gates
    "CX": GateKind.CX,
    "CY": GateKind.CY,
    "CZ": GateKind.CZ,
    "CH": GateKind.CH,
    # Rotation gates (single-qubit)
    "Rx": GateKind.RX,
    "Ry": GateKind.RY,
    "Rz": GateKind.RZ,
    # Rotation gates (two-qubit)
    "Rzz": GateKind.RZZ,
}

# Operations for qubit lifecycle
ALLOC_OPERATIONS = {"QAlloc"}
MEASURE_OPERATIONS = {"Measure", "MeasureFree"}
GATE_OPERATIONS = set(GATE_KIND_MAP.keys())

_TWO_QUBIT_GATES = {"CX", "CY", "CZ", "CH", "Rzz"}

# Gates that carry a rotation angle alongside their qubit operands.  The angle
# is a required operand: emitting one of these without it is not the source
# circuit, so an unresolvable angle is rejected rather than dropped.
_ROTATION_GATES = {"Rx", "Ry", "Rz", "Rzz"}

# HUGR extension op that reads a measurement's classical outcome.
_MEASUREMENT_READ_OPS = {("tket.measurement", "Read")}

# All quantum operations we handle
ALL_QUANTUM_OPERATIONS = GATE_OPERATIONS | ALLOC_OPERATIONS | MEASURE_OPERATIONS


@dataclass
class BlockInfo:
    """Information about a DataflowBlock."""

    node_idx: int
    parent_idx: int
    operations: list[dict] = field(default_factory=list)
    outgoing_edges: list[tuple[int, int, int]] = field(
        default_factory=list,
    )  # (port, target_block, target_port)
    incoming_blocks: set[int] = field(
        default_factory=set,
    )  # Block indices that have edges to this block


@dataclass
class LoopInfo:
    """Information about a detected loop."""

    header_block: int  # The loop header block (receives back-edge)
    body_blocks: list[int]  # Blocks that form the loop body
    exit_block: int | None  # Block to go to when loop exits
    back_edge_source: int  # Block that has the back-edge to header


@dataclass
class CFGStructure:
    """Analyzed CFG structure."""

    blocks: dict[int, BlockInfo]  # block_idx -> BlockInfo
    entry_block: int | None = None
    exit_block: int | None = None
    cfg_node: int | None = None
    is_straight_line: bool = True
    conditional_blocks: list[tuple[int, int, int, int | None]] = field(
        default_factory=list,
    )  # (entry, then, else, continuation)
    loops: list[LoopInfo] = field(default_factory=list)  # Detected loops
    postlude_operations: list[dict] = field(default_factory=list)


class HugrToAstConverter:
    """Converts HUGR to SLR-AST Program."""

    def __init__(self, hugr: Hugr) -> None:
        """Initialize the converter.

        Args:
            hugr: The HUGR to convert.
        """
        self.hugr = hugr
        self.qubit_allocations: dict[int, int] = {}  # HUGR node idx -> qubit index
        self.next_qubit_idx = 0
        self.allocator_name = "q"  # Default allocator name
        self.node_to_qubit: dict[int, int] = {}  # Track qubit for each node
        self.node_port_to_qubit: dict[tuple[int, int], int] = {}  # Track qubit for each node output port
        self.measurement_results: dict[int, str] = {}  # node_idx -> result variable name
        self.next_result_idx = 0
        self.block_input_nodes: dict[int, int] = {}  # block_idx -> Input node idx
        self.block_output_nodes: dict[int, int] = {}  # block_idx -> Output node idx
        self.block_output_qubit_ports: dict[int, dict[int, int]] = {}  # block_idx -> {port: qubit_idx}

    def convert(self) -> Program:
        """Convert the HUGR to an SLR-AST Program.

        Returns:
            An SLR-AST Program representing the quantum circuit.

        Raises:
            UnsupportedHugrStructureError: If the HUGR contains unsupported structures.
        """
        # Analyze CFG structure
        cfg = self._analyze_cfg()

        # Check for unsupported structures (loops)
        self._check_for_loops()

        # Extract all quantum operations across all blocks
        all_operations = self._extract_all_operations(cfg)

        # Build qubit allocation map
        self._build_qubit_map(all_operations)

        # Determine number of qubits
        num_qubits = len(self.qubit_allocations)

        # Build statements based on CFG structure (do this first to populate measurement_results)
        if cfg.is_straight_line:
            statements = self._build_straight_line_statements(cfg)
        else:
            statements = self._build_cfg_statements(cfg)

        # Create declarations
        decl_list: list[Declaration] = []
        if num_qubits > 0:
            decl_list.append(
                AllocatorDecl(name=self.allocator_name, capacity=num_qubits),
            )

        # Add classical register declarations for measurement results
        decl_list.extend(RegisterDecl(name=result_var, size=1) for result_var in self.measurement_results.values())

        declarations = tuple(decl_list)

        # Extract function name
        func_name = self._extract_function_name()

        return Program(
            name=func_name,
            declarations=declarations,
            body=tuple(statements),
        )

    def _analyze_cfg(self) -> CFGStructure:
        """Analyze the CFG structure of the HUGR.

        Returns:
            CFGStructure with block information.
        """
        cfg = CFGStructure(blocks={})

        # Guppy 1 keeps straight-line functions directly below ``FuncDefn``;
        # control-flow functions still use nested ``DataflowBlock`` nodes.
        # Prefer the latter when present so the CFG entry remains unchanged.
        has_dataflow_blocks = any(data.op.__class__.__name__ == "DataflowBlock" for _, data in self.hugr.nodes())
        block_ops = {"DataflowBlock"} if has_dataflow_blocks else {"FuncDefn"}

        # Find all blocks and ExitBlock
        for node, data in self.hugr.nodes():
            op_name = data.op.__class__.__name__
            parent_idx = data.parent.idx if data.parent else None

            if op_name in block_ops:
                block = BlockInfo(node_idx=node.idx, parent_idx=parent_idx or -1)
                cfg.blocks[node.idx] = block
                if op_name == "DataflowBlock":
                    cfg.cfg_node = parent_idx

                # Get outgoing edges
                for out_port, in_ports in self.hugr.outgoing_links(node):
                    for in_port in in_ports:
                        block.outgoing_edges.append(
                            (out_port.offset, in_port.node.idx, in_port.offset),
                        )

            elif op_name == "ExitBlock":
                cfg.exit_block = node.idx

        # Populate incoming_blocks for each block
        for block in cfg.blocks.values():
            for _port, target, _tport in block.outgoing_edges:
                if target in cfg.blocks:
                    cfg.blocks[target].incoming_blocks.add(block.node_idx)

        # Determine entry block (block with no incoming edges from other blocks)
        for block_idx, block in cfg.blocks.items():
            if not block.incoming_blocks:
                cfg.entry_block = block_idx
                break

        # Extract operations and Input/Output nodes for each block
        for node, data in self.hugr.nodes():
            op_name = data.op.__class__.__name__
            parent_idx = data.parent.idx if data.parent else None

            # Track Input/Output nodes for each block
            if parent_idx in cfg.blocks:
                if op_name == "Input":
                    self.block_input_nodes[parent_idx] = node.idx
                elif op_name == "Output":
                    self.block_output_nodes[parent_idx] = node.idx

            if op_name != "ExtOp":
                continue

            # Guppy 1 keeps setup operations directly below FuncDefn before
            # its CFG, but it also keeps return-value operations there after
            # the CFG. Only the former belong to the entry block: trailing
            # gates and measurements must be emitted after the structured
            # control flow.
            operation_parent_idx = parent_idx
            parent_op_name = self.hugr[data.parent].op.__class__.__name__ if data.parent is not None else None
            if has_dataflow_blocks and parent_op_name == "FuncDefn" and cfg.blocks:
                first_cfg_block = min(cfg.blocks)
                last_cfg_block = max(cfg.blocks)
                if node.idx < first_cfg_block:
                    operation_parent_idx = first_cfg_block
                elif node.idx > last_cfg_block:
                    operation_parent_idx = None

            custom_op = data.op.to_custom_op()
            ext_name = custom_op.extension
            ext_op_name = custom_op.op_name

            if ext_name not in QUANTUM_EXTENSIONS or ext_op_name not in ALL_QUANTUM_OPERATIONS:
                continue
            incoming = self._get_incoming_connections(node)
            outgoing = self._get_outgoing_connections(node)
            operation = {
                "node_idx": node.idx,
                "op_name": ext_op_name,
                "parent_idx": operation_parent_idx,
                "incoming": incoming,
                "outgoing": outgoing,
            }

            if operation_parent_idx in cfg.blocks:
                cfg.blocks[operation_parent_idx].operations.append(operation)
            elif has_dataflow_blocks and parent_op_name == "FuncDefn" and node.idx > max(cfg.blocks):
                cfg.postlude_operations.append(operation)

        # Determine if straight-line or has control flow
        if len(cfg.blocks) == 1:
            cfg.is_straight_line = True
        elif len(cfg.blocks) > 1:
            cfg.is_straight_line = False
            # First detect loops (back-edges), then identify conditionals even
            # when a loop is present so unsupported combinations cannot fall
            # through to the lossy loop-only lowering below.
            self._identify_loops(cfg)
            self._identify_conditional_pattern(cfg)
            self._validate_control_flow_shape(cfg)

        return cfg

    def _validate_control_flow_shape(self, cfg: CFGStructure) -> None:
        """Reject CFG forms that need recursive structured lowering.

        The converter deliberately supports one conditional or one simple
        loop.  Continuing with a partial traversal for richer shapes can
        attach a later operation to a plausible but incorrect wire.
        """
        conditional_headers = [
            block_idx
            for block_idx in cfg.blocks
            if self._is_conditional_header(cfg, block_idx)
            and not any(loop.header_block == block_idx for loop in cfg.loops)
        ]
        if cfg.loops and conditional_headers:
            msg = "HUGR CFG combines a loop with a conditional; recursive lowering is not supported"
            raise UnsupportedHugrStructureError(msg)
        # ``_identify_conditional_pattern`` intentionally recognizes only the
        # entry conditional that the flat lowering can emit.  Count every
        # conditional header here instead, otherwise a second conditional in
        # the continuation is silently omitted from the AST.
        if len(conditional_headers) > 1:
            msg = "HUGR CFG has sequential or nested conditionals; recursive lowering is not supported"
            raise UnsupportedHugrStructureError(msg)

    def _identify_loops(self, cfg: CFGStructure) -> None:
        """Identify loop patterns in the CFG.

        A loop is detected when there's a back-edge: an edge from a block
        to a block that appears earlier in traversal order (lower index
        or reachable without going through the target).

        Args:
            cfg: The CFG structure to analyze.
        """
        # Find back-edges: edges where target has lower index than source
        # and target has multiple incoming edges (from entry and from loop body)
        for block_idx, block in cfg.blocks.items():
            for _port, target, _tport in block.outgoing_edges:
                if target in cfg.blocks and target < block_idx:
                    # This is a potential back-edge
                    target_block = cfg.blocks[target]

                    # Verify it's a loop header (has incoming from before and after)
                    has_forward_incoming = any(inc < target for inc in target_block.incoming_blocks)
                    has_forward_incoming |= target == min(cfg.blocks)
                    has_back_edge = block_idx in target_block.incoming_blocks

                    if has_forward_incoming and has_back_edge:
                        # Found a loop! Identify body and exit blocks
                        body_blocks = self._find_loop_body(cfg, target, block_idx)
                        exit_block = self._find_loop_exit(cfg, target, body_blocks)

                        if exit_block is not None:
                            cfg.loops.append(
                                LoopInfo(
                                    header_block=target,
                                    body_blocks=body_blocks,
                                    exit_block=exit_block,
                                    back_edge_source=block_idx,
                                ),
                            )

    def _find_loop_body(
        self,
        cfg: CFGStructure,
        header: int,
        back_edge_source: int,
    ) -> list[int]:
        """Find all blocks that form the loop body.

        Args:
            cfg: The CFG structure.
            header: The loop header block.
            back_edge_source: The block with the back-edge to header.

        Returns:
            List of block indices in the loop body.
        """
        # Start from header, find blocks reachable that lead back to header
        body_blocks = []

        # The back-edge source is definitely in the body
        body_blocks.append(back_edge_source)

        # Find other blocks between header and back-edge source
        # that are part of the loop (lead to back-edge source)
        for block_idx in cfg.blocks:
            if block_idx == header:
                continue
            if block_idx == back_edge_source:
                continue

            # Check if this block leads to back_edge_source
            visited = set()
            stack = [block_idx]
            leads_to_back_edge = False

            while stack and not leads_to_back_edge:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)

                if current == back_edge_source:
                    leads_to_back_edge = True
                    break

                if current == header:
                    continue  # Don't go through header

                if current in cfg.blocks:
                    for _port, target, _tport in cfg.blocks[current].outgoing_edges:
                        if target in cfg.blocks:
                            stack.append(target)

            if leads_to_back_edge and block_idx not in body_blocks:
                body_blocks.append(block_idx)

        return body_blocks

    def _find_loop_exit(
        self,
        cfg: CFGStructure,
        header: int,
        body_blocks: list[int],
    ) -> int | None:
        """Find the exit block for a loop.

        Args:
            cfg: The CFG structure.
            header: The loop header block.
            body_blocks: List of blocks in the loop body.

        Returns:
            The exit block index, or None if not found.
        """
        # The exit block is a target of the header that's not in the body
        header_block = cfg.blocks[header]
        body_set = set(body_blocks)

        for _port, target, _tport in header_block.outgoing_edges:
            if target in cfg.blocks and target not in body_set:
                return target
            if target == cfg.exit_block:
                return target

        return None

    def _identify_conditional_pattern(self, cfg: CFGStructure) -> None:
        """Identify conditional patterns in the CFG.

        A conditional pattern looks like:
        - Entry block with 2 outgoing edges (to then/else blocks)
        - Then and else blocks eventually lead to a continuation block

        Supports nested conditionals by following control flow paths.

        Args:
            cfg: The CFG structure to analyze.
        """
        if cfg.entry_block is None:
            return

        entry = cfg.blocks.get(cfg.entry_block)
        if entry is None:
            return

        # Entry block should have exactly 2 outgoing edges to different blocks
        block_edges = [(port, target) for port, target, _tport in entry.outgoing_edges if target in cfg.blocks]

        if len(block_edges) == 2:
            # Port 0 = else branch, Port 1 = then branch (Guppy convention)
            block_edges.sort(key=lambda x: x[0])
            else_block = block_edges[0][1]
            then_block = block_edges[1][1]

            # Find eventual continuation block (where both branches converge)
            # Follow through nested conditionals
            then_eventual = self._find_eventual_targets(cfg, then_block)
            else_eventual = self._find_eventual_targets(cfg, else_block)

            continuation = then_eventual & else_eventual
            if len(continuation) >= 1:
                # Pick the first reachable common block
                cont_block = min(continuation)
                cfg.conditional_blocks.append(
                    (cfg.entry_block, then_block, else_block, cont_block),
                )
            elif all(self._reaches_exit(cfg, branch) for branch in (then_block, else_block)):
                cfg.conditional_blocks.append(
                    (cfg.entry_block, then_block, else_block, None),
                )

    def _reaches_exit(self, cfg: CFGStructure, start_block: int) -> bool:
        """Return whether a CFG path from ``start_block`` reaches its exit block."""
        visited = set()
        stack = [start_block]

        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            if current not in cfg.blocks:
                continue
            for _port, target, _tport in cfg.blocks[current].outgoing_edges:
                if target == cfg.exit_block:
                    return True
                if target in cfg.blocks:
                    stack.append(target)
        return False

    def _find_eventual_targets(self, cfg: CFGStructure, start_block: int) -> set[int]:
        """Find all blocks eventually reachable from a starting block.

        Follows the control flow through the CFG to find exit points.

        Args:
            cfg: The CFG structure.
            start_block: The block to start from.

        Returns:
            Set of block indices that are eventual targets.
        """
        eventual = set()
        visited = set()
        stack = [start_block]

        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)

            if current not in cfg.blocks:
                continue

            block = cfg.blocks[current]
            targets = [t for _, t, _ in block.outgoing_edges if t in cfg.blocks]

            if not targets:
                # This is a terminal block (leads to exit)
                eventual.add(current)
            elif len(targets) == 1:
                # Single outgoing edge - follow it
                eventual.add(targets[0])
                stack.append(targets[0])
            else:
                # Multiple outgoing edges (nested conditional)
                # Follow all branches to find where they converge
                stack.extend(targets)

        return eventual

    def _check_for_loops(self) -> None:
        """Check for loop structures and raise error if found."""
        for _node, data in self.hugr.nodes():
            if data.op.__class__.__name__ == "TailLoop":
                msg = (
                    "HUGR contains TailLoop structure (while/for loop). "
                    "Loops are not currently supported for HUGR → SLR-AST conversion."
                )
                raise UnsupportedHugrStructureError(msg)

    def _extract_all_operations(self, cfg: CFGStructure) -> list[dict]:
        """Extract all quantum operations from all blocks.

        Args:
            cfg: The CFG structure.

        Returns:
            List of all operations across all blocks.
        """
        operations = []
        for block in cfg.blocks.values():
            operations.extend(block.operations)
        operations.extend(cfg.postlude_operations)
        return operations

    def _build_qubit_map(self, operations: list[dict]) -> None:
        """Build mapping from HUGR QAlloc nodes to qubit indices.

        Args:
            operations: List of quantum operations.
        """
        for op in operations:
            if op["op_name"] == "QAlloc":
                self.qubit_allocations[op["node_idx"]] = self.next_qubit_idx
                self.next_qubit_idx += 1

    def _build_straight_line_statements(self, cfg: CFGStructure) -> list[Statement]:
        """Build statements for a straight-line circuit.

        Args:
            cfg: The CFG structure.

        Returns:
            List of SLR-AST Statement nodes.
        """
        if cfg.entry_block is None:
            return []

        block = cfg.blocks[cfg.entry_block]
        sorted_ops = self._topological_sort_operations(block.operations)
        return self._build_statements_from_ops(sorted_ops)

    def _build_cfg_statements(self, cfg: CFGStructure) -> list[Statement]:
        """Build statements for a CFG with control flow.

        Args:
            cfg: The CFG structure.

        Returns:
            List of SLR-AST Statement nodes.
        """
        statements: list[Statement] = []

        # Handle loops
        if cfg.loops:
            statements = self._build_loop_statements(cfg)
            postlude_ops = self._topological_sort_operations(cfg.postlude_operations)
            statements.extend(self._build_statements_from_ops(postlude_ops))
            return statements

        # Handle conditionals
        if not cfg.conditional_blocks:
            # No recognized pattern - fall back to processing blocks sequentially
            # This handles simple cases but may miss some control flow semantics
            for block in cfg.blocks.values():
                sorted_ops = self._topological_sort_operations(block.operations)
                statements.extend(self._build_statements_from_ops(sorted_ops))
            postlude_ops = self._topological_sort_operations(cfg.postlude_operations)
            statements.extend(self._build_statements_from_ops(postlude_ops))
            return statements

        # Process conditional pattern
        for entry_idx, then_idx, else_idx, cont_idx in cfg.conditional_blocks:
            entry_block = cfg.blocks[entry_idx]
            cfg.blocks[then_idx]
            cfg.blocks[else_idx]
            cont_block = cfg.blocks[cont_idx] if cont_idx is not None else None

            # Process entry block operations (before the conditional)
            entry_ops = self._topological_sort_operations(entry_block.operations)
            entry_stmts = self._build_statements_from_ops(entry_ops)
            statements.extend(entry_stmts)

            # After processing entry block, capture output port -> qubit mappings
            self._capture_block_output_qubits(entry_idx)

            # Read the branch predicate off the entry block's Output wire.
            condition_var = self._resolve_branch_condition(entry_idx)

            # Map then block's Input node to source qubits
            self._map_block_input_qubits(entry_idx, then_idx)

            # Process then block (may contain nested conditional)
            then_stmts = self._build_branch_statements(cfg, then_idx)

            # Map else block's Input node to source qubits
            self._map_block_input_qubits(entry_idx, else_idx)

            # Process else block (may contain nested conditional)
            else_stmts = self._build_branch_statements(cfg, else_idx)

            # Create IfStmt (always create it if we detected a conditional pattern)
            if (
                then_stmts
                or else_stmts
                or self._is_conditional_header(cfg, then_idx)
                or self._is_conditional_header(cfg, else_idx)
            ):
                # Use VarExpr for the condition
                condition = VarExpr(name=condition_var)
                if_stmt = IfStmt(
                    condition=condition,
                    then_body=tuple(then_stmts),
                    else_body=tuple(else_stmts) if else_stmts else None,
                )
                statements.append(if_stmt)

            # Capture output qubits from then/else blocks for continuation
            self._capture_block_output_qubits(then_idx)
            self._capture_block_output_qubits(else_idx)
            self._capture_cfg_output_qubits(cfg, (then_idx, else_idx))

            if cont_idx is None or cont_block is None:
                continue

            # Map continuation block's Input to source qubits (from either branch)
            # Use then block's outputs as reference (they should match else block)
            self._map_block_input_qubits(then_idx, cont_idx)

            # Process continuation block
            cont_ops = self._topological_sort_operations(cont_block.operations)
            cont_stmts = self._build_statements_from_ops(cont_ops)
            statements.extend(cont_stmts)

        postlude_ops = self._topological_sort_operations(cfg.postlude_operations)
        statements.extend(self._build_statements_from_ops(postlude_ops))
        return statements

    def _capture_cfg_qubit_ports(self, cfg: CFGStructure) -> None:
        """Bind a simple CFG's qubit outputs to its typed input wires.

        A Guppy loop is represented by one CFG node: its body blocks carry a
        sum/control value, while the enclosing CFG retains the linear qubit
        wire.  Reading the typed CFG ports is the only reliable way to carry
        that wire into a function-level postlude.
        """
        if cfg.cfg_node is None:
            return

        from hugr import Node  # noqa: PLC0415

        cfg_node = Node(cfg.cfg_node)
        input_qubits: list[int] = []
        for in_port, source_ports in self.hugr.incoming_links(cfg_node):
            if not self._is_qubit_port(in_port):
                continue
            if len(source_ports) != 1:
                msg = f"cannot resolve CFG input port {in_port.offset} for HUGR node {cfg.cfg_node}"
                raise UnsupportedHugrStructureError(msg)
            source_port = source_ports[0]
            qubit_idx = self._trace_qubit_source(source_port.node.idx, source_port.offset)
            if qubit_idx is None:
                msg = f"cannot resolve qubit wire feeding CFG node {cfg.cfg_node} port {in_port.offset}"
                raise UnsupportedHugrStructureError(msg)
            input_qubits.append(qubit_idx)

        output_ports = [
            out_port for out_port, _target_ports in self.hugr.outgoing_links(cfg_node) if self._is_qubit_port(out_port)
        ]
        if not input_qubits or not output_ports:
            return
        if len(input_qubits) != len(output_ports):
            msg = (
                f"cannot match {len(input_qubits)} CFG input qubit wire(s) to "
                f"{len(output_ports)} output wire(s) for HUGR node {cfg.cfg_node}"
            )
            raise UnsupportedHugrStructureError(msg)
        for out_port, qubit_idx in zip(output_ports, input_qubits, strict=True):
            self.node_port_to_qubit[(cfg.cfg_node, out_port.offset)] = qubit_idx

    def _is_conditional_header(self, cfg: CFGStructure, block_idx: int) -> bool:
        """Check if a block is a conditional header (has 2 outgoing block edges).

        Args:
            cfg: The CFG structure.
            block_idx: The block index to check.

        Returns:
            True if the block has exactly 2 outgoing edges to other blocks.
        """
        if block_idx not in cfg.blocks:
            return False
        block = cfg.blocks[block_idx]
        block_targets = [t for _, t, _ in block.outgoing_edges if t in cfg.blocks]
        return len(block_targets) == 2

    def _build_branch_statements(
        self,
        cfg: CFGStructure,
        block_idx: int,
    ) -> list[Statement]:
        """Build statements for one non-nested conditional branch.

        Args:
            cfg: The CFG structure.
            block_idx: The starting block of the branch.

        Returns:
            List of statements for this branch.
        """
        statements: list[Statement] = []

        if block_idx not in cfg.blocks:
            return statements

        block = cfg.blocks[block_idx]

        if self._is_conditional_header(cfg, block_idx):
            msg = "HUGR CFG has sequential or nested conditionals; recursive lowering is not supported"
            raise UnsupportedHugrStructureError(msg)

        # First, process any direct operations in this block
        block_ops = self._topological_sort_operations(block.operations)
        statements.extend(self._build_statements_from_ops(block_ops))

        return statements

    def _build_loop_statements(self, cfg: CFGStructure) -> list[Statement]:
        """Build statements for a CFG with loop control flow.

        Args:
            cfg: The CFG structure with detected loops.

        Returns:
            List of SLR-AST Statement nodes including WhileStmt.
        """
        statements: list[Statement] = []

        for loop in cfg.loops:
            # Find the entry block (block that leads to loop header but isn't in loop)
            entry_block_idx = None
            for block_idx, block in cfg.blocks.items():
                if block_idx == loop.header_block:
                    continue
                if block_idx in loop.body_blocks:
                    continue
                # Check if this block leads to header
                for _port, target, _tport in block.outgoing_edges:
                    if target == loop.header_block:
                        entry_block_idx = block_idx
                        break
                if entry_block_idx:
                    break

            # Process entry block (before loop)
            if entry_block_idx is not None:
                entry_block = cfg.blocks[entry_block_idx]
                entry_ops = self._topological_sort_operations(entry_block.operations)
                entry_stmts = self._build_statements_from_ops(entry_ops)
                statements.extend(entry_stmts)

                # Capture output qubits and map to header
                self._capture_block_output_qubits(entry_block_idx)
                self._map_block_input_qubits(entry_block_idx, loop.header_block)

            # Process loop header (no quantum ops typically, just condition)
            header_block = cfg.blocks[loop.header_block]
            header_ops = self._topological_sort_operations(header_block.operations)
            header_stmts = self._build_statements_from_ops(header_ops)
            statements.extend(header_stmts)
            self._capture_cfg_qubit_ports(cfg)

            # Capture header outputs for body
            self._capture_block_output_qubits(loop.header_block)

            # Build loop body statements
            body_stmts: list[Statement] = []
            for body_block_idx in loop.body_blocks:
                # Map input from header (or previous body block)
                self._map_block_input_qubits(loop.header_block, body_block_idx)

                body_block = cfg.blocks[body_block_idx]
                body_ops = self._topological_sort_operations(body_block.operations)
                body_stmts.extend(self._build_statements_from_ops(body_ops))

                # Capture outputs for next block
                self._capture_block_output_qubits(body_block_idx)

            # Create WhileStmt. The predicate is read off the loop header's
            # Output wire; a counter comparison is rejected there rather than
            # replaced by a fabricated measurement variable (issue #493).
            condition_var = self._resolve_branch_condition(loop.header_block)

            while_stmt = WhileStmt(
                condition=VarExpr(name=condition_var),
                body=tuple(body_stmts),
            )
            statements.append(while_stmt)

            # Process exit block (after loop)
            if loop.exit_block in cfg.blocks:
                # Map from header (where loop exits)
                self._map_block_input_qubits(loop.header_block, loop.exit_block)

                exit_block = cfg.blocks[loop.exit_block]
                exit_ops = self._topological_sort_operations(exit_block.operations)
                exit_stmts = self._build_statements_from_ops(exit_ops)
                statements.extend(exit_stmts)
                self._capture_block_output_qubits(loop.exit_block)
                self._capture_cfg_output_qubits(cfg, (loop.exit_block,))

        return statements

    def _resolve_branch_condition(self, block_idx: int) -> str:
        """Resolve a block's branch predicate to the measurement it reads.

        A ``DataflowBlock`` carries its branch selector on input port 0 of the
        block's ``Output`` node.  The flat lowering can only express a
        predicate that is the direct read of one measurement, so that is the
        one shape accepted here.  Negations, comparisons, purely classical
        expressions and loop counters are rejected: emitting a plausible but
        fabricated ``m<n>`` for them silently changes the circuit (issue #493).

        Args:
            block_idx: The block whose outgoing branch is being lowered.

        Returns:
            The result variable name of the measurement the predicate reads.

        Raises:
            UnsupportedHugrStructureError: If the predicate is anything other
                than the direct read of an already-converted measurement.
        """
        from hugr import Node  # noqa: PLC0415

        output_node_idx = self.block_output_nodes.get(block_idx)
        if output_node_idx is None:
            msg = f"HUGR block {block_idx} has no Output node, so its branch predicate cannot be read"
            raise UnsupportedHugrStructureError(msg)

        sources = [
            out_port
            for in_port, out_ports in self.hugr.incoming_links(Node(output_node_idx))
            if in_port.offset == 0
            for out_port in out_ports
        ]
        if len(sources) != 1:
            msg = (
                f"HUGR block {block_idx} has {len(sources)} source(s) on its branch "
                "predicate wire; exactly one is required"
            )
            raise UnsupportedHugrStructureError(msg)

        predicate_node = sources[0].node.idx
        meas_node = self._trace_predicate_measurement(predicate_node)
        if meas_node is None:
            msg = (
                f"HUGR block {block_idx} branches on {self._describe_node(predicate_node)}, "
                "which is not the direct read of a measurement; negations, comparisons, "
                "classical expressions and loop counters cannot be represented by this "
                "converter"
            )
            raise UnsupportedHugrStructureError(msg)

        condition_var = self.measurement_results.get(meas_node)
        if condition_var is None:
            msg = (
                f"HUGR block {block_idx} branches on the result of HUGR node {meas_node}, "
                "which this converter has not emitted as a measurement"
            )
            raise UnsupportedHugrStructureError(msg)
        return condition_var

    def _trace_predicate_measurement(self, node_idx: int) -> int | None:
        """Return the measurement node a predicate wire reads, or ``None``.

        Only a direct ``tket.measurement.Read`` of a measurement counts.  Any
        intervening classical operation makes the predicate something other
        than that measurement's outcome, and is reported by the caller.

        Args:
            node_idx: The node driving the predicate wire.

        Returns:
            The node index of the measurement being read, or ``None``.
        """
        from hugr import Node  # noqa: PLC0415

        if self._custom_op_id(node_idx) not in _MEASUREMENT_READ_OPS:
            return None

        for _in_port, out_ports in self.hugr.incoming_links(Node(node_idx)):
            for out_port in out_ports:
                if out_port.node.idx in self.measurement_results:
                    return out_port.node.idx
        return None

    def _custom_op_id(self, node_idx: int) -> tuple[str, str] | None:
        """Return ``(extension, op_name)`` for an ExtOp node, else ``None``."""
        from hugr import Node  # noqa: PLC0415

        op = self.hugr[Node(node_idx)].op
        if op.__class__.__name__ != "ExtOp":
            return None
        try:
            custom_op = op.to_custom_op()
        except (AttributeError, ValueError):
            return None
        return (custom_op.extension, custom_op.op_name)

    def _describe_node(self, node_idx: int) -> str:
        """Name a HUGR node for an error message, e.g. ``logic.Not``."""
        from hugr import Node  # noqa: PLC0415

        custom = self._custom_op_id(node_idx)
        if custom is not None:
            return f"HUGR node {node_idx} ({custom[0]}.{custom[1]})"
        return f"HUGR node {node_idx} ({self.hugr[Node(node_idx)].op.__class__.__name__})"

    def _resolve_rotation_params(self, op: dict) -> tuple[Expression, ...]:
        """Resolve a rotation gate's angle operands into AST parameters.

        Guppy lowers a rotation angle to a ``ConstRotation`` value in half
        turns, loaded onto the gate's non-qubit input port.  A rotation whose
        angle cannot be resolved is rejected rather than emitted without it
        (issue #493).

        Args:
            op: The operation info dict.

        Returns:
            One :class:`LiteralExpr` per angle operand, in port order.

        Raises:
            UnsupportedHugrStructureError: If an angle operand is not a
                loaded rotation constant.
        """
        from hugr import Node  # noqa: PLC0415

        node_idx = op["node_idx"]
        angle_sources: list[tuple[int, int]] = []
        for in_port, out_ports in self.hugr.incoming_links(Node(node_idx)):
            if in_port.offset < 0 or self._is_qubit_port(in_port):
                continue
            if len(out_ports) != 1:
                msg = (
                    f"cannot resolve the angle on input port {in_port.offset} of "
                    f"{self._describe_node(node_idx)}: {len(out_ports)} source(s)"
                )
                raise UnsupportedHugrStructureError(msg)
            angle_sources.append((in_port.offset, out_ports[0].node.idx))

        if not angle_sources:
            msg = f"rotation {self._describe_node(node_idx)} has no angle operand"
            raise UnsupportedHugrStructureError(msg)

        angle_sources.sort()
        return tuple(self._resolve_rotation_constant(source) for _offset, source in angle_sources)

    def _resolve_rotation_constant(self, node_idx: int) -> Expression:
        """Read a loaded ``ConstRotation`` as a typed AST angle literal."""
        from hugr import Node  # noqa: PLC0415

        from pecos.slr.angle import turns  # noqa: PLC0415  (avoid import cycle)

        const_idx = node_idx
        if self.hugr[Node(node_idx)].op.__class__.__name__ == "LoadConst":
            const_sources = [
                out_port.node.idx
                for _in_port, out_ports in self.hugr.incoming_links(Node(node_idx))
                for out_port in out_ports
            ]
            if len(const_sources) != 1:
                msg = f"cannot resolve the constant behind {self._describe_node(node_idx)}"
                raise UnsupportedHugrStructureError(msg)
            const_idx = const_sources[0]

        const_op = self.hugr[Node(const_idx)].op
        value = getattr(const_op, "val", None)
        half_turns = getattr(value, "val", None)
        if (
            const_op.__class__.__name__ != "Const"
            or getattr(value, "name", None) != "ConstRotation"
            or not isinstance(half_turns, dict)
            or "half_turns" not in half_turns
        ):
            msg = (
                f"rotation angle is not a constant: {self._describe_node(node_idx)} does not load a ConstRotation value"
            )
            raise UnsupportedHugrStructureError(msg)

        # ``ConstRotation`` counts half turns; `turns` counts full turns.
        return LiteralExpr(value=turns(float(half_turns["half_turns"]) / 2.0))

    def _capture_block_output_qubits(self, block_idx: int) -> None:
        """Capture which qubits are on which output ports of a block.

        After processing a block's operations, this traces which qubit
        ends up on each output port of the block's Output node.

        Args:
            block_idx: The block index.
        """
        from hugr import Node  # noqa: PLC0415

        output_node_idx = self.block_output_nodes.get(block_idx)
        if output_node_idx is None:
            return

        qubit_outputs: list[int] = []
        output_node = Node(output_node_idx)

        # Trace each input to the Output node to find the source qubit
        for in_port, out_ports in self.hugr.incoming_links(output_node):
            if in_port.offset >= 0:
                for out_port in out_ports:
                    if not self._is_qubit_port(out_port):
                        continue
                    qubit_idx = self._trace_qubit_source(out_port.node.idx, out_port.offset)
                    if qubit_idx is not None:
                        qubit_outputs.append(qubit_idx)

        # A DataflowBlock's outer quantum outputs omit its inner classical
        # outputs.  Re-index the filtered inner wires to those outer ports.
        self.block_output_qubit_ports[block_idx] = dict(enumerate(qubit_outputs))

    def _capture_cfg_output_qubits(self, cfg: CFGStructure, source_blocks: tuple[int, ...]) -> None:
        """Bind CFG output ports to qubits preserved by every completed branch."""
        if cfg.cfg_node is None:
            return
        per_port: dict[int, dict[int, int]] = {}
        for block_idx in source_blocks:
            for port, qubit_idx in self.block_output_qubit_ports.get(block_idx, {}).items():
                per_port.setdefault(port, {})[block_idx] = qubit_idx
        for port, branch_qubits in per_port.items():
            # A continuation port is meaningful only if each completed branch
            # supplies it and all branches preserve the same physical qubit.
            if set(branch_qubits) != set(source_blocks):
                continue
            qubit_indices = set(branch_qubits.values())
            if len(qubit_indices) == 1:
                qubit_idx = next(iter(qubit_indices))
                self.node_port_to_qubit[(cfg.cfg_node, port)] = qubit_idx
                # Guppy 1 return-value operations can be attached directly to
                # the enclosing FuncDefn and consume its output wire, rather
                # than the nested CFG node's output wire.
                from hugr import Node  # noqa: PLC0415

                cfg_parent = self.hugr[Node(cfg.cfg_node)].parent
                if cfg_parent is not None:
                    self.node_port_to_qubit[(cfg_parent.idx, port)] = qubit_idx

    def _map_block_input_qubits(
        self,
        source_block_idx: int,
        target_block_idx: int,
    ) -> None:
        """Map a block's Input node outputs to qubits from a source block.

        This maps the target block's Input node outputs to the qubits
        that were on the source block's Output ports, using the CFG edge
        to determine the correct port mapping.

        Args:
            source_block_idx: The block that provides the qubits.
            target_block_idx: The block whose Input node needs mapping.
        """
        input_node_idx = self.block_input_nodes.get(target_block_idx)
        if input_node_idx is None:
            return

        source_ports = self.block_output_qubit_ports.get(source_block_idx, {})
        if not source_ports:
            # Guppy 1 can carry a function-level qubit into a CFG without an
            # explicit block-output link. With one allocated qubit this mapping
            # is unambiguous.
            if len(self.qubit_allocations) == 1:
                self.node_to_qubit[input_node_idx] = next(iter(self.qubit_allocations.values()))
            return

        # Preserve wire identity by the data-port number.  CFG edge ports
        # select a branch, not a qubit target, so using one of them (or the
        # largest output port) as a fallback guesses the target after joins.
        for port, qubit_idx in source_ports.items():
            self.node_port_to_qubit[(input_node_idx, port)] = qubit_idx
        if len(set(source_ports.values())) == 1:
            self.node_to_qubit[input_node_idx] = next(iter(source_ports.values()))

    def _build_statements_from_ops(self, operations: list[dict]) -> list[Statement]:
        """Build SLR-AST statements from a list of operations.

        Args:
            operations: Sorted list of quantum operations.

        Returns:
            List of SLR-AST Statement nodes.
        """
        statements: list[Statement] = []

        for op in operations:
            op_name = op["op_name"]
            node_idx = op["node_idx"]

            if op_name == "QAlloc":
                qubit_idx = self.qubit_allocations[node_idx]
                self.node_to_qubit[node_idx] = qubit_idx
                self._bind_operation_output_qubits(op, [qubit_idx])
                # Add Prepare operation
                statements.append(
                    PrepareOp(allocator=self.allocator_name, slots=(qubit_idx,)),
                )

            elif op_name in GATE_OPERATIONS:
                gate_kind = GATE_KIND_MAP[op_name]
                qubit_indices = self._resolve_qubit_operands(op)
                self._require_qubit_arity(op, qubit_indices, 2 if op_name in _TWO_QUBIT_GATES else 1)
                slot_refs = tuple(SlotRef(allocator=self.allocator_name, index=idx) for idx in qubit_indices)
                params = self._resolve_rotation_params(op) if op_name in _ROTATION_GATES else ()
                statements.append(GateOp(gate=gate_kind, targets=slot_refs, params=params))
                self._bind_operation_output_qubits(op, qubit_indices)

            elif op_name in MEASURE_OPERATIONS:
                qubit_indices = self._resolve_qubit_operands(op)
                self._require_qubit_arity(op, qubit_indices, 1)
                slot_refs = tuple(SlotRef(allocator=self.allocator_name, index=idx) for idx in qubit_indices)

                # Create result variable
                result_var = f"m{self.next_result_idx}"
                self.measurement_results[node_idx] = result_var
                self.next_result_idx += 1

                # Create MeasureOp with result
                result_refs = tuple(BitRef(register=result_var, index=0) for _ in qubit_indices)
                statements.append(MeasureOp(targets=slot_refs, results=result_refs))

        return statements

    def _topological_sort_operations(self, operations: list[dict]) -> list[dict]:
        """Sort operations in topological order based on qubit data flow.

        Args:
            operations: List of quantum operations.

        Returns:
            Operations sorted in execution order.
        """
        if not operations:
            return []

        # Build dependency graph
        node_to_op = {op["node_idx"]: op for op in operations}
        op_indices = {op["node_idx"]: i for i, op in enumerate(operations)}

        # Find dependencies
        dependencies: dict[int, set[int]] = {op["node_idx"]: set() for op in operations}

        for op in operations:
            for src_node_idx, _src_port, _dest_port in op["incoming"]:
                if src_node_idx in node_to_op:
                    dependencies[op["node_idx"]].add(src_node_idx)
                else:
                    source = self._trace_to_quantum_op(src_node_idx, node_to_op)
                    if source is not None:
                        dependencies[op["node_idx"]].add(source)

        # Kahn's algorithm
        in_degree = {node: len(deps) for node, deps in dependencies.items()}
        queue = [node for node, deg in in_degree.items() if deg == 0]
        sorted_nodes = []

        while queue:
            queue.sort(key=lambda n: op_indices.get(n, 0))
            node = queue.pop(0)
            sorted_nodes.append(node)

            for other_node, deps in dependencies.items():
                if node in deps:
                    in_degree[other_node] -= 1
                    if in_degree[other_node] == 0:
                        queue.append(other_node)

        return [node_to_op[n] for n in sorted_nodes]

    def _resolve_qubit_operands(self, op: dict) -> list[int]:
        """Resolve which qubits an operation acts on.

        Args:
            op: The operation info dict.

        Returns:
            List of qubit indices this operation acts on.
        """
        qubit_indices = []

        for src_node_idx, src_port, dest_port in op["incoming"]:
            if (src_node_idx, src_port) in self.node_port_to_qubit:
                qubit_indices.append((dest_port, self.node_port_to_qubit[(src_node_idx, src_port)]))
            elif src_node_idx in self.node_to_qubit:
                qubit_indices.append((dest_port, self.node_to_qubit[src_node_idx]))
            else:
                qubit_idx = self._trace_qubit_source(src_node_idx, src_port)
                if qubit_idx is not None:
                    qubit_indices.append((dest_port, qubit_idx))

        qubit_indices.sort(key=lambda x: x[0])
        return [idx for _port, idx in qubit_indices]

    def _require_qubit_arity(self, op: dict, qubit_indices: list[int], expected: int) -> None:
        """Raise instead of exporting an operation with guessed or missing targets."""
        if len(qubit_indices) != expected:
            msg = (
                f"cannot resolve {expected} qubit operand(s) for HUGR node {op['node_idx']} "
                f"operation {op['op_name']!r}; found {len(qubit_indices)}"
            )
            raise UnsupportedHugrStructureError(msg)

    def _is_qubit_port(self, port: object) -> bool:
        """Return whether a HUGR wire carries a qubit rather than a result."""
        try:
            return str(self.hugr.port_type(port)) == "Qubit"
        except ValueError:
            # Static/order ports have no value type and cannot be qubit wires.
            return False

    def _bind_operation_output_qubits(self, op: dict, qubit_indices: list[int]) -> None:
        """Record wire identity for every quantum output port of an operation."""
        output_ports = sorted({source_port for source_port, _target, _target_port in op["outgoing"]})
        if not output_ports:
            return
        if len(output_ports) == len(qubit_indices):
            bindings = zip(output_ports, qubit_indices, strict=True)
        elif len(qubit_indices) == 1:
            bindings = ((port, qubit_indices[0]) for port in output_ports)
        else:
            msg = (
                f"cannot match HUGR node {op['node_idx']} operation {op['op_name']!r} "
                f"outputs {output_ports} to {len(qubit_indices)} qubit operand(s)"
            )
            raise UnsupportedHugrStructureError(msg)
        for port, qubit_idx in bindings:
            self.node_port_to_qubit[(op["node_idx"], port)] = qubit_idx
        if len(set(qubit_indices)) == 1:
            self.node_to_qubit[op["node_idx"]] = qubit_indices[0]

    def _trace_qubit_source(self, node_idx: int, port_idx: int | None = None) -> int | None:
        """Trace backwards to find which qubit a wire represents.

        Args:
            node_idx: Starting node index.
            port_idx: Optional output port on the starting node.

        Returns:
            The qubit index, or None if not found.
        """
        from hugr import Node  # noqa: PLC0415

        visited: set[tuple[int, int | None]] = set()
        stack = [(node_idx, port_idx)]

        while stack:
            current, current_port = stack.pop()
            key = (current, current_port)
            if key in visited:
                continue
            visited.add(key)

            if current_port is not None and (current, current_port) in self.node_port_to_qubit:
                return self.node_port_to_qubit[(current, current_port)]

            if current in self.node_to_qubit and current_port is None:
                return self.node_to_qubit[current]

            try:
                node = Node(current)
                for in_port, out_ports in self.hugr.incoming_links(node):
                    if current_port is not None and in_port.offset != current_port:
                        continue
                    stack.extend((out_port.node.idx, out_port.offset) for out_port in out_ports)
            except (KeyError, IndexError):
                continue

        return None

    def _trace_to_quantum_op(
        self,
        node_idx: int,
        node_to_op: dict[int, dict],
    ) -> int | None:
        """Trace backwards to find source quantum operation.

        Args:
            node_idx: Starting node index.
            node_to_op: Mapping from node index to operation.

        Returns:
            The node index of the source quantum op, or None.
        """
        from hugr import Node  # noqa: PLC0415

        visited: set[int] = set()
        stack = [node_idx]

        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)

            if current in node_to_op:
                return current

            try:
                node = Node(current)
                for _in_port, out_ports in self.hugr.incoming_links(node):
                    stack.extend(out_port.node.idx for out_port in out_ports)
            except (KeyError, IndexError):
                continue

        return None

    def _get_incoming_connections(self, node: Node) -> list[tuple[int, int, int]]:
        """Get incoming connections for a node.

        Returns:
            List of (source_node_idx, source_port, dest_port) tuples.
        """
        return [
            (out_port.node.idx, out_port.offset, in_port.offset)
            for in_port, out_ports in self.hugr.incoming_links(node)
            for out_port in out_ports
            if in_port.offset >= 0 and out_port.offset >= 0
        ]

    def _get_outgoing_connections(self, node: Node) -> list[tuple[int, int, int]]:
        """Get outgoing connections for a node.

        Returns:
            List of (source_port, dest_node_idx, dest_port) tuples.
        """
        return [
            (out_port.offset, in_port.node.idx, in_port.offset)
            for out_port, in_ports in self.hugr.outgoing_links(node)
            for in_port in in_ports
            if out_port.offset >= 0 and in_port.offset >= 0
        ]

    def _extract_function_name(self) -> str:
        """Extract the function name from the HUGR."""
        for _node, data in self.hugr.nodes():
            if data.op.__class__.__name__ == "FuncDefn":
                if hasattr(data.op, "f_name"):
                    return data.op.f_name
                if hasattr(data.op, "name"):
                    name = data.op.name
                    if callable(name):
                        return name()
                    return name
        return "guppy_circuit"


def hugr_to_ast(
    hugr: Hugr,
    *,
    allocator_name: str = "q",
) -> Program:
    """Convert a HUGR quantum circuit to an SLR-AST Program.

    Supports straight-line circuits and simple conditionals.

    Args:
        hugr: A HUGR containing a compiled quantum circuit.
        allocator_name: Name for the qubit allocator (default: "q").

    Returns:
        An SLR-AST Program representing the quantum circuit.

    Raises:
        UnsupportedHugrStructureError: If the HUGR contains unsupported structures
            like loops.

    Example:
        >>> from guppylang import guppy
        >>> from guppylang.std.quantum import h, qubit, measure
        >>> @guppy
        ... def simple() -> bool:
        ...     q = qubit()
        ...     h(q)
        ...     return measure(q).read()
        ...
        >>> package = simple.compile()
        >>> ast = hugr_to_ast(package.modules[0])
        >>> len(ast.body)  # PZ + H + Measure
        3
    """
    converter = HugrToAstConverter(hugr)
    converter.allocator_name = allocator_name
    return converter.convert()


def guppy_to_ast(
    guppy_func: GuppyFunction,
    *,
    allocator_name: str = "q",
) -> Program:
    """Convert a Guppy-decorated function to an SLR-AST Program.

    Supports straight-line circuits and simple conditionals.

    Args:
        guppy_func: A function decorated with @guppy.
        allocator_name: Name for the qubit allocator (default: "q").

    Returns:
        An SLR-AST Program representing the quantum circuit.

    Raises:
        UnsupportedHugrStructureError: If the HUGR contains unsupported structures
            like loops.

    Example::

        from guppylang import guppy
        from guppylang.std.quantum import h, x, qubit, measure
        from pecos.circuit_converters import guppy_to_ast

        @guppy
        def conditional() -> bool:
            q = qubit()
            h(q)
            result = measure(q).read()
            q2 = qubit()
            if result:
                x(q2)
            return measure(q2).read()

        ast = guppy_to_ast(conditional)
        # ast now contains an IfStmt for the conditional
    """
    package = guppy_func.compile()
    hugr = package.modules[0]
    return hugr_to_ast(hugr, allocator_name=allocator_name)
