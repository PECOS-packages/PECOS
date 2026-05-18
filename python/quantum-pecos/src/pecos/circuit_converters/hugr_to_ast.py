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
- Conditionals (if/else based on measurement results)
- Nested conditionals (if/else within if/else branches)
- While loops with classical conditions
- Two-qubit gates (CX, CZ, etc.)

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
    ...     return measure(q0), measure(q1)
    ...
    >>>
    >>> ast = guppy_to_ast(bell)
    >>> # Use ast with SLR-AST analysis, optimization, or code generation

    Conditional circuit with measurement feedback:

    >>> @guppy
    ... def conditional() -> bool:
    ...     q = qubit()
    ...     h(q)
    ...     result = measure(q)
    ...     q2 = qubit()
    ...     if result:
    ...         x(q2)
    ...     return measure(q2)
    ...
    >>>
    >>> ast = guppy_to_ast(conditional)
    >>> # AST contains IfStmt node for the conditional

    Loop circuit:

    >>> @guppy
    ... def loop_circuit() -> bool:
    ...     q = qubit()
    ...     h(q)
    ...     count = 0
    ...     while count < 3:
    ...         x(q)
    ...         count = count + 1
    ...     return measure(q)
    ...
    >>>
    >>> ast = guppy_to_ast(loop_circuit)
    >>> # AST contains WhileStmt node for the loop
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
    - Conditionals (if/else based on measurements)
    - Nested conditionals
    - While loops with classical conditions

    Unsupported structures include:
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
    exit_block: int  # Block to go to when loop exits
    back_edge_source: int  # Block that has the back-edge to header


@dataclass
class CFGStructure:
    """Analyzed CFG structure."""

    blocks: dict[int, BlockInfo]  # block_idx -> BlockInfo
    entry_block: int | None = None
    exit_block: int | None = None
    is_straight_line: bool = True
    conditional_blocks: list[tuple[int, int, int, int]] = field(
        default_factory=list,
    )  # (entry, then, else, continuation)
    loops: list[LoopInfo] = field(default_factory=list)  # Detected loops


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

        # Find all DataflowBlocks and ExitBlock
        for node, data in self.hugr.nodes():
            op_name = data.op.__class__.__name__
            parent_idx = data.parent.idx if data.parent else None

            if op_name == "DataflowBlock":
                block = BlockInfo(node_idx=node.idx, parent_idx=parent_idx or -1)
                cfg.blocks[node.idx] = block

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

            if parent_idx in cfg.blocks:
                custom_op = data.op.to_custom_op()
                ext_name = custom_op.extension
                ext_op_name = custom_op.op_name

                if ext_name in QUANTUM_EXTENSIONS and ext_op_name in ALL_QUANTUM_OPERATIONS:
                    incoming = self._get_incoming_connections(node)
                    outgoing = self._get_outgoing_connections(node)

                    cfg.blocks[parent_idx].operations.append(
                        {
                            "node_idx": node.idx,
                            "op_name": ext_op_name,
                            "parent_idx": parent_idx,
                            "incoming": incoming,
                            "outgoing": outgoing,
                        },
                    )

        # Determine if straight-line or has control flow
        if len(cfg.blocks) == 1:
            cfg.is_straight_line = True
        elif len(cfg.blocks) > 1:
            cfg.is_straight_line = False
            # First detect loops (back-edges)
            self._identify_loops(cfg)
            # Then try to identify conditional patterns (if no loops found)
            if not cfg.loops:
                self._identify_conditional_pattern(cfg)

        return cfg

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
            return self._build_loop_statements(cfg)

        # Handle conditionals
        if not cfg.conditional_blocks:
            # No recognized pattern - fall back to processing blocks sequentially
            # This handles simple cases but may miss some control flow semantics
            for block in cfg.blocks.values():
                sorted_ops = self._topological_sort_operations(block.operations)
                statements.extend(self._build_statements_from_ops(sorted_ops))
            return statements

        # Process conditional pattern
        for entry_idx, then_idx, else_idx, cont_idx in cfg.conditional_blocks:
            entry_block = cfg.blocks[entry_idx]
            cfg.blocks[then_idx]
            cfg.blocks[else_idx]
            cont_block = cfg.blocks[cont_idx]

            # Process entry block operations (before the conditional)
            entry_ops = self._topological_sort_operations(entry_block.operations)
            entry_stmts = self._build_statements_from_ops(entry_ops)
            statements.extend(entry_stmts)

            # After processing entry block, capture output port -> qubit mappings
            self._capture_block_output_qubits(entry_idx)

            # Get the condition (last measurement result)
            condition_var = self._get_condition_variable()

            # Map then block's Input node to source qubits
            self._map_block_input_qubits(entry_idx, then_idx, cfg)

            # Process then block (may contain nested conditional)
            then_stmts = self._build_branch_statements(cfg, then_idx, cont_idx)

            # Map else block's Input node to source qubits
            self._map_block_input_qubits(entry_idx, else_idx, cfg)

            # Process else block (may contain nested conditional)
            else_stmts = self._build_branch_statements(cfg, else_idx, cont_idx)

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

            # Map continuation block's Input to source qubits (from either branch)
            # Use then block's outputs as reference (they should match else block)
            self._map_block_input_qubits(then_idx, cont_idx, cfg)

            # Process continuation block
            cont_ops = self._topological_sort_operations(cont_block.operations)
            cont_stmts = self._build_statements_from_ops(cont_ops)
            statements.extend(cont_stmts)

        return statements

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
        stop_at: int,
    ) -> list[Statement]:
        """Build statements for a branch, handling nested conditionals.

        Args:
            cfg: The CFG structure.
            block_idx: The starting block of the branch.
            stop_at: The block to stop at (continuation block).

        Returns:
            List of statements for this branch.
        """
        statements: list[Statement] = []

        if block_idx not in cfg.blocks:
            return statements

        block = cfg.blocks[block_idx]

        # First, process any direct operations in this block
        block_ops = self._topological_sort_operations(block.operations)
        statements.extend(self._build_statements_from_ops(block_ops))

        # Check if this block is a conditional header (nested conditional)
        if self._is_conditional_header(cfg, block_idx):
            # Get the branches
            block_edges = [(port, target) for port, target, _tport in block.outgoing_edges if target in cfg.blocks]
            block_edges.sort(key=lambda x: x[0])
            nested_else = block_edges[0][1]
            nested_then = block_edges[1][1]

            # Find nested continuation (where both branches meet)
            nested_then_targets = self._find_eventual_targets(cfg, nested_then)
            nested_else_targets = self._find_eventual_targets(cfg, nested_else)
            nested_cont_candidates = nested_then_targets & nested_else_targets

            if nested_cont_candidates:
                nested_cont = min(nested_cont_candidates)

                # Capture outputs and map inputs for nested branches
                self._capture_block_output_qubits(block_idx)

                # Get condition for nested conditional
                condition_var = self._get_condition_variable()

                # Process nested then branch
                self._map_block_input_qubits(block_idx, nested_then, cfg)
                nested_then_stmts = self._build_branch_statements(
                    cfg,
                    nested_then,
                    nested_cont,
                )

                # Process nested else branch
                self._map_block_input_qubits(block_idx, nested_else, cfg)
                nested_else_stmts = self._build_branch_statements(
                    cfg,
                    nested_else,
                    nested_cont,
                )

                # Create nested IfStmt
                if nested_then_stmts or nested_else_stmts:
                    nested_condition = VarExpr(name=condition_var)
                    nested_if = IfStmt(
                        condition=nested_condition,
                        then_body=tuple(nested_then_stmts),
                        else_body=(tuple(nested_else_stmts) if nested_else_stmts else None),
                    )
                    statements.append(nested_if)

                # Process nested continuation (up to stop_at)
                if nested_cont != stop_at and nested_cont in cfg.blocks:
                    self._capture_block_output_qubits(nested_then)
                    self._map_block_input_qubits(nested_then, nested_cont, cfg)
                    cont_stmts = self._build_branch_statements(
                        cfg,
                        nested_cont,
                        stop_at,
                    )
                    statements.extend(cont_stmts)

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
                self._map_block_input_qubits(entry_block_idx, loop.header_block, cfg)

            # Process loop header (no quantum ops typically, just condition)
            header_block = cfg.blocks[loop.header_block]
            header_ops = self._topological_sort_operations(header_block.operations)
            header_stmts = self._build_statements_from_ops(header_ops)
            statements.extend(header_stmts)

            # Capture header outputs for body
            self._capture_block_output_qubits(loop.header_block)

            # Build loop body statements
            body_stmts: list[Statement] = []
            for body_block_idx in loop.body_blocks:
                # Map input from header (or previous body block)
                self._map_block_input_qubits(loop.header_block, body_block_idx, cfg)

                body_block = cfg.blocks[body_block_idx]
                body_ops = self._topological_sort_operations(body_block.operations)
                body_stmts.extend(self._build_statements_from_ops(body_ops))

                # Capture outputs for next block
                self._capture_block_output_qubits(body_block_idx)

            # Create WhileStmt
            # For quantum loops, the condition is typically based on a measurement result
            # or a classical counter. Use a placeholder variable for now.
            condition_var = self._get_condition_variable()
            # Use measurement variable if available, else generic condition
            condition = VarExpr(name=condition_var) if condition_var.startswith("m") else VarExpr(name="loop_condition")

            while_stmt = WhileStmt(
                condition=condition,
                body=tuple(body_stmts),
            )
            statements.append(while_stmt)

            # Process exit block (after loop)
            if loop.exit_block in cfg.blocks:
                # Map from header (where loop exits)
                self._map_block_input_qubits(loop.header_block, loop.exit_block, cfg)

                exit_block = cfg.blocks[loop.exit_block]
                exit_ops = self._topological_sort_operations(exit_block.operations)
                exit_stmts = self._build_statements_from_ops(exit_ops)
                statements.extend(exit_stmts)

        return statements

    def _get_condition_variable(self) -> str:
        """Get the variable name for the last measurement result.

        Returns:
            Variable name like "m0", "m1", etc.
        """
        # Use the last assigned measurement result
        if self.measurement_results:
            return list(self.measurement_results.values())[-1]
        return f"m{self.next_result_idx}"

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

        port_to_qubit: dict[int, int] = {}
        output_node = Node(output_node_idx)

        # Trace each input to the Output node to find the source qubit
        for in_port, out_ports in self.hugr.incoming_links(output_node):
            if in_port.offset >= 0:
                for out_port in out_ports:
                    qubit_idx = self._trace_qubit_source(out_port.node.idx)
                    if qubit_idx is not None:
                        port_to_qubit[in_port.offset] = qubit_idx

        self.block_output_qubit_ports[block_idx] = port_to_qubit

    def _map_block_input_qubits(
        self,
        source_block_idx: int,
        target_block_idx: int,
        cfg: CFGStructure | None = None,
    ) -> None:
        """Map a block's Input node outputs to qubits from a source block.

        This maps the target block's Input node outputs to the qubits
        that were on the source block's Output ports, using the CFG edge
        to determine the correct port mapping.

        Args:
            source_block_idx: The block that provides the qubits.
            target_block_idx: The block whose Input node needs mapping.
            cfg: Optional CFG structure for edge lookup.
        """
        input_node_idx = self.block_input_nodes.get(target_block_idx)
        if input_node_idx is None:
            return

        source_ports = self.block_output_qubit_ports.get(source_block_idx, {})
        if not source_ports:
            return

        # Find which output port of source block connects to target block
        # by looking at the CFG edges
        source_cfg_port = None
        if cfg and source_block_idx in cfg.blocks:
            for port, target, _tport in cfg.blocks[source_block_idx].outgoing_edges:
                if target == target_block_idx:
                    source_cfg_port = port
                    break

        # The CFG edge port corresponds to the Output node port that carries the qubit
        # In Guppy's conditional pattern, each branch gets the same qubit on different CFG ports
        # The qubit is on the Output port matching the CFG port
        if source_cfg_port is not None and source_cfg_port in source_ports:
            qubit_idx = source_ports[source_cfg_port]
            self.node_to_qubit[input_node_idx] = qubit_idx
        elif source_ports:
            # Fallback: use the highest-numbered port (typically the non-control data)
            max_port = max(source_ports.keys())
            self.node_to_qubit[input_node_idx] = source_ports[max_port]

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
                # Add Prepare operation
                statements.append(
                    PrepareOp(allocator=self.allocator_name, slots=(qubit_idx,)),
                )

            elif op_name in GATE_OPERATIONS:
                gate_kind = GATE_KIND_MAP[op_name]
                qubit_indices = self._resolve_qubit_operands(op)

                if qubit_indices:
                    slot_refs = tuple(SlotRef(allocator=self.allocator_name, index=idx) for idx in qubit_indices)
                    statements.append(GateOp(gate=gate_kind, targets=slot_refs))

                    # Update node_to_qubit for outputs
                    for qubit_idx in qubit_indices:
                        self.node_to_qubit[node_idx] = qubit_idx

            elif op_name in MEASURE_OPERATIONS:
                qubit_indices = self._resolve_qubit_operands(op)
                if qubit_indices:
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

        for src_node_idx, _src_port, dest_port in op["incoming"]:
            if src_node_idx in self.node_to_qubit:
                qubit_indices.append((dest_port, self.node_to_qubit[src_node_idx]))
            else:
                qubit_idx = self._trace_qubit_source(src_node_idx)
                if qubit_idx is not None:
                    qubit_indices.append((dest_port, qubit_idx))

        qubit_indices.sort(key=lambda x: x[0])
        return [idx for _port, idx in qubit_indices]

    def _trace_qubit_source(self, node_idx: int) -> int | None:
        """Trace backwards to find which qubit a wire represents.

        Args:
            node_idx: Starting node index.

        Returns:
            The qubit index, or None if not found.
        """
        from hugr import Node  # noqa: PLC0415

        visited: set[int] = set()
        stack = [node_idx]

        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)

            if current in self.node_to_qubit:
                return self.node_to_qubit[current]

            try:
                node = Node(current)
                for _in_port, out_ports in self.hugr.incoming_links(node):
                    stack.extend(out_port.node.idx for out_port in out_ports)
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
        ...     return measure(q)
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
            result = measure(q)
            q2 = qubit()
            if result:
                x(q2)
            return measure(q2)

        ast = guppy_to_ast(conditional)
        # ast now contains an IfStmt for the conditional
    """
    package = guppy_func.compile()
    hugr = package.modules[0]
    return hugr_to_ast(hugr, allocator_name=allocator_name)
