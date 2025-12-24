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

"""Optimizer for Parallel blocks that reorders operations for maximum parallelism."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pecos.slr.fund import Node

from pecos.slr.block import Block
from pecos.slr.cond_block import If, Repeat
from pecos.slr.main import Main
from pecos.slr.misc import Parallel


class ParallelOptimizer:
    """Optimizes Parallel blocks by reordering operations for maximum parallelism.

    This transformation pass analyzes operations within Parallel blocks and reorders
    them to maximize parallelism while respecting quantum gate dependencies. Gates
    that act on disjoint sets of qubits can be grouped together for parallel execution.

    The optimizer is conservative and will not optimize Parallel blocks containing
    control flow (If/Repeat blocks) to ensure correctness.
    """

    def transform(self, block: Block) -> Block:
        """Transform a block by optimizing any Parallel blocks within it.

        Args:
            block: The block to transform

        Returns:
            A new block with optimized Parallel blocks
        """
        return self._transform_block(block)

    def _transform_block(self, block: Block) -> Block:
        """Transform a block bottom-up, processing inner blocks first."""
        new_ops = []

        for op in block.ops:
            if isinstance(op, Parallel):
                # Transform the parallel block
                optimized = self._transform_parallel(op)
                new_ops.append(optimized)

            elif isinstance(op, Block):
                # Recursively transform nested blocks
                new_ops.append(self._transform_block(op))
            else:
                new_ops.append(op)

        # Handle special block types differently
        if isinstance(block, If):
            # Create new If block preserving condition
            new_block = If(block.cond)
            if new_ops:
                new_block.Then(*new_ops)
            # Transform else block if it exists
            if block.else_block:
                new_block.else_block = self._transform_block(block.else_block)
        elif isinstance(block, Repeat):
            # Create new Repeat block preserving condition
            new_block = Repeat(block.cond)
            if new_ops:
                new_block.block(*new_ops)
        # Only reconstruct certain block types
        elif isinstance(block, Parallel) and type(block) is Parallel:
            new_block = Parallel(*new_ops)
        elif isinstance(block, Main) and type(block) is Main:
            new_block = Main(*new_ops)
        elif isinstance(block, Block):
            # Use isinstance to handle Block subclasses
            new_block = Block(*new_ops)
            # Preserve block metadata if available
            if hasattr(block, "block_name"):
                new_block.block_name = block.block_name
            if hasattr(block, "block_module"):
                new_block.block_module = block.block_module
            if hasattr(block, "__slr_return_type__"):
                new_block.__slr_return_type__ = block.__slr_return_type__
        else:
            # For non-Block types, don't transform them
            # They may have specific initialization requirements
            return block

        # Copy over any additional attributes
        if hasattr(block, "vars"):
            new_block.vars = block.vars

        return new_block

    def _transform_parallel(self, parallel: Parallel) -> Block | Parallel:
        """Transform a Parallel block into an optimized structure.

        If the block contains control flow, returns it unchanged.
        Otherwise, reorders operations for maximum parallelism.
        """
        # Check if optimization is safe
        if not self._can_optimize_parallel(parallel):
            return parallel

        # Collect all operations from the Parallel block
        operations = self._collect_operations(parallel)

        if not operations:
            return parallel

        # Build dependency graph
        dep_graph = self._build_dependency_graph(operations)

        # Group operations by gate type and dependencies
        grouped_ops = self._group_operations(operations, dep_graph)

        # If only one group, no optimization needed
        if len(grouped_ops) <= 1:
            return parallel

        # Create new structure with nested Parallel blocks
        new_ops = []
        for group in grouped_ops:
            if len(group) == 1:
                new_ops.append(group[0])
            else:
                new_ops.append(Parallel(*group))

        # Return a Block containing the optimized structure
        return Block(*new_ops)

    def _can_optimize_parallel(self, parallel: Parallel) -> bool:
        """Check if a Parallel block can be safely optimized.

        Returns False if the block contains control flow (If/Repeat).
        """
        for op in parallel.ops:
            if isinstance(op, If | Repeat):
                return False
            if isinstance(op, Block) and self._contains_control_flow(op):
                return False
        return True

    def _contains_control_flow(self, block: Block) -> bool:
        """Check if a block contains any control flow operations."""
        for op in block.ops:
            if isinstance(op, If | Repeat):
                return True
            if isinstance(op, Block) and self._contains_control_flow(op):
                return True
        return False

    def _collect_operations(self, parallel: Parallel) -> list[Node]:
        """Recursively collect all operations from a Parallel block."""
        operations = []

        for op in parallel.ops:
            if isinstance(op, Block):
                # Recursively collect from nested blocks
                operations.extend(self._collect_from_block(op))
            else:
                operations.append(op)

        return operations

    def _collect_from_block(self, block: Block) -> list[Node]:
        """Recursively collect all operations from a Block."""
        operations = []

        for op in block.ops:
            if isinstance(op, Block):
                operations.extend(self._collect_from_block(op))
            else:
                operations.append(op)

        return operations

    def _build_dependency_graph(self, operations: list[Node]) -> dict[int, set[int]]:
        """Build a dependency graph based on qubit usage.

        Returns a dict mapping operation index to set of indices it depends on.
        """
        dep_graph = defaultdict(set)

        # Track which qubits are used by each operation
        qubit_usage = {}
        for i, op in enumerate(operations):
            qubits = self._get_qubits(op)
            qubit_usage[i] = qubits

        # Build dependencies: op_i depends on op_j if they share qubits and j < i
        for i in range(len(operations)):
            for j in range(i):
                if qubit_usage[i] & qubit_usage[j]:  # Intersection of qubit sets
                    dep_graph[i].add(j)

        return dict(dep_graph)

    def _get_qubits(self, op: Node) -> set:
        """Extract the set of qubits an operation acts on."""
        qubits = set()

        # Handle quantum gates with qargs
        if hasattr(op, "qargs"):
            for qarg in op.qargs:
                qubits.add(self._qubit_to_key(qarg))

        # Handle measurements
        elif hasattr(op, "qin"):
            for q in op.qin:
                qubits.add(self._qubit_to_key(q))

        return qubits

    def _qubit_to_key(self, qubit) -> tuple:
        """Convert a qubit to a hashable key."""
        if hasattr(qubit, "reg") and hasattr(qubit, "index"):
            return (qubit.reg.sym, qubit.index)
        if hasattr(qubit, "sym"):
            return (qubit.sym,)
        return (str(qubit),)

    def _group_operations(
        self,
        operations: list[Node],
        dep_graph: dict[int, set[int]],
    ) -> list[list[Node]]:
        """Group operations by gate type and dependencies.

        Returns a list of groups, where each group contains operations that can
        execute in parallel.
        """
        # Perform topological sort to find dependency levels
        levels = self._topological_levels(len(operations), dep_graph)

        # Within each level, group by operation type
        grouped = []
        for level in levels:
            # Group operations at this level by type
            type_groups = defaultdict(list)
            for idx in level:
                op = operations[idx]
                op_type = self._get_op_type(op)
                type_groups[op_type].append(op)

            # Add groups in a consistent order (H, then other single-qubit, then two-qubit, then measurements)
            order = [
                "H",
                "X",
                "Y",
                "Z",
                "S",
                "T",
                "RX",
                "RY",
                "RZ",
                "CX",
                "CY",
                "CZ",
                "Measure",
                "Other",
            ]
            # Use list comprehension to build groups from ordered types
            grouped.extend(
                type_groups[op_type] for op_type in order if op_type in type_groups
            )

            # Add any remaining types
            grouped.extend(
                ops for op_type, ops in type_groups.items() if op_type not in order
            )

        return grouped

    def _topological_levels(
        self,
        n: int,
        dep_graph: dict[int, set[int]],
    ) -> list[list[int]]:
        """Compute topological levels for scheduling.

        Returns a list of levels, where each level contains indices of operations
        that can execute in parallel.
        """
        # Compute in-degree for each node
        in_degree = [0] * n
        for deps in dep_graph.values():
            for dep in deps:
                in_degree[dep] += 1

        # Find all nodes with no dependencies
        queue = [i for i in range(n) if i not in dep_graph or not dep_graph[i]]
        levels = []

        while queue:
            # Current level contains all nodes with satisfied dependencies
            current_level = list(queue)
            levels.append(current_level)

            # Find next level
            next_queue = []
            for node in current_level:
                # Check all nodes that might depend on this one
                for i in range(n):
                    if i in dep_graph and node in dep_graph[i]:
                        dep_graph[i].remove(node)
                        if not dep_graph[i]:
                            next_queue.append(i)

            queue = next_queue

        return levels

    def _get_op_type(self, op: Node) -> str:
        """Get a string identifier for the operation type."""
        return type(op).__name__
