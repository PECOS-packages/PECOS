# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""AST to Stim circuit code generator.

This module transforms AST nodes into Stim circuit format.
Stim is a high-performance stabilizer circuit simulator.

Example:
    from pecos.slr.ast import slr_to_ast
    from pecos.slr.ast.codegen import AstToStim

    ast = slr_to_ast(slr_program)
    generator = AstToStim()
    circuit = generator.generate(ast)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pecos.slr.ast.nodes import (
    AllocatorDecl,
    BarrierOp,
    ForStmt,
    GateKind,
    GateOp,
    IfStmt,
    MeasureOp,
    ParallelBlock,
    PermuteOp,
    PrepareOp,
    Program,
    RegisterDecl,
    RepeatStmt,
    SlotRef,
    WhileStmt,
)

if TYPE_CHECKING:
    import stim

    from pecos.slr.ast.nodes import Statement

# Mapping from AST GateKind to Stim gate names
GATE_TO_STIM: dict[GateKind, str] = {
    # Single-qubit Paulis
    GateKind.X: "X",
    GateKind.Y: "Y",
    GateKind.Z: "Z",
    # Hadamard
    GateKind.H: "H",
    # Phase gates
    GateKind.S: "S",
    GateKind.Sdg: "S_DAG",
    GateKind.T: "T",
    GateKind.Tdg: "T_DAG",
    # Square root gates (mapped to S variants)
    GateKind.SZ: "S",
    GateKind.SZdg: "S_DAG",
    GateKind.SX: "SQRT_X",
    GateKind.SXdg: "SQRT_X_DAG",
    GateKind.SY: "SQRT_Y",
    GateKind.SYdg: "SQRT_Y_DAG",
    # Two-qubit gates
    GateKind.CX: "CX",
    GateKind.CY: "CY",
    GateKind.CZ: "CZ",
    # Two-qubit sqrt gates
    GateKind.SXX: "SQRT_XX",
    GateKind.SYY: "SQRT_YY",
    GateKind.SZZ: "SQRT_ZZ",
    GateKind.SXXdg: "SQRT_XX_DAG",
    GateKind.SYYdg: "SQRT_YY_DAG",
    GateKind.SZZdg: "SQRT_ZZ_DAG",
}

# Two-qubit gate kinds for special handling
TWO_QUBIT_GATES = {
    GateKind.CX,
    GateKind.CY,
    GateKind.CZ,
    GateKind.CH,
    GateKind.SXX,
    GateKind.SYY,
    GateKind.SZZ,
    GateKind.SXXdg,
    GateKind.SYYdg,
    GateKind.SZZdg,
    GateKind.RZZ,
}


@dataclass
class StimCodeGenContext:
    """Context for Stim code generation."""

    qubit_map: dict[tuple[str, int], int] = field(default_factory=dict)
    next_qubit_id: int = 0
    measurement_count: int = 0
    allocator_parents: dict[str, str | None] = field(default_factory=dict)
    allocator_offsets: dict[str, int] = field(default_factory=dict)

    def get_root_allocator(self, name: str) -> str:
        """Get the root allocator for a given allocator name."""
        current = name
        while self.allocator_parents.get(current) is not None:
            current = self.allocator_parents[current]
        return current

    def get_absolute_index(self, allocator: str, index: int) -> int:
        """Get the absolute index in the root allocator."""
        offset = self.allocator_offsets.get(allocator, 0)
        return offset + index

    def get_qubit(self, allocator: str, index: int) -> int:
        """Get or allocate a qubit ID for an allocator slot.

        For child allocators, translates to root allocator with computed offset.
        """
        # Translate to root allocator and absolute index
        root = self.get_root_allocator(allocator)
        abs_index = self.get_absolute_index(allocator, index)

        key = (root, abs_index)
        if key not in self.qubit_map:
            self.qubit_map[key] = self.next_qubit_id
            self.next_qubit_id += 1
        return self.qubit_map[key]


class AstToStim:
    """Transforms AST programs into Stim circuits using recursive descent.

    Generates Stim circuit objects suitable for stabilizer simulation.

    Usage:
        generator = AstToStim()
        circuit = generator.generate(ast_program)
    """

    def __init__(self) -> None:
        """Initialize the generator."""
        self.context = StimCodeGenContext()
        self.circuit: stim.Circuit | None = None

    def generate(self, program: Program) -> stim.Circuit:
        """Generate a Stim circuit for a program.

        Args:
            program: The AST Program to generate code for.

        Returns:
            A stim.Circuit object.
        """
        import stim

        self.context = StimCodeGenContext()
        self.circuit = stim.Circuit()

        # Process declarations to allocate qubits
        self._process_declarations(program)

        # Process body statements
        for stmt in program.body:
            self._process_statement(stmt)

        return self.circuit

    def _process_declarations(self, program: Program) -> None:
        """Process declarations to allocate qubits."""
        # First pass: collect allocator parent info
        for decl in program.declarations:
            if isinstance(decl, AllocatorDecl):
                self.context.allocator_parents[decl.name] = decl.parent

        if program.allocator:
            self.context.allocator_parents[program.allocator.name] = program.allocator.parent

        # Calculate offsets for child allocators
        self._calculate_allocator_offsets(program)

        # Allocate qubits only for root allocators
        for decl in program.declarations:
            if isinstance(decl, AllocatorDecl):
                # Only allocate for root allocators (those without parents)
                if decl.parent is None:
                    for i in range(decl.capacity):
                        self.context.get_qubit(decl.name, i)
            elif isinstance(decl, RegisterDecl):
                pass  # Classical registers don't need qubit allocation

        if program.allocator and program.allocator.parent is None:
            for i in range(program.allocator.capacity):
                self.context.get_qubit(program.allocator.name, i)

    def _calculate_allocator_offsets(self, program: Program) -> None:
        """Calculate the offset of each child allocator within its parent."""
        parent_next_offset: dict[str, int] = {}

        # Root allocators have offset 0
        for decl in program.declarations:
            if isinstance(decl, AllocatorDecl) and decl.parent is None:
                self.context.allocator_offsets[decl.name] = 0

        if program.allocator and program.allocator.parent is None:
            self.context.allocator_offsets[program.allocator.name] = 0

        # Process child allocators in declaration order
        for decl in program.declarations:
            if isinstance(decl, AllocatorDecl) and decl.parent is not None:
                parent = decl.parent
                if parent not in parent_next_offset:
                    parent_next_offset[parent] = 0

                parent_offset = self.context.allocator_offsets.get(parent, 0)
                self.context.allocator_offsets[decl.name] = (
                    parent_offset + parent_next_offset[parent]
                )
                parent_next_offset[parent] += decl.capacity

    def _process_statement(self, stmt: Statement) -> None:
        """Process a statement using recursive descent."""
        if isinstance(stmt, GateOp):
            self._process_gate(stmt)
        elif isinstance(stmt, MeasureOp):
            self._process_measure(stmt)
        elif isinstance(stmt, PrepareOp):
            self._process_prepare(stmt)
        elif isinstance(stmt, BarrierOp):
            self._process_barrier()
        elif isinstance(stmt, IfStmt):
            self._process_if(stmt)
        elif isinstance(stmt, WhileStmt):
            self._process_while(stmt)
        elif isinstance(stmt, ForStmt):
            self._process_for(stmt)
        elif isinstance(stmt, RepeatStmt):
            self._process_repeat(stmt)
        elif isinstance(stmt, ParallelBlock):
            self._process_parallel(stmt)
        elif isinstance(stmt, PermuteOp):
            self._process_permute(stmt)
        # Other statement types (Comment, Assign, Return) don't generate Stim output

    def _process_gate(self, node: GateOp) -> None:
        """Process a gate operation."""
        stim_gate = GATE_TO_STIM.get(node.gate)
        if stim_gate is None:
            # Skip unsupported gates
            return

        if node.gate in TWO_QUBIT_GATES:
            self._process_two_qubit_gate(node, stim_gate)
        else:
            self._process_single_qubit_gate(node, stim_gate)

    def _process_single_qubit_gate(self, node: GateOp, stim_gate: str) -> None:
        """Process a single-qubit gate."""
        qubits = [
            self.context.get_qubit(t.allocator, t.index) for t in node.targets
        ]
        self.circuit.append_operation(stim_gate, qubits)

    def _process_two_qubit_gate(self, node: GateOp, stim_gate: str) -> None:
        """Process a two-qubit gate."""
        if len(node.targets) >= 2:
            q0 = self.context.get_qubit(node.targets[0].allocator, node.targets[0].index)
            q1 = self.context.get_qubit(node.targets[1].allocator, node.targets[1].index)
            self.circuit.append_operation(stim_gate, [q0, q1])
        elif len(node.targets) % 2 == 0:
            # Process pairs
            for i in range(0, len(node.targets), 2):
                q0 = self.context.get_qubit(
                    node.targets[i].allocator, node.targets[i].index
                )
                q1 = self.context.get_qubit(
                    node.targets[i + 1].allocator, node.targets[i + 1].index
                )
                self.circuit.append_operation(stim_gate, [q0, q1])

    def _process_measure(self, node: MeasureOp) -> None:
        """Process a measurement operation."""
        qubits = [
            self.context.get_qubit(t.allocator, t.index) for t in node.targets
        ]
        self.circuit.append_operation("M", qubits)
        self.context.measurement_count += len(qubits)

    def _process_prepare(self, node: PrepareOp) -> None:
        """Process a prepare/reset operation."""
        if node.slots is None:
            return

        qubits = [self.context.get_qubit(node.allocator, slot) for slot in node.slots]
        self.circuit.append_operation("R", qubits)

    def _process_barrier(self) -> None:
        """Process a barrier as TICK."""
        self.circuit.append("TICK")

    def _process_if(self, node: IfStmt) -> None:
        """Process an if statement."""
        # Stim doesn't directly support conditionals
        # Process both branches with TICK markers
        self.circuit.append("TICK")

        for stmt in node.then_body:
            self._process_statement(stmt)

        if node.else_body:
            self.circuit.append("TICK")
            for stmt in node.else_body:
                self._process_statement(stmt)

    def _process_while(self, node: WhileStmt) -> None:
        """Process a while loop."""
        # Stim doesn't support runtime loops - process body once
        self.circuit.append("TICK")
        for stmt in node.body:
            self._process_statement(stmt)

    def _process_for(self, node: ForStmt) -> None:
        """Process a for loop."""
        # Try to unroll if bounds are static
        if isinstance(node.start, int) and isinstance(node.stop, int):
            step = node.step if isinstance(node.step, int) else 1
            for _ in range(node.start, node.stop, step):
                for stmt in node.body:
                    self._process_statement(stmt)
        else:
            # Can't unroll - process body once
            for stmt in node.body:
                self._process_statement(stmt)

    def _process_repeat(self, node: RepeatStmt) -> None:
        """Process a repeat loop using Stim's REPEAT block."""
        import stim

        if node.count <= 0:
            return

        # Build sub-circuit for repeat body
        original_circuit = self.circuit
        self.circuit = stim.Circuit()

        for stmt in node.body:
            self._process_statement(stmt)

        sub_circuit = self.circuit
        self.circuit = original_circuit

        # Add repeat block if sub-circuit has content
        if len(sub_circuit) > 0:
            self.circuit.append(stim.CircuitRepeatBlock(node.count, sub_circuit))

    def _process_parallel(self, node: ParallelBlock) -> None:
        """Process a parallel block."""
        # In Stim, operations within a block are naturally parallel
        for stmt in node.body:
            self._process_statement(stmt)

    def _process_permute(self, node: PermuteOp) -> None:
        """Process a permutation operation.

        Updates the internal allocator mapping to swap qubit references.
        Stim doesn't have a permute instruction, so this just updates
        how we map allocator names to qubit indices.
        """
        # Swap the allocator mappings
        for src, tgt in zip(node.sources, node.targets, strict=False):
            # Get current offsets
            src_offset = self.context.allocator_offsets.get(src, 0)
            tgt_offset = self.context.allocator_offsets.get(tgt, 0)
            # Swap them
            self.context.allocator_offsets[src] = tgt_offset
            self.context.allocator_offsets[tgt] = src_offset


def ast_to_stim(program: Program) -> stim.Circuit:
    """Convert an AST Program to a Stim circuit.

    Convenience function for simple code generation.

    Args:
        program: The AST Program to convert.

    Returns:
        A stim.Circuit object.
    """
    generator = AstToStim()
    return generator.generate(program)


def ast_to_stim_str(program: Program) -> str:
    """Convert an AST Program to a Stim circuit string.

    Convenience function for getting string output.

    Args:
        program: The AST Program to convert.

    Returns:
        Stim circuit as a string.
    """
    circuit = ast_to_stim(program)
    return str(circuit)
