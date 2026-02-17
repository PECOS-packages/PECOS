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

"""AST to PECOS QuantumCircuit code generator.

This module transforms AST nodes into PECOS QuantumCircuit format.
QuantumCircuit is PECOS's internal circuit representation.

Example:
    from pecos.slr.ast import slr_to_ast
    from pecos.slr.ast.codegen import AstToQuantumCircuit

    ast = slr_to_ast(slr_program)
    generator = AstToQuantumCircuit()
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
    RegisterDecl,
    RepeatStmt,
    WhileStmt,
)

if TYPE_CHECKING:
    from pecos.circuits.quantum_circuit import QuantumCircuit
    from pecos.slr.ast.nodes import (
        Program,
        Statement,
    )

# Mapping from AST GateKind to QuantumCircuit gate names
GATE_TO_QC: dict[GateKind, str] = {
    # Single-qubit Paulis
    GateKind.X: "X",
    GateKind.Y: "Y",
    GateKind.Z: "Z",
    # Hadamard
    GateKind.H: "H",
    # Phase gates
    GateKind.S: "S",
    GateKind.Sdg: "SDG",
    GateKind.T: "T",
    GateKind.Tdg: "TDG",
    # Square root gates
    GateKind.SX: "SX",
    GateKind.SY: "SY",
    GateKind.SZ: "S",  # SZ is S in many conventions
    GateKind.SXdg: "SXDG",
    GateKind.SYdg: "SYDG",
    GateKind.SZdg: "SDG",  # SZdg is Sdg
    # Rotation gates
    GateKind.RX: "RX",
    GateKind.RY: "RY",
    GateKind.RZ: "RZ",
    # Two-qubit gates
    GateKind.CX: "CX",
    GateKind.CY: "CY",
    GateKind.CZ: "CZ",
    GateKind.CH: "CH",
    # Two-qubit sqrt gates
    GateKind.SXX: "SXX",
    GateKind.SYY: "SYY",
    GateKind.SZZ: "SZZ",
    GateKind.RZZ: "RZZ",
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
class QCCodeGenContext:
    """Context for QuantumCircuit code generation."""

    qubit_map: dict[tuple[str, int], int] = field(default_factory=dict)
    next_qubit_id: int = 0
    current_tick: dict[str, set] = field(default_factory=dict)
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


class AstToQuantumCircuit:
    """Transforms AST programs into PECOS QuantumCircuit using recursive descent.

    Usage:
        generator = AstToQuantumCircuit()
        circuit = generator.generate(ast_program)
    """

    def __init__(self) -> None:
        """Initialize the generator."""
        self.context = QCCodeGenContext()
        self.circuit: QuantumCircuit | None = None
        self._in_parallel = False

    def generate(self, program: Program) -> QuantumCircuit:
        """Generate a QuantumCircuit for a program.

        Args:
            program: The AST Program to generate code for.

        Returns:
            A QuantumCircuit object.
        """
        from pecos.circuits.quantum_circuit import QuantumCircuit  # noqa: PLC0415

        self.context = QCCodeGenContext()
        self.circuit = QuantumCircuit()
        self._in_parallel = False

        # Process declarations to allocate qubits
        self._process_declarations(program)

        # Process body statements
        for stmt in program.body:
            self._process_statement(stmt)

        # Flush any remaining operations
        self._flush_tick()

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
                self.context.allocator_offsets[decl.name] = parent_offset + parent_next_offset[parent]
                parent_next_offset[parent] += decl.capacity

    def _process_statement(self, stmt: Statement) -> None:
        """Process a statement using recursive descent."""
        if isinstance(stmt, GateOp):
            self._process_gate(stmt)
            if not self._in_parallel:
                self._flush_tick()
        elif isinstance(stmt, MeasureOp):
            self._process_measure(stmt)
            if not self._in_parallel:
                self._flush_tick()
        elif isinstance(stmt, PrepareOp):
            self._process_prepare(stmt)
            if not self._in_parallel:
                self._flush_tick()
        elif isinstance(stmt, BarrierOp):
            self._flush_tick()
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

    def _process_gate(self, node: GateOp) -> None:
        """Process a gate operation."""
        gate_name = GATE_TO_QC.get(node.gate, node.gate.name)

        if node.gate in TWO_QUBIT_GATES:
            self._process_two_qubit_gate(node, gate_name)
        else:
            self._process_single_qubit_gate(node, gate_name)

    def _process_single_qubit_gate(self, node: GateOp, gate_name: str) -> None:
        """Process a single-qubit gate."""
        for target in node.targets:
            qubit = self.context.get_qubit(target.allocator, target.index)
            self._add_to_tick(gate_name, qubit)

    def _process_two_qubit_gate(self, node: GateOp, gate_name: str) -> None:
        """Process a two-qubit gate."""
        if len(node.targets) >= 2:
            q0 = self.context.get_qubit(
                node.targets[0].allocator,
                node.targets[0].index,
            )
            q1 = self.context.get_qubit(
                node.targets[1].allocator,
                node.targets[1].index,
            )
            self._add_to_tick(gate_name, (q0, q1))
        elif len(node.targets) % 2 == 0:
            # Process pairs
            for i in range(0, len(node.targets), 2):
                q0 = self.context.get_qubit(
                    node.targets[i].allocator,
                    node.targets[i].index,
                )
                q1 = self.context.get_qubit(
                    node.targets[i + 1].allocator,
                    node.targets[i + 1].index,
                )
                self._add_to_tick(gate_name, (q0, q1))

    def _process_measure(self, node: MeasureOp) -> None:
        """Process a measurement operation."""
        for target in node.targets:
            qubit = self.context.get_qubit(target.allocator, target.index)
            self._add_to_tick("Measure", qubit)

    def _process_prepare(self, node: PrepareOp) -> None:
        """Process a prepare/reset operation."""
        if node.slots is None:
            return

        for slot in node.slots:
            qubit = self.context.get_qubit(node.allocator, slot)
            self._add_to_tick("RESET", qubit)

    def _add_to_tick(self, gate_name: str, target: int | tuple[int, int]) -> None:
        """Add a gate to the current tick."""
        if gate_name not in self.context.current_tick:
            self.context.current_tick[gate_name] = set()
        self.context.current_tick[gate_name].add(target)

    def _flush_tick(self) -> None:
        """Flush the current tick to the circuit."""
        if self.context.current_tick:
            self.circuit.append(dict(self.context.current_tick))
            self.context.current_tick = {}

    def _process_if(self, node: IfStmt) -> None:
        """Process an if statement."""
        # QuantumCircuit doesn't support conditionals directly
        # Process both branches
        self._flush_tick()

        for stmt in node.then_body:
            self._process_statement(stmt)

        if node.else_body:
            self._flush_tick()
            for stmt in node.else_body:
                self._process_statement(stmt)

    def _process_while(self, node: WhileStmt) -> None:
        """Process a while loop."""
        msg = (
            "While loops cannot be converted to QuantumCircuit format as they require "
            "runtime condition evaluation. Use For or Repeat blocks with static bounds instead."
        )
        raise NotImplementedError(msg)

    def _process_for(self, node: ForStmt) -> None:
        """Process a for loop by unrolling."""
        # Unroll if bounds are static
        if isinstance(node.start, int) and isinstance(node.stop, int):
            step = node.step if isinstance(node.step, int) else 1
            for _ in range(node.start, node.stop, step):
                for stmt in node.body:
                    self._process_statement(stmt)
        else:
            msg = f"Cannot unroll For loop with non-integer bounds: start={node.start}, stop={node.stop}"
            raise TypeError(msg)

    def _process_repeat(self, node: RepeatStmt) -> None:
        """Process a repeat loop by unrolling."""
        if not isinstance(node.count, int):
            msg = f"Cannot unroll Repeat block with non-integer count: {node.count}"
            raise TypeError(msg)

        for _ in range(node.count):
            for stmt in node.body:
                self._process_statement(stmt)

    def _process_parallel(self, node: ParallelBlock) -> None:
        """Process a parallel block."""
        self._in_parallel = True

        for stmt in node.body:
            self._process_statement(stmt)

        self._in_parallel = False
        self._flush_tick()

    def _process_permute(self, node: PermuteOp) -> None:
        """Process a permutation operation.

        Updates the internal allocator mapping to swap qubit references.
        QuantumCircuit doesn't have a permute instruction, so this just updates
        how we map allocator names to qubit indices.
        """
        # Swap the allocator offsets
        for src, tgt in zip(node.sources, node.targets, strict=False):
            # Get current offsets
            src_offset = self.context.allocator_offsets.get(src, 0)
            tgt_offset = self.context.allocator_offsets.get(tgt, 0)
            # Swap them
            self.context.allocator_offsets[src] = tgt_offset
            self.context.allocator_offsets[tgt] = src_offset


def ast_to_quantum_circuit(program: Program) -> QuantumCircuit:
    """Convert an AST Program to a QuantumCircuit.

    Convenience function for simple code generation.

    Args:
        program: The AST Program to convert.

    Returns:
        A QuantumCircuit object.
    """
    generator = AstToQuantumCircuit()
    return generator.generate(program)


def ast_to_quantum_circuit_str(program: Program) -> str:
    """Convert an AST Program to a QuantumCircuit string representation.

    Convenience function for getting string output.

    Args:
        program: The AST Program to convert.

    Returns:
        QuantumCircuit as a string.
    """
    circuit = ast_to_quantum_circuit(program)
    return str(circuit)
