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

from pecos.slr.ast.codegen._block_flatten import flatten_block_calls
from pecos.slr.ast.codegen._prep_tail import prep_tail
from pecos.slr.ast.nodes import (
    AllocatorDecl,
    BarrierOp,
    ForStmt,
    GateKind,
    GateOp,
    IfStmt,
    LiteralExpr,
    MeasureOp,
    ParallelBlock,
    PermuteOp,
    PrepareOp,
    PrintOp,
    RegisterDecl,
    RepeatStmt,
    WhileStmt,
)

if TYPE_CHECKING:
    import stim

    from pecos.slr.ast.nodes import (
        Program,
        Statement,
    )

# Mapping from AST GateKind to Stim gate names
GATE_TO_STIM: dict[GateKind, str] = {
    # Single-qubit Paulis
    GateKind.X: "X",
    GateKind.Y: "Y",
    GateKind.Z: "Z",
    # Hadamard
    GateKind.H: "H",
    # Phase gates
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

# Decomposition table: Clifford-only PECOS gates with no direct Stim
# primitive, lowered into the Stim-native gate set. Each entry is a
# sequence of (Stim gate name, qubit-index tuple-into-targets) steps
# in CIRCUIT order (first applied first). The compositions mirror the
# already-verified QIR `_GATE_DECOMP` entries: each H/S/S_DAG step is
# itself a Stim primitive, so a decomp using only those primitives is
# correct iff the QIR-side decomp is correct (which was verified
# up-to-phase against the PECOS oracle and end-to-end via selene).
#
# Stim is a Clifford-only stabilizer simulator: gates that involve
# non-Clifford rotations (CH, CRX/CRY/CRZ, T-decomposable forms) or
# arbitrary continuous rotations (RX/RY/RZ/RZZ at non-pi/2 angles)
# remain fail-loud here -- there is no Clifford-only decomposition,
# and the qir-qis-style "decompose to a Clifford+T target" path is
# fundamentally unavailable to Stim (the user's "support all gates
# in all languages, IF DECOMPOSABLE" directive admits this caveat).
_GATE_DECOMP: dict[GateKind, tuple[tuple[str, tuple[int, ...]], ...]] = {
    # F = H . SZdg (matrix product; circuit-time: SZdg first, then H).
    # F cycles Paulis X -> Y -> Z -> X (face rotation of the Bloch cube).
    GateKind.F: (("S_DAG", (0,)), ("H", (0,))),
    # Fdg = SZ . H (inverse of F; cycles X <- Y <- Z <- X).
    GateKind.Fdg: (("H", (0,)), ("S", (0,))),
    # F4 = SZdg . H -- the F-rotation around a different face axis.
    GateKind.F4: (("H", (0,)), ("S_DAG", (0,))),
    # F4dg = H . SZ -- inverse of F4.
    GateKind.F4dg: (("S", (0,)), ("H", (0,))),
}


@dataclass
class StimCodeGenContext:
    """Context for Stim code generation."""

    qubit_map: dict[tuple[str, int], int] = field(default_factory=dict)
    next_qubit_id: int = 0
    measurement_count: int = 0
    allocator_parents: dict[str, str | None] = field(default_factory=dict)
    allocator_offsets: dict[str, int] = field(default_factory=dict)
    qreg_sizes: dict[str, int] = field(default_factory=dict)  # name -> capacity
    # Static logical permutation (same model as the QIR codegen /
    # the Guppy linearity tracker -- Stim has no permute instruction).
    # Maps a logical (reg, index) ref to the (reg, index) whose qubit
    # it resolves to. Consulted at every qubit-ref lowering.
    permutation_map: dict[tuple[str, int], tuple[str, int]] = field(default_factory=dict)

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
        # Resolve any active logical permutation first (identity until
        # a Permute runs; decl-time pre-population sees the empty map,
        # so real qubits are still allocated 1:1).
        allocator, index = self.permutation_map.get((allocator, index), (allocator, index))

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
        import stim  # noqa: PLC0415

        program = flatten_block_calls(program)

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
                self.context.qreg_sizes[decl.name] = decl.capacity
                # Only allocate for root allocators (those without parents)
                if decl.parent is None:
                    for i in range(decl.capacity):
                        self.context.get_qubit(decl.name, i)
            elif isinstance(decl, RegisterDecl):
                pass  # Classical registers don't need qubit allocation

        if program.allocator:
            self.context.qreg_sizes[program.allocator.name] = program.allocator.capacity
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
        elif isinstance(stmt, PrintOp):
            # Classical-output streaming is unimplemented in the Stim
            # backend. Silently dropping it loses observable program
            # output -- fail LOUD (same decision as the QIR backend).
            msg = (
                "Stim codegen does not support Print (classical output "
                "streaming is unimplemented; silently dropping it would "
                "lose observable program output)."
            )
            raise NotImplementedError(msg)
        # Other statement types (Comment, Assign, Return) don't generate Stim output

    def _process_gate(self, node: GateOp) -> None:
        """Process a gate operation."""
        stim_gate = GATE_TO_STIM.get(node.gate)
        if stim_gate is None and node.gate in _GATE_DECOMP:
            # A Clifford gate with no direct Stim primitive but a
            # verified decomposition into Stim-native primitives.
            # Emit each step in circuit order.
            qubits = [self.context.get_qubit(t.allocator, t.index) for t in node.targets]
            for prim_name, idxs in _GATE_DECOMP[node.gate]:
                prim_qubits = [qubits[i] for i in idxs]
                self.circuit.append_operation(prim_name, prim_qubits)
            return
        if stim_gate is None:
            # A gate with no GATE_TO_STIM entry was SILENTLY DROPPED
            # -- the emitted Stim circuit ran but with wrong
            # semantics (a silent miscompile,
            # uncatchable downstream). Fail loud instead. Stim is
            # Clifford-only, so non-Clifford rotations
            # (RX/RY/RZ/RZZ/CR*) are fundamentally unrepresentable
            # here; CH is non-Clifford too. Gates that are Clifford
            # but lack a direct Stim primitive get a verified
            # `_GATE_DECOMP` entry above; anything that reaches this
            # raise has no representable form.
            gate_name = getattr(node.gate, "name", node.gate)
            msg = (
                f"Stim codegen: gate {gate_name!r} has no Stim lowering "
                "(not in GATE_TO_STIM, no Clifford decomposition in "
                "_GATE_DECOMP). Emitting the circuit without it would be "
                "a silent miscompile; it is not supported by the Stim "
                "backend (non-Clifford gates like CH, CR*, continuous "
                "rotations are fundamentally unrepresentable here)."
            )
            raise NotImplementedError(msg)

        if node.gate in TWO_QUBIT_GATES:
            self._process_two_qubit_gate(node, stim_gate)
        else:
            self._process_single_qubit_gate(node, stim_gate)

    def _process_single_qubit_gate(self, node: GateOp, stim_gate: str) -> None:
        """Process a single-qubit gate."""
        qubits = [self.context.get_qubit(t.allocator, t.index) for t in node.targets]
        self.circuit.append_operation(stim_gate, qubits)

    def _process_two_qubit_gate(self, node: GateOp, stim_gate: str) -> None:
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
            self.circuit.append_operation(stim_gate, [q0, q1])
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
                self.circuit.append_operation(stim_gate, [q0, q1])

    def _process_measure(self, node: MeasureOp) -> None:
        """Process a measurement operation."""
        qubits = [self.context.get_qubit(t.allocator, t.index) for t in node.targets]
        self.circuit.append_operation("M", qubits)
        self.context.measurement_count += len(qubits)

    def _process_prepare(self, node: PrepareOp) -> None:
        """Process a prepare/reset operation (Z-reset + canonical basis tail)."""
        tail = prep_tail(node.basis)
        if node.slots is None:
            if tail:
                msg = f"Stim codegen: prepare_all with non-PZ basis {node.basis!r} is not supported"
                raise NotImplementedError(msg)
            return

        qubits = [self.context.get_qubit(node.allocator, slot) for slot in node.slots]
        self.circuit.append_operation("R", qubits)
        for gk in tail:
            self.circuit.append_operation(GATE_TO_STIM[gk], qubits)

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
        """`While` is not supported by the Stim backend.

        Stim has no runtime loop. The previous "process body once + TICK"
        silently dropped the loop condition and all iterations -- a
        miscompile. Fail LOUD instead (same decision as the QIR backend;
        real While is out of scope).
        """
        _ = node
        msg = (
            "Stim codegen does not support While loops (Stim has no "
            "runtime loop; a single-pass approximation would be a silent "
            "miscompile)."
        )
        raise NotImplementedError(msg)

    def _static_int_bound(self, expr: object, which: str) -> int:
        """Resolve a static integer `For` bound.

        The AST converter wraps integer range bounds in `LiteralExpr`
        (`converter.py` `_convert_for`), so the bound is never a raw
        `int` -- the old `isinstance(int)` guard was always false and
        silently dropped every `For` body. A non-literal / non-int
        bound is a symbolic/dynamic `For`: fail LOUD, never drop.
        """
        if isinstance(expr, LiteralExpr) and isinstance(expr.value, int) and not isinstance(expr.value, bool):
            return expr.value
        msg = (
            f"Stim codegen: For loop {which} bound is not a static integer "
            f"({type(expr).__name__}); only fixed-bound `For(i, <int>, "
            "<int>)` is supported (symbolic/dynamic For is out of scope -- "
            "and must not silently drop the loop body)."
        )
        raise NotImplementedError(msg)

    def _process_for(self, node: ForStmt) -> None:
        """Unroll a static fixed-bound `For` (v1-supported)."""
        start = self._static_int_bound(node.start, "start")
        stop = self._static_int_bound(node.stop, "stop")
        step = 1 if node.step is None else self._static_int_bound(node.step, "step")
        if step == 0:
            msg = "Stim codegen: For loop step is 0 (infinite loop); only a non-zero static step is supported."
            raise NotImplementedError(msg)
        for _ in range(start, stop, step):
            for stmt in node.body:
                self._process_statement(stmt)

    def _process_repeat(self, node: RepeatStmt) -> None:
        """Process a repeat loop using Stim's REPEAT block."""
        import stim  # noqa: PLC0415

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

    def _expand_permute_ref(self, ref: str) -> list[tuple[str, int]]:
        """Expand a Permute ref string to logical (reg, index) pairs.

        `name[idx]` -> a single element; bare `name` -> every element
        of the qubit register. Stim has no classical-register model,
        so a bare CReg permute is not realizable -> fail loud (never
        a silent no-op). Mirrors the QIR codegen's helper.
        """
        if ref.endswith("]") and "[" in ref:
            name, idx = ref[:-1].split("[", 1)
            return [(name, int(idx))]
        if ref in self.context.qreg_sizes:
            return [(ref, i) for i in range(self.context.qreg_sizes[ref])]
        msg = (
            f"Stim codegen: whole-register Permute of {ref!r} is not "
            "supported (no classical-register model in Stim); a "
            "qubit-register or element-wise Permute is realizable."
        )
        raise NotImplementedError(msg)

    def _process_permute(self, node: PermuteOp) -> None:
        """Realize a Permute as a static logical relabel.

        Stim has no permute instruction, so -- exactly like the QIR
        codegen and the Guppy linearity tracker -- a Permute is
        realized at compile time by relabelling which qubit each
        logical (reg, index) ref resolves to (consulted in
        `get_qubit`). The old `allocator_offsets` swap was a no-op
        for element-wise refs and self-cancelling for a whole-register
        (a,b)/(b,a) pair -- a silent miscompile.
        """
        if len(node.sources) != len(node.targets):
            msg = "Stim codegen: Permute source/target length mismatch"
            raise NotImplementedError(msg)

        # Validate the expanded ref lists BEFORE building the dict (a
        # dict would silently collapse a duplicate source).
        src_all: list[tuple[str, int]] = []
        tgt_all: list[tuple[str, int]] = []
        for source, target in zip(node.sources, node.targets, strict=True):
            src_refs = self._expand_permute_ref(source)
            tgt_refs = self._expand_permute_ref(target)
            if len(src_refs) != len(tgt_refs):
                msg = f"Stim codegen: Permute element count mismatch for {source!r} -> {target!r}"
                raise NotImplementedError(msg)
            src_all.extend(src_refs)
            tgt_all.extend(tgt_refs)

        if len(src_all) != len(set(src_all)):
            msg = "Stim codegen: Permute has a duplicate source ref (not a permutation)"
            raise NotImplementedError(msg)
        if len(tgt_all) != len(set(tgt_all)):
            msg = "Stim codegen: Permute has a duplicate target ref (not a permutation)"
            raise NotImplementedError(msg)
        if set(src_all) != set(tgt_all):
            msg = "Stim codegen: Permute must be bijective over the same ref set"
            raise NotImplementedError(msg)

        # Compose ATOMICALLY (snapshot old, then map[s] = old.get(t, t))
        # so a whole-register (a,b)/(b,a) pair applies once.
        old = dict(self.context.permutation_map)
        self.context.permutation_map.update({s: old.get(t, t) for s, t in zip(src_all, tgt_all, strict=True)})


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
