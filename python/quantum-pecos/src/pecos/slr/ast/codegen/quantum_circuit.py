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

from pecos.slr.ast.codegen._allocator_layout import allocator_offsets
from pecos.slr.ast.codegen._block_flatten import flatten_block_calls
from pecos.slr.ast.codegen._loop_bounds import static_for_range
from pecos.slr.ast.codegen._permutation import parse_permute_ref, reject_permute_in_region
from pecos.slr.ast.codegen._prep_tail import prep_tail
from pecos.slr.ast.nodes import (
    AllocatorDecl,
    AssignOp,
    BarrierOp,
    BlockCall,
    CommentOp,
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
    ReturnOp,
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
    GateKind.CRX,
    GateKind.CRY,
    GateKind.CRZ,
}
CONTROLLED_ROTATION_GATES = {GateKind.CRX, GateKind.CRY, GateKind.CRZ}


@dataclass
class QCCodeGenContext:
    """Context for QuantumCircuit code generation."""

    is_fragment: bool = False
    qubit_map: dict[tuple[str, int], int] = field(default_factory=dict)
    next_qubit_id: int = 0
    current_tick: dict[str, set] = field(default_factory=dict)
    # Insertion order preserves measurement order within a parallel tick.
    measurement_results: dict[int, tuple[str, int] | None] = field(default_factory=dict)
    allocator_parents: dict[str, str | None] = field(default_factory=dict)
    allocator_offsets: dict[str, int] = field(default_factory=dict)
    qreg_sizes: dict[str, int] = field(default_factory=dict)  # name -> capacity
    # Static logical permutation (same model as the QIR codegen /
    # the Guppy linearity tracker -- QuantumCircuit has no permute
    # instruction). Maps a logical (reg, index) ref to the (reg,
    # index) whose qubit it resolves to; consulted in `get_qubit`.
    permutation_map: dict[tuple[str, int], tuple[str, int]] = field(default_factory=dict)

    def get_root_allocator(self, name: str) -> str:
        """Get the root allocator for a given allocator name."""
        current = name
        while self.allocator_parents.get(current) is not None:
            current = self.allocator_parents[current]
        if current not in self.allocator_parents and not self.is_fragment:
            msg = f"Undeclared allocator {current!r}; declare it before generating code."
            raise ValueError(msg)
        return current

    def get_absolute_index(self, allocator: str, index: int) -> int:
        """Get the absolute index in the root allocator."""
        if self.is_fragment and allocator not in self.allocator_offsets:
            return index
        offset = self.allocator_offsets[allocator]
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

        program = flatten_block_calls(program)

        self.context = QCCodeGenContext(is_fragment=program.is_fragment)
        self.circuit = QuantumCircuit(cvar_spec={})
        self._in_parallel = False

        # Process declarations to allocate qubits
        self._process_declarations(program)

        # Process body statements
        for stmt in program.body:
            self._process_statement(stmt)

        # Flush any remaining operations
        self._flush_tick()

        self.circuit.metadata["qvars"] = {qubit: ref for ref, qubit in self.context.qubit_map.items()}
        qvar_spec: dict[str, int] = {}
        for name, index in self.context.qubit_map:
            qvar_spec[name] = max(qvar_spec.get(name, 0), index + 1)
        self.circuit.metadata["qvar_spec"] = qvar_spec

        return self.circuit

    def _process_declarations(self, program: Program) -> None:
        """Process declarations to allocate qubits and declare classical storage."""
        # First pass: collect allocator parent info
        for decl in program.declarations:
            if isinstance(decl, AllocatorDecl):
                self.context.allocator_parents[decl.name] = decl.parent

        if program.allocator:
            self.context.allocator_parents[program.allocator.name] = program.allocator.parent

        # Calculate offsets for child allocators
        self.context.allocator_offsets = allocator_offsets(program)

        # Allocate qubits only for root allocators
        for decl in program.declarations:
            if isinstance(decl, AllocatorDecl):
                self.context.qreg_sizes[decl.name] = decl.capacity
                if decl.parent is None:
                    for i in range(decl.capacity):
                        self.context.get_qubit(decl.name, i)
            elif isinstance(decl, RegisterDecl):
                self.circuit.metadata["cvar_spec"][decl.name] = decl.size

        if program.allocator:
            self.context.qreg_sizes[program.allocator.name] = program.allocator.capacity
        if program.allocator and program.allocator.parent is None:
            for i in range(program.allocator.capacity):
                self.context.get_qubit(program.allocator.name, i)

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
        elif isinstance(stmt, AssignOp):
            msg = (
                f"QuantumCircuit codegen does not yet support {type(stmt).__name__}: "
                "classical assignments are not lowered. "
                "Use the QASM, QIR or Guppy targets for these operations."
            )
            raise NotImplementedError(msg)
        elif isinstance(stmt, BlockCall):
            msg = (
                f"QuantumCircuit codegen cannot lower BlockCall {stmt.callee!r}; "
                "provide its block declaration so the call can be flattened."
            )
            raise NotImplementedError(msg)
        elif isinstance(stmt, CommentOp | ReturnOp):
            pass  # Comments and output declarations have no circuit effect.
        elif isinstance(stmt, PrintOp):
            # Print is not yet implemented for the QuantumCircuit
            # backend. Fail LOUD rather than silently drop observable
            # program output (fail-loud principle). Unlike Stim, PECOS owns
            # the QuantumCircuit format, so a `Print` representation
            # could be added later -- this is "not yet", not "cannot".
            msg = (
                "QuantumCircuit codegen does not yet support Print "
                "(classical output streaming is not implemented for this "
                "backend; it is silently-drop-free by design -- fail "
                "loud. PECOS controls this format, so Print support may "
                "be added in future)."
            )
            raise NotImplementedError(msg)
        else:
            msg = (
                f"QuantumCircuit codegen does not support statement {type(stmt).__name__}; "
                "lower it to supported operations before conversion."
            )
            raise NotImplementedError(msg)

    def _process_gate(self, node: GateOp) -> None:
        """Process a gate operation."""
        gate_name = GATE_TO_QC.get(node.gate, node.gate.name)

        if node.params:
            # Parameterized gates (RX/RY/RZ/RZZ/CRX/CRY/CRZ etc.):
            # PECOS QuantumCircuit ticks are parallel sets keyed on
            # (gate_name, params), so a tick mixing different param
            # values would lose information. Flush the current tick,
            # then emit the parameterized gate as its own tick via
            # `circuit.append(gate, locations, angles=[...])` which the
            # Rust gate registry routes to the typed-param dispatcher.
            self._process_parameterized_gate(node, gate_name)
            return

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

    def _process_parameterized_gate(self, node: GateOp, gate_name: str) -> None:
        """Emit a parameterized gate (rotation angle threaded through).

        Resolves `LiteralExpr` bracket-params to raw floats (the AST
        converter wraps the `qb.RZ(0.5, q)` angle as a `LiteralExpr`, so a bare
        `float(p)` would fail). The QC `circuit.append(...,
        angles=...)` path forwards the angle list to Rust's typed-
        parameter dispatcher (e.g. `RZ` requires 1 angle, `RXXRYYRZZ`
        requires 3).

        Flushes the current tick before emission so the parameterized
        gate gets its own tick -- mixing different param values within
        a single tick would lose information (tick batches by
        `(gate_name, target)` only). Non-literal expressions (VarExpr,
        BinaryExpr at gate-param position) are not yet supported for
        QC parameterized gates; they fail loud here.
        """
        from pecos.slr.angle import Angle  # noqa: PLC0415  (avoid import cycle)

        # Stored rotations carry typed `Angle` literals. Controlled-rotation
        # boundary spellings carry unreduced real radians literals.
        for p in node.params:
            if node.gate in CONTROLLED_ROTATION_GATES:
                valid = (
                    isinstance(p, LiteralExpr) and isinstance(p.value, int | float) and not isinstance(p.value, bool)
                )
            else:
                valid = isinstance(p, LiteralExpr) and isinstance(p.value, Angle)
            if not valid:
                expected = (
                    "an unreduced real radians literal"
                    if node.gate in CONTROLLED_ROTATION_GATES
                    else "typed `Angle` params (use `rad(...)` / `turns(...)` in SLR)"
                )
                msg = f"QuantumCircuit codegen: parameterized gate {gate_name!r} requires {expected}; got {p!r}."
                raise NotImplementedError(msg)
        angles = [
            float(p.value) if node.gate in CONTROLLED_ROTATION_GATES else p.value.value.to_radians_signed()
            for p in node.params
        ]

        self._flush_tick()
        if node.gate in TWO_QUBIT_GATES:
            if len(node.targets) < 2:
                msg = f"QuantumCircuit codegen: two-qubit gate {gate_name!r} needs >=2 targets, got {len(node.targets)}"
                raise ValueError(msg)
            # Emit each consecutive pair (mirrors the un-parameterized
            # two-qubit path which iterates over target pairs).
            for i in range(0, len(node.targets) - 1, 2):
                q0 = self.context.get_qubit(node.targets[i].allocator, node.targets[i].index)
                q1 = self.context.get_qubit(node.targets[i + 1].allocator, node.targets[i + 1].index)
                if node.gate in CONTROLLED_ROTATION_GATES:
                    theta = angles[0]
                    if node.gate is GateKind.CRX:
                        self.circuit.append("H", {q1})
                    elif node.gate is GateKind.CRY:
                        self.circuit.append("SX", {q1})
                    self.circuit.append("RZZ", {(q0, q1)}, angles=[-theta / 2])
                    self.circuit.append("RZ", {q1}, angles=[theta / 2])
                    if node.gate is GateKind.CRX:
                        self.circuit.append("H", {q1})
                    elif node.gate is GateKind.CRY:
                        self.circuit.append("SXdg", {q1})
                else:
                    self.circuit.append(gate_name, {(q0, q1)}, angles=angles)
        else:
            for target in node.targets:
                qubit = self.context.get_qubit(target.allocator, target.index)
                self.circuit.append(gate_name, {qubit}, angles=angles)

    def _process_measure(self, node: MeasureOp) -> None:
        """Process a measurement operation."""
        for i, target in enumerate(node.targets):
            qubit = self.context.get_qubit(target.allocator, target.index)
            result = node.results[i] if i < len(node.results) else None
            self._add_to_tick("Measure", qubit, result=None if result is None else (result.register, result.index))

    def _process_prepare(self, node: PrepareOp) -> None:
        """Process a prepare/reset operation (Z-reset + canonical basis tail).

        QC ticks are parallel sets; reset and the Clifford tail MUST
        be sequential, so each is its own flushed tick (PZ has no
        tail -> byte-identical to the prior behaviour).
        """
        tail = prep_tail(node.basis)
        if node.slots is None and node.allocator not in self.context.qreg_sizes:
            msg = (
                f"QuantumCircuit codegen: prepare_all references unknown allocator {node.allocator!r}; "
                "declare the allocator before preparing it."
            )
            raise ValueError(msg)
        slots = range(self.context.qreg_sizes[node.allocator]) if node.slots is None else node.slots
        qubits = [self.context.get_qubit(node.allocator, slot) for slot in slots]
        for qubit in qubits:
            self._add_to_tick("RESET", qubit)
        for gk in tail:
            self._flush_tick()  # sequence reset/prev-tail before this gate
            for qubit in qubits:
                self._add_to_tick(GATE_TO_QC[gk], qubit)

    def _add_to_tick(
        self,
        gate_name: str,
        target: int | tuple[int, int],
        *,
        result: tuple[str, int] | None = None,
    ) -> None:
        """Add a gate, flushing any overlapping operation to preserve qubit order."""
        qubits = {target} if isinstance(target, int) else set(target)
        occupied = {
            qubit
            for targets in self.context.current_tick.values()
            for location in targets
            for qubit in ((location,) if isinstance(location, int) else location)
        }
        if qubits.intersection(occupied):
            self._flush_tick()
        if gate_name == "Measure":
            self.context.measurement_results[target] = result
        if gate_name not in self.context.current_tick:
            self.context.current_tick[gate_name] = set()
        self.context.current_tick[gate_name].add(target)

    def _flush_tick(self) -> None:
        """Flush the current tick to the circuit."""
        if self.context.current_tick:
            tick = len(self.circuit)
            for gate_name, targets in self.context.current_tick.items():
                if gate_name == "Measure":
                    for qubit, result in self.context.measurement_results.items():
                        params = {} if result is None else {"var_output": {qubit: result}}
                        self.circuit.update(gate_name, {qubit}, tick=tick, emptyappend=True, **params)
                else:
                    self.circuit.update(gate_name, targets, tick=tick, emptyappend=True)
            self.context.current_tick = {}
            self.context.measurement_results = {}

    def _process_if(self, node: IfStmt) -> None:
        """Reject an if statement: classical control flow is not lowered here."""
        reject_permute_in_region(node, "QuantumCircuit")
        msg = (
            "If cannot be converted to QuantumCircuit format: this backend does not lower "
            "classical conditions onto branch operations. "
            "Use the QASM, QIR or Guppy targets for programs with conditionals."
        )
        raise NotImplementedError(msg)

    def _process_while(self, node: WhileStmt) -> None:
        """Process a while loop."""
        reject_permute_in_region(node, "QuantumCircuit")
        msg = (
            "While loops cannot be converted to QuantumCircuit format as they require "
            "runtime condition evaluation. Use For or Repeat blocks with static bounds instead."
        )
        raise NotImplementedError(msg)

    def _process_for(self, node: ForStmt) -> None:
        """Unroll a static fixed-bound For."""
        for _ in static_for_range(node, "QuantumCircuit"):
            for stmt in node.body:
                self._process_statement(stmt)

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
        was_parallel = self._in_parallel
        self._in_parallel = True

        for stmt in node.body:
            self._process_statement(stmt)

        self._in_parallel = was_parallel
        if not self._in_parallel:
            self._flush_tick()

    def _expand_permute_ref(self, ref: str) -> list[tuple[str, int]]:
        """Expand a Permute ref string to logical (reg, index) pairs.

        `name[idx]` -> a single element; bare `name` -> every element
        of the qubit register. Classical Permute requires runtime
        assignments, which this backend does not lower.
        """
        parsed = parse_permute_ref(ref, "QuantumCircuit")
        name = parsed[0] if parsed is not None else ref
        if name in self.circuit.metadata.get("cvar_spec", {}):
            msg = (
                f"QuantumCircuit codegen: classical Permute of {ref!r} is not supported; "
                "use the QASM, QIR or Guppy targets for classical permutations."
            )
            raise NotImplementedError(msg)
        if name not in self.context.qreg_sizes:
            msg = (
                f"QuantumCircuit codegen: Permute reference {ref!r} is not a known qubit allocator; "
                "declare a qubit allocator or use the QASM, QIR or Guppy targets for classical permutations."
            )
            raise ValueError(msg)
        if parsed is not None:
            return [parsed]
        return [(ref, i) for i in range(self.context.qreg_sizes[ref])]

    def _process_permute(self, node: PermuteOp) -> None:
        """Realize a Permute as a static logical relabel.

        QuantumCircuit has no permute instruction, so -- exactly like
        the QIR codegen and the Guppy linearity tracker -- a
        Permute is realized at compile time by relabelling which qubit
        each logical (reg, index) ref resolves to (consulted in
        `get_qubit`). The old `allocator_offsets` swap was a no-op for
        element-wise refs and self-cancelling for a whole-register
        (a,b)/(b,a) pair -- a silent miscompile.
        """
        if len(node.sources) != len(node.targets):
            msg = "QuantumCircuit codegen: Permute source/target length mismatch"
            raise NotImplementedError(msg)

        # Validate the expanded ref lists BEFORE building the dict (a
        # dict would silently collapse a duplicate source).
        src_all: list[tuple[str, int]] = []
        tgt_all: list[tuple[str, int]] = []
        for source, target in zip(node.sources, node.targets, strict=True):
            src_refs = self._expand_permute_ref(source)
            tgt_refs = self._expand_permute_ref(target)
            if len(src_refs) != len(tgt_refs):
                msg = f"QuantumCircuit codegen: Permute element count mismatch for {source!r} -> {target!r}"
                raise NotImplementedError(msg)
            src_all.extend(src_refs)
            tgt_all.extend(tgt_refs)

        if len(src_all) != len(set(src_all)):
            msg = "QuantumCircuit codegen: Permute has a duplicate source ref (not a permutation)"
            raise NotImplementedError(msg)
        if len(tgt_all) != len(set(tgt_all)):
            msg = "QuantumCircuit codegen: Permute has a duplicate target ref (not a permutation)"
            raise NotImplementedError(msg)
        if set(src_all) != set(tgt_all):
            msg = "QuantumCircuit codegen: Permute must be bijective over the same ref set"
            raise NotImplementedError(msg)

        # Compose ATOMICALLY (snapshot old, then map[s] = old.get(t, t))
        # so a whole-register (a,b)/(b,a) pair applies once.
        old = dict(self.context.permutation_map)
        self.context.permutation_map.update({s: old.get(t, t) for s, t in zip(src_all, tgt_all, strict=True)})


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
