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

"""Tests for the HUGR to AST converter."""

from __future__ import annotations

import pytest

# Check if guppylang is available
try:
    from guppylang import guppy
    from guppylang.std.quantum import cx, cz, h, measure, qubit, s, t, x, y, z

    HAS_GUPPYLANG = True
except ImportError:
    HAS_GUPPYLANG = False

# Check if hugr_to_ast is available
try:
    from pecos.circuit_converters.hugr_to_ast import (
        UnsupportedHugrStructureError,
        guppy_to_ast,
        hugr_to_ast,
    )
    from pecos.slr.ast.nodes import (
        AllocatorDecl,
        GateKind,
        GateOp,
        IfStmt,
        MeasureOp,
        PrepareOp,
        Program,
    )

    HAS_HUGR_TO_AST = True
except ImportError:
    HAS_HUGR_TO_AST = False

pytestmark = pytest.mark.skipif(
    not (HAS_GUPPYLANG and HAS_HUGR_TO_AST),
    reason="guppylang or hugr_to_ast not available",
)


class TestBasicConversion:
    """Tests for basic HUGR to AST conversion."""

    def test_single_qubit_circuit(self) -> None:
        """Test conversion of a single-qubit circuit."""

        @guppy
        def single_h() -> bool:
            q = qubit()
            h(q)
            return measure(q)

        ast = guppy_to_ast(single_h)

        # Check it's a Program
        assert isinstance(ast, Program)

        # Check declaration (should have AllocatorDecl and RegisterDecl)
        alloc_decls = [d for d in ast.declarations if isinstance(d, AllocatorDecl)]
        assert len(alloc_decls) == 1
        assert alloc_decls[0].name == "q"
        assert alloc_decls[0].capacity == 1

        # Check body has Prep, H, Measure
        assert len(ast.body) == 3

        # First should be PrepareOp
        assert isinstance(ast.body[0], PrepareOp)
        assert ast.body[0].allocator == "q"

        # Second should be H gate
        assert isinstance(ast.body[1], GateOp)
        assert ast.body[1].gate == GateKind.H

        # Third should be MeasureOp
        assert isinstance(ast.body[2], MeasureOp)

    def test_bell_state_circuit(self) -> None:
        """Test conversion of a Bell state circuit."""

        @guppy
        def bell() -> tuple[bool, bool]:
            q0 = qubit()
            q1 = qubit()
            h(q0)
            cx(q0, q1)
            return measure(q0), measure(q1)

        ast = guppy_to_ast(bell)

        # Should have 2 qubits
        assert ast.declarations[0].capacity == 2

        # Count operations by type
        preps = [s for s in ast.body if isinstance(s, PrepareOp)]
        gates = [s for s in ast.body if isinstance(s, GateOp)]
        measures = [s for s in ast.body if isinstance(s, MeasureOp)]

        assert len(preps) == 2  # Two qubit preparations
        assert len(gates) == 2  # H and CX
        assert len(measures) == 2  # Two measurements

        # Check gate types
        gate_kinds = [g.gate for g in gates]
        assert GateKind.H in gate_kinds
        assert GateKind.CX in gate_kinds

    def test_multi_gate_circuit(self) -> None:
        """Test conversion of a circuit with multiple gate types."""

        @guppy
        def multi_gate() -> bool:
            q = qubit()
            h(q)
            t(q)
            s(q)
            x(q)
            y(q)
            z(q)
            return measure(q)

        ast = guppy_to_ast(multi_gate)

        # Get all gates
        gates = [s for s in ast.body if isinstance(s, GateOp)]
        gate_kinds = {g.gate for g in gates}

        # Check all gate types are present
        assert GateKind.H in gate_kinds
        assert GateKind.T in gate_kinds
        assert GateKind.SZ in gate_kinds  # S maps to SZ
        assert GateKind.X in gate_kinds
        assert GateKind.Y in gate_kinds
        assert GateKind.Z in gate_kinds

    def test_two_qubit_gates(self) -> None:
        """Test conversion with two-qubit gates."""

        @guppy
        def two_qubit() -> tuple[bool, bool]:
            q0 = qubit()
            q1 = qubit()
            cx(q0, q1)
            cz(q0, q1)
            return measure(q0), measure(q1)

        ast = guppy_to_ast(two_qubit)

        gates = [s for s in ast.body if isinstance(s, GateOp)]
        gate_kinds = {g.gate for g in gates}

        assert GateKind.CX in gate_kinds
        assert GateKind.CZ in gate_kinds


class TestProgramMetadata:
    """Tests for program metadata extraction."""

    def test_function_name_extraction(self) -> None:
        """Test that function name is extracted correctly."""

        @guppy
        def my_quantum_circuit() -> bool:
            q = qubit()
            h(q)
            return measure(q)

        ast = guppy_to_ast(my_quantum_circuit)
        assert ast.name == "my_quantum_circuit"

    def test_custom_allocator_name(self) -> None:
        """Test custom allocator name."""

        @guppy
        def circuit() -> bool:
            q = qubit()
            h(q)
            return measure(q)

        ast = guppy_to_ast(circuit, allocator_name="qubits")

        assert ast.declarations[0].name == "qubits"

        # Check SlotRefs use the custom name
        gates = [s for s in ast.body if isinstance(s, GateOp)]
        for gate in gates:
            for slot_ref in gate.targets:
                assert slot_ref.allocator == "qubits"


class TestSlotReferences:
    """Tests for correct SlotRef generation."""

    def test_single_qubit_slot_refs(self) -> None:
        """Test SlotRef indices for single-qubit gates."""

        @guppy
        def circuit() -> bool:
            q = qubit()
            h(q)
            x(q)
            return measure(q)

        ast = guppy_to_ast(circuit)

        gates = [s for s in ast.body if isinstance(s, GateOp)]

        # All gates should target qubit 0
        for gate in gates:
            assert len(gate.targets) == 1
            assert gate.targets[0].index == 0

    def test_two_qubit_slot_refs(self) -> None:
        """Test SlotRef indices for two-qubit gates."""

        @guppy
        def circuit() -> tuple[bool, bool]:
            q0 = qubit()
            q1 = qubit()
            cx(q0, q1)
            return measure(q0), measure(q1)

        ast = guppy_to_ast(circuit)

        cx_gates = [s for s in ast.body if isinstance(s, GateOp) and s.gate == GateKind.CX]

        assert len(cx_gates) == 1
        cx_gate = cx_gates[0]

        # CX should have 2 targets
        assert len(cx_gate.targets) == 2

        # Check indices are 0 and 1
        indices = {t.index for t in cx_gate.targets}
        assert indices == {0, 1}


class TestMinimalCircuits:
    """Tests for edge cases with minimal circuits."""

    def test_single_allocation_and_measure(self) -> None:
        """Test circuit with just allocation and measurement."""

        @guppy
        def minimal() -> bool:
            q = qubit()
            return measure(q)

        ast = guppy_to_ast(minimal)

        # Should have Prep and Measure only
        assert len(ast.body) == 2
        assert isinstance(ast.body[0], PrepareOp)
        assert isinstance(ast.body[1], MeasureOp)


class TestGuppyToAstWrapper:
    """Tests for the guppy_to_ast convenience wrapper."""

    def test_basic_usage(self) -> None:
        """Test basic guppy_to_ast usage."""

        @guppy
        def circuit() -> bool:
            q = qubit()
            h(q)
            return measure(q)

        ast = guppy_to_ast(circuit)

        assert isinstance(ast, Program)
        alloc_decls = [d for d in ast.declarations if isinstance(d, AllocatorDecl)]
        assert len(alloc_decls) == 1
        assert len(ast.body) == 3

    def test_equivalent_to_manual_compilation(self) -> None:
        """Test that guppy_to_ast produces same result as manual compilation."""

        @guppy
        def circuit() -> tuple[bool, bool]:
            q0 = qubit()
            q1 = qubit()
            h(q0)
            cx(q0, q1)
            return measure(q0), measure(q1)

        # Using guppy_to_ast
        ast1 = guppy_to_ast(circuit)

        # Manual compilation
        package = circuit.compile()
        ast2 = hugr_to_ast(package.modules[0])

        # Should have same structure
        assert len(ast1.declarations) == len(ast2.declarations)
        assert len(ast1.body) == len(ast2.body)

        # Same declaration
        assert ast1.declarations[0].capacity == ast2.declarations[0].capacity


class TestAstValidation:
    """Tests that generated AST passes validation."""

    def test_generated_ast_validates(self) -> None:
        """Test that generated AST passes SLR-AST validation."""
        from pecos.slr.ast.validation import validate

        @guppy
        def circuit() -> tuple[bool, bool]:
            q0 = qubit()
            q1 = qubit()
            h(q0)
            cx(q0, q1)
            t(q1)
            return measure(q0), measure(q1)

        ast = guppy_to_ast(circuit)

        # Validate the AST
        result = validate(ast)
        assert result.valid, f"Validation failed: {result}"


class TestAstCodeGeneration:
    """Tests that generated AST can be used for code generation."""

    def test_generate_qasm(self) -> None:
        """Test generating QASM from converted AST."""
        from pecos.slr.ast.codegen import generate

        @guppy
        def circuit() -> tuple[bool, bool]:
            q0 = qubit()
            q1 = qubit()
            h(q0)
            cx(q0, q1)
            return measure(q0), measure(q1)

        ast = guppy_to_ast(circuit)
        qasm = generate(ast, "qasm")

        # Check QASM contains expected elements
        assert "OPENQASM" in qasm
        assert "qreg q[2]" in qasm
        assert "h q[0]" in qasm
        assert "cx q[0], q[1]" in qasm

    def test_generate_stim(self) -> None:
        """Test generating Stim from converted AST."""
        from pecos.slr.ast.codegen import generate

        @guppy
        def circuit() -> bool:
            q = qubit()
            h(q)
            s(q)  # Use S gate (Clifford) instead of T (non-Clifford)
            return measure(q)

        ast = guppy_to_ast(circuit)
        stim = generate(ast, "stim")

        # Check Stim contains expected elements
        assert "H" in stim
        assert "S" in stim
        assert "M" in stim


class TestConditionalCircuits:
    """Tests for conditional circuit conversion."""

    def test_conditional_circuit(self) -> None:
        """Test conversion of a conditional circuit."""

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

        # Should have 2 qubits
        assert ast.declarations[0].capacity == 2

        # Should have an IfStmt in the body
        if_stmts = [s for s in ast.body if isinstance(s, IfStmt)]
        assert len(if_stmts) == 1

        # The then body should have an X gate
        if_stmt = if_stmts[0]
        then_gates = [s for s in if_stmt.then_body if isinstance(s, GateOp)]
        assert len(then_gates) == 1
        assert then_gates[0].gate == GateKind.X


class TestLoopCircuits:
    """Tests for loop circuit conversion."""

    def test_simple_loop(self) -> None:
        """Test conversion of a simple loop circuit."""
        from pecos.slr.ast.nodes import WhileStmt

        @guppy
        def simple_loop() -> bool:
            q = qubit()
            h(q)
            count = 0
            while count < 3:
                x(q)
                count = count + 1
            return measure(q)

        ast = guppy_to_ast(simple_loop)

        # Should have 1 qubit
        alloc_decls = [d for d in ast.declarations if isinstance(d, AllocatorDecl)]
        assert len(alloc_decls) == 1
        assert alloc_decls[0].capacity == 1

        # Should have a WhileStmt in the body
        while_stmts = [s for s in ast.body if isinstance(s, WhileStmt)]
        assert len(while_stmts) == 1

        # The loop body should have an X gate
        while_stmt = while_stmts[0]
        body_gates = [s for s in while_stmt.body if isinstance(s, GateOp)]
        assert len(body_gates) == 1
        assert body_gates[0].gate == GateKind.X

        # X gate should target qubit 0
        assert body_gates[0].targets[0].index == 0


class TestNestedConditionalCircuits:
    """Tests for nested conditional circuit conversion."""

    def test_nested_conditional_circuit(self) -> None:
        """Test conversion of a nested conditional circuit."""
        from pecos.slr.ast.nodes import WhileStmt

        @guppy
        def nested_conditional() -> bool:
            q0 = qubit()
            q1 = qubit()
            h(q0)
            h(q1)
            r0 = measure(q0)
            r1 = measure(q1)

            q2 = qubit()
            if r0:
                if r1:
                    x(q2)
                else:
                    z(q2)

            return measure(q2)

        ast = guppy_to_ast(nested_conditional)

        # Should have 3 qubits
        alloc_decls = [d for d in ast.declarations if isinstance(d, AllocatorDecl)]
        assert len(alloc_decls) == 1
        assert alloc_decls[0].capacity == 3

        # Should have an outer IfStmt
        outer_if_stmts = [s for s in ast.body if isinstance(s, IfStmt)]
        assert len(outer_if_stmts) == 1

        # The outer then body should contain a nested IfStmt
        outer_if = outer_if_stmts[0]
        inner_if_stmts = [s for s in outer_if.then_body if isinstance(s, IfStmt)]
        assert len(inner_if_stmts) == 1

        # The inner IfStmt should have X in then and Z in else
        inner_if = inner_if_stmts[0]
        inner_then_gates = [s for s in inner_if.then_body if isinstance(s, GateOp)]
        assert len(inner_then_gates) == 1
        assert inner_then_gates[0].gate == GateKind.X

        assert inner_if.else_body is not None
        inner_else_gates = [s for s in inner_if.else_body if isinstance(s, GateOp)]
        assert len(inner_else_gates) == 1
        assert inner_else_gates[0].gate == GateKind.Z
