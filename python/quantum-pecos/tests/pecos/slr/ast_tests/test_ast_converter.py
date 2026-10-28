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

"""Tests for SLR to AST converter."""

import pytest
import stim
from pecos.slr import CReg, If, Main, QReg, Repeat, SlrConverter, rad
from pecos.slr.ast import (
    AllocatorDecl,
    BinaryOp,
    GateKind,
    GateOp,
    IfStmt,
    MeasureOp,
    PrepareOp,
    Program,
    RegisterDecl,
    RepeatStmt,
    SlotRef,
    SlrToAst,
    slr_to_ast,
)
from pecos.slr.qalloc import QAlloc
from pecos.slr.qeclib import qubit as qb


class TestSlrToAstBasic:
    """Basic conversion tests."""

    def test_empty_program(self) -> None:
        """Empty program converts to Program with no body."""
        prog = Main()

        ast = slr_to_ast(prog)

        assert isinstance(ast, Program)
        assert ast.name == "Main"
        assert ast.body == ()

    def test_program_with_qreg(self) -> None:
        """QReg converts to AllocatorDecl."""
        prog = Main(
            _q := QReg("q", 2),
        )

        ast = slr_to_ast(prog)

        assert len(ast.declarations) == 1
        decl = ast.declarations[0]
        assert isinstance(decl, AllocatorDecl)
        assert decl.name == "q"
        assert decl.capacity == 2

    def test_program_with_creg(self) -> None:
        """CReg converts to RegisterDecl."""
        prog = Main(
            _c := CReg("c", 3),
        )

        ast = slr_to_ast(prog)

        assert len(ast.declarations) == 1
        decl = ast.declarations[0]
        assert isinstance(decl, RegisterDecl)
        assert decl.name == "c"
        assert decl.size == 3

    def test_program_with_both_regs(self) -> None:
        """Program with QReg and CReg has both declarations."""
        prog = Main(
            _q := QReg("q", 2),
            _c := CReg("c", 2),
        )

        ast = slr_to_ast(prog)

        assert len(ast.declarations) == 2
        names = [d.name for d in ast.declarations]
        assert "q" in names
        assert "c" in names


class TestSlrToAstGates:
    """Gate conversion tests."""

    def test_single_qubit_gate(self) -> None:
        """Single-qubit gate converts to GateOp with correct kind and target."""
        prog = Main(
            q := QReg("q", 1),
            qb.H(q[0]),
        )

        ast = slr_to_ast(prog)

        assert len(ast.body) == 1
        gate = ast.body[0]
        assert isinstance(gate, GateOp)
        assert gate.gate == GateKind.H
        assert len(gate.targets) == 1
        assert gate.targets[0].allocator == "q"
        assert gate.targets[0].index == 0

    def test_two_qubit_gate(self) -> None:
        """Two-qubit gate converts to GateOp with two targets."""
        prog = Main(
            q := QReg("q", 2),
            qb.CX(q[0], q[1]),
        )

        ast = slr_to_ast(prog)

        assert len(ast.body) == 1
        gate = ast.body[0]
        assert isinstance(gate, GateOp)
        assert gate.gate == GateKind.CX
        assert len(gate.targets) == 2
        assert gate.targets[0].index == 0
        assert gate.targets[1].index == 1

    def test_multiple_gates(self) -> None:
        """Multiple gates convert to multiple GateOps in sequence."""
        prog = Main(
            q := QReg("q", 2),
            qb.H(q[0]),
            qb.X(q[1]),
            qb.CZ(q[0], q[1]),
        )

        ast = slr_to_ast(prog)

        assert len(ast.body) == 3
        assert ast.body[0].gate == GateKind.H
        assert ast.body[1].gate == GateKind.X
        assert ast.body[2].gate == GateKind.CZ


class TestMultiQubitArguments:
    """Multi-qubit applications preserve the user's target grouping."""

    @pytest.mark.parametrize("gate_name", ["CX", "CY", "CZ"])
    @pytest.mark.parametrize("tuple_form", [False, True])
    @pytest.mark.parametrize("output", ["ast", "qasm", "stim"])
    def test_register_pairing(self, gate_name: str, tuple_form: bool, output: str) -> None:
        """Whole registers apply the gate to matching indices in order."""
        q, r = QReg("q", 3), QReg("r", 3)
        gate = getattr(qb, gate_name)
        op = gate((q, r)) if tuple_form else gate(q, r)
        prog = Main(q, r, op)

        if output == "ast":
            assert slr_to_ast(prog).body == tuple(
                GateOp(
                    gate=GateKind[gate_name],
                    targets=(SlotRef(allocator="q", index=i), SlotRef(allocator="r", index=i)),
                )
                for i in range(3)
            )
        elif output == "qasm":
            lines = SlrConverter(prog).qasm(skip_headers=True).splitlines()
            assert [line for line in lines if line.startswith(f"{gate_name.lower()} ")] == [
                f"{gate_name.lower()} q[{i}], r[{i}];" for i in range(3)
            ]
        else:
            expected = stim.Circuit("\n".join(f"{gate_name} {i} {i + 3}" for i in range(3)))
            assert SlrConverter(prog).stim().to_tableau() == expected.to_tableau()

    @pytest.mark.parametrize("tuple_form", [False, True])
    def test_register_size_mismatch(self, tuple_form: bool) -> None:
        """Mismatched sizes report the gate and every register's size."""
        q, r = QReg("q", 2), QReg("r", 3)
        op = qb.CX((q, r)) if tuple_form else qb.CX(q, r)
        with pytest.raises(ValueError, match=r"CX.*q.*2.*r.*3"):
            slr_to_ast(Main(q, r, op))

    @pytest.mark.parametrize(
        "shape",
        [
            "mixed",
            "mixed_reverse",
            "list",
            "slice",
            "lists",
            "slices",
            "too_many",
            "too_few",
            "empty",
            "one_register",
            "three_registers",
            "stray_string",
            "classical",
        ],
    )
    @pytest.mark.parametrize("tuple_form", [False, True])
    def test_invalid_shapes(self, shape: str, tuple_form: bool) -> None:
        """Unsupported argument shapes fail without dropping or inventing pairs."""
        q, r, s = QReg("q", 2), QReg("r", 2), QReg("s", 2)
        c = CReg("c", 2)
        shapes = {
            "mixed": (q[0], r),
            "mixed_reverse": (q, r[0]),
            "list": ([q[0], r[0]],),
            "slice": (q[:],),
            "lists": ([q[0]], [r[0]]),
            "slices": (q[:1], r[:1]),
            "too_many": (q[0], q[1], r[0], r[1]),
            "too_few": (q[0],),
            "empty": (),
            "one_register": (q,),
            "three_registers": (q, r, s),
            "stray_string": (q[0], r[0], "ignored"),
            "classical": (c[0], q[0]),
        }
        args = (shapes[shape],) if tuple_form else shapes[shape]
        with pytest.raises(ValueError, match=r"CX.*arguments") as exc:
            slr_to_ast(Main(q, r, s, c, qb.CX(*args)))
        assert repr(args) in str(exc.value)

    @pytest.mark.parametrize("tuple_first", [False, True])
    def test_mixed_tuple_and_qubit(self, tuple_first: bool) -> None:
        """The tuple form requires every application to be a tuple."""
        q = QReg("q", 3)
        args = ((q[0], q[1]), q[2]) if tuple_first else (q[2], (q[0], q[1]))
        with pytest.raises(ValueError, match=r"CX.*arguments") as exc:
            slr_to_ast(Main(q, qb.CX(*args)))
        assert repr(args) in str(exc.value)

    @pytest.mark.parametrize("tuple_form", [False, True])
    def test_empty_registers(self, tuple_form: bool) -> None:
        """Equal empty registers have no indices to apply a gate to."""
        q, r = QReg("q", 0), QReg("r", 0)
        op = qb.CX((q, r)) if tuple_form else qb.CX(q, r)
        assert slr_to_ast(Main(q, r, op)).body == ()

    def test_register_and_explicit_tuple_applications(self) -> None:
        """Each tuple is expanded independently while preserving application order."""
        q, r = QReg("q", 2), QReg("r", 2)
        ast = slr_to_ast(Main(q, r, qb.CX((q, r), (r[1], q[0]))))
        assert [gate.targets for gate in ast.body] == [
            (SlotRef(allocator="q", index=0), SlotRef(allocator="r", index=0)),
            (SlotRef(allocator="q", index=1), SlotRef(allocator="r", index=1)),
            (SlotRef(allocator="r", index=1), SlotRef(allocator="q", index=0)),
        ]

    def test_parameterized_register_pairing(self) -> None:
        """Each expanded operation retains its angle parameter."""
        q, r = QReg("q", 2), QReg("r", 2)
        ast = slr_to_ast(Main(q, r, qb.RZZ(rad(0.5), q, r)))
        expected = slr_to_ast(Main(q, r, *(qb.RZZ(rad(0.5), q[i], r[i]) for i in range(2))))
        assert ast.body == expected.body

    @pytest.mark.parametrize("gate_kind", [gate for gate in GateKind if gate.arity > 1])
    @pytest.mark.parametrize("count", [0, 1, 3, 4])
    def test_gate_op_rejects_wrong_arity(self, gate_kind: GateKind, count: int) -> None:
        """Every multi-qubit node enforces arity before any backend sees it."""
        targets = tuple(SlotRef(allocator="q", index=i) for i in range(count))
        with pytest.raises(ValueError, match=rf"{gate_kind.name}.*2.*{count}"):
            GateOp(gate=gate_kind, targets=targets)

    @pytest.mark.parametrize("tuple_form", [False, True])
    def test_explicit_pair_compatibility(self, tuple_form: bool) -> None:
        """Explicit tuples and a single pair keep their existing AST representation."""
        q, r = QReg("q", 2), QReg("r", 2)
        op = qb.CX((q[0], r[0]), (q[1], r[1])) if tuple_form else qb.CX(q[0], r[0])
        assert slr_to_ast(Main(q, r, op)).body == tuple(
            GateOp(
                gate=GateKind.CX,
                targets=(SlotRef(allocator="q", index=i), SlotRef(allocator="r", index=i)),
            )
            for i in range(2 if tuple_form else 1)
        )

    def test_allocator_ref_compatibility(self) -> None:
        """QAlloc's individual QubitRefs remain valid gate arguments."""
        q = QAlloc(2, name="q")
        assert slr_to_ast(Main(q, qb.CX(q[0], q[1]))).body == (
            GateOp(
                gate=GateKind.CX,
                targets=(SlotRef(allocator="q", index=0), SlotRef(allocator="q", index=1)),
            ),
        )


class TestSlrToAstPrepMeasure:
    """PZ and Measure conversion tests."""

    def test_prep_operation(self) -> None:
        """PZ converts to PrepareOp with correct allocator and slots."""
        prog = Main(
            q := QReg("q", 2),
            qb.PZ(q[0]),
        )

        ast = slr_to_ast(prog)

        assert len(ast.body) == 1
        prep = ast.body[0]
        assert isinstance(prep, PrepareOp)
        assert prep.allocator == "q"
        assert prep.slots == (0,)

    def test_measure_operation(self) -> None:
        """Measure converts to MeasureOp with targets and results."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            qb.Measure(q[0]) > c[0],
        )

        ast = slr_to_ast(prog)

        assert len(ast.body) == 1
        measure = ast.body[0]
        assert isinstance(measure, MeasureOp)
        assert len(measure.targets) == 1
        assert measure.targets[0].allocator == "q"
        assert measure.targets[0].index == 0
        assert len(measure.results) == 1
        assert measure.results[0].register == "c"
        assert measure.results[0].index == 0


class TestSlrToAstControlFlow:
    """Control flow conversion tests."""

    def test_if_statement(self) -> None:
        """If statement converts to IfStmt with then_body."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            If(c[0] == 1).Then(
                qb.H(q[0]),
            ),
        )

        ast = slr_to_ast(prog)

        assert len(ast.body) == 1
        if_stmt = ast.body[0]
        assert isinstance(if_stmt, IfStmt)
        assert len(if_stmt.then_body) == 1
        assert isinstance(if_stmt.then_body[0], GateOp)

    def test_if_else_statement(self) -> None:
        """If-else statement converts to IfStmt with both branches."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            If(c[0] == 1)
            .Then(
                qb.H(q[0]),
            )
            .Else(
                qb.X(q[0]),
            ),
        )

        ast = slr_to_ast(prog)

        if_stmt = ast.body[0]
        assert isinstance(if_stmt, IfStmt)
        assert len(if_stmt.then_body) == 1
        assert len(if_stmt.else_body) == 1
        assert if_stmt.then_body[0].gate == GateKind.H
        assert if_stmt.else_body[0].gate == GateKind.X

    def test_condition_expression(self) -> None:
        """Condition converts to BinaryExpr with correct operator."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            If(c[0] == 1).Then(
                qb.H(q[0]),
            ),
        )

        ast = slr_to_ast(prog)

        if_stmt = ast.body[0]
        # Check the condition is a binary expression
        from pecos.slr.ast import BinaryExpr

        assert isinstance(if_stmt.condition, BinaryExpr)
        assert if_stmt.condition.op == BinaryOp.EQ

    def test_repeat_statement(self) -> None:
        """Repeat statement converts to RepeatStmt with count and body."""
        prog = Main(
            q := QReg("q", 1),
            Repeat(cond=5).block(
                qb.H(q[0]),
            ),
        )

        ast = slr_to_ast(prog)

        assert len(ast.body) == 1
        repeat = ast.body[0]
        assert isinstance(repeat, RepeatStmt)
        assert repeat.count == 5
        assert len(repeat.body) == 1


class TestSlrToAstConverter:
    """Tests for SlrToAst converter class."""

    def test_converter_reusable(self) -> None:
        """Converter can be reused for multiple programs."""
        converter = SlrToAst()

        prog1 = Main(
            q := QReg("q", 1),
            qb.H(q[0]),
        )

        prog2 = Main(
            r := QReg("r", 2),
            qb.X(r[0]),
            qb.X(r[1]),
        )

        ast1 = converter.convert(prog1)
        ast2 = converter.convert(prog2)

        assert len(ast1.body) == 1
        assert len(ast2.body) == 2
        assert ast1.declarations[0].name == "q"
        assert ast2.declarations[0].name == "r"


class TestSlrToAstQEC:
    """Tests for QEC-related patterns."""

    def test_syndrome_extraction_pattern(self) -> None:
        """Syndrome extraction converts with correct operations."""
        prog = Main(
            data := QReg("data", 2),
            ancilla := QReg("ancilla", 1),
            c := CReg("c", 1),
            # Initialize
            qb.PZ(data[0]),
            qb.PZ(data[1]),
            qb.PZ(ancilla[0]),
            # Syndrome extraction
            qb.CX(data[0], ancilla[0]),
            qb.CX(data[1], ancilla[0]),
            qb.Measure(ancilla[0]) > c[0],
        )

        ast = slr_to_ast(prog)

        # Check declarations
        assert len(ast.declarations) == 3
        alloc_names = [d.name for d in ast.declarations if isinstance(d, AllocatorDecl)]
        assert "data" in alloc_names
        assert "ancilla" in alloc_names

        # Check body operations
        assert len(ast.body) == 6
        preps = [op for op in ast.body if isinstance(op, PrepareOp)]
        gates = [op for op in ast.body if isinstance(op, GateOp)]
        measures = [op for op in ast.body if isinstance(op, MeasureOp)]

        assert len(preps) == 3
        assert len(gates) == 2
        assert len(measures) == 1

    def test_round_trip_preserves_structure(self) -> None:
        """Test that conversion preserves the logical structure."""
        prog = Main(
            q := QReg("q", 3),
            c := CReg("c", 1),
            qb.PZ(q[0]),
            qb.PZ(q[1]),
            qb.PZ(q[2]),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
            qb.CX(q[1], q[2]),
            If(c[0] == 1).Then(
                qb.X(q[0]),
            ),
        )

        ast = slr_to_ast(prog)

        # Verify structure
        assert ast.name == "Main"
        assert len(ast.declarations) == 2

        # Count operation types
        gate_count = sum(1 for op in ast.body if isinstance(op, GateOp))
        prep_count = sum(1 for op in ast.body if isinstance(op, PrepareOp))
        if_count = sum(1 for op in ast.body if isinstance(op, IfStmt))

        assert gate_count == 3  # H, CX, CX
        assert prep_count == 3  # 3 Preps
        assert if_count == 1  # 1 If


class TestSlrToAstQAlloc:
    """Tests for QAlloc (hierarchical allocator) conversion."""

    def test_single_qalloc(self) -> None:
        """Test conversion of a single QAlloc."""
        prog = Main(
            base := QAlloc(4, name="base"),
            base.prepare_all(),
            qb.H(base[0]),
        )

        ast = slr_to_ast(prog)

        assert len(ast.declarations) == 1
        decl = ast.declarations[0]
        assert isinstance(decl, AllocatorDecl)
        assert decl.name == "base"
        assert decl.capacity == 4
        assert decl.parent is None

    def test_hierarchical_qalloc(self) -> None:
        """Test conversion of hierarchical QAllocs (parent-child)."""
        prog = Main(
            base := QAlloc(10, name="base"),
            data := base.child(4, name="data"),
            ancilla := base.child(4, name="ancilla"),
            data.prepare_all(),
            ancilla.prepare_all(),
            qb.H(data[0]),
            qb.CX(data[0], ancilla[0]),
        )

        ast = slr_to_ast(prog)

        # Should have 3 declarations
        assert len(ast.declarations) == 3
        decl_by_name = {d.name: d for d in ast.declarations}

        # Check base allocator
        assert "base" in decl_by_name
        assert decl_by_name["base"].capacity == 10
        assert decl_by_name["base"].parent is None

        # Check child allocators
        assert "data" in decl_by_name
        assert decl_by_name["data"].capacity == 4
        assert decl_by_name["data"].parent == "base"

        assert "ancilla" in decl_by_name
        assert decl_by_name["ancilla"].capacity == 4
        assert decl_by_name["ancilla"].parent == "base"

    def test_qalloc_gates_use_correct_allocator_names(self) -> None:
        """Test that gates on QAlloc qubits reference correct allocator names."""
        prog = Main(
            base := QAlloc(10, name="base"),
            data := base.child(4, name="data"),
            data.prepare_all(),
            qb.H(data[0]),
            qb.CX(data[0], data[1]),
        )

        ast = slr_to_ast(prog)

        # Check that gate targets reference "data" allocator
        assert len(ast.body) == 2
        h_gate = ast.body[0]
        cx_gate = ast.body[1]

        assert h_gate.gate == GateKind.H
        assert h_gate.targets[0].allocator == "data"
        assert h_gate.targets[0].index == 0

        assert cx_gate.gate == GateKind.CX
        assert cx_gate.targets[0].allocator == "data"
        assert cx_gate.targets[1].allocator == "data"

    def test_qalloc_no_duplicates(self) -> None:
        """Test that QAllocs appearing in both vars and ops aren't duplicated."""
        # When using walrus operator, QAllocs end up in ops
        prog = Main(
            base := QAlloc(4, name="base"),
            base.prepare_all(),
            qb.H(base[0]),
        )

        ast = slr_to_ast(prog)

        # Should have exactly 1 declaration, not duplicated
        assert len(ast.declarations) == 1
        names = [d.name for d in ast.declarations]
        assert names.count("base") == 1
