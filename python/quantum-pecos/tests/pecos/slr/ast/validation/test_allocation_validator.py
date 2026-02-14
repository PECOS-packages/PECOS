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

"""Tests for allocation validator."""

from pecos.slr import CReg, If, Main, QReg, Repeat
from pecos.slr.ast import slr_to_ast
from pecos.slr.ast.nodes import (
    AllocatorDecl,
    GateKind,
    GateOp,
    PrepareOp,
    Program,
    SlotRef,
)
from pecos.slr.ast.validation import AllocationValidator, validate_allocations
from pecos.slr.qeclib import qubit as qb


class TestAllocationValidatorValid:
    """Tests for valid allocations."""

    def test_valid_single_allocator(self):
        """Single valid allocator."""
        prog = Main(
            q := QReg("q", 2),
            qb.H(q[0]),
            qb.X(q[1]),
        )

        ast = slr_to_ast(prog)
        result = validate_allocations(ast)

        assert result.valid is True
        assert len(result.errors) == 0

    def test_valid_multiple_allocators(self):
        """Multiple valid allocators."""
        prog = Main(
            q := QReg("q", 2),
            a := QReg("a", 3),
            qb.H(q[0]),
            qb.X(a[0]),
            qb.CX(q[1], a[1]),
        )

        ast = slr_to_ast(prog)
        result = validate_allocations(ast)

        assert result.valid is True


class TestAllocationValidatorErrors:
    """Tests for allocation errors."""

    def test_duplicate_allocator_names(self):
        """Duplicate allocator names."""
        prog = Program(
            name="test",
            allocator=AllocatorDecl(name="q", capacity=2),
            declarations=(AllocatorDecl(name="q", capacity=3),),  # Duplicate
            body=(),
        )

        result = validate_allocations(prog)

        assert result.valid is False
        assert "Duplicate allocator name" in result.errors[0].message
        assert result.errors[0].code == "E301"

    def test_undeclared_allocator_reference(self):
        """Reference to undeclared allocator."""
        prog = Program(
            name="test",
            allocator=AllocatorDecl(name="q", capacity=2),
            body=(
                GateOp(
                    gate=GateKind.H,
                    targets=(SlotRef(allocator="unknown", index=0),),
                ),
            ),
        )

        result = validate_allocations(prog)

        assert result.valid is False
        assert "undeclared allocator" in result.errors[0].message
        assert result.errors[0].code == "E305"

    def test_zero_capacity(self):
        """Allocator with zero capacity."""
        prog = Program(
            name="test",
            allocator=AllocatorDecl(name="q", capacity=0),
            body=(),
        )

        result = validate_allocations(prog)

        assert result.valid is False
        assert "non-positive capacity" in result.errors[0].message
        assert result.errors[0].code == "E303"

    def test_negative_capacity(self):
        """Allocator with negative capacity."""
        prog = Program(
            name="test",
            allocator=AllocatorDecl(name="q", capacity=-5),
            body=(),
        )

        result = validate_allocations(prog)

        assert result.valid is False
        assert "non-positive capacity" in result.errors[0].message


class TestAllocationValidatorParentHierarchy:
    """Tests for parent allocator hierarchy."""

    def test_unknown_parent(self):
        """Reference to unknown parent allocator."""
        prog = Program(
            name="test",
            allocator=AllocatorDecl(name="q", capacity=5),
            declarations=(
                AllocatorDecl(name="child", capacity=2, parent="nonexistent"),
            ),
            body=(),
        )

        result = validate_allocations(prog)

        assert result.valid is False
        assert "unknown parent" in result.errors[0].message
        assert result.errors[0].code == "E302"

    def test_valid_parent_reference(self):
        """Valid parent allocator reference."""
        prog = Program(
            name="test",
            allocator=AllocatorDecl(name="parent", capacity=5),
            declarations=(
                AllocatorDecl(name="child", capacity=2, parent="parent"),
            ),
            body=(
                GateOp(gate=GateKind.H, targets=(SlotRef(allocator="child", index=0),)),
            ),
        )

        result = validate_allocations(prog)

        assert result.valid is True

    def test_parent_cycle_detection(self):
        """Detect cycles in parent hierarchy."""
        prog = Program(
            name="test",
            declarations=(
                AllocatorDecl(name="a", capacity=2, parent="b"),
                AllocatorDecl(name="b", capacity=2, parent="a"),  # Cycle
            ),
            body=(),
        )

        result = validate_allocations(prog)

        assert result.valid is False
        assert "Cycle detected" in result.errors[0].message
        assert result.errors[0].code == "E304"


class TestAllocationValidatorWarnings:
    """Tests for allocation warnings."""

    def test_unused_allocator_warning(self):
        """Unused allocator generates warning."""
        prog = Program(
            name="test",
            allocator=AllocatorDecl(name="used", capacity=2),
            declarations=(AllocatorDecl(name="unused", capacity=3),),
            body=(
                GateOp(gate=GateKind.H, targets=(SlotRef(allocator="used", index=0),)),
            ),
        )

        result = validate_allocations(prog)

        assert result.valid is True
        assert len(result.warnings) == 1
        assert "never used" in result.warnings[0].message


class TestAllocationValidatorControlFlow:
    """Allocation validation in control flow."""

    def test_allocation_inside_if(self):
        """Allocator references inside if statements."""
        prog = Main(
            q := QReg("q", 2),
            c := CReg("c", 1),
            If(c[0] == 1).Then(
                qb.H(q[0]),
            ),
        )

        ast = slr_to_ast(prog)
        result = validate_allocations(ast)

        assert result.valid is True

    def test_allocation_inside_repeat(self):
        """Allocator references inside repeat loops."""
        prog = Main(
            q := QReg("q", 1),
            Repeat(cond=3).block(
                qb.X(q[0]),
            ),
        )

        ast = slr_to_ast(prog)
        result = validate_allocations(ast)

        assert result.valid is True


class TestAllocationValidatorPrepare:
    """Tests for prepare operation validation."""

    def test_valid_prepare(self):
        """Valid prepare references declared allocator."""
        prog = Program(
            name="test",
            allocator=AllocatorDecl(name="q", capacity=3),
            body=(PrepareOp(allocator="q", slots=(0, 1, 2)),),
        )

        result = validate_allocations(prog)

        assert result.valid is True

    def test_prepare_undeclared_allocator(self):
        """Prepare references undeclared allocator."""
        prog = Program(
            name="test",
            allocator=AllocatorDecl(name="q", capacity=3),
            body=(PrepareOp(allocator="unknown", slots=(0,)),),
        )

        result = validate_allocations(prog)

        assert result.valid is False
        assert "undeclared allocator" in result.errors[0].message


class TestAllocationValidatorClass:
    """Tests for AllocationValidator class."""

    def test_validator_reuse(self):
        """Validator can be reused."""
        validator = AllocationValidator()

        prog1 = Main(q := QReg("q", 1), qb.H(q[0]))
        prog2 = Program(
            name="test",
            allocator=AllocatorDecl(name="q", capacity=2),
            body=(GateOp(gate=GateKind.H, targets=(SlotRef(allocator="unknown", index=0),)),),
        )

        ast1 = slr_to_ast(prog1)

        result1 = validator.validate(ast1)
        result2 = validator.validate(prog2)

        assert result1.valid is True
        assert result2.valid is False

    def test_passes_applied(self):
        """Pass name is tracked."""
        prog = Main(q := QReg("q", 1), qb.H(q[0]))
        ast = slr_to_ast(prog)

        result = validate_allocations(ast)

        assert "allocation_validator" in result.passes_applied
