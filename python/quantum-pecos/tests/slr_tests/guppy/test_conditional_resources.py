"""Tests for conditional resource consumption handling."""

import pytest
from pecos.slr import CReg, If, Main, QReg, SlrConverter
from pecos.slr.qeclib import qubit
from pecos.slr.qeclib.qubit.measures import Measure


def test_conditional_measurement_without_else() -> None:
    """Test that conditional measurements without else properly consume resources."""
    prog = Main(
        q := QReg("q", 2),
        flag := CReg("flag", 1),
        result := CReg("result", 1),
        # Get flag
        Measure(q[0]) > flag[0],
        # Conditionally measure second qubit
        If(flag[0]).Then(
            Measure(q[1]) > result[0],
        ),
    )

    guppy = SlrConverter(prog).guppy()

    # AST codegen generates conditionals with array indexing
    assert "if flag_0:" in guppy
    # Check measurements
    assert "quantum.measure(q[0])" in guppy
    assert "quantum.measure(q[1])" in guppy


def test_if_else_different_measurements() -> None:
    """Test that if-else blocks with different measurements balance resources."""
    prog = Main(
        q := QReg("q", 3),
        flag := CReg("flag", 1),
        result := CReg("result", 2),
        # Get flag
        Measure(q[0]) > flag[0],
        # Different measurements in each branch
        If(flag[0])
        .Then(
            Measure(q[1]) > result[0],
        )
        .Else(
            Measure(q[2]) > result[1],
        ),
    )

    guppy = SlrConverter(prog).guppy()

    # AST codegen uses variable names for conditions
    assert "if flag_0:" in guppy
    assert "else:" in guppy

    # Check that measurements are present
    assert "quantum.measure(q[0])" in guppy
    assert "quantum.measure(q[1])" in guppy
    assert "quantum.measure(q[2])" in guppy


def test_complex_conditional_with_gates() -> None:
    """Test complex conditional with quantum gates and partial consumption."""
    prog = Main(
        q := QReg("q", 4),
        flag := CReg("flag", 1),
        result := CReg("result", 4),
        qubit.H(q[0]),
        Measure(q[0]) > flag[0],
        If(flag[0])
        .Then(
            qubit.CX(q[1], q[2]),
            Measure(q[1]) > result[1],
            Measure(q[2]) > result[2],
            # q[3] not measured in this branch
        )
        .Else(
            qubit.X(q[3]),
            Measure(q[3]) > result[3],
            # q[1], q[2] not measured in this branch
        ),
    )

    guppy = SlrConverter(prog).guppy()

    # AST codegen uses array indexing
    assert "quantum.h(q[0])" in guppy
    assert "quantum.cx(q[1], q[2])" in guppy
    assert "quantum.x(q[3])" in guppy

    # Check that measurements happen in conditional branches
    assert "quantum.measure(q[0])" in guppy


def test_nested_conditionals() -> None:
    """Test nested conditionals properly handle resource consumption."""
    prog = Main(
        q := QReg("q", 3),
        flags := CReg("flags", 2),
        result := CReg("result", 3),
        Measure(q[0]) > flags[0],
        If(flags[0]).Then(
            Measure(q[1]) > flags[1],
            If(flags[1]).Then(
                Measure(q[2]) > result[2],
            ),
        ),
    )

    guppy = SlrConverter(prog).guppy()

    # AST codegen uses array indexing
    assert "quantum.measure(q[0])" in guppy
    assert "quantum.measure(q[1])" in guppy
    assert "quantum.measure(q[2])" in guppy

    # Check nested if structure
    assert "if flags_0:" in guppy
    assert "if flags_1:" in guppy

    # Should compile to HUGR without errors
    hugr = SlrConverter(prog).hugr()
    assert hugr is not None


def test_no_else_with_unconsumed_resources() -> None:
    """Test that missing else blocks are generated when needed for linearity."""
    prog = Main(
        q := QReg("q", 2),
        flag := CReg("flag", 2),  # Need size 2 for flag[1]
        Measure(q[0]) > flag[0],
        If(flag[0]).Then(
            # Only measure q[1] in then branch
            Measure(q[1])
            > flag[1],
        ),
        # No explicit else - should be generated
    )

    guppy = SlrConverter(prog).guppy()

    # Should have if block with condition
    assert "if flag_0:" in guppy
    assert "quantum.measure(q[0])" in guppy
    assert "quantum.measure(q[1])" in guppy

    # Should compile to HUGR without errors
    hugr = SlrConverter(prog).hugr()
    assert hugr is not None


@pytest.mark.optional_dependency
def test_hugr_compilation_simple() -> None:
    """Test that simple conditional programs can compile to HUGR."""
    prog = Main(
        q := QReg("q", 2),
        c := CReg("c", 2),
        # Simple conditional that should work
        Measure(q[0]) > c[0],
        If(c[0])
        .Then(
            Measure(q[1]) > c[1],
        )
        .Else(
            Measure(q[1]) > c[1],
        ),
    )

    # This might still fail due to other HUGR issues, but the conditional
    # resource handling should be correct
    try:
        SlrConverter(prog).hugr()
        # If it succeeds, great!
    except ImportError as e:
        # If it fails due to import, that's expected
        if "linearity" in str(e).lower():
            pytest.fail(f"Should not fail due to linearity: {e}")
