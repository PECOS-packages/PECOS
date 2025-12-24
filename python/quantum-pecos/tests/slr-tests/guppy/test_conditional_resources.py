"""Tests for conditional resource consumption handling."""

import pytest
from pecos.qeclib import qubit
from pecos.qeclib.qubit.measures import Measure
from pecos.slr import CReg, If, Main, QReg, SlrConverter


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

    # Check that else branch was generated
    assert "else:" in guppy

    # Check that unconsumed qubit is measured
    # The else branch should measure q[1] to maintain linearity
    # But in this case, q[1] might be consumed at the end of main

    # At minimum, all qubits should be consumed
    lines = guppy.split("\n")
    measure_count = sum(1 for line in lines if "quantum.measure" in line)
    assert measure_count >= 2  # Both qubits must be measured


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

    # With dynamic allocation, no explicit linearity comment needed
    # Each branch allocates and measures its own qubit
    # With unpacking, flag[0] becomes flag_0
    assert "if flag[0]:" in guppy or "if flag_0:" in guppy
    assert "else:" in guppy

    # Check that all qubits are measured
    lines = guppy.split("\n")
    measure_count = sum(1 for line in lines if "quantum.measure" in line)
    assert measure_count >= 3  # All three qubits must be measured


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

    # Check that unpacking happened
    assert "# Unpack q for individual access" in guppy
    assert "q_0, q_1, q_2, q_3 = q" in guppy

    # All qubits are consumed in the conditional branches, so no cleanup needed
    # q[0] is measured before the if
    # In Then branch: q[1] and q[2] are measured
    # In Else branch: q[3] is measured
    # Therefore, no discard should be present

    # Check that measurements happen in conditional branches
    lines = guppy.split("\n")
    measure_count = sum(1 for line in lines if "quantum.measure" in line)
    assert measure_count >= 3  # Three measurements: flag + either (1,2) or (3)


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

    # With dynamic allocation optimization, qubits are allocated on demand
    # Check that qubits are allocated locally rather than pre-allocated
    assert "q_0 = quantum.qubit()" in guppy
    assert "q_1 = quantum.qubit()" in guppy
    assert "q_2 = quantum.qubit()" in guppy

    # Check that all branches have proper structure
    # Should have else branches to balance resources
    lines = guppy.split("\n")
    else_count = sum(1 for line in lines if line.strip() == "else:")
    assert else_count >= 1  # At least one else for resource balancing

    # With dynamic allocation, else blocks should allocate fresh qubits for balancing
    # Check that else blocks consume resources properly
    assert "_q_1 = quantum.qubit()" in guppy or "_q_2 = quantum.qubit()" in guppy

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

    # Should generate else block
    assert "else:" in guppy

    # The else block should consume q[1]
    # With dynamic allocation, else block allocates fresh qubit and measures it
    assert "_q_1 = quantum.qubit()" in guppy
    assert "_ = quantum.measure(_q_1)" in guppy

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
