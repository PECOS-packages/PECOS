"""Test loop generation for register-wide operations."""

from pecos.qeclib import qubit
from pecos.qeclib.qubit.measures import Measure
from pecos.slr import Block, CReg, Main, QReg, SlrConverter


def test_consecutive_gate_applications() -> None:
    """Test that gates applied individually remain individual."""
    prog = Main(
        q := QReg("q", 5),
        c := CReg("c", 5),
        # Apply gates to consecutive elements individually
        qubit.H(q[0]),
        qubit.H(q[1]),
        qubit.H(q[2]),
        qubit.H(q[3]),
        qubit.H(q[4]),
        Measure(q) > c,
    )

    guppy_code = SlrConverter(prog).guppy()

    # Individual applications remain individual (not merged into loops)
    assert "quantum.h(q[0])" in guppy_code
    assert "quantum.h(q[1])" in guppy_code
    assert "quantum.h(q[2])" in guppy_code
    assert "quantum.h(q[3])" in guppy_code
    assert "quantum.h(q[4])" in guppy_code


def test_register_wide_generates_loop() -> None:
    """Test that register-wide operations generate loops."""
    prog = Main(
        q := QReg("q", 5),
        c := CReg("c", 5),
        # Apply gate to entire register
        qubit.H(q),  # This should generate a loop
        Measure(q) > c,
    )

    guppy_code = SlrConverter(prog).guppy()

    # Should generate a loop for register-wide operation
    assert "for i in range(0, 5):" in guppy_code
    assert "quantum.h(q[i])" in guppy_code


def test_mixed_individual_and_register_wide() -> None:
    """Test mixing individual and register-wide operations."""
    prog = Main(
        q := QReg("q", 4),
        c := CReg("c", 4),
        # Mix register-wide and individual operations
        qubit.H(q),  # Register-wide - should be a loop
        qubit.X(q[0]),  # Individual
        qubit.X(q[2]),  # Individual
        qubit.Z(q),  # Register-wide - should be a loop
        Measure(q) > c,
    )

    guppy_code = SlrConverter(prog).guppy()

    # Should have loops for H and Z
    assert "for i in range(0, 4):" in guppy_code
    assert "quantum.h(q[i])" in guppy_code
    assert "quantum.z(q[i])" in guppy_code

    # Should have individual X operations
    assert "quantum.x(q[0])" in guppy_code
    assert "quantum.x(q[2])" in guppy_code


def test_loop_in_function() -> None:
    """Test register-wide operations in a function block.

    Note: With Guppy's linear type system and @owned arrays, we can't use
    loops with array indexing (q[i]) because that would cause MoveOutOfSubscriptError.
    Instead, we unpack the array and apply operations to individual elements.
    This generates unrolled code, which is the correct behavior for @owned arrays.
    """

    class ApplyHadamards(Block):
        def __init__(self, q: QReg) -> None:
            super().__init__()
            self.q = q
            self.ops = [
                qubit.H(q),  # Apply to entire register
            ]

    prog = Main(
        q := QReg("q", 4),
        c := CReg("c", 4),
        ApplyHadamards(q),
        Measure(q) > c,
    )

    guppy_code = SlrConverter(prog).guppy()

    # Function should be created
    assert (
        "def test_loop_generation_apply_hadamards" in guppy_code
        or "def apply_hadamards" in guppy_code
    )

    # With @owned arrays, we unpack and unroll instead of using loops
    # Verify the function unpacks the array
    assert "q_0, q_1, q_2, q_3 = q" in guppy_code

    # Verify H is applied to all elements (unrolled)
    assert "quantum.h(q_0)" in guppy_code
    assert "quantum.h(q_1)" in guppy_code
    assert "quantum.h(q_2)" in guppy_code
    assert "quantum.h(q_3)" in guppy_code

    # Verify it compiles to HUGR (the real test of correctness)
    hugr = SlrConverter(prog).hugr()
    assert hugr is not None


def test_different_gates_separate_loops() -> None:
    """Test that different gates generate separate loops."""
    prog = Main(
        q := QReg("q", 3),
        c := CReg("c", 3),
        # Different gates on same register
        qubit.H(q),
        qubit.X(q),
        qubit.Y(q),
        qubit.Z(q),
        Measure(q) > c,
    )

    guppy_code = SlrConverter(prog).guppy()

    # Should have separate loops for each gate type
    loop_count = guppy_code.count("for i in range(0, 3):")
    assert loop_count == 4, f"Expected 4 loops, got {loop_count}"

    # Each gate should be in its own loop
    assert "quantum.h(q[i])" in guppy_code
    assert "quantum.x(q[i])" in guppy_code
    assert "quantum.y(q[i])" in guppy_code
    assert "quantum.z(q[i])" in guppy_code


def test_multiple_registers() -> None:
    """Test loop generation with multiple registers."""
    prog = Main(
        q1 := QReg("q1", 3),
        q2 := QReg("q2", 3),
        c := CReg("c", 6),
        # Apply gates to both registers
        qubit.H(q1),  # Should be a loop
        qubit.X(q2),  # Should be a loop
        Measure(q1) > c[0:3],
        Measure(q2) > c[3:6],
    )

    guppy_code = SlrConverter(prog).guppy()

    # Should generate loops for both operations
    assert "for i in range(0, 3):" in guppy_code
    assert "quantum.h(q1[i])" in guppy_code
    assert "quantum.x(q2[i])" in guppy_code
