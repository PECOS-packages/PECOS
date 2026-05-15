"""Test loop generation for register-wide operations."""

from pecos.slr import Block, CReg, Main, QReg, SlrConverter
from pecos.slr.qeclib import qubit
from pecos.slr.qeclib.qubit.measures import Measure


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
    """Test that register-wide operations are handled (loop or expanded)."""
    prog = Main(
        q := QReg("q", 5),
        c := CReg("c", 5),
        # Apply gate to entire register
        qubit.H(q),  # May generate loop or expand to individual
        Measure(q) > c,
    )

    guppy_code = SlrConverter(prog).guppy()

    # AST codegen expands register-wide ops to individual operations
    # Check that all qubits have H applied
    h_count = guppy_code.count("quantum.h")
    assert h_count >= 5, f"Expected at least 5 H gates, got {h_count}"


def test_mixed_individual_and_register_wide() -> None:
    """Test mixing individual and register-wide operations."""
    prog = Main(
        q := QReg("q", 4),
        c := CReg("c", 4),
        # Mix register-wide and individual operations
        qubit.H(q),  # Register-wide
        qubit.X(q[0]),  # Individual
        qubit.X(q[2]),  # Individual
        qubit.Z(q),  # Register-wide
        Measure(q) > c,
    )

    guppy_code = SlrConverter(prog).guppy()

    # Should have H and Z applied to all qubits
    h_count = guppy_code.count("quantum.h")
    z_count = guppy_code.count("quantum.z")
    assert h_count >= 4, f"Expected at least 4 H gates, got {h_count}"
    assert z_count >= 4, f"Expected at least 4 Z gates, got {z_count}"

    # Should have individual X operations
    assert "quantum.x(q[0])" in guppy_code
    assert "quantum.x(q[2])" in guppy_code


def test_loop_in_function() -> None:
    """Test register-wide operations in a function block.

    Note: AST codegen flattens blocks into main and expands register-wide
    operations to individual operations.
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

    # AST codegen flattens blocks and expands register-wide ops
    # Verify H is applied to all elements
    h_count = guppy_code.count("quantum.h")
    assert h_count >= 4, f"Expected at least 4 H gates, got {h_count}"

    # Verify measurements
    assert "quantum.measure(q[0])" in guppy_code
    assert "quantum.measure(q[1])" in guppy_code

    # Verify it compiles to HUGR (the real test of correctness)
    hugr = SlrConverter(prog).hugr()
    assert hugr is not None


def test_different_gates_separate_loops() -> None:
    """Test that different gates are applied to all qubits."""
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

    # Each gate type should be applied 3 times (once per qubit)
    h_count = guppy_code.count("quantum.h")
    x_count = guppy_code.count("quantum.x")
    y_count = guppy_code.count("quantum.y")
    z_count = guppy_code.count("quantum.z")

    assert h_count >= 3, f"Expected at least 3 H gates, got {h_count}"
    assert x_count >= 3, f"Expected at least 3 X gates, got {x_count}"
    assert y_count >= 3, f"Expected at least 3 Y gates, got {y_count}"
    assert z_count >= 3, f"Expected at least 3 Z gates, got {z_count}"


def test_multiple_registers() -> None:
    """Test operations on multiple registers."""
    prog = Main(
        q1 := QReg("q1", 3),
        q2 := QReg("q2", 3),
        c := CReg("c", 6),
        # Apply gates to both registers
        qubit.H(q1),
        qubit.X(q2),
        Measure(q1) > c[0:3],
        Measure(q2) > c[3:6],
    )

    guppy_code = SlrConverter(prog).guppy()

    # Should have H applied to q1 and X applied to q2
    assert "quantum.h(q1[0])" in guppy_code
    assert "quantum.h(q1[1])" in guppy_code
    assert "quantum.h(q1[2])" in guppy_code
    assert "quantum.x(q2[0])" in guppy_code
    assert "quantum.x(q2[1])" in guppy_code
    assert "quantum.x(q2[2])" in guppy_code
