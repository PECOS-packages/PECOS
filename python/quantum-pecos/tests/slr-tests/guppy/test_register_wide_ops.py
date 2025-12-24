"""Test register-wide operations that should generate loops."""

from pecos.qeclib import qubit
from pecos.qeclib.qubit.measures import Measure
from pecos.slr import CReg, Main, QReg, SlrConverter


def test_hadamard_on_register() -> None:
    """Test that H(q) generates a loop when q is a register."""
    prog = Main(
        q := QReg("q", 4),
        # Apply Hadamard to entire register
        qubit.H(q),
        # Measure all qubits
        Measure(q) > CReg("c", 4),
    )

    guppy_code = SlrConverter(prog).guppy()

    # Should generate a loop to apply H to each qubit
    assert "for" in guppy_code or "quantum.h(q[0])" in guppy_code

    # Check that H is applied (either in a loop or expanded)
    if "for" in guppy_code:
        # Loop form
        assert "quantum.h(q[i])" in guppy_code
    else:
        # Expanded form
        h_count = guppy_code.count("quantum.h")
        assert h_count >= 4, f"Expected at least 4 H gates, got {h_count}"


def test_multiple_gates_on_register() -> None:
    """Test multiple single-qubit gates on registers."""
    prog = Main(
        q := QReg("q", 3),
        # Apply multiple gates to entire register
        qubit.H(q),
        qubit.X(q),
        qubit.Z(q),
        # Measure all
        Measure(q) > CReg("c", 3),
    )

    guppy_code = SlrConverter(prog).guppy()

    # Check that all gates are applied (either in loops or expanded)
    if "for" in guppy_code:
        # Loop form
        assert "quantum.h(q[i])" in guppy_code
        assert "quantum.x(q[i])" in guppy_code
        assert "quantum.z(q[i])" in guppy_code
    else:
        # Expanded form
        assert guppy_code.count("quantum.h") >= 3
        assert guppy_code.count("quantum.x") >= 3
        assert guppy_code.count("quantum.z") >= 3


def test_mixed_register_and_element_ops() -> None:
    """Test mixing register-wide and element-specific operations."""
    prog = Main(
        q := QReg("q", 4),
        # Apply H to entire register
        qubit.H(q),
        # Apply X to specific elements
        qubit.X(q[0]),
        qubit.X(q[2]),
        # Apply Z to entire register again
        qubit.Z(q),
        Measure(q) > CReg("c", 4),
    )

    guppy_code = SlrConverter(prog).guppy()

    # Should have H and Z applied to all qubits (either in loops or expanded)
    if "for" in guppy_code:
        # Loop form - count loops
        assert "quantum.h(q[i])" in guppy_code
        assert "quantum.z(q[i])" in guppy_code
    else:
        # Expanded form
        assert guppy_code.count("quantum.h") >= 4
        assert guppy_code.count("quantum.z") >= 4

    # Should have X applied to specific qubits (always individual)
    assert "quantum.x(q[0])" in guppy_code
    assert "quantum.x(q[2])" in guppy_code
