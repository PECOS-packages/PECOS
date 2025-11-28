"""Test Permute operation in IR generator."""

from pecos.qeclib import qubit
from pecos.qeclib.qubit.measures import Measure
from pecos.slr import CReg, Main, Permute, QReg
from pecos.slr.gen_codes.guppy.ir_generator import IRGuppyGenerator


def test_ir_simple_permute() -> None:
    """Test simple register swap with IR generator."""
    prog = Main(
        a := QReg("a", 2),
        b := QReg("b", 2),
        # Initialize
        qubit.H(a[0]),
        qubit.X(b[1]),
        # Swap registers
        Permute(a, b),
        # Measure
        Measure(a) > CReg("c_a", 2),
        Measure(b) > CReg("c_b", 2),
    )

    gen = IRGuppyGenerator()
    gen.generate_block(prog)
    code = gen.get_output()

    # Check that swap comment is added
    assert "# Swap a and b" in code

    # Check that temporary variable is used
    assert "_temp_a" in code

    # Check the swap sequence
    assert "_temp_a = a" in code
    assert "a = b" in code
    assert "b = _temp_a" in code


def test_ir_permute_with_operations() -> None:
    """Test Permute with operations before and after."""
    prog = Main(
        q1 := QReg("q1", 3),
        q2 := QReg("q2", 3),
        # Operations on q1
        qubit.H(q1[0]),
        qubit.CX(q1[0], q1[1]),
        # Operations on q2
        qubit.X(q2[0]),
        qubit.Y(q2[1]),
        # Swap the registers
        Permute(q1, q2),
        # Now q1 has what was in q2 and vice versa
        # Measure them
        Measure(q1) > CReg("c1", 3),
        Measure(q2) > CReg("c2", 3),
    )

    gen = IRGuppyGenerator()
    gen.generate_block(prog)
    code = gen.get_output()

    # Check operations are generated
    assert "quantum.h" in code
    assert "quantum.cx" in code
    assert "quantum.x" in code
    assert "quantum.y" in code

    # Check swap
    assert "# Swap q1 and q2" in code
    assert "_temp_q1" in code


def test_ir_complex_permute_cycle() -> None:
    """Test complex permutation with cycle pattern."""
    # Cycle permutation: a[0] -> a[1] -> a[2] -> a[0]
    prog = Main(
        a := QReg("a", 3),
        # Initialize with different states
        qubit.X(a[0]),
        qubit.Y(a[1]),
        qubit.Z(a[2]),
        # Cyclic permutation: 0->1, 1->2, 2->0
        Permute([a[0], a[1], a[2]], [a[1], a[2], a[0]]),
        Measure(a) > CReg("result", 3),
    )

    gen = IRGuppyGenerator()
    gen.generate_block(prog)
    code = gen.get_output()

    # Should have permutation comment
    assert "# Permute 3 elements" in code
    # Should use temporary variable for cycle
    assert "_temp_cycle" in code


def test_ir_complex_permute_multiple_swaps() -> None:
    """Test permutation that decomposes into multiple swaps."""
    prog = Main(
        q := QReg("q", 4),
        # Permutation: swap 0<->3 and 1<->2
        Permute([q[0], q[1], q[2], q[3]], [q[3], q[2], q[1], q[0]]),
        Measure(q) > CReg("result", 4),
    )

    gen = IRGuppyGenerator()
    gen.generate_block(prog)
    code = gen.get_output()

    # Should generate swap operations
    assert "_temp_swap" in code
