"""Test complex permutation patterns in quantum circuits."""

from pecos.qeclib import qubit
from pecos.qeclib.qubit.measures import Measure
from pecos.slr import CReg, Main, Permute, QReg
from pecos.slr.gen_codes.guppy.ir_generator import IRGuppyGenerator


def test_permute_identity() -> None:
    """Test identity permutation (no change)."""
    prog = Main(
        q := QReg("q", 3),
        # Identity permutation
        Permute([q[0], q[1], q[2]], [q[0], q[1], q[2]]),
        Measure(q) > CReg("c", 3),
    )

    gen = IRGuppyGenerator()
    gen.generate_block(prog)
    code = gen.get_output()

    # Identity permutation should be recognized
    # Currently generates cycles but they're all fixed points
    assert "quantum.measure_array(q)" in code


def test_permute_reverse() -> None:
    """Test reversing array elements."""
    prog = Main(
        q := QReg("q", 5),
        # Initialize with pattern
        qubit.X(q[0]),
        qubit.Y(q[1]),
        qubit.Z(q[2]),
        qubit.H(q[3]),
        qubit.SZ(q[4]),
        # Reverse the array
        Permute([q[0], q[1], q[2], q[3], q[4]], [q[4], q[3], q[2], q[1], q[0]]),
        Measure(q) > CReg("c", 5),
    )

    gen = IRGuppyGenerator()
    gen.generate_block(prog)
    code = gen.get_output()

    # Should generate swaps
    assert "_temp_swap" in code
    assert "# Permute 5 elements" in code


def test_permute_rotate() -> None:
    """Test rotation of elements."""
    prog = Main(
        q := QReg("q", 4),
        # Rotate left by 1: [0,1,2,3] -> [1,2,3,0]
        Permute([q[0], q[1], q[2], q[3]], [q[1], q[2], q[3], q[0]]),
        Measure(q) > CReg("c", 4),
    )

    gen = IRGuppyGenerator()
    gen.generate_block(prog)
    code = gen.get_output()

    # Should generate a cycle
    assert "_temp_cycle" in code


def test_permute_complex_pattern() -> None:
    """Test complex permutation with multiple cycles."""
    prog = Main(
        a := QReg("a", 6),
        # Complex permutation:
        # 0->2, 2->4, 4->0 (cycle 1)
        # 1->3, 3->5, 5->1 (cycle 2)
        Permute(
            [a[0], a[1], a[2], a[3], a[4], a[5]],
            [a[2], a[3], a[4], a[5], a[0], a[1]],
        ),
        Measure(a) > CReg("c", 6),
    )

    gen = IRGuppyGenerator()
    gen.generate_block(prog)
    code = gen.get_output()

    # Should have permutation operations
    assert "# Permute 6 elements" in code
    assert "_temp_" in code  # Either swap or cycle


def test_permute_partial_registers() -> None:
    """Test permuting parts of different registers."""
    prog = Main(
        x := QReg("x", 3),
        y := QReg("y", 3),
        # Mix elements from both registers
        Permute([x[0], x[1], y[0], y[1]], [y[1], x[0], y[0], x[1]]),
        Measure(x) > CReg("cx", 3),
        Measure(y) > CReg("cy", 3),
    )

    gen = IRGuppyGenerator()
    gen.generate_block(prog)
    code = gen.get_output()

    # Should handle cross-register permutations
    assert "# Permute 4 elements" in code


def test_permute_with_gates() -> None:
    """Test permutation interleaved with quantum gates."""
    prog = Main(
        q := QReg("q", 3),
        # Initial gates
        qubit.H(q[0]),
        qubit.CX(q[0], q[1]),
        # First permutation: rotate
        Permute([q[0], q[1], q[2]], [q[1], q[2], q[0]]),
        # More gates on permuted qubits
        qubit.CX(q[0], q[1]),  # Now acts on what was q[1] and q[2]
        # Second permutation: swap first two
        Permute([q[0], q[1], q[2]], [q[1], q[0], q[2]]),
        Measure(q) > CReg("c", 3),
    )

    gen = IRGuppyGenerator()
    gen.generate_block(prog)
    code = gen.get_output()

    # Should have multiple permutations
    assert code.count("# Permute") >= 2


def test_permute_error_mismatched_elements() -> None:
    """Test error handling for mismatched element lists."""
    prog = Main(
        a := QReg("a", 3),
        b := QReg("b", 2),
        # Try to permute with different elements
        Permute([a[0], a[1], a[2]], [b[0], b[1], a[0]]),
        Measure(a) > CReg("ca", 3),
        Measure(b) > CReg("cb", 2),
    )

    gen = IRGuppyGenerator()
    gen.generate_block(prog)
    code = gen.get_output()

    # Should generate error comment
    assert "ERROR: Invalid permutation" in code


def test_permute_single_cycle() -> None:
    """Test single large cycle permutation."""
    prog = Main(
        q := QReg("q", 7),
        # Single cycle touching all elements: 0->1->2->3->4->5->6->0
        Permute(
            [q[0], q[1], q[2], q[3], q[4], q[5], q[6]],
            [q[1], q[2], q[3], q[4], q[5], q[6], q[0]],
        ),
        Measure(q) > CReg("c", 7),
    )

    gen = IRGuppyGenerator()
    gen.generate_block(prog)
    code = gen.get_output()

    # Should use cycle temporary
    assert "_temp_cycle" in code
    # Should have exactly 7 assignments in the cycle
    assert code.count(" = q[") == 7  # 6 shifts + 1 from temp
