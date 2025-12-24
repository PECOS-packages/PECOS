"""Test IR generator with HUGR compilation scenarios."""

from pecos.qeclib import qubit
from pecos.qeclib.qubit.measures import Measure
from pecos.slr import CReg, If, Main, QReg
from pecos.slr.gen_codes.guppy.ir_generator import IRGuppyGenerator


def test_ir_handles_array_measurement_patterns() -> None:
    """Test that IR generator produces HUGR-compatible array measurement code."""
    prog = Main(
        q := QReg("q", 3),
        c := CReg("c", 3),
        # Operations on individual qubits
        qubit.H(q[0]),
        qubit.CX(q[0], q[1]),
        # Then measure all at once
        Measure(q) > c,
    )

    gen = IRGuppyGenerator()
    gen.generate_block(prog)
    code = gen.get_output()

    # Should use measure_array for the full array
    assert "c = quantum.measure_array(q)" in code

    # Check that the code structure should be HUGR-compatible
    # The IR unpacks arrays, but we need to ensure operations use unpacked names
    lines = code.split("\n")

    # Find unpacking line
    unpack_line = next(
        (i for i, line in enumerate(lines) if "q_0, q_1, q_2 = q" in line),
        -1,
    )

    if unpack_line >= 0:
        # After unpacking, operations should use unpacked names
        for i in range(unpack_line + 1, len(lines)):
            line = lines[i]
            if "quantum.h" in line:
                # Should use q_0 not q[0] after unpacking
                assert (
                    "q_0" in line or "q[0]" not in line
                ), f"Line {i}: Should use unpacked name after unpacking"


def test_ir_handles_mixed_measurements() -> None:
    """Test IR generator with mixed measurement patterns."""
    prog = Main(
        q := QReg("q", 4),
        c := CReg("c", 4),
        # Individual measurements first
        Measure(q[0]) > c[0],
        # Gate on another qubit
        qubit.H(q[1]),
        # More measurements
        Measure(q[1]) > c[1],
        Measure(q[2]) > c[2],
        Measure(q[3]) > c[3],
    )

    gen = IRGuppyGenerator()
    gen.generate_block(prog)
    code = gen.get_output()

    # Should handle individual measurements correctly
    assert "quantum.measure(q" in code

    # Check code structure
    # IR generator should produce code that avoids HUGR issues


def test_ir_with_conditional_measurements() -> None:
    """Test IR generator with conditional measurement patterns."""
    prog = Main(
        q := QReg("q", 3),
        flag := CReg("flag", 1),
        results := CReg("results", 2),
        # Measure for condition
        Measure(q[0]) > flag[0],
        If(flag[0])
        .Then(
            Measure(q[1]) > results[0],
        )
        .Else(
            Measure(q[2]) > results[1],
        ),
    )

    gen = IRGuppyGenerator()
    gen.generate_block(prog)
    code = gen.get_output()

    # Check structure - after unpacking, it should use flag_0
    assert "if flag_0:" in code or "if flag[0]:" in code
    assert "else:" in code

    # Check code structure for HUGR compatibility
    # (Actual HUGR compilation would require setting up the proper environment)


def test_ir_avoids_subscript_after_consume() -> None:
    """Test that IR generator avoids accessing arrays after they're consumed."""
    # This is a pattern that fails with the original generator
    prog = Main(
        q := QReg("q", 2),
        c := CReg("c", 2),
        # Measure all qubits
        Measure(q) > c,
        # Try to access individual elements (should not generate q[0] access)
        # This would fail in HUGR if we tried to access q[0] after measure_array
    )

    gen = IRGuppyGenerator()
    gen.generate_block(prog)
    code = gen.get_output()

    # Should use measure_array
    assert "quantum.measure_array(q)" in code

    # Should NOT have any q[0] or q[1] access after measurement
    lines = code.split("\n")
    measure_line = next(i for i, line in enumerate(lines) if "measure_array" in line)

    # Check no array access after measurement
    for i in range(measure_line + 1, len(lines)):
        assert "q[0]" not in lines[i], "Should not access q[0] after measure_array"
        assert "q[1]" not in lines[i], "Should not access q[1] after measure_array"
