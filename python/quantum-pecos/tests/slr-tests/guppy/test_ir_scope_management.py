"""Test scope management in IR generator."""

from pecos.qeclib.qubit.measures import Measure
from pecos.slr import CReg, If, Main, QReg
from pecos.slr.gen_codes.guppy.ir_generator import IRGuppyGenerator


def test_conditional_resource_balancing() -> None:
    """Test that resources are balanced across conditional branches."""
    prog = Main(
        q := QReg("q", 3),
        flag := CReg("flag", 1),
        result := CReg("result", 2),
        # Measure first qubit for condition
        Measure(q[0]) > flag[0],
        # Conditional that consumes different resources in each branch
        If(flag[0])
        .Then(
            # Then branch: measure q[1]
            Measure(q[1])
            > result[0],
        )
        .Else(
            # Else branch: measure q[2]
            Measure(q[2])
            > result[1],
        ),
    )

    gen = IRGuppyGenerator()
    gen.generate_block(prog)
    code = gen.get_output()

    # print("Generated code with conditional resource balancing:")
    # print(code)

    # Both branches should exist
    # With unpacking, flag[0] becomes flag_0
    assert "if flag[0]:" in code or "if flag_0:" in code
    assert "else:" in code

    # Check measurements in branches
    lines = code.split("\n")

    # Find the if and else blocks
    # Support both array access and unpacked variable
    if_idx = next(
        (
            i
            for i, line in enumerate(lines)
            if ("if flag[0]:" in line or "if flag_0:" in line)
        ),
        -1,
    )
    if if_idx == -1:
        msg = "Could not find if statement"
        raise AssertionError(msg)
    else_idx = next(i for i, line in enumerate(lines) if line.strip() == "else:")

    # Check that both branches have measurements
    then_block = lines[if_idx + 1 : else_idx]
    else_block = lines[else_idx + 1 :]

    # Then branch should measure q[1] (result was renamed to result_reg)
    # With dynamic allocation, it uses individual variables instead of array access
    assert any("result_0 = quantum.measure" in line for line in then_block)

    # Else branch should measure q[2]
    assert any("result_1 = quantum.measure" in line for line in else_block)


def test_nested_conditional_scopes() -> None:
    """Test nested conditional scopes."""
    prog = Main(
        q := QReg("q", 4),
        flags := CReg("flags", 2),
        c := CReg("c", 1),
        # Outer condition
        Measure(q[0]) > flags[0],
        If(flags[0]).Then(
            # Inner condition
            Measure(q[1]) > flags[1],
            If(flags[1]).Then(
                # Nested then: measure q[2]
                Measure(q[2])
                > c[0],
            ),
            # q[3] not measured in inner if
        ),
        # q[2] and q[3] might not be measured
    )

    gen = IRGuppyGenerator()
    gen.generate_block(prog)
    code = gen.get_output()

    # Should have nested if statements
    assert code.count("if flags") >= 2

    # All qubits should eventually be consumed
    # The IR generator uses discard_array at the end
    assert "# Discard q" in code
    assert "quantum.discard_array(q)" in code


def test_function_scope_returns() -> None:
    """Test that function scopes properly track returned resources."""
    # This would test function-level scope management
    # For now, just test that main function works
    prog = Main(
        q := QReg("q", 2),
        c := CReg("c", 1),
        # Only measure first qubit
        Measure(q[0]) > c[0],
        # q[1] should be cleaned up
    )

    gen = IRGuppyGenerator()
    gen.generate_block(prog)
    code = gen.get_output()

    # With dynamic allocation, only q_0 is allocated and measured, no cleanup needed for q_1
    # Check that the measurement happened correctly (may be q_0, q_0_local, or c_0)
    assert (
        "c[0] = quantum.measure(q_0)" in code
        or "c_0 = quantum.measure(q_0)" in code
        or "c[0] = quantum.measure(q_0_local)" in code
        or "c_0 = quantum.measure(q_0_local)" in code
    )
    # Check that result is generated
    assert 'result("c", c)' in code
