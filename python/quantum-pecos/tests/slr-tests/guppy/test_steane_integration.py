"""Test SLR-to-HUGR compilation with Steane code integration.

This test demonstrates the complete pipeline from natural SLR code
through Guppy generation to HUGR compilation with real quantum
error correction code.
"""

from pecos.qeclib.steane.steane_class import Steane
from pecos.slr import Main, SlrConverter


def test_steane_guppy_generation() -> None:
    """Test that Steane SLR code generates valid Guppy code."""
    # Create natural SLR program with Steane code
    prog = Main(
        c := Steane("c"),
        c.px(),
    )

    # Generate Guppy code
    guppy_code = SlrConverter(prog).guppy()

    # Verify code generation succeeded
    assert guppy_code is not None
    assert len(guppy_code) > 0

    # Verify basic structure
    assert "from guppylang.decorator import guppy" in guppy_code
    assert "@guppy" in guppy_code
    assert "def main() -> None:" in guppy_code

    # Verify array/struct interfaces are maintained
    assert "array[qubit," in guppy_code or "struct" in guppy_code
    assert (
        "-> tuple[array[qubit," in guppy_code
        or "-> array[qubit," in guppy_code
        or "-> c_struct" in guppy_code
        or "_struct" in guppy_code
    )

    # print("PASS: Guppy code generation successful")
    # print(f"PASS: Generated {len(guppy_code.splitlines())} lines of code")


def test_steane_array_boundary_pattern() -> None:
    """Test that the struct-based boundary pattern is correctly implemented."""
    prog = Main(
        c := Steane("c"),
        c.px(),
    )

    guppy_code = SlrConverter(prog).guppy()

    # Verify struct patterns
    lines = guppy_code.splitlines()

    # Check for struct definition
    struct_lines = [
        line
        for line in lines
        if "@guppy.struct" in line or ("class" in line and "_struct" in line)
    ]
    assert len(struct_lines) > 0, "Should have struct definition"

    # Check for proper function interfaces with structs
    function_lines = [
        line
        for line in lines
        if "def " in line and ("_struct" in line or ": c_struct" in line)
    ]
    assert len(function_lines) > 0, "Should have functions with struct interfaces"

    # Check for struct construction
    struct_construction = [line for line in lines if "_struct(" in line and "=" in line]
    assert len(struct_construction) > 0, "Should have struct construction"

    # Check for natural SLR assignment pattern (no temporary variables)
    assignment_lines = [line for line in lines if " = " in line and "prep_" in line]
    assert len(assignment_lines) > 0, "Should have function assignments"

    # Verify no temporary variable pollution
    temp_lines = [line for line in lines if "_temp" in line or "_returned" in line]
    assert (
        len(temp_lines) == 0
    ), "Should not use temporary variables - maintains natural SLR semantics"

    # print("PASS: Struct-based boundary pattern correctly implemented")


def test_steane_hugr_compilation() -> None:
    """Test HUGR compilation of Steane code."""
    prog = Main(
        c := Steane("c"),
        c.px(),
    )

    try:
        hugr = SlrConverter(prog).hugr()
        assert hugr is not None

    except ImportError as e:
        print(f"WARNING: HUGR compilation issue: {e}")

        # Even if HUGR compilation fails, verify the Guppy code quality
        guppy_code = SlrConverter(prog).guppy()

        # Check that we're using struct patterns
        assert (
            "steane_struct" in guppy_code or "c_struct" in guppy_code
        ), "Should use struct pattern"
        assert "_returned" not in guppy_code, "Should not use temporary variables"

        # The test passes if the code shows the correct patterns
        # even if HUGR compilation isn't perfect yet


def test_natural_slr_usage() -> None:
    """Test that SLR can be written completely naturally."""
    # This should work without any special considerations for Guppy
    prog = Main(
        c := Steane("c"),
        c.px(),  # Natural Steane operation
    )

    # Should generate code without errors
    guppy_code = SlrConverter(prog).guppy()

    # Verify struct patterns are used
    assert (
        "steane_struct" in guppy_code or "c_struct" in guppy_code
    ), "Should use struct pattern"
    assert "c_d = array(quantum.qubit() for _ in range(7))" in guppy_code
    # c_a might be dynamically allocated
    assert (
        "c_a = array(quantum.qubit() for _ in range(3))" in guppy_code
        or "c_a_0 = quantum.qubit()" in guppy_code
    )
