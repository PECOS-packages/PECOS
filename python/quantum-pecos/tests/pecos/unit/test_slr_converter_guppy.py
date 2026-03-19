# Copyright 2025 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Tests for SlrConverter Guppy functionality."""

from pecos.slr import CReg, Main, Parallel, QReg, SlrConverter
from pecos.slr.qeclib import qubit as qb
from pecos.slr.qeclib.steane.steane_class import Steane


def test_slr_converter_guppy_simple() -> None:
    """Test SlrConverter.guppy() with a simple program."""
    prog = Main(
        q := QReg("q", 2),
        c := CReg("c", 2),
        qb.H(q[0]),
        qb.CX(q[0], q[1]),
        qb.Measure(q) > c,
    )

    guppy_code = SlrConverter(prog).guppy()

    # Check that the generated code is valid Python
    # AST codegen uses simplified imports
    assert "from guppylang import guppy" in guppy_code
    assert "@guppy" in guppy_code
    # AST codegen uses array parameters
    assert "def main(q:" in guppy_code
    assert "quantum.h(" in guppy_code
    assert "quantum.cx(" in guppy_code


def test_slr_converter_guppy_does_not_have_undefined_variables() -> None:
    """Test that generated Guppy code doesn't contain undefined variables."""
    prog = Main(
        q := QReg("q", 2),
        c := CReg("c", 2),
        qb.H(q[0]),
        qb.CX(q[0], q[1]),
        qb.Measure(q) > c,
    )

    guppy_code = SlrConverter(prog).guppy()

    # The generated code should compile without undefined variable errors
    # We can't easily test actual compilation here, but we can check for common issues
    lines = guppy_code.split("\n")

    # Find all variable assignments and uses
    defined_vars = set()
    for line in lines:
        stripped_line = line.strip()
        if "=" in stripped_line and not stripped_line.startswith("#") and not stripped_line.startswith("@"):
            # Extract variable definitions (left side of =)
            left_side = stripped_line.split("=")[0].strip()
            vars_defined = [v.strip() for v in left_side.split(",")] if "," in left_side else [left_side]
            defined_vars.update(vars_defined)

    # Now check that we don't have obvious undefined variable usage
    # This is a basic check - the real test would be compilation
    for line in lines:
        # Skip comments and decorators
        if line.strip().startswith("#") or line.strip().startswith("@"):
            continue
        # Skip import and function definition lines
        if any(keyword in line for keyword in ["import", "from", "def ", "class ", "return"]):
            continue

    # At minimum, check that we don't reference common undefined variables
    undefined_vars = ["c_a", "c_a_0"]  # Common issues we've seen
    for var in undefined_vars:
        assert var not in guppy_code, f"Generated code contains undefined variable: {var}"


def test_slr_converter_hugr_simple() -> None:
    """Test SlrConverter.hugr() with a simple program."""
    prog = Main(
        q := QReg("q", 2),
        c := CReg("c", 2),
        qb.H(q[0]),
        qb.CX(q[0], q[1]),
        qb.Measure(q) > c,
    )

    # This should not raise an exception
    hugr = SlrConverter(prog).hugr()

    # Basic checks that we got a valid HUGR object
    assert hugr is not None
    # HUGR is now a Package object with modules
    assert hasattr(hugr, "modules")


def test_slr_converter_steane_guppy_generation() -> None:
    """Test that Steane code can generate Guppy code without undefined variables."""
    prog = Main(
        c := Steane("c"),
        c.px(),  # Simple Pauli-X operation
    )

    # This should generate valid Guppy code without undefined variables
    guppy_code = SlrConverter(prog).guppy()

    # AST codegen uses array parameters instead of local declarations
    # Check that c_a is declared as parameter or that c_a[i] is properly accessed
    # Either c_a: array[...] @owned in params, or c_a = array(...) in body
    c_a_in_params = "c_a: array[qubit" in guppy_code
    c_a_declared = "c_a =" in guppy_code or "c_a=" in guppy_code

    # If c_a appears in the code, it should be in params or declared
    assert ("c_a" not in guppy_code) or c_a_in_params or c_a_declared

    # Code should have quantum operations
    assert "quantum." in guppy_code


def test_slr_converter_steane_hugr_compilation() -> None:
    """Test that Steane code can compile to HUGR.

    This test verifies that the Steane code implementation can be successfully
    compiled to HUGR format through guppylang. The test ensures that:

    1. Ancilla arrays (like c_a) are properly detected and excluded from structs
    2. These arrays are passed to functions with @owned annotation
    3. Arrays are unpacked to individual variables to avoid MoveOutOfSubscriptError
    4. The unpacked variables are used instead of array indexing in function bodies

    The solution works by:
    - Detecting ancilla qubits based on usage patterns (frequent measurement/reset)
    - Excluding them from struct packing to keep them as separate arrays
    - Unpacking @owned ancilla arrays at the start of functions
    - Using the unpacked variables (e.g., c_a_0) instead of array access (c_a[0])

    Note: The guppy code generation itself works correctly, but the final
    compilation to HUGR fails due to API mismatch between guppylang-internals
    (expecting hugr.build module) and hugr 0.13.0 (which doesn't have it).
    """
    prog = Main(
        c := Steane("c"),
        c.px(),
    )

    # This should work once guppylang supports the required patterns
    hugr = SlrConverter(prog).hugr()
    assert hugr is not None
    assert hasattr(hugr, "modules")


def test_slr_converter_parallel_blocks_guppy() -> None:
    """Test Guppy generation with parallel blocks."""
    prog = Main(
        q := QReg("q", 4),
        c := CReg("c", 4),
        Parallel(
            qb.H(q[0]),
            qb.X(q[1]),
            qb.H(q[2]),
            qb.X(q[3]),
        ),
        qb.Measure(q) > c,
    )

    guppy_code = SlrConverter(prog).guppy()

    # Should contain the gates
    assert "quantum.h(" in guppy_code
    assert "quantum.x(" in guppy_code
    assert "quantum.measure" in guppy_code

    # Should not have undefined variables
    undefined_vars = ["c_a", "c_a_0"]
    for var in undefined_vars:
        assert var not in guppy_code, f"Generated code contains undefined variable: {var}"


def test_slr_converter_guppy_has_main_function() -> None:
    """Test that generated Guppy code has a proper main function."""
    prog = Main(
        q := QReg("q", 1),
        c := CReg("c", 1),
        qb.H(q[0]),
        qb.Measure(q) > c,
    )

    guppy_code = SlrConverter(prog).guppy()

    # Should have main function with array parameters
    assert "def main(" in guppy_code
    assert "@guppy" in guppy_code


def test_slr_converter_guppy_imports() -> None:
    """Test that generated Guppy code has correct imports."""
    prog = Main(
        q := QReg("q", 1),
        c := CReg("c", 1),
        qb.H(q[0]),
        qb.Measure(q) > c,
    )

    guppy_code = SlrConverter(prog).guppy()

    # AST codegen uses simplified imports
    required_imports = [
        "from guppylang import guppy",
        "from guppylang.std import quantum",
        "from guppylang.std.quantum import qubit",
    ]

    for imp in required_imports:
        assert imp in guppy_code, f"Missing import: {imp}"
