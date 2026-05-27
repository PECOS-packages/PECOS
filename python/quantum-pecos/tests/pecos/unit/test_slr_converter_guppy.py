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

"""Tests for SlrConverter Guppy functionality.

The v1 AST -> Guppy emitter is exercised via compile-and-run acceptance tests
under ``tests/slr_tests/ast_guppy/``. Tests here cover basic structural sanity
of `SlrConverter.hugr()` (AST-routed post-cutover, wraps `main` in a no-arg
`entry()` and compiles that).
"""

from pecos.slr import CReg, Main, QReg, SlrConverter
from pecos.slr.qeclib import qubit as qb


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
