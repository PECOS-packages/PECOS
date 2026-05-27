"""Tests for conditional resource consumption handling.

Most legacy tests in this file exercised divergent post-state branches
(e.g. ``If(c).Then(Measure(q[1]))`` with no else, or ``Then`` and
``Else`` measuring different qubits). The AST -> Guppy v1 emitter
explicitly rejects those patterns -- the v1 acceptance corpus covers
the supported state-preserving conditional in
``tests/slr_tests/ast_guppy/test_v1_acceptance.py::TestControlFlow``.

The remaining test below exercises a supported state-preserving
conditional through ``SlrConverter.hugr()`` (AST-routed post-cutover).
"""

import pytest
from pecos.slr import CReg, If, Main, QReg, Return, SlrConverter
from pecos.slr.qeclib.qubit.measures import Measure


@pytest.mark.optional_dependency
def test_hugr_compilation_simple() -> None:
    """Test that simple conditional programs can compile to HUGR."""
    prog = Main(
        q := QReg("q", 2),
        c := CReg("c", 2),
        # Simple conditional that should work
        Measure(q[0]) > c[0],
        If(c[0])
        .Then(
            Measure(q[1]) > c[1],
        )
        .Else(
            Measure(q[1]) > c[1],
        ),
        Return(c),
    )

    # This might still fail due to other HUGR issues, but the conditional
    # resource handling should be correct
    try:
        SlrConverter(prog).hugr()
        # If it succeeds, great!
    except ImportError as e:
        # If it fails due to import, that's expected
        if "linearity" in str(e).lower():
            pytest.fail(f"Should not fail due to linearity: {e}")
