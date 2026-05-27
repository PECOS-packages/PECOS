"""Tests for Guppy code generation from SLR programs.

The basic-circuit / conditional / repeat / measurement / various-gate
coverage is in the v1 acceptance corpus
(``tests/slr_tests/ast_guppy/test_v1_acceptance.py``). What survives
here are larger end-to-end SLR patterns (Steane-style multi-pair CX,
PZ across multiple qubits, mixed quantum/classical Permute) that the
acceptance set does not exercise but that v1 supports.

The Steane(...) prep tests and the conditional X-after-measure tests
were deleted: Steane prep is v2 (BlockCall + nested-return), and
``If(c[0]).Then(X(q[0]))`` after ``Measure(q[0])`` is the use-after-
measurement pattern v1 explicitly rejects.
"""

import sys
from pathlib import Path

# ``tests/slr_tests/ast_guppy`` is a proper package with the v1 compile
# harness, but this file lives under ``tests/slr_tests/pecos/unit/slr/``
# where adding ``__init__.py`` would shadow the installed ``pecos``
# package. Instead, put ``tests/slr_tests`` on ``sys.path`` so the
# absolute import below resolves.
_SLR_TESTS_ROOT = Path(__file__).resolve().parents[3]
if str(_SLR_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_SLR_TESTS_ROOT))

from ast_guppy._harness import assert_ast_guppy_compiles  # noqa: E402
from pecos.slr import CReg, Main, QReg, Repeat, Return  # noqa: E402
from pecos.slr.misc import Permute  # noqa: E402
from pecos.slr.qeclib import qubit as qb  # noqa: E402


def test_bitwise_operations() -> None:
    """Test generation of bitwise operations."""
    from pecos.slr import SlrConverter

    prog = Main(
        c := CReg("c", 8),
        # Initialize some bits
        c[0].set(1),
        c[1].set(0),
        c[2].set(1),
        # Test bitwise in assignments
        c[3].set(c[0] ^ c[1]),
        c[4].set(c[0] & c[2]),
        c[5].set(c[1] | c[2]),
        # Test NOT operation
        c[6].set(~c[0]),
        c[7].set((c[0] | c[1]) & ~c[2]),
        Return(c),
    )

    guppy_code = SlrConverter(prog).guppy()

    # Check bitwise operations - AST codegen uses underscore naming for expressions
    # XOR uses ^ operator
    assert "c[3] = " in guppy_code
    assert "^" in guppy_code
    # AND uses 'and' operator
    assert "c[4] = " in guppy_code
    assert "and" in guppy_code
    # OR uses 'or' operator
    assert "c[5] = " in guppy_code
    assert "or" in guppy_code
    # NOT uses 'not' operator
    assert "c[6] = " in guppy_code
    assert "not" in guppy_code
    # Complex expression with multiple operators
    assert "c[7] = " in guppy_code


def test_repeat_loop() -> None:
    """Repeat with a state-preserving body compiles."""
    prog = Main(
        q := QReg("q", 1),
        Repeat(3).block(
            qb.H(q[0]),
            qb.H(q[0]),
        ),
    )
    assert_ast_guppy_compiles(prog)


def test_register_operations() -> None:
    """Mixed register-wide and element-wise operations on one QReg."""
    prog = Main(
        q := QReg("q", 4),
        _c := CReg("c", 4),
        qb.H(q),
        qb.X(q[0]),
        qb.X(q[2]),
        qb.CX(q[0], q[1]),
        qb.CX(q[2], q[3]),
        Return(_c),
    )
    assert_ast_guppy_compiles(prog)


def test_steane_encoding_circuit_pattern() -> None:
    """Multi-pair CX pattern from the Steane encoding circuit + PZ set."""
    prog = Main(
        q := QReg("q", 7),
        qb.PZ(q[0], q[1], q[2], q[3], q[4], q[5]),
        qb.CX(q[6], q[5]),
        qb.H(q[1]),
        qb.CX(q[1], q[0]),
        qb.H(q[2]),
        qb.CX(q[2], q[4]),
        qb.H(q[3]),
        qb.CX(
            (q[3], q[5]),
            (q[2], q[0]),
            (q[6], q[4]),
        ),
        qb.CX(
            (q[2], q[6]),
            (q[3], q[4]),
            (q[1], q[5]),
        ),
        qb.CX(
            (q[1], q[6]),
            (q[3], q[0]),
        ),
    )
    assert_ast_guppy_compiles(prog)


def test_reset_operations() -> None:
    """PZ across single and multi-qubit forms compiles."""
    prog = Main(
        q := QReg("q", 3),
        _c := CReg("c", 3),
        qb.PZ(q[0]),
        qb.H(q[0]),
        qb.CX(q[0], q[1]),
        qb.PZ(q[1], q[2]),
        qb.X(q[0]),
        qb.Y(q[1]),
        qb.Z(q[2]),
        Return(_c),
    )
    assert_ast_guppy_compiles(prog)


def test_permute_operations() -> None:
    """Element, multi-element, and whole-register Permute compile together."""
    prog = Main(
        a := QReg("a", 3),
        b := QReg("b", 3),
        c := CReg("c", 2),
        d := CReg("d", 2),
        Permute([a[0], b[1]], [b[1], a[0]]),
        Permute([a[0], a[1], a[2]], [a[2], a[0], a[1]]),
        Permute(c, d),
        qb.H(a[0]),
        qb.X(b[1]),
        Return(c, d),
    )
    assert_ast_guppy_compiles(prog)
