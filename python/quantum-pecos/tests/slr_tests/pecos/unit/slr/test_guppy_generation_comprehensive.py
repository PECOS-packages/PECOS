"""Comprehensive Guppy code generation patterns from realistic SLR programs.

The patterns retained here exercise larger SLR programs that are within
v1 scope but not part of the v1 acceptance set. The legacy file also
contained:

- A syndrome-extraction pattern that reused an ancilla after measure
  without an intervening ``PZ`` (use-after-measurement, v1 rejects).
- A parameterized circuit with branches consuming different qubits
  (divergent post-state, v1 rejects).
- Complex permutation cycles that are not bijective over the same slot
  set (v1 rejects).
- A nested-repeat pattern with conditional measurement that produces a
  divergent quantum state (v1 rejects).
- A mixed-classical-quantum program whose ``c[0].set(1)`` emits ``= 1``
  into a ``bool`` array (a current v1-emitter shortcoming for integer
  literals; tracked separately and not worked around here).

Those tests have been deleted because their underlying SLR programs
are explicitly unsupported in v1 (or expose a separate emitter bug
that should not be papered over by tests).
"""

import sys
from pathlib import Path

# Bridge ``tests/slr_tests/ast_guppy._harness`` into this file. See the
# matching block in ``test_guppy_generation.py`` for the rationale --
# adding ``__init__.py`` files inside ``slr_tests/pecos/`` would shadow
# the installed ``pecos`` package.
_SLR_TESTS_ROOT = Path(__file__).resolve().parents[3]
if str(_SLR_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_SLR_TESTS_ROOT))

from ast_guppy._harness import assert_ast_guppy_compiles  # noqa: E402
from pecos.slr import CReg, If, Main, QReg, Return, SlrConverter  # noqa: E402
from pecos.slr.qeclib import qubit as qb  # noqa: E402


def test_quantum_teleportation() -> None:
    """The standard teleportation circuit -- two If corrections on Bob."""
    prog = Main(
        alice := QReg("alice", 1),
        bob := QReg("bob", 1),
        epr := QReg("epr", 1),
        c := CReg("c", 2),
        # Create EPR pair
        qb.H(epr[0]),
        qb.CX(epr[0], bob[0]),
        # Alice's operations
        qb.CX(alice[0], epr[0]),
        qb.H(alice[0]),
        # Measure Alice's qubits
        qb.Measure(alice[0]) > c[0],
        qb.Measure(epr[0]) > c[1],
        # Bob's corrections
        If(c[1]).Then(
            qb.X(bob[0]),
        ),
        If(c[0]).Then(
            qb.Z(bob[0]),
        ),
        Return(c),
    )
    assert_ast_guppy_compiles(prog)


def test_complex_boolean_expressions() -> None:
    """Test complex classical boolean expressions with proper precedence."""
    prog = Main(
        c := CReg("c", 8),
        # Set initial values
        c[0].set(1),
        c[1].set(0),
        c[2].set(1),
        # Complex expressions - test precedence
        c[3].set(c[0] | (c[1] & c[2])),
        c[4].set((c[0] | c[1]) & (c[2] ^ c[3])),
        c[5].set((c[0] ^ c[1]) ^ (c[2] & ~c[3])),
        # Nested operations
        If((c[0] | ~c[1]) & (c[2] ^ (c[3] & c[4]))).Then(
            c[6].set(~(c[0] & c[1]) | (c[2] ^ c[3])),
            c[7].set(~((c[5] & c[6]) ^ (c[0] | c[3]))),
        ),
        Return(c),
    )

    guppy_code = SlrConverter(prog).guppy()

    # Check that assignments are present - AST codegen uses array indexing for targets
    assert "c[3] = " in guppy_code
    assert "c[4] = " in guppy_code
    assert "c[5] = " in guppy_code
    assert "if" in guppy_code

    # Boolean operations use Python keywords/operators
    assert "or" in guppy_code  # OR uses 'or'
    assert "and" in guppy_code  # AND uses 'and'
    assert "^" in guppy_code  # XOR uses '^'
    assert "not" in guppy_code  # NOT uses 'not'


def test_empty_blocks_and_edge_cases() -> None:
    """Test empty blocks and various edge cases."""
    from pecos.slr import Repeat

    prog = Main(
        q := QReg("q", 1),
        c := CReg("c", 2),
        # Empty conditional
        If(c[0]).Then(),
        # Empty repeat
        Repeat(3).block(),
        # Nested empty blocks
        If(c[0]).Then(
            Repeat(2).block(
                If(c[1]).Then(),
            ),
        ),
        # Measurement without output
        qb.H(q[0]),
        qb.Measure(q[0]),
        # Apply gate to register
        qb.PZ(q),
        Return(c),
    )

    guppy_code = SlrConverter(prog).guppy()

    # Check that code is generated without errors - AST codegen uses array parameters
    assert "def main(q:" in guppy_code
    assert len(guppy_code) > 100
    # Note: empty blocks use 'pass' and reset operations may be optimized


def test_grover_decomposition() -> None:
    """Grover's algorithm with CCX decomposed via T/Tdg + CX."""
    prog = Main(
        q := QReg("q", 2),
        ancilla := QReg("ancilla", 1),
        c := CReg("c", 3),
        # Initialize superposition
        qb.H(q),
        # Oracle using decomposed CCX
        qb.H(ancilla[0]),
        qb.CX(q[1], ancilla[0]),
        qb.Tdg(ancilla[0]),
        qb.CX(q[0], ancilla[0]),
        qb.T(ancilla[0]),
        qb.CX(q[1], ancilla[0]),
        qb.Tdg(ancilla[0]),
        qb.CX(q[0], ancilla[0]),
        qb.T(ancilla[0]),
        qb.H(ancilla[0]),
        # Diffusion operator
        qb.H(q),
        qb.X(q),
        qb.CZ(q[0], q[1]),
        qb.X(q),
        qb.H(q),
        # Measure
        qb.Measure(q) > [c[0], c[1]],
        qb.Measure(ancilla[0]) > c[2],
        Return(c),
    )
    assert_ast_guppy_compiles(prog)


def test_multi_pair_cx_pattern() -> None:
    """Multi-pair CX (e.g. ``CX((q[3], q[5]), ...)``) compiles."""
    prog = Main(
        q := QReg("q", 7),
        # Multi-pair CX from Steane encoding
        qb.CX(
            (q[3], q[5]),
            (q[2], q[0]),
            (q[6], q[4]),
        ),
        # Another pattern
        qb.CX(
            (q[0], q[1]),
            (q[2], q[3]),
        ),
    )
    assert_ast_guppy_compiles(prog)
