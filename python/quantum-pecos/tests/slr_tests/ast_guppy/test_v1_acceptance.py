"""v1 acceptance tests for the AST -> Guppy emitter.

Each test is the spec for one feature in the v1 supported set. All
marked `xfail(strict=True)` until the emitter rewrite lands the
corresponding feature; xfail comes off as features ship.

Test layout follows the practical v1 acceptance set plus the
coverage gaps for final-root-return, static For,
Parallel, PZ-after-measure, mixed Permute, gates beyond CX,
SZ/SZdg mapping, measurement-without-output, and targeted unsupported
errors.
"""

from __future__ import annotations

from pecos.slr import Block, CReg, If, Main, QReg, Repeat, Return
from pecos.slr.qeclib import qubit as qb
from pecos.slr.qeclib.qubit.measures import Measure

from ._harness import assert_ast_guppy_compiles  # noqa: TID252


class TestStraightLine:
    """Bell, GHZ, simple-reset, multi-register; the basics."""

    def test_bell(self) -> None:
        prog = Main(
            q := QReg("q", 2),
            c := CReg("c", 2),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
            Measure(q) > c,
            Return(c),
        )
        assert_ast_guppy_compiles(prog)

    def test_ghz_three(self) -> None:
        prog = Main(
            q := QReg("q", 3),
            c := CReg("c", 3),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
            qb.CX(q[1], q[2]),
            Measure(q) > c,
            Return(c),
        )
        assert_ast_guppy_compiles(prog)

    def test_multi_register(self) -> None:
        prog = Main(
            a := QReg("a", 2),
            b := QReg("b", 2),
            ca := CReg("ca", 2),
            cb := CReg("cb", 2),
            qb.H(a[0]),
            qb.CX(a[0], a[1]),
            qb.X(b[0]),
            qb.CX(b[0], b[1]),
            Measure(a) > ca,
            Measure(b) > cb,
            Return(ca, cb),
        )
        assert_ast_guppy_compiles(prog)


class TestMeasurement:
    """Partial / full / no-output / individual measurement patterns."""

    def test_partial_measurement_lives_discarded(self) -> None:
        """q[0] measured; q[1], q[2] live -> codegen discards them at exit."""
        prog = Main(
            q := QReg("q", 3),
            c := CReg("c", 1),
            qb.H(q[0]),
            Measure(q[0]) > c[0],
            Return(c),
        )
        assert_ast_guppy_compiles(prog)

    def test_full_register_measurement(self) -> None:
        prog = Main(
            q := QReg("q", 4),
            c := CReg("c", 4),
            qb.H(q[0]),
            qb.H(q[1]),
            qb.H(q[2]),
            qb.H(q[3]),
            Measure(q) > c,
            Return(c),
        )
        assert_ast_guppy_compiles(prog)

    def test_individual_measurements(self) -> None:
        prog = Main(
            q := QReg("q", 3),
            c := CReg("c", 3),
            qb.X(q[0]),
            Measure(q[0]) > c[0],
            qb.X(q[1]),
            Measure(q[1]) > c[1],
            Measure(q[2]) > c[2],
            Return(c),
        )
        assert_ast_guppy_compiles(prog)

    def test_measurement_without_output(self) -> None:
        """Measure with no `> creg` consumes the qubit, discards the result."""
        prog = Main(
            q := QReg("q", 2),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
            Measure(q),
        )
        assert_ast_guppy_compiles(prog)


class TestClassical:
    """Classical bit operations that must typecheck against bool arrays."""

    def test_creg_bit_set_int_literal(self) -> None:
        prog = Main(
            c := CReg("c", 2),
            c[0].set(1),
            c[1].set(0),
            Return(c),
        )
        assert_ast_guppy_compiles(prog)


class TestPrep:
    """PZ as reset (live slot) or fresh allocation (consumed slot)."""

    def test_prep_resets_live_slot(self) -> None:
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 2),
            qb.H(q[0]),
            Measure(q[0]) > c[0],
            qb.PZ(q[0]),  # consumed -> fresh qubit()
            Measure(q[0]) > c[1],
            Return(c),
        )
        assert_ast_guppy_compiles(prog)


class TestControlFlow:
    """Conditional + loop patterns within v1 supported semantics."""

    def test_conditional_x_state_preserving(self) -> None:
        """If-then where then preserves slot state; identity else is implicit."""
        prog = Main(
            q := QReg("q", 2),
            c := CReg("c", 2),
            Measure(q[0]) > c[0],
            If(c[0]).Then(qb.X(q[1])),
            Measure(q[1]) > c[1],
            Return(c),
        )
        assert_ast_guppy_compiles(prog)

    def test_repeat_state_preserving_body(self) -> None:
        """Repeat whose body leaves slot state unchanged."""
        prog = Main(
            q := QReg("q", 1),
            Repeat(3).block(qb.H(q[0]), qb.H(q[0])),
            Measure(q[0]),
        )
        assert_ast_guppy_compiles(prog)


class TestGatesBeyondCX:
    """Core gates per matrix: H, X, Y, Z, S/Sdg, T/Tdg, CX, CY, CZ, CH."""

    def test_pauli_set(self) -> None:
        prog = Main(
            q := QReg("q", 4),
            c := CReg("c", 4),
            qb.X(q[0]),
            qb.Y(q[1]),
            qb.Z(q[2]),
            qb.H(q[3]),
            Measure(q) > c,
            Return(c),
        )
        assert_ast_guppy_compiles(prog)

    def test_szdg_t_tdg(self) -> None:
        """SZ/SZdg map to s/sdg in Guppy; T/Tdg are direct."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            qb.H(q[0]),
            qb.SZ(q[0]),
            qb.T(q[0]),
            qb.Tdg(q[0]),
            qb.SZdg(q[0]),
            qb.H(q[0]),
            Measure(q[0]) > c[0],
            Return(c),
        )
        assert_ast_guppy_compiles(prog)

    def test_cy_cz(self) -> None:
        prog = Main(
            q := QReg("q", 2),
            c := CReg("c", 2),
            qb.H(q[0]),
            qb.CY(q[0], q[1]),
            qb.CZ(q[0], q[1]),
            Measure(q) > c,
            Return(c),
        )
        assert_ast_guppy_compiles(prog)


class TestReturn:
    """Final root-level Return (stage 5 finding 5)."""

    def test_final_root_return_qubit_array(self) -> None:
        """Explicit Return -> generated function returns the live array."""
        from pecos.slr.misc import Return

        prog = Main(
            q := QReg("q", 1),
            qb.H(q[0]),
            Return(q),
        )
        assert_ast_guppy_compiles(prog)


# Rejection tests for divergent control flow + unsupported gates land
# in a follow-up once the emitter has a specific `LinearityError`
# (or analogous typed error) to assert against. Asserting `Exception`
# today would silently pass on the AST path's pre-existing breakage --
# a fallback the design philosophy explicitly disallows.
