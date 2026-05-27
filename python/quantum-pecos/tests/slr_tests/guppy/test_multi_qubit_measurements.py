"""Multi-qubit ``Measure(...) > (c[0], c[1], ...)`` patterns.

The legacy tests instantiated a bare ``Block`` and called
``SlrConverter(block).guppy()`` directly, which the v1 AST -> Guppy
emitter does not accept (root allocators must be declared at the
Program level). These rewrites wrap each pattern in a ``Main`` and
verify the generated Guppy compiles via the v1 harness.
"""

from pecos.slr import CReg, Main, QReg, Return
from pecos.slr.qeclib import qubit

from ..ast_guppy._harness import assert_ast_guppy_compiles  # noqa: TID252


class TestMultiQubitMeasurements:
    """Multi-qubit ``Measure`` with classical outputs."""

    def test_multi_qubit_with_outputs(self) -> None:
        prog = Main(
            q := QReg("q", 3),
            c := CReg("c", 3),
            qubit.Measure(q[0], q[1], q[2]) > (c[0], c[1], c[2]),
            Return(c),
        )
        assert_ast_guppy_compiles(prog)

    def test_multi_qubit_without_outputs(self) -> None:
        prog = Main(
            q := QReg("q", 3),
            qubit.Measure(q[0], q[1], q[2]),
        )
        assert_ast_guppy_compiles(prog)

    def test_mixed_measurements(self) -> None:
        prog = Main(
            q := QReg("q", 5),
            c := CReg("c", 5),
            qubit.Measure(q[0]) > c[0],
            qubit.Measure(q[1], q[2], q[3]) > (c[1], c[2], c[3]),
            qubit.Measure(q[4]) > c[4],
            Return(c),
        )
        assert_ast_guppy_compiles(prog)


class TestMultiQubitMeasurementEdgeCases:
    """Edge cases in multi-qubit measurement handling."""

    def test_two_qubit_measurement(self) -> None:
        prog = Main(
            q := QReg("q", 2),
            c := CReg("c", 2),
            qubit.Measure(q[0], q[1]) > (c[0], c[1]),
            Return(c),
        )
        assert_ast_guppy_compiles(prog)

    def test_many_qubit_measurement(self) -> None:
        prog = Main(
            q := QReg("q", 7),
            c := CReg("c", 7),
            qubit.Measure(q[0], q[1], q[2], q[3], q[4], q[5], q[6]) > (c[0], c[1], c[2], c[3], c[4], c[5], c[6]),
            Return(c),
        )
        assert_ast_guppy_compiles(prog)


class TestSingleQubitMeasurementRegression:
    """Single-qubit ``Measure`` regression coverage at the Main level."""

    def test_single_qubit_with_output(self) -> None:
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            qubit.Measure(q[0]) > c[0],
            Return(c),
        )
        assert_ast_guppy_compiles(prog)

    def test_single_qubit_without_output(self) -> None:
        prog = Main(
            q := QReg("q", 1),
            qubit.Measure(q[0]),
        )
        assert_ast_guppy_compiles(prog)
