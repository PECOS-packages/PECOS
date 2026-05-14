"""Audit-only `SlrConverter.hugr(_force_ast=True)` hook tests."""

from __future__ import annotations

from pecos.slr import CReg, Main, QReg, SlrConverter
from pecos.slr.qeclib import qubit as qb
from pecos.slr.qeclib.qubit.measures import Measure


def test_force_ast_hugr_compiles_via_ast_guppy_path() -> None:
    prog = Main(
        q := QReg("q", 2),
        c := CReg("c", 2),
        qb.H(q[0]),
        qb.CX(q[0], q[1]),
        Measure(q) > c,
    )

    hugr = SlrConverter(prog).hugr(_force_ast=True)

    assert hugr is not None
