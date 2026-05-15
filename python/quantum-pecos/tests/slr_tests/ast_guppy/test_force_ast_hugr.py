"""Audit-only `SlrConverter.hugr(_force_ast=True)` hook tests."""

from __future__ import annotations

import pytest
from pecos.slr import CReg, For, LoopVar, Main, QReg, SlrConverter, While
from pecos.slr.ast.codegen.guppy import GuppyCodegenError
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


def test_force_ast_rejects_while_before_parallel_optimizer_erases_it() -> None:
    prog = Main(
        q := QReg("q", 1),
        c := CReg("c", 1),
        While(c[0] == 0).Do(
            qb.H(q[0]),
            Measure(q[0]) > c[0],
        ),
    )

    with pytest.raises(GuppyCodegenError, match="does not support While loops"):
        SlrConverter(prog).hugr(_force_ast=True)


def test_force_ast_rejects_non_z_prep_basis_before_ast_drops_it() -> None:
    prog = Main(
        q := QReg("q", 1),
        qb.Prep(q[0], "X"),
    )

    with pytest.raises(GuppyCodegenError, match="supports only Z-basis Prep"):
        SlrConverter(prog).hugr(_force_ast=True)


def test_force_ast_rejects_symbolic_loopvar_indexing_cleanly() -> None:
    i = LoopVar("i")
    prog = Main(
        q := QReg("q", 4),
        For(i, range(4)).Do(qb.H(q[i])),
    )

    with pytest.raises(GuppyCodegenError, match="symbolic LoopVar indexing"):
        SlrConverter(prog).hugr(_force_ast=True)


def test_force_ast_accepts_inline_measure_creg_result() -> None:
    prog = Main(
        q := QReg("q", 2),
        Measure(q) > CReg("final", 2),
    )

    hugr = SlrConverter(prog).hugr(_force_ast=True)

    assert hugr is not None
