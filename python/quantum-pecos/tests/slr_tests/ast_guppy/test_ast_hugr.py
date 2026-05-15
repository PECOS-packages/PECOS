"""Lock-in tests for `SlrConverter.hugr()` AST-routed path (post-cutover)."""

from __future__ import annotations

import pytest
from pecos.slr import CReg, For, LoopVar, Main, QReg, Repeat, Return, SlrConverter, While
from pecos.slr.ast.codegen.guppy import GuppyCodegenError
from pecos.slr.qeclib import qubit as qb
from pecos.slr.qeclib.qubit.measures import Measure


def test_hugr_compiles_via_ast_guppy_path() -> None:
    prog = Main(
        q := QReg("q", 2),
        c := CReg("c", 2),
        qb.H(q[0]),
        qb.CX(q[0], q[1]),
        Measure(q) > c,
    )

    hugr = SlrConverter(prog).hugr()

    assert hugr is not None


def test_hugr_rejects_while_before_parallel_optimizer_erases_it() -> None:
    prog = Main(
        q := QReg("q", 1),
        c := CReg("c", 1),
        While(c[0] == 0).Do(
            qb.H(q[0]),
            Measure(q[0]) > c[0],
        ),
    )

    with pytest.raises(GuppyCodegenError, match="does not support While loops"):
        SlrConverter(prog).hugr()


def test_hugr_rejects_non_z_prep_basis_before_ast_drops_it() -> None:
    prog = Main(
        q := QReg("q", 1),
        qb.Prep(q[0], "X"),
    )

    with pytest.raises(GuppyCodegenError, match="supports only Z-basis Prep"):
        SlrConverter(prog).hugr()


def test_hugr_rejects_symbolic_loopvar_indexing_cleanly() -> None:
    i = LoopVar("i")
    prog = Main(
        q := QReg("q", 4),
        For(i, range(4)).Do(qb.H(q[i])),
    )

    with pytest.raises(GuppyCodegenError, match="symbolic LoopVar indexing"):
        SlrConverter(prog).hugr()


def test_hugr_accepts_inline_measure_creg_result() -> None:
    prog = Main(
        q := QReg("q", 2),
        Measure(q) > CReg("final", 2),
    )

    hugr = SlrConverter(prog).hugr()

    assert hugr is not None


def test_hugr_returns_no_arg_entrypoint_runnable_via_hugr_adapter() -> None:
    """Pin the post-cutover entrypoint contract.

    `SlrConverter.hugr()` must return a Package whose `to_str()` produces
    HUGR JSON for a no-arg entrypoint -- the same shape `Guppy(func).compile()`
    would produce. This is required for downstream consumers (`pecos.Hugr(bytes)`,
    Selene runtime, `pecos_rslib.HugrProgram`) that expect a runnable program,
    not a parameterized function definition.
    """
    from pecos import Hugr, selene_engine, sim

    prog = Main(
        q := QReg("q", 2),
        c := CReg("c", 2),
        qb.H(q[0]),
        qb.CX(q[0], q[1]),
        Measure(q) > c,
    )

    package = SlrConverter(prog).hugr()
    hugr_bytes = package.to_str().encode("utf-8")

    result = sim(Hugr(hugr_bytes)).classical(selene_engine()).qubits(2).seed(42).run(10)

    raw = result.to_dict() if hasattr(result, "to_dict") else result
    assert raw, "Hugr adapter produced no measurement records from .hugr() output"


def test_hugr_supports_explicit_return_of_root_allocator() -> None:
    """Regression for Codex finding #1.

    A v1 program with an explicit `Return(q)` (no result CRegs) must compile.
    The wrapper must pass main's return value through, not silently discard it
    (which produces Guppy `UnnamedExprNotUsedError`).
    """
    prog = Main(
        q := QReg("q", 1),
        qb.H(q[0]),
        Return(q),
    )

    hugr = SlrConverter(prog).hugr()

    assert hugr is not None


def test_hugr_inline_measure_creg_round_trips_through_selene() -> None:
    """Regression for Codex finding #2.

    The AST emitter infers inline `CReg("final", n)` registers from
    `Measure(q) > CReg(...)` and main returns them. The entry wrapper must
    capture and flatten those results -- not discard them -- so downstream
    consumers see measurement records.
    """
    from pecos import Hugr, selene_engine, sim

    prog = Main(
        q := QReg("qi", 2),
        qb.H(q[0]),
        qb.CX(q[0], q[1]),
        Measure(q) > CReg("final", 2),
    )

    package = SlrConverter(prog).hugr()
    hugr_bytes = package.to_str().encode("utf-8")

    result = sim(Hugr(hugr_bytes)).classical(selene_engine()).qubits(2).seed(42).run(10)

    raw = result.to_dict() if hasattr(result, "to_dict") else result
    assert raw, "Inline-CReg .hugr() output produced no measurement records"


def test_hugr_inline_measure_creg_inside_nested_repeat() -> None:
    """Walker must descend into nested control-flow bodies (Repeat/If/For/While/Parallel).

    `Measure(q[0]) > CReg("flag", 1)` is buried inside a Repeat body. The wrapper's
    `_walk_for_measure_results` must still find "flag" as an inline result register
    and the entry wrapper must capture+flatten it -- otherwise the package compiles
    but downstream sees no measurement record from the nested Measure.
    """
    from pecos import Hugr, selene_engine, sim

    prog = Main(
        q := QReg("q", 1),
        Repeat(2).block(
            qb.X(q[0]),
            Measure(q[0]) > CReg("flag", 1),
            qb.Prep(q[0]),
        ),
    )

    package = SlrConverter(prog).hugr()
    hugr_bytes = package.to_str().encode("utf-8")

    result = sim(Hugr(hugr_bytes)).classical(selene_engine()).qubits(1).seed(42).run(5)

    raw = result.to_dict() if hasattr(result, "to_dict") else result
    assert raw, "Nested inline-CReg .hugr() output produced no measurement records"


def test_hugr_explicit_return_of_non_result_creg() -> None:
    """Pin entry_wrapper / emitter parity on declared non-result CRegs.

    The emitter's `_return_value_type` resolves any declared CReg name in
    `Return(...)` -- `is_result` is not consulted. The wrapper must mirror
    this exactly so it can't silently diverge. Without this fix the wrapper's
    `_explicit_return_type` looked up only `info.result_cregs` (filtered to
    `is_result=True`), so `Return(c)` for `c = CReg(..., result=False)` would
    raise `ValueError` while the emitter happily compiled.

    Note: the combo `result=False` + `Return(c)` is itself a design quirk
    (the two flags say opposite things). Whether v1 should reject this in
    preflight, or whether SLR should adopt a `result()` / explicit-output
    mechanism instead, is a separate v1-followup question. This test only
    pins wrapper/emitter parity for the current semantics.
    """
    prog = Main(
        q := QReg("q", 1),
        c := CReg("c", 1, result=False),
        qb.Prep(q[0]),
        Measure(q[0]) > c[0],
        Return(c),
    )

    hugr = SlrConverter(prog).hugr()

    assert hugr is not None
