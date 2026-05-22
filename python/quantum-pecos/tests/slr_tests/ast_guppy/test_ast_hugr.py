"""Lock-in tests for `SlrConverter.hugr()` AST-routed path (post-cutover)."""

from __future__ import annotations

import pytest
from pecos.slr import CReg, For, LoopVar, Main, QReg, Repeat, Return, SlrConverter, While, rad
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
        Return(c),
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
        Return(c),
    )

    with pytest.raises(GuppyCodegenError, match="does not support While loops"):
        SlrConverter(prog).hugr()


def test_hugr_rejects_stray_prep_basis_string() -> None:
    # The prep basis is the gate IDENTITY (PZ/PNZ/PX/PNX/PY/PNY),
    # not a string argument. A stray string qarg on any prep gate is
    # rejected loudly at the shared converter root (NotImplementedError),
    # so it never silently reaches the codegen -- superseding the old
    # Guppy-preflight non-Z-Prep reject (removed along
    # with the `Prep` alias).
    prog = Main(
        q := QReg("q", 1),
        qb.PZ(q[0], "X"),
    )

    with pytest.raises(NotImplementedError, match="stray string argument"):
        SlrConverter(prog).hugr()


def test_hugr_rejects_symbolic_loopvar_indexing_cleanly() -> None:
    i = LoopVar("i")
    prog = Main(
        q := QReg("q", 4),
        For(i, range(4)).Do(qb.H(q[i])),
    )

    with pytest.raises(GuppyCodegenError, match="symbolic LoopVar indexing"):
        SlrConverter(prog).hugr()


def test_angle_first_misuse_fails_loud_at_call() -> None:
    # Angle-first API: a
    # mis-ordered `gate(qubit, angle)` call -- the old silent-
    # angle-drop footgun -- must fail loud at the SLR call, never
    # compile to a silent no-op.
    q = QReg("q", 2)

    # (1) Too few arguments (no angle at all).
    for gate_obj, name in [(qb.RX, "RX"), (qb.RZZ, "RZZ"), (qb.CRX, "CRX"), (qb.CRZ, "CRZ")]:
        with pytest.raises(TypeError, match=f"{name} is a parameterized gate"):
            gate_obj()

    # (2) Qubit reference in an angle slot (the classic `RX(q, 0.5)` /
    #     `RZZ(q0, q1)` mis-order). Caught at the call by the base class.
    with pytest.raises(TypeError, match=r"RX: a register/qubit reference .* angle position"):
        qb.RX(q[0], 0.5)
    for gate_obj, name in [(qb.RZZ, "RZZ"), (qb.CRX, "CRX"), (qb.CRZ, "CRZ")]:
        with pytest.raises(TypeError, match=f"{name}: a register/qubit reference .* angle position"):
            gate_obj(q[0], q[1])

    # (3) Non-qubit in a qubit slot (`RX(0.5, 0.7)`).
    with pytest.raises(TypeError, match=r"RX: a non-qubit .* qubit position"):
        qb.RX(rad(0.5), 0.7)

    # (4) Classical register/bit in a qubit slot:
    #     `CReg`/`Bit`/`SymbolicBit` are `Var` subclasses
    #     too, so a broad `isinstance(_, Var)` qubit check let
    #     `RX(0.5, c[0])` through and QASM/QIR lowered `rx(0.5) c[0];`
    #     against a classical register. Qubit slots accept ONLY
    #     quantum qubit shapes (Qubit/QReg/SymbolicQubit).
    c = CReg("c", 1)
    with pytest.raises(TypeError, match=r"RX: a classical register/bit .* qubit position"):
        qb.RX(rad(0.5), c[0])
    with pytest.raises(TypeError, match=r"RX: a classical register/bit .* qubit position"):
        qb.RX(rad(0.5), c)
    with pytest.raises(TypeError, match=r"RZZ: a classical register/bit .* qubit position"):
        qb.RZZ(rad(0.5), q[0], c[0])

    # (5) v2 typed-angle: a bare numeric angle (no rad()/turns()) is
    #     rejected at the call -- even though the Rust `AngleParam`
    #     extractor would accept a bare f64, SLR requires a typed Angle.
    with pytest.raises(TypeError, match=r"RX: bare numeric angle .* no longer accepted"):
        qb.RX(0.5, q[0])
    with pytest.raises(TypeError, match=r"RZZ: bare numeric angle .* no longer accepted"):
        qb.RZZ(0.5, q[0], q[1])


def test_codegen_arity_guard_rejects_malformed_param_gate() -> None:
    # Even if a malformed
    # parameterized GateOp reaches a codegen from a non-SLR path
    # (the SLR `__call__` guard above blocks the user-facing route),
    # QIR and QASM must FAIL LOUD on too-few-targets rather than
    # silently emitting no operation (the pre-fold behavior: the
    # single-qubit emit loop iterated zero targets; the two-qubit
    # path was gated on `len(targets) >= 2`; the QIR decomposition
    # path raised a raw IndexError).
    from pecos.slr.ast.codegen.qasm import AstToQasm
    from pecos.slr.ast.codegen.qir import AstToQir
    from pecos.slr.ast.nodes import GateKind, GateOp, LiteralExpr, SlotRef

    one = (SlotRef(allocator="q", index=0),)
    malformed = [
        ("RX", GateOp(gate=GateKind.RX, targets=(), params=(LiteralExpr(value=0.5),))),
        ("RZZ", GateOp(gate=GateKind.RZZ, targets=one, params=(LiteralExpr(value=0.5),))),
        ("CRX", GateOp(gate=GateKind.CRX, targets=one, params=(LiteralExpr(value=0.5),))),
        ("CRZ", GateOp(gate=GateKind.CRZ, targets=one, params=(LiteralExpr(value=0.5),))),
    ]
    for name, node in malformed:
        with pytest.raises(NotImplementedError, match=f"QIR codegen: parameterized gate {name!r}"):
            AstToQir()._process_gate(node)  # noqa: SLF001
        with pytest.raises(NotImplementedError, match=f"QASM codegen: parameterized gate {name!r}"):
            AstToQasm().visit_gate(node)


def test_hugr_accepts_inline_measure_creg_result() -> None:
    final = CReg("final", 2)
    prog = Main(
        q := QReg("q", 2),
        Measure(q) > final,
        Return(final),
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
        Return(c),
    )

    package = SlrConverter(prog).hugr()
    hugr_bytes = package.to_str().encode("utf-8")

    result = sim(Hugr(hugr_bytes)).classical(selene_engine()).qubits(2).seed(42).run(10)

    raw = result.to_dict() if hasattr(result, "to_dict") else result
    assert raw, "Hugr adapter produced no measurement records from .hugr() output"


def test_hugr_supports_explicit_return_of_root_allocator() -> None:
    """Regression: explicit-return wrapper passthrough.

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
    """Regression: inline-CReg result capture through the entry wrapper.

    The AST emitter infers inline `CReg("final", n)` registers from
    `Measure(q) > CReg(...)` and main returns them. The entry wrapper must
    capture and flatten those results -- not discard them -- so downstream
    consumers see measurement records.
    """
    from pecos import Hugr, selene_engine, sim

    final = CReg("final", 2)
    prog = Main(
        q := QReg("qi", 2),
        qb.H(q[0]),
        qb.CX(q[0], q[1]),
        Measure(q) > final,
        Return(final),
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

    flag = CReg("flag", 1)
    prog = Main(
        q := QReg("q", 1),
        Repeat(2).block(
            qb.X(q[0]),
            Measure(q[0]) > flag[0],
            qb.PZ(q[0]),
        ),
        Return(flag),
    )

    package = SlrConverter(prog).hugr()
    hugr_bytes = package.to_str().encode("utf-8")

    result = sim(Hugr(hugr_bytes)).classical(selene_engine()).qubits(1).seed(42).run(5)

    raw = result.to_dict() if hasattr(result, "to_dict") else result
    assert raw, "Nested inline-CReg .hugr() output produced no measurement records"


def test_hugr_explicit_return_of_declared_creg() -> None:
    """Pin entry_wrapper / emitter parity on declared CRegs in `Return(...)`.

    The emitter's `_return_value_type` resolves any declared CReg name in
    `Return(...)`. The no-arg `entry()` wrapper's `_explicit_return_type`
    must mirror this exactly (resolving via `info.all_creg_sizes`) so the
    wrapper signature can't silently diverge from `main(...)`.
    """
    prog = Main(
        q := QReg("q", 1),
        c := CReg("c", 1),
        qb.PZ(q[0]),
        Measure(q[0]) > c[0],
        Return(c),
    )

    hugr = SlrConverter(prog).hugr()

    assert hugr is not None
