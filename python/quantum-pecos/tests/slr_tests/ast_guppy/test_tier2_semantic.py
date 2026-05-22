# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Semantic assurance for the static CReg lowering.

Three independent layers per representative program (the structural
`test_qir_spec_compliance.py` gate covers the whole corpus; this is the
deeper *semantic* proof for the load-bearing CReg shapes):

  A. **Real-compiler acceptance.** Quantinuum's `qir-qis` (a real
     ~10k-LOC QIR->QIS compiler: type-checking, call allowlist,
     profile + result-slot validation, full QIS lowering) must
     `validate_qir` AND `qir_to_qis` the emitted QIR. Acceptance by an
     independent production compiler is strong evidence the emitted
     QIR is well-formed and semantically valid.

  B. **Emitted-QIR structural invariants.** The static classical
     model is deterministic bit-shuffling; its correctness is exactly
     checkable on the emitted IR: per-CReg entry-block `alloca [N x
     i1]` + `zeroinitializer` store; measurements use monotonic
     0-based `%Result*` slots; `read_result` reuses the measurement's
     own slot; result bits `store`d at the `gep` matching the CReg
     bit index; record pack is LSB-first `zext`/`shl`/`or`;
     whole-CReg `c.set(int)` unpacks `lshr`+`trunc` per bit; and NO
     bespoke `create_creg`/`*_creg_bit`/... runtime helpers remain.

  C. **Independent executable cross-anchor.** The SAME SLR programs
     run through the AST->Guppy->Selene path
     (`run_ast_guppy_via_selene`); for the deterministic programs the
     exact classical outcome is asserted. The emitted QIR is a
     deterministic sibling lowering (proven structurally in B) of the
     same AST whose classical semantics C executes.

  D. **Executable differential.** `selene_sim` natively
     executes `qir_qis.qir_to_qis`'s LLVM-21 opaque-pointer QIS
     bitcode via the bundled `selene_helios_qis_plugin` (+ Helios
     QIR runtime) -- `selene_sim.build(BitcodeString(...)) ->
     run_shots(Stim)`. So the direct `qir_to_qis -> Selene`
     differential long claimed "blocked" (an
     alleged LLVM 14<->21 / opaque-vs-typed bridge) is in fact
     available with zero PECOS LLVM work: PECOS-Rust stays LLVM-14;
     the LLVM-21 capability lives entirely in the qir-qis +
     selene_sim *Python* deps. Layer D
     (`test_tier2_executable_differential`) lands the
     **representative** executable differential (deterministic
     programs to exact records; Bell by entanglement-correlation
     property; one clean cross-path vs the AST->Guppy->Selene
     oracle) -- not an exhaustive proof, the representative shapes.

Run standalone: `uv run python -m pytest <thisfile> -q` or
`uv run python .../tier2_semantic.py`. The pytest entry is marked
`slow` (it builds the Selene runtime) so the default fast lane skips
it; it still runs in the full sweep.
"""

from __future__ import annotations

import re
import sys
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable
import qir_qis
import selene_sim
from pecos.slr import CReg, For, If, Main, Print, QReg, Return, SlrConverter, While
from pecos.slr.ast.codegen.qir import AstToQir
from pecos.slr.ast.nodes import VarExpr
from pecos.slr.qeclib import qubit as qb
from pecos.slr.qeclib.qubit.measures import Measure

from ._selene_harness import run_ast_guppy_via_selene  # noqa: TID252
from .audit_runner import _curated_cases  # noqa: TID252

_SHOTS = 8
_SEED = 42
_BESPOKE = re.compile(r"create_creg|get_creg_bit|set_creg_bit|get_int_from_creg|set_creg_to_int|mz_to_creg_bit")
# Ordered mz/read_result events (slot in g1 for mz, g2 for read_result) so
# we can pin per-measurement slot correspondence, not just set membership.
_MZ_RR_EVENT = re.compile(
    r"@__quantum__qis__mz__body\(%Qubit\* [^,]+, "
    r"%Result\* inttoptr \(i64 (\d+) to %Result\*\)\)"
    r"|@__quantum__rt__read_result\(%Result\* inttoptr \(i64 (\d+) to %Result\*\)\)",
)


def _assert_mz_rr_pairing(label: str, ir: str) -> None:
    """Per-measurement slot correspondence: mz `%Result*` slots are
    monotonic 0..n-1, and EVERY `read_result` is immediately preceded
    (in emission order) by the `mz__body` of the SAME slot -- i.e. a
    read_result reuses *its own* measurement's static slot, not merely
    some slot in the mz set (stronger than a subset check)."""
    events: list[tuple[str, int]] = []
    for m in _MZ_RR_EVENT.finditer(ir):
        if m.group(1) is not None:
            events.append(("mz", int(m.group(1))))
        else:
            events.append(("rr", int(m.group(2))))
    mz_slots = [s for kind, s in events if kind == "mz"]
    assert mz_slots == list(range(len(mz_slots))), f"{label}: mz %Result* slots not monotonic 0..n-1: {mz_slots}"
    for i, (kind, slot) in enumerate(events):
        if kind != "rr":
            continue
        assert i > 0, f"{label}: read_result(slot {slot}) emitted before any mz; events={events}"
        assert events[i - 1] == ("mz", slot), (
            f"{label}: read_result(slot {slot}) is not immediately preceded "
            f"by its own mz(slot {slot}); event sequence={events}"
        )


def _case(label: str) -> Main:
    for c in _curated_cases():
        if c.label == label:
            return c.factory()
    msg = f"audit case {label!r} not found"
    raise KeyError(msg)


# --- extra programs not in the corpus (plan amendment 8 coverage) ---


def _set_int() -> Main:
    """Whole-CReg `c.set(int)` -> per-bit lshr/trunc unpack."""
    return Main(
        q := QReg("q", 1),
        c := CReg("c", 4),
        qb.X(q[0]),
        c.set(0b1011),
        Return(c),
    )


def _zero_init_safety() -> Main:
    """`If(c[0])` BEFORE any write -- must read the zeroinit 0, so the
    Then-branch is never taken (undef would be nondeterministic)."""
    return Main(
        q := QReg("q", 1),
        c := CReg("c", 1),
        d := CReg("d", 1),
        If(c[0]).Then(qb.X(q[0])),
        Measure(q[0]) > d[0],
        Return(d),
    )


def _multi_creg() -> Main:
    """Two CRegs: one set classically, one filled by measurement."""
    return Main(
        q := QReg("q", 2),
        a := CReg("a", 2),
        b := CReg("b", 2),
        qb.X(q[0]),
        a.set(0b10),
        Measure(q[0]) > b[0],
        Measure(q[1]) > b[1],
        Return(a, b),
    )


def _creg_int(rec: dict[str, int], n: int) -> int:
    """Pack a `measurement_N`-keyed shot record LSB-first into an int."""
    return sum((rec.get(f"measurement_{i}", 0) & 1) << i for i in range(n))


# --- the three semantic layers -------------------------------------


def _layer_a_compiler_accepts(prog: Main, label: str) -> None:
    bc = SlrConverter(prog).qir_bc()
    qir_qis.validate_qir(bc)
    qis = qir_qis.qir_to_qis(bc)
    assert isinstance(qis, (bytes, bytearray)), f"{label}: qir_to_qis returned {type(qis).__name__}, not bytes"
    assert qis[:4] == b"BC\xc0\xde", f"{label}: qir_to_qis output is not QIS LLVM bitcode (bad magic)"


def _layer_b_structural(prog: Main, label: str, *, creg_sizes: dict[str, int]) -> None:
    ir = SlrConverter(prog).qir()

    assert not _BESPOKE.search(ir), f"{label}: bespoke CReg helper still emitted"

    for name, size in creg_sizes.items():
        assert f"%{name} = alloca [{size} x i1]" in ir, f"{label}: missing entry-block alloca for CReg {name!r}"
        assert (
            f"store [{size} x i1] zeroinitializer, [{size} x i1]* %{name}" in ir
        ), f"{label}: missing zeroinitializer for CReg {name!r} (unset bits must read 0, not undef)"

    _assert_mz_rr_pairing(label, ir)


def _assert_set_int_unpack(label: str, ir: str, *, name: str, size: int, value: int) -> None:
    """`c.set(<const int>)` lowers, after LLVM constant-folds the
    per-bit `lshr`/`trunc`, to a direct `store i1 <bit_i>` at `gep
    c[i]` for each bit (LSB-first). Assert the EXACT stored bits --
    stronger than checking for the (folded-away) shift instructions.

    (`extra.set_int` is verified here + by `qir_to_qis` acceptance,
    NOT by the Guppy oracle: the AST->Guppy return wrapper cannot
    return a `set(int)`-assigned CReg -- it models the CReg as `int`
    while the wrapper expects `array[bool, N]`. That is a Guppy-path
    limitation orthogonal to the QIR, whose lowering is exact and
    checked structurally here.)
    """
    for i in range(size):
        bit = (value >> i) & 1
        pat = (
            rf"getelementptr \[{size} x i1\], \[{size} x i1\]\* %{name}, "
            rf"i64 0, i64 {i}\n\s*store i1 {bit}, i1\* %[.\w]+"
        )
        assert re.search(
            pat,
            ir,
        ), f"{label}: c.set({value:#b}) bit {i} not stored as `i1 {bit}` at gep {name}[{i}] (LSB-first)"
    # Record pack: per-bit load/zext, `shl i64 _, i` for i>0, or-chain.
    assert ir.count("load i1") >= size, f"{label}: record pack missing per-bit `load i1`"
    assert ir.count("zext i1") >= size, f"{label}: record pack missing per-bit `zext i1`"
    for i in range(1, size):
        assert re.search(rf"shl i64 %[.\w]+, {i}\b", ir), f"{label}: record pack missing `shl i64 _, {i}`"
    assert ir.count("or i64") >= size, f"{label}: record pack missing or-chain"


def _assert_zero_init_predicate(label: str, ir: str, *, name: str) -> None:
    """`If(c[i])` before any write must branch on a `load` of the
    zero-initialised buffer (so the predicate is deterministically 0,
    never `undef`)."""
    assert (
        f"store [1 x i1] zeroinitializer, [1 x i1]* %{name}" in ir
    ), f"{label}: missing zeroinitializer for the pre-read CReg {name!r}"
    assert re.search(
        rf"getelementptr \[1 x i1\], \[1 x i1\]\* %{name}, i64 0, i64 0\n\s*%[.\w]+ = load i1",
        ir,
    ), f"{label}: If(c[0]) predicate is not a load of the zero-inited buffer"


def _layer_c_oracle(prog: Main, label: str) -> list[dict[str, int]]:
    recs = run_ast_guppy_via_selene(prog, shots=_SHOTS, seed=_SEED)
    assert len(recs) == _SHOTS, f"{label}: expected {_SHOTS} shots, got {len(recs)}"
    return recs


# --- representative programs + their deterministic expectations ----


def _programs() -> list[tuple[str, Main, dict[str, int]]]:
    return [
        ("v1.bell", _case("v1.bell"), {"c": 2}),
        ("v1.conditional_correction", _case("v1.conditional_correction"), {"c": 2}),
        ("v1.repeat_idle", _case("v1.repeat_idle"), {}),
        (
            "legacy.partial_consumption_with_block",
            _case("legacy.partial_consumption_with_block"),
            {},
        ),
        ("examples.surface_d3_z_1round", _case("examples.surface_d3_z_1round"), {}),
        ("extra.set_int", _set_int(), {"c": 4}),
        ("extra.zero_init_safety", _zero_init_safety(), {"c": 1, "d": 1}),
        ("extra.multi_creg", _multi_creg(), {"a": 2, "b": 2}),
    ]


def run() -> int:
    """Standalone driver: returns process exit code."""
    failures: list[str] = []
    for label, prog, creg_sizes in _programs():
        try:
            _layer_a_compiler_accepts(prog, label)
            ir = SlrConverter(prog).qir()
            assert not _BESPOKE.search(ir), f"{label}: bespoke helper emitted"
            _assert_mz_rr_pairing(label, ir)
            if creg_sizes:
                _layer_b_structural(prog, label, creg_sizes=creg_sizes)
            if label == "extra.set_int":
                _assert_set_int_unpack(label, ir, name="c", size=4, value=0b1011)
            if label == "extra.zero_init_safety":
                _assert_zero_init_predicate(label, ir, name="c")
            print(f"[A/B OK] {label}")
        except AssertionError as exc:  # noqa: PERF203
            failures.append(f"{label}: {exc}")
            print(f"[A/B FAIL] {label}: {exc}")

    # Layer C -- executable deterministic cross-anchor via the
    # dual-reviewed AST->Guppy->Selene oracle. Restricted to programs
    # that path supports; `extra.set_int` / `extra.zero_init_safety`
    # are fully covered by Layer A (real qir-qis compiler acceptance)
    # + Layer B (exact emitted-QIR structure: zeroinitializer and
    # per-bit lshr/trunc unpack). The AST->Guppy return wrapper
    # cannot return a `set(int)`-assigned CReg (it models the CReg as
    # `int`, the wrapper expects `array[bool, N]`) -- a Guppy-path
    # limitation orthogonal to the QIR, so Guppy is not the right
    # oracle for those two shapes.
    det_checks: list[tuple[str, Main, str]] = [
        ("v1.bell", _case("v1.bell"), "bell"),
        ("v1.conditional_correction", _case("v1.conditional_correction"), "all_zero"),
    ]
    for label, prog, expect in det_checks:
        try:
            recs = _layer_c_oracle(prog, label)
            if expect == "bell":
                for r in recs:
                    assert r.get("measurement_0") == r.get("measurement_1"), f"{label}: Bell pair not correlated: {r}"
            else:  # all_zero
                for r in recs:
                    assert set(r.values()) <= {0}, f"{label}: expected all-zero, got {r}"
            print(f"[C OK]   {label}")
        # Per-program isolation is required in this reporting harness
        # (one Guppy/Selene failure must not abort the others); the
        # loop is ~2 items so the PERF203 try/except cost is irrelevant.
        except Exception as exc:  # noqa: PERF203
            failures.append(f"{label} (oracle): {type(exc).__name__}: {exc}")
            print(f"[C FAIL] {label}: {type(exc).__name__}: {exc}")

    if failures:
        print(f"\nTIER-2 SEMANTIC: {len(failures)} FAILURE(S)")
        for f in failures:
            print("  -", f)
        return 1
    print("\nTIER-2 SEMANTIC: PASS")
    return 0


def test_oversize_creg_raises_loud() -> None:
    """A >64-bit CReg must FAIL LOUD at QIR codegen, not silently drop
    its storage/output. 64 is the cap
    (single-i64 record pack); 65 must raise a clear error."""
    ok = Main(q := QReg("q", 1), c := CReg("c", 64), Measure(q[0]) > c[0], Return(c))
    SlrConverter(ok).qir_bc()  # 64 is allowed (boundary)

    over = Main(q := QReg("q", 1), c := CReg("c", 65), Measure(q[0]) > c[0], Return(c))
    with pytest.raises(NotImplementedError, match=r"CReg 'c' has 65 bits.*64-bit cap"):
        SlrConverter(over).qir_bc()


def test_while_raises_loud() -> None:
    """The QIR backend must FAIL LOUD on `While`, not silently
    emit a one-pass approximation (qir-qis cannot catch that -- one
    pass is valid QIR, wrong semantics). Aligns with the Guppy path,
    which already rejects `While` (real While is
    out of scope for the sound emitter)."""
    prog = Main(
        q := QReg("q", 1),
        c := CReg("c", 1),
        While(c[0] == 0).Do(qb.H(q[0]), Measure(q[0]) > c[0]),
        Return(c),
    )
    with pytest.raises(NotImplementedError, match=r"does not support While loops"):
        SlrConverter(prog).qir_bc()


def test_print_raises_loud() -> None:
    """The QIR backend must FAIL LOUD on `Print`, not silently
    drop it (silently losing observable program output)."""
    prog = Main(
        q := QReg("q", 1),
        c := CReg("c", 1),
        Measure(q[0]) > c[0],
        Print(c),
        Return(c),
    )
    with pytest.raises(NotImplementedError, match=r"does not support Print"):
        SlrConverter(prog).qir_bc()


def test_static_for_unrolls_body_not_dropped() -> None:
    """A static `For` must UNROLL its body, not silently
    drop it. The old `_process_for` guard `isinstance(node.start, int)`
    was always false (the converter wraps bounds in `LiteralExpr`), so
    every `For` body was silently dropped -- valid QIR, wrong
    semantics, qir-qis-uncatchable (the exact bug class this targets)."""
    prog = Main(
        q := QReg("q", 1),
        c := CReg("c", 1),
        For("i", 0, 3).Do(qb.X(q[0])),
        Measure(q[0]) > c[0],
        Return(c),
    )
    ir = SlrConverter(prog).qir()
    # Count CALL sites only -- `@__quantum__qis__x__body(` also matches
    # the one `declare` line. `For("i", 0, 3)` -> range(0, 3) -> 3 (the
    # canonical exclusive semantics, matching gen_quantum_circuit).
    n = ir.count("call void @__quantum__qis__x__body(")
    assert n == 3, f"static For body must unroll 3x; got {n} X call(s) (0 == silently dropped)"
    # and the unrolled QIR still lowers via the real qir-qis compiler
    qir_qis.qir_to_qis(SlrConverter(prog).qir_bc())


def test_varexpr_raises_loud() -> None:
    """A classical `VarExpr` whose name is not a declared
    Main-scope CReg must FAIL LOUD, not silently evaluate to constant
    0 (a value miscompile qir-qis cannot catch). The
    VarExpr arm packs the CReg
    via `_pack_creg` if known and fails loud via `_require_creg`
    otherwise. The stray-name path still fails loud, just with the
    `_require_creg` "not declared at Main scope" message (the
    anti-silent-0 invariant is preserved)."""
    with pytest.raises(NotImplementedError, match=r"classical register 'x'.*not.*declared at Main scope"):
        AstToQir()._eval_expression(VarExpr(name="x"))  # noqa: SLF001


def test_inline_returned_creg_raises_loud() -> None:
    """A CReg that is measured/returned but NOT declared at Main
    scope must FAIL LOUD. Only Main-scope CRegs get an entry-block
    `alloca [N x i1]`; an inline/local CReg surfaced only via
    `Return(creg)` had no storage, so the measure-store was SILENTLY
    skipped and the explicit returned value vanished from the QIS
    records (the `docs.inline_measure_creg` defect surfaced --
    QIS recorded `[]` for an explicit `Return(final)`). Mirrors the
    fail-loud doctrine: silent miscompile -> loud NotImplementedError."""
    final = CReg("final", 2)
    prog = Main(q := QReg("q", 2), Measure(q) > final, Return(final))
    with pytest.raises(NotImplementedError, match=r"classical register 'final'.*not.*declared at Main scope"):
        SlrConverter(prog).qir_bc()


def test_prep_stray_string_arg_raises_loud() -> None:
    """The prep basis is the GATE IDENTITY
    (`PZ`/`PNZ`/`PX`/`PNX`/`PY`/`PNY`), not a string argument.
    `_expand_qubit_args` silently drops a string qarg, so a basis
    string on ANY prep gate -- the legacy `PZ(q, "X")`, or
    `PZ(q, "X")`, even `PZ(q, "Z")` -- would be silently dropped
    and lowered as the plain basis (a miscompile). The
    converter rejects ANY stray string qarg on EVERY prep gate
    (the prep-basis symmetry rule). A bare prep gate (no string) is fine."""
    # No-string preps build (PZ default; dedicated gate identity).
    # No-string prep on every dedicated gate builds (basis = identity).
    for prep in (qb.PZ, qb.PNZ, qb.PX, qb.PNX, qb.PY, qb.PNY):
        ok = Main(q := QReg("q", 1), c := CReg("c", 1), prep(q[0]), Measure(q) > c, Return(c))
        SlrConverter(ok).qir_bc()

    # ANY string qarg on EVERY prep gate fails loud (incl. "Z" -- the
    # basis is the gate identity, never an argument).
    for prep, s in (
        (qb.PZ, "X"),
        (qb.PNZ, "Z"),
        (qb.PX, "ignored"),
        (qb.PNX, "X"),
        (qb.PY, "Z"),
        (qb.PNY, "Y"),
    ):
        bad = Main(q := QReg("q", 1), c := CReg("c", 1), prep(q[0], s), Measure(q) > c, Return(c))
        with pytest.raises(NotImplementedError, match=r"stray string argument"):
            SlrConverter(bad).qir_bc()


def test_return_only_inline_creg_raises_loud() -> None:
    """A CReg surfaced ONLY via
    `Return(creg)` -- never measured/assigned/read -- reached no
    point-of-use `_require_creg` site, so `_generate_results` (which
    records only Main-declared CRegs) emitted ZERO recorded output
    for an explicit `Return`. The build succeeded and validated; the
    program silently returned nothing. `_process_return` now
    validates returned classical registers too. (The
    measure-then-return case is `test_inline_returned_creg_raises_loud`;
    this is the no-other-use bypass it did not cover.) A qubit return
    must NOT be false-rejected (QIR records no classical output for
    it)."""
    ret_only = CReg("ret_only", 1)
    prog = Main(q := QReg("q", 1), Return(ret_only))
    with pytest.raises(NotImplementedError, match=r"classical register 'ret_only'.*not.*declared at Main scope"):
        SlrConverter(prog).qir_bc()

    # A declared CReg returned (no other use) is fine; a bare qubit
    # return must not trip the classical-storage check.
    ok = Main(q := QReg("q", 1), c := CReg("c", 1), Return(c))
    SlrConverter(ok).qir_bc()
    ok_q = Main(q := QReg("q", 1), Return(q))
    SlrConverter(ok_q).qir_bc()

    # An inline CReg whose name
    # collides with a declared QReg was misclassified as a qubit
    # return by the old name-membership skip and silently dropped
    # (same silent-output-loss class, reachable via public SLR).
    # Provenance from `_convert_return` (real QReg/CReg object) now
    # disambiguates -> the inline CReg still fails loud.
    collide = Main(q := QReg("q", 1), Return(CReg("q", 1)))
    with pytest.raises(NotImplementedError, match=r"classical register 'q'.*not.*declared at Main scope"):
        SlrConverter(collide).qir_bc()


def _qis_exec_records(
    prog: Main,
    n_qubits: int,
    *,
    shots: int = _SHOTS,
    seed: int = _SEED,
    simulator_factory: Callable[[int], object] | None = None,
) -> list[list[int]]:
    """The real EXECUTABLE differential.

    `AST QIR -> qir_qis.qir_to_qis` (LLVM-21 QIS bitcode) ->
    `selene_sim.build` (selene ingests it natively via
    `selene_helios_qis_plugin`; the long-claimed "blocked on an LLVM
    14<->21 bridge" was false) -> run on the selected backend.
    Returns per-shot lists of the recorded
    `__quantum__rt__int_record_output` *values*, in call order ==
    CReg *declaration* order (`_generate_results` records every
    declared CReg; the QIS tag is empty post-qir_to_qis so order is
    the key). This is the executable proof that the emitted QIR, lowered
    by the real Quantinuum compiler and run, computes the correct
    classical results -- upgrading A+B+C to A+B+C+D.

    `simulator_factory` is a callable `seed -> Simulator` -- defaults
    to `selene_sim.Stim(random_seed=seed)`. Non-Clifford circuits
    (e.g. those containing CH, T, arbitrary rotations) must pass a
    statevector backend such as `lambda s: selene_sim.Quest(
    random_seed=s)` since Stim rejects non-Clifford operations.
    """
    if simulator_factory is None:
        simulator_factory = lambda s: selene_sim.Stim(random_seed=s)  # noqa: E731
    qis = qir_qis.qir_to_qis(SlrConverter(prog).qir_bc())
    inst = selene_sim.build(selene_sim.BitcodeString(qis))
    shots_out = inst.run_shots(simulator_factory(seed), n_qubits=n_qubits, n_shots=shots)
    return [[value for (_tag, value) in shot] for shot in shots_out]


@pytest.mark.slow
def test_tier2_executable_differential() -> None:
    """Deterministic representative programs must EXECUTE through
    QIR -> qir_to_qis -> QIS -> selene to the exact known classical
    record (records are all declared CRegs, declaration order, packed
    LSB-first -- empirically pinned). Oracle-independent: this is the
    real executable equivalence that A+B+C otherwise substituted for.
    """
    cases: list[tuple[str, Main, int, list[int]]] = [
        # c.set(0b1011); Return(c) -> [11] every shot
        ("extra.set_int", _set_int(), 1, [0b1011]),
        # decl(c,d); If(c[0]) before any write reads zero-init 0 -> X
        # not applied; Measure |0> -> d=0. Records [c, d] = [0, 0].
        ("extra.zero_init_safety", _zero_init_safety(), 1, [0, 0]),
        # a.set(0b10)=2; X q0 -> b0=1,b1=0 -> b=0b01=1. Decl order
        # [a, b] -> [2, 1].
        ("extra.multi_creg", _multi_creg(), 2, [2, 1]),
        # |0> measured -> c0=0; If(c0) not taken; |0> -> c1=0 -> [0]
        ("v1.conditional_correction", _case("v1.conditional_correction"), 2, [0]),
    ]
    failures: list[str] = []
    for label, prog, n_qubits, expected in cases:
        recs = _qis_exec_records(prog, n_qubits)
        if all(shot == expected for shot in recs):
            print(f"[D OK]   {label} -> {expected} x{len(recs)}")
        else:
            failures.append(f"{label}: QIS-exec {recs} != expected {expected} (all shots)")
            print(f"[D FAIL] {failures[-1]}")

    # Bell: genuinely quantum -> property check, not a fixed value.
    # NECESSARY: every shot records c[0]==c[1] (single record, packed
    # in {0b00, 0b11}, never 1/2) -- a broken/decorrelating CX trips
    # this. SUFFICIENT: the aggregate over fixed
    # seeds must contain BOTH 0b00 AND 0b11 -- a dropped H / dropped
    # CX / no-op Bell lowering yields all-0, which a subset-only check
    # (`in {0,3}`) would wrongly pass. Fixed seeds -> deterministic;
    # 1/2/7/42 were verified to jointly span both outcomes.
    bell = _case("v1.bell")
    bell_seeds = (1, 2, 7, 42)
    bell_recs = {seed: _qis_exec_records(bell, 2, seed=seed) for seed in bell_seeds}
    well_formed = all(len(shot) == 1 for recs in bell_recs.values() for shot in recs)
    observed = {shot[0] for recs in bell_recs.values() for shot in recs if len(shot) == 1}
    if well_formed and observed == {0b00, 0b11}:
        print(f"[D OK]   v1.bell (c[0]==c[1] always; both 00 & 11 occur over seeds {bell_seeds})")
    else:
        failures.append(
            f"v1.bell: require each shot a single record in {{0,3}} AND both 0 & 3 "
            f"to occur across seeds {bell_seeds} (catches dropped H/CX -> all-0); "
            f"observed={sorted(observed)} well_formed={well_formed} recs={bell_recs}",
        )
        print(f"[D FAIL] {failures[-1]}")

    # Cross-path executable differential vs the dual-reviewed
    # AST->Guppy->Selene oracle. Only `v1.conditional_correction`
    # qualifies cleanly: single declared CReg that is also the
    # returned one (QIS-records-all-declared == oracle-records-returned)
    # AND Guppy-supported. (`set_int`/`multi_creg` use `c.set(int)`,
    # which the AST->Guppy return wrapper cannot return -- the same
    # Layer-C limitation; `zero_init` declares != returns. Those are
    # covered by the exact known-record checks above, which are
    # themselves the executable proof for those shapes.)
    cc = _case("v1.conditional_correction")
    qis_vals = [shot[0] for shot in _qis_exec_records(cc, 2)]
    oracle_vals = [_creg_int(rec, 2) for rec in run_ast_guppy_via_selene(cc, shots=_SHOTS, seed=_SEED)]
    if qis_vals == oracle_vals:
        print(f"[D~oracle OK] v1.conditional_correction: QIS-exec == Guppy-oracle == {qis_vals}")
    else:
        failures.append(f"v1.conditional_correction: QIS-exec {qis_vals} != Guppy-oracle {oracle_vals}")
        print(f"[D~oracle FAIL] {failures[-1]}")

    assert not failures, "Executable differential failures:\n" + "\n".join(failures)


@pytest.mark.slow
def test_tier2_semantic_b2() -> None:
    """Pytest entry (slow: builds the Selene runtime)."""
    assert run() == 0, "Semantic assurance failed (see stdout)"


if __name__ == "__main__":
    sys.exit(run())
