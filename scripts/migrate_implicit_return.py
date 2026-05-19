"""Phase 3b S1 codemod: make implicit-return `Main(...)` explicit.

For every `Main(...)` call that does NOT already contain a `Return(...)`
argument, append `Return(<result CRegs>)` listing the CRegs that were
implicitly returned -- i.e. CRegs bound by a walrus assignment directly
in the `Main(...)` arguments via `name := CReg(...)` whose `CReg(...)`
call does NOT pass `result=False`, in order of appearance (mirrors the
v1 implicit-return tuple order).

This is behavior-preserving under the *current* implementation (proven:
explicit `Return(c)` is QASM-SHA + Selene-record identical to the
implicit path and suppresses the deprecation warning). It does NOT
touch `result=` kwargs (that is S3) -- so byte-identity is exact.

Sites whose result-CReg set is not statically determinable from
walrus-in-`Main` args (e.g. `Main(*ops, ...)`, CRegs declared outside
the call) are left untouched and reported; they are hand-migrated and
the byte-identity + `-W error::DeprecationWarning` gate is the
authoritative verifier.

Run: `uv run --with libcst python scripts/migrate_implicit_return.py PATH...`
"""

from __future__ import annotations

import sys
from pathlib import Path

import libcst as cst
import libcst.matchers as m


class _MainReturnTransformer(cst.CSTTransformer):
    def __init__(self) -> None:
        self.migrated = 0
        self.skipped_existing_return = 0
        self.residual: list[str] = []
        self._needs_return_import = False

    @staticmethod
    def _is_result_creg_call(call: cst.Call) -> bool:
        """A `CReg(...)` call counts as a result CReg unless it explicitly
        passes `result=False`."""
        for arg in call.args:
            if arg.keyword is not None and arg.keyword.value == "result" and m.matches(arg.value, m.Name("False")):
                return False
        return True

    # libcst dispatches transformer hooks by the exact method name
    # `leave_<CSTNodeClassName>` and calls them positionally; the
    # CamelCase name is the libcst API contract, not ours -- renaming
    # it breaks dispatch (hence the scoped N802 allow). `original` is
    # unused, so `_original` silences ARG002 without a suppression.
    def leave_Call(self, _original: cst.Call, updated: cst.Call) -> cst.BaseExpression:  # noqa: N802
        # Match bare `Main(...)` AND any qualified form (`pecos.slr.Main(...)`,
        # `slr.Main(...)`) -- the func is then an Attribute whose final attr
        # is `Main` (qualified calls were previously missed).
        if not m.matches(
            updated.func,
            m.Name("Main") | m.Attribute(attr=m.Name("Main")),
        ):
            return updated

        has_return = any(m.matches(a.value, m.Call(func=m.Name("Return"))) for a in updated.args)
        if has_return:
            self.skipped_existing_return += 1
            return updated

        result_cregs: list[str] = []
        saw_unresolvable = False
        for a in updated.args:
            if a.star:  # `Main(*ops, ...)` -- can't see the CRegs
                saw_unresolvable = True
                continue
            v = a.value
            if m.matches(v, m.NamedExpr()) and m.matches(
                v.value,
                m.Call(func=m.Name("CReg")),
            ):
                target = v.target
                if isinstance(target, cst.Name) and self._is_result_creg_call(v.value):
                    result_cregs.append(target.value)

        if not result_cregs:
            if saw_unresolvable:
                self.residual.append("Main(*...) / non-walrus CRegs")
            return updated

        self.migrated += 1
        # Mirror the qualifier of the matched `Main` call: `pecos.slr.Main`
        # -> `pecos.slr.Return` (no import needed); bare `Main` -> bare
        # `Return` (+ ensure `from pecos.slr import Return`).
        if isinstance(updated.func, cst.Attribute):
            return_func: cst.BaseExpression = updated.func.with_changes(
                attr=cst.Name("Return"),
            )
        else:
            return_func = cst.Name("Return")
            self._needs_return_import = True
        # Trailing comma on the appended arg preserves the "magic trailing
        # comma" so the formatter keeps the call's original multi-line vs
        # single-line shape (minimal diff -- no whole-call reflow).
        ret = cst.Arg(
            value=cst.Call(
                func=return_func,
                args=[cst.Arg(value=cst.Name(n)) for n in result_cregs],
            ),
            comma=cst.Comma(),
        )
        return updated.with_changes(args=[*updated.args, ret])


def _ensure_return_import(module: cst.Module) -> cst.Module:
    """Add `Return` to an existing `from pecos.slr import ...` if absent."""

    class _ImportFixer(cst.CSTTransformer):
        # See `leave_Call` above: libcst mandates the `leave_<Node>`
        # name (scoped N802) and positional call; `_original` silences
        # ARG002 without a suppression.
        def leave_ImportFrom(  # noqa: N802
            self,
            _original: cst.ImportFrom,
            updated: cst.ImportFrom,
        ) -> cst.ImportFrom:
            mod = updated.module
            if not (m.matches(mod, m.Attribute()) and cst.Module([]).code_for_node(mod) == "pecos.slr"):
                return updated
            if isinstance(updated.names, cst.ImportStar):
                return updated
            names = list(updated.names)
            present = {n.name.value for n in names if isinstance(n.name, cst.Name)}
            if "Return" in present or "Main" not in present:
                return updated
            names.append(cst.ImportAlias(name=cst.Name("Return")))
            names.sort(key=lambda n: n.name.value if isinstance(n.name, cst.Name) else "")
            # Re-add commas (sort drops them); libcst normalizes.
            fixed = [a.with_changes(comma=cst.MaybeSentinel.DEFAULT) for a in names]
            return updated.with_changes(names=fixed)

    return module.visit(_ImportFixer())


def main(paths: list[str]) -> int:
    total_migrated = 0
    total_residual: list[str] = []
    for p in paths:
        path = Path(p)
        src = path.read_text()
        mod = cst.parse_module(src)
        tx = _MainReturnTransformer()
        new = mod.visit(tx)
        if tx.migrated:
            new = _ensure_return_import(new)
        if new.code != src:
            path.write_text(new.code)
        if tx.migrated or tx.residual:
            print(
                f"{p}: migrated={tx.migrated} "
                f"skipped_existing_return={tx.skipped_existing_return} "
                f"residual={len(tx.residual)}",
            )
        total_migrated += tx.migrated
        total_residual += [f"{p}: {r}" for r in tx.residual]
    print(f"\nTOTAL migrated Main() sites: {total_migrated}")
    if total_residual:
        print(f"RESIDUAL (hand-migrate, gate will flag): {len(total_residual)}")
        for r in total_residual:
            print(f"  {r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
