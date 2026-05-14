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

"""Audit runner for Step 4 Workstream B (cutover gap discovery).

Iterates a curated list of `(source_label, slr_program_factory)`
pairs from PECOS examples, qeclib, and existing test fixtures.
Runs each through the AST -> Guppy path via
`SlrConverter.hugr(_force_ast=True)` (private kwarg added by
Codex; see step4-cutover-plan.md) and captures any failures.

This is NOT a pytest test file. It's an audit tool grug runs
during Workstream B and at cutover. Output is the seed for new
rows in `~/Repos/pecos-docs/design/slr/v1-audit-manifest.md`.

Invocation:
    cd /home/ciaranra/Repos/PECOS
    uv run python python/quantum-pecos/tests/slr_tests/ast_guppy/audit_runner.py

For each program: emits one of
- OK   <label>
- FAIL <label> <ExceptionType>: <truncated message>

The curated list is intentionally small at first and grows as we
identify canonical examples worth exercising. The list is the audit
surface; growing it is part of Workstream B.
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pecos.slr import CReg, If, Main, QReg, Repeat, SlrConverter
from pecos.slr.qeclib import qubit as qb
from pecos.slr.qeclib.qubit.measures import Measure

if TYPE_CHECKING:
    from collections.abc import Callable

    from pecos.slr import Block


@dataclass(frozen=True)
class AuditCase:
    """One audit entry: a label and a factory that builds the SLR program."""

    label: str
    factory: Callable[[], Block]


@dataclass(frozen=True)
class AuditResult:
    """Audit outcome for one program."""

    label: str
    passed: bool
    exception_type: str | None = None
    exception_message: str | None = None


def _bell() -> Block:
    return Main(
        q := QReg("q", 2),
        c := CReg("c", 2),
        qb.H(q[0]),
        qb.CX(q[0], q[1]),
        Measure(q) > c,
    )


def _ghz_three() -> Block:
    return Main(
        q := QReg("q", 3),
        c := CReg("c", 3),
        qb.H(q[0]),
        qb.CX(q[0], q[1]),
        qb.CX(q[1], q[2]),
        Measure(q) > c,
    )


def _conditional_correction() -> Block:
    return Main(
        q := QReg("q", 2),
        c := CReg("c", 2),
        Measure(q[0]) > c[0],
        If(c[0]).Then(qb.X(q[1])),
        Measure(q[1]) > c[1],
    )


def _repeat_idle() -> Block:
    return Main(
        q := QReg("q", 1),
        Repeat(3).block(qb.H(q[0]), qb.H(q[0])),
        Measure(q[0]),
    )


def _curated_cases() -> list[AuditCase]:
    """v1 acceptance baseline; grows during Workstream B as gaps surface.

    Initial list mirrors the v1 acceptance corpus. As the audit
    runner is exercised against examples/ and qeclib, additional
    cases are added here. Each new case is a candidate manifest row.
    """
    return [
        AuditCase("v1.bell", _bell),
        AuditCase("v1.ghz_three", _ghz_three),
        AuditCase("v1.conditional_correction", _conditional_correction),
        AuditCase("v1.repeat_idle", _repeat_idle),
    ]


def _run_case(case: AuditCase) -> AuditResult:
    """Run one program through the AST path and capture pass/fail."""
    try:
        prog = case.factory()
    except BaseException as exc:
        return AuditResult(
            label=case.label,
            passed=False,
            exception_type=type(exc).__name__,
            exception_message=f"factory raised: {exc}",
        )

    try:
        # _force_ast is a private kwarg that routes SlrConverter.hugr()
        # through the AST path even before cutover. Audit-only.
        SlrConverter(prog).hugr(_force_ast=True)
    except TypeError as exc:
        # _force_ast not yet plumbed -- emit a clear marker so the runner
        # output points at the missing kwarg instead of looking like a
        # real audit failure.
        if "_force_ast" in str(exc):
            return AuditResult(
                label=case.label,
                passed=False,
                exception_type="MissingKwarg",
                exception_message="SlrConverter.hugr() does not accept _force_ast yet (Codex emitter PR pending)",
            )
        return AuditResult(
            label=case.label,
            passed=False,
            exception_type=type(exc).__name__,
            exception_message=str(exc).splitlines()[0][:200],
        )
    except BaseException as exc:
        return AuditResult(
            label=case.label,
            passed=False,
            exception_type=type(exc).__name__,
            exception_message=str(exc).splitlines()[0][:200],
        )

    return AuditResult(label=case.label, passed=True)


def run_audit() -> list[AuditResult]:
    """Run the full curated list and return all results."""
    return [_run_case(case) for case in _curated_cases()]


def main() -> int:
    """CLI entrypoint. Returns 0 if all pass; non-zero if any fail."""
    results = run_audit()
    for r in results:
        if r.passed:
            print(f"OK   {r.label}")
        else:
            print(f"FAIL {r.label} {r.exception_type}: {r.exception_message}")

    failures = sum(1 for r in results if not r.passed)
    total = len(results)
    print()
    print(f"Audit summary: {total - failures}/{total} passed; {failures} failed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
