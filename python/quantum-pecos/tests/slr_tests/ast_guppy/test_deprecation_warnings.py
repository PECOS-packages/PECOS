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

"""Post-S2 contracts for the Phase 3b output-model migration.

S2 removes the Phase-2 deprecation warnings only. It intentionally does not
remove the `CReg(result=...)` kwarg and does not yet remove the implicit-return
code path; those hard breaking changes are S3. These tests pin the S2 boundary:
old warning-producing programs still work, but no Phase-2 DeprecationWarning is
emitted.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, TypeVar

from pecos.slr import CReg, Main, QReg, Return, SlrConverter
from pecos.slr.qeclib import qubit as qb
from pecos.slr.qeclib.qubit.measures import Measure

from ._selene_harness import run_ast_guppy_via_selene  # noqa: TID252

if TYPE_CHECKING:
    from collections.abc import Callable

T = TypeVar("T")


def _run_without_deprecation_warning(func: Callable[[], T]) -> T:
    """Run `func` and fail if any DeprecationWarning is emitted."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        result = func()

    deprecations = [warning for warning in caught if issubclass(warning.category, DeprecationWarning)]
    assert deprecations == [], f"expected no DeprecationWarning, got {[str(w.message) for w in deprecations]}"
    return result


class TestCRegResultKwargPostS2:
    """`result=` is still accepted in S2, but no longer warns."""

    def test_creg_with_result_false_does_not_warn(self) -> None:
        c = _run_without_deprecation_warning(lambda: CReg("c", 1, result=False))
        assert c.result is False

    def test_creg_with_explicit_result_true_does_not_warn(self) -> None:
        c = _run_without_deprecation_warning(lambda: CReg("c", 1, result=True))
        assert c.result is True

    def test_creg_with_default_result_does_not_warn(self) -> None:
        c = _run_without_deprecation_warning(lambda: CReg("c", 1))
        assert c.result is True


class TestImplicitReturnPostS2:
    """Implicit-return programs still work in S2, but no longer warn."""

    def test_implicit_return_program_still_compiles_without_warning(self) -> None:
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            qb.X(q[0]),
            Measure(q[0]) > c[0],
        )

        source = _run_without_deprecation_warning(lambda: SlrConverter(prog).guppy())
        assert "result.c" not in source

    def test_implicit_return_hugr_still_compiles_without_warning(self) -> None:
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            qb.X(q[0]),
            Measure(q[0]) > c[0],
        )

        package = _run_without_deprecation_warning(lambda: SlrConverter(prog).hugr())
        assert package is not None

    def test_explicit_return_program_still_runs_through_selene(self) -> None:
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            qb.X(q[0]),
            Measure(q[0]) > c[0],
            Return(c),
        )

        records = _run_without_deprecation_warning(lambda: run_ast_guppy_via_selene(prog, shots=10))
        assert all(record["measurement_0"] == 1 for record in records)

    def test_inline_whole_register_measure_still_compiles_without_warning(self) -> None:
        """S1b's whole-register cout shape no longer has a warning detector."""
        prog = Main(
            q := QReg("q", 2),
            Measure(q) > CReg("inline", 2),
        )

        source = _run_without_deprecation_warning(lambda: SlrConverter(prog).guppy())
        assert "inline" in source

    def test_inline_slice_measure_still_compiles_without_warning(self) -> None:
        """S1b's slice cout shape no longer has a warning detector."""
        inline = CReg("inline", 2)
        prog = Main(
            q := QReg("q", 2),
            Measure(q) > inline[0:2],
        )

        source = _run_without_deprecation_warning(lambda: SlrConverter(prog).guppy())
        assert "inline" in source

    def test_result_false_only_measure_still_compiles_without_warning(self) -> None:
        def build_and_compile() -> str:
            q = QReg("q", 1)
            scratch = CReg("scratch", 1, result=False)
            prog = Main(
                q,
                scratch,
                qb.X(q[0]),
                Measure(q[0]) > scratch[0],
            )
            return SlrConverter(prog).guppy()

        source = _run_without_deprecation_warning(build_and_compile)
        assert "scratch" in source
