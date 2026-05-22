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

"""Hard-contract tests for the v2 output model.

Removed the `CReg(result=...)` kwarg, the `RegisterDecl.is_result`
field, and the v1 implicit return of result-flagged CRegs. These tests
pin those removals as hard contracts: the old knobs now raise, and a
program must use an explicit `Return(...)` to produce any output.
"""

from __future__ import annotations

import pytest
from pecos.slr import CReg, Main, QReg, Return, SlrConverter
from pecos.slr.ast.nodes import RegisterDecl
from pecos.slr.qeclib import qubit as qb
from pecos.slr.qeclib.qubit.measures import Measure

from ._selene_harness import run_ast_guppy_via_selene  # noqa: TID252


class TestCRegResultKwargRemoved:
    """The `CReg(result=...)` kwarg is gone -> hard `TypeError`."""

    def test_creg_result_false_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            CReg("c", 1, result=False)

    def test_creg_result_true_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            CReg("c", 1, result=True)

    def test_creg_without_result_still_constructs(self) -> None:
        c = CReg("c", 1)
        assert c.sym == "c"


class TestRegisterDeclIsResultRemoved:
    """The `RegisterDecl.is_result` field is gone -> hard `TypeError`."""

    def test_register_decl_is_result_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            RegisterDecl(name="scratch", size=2, is_result=False)

    def test_register_decl_without_is_result_still_constructs(self) -> None:
        decl = RegisterDecl(name="c", size=2)
        assert decl.name == "c"
        with pytest.raises(AttributeError):
            _ = decl.is_result


class TestNoImplicitReturn:
    """No `Return(...)` means no output (the v1 implicit path is gone)."""

    def test_no_return_compiles_to_main_returning_none(self) -> None:
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            qb.X(q[0]),
            Measure(q[0]) > c[0],
        )

        source = SlrConverter(prog).guppy()
        assert "-> None:" in source
        # No implicit `return <result cregs>` was emitted.
        assert "\n    return " not in source

    def test_no_return_hugr_compiles(self) -> None:
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            qb.X(q[0]),
            Measure(q[0]) > c[0],
        )

        package = SlrConverter(prog).hugr()
        assert package is not None

    def test_no_return_selene_fails_fast(self) -> None:
        """No `Return` -> no measurement record -> harness raises clearly."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            qb.X(q[0]),
            Measure(q[0]) > c[0],
        )

        with pytest.raises(ValueError, match="requires an explicit"):
            run_ast_guppy_via_selene(prog, shots=1)

    def test_explicit_return_runs_through_selene(self) -> None:
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            qb.X(q[0]),
            Measure(q[0]) > c[0],
            Return(c),
        )

        records = run_ast_guppy_via_selene(prog, shots=10)
        assert all(record["measurement_0"] == 1 for record in records)
