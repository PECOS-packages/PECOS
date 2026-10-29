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

"""Static For range resolution shared by the unrolling backends."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pecos.slr.ast.nodes import LiteralExpr

if TYPE_CHECKING:
    from pecos.slr.ast.nodes import ForStmt


def _static_int_bound(expr: object, which: str, backend: str) -> int:
    """Resolve converter-produced integer literals, excluding booleans."""
    if isinstance(expr, LiteralExpr) and isinstance(expr.value, int) and not isinstance(expr.value, bool):
        return expr.value
    msg = (
        f"{backend} codegen: For loop {which} bound is not a static integer "
        f"({expr!r}); only fixed-bound `For(i, <int>, <int>)` is supported."
    )
    raise NotImplementedError(msg)


def static_for_range(node: ForStmt, backend: str) -> range:
    """Resolve all bounds before iterating, including an empty range.

    The SLR converter wraps bounds in LiteralExpr. This preserves the Stim
    unroller's Python range semantics, including negative steps, without
    attempting to evaluate dynamic expressions or bind the loop variable.
    """
    start = _static_int_bound(node.start, "start", backend)
    stop = _static_int_bound(node.stop, "stop", backend)
    step = 1 if node.step is None else _static_int_bound(node.step, "step", backend)
    if step == 0:
        msg = f"{backend} codegen: For loop step is 0; only a non-zero static step is supported."
        raise NotImplementedError(msg)
    return range(start, stop, step)
