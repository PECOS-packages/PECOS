# Copyright 2023 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
from __future__ import annotations

import re
from typing import TYPE_CHECKING

from pecos.slr.block import Block
from pecos.slr.fund import Statement
from pecos.slr.vars import Bit, CReg, SymbolicBit

if TYPE_CHECKING:
    from pecos.slr.vars import Elem, QReg, Qubit, Reg

_TAG_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class Barrier(Statement):
    def __init__(self, *qregs: QReg | tuple[QReg] | Qubit) -> None:
        self.qregs = qregs


class Comment(Statement):
    """A comment for human readability of output qasm."""

    def __init__(self, *txt, space: bool = True, newline: bool = True) -> None:
        self.space = space
        self.newline = newline
        self.txt = "\n".join(txt)


class Parallel(Block):
    """A block that indicates the contained statements can be executed in parallel.

    This is a hint to the compiler/simulator that the operations within this block
    are independent and can be executed simultaneously.
    """

    def __init__(self, *statements: Statement) -> None:
        super().__init__()
        self.extend(*statements)


class Permute(Statement):
    """Permutes the indices that the elements of the register so that Reg[i] now refers to Reg[j]."""

    def __init__(
        self,
        elems_i: list[Elem] | Reg,
        elems_f: list[Elem] | Reg,
        *,
        comment: bool = True,
    ) -> None:
        self.elems_i = elems_i
        self.elems_f = elems_f
        self.comment = comment


class Return(Statement):
    """Explicitly declares which variables a block returns.

    This operation is similar to Python's return statement and works in conjunction with
    the block_returns annotation (similar to Python's -> type annotation).

    Example:
        from pecos.slr import Block, QReg
        from pecos.slr.types import Array, QubitType
        from pecos.slr.misc import Return

        class MyBlock(Block):
            # Type annotation (like -> Type)
            block_returns = (Array[QubitType, 2], Array[QubitType, 7])

            def __init__(self, data, ancilla):
                super().__init__()
                # ... operations ...
                # Explicit return statement
                self.extend(Return(ancilla, data))
    """

    def __init__(self, *return_vars) -> None:
        """Initialize Return operation with variables to return.

        Args:
            *return_vars: Variables to return, in order. Can be QReg, Qubit, Bit, or other variables.
        """
        self.return_vars = return_vars


class Print(Statement):
    """Emit an intermediate streamed value at the call site.

    Lowers to Guppy's ``result(name, value)``. Scope-orthogonal side-effect:
    does not touch block ownership or compile-time return shape.

    The emitted Guppy tag is ``f"{namespace}.{tag}"``. Default namespace is
    ``"result"``. If ``tag`` is not provided it is derived from ``value``'s
    name (CReg name, or ``f"{reg}_{index}"`` for a Bit).

    Args:
        value: A CReg or Bit (CReg element). Only these are supported.
            Expression values (e.g. ``c[0] ^ c[1]``), SymbolicBit, and other
            types are rejected at construction time.
        tag: Explicit tag string overriding the derived name. Must match
            ``[A-Za-z_][A-Za-z0-9_]*`` (Python identifier rules).
        namespace: Tag prefix. Default ``"result"``. Must match
            ``[A-Za-z_][A-Za-z0-9_]*``.

    Example:
        Main(
            c := CReg("c", 2),
            ...,
            Print(c),                       # tag "result.c"
            Print(c[0], tag="first"),       # tag "result.first"
            Print(c, namespace="debug"),    # tag "debug.c"
        )
    """

    def __init__(self, value, *, tag: str | None = None, namespace: str = "result") -> None:
        """Construction-time validation per `v2-print.md`.

        Validates value type, tag/namespace character rules, and derives the
        default tag from the value's name. AST/Guppy-level checks (path-
        signature consistency in If/Elif, inline-CReg definite-assignment)
        run later during emission.
        """
        if isinstance(value, SymbolicBit):
            msg = "Print does not support SymbolicBit (LoopVar-indexed) values."
            raise TypeError(msg)
        if not isinstance(value, (CReg, Bit)):
            msg = (
                f"Print(value, ...) requires a CReg or Bit value; got {type(value).__name__}. "
                "Expression values (e.g. c[0] ^ c[1]) are deferred and must be passed with explicit tag=...; "
                "Only CReg and Bit values are supported."
            )
            raise TypeError(msg)

        if not _TAG_RE.match(namespace):
            msg = (
                f"Print namespace {namespace!r} must match [A-Za-z_][A-Za-z0-9_]* "
                "(Python identifier rules). The dot is reserved as the namespace-tag separator."
            )
            raise ValueError(msg)

        if tag is None:
            tag = self._derive_tag(value)
        if not _TAG_RE.match(tag):
            msg = (
                f"Print tag {tag!r} must match [A-Za-z_][A-Za-z0-9_]* "
                "(Python identifier rules). The dot is reserved as the namespace-tag separator. "
                "Tags derived from non-identifier register names are rejected; pass tag=... explicitly."
            )
            raise ValueError(msg)

        self.value = value
        self.tag = tag
        self.namespace = namespace

    @staticmethod
    def _derive_tag(value: CReg | Bit) -> str:
        if isinstance(value, CReg):
            return value.sym
        if isinstance(value, Bit):
            return f"{value.reg.sym}_{value.index}"
        msg = f"Cannot derive Print tag from {type(value).__name__}"
        raise TypeError(msg)
