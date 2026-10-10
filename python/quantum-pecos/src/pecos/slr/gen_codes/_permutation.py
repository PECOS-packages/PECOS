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

"""Permutation checks shared by the deprecated direct SLR generators."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pecos.slr.misc import Permute
from pecos.slr.vars import CReg, Reg

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pecos.slr import Block


def iter_permutes(block: Block) -> Iterator[Permute]:
    """Search nested blocks and both If branches, including skipped bodies."""
    for op in block.ops:
        if isinstance(op, Permute):
            yield op
        elif hasattr(op, "ops"):
            yield from iter_permutes(op)
    otherwise = getattr(block, "else_block", None)
    if otherwise is not None:
        yield from iter_permutes(otherwise)


def is_classical_permute(op: Permute) -> bool:
    """Whether every source and target is a classical register or bit."""

    def classical(refs: Reg | list) -> bool:
        if isinstance(refs, Reg):
            return isinstance(refs, CReg)
        return all(isinstance(getattr(ref, "reg", ref), CReg) for ref in refs)

    return classical(op.elems_i) and classical(op.elems_f)


def reject_permute_in_region(block: Block, backend: str, *, classical_runtime: bool = False) -> None:
    """Reject relabels in a region whose execution cannot be resolved statically.

    Legacy QIR emits actual classical swaps under its LLVM branch, so it
    can retain classical permutations in If bodies.
    """
    for op in iter_permutes(block):
        if classical_runtime and is_classical_permute(op):
            continue
        msg = (
            f"{backend}: Permute in {type(block).__name__} is not supported: "
            "compile-time relabelling (or unguarded classical swaps) cannot "
            "represent runtime-dependent execution or repeated loop iterations."
        )
        raise NotImplementedError(msg)
