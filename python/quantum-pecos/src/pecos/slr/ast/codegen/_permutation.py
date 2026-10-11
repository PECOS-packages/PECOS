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

"""Control-flow checks for codegens which implement Permute as a static relabel."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pecos.slr.ast.nodes import PermuteOp

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pecos.slr.ast.nodes import AstNode


def iter_permutes(node: AstNode) -> Iterator[PermuteOp]:
    """Find permutations even in nested regions and otherwise skipped branches.

    Callers run after BlockCall flattening, so this also covers block bodies.
    """
    if isinstance(node, PermuteOp):
        yield node
    for child in node.children():
        yield from iter_permutes(child)


def reject_permute_in_region(node: AstNode, backend: str) -> None:
    """Reject a static permutation before emitting any part of a runtime region."""
    if any(iter_permutes(node)):
        construct = type(node).__name__.removesuffix("Stmt")
        msg = (
            f"{backend} codegen: Permute in {construct} is not supported: "
            "compile-time relabelling (or unguarded classical swaps) cannot "
            "represent runtime-dependent execution or repeated loop iterations."
        )
        raise NotImplementedError(msg)
