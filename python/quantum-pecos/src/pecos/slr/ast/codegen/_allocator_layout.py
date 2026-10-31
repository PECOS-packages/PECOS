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

"""Resolve explicit allocator placements independently of declaration order."""

from pecos.slr.ast.nodes import AllocatorDecl, Program


def allocator_offsets(program: Program) -> dict[str, int]:
    """Return offsets within root storage, validating each declared parent chain."""
    declarations = {decl.name: decl for decl in program.declarations if isinstance(decl, AllocatorDecl)}
    if program.allocator is not None:
        declarations[program.allocator.name] = program.allocator
    offsets: dict[str, int] = {}
    visiting: set[str] = set()

    def resolve(name: str) -> int:
        if name in offsets:
            return offsets[name]
        if name not in declarations:
            if program.is_fragment:
                # An undeclared parent belongs to the caller and is a root.
                return 0
            msg = f"Undeclared allocator {name!r}; declare it before generating code."
            raise ValueError(msg)
        if name in visiting:
            msg = f"Allocator {name!r} has a cyclic parent chain."
            raise ValueError(msg)
        visiting.add(name)
        decl = declarations[name]
        if decl.parent is None:
            if decl.offset != 0:
                msg = f"Root allocator {name!r} must have offset zero."
                raise ValueError(msg)
            offset = 0
        else:
            offset = resolve(decl.parent) + decl.offset
            parent = declarations.get(decl.parent)
            if decl.offset < 0 or (parent is not None and decl.offset + decl.capacity > parent.capacity):
                msg = f"Allocator {name!r} extends outside its parent {decl.parent!r}."
                raise ValueError(msg)
        visiting.remove(name)
        offsets[name] = offset
        return offset

    for name in declarations:
        resolve(name)
    return offsets
