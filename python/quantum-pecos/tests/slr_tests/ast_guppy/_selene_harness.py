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

"""Selene behavioral test harness for the AST -> Guppy v1 emitter.

Compile-only tests via `_harness.assert_ast_guppy_compiles` prove
linearity + HUGR construction. They do not prove that observable
outcomes match SLR intent (wrong CReg ordering, wrong permutation
mapping, swapped reset/discard semantics all type-check).

This harness runs an SLR program through the AST path and executes
the result via Selene
(`pecos.sim(pecos.Guppy(entry)).classical(pecos.selene_engine())`),
returning per-shot measurement bits as a list of dicts.

Behavioral assertions on the result table are the v1 oracle.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from pecos import Guppy, selene_engine, sim
from pecos.slr import SlrConverter
from pecos.slr.ast import AllocatorDecl, RegisterDecl, slr_to_ast

if TYPE_CHECKING:
    import pecos_rslib
    from pecos.slr import Block


_DEFAULT_SHOTS = 100
_DEFAULT_SEED = 42


def run_ast_guppy_via_selene(
    slr_program: Block,
    *,
    shots: int = _DEFAULT_SHOTS,
    seed: int = _DEFAULT_SEED,
) -> list[dict[str, int]]:
    """Run an SLR program through the AST -> Guppy -> Selene path.

    Returns a list of per-shot measurement records. Each record is a
    `dict[str, int]` keyed by Guppy result names ("measurement_0",
    "measurement_1", ...) with bit values 0 or 1.

    The AST-emitted `main(q: array[qubit, N] @ owned) -> ...` is
    wrapped in a no-arg `entry()` that allocates the qubits, calls
    main, and returns the result CRegs unpacked as a flat tuple of
    bools. Selene's Guppy adapter requires a no-arg entrypoint.
    """
    ast_source = SlrConverter(slr_program).guppy()
    program = slr_to_ast(slr_program)

    allocator_sizes = _allocator_sizes(program)
    cregs = _result_cregs(program)
    if not cregs:
        msg = (
            "Behavioral test requires at least one result CReg. "
            "v1 acceptance Selene tests should declare CRegs and write measurement bits into them."
        )
        raise ValueError(msg)

    wrapper = _build_entry_wrapper(allocator_sizes, cregs)
    full_source = ast_source + wrapper

    entry_func = _import_entry_function(full_source)
    total_qubits = sum(allocator_sizes.values())

    result = (
        sim(Guppy(entry_func))
        .classical(selene_engine())
        .qubits(max(total_qubits, 1))
        .seed(seed)
        .run(shots)
    )

    return _shot_records(result, _result_keys(cregs))


def _allocator_sizes(program: object) -> dict[str, int]:
    """Map root allocator name -> capacity. Same iteration order as the emitter."""
    sizes: dict[str, int] = {}
    for decl in getattr(program, "declarations", ()):
        if isinstance(decl, AllocatorDecl) and decl.parent is None:
            sizes.setdefault(decl.name, decl.capacity)
    if getattr(program, "allocator", None) is not None:
        decl = program.allocator
        if isinstance(decl, AllocatorDecl) and decl.parent is None:
            sizes.setdefault(decl.name, decl.capacity)
    return sizes


def _result_cregs(program: object) -> list[RegisterDecl]:
    """Return result-flagged CReg declarations in declaration order."""
    return [
        decl
        for decl in getattr(program, "declarations", ())
        if isinstance(decl, RegisterDecl) and decl.is_result
    ]


def _result_keys(cregs: list[RegisterDecl]) -> list[str]:
    """Names the Selene runtime uses for each bit in the entry tuple.

    Selene emits "measurement_0", "measurement_1", ... in tuple-position order.
    The wrapper returns CReg bits in declaration order, so we generate the keys
    in that same order.
    """
    keys: list[str] = []
    counter = 0
    for decl in cregs:
        for _ in range(decl.size):
            keys.append(f"measurement_{counter}")
            counter += 1
    return keys


def _build_entry_wrapper(
    allocator_sizes: dict[str, int],
    cregs: list[RegisterDecl],
) -> str:
    """Generate the no-arg `entry()` wrapper that Selene needs.

    Wrapper allocates each root allocator's qubits, calls the AST-
    emitted `main(...)` with them, and unpacks the returned CReg
    arrays into a flat tuple of bools.
    """
    if not cregs:
        msg = "Refusing to build wrapper without result CRegs"
        raise ValueError(msg)

    bool_count = sum(decl.size for decl in cregs)
    return_ann = "tuple[bool]" if bool_count == 1 else "tuple[" + ", ".join(["bool"] * bool_count) + "]"

    body_lines: list[str] = [
        f"    {allocator} = array(qubit() for _ in range({size}))" for allocator, size in allocator_sizes.items()
    ]

    call_args = ", ".join(allocator_sizes.keys())
    if len(cregs) == 1:
        body_lines.append(f"    {cregs[0].name} = main({call_args})")
    else:
        result_names = ", ".join(decl.name for decl in cregs)
        body_lines.append(f"    {result_names} = main({call_args})")

    return_parts: list[str] = []
    for decl in cregs:
        return_parts.extend(f"{decl.name}[{i}]" for i in range(decl.size))
    return_expr = ", ".join(return_parts)
    if len(return_parts) == 1:
        return_expr += ","
    body_lines.append(f"    return {return_expr}")

    body = "\n".join(body_lines)
    return f"\n\n@guppy\ndef entry() -> {return_ann}:\n{body}\n"


def _import_entry_function(source: str) -> object:
    """Write source to a temp file, import, and return the `entry` callable."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        path = Path(f.name)
        f.write(source)

    spec = importlib.util.spec_from_file_location(f"_selene_test_{path.stem}", path)
    if spec is None or spec.loader is None:
        msg = f"Failed to create import spec for generated source at {path}"
        raise RuntimeError(msg)

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    entry = getattr(module, "entry", None)
    if entry is None:
        msg = "Wrapped Guppy source has no `entry` function"
        raise RuntimeError(msg)
    return entry


def _shot_records(result: pecos_rslib.ShotVec, keys: list[str]) -> list[dict[str, int]]:
    """Convert ShotVec to a list of per-shot measurement records."""
    raw = result.to_dict() if hasattr(result, "to_dict") else result
    if not isinstance(raw, dict):
        msg = f"Unexpected Selene result shape: {type(raw).__name__}"
        raise TypeError(msg)

    shot_count = len(next(iter(raw.values()))) if raw else 0
    records: list[dict[str, int]] = []
    for shot_idx in range(shot_count):
        record: dict[str, int] = {}
        for key in keys:
            record[key] = int(raw[key][shot_idx])
        records.append(record)
    return records
