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
from pecos.slr.ast import RegisterDecl, slr_to_ast
from pecos.slr.ast.codegen.entry_wrapper import RETURN_TAG_NAMESPACE, build_no_arg_entry_wrapper

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

    # Opt-in named return tags. The wrapper emits
    # `result("__pecos_return.<creg>", <creg>)` per returned CReg, so Selene
    # keys outputs by CReg NAME -- immune to internal (non-returned)
    # measurements like the Steane RUS verify. `_shot_records` reads those
    # tags and re-exports the existing public `measurement_N` shape, so all
    # `run_ast_guppy_via_selene` consumers stay unchanged. The production
    # wrapper (default `emit_return_result_tags=False`) is untouched.
    wrapper, info = build_no_arg_entry_wrapper(program, emit_return_result_tags=True)
    # The returned CRegs (explicit `Return(...)`, in listed order) are the
    # source of truth for the public `measurement_N` order. The implicit
    # result-CReg path was removed, so a program with no `Return` has no
    # measurement record at all.
    if info.explicit_return is None:
        msg = (
            "Selene behavioral test requires an explicit `Return(<creg>...)`. "
            "The implicit result-CReg return was removed; a program with "
            "no `Return` compiles to `entry() -> None` and has no measurement "
            "record."
        )
        raise ValueError(msg)
    record_cregs: list[RegisterDecl] = []
    for value in info.explicit_return.values:
        name = value if isinstance(value, str) else getattr(value, "name", None)
        if name in info.all_creg_sizes:
            record_cregs.append(
                RegisterDecl(name=name, size=info.all_creg_sizes[name]),
            )
    if not record_cregs:
        # Strict: an explicit `Return(...)`
        # with no CRegs (e.g. `Return(q)`) has no measurement record --
        # fail loudly rather than silently mis-shape the result table.
        msg = (
            "Selene behavioral test requires at least one returned CReg "
            "(explicit `Return(<creg>...)`). Returning only QRegs/values "
            "yields no measurement record. Declare CRegs and write "
            "measurement bits into them."
        )
        raise ValueError(msg)

    full_source = ast_source + wrapper

    entry_func = _import_entry_function(full_source)
    total_qubits = sum(info.allocator_sizes.values())

    result = sim(Guppy(entry_func)).classical(selene_engine()).qubits(max(total_qubits, 1)).seed(seed).run(shots)

    return _shot_records(result, record_cregs)


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


def _shot_records(result: pecos_rslib.ShotVec, record_cregs: list[RegisterDecl]) -> list[dict[str, int]]:
    """Re-export the named `__pecos_return.<creg>` tags as the public shape.

    The wrapper emits `result("__pecos_return.<creg>", <creg>)` per returned
    CReg, so Selene's `to_dict()` keys outputs by CReg name and each
    shot value is a list of that CReg's bits in declaration order. We flatten
    the returned CRegs (in `Return(...)` order) into the historical public
    shape `{"measurement_0": .., "measurement_1": .., ...}` so existing
    `run_ast_guppy_via_selene` consumers are unchanged -- but now reading the
    correct bits (immune to internal, non-returned measurements).
    """
    raw = result.to_dict() if hasattr(result, "to_dict") else result
    if not isinstance(raw, dict):
        msg = f"Unexpected Selene result shape: {type(raw).__name__}"
        raise TypeError(msg)

    tag_keys = [f"{RETURN_TAG_NAMESPACE}.{decl.name}" for decl in record_cregs]
    missing = [k for k in tag_keys if k not in raw]
    if missing:
        msg = (
            f"Selene result is missing return tags {missing}; got keys "
            f"{sorted(raw)}. Expected the opt-in wrapper to emit "
            f'`result("{RETURN_TAG_NAMESPACE}.<creg>", <creg>)` per '
            "returned CReg."
        )
        raise KeyError(msg)

    shot_count = len(raw[tag_keys[0]]) if tag_keys else 0
    records: list[dict[str, int]] = []
    for shot_idx in range(shot_count):
        record: dict[str, int] = {}
        counter = 0
        for decl, key in zip(record_cregs, tag_keys, strict=True):
            shot_val = raw[key][shot_idx]
            # Selene shapes a size-1 CReg result tag as a scalar int per
            # shot, and a size>1 CReg as a list of `size` ints per shot.
            # Be explicit and fail LOUD on any other shape -- a silent
            # mis-count is the exact bug class this guard prevents (do not let an
            # unexpected Selene type, e.g. a numpy array/generator, be
            # silently wrapped as one bit).
            if isinstance(shot_val, (list, tuple)):
                bits = list(shot_val)
            elif isinstance(shot_val, int):  # bool is an int subclass
                bits = [shot_val]
            else:
                msg = (
                    f"Return tag {key!r} shot {shot_idx}: unexpected Selene "
                    f"value shape {type(shot_val).__name__} ({shot_val!r}); "
                    "expected a scalar int (size-1 CReg) or a list of ints "
                    "(size>1 CReg). Selene's result-tag output shape may "
                    "have changed -- update _shot_records deliberately."
                )
                raise TypeError(msg)
            if len(bits) != decl.size:
                msg = (
                    f"Return tag {key!r} shot {shot_idx} has {len(bits)} bits, "
                    f"expected {decl.size} (CReg {decl.name!r})."
                )
                raise ValueError(msg)
            for bit in bits:
                record[f"measurement_{counter}"] = int(bit)
                counter += 1
        records.append(record)
    return records
