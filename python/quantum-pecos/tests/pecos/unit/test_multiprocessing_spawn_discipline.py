# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

"""Prevent platform-default or explicit-fork multiprocessing in PECOS source."""

from __future__ import annotations

import ast
from pathlib import Path

_SOURCE_ROOT = Path(__file__).resolve().parents[3] / "src" / "pecos"
_ALLOWLIST: frozenset[Path] = frozenset()


def _qualified_name(expression: ast.expr) -> str | None:
    """Return a dotted name for a simple name or attribute expression."""
    if isinstance(expression, ast.Name):
        return expression.id
    if isinstance(expression, ast.Attribute):
        prefix = _qualified_name(expression.value)
        if prefix is not None:
            return f"{prefix}.{expression.attr}"
    return None


def _string_argument(call: ast.Call, keyword_name: str) -> str | None:
    """Return a call's first positional or selected keyword string argument."""
    argument = (
        call.args[0]
        if call.args
        else next(
            (keyword.value for keyword in call.keywords if keyword.arg == keyword_name),
            None,
        )
    )
    if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
        return argument.value
    return None


def _find_fork_hazards(source: str) -> list[tuple[int, str]]:
    """Find fork-prone process creation in parsed Python source."""
    tree = ast.parse(source)
    source_lines = source.splitlines()
    multiprocessing_aliases = {"multiprocessing", "mp"}
    os_aliases = {"os"}
    multiprocessing_imports: dict[str, str] = {}
    process_pool_executor_names = {"ProcessPoolExecutor"}
    hazards: set[tuple[int, str]] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "multiprocessing":
                    multiprocessing_aliases.add(alias.asname or alias.name)
                elif alias.name == "os":
                    os_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module == "multiprocessing":
                for alias in node.names:
                    local_name = alias.asname or alias.name
                    multiprocessing_imports[local_name] = alias.name
                    if alias.name in {"Pool", "Process"}:
                        hazards.add((node.lineno, f"from multiprocessing import {alias.name}"))
            elif node.module == "concurrent.futures":
                for alias in node.names:
                    if alias.name == "ProcessPoolExecutor":
                        process_pool_executor_names.add(alias.asname or alias.name)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function_name = _qualified_name(node.func)
        if function_name is None:
            continue
        name_parts = function_name.split(".")
        root_name = name_parts[0]
        leaf_name = name_parts[-1]
        source_line = source_lines[node.lineno - 1].lstrip() if source_lines else ""
        definition_line = source_line.startswith(("def ", "async def ", "class "))

        imported_name = multiprocessing_imports.get(function_name)
        is_multiprocessing_member = root_name in multiprocessing_aliases and len(name_parts) == 2
        if leaf_name == "get_context" or imported_name == "get_context":
            method = _string_argument(node, "method")
            if not node.args and method is None:
                hazards.add((node.lineno, "get_context() uses the platform default"))
            elif method == "fork":
                hazards.add((node.lineno, 'get_context("fork")'))
        elif leaf_name == "set_start_method" or imported_name == "set_start_method":
            if _string_argument(node, "method") == "fork":
                hazards.add((node.lineno, 'set_start_method("fork")'))
        elif root_name in os_aliases and len(name_parts) == 2 and leaf_name == "fork":
            hazards.add((node.lineno, "os.fork()"))
        elif is_multiprocessing_member and leaf_name in {"Pool", "Process"}:
            hazards.add((node.lineno, f"{root_name}.{leaf_name}()"))
        elif isinstance(node.func, ast.Name) and not definition_line:
            if function_name == "Pool" or imported_name == "Pool":
                hazards.add((node.lineno, f"bare or imported {function_name}()"))
            elif imported_name == "Process":
                hazards.add((node.lineno, f"imported {function_name}()"))

        is_process_pool_executor = leaf_name == "ProcessPoolExecutor" or function_name in process_pool_executor_names
        if is_process_pool_executor and not any(keyword.arg == "mp_context" for keyword in node.keywords):
            hazards.add((node.lineno, "ProcessPoolExecutor() without mp_context"))

    return sorted(hazards)


def test_hazard_detector_catches_variants_without_safe_false_positives() -> None:
    """Pin supported syntax variants and the intentional precision exclusions."""
    hazardous_sources = [
        'multiprocessing.get_context("fork", force=True)',
        "mp.get_context( 'fork' , extra=True)",
        'multiprocessing.set_start_method("fork", force=True)',
        "multiprocessing.get_context()",
        "os.fork()",
        "multiprocessing.Process(target=work)",
        "mp.Pool(processes=2)",
        "Pool(processes=2)",
        "from multiprocessing import Pool as WorkerPool",
        "from multiprocessing import Process as WorkerProcess",
        "ProcessPoolExecutor(max_workers=2)",
        "ProcessPoolExecutor(\n    max_workers=2,\n)",
    ]
    for source in hazardous_sources:
        assert _find_fork_hazards(source), source

    safe_sources = [
        "# multiprocessing.Pool(processes=2)",
        "def Pool(processes):\n    return processes",
        "class ProcessPoolExecutor:\n    pass",
        'multiprocessing.get_context("spawn").Pool(processes=2)',
        "ProcessPoolExecutor(max_workers=2, mp_context=spawn_context)",
        "ProcessPoolExecutor(\n    max_workers=2,\n    mp_context=spawn_context,\n)",
    ]
    for source in safe_sources:
        assert not _find_fork_hazards(source), source


def test_pecos_source_avoids_fork_based_multiprocessing() -> None:
    """Require explicit non-fork multiprocessing throughout PECOS Python source."""
    assert _SOURCE_ROOT.is_dir(), f"PECOS source root does not exist: {_SOURCE_ROOT}"
    source_paths = sorted(_SOURCE_ROOT.rglob("*.py"))
    assert source_paths, f"No Python source files found under {_SOURCE_ROOT}"

    violations = []
    for source_path in source_paths:
        relative_path = source_path.relative_to(_SOURCE_ROOT)
        if relative_path in _ALLOWLIST:
            continue
        source = source_path.read_text(encoding="utf-8")
        for line_number, hazard in _find_fork_hazards(source):
            source_line = source.splitlines()[line_number - 1].strip()
            violations.append(f"{relative_path}:{line_number}: {hazard}: {source_line}")

    assert not violations, "Fork-unsafe multiprocessing usage found:\n" + "\n".join(violations)
