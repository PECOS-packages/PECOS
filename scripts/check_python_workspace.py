#!/usr/bin/env python3
# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0
"""Validate Python workspace metadata."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        print("Python 3.11+ or tomli is required to parse pyproject.toml files", file=sys.stderr)
        sys.exit(2)


PROJECT_NAME_RE = re.compile(r"^\s*([A-Za-z0-9_.-]+)")


def normalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def project_metadata(path: Path, errors: list[str]) -> tuple[str, str] | None:
    data = load_toml(path)
    project = data.get("project")
    if not isinstance(project, dict):
        errors.append(f"{path}: missing [project] table")
        return None

    name = project.get("name")
    version = project.get("version")
    if not isinstance(name, str) or not name:
        errors.append(f"{path}: missing project.name")
        return None
    if not isinstance(version, str) or not version:
        errors.append(f"{path}: missing project.version")
        return None
    return name, version


def dependency_strings(pyproject: dict[str, Any]) -> list[str]:
    project = pyproject.get("project")
    if not isinstance(project, dict):
        return []

    out: list[str] = []
    deps = project.get("dependencies", [])
    if isinstance(deps, list):
        out.extend(dep for dep in deps if isinstance(dep, str))

    optional = project.get("optional-dependencies", {})
    if isinstance(optional, dict):
        for values in optional.values():
            if isinstance(values, list):
                out.extend(dep for dep in values if isinstance(dep, str))
    return out


def dependency_name(requirement: str) -> str | None:
    match = PROJECT_NAME_RE.match(requirement)
    if match is None:
        return None
    return normalize_name(match.group(1))


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    root_pyproject = repo_root / "pyproject.toml"
    python_pyprojects = sorted((repo_root / "python").rglob("pyproject.toml"))
    all_pyprojects = [root_pyproject, *python_pyprojects]

    errors: list[str] = []

    versions: dict[str, str] = {}
    project_paths: dict[str, Path] = {}
    for path in all_pyprojects:
        metadata = project_metadata(path, errors)
        if metadata is None:
            continue
        name, version = metadata
        normalized = normalize_name(name)
        versions[normalized] = version
        project_paths[normalized] = path

    if versions:
        expected_version = versions.get("pecos-workspace") or next(iter(versions.values()))
        for name, version in sorted(versions.items()):
            if version != expected_version:
                errors.append(
                    f"{project_paths[name].relative_to(repo_root)}: version {version} "
                    f"does not match workspace version {expected_version}",
                )

    root_data = load_toml(root_pyproject)
    uv_workspace = root_data.get("tool", {}).get("uv", {}).get("workspace", {})
    uv_members = uv_workspace.get("members", [])
    if not isinstance(uv_members, list) or not all(isinstance(member, str) for member in uv_members):
        errors.append("pyproject.toml: [tool.uv.workspace].members must be a list of strings")
    else:
        actual_members = {path.parent.relative_to(repo_root).as_posix() for path in python_pyprojects}
        configured_members = set(uv_members)
        missing = sorted(actual_members - configured_members)
        extra = sorted(configured_members - actual_members)
        if missing:
            errors.append(f"pyproject.toml: missing uv workspace members: {missing}")
        if extra:
            errors.append(f"pyproject.toml: unknown uv workspace members: {extra}")

    for name, path in sorted(project_paths.items()):
        if name == "pecos-workspace":
            continue
        pyproject = load_toml(path)
        for requirement in dependency_strings(pyproject):
            dep_name = dependency_name(requirement)
            if dep_name is None or dep_name == name or dep_name not in versions:
                continue
            expected = versions[dep_name]
            if f"=={expected}" not in requirement:
                errors.append(
                    f"{path.relative_to(repo_root)}: internal dependency {requirement!r} "
                    f"must pin {dep_name}=={expected}",
                )

    if errors:
        print("Python workspace metadata check failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print(f"Python workspace metadata is consistent across {len(all_pyprojects)} pyproject.toml files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
