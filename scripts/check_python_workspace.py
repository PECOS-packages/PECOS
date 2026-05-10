#!/usr/bin/env python3
# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0
"""Validate PECOS Python workspace metadata.

This check is intentionally narrower than a full packaging linter. It guards
the invariants that tend to drift in this repository: package versions,
workspace membership, internal dependency pins, and uv workspace sources.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        print("error: Python 3.11+ or the 'tomli' package is required", file=sys.stderr)
        sys.exit(2)


REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT_PYPROJECT = REPO_ROOT / "pyproject.toml"
DEPENDENCY_NAME_RE = re.compile(r"^\s*([A-Za-z0-9_.-]+)")


@dataclass(frozen=True)
class Package:
    path: Path
    rel_dir: str
    name: str
    normalized_name: str
    version: str
    data: dict[str, Any]


def normalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def load_package(path: Path, errors: list[str]) -> Package | None:
    data = load_toml(path)
    project = data.get("project")
    if not isinstance(project, dict):
        fail(errors, f"{rel(path)}: missing [project] table")
        return None

    name = project.get("name")
    version = project.get("version")
    if not isinstance(name, str) or not name:
        fail(errors, f"{rel(path)}: missing [project].name")
        return None
    if not isinstance(version, str) or not version:
        fail(errors, f"{rel(path)}: missing [project].version")
        return None

    return Package(
        path=path,
        rel_dir=rel(path.parent),
        name=name,
        normalized_name=normalize_name(name),
        version=version,
        data=data,
    )


def dependency_name(requirement: str) -> str | None:
    match = DEPENDENCY_NAME_RE.match(requirement)
    if match is None:
        return None
    return normalize_name(match.group(1))


def has_exact_version_pin(requirement: str, version: str) -> bool:
    return re.search(rf"(^|[^=!<>~])==\s*{re.escape(version)}(\s*(;|,|$))", requirement) is not None


def iter_dependency_lists(data: dict[str, Any]) -> list[tuple[str, list[Any]]]:
    lists: list[tuple[str, list[Any]]] = []
    project = data.get("project", {})
    if isinstance(project, dict):
        dependencies = project.get("dependencies", [])
        if isinstance(dependencies, list):
            lists.append(("[project].dependencies", dependencies))

        optional = project.get("optional-dependencies", {})
        if isinstance(optional, dict):
            for extra, deps in sorted(optional.items()):
                if isinstance(deps, list):
                    lists.append((f"[project.optional-dependencies].{extra}", deps))

    dependency_groups = data.get("dependency-groups", {})
    if isinstance(dependency_groups, dict):
        for group, deps in sorted(dependency_groups.items()):
            if isinstance(deps, list):
                lists.append((f"[dependency-groups].{group}", deps))

    return lists


def internal_dependencies(package: Package, workspace_names: set[str], errors: list[str]) -> set[str]:
    internal: set[str] = set()
    for section, deps in iter_dependency_lists(package.data):
        for dep in deps:
            if not isinstance(dep, str):
                fail(errors, f"{rel(package.path)}: {section} contains non-string dependency {dep!r}")
                continue
            dep_name = dependency_name(dep)
            if dep_name is None or dep_name not in workspace_names or dep_name == package.normalized_name:
                continue
            internal.add(dep_name)
            if not has_exact_version_pin(dep, package.version):
                fail(
                    errors,
                    f"{rel(package.path)}: {section} dependency {dep!r} must pin "
                    f"workspace package version =={package.version}",
                )
    return internal


def workspace_sources(package: Package, errors: list[str]) -> set[str]:
    tool = package.data.get("tool", {})
    uv = tool.get("uv", {}) if isinstance(tool, dict) else {}
    sources = uv.get("sources", {}) if isinstance(uv, dict) else {}
    if not isinstance(sources, dict):
        fail(errors, f"{rel(package.path)}: [tool.uv.sources] must be a table")
        return set()

    names: set[str] = set()
    for name, source in sources.items():
        normalized = normalize_name(name)
        if not isinstance(source, dict) or source.get("workspace") is not True:
            continue
        names.add(normalized)
    return names


def check_cuda_extra_group(root_data: dict[str, Any], errors: list[str]) -> None:
    project = root_data.get("project", {})
    optional = project.get("optional-dependencies", {}) if isinstance(project, dict) else {}
    dependency_groups = root_data.get("dependency-groups", {})
    cuda_extra = optional.get("cuda") if isinstance(optional, dict) else None
    cuda_group = dependency_groups.get("cuda") if isinstance(dependency_groups, dict) else None

    if cuda_extra is None or cuda_group is None:
        return
    if cuda_extra != cuda_group:
        fail(
            errors,
            "pyproject.toml: [project.optional-dependencies].cuda and [dependency-groups].cuda must stay identical",
        )


def main() -> int:
    errors: list[str] = []

    root = load_package(ROOT_PYPROJECT, errors)
    package_paths = sorted((REPO_ROOT / "python").rglob("pyproject.toml"))
    packages = [pkg for path in package_paths if (pkg := load_package(path, errors)) is not None]
    if root is None:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1

    all_packages = [root, *packages]
    workspace_names = {pkg.normalized_name for pkg in all_packages}

    for pkg in all_packages:
        if pkg.version != root.version:
            fail(
                errors,
                f"{rel(pkg.path)}: version {pkg.version!r} does not match root version {root.version!r}",
            )

    root_tool = root.data.get("tool", {})
    root_uv = root_tool.get("uv", {}) if isinstance(root_tool, dict) else {}
    workspace = root_uv.get("workspace", {}) if isinstance(root_uv, dict) else {}
    members = workspace.get("members") if isinstance(workspace, dict) else None
    expected_members = sorted(pkg.rel_dir for pkg in packages)
    if not isinstance(members, list) or any(not isinstance(member, str) for member in members):
        fail(errors, "pyproject.toml: [tool.uv.workspace].members must be a string list")
    elif sorted(members) != expected_members:
        fail(
            errors,
            "pyproject.toml: [tool.uv.workspace].members does not match Python package directories\n"
            f"  expected: {expected_members}\n"
            f"  found:    {sorted(members)}",
        )

    check_cuda_extra_group(root.data, errors)

    for pkg in all_packages:
        internal = internal_dependencies(pkg, workspace_names, errors)
        sources = workspace_sources(pkg, errors)
        missing_sources = sorted(internal - sources)
        extra_sources = sorted((sources & workspace_names) - internal)
        if missing_sources:
            fail(
                errors,
                f"{rel(pkg.path)}: missing [tool.uv.sources] workspace entries for {missing_sources}",
            )
        if extra_sources:
            fail(
                errors,
                f"{rel(pkg.path)}: unused internal [tool.uv.sources] workspace entries {extra_sources}",
            )

    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1

    print(
        f"Python workspace metadata OK: {len(packages)} packages, "
        f"version {root.version}, {len(expected_members)} uv workspace members",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
