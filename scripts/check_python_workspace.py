#!/usr/bin/env python3
# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0
"""Validate PECOS Python workspace metadata.

This check is intentionally narrower than a full packaging linter. It guards
the invariants that tend to drift in this repository: package versions,
workspace membership, Python-version metadata, release ABI targets, internal
dependency pins, and uv workspace sources.
"""

from __future__ import annotations

import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT_PYPROJECT = REPO_ROOT / "pyproject.toml"
DEPENDENCY_NAME_RE = re.compile(r"^\s*([A-Za-z0-9_.-]+)")
MINIMUM_PYTHON = "3.12"
EXPECTED_PYTHON_CLASSIFIERS = {"3", "3.12", "3.13", "3.14"}
ABI3_MANIFESTS = (
    REPO_ROOT / "python/pecos-rslib/Cargo.toml",
    REPO_ROOT / "python/pecos-rslib-llvm/Cargo.toml",
    REPO_ROOT / "python/pecos-rslib-cuda/Cargo.toml",
    REPO_ROOT / "python/pecos-rslib-exp/Cargo.toml",
    REPO_ROOT / "exp/zluppy/Cargo.toml",
)
RELEASE_WORKFLOW = REPO_ROOT / ".github/workflows/python-release.yml"


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
    if not isinstance(optional, dict) or not isinstance(dependency_groups, dict):
        return

    # The CUDA stack is split by toolkit major; each major must be defined as BOTH a
    # `[project.optional-dependencies]` extra AND a matching `[dependency-groups]`
    # group, kept identical, so `pip install .[cuda13]` and `uv sync --group cuda13`
    # resolve to the same packages. Both majors are required: the `pecos` CLI selects
    # cuda12 or cuda13 by the detected toolkit (cuda_python_group), so deleting either
    # the extra or the group breaks CUDA setup on the corresponding host -- a missing
    # side is an error, not a silent skip.
    for cuda_name in ("cuda12", "cuda13"):
        cuda_extra = optional.get(cuda_name)
        cuda_group = dependency_groups.get(cuda_name)
        if cuda_extra is None:
            fail(
                errors,
                f"pyproject.toml: missing required [project.optional-dependencies].{cuda_name}",
            )
        if cuda_group is None:
            fail(
                errors,
                f"pyproject.toml: missing required [dependency-groups].{cuda_name}",
            )
        if cuda_extra is not None and cuda_group is not None and cuda_extra != cuda_group:
            fail(
                errors,
                f"pyproject.toml: [project.optional-dependencies].{cuda_name} and "
                f"[dependency-groups].{cuda_name} must stay identical",
            )


def check_python_floor(path: Path, data: dict[str, Any], errors: list[str]) -> None:
    """Ensure Python package and wheel metadata agree on the supported floor."""
    project = data.get("project", {})
    if not isinstance(project, dict):
        return

    requires_python = project.get("requires-python")
    normalized_requirement = re.sub(r"\s+", "", requires_python) if isinstance(requires_python, str) else ""
    if normalized_requirement != f">={MINIMUM_PYTHON}":
        fail(
            errors,
            f"{rel(path)}: [project].requires-python must be >={MINIMUM_PYTHON!s}",
        )

    classifiers = project.get("classifiers", [])
    if not isinstance(classifiers, list):
        fail(errors, f"{rel(path)}: [project].classifiers must be a list")
        return
    python_classifiers: set[str] = set()
    for classifier in classifiers:
        if not isinstance(classifier, str):
            continue
        version = classifier.removeprefix("Programming Language :: Python :: ")
        if classifier.startswith("Programming Language :: Python :: ") and (
            version == "3" or re.fullmatch(r"\d+\.\d+", version)
        ):
            python_classifiers.add(version)
    if python_classifiers and python_classifiers != EXPECTED_PYTHON_CLASSIFIERS:
        fail(
            errors,
            f"{rel(path)}: Python classifiers must be {sorted(EXPECTED_PYTHON_CLASSIFIERS)!r}",
        )


def check_release_python_abi(errors: list[str]) -> None:
    """Keep ABI3, cibuildwheel, and the release smoke test on one floor."""
    expected_abi3 = f"abi3-py{MINIMUM_PYTHON.replace('.', '')}"
    for manifest in ABI3_MANIFESTS:
        abi3_features = set(re.findall(r"abi3-py\d+", manifest.read_text()))
        if abi3_features != {expected_abi3}:
            fail(
                errors,
                f"{rel(manifest)}: ABI3 feature must be exactly {expected_abi3!r}, found {sorted(abi3_features)!r}",
            )

    workflow = RELEASE_WORKFLOW.read_text()
    cibw_target = f'CIBW_BUILD: "cp{MINIMUM_PYTHON.replace(".", "")}-*"'
    if workflow.count(cibw_target) != 2:
        fail(errors, f"{rel(RELEASE_WORKFLOW)}: expected two {cibw_target!r} release-wheel targets")

    python_abi = MINIMUM_PYTHON.replace(".", "")
    smoke_interpreter = f"/opt/python/cp{python_abi}-cp{python_abi}/bin/python"
    if workflow.count(smoke_interpreter) != 4:
        fail(errors, f"{rel(RELEASE_WORKFLOW)}: expected four {smoke_interpreter!r} smoke commands")


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

    project_paths = [
        ROOT_PYPROJECT,
        *package_paths,
        REPO_ROOT / "exp/zluppy/pyproject.toml",
        REPO_ROOT / "exp/zlup/pyproject.toml",
    ]
    for path in project_paths:
        check_python_floor(path, load_toml(path), errors)
    check_release_python_abi(errors)

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
