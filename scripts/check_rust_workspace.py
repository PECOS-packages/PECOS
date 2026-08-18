#!/usr/bin/env python3
# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0
"""Validate PECOS Rust workspace crate versions.

Crate versions are a single train: `[workspace.package].version` in the root
`Cargo.toml`, inherited by every member through `version.workspace = true`. A member
that hardcodes its version keeps whatever literal it was born with and silently falls
off the train at the next bump, so a literal version is an error unless the member is
listed below as deliberately independent.

This train is separate from the Python distribution versions, which ride
`[project].version` and are checked by `scripts/check_python_workspace.py`.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT_CARGO = REPO_ROOT / "Cargo.toml"
SCRIPT_NAME = Path(__file__).name
INHERITED = {"workspace": True}


@dataclass(frozen=True)
class ForeignTrain:
    """A version train owned by a language binding rather than by the Rust workspace."""

    language: str
    # Another file on that train whose `version` this crate must equal, if one exists.
    version_source: Path | None


# Members deliberately off the crate train: each FFI shim is versioned with the language
# package it ships inside. The Julia binding declares its version in three places, so the
# crate is checked against `PECOS.jl` here and `julia-version-consistency.yml` additionally
# covers `build_tarballs.jl`. The Go binding declares its version only in this crate --
# `pecos_version()` derives the string users see from `CARGO_PKG_VERSION` and Go modules are
# versioned by git tag -- so there is no second declaration to drift against.
INDEPENDENT_MEMBERS = {
    "go/pecos-go-ffi": ForeignTrain("Go", None),
    "julia/pecos-julia-ffi": ForeignTrain("Julia", REPO_ROOT / "julia/PECOS.jl/Project.toml"),
}

# Manifests that are deliberately not workspace members: each declares its own `[workspace]`
# or is a fixture built on its own. Everything else that is tracked must be a member --
# without this, dropping a crate from `[workspace].members` (or adding it to `exclude`)
# would hide it from `cargo metadata` and silently take it off the train.
STANDALONE_MANIFESTS = {
    "crates/pecos-pymatching/tests/pymatching/crates/pecos-chromobius",
    "crates/pecos-pymatching/tests/pymatching/crates/pecos-tesseract",
    "exp/zlup/ffi/zlup-ffi",
    "exp/zlup/fuzz",
    "python/quantum-pecos/tests/docs/rust_crate",
    "scripts/native_bench/bench_pecos",
}


def rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def tracked_manifests() -> list[Path]:
    """Every `Cargo.toml` tracked by git, so build output and vendored trees stay out."""
    git = shutil.which("git")
    if git is None:
        msg = "git not found on PATH"
        raise RuntimeError(msg)

    result = subprocess.run(
        [git, "ls-files", "*Cargo.toml"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [REPO_ROOT / line for line in result.stdout.split()]


def member_manifests() -> list[Path]:
    """Workspace member manifests, as cargo resolves them (globs and path deps included)."""
    cargo = shutil.which("cargo")
    if cargo is None:
        msg = "cargo not found on PATH"
        raise RuntimeError(msg)

    result = subprocess.run(
        [cargo, "metadata", "--no-deps", "--offline", "--format-version", "1"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    metadata = json.loads(result.stdout)
    return sorted(Path(package["manifest_path"]) for package in metadata["packages"])


def main() -> int:
    workspace_version = load_toml(ROOT_CARGO).get("workspace", {}).get("package", {}).get("version")
    if not isinstance(workspace_version, str) or not workspace_version:
        print("error: Cargo.toml: missing [workspace.package].version", file=sys.stderr)
        return 1

    errors: list[str] = []
    inherited = 0
    independent_seen: set[str] = set()

    manifests = member_manifests()
    for manifest in manifests:
        rel_dir = manifest.relative_to(REPO_ROOT).parent.as_posix()
        version = load_toml(manifest).get("package", {}).get("version")

        if (train := INDEPENDENT_MEMBERS.get(rel_dir)) is not None:
            independent_seen.add(rel_dir)
            if version == INHERITED:
                errors.append(
                    f"{rel_dir}/Cargo.toml: inherits the workspace version but is listed as "
                    f"independent in {SCRIPT_NAME}; drop it from INDEPENDENT_MEMBERS",
                )
            elif train.version_source is not None:
                owner_version = load_toml(train.version_source).get("version")
                if version != owner_version:
                    errors.append(
                        f"{rel_dir}/Cargo.toml: version {version!r} is on the {train.language} "
                        f"train but {rel(train.version_source)} declares {owner_version!r}",
                    )
            continue

        if version == INHERITED:
            inherited += 1
        else:
            errors.append(
                f"{rel_dir}/Cargo.toml: [package].version is {version!r}; workspace members "
                f"must use `version.workspace = true` so they track {workspace_version}",
            )

    errors.extend(
        f"{SCRIPT_NAME}: INDEPENDENT_MEMBERS lists {rel_dir}, which is no longer a workspace member"
        for rel_dir in sorted(INDEPENDENT_MEMBERS.keys() - independent_seen)
    )

    # Membership itself is a checked property: `cargo metadata` reports what the workspace
    # currently contains, which says nothing about what it should contain.
    member_dirs = {manifest.relative_to(REPO_ROOT).parent.as_posix() for manifest in manifests}
    tracked_dirs = {manifest.relative_to(REPO_ROOT).parent.as_posix() for manifest in tracked_manifests()}
    unclaimed = tracked_dirs - member_dirs - STANDALONE_MANIFESTS - {"."}
    errors.extend(
        f"{rel_dir}/Cargo.toml: tracked but not a workspace member; add it to [workspace].members "
        f"or to STANDALONE_MANIFESTS in {SCRIPT_NAME} with the reason"
        for rel_dir in sorted(unclaimed)
    )
    errors.extend(
        f"{SCRIPT_NAME}: STANDALONE_MANIFESTS lists {rel_dir}, which has no tracked Cargo.toml"
        for rel_dir in sorted(STANDALONE_MANIFESTS - tracked_dirs)
    )
    errors.extend(
        f"{SCRIPT_NAME}: STANDALONE_MANIFESTS lists {rel_dir}, which is now a workspace member"
        for rel_dir in sorted(STANDALONE_MANIFESTS & member_dirs)
    )

    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1

    print(
        f"Rust workspace versions OK: {inherited} members inherit {workspace_version}, "
        f"{len(independent_seen)} on language trains "
        f"({', '.join(sorted(train.language for train in INDEPENDENT_MEMBERS.values()))})",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
