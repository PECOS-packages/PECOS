#!/usr/bin/env python3
# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0
"""Validate the Julia binding's version train.

`PECOS.jl` declares its version in three places -- the package manifest, the Rust FFI crate
it ships, and the BinaryBuilder recipe that produces the tarball -- and all three must
agree or the registered package points at a build of a different version. The Julia
compatibility bound is declared twice for the same reason.

The Rust workspace guard (`scripts/check_rust_workspace.py`) separately asserts that the
FFI crate matches `Project.toml`, which is how that crate justifies sitting off the Rust
crate train.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_TOML = REPO_ROOT / "julia/PECOS.jl/Project.toml"
FFI_CARGO = REPO_ROOT / "julia/pecos-julia-ffi/Cargo.toml"
BUILD_TARBALLS = REPO_ROOT / "julia/PECOS.jl/deps/build_tarballs.jl"
FFI_CRATE_NAME = "pecos-julia-ffi"

# Julia tolerates any whitespace around `=`, and the last assignment is the one that
# reaches `build_tarballs`, so the patterns have to see every legal spelling to count them.
BUILD_TARBALLS_VERSION_RE = re.compile(r'^\s*version\s*=\s*v"([^"]+)"', re.MULTILINE)
BUILD_TARBALLS_COMPAT_RE = re.compile(r'^\s*julia_compat\s*=\s*"([^"]+)"', re.MULTILINE)


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def extract(pattern: re.Pattern[str], path: Path, what: str, errors: list[str]) -> str | None:
    """The one value `pattern` captures in `path`.

    A second declaration is an error rather than something to ignore: Julia executes the
    file, so a later assignment is what reaches `build_tarballs`, and reading only the first
    match would compare a value the build never uses.
    """
    matches = pattern.findall(path.read_text())
    if not matches:
        errors.append(f"{rel(path)}: no {what} found")
        return None
    if len(matches) > 1:
        errors.append(f"{rel(path)}: {len(matches)} {what} declarations found, expected one: {matches}")
        return None
    return matches[0]


def main() -> int:
    errors: list[str] = []

    project = load_toml(PROJECT_TOML)
    package_version = project.get("version")
    package_compat = project.get("compat", {}).get("julia")
    crate = load_toml(FFI_CARGO).get("package", {})
    crate_version = crate.get("version")
    crate_name = crate.get("name")
    tarball_version = extract(BUILD_TARBALLS_VERSION_RE, BUILD_TARBALLS, 'version = v"..."', errors)
    tarball_compat = extract(BUILD_TARBALLS_COMPAT_RE, BUILD_TARBALLS, 'julia_compat = "..."', errors)

    if not isinstance(package_version, str) or not package_version:
        print(f"error: {rel(PROJECT_TOML)}: missing version", file=sys.stderr)
        return 1

    if not isinstance(crate_version, str) or not crate_version:
        errors.append(f"{rel(FFI_CARGO)}: missing [package].version")
        crate_version = None

    for path, version in ((FFI_CARGO, crate_version), (BUILD_TARBALLS, tarball_version)):
        if version is not None and version != package_version:
            errors.append(
                f"{rel(path)}: version {version!r} does not match {rel(PROJECT_TOML)} version {package_version!r}",
            )

    # The compat bound is what the registry promises and what the tarball is actually built
    # against; a mismatch ships a package that claims support it was never built for.
    if tarball_compat is not None and tarball_compat != package_compat:
        errors.append(
            f"{rel(BUILD_TARBALLS)}: julia_compat {tarball_compat!r} does not match "
            f"{rel(PROJECT_TOML)} [compat].julia {package_compat!r}",
        )

    if crate_name != FFI_CRATE_NAME:
        errors.append(f"{rel(FFI_CARGO)}: [package].name is {crate_name!r}, expected {FFI_CRATE_NAME!r}")

    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"Julia version train OK: {package_version} across 3 files, julia compat {package_compat}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
