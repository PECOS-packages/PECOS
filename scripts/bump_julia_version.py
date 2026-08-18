#!/usr/bin/env python3
# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0
"""Move the Julia binding onto a new version.

`PECOS.jl`'s version is a literal in three coordinated files: the package manifest, the
Rust FFI crate it ships, and the BinaryBuilder recipe. This rewrites all three together;
`scripts/check_julia_versions.py` is what catches them drifting apart.
"""

from __future__ import annotations

import argparse
import re
import sys

from check_julia_versions import (
    BUILD_TARBALLS,
    BUILD_TARBALLS_VERSION_RE,
    FFI_CARGO,
    PROJECT_TOML,
    load_toml,
    rel,
)
from version_bump import rewrite_version

# Julia's VersionNumber is SemVer 2.0.0; this is the official recommended regex from
# https://semver.org, which rejects the leading `v` and the loose two-part forms that a
# Julia manifest will not accept.
SEMVER_RE = re.compile(
    r"^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)"
    r"(?:-(?P<prerelease>(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)"
    r"(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?"
    r"(?:\+(?P<buildmetadata>[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$",
    # ASCII only: Python's `\d` also matches non-ASCII digits, which SemVer forbids and
    # which would otherwise be written straight into a Julia manifest.
    re.ASCII,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="the new version, e.g. 0.2.0-dev0")
    args = parser.parse_args()

    if SEMVER_RE.match(args.version) is None:
        print(f"error: {args.version!r} is not a valid SemVer version", file=sys.stderr)
        return 1

    old = load_toml(PROJECT_TOML).get("version")
    if not isinstance(old, str) or not old:
        print(f"error: {rel(PROJECT_TOML)}: missing version", file=sys.stderr)
        return 1
    if old == args.version:
        print(f"error: already on version {old}", file=sys.stderr)
        return 1

    try:
        stale, changes = rewrite_version([PROJECT_TOML, FFI_CARGO, BUILD_TARBALLS], old, args.version)
    except OSError as err:
        print(f"error: {err}; any files already written were restored", file=sys.stderr)
        return 1
    if stale:
        for path in stale:
            print(f"error: {rel(path)} does not carry version {old}; nothing was changed", file=sys.stderr)
        return 1

    # Confirm the three declarations themselves moved, not just some other occurrence.
    declared = {
        PROJECT_TOML: load_toml(PROJECT_TOML).get("version"),
        FFI_CARGO: load_toml(FFI_CARGO).get("package", {}).get("version"),
        BUILD_TARBALLS: next(
            iter(BUILD_TARBALLS_VERSION_RE.findall(BUILD_TARBALLS.read_text())),
            None,
        ),
    }
    undeclared = [path for path, version in declared.items() if version != args.version]
    if undeclared:
        for path in undeclared:
            print(f"error: {rel(path)}: version is still not {args.version}", file=sys.stderr)
        return 1

    for path, count in changes:
        print(f"  {rel(path)} ({count})")
    print(f"{old} -> {args.version} in {len(changes)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
