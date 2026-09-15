#!/usr/bin/env python3
# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0
"""Build the bare-Wasm Frontier decoder with an optional embedded Stim DEM."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "pecos-frontier-wasm"
TARGET = "wasm32-unknown-unknown"
MODULE = "pecos_frontier_wasm.wasm"
GETRANDOM_CFG = '--cfg getrandom_backend="unsupported"'


def cargo_executable() -> str:
    """Return cargo from PATH or fail with a useful message."""
    cargo = shutil.which("cargo")
    if cargo is None:
        message = "cargo was not found on PATH"
        raise RuntimeError(message)
    return cargo


def target_directory(cargo: str) -> Path:
    """Ask Cargo for its configured target directory."""
    result = subprocess.run(
        [cargo, "metadata", "--no-deps", "--format-version", "1"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(json.loads(result.stdout)["target_directory"])


def build(dem_path: Path | None) -> Path:
    """Build and copy the module, returning the distribution path."""
    cargo = cargo_executable()
    target_dir = target_directory(cargo)
    environment = os.environ.copy()
    environment["RUSTFLAGS"] = " ".join(value for value in (environment.get("RUSTFLAGS"), GETRANDOM_CFG) if value)
    if dem_path is None:
        environment.pop("FRONTIER_DEM_PATH", None)
    else:
        resolved_dem = dem_path.resolve(strict=True)
        if not resolved_dem.is_file():
            message = f"DEM path is not a file: {resolved_dem}"
            raise ValueError(message)
        environment["FRONTIER_DEM_PATH"] = str(resolved_dem)

    subprocess.run(
        [cargo, "build", "--release", "--target", TARGET, "-p", PACKAGE],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
    )

    source = target_dir / TARGET / "release" / MODULE
    destination = REPO_ROOT / "dist" / MODULE
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dem",
        nargs="?",
        help="flattened Stim detector error model to embed",
    )
    args = parser.parse_args()
    dem_path = Path(args.dem) if args.dem else None
    output = build(dem_path)
    print(f"Built {output} ({output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
