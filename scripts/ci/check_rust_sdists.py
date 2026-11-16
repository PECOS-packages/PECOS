#!/usr/bin/env python3
# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0
"""Build Maturin sdists and verify their Cargo patches outside the checkout.

Use the configured PEP 517 backend, including pecos-rslib's wrapper. Hatchling
plugins have their own build hooks and are intentionally not Maturin inputs.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUILD_SDIST = """
import importlib
import sys
import tomllib
from pathlib import Path

config = tomllib.loads(Path("pyproject.toml").read_text())["build-system"]
sys.path[:0] = config.get("backend-path", [])
backend = importlib.import_module(config["build-backend"])
backend.build_sdist(sys.argv[1], config_settings={})
"""


def tool(name: str) -> str:
    """Resolve a required build tool from the active environment."""
    executable = shutil.which(name)
    if executable is None:
        msg = f"Required build tool not found: {name}"
        raise RuntimeError(msg)
    return executable


def load_toml(path: Path) -> dict:
    """Read a source or archived manifest."""
    return tomllib.loads(path.read_text())


def local_patches(manifest: dict) -> set[str]:
    """Collect paths from every patch source, including Git patches."""
    return {
        spec["path"] for patches in manifest.get("patch", {}).values() for spec in patches.values() if "path" in spec
    }


def package_pins(lock: dict) -> set[tuple]:
    """Keep the immutable identity of every resolved package."""
    return {
        (package["name"], package["version"], package.get("source"), package.get("checksum"))
        for package in lock["package"]
    }


def check_sdist(project: Path, output: Path, patches: set[str]) -> None:
    """Resolve an extracted sdist without relying on the original checkout."""
    output.mkdir()
    subprocess.run([sys.executable, "-c", BUILD_SDIST, str(output)], cwd=project, check=True)
    archives = list(output.glob("*.tar.gz"))
    if len(archives) != 1:
        msg = f"Expected one source archive for {project}, found {archives}"
        raise RuntimeError(msg)
    unpacked = output / "unpacked"
    with tarfile.open(archives[0]) as archive:
        archive.extractall(unpacked, filter="data")
    roots = list(unpacked.iterdir())
    if len(roots) != 1:
        msg = f"Expected one archive root in {archives[0]}"
        raise RuntimeError(msg)
    root = roots[0]
    manifest = root / "Cargo.toml"
    if local_patches(load_toml(manifest)) != patches:
        msg = f"{project}: source archive changed the workspace's local Cargo patches"
        raise RuntimeError(msg)
    missing = sorted(patch for patch in patches if not (root / patch / "Cargo.toml").is_file())
    if missing:
        msg = f"{project}: source archive omitted Cargo patch targets: {', '.join(missing)}"
        raise RuntimeError(msg)
    # Maturin copies the full workspace lockfile but narrows workspace members.
    # Let Cargo prune that lock offline, then require that it introduced no new
    # package version/source/checksum and that the resulting graph stays locked.
    lock = root / "Cargo.lock"
    original_pins = package_pins(load_toml(lock))
    subprocess.run(
        [tool("cargo"), "metadata", "--offline", "--all-features", "--format-version", "1"],
        cwd=root,
        stdout=subprocess.DEVNULL,
        check=True,
    )
    if not package_pins(load_toml(lock)) <= original_pins:
        msg = f"{project}: source archive requires packages outside the workspace lockfile"
        raise RuntimeError(msg)
    metadata = json.loads(
        subprocess.check_output(
            [tool("cargo"), "metadata", "--locked", "--all-features", "--format-version", "1"],
            cwd=root,
            text=True,
        ),
    )
    for package in metadata["packages"]:
        if package["source"] is None:
            # A successful resolution must not obtain local sources from the checkout.
            Path(package["manifest_path"]).resolve().relative_to(root.resolve())
        if package["name"] == "zip" and int(package["version"].split(".")[0]) < 9:
            msg = f"{project}: source archive resolves vulnerable ZIP {package['version']}"
            raise RuntimeError(msg)
    print(f"Verified {project.relative_to(ROOT)}: local patches and locked Cargo graph", flush=True)


def main() -> None:
    """Discover every tracked Maturin project and check its source distribution."""
    manifests = subprocess.check_output(
        [tool("git"), "ls-files", "**/pyproject.toml"],
        cwd=ROOT,
        text=True,
    ).splitlines()
    projects = [(ROOT / path).parent for path in manifests if "maturin" in load_toml(ROOT / path).get("tool", {})]
    if not projects:
        msg = "No Maturin projects found"
        raise RuntimeError(msg)
    patches = local_patches(load_toml(ROOT / "Cargo.toml"))
    with tempfile.TemporaryDirectory(prefix="pecos-sdists-") as directory:
        for index, project in enumerate(projects):
            check_sdist(project, Path(directory) / str(index), patches)
    print(f"Verified {len(projects)} Maturin source distributions")


if __name__ == "__main__":
    main()
