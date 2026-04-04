#!/usr/bin/env python3
"""Cross-platform cleaning script for PECOS build artifacts.

This script handles cleaning of build artifacts across Windows, macOS, and Linux
without requiring Unix-specific tools like `find` or `rm`.

Usage:
    uv run python scripts/clean.py [options]

Options:
    --cache     Clean ~/.pecos/cache/ and ~/.pecos/tmp/
    --deps      Clean ~/.pecos/deps/ (LLVM, CUDA, cuQuantum)
    --selene    Clean only Selene plugin artifacts
    --all       Clean everything (project + selene + cache + deps)
    --dry-run   Show what would be deleted without deleting
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def rmtree_safe(path: Path, *, dry_run: bool = False) -> bool:
    """Safely remove a directory tree, returning True if something was removed."""
    if path.exists():
        if dry_run:
            print(f"  Would remove: {path}")
        else:
            shutil.rmtree(path, ignore_errors=True)
            if not path.exists():
                print(f"  Removed: {path}")
                return True
            print(f"  Failed to remove: {path}")
        return True
    return False


def rm_safe(path: Path, *, dry_run: bool = False) -> bool:
    """Safely remove a file, returning True if something was removed."""
    if path.exists() and path.is_file():
        if dry_run:
            print(f"  Would remove: {path}")
        else:
            try:
                path.unlink()
            except OSError as e:
                print(f"  Failed to remove {path}: {e}")
            else:
                print(f"  Removed: {path}")
                return True
        return True
    return False


def find_and_remove_dirs(
    root: Path,
    name: str,
    *,
    dry_run: bool = False,
    skip_venv: bool = True,
) -> int:
    """Find and remove all directories with the given name under root."""
    count = 0
    if not root.exists():
        return count

    for path in root.rglob(name):
        if path.is_dir() and ".git" not in path.parts:
            if skip_venv and ".venv" in path.parts:
                continue
            if rmtree_safe(path, dry_run=dry_run):
                count += 1
    return count


def find_and_remove_files(
    root: Path,
    pattern: str,
    *,
    dry_run: bool = False,
    skip_venv: bool = True,
) -> int:
    """Find and remove all files matching the pattern under root."""
    count = 0
    if not root.exists():
        return count

    for path in root.rglob(pattern):
        if path.is_file():
            if skip_venv and ".venv" in path.parts:
                continue
            if rm_safe(path, dry_run=dry_run):
                count += 1
    return count


def run_command(cmd: list[str], *, quiet: bool = True) -> bool:
    """Run a command, returning True if successful."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=quiet,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return False
    else:
        return result.returncode == 0


def clean_project(root: Path, *, dry_run: bool = False) -> None:
    """Clean project build artifacts."""
    print("Cleaning project build artifacts...")

    # Cargo clean
    if not dry_run:
        print("  Running cargo clean...")
        run_command(["cargo", "clean", "-q"])
    else:
        print("  Would run: cargo clean")

    # Top-level directories
    for dirname in ["dist", "site", ".ruff_cache"]:
        rmtree_safe(root / dirname, dry_run=dry_run)

    # Python docs build
    rmtree_safe(root / "python" / "docs" / "_build", dry_run=dry_run)

    # Find and remove common build directories
    dir_patterns = [
        "*.egg-info",
        "build",
        ".pytest_cache",
        ".ipynb_checkpoints",
        ".hypothesis",
        "junit",
        "__pycache__",
    ]
    for pattern in dir_patterns:
        count = find_and_remove_dirs(root, pattern, dry_run=dry_run)
        if count > 0 and not dry_run:
            print(f"  Removed {count} '{pattern}' directories")

    # Compiled Python extensions
    python_dir = root / "python"
    if python_dir.exists():
        so_count = find_and_remove_files(python_dir, "*.so", dry_run=dry_run)
        pyd_count = find_and_remove_files(python_dir, "*.pyd", dry_run=dry_run)
        if (so_count + pyd_count) > 0 and not dry_run:
            print(f"  Removed {so_count + pyd_count} compiled extensions")

    # Julia artifacts
    julia_dir = root / "julia"
    if julia_dir.exists():
        rm_safe(julia_dir / "PECOS.jl" / "Manifest.toml", dry_run=dry_run)
        rm_safe(
            julia_dir / "PECOS.jl" / "dev" / "PECOS_julia_jll" / "Manifest.toml",
            dry_run=dry_run,
        )
        find_and_remove_files(julia_dir, "*.jl.*.cov", dry_run=dry_run)
        find_and_remove_files(julia_dir, "*.jl.cov", dry_run=dry_run)
        find_and_remove_files(julia_dir, "*.jl.mem", dry_run=dry_run)

    # Clean pecos_rslib from venv
    venv_dir = root / ".venv"
    if venv_dir.exists():
        for site_packages in venv_dir.rglob("site-packages"):
            for pecos_rslib in site_packages.glob("pecos_rslib*"):
                rmtree_safe(pecos_rslib, dry_run=dry_run)

    # Clean uv cache for pecos-rslib (use --force to avoid blocking on cache lock)
    if not dry_run:
        run_command(["uv", "cache", "clean", "--force", "pecos-rslib"])
    else:
        print("  Would run: uv cache clean --force pecos-rslib")


def clean_selene(root: Path, *, dry_run: bool = False) -> None:
    """Clean Selene plugin artifacts."""
    print("Cleaning Selene plugin artifacts...")
    selene_dir = root / "python" / "selene-plugins"
    if selene_dir.exists():
        count = 0
        for plugin_dir in selene_dir.iterdir():
            if plugin_dir.is_dir():
                for python_pkg in (plugin_dir / "python").glob("*"):
                    dist_dir = python_pkg / "_dist"
                    if rmtree_safe(dist_dir, dry_run=dry_run):
                        count += 1
        if count > 0:
            print(f"  Removed {count} _dist directories")


def clean_pecos_home(what: str, *, dry_run: bool = False) -> None:
    """Clean ~/.pecos/ directories."""
    pecos_home = Path.home() / ".pecos"

    if what == "cache":
        print("Cleaning ~/.pecos/cache/ and ~/.pecos/tmp/...")
        rmtree_safe(pecos_home / "cache", dry_run=dry_run)
        rmtree_safe(pecos_home / "tmp", dry_run=dry_run)
    elif what == "deps":
        print("Cleaning ~/.pecos/deps/...")
        rmtree_safe(pecos_home / "deps", dry_run=dry_run)


def main() -> int:
    """Entry point for the cleaning script."""
    parser = argparse.ArgumentParser(
        description="Cross-platform cleaning script for PECOS build artifacts",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Clean ~/.pecos/cache/ and ~/.pecos/tmp/",
    )
    parser.add_argument(
        "--deps",
        action="store_true",
        help="Clean ~/.pecos/deps/ (LLVM, CUDA, cuQuantum)",
    )
    parser.add_argument(
        "--selene",
        action="store_true",
        help="Clean only Selene plugin artifacts",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Clean everything (project + selene + cache + deps)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without deleting",
    )

    args = parser.parse_args()

    # Find project root (directory containing Cargo.toml)
    root = Path(__file__).parent.parent.resolve()
    if not (root / "Cargo.toml").exists():
        print(f"Error: Could not find project root (no Cargo.toml in {root})")
        return 1

    if args.dry_run:
        print("DRY RUN - showing what would be deleted\n")

    if args.all:
        clean_project(root, dry_run=args.dry_run)
        clean_selene(root, dry_run=args.dry_run)
        clean_pecos_home("cache", dry_run=args.dry_run)
        clean_pecos_home("deps", dry_run=args.dry_run)
    elif args.selene:
        clean_selene(root, dry_run=args.dry_run)
    elif args.cache or args.deps:
        if args.cache:
            clean_pecos_home("cache", dry_run=args.dry_run)
        if args.deps:
            clean_pecos_home("deps", dry_run=args.dry_run)
    else:
        # Default: clean project artifacts only
        clean_project(root, dry_run=args.dry_run)
        clean_selene(root, dry_run=args.dry_run)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
