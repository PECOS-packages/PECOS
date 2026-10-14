#!/usr/bin/env python3
# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0
"""Require exactly one Selene wheel per discovered plugin and release runner."""

from __future__ import annotations

import argparse
from fnmatch import fnmatchcase
from pathlib import Path

from check_wheel_platform_tags import platform_tags


def main() -> None:
    """Validate the merged artifacts before publishing the combined wheel set."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheels", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    plugins = sorted(path.name for path in (root / "python/selene-plugins").glob("pecos-selene-*/") if path.is_dir())
    if not plugins:
        message = "No Selene plugins discovered"
        raise SystemExit(message)

    # These are the four build-wheels runners. Platform tags distinguish both
    # macOS architectures even when their deployment target versions change.
    platforms = {
        "ubuntu-latest": ("manylinux*_x86_64", "linux_x86_64"),
        "macos-latest": ("macosx_*_arm64",),
        "macos-15-intel": ("macosx_*_x86_64",),
        "windows-2022": ("win_amd64",),
    }
    distributions = {plugin.replace("-", "_"): plugin for plugin in plugins}
    matches: dict[tuple[str, str], list[str]] = {(plugin, runner): [] for plugin in plugins for runner in platforms}
    failures = []
    wheels = sorted(args.wheels.glob("*.whl"))
    for wheel in wheels:
        plugin = distributions.get(wheel.name.split("-", maxsplit=1)[0])
        tags = platform_tags(wheel)
        runners = [
            runner
            for runner, patterns in platforms.items()
            if tags and any(fnmatchcase(tag, pattern) for tag in tags for pattern in patterns)
        ]
        if plugin is None or len(runners) != 1:
            failures.append(f"Unexpected wheel: {wheel.name}")
            continue
        matches[plugin, runners[0]].append(wheel.name)

    for (plugin, runner), filenames in matches.items():
        if not filenames:
            failures.append(f"Missing wheel: {plugin} on {runner}")
        elif len(filenames) != 1:
            failures.append(f"Expected exactly one wheel: {plugin} on {runner}; found {filenames}")
    if failures:
        raise SystemExit("Selene wheel completeness check failed:\n- " + "\n- ".join(failures))
    print(f"Validated {len(wheels)} wheels: {len(plugins)} plugins across {len(platforms)} runners")


if __name__ == "__main__":
    main()
